from __future__ import annotations

import dataclasses
import sqlite3
from typing import Callable, Optional

from ..config import settings
from ..progress import NULL_REPORTER, ProgressReporter
from ..qe import ExpansionResult
from ..qe import expand as qe_expand
from ..rerank import DEFAULT_RERANKER, rerank_scores
from ..store import get_meta
from .trisearch import SearchResult, trisearch

# AGENTS.md 準拠: RRF パラメータ既定 k=60（JAQMD_TUNING_RRF_K / 設定ファイルで変更可能）
RRF_K = settings.rrf_k


def _blend_scores(
    candidates: list[SearchResult],
    rerank_scores_list: Optional[list[float]],
) -> list[SearchResult]:
    """tobi/qmd 風の位置依存ブレンドで最終スコアを算出し、降順ソートして返す。

    融合順位 rrfRank（candidates 内の 1 始まり順位）の逆数 1/rrfRank と、
    rerankScore（sigmoid 正規化済み）を位置依存の重みで加重合成する。
    上位ほど RRF 順位を重視し（reranker の乱れから保護）、下位ほど rerank を重視する。

        rrf_score  = 1 / rrfRank
        rrf_weight = 0.75 (rrfRank<=3) / 0.60 (<=10) / 0.40 (それ以外)
        blended    = rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score

    Args:
        candidates: RRF 融合順（スコア降順）の SearchResult リスト。
        rerank_scores_list: candidates と同順の rerankScore 列。None なら reranker
            無効/失敗の degrade とみなし rerank_score = rrf_score を使う
            （結果として blended = 1/rrfRank となり RRF 融合順位がそのまま残る）。

    Returns:
        blended スコアを score に持つ SearchResult リスト（スコア降順）。
    """
    blended: list[SearchResult] = []
    for i, r in enumerate(candidates):
        rrf_rank = i + 1
        rrf_score = 1.0 / rrf_rank
        if rrf_rank <= 3:
            rrf_weight = 0.75
        elif rrf_rank <= 10:
            rrf_weight = 0.60
        else:
            rrf_weight = 0.40
        rerank_score = (
            rerank_scores_list[i] if rerank_scores_list is not None else rrf_score
        )
        score = rrf_weight * rrf_score + (1.0 - rrf_weight) * rerank_score
        blended.append(dataclasses.replace(r, score=score))

    blended.sort(key=lambda r: r.score, reverse=True)
    return blended


def _rrf_fuse(
    result_lists: list[list[SearchResult]],
    k: int = RRF_K,
    weights: Optional[list[float]] = None,
    top_rank_bonus: bool = True,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion で複数の検索結果リストを融合する。

    各リストの rank（0始まり）位置の結果に weight / (k + rank + 1) を加算する。
    代表 SearchResult は docid 初出時のもの（リスト順）を保持し、
    RRF スコアを score フィールドに差し替えて返す。

    さらに tobi/qmd に倣い top-rank ボーナスを加える: いずれかのリストで
    最上位（rank 0）に現れた doc に +0.05、上位（rank <= 2）に +0.02。
    最終スコアは後段の位置依存ブレンド（_blend_scores）で 1/rrfRank ベースに
    差し替わるため、このボーナスは融合順位（＝ rrfRank と候補プールの切り出し）
    にのみ効き、最終スコアの値そのものには影響しない。

    Args:
        result_lists: 融合する SearchResult リストのリスト（trigram→morph→vec の順を推奨）。
        k: RRF パラメータ（デフォルト 60）。
        weights: result_lists と同じ長さの重みリスト。None なら全リスト重み 1.0
            （tobi/qmd 風の「先頭 search を 2x 優遇」等の実装に利用する）。
        top_rank_bonus: True なら top-rank ボーナスを加算する（既定）。生の RRF 値を
            検証したい場合は False。

    Returns:
        RRF スコア降順の SearchResult リスト。
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}
    top_rank: dict[str, int] = {}

    for weight, result_list in zip(weights, result_lists):
        for rank, result in enumerate(result_list):
            docid = result.docid
            scores[docid] = scores.get(docid, 0.0) + weight / (k + rank + 1)
            if docid not in representatives:
                representatives[docid] = result
            if docid not in top_rank or rank < top_rank[docid]:
                top_rank[docid] = rank

    if top_rank_bonus:
        for docid, tr in top_rank.items():
            if tr == 0:
                scores[docid] += 0.05
            elif tr <= 2:
                scores[docid] += 0.02

    fused = [
        dataclasses.replace(representatives[docid], score=score)
        for docid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def _finalize(
    fused: list[SearchResult],
    *,
    query_for_rerank: str,
    rerank_enabled: bool,
    rerank_model: str,
    all_results: bool,
    n: int,
    min_score: Optional[float],
    candidate_limit: Optional[int],
    reporter: Optional[ProgressReporter] = None,
) -> list[SearchResult]:
    """RRF 融合後の共通後処理: 候補確定 → rerank → 位置依存ブレンド → 足切り → n 制限。

    `query()` と `query_searches()` で共有する（重複回避）。

    tobi/qmd に倣い、候補プール（RRF 融合順の先頭 candidate_limit 件）全体を
    reranker にかけ、`_blend_scores` で 1/rrfRank と rerankScore を位置依存合成する。
    min-max 正規化は行わない（min_score はブレンド絶対スコアへの閾値）。

    all_results=True でも reranker 対象は先頭 candidate_limit 件に制限する
    （全ヒット×全文の一括推論による OOM 防止）。tail は rerank_score = 1/rrfRank
    の degrade 扱いでブレンドし、RRF 融合順位を保つ。
    """
    pool = (
        fused if all_results else fused[:candidate_limit] if candidate_limit else fused
    )
    head = pool[:candidate_limit] if candidate_limit else pool
    rr = rerank_scores(
        query_for_rerank,
        head,
        enabled=rerank_enabled,
        model=rerank_model,
        reporter=reporter,
    )
    if rr is not None and len(rr) < len(pool):
        rr = rr + [1.0 / (i + 1) for i in range(len(rr), len(pool))]
    blended = _blend_scores(pool, rr)

    if min_score is not None:
        blended = [r for r in blended if r.score >= min_score]

    if not all_results:
        blended = blended[:n]

    return blended


def query(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    n: int = 5,
    collection: Optional[str] = None,
    min_score: Optional[float] = None,
    all_results: bool = False,
    rerank_enabled: bool = True,
    rerank_model: str = DEFAULT_RERANKER,
    qe_enabled: bool = True,
    snippet_chars: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None,
    on_expansion: Optional[Callable[[Optional[ExpansionResult]], None]] = None,
) -> list[SearchResult]:
    """ハイブリッド検索: RRF 融合による trigram / morph / vector 統合検索。

    利用可能なインデックス（index_meta）に基づいて動的に検索を組み合わせる。
    morph / vec 未構築でも trigram のみで degrade して動作する。

    Args:
        conn: sqlite3 接続（row_factory = sqlite3.Row 前提）。
        query_text: 検索クエリ文字列。
        n: 返却件数（all_results=True の場合は無視）。
        collection: コレクション名での絞り込み（None で全コレクション）。
        min_score: ブレンド後スコアの最小閾値（None で足切りなし）。
        all_results: True なら全件返却（n・min_score は適用しない）。
        rerank_enabled: False なら reranker を無効化（RRF 順のまま）。
        rerank_model: 使用する reranker モデルキー（既定 "default"）。
        qe_enabled: False なら Query Expansion を無効化（raw クエリのみ使用）。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。
        on_expansion: 指定すると Query Expansion 完了直後（trigram/morph/vector
            検索の実行前）に、結果（ExpansionResult か None）を引数に1度だけ呼ばれる。
            呼び出し側が展開結果を即座に表示する等に利用する。

    Returns:
        RRF 融合 + rerank 後の SearchResult リスト（スコア降順）。
    """
    reporter = reporter or NULL_REPORTER
    candidate_n = max(n * 5, 20)

    # Query Expansion: lex/vec/hyde に展開する（未導入・失敗時は None で raw に degrade）
    exp = qe_expand(conn, query_text, reporter=reporter) if qe_enabled else None
    if on_expansion is not None:
        on_expansion(exp)
    lex_query = " ".join([query_text, *exp.lex]) if exp else query_text
    vec_query = exp.vec if exp and exp.vec else query_text
    hyde_text = exp.hyde if exp and exp.hyde else None

    result_lists: list[list[SearchResult]] = []

    # trigram は常に実行（lex 展開語を付加）
    tri_results = trisearch(
        conn,
        lex_query,
        n=candidate_n,
        collection=collection,
        all_results=True,
        snippet_chars=snippet_chars,
        reporter=reporter,
    )
    result_lists.append(tri_results)

    # morph: morph_indexed が立っているときのみ（lex 展開語を付加）
    if get_meta(conn, "morph_indexed") == "1":
        from .mosearch import mosearch

        mo_results = mosearch(
            conn,
            lex_query,
            n=candidate_n,
            collection=collection,
            all_results=True,
            snippet_chars=snippet_chars,
            reporter=reporter,
        )
        result_lists.append(mo_results)

    # vector: vec_indexed が立っているときのみ（vec 展開文で検索）
    if get_meta(conn, "vec_indexed") == "1":
        from .vsearch import vsearch

        vs_results = vsearch(
            conn,
            vec_query,
            n=candidate_n,
            collection=collection,
            all_results=True,
            snippet_chars=snippet_chars,
            reporter=reporter,
        )
        result_lists.append(vs_results)

        # HyDE: 仮想文書が得られたときのみ追加のベクトル検索リストとして融合
        if hyde_text:
            hyde_results = vsearch(
                conn,
                hyde_text,
                n=candidate_n,
                collection=collection,
                all_results=True,
                snippet_chars=snippet_chars,
                reporter=reporter,
            )
            result_lists.append(hyde_results)

    fused = _rrf_fuse(result_lists, k=RRF_K)

    # reranker（候補プールを再スコアして位置依存ブレンド。--all 時は全件を再スコア）
    return _finalize(
        fused,
        query_for_rerank=query_text,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        all_results=all_results,
        n=n,
        min_score=min_score,
        candidate_limit=settings.rerank_candidate_limit,
        reporter=reporter,
    )


# tobi/qmd 風の typed search 種別
SearchType = str  # "lex" | "vec" | "hyde"


def query_searches(
    conn: sqlite3.Connection,
    searches: list[tuple[str, str]],
    *,
    collections: Optional[list[str]] = None,
    limit: int = 10,
    min_score: float = 0.0,
    candidate_limit: int = 40,
    rerank_enabled: bool = True,
    rerank_model: str = DEFAULT_RERANKER,
    snippet_chars: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None,
) -> list[SearchResult]:
    """tobi/qmd 風の typed searches 配列によるハイブリッド検索（MCP query ツール用）。

    `query()` が単一クエリ文字列＋内部 Query Expansion で lex/vec/hyde を自動生成するのに対し、
    こちらは呼び出し側（MCP クライアント）が明示的に型付きサブクエリを渡す。

    Args:
        conn: sqlite3 接続。
        searches: `(type, text)` のリスト（type は "lex" / "vec" / "hyde"）。1〜10 件を想定。
            先頭の search は RRF 融合時に weight=2.0 で優遇される（tobi/qmd 仕様）。
            "lex" は trigram（常時）＋ morph（morph_indexed なら）の両方で検索する。
            "vec" / "hyde" は vec_indexed のときのみ vsearch で検索する（未構築時は無視して degrade）。
        collections: 絞り込むコレクション名のリスト（OR）。None で全コレクション。
        limit: 返却件数。
        min_score: ブレンド後スコアの最小閾値。
        candidate_limit: reranker に渡す融合プールの上位候補数（rerank+ブレンド対象）。
        rerank_enabled: False なら reranker を無効化（RRF 順のまま）。
        rerank_model: 使用する reranker モデルキー。
        reporter: 進捗表示用の ProgressReporter。

    Returns:
        RRF 融合 + rerank 後の SearchResult リスト（スコア降順、最大 limit 件）。
    """
    reporter = reporter or NULL_REPORTER
    if not searches:
        return []

    morph_available = get_meta(conn, "morph_indexed") == "1"
    vec_indexed = get_meta(conn, "vec_indexed") == "1"

    result_lists: list[list[SearchResult]] = []
    weights: list[float] = []

    for i, (stype, text) in enumerate(searches):
        weight = 2.0 if i == 0 else 1.0

        if stype == "lex":
            tri_results = trisearch(
                conn,
                text,
                all_results=True,
                snippet_chars=snippet_chars,
                reporter=reporter,
            )
            result_lists.append(tri_results)
            weights.append(weight)

            if morph_available:
                from .mosearch import mosearch

                mo_results = mosearch(
                    conn,
                    text,
                    all_results=True,
                    snippet_chars=snippet_chars,
                    reporter=reporter,
                )
                result_lists.append(mo_results)
                weights.append(weight)

        elif stype in ("vec", "hyde"):
            if vec_indexed:
                from .vsearch import vsearch

                vs_results = vsearch(
                    conn,
                    text,
                    all_results=True,
                    snippet_chars=snippet_chars,
                    reporter=reporter,
                )
                result_lists.append(vs_results)
                weights.append(weight)

        else:
            raise ValueError(
                f"未知の search type です: {stype!r}（lex/vec/hyde のいずれか）"
            )

    if not result_lists:
        return []

    fused = _rrf_fuse(result_lists, k=RRF_K, weights=weights)

    # collections フィルタ（OR）。filepath は "collection/path" 構成のため先頭セグメントで判定する。
    if collections:
        collection_set = set(collections)
        fused = [r for r in fused if r.filepath.split("/", 1)[0] in collection_set]

    return _finalize(
        fused,
        query_for_rerank=searches[0][1],
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        all_results=False,
        n=limit,
        min_score=min_score,
        candidate_limit=candidate_limit,
        reporter=reporter,
    )
