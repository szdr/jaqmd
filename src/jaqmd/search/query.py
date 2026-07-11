from __future__ import annotations

import dataclasses
import sqlite3
from typing import Callable, Optional

from .trisearch import SearchResult, trisearch
from ..store import get_meta
from ..rerank import rerank, RERANK_TOP_K, DEFAULT_RERANKER
from ..qe import expand as qe_expand, ExpansionResult
from ..progress import NULL_REPORTER, ProgressReporter

# AGENTS.md 準拠: RRF パラメータ k=60
RRF_K = 60


def _minmax_scale(results: list[SearchResult]) -> list[SearchResult]:
    """RRF スコアを min-max スケーリングで 0-1 に正規化する。

    融合済み全候補の min/max を使って正規化するため、
    最も関連度が高い結果が 1.0、最も低い結果が 0.0 になる。

    Args:
        results: RRF 融合済みの SearchResult リスト（スコア降順）。

    Returns:
        score を 0-1 に正規化した SearchResult リスト（並び順は維持）。
    """
    if not results:
        return results
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # 全件同スコア（1件のみ含む）はゼロ除算を避けて全件 1.0
        return [dataclasses.replace(r, score=1.0) for r in results]
    span = hi - lo
    return [dataclasses.replace(r, score=(r.score - lo) / span) for r in results]


def _rrf_fuse(
    result_lists: list[list[SearchResult]],
    k: int = RRF_K,
    weights: Optional[list[float]] = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion で複数の検索結果リストを融合する。

    各リストの rank（0始まり）位置の結果に weight / (k + rank + 1) を加算する。
    代表 SearchResult は docid 初出時のもの（リスト順）を保持し、
    RRF スコアを score フィールドに差し替えて返す。

    Args:
        result_lists: 融合する SearchResult リストのリスト（trigram→morph→vec の順を推奨）。
        k: RRF パラメータ（デフォルト 60）。
        weights: result_lists と同じ長さの重みリスト。None なら全リスト重み 1.0
            （tobi/qmd 風の「先頭 search を 2x 優遇」等の実装に利用する）。

    Returns:
        RRF スコア降順の SearchResult リスト。
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}

    for weight, result_list in zip(weights, result_lists):
        for rank, result in enumerate(result_list):
            docid = result.docid
            scores[docid] = scores.get(docid, 0.0) + weight / (k + rank + 1)
            if docid not in representatives:
                representatives[docid] = result

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
    candidate_top_k: Optional[int],
    reporter: Optional[ProgressReporter] = None,
) -> list[SearchResult]:
    """RRF 融合後の共通後処理: rerank → min-max 正規化 → 足切り → n 制限。

    `query()` と `query_searches()` で共有する（重複回避）。
    """
    top_k = None if all_results else candidate_top_k
    fused = rerank(
        query_for_rerank,
        fused,
        enabled=rerank_enabled,
        model=rerank_model,
        top_k=top_k,
        reporter=reporter,
    )

    fused = _minmax_scale(fused)

    if min_score is not None:
        fused = [r for r in fused if r.score >= min_score]

    if not all_results:
        fused = fused[:n]

    return fused


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
        min_score: RRF スコアの最小閾値（None で足切りなし）。
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
                reporter=reporter,
            )
            result_lists.append(hyde_results)

    fused = _rrf_fuse(result_lists, k=RRF_K)

    # reranker（融合プール全体に適用してから n 制限する。--all 時は全件を再スコア）
    return _finalize(
        fused,
        query_for_rerank=query_text,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        all_results=all_results,
        n=n,
        min_score=min_score,
        candidate_top_k=RERANK_TOP_K,
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
        min_score: 0-1 正規化後のスコア閾値。
        candidate_limit: reranker に渡す融合プールの上位候補数。
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
            tri_results = trisearch(conn, text, all_results=True, reporter=reporter)
            result_lists.append(tri_results)
            weights.append(weight)

            if morph_available:
                from .mosearch import mosearch

                mo_results = mosearch(conn, text, all_results=True, reporter=reporter)
                result_lists.append(mo_results)
                weights.append(weight)

        elif stype in ("vec", "hyde"):
            if vec_indexed:
                from .vsearch import vsearch

                vs_results = vsearch(conn, text, all_results=True, reporter=reporter)
                result_lists.append(vs_results)
                weights.append(weight)

        else:
            raise ValueError(f"未知の search type です: {stype!r}（lex/vec/hyde のいずれか）")

    if not result_lists:
        return []

    fused = _rrf_fuse(result_lists, k=RRF_K, weights=weights)

    # collections フィルタ（OR）。filepath は "collection/path" 構成のため先頭セグメントで判定する。
    if collections:
        collection_set = set(collections)
        fused = [
            r for r in fused if r.filepath.split("/", 1)[0] in collection_set
        ]

    return _finalize(
        fused,
        query_for_rerank=searches[0][1],
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        all_results=False,
        n=limit,
        min_score=min_score,
        candidate_top_k=candidate_limit,
        reporter=reporter,
    )
