from __future__ import annotations

import dataclasses
import sqlite3
from typing import Optional

from .trisearch import SearchResult, trisearch
from ..store import get_meta
from ..rerank import rerank, RERANK_TOP_K
from ..qe import expand as qe_expand
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
    result_lists: list[list[SearchResult]], k: int = RRF_K
) -> list[SearchResult]:
    """Reciprocal Rank Fusion で複数の検索結果リストを融合する。

    各リストの rank（0始まり）位置の結果に 1/(k + rank + 1) を加算する。
    代表 SearchResult は docid 初出時のもの（リスト順）を保持し、
    RRF スコアを score フィールドに差し替えて返す。

    Args:
        result_lists: 融合する SearchResult リストのリスト（trigram→morph→vec の順を推奨）。
        k: RRF パラメータ（デフォルト 60）。

    Returns:
        RRF スコア降順の SearchResult リスト。
    """
    scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            docid = result.docid
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank + 1)
            if docid not in representatives:
                representatives[docid] = result

    fused = [
        dataclasses.replace(representatives[docid], score=score)
        for docid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
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
    qe_enabled: bool = True,
    reporter: Optional[ProgressReporter] = None,
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
        qe_enabled: False なら Query Expansion を無効化（raw クエリのみ使用）。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。

    Returns:
        RRF 融合 + rerank 後の SearchResult リスト（スコア降順）。
    """
    reporter = reporter or NULL_REPORTER
    candidate_n = max(n * 5, 20)

    # Query Expansion: lex/vec/hyde に展開する（未導入・失敗時は None で raw に degrade）
    exp = qe_expand(conn, query_text, reporter=reporter) if qe_enabled else None
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
    top_k = None if all_results else RERANK_TOP_K
    fused = rerank(
        query_text, fused, enabled=rerank_enabled, top_k=top_k, reporter=reporter
    )

    fused = _minmax_scale(fused)

    # min_score 足切り（0-1 正規化後のスコアと比較、all_results の有無に関わらず適用）
    if min_score is not None:
        fused = [r for r in fused if r.score >= min_score]

    # n 制限（all_results=True なら適用しない）
    if not all_results:
        fused = fused[:n]

    return fused
