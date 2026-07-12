from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from ..progress import NULL_REPORTER, ProgressReporter
from ..tokenize.trigram import to_fts_query


@dataclass
class SearchResult:
    docid: str
    score: float
    filepath: str
    title: str
    snippet: str
    body: str = ""


def trisearch(
    conn: sqlite3.Connection,
    query: str,
    *,
    n: int = 5,
    collection: Optional[str] = None,
    min_score: Optional[float] = None,
    all_results: bool = False,
    reporter: Optional[ProgressReporter] = None,
) -> list[SearchResult]:
    """trigram BM25 検索を実行する。"""
    reporter = reporter or NULL_REPORTER
    fts_query = to_fts_query(query)
    if not fts_query:
        return []

    where_clauses = ["docs_fts_trigram MATCH ?"]
    params: list = [fts_query]

    if collection:
        # 注意: docs_fts_trigram は tokenize='trigram' のため、filepath (UNINDEXED)
        # に対する LIKE/GLOB はトライグラム索引経由の最適化に誤って解釈され、
        # 3文字以上の連続一致パターンで 0 件になる。documents 側で絞り込む。
        where_clauses.append(
            "docid IN (SELECT docid FROM documents WHERE collection = ?)"
        )
        params.append(collection)

    where_sql = " AND ".join(where_clauses)
    limit_sql = "" if all_results else f"LIMIT {n}"

    sql = f"""
        SELECT
            docid,
            filepath,
            title,
            body,
            snippet(docs_fts_trigram, 3, '', '', '...', 40) AS snippet,
            bm25(docs_fts_trigram) AS score
        FROM docs_fts_trigram
        WHERE {where_sql}
        ORDER BY bm25(docs_fts_trigram)
        {limit_sql}
    """

    with reporter.step("trigram 検索"):
        rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        score = -float(row["score"])  # bm25() は負値（小さいほど良い）を返す
        if min_score is not None and score < min_score:
            continue
        results.append(
            SearchResult(
                docid=row["docid"],
                score=score,
                filepath=row["filepath"],
                title=row["title"] or "",
                snippet=row["snippet"] or "",
                body=row["body"] or "",
            )
        )
    return results
