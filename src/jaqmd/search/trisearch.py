from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from ..tokenize.trigram import to_fts_query


@dataclass
class SearchResult:
    docid: str
    score: float
    filepath: str
    title: str
    snippet: str


def trisearch(
    conn: sqlite3.Connection,
    query: str,
    *,
    n: int = 5,
    collection: Optional[str] = None,
    min_score: Optional[float] = None,
    all_results: bool = False,
) -> list[SearchResult]:
    """trigram BM25 検索を実行する。"""
    fts_query = to_fts_query(query)
    if not fts_query:
        return []

    where_clauses = ["docs_fts_trigram MATCH ?"]
    params: list = [fts_query]

    if collection:
        where_clauses.append("filepath LIKE ?")
        params.append(f"{collection}/%")

    where_sql = " AND ".join(where_clauses)
    limit_sql = "" if all_results else f"LIMIT {n}"

    sql = f"""
        SELECT
            docid,
            filepath,
            title,
            snippet(docs_fts_trigram, 3, '', '', '...', 20) AS snippet,
            bm25(docs_fts_trigram) AS score
        FROM docs_fts_trigram
        WHERE {where_sql}
        ORDER BY bm25(docs_fts_trigram)
        {limit_sql}
    """

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
            )
        )
    return results
