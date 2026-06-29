from __future__ import annotations

import sqlite3
from typing import Optional

from ..tokenize.morph import snippet_terms, to_fts_query
from .snippet import extract_snippet
from .trisearch import SearchResult


def mosearch(
    conn: sqlite3.Connection,
    query: str,
    *,
    n: int = 5,
    collection: Optional[str] = None,
    min_score: Optional[float] = None,
    all_results: bool = False,
) -> list[SearchResult]:
    """形態素 BM25 検索を実行する。スニペットは原文ベースで生成する。"""
    fts_query = to_fts_query(query)
    if not fts_query:
        return []

    where_clauses = ["docs_fts_morph MATCH ?"]
    params: list = [fts_query]

    if collection:
        where_clauses.append("m.filepath LIKE ?")
        params.append(f"{collection}/%")

    where_sql = " AND ".join(where_clauses)
    limit_sql = "" if all_results else f"LIMIT {n}"

    # docs_fts_morph は正規化形テキストを格納しているため、
    # 原文 (content.body) と原文タイトル (documents.title) を JOIN で取得する。
    sql = f"""
        SELECT
            m.docid,
            m.filepath,
            d.title   AS title,
            c.body    AS body,
            bm25(docs_fts_morph) AS score
        FROM docs_fts_morph m
        JOIN documents d ON d.docid = m.docid AND d.active = 1
        JOIN content   c ON c.hash  = d.hash
        WHERE {where_sql}
        ORDER BY bm25(docs_fts_morph)
        {limit_sql}
    """

    terms = snippet_terms(query)
    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        score = -float(row["score"])
        if min_score is not None and score < min_score:
            continue
        body = row["body"] or ""
        results.append(
            SearchResult(
                docid=row["docid"],
                score=score,
                filepath=row["filepath"],
                title=row["title"] or "",
                snippet=extract_snippet(body, terms, max_chars=160),
                body=body,
            )
        )
    return results
