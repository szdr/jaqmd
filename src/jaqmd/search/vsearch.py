from __future__ import annotations

import sqlite3
from typing import Optional

from ..config import settings
from ..progress import NULL_REPORTER, ProgressReporter
from .snippet import extract_snippet
from .trisearch import SearchResult


def vsearch(
    conn: sqlite3.Connection,
    query: str,
    *,
    n: int = 5,
    collection: Optional[str] = None,
    min_score: Optional[float] = None,
    all_results: bool = False,
    snippet_chars: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None,
) -> list[SearchResult]:
    """ベクトル KNN 検索を実行する。ドキュメント単位（最良チャンク）で結果を返す。

    score は cosine 類似度近似値（高いほど良い）。
    """
    reporter = reporter or NULL_REPORTER
    if snippet_chars is None:
        snippet_chars = settings.search_snippet_chars
    if not query.strip():
        return []

    try:
        import sqlite_vec

        from ..embed import embed_query
    except ImportError as e:
        raise RuntimeError(
            "ベクトル検索に必要なライブラリが見つかりません: " + str(e)
        ) from e

    # vectors_vec テーブルの存在確認（sqlite-vec 拡張がロードされているか）
    try:
        conn.execute("SELECT 1 FROM vectors_vec LIMIT 0")
    except Exception as e:
        raise RuntimeError(
            "ベクトルインデックスが利用できません。\n"
            "sqlite-vec 拡張のロードに失敗している可能性があります。"
        ) from e

    vec = embed_query(query, reporter=reporter)
    vec_bytes = sqlite_vec.serialize_float32(vec)

    # 集約前に多めに取得（collection フィルタ後に n 件確保するため）
    k = 500 if all_results else max(n * 5, 50)

    with reporter.step("ベクトル検索"):
        knn_rows = conn.execute(
            "SELECT chunk_id, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_bytes, k),
        ).fetchall()

    if not knn_rows:
        return []

    chunk_ids = [r["chunk_id"] for r in knn_rows]
    distances = {r["chunk_id"]: r["distance"] for r in knn_rows}

    # chunk_vectors → documents（active=1）→ content を JOIN して情報取得
    placeholders = ",".join("?" * len(chunk_ids))
    collection_clause = "AND d.collection = ?" if collection else ""
    sql = f"""
        SELECT
            cv.id        AS chunk_id,
            cv.docid,
            cv.chunk_text,
            d.collection || '/' || d.path AS filepath,
            d.title,
            c.body
        FROM chunk_vectors cv
        JOIN documents d ON d.docid = cv.docid AND d.active = 1
        JOIN content   c ON c.hash  = d.hash
        WHERE cv.id IN ({placeholders})
        {collection_clause}
    """
    params = chunk_ids + ([collection] if collection else [])
    rows = conn.execute(sql, params).fetchall()
    row_map = {r["chunk_id"]: r for r in rows}

    # KNN 距離昇順（= 類似度降順）で走査し docid 初出のみ採用（最良チャンク）
    seen_docids: set[str] = set()
    results: list[SearchResult] = []

    for knn_row in knn_rows:
        cid = knn_row["chunk_id"]
        row = row_map.get(cid)
        if row is None:
            continue  # active=1 でない or collection フィルタ外
        docid = row["docid"]
        if docid in seen_docids:
            continue
        seen_docids.add(docid)

        # 正規化 embedding（normalization=True）では距離 ∈ [0, 2]
        # score = 1 - distance/2 ≈ cosine 類似度（1が完全一致、0が直交）
        distance = distances[cid]
        score = 1.0 - distance / 2.0

        if min_score is not None and score < min_score:
            continue

        results.append(
            SearchResult(
                docid=docid,
                score=score,
                filepath=row["filepath"],
                title=row["title"] or "",
                snippet=extract_snippet(
                    row["chunk_text"] or "", query.split(), max_chars=snippet_chars
                ),
                body=row["body"] or "",
            )
        )

        if not all_results and len(results) >= n:
            break

    return results
