from __future__ import annotations

import sqlite3
from typing import Literal, Optional

from ..progress import NULL_REPORTER
from ..search.query import query_searches
from ..search.trisearch import SearchResult
from ..store import connect, get_document, get_meta, get_stats

# ---------------------------------------------------------------------------
# 純粋ロジック関数（conn と検証済み引数を受け、JSON-serializable な dict/list を返す）
#
# FastMCP のツールハンドラから薄くラップして呼び出す。サーバーを起動せずに
# 単体テストできるよう、MCP SDK には依存しない形にしている。
# ---------------------------------------------------------------------------


def _result_to_dict(r: SearchResult) -> dict:
    return {
        "docid": r.docid,
        "score": r.score,
        "filepath": r.filepath,
        "title": r.title,
        "snippet": r.snippet,
    }


def run_query(
    conn: sqlite3.Connection,
    searches: list[tuple[str, str]],
    *,
    collections: Optional[list[str]] = None,
    limit: int = 10,
    min_score: float = 0.0,
    candidate_limit: int = 40,
    rerank: bool = True,
) -> list[dict]:
    """ハイブリッド検索（RRF 融合: trigram / morph / vector + rerank）を実行する。

    Args:
        searches: `(type, text)` のリスト（type は "lex" / "vec" / "hyde"）。1〜10 件。
            先頭の search は融合時に 2 倍の重みを持つ。
        collections: 絞り込むコレクション名（OR）。None で全コレクション。
        limit: 返却件数。
        min_score: 0-1 正規化後のスコア閾値。
        candidate_limit: reranker に渡す融合プールの上位候補数。
        rerank: False なら reranker を無効化。

    Returns:
        docid/score/filepath/title/snippet を持つ dict のリスト（スコア降順）。

    Raises:
        ValueError: searches が空/11件以上、type が不正、trigram インデックス未構築。
    """
    if get_meta(conn, "trigram_indexed") != "1":
        raise ValueError(
            "trigram インデックスが構築されていません。`jaqmd update` を実行してください。"
        )
    if not (1 <= len(searches) <= 10):
        raise ValueError("searches は 1〜10 件で指定してください。")

    results = query_searches(
        conn,
        searches,
        collections=collections,
        limit=limit,
        min_score=min_score,
        candidate_limit=candidate_limit,
        rerank_enabled=rerank,
        reporter=NULL_REPORTER,
    )
    return [_result_to_dict(r) for r in results]


def run_get(conn: sqlite3.Connection, file: str) -> dict:
    """パス・docid（`:line` サフィックスは無視）でドキュメントを1件取得する。

    Raises:
        ValueError: ドキュメントが見つからない場合。
    """
    ref = file.split(":")[0]
    row = get_document(conn, ref)
    if row is None:
        raise ValueError(f"ドキュメントが見つかりません: {file}")
    return {
        "docid": row["docid"],
        "collection": row["collection"],
        "path": row["path"],
        "title": row["title"],
        "body": row["body"],
    }


def run_multi_get(conn: sqlite3.Connection, pattern: str) -> list[dict]:
    """glob パターンまたはカンマ区切り docid/パスで複数ドキュメントを取得する。

    見つからない参照はスキップする（`jaqmd multi-get` の挙動を踏襲）。
    """
    if "," in pattern:
        out = []
        for ref in (r.strip() for r in pattern.split(",")):
            row = get_document(conn, ref)
            if row is None:
                continue
            out.append(
                {
                    "docid": row["docid"],
                    "collection": row["collection"],
                    "path": row["path"],
                    "title": row["title"],
                    "body": row["body"],
                }
            )
        return out

    rows = conn.execute(
        """SELECT d.docid, d.collection, d.path, d.title, c.body
           FROM documents d JOIN content c ON d.hash = c.hash
           WHERE d.path GLOB ? AND d.active = 1""",
        (pattern,),
    ).fetchall()
    return [
        {
            "docid": row["docid"],
            "collection": row["collection"],
            "path": row["path"],
            "title": row["title"],
            "body": row["body"],
        }
        for row in rows
    ]


def run_status(conn: sqlite3.Connection) -> dict:
    """インデックスの構築状態・コレクション一覧を返す。"""
    stats = get_stats(conn)

    collections = []
    for col in stats["collections"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE collection = ? AND active = 1",
            (col["name"],),
        ).fetchone()[0]
        collections.append(
            {"name": col["name"], "path": col["path"], "documents": count}
        )

    available = ["search"]
    if stats["morph_indexed"]:
        available.append("mosearch")
    if stats["vec_indexed"]:
        available.append("vsearch")
    available.append("query")

    return {
        "collections": collections,
        "total_documents": stats["total"],
        "trigram_count": stats["trigram"],
        "morph_indexed": stats["morph_indexed"],
        "vec_indexed": stats["vec_indexed"],
        "available": available,
    }


# ---------------------------------------------------------------------------
# FastMCP 登録
# ---------------------------------------------------------------------------


def build_server():
    """FastMCP サーバーを構築し、query/get/multi_get/status の4ツールを登録して返す。"""
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel, Field

    mcp = FastMCP(
        "jaqmd",
        instructions="日本語ドキュメント検索エンジン jaqmd の MCP サーバー（tobi/qmd 準拠）。",
    )

    # `from __future__ import annotations` によりツール関数の型注釈は文字列として
    # 遅延評価される。FastMCP はそれをこのモジュールの __globals__ に対して eval する
    # ため、ローカル変数のままでは見えない。global 宣言でモジュール名前空間に置く。
    global SearchItem

    class SearchItem(BaseModel):
        type: Literal["lex", "vec", "hyde"] = Field(
            description=(
                "サブクエリの種類。lex: BM25 語彙検索（trigram + 形態素）、"
                "vec: ベクトル意味検索、hyde: 仮想文書によるベクトル検索。"
            )
        )
        text: str = Field(description="サブクエリの文字列。")

    @mcp.tool(
        description=(
            "typed searches（lex/vec/hyde の型付きサブクエリ配列）による"
            "ハイブリッド検索（RRF 融合 + rerank）を実行する。"
        )
    )
    def query(
        searches: list[SearchItem],
        collections: Optional[list[str]] = None,
        limit: int = 10,
        minScore: float = 0.0,
        candidateLimit: int = 40,
        rerank: bool = True,
    ) -> list[dict]:
        conn = connect()
        try:
            return run_query(
                conn,
                [(s.type, s.text) for s in searches],
                collections=collections,
                limit=limit,
                min_score=minScore,
                candidate_limit=candidateLimit,
                rerank=rerank,
            )
        finally:
            conn.close()

    @mcp.tool(description="パス・docid（`#abc123` 形式）でドキュメントを1件取得する。")
    def get(file: str) -> dict:
        conn = connect()
        try:
            return run_get(conn, file)
        finally:
            conn.close()

    @mcp.tool(
        description="glob パターンまたはカンマ区切りのパス/docid で複数ドキュメントを取得する。"
    )
    def multi_get(pattern: str) -> list[dict]:
        conn = connect()
        try:
            return run_multi_get(conn, pattern)
        finally:
            conn.close()

    @mcp.tool(description="インデックスの構築状態・コレクション一覧を取得する。")
    def status() -> dict:
        conn = connect()
        try:
            return run_status(conn)
        finally:
            conn.close()

    return mcp


def serve_stdio() -> None:
    """MCP サーバーを stdio トランスポートで起動する（ブロッキング）。"""
    build_server().run(transport="stdio")
