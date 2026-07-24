from __future__ import annotations

import sqlite3
from typing import Literal, Optional

from ..progress import NULL_REPORTER
from ..search.query import query_searches
from ..search.trisearch import SearchResult
from ..store import connect, find_documents_glob, get_document, get_meta, get_stats

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
        min_score: ブレンド後の絶対スコアに対する閾値（min-max 正規化は行わない）。
            0.0 で足切りなし。
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


def run_multi_get(conn: sqlite3.Connection, pattern: str) -> dict:
    """glob パターンまたはカンマ区切り docid/パスで複数ドキュメントを取得する。

    Returns:
        `{"results": [...], "not_found": [...]}`。カンマ区切り指定で解決でき
        なかった参照は not_found に入力順で入る。glob モードのマッチ0件は
        「参照の未解決」ではなく正当な空集合なので not_found は常に空。
    """
    if "," in pattern:
        results = []
        not_found = []
        for ref in (r.strip() for r in pattern.split(",")):
            row = get_document(conn, ref.split(":")[0])
            if row is None:
                not_found.append(ref)
                continue
            results.append(
                {
                    "docid": row["docid"],
                    "collection": row["collection"],
                    "path": row["path"],
                    "title": row["title"],
                    "body": row["body"],
                }
            )
        return {"results": results, "not_found": not_found}

    rows = find_documents_glob(conn, pattern)
    return {
        "results": [
            {
                "docid": row["docid"],
                "collection": row["collection"],
                "path": row["path"],
                "title": row["title"],
                "body": row["body"],
            }
            for row in rows
        ],
        "not_found": [],
    }


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
            "score は RRF 融合順位の逆数と reranker スコアの位置依存加重和による"
            "絶対スコア（正規化なし）。1位 ≈ 0.8〜1.0、2位 ≈ 0.4〜0.6、"
            "10位前後 ≈ 0.1〜0.2 と順位に応じて急減衰するため、"
            "同一レスポンス内の相対的な確からしさとして解釈し、"
            "別クエリのスコアと比較しないこと。"
            "足切りの目安: minScore=0.3 で概ね上位2〜3件、0.15 で上位7件程度。"
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

    @mcp.tool(
        description=(
            "パス・docid でドキュメントを1件取得する。"
            "file には docid、path、または query が返す filepath"
            "（`collection/path` 形式）をそのまま指定できる。"
            "docid は query/multi_get のレスポンスに含まれる `docid` の値をそのまま指定する"
            "（例: `abc123`。先頭に `#` は付けない）。"
        )
    )
    def get(file: str) -> dict:
        conn = connect()
        try:
            return run_get(conn, file)
        finally:
            conn.close()

    @mcp.tool(
        description=(
            "glob パターンまたはカンマ区切りのパス/docid で複数ドキュメントを取得する。"
            "パスは query が返す filepath（`collection/path` 形式）をそのまま指定できる。"
            "glob は path と `collection/path` の両方に対して照合する。"
            '返り値は `{"results": [...], "not_found": [...]}` で、'
            "カンマ区切り指定で解決できなかった参照は not_found に入る"
            "（glob でマッチ0件の場合は results が空になるだけで not_found は常に空）。"
        )
    )
    def multi_get(pattern: str) -> dict:
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
