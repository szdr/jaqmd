from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .format import format_results
from .scan import scan_collection
from .search.trisearch import trisearch as do_trisearch
from .search.mosearch import mosearch as do_mosearch
from .search.vsearch import vsearch as do_vsearch
from .search.query import query as do_query
from .store import (
    add_collection,
    connect,
    get_document,
    get_meta,
    get_stats,
    list_active_paths,
    list_collections,
    remove_collection,
    set_meta,
    soft_delete_path,
    upsert_document,
)

app = typer.Typer(help="jaqmd — 日本語ドキュメント検索エンジン", no_args_is_help=True)
collection_app = typer.Typer(no_args_is_help=True)
app.add_typer(collection_app, name="collection", help="コレクションの管理")


def main() -> None:
    app()


# ---------------------------------------------------------------------------
# collection サブコマンド
# ---------------------------------------------------------------------------

@collection_app.command("add")
def collection_add(
    path: str = typer.Argument(..., help="コレクションのディレクトリパス"),
    name: str = typer.Option(..., "--name", "-n", help="コレクション名"),
    glob: str = typer.Option("**/*.md", "--glob", help="glob パターン"),
) -> None:
    """コレクションを追加します。"""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        typer.echo(f"エラー: ディレクトリが存在しません: {path}", err=True)
        raise typer.Exit(1)
    conn = connect()
    try:
        add_collection(conn, name, str(p), glob)
    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"コレクションを追加しました: {name} → {p}")


@collection_app.command("list")
def collection_list() -> None:
    """コレクション一覧を表示します。"""
    conn = connect()
    rows = list_collections(conn)
    if not rows:
        typer.echo("コレクションがありません。")
        return
    for r in rows:
        typer.echo(f"  {r['name']:<20} {r['path']}  ({r['glob_mask']})")


@collection_app.command("remove")
def collection_remove(
    name: str = typer.Argument(..., help="削除するコレクション名"),
) -> None:
    """コレクションを削除します。"""
    conn = connect()
    remove_collection(conn, name)
    typer.echo(f"コレクションを削除しました: {name}")


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

@app.command()
def update(
    pull: bool = typer.Option(False, "--pull", help="（予約）"),
) -> None:
    """ファイルをスキャンして trigram FTS インデックスを構築します。"""
    conn = connect()
    collections = list_collections(conn)
    if not collections:
        typer.echo(
            "コレクションがありません。先に `jaqmd collection add` を実行してください。"
        )
        raise typer.Exit(1)

    total_added = 0
    total_deleted = 0

    for col in collections:
        name = col["name"]
        col_path = col["path"]
        glob_mask = col["glob_mask"]

        if not Path(col_path).is_dir():
            typer.echo(f"警告: コレクションパスが見つかりません（スキップ）: {col_path}", err=True)
            continue

        typer.echo(f"スキャン中: {name} ({col_path})")
        existing_paths = list_active_paths(conn, name)
        scanned_paths: set[str] = set()

        for f in scan_collection(col_path, glob_mask):
            upsert_document(
                conn,
                collection=name,
                path=f["path"],
                body=f["body"],
                title=f["title"],
                mtime=f["mtime"],
            )
            scanned_paths.add(f["path"])
            total_added += 1

        for gone in existing_paths - scanned_paths:
            soft_delete_path(conn, name, gone)
            total_deleted += 1

        conn.commit()

    set_meta(conn, "trigram_indexed", "1")
    conn.commit()

    typer.echo(f"完了: {total_added} ファイル処理、{total_deleted} ファイル削除")


# ---------------------------------------------------------------------------
# 検索コマンド共通ロジック
# ---------------------------------------------------------------------------

def _run_search(
    query: str,
    *,
    n: int,
    collection: Optional[str],
    min_score: Optional[float],
    all_results: bool,
    full: bool,
    json_out: bool,
    md: bool,
    xml: bool,
    files: bool,
    search_fn=None,
    meta_key: str = "trigram_indexed",
    meta_missing_msg: str = (
        "エラー: trigram インデックスが構築されていません。\n"
        "→ `jaqmd update` を実行してください。"
    ),
) -> None:
    conn = connect()
    if get_meta(conn, meta_key) != "1":
        typer.echo(meta_missing_msg, err=True)
        raise typer.Exit(1)

    fn = search_fn or do_trisearch
    results = fn(
        conn, query, n=n, collection=collection,
        min_score=min_score, all_results=all_results,
    )

    fmt = "default"
    if json_out:
        fmt = "json"
    elif md:
        fmt = "md"
    elif xml:
        fmt = "xml"
    elif files:
        fmt = "files"

    typer.echo(format_results(results, fmt=fmt, full=full))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command(name="search")
def search_command(
    query: str = typer.Argument(..., help="検索クエリ"),
    n: int = typer.Option(5, "-n", help="結果件数"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="コレクション絞り込み"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help="スコア閾値"),
    all_results: bool = typer.Option(False, "--all", help="全件返却"),
    full: bool = typer.Option(False, "--full", help="全文表示"),
    json_out: bool = typer.Option(False, "--json", help="JSON 出力"),
    md: bool = typer.Option(False, "--md", help="Markdown 出力"),
    xml: bool = typer.Option(False, "--xml", help="XML 出力"),
    files: bool = typer.Option(False, "--files", help="files 形式出力"),
) -> None:
    """trigram BM25 検索を実行します。"""
    _run_search(
        query, n=n, collection=collection, min_score=min_score,
        all_results=all_results, full=full, json_out=json_out,
        md=md, xml=xml, files=files,
    )


# ---------------------------------------------------------------------------
# get / multi-get
# ---------------------------------------------------------------------------

@app.command(name="get")
def get_command(
    ref: str = typer.Argument(..., help="ファイルパスまたは docid"),
) -> None:
    """パスまたは docid でドキュメントを1件取得します。"""
    conn = connect()
    # :line サフィックスを除去
    ref = ref.split(":")[0]
    row = get_document(conn, ref)
    if row is None:
        typer.echo(f"エラー: ドキュメントが見つかりません: {ref}", err=True)
        raise typer.Exit(1)
    typer.echo(f"# {row['title'] or row['path']}")
    typer.echo(f"docid: {row['docid']}  path: {row['collection']}/{row['path']}")
    typer.echo("---")
    typer.echo(row["body"])


@app.command(name="multi-get")
def multi_get_command(
    pattern: str = typer.Argument(..., help="glob パターンまたはカンマ区切り docid"),
) -> None:
    """複数のドキュメントを取得します。"""
    conn = connect()
    if "," in pattern:
        for ref in (r.strip() for r in pattern.split(",")):
            row = get_document(conn, ref)
            if row is None:
                typer.echo(f"警告: 見つかりません: {ref}", err=True)
                continue
            typer.echo(f"--- {row['docid']} ---")
            typer.echo(row["body"])
        return

    rows = conn.execute(
        """SELECT d.docid, d.collection, d.path, d.title, c.body
           FROM documents d JOIN content c ON d.hash = c.hash
           WHERE d.path GLOB ? AND d.active = 1""",
        (pattern,),
    ).fetchall()
    if not rows:
        typer.echo(f"エラー: マッチするドキュメントが見つかりません: {pattern}", err=True)
        raise typer.Exit(1)
    for row in rows:
        typer.echo(f"--- {row['docid']} ---")
        typer.echo(row["body"])


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

@app.command(name="ls")
def ls_command(
    collection: Optional[str] = typer.Argument(None, help="コレクション名"),
) -> None:
    """コレクション内のファイル一覧を表示します。"""
    conn = connect()
    if collection:
        rows = conn.execute(
            """SELECT docid, path, title FROM documents
               WHERE collection = ? AND active = 1 ORDER BY path""",
            (collection,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT docid, collection, path, title FROM documents
               WHERE active = 1 ORDER BY collection, path"""
        ).fetchall()

    if not rows:
        typer.echo("ドキュメントがありません。")
        return

    for r in rows:
        prefix = r["path"] if collection else f"{r['collection']}/{r['path']}"
        typer.echo(f"  {r['docid']}  {prefix}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status() -> None:
    """インデックスの構築状態を表示します。"""
    conn = connect()
    stats = get_stats(conn)

    col_names = ", ".join(r["name"] for r in stats["collections"])
    col_count = len(stats["collections"])
    sep = "─" * 45

    typer.echo(f"Collections : {col_count} ({col_names})")
    typer.echo(f"Documents   : {stats['total']:,}")
    typer.echo(sep)

    trigram_ok = stats["trigram"] > 0 or stats["total"] == 0
    trigram_mark = "✓" if trigram_ok else "✗"
    typer.echo(f"trigram FTS : {trigram_mark} {stats['trigram']:,} docs    (jaqmd update)")

    morph_mark = "✓" if stats["morph_indexed"] else "✗"
    morph_hint = "" if stats["morph_indexed"] else "   → run: jaqmd morph"
    typer.echo(f"morph  FTS  : {morph_mark} {'indexed' if stats['morph_indexed'] else 'not indexed'}{morph_hint}")

    vec_mark = "✓" if stats["vec_indexed"] else "✗"
    vec_hint = "" if stats["vec_indexed"] else "   → run: jaqmd embed"
    typer.echo(f"vectors     : {vec_mark} {'indexed' if stats['vec_indexed'] else 'not indexed'}{vec_hint}")

    typer.echo(sep)

    available = ["search"]
    unavailable = []
    if stats["morph_indexed"]:
        available.append("mosearch")
    else:
        unavailable.append("mosearch")
    if stats["vec_indexed"]:
        available.append("vsearch")
    else:
        unavailable.append("vsearch")
        unavailable.append("query(full)")

    typer.echo(f"Available   : {', '.join(available)}")
    if unavailable:
        typer.echo(f"Unavailable : {', '.join(unavailable)}")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

@app.command()
def cleanup() -> None:
    """論理削除済みドキュメントを削除して DB を最適化します。"""
    conn = connect()
    deleted = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE active = 0"
    ).fetchone()[0]
    conn.execute("DELETE FROM documents WHERE active = 0")
    conn.commit()
    conn.execute("VACUUM")
    typer.echo(f"クリーンアップ完了: {deleted} 件の論理削除済みエントリを削除しました。")


# ---------------------------------------------------------------------------
# 未実装コマンド（次イテレーション）
# ---------------------------------------------------------------------------

@app.command()
def morph() -> None:
    """形態素解析インデックスを構築します。"""
    try:
        from .tokenize.morph import tokenize_text
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    conn = connect()
    if get_meta(conn, "trigram_indexed") != "1":
        typer.echo(
            "エラー: trigram インデックスが構築されていません。\n"
            "→ 先に `jaqmd update` を実行してください。",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("形態素インデックスを構築中...")

    # 既存エントリを削除してから再投入（冪等）
    conn.execute("DELETE FROM docs_fts_morph")

    rows = conn.execute(
        """SELECT d.docid, d.collection, d.path, d.title, c.body
           FROM documents d JOIN content c ON d.hash = c.hash
           WHERE d.active = 1"""
    ).fetchall()

    for row in rows:
        tokenized_title = tokenize_text(row["title"] or "")
        tokenized_body = tokenize_text(row["body"] or "")
        conn.execute(
            "INSERT INTO docs_fts_morph(docid, filepath, title, body) VALUES (?, ?, ?, ?)",
            (
                row["docid"],
                row["collection"] + "/" + row["path"],
                tokenized_title,
                tokenized_body,
            ),
        )

    set_meta(conn, "morph_indexed", "1")
    set_meta(conn, "morph_tokenizer", "sudachipy/normalized_form")
    conn.commit()

    typer.echo(f"完了: {len(rows)} 件の形態素インデックスを構築しました。")


@app.command()
def embed(
    force: bool = typer.Option(False, "-f", "--force", help="既存ベクトルを削除して全再構築"),
) -> None:
    """ベクトルインデックスを構築します。"""
    try:
        from .chunk import chunk_document
        from .embed import EMBED_DIM, EMBED_MODEL, count_tokens, embed_documents
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    conn = connect()

    from .store import vec_available
    if not vec_available(conn):
        typer.echo(
            "エラー: sqlite-vec 拡張をロードできません。\n"
            "→ `pip install sqlite-vec` またはビルド環境を確認してください。",
            err=True,
        )
        raise typer.Exit(1)

    if get_meta(conn, "trigram_indexed") != "1":
        typer.echo(
            "エラー: trigram インデックスが構築されていません。\n"
            "→ 先に `jaqmd update` を実行してください。",
            err=True,
        )
        raise typer.Exit(1)

    if force:
        typer.echo("既存ベクトルを削除して全再構築します...")
        conn.execute("DELETE FROM vectors_vec")
        conn.execute("DELETE FROM chunk_vectors")
        conn.commit()

    # 差分: chunk_vectors に未登録の active ドキュメントのみ処理
    if force:
        rows = conn.execute(
            """SELECT d.id, d.docid, d.collection, d.path, c.body
               FROM documents d JOIN content c ON d.hash = c.hash
               WHERE d.active = 1"""
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT d.id, d.docid, d.collection, d.path, c.body
               FROM documents d JOIN content c ON d.hash = c.hash
               LEFT JOIN chunk_vectors cv ON cv.docid = d.docid
               WHERE d.active = 1 AND cv.docid IS NULL"""
        ).fetchall()

    if not rows:
        typer.echo("新しくベクトル化するドキュメントはありません。")
        set_meta(conn, "vec_indexed", "1")
        set_meta(conn, "embed_model", EMBED_MODEL)
        set_meta(conn, "embed_dim", str(EMBED_DIM))
        conn.commit()
        return

    typer.echo(f"ベクトルインデックスを構築中... ({len(rows)} 件)")

    import sqlite_vec

    for doc in rows:
        body = doc["body"] or ""
        chunks = chunk_document(body, count_tokens=count_tokens)
        if not chunks:
            continue

        chunk_texts = [ct for _, _, ct in chunks]
        vectors = embed_documents(chunk_texts)

        for (chunk_seq, chunk_pos, chunk_text), vec in zip(chunks, vectors):
            cur = conn.execute(
                """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc["id"], doc["docid"], chunk_seq, chunk_pos, chunk_text, EMBED_MODEL),
            )
            chunk_id = cur.lastrowid
            conn.execute(
                "INSERT INTO vectors_vec(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vec)),
            )

        typer.echo(f"  {doc['collection']}/{doc['path']} ({len(chunks)} チャンク)")

    set_meta(conn, "vec_indexed", "1")
    set_meta(conn, "embed_model", EMBED_MODEL)
    set_meta(conn, "embed_dim", str(EMBED_DIM))
    conn.commit()

    typer.echo(f"完了: {len(rows)} 件のドキュメントをベクトル化しました。")


@app.command()
def mosearch(
    query: str = typer.Argument(..., help="検索クエリ"),
    n: int = typer.Option(5, "-n", help="結果件数"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="コレクション絞り込み"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help="スコア閾値"),
    all_results: bool = typer.Option(False, "--all", help="全件返却"),
    full: bool = typer.Option(False, "--full", help="全文表示"),
    json_out: bool = typer.Option(False, "--json", help="JSON 出力"),
    md: bool = typer.Option(False, "--md", help="Markdown 出力"),
    xml: bool = typer.Option(False, "--xml", help="XML 出力"),
    files: bool = typer.Option(False, "--files", help="files 形式出力"),
) -> None:
    """形態素 BM25 検索を実行します。"""
    _run_search(
        query, n=n, collection=collection, min_score=min_score,
        all_results=all_results, full=full, json_out=json_out,
        md=md, xml=xml, files=files,
        search_fn=do_mosearch,
        meta_key="morph_indexed",
        meta_missing_msg=(
            "エラー: 形態素インデックスが構築されていません。\n"
            "→ `jaqmd morph` を実行してください。"
        ),
    )


@app.command()
def vsearch(
    query: str = typer.Argument(..., help="検索クエリ"),
    n: int = typer.Option(5, "-n", help="結果件数"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="コレクション絞り込み"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help="スコア閾値"),
    all_results: bool = typer.Option(False, "--all", help="全件返却"),
    full: bool = typer.Option(False, "--full", help="全文表示"),
    json_out: bool = typer.Option(False, "--json", help="JSON 出力"),
    md: bool = typer.Option(False, "--md", help="Markdown 出力"),
    xml: bool = typer.Option(False, "--xml", help="XML 出力"),
    files: bool = typer.Option(False, "--files", help="files 形式出力"),
) -> None:
    """ベクトル意味検索を実行します。"""
    _run_search(
        query, n=n, collection=collection, min_score=min_score,
        all_results=all_results, full=full, json_out=json_out,
        md=md, xml=xml, files=files,
        search_fn=do_vsearch,
        meta_key="vec_indexed",
        meta_missing_msg=(
            "エラー: ベクトルインデックスが構築されていません。\n"
            "→ `jaqmd embed` を実行してください。"
        ),
    )


@app.command()
def query(
    q: str = typer.Argument(..., help="検索クエリ"),
    n: int = typer.Option(5, "-n", help="結果件数"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="コレクション絞り込み"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help="スコア閾値"),
    all_results: bool = typer.Option(False, "--all", help="全件返却"),
    full: bool = typer.Option(False, "--full", help="全文表示"),
    json_out: bool = typer.Option(False, "--json", help="JSON 出力"),
    md: bool = typer.Option(False, "--md", help="Markdown 出力"),
    xml: bool = typer.Option(False, "--xml", help="XML 出力"),
    files: bool = typer.Option(False, "--files", help="files 形式出力"),
) -> None:
    """ハイブリッド検索（RRF 融合: trigram / morph / vector）。"""
    _run_search(
        q, n=n, collection=collection, min_score=min_score,
        all_results=all_results, full=full, json_out=json_out,
        md=md, xml=xml, files=files,
        search_fn=do_query,
        meta_key="trigram_indexed",
        meta_missing_msg=(
            "エラー: trigram インデックスが構築されていません。\n"
            "→ `jaqmd update` を実行してください。"
        ),
    )


@app.command()
def mcp(
    http: bool = typer.Option(False, "--http", help="HTTP モード"),
) -> None:
    """MCP サーバーを起動します（次イテレーション対応予定）。"""
    typer.echo(
        "エラー: MCP サーバー機能は次イテレーション対応予定です。",
        err=True,
    )
    raise typer.Exit(1)
