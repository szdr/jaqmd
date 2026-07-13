from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

from .paths import db_path

_SCHEMA = Path(__file__).parent / "schema.sql"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(conn)
    _load_vec(conn)  # 失敗しても握りつぶす（trigram/morph は動作継続）
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text())


def _load_vec(conn: sqlite3.Connection) -> bool:
    """sqlite-vec 拡張をロードし、vectors_vec テーブルを作成する。
    拡張が利用できない環境ではサイレントに失敗して False を返す。
    """
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec "
            "USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[768])"
        )
        return True
    except Exception:
        return False


def vec_available(conn: sqlite3.Connection) -> bool:
    """vectors_vec テーブルが利用可能かどうかを確認する。"""
    try:
        conn.execute("SELECT 1 FROM vectors_vec LIMIT 0")
        return True
    except Exception:
        return False


# --- collections ---


def add_collection(
    conn: sqlite3.Connection,
    name: str,
    path: str,
    glob_mask: str = "**/*.md",
) -> None:
    conn.execute(
        "INSERT INTO collections(name, path, glob_mask) VALUES (?, ?, ?)",
        (name, path, glob_mask),
    )
    conn.commit()


def list_collections(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM collections ORDER BY name").fetchall()


def get_collection(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()


def remove_collection(conn: sqlite3.Connection, name: str) -> None:
    # documents 削除前に FTS / ベクトルデータを一括削除しておく。
    # documents_delete トリガーは行単位で docid = ? の DELETE を発行するため、
    # 先に空にしておかないと UNINDEXED な docid 列の全表スキャンが件数分繰り返され
    # O(N^2) になってしまう。
    conn.execute(
        """DELETE FROM docs_fts_trigram WHERE docid IN (
               SELECT docid FROM documents WHERE collection = ?
           )""",
        (name,),
    )
    conn.execute(
        """DELETE FROM docs_fts_morph WHERE docid IN (
               SELECT docid FROM documents WHERE collection = ?
           )""",
        (name,),
    )
    if vec_available(conn):
        conn.execute(
            """DELETE FROM vectors_vec WHERE chunk_id IN (
                   SELECT cv.id FROM chunk_vectors cv
                     JOIN documents d ON cv.docid = d.docid
                    WHERE d.collection = ?
               )""",
            (name,),
        )
    conn.execute(
        """DELETE FROM chunk_vectors WHERE docid IN (
               SELECT docid FROM documents WHERE collection = ?
           )""",
        (name,),
    )
    conn.execute("DELETE FROM documents WHERE collection = ?", (name,))
    conn.execute("DELETE FROM collections WHERE name = ?", (name,))
    conn.commit()


# --- documents ---


def _make_docid(hash_: str, conn: sqlite3.Connection) -> str:
    """hash プレフィックスから重複しない docid を生成する。"""
    for length in range(6, len(hash_) + 1):
        candidate = hash_[:length]
        existing = conn.execute(
            "SELECT 1 FROM documents WHERE docid = ?", (candidate,)
        ).fetchone()
        if existing is None:
            return candidate
    return hash_


def upsert_document(
    conn: sqlite3.Connection,
    *,
    collection: str,
    path: str,
    body: str,
    title: Optional[str],
    mtime: int,
) -> str:
    """ドキュメントを upsert して docid を返す。commit は呼び出し側の責務。"""
    hash_ = hashlib.sha256(body.encode()).hexdigest()

    conn.execute(
        "INSERT OR IGNORE INTO content(hash, body) VALUES (?, ?)", (hash_, body)
    )

    existing = conn.execute(
        "SELECT id, hash, docid, active FROM documents WHERE collection = ? AND path = ?",
        (collection, path),
    ).fetchone()

    if existing is None:
        docid = _make_docid(hash_, conn)
        conn.execute(
            """INSERT INTO documents(collection, path, hash, docid, title, mtime, active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (collection, path, hash_, docid, title, mtime),
        )
        return docid

    if existing["hash"] != hash_ or existing["active"] == 0:
        new_docid = (
            _make_docid(hash_, conn) if existing["hash"] != hash_ else existing["docid"]
        )
        conn.execute(
            """UPDATE documents
               SET hash = ?, docid = ?, title = ?, mtime = ?, active = 1,
                   indexed_at = unixepoch()
               WHERE id = ?""",
            (hash_, new_docid, title, mtime, existing["id"]),
        )
        return new_docid

    conn.execute(
        "UPDATE documents SET title = ?, mtime = ? WHERE id = ?",
        (title, mtime, existing["id"]),
    )
    return existing["docid"]


def soft_delete_path(conn: sqlite3.Connection, collection: str, path: str) -> None:
    conn.execute(
        "UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1",
        (collection, path),
    )


def list_active_paths(conn: sqlite3.Connection, collection: str) -> set[str]:
    rows = conn.execute(
        "SELECT path FROM documents WHERE collection = ? AND active = 1", (collection,)
    ).fetchall()
    return {r["path"] for r in rows}


def get_document(conn: sqlite3.Connection, ref: str) -> Optional[sqlite3.Row]:
    """docid またはパスでドキュメントを1件取得する。

    docid は先頭に "#" を付けずに指定する仕様だが、誤って付与された場合も
    救済できるよう剥がしてから照合する。
    """
    ref = ref.removeprefix("#")
    row = conn.execute(
        """SELECT d.id, d.docid, d.collection, d.path, d.title, c.body
           FROM documents d JOIN content c ON d.hash = c.hash
           WHERE d.docid = ? AND d.active = 1""",
        (ref,),
    ).fetchone()
    if row is not None:
        return row
    return conn.execute(
        """SELECT d.id, d.docid, d.collection, d.path, d.title, c.body
           FROM documents d JOIN content c ON d.hash = c.hash
           WHERE d.path = ? AND d.active = 1
           LIMIT 1""",
        (ref,),
    ).fetchone()


# --- index_meta ---


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """INSERT INTO index_meta(key, value) VALUES (?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        (key, value),
    )


# --- qe_cache ---


def get_qe_cache(
    conn: sqlite3.Connection, query_hash: str, model_id: str
) -> Optional[sqlite3.Row]:
    """有効期限内（created_at + ttl > now）の QE キャッシュを取得する。"""
    return conn.execute(
        """SELECT * FROM qe_cache
           WHERE query_hash = ? AND model_id = ?
             AND (created_at + ttl) > unixepoch()""",
        (query_hash, model_id),
    ).fetchone()


def set_qe_cache(
    conn: sqlite3.Connection,
    query_hash: str,
    query_raw: str,
    lex_query: Optional[str],
    vec_query: Optional[str],
    hyde_text: Optional[str],
    model_id: str,
) -> None:
    conn.execute(
        """INSERT INTO qe_cache(query_hash, query_raw, lex_query, vec_query, hyde_text, model_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, unixepoch())
           ON CONFLICT(query_hash) DO UPDATE SET
               query_raw  = excluded.query_raw,
               lex_query  = excluded.lex_query,
               vec_query  = excluded.vec_query,
               hyde_text  = excluded.hyde_text,
               model_id   = excluded.model_id,
               created_at = excluded.created_at""",
        (query_hash, query_raw, lex_query, vec_query, hyde_text, model_id),
    )


def get_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM documents WHERE active = 1").fetchone()[
        0
    ]
    trigram_count = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]
    collections = list_collections(conn)
    morph_indexed = get_meta(conn, "morph_indexed") == "1"
    vec_indexed = get_meta(conn, "vec_indexed") == "1"
    return {
        "total": total,
        "trigram": trigram_count,
        "morph_indexed": morph_indexed,
        "vec_indexed": vec_indexed,
        "collections": collections,
    }
