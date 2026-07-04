import sqlite3

import pytest
from jaqmd.store import (
    _SCHEMA,
    add_collection,
    get_meta,
    list_active_paths,
    remove_collection,
    set_meta,
    soft_delete_path,
    upsert_document,
)


def test_content_addressable(conn, doc_dir):
    """同一本文は content テーブルに1行だけ登録される。"""
    add_collection(conn, "test", str(doc_dir))
    body = "テストコンテンツ本文"
    upsert_document(conn, collection="test", path="a.md", body=body, title="A", mtime=1000)
    upsert_document(conn, collection="test", path="b.md", body=body, title="B", mtime=1001)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    assert count == 1

    doc_count = conn.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0]
    assert doc_count == 2


def test_docid_generated(conn, doc_dir):
    """docid は hash の先頭6文字で生成される。"""
    add_collection(conn, "test", str(doc_dir))
    docid = upsert_document(conn, collection="test", path="a.md", body="内容A", title="A", mtime=1000)
    conn.commit()

    assert len(docid) >= 6
    row = conn.execute("SELECT docid FROM documents WHERE path='a.md'").fetchone()
    assert row["docid"] == docid


def test_soft_delete(conn, doc_dir):
    """ファイル削除時は active=0 になる。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(conn, collection="test", path="a.md", body="内容", title="A", mtime=1000)
    conn.commit()

    soft_delete_path(conn, "test", "a.md")
    conn.commit()

    row = conn.execute("SELECT active FROM documents WHERE path='a.md'").fetchone()
    assert row["active"] == 0


def test_soft_delete_removes_from_fts(conn, doc_dir):
    """論理削除で trigram FTS からもエントリが削除される。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(
        conn, collection="test", path="a.md",
        body="日本語検索エンジンの実装", title="A", mtime=1000,
    )
    conn.commit()

    fts_before = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]
    soft_delete_path(conn, "test", "a.md")
    conn.commit()
    fts_after = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]

    assert fts_before == 1
    assert fts_after == 0


def test_hash_change_updates_docid(conn, doc_dir):
    """本文変更で docid が更新され、FTS も更新される。"""
    add_collection(conn, "test", str(doc_dir))
    docid1 = upsert_document(conn, collection="test", path="a.md", body="内容A", title="A", mtime=1000)
    conn.commit()

    docid2 = upsert_document(conn, collection="test", path="a.md", body="内容Bに変更", title="A", mtime=1001)
    conn.commit()

    assert docid1 != docid2
    # active なドキュメントは 1 件のみ
    count = conn.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0]
    assert count == 1
    # FTS も 1 件
    fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]
    assert fts_count == 1


def test_reactivate(conn, doc_dir):
    """soft delete 後に同じファイルが戻ると active=1 に再活性化される。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(conn, collection="test", path="a.md", body="内容", title="A", mtime=1000)
    conn.commit()
    soft_delete_path(conn, "test", "a.md")
    conn.commit()

    upsert_document(conn, collection="test", path="a.md", body="内容", title="A", mtime=1002)
    conn.commit()

    row = conn.execute("SELECT active FROM documents WHERE path='a.md'").fetchone()
    assert row["active"] == 1
    fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]
    assert fts_count == 1


def test_remove_collection_cascades(conn, doc_dir):
    """コレクション削除でドキュメントと FTS・ベクトルが全て削除される。"""
    add_collection(conn, "test", str(doc_dir))
    docid_a = upsert_document(conn, collection="test", path="a.md", body="内容A", title="A", mtime=1000)
    docid_b = upsert_document(conn, collection="test", path="b.md", body="内容B", title="B", mtime=1001)
    conn.execute(
        "INSERT INTO docs_fts_morph(docid, filepath, title, body) VALUES (?, ?, ?, ?)",
        (docid_a, "test/a.md", "A", "内容A"),
    )
    conn.execute(
        "INSERT INTO docs_fts_morph(docid, filepath, title, body) VALUES (?, ?, ?, ?)",
        (docid_b, "test/b.md", "B", "内容B"),
    )
    doc_row = conn.execute(
        "SELECT id FROM documents WHERE docid = ?", (docid_a,)
    ).fetchone()
    conn.execute(
        """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
           VALUES (?, ?, 0, 0, ?, 'test-model')""",
        (doc_row["id"], docid_a, "内容A"),
    )
    conn.commit()

    remove_collection(conn, "test")

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    fts_trigram_count = conn.execute("SELECT COUNT(*) FROM docs_fts_trigram").fetchone()[0]
    fts_morph_count = conn.execute("SELECT COUNT(*) FROM docs_fts_morph").fetchone()[0]
    chunk_vectors_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    assert doc_count == 0
    assert fts_trigram_count == 0
    assert fts_morph_count == 0
    assert chunk_vectors_count == 0


def test_index_meta(conn):
    """index_meta の get/set が正しく動作する。"""
    set_meta(conn, "trigram_indexed", "1")
    conn.commit()
    assert get_meta(conn, "trigram_indexed") == "1"
    assert get_meta(conn, "nonexistent") is None

    set_meta(conn, "trigram_indexed", "0")
    conn.commit()
    assert get_meta(conn, "trigram_indexed") == "0"


def test_idempotent_schema_on_legacy_db(tmp_cache, monkeypatch):
    """旧バージョンDB（trigram のみ）に再接続すると新テーブルが自動作成される。"""
    from jaqmd.paths import db_path

    # 旧スキーマ（docs_fts_trigram のみ・index_meta なし）を手動で作成
    legacy_sql = """
        CREATE TABLE content (hash TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, collection TEXT NOT NULL, path TEXT NOT NULL,
            hash TEXT NOT NULL, docid TEXT UNIQUE NOT NULL, title TEXT,
            mtime INTEGER, active INTEGER DEFAULT 1, indexed_at INTEGER DEFAULT (unixepoch())
        );
        CREATE VIRTUAL TABLE docs_fts_trigram USING fts5(
            docid UNINDEXED, filepath UNINDEXED, title, body, tokenize = 'trigram'
        );
    """
    db = db_path()
    legacy_conn = sqlite3.connect(db)
    legacy_conn.executescript(legacy_sql)
    legacy_conn.close()

    # 新しい connect() で接続 → schema.sql が冪等に適用される
    from jaqmd.store import connect
    conn = connect()

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "index_meta" in tables
    assert "collections" in tables
    assert "path_contexts" in tables

    triggers = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    assert "documents_ai" in triggers
    assert "documents_soft_delete" in triggers

    conn.close()

    # 2回目の接続でもエラーにならない（冪等性）
    conn2 = connect()
    conn2.close()


def test_list_active_paths(conn, doc_dir):
    """list_active_paths は active=1 のパスのみ返す。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(conn, collection="test", path="a.md", body="内容A", title="A", mtime=1000)
    upsert_document(conn, collection="test", path="b.md", body="内容B", title="B", mtime=1001)
    conn.commit()
    soft_delete_path(conn, "test", "b.md")
    conn.commit()

    paths = list_active_paths(conn, "test")
    assert paths == {"a.md"}
