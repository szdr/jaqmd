import pytest
from jaqmd.store import (
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
    """論理削除で bigram FTS からもエントリが削除される。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(
        conn, collection="test", path="a.md",
        body="日本語検索エンジンの実装", title="A", mtime=1000,
    )
    conn.commit()

    fts_before = conn.execute("SELECT COUNT(*) FROM docs_fts_bigram").fetchone()[0]
    soft_delete_path(conn, "test", "a.md")
    conn.commit()
    fts_after = conn.execute("SELECT COUNT(*) FROM docs_fts_bigram").fetchone()[0]

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
    fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts_bigram").fetchone()[0]
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
    fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts_bigram").fetchone()[0]
    assert fts_count == 1


def test_remove_collection_cascades(conn, doc_dir):
    """コレクション削除でドキュメントと FTS が全て削除される。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(conn, collection="test", path="a.md", body="内容A", title="A", mtime=1000)
    upsert_document(conn, collection="test", path="b.md", body="内容B", title="B", mtime=1001)
    conn.commit()

    remove_collection(conn, "test")

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM docs_fts_bigram").fetchone()[0]
    assert doc_count == 0
    assert fts_count == 0


def test_index_meta(conn):
    """index_meta の get/set が正しく動作する。"""
    set_meta(conn, "bigram_indexed", "1")
    conn.commit()
    assert get_meta(conn, "bigram_indexed") == "1"
    assert get_meta(conn, "nonexistent") is None

    set_meta(conn, "bigram_indexed", "0")
    conn.commit()
    assert get_meta(conn, "bigram_indexed") == "0"


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
