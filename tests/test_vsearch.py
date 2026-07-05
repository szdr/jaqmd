"""vsearch の単体テスト — sqlite-vec に既知ベクトルを直接 INSERT してモデルなしで検証。"""

from __future__ import annotations

import pytest

sqlite_vec = pytest.importorskip("sqlite_vec")

from jaqmd.store import add_collection, set_meta, upsert_document, vec_available
from jaqmd.search.vsearch import vsearch


def _insert_vec(conn, collection, path, title, body, vector):
    """テスト用: ドキュメントを upsert して chunk_vectors + vectors_vec に直接 INSERT する。"""
    upsert_document(
        conn, collection=collection, path=path, body=body, title=title, mtime=1000
    )
    row = conn.execute(
        "SELECT id, docid FROM documents WHERE collection=? AND path=?",
        (collection, path),
    ).fetchone()
    doc_id, docid = row["id"], row["docid"]

    cur = conn.execute(
        """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
           VALUES (?, ?, 0, 0, ?, 'test-model')""",
        (doc_id, docid, body[:100]),
    )
    chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO vectors_vec(chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, sqlite_vec.serialize_float32(vector)),
    )
    return docid


def _unit_vec(dim, idx):
    """dim 次元の単位ベクトル（idx 番目の要素のみ 1.0）を返す。"""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


@pytest.fixture
def vec_conn(conn, doc_dir):
    """3件のドキュメントと既知ベクトルを持つ接続。"""
    if not vec_available(conn):
        pytest.skip("sqlite-vec 拡張が利用できません")
    add_collection(conn, "test", str(doc_dir))

    # ベクトル: 各ドキュメントを互いに直交する方向に配置
    DIM = 768
    _insert_vec(conn, "test", "a.md", "ドキュメントA", "AAAの内容", _unit_vec(DIM, 0))
    _insert_vec(conn, "test", "b.md", "ドキュメントB", "BBBの内容", _unit_vec(DIM, 1))
    _insert_vec(conn, "test", "c.md", "ドキュメントC", "CCCの内容", _unit_vec(DIM, 2))
    conn.commit()
    set_meta(conn, "vec_indexed", "1")
    conn.commit()
    return conn


def _mock_embed_query(monkeypatch, vector):
    """vsearch 内の embed_query をモックに差し替える。"""
    import jaqmd.search.vsearch as vsearch_mod

    monkeypatch.setattr(vsearch_mod, "embed_query", lambda q: vector, raising=False)

    # vsearch モジュール内で from ..embed import embed_query しているので直接パッチ
    import importlib
    import jaqmd.search.vsearch

    importlib.import_module("jaqmd.search.vsearch")

    # モンキーパッチ: vsearch 関数の embed_query 呼び出しを差し替える
    original_vsearch = jaqmd.search.vsearch.vsearch

    def patched_vsearch(conn, query, **kwargs):
        # embed_query をモックしてから呼び出す
        import jaqmd.embed as embed_mod

        orig = getattr(embed_mod, "embed_query", None)
        embed_mod.embed_query = lambda q: vector
        try:
            return original_vsearch(conn, query, **kwargs)
        finally:
            if orig is not None:
                embed_mod.embed_query = orig

    return patched_vsearch


def test_basic_vsearch(vec_conn, monkeypatch):
    """クエリに最も近いドキュメントが先頭に来ること。"""
    DIM = 768
    # ドキュメントAの方向にクエリを向ける
    query_vec = _unit_vec(DIM, 0)

    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: query_vec)

    results = vsearch(vec_conn, "テストクエリ", n=3)
    assert len(results) >= 1
    assert results[0].filepath.endswith("a.md"), (
        f"最も近いドキュメントが先頭に来ていない: {results[0].filepath}"
    )


def test_doc_unit_aggregation(vec_conn, monkeypatch):
    """同一 docid は最良チャンクのみ（重複なし）。"""
    DIM = 768

    # ドキュメントBに追加チャンクを挿入（同じ docid で chunk_seq=1）
    row = vec_conn.execute(
        "SELECT id, docid FROM documents WHERE collection='test' AND path='b.md'"
    ).fetchone()
    doc_id, docid = row["id"], row["docid"]
    cur = vec_conn.execute(
        """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
           VALUES (?, ?, 1, 50, 'BBBチャンク2', 'test-model')""",
        (doc_id, docid),
    )
    chunk_id2 = cur.lastrowid
    # ゼロベクトル（クエリ方向とは直交）
    vec_conn.execute(
        "INSERT INTO vectors_vec(chunk_id, embedding) VALUES (?, ?)",
        (chunk_id2, sqlite_vec.serialize_float32([0.0] * DIM)),
    )
    vec_conn.commit()

    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 1))

    results = vsearch(vec_conn, "テストクエリ", n=5)
    docids = [r.docid for r in results]
    assert len(docids) == len(set(docids)), "同一 docid が重複して返っている"


def test_score_range(vec_conn, monkeypatch):
    """score が [0, 1] の範囲に収まること（正規化 embedding の場合）。"""
    DIM = 768
    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 0))

    results = vsearch(vec_conn, "テスト", n=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"score が範囲外: {r.score}"


def test_n_limit(vec_conn, monkeypatch):
    """n を指定したとき最大 n 件しか返さないこと。"""
    DIM = 768
    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 0))

    results = vsearch(vec_conn, "テスト", n=2)
    assert len(results) <= 2


def test_all_results(vec_conn, monkeypatch):
    """all_results=True のとき n を超えて返すこと。"""
    DIM = 768
    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 0))

    results = vsearch(vec_conn, "テスト", n=1, all_results=True)
    assert len(results) > 1


def test_empty_query(vec_conn):
    """空クエリは空リストを返すこと。"""
    results = vsearch(vec_conn, "")
    assert results == []
    results = vsearch(vec_conn, "   ")
    assert results == []


def test_soft_deleted_excluded(vec_conn, monkeypatch):
    """soft-delete されたドキュメントは結果に含まれないこと。"""
    DIM = 768
    # ドキュメントAを論理削除
    vec_conn.execute(
        "UPDATE documents SET active = 0 WHERE collection = 'test' AND path = 'a.md'"
    )
    vec_conn.commit()

    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 0))

    results = vsearch(vec_conn, "テスト", n=5)
    filepaths = [r.filepath for r in results]
    assert not any("a.md" in fp for fp in filepaths), (
        "soft-delete 済みドキュメントが結果に含まれている"
    )


def test_collection_filter(vec_conn, monkeypatch, doc_dir):
    """collection フィルタが正しく機能すること。"""
    DIM = 768
    # 別コレクションにドキュメントを追加
    vec_conn.execute(
        "INSERT INTO collections(name, path, glob_mask) VALUES ('other', ?, '**/*.md')",
        (str(doc_dir),),
    )
    upsert_document(
        vec_conn,
        collection="other",
        path="x.md",
        body="XXXの内容",
        title="X",
        mtime=1000,
    )
    xrow = vec_conn.execute(
        "SELECT id, docid FROM documents WHERE collection='other' AND path='x.md'"
    ).fetchone()
    cur = vec_conn.execute(
        """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
           VALUES (?, ?, 0, 0, 'XXX', 'test-model')""",
        (xrow["id"], xrow["docid"]),
    )
    vec_conn.execute(
        "INSERT INTO vectors_vec(chunk_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(_unit_vec(DIM, 0))),
    )
    vec_conn.commit()

    import jaqmd.embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_query", lambda q: _unit_vec(DIM, 0))

    results = vsearch(vec_conn, "テスト", n=5, collection="test")
    collections = {r.filepath.split("/")[0] for r in results}
    assert collections == {"test"}, f"他のコレクションが混入: {collections}"


@pytest.mark.integration
def test_vsearch_with_real_model(tmp_cache):
    """実モデルを使った統合テスト（fastembed + ruri-v3-310m のダウンロードが必要）。"""
    pytest.importorskip("fastembed")
    from jaqmd.store import connect, add_collection, set_meta, upsert_document

    conn = connect()
    if not vec_available(conn):
        pytest.skip("sqlite-vec 拡張が利用できません")

    add_collection(conn, "integ", "/tmp/integ_docs")

    from jaqmd.embed import embed_documents, count_tokens, EMBED_MODEL
    from jaqmd.chunk import chunk_document

    body = "東京は日本の首都です。大阪は関西の中心都市です。"
    upsert_document(
        conn, collection="integ", path="test.md", body=body, title="テスト", mtime=1000
    )
    row = conn.execute(
        "SELECT id, docid FROM documents WHERE collection='integ' AND path='test.md'"
    ).fetchone()

    chunks = chunk_document(body, count_tokens=count_tokens)
    vecs = embed_documents([ct for _, _, ct in chunks])

    import sqlite_vec as sv

    for (seq, pos, ct), vec in zip(chunks, vecs):
        cur = conn.execute(
            """INSERT INTO chunk_vectors(doc_id, docid, chunk_seq, chunk_pos, chunk_text, embed_model)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["id"], row["docid"], seq, pos, ct, EMBED_MODEL),
        )
        conn.execute(
            "INSERT INTO vectors_vec(chunk_id, embedding) VALUES (?, ?)",
            (cur.lastrowid, sv.serialize_float32(vec)),
        )
    set_meta(conn, "vec_indexed", "1")
    conn.commit()

    results = vsearch(conn, "日本の首都")
    assert len(results) >= 1
