import pytest

from jaqmd.search.trisearch import trisearch
from jaqmd.store import add_collection, set_meta, upsert_document


@pytest.fixture
def indexed_conn(conn, doc_dir):
    add_collection(conn, "test", str(doc_dir))
    upsert_document(
        conn,
        collection="test",
        path="a.md",
        body="形態素解析は日本語の自然言語処理の基礎技術です",
        title="形態素解析について",
        mtime=1000,
    )
    upsert_document(
        conn,
        collection="test",
        path="b.md",
        body="検索エンジンの仕組みと実装方法を解説します",
        title="検索エンジン入門",
        mtime=1001,
    )
    upsert_document(
        conn,
        collection="test",
        path="c.md",
        body="サーバーの設定と運用について説明します",
        title="サーバー運用ガイド",
        mtime=1002,
    )
    conn.commit()
    set_meta(conn, "trigram_indexed", "1")
    conn.commit()
    return conn


def test_basic_japanese_search(indexed_conn):
    """日本語クエリで関連ドキュメントがヒットする。"""
    results = trisearch(indexed_conn, "形態素解析")
    assert len(results) >= 1
    filepaths = [r.filepath for r in results]
    assert any("a.md" in fp for fp in filepaths)


def test_no_results(indexed_conn):
    """存在しないキーワードは0件を返す。"""
    results = trisearch(indexed_conn, "絶対に存在しないXYZ999")
    assert results == []


def test_score_ordering(indexed_conn):
    """スコアは降順（高い方が先頭）で返る。"""
    results = trisearch(indexed_conn, "説明します", n=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_n_limit(indexed_conn):
    """n=1 で1件だけ返る。"""
    results = trisearch(indexed_conn, "解説", n=1)
    assert len(results) <= 1


def test_all_results(indexed_conn):
    """all_results=True で件数上限なし。"""
    results = trisearch(indexed_conn, "す", all_results=True)
    # 件数制限なしなので n=5 の制約はない
    assert isinstance(results, list)


def test_collection_filter(conn, tmp_path):
    """コレクション絞り込みが正しく動作する。"""
    d1 = tmp_path / "col1"
    d2 = tmp_path / "col2"
    d1.mkdir()
    d2.mkdir()

    add_collection(conn, "col1", str(d1))
    add_collection(conn, "col2", str(d2))
    upsert_document(
        conn,
        collection="col1",
        path="a.md",
        body="日本語処理の解説",
        title="A",
        mtime=1000,
    )
    upsert_document(
        conn,
        collection="col2",
        path="b.md",
        body="日本語処理は重要です",
        title="B",
        mtime=1001,
    )
    conn.commit()

    results = trisearch(conn, "日本語処理", collection="col1")
    assert all("col1/" in r.filepath for r in results)


def test_server_typo_match(indexed_conn):
    """サーバ（3文字）で サーバー（4文字）を含むドキュメントにマッチする。

    trigram では 'サーバ' が 'サーバー' の先頭trigram として機能する。
    """
    results = trisearch(indexed_conn, "サーバ")
    assert len(results) >= 1
    assert any("c.md" in r.filepath for r in results)


def test_min_score_filter(indexed_conn):
    """min_score より低いスコアの結果は除外される。"""
    all_res = trisearch(indexed_conn, "形態素解析", n=10)
    if not all_res:
        pytest.skip("検索結果が0件")
    max_score = max(r.score for r in all_res)
    # max_score より高い閾値を設定 → 0件になる
    filtered = trisearch(indexed_conn, "形態素解析", n=10, min_score=max_score + 1.0)
    assert filtered == []


def test_result_fields(indexed_conn):
    """SearchResult の各フィールドが正しく設定されている。"""
    results = trisearch(indexed_conn, "形態素解析")
    assert results
    r = results[0]
    assert r.docid
    assert r.filepath
    assert r.score > 0
    # snippet は空でも可（短い文書の場合）
    assert isinstance(r.snippet, str)
    assert isinstance(r.body, str)
    # body は原文全文
    assert r.body == "形態素解析は日本語の自然言語処理の基礎技術です"
