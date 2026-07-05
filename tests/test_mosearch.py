import pytest

sudachipy = pytest.importorskip("sudachipy")

from jaqmd.store import add_collection, set_meta, upsert_document
from jaqmd.tokenize.morph import tokenize_text
from jaqmd.search.mosearch import mosearch


def _insert_morph(conn, collection, path, title, body):
    """テスト用: ドキュメントを upsert して docs_fts_morph にも投入する。"""
    upsert_document(
        conn, collection=collection, path=path, body=body, title=title, mtime=1000
    )
    row = conn.execute(
        "SELECT docid FROM documents WHERE collection=? AND path=?",
        (collection, path),
    ).fetchone()
    docid = row["docid"]
    conn.execute(
        "INSERT INTO docs_fts_morph(docid, filepath, title, body) VALUES (?, ?, ?, ?)",
        (
            docid,
            f"{collection}/{path}",
            tokenize_text(title or ""),
            tokenize_text(body),
        ),
    )
    return docid


@pytest.fixture
def morph_conn(conn, doc_dir):
    add_collection(conn, "test", str(doc_dir))
    _insert_morph(
        conn,
        "test",
        "a.md",
        "形態素解析について",
        "形態素解析は日本語の自然言語処理の基礎技術です",
    )
    _insert_morph(
        conn,
        "test",
        "b.md",
        "検索エンジン入門",
        "検索エンジンの仕組みと実装方法を解説します",
    )
    _insert_morph(
        conn,
        "test",
        "c.md",
        "サーバー運用ガイド",
        "サーバーの設定と運用について説明します",
    )
    conn.commit()
    set_meta(conn, "morph_indexed", "1")
    conn.commit()
    return conn


def test_basic_search(morph_conn):
    results = mosearch(morph_conn, "形態素解析")
    assert len(results) >= 1
    assert any("a.md" in r.filepath for r in results)


def test_server_variant_match(morph_conn):
    """サーバ（短縮形）でサーバー（長音あり）の文書がヒットする（正規化形一致）。"""
    results_short = mosearch(morph_conn, "サーバ")
    results_long = mosearch(morph_conn, "サーバー")
    # 両方同じ文書にヒットする
    assert len(results_short) >= 1
    assert len(results_long) >= 1
    filepaths_short = {r.filepath for r in results_short}
    filepaths_long = {r.filepath for r in results_long}
    assert filepaths_short == filepaths_long


def test_trigram_difference(morph_conn):
    """trigram では3文字未満でスキップされるが morph は1形態素でもヒットする。"""
    from jaqmd.search.trisearch import trisearch

    # 「する」は2文字なので trigram は空クエリになりヒットしない
    trigram_results = trisearch(morph_conn, "する")
    morph_results = mosearch(morph_conn, "する")
    # morph は1形態素でも検索できる（trigram より多くヒットする可能性）
    assert len(morph_results) >= len(trigram_results)


def test_no_results(morph_conn):
    # 辞書に存在しないアルファベット列のみ → 正規化形も同じ文字列 → 文書にヒットしない
    results = mosearch(morph_conn, "XYZNONEXISTENT999ZZZZZ")
    assert results == []


def test_score_ordering(morph_conn):
    results = mosearch(morph_conn, "解析", n=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_n_limit(morph_conn):
    results = mosearch(morph_conn, "す", n=1)
    assert len(results) <= 1


def test_result_fields(morph_conn):
    results = mosearch(morph_conn, "形態素解析")
    assert results
    r = results[0]
    assert r.docid
    assert r.filepath
    assert r.score > 0
    assert isinstance(r.snippet, str)
    assert isinstance(r.body, str)


def test_snippet_is_original_text(morph_conn):
    """スニペットが正規化形の分かち書きではなく原文テキストを含む。"""
    results = mosearch(morph_conn, "形態素解析")
    assert results
    r = results[0]
    # 正規化形の分かち書き（空白区切りのひらがな・カタカナが連続）は含まれない
    # 原文に含まれる文字列がスニペットに現れる
    assert "形態素解析" in r.snippet or "日本語" in r.snippet


def test_body_is_original_text(morph_conn):
    """body フィールドが原文全文を保持している。"""
    results = mosearch(morph_conn, "形態素解析")
    assert results
    r = results[0]
    # body は原文そのもの（正規化形分かち書きではない）
    assert r.body == "形態素解析は日本語の自然言語処理の基礎技術です"


def test_collection_filter(conn, tmp_path):
    d1 = tmp_path / "col1"
    d2 = tmp_path / "col2"
    d1.mkdir()
    d2.mkdir()
    add_collection(conn, "col1", str(d1))
    add_collection(conn, "col2", str(d2))
    _insert_morph(conn, "col1", "a.md", "A", "日本語処理の解説")
    _insert_morph(conn, "col2", "b.md", "B", "日本語処理は重要です")
    conn.commit()

    results = mosearch(conn, "日本語", collection="col1")
    assert all("col1/" in r.filepath for r in results)
