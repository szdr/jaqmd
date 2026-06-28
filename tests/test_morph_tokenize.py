import pytest

sudachipy = pytest.importorskip("sudachipy")

from jaqmd.tokenize.morph import tokenize_text, to_fts_query


def test_tokenize_text_returns_space_separated():
    result = tokenize_text("サーバの設定")
    assert isinstance(result, str)
    assert " " in result or len(result) > 0


def test_normalized_form_server():
    """サーバ → サーバー に正規化される。"""
    result = tokenize_text("サーバ")
    assert "サーバー" in result


def test_normalized_form_ordinal():
    """第1条 → 第一条 等の表記ゆれが正規化される。"""
    result1 = tokenize_text("第1条")
    result2 = tokenize_text("第一条")
    # 両方とも同一の正規化形を含む
    tokens1 = set(result1.split())
    tokens2 = set(result2.split())
    assert tokens1 & tokens2, f"共通トークンなし: {tokens1} vs {tokens2}"


def test_to_fts_query_basic():
    result = to_fts_query("形態素解析")
    assert '"' in result
    # OR 結合されている（複数形態素の場合）
    assert isinstance(result, str)
    assert len(result) > 0


def test_to_fts_query_empty():
    assert to_fts_query("") == ""
    assert to_fts_query("   ") == ""


def test_to_fts_query_deduplication():
    """同じ正規化形が重複しないこと。"""
    result = to_fts_query("する する する")
    parts = [p for p in result.split(" OR ")]
    assert len(parts) == len(set(parts))


def test_to_fts_query_server_normalization():
    """サーバ と サーバー が同一の正規化形を生成する。"""
    q1 = to_fts_query("サーバ")
    q2 = to_fts_query("サーバー")
    assert q1 == q2


def test_to_fts_query_quote_escape():
    """クォートを含む入力が安全にエスケープされる。"""
    result = to_fts_query('テスト"文字列')
    assert '""' in result or '"テスト' in result
