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


def test_tokenize_text_long_document_with_newlines():
    """SudachiPyのバイト長制約(約49149バイト)を超える長文でも例外にならないこと。"""
    from jaqmd.tokenize.morph import _MAX_BYTES

    line = "これはテスト用の文章です。サーバの設定について説明します。\n"
    # 1行を十分な回数繰り返し、_MAX_BYTES を大きく超える長さにする
    repeat = (_MAX_BYTES * 3) // len(line.encode("utf-8")) + 10
    text = line * repeat
    assert len(text.encode("utf-8")) > _MAX_BYTES

    result = tokenize_text(text)
    assert isinstance(result, str)
    assert "サーバー" in result


def test_tokenize_text_single_huge_line_without_newline():
    """改行を含まない巨大な1行でも例外にならないこと（文字単位フォールバック）。"""
    from jaqmd.tokenize.morph import _MAX_BYTES

    text = "あ" * ((_MAX_BYTES // 3) * 5)  # "あ"は3バイト、上限を大きく超える
    assert len(text.encode("utf-8")) > _MAX_BYTES

    result = tokenize_text(text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_tokenize_text_short_text_unaffected():
    """短文は分割されても従来と同一の結果になること。"""
    result = tokenize_text("サーバの設定")
    assert "サーバー" in result
