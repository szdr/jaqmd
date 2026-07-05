"""extract_snippet の単体テスト（埋め込み不要・高速）。"""

from __future__ import annotations

import pytest

from jaqmd.search.snippet import extract_snippet


# ---------------------------------------------------------------------------
# 基本
# ---------------------------------------------------------------------------


def test_short_text_returned_as_is():
    """max_chars 以下のテキストはそのまま返す。"""
    text = "短いテキスト。"
    assert extract_snippet(text, ["テキスト"], max_chars=200) == text


def test_empty_text():
    """空テキストは空文字列を返す。"""
    assert extract_snippet("", ["クエリ"]) == ""


def test_query_relevant_sentence_included():
    """クエリに関連する文が snippet に含まれること。"""
    filler = "関係のない文が続きます。" * 5
    text = filler + "東京は日本の首都です。" + filler
    snippet = extract_snippet(text, ["東京", "首都"], max_chars=200)
    assert "東京" in snippet


def test_no_overlap_falls_back_to_start():
    """クエリ語が一切含まれない場合は先頭付近の文が返り、クラッシュしない。"""
    text = "あいうえお。\n" * 30
    snippet = extract_snippet(text, ["xyz", "unknown"], max_chars=200)
    assert isinstance(snippet, str)
    assert len(snippet) > 0


def test_output_body_within_max_chars():
    """省略記号を除いた本文部分が max_chars 以内に収まること。"""
    sents = [f"文番号{i:02d}はここです。" for i in range(30)]
    text = "\n".join(sents)
    max_chars = 150
    snippet = extract_snippet(text, ["文番号15"], max_chars=max_chars)
    body = snippet.strip(".")
    assert len(body) <= max_chars


def test_ellipsis_prefix_when_not_at_start():
    """窓がテキスト先頭に届いていなければ先頭に '...' が付く。"""
    prefix_filler = "無関係な文が続く。" * 10
    text = prefix_filler + "ターゲット文があります。"
    snippet = extract_snippet(text, ["ターゲット"], max_chars=100)
    assert snippet.startswith("...")


def test_no_ellipsis_suffix_at_end():
    """窓がテキスト末尾に届いていれば末尾に '...' は付かない。"""
    prefix_filler = "無関係。" * 20
    target = "ターゲット文です。"
    text = prefix_filler + target
    snippet = extract_snippet(text, ["ターゲット"], max_chars=50)
    assert not snippet.endswith("...")


def test_no_ellipsis_prefix_at_start():
    """窓がテキスト先頭ならば先頭に '...' は付かない。"""
    target = "ターゲット文です。"
    suffix_filler = "無関係。" * 20
    text = target + suffix_filler
    snippet = extract_snippet(text, ["ターゲット"], max_chars=50)
    assert not snippet.startswith("...")


def test_short_query_fallback():
    """2文字以下のクエリ（trigram が生成できない）でもクラッシュしない。"""
    text = "あいうえお。\n" * 20
    snippet = extract_snippet(text, ["あ"], max_chars=100)
    assert isinstance(snippet, str)


def test_multitoken_query_best_sentence():
    """複数トークンのうち最も多く重なる文が採用されること。"""
    sents = [
        "Aの話題が続きます。",
        "Bの話題も出てきます。",
        "東京と大阪の比較が重要です。",
        "Cの話題が続きます。",
        "Dの話題も続きます。",
    ]
    text = "\n".join(sents * 5)
    snippet = extract_snippet(text, ["東京", "大阪", "比較"], max_chars=100)
    assert "東京" in snippet or "大阪" in snippet or "比較" in snippet


def test_empty_terms():
    """terms が空リストでもクラッシュせず先頭付近を返す。"""
    sents = ["文が続きます。"] * 30
    text = "\n".join(sents)
    snippet = extract_snippet(text, [], max_chars=50)
    assert isinstance(snippet, str)
    assert len(snippet) > 0
