from jaqmd.format import format_results
from jaqmd.search.trisearch import SearchResult

ESC = "\x1b["


def _results():
    return [
        SearchResult(
            docid="a1b2c3",
            score=0.812,
            filepath="docs/nlp/morph.md",
            title="形態素解析の基礎",
            snippet="...この文書は形態素解析について説明する...",
            body="本文",
        ),
        SearchResult(
            docid="d4e5f6",
            score=0.771,
            filepath="docs/tokenize.md",
            title="トークナイザ設計",
            snippet="...トークナイザの設計について...",
            body="本文",
        ),
    ]


def test_default_no_color_is_backward_compatible():
    out = format_results(_results(), query="形態素解析", color=False)
    assert ESC not in out
    assert "─" not in out
    assert "[a1b2c3] 形態素解析の基礎  (score: 0.812)" in out
    assert "  docs/nlp/morph.md" in out


def test_default_color_adds_ansi_and_rule():
    out = format_results(_results(), query="形態素解析", color=True)
    assert ESC in out
    # 件と件の間に罫線が入る
    assert "─" in out
    # docid・パス・タイトルの文字自体は残る
    assert "a1b2c3" in out
    assert "docs/nlp/morph.md" in out


def test_default_color_highlights_query_term():
    out = format_results(_results(), query="形態素解析", color=True)
    # マッチ語がスタイル付きで出力される（語の直前に ANSI エスケープ）
    assert f"{ESC}" in out
    assert "形態素解析" in out
    # ハイライトなし（color=False）では語の直前に ANSI が付かない
    plain = format_results(_results(), query="形態素解析", color=False)
    assert f"{ESC}" not in plain


def test_json_ignores_query_and_color():
    colored = format_results(_results(), fmt="json", query="形態素解析", color=True)
    plain = format_results(_results(), fmt="json")
    assert ESC not in colored
    assert colored == plain


def test_files_ignores_query_and_color():
    colored = format_results(_results(), fmt="files", query="形態素解析", color=True)
    plain = format_results(_results(), fmt="files")
    assert ESC not in colored
    assert colored == plain


def test_no_results():
    assert format_results([], color=True) == "No results found."
