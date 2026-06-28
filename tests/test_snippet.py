from jaqmd.snippet import make_snippet


def test_match_center():
    """マッチ箇所が中心に来るようスニペットを切り出す。"""
    body = "あ" * 40 + "機械学習" + "い" * 40
    snippet = make_snippet(body, ["機械学習"], width=20)
    assert "機械学習" in snippet


def test_fallback_to_head():
    """マッチしない場合は先頭から切り出す。"""
    body = "これはテストの文書です。" * 5
    snippet = make_snippet(body, ["NOTFOUND"], width=10)
    # 先頭付近の文字列が含まれる
    assert "これは" in snippet


def test_suffix_ellipsis():
    """本文が width を超える場合、末尾に ... が付く。"""
    body = "a" * 200
    snippet = make_snippet(body, [], width=10)
    assert snippet.endswith("...")


def test_no_prefix_ellipsis_at_start():
    """先頭から切り出した場合、先頭の ... は付かない。"""
    body = "abc" * 5
    snippet = make_snippet(body, ["abc"], width=30)
    assert not snippet.startswith("...")


def test_newline_normalized():
    """改行・連続空白が1スペースに正規化される。"""
    body = "foo\nbar\nbaz"
    snippet = make_snippet(body, ["bar"], width=20)
    assert "\n" not in snippet


def test_empty_body():
    """空文字列は空文字列を返す。"""
    assert make_snippet("", ["foo"]) == ""


def test_empty_terms_fallback():
    """terms が空でも先頭フォールバックで動作する。"""
    body = "これはサンプルテキストです。"
    snippet = make_snippet(body, [], width=10)
    assert len(snippet) > 0


def test_case_insensitive():
    """大文字小文字を区別しないマッチ。"""
    body = "The Quick Brown Fox"
    snippet = make_snippet(body, ["quick"], width=20)
    assert "Quick" in snippet
