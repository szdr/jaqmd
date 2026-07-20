"""config の設定解決テスト（snippet 長など）。"""

from __future__ import annotations

from jaqmd import config


def test_snippet_chars_default():
    """未設定なら既定値 300。"""
    settings = config.reload()
    assert settings.search_snippet_chars == 300


def test_snippet_chars_from_env(monkeypatch):
    """環境変数 JAQMD_SEARCH_SNIPPET_CHARS で上書きできる。"""
    monkeypatch.setenv("JAQMD_SEARCH_SNIPPET_CHARS", "500")
    settings = config.reload()
    assert settings.search_snippet_chars == 500


def test_snippet_chars_invalid_falls_back_to_default(monkeypatch):
    """不正値なら既定値 300 にフォールバックする。"""
    monkeypatch.setenv("JAQMD_SEARCH_SNIPPET_CHARS", "not-a-number")
    settings = config.reload()
    assert settings.search_snippet_chars == 300
