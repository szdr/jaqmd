from __future__ import annotations

import json

import pytest

from jaqmd.qe import ExpansionResult, _extract_json, expand
from jaqmd.store import get_qe_cache


# ---------------------------------------------------------------------------
# _extract_json: 堅牢な JSON 抽出
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    text = '{"lex":["a","b"],"vec":"v","hyde":"h"}'
    assert _extract_json(text) == {"lex": ["a", "b"], "vec": "v", "hyde": "h"}


def test_extract_json_with_surrounding_noise():
    text = 'ここが答えです:\n{"lex":["a"],"vec":"v","hyde":"h"}\nありがとう'
    assert _extract_json(text) == {"lex": ["a"], "vec": "v", "hyde": "h"}


def test_extract_json_with_nested_braces():
    text = '{"lex":["a"],"vec":"v","hyde":"h","meta":{"n":1}}'
    obj = _extract_json(text)
    assert obj["meta"] == {"n": 1}


def test_extract_json_no_brace_returns_none():
    assert _extract_json("no json here") is None


def test_extract_json_malformed_returns_none():
    assert _extract_json('{"lex": [oops}') is None


def test_extract_json_unclosed_returns_none():
    assert _extract_json('{"lex": ["a"') is None


# ---------------------------------------------------------------------------
# expand(): degrade 経路（llama_cpp 未導入/ロード失敗）
# ---------------------------------------------------------------------------


def test_expand_returns_none_when_llm_unavailable(conn, monkeypatch):
    monkeypatch.setattr("jaqmd.qe._get_llm", lambda: None)
    assert expand(conn, "木魚") is None


# ---------------------------------------------------------------------------
# expand(): ダミー LLM でのキャッシュ miss → 保存 → hit
# ---------------------------------------------------------------------------


class _DummyLlm:
    """canned JSON を返すダミー LLM（llama_cpp.Llama 互換の最小インターフェース）。"""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[str] = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append(messages[0]["content"])
        return {"choices": [{"message": {"content": self.content}}]}


def test_expand_cache_miss_then_hit(conn, monkeypatch):
    canned = json.dumps(
        {"lex": ["木魚", "もくぎょ"], "vec": "木魚とは何か", "hyde": "木魚は仏具の一種..."},
        ensure_ascii=False,
    )
    llm = _DummyLlm(canned)
    monkeypatch.setattr("jaqmd.qe._get_llm", lambda: llm)

    result = expand(conn, "木魚")
    assert result == ExpansionResult(
        lex=["木魚", "もくぎょ"], vec="木魚とは何か", hyde="木魚は仏具の一種..."
    )
    assert len(llm.calls) == 1

    # キャッシュに保存されていること
    import hashlib

    query_hash = hashlib.sha256("木魚".encode("utf-8")).hexdigest()
    row = get_qe_cache(conn, query_hash, "szdr/jaqmd-qe-gemma-4-e2b-it")
    assert row is not None
    assert row["query_raw"] == "木魚"

    # 2 回目はキャッシュ hit で LLM が呼ばれない
    result2 = expand(conn, "木魚")
    assert result2 == result
    assert len(llm.calls) == 1


def test_expand_returns_none_on_malformed_response(conn, monkeypatch):
    llm = _DummyLlm("not json at all")
    monkeypatch.setattr("jaqmd.qe._get_llm", lambda: llm)
    assert expand(conn, "テストクエリ") is None


def test_expand_returns_none_on_missing_keys(conn, monkeypatch):
    llm = _DummyLlm(json.dumps({"lex": ["a"]}))
    monkeypatch.setattr("jaqmd.qe._get_llm", lambda: llm)
    assert expand(conn, "テストクエリ2") is None


def test_expand_returns_none_when_inference_raises(conn, monkeypatch):
    class _RaisingLlm:
        def create_chat_completion(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr("jaqmd.qe._get_llm", lambda: _RaisingLlm())
    assert expand(conn, "テストクエリ3") is None


# ---------------------------------------------------------------------------
# 統合テスト: 実モデルをロード（重い・既定スキップ）
# ---------------------------------------------------------------------------

pytest.importorskip("llama_cpp")


@pytest.mark.integration
def test_expand_integration_real_model_returns_json_keys(conn):
    result = expand(conn, "木魚")
    assert result is not None
    assert isinstance(result.lex, list)
    assert isinstance(result.vec, str)
    assert isinstance(result.hyde, str)
