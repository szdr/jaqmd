from __future__ import annotations

import pytest

from jaqmd.search.trisearch import SearchResult
from jaqmd.rerank import rerank, _doc_text


def _make_result(
    docid: str, score: float = 1.0, body: str = "", snippet: str = "snippet"
) -> SearchResult:
    return SearchResult(
        docid=docid,
        score=score,
        filepath=f"{docid}.md",
        title=docid,
        snippet=snippet,
        body=body,
    )


class _DummyEncoder:
    """docid ごとに固定スコアを返すダミー cross-encoder。"""

    def __init__(self, score_map: dict[str, float]):
        self.score_map = score_map
        self.calls = []

    def rerank(self, query, documents, **kwargs):
        self.calls.append((query, list(documents)))
        return [self.score_map[doc] for doc in documents]


# ---------------------------------------------------------------------------
# _doc_text
# ---------------------------------------------------------------------------


def test_doc_text_prefers_body():
    r = _make_result("a", body="本文", snippet="スニペット")
    assert _doc_text(r) == "本文"


def test_doc_text_falls_back_to_snippet():
    r = _make_result("a", body="", snippet="スニペット")
    assert _doc_text(r) == "スニペット"


# ---------------------------------------------------------------------------
# rerank: 恒等フォールバック経路
# ---------------------------------------------------------------------------


def test_rerank_disabled_returns_identity():
    results = [_make_result("a"), _make_result("b")]
    out = rerank("q", results, enabled=False)
    assert out == results


def test_rerank_empty_results():
    assert rerank("q", []) == []


def test_rerank_no_encoder_returns_identity(monkeypatch):
    monkeypatch.setattr("jaqmd.rerank._get_encoder", lambda model=None, reporter=None: None)
    results = [_make_result("a"), _make_result("b")]
    out = rerank("q", results)
    assert out == results


def test_rerank_disabled_respects_n():
    results = [_make_result("a"), _make_result("b"), _make_result("c")]
    out = rerank("q", results, enabled=False, n=2)
    assert out == results[:2]


# ---------------------------------------------------------------------------
# rerank: エンコーダあり（ダミー）
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_encoder_score(monkeypatch):
    results = [
        _make_result("a", body="a本文"),
        _make_result("b", body="b本文"),
        _make_result("c", body="c本文"),
    ]
    # b が最も高スコアになるようにする
    encoder = _DummyEncoder({"a本文": 0.1, "b本文": 0.9, "c本文": 0.5})
    monkeypatch.setattr("jaqmd.rerank._get_encoder", lambda model=None, reporter=None: encoder)

    out = rerank("q", results, top_k=None)
    assert [r.docid for r in out] == ["b", "c", "a"]
    assert out[0].score == pytest.approx(0.9)
    assert out[1].score == pytest.approx(0.5)
    assert out[2].score == pytest.approx(0.1)


def test_rerank_top_k_splits_head_and_tail(monkeypatch):
    results = [
        _make_result("a", body="a本文", score=10.0),
        _make_result("b", body="b本文", score=9.0),
        _make_result("c", body="c本文", score=8.0),
    ]
    # head = [a, b]（top_k=2）が再スコア対象。b がトップになるよう仕込む。
    encoder = _DummyEncoder({"a本文": 0.2, "b本文": 0.8})
    monkeypatch.setattr("jaqmd.rerank._get_encoder", lambda model=None, reporter=None: encoder)

    out = rerank("q", results, top_k=2)
    # head 部分は再スコアされ b が先頭、tail の c はそのまま末尾に温存
    assert [r.docid for r in out] == ["b", "a", "c"]
    assert out[0].score == pytest.approx(0.8)
    assert out[1].score == pytest.approx(0.2)
    # tail はスコア変更なし
    assert out[2].score == pytest.approx(8.0)
    # エンコーダには head の2件のみ渡された
    assert len(encoder.calls[0][1]) == 2


def test_rerank_respects_n_after_reorder(monkeypatch):
    results = [_make_result("a", body="a"), _make_result("b", body="b")]
    encoder = _DummyEncoder({"a": 0.1, "b": 0.9})
    monkeypatch.setattr("jaqmd.rerank._get_encoder", lambda model=None, reporter=None: encoder)

    out = rerank("q", results, n=1)
    assert len(out) == 1
    assert out[0].docid == "b"


# ---------------------------------------------------------------------------
# 統合テスト: 実モデルをロード（重い・既定スキップ）
# ---------------------------------------------------------------------------

pytest.importorskip("fastembed")


@pytest.mark.integration
def test_rerank_integration_real_model_reorders():
    """実モデルで関連度の高い文書が最上位に来ることを検証する。"""
    results = [
        _make_result(
            "irrelevant",
            score=1.0,
            body="今日の天気は晴れで、気温は25度前後の見込みです。",
        ),
        _make_result(
            "relevant",
            score=0.5,
            body="瑠璃色（るりいろ）は、紫みを帯びた濃い青。瑠璃の色から。",
        ),
    ]
    out = rerank("瑠璃色はどんな色？", results, top_k=None)
    assert out[0].docid == "relevant"


@pytest.mark.integration
def test_rerank_integration_handles_mixed_length_documents():
    """トークン長がバラバラな文書が混在してもクラッシュせず処理できることを検証する。

    reranker モデル（szdr/ruri-v3-reranker-310m-onnx）の tokenizer.json は
    以前 padding=Fixed(32) / truncation=32 という壊れた設定を持っており、
    fastembed が truncation のみ model_max_length へ上書きするため、
    32 トークン超の文書が可変長のまま np.array 化されて
    ValueError（inhomogeneous shape）になる不具合があった（上流で修正済み）。
    短文・長文を混在させることで設定リグレッションを検知する回帰テスト。
    """
    results = [
        _make_result("short", body="今日の天気は晴れです。"),
        _make_result(
            "long",
            body="瑠璃色（るりいろ）は、紫みを帯びた濃い青。瑠璃の色から。" * 20,
        ),
        _make_result("short2", body="猫が好きです。"),
    ]
    out = rerank("瑠璃色はどんな色？", results, top_k=None)
    assert {r.docid for r in out} == {"short", "long", "short2"}
    assert out[0].docid == "long"
