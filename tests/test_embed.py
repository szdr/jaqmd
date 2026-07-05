"""embed.py の単体テスト — prefix 付与の検証（fastembed 本体はモック）。"""
from __future__ import annotations

import pytest


# fastembed が未インストールの場合はモックを強制使用するため importorskip しない
# 代わりに fastembed の TextEmbedding 自体をモックする

class _FakeEmbedding:
    """fastembed.TextEmbedding の最小モック。"""

    add_custom_model_calls: list[dict] = []
    embed_calls: list[list[str]] = []
    embed_kwargs: list[dict] = []

    @classmethod
    def add_custom_model(cls, **kwargs):
        cls.add_custom_model_calls.append(kwargs)

    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name

    def embed(self, texts, batch_size=256):
        texts = list(texts)
        self.__class__.embed_calls.append(texts)
        self.__class__.embed_kwargs.append({"batch_size": batch_size})
        # 768次元のゼロベクトルを返す
        for _ in texts:
            yield [0.0] * 768


class _FakePoolingType:
    MEAN = "mean"


class _FakeModelSource:
    def __init__(self, hf=None):
        self.hf = hf


@pytest.fixture(autouse=True)
def patch_fastembed(monkeypatch):
    """fastembed モジュールをモックで差し替える。"""
    import sys
    import types

    # fastembed パッケージをモックモジュールとして登録
    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = _FakeEmbedding

    fake_common = types.ModuleType("fastembed.common")
    fake_model_desc = types.ModuleType("fastembed.common.model_description")
    fake_model_desc.PoolingType = _FakePoolingType
    fake_model_desc.ModelSource = _FakeModelSource

    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
    monkeypatch.setitem(sys.modules, "fastembed.common", fake_common)
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_model_desc)

    # _model キャッシュをリセット
    import jaqmd.embed as embed_mod
    monkeypatch.setattr(embed_mod, "_model", None)

    _FakeEmbedding.add_custom_model_calls = []
    _FakeEmbedding.embed_calls = []
    _FakeEmbedding.embed_kwargs = []

    yield


def test_embed_documents_applies_doc_prefix():
    """embed_documents は各テキストに DOC_PREFIX を付与して embed に渡す。"""
    from jaqmd.embed import DOC_PREFIX, embed_documents

    texts = ["テキストA", "テキストB"]
    list(embed_documents(texts))  # embed はジェネレータなので消費する

    assert len(_FakeEmbedding.embed_calls) == 1
    passed = _FakeEmbedding.embed_calls[0]
    assert len(passed) == 2
    for original, sent in zip(texts, passed):
        assert sent == DOC_PREFIX + original, (
            f"DOC_PREFIX が付与されていない: expected '{DOC_PREFIX + original}', got '{sent}'"
        )


def test_embed_query_applies_query_prefix():
    """embed_query はテキストに QUERY_PREFIX を付与して embed に渡す。"""
    from jaqmd.embed import QUERY_PREFIX, embed_query

    embed_query("検索クエリテスト")

    assert len(_FakeEmbedding.embed_calls) == 1
    passed = _FakeEmbedding.embed_calls[0]
    assert len(passed) == 1
    assert passed[0] == QUERY_PREFIX + "検索クエリテスト", (
        f"QUERY_PREFIX が付与されていない: got '{passed[0]}'"
    )


def test_embed_documents_empty():
    """空リストを渡したら空リストが返る（モデルは呼ばれない）。"""
    from jaqmd.embed import embed_documents

    result = embed_documents([])
    assert result == []
    assert _FakeEmbedding.embed_calls == []


def test_embed_documents_returns_list_of_floats():
    """embed_documents の戻り値がベクトルのイテラブルであること（各ベクトルは EMBED_DIM 次元）。"""
    from jaqmd.embed import EMBED_DIM, embed_documents

    result = list(embed_documents(["テスト"]))
    assert len(result) == 1
    assert len(result[0]) == EMBED_DIM


def test_embed_documents_passes_batch_size():
    """embed_documents は batch_size を fastembed の embed にそのまま渡す。"""
    from jaqmd.embed import embed_documents

    list(embed_documents(["テスト"], batch_size=32))
    assert _FakeEmbedding.embed_kwargs[-1] == {"batch_size": 32}


def test_embed_query_returns_list_of_floats():
    """embed_query の戻り値が list[float] であること。"""
    from jaqmd.embed import EMBED_DIM, embed_query

    result = embed_query("テスト")
    assert isinstance(result, list)
    assert len(result) == EMBED_DIM


def test_doc_and_query_prefix_are_different():
    """DOC_PREFIX と QUERY_PREFIX が異なること（混同防止）。"""
    from jaqmd.embed import DOC_PREFIX, QUERY_PREFIX

    assert DOC_PREFIX != QUERY_PREFIX
    assert "文書" in DOC_PREFIX
    assert "クエリ" in QUERY_PREFIX
