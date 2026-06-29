from __future__ import annotations

from pathlib import Path

EMBED_MODEL = "cl-nagoya/ruri-v3-310m"
EMBED_DIM = 768
DOC_PREFIX = "検索文書: "
QUERY_PREFIX = "検索クエリ: "

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as e:
            raise ImportError(
                "fastembed が見つかりません。\n"
                "→ pip install 'jaqmd[vector]' を実行してください。"
            ) from e

        # ruri-v3-310m のカスタムモデル登録（ONNX 版）
        TextEmbedding.add_custom_model(
            model=EMBED_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="sirasagi62/ruri-v3-310m-ONNX"),
            dim=EMBED_DIM,
            model_file="onnx/model.onnx",
        )
        cache_dir = Path.home() / ".cache" / "jaqmd" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(cache_dir))
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """文書テキストのリストを embedding する（DOC_PREFIX を自前付与）。"""
    if not texts:
        return []
    model = _get_model()
    prefixed = [DOC_PREFIX + t for t in texts]
    return [list(v) for v in model.embed(prefixed)]


def embed_query(text: str) -> list[float]:
    """クエリテキストを embedding する（QUERY_PREFIX を自前付与）。"""
    model = _get_model()
    prefixed = QUERY_PREFIX + text
    return list(next(model.embed([prefixed])))


def count_tokens(text: str) -> int:
    """ruri-v3 トークナイザでテキストのトークン数を返す。

    fastembed の内部 tokenizer にアクセスできない場合は文字数ベースのフォールバックを使用する。
    """
    model = _get_model()
    # fastembed は TextEmbedding.model 内部に tokenizer を持つ
    # バージョンによりアクセスパスが異なるため複数の場所を試みる
    tokenizer = (
        getattr(model, "tokenizer", None)
        or getattr(getattr(model, "model", None), "tokenizer", None)
    )
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text)
            return len(tokens) if hasattr(tokens, "__len__") else len(list(tokens))
        except Exception:
            pass
    # フォールバック: 文字数ベース概算
    return len(text)
