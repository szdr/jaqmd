from __future__ import annotations

from typing import Optional

from .config import settings
from .modelcache import is_model_cached
from .paths import models_dir
from .progress import NULL_REPORTER, ProgressReporter

EMBED_MODEL = settings.embed_model
EMBED_DIM = 768
DOC_PREFIX = "検索文書: "
QUERY_PREFIX = "検索クエリ: "

_model = None


def _get_model(reporter: Optional[ProgressReporter] = None):
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

        reporter = reporter or NULL_REPORTER
        cache_dir = models_dir()
        label = "Embedding モデルをロード中"
        if not is_model_cached(EMBED_MODEL, "onnx/model.onnx", str(cache_dir)):
            label += "（初回はダウンロードのため数分かかる場合があります）"
        with reporter.step(label):
            # ruri-v3-310m のカスタムモデル登録（ONNX 版）
            TextEmbedding.add_custom_model(
                model=EMBED_MODEL,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=EMBED_MODEL),
                dim=EMBED_DIM,
                model_file="onnx/model.onnx",
            )
            _model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(cache_dir))
    return _model


def embed_documents(
    texts: list[str],
    *,
    batch_size: int = 1,
    reporter: Optional[ProgressReporter] = None,
):
    """文書テキストのリストを embedding する（DOC_PREFIX を自前付与）。

    Args:
        texts: 文書テキストのリスト。
        batch_size: fastembed に渡すバッチサイズ。大きいほど高速だがメモリを使う。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。
    """
    if not texts:
        return []
    model = _get_model(reporter)
    prefixed = [DOC_PREFIX + t for t in texts]
    return model.embed(prefixed, batch_size=batch_size)


def embed_query(text: str, *, reporter: Optional[ProgressReporter] = None) -> list[float]:
    """クエリテキストを embedding する（QUERY_PREFIX を自前付与）。"""
    reporter = reporter or NULL_REPORTER
    model = _get_model(reporter)
    with reporter.step("クエリをベクトル化"):
        prefixed = QUERY_PREFIX + text
        return list(next(model.embed([prefixed])))


def count_tokens(text: str) -> int:
    """ruri-v3 トークナイザでテキストのトークン数を返す。

    fastembed の内部 tokenizer にアクセスできない場合は文字数ベースのフォールバックを使用する。
    """
    model = _get_model()
    # fastembed は TextEmbedding.model 内部に tokenizer を持つ
    # バージョンによりアクセスパスが異なるため複数の場所を試みる
    tokenizer = getattr(model, "tokenizer", None) or getattr(
        getattr(model, "model", None), "tokenizer", None
    )
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text)
            return len(tokens) if hasattr(tokens, "__len__") else len(list(tokens))
        except Exception:
            pass
    # フォールバック: 文字数ベース概算
    return len(text)
