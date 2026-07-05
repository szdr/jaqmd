from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .progress import NULL_REPORTER, ProgressReporter

if TYPE_CHECKING:
    from .search.trisearch import SearchResult

RERANK_MODEL = "szdr/ruri-v3-reranker-310m-onnx"
RERANK_TOP_K = 50

_encoder = None
_encoder_load_attempted = False


def _get_encoder():
    """TextCrossEncoder をロードして返す。失敗時は None（恒等フォールバック用）。

    fastembed 未導入・モデルロード失敗時は例外を投げず None を返す。
    embed.py の _get_model() と同じカスタムモデル登録パターンを使う。
    """
    global _encoder, _encoder_load_attempted
    if _encoder is not None:
        return _encoder
    if _encoder_load_attempted:
        return None
    _encoder_load_attempted = True

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from fastembed.common.model_description import ModelSource
    except ImportError:
        print(
            "警告: fastembed が見つかりません。reranker を無効化して続行します。\n"
            "→ pip install 'jaqmd[vector]' を実行すると reranker が有効になります。",
            file=sys.stderr,
        )
        return None

    try:
        # ruri-v3-reranker-310m の ONNX 版カスタムモデル登録
        TextCrossEncoder.add_custom_model(
            model=RERANK_MODEL,
            sources=ModelSource(hf="szdr/ruri-v3-reranker-310m-onnx"),
            model_file="model.onnx",
            additional_files=["model.onnx.data"],
        )
        cache_dir = Path.home() / ".cache" / "jaqmd" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _encoder = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=str(cache_dir))
    except Exception as e:
        print(
            f"警告: reranker モデルのロードに失敗しました（{e}）。無効化して続行します。",
            file=sys.stderr,
        )
        return None
    return _encoder


def _doc_text(r: "SearchResult") -> str:
    """rerank に渡す文書テキスト。body があればそれ、なければ snippet。"""
    return r.body if r.body else r.snippet


def rerank(
    query: str,
    results: list["SearchResult"],
    *,
    enabled: bool = True,
    top_k: Optional[int] = RERANK_TOP_K,
    n: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None,
) -> list["SearchResult"]:
    """結果を ruri-v3-reranker（cross-encoder）でリランクして返す。

    fastembed 未導入・モデルロード失敗・enabled=False の場合は恒等（入力順）
    で返す（degrade）。

    Args:
        query: 検索クエリ。
        results: RRF 融合済みの SearchResult リスト（スコア降順）。
        enabled: False なら reranker を使わず恒等フォールバック。
        top_k: 再スコア対象を先頭 top_k 件に限定する。None なら全件。
        n: 上位 n 件に絞る場合は指定する。None なら全件返却。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。

    Returns:
        リランク済み（無効時は入力順）の SearchResult リスト。
    """
    reporter = reporter or NULL_REPORTER
    if not results or not enabled:
        return results if n is None else results[:n]

    encoder = _get_encoder()
    if encoder is None:
        return results if n is None else results[:n]

    if top_k is None:
        head, tail = results, []
    else:
        head, tail = results[:top_k], results[top_k:]

    with reporter.step(f"リランク ({len(head)} 件)"):
        scores = list(encoder.rerank(query, [_doc_text(r) for r in head]))
    rescored = [
        dataclasses.replace(r, score=float(s))
        for r, s in zip(head, scores)
    ]
    rescored.sort(key=lambda r: r.score, reverse=True)

    out = rescored + tail
    return out if n is None else out[:n]
