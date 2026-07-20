from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import TYPE_CHECKING, Optional

from .config import settings
from .modelcache import is_model_cached
from .paths import models_dir
from .progress import NULL_REPORTER, ProgressReporter

if TYPE_CHECKING:
    from .search.trisearch import SearchResult

RERANK_TOP_K = settings.rerank_top_k

DEFAULT_RERANKER = "default"

# reranker モデルレジストリ。fastembed の add_custom_model 登録に必要な情報を保持する。
# int8 版は model.onnx 単体（additional_files 無し）でフル版とファイル構成が異なるため、
# モデルごとに登録メタ情報を分けて持つ。
RERANKER_MODELS: dict[str, dict] = {
    "default": {
        "hf": settings.reranker_model,
        "model_file": "model.onnx",
        "additional_files": ["model.onnx.data"],
    },
    "int8": {
        "hf": "szdr/ruri-v3-reranker-310m-onnx_int8_arm64",
        "model_file": "model.onnx",
        "additional_files": None,
    },
}

_encoders: dict[str, object] = {}
_load_attempted: set[str] = set()


def _get_encoder(
    model: str = DEFAULT_RERANKER, reporter: Optional[ProgressReporter] = None
):
    """TextCrossEncoder をロードして返す。失敗時は None（恒等フォールバック用)。

    fastembed 未導入・モデルロード失敗時は例外を投げず None を返す。
    embed.py の _get_model() と同じカスタムモデル登録パターンを使う。
    モデルキーごとにエンコーダをキャッシュする（同一プロセス内で複数モデルを併用可能）。
    """
    global _encoders, _load_attempted
    reporter = reporter or NULL_REPORTER
    if model not in RERANKER_MODELS:
        print(
            f"警告: 未知の reranker モデル指定です（{model}）。'{DEFAULT_RERANKER}' にフォールバックします。",
            file=sys.stderr,
        )
        model = DEFAULT_RERANKER

    if model in _encoders:
        return _encoders[model]
    if model in _load_attempted:
        return None
    _load_attempted.add(model)

    try:
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        print(
            "警告: fastembed が見つかりません。reranker を無効化して続行します。\n"
            "→ pip install 'jaqmd[vector]' を実行すると reranker が有効になります。",
            file=sys.stderr,
        )
        return None

    spec = RERANKER_MODELS[model]
    cache_dir = models_dir()
    label = f"Reranker モデル({model})をロード中"
    if not is_model_cached(spec["hf"], spec["model_file"], str(cache_dir)):
        label += "(初回はダウンロードのため数分かかる場合があります)"
    try:
        with reporter.step(label):
            # ruri-v3-reranker-310m の ONNX 版カスタムモデル登録
            TextCrossEncoder.add_custom_model(
                model=model,
                sources=ModelSource(hf=spec["hf"]),
                model_file=spec["model_file"],
                additional_files=spec["additional_files"],
            )
            _encoders[model] = TextCrossEncoder(
                model_name=model, cache_dir=str(cache_dir), threads=os.cpu_count()
            )
    except Exception as e:
        print(
            f"警告: reranker モデルのロードに失敗しました（{e}）。無効化して続行します。",
            file=sys.stderr,
        )
        return None
    return _encoders[model]


def _doc_text(r: "SearchResult") -> str:
    """rerank に渡す文書テキスト。body があればそれ、なければ snippet。"""
    return r.body if r.body else r.snippet


def _sigmoid(x: float) -> float:
    """生ロジットを (0, 1) に写像する。オーバーフローを避ける安定版。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def rerank_scores(
    query: str,
    results: list["SearchResult"],
    *,
    enabled: bool = True,
    model: str = DEFAULT_RERANKER,
    reporter: Optional[ProgressReporter] = None,
) -> Optional[list[float]]:
    """results と同じ長さ・同じ順序の rerank スコア列（sigmoid 正規化済み）を返す。

    `rerank()` と異なりソートも score 差し替えもしない純粋関数。位置依存ブレンド
    （query._blend_scores）で rerankScore として使うために、融合順位を保ったまま
    スコアだけを取り出す用途。

    reranker は生ロジットを返すため、`1/rrfRank`（0-1）と加重合成できるよう
    sigmoid で (0, 1) に写像する。

    Args:
        query: 検索クエリ。
        results: rerank 対象の SearchResult リスト（RRF 融合順）。
        enabled: False なら reranker を使わず None を返す。
        model: 使用する reranker モデルキー（RERANKER_MODELS 参照。既定 "default"）。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。

    Returns:
        results と同順の sigmoid 正規化済みスコア列。
        enabled=False / fastembed 未導入 / モデルロード失敗 / 空入力 の場合は None
        （呼び出し側で rerankScore 抜きの degrade ブレンドに使う）。
    """
    reporter = reporter or NULL_REPORTER
    if not results or not enabled:
        return None

    encoder = _get_encoder(model, reporter)
    if encoder is None:
        return None

    with reporter.step(f"リランク ({len(results)} 件)"):
        raw = list(encoder.rerank(query, [_doc_text(r) for r in results]))
    return [_sigmoid(float(s)) for s in raw]


def rerank(
    query: str,
    results: list["SearchResult"],
    *,
    enabled: bool = True,
    model: str = DEFAULT_RERANKER,
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
        model: 使用する reranker モデルキー（RERANKER_MODELS 参照。既定 "default"）。
        top_k: 再スコア対象を先頭 top_k 件に限定する。None なら全件。
        n: 上位 n 件に絞る場合は指定する。None なら全件返却。
        reporter: 進捗表示用の ProgressReporter（None なら無効）。

    Returns:
        リランク済み（無効時は入力順）の SearchResult リスト。
    """
    reporter = reporter or NULL_REPORTER
    if not results or not enabled:
        return results if n is None else results[:n]

    encoder = _get_encoder(model, reporter)
    if encoder is None:
        return results if n is None else results[:n]

    if top_k is None:
        head, tail = results, []
    else:
        head, tail = results[:top_k], results[top_k:]

    with reporter.step(f"リランク ({len(head)} 件)"):
        scores = list(encoder.rerank(query, [_doc_text(r) for r in head]))
    rescored = [dataclasses.replace(r, score=float(s)) for r, s in zip(head, scores)]
    rescored.sort(key=lambda r: r.score, reverse=True)

    out = rescored + tail
    return out if n is None else out[:n]
