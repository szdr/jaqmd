from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .search.trisearch import SearchResult


def rerank(
    query: str,
    results: list["SearchResult"],
    *,
    n: Optional[int] = None,
) -> list["SearchResult"]:
    """結果をリランクして返す。

    現状は RRF 順をそのまま素通し（恒等フォールバック）。
    ruri-v3-reranker-310m の ONNX 統合は別タスク。
    実モデルを導入する際はこの関数内を差し替えること。

    Args:
        query: 検索クエリ（将来の reranker に渡すために受け取る）。
        results: RRF 融合済みの SearchResult リスト（スコア降順）。
        n: 上位 n 件に絞る場合は指定する。None なら全件返却。

    Returns:
        リランク済み（現状は入力順）の SearchResult リスト。
    """
    out = results if n is None else results[:n]
    return out
