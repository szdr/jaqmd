from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text

_BAR_WIDTH = 24


class _ElapsedColumn(ProgressColumn):
    """経過秒を "(12.3s)" 形式で表示する。"""

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        return Text(f"({elapsed or 0.0:.1f}s)")


class _CountColumn(ProgressColumn):
    """総数不明時の走行カウンタ表示 "12 件"。"""

    def render(self, task: Task) -> Text:
        return Text(f"{int(task.completed)} 件")


def _columns(*, with_bar: bool, with_count: bool) -> tuple:
    """全コマンド共通の列構成: スピナー＋説明＋(バー or 件数)＋経過秒。"""
    columns: list = [SpinnerColumn(), TextColumn("{task.description}...")]
    if with_bar:
        columns += [
            BarColumn(bar_width=_BAR_WIDTH),
            MofNCompleteColumn(),
            TaskProgressColumn(),
        ]
    elif with_count:
        columns.append(_CountColumn())
    columns.append(_ElapsedColumn())
    return tuple(columns)


class ProgressReporter:
    """処理の進捗を stderr に rich.progress で表示する。

    enabled=False の場合は何も出力しない（デフォルトのフォールバック）。
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        """処理中はスピナーで経過秒をライブ更新し、終了時に確定行を残す。"""
        if not self.enabled:
            yield
            return

        console = Console(file=sys.stderr)
        with Progress(
            *_columns(with_bar=False, with_count=False),
            console=console,
            transient=False,
        ) as progress:
            progress.add_task(label)
            yield

    @contextmanager
    def track(
        self, label: str, total: Optional[int] = None
    ) -> Iterator[Callable[[int], None]]:
        """件数付きループの進捗を stderr にライブ表示する。

        yield された callable を各反復後に呼び出してカウントを進める
        （引数省略時は 1 進む）。total 指定時はバー＋件数＋%、
        total が None（総数不明）の場合は走行カウンタのみを表示する。
        """
        if not self.enabled:

            def _noop(step: int = 1) -> None:
                return None

            yield _noop
            return

        console = Console(file=sys.stderr)
        columns = _columns(with_bar=bool(total), with_count=not total)

        with Progress(*columns, console=console, transient=False) as progress:
            task_id = progress.add_task(label, total=total)

            def _advance(step: int = 1) -> None:
                progress.advance(task_id, step)

            yield _advance


NULL_REPORTER = ProgressReporter(enabled=False)
