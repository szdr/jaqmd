from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TICK_SECONDS = 0.1
_BAR_WIDTH = 24


class ProgressReporter:
    """検索処理の進捗を stderr にスピナー＋経過秒で表示する。

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

        stop = threading.Event()
        start = time.monotonic()

        def _spin() -> None:
            i = 0
            while not stop.is_set():
                elapsed = time.monotonic() - start
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                sys.stderr.write(f"\r{frame} {label}... ({elapsed:.1f}s)")
                sys.stderr.flush()
                i += 1
                stop.wait(_TICK_SECONDS)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            elapsed = time.monotonic() - start
            sys.stderr.write(f"\r\x1b[2K{label}... ({elapsed:.1f}s)\n")
            sys.stderr.flush()

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

        stop = threading.Event()
        start = time.monotonic()
        lock = threading.Lock()
        done = 0

        def _advance(step: int = 1) -> None:
            nonlocal done
            with lock:
                done += step

        def _render() -> str:
            with lock:
                current = done
            elapsed = time.monotonic() - start
            if total:
                pct = min(current / total, 1.0)
                filled = int(_BAR_WIDTH * pct)
                bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
                return f"{label}... [{bar}] {current}/{total} ({pct:.0%}) ({elapsed:.1f}s)"
            return f"{label}... {current} 件 ({elapsed:.1f}s)"

        def _spin() -> None:
            i = 0
            while not stop.is_set():
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                sys.stderr.write(f"\r\x1b[2K{frame} {_render()}")
                sys.stderr.flush()
                i += 1
                stop.wait(_TICK_SECONDS)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()
        try:
            yield _advance
        finally:
            stop.set()
            thread.join()
            sys.stderr.write(f"\r\x1b[2K{_render()}\n")
            sys.stderr.flush()


NULL_REPORTER = ProgressReporter(enabled=False)
