from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TICK_SECONDS = 0.1


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


NULL_REPORTER = ProgressReporter(enabled=False)
