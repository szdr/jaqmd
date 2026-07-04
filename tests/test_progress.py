from __future__ import annotations

from jaqmd.progress import NULL_REPORTER, ProgressReporter


def test_disabled_reporter_emits_nothing(capsys):
    """enabled=False の場合、step() は何も出力しない。"""
    with NULL_REPORTER.step("何か処理"):
        pass
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_disabled_reporter_still_runs_body():
    """enabled=False でも with ブロックの中身は実行される。"""
    ran = False
    with NULL_REPORTER.step("何か処理"):
        ran = True
    assert ran


def test_enabled_reporter_writes_final_line(capsys):
    """enabled=True の場合、確定行にラベルと経過秒 (s)) が含まれる。"""
    reporter = ProgressReporter(enabled=True)
    with reporter.step("テスト処理"):
        pass
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "テスト処理" in captured.err
    assert "s)" in captured.err


def test_enabled_reporter_propagates_exception(capsys):
    """例外が発生してもスピナースレッドが確実に停止し、例外は伝播する。"""
    reporter = ProgressReporter(enabled=True)
    try:
        with reporter.step("失敗する処理"):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError が伝播しなかった")


def test_disabled_reporter_track_emits_nothing(capsys):
    """enabled=False の場合、track() は何も出力しない。"""
    with NULL_REPORTER.track("何か処理", total=3) as advance:
        advance()
        advance()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_disabled_reporter_track_advance_is_noop():
    """enabled=False でも advance() 呼び出しは安全に無視される。"""
    with NULL_REPORTER.track("何か処理") as advance:
        advance()
        advance(2)


def test_enabled_reporter_track_writes_final_line_with_total(capsys):
    """total 指定時、確定行に件数・% ・経過秒が含まれる。"""
    reporter = ProgressReporter(enabled=True)
    with reporter.track("テスト処理", total=2) as advance:
        advance()
        advance()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "テスト処理" in captured.err
    assert "2/2" in captured.err
    assert "100%" in captured.err
    assert "s)" in captured.err


def test_enabled_reporter_track_writes_count_without_total(capsys):
    """total 未指定時は件数のみの走行カウンタ表示になる。"""
    reporter = ProgressReporter(enabled=True)
    with reporter.track("テスト処理") as advance:
        advance()
        advance()
        advance()
    captured = capsys.readouterr()
    assert "3 件" in captured.err
    assert "s)" in captured.err


def test_enabled_reporter_track_propagates_exception():
    """track() 内で例外が発生してもスピナースレッドが停止し、例外は伝播する。"""
    reporter = ProgressReporter(enabled=True)
    try:
        with reporter.track("失敗する処理", total=5) as advance:
            advance()
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError が伝播しなかった")
