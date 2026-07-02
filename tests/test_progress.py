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
