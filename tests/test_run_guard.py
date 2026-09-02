import pytest

from core.run_guard import notify_on_completion


def test_success_sends_one_done_message():
    sent = []

    with notify_on_completion("test job", alert_fn=sent.append):
        pass

    assert len(sent) == 1
    assert "test job" in sent[0]
    assert "DONE" in sent[0]


def test_success_includes_the_caller_supplied_headline():
    sent = []

    with notify_on_completion("S1 overnight", alert_fn=sent.append) as note:
        note.headline = "best arm: +0.031R net, confirmed +0.028R"

    assert "best arm: +0.031R net, confirmed +0.028R" in sent[0]


def test_success_with_no_headline_still_sends_a_message():
    sent = []

    with notify_on_completion("plain job", alert_fn=sent.append) as note:
        pass  # note.headline left unset

    assert len(sent) == 1
    assert "plain job" in sent[0]


def test_failure_sends_a_failed_message_and_reraises():
    sent = []

    with pytest.raises(RuntimeError, match="boom"):
        with notify_on_completion("test job", alert_fn=sent.append):
            raise RuntimeError("boom")

    assert len(sent) == 1
    assert "test job" in sent[0]
    assert "FAILED" in sent[0]
    assert "RuntimeError" in sent[0]
    assert "boom" in sent[0]


def test_a_broken_alert_fn_does_not_replace_the_original_exception():
    """weekly_review/main.py's own rule: a failing alert path must not
    swallow or replace the real error - the log is the last resort."""

    def _broken_alert(text):
        raise ConnectionError("telegram is down")

    with pytest.raises(RuntimeError, match="the real failure"):
        with notify_on_completion("test job", alert_fn=_broken_alert):
            raise RuntimeError("the real failure")


def test_a_broken_alert_fn_does_not_crash_the_success_path():
    def _broken_alert(text):
        raise ConnectionError("telegram is down")

    with notify_on_completion("test job", alert_fn=_broken_alert):
        pass  # must not raise


def test_elapsed_time_is_reported():
    sent = []

    with notify_on_completion("test job", alert_fn=sent.append):
        pass

    assert "m" in sent[0]  # minutes, however small
