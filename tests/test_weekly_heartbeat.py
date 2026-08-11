"""The weekly report must not be able to fail silently.

It ran on schedule every Sunday from 2026-08-02, crashed before sending every
time, and nobody knew for two weeks - cron appends to a log file nobody reads,
and an absent report looks exactly like a quiet week.
"""

from datetime import datetime, timedelta, timezone

import pytest

from weekly_review import heartbeat


def test_a_report_that_has_never_run_is_not_reported_as_late(tmp_path):
    """None means "no record", which is what the first two weeks looked like.

    Treating that as overdue would have alerted every hour from the moment the
    feature shipped, before a single Sunday had come round.
    """
    db = str(tmp_path / "trades.db")

    assert heartbeat.last_success(db) is None
    assert heartbeat.overdue_by(db, 8.0) is None


def test_a_recent_success_is_not_overdue(tmp_path):
    db = str(tmp_path / "trades.db")
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    heartbeat.record_success(db, now=now - timedelta(days=2))

    assert heartbeat.overdue_by(db, 8.0, now=now) is None


def test_a_missed_sunday_is_reported_as_overdue(tmp_path):
    """One skipped run, not a late one. The window is 8 days precisely so a
    report that arrives a few hours late is silent."""
    db = str(tmp_path / "trades.db")
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    heartbeat.record_success(db, now=now - timedelta(days=9))

    overdue = heartbeat.overdue_by(db, 8.0, now=now)

    assert overdue == pytest.approx(1.0)


def test_a_run_a_few_hours_late_does_not_alert(tmp_path):
    db = str(tmp_path / "trades.db")
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    heartbeat.record_success(db, now=now - timedelta(days=7, hours=6))

    assert heartbeat.overdue_by(db, 8.0, now=now) is None


def test_the_marker_survives_a_round_trip(tmp_path):
    db = str(tmp_path / "nested" / "trades.db")
    when = datetime(2026, 8, 9, 17, 0, tzinfo=timezone.utc)

    heartbeat.record_success(db, now=when)

    assert heartbeat.last_success(db) == when


def test_a_corrupt_marker_reads_as_never_run_rather_than_raising(tmp_path):
    """This is watched by the always-on scanner loop. A garbled file must not
    take down the thing whose job is to notice problems."""
    db = str(tmp_path / "trades.db")
    heartbeat.record_success(db)
    (tmp_path / "weekly_review_last_success").write_text("not a timestamp", encoding="utf-8")

    assert heartbeat.last_success(db) is None
    assert heartbeat.overdue_by(db, 8.0) is None
