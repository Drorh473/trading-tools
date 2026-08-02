import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from core.storage import Storage
from weekly_review.analyze import analyze, render, start_of_week


def _backdate_trade(db_path, trade_id, when: date):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE trades SET תאריך = ? WHERE מספר_עסקה = ?", (when.isoformat(), trade_id))
    conn.commit()
    conn.close()


def _open_and_close(storage, symbol, direction, entry_price, size, stop, target, exit_price, strategy_tag):
    trade_id = storage.create_pending(symbol=symbol, direction=direction, strategy_tag=strategy_tag)
    storage.confirm_entry(
        trade_id, entry_price=entry_price, position_size=size, actual_stop=stop, actual_target=target, leverage=1.0
    )
    storage.close_trade(trade_id, exit_price=exit_price)
    return trade_id


def _resolved_signal(storage, db_path, dispatched_at, strategy_tag, direction, paper_r, decision=None):
    signal_id = storage.log_signal(
        symbol="BTCUSDT",
        direction=direction,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        strategy_tag=strategy_tag,
    )
    storage.record_paper_outcome(signal_id, paper_r)
    if decision:
        storage.mark_signal_decision(signal_id, decision)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE signals SET dispatched_at = ? WHERE id = ?", (dispatched_at, signal_id))
    conn.commit()
    conn.close()
    return signal_id


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)

    week_start = start_of_week()
    last_week = week_start - timedelta(days=1)
    this_week_ts = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    last_week_ts = datetime.combine(last_week, datetime.min.time(), tzinfo=timezone.utc).isoformat()

    # closed trade from before this week: a loss on strategy A
    old = _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=90, strategy_tag="A")
    _backdate_trade(db_path, old, last_week)

    # this week: two wins on strategy B
    _open_and_close(s, "ETHUSDT", "long", 50, 2, 45, 60, exit_price=60, strategy_tag="B")
    _open_and_close(s, "SOLUSDT", "short", 20, 5, 22, 14, exit_price=14, strategy_tag="B")

    # paper signals: one from last week, three this week across decisions
    _resolved_signal(s, db_path, last_week_ts, "Strategy 1 1H", "long", -1.0, decision="approved")
    _resolved_signal(s, db_path, this_week_ts, "Strategy 1 1H", "long", 2.0, decision="approved")
    _resolved_signal(s, db_path, this_week_ts, "Strategy 1 1H", "long", -1.0, decision="rejected")
    _resolved_signal(s, db_path, this_week_ts, "Strategy 2 1H/15m", "short", 1.5, decision=None)

    return s


def test_weekly_report_real_section(storage):
    report = analyze(storage)

    assert report.real_all_time.total_closed == 3
    assert report.real_this_week.total_closed == 2
    assert len(report.week_trades) == 2
    assert report.best_strategy_this_week == "B"
    assert report.current_streak_len == 2
    assert report.current_streak_type == "win"


def test_weekly_report_paper_section(storage):
    report = analyze(storage)

    assert report.paper_all_time.total_resolved == 4
    assert report.paper_this_week.total_resolved == 3
    assert report.paper_this_week.by_decision["approved"].count == 1
    assert report.paper_this_week.by_decision["rejected"].count == 1
    assert report.paper_this_week.by_decision["ignored"].count == 1
    assert "Strategy 1 1H long" in report.paper_this_week.by_strategy_direction
    assert "Strategy 2 1H/15m short" in report.paper_this_week.by_strategy_direction


def test_render_includes_both_sections(storage):
    text = render(analyze(storage))

    assert "## Real trades this week" in text
    assert "ETHUSDT long" in text
    assert "## Paper-simulated signals this week" in text
    assert "By decision" in text
    assert "rejected: 1 signals" in text


def test_weekly_report_with_nothing_this_week(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    old = _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=120, strategy_tag="A")
    _backdate_trade(db_path, old, week_start - timedelta(days=1))

    report = analyze(s)
    assert report.real_this_week.total_closed == 0
    assert report.best_strategy_this_week is None

    text = render(report)
    assert "None closed this week." in text
    assert "None resolved this week." in text


def test_start_of_week_is_sunday_not_monday():
    # Wednesday, Aug 5 2026 should back up to Sunday Aug 2, not Monday Aug 3 -
    # the ISO week (Monday-start) was the bug this replaces.
    wednesday = date(2026, 8, 5)
    result = start_of_week(wednesday)
    assert result == date(2026, 8, 2)
    assert result.weekday() == 6  # Sunday, in Python's Monday=0 numbering
