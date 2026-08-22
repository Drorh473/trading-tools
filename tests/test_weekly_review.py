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
    assert "## Too small to execute this week" in text
    assert "None this week." in text


def test_too_small_signals_get_their_own_section_and_dont_pollute_the_headline(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    this_week_ts = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()

    # No normal resolved signals this week at all - only a too_small one. The
    # section must still render independently of the main paper section's
    # "None resolved" early return.
    _resolved_signal(s, db_path, this_week_ts, "Strategy 1 1D", "short", -5.0, decision="too_small")

    report = analyze(s)
    assert report.paper_this_week.total_resolved == 0  # not polluted by the too_small outcome

    text = render(report)
    assert "## Paper-simulated signals this week" in text
    assert "None resolved this week." in text
    assert "## Too small to execute this week" in text
    assert "1 signal(s) couldn't be split-entered" in text
    assert "-5.00R" in text


def test_swing_slots_full_signals_get_their_own_section_and_dont_pollute_the_headline(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    this_week_ts = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()

    _resolved_signal(s, db_path, this_week_ts, "Strategy 1 1D", "long", -3.0, decision="swing_slots_full")

    report = analyze(s)
    assert report.paper_this_week.total_resolved == 0  # not polluted by the swing_slots_full outcome

    text = render(report)
    assert "## Paper-simulated signals this week" in text
    assert "None resolved this week." in text
    assert "## Swing slots full this week" in text
    assert "1 signal(s) suppressed because both swing slots were taken" in text
    assert "-3.00R" in text


def test_start_of_week_is_sunday_not_monday():
    # Wednesday, Aug 5 2026 should back up to Sunday Aug 2, not Monday Aug 3 -
    # the ISO week (Monday-start) was the bug this replaces.
    wednesday = date(2026, 8, 5)
    result = start_of_week(wednesday)
    assert result == date(2026, 8, 2)
    assert result.weekday() == 6  # Sunday, in Python's Monday=0 numbering


def test_the_review_counter_counts_only_closed_trades_of_the_watched_tag(tmp_path):
    """Strategy 1 1H was left live and unchanged on 2026-08-20, to be revisited
    at 100 closed trades. That decision had no mechanism behind it, and
    "revisit later" with nothing counting becomes never."""
    from weekly_review.analyze import REVIEW_THRESHOLD, analyze, render

    storage = Storage(str(tmp_path / "trades.db"))
    for i in range(3):
        tid = storage.create_pending(symbol=f"A{i}USDT", direction="long",
                                     strategy_tag="Strategy 1 1H")
        storage.confirm_entry(tid, entry_price=100.0, position_size=1.0,
                              actual_stop=99.0, actual_target=102.0, leverage=10.0)
        storage.close_trade(tid, exit_price=102.0)
    # Still open, and a different strategy: neither counts.
    open_id = storage.create_pending(symbol="BUSDT", direction="long",
                                     strategy_tag="Strategy 1 1H")
    storage.confirm_entry(open_id, entry_price=100.0, position_size=1.0,
                          actual_stop=99.0, actual_target=102.0, leverage=10.0)
    other = storage.create_pending(symbol="CUSDT", direction="long",
                                   strategy_tag="Strategy 2.1 1H")
    storage.confirm_entry(other, entry_price=100.0, position_size=1.0,
                          actual_stop=99.0, actual_target=102.0, leverage=10.0)
    storage.close_trade(other, exit_price=102.0)

    report = analyze(storage)
    assert report.review_progress == {"Strategy 1 1H": 3}
    assert f"3 of {REVIEW_THRESHOLD} closed trades toward review" in render(report)


def test_reaching_the_threshold_says_so_loudly(tmp_path, monkeypatch):
    """The point is that it announces itself rather than being remembered."""
    from weekly_review import analyze as A

    monkeypatch.setattr(A, "REVIEW_THRESHOLD", 2)
    storage = Storage(str(tmp_path / "trades.db"))
    for i in range(2):
        tid = storage.create_pending(symbol=f"A{i}USDT", direction="long",
                                     strategy_tag="Strategy 1 1H")
        storage.confirm_entry(tid, entry_price=100.0, position_size=1.0,
                              actual_stop=99.0, actual_target=102.0, leverage=10.0)
        storage.close_trade(tid, exit_price=102.0)

    text = A.render(A.analyze(storage))
    assert "THE REVIEW IS DUE" in text


def test_the_report_names_every_gap_however_small(tmp_path):
    """Dror asked for it to report "if the bot was down for even a small time",
    so a restart shows up too - a deploy IS a window where nothing watched the
    market. The duration is printed so a short bounce reads differently from a
    real outage."""
    from datetime import datetime

    from core import clock
    from weekly_review.analyze import analyze, render

    storage = Storage(str(tmp_path / "trades.db"))
    start = datetime.now(clock.LOCAL_TZ).replace(hour=1, minute=0, second=0, microsecond=0)
    t = start.timestamp()
    for n in range(6):                       # six on-time cycles
        storage.record_heartbeat(t + n * 900, t + (n + 1) * 900)
    storage.record_heartbeat(t + 5 * 900 + 3000, t + 5 * 900 + 3900)  # a gap

    text = render(analyze(storage))
    assert "## Bot availability" in text
    assert "1 gap(s)" in text
    assert "unwatched" in text
    assert "% up)" in text


def test_a_week_with_no_heartbeat_says_unknown_not_perfect(tmp_path):
    """The capability ledger's "all watched capabilities have worked recently"
    was true by its rules and false in substance. This must not repeat it."""
    from weekly_review.analyze import analyze, render

    storage = Storage(str(tmp_path / "trades.db"))
    text = render(analyze(storage))
    assert "UNKNOWN, not perfect" in text


def test_a_clean_week_says_how_long_it_actually_watched(tmp_path):
    """"No gaps" over two hours of heartbeat is a different claim from "no
    gaps" over a full week, so the covered period is always stated."""
    from datetime import datetime

    from core import clock
    from weekly_review.analyze import analyze, render

    storage = Storage(str(tmp_path / "trades.db"))
    t = datetime.now(clock.LOCAL_TZ).replace(hour=2, minute=0, second=0, microsecond=0).timestamp()
    for n in range(9):
        storage.record_heartbeat(t + n * 900, t + (n + 1) * 900)
    text = render(analyze(storage))
    assert "No gaps" in text and "Watched continuously for" in text
