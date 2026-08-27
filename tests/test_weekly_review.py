import dataclasses
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

import weekly_review.main as WM
from config import settings as real_settings
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


def test_render_no_longer_includes_the_removed_real_trades_section(storage):
    """Dror, 2026-08-27: "remove the real trade section". analyze() still
    computes real_this_week/real_all_time/etc (test_weekly_report_real_section
    still pins that), just render() no longer prints them."""
    text = render(analyze(storage))

    assert "## Real trades this week" not in text
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
    assert "## Real trades this week" not in text
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


def test_swing_slots_full_signals_are_tracked_but_no_longer_rendered(tmp_path):
    """The section itself was removed from the report (Dror, 2026-08-27:
    "remove... swing slots full"), but the underlying decision is still
    recorded in paper_this_week.by_decision (nothing about how signals are
    resolved or logged changed) - only render() stopped printing a section
    for it. Still must not pollute the main "resolved" headline: a
    suppressed signal is not a resolved one."""
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    this_week_ts = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()

    _resolved_signal(s, db_path, this_week_ts, "Strategy 1 1D", "long", -3.0, decision="swing_slots_full")

    report = analyze(s)
    assert report.paper_this_week.total_resolved == 0  # not polluted by the swing_slots_full outcome
    assert report.paper_this_week.by_decision["swing_slots_full"].count == 1  # still tracked, just not rendered

    text = render(report)
    assert "## Paper-simulated signals this week" in text
    assert "None resolved this week." in text
    assert "## Swing slots full this week" not in text
    assert "suppressed because both swing slots" not in text


def test_refused_at_fill_signals_get_their_own_section_and_dont_pollute_the_headline(tmp_path):
    """Dror, 2026-08-27: "add [a refused-at-fill section]" - the fee-
    domination gates added the same session produce this decision, and
    without a section for it, a signal a gate caught was invisible in the
    report (present in by_decision, per journal/stats.py's own docstring,
    just never rendered anywhere)."""
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    this_week_ts = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc).isoformat()

    _resolved_signal(s, db_path, this_week_ts, "Strategy 2.1 15m", "short", -1.2, decision="refused_at_fill")

    report = analyze(s)
    assert report.paper_this_week.total_resolved == 0  # not polluted by the refusal

    text = render(report)
    assert "## Paper-simulated signals this week" in text
    assert "None resolved this week." in text
    assert "## Refused at the fill this week" in text
    assert "1 signal(s)" in text
    assert "-1.20R" in text


def test_refused_at_fill_section_says_none_when_nothing_was_refused(storage):
    text = render(analyze(storage))
    assert "## Refused at the fill this week" in text
    assert "None this week." in text


class FakeBitgetFees:
    def __init__(self, total):
        self._total = total
        self.calls = []

    def get_fees_paid(self, start_ms, end_ms):
        self.calls.append((start_ms, end_ms))
        return self._total


def test_fee_paid_this_week_is_none_without_a_bitget_client(storage):
    """No bitget client means no network call was made, so the report must
    say "not available", not silently claim zero fees were paid."""
    report = analyze(storage)
    assert report.fee_paid_this_week is None

    text = render(report)
    assert "## Fees paid this week" in text
    assert "Not available this run." in text


def test_fee_paid_this_week_comes_from_bitgets_real_fills(storage):
    """The total is the REAL charged amount from Bitget's own fills, not
    sizing's estimate of it - queried for the exact week window."""
    bitget = FakeBitgetFees(11.44)

    report = analyze(storage, bitget=bitget)

    assert report.fee_paid_this_week == pytest.approx(11.44)
    assert len(bitget.calls) == 1
    start_ms, end_ms = bitget.calls[0]
    week_start_ts = datetime.combine(
        start_of_week(), datetime.min.time(), tzinfo=timezone.utc
    ).timestamp()
    # Jerusalem is ahead of UTC, so the local week start is a few hours
    # earlier in absolute time - just check it's the same calendar day's
    # start within a day's slack, not the exact UTC instant.
    assert abs(start_ms / 1000 - week_start_ts) < 86400
    assert end_ms > start_ms

    text = render(report)
    assert "## Fees paid this week" in text
    assert "$11.44" in text


def test_fee_by_strategy_is_estimated_from_each_strategys_own_fee_basis(tmp_path):
    """Each strategy's true round-trip fee differs (Strategy 2.1 is 100%
    market, Strategy 4 is 100% limit) - the estimate must use each trade's
    OWN strategy's fee basis, not one flat number for everyone (the exact
    bug notifier/scanner.py's sizing had until 2026-08-27)."""
    from notifier.risk_sizing import round_trip_fee_for
    from notifier.strategies.ema_trend_v2 import MARKET_FRACTION as V21_FRACTION
    from notifier.strategies.order_block import MARKET_FRACTION as S4_FRACTION

    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)

    _open_and_close(s, "BTCUSDT", "long", 100.0, 1.0, 95.0, 110.0, exit_price=110.0, strategy_tag="Strategy 2.1 15m")
    _open_and_close(s, "ETHUSDT", "long", 50.0, 2.0, 48.0, 56.0, exit_price=56.0, strategy_tag="Strategy 4 1H OB2.0")

    report = analyze(s)

    v21_count, v21_fee = report.fee_by_strategy["Strategy 2.1 15m"]
    s4_count, s4_fee = report.fee_by_strategy["Strategy 4 1H OB2.0"]

    assert v21_count == 1
    assert v21_fee == pytest.approx(round_trip_fee_for(V21_FRACTION) * 100.0 * 1.0)
    assert s4_count == 1
    assert s4_fee == pytest.approx(round_trip_fee_for(S4_FRACTION) * 50.0 * 2.0)
    # Strategy 2.1 is 100% market (taker both legs) and Strategy 4 is 100%
    # limit (maker in) - same notional-equivalent trade must NOT cost the
    # same fee, or the estimate collapsed back to one flat assumption.
    assert (v21_fee / 100.0) != pytest.approx(s4_fee / 100.0)

    text = render(report)
    assert "Strategy 2.1 15m: 1 trade(s)" in text
    assert "Strategy 4 1H OB2.0: 1 trade(s)" in text


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
    assert "No scan was missed or late" in text and "Watched continuously for" in text


def test_the_weekly_run_prunes_heartbeats_it_will_never_read(tmp_path):
    """prune_heartbeats existed and nothing called it, so the table grew
    forever - ~96 rows a day, indefinitely, for a report that never looks
    further back than one week. A retention policy that is never applied is
    the same as not having one.
    """
    import time

    from core.storage import HEARTBEAT_PRUNE_DAYS
    from weekly_review.analyze import prune_stale_heartbeats

    storage = Storage(str(tmp_path / "trades.db"))
    now = time.time()
    old = now - (HEARTBEAT_PRUNE_DAYS + 5) * 86400
    recent = now - 3600
    storage.record_heartbeat(old, old + 900)
    storage.record_heartbeat(recent, recent + 900)

    prune_stale_heartbeats(storage, now=now)

    span = storage.heartbeat_span()
    assert span is not None, "the recent heartbeat must survive"
    assert span[0] == recent, "everything older than the retention window goes"


def _fake_settings(tmp_path):
    return dataclasses.replace(
        real_settings,
        trades_db_path=str(tmp_path / "trades.db"),
        telegram_bot_token="test-token",
        telegram_chat_id="test-chat",
    )


def test_report_today_is_the_day_before_now_not_today(monkeypatch):
    """The cron fires Sunday 20:00 Jerusalem - at that instant, "today" is day
    ZERO of a new week, and start_of_week(today) resolves to itself, so the
    report would cover only the hours elapsed since midnight rather than the
    week that just finished. Proven wrong against a REAL production log: a
    past Sunday run logged "Service started 1x this week: Sun 11:32...
    Watched continuously for 20.0h" - a 20-hour "week". Subtracting a day
    makes a Sunday-evening run resolve back to the Sunday that started the
    week which just ended."""
    from datetime import date, datetime

    class _FixedNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 30, 20, 0, 0, tzinfo=tz)  # a real Sunday, 20:00

    monkeypatch.setattr(WM, "datetime", _FixedNow)

    assert WM._report_today() == date(2026, 8, 29)


def test_main_asks_analyze_for_the_week_that_just_ended(tmp_path, monkeypatch):
    """main() must pass _report_today()'s result into analyze(), not let it
    default to "right now" - otherwise the fix above never actually reaches
    the report."""
    from datetime import date

    captured = {}

    def _spy_analyze(storage, today=None, bitget=None):
        captured["today"] = today
        return "fake-report"

    fake_settings = _fake_settings(tmp_path)
    monkeypatch.setattr(WM, "settings", fake_settings)
    monkeypatch.setattr(WM, "client_from_settings", lambda settings: "fake-bitget-client")
    monkeypatch.setattr(WM, "resolve_pending", lambda storage, bitget: 3)
    monkeypatch.setattr(WM, "analyze", _spy_analyze)
    monkeypatch.setattr(WM, "render", lambda report: "REPORT TEXT")
    monkeypatch.setattr(WM, "send_message", lambda token, chat_id, text: None)
    monkeypatch.setattr(WM, "_report_today", lambda: date(2026, 8, 29))

    WM.main()

    assert captured["today"] == date(2026, 8, 29)


def test_main_sends_the_report_and_records_success(tmp_path, monkeypatch):
    """The ordinary path: build the report, send it, then record that this run
    happened - via both watchers (heartbeat.py's file and the capability
    ledger), since notifier/scanner.py checks the former and the weekly
    report itself checks the latter."""
    from core import ledger
    from weekly_review import heartbeat

    sent = []
    fake_settings = _fake_settings(tmp_path)
    monkeypatch.setattr(WM, "settings", fake_settings)
    monkeypatch.setattr(WM, "client_from_settings", lambda settings: "fake-bitget-client")
    monkeypatch.setattr(WM, "resolve_pending", lambda storage, bitget: 3)
    monkeypatch.setattr(WM, "analyze", lambda storage, today=None, bitget=None: "fake-report")
    monkeypatch.setattr(WM, "render", lambda report: "REPORT TEXT")
    monkeypatch.setattr(WM, "send_message", lambda token, chat_id, text: sent.append(text))

    WM.main()

    assert len(sent) == 1
    assert "REPORT TEXT" in sent[0]
    assert heartbeat.last_success(fake_settings.trades_db_path) is not None
    assert ledger.last_success(fake_settings.trades_db_path, ledger.WEEKLY_REPORT) is not None


def test_main_alerts_and_reraises_when_the_report_fails(tmp_path, monkeypatch):
    """The exact fix for the 2026-08-02 incident: two weeks of missing reports
    that left no trace but a traceback nobody read. A failure must both reach
    Telegram and still fail the run (non-zero exit) - and must not reach the
    success bookkeeping below it."""
    from core import ledger
    from weekly_review import heartbeat

    sent = []
    fake_settings = _fake_settings(tmp_path)
    monkeypatch.setattr(WM, "settings", fake_settings)
    monkeypatch.setattr(WM, "client_from_settings", lambda settings: "fake-bitget-client")
    monkeypatch.setattr(WM, "resolve_pending", lambda storage, bitget: 3)

    def _boom(storage, today=None, bitget=None):
        raise RuntimeError("simulated Bitget 400")

    monkeypatch.setattr(WM, "analyze", _boom)
    monkeypatch.setattr(WM, "send_message", lambda token, chat_id, text: sent.append(text))

    with pytest.raises(RuntimeError, match="simulated Bitget 400"):
        WM.main()

    assert len(sent) == 1
    assert "WEEKLY REPORT FAILED" in sent[0]
    assert "RuntimeError" in sent[0]
    assert "simulated Bitget 400" in sent[0]
    assert heartbeat.last_success(fake_settings.trades_db_path) is None
    assert ledger.last_success(fake_settings.trades_db_path, ledger.WEEKLY_REPORT) is None


def test_a_failing_alert_does_not_swallow_the_original_error(tmp_path, monkeypatch, capsys):
    """The failure alert is itself a network call and can itself fail. If that
    replaced the original exception, the log would show a Telegram error where
    the real cause - the thing that actually broke - used to be."""
    fake_settings = _fake_settings(tmp_path)
    monkeypatch.setattr(WM, "settings", fake_settings)
    monkeypatch.setattr(WM, "client_from_settings", lambda settings: "fake-bitget-client")
    monkeypatch.setattr(WM, "resolve_pending", lambda storage, bitget: 3)

    def _boom(storage, today=None, bitget=None):
        raise RuntimeError("simulated Bitget 400")

    def _alert_boom(token, chat_id, text):
        raise ConnectionError("telegram is also down")

    monkeypatch.setattr(WM, "analyze", _boom)
    monkeypatch.setattr(WM, "send_message", _alert_boom)

    with pytest.raises(RuntimeError, match="simulated Bitget 400"):
        WM.main()

    # the alert's own failure still reaches the log, as the last resort
    assert "telegram is also down" in capsys.readouterr().err


def test_main_still_records_success_when_heartbeat_pruning_fails(tmp_path, monkeypatch):
    """Housekeeping runs after the report is already out; it must not be able
    to take the run's success bookkeeping down with it."""
    from core import ledger
    from weekly_review import heartbeat

    fake_settings = _fake_settings(tmp_path)
    monkeypatch.setattr(WM, "settings", fake_settings)
    monkeypatch.setattr(WM, "client_from_settings", lambda settings: "fake-bitget-client")
    monkeypatch.setattr(WM, "resolve_pending", lambda storage, bitget: 3)
    monkeypatch.setattr(WM, "analyze", lambda storage, today=None, bitget=None: "fake-report")
    monkeypatch.setattr(WM, "render", lambda report: "REPORT TEXT")
    monkeypatch.setattr(WM, "send_message", lambda token, chat_id, text: None)

    def _prune_boom(storage):
        raise RuntimeError("simulated prune failure")

    monkeypatch.setattr(WM, "prune_stale_heartbeats", _prune_boom)

    WM.main()  # must not raise

    assert heartbeat.last_success(fake_settings.trades_db_path) is not None
    assert ledger.last_success(fake_settings.trades_db_path, ledger.WEEKLY_REPORT) is not None


def test_the_report_names_a_restart_that_cost_no_scan(tmp_path):
    """A bounce inside a sleep window misses nothing, so it shows no gap - and
    still has to appear, because Dror asked to know the bot was down at all."""
    from datetime import datetime

    from core import clock
    from weekly_review.analyze import analyze, render

    storage = Storage(str(tmp_path / "trades.db"))
    t = datetime.now(clock.LOCAL_TZ).replace(hour=3, minute=0, second=0, microsecond=0).timestamp()
    for n in range(6):
        storage.record_heartbeat(t + n * 900, t + (n + 1) * 900)
    storage.record_service_start(t + 200)

    text = render(analyze(storage))
    assert "Service started 1x this week" in text
    assert "No scan was missed or late" in text, "nothing was actually missed"
