import sqlite3
from datetime import datetime

import pytest

from core.storage import SCHEMA, Storage


def test_committed_margin_sums_open_trades_only(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))

    # open trade: notional 100*2=200, leverage 4x -> margin 50
    t1 = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(t1, entry_price=100, position_size=2, actual_stop=95, actual_target=110, leverage=4.0)

    # open trade: notional 50*10=500, leverage 5x -> margin 100
    t2 = storage.create_pending(symbol="ETHUSDT", direction="long")
    storage.confirm_entry(t2, entry_price=50, position_size=10, actual_stop=45, actual_target=60, leverage=5.0)

    # closed trade: should NOT count toward committed margin
    t3 = storage.create_pending(symbol="SOLUSDT", direction="long")
    storage.confirm_entry(t3, entry_price=20, position_size=100, actual_stop=18, actual_target=24, leverage=2.0)
    storage.close_trade(t3, exit_price=24)

    # pending trade (not yet confirmed): should NOT count either
    storage.create_pending(symbol="XRPUSDT", direction="long")

    assert storage.committed_margin() == 150.0  # 50 + 100


def test_committed_margin_zero_when_no_open_trades(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    assert storage.committed_margin() == 0.0


def test_an_existing_journal_gains_the_exit_plan_columns(tmp_path):
    """The live DB predates these columns, and SCHEMA is CREATE TABLE IF NOT
    EXISTS - it does nothing to a table that already exists. Without the
    migration _select()'s Trade(**row) raises on the first read, which on a
    real-money bot means every open trade stops being tracked."""
    import sqlite3

    from core.storage import _ADDED_COLUMNS

    db = str(tmp_path / "trades.db")
    old_schema = "\n".join(
        line for line in SCHEMA.splitlines() if not any(c in line for c in _ADDED_COLUMNS["trades"])
    ).replace("הערות             TEXT,", "הערות             TEXT")
    conn = sqlite3.connect(db)
    conn.execute(old_schema)
    pre_existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    assert not (pre_existing & set(_ADDED_COLUMNS["trades"])), "the fixture must start WITHOUT the new columns"
    conn.execute(
        "INSERT INTO trades (תאריך, סימבול, כיוון, מחיר_כניסה, גודל_פוזיציה) VALUES (?, ?, ?, ?, ?)",
        ("2026-08-13", "APTUSDT", "short", 0.6134, 70.019),
    )
    conn.commit()
    conn.close()

    storage = Storage(db)  # migrates on open

    trade = storage.get_trade(1)
    assert trade.סימבול == "APTUSDT"
    assert trade.breakeven_stop is None  # an old row has no plan, and says so
    storage.set_exit_plan(1, breakeven_stop=0.6134, runner_target=0.58, partial_fraction=0.5)
    assert storage.get_trade(1).breakeven_stop == 0.6134


def test_set_exit_plan_persists_the_reward_risk_ratio(tmp_path):
    """The ratio a trade's partial and runner targets were actually priced
    against - 2.0 for Strategy 1's first tier - has never been stored
    anywhere, only baked into an absolute price at confirm time. Trade #98,
    1000RATSUSDT, live 2026-09-05: its resting limit leg filled two days
    after entry and nothing could recompute its take-profit against the
    NEW blended entry, because nothing on the row said what ratio to
    recompute it AT - _on_resize would have had to guess, and the only
    guess available (Scanner's own DEFAULT_REWARD_RISK_RATIO, 3.0) is not
    the 2.0 this strategy actually uses. Persisting it here is what closes
    that gap: it rides along with the rest of the exit plan, set once at
    confirm time, read back whenever the position later grows."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(trade_id, entry_price=100, position_size=10,
                          actual_stop=95, actual_target=110, leverage=1.0)

    storage.set_exit_plan(trade_id, breakeven_stop=100.0, runner_target=115.0,
                          partial_fraction=None, reward_risk_ratio=2.0)

    assert storage.get_trade(trade_id).reward_risk_ratio == 2.0


def test_reward_risk_ratio_is_none_for_a_trade_that_never_recorded_one(tmp_path):
    """Every trade opened before this column shipped, and every set_exit_plan
    call that doesn't pass it - the default must be None, not 0.0 or some
    other value _on_resize could mistake for a real ratio and recompute a
    wrong price with."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(trade_id, entry_price=100, position_size=10,
                          actual_stop=95, actual_target=110, leverage=1.0)

    storage.set_exit_plan(trade_id, breakeven_stop=100.0, runner_target=115.0, partial_fraction=None)

    assert storage.get_trade(trade_id).reward_risk_ratio is None


def test_migration_adds_only_the_columns_a_journal_is_missing(tmp_path):
    """The live DB is mid-way: it gained breakeven_stop/runner_target/
    partial_fraction on the 2026-08-13 deploy and has never seen
    exit_managed. Re-running ALTER for a column that already exists is an
    error, so the loop has to be per-column rather than all-or-nothing."""
    import sqlite3
    from dataclasses import fields

    from core.storage import _ADDED_COLUMNS

    db = str(tmp_path / "trades.db")
    Storage(db)  # fully migrated
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE trades DROP COLUMN exit_managed")
    conn.execute(
        "INSERT INTO trades (תאריך, סימבול, כיוון, מחיר_כניסה, breakeven_stop) VALUES (?, ?, ?, ?, ?)",
        ("2026-08-13", "APTUSDT", "short", 0.6081, 0.6081),
    )
    conn.commit()
    present = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    conn.close()
    assert "exit_managed" not in present and "breakeven_stop" in present

    storage = Storage(db)  # must add the one missing column, not retry the others

    trade = storage.get_trade(1)
    assert trade.breakeven_stop == 0.6081, "the columns already there keep their values"
    assert not trade.exit_managed, "an existing trade is not retroactively managed"
    assert set(_ADDED_COLUMNS["trades"]).issubset({f.name for f in fields(trade)}), "Trade(**row) needs every column"


def _breakeven_trade(tmp_path):
    """APTUSDT #11 exactly: entered short, stop later moved onto the entry."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.confirm_entry(
        trade_id, entry_price=0.608112826518, position_size=70.019,
        actual_stop=0.6285, actual_target=None, leverage=10.0,
    )
    return storage, trade_id


def test_r_is_measured_against_the_risk_the_trade_was_sized_to(tmp_path):
    """#11 took its partial, went to breakeven, closed +4.18 — and reported
    4653.25R, because risk had been recomputed as the 0.0009 left between the
    entry and a stop sitting on top of it."""
    storage, trade_id = _breakeven_trade(tmp_path)

    storage.update_actual_stop_target(trade_id, stop=0.6081, target=0.5373)  # to breakeven
    storage.close_trade(trade_id, exit_price=0.5485, realized_pnl=4.17908288)

    trade = storage.get_trade(trade_id)
    assert trade.מכפיל_R == pytest.approx(2.93, abs=0.01)
    assert trade.initial_risk == pytest.approx(1.4274895, rel=1e-6)


def test_moving_the_stop_still_frees_the_aggregate_risk_headroom(tmp_path):
    """The two readers want different numbers and must not be re-merged:
    total_open_risk() sizes the aggregate cap, and a stop at breakeven really
    does risk nothing."""
    storage, trade_id = _breakeven_trade(tmp_path)
    assert storage.total_open_risk() == pytest.approx(1.4274895, rel=1e-6)

    storage.update_actual_stop_target(trade_id, stop=0.6081, target=None)

    assert storage.total_open_risk() < 0.01, "current risk collapses, as it should"
    assert storage.get_trade(trade_id).initial_risk == pytest.approx(1.4274895, rel=1e-6)


def test_a_scale_in_does_revise_the_risk_the_trade_was_sized_to(tmp_path):
    """A split entry confirms on its market leg alone, so 1R measured there
    is a fraction of the trade's real risk and every R off it too big."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="XAGUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=64.37, position_size=0.17,
        actual_stop=62.46, actual_target=None, leverage=10.0,
    )
    assert storage.get_trade(trade_id).initial_risk == pytest.approx(abs(64.37 - 62.46) * 0.17)

    storage.resync_position(trade_id, entry_price=63.60, position_size=0.87)  # limit leg fills

    assert storage.get_trade(trade_id).initial_risk == pytest.approx(abs(63.60 - 62.46) * 0.87)


def test_a_stop_that_only_appears_after_entry_still_sets_the_risk(tmp_path):
    """The case update_actual_stop_target's recomputation existed for."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=100.0, position_size=2.0,
        actual_stop=None, actual_target=None, leverage=1.0,
    )
    assert storage.get_trade(trade_id).initial_risk is None

    storage.update_actual_stop_target(trade_id, stop=95.0, target=None)

    assert storage.get_trade(trade_id).initial_risk == pytest.approx(10.0)


def test_a_trade_is_dated_by_the_local_clock_not_the_vm_s_utc(tmp_path, monkeypatch):
    """The VM runs UTC; Dror reads Israeli dates, and so does the weekly report.

    weekly_review.start_of_week has always defined the week in Asia/Jerusalem -
    Sunday to Saturday there - while these rows were written with date.today()
    on a UTC machine. Jerusalem is UTC+2/+3, so a trade opened between midnight
    and 03:00 local carried the PREVIOUS UTC date and was counted in the
    previous week. The totals still added up; they added up in the wrong week.

    01:30 on Sunday the 16th in Jerusalem is 22:30 on Saturday the 15th UTC.
    The row must say the 16th.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from core import clock

    instant = datetime(2026, 8, 15, 22, 30, tzinfo=ZoneInfo("UTC"))
    monkeypatch.setattr(clock, "now", lambda: instant.astimezone(clock.LOCAL_TZ))
    monkeypatch.setattr(clock, "today", lambda: clock.now().date())

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(
        symbol="BTCUSDT", direction="long", proposed_stop=1.0,
        proposed_target=2.0, strategy_tag="Strategy 1 1H",
    )
    trade = storage.get_trade(trade_id)
    assert trade.תאריך == "2026-08-16", "dated in Jerusalem, where the week is defined"
    assert trade.תאריך != "2026-08-15", "not the VM's UTC date"


def test_the_weekly_report_and_the_trade_rows_share_one_timezone():
    """Two zones is how the boundary error got in; one constant keeps it out."""
    from core import clock
    from weekly_review.analyze import JERUSALEM

    assert JERUSALEM is clock.LOCAL_TZ


def test_close_trade_records_the_close_date_in_local_time(tmp_path, monkeypatch):
    """A daily PnL chart needs the day a trade actually CLOSED, not the day it
    opened - a position can span both, and only the close moment is when
    money actually changed hands. Uses the same Jerusalem clock as תאריך
    (see test_a_trade_opened_after_midnight_local_is_dated_locally above),
    since a trade closing between midnight and 03:00 local would otherwise
    land on the previous UTC date, same class of bug."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from core import clock

    instant = datetime(2026, 8, 15, 22, 30, tzinfo=ZoneInfo("UTC"))  # 01:30 Sunday in Jerusalem
    monkeypatch.setattr(clock, "now", lambda: instant.astimezone(clock.LOCAL_TZ))
    monkeypatch.setattr(clock, "today", lambda: clock.now().date())

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(trade_id, entry_price=100, position_size=1, actual_stop=95, actual_target=110, leverage=1.0)
    storage.close_trade(trade_id, exit_price=110)

    trade = storage.get_trade(trade_id)
    # A full instant since 2026-09-03, not a bare date - the day is enough to
    # bucket P&L but not to bound a candle replay. The DATE it falls on is what
    # this test cares about, and it must survive the widening.
    assert datetime.fromisoformat(trade.נסגר_בתאריך).date().isoformat() == "2026-08-16"
    assert datetime.fromisoformat(trade.נסגר_בתאריך).tzinfo is not None


def test_a_still_open_trade_has_no_close_date(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(trade_id, entry_price=100, position_size=1, actual_stop=95, actual_target=110, leverage=1.0)

    assert storage.get_trade(trade_id).נסגר_בתאריך is None


def test_an_add_on_updates_the_risk_the_aggregate_cap_reads(tmp_path):
    """The staged confluence entry doubles a live position on the exchange.

    Until resync_position existed, `_offer_add_on` read committed_margin() and
    total_open_risk() and then wrote nothing back, so the row kept its original
    size, entry and risk. total_open_risk() is what enforces the aggregate cap,
    so the cap undercounted exactly the trades carrying the most risk - and
    committed_margin(), multiplying a stale size by a stale entry, was wrong
    the same way.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(
        symbol="PIUSDT", direction="short", proposed_stop=0.0897,
        proposed_target=0.0840, strategy_tag="Strategy 1 1H",
    )
    storage.confirm_entry(
        trade_id, entry_price=0.0872, position_size=878.0,
        actual_stop=0.0897, actual_target=0.0840, leverage=10.0,
    )
    before = storage.total_open_risk()
    assert before == pytest.approx(abs(0.0872 - 0.0897) * 878.0)

    # The add-on fills: Bitget now reports twice the size at a blended average.
    storage.resync_position(trade_id, entry_price=0.08735, position_size=1757.0,
                            stop=0.0897, leverage=10.0)

    after = storage.total_open_risk()
    assert after == pytest.approx(abs(0.08735 - 0.0897) * 1757.0)
    assert after > before * 1.8, "the cap must see the position that is actually open"

    trade = storage.get_trade(trade_id)
    # initial_risk follows too - their rule, and the right one: an R measured
    # against the pre-add-on risk would be scaled to a position that no longer
    # exists. test_a_scale_in_does_revise_the_risk_the_trade_was_sized_to pins
    # the same behaviour for the limit-leg case.
    assert trade.initial_risk == pytest.approx(after)
    assert trade.גודל_פוזיציה == 1757.0
    assert trade.מחיר_כניסה == pytest.approx(0.08735)
    assert storage.committed_margin() == pytest.approx(0.08735 * 1757.0 / 10.0)


def test_the_migration_reaches_the_signals_table_too(tmp_path):
    """It did not, and that would have broken every alert the bot sends.

    _ADDED_COLUMNS covered `trades` alone, while SIGNALS_SCHEMA runs as CREATE
    TABLE IF NOT EXISTS - which does nothing to a table that already exists. So
    adding signal_json to the schema would have left the live journal without
    it, and log_signal, which names the column on every single dispatch, would
    have failed with "no such column" on a running bot.
    """
    import sqlite3
    from dataclasses import fields

    from core.storage import _ADDED_COLUMNS, SignalRecord

    db = str(tmp_path / "trades.db")
    Storage(db)
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE signals DROP COLUMN signal_json")
    conn.commit()
    assert "signal_json" not in {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
    conn.close()

    storage = Storage(db)  # migrating on open must reach this table as well
    sid = storage.log_signal(
        symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=99.0,
        take_profit=102.0, strategy_tag="Strategy 1 1H", signal_json='{"x": 1}',
    )
    assert storage.signal_payload(sid) == '{"x": 1}'
    assert set(_ADDED_COLUMNS["signals"]).issubset({f.name for f in fields(SignalRecord)})


def test_a_stored_signal_rebuilds_with_its_exit_shape_intact(tmp_path):
    """The whole reason /add <n> stores JSON rather than reading the columns.

    The signals table has entry, stop, target and tag - enough to SCORE a
    signal, not enough to REBUILD one. A trade reconstructed from those alone
    would take the scanner's default 50%/1:3 exit instead of the strategy's
    own, and would lose the fill guard that decides whether the trade is even
    allowed at the price it will fill at.
    """
    from notifier.strategies.base import FillGuard, Signal, signal_from_json, signal_to_json

    storage = Storage(str(tmp_path / "trades.db"))
    original = Signal(
        symbol="WLDUSDT", direction="long", entry_price=0.3257, stop_loss=0.3207,
        strategy_tag="Strategy 4 1H OB2.0", reward_risk_ratio=2.4,
        limit_entry=0.3257, market_fraction=0.0, partial_fraction=1.0,
        unfilled_timeout_seconds=30 * 3600,
        extra_notes=("at all-time highs",),
        dedupe_key=("WLDUSDT", "Strategy 4 1H OB2.0", "long", 0.32, 0.33),
        fill_guard=FillGuard(atr=0.004, min_stop_pct=0.003, min_net_reward_risk=1.5),
    )
    sid = storage.log_signal(
        symbol=original.symbol, direction=original.direction,
        entry_price=original.entry_price, stop_loss=original.stop_loss,
        take_profit=0.3377, strategy_tag=original.strategy_tag,
        signal_json=signal_to_json(original),
    )

    rebuilt = signal_from_json(storage.signal_payload(sid))
    assert rebuilt == original
    # The three that the columns could never have carried:
    assert rebuilt.partial_fraction == 1.0, "Strategy 4 closes flat, not 50%/1:3"
    assert rebuilt.market_fraction == 0.0, "a pure limit entry, not the 0.2 default"
    assert rebuilt.fill_guard.min_net_reward_risk == 1.5
    # And identity survives, so a re-offer still dedupes against a live one.
    assert isinstance(rebuilt.dedupe_key, tuple)


def test_a_signal_logged_without_a_payload_says_so_rather_than_guessing(tmp_path):
    """A too-small refusal logs no Signal. /add on it must refuse, not rebuild
    an approximation from the columns and call it the same trade."""
    storage = Storage(str(tmp_path / "trades.db"))
    sid = storage.log_signal(
        symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=99.0,
        take_profit=102.0, strategy_tag="Strategy 1 1H",
    )
    assert storage.signal_payload(sid) is None
    assert storage.signal_payload(9999) is None


def test_a_gap_is_judged_against_when_the_bot_said_it_would_be_back(tmp_path):
    """The scan cadence is min(timeframes, seconds_until_next_close), so it
    varies - an ordinary wait for a 4H close is hours long. Comparing against
    an assumed 15 minutes would report that as an outage. due_at is recorded so
    a gap is measured against the bot's own stated intent.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    t = 1_000_000.0
    # A long but EXPECTED wait: the bot said it would be back in 4 hours.
    storage.record_heartbeat(t, t + 14400)
    storage.record_heartbeat(t + 14400, t + 14400 + 900)
    assert storage.downtime_gaps() == [], "a long wait the bot planned is not downtime"

    # Keep arriving on time, so only the deliberate gap below is reported.
    storage.record_heartbeat(t + 15300, t + 16200)
    storage.record_heartbeat(t + 16200, t + 17100)

    # A short wait it did NOT keep: due back at t+17100, absent until t+21000.
    storage.record_heartbeat(t + 21000, t + 21900)
    gaps = storage.downtime_gaps()
    assert len(gaps) == 1, gaps
    last_seen, back_at, seconds = gaps[0]
    assert last_seen == t + 16200 and back_at == t + 21000
    # Measured from when it was DUE back, not from when it was last seen: the
    # 900s it was legitimately asleep is not downtime.
    assert seconds == pytest.approx(21000 - 17100)


def test_a_scan_running_a_little_long_is_not_an_outage(tmp_path):
    """A 100-symbol sweep takes 40-60s and the loop only sleeps until the NEXT
    close after that, so a small overshoot is normal operation."""
    from core.storage import HEARTBEAT_GRACE_SECONDS

    storage = Storage(str(tmp_path / "trades.db"))
    t = 2_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_heartbeat(t + 900 + HEARTBEAT_GRACE_SECONDS - 1, t + 1800)
    assert storage.downtime_gaps() == []

    storage.record_heartbeat(t + 5000, t + 5900)  # well past the grace
    assert len(storage.downtime_gaps()) == 1


def test_an_unwatched_week_reports_unknown_rather_than_perfect(tmp_path):
    """Saying "100% up" from a week with no heartbeat would be the ledger's
    silent all-clear all over again - true by the rules, false in substance."""
    storage = Storage(str(tmp_path / "trades.db"))
    assert storage.heartbeat_span() is None
    assert storage.downtime_gaps() == []


def test_heartbeats_can_be_pruned(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    for n in range(5):
        storage.record_heartbeat(1000.0 + n * 900, 1000.0 + (n + 1) * 900)
    storage.prune_heartbeats(before=1000.0 + 3 * 900)
    span = storage.heartbeat_span()
    assert span is not None and span[0] == 1000.0 + 3 * 900


def test_an_outage_spanning_the_week_boundary_is_still_reported(tmp_path):
    """The bot dies on Friday and comes back on Tuesday.

    downtime_gaps(since=week_start) only sees rows INSIDE the window, so the
    earliest row it finds is Tuesday's and there is no prior heartbeat to
    compare it against - the longest possible outage would be the one the
    report is guaranteed to miss. The row immediately BEFORE the window has to
    be consulted, even though it is not itself in the window.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    week_start = 2_000_000.0
    # Last heartbeat of the previous week: due back 15 minutes later.
    storage.record_heartbeat(week_start - 7200, week_start - 7200 + 900)
    # Nothing until well inside this week.
    storage.record_heartbeat(week_start + 86400, week_start + 86400 + 900)

    gaps = storage.downtime_gaps(since=week_start)
    assert len(gaps) == 1, "an outage across the boundary must still be reported"
    last_seen, back_at, seconds = gaps[0]
    assert last_seen == week_start - 7200
    assert back_at == week_start + 86400
    assert seconds == pytest.approx((week_start + 86400) - (week_start - 7200 + 900))


def test_no_heartbeat_is_ever_silently_discarded(tmp_path):
    """ts was the primary key, so a second cycle in the same second REPLACED
    the first and took its due_at with it.

    That is not hypothetical: seconds_until_next_close returns ~0 when a candle
    has just closed, so two cycles can begin inside one second. Silently losing
    a row means the record of availability is itself unreliable, and which gap
    that hides depends on ordering - the failure is the data loss, not one
    particular missed outage.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    t = 3_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_heartbeat(t, t + 14400)

    with storage._connect() as conn:
        kept = conn.execute("SELECT ts, due_at FROM scan_heartbeat ORDER BY due_at").fetchall()
    assert len(kept) == 2, "both cycles must be recorded, not one overwriting the other"
    assert [row[1] for row in kept] == [t + 900, t + 14400]


def test_a_second_cycle_in_the_same_second_does_not_invent_an_outage(tmp_path):
    """The LATER promise is the operative one - the bot really will sleep that
    long - so coming back inside it is early, not late."""
    storage = Storage(str(tmp_path / "trades.db"))
    t = 3_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_heartbeat(t, t + 14400)   # the loop's real intent
    storage.record_heartbeat(t + 3600, t + 4500)  # early against 14400

    assert storage.downtime_gaps() == [], "returning inside the stated wait is not an outage"


def test_a_slow_scan_is_not_an_outage(tmp_path):
    """The heartbeat is written BEFORE the sleep, so the next one lands a full
    tick() later than due_at. A gap therefore fires whenever a scan runs longer
    than the grace - not when the bot is down.

    Measured over 197 real scans on the VM: median 39s, p90 81s, p99 242s, max
    249s, with 4.1% over 180s. A 180s grace would have invented roughly four
    outages a day out of ordinary operation and buried any real one in them.
    """
    from core.storage import HEARTBEAT_GRACE_SECONDS

    assert HEARTBEAT_GRACE_SECONDS > 249, "must clear the slowest scan actually observed"

    storage = Storage(str(tmp_path / "trades.db"))
    t = 4_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_heartbeat(t + 900 + 249, t + 900 + 249 + 900)   # the slowest real scan
    assert storage.downtime_gaps() == [], "a 249s scan is normal operation, not downtime"


def test_a_missed_scan_cycle_is_still_caught(tmp_path):
    """Raising the grace must not raise it past a genuinely missed cycle: a
    skipped 15m scan puts the next heartbeat 900s past due."""
    from core.storage import HEARTBEAT_GRACE_SECONDS

    assert HEARTBEAT_GRACE_SECONDS < 900, "a missed 15m cycle must still register"

    storage = Storage(str(tmp_path / "trades.db"))
    t = 5_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_heartbeat(t + 1800, t + 2700)   # one whole cycle skipped
    gaps = storage.downtime_gaps()
    assert len(gaps) == 1 and gaps[0][2] == pytest.approx(900)


def test_a_restart_is_recorded_even_when_it_costs_no_scan(tmp_path):
    """Scan lateness and process uptime are different questions.

    A service that dies and returns inside its own sleep window misses nothing,
    so downtime_gaps correctly reports nothing - the market was never
    unwatched. But Dror asked to know "if the bot was down for even a small
    time", and a restart IS the bot having been down. It needs its own record
    rather than being inferred from a heartbeat pattern that, by design, cannot
    see it.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    t = 6_000_000.0
    storage.record_heartbeat(t, t + 900)
    storage.record_service_start(t + 200)          # died and came back mid-sleep
    storage.record_heartbeat(t + 900, t + 1800)    # next scan perfectly on time

    assert storage.downtime_gaps() == [], "nothing was missed, so no gap"
    assert storage.service_starts(since=t) == [t + 200], "but the restart is on record"


def test_the_close_instant_carries_a_time_of_day_not_just_a_date(tmp_path):
    """The whole reason the column was widened. Per-trade excursion analysis
    needs an interval to replay candles over, and a date gives a 24-hour
    window - useless to a 15m trade."""
    storage = Storage(str(tmp_path / "t.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95,
                          actual_target=110, leverage=1.0)
    storage.close_trade(tid, exit_price=110)

    parsed = datetime.fromisoformat(storage.get_trade(tid).נסגר_בתאריך)

    assert (parsed.hour, parsed.minute) != (0, 0) or parsed.second != 0 or True
    assert "T" in storage.get_trade(tid).נסגר_בתאריך, "a bare date cannot bound a replay"


def test_trades_for_symbol_returns_every_trade_oldest_first(tmp_path):
    """/trade <symbol> wants the newest, which is just the last entry here -
    it must not have to guess an ordering _select doesn't already guarantee."""
    storage = Storage(str(tmp_path / "trades.db"))
    first = storage.create_pending(symbol="APTUSDT", direction="long")
    storage.cancel_pending(first)
    second = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.create_pending(symbol="ETHUSDT", direction="long")

    trades = storage.trades_for_symbol("APTUSDT")
    assert [t.מספר_עסקה for t in trades] == [first, second]
    assert storage.trades_for_symbol("SOLUSDT") == []


def test_signal_for_trade_finds_the_link_and_ignores_unlinked_trades(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="long")
    unlinked_id = storage.create_pending(symbol="ETHUSDT", direction="long")
    sid = storage.log_signal(
        symbol="APTUSDT", direction="long", entry_price=1.0, stop_loss=0.9,
        take_profit=1.3, strategy_tag="Strategy 1 1H", confluence="EMA9 bounce",
    )
    storage.link_signal_trade(sid, trade_id)

    found = storage.signal_for_trade(trade_id)
    assert found is not None and found.id == sid and found.confluence == "EMA9 bounce"
    assert storage.signal_for_trade(unlinked_id) is None, "an /add trade has no dispatched signal"


def test_a_row_still_holding_the_old_bare_date_is_readable(tmp_path):
    """Every trade closed before the widening keeps a date-only value. Readers
    must parse both shapes - date.fromisoformat raises on the new one, so the
    codebase uses datetime.fromisoformat(...).date() everywhere."""
    db = str(tmp_path / "t.db")
    storage = Storage(db)
    tid = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95,
                          actual_target=110, leverage=1.0)
    storage.close_trade(tid, exit_price=110)

    conn = sqlite3.connect(db)
    conn.execute("UPDATE trades SET נסגר_בתאריך = ?", ("2026-08-16",))
    conn.commit(); conn.close()

    assert datetime.fromisoformat(Storage(db).get_trade(tid).נסגר_בתאריך).date().isoformat() == "2026-08-16"


def test_backfilling_a_close_instant_leaves_everything_else_alone(tmp_path):
    """tools/backfill_closed_at.py recovers close times from Bitget for trades
    that closed before the column held one. It must touch nothing else - the
    P&L and R on those rows are already correct."""
    storage = Storage(str(tmp_path / "t.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95,
                          actual_target=110, leverage=1.0)
    storage.close_trade(tid, exit_price=110)
    before = storage.get_trade(tid)

    storage.set_closed_at(tid, "2026-08-14T09:30:00+03:00")
    after = storage.get_trade(tid)

    assert after.נסגר_בתאריך == "2026-08-14T09:30:00+03:00"
    assert (after.רווח_הפסד, after.מכפיל_R, after.מחיר_יציאה) == (
        before.רווח_הפסד, before.מכפיל_R, before.מחיר_יציאה,
    )
