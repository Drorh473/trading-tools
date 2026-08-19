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
        line for line in SCHEMA.splitlines() if not any(c in line for c in _ADDED_COLUMNS)
    ).replace("הערות             TEXT,", "הערות             TEXT")
    conn = sqlite3.connect(db)
    conn.execute(old_schema)
    pre_existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    assert not (pre_existing & set(_ADDED_COLUMNS)), "the fixture must start WITHOUT the new columns"
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
    assert set(_ADDED_COLUMNS).issubset({f.name for f in fields(trade)}), "Trade(**row) needs every column"


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
