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
