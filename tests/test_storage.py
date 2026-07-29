from core.storage import Storage


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
