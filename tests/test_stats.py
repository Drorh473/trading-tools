import pytest

from core.storage import Storage
from journal.stats import compute_stats


def _open_and_close(storage, symbol, direction, entry_price, size, stop, target, exit_price, strategy_tag):
    trade_id = storage.create_pending(symbol=symbol, direction=direction, strategy_tag=strategy_tag)
    storage.confirm_entry(
        trade_id, entry_price=entry_price, position_size=size, actual_stop=stop, actual_target=target, leverage=1.0
    )
    storage.close_trade(trade_id, exit_price=exit_price)
    return trade_id


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "trades.db"))

    # +2R win on strategy A
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=120, strategy_tag="A")

    # -1R loss on strategy A
    _open_and_close(s, "ETHUSDT", "long", 50, 2, 45, 60, exit_price=45, strategy_tag="A")

    # +3R win on strategy B
    _open_and_close(s, "SOLUSDT", "short", 20, 5, 22, 14, exit_price=14, strategy_tag="B")

    # still open, should be excluded
    trade_id = s.create_pending(symbol="XRPUSDT", direction="long", strategy_tag="B")
    s.confirm_entry(trade_id, entry_price=1, position_size=100, actual_stop=0.9, actual_target=1.2, leverage=1.0)

    return s


def test_compute_stats_overall(storage):
    stats = compute_stats(storage.read_all())

    assert stats.total_closed == 3
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.expectancy == pytest.approx((2 - 1 + 3) / 3)
    assert stats.total_pnl == pytest.approx(20 - 10 + 30)
    assert stats.equity_curve == pytest.approx([20, 10, 40])
    assert stats.max_drawdown == pytest.approx(10)  # peak 20 -> dip to 10
    assert stats.best_trade.סימבול == "SOLUSDT"
    assert stats.worst_trade.סימבול == "ETHUSDT"


def test_compute_stats_by_strategy(storage):
    stats = compute_stats(storage.read_all())

    a = stats.by_strategy["A"]
    assert a.count == 2
    assert a.win_rate == pytest.approx(0.5)
    assert a.expectancy == pytest.approx((2 - 1) / 2)
    assert a.total_pnl == pytest.approx(10)

    b = stats.by_strategy["B"]
    assert b.count == 1
    assert b.win_rate == 1.0
    assert b.total_pnl == pytest.approx(30)


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats.total_closed == 0
    assert stats.best_trade is None
