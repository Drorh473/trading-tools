import itertools

import pytest

from core.storage import Storage
from execution.tracker import (
    check_position_now,
    track_position,
    wait_for_signal_position,
)


class FakeBitget:
    def __init__(self, position_sequence, history=None, mark_price=100.0):
        self._positions = itertools.chain(position_sequence, itertools.repeat(position_sequence[-1]))
        self._history = history or []
        self._mark_price = mark_price

    def get_position(self, symbol):
        return next(self._positions)

    def get_position_history(self, symbol, limit=5):
        return self._history

    def get_mark_price(self, symbol):
        return self._mark_price


def make_position(direction="long", entry_price=100.0, size=2.0, stop=95.0, target=115.0):
    return {
        "symbol": "BTCUSDT",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "stop_loss": stop,
        "take_profit": target,
        "unrealized_pnl": 0.0,
        "leverage": 1.0,
        "raw": {},
    }


async def test_wait_for_signal_position_matches_within_tolerance():
    bitget = FakeBitget([None, make_position(entry_price=100.4, size=2.05)])
    position = await wait_for_signal_position(
        bitget, "BTCUSDT", "long", entry_price=100.0, size=2.0, poll_interval=0, timeout_seconds=10
    )
    assert position is not None
    assert position["entry_price"] == 100.4


async def test_wait_for_signal_position_rejects_wrong_direction():
    bitget = FakeBitget([make_position(direction="short")])
    position = await wait_for_signal_position(
        bitget, "BTCUSDT", "long", entry_price=100.0, size=2.0, poll_interval=0.001, timeout_seconds=0.01
    )
    assert position is None


async def test_wait_for_signal_position_rejects_price_outside_tolerance():
    # 5% off, tolerance is 1%
    bitget = FakeBitget([make_position(entry_price=105.0)])
    position = await wait_for_signal_position(
        bitget, "BTCUSDT", "long", entry_price=100.0, size=2.0, poll_interval=0.001, timeout_seconds=0.01
    )
    assert position is None


def test_check_position_now_returns_whatever_is_open():
    bitget = FakeBitget([make_position(direction="short", entry_price=50.0)])
    position = check_position_now(bitget, "BTCUSDT")
    assert position["direction"] == "short"


async def test_track_position_detects_sl_tp_drift_then_close(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95, proposed_target=115)
    storage.confirm_entry(trade_id, entry_price=100, position_size=2, actual_stop=95, actual_target=115, leverage=1.0)

    bitget = FakeBitget(
        position_sequence=[
            make_position(stop=95, target=115),  # unchanged
            make_position(stop=97, target=115),  # user moved the stop up
            None,  # position closed
        ],
        history=[{"exit_price": 118.0, "realized_pnl": 36.0}],
    )

    closed = {}
    await track_position(
        storage, bitget, trade_id, "BTCUSDT", poll_interval=0, on_close=lambda tid, price: closed.update(id=tid, price=price)
    )

    trade = storage.get_trade(trade_id)
    assert trade.סטופ_לוס_בפועל == 97
    assert trade.changed_from_plan is True
    assert trade.מחיר_יציאה == 118.0
    assert trade.רווח_הפסד == 36.0
    assert closed == {"id": trade_id, "price": 118.0}
