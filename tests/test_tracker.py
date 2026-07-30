import itertools

import pytest

from core.storage import Storage
from execution.tracker import (
    check_position_now,
    matches_expected,
    track_position,
    wait_for_signal_position,
)


def make_position(direction="long", entry_price=100.0, size=2.0, stop=95.0, target=115.0, realized=0.0):
    return {
        "symbol": "BTCUSDT",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "stop_loss": stop,
        "take_profit": target,
        "unrealized_pnl": 0.0,
        "realized_pnl": realized,
        "leverage": 1.0,
        "raw": {},
    }


class FakeBitget:
    def __init__(self, position_sequence, stop_target=(95.0, 115.0), closed=None, mark_price=100.0):
        self._positions = itertools.chain(position_sequence, itertools.repeat(position_sequence[-1]))
        self._stop_target = stop_target
        self._closed = closed
        self._mark_price = mark_price

    def get_position(self, symbol, direction=None):
        pos = next(self._positions)
        if pos and direction and pos["direction"] != direction:
            return None
        return pos

    def get_stop_target(self, symbol, direction):
        st = self._stop_target
        return st.pop(0) if isinstance(st, list) else st

    def find_closed_position(self, symbol, direction):
        return self._closed

    def get_mark_price(self, symbol):
        return self._mark_price


async def test_wait_for_signal_position_returns_matching_side():
    bitget = FakeBitget([None, make_position(direction="long")])
    position = await wait_for_signal_position(bitget, "BTCUSDT", "long", poll_interval=0, timeout_seconds=10)
    assert position is not None
    assert position["direction"] == "long"


async def test_wait_for_signal_position_ignores_opposite_side():
    bitget = FakeBitget([make_position(direction="short")])
    position = await wait_for_signal_position(bitget, "BTCUSDT", "long", poll_interval=0.001, timeout_seconds=0.01)
    assert position is None


async def test_wait_for_signal_position_accepts_any_price_and_size():
    # Split entry: first tranche fills at market for a fraction of planned size.
    # That must still confirm — the tolerances were dropped for signal trades.
    bitget = FakeBitget([make_position(entry_price=142.0, size=0.3)])
    position = await wait_for_signal_position(bitget, "BTCUSDT", "long", poll_interval=0, timeout_seconds=10)
    assert position is not None
    assert position["entry_price"] == 142.0


async def test_wait_for_signal_position_times_out():
    bitget = FakeBitget([None])
    position = await wait_for_signal_position(bitget, "BTCUSDT", "long", poll_interval=0.001, timeout_seconds=0.01)
    assert position is None


def test_matches_expected_still_enforces_tolerance_for_add():
    position = make_position(entry_price=100.4, size=2.05)
    assert matches_expected(position, "long", entry_price=100.0, size=2.0)
    assert not matches_expected(position, "short", entry_price=100.0, size=2.0)
    assert not matches_expected(make_position(entry_price=105.0), "long", entry_price=100.0, size=2.0)


def test_check_position_now_returns_open_position():
    bitget = FakeBitget([make_position(direction="short", entry_price=50.0)])
    assert check_position_now(bitget, "BTCUSDT")["direction"] == "short"


async def test_track_position_syncs_stop_change_then_closes(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95, proposed_target=115)
    storage.confirm_entry(trade_id, entry_price=100, position_size=2, actual_stop=95, actual_target=115, leverage=1.0)

    bitget = FakeBitget(
        position_sequence=[make_position(), make_position(), None],
        stop_target=[(95.0, 115.0), (97.0, 115.0)],
        closed={"exit_price": 118.0, "realized_pnl": 35.5},
    )

    closed = {}
    await track_position(
        storage,
        bitget,
        trade_id,
        "BTCUSDT",
        "long",
        poll_interval=0,
        on_close=lambda tid, price: closed.update(id=tid, price=price),
    )

    trade = storage.get_trade(trade_id)
    assert trade.סטופ_לוס_בפועל == 97
    assert trade.changed_from_plan is True
    assert trade.מחיר_יציאה == 118.0
    assert trade.רווח_הפסד == 35.5  # netProfit from history, not derived
    assert closed == {"id": trade_id, "price": 118.0}


async def test_track_position_records_partial_exit(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95, proposed_target=115)
    storage.confirm_entry(trade_id, entry_price=100, position_size=2, actual_stop=95, actual_target=115, leverage=1.0)

    bitget = FakeBitget(
        position_sequence=[
            make_position(size=2.0),
            make_position(size=1.0, realized=15.0),  # half closed
            None,  # runner closed
        ],
        closed={"exit_price": 100.0, "realized_pnl": 15.0},
    )

    partials = []
    await track_position(
        storage,
        bitget,
        trade_id,
        "BTCUSDT",
        "long",
        poll_interval=0,
        on_partial=lambda tid, size, pnl: partials.append((tid, size, pnl)),
    )

    assert partials == [(trade_id, 1.0, 15.0)]
    trade = storage.get_trade(trade_id)
    assert trade.מחיר_יציאה == 100.0
    assert trade.רווח_הפסד == 15.0


async def test_track_position_resyncs_scale_in(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95, proposed_target=115)
    storage.confirm_entry(trade_id, entry_price=100, position_size=0.4, actual_stop=95, actual_target=115, leverage=1.0)

    # limit tranche fills: size grows and average entry moves
    bitget = FakeBitget(
        position_sequence=[make_position(entry_price=98.0, size=2.0), None],
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0)

    trade = storage.get_trade(trade_id)
    assert trade.מחיר_כניסה == 98.0
    assert trade.גודל_פוזיציה == 2.0
    assert trade.סכום_סיכון == pytest.approx(abs(98.0 - 95.0) * 2.0)  # risk recomputed off real numbers
