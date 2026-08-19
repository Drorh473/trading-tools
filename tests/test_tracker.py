import itertools

import pytest

from core.storage import Storage
from execution.tracker import (
    check_position_now,
    format_partial_message,
    format_scale_in_message,
    matches_expected,
    take_profit_coverage,
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


def _scaling_trade(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95, proposed_target=115)
    storage.confirm_entry(trade_id, entry_price=100, position_size=0.4, actual_stop=95, actual_target=115, leverage=1.0)
    return storage, trade_id


async def test_a_fill_arriving_in_pieces_notifies_once(tmp_path):
    """One logical fill, one message.

    A limit leg can fill in parts, and a message per part would be noise. The
    notification is held until a poll passes with no further growth.
    """
    storage, trade_id = _scaling_trade(tmp_path)
    scaled = []
    bitget = FakeBitget(
        position_sequence=[
            make_position(size=1.0),  # first piece
            make_position(size=1.6),  # still growing - must not fire yet
            make_position(size=2.0),  # still growing
            make_position(size=2.0),  # settled -> fires here
            None,
        ],
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(
        storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0,
        on_scale_in=lambda tid: scaled.append(storage.get_trade(tid).גודל_פוזיציה),
    )

    assert scaled == [2.0], "three growth polls should collapse into one message at the final size"


async def test_a_scale_in_is_not_announced_after_the_trade_closes(tmp_path):
    """The close message supersedes it.

    A position that closes while the notification is still held would produce
    "your position grew to X" describing something that no longer exists - and
    the close message already carries the final entry, size and P&L.
    """
    storage, trade_id = _scaling_trade(tmp_path)
    scaled, closed = [], []
    bitget = FakeBitget(
        position_sequence=[make_position(size=2.0), None],  # grows, then gone before it settles
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(
        storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0,
        on_scale_in=lambda tid: scaled.append(tid),
        on_close=lambda tid, price: closed.append(tid),
    )

    assert scaled == []
    assert closed == [trade_id]


async def test_a_scale_out_overtaking_the_fill_cancels_the_message(tmp_path):
    """If the position shrank before the fill settled, the figures are stale."""
    storage, trade_id = _scaling_trade(tmp_path)
    scaled = []
    bitget = FakeBitget(
        position_sequence=[
            make_position(size=2.0),  # grew
            make_position(size=1.0),  # partial exit overtook it
            None,
        ],
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(
        storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0,
        on_scale_in=lambda tid: scaled.append(tid),
    )

    assert scaled == []


async def test_a_leg_that_filled_during_downtime_still_notifies(tmp_path):
    """Restart case: the DB is behind, the position is already larger.

    Deliberately NOT suppressed the way the pending-break watch is on restart.
    That watch offers an action based on a past event; this reports the
    position as it stands right now, so it cannot be acted on stale.
    """
    storage, trade_id = _scaling_trade(tmp_path)  # DB says 0.4
    scaled = []
    bitget = FakeBitget(
        position_sequence=[
            make_position(size=2.0),  # first poll after restart already bigger
            make_position(size=2.0),
            None,
        ],
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(
        storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0,
        on_scale_in=lambda tid: scaled.append(tid),
    )

    assert scaled == [trade_id]


async def test_no_scale_in_message_when_the_position_never_grows(tmp_path):
    storage, trade_id = _scaling_trade(tmp_path)
    scaled = []
    bitget = FakeBitget(
        position_sequence=[make_position(size=0.4), make_position(size=0.4), None],
        closed={"exit_price": 115.0, "realized_pnl": 34.0},
    )

    await track_position(
        storage, bitget, trade_id, "BTCUSDT", "long", poll_interval=0,
        on_scale_in=lambda tid: scaled.append(tid),
    )

    assert scaled == []


class CoverageBitget:
    """Reports plan orders and resting orders independently, since a target can
    live in either and they come from different endpoints."""

    def __init__(self, plan_orders=(), open_orders=()):
        self._plan_orders = list(plan_orders)
        self._open_orders = list(open_orders)

    def get_plan_orders(self, symbol, direction):
        return self._plan_orders

    def get_open_orders(self, symbol=None):
        return self._open_orders


def _plan(is_target, size, price=110.0):
    return {
        "plan_type": "pos_profit" if is_target else "pos_loss",
        "is_stop": not is_target,
        "is_target": is_target,
        "trigger_price": price,
        "size": size,
    }


def test_a_position_level_target_covers_whatever_the_position_grew_to():
    """size 0 is Bitget's "all closable" sentinel, not a missing value.

    This is what Dror's hand-set targets are, and why they survive a scale-in
    without anyone touching them.
    """
    bitget = CoverageBitget(plan_orders=[_plan(is_target=True, size=0)])

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == 0.51


def test_a_fixed_size_target_does_not_grow_with_the_position():
    """The gap this whole notification exists to surface.

    A target sized to the first leg still covers only that leg after the
    second one fills - the position doubled, the protection did not.
    """
    bitget = CoverageBitget(plan_orders=[_plan(is_target=True, size=0.1)])

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == 0.1


def test_per_leg_targets_add_up():
    bitget = CoverageBitget(
        plan_orders=[_plan(is_target=True, size=0.1), _plan(is_target=True, size=0.41)]
    )

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == pytest.approx(0.51)


def test_a_stop_is_not_counted_as_take_profit_coverage():
    bitget = CoverageBitget(plan_orders=[_plan(is_target=False, size=0)])

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == 0.0


def test_a_resting_reduce_only_limit_counts_but_an_entry_leg_does_not():
    """The bot's own partial take-profit is a resting reduce-only limit, not a
    plan order. An unfilled ENTRY leg sits on the same endpoint and must not be
    mistaken for protection - it would open more position, not close any."""
    bitget = CoverageBitget(
        open_orders=[
            {"tradeSide": "close", "posSide": "long", "size": "0.25"},
            {"tradeSide": "open", "posSide": "long", "size": "0.42"},
        ]
    )

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == 0.25


def test_coverage_ignores_the_other_side_of_a_hedged_symbol():
    bitget = CoverageBitget(
        open_orders=[{"tradeSide": "close", "posSide": "short", "size": "0.25"}]
    )

    assert take_profit_coverage(bitget, "BTCUSDT", "long", position_size=0.51) == 0.0


def test_scale_in_message_states_the_new_position_without_deltas(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="AAPLUSDT", direction="short", proposed_stop=310.94)
    storage.confirm_entry(
        trade_id, entry_price=309.03, position_size=0.51, actual_stop=310.94, actual_target=None, leverage=10.0
    )

    text = format_scale_in_message(storage.get_trade(trade_id), covered=0.51)

    assert "limit leg filled" in text
    assert "0.51 @ 309.03" in text
    assert "stop 310.94" in text
    assert "Breakeven is now 309.03" in text
    assert "Take-profit covers 0.51 of 0.51" in text
    # New state only - Dror's standing preference, same call he made for the
    # weekly report's real-trades section.
    assert "→" not in text and "->" not in text


def test_scale_in_message_survives_a_sub_penny_symbol(tmp_path):
    """The older formatters use .2f, which prints PEPEUSDT's entry as 0.00."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="PEPEUSDT", direction="long", proposed_stop=2.7267e-06)
    storage.confirm_entry(
        trade_id, entry_price=2.8377e-06, position_size=3262000, actual_stop=2.7267e-06,
        actual_target=None, leverage=10.0,
    )

    text = format_scale_in_message(storage.get_trade(trade_id), covered=None)

    assert "2.8377e-06" in text
    assert "0.00 " not in text
    assert "Take-profit covers" not in text  # omitted when coverage is unknown


def _short_with_partial(tmp_path, breakeven=None):
    """The APTUSDT short of 2026-08-13, at the point its partial filled."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short", proposed_stop=0.6312)
    storage.confirm_entry(
        trade_id, entry_price=0.6134, position_size=70.019,
        actual_stop=0.6312, actual_target=None, leverage=10.0,
    )
    if breakeven is not None:
        storage.set_exit_plan(trade_id, breakeven_stop=breakeven, runner_target=0.58, partial_fraction=0.5)
    return storage.get_trade(trade_id)


def test_the_partial_message_no_longer_claims_a_stop_move_nobody_made(tmp_path):
    """The line it used to end on - "stop should already be at entry (0.61)" -
    was unconditional, so it printed on trades that had no breakeven handler
    attached at all. That is how APTUSDT #11 rode its remainder on the
    original stop while the message said otherwise."""
    trade = _short_with_partial(tmp_path)  # no exit plan recorded

    text = format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)

    assert "should already be at entry" not in text
    assert "does NOT manage this trade" in text
    assert "by hand" in text


def test_the_partial_message_states_the_move_when_the_bot_owns_the_exits(tmp_path):
    trade = _short_with_partial(tmp_path, breakeven=0.6134)

    text = format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)

    assert "0.6134 breakeven" in text
    assert "by hand" not in text


def test_the_partial_message_prints_the_breakeven_at_full_precision(tmp_path):
    """.2f rounded this 0.6134 short's breakeven to 0.61 - 0.8% of the price,
    on a stop, which is the difference between breakeven and a small loss."""
    trade = _short_with_partial(tmp_path, breakeven=0.6134)

    assert "0.61)" not in format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)


def test_breakeven_price_prefers_the_positions_real_entry(tmp_path):
    """The stored plan is the alert's blend, made before anything filled."""
    from execution.tracker import breakeven_price

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="XAGUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=64.37, position_size=0.17,
        actual_stop=62.46, actual_target=None, leverage=10.0,
    )
    storage.set_exit_plan(trade_id, breakeven_stop=63.66, runner_target=None, partial_fraction=None)

    assert breakeven_price(storage.get_trade(trade_id)) == 64.37


def test_breakeven_price_keeps_a_hand_typed_price(tmp_path):
    from execution.tracker import breakeven_price

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.confirm_entry(
        trade_id, entry_price=0.65, position_size=70.0,
        actual_stop=0.63, actual_target=None, leverage=10.0,
    )
    storage.set_exit_plan(trade_id, breakeven_stop=0.6081, runner_target=None, partial_fraction=None)
    storage.set_exit_managed(trade_id, True)

    assert breakeven_price(storage.get_trade(trade_id)) == 0.6081


def test_breakeven_price_is_none_when_nothing_is_armed(tmp_path):
    from execution.tracker import breakeven_price

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.confirm_entry(
        trade_id, entry_price=0.6081, position_size=70.0,
        actual_stop=0.63, actual_target=None, leverage=10.0,
    )

    assert breakeven_price(storage.get_trade(trade_id)) is None


def _closed_apt(tmp_path):
    """#11 as it actually closed."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.confirm_entry(
        trade_id, entry_price=0.608112826518, position_size=70.019,
        actual_stop=0.6285, actual_target=None, leverage=10.0,
    )
    storage.close_trade(trade_id, exit_price=0.5485, realized_pnl=4.17908288)
    return storage.get_trade(trade_id)


def test_the_close_message_breaks_out_each_exit(tmp_path):
    """"Exit: 0.55" described a price nothing traded at: it is the average of
    a 0.5608 partial and a 0.5360 runner."""
    from execution.tracker import format_close_message

    exits = [
        {"size": 35.010, "price": 0.5608, "profit": 1.65642205, "at": 1},
        {"size": 35.009, "price": 0.5360, "profit": 2.52459794, "at": 2},
    ]

    text = format_close_message(_closed_apt(tmp_path), exits)

    assert "35.01 @ 0.5608" in text
    assert "35.009 @ 0.536" in text
    assert "avg of the closes below" in text


def test_the_close_message_does_not_call_a_single_exit_an_average(tmp_path):
    """With one close the average IS the fill, so the breakdown is noise."""
    from execution.tracker import format_close_message

    text = format_close_message(_closed_apt(tmp_path), [{"size": 70.019, "price": 0.5485, "profit": 4.0, "at": 1}])

    assert "avg of the closes below" not in text
    assert "@ 0.5485" not in text


def test_the_close_message_prints_prices_at_full_precision(tmp_path):
    """.2f rendered this entry as 0.61 and this exit as 0.55."""
    from execution.tracker import format_close_message

    text = format_close_message(_closed_apt(tmp_path))

    assert "0.608113" in text and "0.5485" in text
    assert "Entry: 0.61 " not in text


def test_closing_exits_never_lets_a_fill_lookup_break_the_close_message(tmp_path):
    """The trade has closed whether or not the fill history can be read."""
    from execution.tracker import closing_exits

    class Broken:
        def get_closing_exits(self, symbol, position_size):
            raise RuntimeError("bitget down")

    assert closing_exits(Broken(), _closed_apt(tmp_path)) == []


async def test_a_strategy_with_its_own_unfilled_window_gets_it():
    """Strategy 4 measured 30 candles and then could not use the number.

    ENTRY_TIMEOUT_SECONDS is a flat 4 hours, chosen for Strategy 1's limit
    tranche. Strategy 4's 1H instance therefore got 4 candles of the 30 its
    fill curve was calibrated on, and its 15m instance 16 - so the calibration
    described a strategy that was not the one running. The proximity gate was
    removed precisely because a resting limit is SUPPOSED to wait, which is
    what made the mismatch load-bearing rather than cosmetic.
    """
    from execution.tracker import ENTRY_TIMEOUT_SECONDS
    from notifier.strategies.base import TIMEFRAME_SECONDS
    from notifier.strategies.order_block import UNFILLED_CANDLES, OrderBlockStrategy

    for tf, expected_hours in (("15m", 7.5), ("1H", 30.0)):
        seconds = UNFILLED_CANDLES * TIMEFRAME_SECONDS[tf]
        assert seconds / 3600 == expected_hours
        assert seconds != ENTRY_TIMEOUT_SECONDS, f"{tf} must not silently match the flat default"
        assert OrderBlockStrategy(tf).timeframe == tf


async def test_a_strategy_without_an_opinion_keeps_the_flat_default():
    from notifier.strategies.base import Signal

    sig = Signal(symbol="X", direction="long", entry_price=1.0, stop_loss=0.9,
                 strategy_tag="Strategy 1 1H")
    assert sig.unfilled_timeout_seconds is None
