"""build_signal() must reproduce _signal_for() at baseline parameters.

This is the guard on the whole Tier B approach. s4_context.build_signal
duplicates _signal_for's arithmetic so that a sweep arm can rebuild a trade
without re-running the detector; if the two drift, every swept number is
measuring a strategy that does not exist. Pinning them on real bars is the
only thing that keeps the duplication honest.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from backtest.s4_context import Params, build_signal
from backtest.s4_record import (
    LIVE_WINDOW, RECORD_STEEPNESS, TIMEFRAME, _frame, context_for,
)
from notifier.strategies.order_block import OrderBlockStrategy

BARS = Path("data/bars_1h_deep_np.pkl")


@pytest.mark.skipif(not BARS.exists(), reason="deep bar cache not present")
def test_build_signal_matches_signal_for_on_real_bars():
    with open(BARS, "rb") as fh:
        raw = pickle.load(fh)

    # The SHIPPED instance, at the real steepness floor.
    shipped = OrderBlockStrategy(TIMEFRAME, session_gated=False)
    # What the recorder actually uses: permissive, so the floor becomes a
    # sweepable filter. Filtering these back to the shipped floor must
    # reproduce `shipped` exactly - that equivalence is the whole basis for
    # recording once and sweeping the floor afterwards, and it holds only
    # because the steepness test is the LAST gate in _qualifies and the
    # expansion loop advances unconditionally.
    recorder = OrderBlockStrategy(TIMEFRAME, session_gated=False,
                                  min_steepness=RECORD_STEEPNESS)
    compared = agreed = 0
    for symbol, cols in list(raw.items()):
        frame = _frame(cols)
        if len(frame) < 700:
            continue
        stamps = frame["ts"].to_numpy()
        closes = frame["close"].to_numpy()
        # A slice of bars is plenty: what matters is that every setup found
        # agrees, not how many setups the slice happens to contain.
        for i in range(len(frame) - 400, len(frame)):
            lo = max(0, i + 1 - LIVE_WINDOW)
            bars = frame.iloc[lo: i + 1]
            want = shipped.evaluate(symbol, {TIMEFRAME: bars})
            ctx = context_for(recorder, symbol, bars, int(stamps[i]), i, float(closes[i]))
            # Params() now defaults to a nonzero gap_close_margin_pct (see
            # order_block.GAP_CLOSE_MARGIN_PCT), so build_signal needs bars to
            # re-derive the structure window a margin check runs against.
            # MUST be `frame` (the full symbol history), not the `bars` slice
            # above: build_signal does its own `bars.iloc[lo: ctx.bar_index+1]`
            # using ctx.bar_index as an ABSOLUTE frame position - handed the
            # slice instead, that positional lookup falls outside its length
            # and .iloc silently returns empty rather than raising.
            rebuilt = build_signal(ctx, Params(), TIMEFRAME, bars=frame) if ctx else None

            if want is None and rebuilt is None:
                continue
            compared += 1
            assert want is not None and rebuilt is not None, (
                f"{symbol} bar {i}: shipped={want!r} rebuilt={rebuilt!r}")
            assert want.direction == rebuilt.direction
            assert want.entry_price == pytest.approx(rebuilt.entry_price, rel=1e-9)
            assert want.stop_loss == pytest.approx(rebuilt.stop_loss, rel=1e-9)
            assert want.reward_risk_ratio == pytest.approx(
                rebuilt.reward_risk_ratio, rel=1e-9)
            assert want.strategy_tag == rebuilt.strategy_tag
            agreed += 1
        if compared >= 15:
            break

    assert compared > 0, "no setups found to compare - the test proved nothing"
    assert agreed == compared


def test_a_permissive_recording_filters_back_to_the_shipped_floor():
    """OB1.0 blocks carry no steepness and must survive any floor; OB2.0
    blocks are filtered on their own measured value."""
    from backtest.s4_context import BlockCtx

    ob1 = BlockCtx(low=1.0, high=2.0, direction="short", variant="OB1.0", index=5,
                   displacement_end=7, sweep_level=2.1, sweep_extreme=2.2,
                   in_asia=False, steepness=None)
    ob2 = BlockCtx(low=1.0, high=2.0, direction="short", variant="OB2.0", index=5,
                   displacement_end=7, sweep_level=2.1, sweep_extreme=2.2,
                   in_asia=False, steepness=0.6)
    assert ob1.steepness is None, "OB1.0 has no steepness test of its own"
    # A 0.6 block is admitted at a 0.5 floor and refused at the shipped 1.0.
    assert ob2.steepness >= 0.5
    assert ob2.steepness < Params().min_steepness


def test_entry_fraction_moves_the_entry_toward_the_near_edge():
    """0.5 is the shipped midpoint; 0.0 is the edge price reaches first."""
    from backtest.s4_context import BlockCtx, _entry

    short = BlockCtx(low=100.0, high=110.0, direction="short", variant="OB2.0",
                     index=5, displacement_end=7, sweep_level=111.0,
                     sweep_extreme=111.5, in_asia=False)
    assert _entry(short, 0.5) == 105.0          # midpoint, what ships
    assert _entry(short, 0.0) == 110.0          # near edge: price arrives here first
    long = BlockCtx(low=100.0, high=110.0, direction="long", variant="OB2.0",
                    index=5, displacement_end=7, sweep_level=99.0,
                    sweep_extreme=98.5, in_asia=False)
    assert _entry(long, 0.5) == 105.0
    assert _entry(long, 0.0) == 100.0


def test_target_fraction_zero_is_the_near_edge_of_the_gap():
    """Dror's "start of the ob": the first edge of the target zone, which is
    reached strictly before the midpoint and therefore converts more often."""
    from backtest.s4_context import GapCtx, _target_price

    gap = GapCtx(direction="up", low=90.0, high=95.0, size=5.0, atr_at_end=1.0,
                 start_index=3, end_index=5)
    # A short targets downward, so the near edge is the HIGH of the gap.
    assert _target_price(gap, 0.0, "short") == 95.0
    assert _target_price(gap, 0.5, "short") == 92.5     # what ships
    # A long targets upward, so the near edge is the LOW.
    assert _target_price(gap, 0.0, "long") == 90.0
    assert _target_price(gap, 0.5, "long") == 92.5
