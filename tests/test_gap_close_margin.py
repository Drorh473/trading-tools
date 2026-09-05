"""gap_is_closed's margin parameter, and the LTCUSDT case it exists for.

LTCUSDT signal 2023-11-09 20:00: the target gap (67.40-67.91) sat inside the
same flash-crash candle that built the block's own displacement. The wick
reached 67.52, 0.12 short of the gap's low, 24% of the gap's own height. The
strict all-or-nothing test called that "still open" and targeted a level the
setup's own formation had almost already revisited.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.s4_context import Params, build_signal, pick_target
from notifier.strategies.order_block import Gap, gap_closest_approach, gap_is_closed


def test_margin_zero_reproduces_the_original_all_or_nothing_test():
    gap = Gap("up", low=67.40, high=67.91, start_index=0, end_index=2)
    # end_index=2, so bars.iloc[3:] is what gap_is_closed scans; the wick to
    # 67.52 never crosses 67.40, so this must read as open at margin 0.
    bars = pd.DataFrame({"low": [0, 0, 0, 67.52, 67.60], "high": [0, 0, 0, 70.0, 71.0]})
    assert gap_is_closed(gap, bars, margin_pct=0.0) is False


def test_the_ltc_near_miss_flips_to_closed_at_a_real_margin():
    """A wick to 67.52 against a 67.40-67.91 gap is 0.12 short of closing -
    24% of the gap's 0.51 height. margin_pct=0.25 must call that closed;
    margin_pct=0.20 must not."""
    gap = Gap("up", low=67.40, high=67.91, start_index=0, end_index=2)
    bars = pd.DataFrame({"low": [0, 0, 0, 67.52], "high": [0, 0, 0, 75.26]})
    assert gap_is_closed(gap, bars, margin_pct=0.0) is False
    assert gap_is_closed(gap, bars, margin_pct=0.20) is False
    assert gap_is_closed(gap, bars, margin_pct=0.25) is True
    assert gap_is_closed(gap, bars, margin_pct=1.0) is True


def test_a_true_full_close_is_closed_at_every_margin_including_zero():
    gap = Gap("up", low=67.40, high=67.91, start_index=0, end_index=2)
    bars = pd.DataFrame({"low": [0, 0, 0, 60.0], "high": [0, 0, 0, 65.0]})
    for margin in (0.0, 0.1, 0.5, 1.0):
        assert gap_is_closed(gap, bars, margin_pct=margin) is True


def test_a_gap_with_no_bars_after_it_yet_is_open_at_any_margin():
    """No price action since the gap printed - can't have approached
    anything, so it must not read as closed no matter how generous the
    margin. Matches the original function's `after.empty -> False`."""
    gap = Gap("up", low=67.40, high=67.91, start_index=0, end_index=2)
    bars = pd.DataFrame({"low": [0, 0, 0], "high": [0, 0, 0]})  # nothing past end_index+1
    for margin in (0.0, 0.5, 1_000.0):
        assert gap_is_closed(gap, bars, margin_pct=margin) is False


def test_gap_closest_approach_matches_the_direction():
    up = Gap("up", low=10.0, high=11.0, start_index=0, end_index=2)
    down = Gap("down", low=10.0, high=11.0, start_index=0, end_index=2)
    bars = pd.DataFrame({"low": [0, 0, 0, 9.5, 9.0], "high": [0, 0, 0, 12.0, 13.0]})
    assert gap_closest_approach(up, bars) == 9.0     # lowest low after the gap
    assert gap_closest_approach(down, bars) == 13.0  # highest high after the gap


def test_params_default_matches_the_live_strategy_default():
    """Params() must equal today's shipped behaviour exactly - the whole
    Tier B approach is pinned on that (test_s4_context.py). Adopted
    2026-09-05 at 0.25 after the full-universe measurement in
    order_block.GAP_CLOSE_MARGIN_PCT's own comment; was 0.0 before that."""
    from notifier.strategies.order_block import GAP_CLOSE_MARGIN_PCT

    assert Params().gap_close_margin_pct == GAP_CLOSE_MARGIN_PCT
    assert GAP_CLOSE_MARGIN_PCT > 0.0, (
        "this test's OWN point is that the gate is now on by default - "
        "if it is ever reverted to 0.0, delete this assertion deliberately, "
        "don't just let it start failing"
    )


def test_pick_target_ignores_margin_without_a_window():
    """margin_pct > 0 with no window supplied must not silently do nothing -
    every recorded gap is already guaranteed unclosed at margin 0, so this
    degrades to margin-0 behaviour rather than raising, which is correct for
    every OTHER filter pick_target applies but must not be mistaken for the
    margin having been checked."""
    from backtest.s4_context import GapCtx, BlockCtx

    block = BlockCtx(low=100.0, high=110.0, direction="short", variant="OB2.0",
                     index=5, displacement_end=7, sweep_level=111.0, sweep_extreme=111.5,
                     in_asia=False)
    gap = GapCtx(direction="up", low=90.0, high=95.0, size=5.0, atr_at_end=1.0,
                start_index=1, end_index=3)
    ctx = __import__("backtest.s4_context", fromlist=["SetupCtx"]).SetupCtx(
        symbol="X", ts=0, bar_index=100, close=105.0, price=105.0,
        atr_now=1.0, equilibrium=102.0, blocks=[block], gaps=[gap])
    p = Params(gap_close_margin_pct=0.5)
    # window=None -> the margin check is skipped (not evaluated as "closed"),
    # so the gap still qualifies on every other ground.
    assert pick_target(ctx, block, entry=105.0, p=p, window=None) is not None


def test_build_signal_raises_without_bars_when_margin_requested():
    from backtest.s4_context import BlockCtx, GapCtx, SetupCtx

    block = BlockCtx(low=100.0, high=110.0, direction="short", variant="OB2.0",
                     index=5, displacement_end=7, sweep_level=111.0, sweep_extreme=111.5,
                     in_asia=False)
    gap = GapCtx(direction="up", low=90.0, high=95.0, size=5.0, atr_at_end=1.0,
                start_index=1, end_index=3)
    ctx = SetupCtx(symbol="X", ts=0, bar_index=100, close=105.0, price=105.0,
                   atr_now=1.0, equilibrium=102.0, blocks=[block], gaps=[gap])
    with pytest.raises(ValueError, match="gap_close_margin_pct"):
        build_signal(ctx, Params(gap_close_margin_pct=0.3), "1H")
