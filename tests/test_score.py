"""The one trade scorer, tested against cases whose answers are arithmetic.

This file exists because there used to be TWO scorers and they disagreed by
0.06R on 5,723 trades - larger than most of the effects being compared with
them. The bug was that generate_v2._walk kept the ORIGINAL stop for the whole
trade while the live tracker moves it to breakeven when the partial fills, so it
credited "both targets" to runners which had retraced through breakeven and
recovered. test_a_runner_that_retraces_through_breakeven_ends_there pins exactly
that case.
"""

import pandas as pd
import pytest

from backtest.score import MAKER, TAKER, simulate
from notifier.strategies.base import Signal


def _bars(rows) -> pd.DataFrame:
    """rows = [(high, low, close)], one per bar, starting at the ENTRY bar."""
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(rows), freq="h"),
            "open": [r[2] for r in rows],
            "high": [r[0] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[2] for r in rows],
        }
    )


def _sig(entry=100.0, stop=99.0, rr=2.0, remainder=103.0, direction="long") -> Signal:
    return Signal(
        symbol="TESTUSDT",
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        strategy_tag="test",
        reward_risk_ratio=rr,
        remainder_target=remainder,
    )


# entry 100, stop 99 -> 1R = 1.0 in price. target1 102 (2R), target2 103 (3R).


def test_straight_to_the_stop_is_minus_one_r():
    bars = _bars([(100, 100, 100), (100, 98.5, 98.6)])
    s = simulate(bars, 0, _sig())
    assert s.result == "stop"
    assert s.r_gross == pytest.approx(-1.0)


def test_both_targets_pays_half_at_each():
    bars = _bars([(100, 100, 100), (102.5, 100, 102.2), (103.5, 102, 103.2)])
    s = simulate(bars, 0, _sig())
    assert s.result == "both targets"
    # 0.5 x 2R at target1, 0.5 x 3R at target2
    assert s.r_gross == pytest.approx(0.5 * 2.0 + 0.5 * 3.0)


def test_a_runner_that_retraces_through_breakeven_ends_there():
    """THE BUG THIS FILE EXISTS FOR. Target 1 fills, the stop moves to
    breakeven, price retraces to 100 - the runner is closed at zero. The old
    scorer kept the ORIGINAL stop at 99, so this trade stayed alive and was
    credited "both targets" when price later reached 103."""
    bars = _bars([
        (100, 100, 100),
        (102.5, 100, 102.2),   # target 1 fills, stop -> breakeven at 100
        (102.0, 99.5, 99.8),   # retraces THROUGH breakeven: runner closed at 0
        (104.0, 100, 103.5),   # would have reached target 2 - too late
    ])
    s = simulate(bars, 0, _sig())
    assert s.result == "target1 then stop"
    assert s.r_gross == pytest.approx(0.5 * 2.0), "only the banked half survives"


def test_the_stop_wins_an_ambiguous_candle():
    """One bar touching both stop and target resolves as the stop - the
    conservative reading, since 1H bars cannot say which came first."""
    bars = _bars([(100, 100, 100), (103.5, 98.0, 101.0)])
    s = simulate(bars, 0, _sig())
    assert s.result == "stop"
    assert s.r_gross == pytest.approx(-1.0)


def test_an_unfilled_runner_is_valued_where_it_stands():
    """Not credited a target it never reached, and not written off either."""
    bars = _bars([(100, 100, 100), (102.5, 100, 102.2), (102.6, 101.5, 102.5)])
    s = simulate(bars, 0, _sig())
    assert s.result == "runner open"
    assert s.r_gross == pytest.approx(0.5 * 2.0 + 0.5 * 2.5)


def test_a_trade_that_never_resolves_scores_zero():
    bars = _bars([(100, 100, 100), (101.0, 99.5, 100.5), (101.2, 99.6, 100.8)])
    s = simulate(bars, 0, _sig())
    assert s.result == "unresolved"
    assert s.r_gross == pytest.approx(0.0)


def test_shorts_mirror_longs():
    sig = _sig(entry=100.0, stop=101.0, rr=2.0, remainder=97.0, direction="short")
    # the highs stay UNDER 100 after the partial: on a short, breakeven is
    # above price, so a bar tagging 100 exactly would close the runner - which
    # is correct behaviour and was what this fixture accidentally tested first.
    bars = _bars([(100, 100, 100), (100, 97.5, 97.8), (98.5, 96.5, 96.8)])
    s = simulate(bars, 0, sig)
    assert s.result == "both targets"
    assert s.r_gross == pytest.approx(0.5 * 2.0 + 0.5 * 3.0)


# ---- fees and slippage ----


def test_fees_are_maker_in_and_taker_out_at_the_stop():
    bars = _bars([(100, 100, 100), (100, 98.5, 98.6)])
    s = simulate(bars, 0, _sig())
    per_r = 100.0 / 1.0  # entry / risk
    assert s.r_net == pytest.approx(s.r_gross - (MAKER + TAKER) * per_r)


def test_both_targets_pays_no_taker_because_both_legs_are_limits():
    bars = _bars([(100, 100, 100), (102.5, 100, 102.2), (103.5, 102, 103.2)])
    s = simulate(bars, 0, _sig())
    per_r = 100.0
    assert s.r_net == pytest.approx(s.r_gross - MAKER * per_r * 2)


def test_slippage_is_charged_on_the_stop_only():
    bars = _bars([(100, 100, 100), (100, 98.5, 98.6)])
    clean = simulate(bars, 0, _sig())
    slipped = simulate(bars, 0, _sig(), slippage=0.001)
    assert slipped.r_gross == pytest.approx(clean.r_gross - 0.001 * 100.0)

    winner = _bars([(100, 100, 100), (102.5, 100, 102.2), (103.5, 102, 103.2)])
    assert simulate(winner, 0, _sig(), slippage=0.001).r_gross == pytest.approx(
        simulate(winner, 0, _sig()).r_gross
    ), "a resting limit cannot fill worse than its own price"


# ---- the CHoCH runner ----


def test_the_choch_runner_ratchets_its_stop_to_confirmed_swings():
    bars = _bars([
        (100, 100, 100),
        (102.5, 100, 102.2),   # target 1 -> stop to breakeven
        (104.0, 102.0, 103.8),
        (105.0, 103.0, 104.5),
        (105.0, 101.5, 101.6),  # breaks the protected 102.0
    ])
    # a swing low at 102.0 becomes usable from bar 3
    pivots = [(3, 102.0, False)]
    s = simulate(bars, 0, _sig(remainder=None), runner="choch", pivots=pivots)
    assert s.result == "target1 then stop"
    assert s.r_gross == pytest.approx(0.5 * 2.0 + 0.5 * 2.0), "runner exits at the 102 swing"


def test_a_pivot_is_unusable_before_it_is_confirmed():
    """Dating a pivot by when it PRINTED rather than when it was confirmed
    would let the runner sit behind a level nobody could have known was one."""
    bars = _bars([
        (100, 100, 100),
        (102.5, 100, 102.2),
        (104.0, 102.0, 103.8),
        (105.0, 101.5, 101.6),
    ])
    late = simulate(bars, 0, _sig(remainder=None), runner="choch", pivots=[(99, 102.0, False)])
    assert late.result != "target1 then stop" or late.r_gross == pytest.approx(1.0)


# ---- the retest fill ----


def test_a_retest_that_never_returns_is_unfilled():
    """The rejection candle is the SIGNAL, not the fill. If price never comes
    back to the level the trade simply does not happen - it is not a loss and
    it is not a win."""
    bars = _bars([(100, 100, 100), (101.5, 100.5, 101.2), (103.5, 101.0, 103.0)])
    s = simulate(bars, 0, _sig(), fill_within=4)
    assert s.result == "unfilled"
    assert s.r_net == pytest.approx(0.0)


def test_a_retest_fills_when_price_comes_back_and_then_trades_normally():
    bars = _bars([
        (100, 100, 100),        # the rejection candle - signal only
        (101.5, 100.5, 101.2),  # away from the level
        (101.0, 99.9, 100.2),   # returns to 100 -> filled here
        (102.5, 100, 102.2),    # target 1
        (103.5, 102, 103.2),    # target 2
    ])
    s = simulate(bars, 0, _sig(), fill_within=4)
    assert s.result == "both targets"
    assert s.r_gross == pytest.approx(0.5 * 2.0 + 0.5 * 3.0)


def test_the_fill_bar_itself_can_stop_the_trade_out():
    """Price can keep moving after the limit fills, including straight to the
    stop - the fill bar is walked, not skipped."""
    bars = _bars([(100, 100, 100), (101.5, 100.5, 101.2), (100.5, 98.0, 98.2)])
    s = simulate(bars, 0, _sig(), fill_within=4)
    assert s.result == "stop"
    assert s.r_gross == pytest.approx(-1.0)


def test_without_fill_within_the_entry_is_already_resting():
    """The pre-placed model still works, for comparing the two entry styles."""
    bars = _bars([(100, 100, 100), (102.5, 100, 102.2), (103.5, 102, 103.2)])
    assert simulate(bars, 0, _sig()).result == "both targets"
