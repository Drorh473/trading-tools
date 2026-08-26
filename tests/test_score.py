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


def test_generators_measure_exactly_what_ships():
    """The measured instance set must equal the live one.

    This is not hypothetical drift. generate_v2.MEASURABLE kept ("1H","4H") and
    ("4H","1D") for a full run after the paired instances were deleted from the
    strategy, and generate_15m kept ("15m","1H"). A measurement of a variant
    that cannot fire live is worse than no measurement, because it reports a
    number under the name of the thing that shipped.
    """
    from backtest.generate_15m import INSTANCES as FIFTEEN
    from backtest.generate_v2 import MEASURABLE
    from notifier.strategies.ema_trend_v2 import INSTANCES as SHIPPING

    assert set(MEASURABLE) | set(FIFTEEN) == set(SHIPPING)
    assert not set(MEASURABLE) & set(FIFTEEN), "an instance measured by both generators"


def test_both_generators_disable_the_same_thresholds():
    """Two populations are only comparable if generated under one rule set.

    Generation runs with the swept thresholds OFF and records what each setup
    actually had. If one generator left a threshold ON, its population would be
    pre-filtered and the sweep would silently compare a subset against a whole.

    Both generators zero these at IMPORT time, on the shared strategy module.
    That mutation outlives the import, so this test restores the shipping values
    from source afterwards - otherwise merely importing a generator anywhere in
    the suite would leave every later strategy test running against a strategy
    with no thresholds, and passing for the wrong reason.
    """
    import ast
    import pathlib

    import notifier.strategies.ema_trend_v2 as v2

    swept = [
        "EMA9_HOLD_BARS",
        "MIN_STOP_PCT",
        "MIN_NET_REWARD_RISK",
        "MIN_PIVOT_SPAN_BARS",
        "MIN_SWING_DRIFT_ATR",
        "MAX_EMA9_CROSSINGS",
        # Added 2026-08-21 with the gate. While it shipped False this was
        # harmless to omit; the moment it went True, a generator that did not
        # neutralise it would PRE-FILTER the population and every sweep over
        # the result would compare a subset against a whole - including the
        # sweep that justified turning the gate on in the first place.
        "REQUIRE_STRUCTURE_TREND",
    ]
    src = ast.parse(pathlib.Path(v2.__file__).read_text(encoding="utf-8"))
    shipping = {
        t.id: node.value.value
        for node in src.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id in swept
    }
    assert set(shipping) == set(swept), "a swept threshold vanished from the strategy"

    try:
        import importlib

        import backtest.generate_15m
        import backtest.generate_v2

        # The override runs as a module-level side effect at IMPORT time, so a
        # plain `import` here is a no-op once some other test (or this one, on
        # a re-run) has already imported the same module in this process -
        # Python caches modules, it does not re-execute them. reload() forces
        # the override to actually apply again, so this test's result does not
        # depend on whether it happens to be the first thing in the suite to
        # touch these generators.
        importlib.reload(backtest.generate_15m)
        importlib.reload(backtest.generate_v2)

        # MAX_EMA9_CROSSINGS is a ceiling, so "off" is a large number, not 0;
        # the gate is a flag, so "off" is False.
        off = {name: 0 for name in swept} | {"MAX_EMA9_CROSSINGS": 999,
                                             "REQUIRE_STRUCTURE_TREND": False}
        for name in swept:
            assert getattr(v2, name) == off[name], f"{name} was left on during generation"
    finally:
        for name, value in shipping.items():
            setattr(v2, name, value)


def test_a_market_entry_is_scored_at_the_fill_not_at_the_level_that_selected_it():
    """Strategy 2.1's entry_price is the EMA9; the order fills past it.

    entry_price 100.00 against a 95.00 stop looks like 5.00 of risk and puts
    1:2 at 110.00. Filled at 102.00 the real risk is 7.00 and the honest 1:2 is
    116.00 - so a bar reaching 110.00 has not paid 2R, it has paid 1.14R.
    Scoring at the reference price hands the trade the difference.
    """
    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=95.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
        remainder_target=None,
    )
    bars = _bars([
        (102.0, 100.0, 102.0),   # entry bar, not walked
        (111.0, 103.0, 110.0),   # reaches 110.00 - target 1 only if entry was 100.00
        (105.0, 94.0, 95.0),     # then back through both stops
    ])

    at_reference = simulate(bars, 0, sig, runner="choch", pivots=[])
    at_fill = simulate(bars, 0, sig, runner="choch", pivots=[], fill_at=102.0)

    assert at_reference.result == "target1 then stop", "banked half at 110, trailed out"
    assert at_fill.result == "stop", "110.00 is not target 1 for a fill at 102.00"
    assert at_reference.r_gross > 0 > at_fill.r_gross


def test_a_fill_already_through_the_stop_is_not_a_trade():
    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=95.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
    )
    bars = _bars([(101.0, 99.0, 100.0), (101.0, 99.0, 100.0)])
    assert simulate(bars, 0, sig, fill_at=94.0).result == "invalid"


def test_a_market_fill_pays_taker_on_the_way_in():
    """A resting limit is maker; a market order is not, and at these stop
    widths the difference is a real fraction of an R."""
    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=99.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
    )
    bars = _bars([(100.0, 100.0, 100.0), (99.5, 98.0, 98.5)])
    limit = simulate(bars, 0, sig)
    market = simulate(bars, 0, sig, fill_at=100.0)

    assert limit.result == market.result == "stop"
    assert limit.r_gross == market.r_gross
    assert market.r_net < limit.r_net
    # per_r is entry/risk = 100, so the 0.04% fee gap is 0.04R - on a 1% stop,
    # four percent of the trade's whole risk budget, paid on the way in.
    assert (limit.r_net - market.r_net) == pytest.approx((TAKER - MAKER) * 100.0, rel=1e-6)
