"""Strategy 2 v2 - the EMA9 pullback with the target decoupled from the stop.

THE FIXTURES ARE STAIRCASES, NOT RAMPS, and that is not incidental. v2's trend
read is the strict last-3 rule: three ascending swing highs AND three ascending
swing lows. A monotonic ramp reverses nowhere, so zigzag_pivots finds no pivots
at all, the read returns None, and None BLOCKS the trade. v1's fixtures were
ramps because its reader tolerated them; reusing them here would have produced
a test suite in which nothing ever fires. test_a_ramp_has_no_pivots_so_no_trade
pins that down so the next person does not rediscover it.
"""

import pandas as pd
import pytest

from notifier.risk_sizing import TAKER_FEE_PCT
from notifier.strategies import ema_trend_v2 as v2
from notifier.strategies.ema_trend_v2 import (
    INSTANCES,
    EmaTrendV2,
    build_instances,
    structure_metrics,
)
from notifier.strategies.indicators import ema


def _bars(closes: list[float], freq: str = "h", last_low=None, last_high=None, volume=1.0) -> pd.DataFrame:
    s = pd.Series(closes)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq=freq),
            "open": s,
            "high": s,
            "low": s,
            "close": s,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )
    if last_low is not None:
        df.loc[df.index[-1], "low"] = last_low
    if last_high is not None:
        df.loc[df.index[-1], "high"] = last_high
    df.loc[df.index[-1], "base_vol"] = volume
    return df


def _staircase(cycles=12, up_bars=14, up=1.2, dn_bars=6, dn=1.1, start=100.0, sign=1.0) -> list[float]:
    """Rising (or falling) in legs, so each cycle prints one confirmed swing
    high and one confirmed swing low, each beyond the last."""
    c = [start]
    for _ in range(cycles):
        for _ in range(up_bars):
            c.append(c[-1] + up * sign)
        for _ in range(dn_bars):
            c.append(c[-1] - dn * sign)
    return c


# Long enough that the last _hold_bars("1H") closes all sit on the trend side
# of their own EMA9 before the pullback. A shorter tail leaves part of the
# staircase's down-leg inside the hold window, and the setup is correctly
# refused - which is what the hold is for.
#
# DERIVED from the hold, not a constant. It was a hardcoded 14, and when 1H's
# hold moved 10 -> 20 on 2026-08-29 that refused TWENTY of these fixtures at
# once - every one of them reading as "the strategy stopped firing" when what
# had happened is that the fixture became too short to satisfy its own gate.
_TAIL = v2._hold_bars("1H") + 4


def uptrend(freq: str = "h", **kw) -> pd.DataFrame:
    """An ascending staircase, then a sustained hold above EMA9, then a touch
    of it from above - the EMA9 acting as support."""
    closes = _staircase(**kw)
    closes += [closes[-1] + 1.2 * i for i in range(1, _TAIL + 1)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, freq=freq, last_low=e9_prev * 0.9999)


def downtrend(freq: str = "h", volume: float = 1.0) -> pd.DataFrame:
    closes = _staircase(start=400.0, sign=-1.0)
    closes += [closes[-1] - 1.2 * i for i in range(1, _TAIL + 1)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, freq=freq, last_high=e9_prev * 1.0001, volume=volume)


def uptrend_short_hold() -> pd.DataFrame:
    """Structure and touch intact, but the EMA9 has only been held a bar or
    two - a young trend. Enough for a trigger timeframe, not enough to claim
    the level is respected."""
    closes = _staircase()
    closes += [closes[-1] + 1.2 * i for i in range(1, 4)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, last_low=e9_prev * 0.9999)


def broken_resistance() -> pd.DataFrame:
    """The ETHUSDT shape: price BELOW its EMA9 for a long stretch, then a
    rally that carries it up through the level. The final candle satisfies
    `low <= EMA9 and close > EMA9` without the EMA9 ever having been support."""
    closes = _staircase()
    # sag beneath the EMA9, then ONE candle spiking back through it. Only one
    # close lands above the level, so hold_run is 1 - which is what makes this
    # distinguishable from a real pullback now that EMA9_HOLD_BARS is 2.
    closes += [closes[-1] - 1.6 * i for i in range(1, 15)]
    closes += [closes[-1] + 26.0]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, last_low=e9_prev * 0.999)


def _metrics(trend):
    """A structure_metrics result with the scale measures wide open, so a test
    about the TREND is not accidentally a test about span or drift."""
    return {"trend": trend, "span": 999, "drift_high": 99.0, "drift_low": 99.0, "crossings": 0}


def ramp(n: int = 250) -> pd.DataFrame:
    """Rising every single bar - stacked, but structurally unreadable."""
    closes = [100 + i * 0.8 for i in range(n)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, last_low=e9_prev * 0.9999)


# ---- the trade it produces ----


def test_fires_a_standalone_long_on_a_clean_staircase():
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 2.1 1H"
    assert signal.stop_loss < signal.entry_price


def test_the_entry_price_is_built_from_the_PREVIOUS_bar_only():
    """The lookahead guard. The EMA9 the trade is priced from comes from the bar
    BEFORE the rejection, never from the rejection candle itself - measuring it
    that way is what made four separate variants look better than they were.

    In next_open mode the FILL is a market order on the following candle, so
    entry_price is what the plan is sized from rather than a resting price."""
    bars = uptrend()
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars})
    prev = ema(bars["close"].iloc[:-1], 9).iloc[-1]
    current = ema(bars["close"], 9).iloc[-1]
    assert signal.entry_price == pytest.approx(prev)
    assert signal.entry_price != pytest.approx(current)


def test_next_open_mode_enters_at_market():
    """The rejection is only known at the close, so the first price actually
    available afterwards is the next candle. That means a MARKET order, not a
    resting limit - and no limit means no fill risk, at the cost of the entry
    price. Measured best of the faithful constructions at -0.020R against
    -0.086R for a limit at the EMA9 that fills 47% of the time."""
    assert v2.ENTRY_MODE == "next_open"
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    assert signal.market_fraction == 1.0
    assert signal.limit_entry is None


def test_shorts_mirror_longs_and_need_no_volume():
    """v1 required above-average volume on shorts and nothing on longs. That
    asymmetry is removed, so a short fires on volume no better than average."""
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": downtrend(volume=1.0)})
    assert signal is not None
    assert signal.direction == "short"
    assert signal.stop_loss > signal.entry_price


# ---- the split that defines v2 ----


def test_the_paired_target_comes_from_the_reference_not_the_trades_own_risk():
    """The whole point of v2. A standalone trade's target is 2x its own stop;
    a paired trade's target is a PRICE the higher timeframe set, and it does
    not move when the lower timeframe offers a better entry."""
    base = uptrend()
    # A steeper reference, so its EMA9/EMA20 gap cannot be mistaken for the
    # base's own risk.
    ref = uptrend(freq="4h", up=2.4, dn=2.2)
    signal = EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref})
    assert signal is not None
    assert signal.strategy_tag == "Strategy 2.1 4H/1H"

    e9, e20 = v2._levels(ref.iloc[:-1])
    gap = e9 - e20
    risk = signal.entry_price - signal.stop_loss
    target_1 = signal.entry_price + signal.reward_risk_ratio * risk
    assert target_1 == pytest.approx(e9 + v2.TARGET_1_RATIO * gap)
    # Derived from the reference, so it is NOT the standalone's fixed 2.0.
    assert signal.reward_risk_ratio != pytest.approx(v2.TARGET_1_RATIO)


def test_the_first_target_is_a_multiple_of_its_own_stop():
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    assert signal.reward_risk_ratio == pytest.approx(v2.TARGET_1_RATIO)


def test_the_reference_levels_come_from_closed_bars_only():
    """The forming candle decides whether the reference is TOUCHING its EMA9.
    It must not move the EMA9 the target is measured from, or the target
    drifts while the candle builds."""
    base = uptrend()
    ref = uptrend(freq="4h", up=2.4, dn=2.2)
    first = EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref})

    moved = ref.copy()
    moved.loc[moved.index[-1], "close"] = float(moved["close"].iloc[-1]) * 1.02
    second = EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": moved})

    assert first is not None and second is not None
    assert second.remainder_target == pytest.approx(first.remainder_target)


# ---- the structure rule, and its changed semantics ----


def test_with_the_gate_off_the_stack_alone_decides(monkeypatch):
    """The fallback, kept so switching the gate off again is a one-line change
    with a test behind it rather than an unknown.

    An unbroken ramp contains no confirmed swings, so the structure read is
    None. The gate is ON by default now and refuses it - see
    test_the_structure_gate_is_on_by_default. Switched off, the four-MA stack
    decides on its own and the trade fires."""
    bars = ramp()
    assert v2._stack(bars.iloc[:-1]) == "up"
    assert v2._last3_trend(bars.iloc[:-1]) is None
    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is None

    monkeypatch.setattr(v2, "REQUIRE_STRUCTURE_TREND", False)
    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is not None


def test_unreadable_structure_blocks_the_trade(monkeypatch):
    """THE SEMANTICS CHANGED. v1 treated None as permission, which was safe
    when its reader returned None in 0% of 832 sampled reads. This one
    abstains 78% of the time, and permission would make the gate a no-op."""
    monkeypatch.setattr(v2, "REQUIRE_STRUCTURE_TREND", True)
    monkeypatch.setattr(v2, "structure_metrics", lambda bars: _metrics(None))
    # 15m, because the gate is per instance and 1H opts out - see
    # REQUIRE_STRUCTURE_TREND_BY_TIMEFRAME.
    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": uptrend()}) is None


def test_counter_structure_blocks_the_trade(monkeypatch):
    monkeypatch.setattr(v2, "REQUIRE_STRUCTURE_TREND", True)
    monkeypatch.setattr(v2, "structure_metrics", lambda bars: _metrics("down"))
    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": uptrend()}) is None


def test_last3_reads_a_descending_staircase_as_down():
    assert v2._last3_trend(downtrend().iloc[:-1]) == "down"


def test_an_unstacked_market_never_fires(monkeypatch):
    monkeypatch.setattr(v2, "_stack", lambda bars: None)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()}) is None


# ---- the gates ----


def test_a_stop_tighter_than_the_floor_is_refused(monkeypatch):
    """The gate that matters. A stop tighter than the spread is not a stop, it
    is a coin flip with a flattering denominator - and because R is measured
    against it, such trades report enormous R:R while being unholdable. The
    first scan measured +26.9R average below 0.15% of price and negative in
    every bucket a trade could actually be held."""
    bars = uptrend()
    entry, stop = v2._trigger(bars, "up")
    actual = abs(entry - stop) / entry
    monkeypatch.setattr(v2, "MIN_STOP_PCT", actual * 1.5)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_a_crash_wide_stop_is_refused(monkeypatch):
    bars = uptrend()
    entry, stop = v2._trigger(bars, "up")
    monkeypatch.setattr(v2, "MAX_STOP_PCT", abs(entry - stop) / entry * 0.5)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_the_net_floor_must_stay_below_two_or_no_standalone_can_ever_fire():
    """Regression guard for a bug that silently zeroed four of seven instances.

    A standalone targets 1:2 of its own stop, so gross reward:risk is exactly
    2.0 and net - (2r - maker) / (r + fees) - approaches 2.0 from BELOW for
    every stop width without ever reaching it. A 2.0 floor is not strict, it
    is unsatisfiable, and the first generation run produced zero standalone
    signals across every symbol before this was found.
    """
    assert v2.MIN_NET_REWARD_RISK < v2.TARGET_1_RATIO

    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    entry, stop = signal.entry_price, signal.stop_loss
    risk = entry - stop
    net = (v2.TARGET_1_RATIO * risk - v2.ENTRY_FEE_PCT * entry) / (
        risk + v2.round_trip_fee_for(v2.MARKET_FRACTION) * entry
    )
    assert net < v2.TARGET_1_RATIO


def test_the_net_gate_is_computed_against_the_true_taker_in_fee_not_maker():
    """The gate must use ENTRY_FEE_PCT/round_trip_fee_for(MARKET_FRACTION),
    not the flat MAKER_FEE_PCT/(MAKER_FEE_PCT+TAKER_FEE_PCT) pair every other
    strategy is close enough to. Using the wrong (maker-in) basis makes the
    gate MORE lenient than reality - it would pass signals whose true net
    reward:risk, after the real taker-in fee, is actually lower. Guards
    against silently reverting to the maker-in formula this replaced."""
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    entry, stop = signal.entry_price, signal.stop_loss
    risk = entry - stop
    reward = v2.TARGET_1_RATIO * risk

    net_correct = (reward - v2.ENTRY_FEE_PCT * entry) / (
        risk + v2.round_trip_fee_for(v2.MARKET_FRACTION) * entry
    )
    net_if_wrongly_assumed_maker = (reward - v2.MAKER_FEE_PCT * entry) / (
        risk + (v2.MAKER_FEE_PCT + v2.TAKER_FEE_PCT) * entry
    )
    assert net_correct < net_if_wrongly_assumed_maker, (
        "the true (taker-in) net must be strictly tighter than the wrong "
        "maker-in basis this replaced, or the fix has no effect"
    )

    fill_guard_net = (reward - signal.fill_guard.maker_fee_pct * entry) / (
        risk + signal.fill_guard.round_trip_fee_pct * entry
    )
    assert fill_guard_net == pytest.approx(net_correct), (
        "the FillGuard the scanner re-checks at the real fill price must use "
        "the same corrected basis as this strategy's own pre-filter"
    )


def test_a_trade_that_cannot_pay_for_itself_is_refused(monkeypatch):
    monkeypatch.setattr(v2, "MIN_NET_REWARD_RISK", 99.0)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()}) is None


def test_the_fee_is_taker_both_legs_because_entry_is_market():
    """This USED to assert the opposite (0.0008, maker in) as a guard against
    being "restored" to 0.0012 - backwards. ENTRY_MODE="next_open" enters
    the WHOLE position at market (confirmed at the actual order placement,
    scanner.py's order_type="market" for market_fraction>=1.0), so both legs
    are taker: 0.0012, not 0.0008. Traced 2026-08-27 while tracking down where
    the bot's fees actually come from; 0.0008 was itself a mistaken 'fix' at
    some earlier point that this test then locked in. Guards against being
    reverted back to 0.0008."""
    assert v2.ENTRY_FEE_PCT == pytest.approx(TAKER_FEE_PCT)
    assert v2.round_trip_fee_for(v2.MARKET_FRACTION) == pytest.approx(2 * TAKER_FEE_PCT)


# ---- what was removed ----


# ---- the EMA9 must be support, not a line being crossed ----


def test_a_broken_resistance_is_refused_even_though_it_touches():
    """The ETHUSDT defect. The final candle satisfies the one-bar touch test
    exactly - low reaches the EMA9, close is above it - but price arrived from
    BELOW, so the EMA9 was resistance that just broke rather than support that
    held. Dror: "if it long the ema9 is supposed to be support not resistence".
    """
    bars = broken_resistance()
    closed = bars.iloc[:-1]
    level = v2._levels(closed)[0]
    assert v2._touching(bars, level, "up") is True, "the one-bar touch test passes"
    assert v2._holding(closed, "up", v2.EMA9_HOLD_BARS) is False, "but the level was never held"
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_holding_requires_every_close_on_the_trend_side_of_its_own_ema9():
    up = uptrend().iloc[:-1]
    assert v2._holding(up, "up", v2.EMA9_HOLD_BARS) is True
    assert v2._holding(up, "down", v2.EMA9_HOLD_BARS) is False
    down = downtrend().iloc[:-1]
    assert v2._holding(down, "down", v2.EMA9_HOLD_BARS) is True


def test_one_close_through_the_level_ends_the_hold():
    bars = uptrend()
    broken = bars.copy()
    # push a close inside the hold window under its own EMA9
    i = broken.index[-4]
    broken.loc[i, "close"] = float(broken.loc[i, "close"]) * 0.90
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": broken}) is None


# ---- a pair needs the condition on BOTH timeframes ----


def test_a_pair_needs_structure_and_touch_on_both_timeframes():
    """Dror: "if the setup is a paired one, in both of them have to be the
    conditions for the trade"."""
    base, ref = uptrend(), uptrend(freq="4h", up=2.4, dn=2.2)
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref}) is not None

    no_touch = base.copy()
    no_touch.loc[no_touch.index[-1], "low"] = float(no_touch["close"].iloc[-1]) * 1.05
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": no_touch, "4H": ref}) is None


def test_only_the_reference_must_prove_the_hold():
    """The hold is a claim about the TREND, and the reference owns the trend.

    Requiring it of the trigger timeframe as well is not a stricter version of
    the same rule - it is fatal. Measured over 16,198 setups, paired runs 219
    with no hold, 47 at one bar, 5 at three and ZERO at ten when demanded of
    both. Paired is the only place v2's target is decoupled from its stop.
    """
    base = uptrend_short_hold()  # structure and touch, but held only 1 bar
    ref = uptrend(freq="4h", up=2.4, dn=2.2)
    assert v2.hold_run(base.iloc[:-1], "up") < v2.EMA9_HOLD_BARS
    assert v2.hold_run(ref.iloc[:-1], "up") >= v2.EMA9_HOLD_BARS

    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref}) is not None
    # the very same bars standalone, where that timeframe IS the trend-setter
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": base}) is None


def test_a_reference_that_never_held_blocks_the_pair():
    base = uptrend()
    ref = uptrend(freq="4h", up=2.4, dn=2.2)
    unheld = ref.copy()
    i = unheld.index[-3]
    unheld.loc[i, "close"] = float(unheld.loc[i, "close"]) * 0.90
    assert v2.hold_run(unheld.iloc[:-1], "up") < v2.EMA9_HOLD_BARS
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": unheld}) is None


# ---- wiring ----


def test_the_instances_are_standalone_only():
    """PAIRED REMOVED. The decoupled target was v2's whole thesis and measured
    worst of everything - -0.171R at t -5.79 for 4H/1H over 5,094 trades. The
    mechanism is understood: pairing tightens the stop by entering on the lower
    timeframe, and a tighter stop is hit more often while fees and noise stay
    fixed.

    5m is absent because declaring it normally drags the WHOLE scanner to a
    5-minute cadence - 28,800 symbol-fetches a day against today's 3,100 - and
    it has never been backtested. It needs armed_timeframes first."""
    assert all(ref is None for _, ref in INSTANCES), "no paired instances"
    # 4H AND 1D RETIRED 2026-08-19 on their own measurement: 4H at -0.058R over
    # 2,030 setups and WORSE after discarding its three best trades, which makes
    # it the population rather than a few bad trades; 1D at -0.271R on 104. 4H
    # alone was 56% of the alert volume. See INSTANCES for the table.
    #
    # 15m RETIRED 2026-08-30 - a real, held-out-confirmed gross edge that
    # cannot currently clear the fee drag its own tighter ATR imposes; see
    # INSTANCES for the full table of levers tried against it.
    assert {b for b, _ in INSTANCES} == {"1H"}
    assert "5m" not in {b for b, _ in INSTANCES}


def test_paired_instances_read_their_reference_on_the_forming_candle():
    for inst in build_instances():
        if inst.paired:
            assert inst.forming_bar_timeframes == (inst.reference_timeframe,)
            assert inst.timeframes == [inst.base_timeframe, inst.reference_timeframe]
        else:
            assert inst.forming_bar_timeframes == ()


def test_every_tag_is_unique_and_distinct_from_v1():
    tags = [i.tag for i in build_instances()]
    assert len(set(tags)) == len(tags)
    assert not any(t.startswith("Strategy 2 ") for t in tags)


# ---- what the scanner needs to be told ----


def test_a_pair_supersedes_its_own_base_timeframes_standalone():
    """They coincide on 26% of standalone triggers with the same entry level and
    the same stop, differing only in where the target sits. Acting on both puts
    2% of equity on one idea in two correlated positions - the tiered risk that
    was deliberately removed, rebuilt by accident out of instances."""
    pair = EmaTrendV2("1H", "4H")
    assert pair.supersedes == ("Strategy 2.1 1H",)
    assert EmaTrendV2("1H").tag == "Strategy 2.1 1H"
    assert EmaTrendV2("1H").supersedes == ()


def test_the_runner_has_NO_target_so_the_trail_takes_it():
    """The remainder is managed by the scanner's trailing stop, which ratchets
    to the last CONFIRMED swing low while structure still makes higher highs -
    the CHoCH exit, and already built.

    poll_trailing_stops only trails positions with NO target: "having no target
    is precisely the case this exists for". Setting one here would silently opt
    out of the trail and leave the runner on a fixed 1:3, which measured ~0.045R
    per trade worse.
    """
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    assert signal.partial_fraction == 0.5, "half still comes off at the first target"
    assert signal.remainder_target is None
    # is_final means FINAL, including final at None. It said False, which the
    # scanner read as "no opinion" and answered by inventing a daily-level
    # target - which turns the trail off for good. UNIUSDT, 2026-08-19.
    assert signal.remainder_target_is_final is True


# ---- the pre-placed limit must not peek at the bar that fills it ----


def test_rejection_mode_refuses_a_breakdown_bar():
    """THE REJECTION IS THE CONDITION. Dror, on three 15m setups where the
    entry candle closed the wrong side of its EMA9: "the candle broke the ema9,
    it didn't get rejected - that is the most important condition."

    An earlier cut filled a PRE-PLACED limit on any touch, which took exactly
    these trades - AAVEUSDT opened at 189.07, fell through its EMA9 at 187.54,
    closed at 187.19 and kept falling for three more candles. The order filled
    on the way down.

    The rejection can only be known at the close, so the trade now waits for a
    RETEST: this candle is the signal, the limit rests after it, and the fill
    happens later or not at all.
    """
    bars = uptrend()
    e9_prev = ema(bars["close"].iloc[:-1], 9).iloc[-1]
    breakdown = bars.copy()
    last = breakdown.index[-1]
    breakdown.loc[last, "low"] = e9_prev * 0.985
    breakdown.loc[last, "close"] = e9_prev * 0.99  # closes BELOW its own EMA9

    assert v2._touching(breakdown, e9_prev, "up") is False
    v2.ENTRY_MODE = "rejection"
    try:
        assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": breakdown}) is None
    finally:
        v2.ENTRY_MODE = "band"


def test_band_mode_does_not_wait_for_the_rejection():
    """In band mode the resting order IS the trigger: it fills before any close
    is known, so the trigger timeframe is not asked for a rejection. That is the
    whole difference from rejection mode, and it is worth 0.2R per trade - in
    the direction of taking trades Dror would refuse on a chart."""
    assert v2.ENTRY_MODE == "band"
    bars = uptrend()
    e9_prev = ema(bars["close"].iloc[:-1], 9).iloc[-1]
    breakdown = bars.copy()
    last = breakdown.index[-1]
    breakdown.loc[last, "low"] = e9_prev * 0.985
    breakdown.loc[last, "close"] = e9_prev * 0.99  # closes the WRONG side

    assert v2._touching(breakdown, e9_prev, "up") is False
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": breakdown}) is not None, (
        "band mode fills on the touch regardless of where the candle closes"
    )


def test_the_trend_read_never_looks_at_the_entry_bar():
    """Every decision comes from bars that closed before the order was placed,
    so moving the entry bar's close cannot change whether the trade exists."""
    bars = uptrend()
    moved = bars.copy()
    last = moved.index[-1]
    moved.loc[last, "close"] = float(moved.loc[last, "close"]) * 1.03

    a = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars})
    b = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": moved})
    assert (a is None) == (b is None)
    if a is not None:
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.stop_loss == pytest.approx(b.stop_loss)


# ---- one trade per unbroken EMA9 run ----


def test_one_run_produces_one_dedupe_key_not_one_per_bar():
    """WLDUSDT fired the SAME 4H long on eight consecutive bars because the key
    included the entry price, and the entry is `ema9_prev` - a level that
    drifts every bar. Eight copies of one setup are not eight observations, so
    this inflated n and overstated every standard error computed over the
    population."""
    closes = _staircase()
    keys = []
    for extra in range(_TAIL, _TAIL + 4):
        c = closes + [closes[-1] + 1.2 * i for i in range(1, extra + 1)]
        e9_prev = ema(pd.Series(c[:-1]), 9).iloc[-1]
        bars = _bars(c, last_low=e9_prev * 0.9999)
        s = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars})
        if s is not None:
            keys.append(s.dedupe_key)
    assert len(keys) >= 2, "fixture must produce several bars of one run"
    assert len(set(keys)) == 1, f"one unbroken run must be one trade, got {set(keys)}"


def test_breaking_the_level_starts_a_new_run():
    a = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    closes = _staircase()
    closes += [closes[-1] + 1.2 * i for i in range(1, 6)]
    closes += [closes[-1] * 0.93]           # a close through the EMA9 ends the run
    closes += [closes[-1] * 1.02 + 1.2 * i for i in range(1, 9)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    b = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": _bars(closes, last_low=e9_prev * 0.9999)})
    if a is not None and b is not None:
        assert a.dedupe_key != b.dedupe_key


# ---- the trigger timeframe's own EMA9 must not already be broken ----


def test_a_base_already_through_its_ema9_is_refused():
    """Dror, on INJUSDT: "the strategy core idea is that the ema9 is a
    resistance or support, so once it broke like that it is no longer valid."
    LABUSDT entered long after the 1H had closed BELOW its EMA9 twice - it
    passed because the hold was only ever asked of the reference."""
    base, ref = uptrend(), uptrend(freq="4h", up=2.4, dn=2.2)
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref}) is not None

    broken = base.copy()
    i = broken.index[-2]                     # the bar BEFORE the entry closes through
    broken.loc[i, "close"] = float(broken.loc[i, "close"]) * 0.90
    assert v2.hold_run(broken.iloc[:-1], "up") < v2.BASE_HOLD_BARS
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": broken, "4H": ref}) is None


def test_a_pair_checks_the_stack_on_the_trigger_timeframe_too(monkeypatch):
    """Decision 3 asked for the stack on both timeframes; only the reference
    was ever checked. Dror found it on a FIGHTUSDT short whose 1H EMA9 and
    EMA20 sat 0.3% apart and visibly crossing."""
    base, ref = uptrend(), uptrend(freq="4h", up=2.4, dn=2.2)
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref}) is not None

    real = v2._stack
    calls = {"n": 0}

    def ref_ok_base_not(bars):
        calls["n"] += 1
        return real(bars) if calls["n"] == 1 else None

    monkeypatch.setattr(v2, "_stack", ref_ok_base_not)
    assert EmaTrendV2("1H", "4H").evaluate("TESTUSDT", {"1H": base, "4H": ref}) is None


# ---- the trend read must have scale, not just shape ----


def test_structure_metrics_reports_span_drift_and_crossings():
    m = v2.structure_metrics(uptrend().iloc[:-1])
    assert m["trend"] == "up"
    assert m["span"] > 0
    assert m["drift_high"] > 0 and m["drift_low"] > 0
    assert m["crossings"] >= 0


def test_drift_is_signed_in_the_trends_direction():
    """So a genuine trend is positive on both series whichever way it runs, and
    one filter covers longs and shorts."""
    up = v2.structure_metrics(uptrend().iloc[:-1])
    down = v2.structure_metrics(downtrend().iloc[:-1])
    assert up["trend"] == "up" and down["trend"] == "down"
    assert min(up["drift_high"], up["drift_low"]) > 0
    assert min(down["drift_high"], down["drift_low"]) > 0


def test_a_structure_forming_too_quickly_is_refused(monkeypatch):
    """LABUSDT's 1H read "up" on healthy drift, but all six pivots landed
    inside TEN HOURS - one leg with wobble in it, not a structure. AVAX was 15
    hours, SOXS 21."""
    bars = uptrend()
    real = v2.structure_metrics(bars.iloc[:-1])
    monkeypatch.setattr(v2, "MIN_PIVOT_SPAN_BARS", real["span"] + 5)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_a_trend_too_gentle_to_be_one_is_refused(monkeypatch):
    """Dror: "the found trend is too gentle." DOGEUSDT's highs drifted 0.58 ATR
    across 26 daily bars and the rule called it a downtrend."""
    bars = uptrend()
    real = v2.structure_metrics(bars.iloc[:-1])
    monkeypatch.setattr(v2, "MIN_SWING_DRIFT_ATR", min(real["drift_high"], real["drift_low"]) + 1)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_a_chopped_level_is_refused(monkeypatch):
    """v1 refused more than ONE EMA9 crossing in 30 bars; decision 5 dropped
    the filter. SOXSUSDT had ten - Dror: "the graph broke 3 times the ema9
    before the setup so it isn't legit"."""
    bars = uptrend()
    monkeypatch.setattr(v2, "MAX_EMA9_CROSSINGS", -1)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_the_scale_conditions_default_to_off():
    """They are swept, not chosen. Shipping a guessed value is what put
    EMA9_HOLD_BARS at 10 and MIN_NET_REWARD_RISK at an unsatisfiable 2.0."""
    assert v2.MIN_PIVOT_SPAN_BARS == 0
    assert v2.MIN_SWING_DRIFT_ATR == 0.0
    assert v2.MAX_EMA9_CROSSINGS >= 999


def test_an_unset_scale_threshold_blocks_nothing(monkeypatch):
    """structure_metrics signs drift by the trend it read, and signs by -1 when
    it read NONE - so an unreadable structure produces a negative drift, and a
    threshold of 0.0 refused it. That re-implemented the structure gate through
    the drift check and kept refusing 7 of 16 setups Dror marked by eye after
    the gate had supposedly been switched off.

    Tested WITH THE GATE OFF, which is the only configuration where the bug can
    bite: with the gate on, an unreadable structure is refused by the gate
    itself before the drift check is ever reached, so the sign never matters.
    The guard is kept for the day the gate goes off again."""
    monkeypatch.setattr(v2, "REQUIRE_STRUCTURE_TREND", False)
    assert v2.MIN_SWING_DRIFT_ATR == 0.0
    bars = ramp()  # stacked, but structurally unreadable -> drift signs negative
    m = v2.structure_metrics(bars.iloc[:-1])
    assert m["trend"] is None
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is not None

    monkeypatch_free = v2.MIN_SWING_DRIFT_ATR
    try:
        v2.MIN_SWING_DRIFT_ATR = 1.0
        assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None
    finally:
        v2.MIN_SWING_DRIFT_ATR = monkeypatch_free


def test_the_touch_band_applies_to_the_trigger_as_well_as_the_condition():
    """The band did nothing at any size until _trigger honoured it too: the
    condition accepted the setup and the trigger then refused it on an exact
    touch. Sweeping 0 to 1.0 ATR and seeing recall not move AT ALL is what
    exposed that - a flat sweep is usually a disconnected wire."""
    closes = _staircase()
    closes += [closes[-1] + 1.2 * i for i in range(1, _TAIL + 1)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    # a low that stops just SHORT of the EMA9 - no touch without a band
    near = _bars(closes, last_low=e9_prev * 1.004)

    v2.EMA9_TOUCH_ATR = 0.0
    try:
        assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": near}) is None
        assert v2._trigger(near, "up") is None
        v2.EMA9_TOUCH_ATR = 1.5
        assert v2._trigger(near, "up") is not None, "the trigger must honour the band"
        assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": near}) is not None
    finally:
        v2.EMA9_TOUCH_ATR = 0.35


def test_every_instance_tag_resolves_to_its_own_trailing_timeframe():
    """The trail reads its timeframe off the tag, so a 15m trade trails 15m
    swings and a 1D trade trails daily ones. Getting this wrong is silent:
    trailing a 1D setup on 15m lows ratchets the stop into the first noise, and
    the reverse never moves the stop at all."""
    from notifier.scanner import Scanner

    for base, _ref in INSTANCES:
        tag = f"Strategy 2.1 {base}"
        assert Scanner.trail_timeframe(None, tag) == base, tag


def test_the_signal_carries_its_gates_to_be_re_asked_at_the_fill():
    """entry_price is the EMA9 that SELECTED the setup, and ENTRY_MODE
    "next_open" opens at market on the following candle - which has closed back
    on the trend side by construction, so the fill is always on the far side of
    the level.

    HYPEUSDT, 2026-08-18: 1.50% stop measured from its EMA9, 0.15% measured
    from the price it would have filled at. It passed this file's own 0.30%
    floor while failing it by half, because the gate and the trade were talking
    about two different prices.

    The thresholds therefore ride along on the signal for the scanner to
    re-apply. The ATR does too: a floor in percent cannot see volatility.
    """
    sig = EmaTrendV2("4H").evaluate("BTCUSDT", {"4H": uptrend(freq="4h")})
    assert sig is not None

    guard = sig.fill_guard
    assert guard is not None, "a level-selected entry must carry its gates"
    assert guard.min_stop_pct == v2.MIN_STOP_PCT
    assert guard.max_stop_pct == v2.MAX_STOP_PCT
    assert guard.atr and guard.atr > 0, "the stop cannot be judged in ATR without one"

    # And it must actually bite: the same stop measured from a fill on the far
    # side of the EMA9 is a different trade.
    assert guard.refuses(sig.entry_price, sig.stop_loss, 2.0) is None
    barely_above = sig.stop_loss * (1 + v2.MIN_STOP_PCT / 2)
    assert guard.refuses(barely_above, sig.stop_loss, 2.0) is not None


def test_the_atr_floor_applies_only_to_the_timeframes_it_was_measured_on():
    """The sweep behind MIN_STOP_ATR walks a 1H cache; 4H and 1D resample from
    it and 15m cannot be derived at all. So 15m has no evidence for this floor
    in either direction, and it is left off there.

    Not a technicality. Applied to 15m as well, 1.5 ATR refuses BTWUSDT at 0.96
    ATR - which ran +36R - while allowing SOLUSDT at 1.63 ATR, the setup Dror
    said had the wrong entry. Both are 15m, and it reads both backwards.
    """
    assert v2.MIN_STOP_ATR["15m"] == 0.0, "15m has never been swept for this"
    assert v2.MIN_STOP_ATR["1H"] > 0, "1H was measured and the floor helps there"
    # 4H measured WORSE at every floor, and 1D's widest stop in two years is
    # 1.41 ATR - so a 1.5 floor there is a shutdown, not a filter.
    #
    # The 4H zero is a DECISION, not an absence of one: Dror was shown that the
    # measurement contradicts his own read of LABUSDT's stop and chose "keep the
    # 4h floor off, trust the measurement" on 2026-08-19. Pinned so it cannot be
    # reintroduced as a side effect of tuning something else.
    assert v2.MIN_STOP_ATR["4H"] == 0.0
    assert v2.MIN_STOP_ATR["1D"] == 0.0

    on_4h = EmaTrendV2("4H").evaluate("BTCUSDT", {"4H": uptrend(freq="4h")})
    on_15m = EmaTrendV2("15m").evaluate("BTCUSDT", {"15m": uptrend(freq="15min")})
    assert on_4h is not None and on_15m is not None
    assert on_4h.fill_guard.min_stop_atr == v2.MIN_STOP_ATR["4H"]
    assert on_15m.fill_guard.min_stop_atr == 0.0


def test_every_instance_that_ships_declares_a_floor_or_deliberately_none():
    """A new instance must not silently inherit 0.0 by falling off the mapping -
    that is how a gate goes quiet without anyone deciding it should."""
    for base, _ref in INSTANCES:
        assert base in v2.MIN_STOP_ATR, (
            f"{base} has no MIN_STOP_ATR entry; set it from its own sweep, "
            "or write 0.0 to say it has not been measured"
        )


def test_the_hold_is_per_instance_and_a_generator_can_still_disable_it():
    """1H carries a 20-bar hold; 15m keeps 2 because it has never been swept.

    The scalar EMA9_HOLD_BARS still wins when it is zeroed, and that is not a
    nicety: both generators zero it at import to build the widest population,
    and an override that survived would pre-filter 1H at 20. Every sweep over
    that population would then compare a subset against a whole while looking
    exactly like a sweep over one population - which is the failure the
    generate-wide-filter-afterwards structure exists to prevent.
    """
    assert v2._hold_bars("1H") == 20
    assert v2._hold_bars("15m") == v2.EMA9_HOLD_BARS == 2
    assert v2._hold_bars(None) == 2, "an unknown timeframe falls back, never to zero"

    original = v2.EMA9_HOLD_BARS
    try:
        v2.EMA9_HOLD_BARS = 0
        assert v2._hold_bars("1H") == 0, "zeroing for generation must beat the override"
    finally:
        v2.EMA9_HOLD_BARS = original


def test_a_retired_instance_keeps_its_exit_permission():
    """Unregistering a tag removes it from LIVE_TAGS instantly, and with it the
    bot's permission to move the stop or place the take-profit on any position
    already open under it - the same silent orphaning as a hand-typed /add tag,
    arriving all at once. Nothing was open under 2.1's 4H or 1D when they were
    retired, but a signal approved between the decision and the deploy would
    have been."""
    from notifier.main import EXIT_MANAGED_TAGS, LIVE_TAGS

    for tag in ("Strategy 2.1 4H", "Strategy 2.1 1D"):
        assert tag not in LIVE_TAGS, "a retired instance must not open new trades"
        assert tag in EXIT_MANAGED_TAGS, "but must keep managing what it opened"


def test_the_reason_reports_the_structure_it_actually_measured():
    """DRAMUSDT #1061, 2026-08-21: the alert said "15m stack and last-3
    structure both up" on a setup whose structure read was literally None -
    lows 58.04 / 58.80 / 58.55, the last one lower - and whose price had
    crossed its EMA9 seven times in thirty bars. Dror, reading the chart:
    "there was a lower high recently that broke the rising highs".

    The string was hardcoded to describe `trend`, which comes from _stack()
    alone, so EVERY signal asserted last-3 agreement whether or not it held -
    and REQUIRE_STRUCTURE_TREND is False, so the claim was never even tested.
    An alert must say what was checked, not what would have been reassuring.
    """
    bars = uptrend()
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars})
    assert signal is not None

    assert "structure both" not in signal.reason, (
        "an unconditional claim of agreement is the bug; the read is not gated at all"
    )
    measured = structure_metrics(bars.iloc[:-1])["trend"]
    assert f"last-3 structure {measured or 'unreadable'}" in signal.reason, (
        f"the reason must name what structure_metrics actually returned ({measured!r}): "
        f"{signal.reason}"
    )


def test_agreeing_structure_still_fires():
    """The other half: the gate must refuse what disagrees WITHOUT refusing
    everything. A staircase that actually makes higher highs and higher lows
    is what the rule is meant to let through."""
    bars = uptrend()
    assert v2.structure_metrics(bars.iloc[:-1])["trend"] == "up"

    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is not None


def test_the_gate_is_per_instance_on_for_15m_off_for_1h():
    """Each instance gets what its OWN measurement asks for - the same pattern
    MIN_STOP_ATR and EMA9_HOLD_BARS_BY_TIMEFRAME already follow.

    Measured 2026-08-21 on each instance's own population, choosing on the
    first half and scoring the held-out half:

        15m  keep -0.120R  drop -0.245R  gap +0.125  t 2.95  n 2,720
        1H   keep -0.020R  drop +0.076R  gap -0.096  t -0.20  n   213

    On 15m the gate refuses the worse trades, significantly. On 1H it keeps
    them - not significantly, but there is no case for paying its ~65% recall
    cost to make an instance no better.

    The +0.100R once quoted for 1H came from a population that pooled 14,694
    1H setups with 3,111 4H and 176 1D ones, both of those instances RETIRED
    on 2026-08-19. Measured on 1H alone the benefit is not there. Anything
    reading signals_v2_ship.pkl must filter on the instance index.
    """
    bars = ramp()  # stacked, structurally unreadable
    assert v2._stack(bars.iloc[:-1]) == "up"
    assert v2._last3_trend(bars.iloc[:-1]) is None

    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is None, "15m gates on structure"
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is not None, "1H does not"


def test_a_generator_switching_the_gate_off_beats_the_per_instance_map():
    """generate_v2 and generate_15m set REQUIRE_STRUCTURE_TREND = False so the
    population stays wide and every candidate rule is a filter applied
    afterwards. A per-instance map that ignored that would silently pre-filter
    15m - the exact failure _hold_bars already guards against for the hold."""
    bars = ramp()
    assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is None

    original = v2.REQUIRE_STRUCTURE_TREND
    try:
        v2.REQUIRE_STRUCTURE_TREND = False
        assert EmaTrendV2("15m").evaluate("TESTUSDT", {"15m": bars}) is not None, (
            "switching the scalar off must disable the gate on EVERY instance"
        )
    finally:
        v2.REQUIRE_STRUCTURE_TREND = original


# --------------------------------------------------------------------------
# market_trend_symbol: the same cross-symbol gate RsiFibReversal has, mirrored
# here (see tests/test_rsi_fib_reversal.py for the template this follows).
# Independent of `reference_timeframe` above - that is this strategy's OWN
# multi-timeframe confluence, this is a check against another symbol entirely.
# --------------------------------------------------------------------------

def _reference_bars(direction: str) -> pd.DataFrame:
    """A reference symbol's bars reading a clean, unambiguous trend by the
    same price-vs-200MA convention the strategy already uses on itself."""
    n = v2.TREND_MA_PERIOD + 10
    closes = [100.0 + i for i in range(n)] if direction == "up" else [200.0 - i for i in range(n)]
    return _bars(closes)


def test_default_instance_has_no_market_trend_gate():
    strat = EmaTrendV2("1H")
    assert strat.market_trend_symbol is None
    assert strat.timeframes == ["1H"]


def test_instance_with_a_reference_symbol_declares_its_compound_timeframe():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "BTCUSDT@1H"]


def test_a_paired_instance_keeps_both_its_own_reference_and_the_market_one():
    strat = EmaTrendV2("1H", "4H", market_trend_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "4H", "BTCUSDT@1H"]


def test_long_fires_when_the_reference_agrees_uptrend():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1H": _reference_bars("up")})
    assert signal is not None
    assert signal.direction == "long"


def test_long_is_gated_when_the_reference_disagrees_downtrend():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1H": _reference_bars("down")})
    assert signal is None


def test_short_fires_when_the_reference_agrees_downtrend():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": downtrend(), "BTCUSDT@1H": _reference_bars("down")})
    assert signal is not None
    assert signal.direction == "short"


def test_short_is_gated_when_the_reference_disagrees_uptrend():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": downtrend(), "BTCUSDT@1H": _reference_bars("up")})
    assert signal is None


def test_missing_reference_data_fails_open_not_closed():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend()})  # no "BTCUSDT@1H" key at all
    assert signal is not None


def test_short_reference_history_fails_open_too():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT")
    thin_reference = _bars([100.0] * 50)  # well under TREND_MA_PERIOD + 1
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1H": thin_reference})
    assert signal is not None


# --------------------------------------------------------------------------
# market_regime_symbol: a DIFFERENT read of the same idea - structure +
# level-proximity on the reference's DAILY chart (daily_regime_from_bars),
# always "@1D" regardless of the instance's own base_timeframe. Independent
# of market_trend_symbol above. Measured 2026-08-28 on this live instance:
# gated -0.331R/-0.447R vs removed -0.474R/-0.824R across both years,
# surviving drop-top-3 and the floor removed - the one gate this session
# actually recommends shipping.
# --------------------------------------------------------------------------

def _daily_bars(closes) -> pd.DataFrame:
    s = pd.Series(closes)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq="D"),
            "open": s,
            "high": s + 2.0,
            "low": s - 2.0,
            "close": s,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _daily_ramp(a: float, b: float, n: int) -> list[float]:
    step = (b - a) / n
    return [a + step * (i + 1) for i in range(n)]


def _confirmed_daily_downtrend() -> list[float]:
    """Verbatim from tests/test_regime.py's own validated fixture - a
    genuine, OBSERVED change of character, not the bootstrap guess."""
    rising = _daily_ramp(100, 200, 40) + _daily_ramp(200, 170, 15) + _daily_ramp(170, 260, 40) + _daily_ramp(260, 235, 10)
    reversal = rising + _daily_ramp(260, 150, 30) + _daily_ramp(150, 200, 15)
    return reversal + _daily_ramp(200, 80, 20)


def _daily_regime_reference(direction: str) -> pd.DataFrame:
    """A confirmed daily uptrend or downtrend at a fresh extreme (nothing
    confirmed ahead of price to gate on), padded well past
    structure_context's MIN_LOOKBACK (200) so the growing-window search has
    somewhere to land. Mirrored via 500-x for the "up" case rather than
    hand-derived, since the down shape is the one already numerically
    verified in tests/test_regime.py."""
    down = _confirmed_daily_downtrend()
    if direction == "down":
        closes = down + _daily_ramp(down[-1], down[-1] - 5, 220)
    else:
        up = [500 - c for c in down]
        closes = up + _daily_ramp(up[-1], up[-1] + 5, 220)
    return _daily_bars(closes)


def test_default_instance_has_no_market_regime_gate():
    strat = EmaTrendV2("1H")
    assert strat.market_regime_symbol is None
    assert strat.timeframes == ["1H"]


def test_instance_with_a_regime_symbol_declares_its_1D_compound_timeframe():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "BTCUSDT@1D"]
    assert "+BTCUSDT(1D)" in strat.tag


def test_a_1h_instance_reads_the_regime_on_1D_not_its_own_timeframe():
    """The regime reference is ALWAYS daily, regardless of the instance's
    own base_timeframe - unlike market_trend_symbol, which follows it."""
    strat = EmaTrendV2("15m", market_regime_symbol="BTCUSDT")
    assert strat.timeframes == ["15m", "BTCUSDT@1D"]


def test_long_fires_when_the_regime_agrees_uptrend():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1D": _daily_regime_reference("up")})
    assert signal is not None
    assert signal.direction == "long"


def test_long_is_gated_when_the_regime_disagrees_downtrend():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1D": _daily_regime_reference("down")})
    assert signal is None


def test_short_fires_when_the_regime_agrees_downtrend():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": downtrend(), "BTCUSDT@1D": _daily_regime_reference("down")})
    assert signal is not None
    assert signal.direction == "short"


def test_short_is_gated_when_the_regime_disagrees_uptrend():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": downtrend(), "BTCUSDT@1D": _daily_regime_reference("up")})
    assert signal is None


def test_no_daily_reading_fails_open_not_closed():
    """A plain ramp with no observed CHoCH reads None (see test_regime.py) -
    not evidence the market disagrees, so the trade still fires."""
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    no_reading = _daily_bars(_daily_ramp(100, 200, 300))
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend(), "BTCUSDT@1D": no_reading})
    assert signal is not None


def test_missing_regime_reference_fails_open_not_closed():
    strat = EmaTrendV2("1H", market_regime_symbol="BTCUSDT")
    signal = strat.evaluate("TESTUSDT", {"1H": uptrend()})  # no "BTCUSDT@1D" key at all
    assert signal is not None


def test_both_gates_can_be_set_independently_at_once():
    strat = EmaTrendV2("1H", market_trend_symbol="BTCUSDT", market_regime_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "BTCUSDT@1H", "BTCUSDT@1D"]
    assert "+BTCUSDT" in strat.tag and "+BTCUSDT(1D)" in strat.tag


# ---------------------------------------------------------------------------
# chart_overlay: the EMA9/EMA20 stack the entry logic itself reads.
# ---------------------------------------------------------------------------


def test_chart_overlay_draws_ema9_and_ema20_over_the_base_timeframe():
    bars = uptrend()
    strategy = EmaTrendV2("1H")
    signal = strategy.evaluate("TESTUSDT", {"1H": bars})
    assert signal is not None  # sanity: this fixture is meant to fire

    overlay = strategy.chart_overlay({"1H": bars}, signal)

    names = [name for name, _series, _color in overlay.series]
    assert "EMA9" in names
    assert "EMA20" in names


def test_chart_overlay_series_are_indexed_like_the_full_base_frame():
    """render() reindexes each series against `bars` before slicing to the
    visible tail (see chart.render) - so the series returned here must share
    bars' own index, not a shorter/re-based one."""
    bars = uptrend()
    strategy = EmaTrendV2("1H")
    signal = strategy.evaluate("TESTUSDT", {"1H": bars})

    overlay = strategy.chart_overlay({"1H": bars}, signal)

    for _name, series, _color in overlay.series:
        assert list(series.index) == list(bars.index)


def test_chart_overlay_returns_none_when_its_own_timeframe_is_missing():
    strategy = EmaTrendV2("1H")
    signal = strategy.evaluate("TESTUSDT", {"1H": uptrend()})

    assert strategy.chart_overlay({}, signal) is None
