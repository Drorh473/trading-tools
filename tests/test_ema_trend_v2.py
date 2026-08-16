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

from notifier.strategies import ema_trend_v2 as v2
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.ema_trend_v2 import INSTANCES, EmaTrendV2, build_instances
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


# Long enough that the last EMA9_HOLD_BARS closes all sit on the trend side of
# their own EMA9 before the pullback. A shorter tail leaves part of the
# staircase's down-leg inside the hold window, and the setup is correctly
# refused - which is what the hold is for.
_TAIL = 14


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
    # sag beneath the EMA9, then spike back through it
    closes += [closes[-1] - 1.6 * i for i in range(1, 13)]
    closes += [closes[-1] + 5.0 * i for i in range(1, 5)]
    e9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, last_low=e9_prev * 0.999)


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
    assert signal.limit_note == "EMA9"
    assert signal.stop_loss < signal.entry_price


def test_the_entry_is_the_PREVIOUS_bars_ema9_not_this_bars():
    """The lookahead guard. A limit cannot rest at a level computed from the
    close of the candle that fills it - measuring it that way is exactly what
    made the first pass of the pre-placed variant look better than it was."""
    bars = uptrend()
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars})
    prev = ema(bars["close"].iloc[:-1], 9).iloc[-1]
    current = ema(bars["close"], 9).iloc[-1]
    assert signal.entry_price == pytest.approx(prev)
    assert signal.entry_price != pytest.approx(current)


def test_there_is_no_market_fraction():
    """Load-bearing, not incidental: measured, a limit fill at EMA9 is 7.1:1
    while a market entry at the same bar's close is 2.3:1."""
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    assert signal.market_fraction == 0.0
    assert signal.limit_entry == signal.entry_price


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
    assert signal.remainder_target == pytest.approx(e9 + v2.TARGET_2_RATIO * gap)

    risk = signal.entry_price - signal.stop_loss
    target_1 = signal.entry_price + signal.reward_risk_ratio * risk
    assert target_1 == pytest.approx(e9 + v2.TARGET_1_RATIO * gap)
    # Derived from the reference, so it is NOT the standalone's fixed 2.0.
    assert signal.reward_risk_ratio != pytest.approx(v2.TARGET_1_RATIO)


def test_a_standalone_target_is_a_multiple_of_its_own_stop():
    signal = EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()})
    risk = signal.entry_price - signal.stop_loss
    assert signal.reward_risk_ratio == pytest.approx(v2.TARGET_1_RATIO)
    assert signal.remainder_target == pytest.approx(signal.entry_price + v2.TARGET_2_RATIO * risk)


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


def test_a_ramp_has_no_pivots_so_no_trade():
    """Documents why every fixture here is a staircase. The four MAs are
    perfectly stacked and price touches EMA9 - and v2 still refuses, because
    an unbroken ramp contains no confirmed swings to read a trend from."""
    bars = ramp()
    assert v2._stack(bars.iloc[:-1]) == "up"
    assert v2._last3_trend(bars.iloc[:-1]) is None
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_unreadable_structure_blocks_the_trade(monkeypatch):
    """THE SEMANTICS CHANGED. v1 treated None as permission, which was safe
    when its reader returned None in 0% of 832 sampled reads. This one
    abstains 78% of the time, and permission would make the gate a no-op."""
    monkeypatch.setattr(v2, "_last3_trend", lambda bars: None)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()}) is None


def test_counter_structure_blocks_the_trade(monkeypatch):
    monkeypatch.setattr(v2, "_last3_trend", lambda bars: "down")
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()}) is None


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
    net = (v2.TARGET_1_RATIO * risk - v2.MAKER_FEE_PCT * entry) / (risk + v2.ROUND_TRIP_FEE_PCT * entry)
    assert net < v2.TARGET_1_RATIO


def test_a_trade_that_cannot_pay_for_itself_is_refused(monkeypatch):
    monkeypatch.setattr(v2, "MIN_NET_REWARD_RISK", 99.0)
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": uptrend()}) is None


def test_the_fee_is_maker_in_taker_out():
    """Same correction as v1: this strategy never places a market order, so
    its entry is a maker fill. Guards against being 'restored' to 0.0012."""
    assert v2.ROUND_TRIP_FEE_PCT == pytest.approx(0.0008)
    assert v2.MAKER_FEE_PCT == pytest.approx(0.0002)


# ---- what was removed ----


def test_v1_refuses_the_setup_v2_takes():
    """The removals, demonstrated rather than asserted. The staircase crosses
    its EMA9 once per cycle, so v1's chop filter (at most one crossing in 30
    bars) rejects it, and its 0.05 x ATR proximity band rejects a touch that
    actually reaches the level. v2 fires on the same bars."""
    bars = uptrend()
    assert EmaTrendFollowing("1H").evaluate("TESTUSDT", {"1H": bars}) is None
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is not None


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
    assert v2._holding(closed, "up") is False, "but the level was never held"
    assert EmaTrendV2("1H").evaluate("TESTUSDT", {"1H": bars}) is None


def test_holding_requires_every_close_on_the_trend_side_of_its_own_ema9():
    up = uptrend().iloc[:-1]
    assert v2._holding(up, "up") is True
    assert v2._holding(up, "down") is False
    down = downtrend().iloc[:-1]
    assert v2._holding(down, "down") is True


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


def test_there_are_seven_instances_four_standalone_and_three_paired():
    assert len(INSTANCES) == 7
    standalone = [i for i in INSTANCES if i[1] is None]
    paired = [i for i in INSTANCES if i[1] is not None]
    assert len(standalone) == 4 and len(paired) == 3
    assert {b for b, _ in standalone} == {"15m", "1H", "4H", "1D"}


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
