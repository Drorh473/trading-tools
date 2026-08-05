import pandas as pd
import pytest

from notifier.strategies import rsi_fib_reversal
from notifier.strategies.rsi_fib_reversal import (
    FIB_ENTRY,
    FIB_STOP,
    RsiFibReversal,
    _downtrend_leg,
    _uptrend_leg,
)


# Every bar carries a high/low range, so ATR reflects the market's own
# volatility. With range-less bars ATR is set entirely by the pullback that
# triggers the signal, which makes any pullback deep enough to move RSI
# automatically deep enough to confirm a reversal pivot and end the leg -
# a fixture artifact rather than anything a real chart does.
WICK = 3.0


def _bars_from_closes(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    series = pd.Series(closes)
    high = pd.Series(highs) if highs is not None else series + WICK
    low = pd.Series(lows) if lows is not None else series - WICK
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(series), freq="h"),
            "open": series,
            "high": high,
            "low": low,
            "close": series,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _evaluate(symbol, closes, highs=None, lows=None):
    return RsiFibReversal().evaluate(symbol, {"1H": _bars_from_closes(closes, highs, lows)})


def _ramp(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


# A rising market with real structure: a first leg up, a genuine interim
# pullback that forms a swing low at 140, then the leg being retraced up to
# 300, then the sharp drop that pushes RSI(10) below 30. The swing detector
# needs an actual reversal to anchor on - a straight line has no pivots.
#
# The interim pullback is deliberately much larger than the one that triggers
# the signal. That gap is the whole point: a reversal big enough to end the
# previous leg has to be distinguishable from a retracement within the current
# one, and a threshold can only tell them apart if the chart itself does.
# The rally flattens before it rolls over, so RSI is already coming off the
# boil when the pullback starts. Without that, only a pullback steep enough to
# also count as a reversal could drag RSI(10) under 30, and the fixture could
# not express "still inside the leg" and "oversold" at the same time.
UPTREND_SWING_LOW = 140.0
UPTREND_PEAK = 300.0
UPTREND = [
    100.0,
    *_ramp(100, 200, 99),
    *_ramp(200, UPTREND_SWING_LOW, 20),
    *_ramp(UPTREND_SWING_LOW, 290, 140),
    *_ramp(290, UPTREND_PEAK, 20),
]
# Ends on the bar RSI(10) actually crosses 30, since evaluate() looks for the
# crossing itself rather than for RSI merely sitting below it.
UPTREND_PULLBACK = [UPTREND_PEAK - i * 3 for i in range(1, 5)]

DOWNTREND_SWING_HIGH = 260.0
DOWNTREND_TROUGH = 100.0
DOWNTREND = [
    300.0,
    *_ramp(300, 200, 99),
    *_ramp(200, DOWNTREND_SWING_HIGH, 20),
    *_ramp(DOWNTREND_SWING_HIGH, 110, 140),
    *_ramp(110, DOWNTREND_TROUGH, 20),
]
DOWNTREND_BOUNCE = [DOWNTREND_TROUGH + i * 3 for i in range(1, 5)]


def test_fires_long_on_oversold_rsi_cross_above_200ma():
    signal = _evaluate("BTCUSDT", UPTREND + UPTREND_PULLBACK)

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 1 1H"
    assert signal.reward_risk_ratio == 2.0
    assert signal.stop_loss < signal.entry_price < UPTREND_PULLBACK[-1]


def test_fires_short_on_overbought_rsi_cross_below_200ma():
    signal = _evaluate("ETHUSDT", DOWNTREND + DOWNTREND_BOUNCE)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.reward_risk_ratio == 2.0
    assert DOWNTREND_BOUNCE[-1] < signal.entry_price < signal.stop_loss


def test_no_signal_without_enough_history():
    assert _evaluate("BTCUSDT", [100.0] * 50) is None


def test_no_signal_when_rsi_not_crossing():
    # Flat price series: RSI stays near 50, never crosses 30/70.
    assert _evaluate("BTCUSDT", [100.0] * 210) is None


def test_uptrend_leg_anchors_on_the_pivot_that_started_the_leg():
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)

    swing_low, swing_high = _uptrend_leg(bars)

    # The interim pullback low, not the 100.0 the whole series started from,
    # and read off the wicks rather than the closes.
    assert swing_low == pytest.approx(UPTREND_SWING_LOW - WICK)
    assert swing_high == pytest.approx(UPTREND_PEAK + WICK)


def test_downtrend_leg_anchors_on_the_pivot_that_started_the_leg():
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high == pytest.approx(DOWNTREND_SWING_HIGH + WICK)
    assert swing_low == pytest.approx(DOWNTREND_TROUGH - WICK)


def test_swing_uses_wick_extremes_not_closes():
    # The pivot candle wicks below its close and the peak candle wicks above
    # its close; both wicks are the real extremes and must anchor the Fib.
    closes = UPTREND + UPTREND_PULLBACK
    highs = [c + WICK for c in closes]
    lows = [c - WICK for c in closes]
    peak_idx = closes.index(max(closes))
    pivot_idx = closes.index(min(closes[100:125]))  # the interim structural low
    highs[peak_idx] = closes[peak_idx] + 8
    lows[pivot_idx] = closes[pivot_idx] - 15

    signal = _evaluate("BTCUSDT", closes, highs=highs, lows=lows)

    assert signal is not None
    expected_range = highs[peak_idx] - lows[pivot_idx]
    assert signal.entry_price == pytest.approx(highs[peak_idx] - expected_range * FIB_ENTRY)
    assert signal.stop_loss == pytest.approx(highs[peak_idx] - expected_range * FIB_STOP)


def test_swing_ignores_deeper_low_after_the_peak():
    # A deeper low during the pullback that triggers the signal is part of the
    # retracement, not the start of the leg being retraced - the anchor stays
    # the structural pivot. The wick is kept shallow enough not to clear the
    # reversal threshold, because one that did would genuinely end the leg.
    closes = UPTREND + UPTREND_PULLBACK
    lows = [c - WICK for c in closes]
    lows[-1] = closes[-1] - 12

    swing_low, swing_high = _uptrend_leg(_bars_from_closes(closes, lows=lows))

    assert swing_low == pytest.approx(UPTREND_SWING_LOW - WICK)
    assert swing_high == pytest.approx(UPTREND_PEAK + WICK)


def test_leg_stops_at_a_price_regime_break():
    # The SNDKUSDT case: an old high-price regime, a one-session ~35% crash,
    # then a new lower regime. Taking the global max and global min over the
    # window draws a Fib straddling the crash, putting entry far above any
    # price the market has traded since. The leg must start after the crash.
    old_regime = [1400.0, *_ramp(1400, 1650, 99)]
    crash = _ramp(1650, 1000, 8)
    # Rallies to 1270, breaks back under the 1050 swing low - which turns the
    # trend down and makes 1270 the anchor - then bounces enough for that low to
    # confirm as a swing. The bounce matters: the trend turns on a CONFIRMED
    # swing, not on price poking through intraday, so a fixture that ends
    # mid-break would be testing a state the detector deliberately ignores.
    new_regime = [
        *_ramp(1000, 1130, 40), *_ramp(1130, 1050, 20),
        *_ramp(1050, 1270, 40), *_ramp(1270, 1010, 30), *_ramp(1010, 1120, 14),
    ]
    bars = _bars_from_closes(old_regime + crash + new_regime)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high < 1400  # not the pre-crash 1650
    assert swing_low > 1000  # not the crash bottom either


def test_a_pullback_that_holds_structure_is_still_a_tradeable_leg():
    """Deliberate reversal of the old rule, and the reason this was rebuilt.

    The previous guard ended a leg the moment an opposite ZigZag pivot
    confirmed. That is what silently blocked AAPLUSDT: the rally off 302.04
    confirmed a swing low, so the setup vanished - while the leg it had
    selected instead was 90% retraced and should have been the dead one. A
    pullback that has NOT broken structure is a retracement, which is exactly
    what this strategy exists to sell into.
    """
    deep = UPTREND + [UPTREND_PEAK - i * 12 for i in range(1, 8)]
    shallow = UPTREND + UPTREND_PULLBACK

    assert _uptrend_leg(_bars_from_closes(shallow)) is not None
    assert _uptrend_leg(_bars_from_closes(deep)) is not None


def test_no_leg_once_price_is_past_its_own_stop():
    """The TSLAUSDT failure the old guard was really protecting against.

    The code kept quoting entry 303.64 / stop 306.31 off a finished leg while
    price walked to 312, so every alert arrived already past its own stop. That
    is now caught by the retracement test rather than by the pivot rule: beyond
    FIB_STOP the trade cannot be entered, whatever the structure says.
    """
    leg = _uptrend_leg(_bars_from_closes(UPTREND + UPTREND_PULLBACK))
    assert leg is not None
    swing_low, swing_high = leg
    past_stop = swing_high - (swing_high - swing_low) * FIB_STOP

    # Walk price just under that level without breaking the structure low.
    beyond = UPTREND + _ramp(UPTREND_PEAK, past_stop - 2, 6)

    assert _uptrend_leg(_bars_from_closes(beyond)) is None


def test_a_symbol_offers_a_leg_in_only_one_direction_at_a_time():
    # A market is either retracing an up-leg or a down-leg. Before the guard
    # both legs resolved at once off unrelated stale pivots, which is what let
    # a short fire on a symbol that had been rallying for days.
    for closes in (UPTREND + UPTREND_PULLBACK, DOWNTREND + DOWNTREND_BOUNCE):
        bars = _bars_from_closes(closes)

        assert (_uptrend_leg(bars) is None) != (_downtrend_leg(bars) is None)


def test_no_leg_without_an_identifiable_reversal():
    # A perfectly monotonic series never reverses, so there is no pivot marking
    # where the current leg began and no Fib can honestly be drawn.
    bars = _bars_from_closes(_ramp(100, 300, 210))

    assert _uptrend_leg(bars) is None
    assert _downtrend_leg(bars) is None


def test_a_leg_too_small_to_clear_the_fee_is_rejected(monkeypatch):
    """MIN_LEG_PCT, the price of the finer pivot threshold.

    Dropping STRUCTURE_ATR_MULTIPLE to 1.25 doubled the rate of fee-dominated
    legs from 8% to 16% across a 41-day replay. APTUSDT 1H was Dror's live
    example: a 4.2% leg giving a 0.70% stop, where the 0.12% round trip eats
    17% of 1R before the market moves. The Fib gap between entry and stop is
    16.8% of the leg, so a 6% leg is about a 1% stop - the boundary.

    Asserted by lifting the constant rather than shrinking a fixture: what
    matters is that the check is wired and binding, and a synthetic leg small
    enough to trip it would also be too small to form pivots at all.
    """
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    assert _uptrend_leg(bars) is not None, "this leg is normally tradeable"

    monkeypatch.setattr(rsi_fib_reversal, "MIN_LEG_PCT", 5.0)  # demand a 500% leg

    assert _uptrend_leg(bars) is None


def test_the_minimum_leg_applies_to_shorts_too(monkeypatch):
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    assert _downtrend_leg(bars) is not None

    monkeypatch.setattr(rsi_fib_reversal, "MIN_LEG_PCT", 5.0)

    assert _downtrend_leg(bars) is None
