import pandas as pd
import pytest

from notifier.strategies.rsi_fib_reversal import (
    FIB_ENTRY,
    FIB_STOP,
    RsiFibReversal,
    _downtrend_leg,
    _uptrend_leg,
)


def _bars_from_closes(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    series = pd.Series(closes)
    high = pd.Series(highs) if highs is not None else series
    low = pd.Series(lows) if lows is not None else series
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
# pullback that forms a swing low at 170, then the leg being retraced up to
# 300, then the sharp drop that pushes RSI(10) below 30. The swing detector
# needs an actual reversal to anchor on - a straight line has no pivots.
UPTREND_SWING_LOW = 170.0
UPTREND_PEAK = 300.0
UPTREND = [100.0, *_ramp(100, 200, 99), *_ramp(200, UPTREND_SWING_LOW, 15), *_ramp(170, UPTREND_PEAK, 85)]
UPTREND_PULLBACK = [UPTREND_PEAK - i * 4 for i in range(1, 8)]

DOWNTREND_SWING_HIGH = 230.0
DOWNTREND_TROUGH = 100.0
DOWNTREND = [300.0, *_ramp(300, 200, 99), *_ramp(200, DOWNTREND_SWING_HIGH, 15), *_ramp(230, DOWNTREND_TROUGH, 85)]
DOWNTREND_BOUNCE = [DOWNTREND_TROUGH + i * 4 for i in range(1, 8)]


def test_fires_long_on_oversold_rsi_cross_above_200ma():
    signal = _evaluate("BTCUSDT", UPTREND + UPTREND_PULLBACK)

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 1"
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

    # The interim pullback low, not the 100.0 the whole series started from.
    assert swing_low == pytest.approx(UPTREND_SWING_LOW)
    assert swing_high == pytest.approx(UPTREND_PEAK)


def test_downtrend_leg_anchors_on_the_pivot_that_started_the_leg():
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high == pytest.approx(DOWNTREND_SWING_HIGH)
    assert swing_low == pytest.approx(DOWNTREND_TROUGH)


def test_swing_uses_wick_extremes_not_closes():
    # The pivot candle wicks below its close and the peak candle wicks above
    # its close; both wicks are the real extremes and must anchor the Fib.
    closes = UPTREND + UPTREND_PULLBACK
    highs, lows = list(closes), list(closes)
    peak_idx = closes.index(max(closes))
    pivot_idx = closes.index(min(UPTREND[100:115]))
    highs[peak_idx] = closes[peak_idx] + 20
    lows[pivot_idx] = closes[pivot_idx] - 15

    signal = _evaluate("BTCUSDT", closes, highs=highs, lows=lows)

    assert signal is not None
    expected_range = highs[peak_idx] - lows[pivot_idx]
    assert signal.entry_price == pytest.approx(highs[peak_idx] - expected_range * FIB_ENTRY)
    assert signal.stop_loss == pytest.approx(highs[peak_idx] - expected_range * FIB_STOP)


def test_swing_ignores_deeper_low_after_the_peak():
    # A deeper low during the pullback that triggers the signal is part of the
    # retracement, not the start of the leg being retraced.
    closes = UPTREND + UPTREND_PULLBACK
    lows = list(closes)
    lows[-1] = 10.0

    signal = _evaluate("BTCUSDT", closes, lows=lows)

    assert signal is not None
    assert signal.stop_loss > lows[-1]


def test_leg_stops_at_a_price_regime_break():
    # The SNDKUSDT case: an old high-price regime, a one-session ~35% crash,
    # then a new lower regime. Taking the global max and global min over the
    # window draws a Fib straddling the crash, putting entry far above any
    # price the market has traded since. The leg must start after the crash.
    old_regime = [1400.0, *_ramp(1400, 1650, 99)]
    crash = _ramp(1650, 1000, 8)
    new_regime = [*_ramp(1000, 1130, 40), *_ramp(1130, 1050, 20), *_ramp(1050, 1270, 40)]
    bars = _bars_from_closes(old_regime + crash + new_regime)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high < 1400  # not the pre-crash 1650
    assert swing_low > 1000  # not the crash bottom either


def test_no_leg_without_an_identifiable_reversal():
    # A perfectly monotonic series never reverses, so there is no pivot marking
    # where the current leg began and no Fib can honestly be drawn.
    bars = _bars_from_closes(_ramp(100, 300, 210))

    assert _uptrend_leg(bars) is None
    assert _downtrend_leg(bars) is None
