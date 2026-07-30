import pandas as pd
import pytest

from notifier.strategies.rsi_fib_reversal import FIB_ENTRY, FIB_STOP, RsiFibReversal


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


def test_fires_long_on_oversold_rsi_cross_above_200ma():
    # 200-bar uptrend (well above its own 200-MA by the end), then a sharp
    # pullback trimmed to land exactly on the bar where RSI(10) crosses below 30.
    uptrend = [100 + i * (200 / 199) for i in range(200)]
    pullback = [uptrend[-1] - i * 3 for i in range(1, 7)]
    closes = uptrend + pullback

    signal = _evaluate("BTCUSDT", closes)

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 1"
    assert signal.reward_risk_ratio == 2.0
    assert signal.stop_loss < signal.entry_price < closes[-1]


def test_fires_short_on_overbought_rsi_cross_below_200ma():
    downtrend = [300 - i * (200 / 199) for i in range(200)]
    bounce = [downtrend[-1] + i * 3 for i in range(1, 7)]
    closes = downtrend + bounce

    signal = _evaluate("ETHUSDT", closes)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.reward_risk_ratio == 2.0
    assert closes[-1] < signal.entry_price < signal.stop_loss


def test_no_signal_without_enough_history():
    assert _evaluate("BTCUSDT", [100.0] * 50) is None


def test_no_signal_when_rsi_not_crossing():
    # Flat price series: RSI stays near 50, never crosses 30/70.
    assert _evaluate("BTCUSDT", [100.0] * 210) is None


def test_swing_uses_wick_extremes_not_closes():
    # Same uptrend+pullback shape as the basic long test (so RSI/trend-filter
    # still fire), but the peak candle wicks well above its close and an
    # early candle (before the peak) wicks well below its close. The swing
    # anchor should be the peak's high and the lowest low BEFORE that peak -
    # i.e. the real wick extremes of the actual trend leg, not the closes.
    uptrend = [100 + i * (200 / 199) for i in range(200)]
    pullback = [uptrend[-1] - i * 3 for i in range(1, 7)]
    closes = uptrend + pullback

    highs = list(closes)
    lows = list(closes)
    highs[199] = uptrend[-1] + 20  # peak candle wicks well above its close
    lows[10] = uptrend[10] - 15  # early candle (before the peak) wicks well below its close

    signal = _evaluate("BTCUSDT", closes, highs=highs, lows=lows)

    assert signal is not None
    expected_swing_high = highs[199]
    expected_swing_low = lows[10]
    expected_range = expected_swing_high - expected_swing_low
    assert signal.entry_price == pytest.approx(expected_swing_high - expected_range * FIB_ENTRY)
    assert signal.stop_loss == pytest.approx(expected_swing_high - expected_range * FIB_STOP)


def test_swing_finds_low_before_peak_not_after():
    # A second, deeper low AFTER the peak (during the pullback that triggers
    # the signal) must not be used as the swing anchor - only lows before
    # the peak mark where that trend leg started.
    uptrend = [100 + i * (200 / 199) for i in range(200)]
    pullback = [uptrend[-1] - i * 3 for i in range(1, 7)]
    closes = uptrend + pullback

    lows = list(closes)
    lows[10] = uptrend[10] - 15  # genuine pre-peak low: the real anchor
    lows[-1] = 10.0  # an absurdly deep wick AFTER the peak - must be ignored

    signal = _evaluate("BTCUSDT", closes, lows=lows)

    assert signal is not None
    assert signal.stop_loss > lows[-1]  # the post-peak wick did not become the anchor
