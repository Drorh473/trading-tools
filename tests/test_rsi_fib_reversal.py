import pandas as pd

from notifier.strategies.rsi_fib_reversal import RsiFibReversal


def _bars_from_closes(closes: list[float]) -> pd.DataFrame:
    series = pd.Series(closes)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(series), freq="h"),
            "open": series,
            "high": series,
            "low": series,
            "close": series,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _evaluate(symbol, closes):
    return RsiFibReversal().evaluate(symbol, {"1H": _bars_from_closes(closes)})


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
