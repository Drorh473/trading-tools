import pandas as pd

from notifier.strategies.ema_trend import EmaTrendFollowing


def _bars(closes: list[float], last_low=None, last_high=None, last_volume=1.0) -> pd.DataFrame:
    series = pd.Series(closes)
    df = pd.DataFrame(
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
    if last_low is not None:
        df.loc[df.index[-1], "low"] = last_low
    if last_high is not None:
        df.loc[df.index[-1], "high"] = last_high
    df.loc[df.index[-1], "base_vol"] = last_volume
    return df


def test_fires_long_on_ema9_support_in_uptrend():
    closes = [100 + i * 0.8 for i in range(210)]
    # last bar wicks down through EMA9 (~264) but closes back above it
    bars = _bars(closes, last_low=263.0)

    signal = EmaTrendFollowing().evaluate("BTCUSDT", bars)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "ema_trend_following"
    assert signal.entry_price == bars["close"].iloc[-1]
    assert signal.stop_loss < signal.entry_price
    assert signal.reward_risk_ratio is None  # uses the scanner-wide default


def test_fires_short_on_ema9_rejection_in_downtrend_with_volume():
    closes = [300 - i * 0.8 for i in range(210)]
    # last bar wicks up through EMA9 (~136) but closes back below it, on high volume
    bars = _bars(closes, last_high=137.0, last_volume=5.0)

    signal = EmaTrendFollowing().evaluate("ETHUSDT", bars)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.stop_loss > signal.entry_price


def test_no_short_signal_without_volume_confirmation():
    closes = [300 - i * 0.8 for i in range(210)]
    # same price setup as the short test, but ordinary (not elevated) volume
    bars = _bars(closes, last_high=137.0, last_volume=1.0)

    assert EmaTrendFollowing().evaluate("ETHUSDT", bars) is None


def test_no_signal_without_enough_history():
    bars = _bars([100.0] * 50)
    assert EmaTrendFollowing().evaluate("BTCUSDT", bars) is None


def test_no_signal_when_price_doesnt_touch_ema9():
    closes = [100 + i * 0.8 for i in range(210)]
    bars = _bars(closes)  # no wick down to EMA9

    assert EmaTrendFollowing().evaluate("BTCUSDT", bars) is None
