import pandas as pd

from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.indicators import ema


def _bars(closes: list[float], freq: str, last_low=None, last_high=None, last_volume=1.0) -> pd.DataFrame:
    series = pd.Series(closes)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(series), freq=freq),
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


def _trend_1h(closes: list[float]) -> pd.DataFrame:
    return _bars(closes, freq="h")


def uptrend_1h(n=210):
    return _trend_1h([100 + i * 0.8 for i in range(n)])


def downtrend_1h(n=210):
    return _trend_1h([300 - i * 0.8 for i in range(n)])


def test_fires_long_on_near_miss_touch_within_proximity_band():
    bars_1h = uptrend_1h()
    closes_15m = [100 + i * 0.5 for i in range(40)]
    # the level being tested is EMA9 as of the PRIOR candle (this candle's own
    # close doesn't get folded in yet); low stays ABOVE it (a strict
    # wick-through check would miss this) but within the proximity band,
    # and close still holds above it
    ema9_prev = ema(pd.Series(closes_15m[:-1]), 9).iloc[-1]
    bars_15m = _bars(closes_15m, freq="15min", last_low=ema9_prev * 1.00003)

    signal = EmaTrendFollowing().evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m})

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 2"
    assert signal.entry_price == bars_15m["close"].iloc[-1]
    assert signal.stop_loss < signal.entry_price
    assert signal.reward_risk_ratio is None  # uses the scanner-wide default


def test_fires_short_on_near_miss_touch_with_volume():
    bars_1h = downtrend_1h()
    closes_15m = [200 - i * 0.5 for i in range(40)]
    ema9_prev = ema(pd.Series(closes_15m[:-1]), 9).iloc[-1]
    # high stays BELOW the prior-candle ema9 (strict cross-through would miss
    # it) but within band
    bars_15m = _bars(closes_15m, freq="15min", last_high=ema9_prev * 0.99997, last_volume=5.0)

    signal = EmaTrendFollowing().evaluate("ETHUSDT", {"1H": bars_1h, "15m": bars_15m})

    assert signal is not None
    assert signal.direction == "short"
    assert signal.stop_loss > signal.entry_price


def test_no_short_signal_without_volume_confirmation():
    bars_1h = downtrend_1h()
    closes_15m = [200 - i * 0.5 for i in range(40)]
    ema9_prev = ema(pd.Series(closes_15m[:-1]), 9).iloc[-1]
    bars_15m = _bars(closes_15m, freq="15min", last_high=ema9_prev * 0.99997, last_volume=1.0)

    assert EmaTrendFollowing().evaluate("ETHUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_no_signal_when_1h_trend_missing():
    # flat 1H (no clear trend) even though 15m shows a valid touch
    bars_1h = _trend_1h([100.0] * 210)
    closes_15m = [100 + i * 0.5 for i in range(40)]
    bars_15m = _bars(closes_15m, freq="15min", last_low=118.0)

    assert EmaTrendFollowing().evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_no_signal_without_enough_1h_history():
    bars_1h = uptrend_1h(n=50)
    bars_15m = _bars([100 + i * 0.5 for i in range(40)], freq="15min", last_low=118.0)
    assert EmaTrendFollowing().evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_no_signal_outside_proximity_band():
    bars_1h = uptrend_1h()
    closes_15m = [100 + i * 0.5 for i in range(40)]
    bars_15m = _bars(closes_15m, freq="15min")  # no wick, no near-miss either

    assert EmaTrendFollowing().evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_wide_breakout_candle_does_not_count_as_a_touch():
    # 39 flat closes at 100 (ema9_prev == 100 exactly) then one huge breakout
    # candle closing at 130. That close alone would drag a same-candle EMA9
    # up to 106, making a low of 106 look like a "touch" under the old
    # (buggy) same-candle EMA9 check. Measured against the level that
    # actually existed before this candle traded (ema9_prev == 100), a low of
    # 106 is nowhere near it - this is a breakout blasting through the level,
    # not a pullback that tested it and reversed.
    bars_1h = uptrend_1h()
    closes_15m = [100.0] * 39 + [130.0]
    bars_15m = _bars(closes_15m, freq="15min", last_low=106.0)

    assert EmaTrendFollowing().evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_1h_trend_is_cached_per_hour():
    strategy = EmaTrendFollowing()
    bars_1h = uptrend_1h()
    closes_15m = [100 + i * 0.5 for i in range(40)]
    bars_15m = _bars(closes_15m, freq="15min", last_low=118.0)

    strategy.evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m})
    cached_ts, cached_trend = strategy._trend_cache["BTCUSDT"]
    assert cached_trend == "up"
    assert cached_ts == str(bars_1h["ts"].iloc[-1])

    # a second call with the SAME 1H candle re-uses the cached reading rather
    # than recomputing (verified by corrupting the cache and confirming it wins)
    strategy._trend_cache["BTCUSDT"] = (cached_ts, "down")
    signal = strategy.evaluate("BTCUSDT", {"1H": bars_1h, "15m": bars_15m})
    assert signal is None  # "down" (from the poisoned cache) doesn't match the long setup
