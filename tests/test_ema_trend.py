import pandas as pd

from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.indicators import ema, sma


def _decline(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * i for i in range(bars)]


def _rally(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


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


def test_no_signal_when_ema50_is_below_the_200ma():
    # The live XAUTUSDT alert: a long decline then a sharp recent rally lifts
    # EMA9 and EMA20 back over the SMA200 while the slower EMA50 is still
    # underneath it. Checking only 9 > 20 > 200 skips straight past the 50 and
    # calls that an uptrend; the user's method does not.
    bars_1h = _trend_1h([*_decline(300, 100, 170), *_rally(100, 250, 40)])
    closes = bars_1h["close"]
    assert ema(closes, 9).iloc[-1] > ema(closes, 20).iloc[-1] > sma(closes, 200).iloc[-1]  # old gate said "up"
    assert ema(closes, 50).iloc[-1] < sma(closes, 200).iloc[-1]  # but the 50 is below the 200

    closes_15m = [100 + i * 0.5 for i in range(40)]
    ema9_prev = ema(pd.Series(closes_15m[:-1]), 9).iloc[-1]
    bars_15m = _bars(closes_15m, freq="15min", last_low=ema9_prev * 1.00003)

    assert EmaTrendFollowing().evaluate("XAUTUSDT", {"1H": bars_1h, "15m": bars_15m}) is None


def test_1h_trend_is_recomputed_rather_than_cached_for_the_hour():
    # An entry firing at :45 must not act on the stack as it stood at :00. The
    # strategy reads the forming hourly candle, so a stack that has since
    # broken has to be seen on this scan, not an hour later.
    strategy = EmaTrendFollowing()
    assert strategy.wants_forming_bar is True

    closes_15m = [100 + i * 0.5 for i in range(40)]
    ema9_prev = ema(pd.Series(closes_15m[:-1]), 9).iloc[-1]
    bars_15m = _bars(closes_15m, freq="15min", last_low=ema9_prev * 1.00003)

    assert strategy.evaluate("BTCUSDT", {"1H": uptrend_1h(), "15m": bars_15m}) is not None

    # Same symbol, same 15m trigger, but the hour in progress has broken the
    # stack: the answer must change immediately.
    broken = uptrend_1h()
    broken.loc[broken.index[-1], "close"] = 100.0
    assert strategy.evaluate("BTCUSDT", {"1H": broken, "15m": bars_15m}) is None


def test_touch_band_scales_with_volatility_not_price():
    # The same chart at two price scales must behave the same way. As a
    # percentage of price it did not: 0.005% came to 32 ticks on BTCUSDT but
    # under a tenth of a tick on COTIUSDT, so an identical setup fired on one
    # and was mathematically unreachable on the other.
    def fires_at(scale: float):
        bars_1h = _trend_1h([(100 + i * 0.8) * scale for i in range(210)])
        closes = [(100 + i * 0.5) * scale for i in range(40)]
        ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
        # a fifth of the way into the band, in ATR terms, at either scale
        bars_15m = _bars(closes, freq="15min", last_low=ema9_prev + 0.5 * scale * 0.01)
        return EmaTrendFollowing().evaluate("X", {"1H": bars_1h, "15m": bars_15m}) is not None

    assert fires_at(1.0)
    assert fires_at(0.00001)
