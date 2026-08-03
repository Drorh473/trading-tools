import pandas as pd

from notifier.strategies import ema_trend
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.indicators import ema, sma


def _bars(closes: list[float], freq: str = "h", last_low=None, last_high=None, last_volume=1.0) -> pd.DataFrame:
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


def _decline(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * i for i in range(bars)]


def _rally(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


def uptrend_with_touch(freq: str = "h", touch_offset: float = 1.00003) -> pd.DataFrame:
    """A long enough rise for the 4-MA stack (170 bars), then a clean 10-bar
    hold above EMA9 before a near-miss touch on the final candle - everything
    a single timeframe now needs to fire entirely on its own."""
    warmup = [100 + i * 0.8 for i in range(170)]
    tail = [warmup[-1] + i * 0.5 for i in range(1, 41)]
    closes = warmup + tail
    ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, freq=freq, last_low=ema9_prev * touch_offset)


def downtrend_with_touch(freq: str = "h", touch_offset: float = 0.99997, volume: float = 5.0) -> pd.DataFrame:
    warmup = [300 - i * 0.8 for i in range(170)]
    tail = [warmup[-1] - i * 0.5 for i in range(1, 41)]
    closes = warmup + tail
    ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    return _bars(closes, freq=freq, last_high=ema9_prev * touch_offset, last_volume=volume)


def uptrend_only(freq: str = "h", n: int = 210) -> pd.DataFrame:
    """Steadily rising and far from its own EMA9 - satisfies the stack but
    not the touch: trending, not pulled back."""
    return _bars([100 + i * 0.8 for i in range(n)], freq=freq)


def downtrend_only(freq: str = "h", n: int = 210) -> pd.DataFrame:
    return _bars([300 - i * 0.8 for i in range(n)], freq=freq)


def test_fires_long_on_near_miss_touch_within_proximity_band():
    base = uptrend_with_touch()
    ema9_prev = ema(base["close"].iloc[:-1], 9).iloc[-1]

    signal = EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base})

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 2 1H"
    # Entry is the EMA9 level itself, not the candle's close - a limit resting
    # there is the whole position, with no market fraction to fall back on.
    assert signal.entry_price == ema9_prev
    assert signal.limit_entry == ema9_prev
    assert signal.limit_note == "EMA9"
    assert signal.market_fraction == 0.0
    assert signal.stop_loss < signal.entry_price
    assert signal.reward_risk_ratio is None  # uses the scanner-wide default
    assert signal.risk_pct_override is None  # base tier: no reference timeframe at all
    assert signal.analysis_timeframes == ("1H",)


def test_fires_short_on_near_miss_touch_with_volume():
    base = downtrend_with_touch()
    ema9_prev = ema(base["close"].iloc[:-1], 9).iloc[-1]

    signal = EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": base})

    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == ema9_prev
    assert signal.market_fraction == 0.0
    assert signal.stop_loss > signal.entry_price


def test_no_short_signal_without_volume_confirmation():
    base = downtrend_with_touch(volume=1.0)  # not above its own rolling average
    assert EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": base}) is None


def test_no_signal_when_trend_missing():
    flat = _bars([100.0] * 210)
    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": flat}) is None


def test_no_signal_without_enough_history():
    short = _bars([100 + i * 0.8 for i in range(50)])
    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": short}) is None


def test_no_signal_outside_proximity_band():
    warmup = [100 + i * 0.8 for i in range(170)]
    tail = [warmup[-1] + i * 0.5 for i in range(1, 41)]
    no_touch = _bars(warmup + tail)  # no wick, no near-miss either
    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": no_touch}) is None


def test_wide_breakout_candle_does_not_count_as_a_touch():
    # A huge breakout candle's own close would drag a same-candle EMA9 toward
    # it, making a low nowhere near the level LOOK like a touch under a naive
    # same-candle check. Measured against the level as it stood before this
    # candle traded, it is not.
    warmup = [100 + i * 0.8 for i in range(170)]
    tail = [warmup[-1]] * 39 + [warmup[-1] + 30]
    base = _bars(warmup + tail, last_low=warmup[-1] + 6)

    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base}) is None


def test_no_signal_when_ema50_is_below_the_200ma():
    # The live XAUTUSDT alert: a long decline then a sharp recent rally lifts
    # EMA9 and EMA20 back over the SMA200 while the slower EMA50 is still
    # underneath it. Checking only 9 > 20 > 200 skips straight past the 50 and
    # calls that an uptrend; the user's method does not.
    closes = [*_decline(300, 100, 170), *_rally(100, 250, 40)]
    base = _bars(closes)
    assert ema(base["close"], 9).iloc[-1] > ema(base["close"], 20).iloc[-1] > sma(base["close"], 200).iloc[-1]
    assert ema(base["close"], 50).iloc[-1] < sma(base["close"], 200).iloc[-1]  # the 50 is below the 200

    ema9_prev = ema(base["close"].iloc[:-1], 9).iloc[-1]
    base.loc[base.index[-1], "low"] = ema9_prev * 1.00003

    assert EmaTrendFollowing("1H").evaluate("XAUTUSDT", {"1H": base}) is None


def test_touch_band_scales_with_volatility_not_price():
    # The same chart at two price scales must behave the same way. As a
    # percentage of price it did not: 0.005% came to 32 ticks on BTCUSDT but
    # under a tenth of a tick on COTIUSDT, so an identical setup fired on one
    # and was mathematically unreachable on the other.
    def fires_at(scale: float) -> bool:
        warmup = [(100 + i * 0.8) * scale for i in range(170)]
        tail = [warmup[-1] + i * 0.5 * scale for i in range(1, 41)]
        closes = warmup + tail
        ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
        # a fifth of the way into the band, in ATR terms, at either scale
        base = _bars(closes, last_low=ema9_prev + 0.5 * scale * 0.01)
        return EmaTrendFollowing("1H").evaluate("X", {"1H": base}) is not None

    assert fires_at(1.0)
    assert fires_at(0.00001)


def test_no_signal_when_price_was_cutting_through_ema9():
    # "The 9ema was breached before, so it doesn't hold anymore." A touch
    # only means support held if price was not already crossing the level -
    # without this the strategy fired on symbols chopping around EMA9, where
    # the "test" was one crossing among several.
    warmup = [100 + i * 0.8 for i in range(170)]

    def fires(chopped: bool) -> bool:
        steady = [warmup[-1] + i * 0.5 for i in range(1, 41)]
        tail = list(steady)
        if chopped:
            tail[-4] = tail[-4] - 4  # one candle closes back under the EMA9
        closes = warmup + tail
        ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
        base = _bars(closes, last_low=ema9_prev * 1.00003)
        return EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base}) is not None

    assert fires(chopped=False)
    assert not fires(chopped=True)


def test_no_signal_after_a_shock_candle_drags_ema9_toward_price():
    # KAITOUSDT/LABUSDT: a single candle far bigger than its own prior ATR, a
    # few bars before the touch, yanked EMA9 toward price by itself. The
    # 10-bar hold window was technically clean (only the wick/range changes
    # here, not the close), but the level in it was chased, not tested.
    base = uptrend_with_touch()
    base.loc[base.index[-5], "high"] += 20.0
    base.loc[base.index[-5], "low"] -= 20.0

    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base}) is None


def test_no_signal_when_ema9_was_choppy_before_the_hold_window():
    # CLUSDT: the 10 candles right before the touch were clean, but EMA9 had
    # been crossed repeatedly in the hours just before that window - the
    # "hold" only looked clean because the window was too short to see the
    # chop right outside it.
    warmup = [100 + i * 0.8 for i in range(145)]
    chop = []
    level = warmup[-1]
    for i in range(25):
        level += 2.0 if i % 2 == 0 else -2.3
        chop.append(level)
    ramp = [chop[-1] + i * 0.5 for i in range(1, 16)]
    closes = warmup + chop + ramp
    ema9_prev = ema(pd.Series(closes[:-1]), 9).iloc[-1]
    base = _bars(closes, last_low=ema9_prev * 1.00003)

    assert EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base}) is None


def test_a_long_never_gets_a_stop_above_its_entry():
    base = uptrend_with_touch()
    signal = EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base})

    assert signal is not None
    assert signal.stop_loss < signal.entry_price


# ---- reference-timeframe tiers ----


def test_standalone_instance_has_no_reference_and_never_gets_a_bonus():
    base = uptrend_with_touch()
    strategy = EmaTrendFollowing("1D")

    assert strategy.tag == "Strategy 2 1D"
    assert strategy.timeframes == ["1D"]

    signal = strategy.evaluate("BTCUSDT", {"1D": base})

    assert signal is not None
    assert signal.risk_pct_override is None
    assert signal.analysis_timeframes == ("1D",)
    assert signal.extra_notes == ()


def test_base_tier_when_reference_does_not_support():
    base = uptrend_with_touch()
    reference = downtrend_only()  # opposite direction: no support at all

    strategy = EmaTrendFollowing("1H", "4H")
    assert strategy.tag == "Strategy 2 4H/1H"

    signal = strategy.evaluate("BTCUSDT", {"1H": base, "4H": reference})

    assert signal is not None
    assert signal.risk_pct_override is None
    assert signal.analysis_timeframes == ("1H",)
    assert signal.extra_notes == ()


def test_trend_support_tier_when_reference_agrees_but_is_not_touching():
    base = uptrend_with_touch()
    reference = uptrend_only()  # same direction, but trending far from its own EMA9

    signal = EmaTrendFollowing("1H", "4H").evaluate("BTCUSDT", {"1H": base, "4H": reference})

    assert signal is not None
    assert signal.risk_pct_override == ema_trend.TREND_SUPPORT_RISK_PCT
    # Prose only at this tier - the reference isn't a second confirmed
    # timeframe, just a supportive trend read.
    assert signal.analysis_timeframes == ("1H",)
    assert len(signal.extra_notes) == 1
    assert "4H trend supports" in signal.extra_notes[0]


def test_both_touching_tier_when_reference_independently_qualifies():
    base = uptrend_with_touch()
    reference = uptrend_with_touch(freq="4h")  # its own full condition also passes

    signal = EmaTrendFollowing("1H", "4H").evaluate("BTCUSDT", {"1H": base, "4H": reference})

    assert signal is not None
    assert signal.risk_pct_override == ema_trend.BOTH_TOUCHING_RISK_PCT
    # Now the reference earns a place in the formal analysis-timeframe tag.
    assert signal.analysis_timeframes == ("1H", "4H")
    assert signal.extra_notes == ()
    # The stop stays anchored to the BASE timeframe's own EMA20 regardless of
    # tier - a bigger picture agreeing is not a reason to widen the stop onto
    # a timeframe the trade isn't actually being read from.
    base_only_signal = EmaTrendFollowing("1H").evaluate("BTCUSDT", {"1H": base})
    assert signal.stop_loss == base_only_signal.stop_loss


def test_missing_reference_bars_falls_back_to_base_tier():
    base = uptrend_with_touch()
    signal = EmaTrendFollowing("1H", "4H").evaluate("BTCUSDT", {"1H": base})  # "4H" simply not fetched this scan

    assert signal is not None
    assert signal.risk_pct_override is None
    assert signal.analysis_timeframes == ("1H",)
