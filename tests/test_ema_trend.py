import pytest
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
    # Stated, not inherited. Leaving this None took the scanner's 3.0 default,
    # which equals REMAINDER_TARGET_RATIO and put both exit tiers on one price.
    assert signal.reward_risk_ratio == 2.0
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


# ---- break-of-structure gate ----
#
# Dror on a live MUUUSDT short: "currently we didnt change yet to downtrend,
# there is rising highs before the signal so it dont fit the condition". The
# confirmed 15m swings ran 26.760 -> 26.864 -> 26.880 on highs and 24.792 ->
# 25.606 -> 26.423 on lows - higher highs AND higher lows - while every
# EMA-stack condition here said short. Nothing in this module had ever asked
# where the market actually turned.


def test_a_signal_against_the_base_timeframes_structure_is_suppressed(monkeypatch):
    """The MUUUSDT case. Every EMA condition fires short; structure says up."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: "up")

    assert EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()}) is None


def test_a_signal_that_agrees_with_structure_still_fires(monkeypatch):
    """The control: the gate must only remove counter-trend signals, not
    quietly mute the strategy."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: "down")

    signal = EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()})

    assert signal is not None and signal.direction == "short"


def test_unknown_structure_does_not_veto(monkeypatch):
    """Too few confirmed pivots to say. Absence of evidence that a trade is
    counter-trend is not evidence that it is - and muting on missing data is
    the same mistake as muting the watchlist on one failed API read."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: None)

    assert EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()}) is not None


def test_structure_reads_rising_highs_and_lows_as_up():
    """_structure_trend itself, on the shape Dror described - so the gate is
    fed a real reading rather than only ever a patched one."""
    closes = (
        [100.0] * 30
        + _rally(100, 130, 20) + _decline(130, 112, 12)
        + _rally(112, 150, 20) + _decline(150, 128, 12)
        + _rally(128, 165, 20) + _decline(165, 140, 12)
    )

    assert ema_trend._structure_trend(_bars(closes)) == "up"


# ---- fee domination ----
#
# The stop sits at EMA20, and on a fast base timeframe EMA9 and EMA20 are
# barely apart, so the trade risks only that gap. Measured watchlist-wide the
# 1H/15m instance's median stop is 0.145% of price against a 0.08% round-trip
# fee - 0.55R at the median. Paying more in fees than is being risked is
# negative expectancy before the market moves at all.
#
# The round trip is 0.08%, not 0.12%: this strategy's entry is entirely a
# resting limit (market_fraction 0.0) and so fills as a maker at 0.02%, and the
# exit a risk gate should price is the taker stop at 0.06%. See
# ROUND_TRIP_FEE_PCT.


def test_fee_fraction_is_the_round_trip_over_the_stop_distance():
    # a 1% stop against a 0.08% round trip = 0.08R
    assert ema_trend._fee_fraction_of_risk(100.0, 99.0) == pytest.approx(0.08, rel=1e-3)
    # a 0.08% stop costs a full 1R in fees
    assert ema_trend._fee_fraction_of_risk(100.0, 99.92) == pytest.approx(1.0, rel=1e-3)


def test_the_entry_is_a_maker_fill_so_the_fee_is_not_taker_both_ways():
    """Guards the constant against being 'corrected' back to 0.0012. The gate
    is only honest if it prices the legs this strategy actually pays: a maker
    limit in, a taker stop out."""
    assert ema_trend.ROUND_TRIP_FEE_PCT == pytest.approx(0.0008)
    # The stop this admits: fee/risk = 0.25 at a 0.32% stop, not 0.48%.
    assert ema_trend._fee_fraction_of_risk(100.0, 99.68) == pytest.approx(
        ema_trend.MAX_FEE_FRACTION_OF_RISK, rel=1e-2
    )


def test_a_zero_distance_stop_is_infinitely_fee_dominated():
    """Guards the division. A stop on the entry is not a free trade, it is one
    whose fees can never be recovered."""
    assert ema_trend._fee_fraction_of_risk(100.0, 100.0) == float("inf")


def test_a_fee_dominated_setup_is_declined(monkeypatch):
    """The real MUUUSDT 1H/15m shape: every condition passes, the stop is
    0.07% away, fees are 1.14R at the corrected 0.08% round trip."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: None)
    # Force the stop to sit a hair from the entry, leaving everything else intact.
    real = ema_trend._touch_and_hold

    def hair_thin(bars, direction):
        result = real(bars, direction)
        if result is None:
            return None
        entry, _ = result
        return entry, entry * (0.9993 if direction == "up" else 1.0007)  # 0.07%

    monkeypatch.setattr(ema_trend, "_touch_and_hold", hair_thin)

    assert EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()}) is None


def test_a_setup_with_room_still_fires(monkeypatch):
    """The control: the gate must only remove uneconomic trades. This fixture
    sits at 0.075R, well inside the cap."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: None)

    signal = EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()})

    assert signal is not None
    assert ema_trend._fee_fraction_of_risk(signal.entry_price, signal.stop_loss) < ema_trend.MAX_FEE_FRACTION_OF_RISK


# ---- the mirror case: a stop too WIDE ----
#
# EMA20 lags while EMA9 tracks price, so a violent move drags them apart.
# LABUSDT 4H fell 0.92 -> 0.336 in ~20 bars, leaving EMA20 at 0.794 against
# EMA9 at 0.358 - a stop 127% ABOVE the entry.


def test_stop_fraction_is_measured_against_price_not_atr():
    assert ema_trend._stop_fraction_of_price(100.0, 80.0) == pytest.approx(0.20)
    assert ema_trend._stop_fraction_of_price(0.358, 0.794) == pytest.approx(1.218, rel=1e-3)


def test_a_crash_widened_stop_is_declined(monkeypatch):
    """The real LABUSDT geometry: entry 0.358, stop 0.794, 121.8% away."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: None)
    real = ema_trend._touch_and_hold

    def crash_widened(bars, direction):
        result = real(bars, direction)
        if result is None:
            return None
        entry, _ = result
        # stop 121.8% away, the LAB ratio, on whichever side the trade needs
        return entry, entry * (1 + 1.218 if direction == "down" else 1 - 1.218)

    monkeypatch.setattr(ema_trend, "_touch_and_hold", crash_widened)

    assert EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()}) is None


def test_a_normal_width_stop_still_fires(monkeypatch):
    """The control. The real fixture sits at 1.6% of price - inside the cap by
    more than a factor of ten, as every measured normal signal was."""
    monkeypatch.setattr(ema_trend, "_structure_trend", lambda bars: None)

    signal = EmaTrendFollowing("1H").evaluate("ETHUSDT", {"1H": downtrend_with_touch()})

    assert signal is not None
    frac = ema_trend._stop_fraction_of_price(signal.entry_price, signal.stop_loss)
    assert frac < ema_trend.MAX_STOP_FRACTION_OF_PRICE


def test_the_wide_stop_gate_is_not_expressible_in_atr(monkeypatch):
    """Why this check exists at all, pinned as a property rather than prose.

    Both LABUSDT stops sat at 1.4 and 1.6 ATR - inside the 1.0-1.9 range every
    NORMAL signal occupied - because its ATR had grown to ~88% of its own
    price. Any ATR-relative bound loose enough to admit normal trades also
    admits the crash ones, so percent-of-price is the only instrument that
    separates them.
    """
    normal_atr_ratios = [1.9, 1.8, 1.0, 1.1, 1.2, 1.3]
    lab_atr_ratios = [1.4, 1.6]

    assert min(lab_atr_ratios) > min(normal_atr_ratios)
    assert max(lab_atr_ratios) < max(normal_atr_ratios)
