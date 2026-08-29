"""daily_regime_read: Dror's own pre-trade BTC check as code - trend
direction (trend_structure) PLUS whether price sits too close to a
confirmed level ahead of it to trust that direction right now
(nearest_level_beyond). Built for the standalone question "is this read
statistically persistent", independent of any specific strategy.

daily_regime_from_bars: the SAME read, but taking raw daily bars and doing
the growing-window search (structure_context) and threshold reconstruction
itself - the one implementation both backtest/btc_daily_regime_persistence.py
and any live strategy gate share, rather than two copies free to drift apart.
"""
import pandas as pd

from notifier.strategies.regime import daily_regime_from_bars, daily_regime_read


def _bars(closes, highs=None, lows=None):
    s = pd.Series(closes)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq="D"),
            "open": s,
            "high": pd.Series(highs) if highs is not None else s + 2.0,
            "low": pd.Series(lows) if lows is not None else s - 2.0,
            "close": s,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _ramp(a, b, n):
    step = (b - a) / n
    return [a + step * (i + 1) for i in range(n)]


def _flat(bars, value=8.0):
    return pd.Series([value] * len(bars))


def _confirmed_downtrend():
    """A genuine, OBSERVED change of character (choch_count == 1), not the
    bootstrap guess: bootstrap up to 260, then a real break back down through
    the 170 swing low, confirmed by a bounce - verbatim from test_structure.py's
    own `_reversal()`. Extended further down past the confirmed low with no
    further pullback, so nothing is confirmed below the fresh low yet."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    return reversal + _ramp(200, 80, 20)


def test_a_confirmed_downtrend_at_a_fresh_low_reads_down():
    """No confirmed level sits ahead of price when it is making new lows -
    there is nothing to be 'too close to', so the read is just the trend."""
    bars = _bars(_confirmed_downtrend())
    assert daily_regime_read(bars, _flat(bars)) == "down"


def test_a_bootstrap_only_trend_with_no_observed_turn_still_reads_none():
    """trend_structure can hand back a real direction string ('up') from the
    bootstrap guess alone, with choch_count == 0 - it inferred the trend from
    the two oldest pivots, not from a turn the market actually made. Distinct
    from the plain-ramp case below (trend is None there too) - this fixture
    has a REAL trend string with zero observed turns, so it is the one
    fixture that actually exercises the choch_count check on its own."""
    closes = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10) + _ramp(235, 400, 20)
    bars = _bars(closes)
    assert daily_regime_read(bars, _flat(bars)) is None


def test_a_plain_ramp_with_no_observed_turn_reads_none():
    """A monotonic ramp reverses nowhere, so zigzag_pivots finds no pivots
    and trend_structure has nothing but the bootstrap guess (choch_count==0)
    - the same 'no evidence, no reading' rule this module inherits, not
    something new."""
    bars = _bars(_ramp(100, 200, 60))
    assert daily_regime_read(bars, _flat(bars)) is None


def test_a_confirmed_trend_far_from_the_next_level_still_reads():
    """A confirmed level DOES exist ahead of price (48, a deeper swing low
    formed after the original reversal) - the gate is about DISTANCE, not
    merely whether a level exists. 52 away against an 8.0 threshold is not
    close, so the direction still reads."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    closes = reversal + _ramp(200, 50, 30) + _ramp(50, 120, 15) + _ramp(120, 100, 10)
    bars = _bars(closes)
    assert daily_regime_read(bars, _flat(bars)) == "down"


def test_a_confirmed_trend_too_close_to_the_next_level_reads_none():
    """The trend is confirmed 'down' - same reversal as above, but stopped
    just short of the 148 low instead of running past it, so price (155)
    sits within one threshold (8.0) of a level that could end the trend
    right here. This is Dror's own 'C': direction alone is not enough if
    price is sitting on top of a level that could reverse it."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    bars = _bars(reversal + _ramp(200, 155, 8))
    assert daily_regime_read(bars, _flat(bars)) is None


# --------------------------------------------------------------------------
# daily_regime_from_bars - the growing-window wrapper live code and the
# standalone measurement share.
# --------------------------------------------------------------------------

def test_from_bars_matches_a_manual_structure_context_call():
    """Given a confirmed downtrend with plenty of history, the wrapper's
    answer must match calling structure_context + daily_regime_read by hand
    - it is a convenience wrapper, not a different rule."""
    from notifier.strategies.structure import structure_context

    bars = _bars(_confirmed_downtrend() + _ramp(80, 90, 250))  # pad past MIN_LOOKBACK
    window, _structure = structure_context(bars, atr_multiple=1.25, atr_period=14)
    from notifier.strategies.indicators import atr as _atr
    thresholds = (_atr(bars, 14) * 1.25).iloc[-len(window):].reset_index(drop=True)
    expected = daily_regime_read(window, thresholds)
    assert daily_regime_from_bars(bars) == expected


def test_from_bars_too_short_for_the_growing_window_reads_none():
    """Fewer bars than structure_context's own min_lookback (200) - nothing
    to search, matching structure_context's own behaviour on short history."""
    bars = _bars(_ramp(100, 200, 40))
    assert daily_regime_from_bars(bars) is None


# --------------------------------------------------------------------------
# proximity_threshold / proximity_atr_multiple - decoupling "how far must
# price move to confirm a pivot is real" (thresholds, used for zigzag/CHoCH
# detection) from "how close counts as too-close-to-trust-the-trend" (the
# level-proximity check). These were accidentally the SAME number
# (STRUCTURE_ATR_MULTIPLE, borrowed from an unrelated job) until Dror asked
# whether that was ever actually tested for the proximity role specifically -
# it wasn't. Default None preserves exactly today's behaviour.
# --------------------------------------------------------------------------

def test_default_proximity_matches_today_exactly():
    """proximity_threshold=None (the default) must be byte-identical to the
    pre-decoupling behaviour: reuse thresholds.iloc[-1] as the distance cutoff."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    bars = _bars(reversal + _ramp(200, 155, 8))  # distance 7 vs default threshold 8.0
    assert daily_regime_read(bars, _flat(bars)) is None
    assert daily_regime_read(bars, _flat(bars), proximity_threshold=None) is None


def test_explicit_proximity_threshold_overrides_the_default():
    """Same fixture as above (distance 7, default threshold 8.0 - too close).
    An EXPLICIT, smaller proximity_threshold=5.0 means 7 >= 5 - no longer
    too close, so the trend reads through instead of returning None."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    bars = _bars(reversal + _ramp(200, 155, 8))
    assert daily_regime_read(bars, _flat(bars), proximity_threshold=5.0) == "down"


def test_a_wider_explicit_proximity_threshold_can_ALSO_gate_more():
    """The reverse direction: a case that reads THROUGH by default (distance
    52 against threshold 8.0, from the far-from-level fixture) gets gated
    when given an explicit proximity_threshold wide enough to reach it."""
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    reversal = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    closes = reversal + _ramp(200, 50, 30) + _ramp(50, 120, 15) + _ramp(120, 100, 10)
    bars = _bars(closes)
    assert daily_regime_read(bars, _flat(bars)) == "down"  # default: 52 away, far enough
    assert daily_regime_read(bars, _flat(bars), proximity_threshold=60.0) is None  # 52 < 60


def test_from_bars_proximity_atr_multiple_default_matches_today():
    bars = _bars(_confirmed_downtrend() + _ramp(80, 90, 250))
    assert daily_regime_from_bars(bars) == daily_regime_from_bars(bars, proximity_atr_multiple=None)


def test_from_bars_proximity_atr_multiple_scales_the_cutoff():
    """A tiny proximity_atr_multiple (effectively 0) should almost never
    trigger 'too close' - the confirmed downtrend fixture, which reads
    'down' by default, must still read 'down' when proximity is squeezed
    toward zero rather than flipping to None."""
    bars = _bars(_confirmed_downtrend() + _ramp(80, 90, 250))
    assert daily_regime_from_bars(bars, proximity_atr_multiple=0.001) == "down"
