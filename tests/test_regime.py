"""daily_regime_read: Dror's own pre-trade BTC check as code - trend
direction (trend_structure) PLUS whether price sits too close to a
confirmed level ahead of it to trust that direction right now
(nearest_level_beyond). Built for the standalone question "is this read
statistically persistent", independent of any specific strategy.
"""
import pandas as pd

from notifier.strategies.regime import daily_regime_read


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
