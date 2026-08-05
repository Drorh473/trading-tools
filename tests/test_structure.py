import pandas as pd

from notifier.strategies.structure import trend_structure, zigzag_pivots


def _bars(closes, highs=None, lows=None):
    s = pd.Series(closes)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq="h"),
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


def test_a_higher_low_inside_a_downtrend_is_not_a_break():
    """The AAPLUSDT correction, in miniature.

    Lows of 100, then 118, 124, then 108 and 104. Compared pairwise the last two
    look like fresh breaks; against the downtrend's actual low of 100 neither
    broke anything. Treating them as breaks decays the protected high and makes
    an ordinary bounce look like a trend change.
    """
    closes = (
        _ramp(200, 100, 30)          # the down leg, low 100
        + _ramp(100, 150, 20)        # bounce, forms the protected high
        + _ramp(150, 118, 12)        # higher low
        + _ramp(118, 140, 10)
        + _ramp(140, 108, 12)        # another higher low - still above 100
        + _ramp(108, 130, 10)
    )
    bars = _bars(closes)
    s = trend_structure(bars, _flat(bars))

    assert s.trend == "down", "higher lows above the trend low must not flip the trend"


def test_the_trend_turns_only_when_the_protected_level_breaks():
    # The pullback off 260 is what makes that high a confirmed swing; without
    # it the peak is still forming and there is no structure to read.
    rising = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40) + _ramp(260, 235, 10)
    bars = _bars(rising)
    s = trend_structure(bars, _flat(bars))
    assert s.trend == "up"

    # Now break back under the 170 swing low, then bounce so it confirms.
    broken = rising + _ramp(260, 150, 30) + _ramp(150, 200, 15)
    s2 = trend_structure(_bars(broken), _flat(_bars(broken)))
    assert s2.trend == "down"


def test_the_anchor_reaches_past_smaller_highs_formed_since():
    """What makes this immune to the pivot threshold.

    After the trend turns down the market prints several smaller highs. None of
    them may become the anchor by merely being the most recent thing found -
    that is precisely the bug that anchored AAPLUSDT on 313.36 and produced a
    0.62% stop the fees ate 19% of.
    """
    closes = (
        _ramp(100, 200, 25)          # an uptrend WITH structure - a monotonic
        + _ramp(200, 170, 12)        # ramp has no pivots for a break to break
        + _ramp(170, 300, 25)        # the high the trend turns from
        + _ramp(300, 150, 30)        # breaks under 170 -> trend down
        + _ramp(150, 240, 15)        # smaller high
        + _ramp(240, 190, 12)
        + _ramp(190, 230, 12)        # smaller high again
        + _ramp(230, 195, 12)
    )
    bars = _bars(closes)
    s = trend_structure(bars, _flat(bars))

    assert s.trend == "down"
    anchor_price = bars["high"].iloc[s.anchor_index]
    assert anchor_price > 290, f"anchor should be the 300 high, got {anchor_price}"


def test_no_trend_without_enough_structure():
    bars = _bars(_ramp(100, 300, 60))
    assert trend_structure(bars, _flat(bars)).trend is None


def test_bootstrap_does_not_stall_on_a_wide_first_swing():
    """Regression: comparing against the RUNNING extreme while no trend exists
    left the trend permanently unset, because the first pivot in the window is
    frequently the widest and nothing after it can exceed it."""
    closes = (
        _ramp(1650, 1000, 10)        # a crash opens the window
        + _ramp(1000, 1130, 40)
        + _ramp(1130, 1050, 20)
        + _ramp(1050, 1270, 40)      # higher high -> must establish an uptrend
        + _ramp(1270, 1200, 12)
    )
    bars = _bars(closes)

    assert trend_structure(bars, _flat(bars, 20.0)).trend is not None
