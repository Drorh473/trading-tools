import pandas as pd
import pytest

from notifier.strategies.patterns import (
    CONFLUENCE_BARS,
    confluence,
    head_and_shoulders,
    inverse_head_and_shoulders,
)


def _bars(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    s = pd.Series(closes)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq="h"),
            "open": s,
            "high": pd.Series(highs) if highs is not None else s + 1.0,
            "low": pd.Series(lows) if lows is not None else s - 1.0,
            "close": s,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _leg(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


# Left shoulder at 80, head at 60, right shoulder at 80, with both intervening
# peaks at 100 forming the neckline, then a break above it. Every leg is large
# enough to clear the 4x ATR pivot threshold.
IHS = [
    100.0,
    *_leg(100, 80, 12),
    *_leg(80, 100, 12),
    *_leg(100, 60, 20),
    *_leg(60, 100, 20),
    *_leg(100, 80, 12),
    *_leg(80, 115, 18),
]


def test_finds_an_inverse_head_and_shoulders():
    found = inverse_head_and_shoulders(_bars(IHS))

    assert found, "the textbook shape should be detected"
    assert found[0].direction == "long"
    assert found[0].name == "inverse head-and-shoulders"


def test_the_upright_pattern_is_the_mirror():
    found = head_and_shoulders(_bars([200 - x for x in IHS]))

    assert found
    assert found[0].direction == "short"


def test_no_pattern_without_a_breakout():
    # Same shape, but price never closes back above the neckline, and the
    # cheatsheet is explicit that entry is on the break and never before it.
    unbroken = IHS[: -18] + list(_leg(80, 95, 18))

    assert inverse_head_and_shoulders(_bars(unbroken)) == []


def test_no_pattern_when_the_head_is_not_the_deepest():
    # A middle low that is shallower than its shoulders is not a head, so the
    # three lows are just a range rather than the pattern.
    flat = [100.0, *_leg(100, 60, 12), *_leg(60, 100, 12), *_leg(100, 80, 20), *_leg(80, 100, 20), *_leg(100, 60, 12), *_leg(60, 115, 18)]

    assert inverse_head_and_shoulders(_bars(flat)) == []


def test_confluence_matches_direction_and_reports_the_timeframe():
    bars = _bars(IHS)

    assert confluence({"1H": bars}, "long") == "inverse head-and-shoulders on 1H"
    assert confluence({"1H": bars}, "short") is None  # bullish pattern, bearish signal


def test_confluence_expires_once_the_breakout_is_old():
    stale = _bars(IHS + list(_leg(115, 118, CONFLUENCE_BARS + 10)))

    assert confluence({"1H": stale}, "long") is None


def test_confluence_is_none_on_structureless_data():
    assert confluence({"1H": _bars([100.0] * 200)}, "long") is None
