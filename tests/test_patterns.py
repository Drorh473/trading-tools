import pandas as pd
import pytest

from notifier.strategies.patterns import (
    CONFLUENCE_BARS,
    confluence,
    cup_and_handle,
    flag,
    head_and_shoulders,
    inverse_head_and_shoulders,
    triangle_or_wedge,
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


def test_confluence_invalidated_once_price_crosses_back_through_the_neckline():
    # AEVOUSDT: a bearish breakout round-tripped 42% back above its own
    # neckline within the same recency window, tracing out the opposite
    # pattern on the way there - yet the alert still cited it as short
    # confirmation. IHS's neckline is 100; giving it back afterward should
    # stop the pattern counting as long confluence too.
    gave_it_back = _bars(IHS + list(_leg(115, 90, 5)))

    assert confluence({"1H": gave_it_back}, "long") is None


# ---- flags / pennants ----

# A sharp 6-bar pole (100 -> 140), a consolidation that dips to 128 (30% of
# the 40pt pole - comfortably under the 50% cap, and still big enough
# relative to ATR for the peak to register as a confirmed pivot), then a
# breakout continuing up.
FLAG_POLE = [100.0] * 30 + _leg(100, 140, 6)
FLAG_CONSOLIDATION = [136.0, 130.0, 133.0, 128.0, 131.0, 129.0, 130.5]
BULL_FLAG = FLAG_POLE + FLAG_CONSOLIDATION + _leg(136, 142, 2)


def test_finds_a_bull_flag():
    found = flag(_bars(BULL_FLAG))

    assert found, "the pole-then-tight-consolidation shape should be detected"
    assert found[0].direction == "long"
    assert found[0].name == "bull flag"


def test_the_bear_flag_is_the_mirror():
    mirrored = [200 - x for x in BULL_FLAG]
    found = flag(_bars(mirrored))

    assert found
    assert found[0].direction == "short"
    assert found[0].name == "bear flag"


def test_no_flag_when_the_consolidation_gives_back_too_much():
    # A consolidation that round-trips most of the way back to the pole's
    # start is a fresh reversal, not a flag continuing it.
    deep_giveback = FLAG_POLE + _leg(140, 102, 10) + _leg(102, 142, 2)

    assert flag(_bars(deep_giveback)) == []


# ---- triangles / wedges ----

# Ascending triangle: three swings up to a flat ~150 resistance, each pullback
# landing on a higher low - a rising support against a flat top.
ASCENDING_TRIANGLE = (
    [100.0] * 20
    + _leg(100, 150, 8)
    + _leg(150, 115, 8)
    + _leg(115, 148, 8)
    + _leg(148, 122, 8)
    + _leg(122, 151, 8)
    + _leg(151, 130, 6)
    + _leg(130, 160, 4)
)


def test_finds_an_ascending_triangle():
    found = triangle_or_wedge(_bars(ASCENDING_TRIANGLE))

    assert found
    assert found[0].name == "ascending triangle"
    assert found[0].direction == "long"


def test_the_descending_triangle_is_the_mirror():
    mirrored = [250 - x for x in ASCENDING_TRIANGLE]
    found = triangle_or_wedge(_bars(mirrored))

    assert found
    assert found[0].name == "descending triangle"
    assert found[0].direction == "short"


# ---- cup and handle ----

CUP_AND_HANDLE = (
    [140.0] * 20
    + _leg(140, 150, 6)  # up to the left rim
    + _leg(150, 100, 15)  # down into the cup
    + [100.0, 99.0, 100.5, 99.5, 100.0, 100.5, 99.0, 100.0]  # rounded base
    + _leg(100, 150, 15)  # back up to the right rim
    + [148.0, 146.0, 147.0, 145.5, 147.5, 146.5]  # the handle - a shallow pullback
    + _leg(148, 158, 4)  # breakout above the rim
)


def test_finds_a_cup_and_handle():
    found = cup_and_handle(_bars(CUP_AND_HANDLE))

    assert found
    assert found[0].direction == "long"
    assert found[0].name == "cup-and-handle"


def test_no_cup_and_handle_on_a_v_shaped_spike():
    # A sharp V is a different pattern (a spike reversal) - the cup requires
    # a base with width, not just depth.
    v_shape = [140.0] * 20 + _leg(140, 150, 6) + list(reversed(_leg(150, 60, 10))) + _leg(60, 150, 10)
    v_shape += [148.0, 146.0, 147.0, 145.5, 147.5, 146.5] + _leg(148, 158, 4)

    assert cup_and_handle(_bars(v_shape)) == []
