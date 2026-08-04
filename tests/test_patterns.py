import pandas as pd
import pytest

from notifier.strategies.patterns import (
    CONFLUENCE_BARS,
    FLAG_MAX_CONSOLIDATION_BARS,
    confluence,
    cup_and_handle,
    flag,
    head_and_shoulders,
    inverse_head_and_shoulders,
    pending,
    pending_cup_and_handle,
    pending_flag,
    pending_inverse_head_and_shoulders,
    pending_triangle_or_wedge,
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


# ---- pending (not yet broken) patterns ----
#
# Same fixtures as above, truncated before their breakout: the shape is
# complete and intact, but the level that would confirm it is still ahead.


def test_pending_flag_found_while_the_consolidation_is_still_running():
    still_coiling = _bars(FLAG_POLE + FLAG_CONSOLIDATION)  # no breakout leg
    found = pending_flag(still_coiling)

    assert found, "a pole with a live consolidation should be a pending flag"
    p = found[0]
    assert p.name == "bull flag"
    assert p.direction == "long"
    # The break level is the consolidation's own running high, so by
    # construction price has not closed through it yet.
    assert p.break_level == pytest.approx(still_coiling["high"].iloc[-13:].max())
    assert p.break_level > still_coiling["close"].iloc[-1]
    assert p.invalidation_level < p.break_level
    assert p.drift_per_bar == 0.0  # a flag's boundary is a fixed price


def test_pending_bear_flag_is_the_mirror():
    mirrored = _bars([200 - x for x in FLAG_POLE + FLAG_CONSOLIDATION])
    found = pending_flag(mirrored)

    assert found
    assert found[0].name == "bear flag"
    assert found[0].direction == "short"
    assert found[0].break_level < mirrored["close"].iloc[-1]


def test_no_pending_flag_once_it_gives_back_too_much():
    # Same rule the broken-out path uses: past half the pole it is a reversal,
    # not a flag still waiting to continue.
    assert pending_flag(_bars(FLAG_POLE + _leg(140, 102, 10))) == []


def test_no_pending_flag_once_the_consolidation_outlives_the_window():
    stalled = FLAG_POLE + FLAG_CONSOLIDATION + [130.0] * (FLAG_MAX_CONSOLIDATION_BARS + 5)
    assert pending_flag(_bars(stalled)) == []


def test_pending_inverse_head_and_shoulders_before_the_neckline_gives_way():
    # IHS's last 18 bars are its breakout; replace them with a rise that
    # confirms the right shoulder as a pivot but stops short of the neckline.
    approaching = _bars(IHS[:-18] + _leg(80, 95, 10))
    found = pending_inverse_head_and_shoulders(approaching)

    assert found, "the five-pivot shape should be found before the neckline breaks"
    p = found[0]
    assert p.direction == "long"
    # The neckline is read off the highs, and _bars puts each high a point
    # above its close, so the 100-close peaks give a 101 neckline.
    assert p.break_level == pytest.approx(101.0)
    # Invalidation is the head, not the neckline - the neckline is what we are
    # waiting to break, so it cannot also be what kills the setup.
    assert p.invalidation_level == pytest.approx(60.0, abs=1.0)


def test_no_pending_ihs_once_the_neckline_has_already_broken():
    assert pending_inverse_head_and_shoulders(_bars(IHS)) == []


def test_pending_triangle_carries_a_moving_break_level():
    coiling = _bars(ASCENDING_TRIANGLE[:-4])  # drop the breakout leg
    found = pending_triangle_or_wedge(coiling)

    assert found
    p = found[0]
    assert p.name == "ascending triangle"
    assert p.direction == "long"
    # Unlike a flag, a fitted trendline moves - that is why the level is
    # recomputed live rather than frozen at alert time.
    assert p.drift_per_bar != 0.0
    assert p.invalidation_level < p.break_level


# CUP_AND_HANDLE can't be reused by truncation here: its right side is a
# steady climb, so the first bar within rim tolerance is found part-way up and
# the climb then closes back above that rim - which is a break, not a pending
# setup. This fixture tops out in one bar instead, so the detected right rim
# IS the peak and the handle stays under it.
PENDING_CUP = (
    [140.0] * 20
    + _leg(140, 150, 6)  # left rim
    + _leg(150, 100, 15)  # down into the cup
    + [100.0, 99.0, 100.5, 99.5, 100.0, 100.5, 99.0, 100.0]  # rounded base
    + _leg(100, 138, 12)  # back up, stopping just under the rim band
    + [150.0]  # the right rim itself
    + [148.0, 146.0, 147.0, 145.5, 147.5, 146.5]  # handle, still forming
)


def test_pending_cup_and_handle_while_the_handle_is_still_forming():
    forming = _bars(PENDING_CUP)
    found = pending_cup_and_handle(forming)

    assert found
    p = found[0]
    assert p.name == "cup-and-handle"
    assert p.direction == "long"
    assert p.break_level > forming["close"].iloc[-1]  # the rim, still overhead
    assert p.drift_per_bar == 0.0


def test_pending_matches_direction_and_reports_the_timeframe():
    still_coiling = _bars(FLAG_POLE + FLAG_CONSOLIDATION)

    result = pending({"1H": still_coiling}, "long")
    assert result is not None
    p, timeframe = result
    assert timeframe == "1H"
    assert p.direction == "long"

    # A long-arguing pattern must never be offered to a short signal.
    assert pending({"1H": still_coiling}, "short") is None


def test_pending_is_none_on_structureless_data():
    assert pending({"1H": _bars([100.0] * 120)}, "long") is None
