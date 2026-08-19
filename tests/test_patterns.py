import pandas as pd
import pytest

from notifier.strategies import patterns
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


def _bars_oc(opens: list[float], closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    """Like _bars, but with an independently specified open per bar - needed
    wherever candle BODY direction matters (_clean_poles), since _bars()
    sets open == close, an every-bar doji that reads as neither up nor
    down.

    High/low are derived from BOTH open and close so the candle actually
    contains its own body. Deriving them from close alone (as _bars does,
    where they coincide) left a big candle's high BELOW its open, which
    understates a pole's high-low extent and silently inflates every
    retrace measured against it.
    """
    o, c = pd.Series(opens), pd.Series(closes)
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    bars = _bars(closes, highs, lows)
    bars["open"] = o
    if highs is None:
        bars["high"] = body_high + 1.0
    if lows is None:
        bars["low"] = body_low - 1.0
    return bars


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

# A pole is a short (<= FLAG_POLE_MAX_BARS), uninterrupted run of same-
# direction candles now (see _clean_poles) - not any confirmed zigzag swing
# a slow multi-day grind could satisfy. QQQUSDT 4H's real "pole" ran 10 bars
# with 2 down candles in it and Dror rejected it on sight from the rendered
# chart, before any threshold was touched: "the pole isn't a real pole".
#
# That means the pole needs a REAL candle body per bar, which _bars() can't
# give it (open == close there, an every-bar doji _clean_poles reads as
# non-directional) - hence separate opens/closes and _bars_oc. A clean
# 2-candle pole 100 -> 140, matching Dror's "a flag pole should be only 1-2
# candles moving aggressively", then a consolidation dipping to 128 (well
# under the 50% retrace cap once the pole's own last candle is excluded from
# the pause), then a breakout continuing up.
FLAG_POLE_OPENS = [100.0] * 30 + [100.0, 120.0]
FLAG_POLE_CLOSES = [100.0] * 30 + [120.0, 140.0]
FLAG_CONSOLIDATION = [136.0, 130.0, 133.0, 128.0, 131.0, 129.0, 130.5]
BULL_FLAG_OPENS = FLAG_POLE_OPENS + FLAG_CONSOLIDATION + [136.0, 139.0]
BULL_FLAG_CLOSES = FLAG_POLE_CLOSES + FLAG_CONSOLIDATION + _leg(136, 142, 2)


def _with_pole(rest_closes: list[float]) -> pd.DataFrame:
    """The clean pole above, followed by `rest_closes` as doji bars -
    direction only matters for the pole itself; everything after it is
    read purely off high/low levels."""
    return _bars_oc(FLAG_POLE_OPENS + rest_closes, FLAG_POLE_CLOSES + rest_closes)


def test_finds_a_bull_flag():
    found = flag(_bars_oc(BULL_FLAG_OPENS, BULL_FLAG_CLOSES))

    assert found, "the pole-then-tight-consolidation shape should be detected"
    assert found[0].direction == "long"
    assert found[0].name == "bull flag"


def test_the_bear_flag_is_the_mirror():
    # Mirroring each of opens/closes independently correctly flips candle
    # direction too: where the original had close > open (up), 200-close <
    # 200-open (down).
    mirrored_opens = [200 - x for x in BULL_FLAG_OPENS]
    mirrored_closes = [200 - x for x in BULL_FLAG_CLOSES]
    found = flag(_bars_oc(mirrored_opens, mirrored_closes))

    assert found
    assert found[0].direction == "short"
    assert found[0].name == "bear flag"


def test_no_flag_when_the_consolidation_gives_back_too_much():
    # A consolidation that round-trips most of the way back to the pole's
    # start is a fresh reversal, not a flag continuing it.
    assert flag(_with_pole(_leg(140, 102, 10) + _leg(102, 142, 2))) == []


def test_no_flag_when_the_breakout_comes_long_after_the_pole():
    """The pole must still be the move this breakout continues.

    Regression for the decoupled window: the consolidation was measured over
    FLAG_MIN_CONSOLIDATION_BARS while the breakout was searched to the end of
    the data, so a pole could be credited with a breakout an arbitrary distance
    later. AAPLUSDT 4H broke out 90 bars - 15 days - past its pole and was
    still called a bull flag. Here the drift stays inside the retrace cap the
    whole way, so ONLY the distance can reject it.
    """
    drift = [134.0, 130.0] * (FLAG_MAX_CONSOLIDATION_BARS + 6) + _leg(136, 145, 3)

    assert flag(_with_pole(drift)) == []


# ---- what counts as a pole (_clean_poles) ----
#
# Tested directly rather than through flag(), because routing a pole question
# through the whole pattern means a fixture can pass for the wrong reason -
# the breakout quietly failing to trigger looks identical to the pole being
# rejected. These assert on the poles themselves.


def _poles(opens, closes):
    bars = _bars_oc(opens, closes)
    return patterns._clean_poles(bars, patterns.atr(bars, patterns.ATR_PERIOD))


def test_a_counter_candle_is_not_part_of_the_pole():
    """QQQUSDT 4H's original failure: a 10-bar span with 2 down candles in it
    passed, because only the SPAN's total range was ever checked against ATR -
    never what happened bar by bar inside it. Dror rejected it on sight.

    Here the up move is interrupted at bar 31. The three runs are therefore
    bar 30 alone, bar 31 alone, and bars 32-33 - never one 100->140 pole. Only
    the last clears 4x ATR on its body, so that is the only pole found.
    """
    opens = [100.0] * 30 + [100.0, 108.0, 104.0, 118.0]
    closes = [100.0] * 30 + [108.0, 104.0, 118.0, 132.0]  # bar 31 closes DOWN

    poles = _poles(opens, closes)

    assert [(p[0], p[1]) for p in poles] == [(32, 33)], "the down candle must split the run, not join it"


def test_a_clean_run_longer_than_the_cap_is_rejected_outright():
    """Not truncated to its last FLAG_POLE_MAX_BARS either - any cut point
    inside a longer run invents a pole start with no reversal behind it
    (Dror's call). Three clean up candles is a trend leg, not a 1-2 candle
    thrust, so it yields NO pole at all rather than its last two bars."""
    opens = [100.0] * 30 + [100.0, 115.0, 130.0]
    closes = [100.0] * 30 + [115.0, 130.0, 145.0]

    assert _poles(opens, closes) == [], "a 3-bar run must not be salvaged by truncation"


def test_one_decisive_candle_is_a_pole():
    """"1-2 candles" includes one. A single candle carrying a 4x ATR body is
    exactly the aggressive thrust the rule is describing."""
    opens = [100.0] * 30 + [100.0]
    closes = [100.0] * 30 + [140.0]

    poles = _poles(opens, closes)

    assert [(p[0], p[1]) for p in poles] == [(30, 30)]


def test_a_pole_made_mostly_of_wick_does_not_qualify():
    """MMTUSDT was 62% wick against a 15% watchlist median, and Dror's fix was
    to stop measuring the wick at all. Explicit highs/lows here put a huge
    spike above a small body: the high-low range clears 4x ATR comfortably,
    the body does not, so it is no longer a pole."""
    opens = [100.0] * 30 + [100.0]
    closes = [100.0] * 30 + [106.0]  # body of 6, under 4x ATR (~8)
    highs = [101.0] * 30 + [150.0]  # but a wick reaching 150
    lows = [99.0] * 30 + [99.0]

    bars = _bars_oc(opens, closes, highs=highs, lows=lows)
    poles = patterns._clean_poles(bars, patterns.atr(bars, patterns.ATR_PERIOD))

    assert poles == [], "wick alone must not carry a pole over the threshold"


def _volatile_lead_in(base_price: float, n: int = 40):
    """n bars chopping with a ~30-point true range around base_price, so ATR
    settles high enough that 4x ATR alone would demand an impossible move -
    NBISUSDT's actual situation (daily ATR ~12% of its own price)."""
    return [base_price] * n, [base_price + 15.0] * n, [base_price - 15.0] * n


def test_the_demand_ceiling_rescues_an_impossible_pole():
    """NBISUSDT's actual shape, reproduced synthetically: ATR proportionally
    so large that 4x ATR alone demands a move nothing could make (real daily
    ATR ~12% of price demanded a 49% one-candle move; 44 of 95 watchlist
    symbols demand over 30% on 1D, 9 demand over 100%).

    Here ATR settles at ~31.2 against a 100 price - 4x ATR demands 124.9,
    impossible. A 45% one-candle move is a genuinely aggressive thrust and
    should count as a pole; without the 35% ceiling it does not (45 < 124.9).
    """
    values, highs, lows = _volatile_lead_in(100.0)
    bars = _bars_oc(values + [100.0], values + [145.0], highs=highs + [146.0], lows=lows + [99.0])

    poles = patterns._clean_poles(bars, patterns.atr(bars, patterns.ATR_PERIOD))

    assert len(poles) == 1 and poles[0][2] == "long"


def test_the_ceiling_loosens_the_bar_but_does_not_remove_it():
    """A move that clears neither 4x ATR nor the 35% ceiling must still fail -
    the cap rescues an impossible demand, it does not delete the rule."""
    values, highs, lows = _volatile_lead_in(100.0)
    bars = _bars_oc(values + [100.0], values + [110.0], highs=highs + [111.0], lows=lows + [99.0])

    assert patterns._clean_poles(bars, patterns.atr(bars, patterns.ATR_PERIOD)) == []


def test_the_ceiling_is_anchored_on_the_correct_side_for_a_short():
    """A short's move starts at body_top and falls, so the % cap's reference
    price must be body_top - using body_bottom (the lower, POST-decline
    price) is the WRONG direction of error: it shrinks the cap and makes
    qualifying EASIER, not harder, so a move must be picked that the correct
    200-based cap (70) rejects while the wrong 140-based cap (49) would
    wrongly accept. A 30% fall from 200 to 140 (body 60) is exactly that:
    60 < 70 (reject, correct) but 60 >= 49 (accept, the bug)."""
    values, highs, lows = _volatile_lead_in(200.0)
    bars = _bars_oc(values + [200.0], values + [140.0], highs=highs + [201.0], lows=lows + [139.0])

    poles = patterns._clean_poles(bars, patterns.atr(bars, patterns.ATR_PERIOD))

    assert poles == [], "a move the correctly-anchored cap rejects must not slip through on the wrong side"


# A pause whose CLOSES all sit under the break level (141, the pole's own
# last-candle wick) but whose WICKS reach far above it. This is the only shape
# tightness can still reject now that the break level is fixed: closes are
# boxed by that level, wicks are not. It is the real QQQUSDT 4H shape, which
# measured 0.777 tightness under these rules - 1 of 20 otherwise-accepted
# flags - and is the reason the check survived a decision to delete it as
# redundant. Retrace deliberately stays shallow (low of 128 against a 141 top
# over a 42-point pole = 0.31, under the 0.5 cap) so ONLY tightness can bite.
_WICK_WIDE_OPENS = [138.0, 134.0, 136.0, 132.0, 137.0]
_WICK_WIDE_CLOSES = [134.0, 136.0, 132.0, 137.0, 135.0]
_WICK_WIDE_HIGHS = [160.0, 140.0, 139.0, 140.0, 139.0]  # 160 pokes well above the level
_WICK_WIDE_LOWS = [132.0, 130.0, 128.0, 131.0, 133.0]


def _wick_wide_bars(with_breakout: bool) -> pd.DataFrame:
    opens = list(FLAG_POLE_OPENS) + _WICK_WIDE_OPENS
    closes = list(FLAG_POLE_CLOSES) + _WICK_WIDE_CLOSES
    highs = [c + 1.0 for c in FLAG_POLE_CLOSES[:30]] + [121.0, 141.0] + _WICK_WIDE_HIGHS
    lows = [c - 1.0 for c in FLAG_POLE_CLOSES[:30]] + [99.0, 119.0] + _WICK_WIDE_LOWS
    if with_breakout:
        opens, closes = opens + [135.0], closes + [175.0]
        highs, lows = highs + [176.0], lows + [134.0]
    return _bars_oc(opens, closes, highs=highs, lows=lows)


def test_no_flag_when_the_consolidation_is_wider_than_its_own_pole():
    """A pause spanning more than its pole is a leg, not a pause.

    Retrace cannot see this: it measures only travel AGAINST the pole, so a
    consolidation whose wicks range far ABOVE the pole's top still scores as
    shallow. Nor can the fixed break level, which only looks at closes.
    """
    assert flag(_wick_wide_bars(with_breakout=True)) == []


def test_the_tightness_ceiling_is_what_rejects_the_over_wide_consolidation(monkeypatch):
    """Guards the test above against passing for the wrong reason.

    A negative control is only worth having if the rule under test is what
    rejects it. Lifting the ceiling must bring the pattern back - an earlier
    fixture here passed while tightness did nothing at all, which proved
    nothing and would have hidden a broken rule.
    """
    monkeypatch.setattr(patterns, "FLAG_MAX_TIGHTNESS", 10.0)

    assert flag(_wick_wide_bars(with_breakout=True)), (
        "with the ceiling lifted this shape must detect, or the fixture is inert"
    )


def test_the_break_level_is_the_poles_wick_and_does_not_ratchet():
    """Dror caught the ratchet from both sides on real charts.

    The level used to be the consolidation's running extreme, so while price
    drifted the level drifted with it. ENAUSDT 4H then broke a bar LATE (wicks
    at 0.0897 / 0.0901 / 0.09037 had dragged the level above his 0.09023
    resumption close) and GOOGLUSDT 4H broke two bars EARLY off a level with
    only three bars behind it.

    Here the consolidation's own highs (up to 139) climb above every close but
    stay under the pole's 141 wick. Under the ratchet the breakout would be
    whichever bar first cleared that climbing 139; against the fixed level it
    is the bar that actually closes above 141 - and the pattern's own
    invalidation is the running low, per "if it is a bull flag the new low
    should be the lowest".
    """
    opens = list(FLAG_POLE_OPENS) + [136.0, 132.0, 135.0, 131.0, 134.0, 138.0]
    closes = list(FLAG_POLE_CLOSES) + [132.0, 135.0, 131.0, 134.0, 138.0, 145.0]
    bars = _bars_oc(opens, closes)

    found = flag(bars)

    assert found, "a pause that never closes past the pole's wick, then does, is a flag"
    p = found[0]
    # 145 is the first close above the pole's 141 wick; the bars before it all
    # closed under it despite making higher highs than each other.
    assert bars["close"].iloc[p.breakout_index] == pytest.approx(145.0)
    assert p.invalidation_level == pytest.approx(bars["low"].iloc[len(FLAG_POLE_CLOSES) : p.breakout_index].min())


def test_a_break_before_the_minimum_pause_kills_the_flag_outright():
    """NBISUSDT 4H (1 bar of pause) and MSFTUSDT 1D (2 bars), both of which
    Dror rejected as "just part of a trend".

    The level is simply gone - price took it before any pause worth the name
    formed - so the pole is discarded rather than kept alive hunting for a
    later close past a level that no longer exists. Here the break comes 1
    bar in, under the 4-bar minimum, and there is a perfectly good later
    breakout at the end which must NOT be credited to this pole.

    The first pause bar has to close DOWN. An earlier version of this fixture
    opened it upward, which extended the pole's own same-direction run to 3
    bars and disqualified the pole before any of this logic ran - the test
    then passed against the reverted code too, proving nothing.
    """
    opens = list(FLAG_POLE_OPENS) + [138.0, 136.0, 150.0, 148.0, 152.0]
    closes = list(FLAG_POLE_CLOSES) + [136.0, 145.0, 148.0, 152.0, 160.0]

    assert flag(_bars_oc(opens, closes)) == []


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
    still_coiling = _with_pole(FLAG_CONSOLIDATION)  # no breakout leg
    found = pending_flag(still_coiling)

    assert found, "a pole with a live consolidation should be a pending flag"
    p = found[0]
    assert p.name == "bull flag"
    assert p.direction == "long"
    # The break level is the POLE's own last-candle wick high, fixed. Using
    # the consolidation's running high instead (as this once did) made
    # "pending" a tautology - no close inside a window can exceed that
    # window's own highest high, so every flag found was pending by
    # construction rather than because price had actually held below a level.
    pole_end = len(FLAG_POLE_CLOSES) - 1
    assert p.break_level == pytest.approx(still_coiling["high"].iloc[pole_end])
    assert p.break_level > still_coiling["high"].iloc[pole_end + 1 :].max(), (
        "the pole's wick, not the consolidation's own high"
    )
    assert p.break_level > still_coiling["close"].iloc[-1]
    assert p.invalidation_level < p.break_level
    assert p.drift_per_bar == 0.0  # a flag's boundary is a fixed price


def test_pending_bear_flag_is_the_mirror():
    mirrored_opens = [200 - x for x in FLAG_POLE_OPENS + FLAG_CONSOLIDATION]
    mirrored_closes = [200 - x for x in FLAG_POLE_CLOSES + FLAG_CONSOLIDATION]
    mirrored = _bars_oc(mirrored_opens, mirrored_closes)
    found = pending_flag(mirrored)

    assert found
    assert found[0].name == "bear flag"
    assert found[0].direction == "short"
    assert found[0].break_level < mirrored["close"].iloc[-1]


def test_no_pending_flag_once_it_gives_back_too_much():
    # Same rule the broken-out path uses: past half the pole it is a reversal,
    # not a flag still waiting to continue.
    assert pending_flag(_with_pole(_leg(140, 102, 10))) == []


def test_no_pending_flag_once_the_consolidation_outlives_the_window():
    stalled = FLAG_CONSOLIDATION + [130.0] * (FLAG_MAX_CONSOLIDATION_BARS + 5)
    assert pending_flag(_with_pole(stalled)) == []


def test_pending_inverse_head_and_shoulders_before_the_neckline_gives_way():
    # IHS's last 18 bars are its breakout; replace them with a rise that
    # confirms the right shoulder as a pivot but stops short of the neckline.
    approaching = _bars(IHS[:-18] + _leg(80, 95, 10))
    found = pending_inverse_head_and_shoulders(approaching)

    assert found, "the five-pivot shape should be found before the neckline breaks"
    p = found[0]
    assert p.direction == "long"
    # Read off the candle BODY now, not the wick. _bars sets open == close,
    # so the 100-close peaks give a 100 neckline - the extra point of wick no
    # longer defines the level.
    assert p.break_level == pytest.approx(100.0)
    # Invalidation is the head, not the neckline - the neckline is what we are
    # waiting to break, so it cannot also be what kills the setup.
    assert p.invalidation_level == pytest.approx(60.0, abs=1.0)


def test_no_pending_ihs_once_the_neckline_has_already_broken():
    assert pending_inverse_head_and_shoulders(_bars(IHS)) == []


def test_pending_triangle_carries_a_moving_break_level():
    # Drop the breakout leg, then add a small bounce. Without it the final low
    # never reverses far enough to be CONFIRMED as a pivot, so the lower
    # boundary has only two touches - and two points define any line, which is
    # exactly what the three-touch rule exists to reject.
    coiling = _bars(ASCENDING_TRIANGLE[:-4] + _leg(130, 140, 3))
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
    still_coiling = _with_pole(FLAG_CONSOLIDATION)

    result = pending({"1H": still_coiling}, "long")
    assert result is not None
    p, timeframe = result
    assert timeframe == "1H"
    assert p.direction == "long"

    # A long-arguing pattern must never be offered to a short signal.
    assert pending({"1H": still_coiling}, "short") is None


def test_pending_is_none_on_structureless_data():
    assert pending({"1H": _bars([100.0] * 120)}, "long") is None


def test_no_head_and_shoulders_when_the_two_necks_are_at_unrelated_prices():
    # The neckline is only worth breaking because the market turned at the
    # SAME place twice - that is what makes it defended support. Taking one
    # neck and ignoring the other accepted two turns at completely different
    # prices and called the line between them major support.
    lopsided = [
        100.0,
        *_leg(100, 80, 12),
        *_leg(80, 130, 12),  # first neck way up at 130...
        *_leg(130, 60, 20),
        *_leg(60, 100, 20),  # ...second one back at 100
        *_leg(100, 80, 12),
        *_leg(80, 115, 18),
    ]
    assert inverse_head_and_shoulders(_bars(lopsided)) == []
    # The balanced original still works, so this rejects lopsidedness rather
    # than the pattern.
    assert inverse_head_and_shoulders(_bars(IHS)) != []


def test_the_neckline_is_horizontal_not_sloped():
    """Dror's call, reading the rendered charts.

    NECKLINE_TOLERANCE already demands the two necks sit at essentially the
    same price - that shared level is the whole reason the line is worth
    breaking - so any slope between them is noise measured over a long base.
    Extrapolating it walks the break level away from the level the market
    actually defended, and for a PENDING pattern that level is quoted in the
    alert and re-derived every five minutes.
    """
    # Necks deliberately a little apart, so a sloped fit would drift visibly.
    necks = (
        [100.0] * 20
        + _leg(100, 130, 6) + _leg(130, 104, 6)      # left shoulder, neck at ~104
        + _leg(104, 155, 8) + _leg(155, 108, 8)      # head, second neck at ~108
        + _leg(108, 132, 6) + _leg(132, 95, 6)       # right shoulder, then the break
    )
    found = head_and_shoulders(_bars(necks))

    assert found, "the shape should still be detected"
    pat = found[0]
    pending = patterns.pending_head_and_shoulders(_bars(necks[: -6]))
    for p in pending:
        assert p.drift_per_bar == 0.0, "a neckline is a level, not a converging line"
    # The invalidation level is the neckline itself and must sit between the
    # two necks rather than being extrapolated past either of them.
    assert 100.0 < pat.invalidation_level < 115.0


# ---- the shape has to survive until the neckline breaks (XAGUSDT, 2026-08-13) ----
#
# The right shoulder is confirmed by a rally that stops short of the neckline,
# so the later dip lands AFTER the pivot instead of becoming it - which is the
# geometry XAGUSDT's 4H chart actually had.
_IHS_RALLY = [*IHS[:-18], *_leg(80, 95, 8)]
# Identical, then straight up through the neckline: the shape held.
IHS_SHOULDERS_HELD = [*_IHS_RALLY, *_leg(95, 115, 22)]
# Identical, but price first drops to 74 - under both 80 shoulders - and only
# then breaks. Same neckline, same break, different structure doing it.
IHS_SHOULDERS_GAVE_WAY = [*_IHS_RALLY, *_leg(95, 74, 11), *_leg(74, 115, 22)]


def test_a_break_only_counts_while_the_shoulders_still_hold():
    """XAGUSDT's 4H "inverse head-and-shoulders" put its right shoulder in on
    2026-07-23 and did not break its neckline until 2026-08-05 - 75 bars, with
    price below BOTH shoulders on 15 of them. The base that broke 60.11 was a
    different structure; the pattern being credited was three weeks gone. The
    break scan ran to the end of the series with nothing checking that the
    shape was still there."""
    assert inverse_head_and_shoulders(_bars(IHS_SHOULDERS_HELD)), "control: the shape held, so it counts"

    assert inverse_head_and_shoulders(_bars(IHS_SHOULDERS_GAVE_WAY)) == []


def test_the_shoulder_rule_holds_for_the_upright_pattern_too():
    """Same detector, mirrored: shoulders giving way UPWARD ends the shape."""
    assert head_and_shoulders(_bars([200 - x for x in IHS_SHOULDERS_HELD]))

    assert head_and_shoulders(_bars([200 - x for x in IHS_SHOULDERS_GAVE_WAY])) == []


# ---- a pattern that already paid out is not confirmation ----

# Neckline 100, head 60, so the measured move is 100 + 40 = 140. This runs to
# 145, past its own objective.
IHS_MOVE_SPENT = [*IHS[:-18], *_leg(80, 145, 22)]


def test_the_pattern_records_its_measured_move():
    found = inverse_head_and_shoulders(_bars(IHS))

    assert found[0].target == pytest.approx(140.0)  # neckline 100 + depth 40


def test_a_pattern_whose_move_is_already_spent_is_not_confluence():
    """XAGUSDT projected 64.72 from a 60.11 neckline, ran to 66.41, and was
    still cited as confirmation for a long entered at 64.64 targeting 66.07 -
    both at or above the objective the pattern itself argued for."""
    bars = _bars(IHS_MOVE_SPENT)

    # The shape is still a real one; it has simply already done its work.
    assert inverse_head_and_shoulders(bars), "the shape itself stays valid"

    assert confluence({"4H": bars}, "long") != "inverse head-and-shoulders on 4H"


def test_a_pattern_short_of_its_target_still_counts():
    """The guard must not swallow every pattern - IHS breaks to 115 against a
    140 objective and stays valid confirmation."""
    assert confluence({"4H": _bars(IHS)}, "long") == "inverse head-and-shoulders on 4H"


def test_pending_prefers_the_most_recent_shape():
    """MMTUSDT 4H, 2026-08-18: the alert quoted a pending head-and-shoulders
    breaking at 0.1566, from a shape whose right shoulder printed 63 bars
    earlier - while a newer one sat in front of price.

    pending() promises "the unbroken pattern sitting in front of price right
    now", but it returned the FIRST match, and detectors walk pivots
    oldest-first. So the oldest live shape won, every time, which is exactly
    backwards.
    """
    old = patterns.PendingPattern(
        name="head-and-shoulders", direction="short",
        break_level=0.1566, invalidation_level=0.2950, formed_at=353,
    )
    recent = patterns.PendingPattern(
        name="head-and-shoulders", direction="short",
        break_level=0.1642, invalidation_level=0.1850, formed_at=410,
    )
    wrong_way = patterns.PendingPattern(
        name="inverse head-and-shoulders", direction="long",
        break_level=0.1800, invalidation_level=0.1500, formed_at=415,
    )

    def detector(bars):
        return [old, recent, wrong_way]      # oldest first, as detectors emit

    original = patterns._PENDING_DETECTORS
    patterns._PENDING_DETECTORS = (detector,)
    try:
        found, tf = patterns.pending({"4H": _bars([1.0] * 30)}, "short")
    finally:
        patterns._PENDING_DETECTORS = original

    assert found is recent, "the newer shape is the one in front of price"
    assert tf == "4H"


def test_pending_still_respects_the_timeframe_search_order():
    """Only the choice WITHIN one detector's results changed. The timeframe and
    detector precedence mirrors confluence()'s deliberately, so a 1D shape
    still outranks a more recently formed 4H one."""
    daily = patterns.PendingPattern(
        name="head-and-shoulders", direction="short",
        break_level=1.0, invalidation_level=2.0, formed_at=5,
    )
    intraday = patterns.PendingPattern(
        name="head-and-shoulders", direction="short",
        break_level=3.0, invalidation_level=4.0, formed_at=999,
    )

    def detector(bars):
        return [daily] if len(bars) == 30 else [intraday]

    original = patterns._PENDING_DETECTORS
    patterns._PENDING_DETECTORS = (detector,)
    try:
        found, tf = patterns.pending(
            {"1D": _bars([1.0] * 30), "4H": _bars([1.0] * 40)}, "short"
        )
    finally:
        patterns._PENDING_DETECTORS = original

    assert found is daily and tf == "1D"
