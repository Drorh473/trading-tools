"""build_levels: a persistent, never-pruned set of significant BTC daily
levels - Dror's own read of a chart, not a per-signal proximity search.
Each confirmed swing either starts a new level or adds a touch to an
existing one (levels are merged by price, not re-created every time price
revisits a zone). A level's role (support/resistance) is not tracked
separately from price - see notifier/strategies/structure.py's own
nearest_level_beyond docstring: "support that broke becomes resistance on
the way back up ... not a nuance but the common case." Levels are never
removed.
"""
import pandas as pd

from notifier.strategies.levels import Level, build_levels


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


def test_a_single_confirmed_swing_becomes_one_level_with_one_touch():
    """Up, then a confirmed pullback down (the low is now a real pivot),
    then up again far enough to confirm the low as a genuine swing."""
    closes = _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40)
    bars = _bars(closes)
    levels = build_levels(bars, _flat(bars))
    assert len(levels) >= 1
    lows = [lv for lv in levels if not lv.is_high]
    assert len(lows) == 1
    assert lows[0].price == 168.0  # close=170.0, and _bars' default low = close - 2.0
    assert lows[0].touches == 1


def test_price_returning_to_an_existing_level_adds_a_touch_not_a_new_level():
    """Same low (170) confirmed twice: once on the way up, once again after
    a second pullback lands within the merge tolerance of the same price."""
    closes = (
        _ramp(100, 200, 40) + _ramp(200, 170, 15) + _ramp(170, 260, 40)   # first touch at 170
        + _ramp(260, 171, 30) + _ramp(171, 300, 40)                       # second touch, ~171 (within tolerance)
    )
    bars = _bars(closes)
    levels = build_levels(bars, _flat(bars))
    lows = [lv for lv in levels if not lv.is_high]
    assert len(lows) == 1, "the second pullback must merge into the SAME level, not create a new one"
    assert lows[0].touches == 2


def test_no_levels_from_a_plain_ramp():
    """A monotonic ramp confirms no pivots at all - zigzag_pivots finds
    nothing, so there is nothing to build a level from."""
    bars = _bars(_ramp(100, 200, 60))
    assert build_levels(bars, _flat(bars)) == []


# --------------------------------------------------------------------------
# score_level: touches + reaction size + durability + round-number
# proximity, each 0-1, summed. "Combination of all of them," per Dror -
# no single factor decides significance on its own.
# --------------------------------------------------------------------------

from notifier.strategies.levels import TOUCH_CAP, REACTION_CAP_ATR, DURABILITY_CAP_DAYS, ROUND_STEP, score_level


def test_more_touches_scores_strictly_higher_all_else_equal():
    weak = Level(price=50000.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)
    strong = Level(price=50000.0, first_index=0, is_high=False, touches=TOUCH_CAP, best_reaction_atr=0.0)
    assert score_level(strong, as_of_index=0) > score_level(weak, as_of_index=0)


def test_touch_score_caps_rather_than_growing_unbounded():
    """A level touched 20 times must not dominate every other level just by
    volume - capped at TOUCH_CAP, matching the same discipline every other
    threshold in this project uses."""
    at_cap = Level(price=50000.0, first_index=0, is_high=False, touches=TOUCH_CAP, best_reaction_atr=0.0)
    way_over = Level(price=50000.0, first_index=0, is_high=False, touches=TOUCH_CAP * 5, best_reaction_atr=0.0)
    assert score_level(at_cap, as_of_index=0) == score_level(way_over, as_of_index=0)


def test_bigger_reaction_scores_higher_up_to_the_cap():
    small = Level(price=50000.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.5)
    big = Level(price=50000.0, first_index=0, is_high=False, touches=1, best_reaction_atr=REACTION_CAP_ATR)
    huge = Level(price=50000.0, first_index=0, is_high=False, touches=1, best_reaction_atr=REACTION_CAP_ATR * 3)
    assert score_level(small, 0) < score_level(big, 0)
    assert score_level(big, 0) == score_level(huge, 0), "reaction score must cap, not keep growing"


def test_older_unbroken_level_scores_higher_for_durability():
    lvl = Level(price=50000.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)
    fresh = score_level(lvl, as_of_index=1)
    old = score_level(lvl, as_of_index=DURABILITY_CAP_DAYS)
    very_old = score_level(lvl, as_of_index=DURABILITY_CAP_DAYS * 10)
    assert fresh < old
    assert old == very_old, "durability score must cap at DURABILITY_CAP_DAYS, not keep growing forever"


def test_a_round_number_scores_higher_than_an_arbitrary_price():
    round_level = Level(price=ROUND_STEP * 50, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)
    arbitrary = Level(price=ROUND_STEP * 50 + ROUND_STEP / 2, first_index=0, is_high=False,
                       touches=1, best_reaction_atr=0.0)
    assert score_level(round_level, 0) > score_level(arbitrary, 0)


def test_score_is_the_sum_of_all_four_components_not_an_average():
    """A level maxed on every factor should score near the TOP of the range
    (4.0), not near 1.0 - confirms this is Dror's 'combination of all of
    them' (additive), not a single blended number that discards how many
    factors actually support the level."""
    maxed = Level(price=ROUND_STEP * 50, first_index=0, is_high=False,
                  touches=TOUCH_CAP, best_reaction_atr=REACTION_CAP_ATR)
    assert score_level(maxed, as_of_index=DURABILITY_CAP_DAYS) > 3.0


# --------------------------------------------------------------------------
# nearest_significant_level / level_held - the two functions the regime
# read will actually call. "Ahead" mirrors nearest_level_beyond's own
# convention: overhead for a long, below for a short. level_held mirrors
# ema_trend_v2.py's _touching(): one rejection candle (wick through, close
# back on the original side) is enough - Dror's Rule 5.
# --------------------------------------------------------------------------

from notifier.strategies.levels import level_held, nearest_significant_level


def test_nearest_significant_level_finds_the_closest_qualifying_level():
    strong = Level(price=145.0, first_index=0, is_high=False, touches=TOUCH_CAP, best_reaction_atr=REACTION_CAP_ATR)
    weak = Level(price=148.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)  # closer, but too weak
    found = nearest_significant_level([weak, strong], price=200.0, direction="short",
                                      as_of_index=DURABILITY_CAP_DAYS, min_score=2.0)
    assert found is strong, "the closer level does not clear min_score - must be skipped, not picked"


def test_nearest_significant_level_none_when_nothing_qualifies():
    weak = Level(price=148.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)
    assert nearest_significant_level([weak], price=200.0, direction="short",
                                     as_of_index=0, min_score=2.0) is None


def test_level_held_on_a_support_bounce():
    """Wick down to/through the level, close back above it - a rejection,
    per _touching's own 'trend == up' branch."""
    level = Level(price=100.0, first_index=0, is_high=False)
    bar = pd.DataFrame({"high": [103.0], "low": [98.0], "close": [101.0], "open": [102.0]})
    assert level_held(bar, level, direction="long") is True


def test_level_held_false_when_price_closes_through_instead():
    """Closes BELOW the support instead of rejecting it - a break, not a hold."""
    level = Level(price=100.0, first_index=0, is_high=False)
    bar = pd.DataFrame({"high": [101.0], "low": [97.0], "close": [98.0], "open": [100.5]})
    assert level_held(bar, level, direction="long") is False


def test_level_held_on_a_resistance_rejection():
    """Mirror: wick up to/through the level, close back below it."""
    level = Level(price=100.0, first_index=0, is_high=True)
    bar = pd.DataFrame({"high": [102.0], "low": [97.0], "close": [99.0], "open": [98.0]})
    assert level_held(bar, level, direction="short") is True


# --------------------------------------------------------------------------
# daily_regime_read_v2 - the full rule set from the 2026-08-28/29 gate
# review: structure_metrics for trend (Rule 2), an externally-maintained
# significant-levels list for both blocking and reversal (Rules 4/6), one
# rejection candle is enough to flip to the opposite direction (Rule 5).
# --------------------------------------------------------------------------

from notifier.strategies.levels import daily_regime_read_v2


def _staircase_bars(last_low=None, last_high=None, last_close=None):
    """A clean structure_metrics 'down' read (six-cycle staircase), with a
    final, fully controllable bar appended - one forming candle does not
    change the last-3 CONFIRMED pivots the trend read is based on."""
    c, price, sign = [100.0], 100.0, -1.0
    for _ in range(6):
        for _ in range(14):
            price += 3.0 * sign
            c.append(price)
        for _ in range(6):
            price -= 1.5 * sign
            c.append(price)
    c.append(c[-1] - 1.0)
    s = pd.Series(c)
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=len(s), freq="D"),
        "open": s, "high": s + 2.0, "low": s - 2.0, "close": s,
        "base_vol": 1.0, "quote_vol": 1.0,
    })
    if last_low is not None:
        df.loc[df.index[-1], "low"] = last_low
    if last_high is not None:
        df.loc[df.index[-1], "high"] = last_high
    if last_close is not None:
        df.loc[df.index[-1], "close"] = last_close
    return df


def test_v2_reads_the_trend_through_when_no_level_is_nearby():
    bars = _staircase_bars(last_low=15.0, last_close=17.0)
    assert daily_regime_read_v2(bars, levels=[], as_of_index=len(bars) - 1) == "down"


def test_v2_blocks_when_a_significant_level_is_ahead_and_not_yet_rejected():
    """Support at 15.0 sits ahead of price (17.0) in the trend's own
    direction (short). The final bar hasn't tested it yet - close (17.0)
    never even reaches it - so this must block, not read through."""
    bars = _staircase_bars(last_low=16.5, last_close=17.0)  # low never reaches 15.0
    support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = daily_regime_read_v2(bars, levels=[support], as_of_index=len(bars) - 1, min_significance=0.5)
    assert result is None


def test_v2_flips_to_the_reversal_when_the_level_holds():
    """Same support, but the final bar wicks down to it AND closes back
    above - a confirmed rejection. Must flip 'down' to 'up', not just
    unblock it back to 'down'."""
    bars = _staircase_bars(last_low=15.0, last_close=17.0, last_high=17.5)
    support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = daily_regime_read_v2(bars, levels=[support], as_of_index=len(bars) - 1, min_significance=0.5)
    assert result == "up"


def test_v2_stays_blocked_when_price_closes_through_instead_of_rejecting():
    """The level is tested and BROKEN (closes below it), not rejected - the
    break is only one bar old (the bar before it closed back above the
    support, at 16.0), not yet confirmed, so this must read None, not flip
    to 'down' just because the level is technically no longer ahead of
    price."""
    bars = _staircase_bars(last_low=14.0, last_close=14.0)
    bars.loc[bars.index[-2], "close"] = 16.0  # still above the 15.0 support, one bar before the break
    support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = daily_regime_read_v2(bars, levels=[support], as_of_index=len(bars) - 1, min_significance=0.5)
    assert result is None


def test_v2_unblocks_once_the_break_has_held_for_break_confirm_bars():
    """Same support broken, but now the last break_confirm_bars closes have
    ALL stayed below it - the break itself has held, Dror's own words:
    'require the break to hold for a bar or two.' Must read the trend
    through again, not stay blocked forever just because a level once sat
    here."""
    bars = _staircase_bars(last_low=14.0, last_close=14.0)
    bars.loc[bars.index[-2], "close"] = 14.5  # second-to-last close also below the 15.0 support
    support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = daily_regime_read_v2(bars, levels=[support], as_of_index=len(bars) - 1,
                                   min_significance=0.5, break_confirm_bars=2)
    assert result == "down"


def test_v2_ignores_a_level_too_weak_to_clear_min_significance():
    """Same support, same rejection shape, but its score does not clear a
    HIGH min_significance bar - must read the trend through, exactly as if
    the level were not on the list at all."""
    bars = _staircase_bars(last_low=15.0, last_close=17.0, last_high=17.5)
    weak_support = Level(price=15.0, first_index=0, is_high=False, touches=1, best_reaction_atr=0.0)
    result = daily_regime_read_v2(bars, levels=[weak_support], as_of_index=len(bars) - 1, min_significance=3.9)
    assert result == "down"


def test_v2_reads_none_when_there_is_no_confirmed_trend_at_all():
    """A plain ramp has no confirmed structure - must read None regardless
    of what levels exist, since there is no trend for a level to modify."""
    bars = _bars(_ramp(100, 200, 60))
    support = Level(price=90.0, first_index=0, is_high=False, touches=4, best_reaction_atr=6.0)
    result = daily_regime_read_v2(bars, levels=[support], as_of_index=len(bars) - 1, min_significance=0.1)
    assert result is None


# --------------------------------------------------------------------------
# mtf_regime_read_agree / mtf_regime_read_timing - Rule 1, "combination of
# timeframes... the 1h candles are also relevant." Two methods to be
# measured and decided between, not one picked by intuition:
#   A (agree)  - trust a direction only when daily AND 1H both show it.
#   B (timing) - daily alone sets direction; the 1H chart's OWN levels set
#                the entry-timing check instead of daily's.
# --------------------------------------------------------------------------

from notifier.strategies.levels import mtf_regime_read_agree, mtf_regime_read_timing


def _staircase_bars_up(last_low=None, last_high=None, last_close=None):
    """Mirror of _staircase_bars: a clean structure_metrics 'up' read."""
    c, price, sign = [100.0], 100.0, 1.0
    for _ in range(6):
        for _ in range(14):
            price += 3.0 * sign
            c.append(price)
        for _ in range(6):
            price -= 1.5 * sign
            c.append(price)
    c.append(c[-1] + 1.0)
    s = pd.Series(c)
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=len(s), freq="h"),
        "open": s, "high": s + 2.0, "low": s - 2.0, "close": s,
        "base_vol": 1.0, "quote_vol": 1.0,
    })
    if last_low is not None:
        df.loc[df.index[-1], "low"] = last_low
    if last_high is not None:
        df.loc[df.index[-1], "high"] = last_high
    if last_close is not None:
        df.loc[df.index[-1], "close"] = last_close
    return df


def test_agree_reads_through_when_daily_and_hourly_match_with_no_level_nearby():
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _staircase_bars(last_low=15.0, last_close=17.0)
    result = mtf_regime_read_agree(daily, hourly, daily_levels=[], as_of_index=len(daily) - 1)
    assert result == "down"


def test_agree_blocks_when_hourly_disagrees_even_though_daily_alone_would_read():
    """Same daily fixture that reads 'down' on its own (single-timeframe
    daily_regime_read_v2) - but the hourly chart shows 'up'. Method A must
    block, not fall back to daily's own opinion."""
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _staircase_bars_up(last_low=17.0, last_close=17.0)
    assert daily_regime_read_v2(daily, levels=[], as_of_index=len(daily) - 1) == "down"
    result = mtf_regime_read_agree(daily, hourly, daily_levels=[], as_of_index=len(daily) - 1)
    assert result is None


def test_agree_blocks_when_hourly_has_no_confirmed_trend():
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _bars(_ramp(100, 200, 60))  # plain ramp, no confirmed structure
    result = mtf_regime_read_agree(daily, hourly, daily_levels=[], as_of_index=len(daily) - 1)
    assert result is None


def test_agree_still_applies_the_daily_levels_check_once_timeframes_agree():
    """Once daily and hourly agree on 'down', a significant DAILY level
    still ahead of price and not yet rejected must still block - Method A
    adds a gate on TOP of the existing levels logic, not instead of it."""
    daily = _staircase_bars(last_low=16.5, last_close=17.0)  # low never reaches the 15.0 support
    hourly = _staircase_bars(last_low=15.0, last_close=17.0)
    support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = mtf_regime_read_agree(daily, hourly, daily_levels=[support],
                                    as_of_index=len(daily) - 1, min_significance=0.5)
    assert result is None


def test_timing_uses_daily_direction_even_when_hourly_trend_disagrees():
    """Daily reads 'down'; the hourly chart's OWN trend is 'up' - Method B
    never consults hourly's trend for direction, only daily's. With no
    hourly level nearby, the daily direction must read through unchanged."""
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _staircase_bars_up(last_low=17.0, last_close=17.0)
    result = mtf_regime_read_timing(daily, hourly, hourly_levels=[], as_of_index=len(hourly) - 1)
    assert result == "down"


def test_timing_blocks_on_an_unrejected_hourly_level_even_though_daily_alone_would_read():
    """The DAILY chart has no level in its way at all (empty daily_levels in
    the single-timeframe case would read straight through) - but the HOURLY
    chart has a significant level ahead of price, not yet rejected. Method B
    must block on the hourly timing check regardless of what daily's own
    chart looks like."""
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _staircase_bars(last_low=16.5, last_close=17.0)  # hourly low never reaches 15.0
    hourly_support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = mtf_regime_read_timing(daily, hourly, hourly_levels=[hourly_support],
                                     as_of_index=len(hourly) - 1, min_significance=0.5)
    assert result is None


def test_timing_flips_to_the_reversal_when_the_hourly_level_holds():
    daily = _staircase_bars(last_low=15.0, last_close=17.0)
    hourly = _staircase_bars(last_low=15.0, last_close=17.0, last_high=17.5)
    hourly_support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = mtf_regime_read_timing(daily, hourly, hourly_levels=[hourly_support],
                                     as_of_index=len(hourly) - 1, min_significance=0.5)
    assert result == "up"


def test_timing_reads_none_when_daily_has_no_confirmed_trend_regardless_of_hourly():
    daily = _bars(_ramp(100, 200, 60))
    hourly = _staircase_bars(last_low=15.0, last_close=17.0, last_high=17.5)
    hourly_support = Level(price=15.0, first_index=0, is_high=False, touches=3, best_reaction_atr=3.0)
    result = mtf_regime_read_timing(daily, hourly, hourly_levels=[hourly_support],
                                     as_of_index=len(hourly) - 1, min_significance=0.5)
    assert result is None
