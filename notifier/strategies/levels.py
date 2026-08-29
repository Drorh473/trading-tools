"""A persistent, never-pruned set of significant price levels - Dror's own
chart-reading habit as code, built during the 2026-08-28/29 gate review:
"save a list of significant levels of the btc daily and use them."

Distinct from notifier/strategies/structure.py's nearest_level_beyond, which
re-searches a window for the single nearest confirmed swing every call. Here
a level is a first-class, accumulating object: it remembers every time price
has come back and reacted, and it is never removed once formed - a broken
level does not vanish, it just changes which side of price it sits on (see
nearest_level_beyond's own docstring: "support that broke becomes resistance
on the way back up ... not a nuance but the common case").
"""
from dataclasses import dataclass, field

import pandas as pd

from notifier.strategies.ema_trend_v2 import STRUCTURE_LOOKBACK, structure_metrics
from notifier.strategies.structure import zigzag_pivots

# How close a new pivot must land to an existing level's price to count as
# another touch of it, rather than a brand new level - in the SAME threshold
# units as pivot confirmation itself, so "close enough to be the same level"
# scales with volatility the same way "far enough to be a real swing" does.
MERGE_ATR_MULTIPLE = 1.0

# score_level's four components, each normalised to [0, 1] and SUMMED (not
# averaged) - "combination of all of them," Dror's own words. A level strong
# on every factor should score near the top of the range, not get diluted
# toward one blended number that hides how many factors actually back it.
# None of these caps has been measured yet; they are reasonable first
# values, explicitly meant to be swept once the mechanism itself works.
TOUCH_CAP = 4                 # 4+ touches = max touches_score
REACTION_CAP_ATR = 6.0        # 6+ ATR reaction = max reaction_score
DURABILITY_CAP_DAYS = 180     # ~6 months unbroken = max durability_score
ROUND_STEP = 1000.0           # BTC "round" numbers, e.g. every $1,000


@dataclass(frozen=True)
class Level:
    price: float
    first_index: int
    is_high: bool
    touches: int = 1
    best_reaction_atr: float = 0.0


def build_levels(bars: pd.DataFrame, thresholds: pd.Series) -> list[Level]:
    """Every confirmed swing in `bars`, merged into levels by price.

    Walking zigzag_pivots' own sequence gives each pivot's reaction for
    free: the distance to the NEXT pivot in the alternating high/low
    sequence is exactly how far price moved away before turning again - no
    separate lookahead search needed, since that distance is already
    latent in the pivot list itself.
    """
    pivots = zigzag_pivots(bars, thresholds)
    levels: list[Level] = []

    def _price_at(idx: int, is_high: bool) -> float:
        return float(bars["high"].iloc[idx] if is_high else bars["low"].iloc[idx])

    for k, (idx, is_high) in enumerate(pivots):
        price = _price_at(idx, is_high)
        merge_tol = float(thresholds.iloc[idx]) * MERGE_ATR_MULTIPLE

        reaction_atr = 0.0
        if k + 1 < len(pivots):
            next_idx, next_is_high = pivots[k + 1]
            next_price = _price_at(next_idx, next_is_high)
            th = float(thresholds.iloc[idx])
            if th > 0:
                reaction_atr = abs(next_price - price) / th

        match = next(
            (i for i, lv in enumerate(levels) if lv.is_high == is_high and abs(lv.price - price) <= merge_tol),
            None,
        )
        if match is None:
            levels.append(Level(price=price, first_index=idx, is_high=is_high,
                                 touches=1, best_reaction_atr=reaction_atr))
        else:
            old = levels[match]
            levels[match] = Level(
                price=old.price,
                first_index=old.first_index,
                is_high=old.is_high,
                touches=old.touches + 1,
                best_reaction_atr=max(old.best_reaction_atr, reaction_atr),
            )

    return levels


def score_level(level: Level, as_of_index: int) -> float:
    """0.0 to 4.0 - the sum of four independently-capped [0,1] scores.
    Additive, not averaged: a level strong on every factor should read near
    the TOP of the range, not get pulled toward the middle by one weak
    dimension - that is what "combination of all of them" means here.
    """
    touches_score = min(level.touches, TOUCH_CAP) / TOUCH_CAP
    reaction_score = min(level.best_reaction_atr, REACTION_CAP_ATR) / REACTION_CAP_ATR

    age_days = max(as_of_index - level.first_index, 0)
    durability_score = min(age_days, DURABILITY_CAP_DAYS) / DURABILITY_CAP_DAYS

    half_step = ROUND_STEP / 2.0
    remainder = level.price % ROUND_STEP
    dist_to_round = min(remainder, ROUND_STEP - remainder)
    round_score = 1.0 - (dist_to_round / half_step)

    return touches_score + reaction_score + durability_score + round_score


def nearest_significant_level(
    levels: list[Level], price: float, direction: str, as_of_index: int, min_score: float
) -> Level | None:
    """The closest level AHEAD of price in `direction` whose score clears
    `min_score` - "ahead" mirrors structure.nearest_level_beyond's own
    convention: overhead for a long, below for a short. A level that exists
    but scores too low to count is skipped, not treated as absent - the
    caller only ever sees levels worth acting on.
    """
    candidates = [
        lv for lv in levels
        if (lv.price > price if direction == "long" else lv.price < price)
        and score_level(lv, as_of_index) >= min_score
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lv: abs(lv.price - price))


def level_held(bars: pd.DataFrame, level: Level, direction: str) -> bool:
    """Whether the LAST bar in `bars` is a rejection candle at `level`: one
    bar wicking through it and closing back on the side price came from -
    exactly ema_trend_v2.py's _touching() convention, mirrored rather than
    redefined. `direction` is long/short the same way daily_regime_read
    speaks it - "long" means testing a level from above (a support bounce),
    "short" means testing one from below (a resistance rejection). One bar
    is enough - Dror's Rule 5, "one rejection is enough."
    """
    last = bars.iloc[-1]
    if direction == "long":
        return bool(last["low"] <= level.price and last["close"] > level.price)
    return bool(last["high"] >= level.price and last["close"] < level.price)


def level_break_confirmed(bars: pd.DataFrame, level: Level, direction: str, confirm_bars: int) -> bool:
    """Whether the last `confirm_bars` closes have ALL stayed on the side
    of `level` that `direction` continues toward - not just the first bar
    that crossed it. `direction` is the direction the BREAK continues, the
    same sense daily_regime_read speaks: "long" means price is now above
    the level and staying there, "short" means below and staying there.

    An old, long-settled break trivially satisfies this (every recent close
    is on the far side regardless of how long ago the cross happened) - it
    only actually gates a FRESH break, which is the point: Dror's rule is
    "require the break to hold for a bar or two," not "forget every level
    price has ever passed."
    """
    if confirm_bars <= 0:
        return True
    recent = bars.iloc[-confirm_bars:]
    on_broken_side = (recent["close"] > level.price) if direction == "long" else (recent["close"] < level.price)
    return bool(on_broken_side.all())


def structure_trend(bars: pd.DataFrame, lookback: int = STRUCTURE_LOOKBACK) -> str | None:
    """"up", "down", or None - structure_metrics's last-3 higher-highs/
    higher-lows read (Rule 2) on the last `lookback` bars. Exposed
    standalone, not just inlined in daily_regime_read_v2, so the timeframe-
    combination methods (Rule 1) can read it independently on the daily AND
    the hourly chart."""
    window = bars.iloc[-lookback:] if lookback else bars
    return structure_metrics(window)["trend"]


def _apply_levels(
    trend: str | None,
    bars: pd.DataFrame,
    levels: list[Level],
    as_of_index: int,
    min_significance: float,
    break_confirm_bars: int,
) -> str | None:
    """The significant-levels block/reversal/break-confirm core (Rules 4-7),
    given an ALREADY-DETERMINED trend direction. Shared by
    daily_regime_read_v2 (trend and levels both from the same chart) and
    the timeframe-combination methods (trend from one chart, levels/timing
    from another) - one implementation of the levels logic, not two."""
    if trend is None:
        return None

    price = float(bars["close"].iloc[-1])
    direction = "long" if trend == "up" else "short"
    opposite = "short" if direction == "long" else "long"

    level = nearest_significant_level(levels, price, direction, as_of_index, min_significance)
    if level is None:
        behind = nearest_significant_level(levels, price, opposite, as_of_index, min_significance)
        if behind is not None and not level_break_confirmed(bars, behind, direction, break_confirm_bars):
            return None
        return trend

    if level_held(bars, level, opposite):
        return "up" if opposite == "long" else "down"
    return None


def daily_regime_read_v2(
    bars: pd.DataFrame,
    levels: list[Level],
    as_of_index: int,
    lookback: int = STRUCTURE_LOOKBACK,
    min_significance: float = 2.0,
    break_confirm_bars: int = 2,
) -> str | None:
    """The full rule set from the 2026-08-28/29 gate review, replacing
    daily_regime_read's break-of-structure detector and ad-hoc nearest-
    level search:

    - trend: structure_metrics's last-3 higher-highs/higher-lows read
      (Rule 2), on the last `lookback` bars (Rule 3, swept - not
      structure_context's growing search for an observed CHoCH).
    - `levels` is a caller-maintained, persistent list (build_levels/
      score_level) - NOT rebuilt here, since it accumulates over years and
      must not be reconstructed from scratch on every call.
    - No confirmed trend -> None, regardless of levels (a level cannot
      manufacture a trend that is not there).
    - A confirmed trend with no significant level ahead of price -> the
      trend reads through unmodified, UNLESS the level directly behind
      price was only just broken: that break itself must hold for
      `break_confirm_bars` closes (Rule 7, "require the break to hold for a
      bar or two") before the continuation is trusted -> None until then.
    - A significant level ahead of price, not yet rejected -> None (still
      testing it, do not trust the trend here).
    - A significant level ahead of price that JUST held (one rejection
      candle, level_held) -> the OPPOSITE direction, not merely unblocked -
      Dror's own example: a downtrend into a support that holds becomes a
      LONG at that level, not a blocked short.

    `min_significance` is the score_level cutoff for "counts as a level at
    all" - also swept, not a chosen constant (same discipline as `lookback`
    and the retired proximity_atr_multiple before it). `break_confirm_bars`
    is likewise a first value (2), not yet measured.
    """
    trend = structure_trend(bars, lookback)
    return _apply_levels(trend, bars, levels, as_of_index, min_significance, break_confirm_bars)


def mtf_regime_read_agree(
    daily_bars: pd.DataFrame,
    hourly_bars: pd.DataFrame,
    daily_levels: list[Level],
    as_of_index: int,
    lookback: int = STRUCTURE_LOOKBACK,
    min_significance: float = 2.0,
    break_confirm_bars: int = 2,
) -> str | None:
    """Rule 1, Method A: "combination of timeframes" as AGREEMENT - trust a
    direction only when the daily chart AND the 1H chart both show the same
    structure_trend. Disagreement (or no trend on either) -> None, even if
    daily_regime_read_v2 alone would have read through. Levels/timing stay
    on the daily chart ("significant levels of the btc daily") once the two
    timeframes agree on direction.
    """
    d_trend = structure_trend(daily_bars, lookback)
    h_trend = structure_trend(hourly_bars, lookback)
    if d_trend is None or d_trend != h_trend:
        return None
    return _apply_levels(d_trend, daily_bars, daily_levels, as_of_index, min_significance, break_confirm_bars)


def mtf_regime_read_timing(
    daily_bars: pd.DataFrame,
    hourly_bars: pd.DataFrame,
    hourly_levels: list[Level],
    as_of_index: int,
    lookback: int = STRUCTURE_LOOKBACK,
    min_significance: float = 2.0,
    break_confirm_bars: int = 2,
) -> str | None:
    """Rule 1, Method B: "combination of timeframes" as DIRECTION+TIMING -
    the daily chart is the ONLY source of direction (its own 1H trend is
    never consulted), but the entry-timing check (nearest significant level,
    held/broken, break-confirm) runs against the 1H chart's OWN levels and
    OWN bars instead of daily's - the daily chart says which way, the 1H
    chart says when.
    """
    d_trend = structure_trend(daily_bars, lookback)
    return _apply_levels(d_trend, hourly_bars, hourly_levels, as_of_index, min_significance, break_confirm_bars)
