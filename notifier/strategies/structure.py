"""Swing structure shared by anything that needs to know where a move began.

Strategy 1 uses these pivots to anchor a Fibonacci retracement; pattern
detection uses the same ones to recognise shapes. Both need the same answer to
"what counts as a swing here?", so the threshold lives in one place rather than
being re-derived per caller.
"""

from dataclasses import dataclass

import pandas as pd

from notifier.strategies.indicators import atr


def zigzag_pivots(window: pd.DataFrame, thresholds: pd.Series) -> list[tuple[int, bool]]:
    """Indices of swing pivots, oldest first, as (index, is_high).

    An extreme is only promoted to a pivot once price reverses away from it by
    at least the local threshold, so a crash or gap terminates the leg before
    it instead of being absorbed into one giant leg spanning both price
    regimes. Two guards matter: the reversal must land on a *later* bar than
    the extreme (otherwise one wide candle confirms itself off its own wick),
    and the window's first bar is never a pivot (we can't see what came before
    it, so it is a boundary artifact rather than observed structure).

    `thresholds` is indexed alongside `window` so the bar being tested supplies
    its own threshold - typically a multiple of ATR, which keeps the definition
    of a swing proportional to how much the symbol normally moves.
    """
    pivots: list[tuple[int, bool]] = []
    falling = None
    high_idx, high_price = 0, window["high"].iloc[0]
    low_idx, low_price = 0, window["low"].iloc[0]

    for i in range(1, len(window)):
        high, low = window["high"].iloc[i], window["low"].iloc[i]
        threshold = thresholds.iloc[i]

        if falling is not True:
            if high >= high_price:
                high_idx, high_price = i, high
            elif i > high_idx and high_price - low >= threshold:
                if high_idx > 0:
                    pivots.append((high_idx, True))
                falling = True
                low_idx, low_price = i, low
                continue

        if falling is not False:
            if low <= low_price:
                low_idx, low_price = i, low
            elif i > low_idx and high - low_price >= threshold:
                if low_idx > 0:
                    pivots.append((low_idx, False))
                falling = False
                high_idx, high_price = i, high

    return pivots


@dataclass(frozen=True)
class TrendStructure:
    """Where the market currently is in break-of-structure terms.

    `anchor_index` is the swing the CURRENT trend began from - the high price
    was rejected from when a downtrend started, or the low it turned up from.
    `protected_index` is the level whose breach would end that trend.
    """

    trend: str | None  # "up", "down", or None while no structure has formed
    anchor_index: int | None
    protected_index: int | None


def trend_structure(window: pd.DataFrame, thresholds: pd.Series) -> TrendStructure:
    """Track trend by break of structure, and report where it turned.

    A trend continues while price keeps breaking the running extreme in its own
    direction (BOS). It ends only when the PROTECTED level breaks - the swing
    standing at the last such break - which is a change of character, and the
    anchor moves there.

    Two things this gets right that "take the most recent pivot" does not:

    Comparing against the trend's RUNNING extreme rather than the previous
    pivot. On AAPLUSDT 1H the lows after 300.55 were 306.52, 308.01, 303.08 and
    302.04. Measured pairwise the last two look like fresh breaks; measured
    against the downtrend's actual low of 300.55 none of them broke anything.
    Treating them as breaks decayed the protected high from 316.11 to 307.93,
    and a 313.82 print then read as a trend change when the trend was intact.

    Reaching PAST intervening swings. The anchor stays at the high the trend
    turned from even when smaller highs have formed since, which is what makes
    it immune to the pivot threshold in a way the old rule was not: a minor
    high cannot become the anchor just by being the most recent thing found.
    """
    pivots = zigzag_pivots(window, thresholds)
    if len(pivots) < 3:
        return TrendStructure(None, None, None)

    trend: str | None = None
    anchor = protected = None
    last_high = last_low = None
    trend_high = trend_low = None  # the running extremes of the current trend

    def price_at(idx: int, is_high: bool) -> float:
        return window["high"].iloc[idx] if is_high else window["low"].iloc[idx]

    for idx, is_high in pivots:
        price = price_at(idx, is_high)

        if is_high:
            if trend == "down":
                if protected is not None and price > price_at(protected, True):
                    trend, anchor, protected = "up", last_low, last_low
                    trend_high = trend_low = None
            elif trend == "up":
                if trend_high is not None and price > price_at(trend_high, True):
                    protected = last_low  # each continuation re-arms the guard
            # Bootstrapping compares against the PREVIOUS high, not the running
            # one. The running extreme is set by the first pivot in the window,
            # which is frequently the widest, so requiring a break of it would
            # leave the trend permanently unset on any chart that has been
            # falling since - and no trend means no anchor at all.
            elif trend is None and last_high is not None and price > price_at(last_high, True):
                trend, anchor, protected = "up", last_low, last_low
            if trend_high is None or price > price_at(trend_high, True):
                trend_high = idx
            last_high = idx
        else:
            if trend == "up":
                if protected is not None and price < price_at(protected, False):
                    trend, anchor, protected = "down", last_high, last_high
                    trend_high = trend_low = None
            elif trend == "down":
                if trend_low is not None and price < price_at(trend_low, False):
                    protected = last_high
            elif trend is None and last_low is not None and price < price_at(last_low, False):
                trend, anchor, protected = "down", last_high, last_high
            if trend_low is None or price < price_at(trend_low, False):
                trend_low = idx
            last_low = idx

    # A trend turns only on a CONFIRMED swing, never on price poking through a
    # level intraday. Reacting to the unconfirmed close was tried and reverted:
    # it read APTUSDT as having turned up because price closed 0.7% above a
    # minor bounce high at 0.5832, discarding a downtrend that was still the
    # operative structure. Lagging by one swing is the cost of not being
    # whipsawed by every marginal poke, and for anchoring a retracement the
    # established trend is what matters, not the one that might be forming.
    return TrendStructure(trend, anchor, protected)
