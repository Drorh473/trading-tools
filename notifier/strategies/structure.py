"""Swing structure shared by anything that needs to know where a move began.

Strategy 1 uses these pivots to anchor a Fibonacci retracement; pattern
detection uses the same ones to recognise shapes. Both need the same answer to
"what counts as a swing here?", so the threshold lives in one place rather than
being re-derived per caller.
"""

import pandas as pd


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
