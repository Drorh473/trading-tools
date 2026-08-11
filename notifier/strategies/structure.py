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


def nearest_level_beyond(
    window: pd.DataFrame, thresholds: pd.Series, price: float, direction: str
) -> float | None:
    """The closest confirmed swing level ahead of `price`, or None.

    "Ahead" means in the direction the trade is going: something overhead for
    a long, something below for a short.

    BOTH pivot types count, whichever way the trade is going. A level is a
    level regardless of which side price approached it from - support that
    broke becomes resistance on the way back up, and that is not a nuance but
    the common case. Restricting a long to pivot HIGHS was the first version
    of this and it could not see SPCXUSDT's 147.24, a daily low tested
    repeatedly in June and July that Dror named immediately as the level the
    runner should aim at ("it was significant support ... so it is support in
    the daily too"). 121 hourly bars interact with that zone; the nearest
    pivot HIGH above price was a minor blip nobody would trade to.

    Fed daily bars in practice, even though the level is read off the 1H
    chart by eye: the nearest 1H pivot is frequently noise, while the daily
    pivot is the one that actually stops a runner. On SPCXUSDT the daily
    answer is 147.24 and the 1H answer was 136.21.
    """
    pivots = zigzag_pivots(window, thresholds)
    levels = [
        float(window["high"].iloc[i] if is_high else window["low"].iloc[i])
        for i, is_high in pivots
    ]
    if direction == "long":
        above = [p for p in levels if p > price]
        return min(above) if above else None
    below = [p for p in levels if p < price]
    return max(below) if below else None


DEFAULT_MIN_LOOKBACK = 200
DEFAULT_LOOKBACK_STEP = 50


@dataclass(frozen=True)
class TrendStructure:
    """Where the market currently is in break-of-structure terms.

    `anchor_index` is the swing the CURRENT trend began from - the high price
    was rejected from when a downtrend started, or the low it turned up from.
    `protected_index` is the level whose breach would end that trend.

    `choch_count` is how many genuine changes of character were OBSERVED in
    this window. Zero means the trend is nothing but the bootstrap guess -
    inferred from the two oldest pivots rather than from any turn the market
    actually made - and callers should treat it as no reading at all. See
    trend_structure for why that distinction is load-bearing.
    """

    trend: str | None  # "up", "down", or None while no structure has formed
    anchor_index: int | None
    protected_index: int | None
    choch_count: int = 0


def trend_structure(window: pd.DataFrame, thresholds: pd.Series) -> TrendStructure:
    """Track trend by break of structure, and report where it turned.

    PROTECTED IS FIXED. It is set once, at bootstrap or at a CHoCH, and holds
    until price actually trades through it - full stop. It does not move on
    ordinary continuation, however many smaller pullbacks form above it.

    An earlier version re-armed protected to the most recent opposite-type
    pivot on every new continuation high/low. On SAMSUNGUSDT that produced six
    separate CHoCH events in under a week, because each intervening pullback -
    169.86, then 164.20, then 156.02, none of them anywhere near the actual
    142.94 anchor - got treated as a fresh trend change on its own. Dror's
    correction: a level only stops protecting the trend when it is genuinely
    broken. Everything that happens above a 142.94 anchor without ever going
    below it is still the same uptrend, no matter how choppy the path there
    was. Confirmed against APTUSDT 4H, where the two versions disagreed on
    the trend itself, not just the anchor - re-arm read it as up, this reads
    it as down, and the fixed-protected read is the one that held up.

    This also means `last_high`/`last_low` are the only bookkeeping needed -
    they are what becomes the new anchor if a genuine CHoCH happens. There is
    no running "trend extreme" to re-arm against any more.
    """
    pivots = zigzag_pivots(window, thresholds)
    if len(pivots) < 3:
        return TrendStructure(None, None, None)

    trend: str | None = None
    anchor = protected = None
    last_high = last_low = None

    def price_at(idx: int, is_high: bool) -> float:
        return window["high"].iloc[idx] if is_high else window["low"].iloc[idx]

    choch_count = 0

    for idx, is_high in pivots:
        price = price_at(idx, is_high)

        if is_high:
            if trend == "down":
                if protected is not None and price > price_at(protected, True):
                    trend, anchor, protected = "up", last_low, last_low
                    choch_count += 1
            # Bootstrapping compares against the PREVIOUS high, not a running
            # extreme. On a chart that has been falling since the window
            # opened, the first high seen IS the running extreme, and
            # requiring a break of itself would leave the trend permanently
            # unset - no trend means no anchor at all.
            elif trend is None and last_high is not None and price > price_at(last_high, True):
                trend, anchor, protected = "up", last_low, last_low
            last_high = idx
        else:
            if trend == "up":
                if protected is not None and price < price_at(protected, False):
                    trend, anchor, protected = "down", last_high, last_high
                    choch_count += 1
            elif trend is None and last_low is not None and price < price_at(last_low, False):
                trend, anchor, protected = "down", last_high, last_high
            last_low = idx

    # A trend turns only on a CONFIRMED swing, never on price poking through a
    # level intraday. Reacting to the unconfirmed close was tried and reverted:
    # it read APTUSDT as having turned up because price closed 0.7% above a
    # minor bounce high at 0.5832, discarding a downtrend that was still the
    # operative structure. Lagging by one swing is the cost of not being
    # whipsawed by every marginal poke, and for anchoring a retracement the
    # established trend is what matters, not the one that might be forming.
    #
    # choch_count separates a trend that was OBSERVED to turn from one merely
    # assumed. The bootstrap above has to exist - without an initial trend no
    # CHoCH branch can ever fire - but it is a guess made from whichever two
    # pivots happen to be oldest in the window, and under the no-rearm rule it
    # then seeds `protected` for everything that follows. Measured on BTCUSDT
    # 1H, sweeping the lookback from 200 to 600 bars over identical price data
    # produced anywhere between 0 and 4 "genuine" CHoCHs and flipped the final
    # trend repeatedly (up up up up down down down up down). A caller that
    # ignores choch_count is reading the window's starting edge, not the
    # market.
    return TrendStructure(trend, anchor, protected, choch_count)


def structure_context(
    bars: pd.DataFrame,
    atr_multiple: float,
    atr_period: int = 14,
    min_lookback: int = DEFAULT_MIN_LOOKBACK,
    lookback_step: int = DEFAULT_LOOKBACK_STEP,
) -> tuple[pd.DataFrame, TrendStructure]:
    """The lookback window plus its break-of-structure read.

    THE TREND MUST HAVE BEEN OBSERVED TO TURN. A window whose read rests only
    on the bootstrap - no CHoCH within it - is discarded and the window grown,
    because that answer describes where the window happens to start rather
    than what the market did. If no window in the available history contains a
    turn, there is no reading and no trade.

    Dror's rule, after the fixed 200-bar window was measured: "require an
    observed choch and make the window larger as long there isnt one".

    Why a fixed lookback could not work, and why growing until a turn appears
    is not just a bigger fixed number: sweeping 200 to 600 bars over identical
    data, 43% of symbols FLIPPED trend direction, 27% flipped two or more
    times, and only 7% gave the same answer at both ends. BTCUSDT 1H ran
    up-up-up-up-down-down-down-up-down across that sweep. The answer was an
    artifact of the window edge, so no choice of edge fixes it - only refusing
    to answer without evidence does. At the old 200-bar setting 19% of
    symbol/timeframes had a trend resting on no observed turn at all.

    ATR is measured over the FULL history in every pass so the threshold is
    warmed up at each window's first bar, and a signal's threshold does not
    change just because the window grew.

    Lives here rather than inside one strategy because two now need the
    identical reading: Strategy 1 anchors its Fib retracement on it, and
    Strategy 4 takes both its direction gate and the dealing range whose
    midpoint separates premium from discount. Two copies would be two
    definitions of "what trend is this", free to drift apart silently.
    """
    thresholds = atr(bars, atr_period) * atr_multiple
    window = bars.iloc[-min_lookback:].reset_index(drop=True)

    # Clamped, or a history SHORTER than the floor produces an empty range and
    # the loop never runs at all - reporting "no trend" for data that may
    # contain a perfectly clear one. Strategy 1 could never reach that (it
    # needs 201 bars for its own trend MA before it asks); a caller with a
    # lower bar count can, and silently getting no reading is the worst
    # possible way to find out.
    start = min(min_lookback, len(bars))
    for lookback in range(start, len(bars) + lookback_step, lookback_step):
        lookback = min(lookback, len(bars))
        window = bars.iloc[-lookback:].reset_index(drop=True)
        structure = trend_structure(window, thresholds.iloc[-lookback:].reset_index(drop=True))
        if structure.choch_count > 0:
            return window, structure
        if lookback >= len(bars):
            break

    # Nothing in the available history turned. Report no trend rather than the
    # bootstrap's guess - the widest window's own window, so callers slicing
    # against it stay consistent.
    return window, TrendStructure(None, None, None)
