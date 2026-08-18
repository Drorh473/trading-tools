"""The ONE trade scorer. Every expectancy number comes through here.

There were two. `generate_v2._walk` resolved a trade by finding the first bar
that touched the stop, the first that touched target 1 and the first that
touched target 2, then comparing those indices - and it used the ORIGINAL stop
for the whole trade. The live tracker moves the stop to breakeven the moment the
partial fills, so that scorer credited "both targets" to runners which had
retraced back through breakeven and recovered: trades the real system would
already have closed at zero.

A second scorer, written later to test runner variants, walked bar by bar with a
moving stop. On the same 5,723 trades the two disagreed by 0.06R - larger than
most of the effects being compared with them.

So: one implementation, tested against cases whose answers can be worked out by
hand. Two scorers cannot disagree if there is only one.

WHAT THIS IS NOT: a portfolio. No competition for capital, no aggregate cap, no
margin. Trades overlap freely, so t-statistics computed over these are
optimistic. It ranks RULE VARIANTS; it does not say what an account earns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAKER, TAKER = 0.0002, 0.0006
WALK_BARS = 200
PARTIAL = 0.5


@dataclass
class Scored:
    result: str  # stop | target1 then stop | both targets | runner open | unresolved
    bars: int
    r_gross: float
    r_net: float  # after fees and slippage


def simulate(
    bars: pd.DataFrame,
    start: int,
    signal,
    runner: str = "target",
    pivots: list | None = None,
    slippage: float = 0.0,
    walk_bars: int = WALK_BARS,
    fill_within: int | None = None,
) -> Scored:
    """One trade, bar by bar, with the stop that the live system would have.

    runner="target"  the remainder exits at signal.remainder_target
    runner="choch"   the remainder's stop becomes the last CONFIRMED swing low
                     (long) or high (short) and ratchets - `pivots` is
                     [(confirmed_at_bar, price, is_high)], each dated by when it
                     could first have been KNOWN, not when it printed.

    fill_within  None    the entry is already filled at `start` (a limit that
                         was resting BEFORE the trigger bar)
                 N       RETEST: the signal bar is the rejection candle, the
                         limit is placed after it closes, and the trade only
                         exists if a LATER bar within N trades back through the
                         level. Otherwise the result is "unfilled".

    The retest is what makes the rejection requirement honest. A candle is only
    known to have been rejected by the EMA9 once it CLOSES back on the trend
    side - so an order that fills during that same candle is filling on
    information it does not have yet. Dror, on three 15m setups: "the candle
    broke the ema9, it didn't get rejected - that is the most important
    condition." Having both the rejection and a fill at the level is what made
    an earlier cut of this strategy look 3,614x better than it was.

    Fees: the entry and the first target are resting limits (maker). A stop is a
    triggered market order (taker) and is the only leg charged slippage, since a
    resting limit cannot fill worse than its own price.
    """
    long = signal.direction == "long"
    entry, stop = signal.entry_price, signal.stop_loss
    risk = abs(entry - stop)
    if risk <= 0 or entry <= 0:
        return Scored("invalid", 0, 0.0, 0.0)
    sign = 1.0 if long else -1.0
    per_r = entry / risk  # convert a price fraction into R
    t1 = entry + signal.reward_risk_ratio * risk * sign
    t2 = signal.remainder_target

    gross = 0.0
    fees = MAKER * per_r  # entry, maker
    slip_r = slippage * per_r
    partialed = False
    cursor = 0
    if pivots:
        while cursor < len(pivots) and pivots[cursor][0] <= start:
            cursor += 1

    first = start
    if fill_within is not None:
        # The limit rests from the bar AFTER the rejection candle.
        probe = bars.iloc[start + 1 : start + 1 + fill_within]
        if not len(probe):
            return Scored("unfilled", 0, 0.0, 0.0)
        touched = (probe["low"].to_numpy() <= entry) if long else (probe["high"].to_numpy() >= entry)
        if not touched.any():
            return Scored("unfilled", len(probe), 0.0, 0.0)
        # walk from the fill bar inclusive: price can keep moving after it fills
        first = start + int(np.argmax(touched))

    window = bars.iloc[first + 1 : first + 1 + walk_bars]
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()
    closes = window["close"].to_numpy()

    for n in range(len(window)):
        # The stop is checked FIRST: a bar that touches both stop and target
        # resolves as the stop, the conservative reading of an ambiguous candle.
        if (lows[n] <= stop) if long else (highs[n] >= stop):
            share = 1.0 - PARTIAL if partialed else 1.0
            gross += share * ((stop - entry) * sign / risk - slip_r)
            fees += TAKER * per_r * share
            return Scored(
                "target1 then stop" if partialed else "stop", n + 1, gross, gross - fees
            )

        if not partialed:
            if (highs[n] >= t1) if long else (lows[n] <= t1):
                partialed = True
                gross += PARTIAL * signal.reward_risk_ratio
                fees += MAKER * per_r * PARTIAL
                stop = entry  # breakeven, the floor the runner starts from
            continue

        if runner == "target":
            if t2 is not None and ((highs[n] >= t2) if long else (lows[n] <= t2)):
                gross += (1.0 - PARTIAL) * (t2 - entry) * sign / risk
                fees += MAKER * per_r * (1.0 - PARTIAL)
                return Scored("both targets", n + 1, gross, gross - fees)
        else:
            while pivots and cursor < len(pivots) and pivots[cursor][0] <= start + 1 + n:
                _at, price, is_high = pivots[cursor]
                cursor += 1
                if long and not is_high:
                    stop = max(stop, price)
                elif not long and is_high:
                    stop = min(stop, price)

    if not len(window):
        return Scored("unresolved", 0, 0.0, 0.0)
    if partialed:
        # Still open at the end of the window: value the remainder where it
        # stands rather than crediting it a target it never reached.
        gross += (1.0 - PARTIAL) * ((closes[-1] - entry) * sign / risk - slip_r)
        fees += TAKER * per_r * (1.0 - PARTIAL)
        return Scored("runner open", len(window), gross, gross - fees)
    return Scored("unresolved", len(window), 0.0, 0.0)


def confirmed_pivots(frame: pd.DataFrame, atr_series: pd.Series, multiple: float = 1.25) -> list:
    """[(confirmed_at_bar, price, is_high)] for the CHoCH runner.

    A pivot is not a level until price has reversed away from it by the local
    threshold, and that happens on a LATER bar. Dating each pivot by when it
    could first have been known is what keeps the runner from sitting behind a
    level nobody could have seen yet.
    """
    from notifier.strategies.structure import zigzag_pivots

    w = frame.reset_index(drop=True)
    th = (atr_series * multiple).to_numpy()
    highs, lows = w["high"].to_numpy(), w["low"].to_numpy()
    out = []
    for i, is_high in zigzag_pivots(w, pd.Series(th)):
        t = th[i]
        if not np.isfinite(t) or t <= 0:
            continue
        later = slice(i + 1, None)
        reached = (lows[later] <= highs[i] - t) if is_high else (highs[later] >= lows[i] + t)
        if reached.any():
            out.append((i + 1 + int(np.argmax(reached)), float(highs[i] if is_high else lows[i]), bool(is_high)))
    out.sort()
    return out
