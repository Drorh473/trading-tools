"""Chart-pattern detection, used to mark existing signals rather than to
generate its own.

That split is deliberate and measured. Traded standalone, inverse
head-and-shoulders showed no edge on any timeframe tested — 1H, 4H and daily,
at two pivot thresholds each. Every one of those five configurations turned
negative once its three largest winners were removed, which is the signature of
a handful of lucky trades rather than an edge. The cheatsheet says as much: it
puts the pattern at roughly a coin flip alone, and only worth trading in
combination with other confirmation.

Used as confirmation, it looks quite different. Strategy 1 longs that followed
a recent inverse H&S breakout measured +0.29R net against -0.2R for those that
did not, and the effect decays smoothly as the pattern gets older — vanishing
by 200 bars — which is the shape a real effect has rather than the erratic
jumping that noise produces. That evidence rests on 22 trades, so it is
suggestive rather than settled, and nothing here suppresses a signal: a missing
pattern costs an alert nothing.

Detection runs on 1H and 4H. Nine samples on 4H against seventeen on 1H cannot
say which timeframe is better, so both count and either one marks the signal.
"""

from dataclasses import dataclass

import pandas as pd

from notifier.strategies.indicators import atr
from notifier.strategies.structure import zigzag_pivots

ATR_PERIOD = 14
PIVOT_ATR_MULTIPLE = 4.0
# How unequal the two shoulders may be, as a share of the head's depth. Real
# patterns are never symmetric; too tight a tolerance finds nothing and too
# loose calls any three lows a pattern.
SHOULDER_TOLERANCE = 0.35
# How long after its breakout a pattern still counts as confirming a signal.
# Measured by bars rather than hours so it scales with the timeframe it was
# found on; the effect fades to nothing by roughly 200 bars either way.
CONFLUENCE_BARS = 50


@dataclass(frozen=True)
class Pattern:
    name: str
    direction: str  # the direction it argues for: "long" or "short"
    breakout_index: int
    bars_since_breakout: int
    neckline: float


def inverse_head_and_shoulders(bars: pd.DataFrame) -> list[Pattern]:
    """Bullish reversal: five alternating pivots whose middle low is deepest.

    The neckline is the higher of the two intervening peaks, and the pattern
    only counts once price has closed above it — per the cheatsheet, entry is
    on the break and never before it.
    """
    return _head_and_shoulders(bars, inverted=True)


def head_and_shoulders(bars: pd.DataFrame) -> list[Pattern]:
    """The bearish mirror: middle high is the tallest, break below the neckline."""
    return _head_and_shoulders(bars, inverted=False)


def _head_and_shoulders(bars: pd.DataFrame, inverted: bool) -> list[Pattern]:
    if len(bars) < ATR_PERIOD + 5:
        return []

    thresholds = atr(bars, ATR_PERIOD) * PIVOT_ATR_MULTIPLE
    pivots = zigzag_pivots(bars, thresholds)
    # Shoulders and head are lows for the inverted pattern, highs for the
    # upright one; the two intervening pivots form the neckline either way.
    want = [inverted is False] * 5
    want[1] = want[3] = not want[0]

    shoulder_col, neck_col = ("low", "high") if inverted else ("high", "low")
    found: list[Pattern] = []
    last = len(bars) - 1

    for i in range(len(pivots) - 4):
        seq = pivots[i : i + 5]
        if [kind for _, kind in seq] != want:
            continue
        left, peak_a, head, peak_b, right = [idx for idx, _ in seq]

        left_p = bars[shoulder_col].iloc[left]
        head_p = bars[shoulder_col].iloc[head]
        right_p = bars[shoulder_col].iloc[right]
        beyond = (head_p < left_p and head_p < right_p) if inverted else (head_p > left_p and head_p > right_p)
        if not beyond:
            continue  # the head must be the extreme of the three

        necks = (bars[neck_col].iloc[peak_a], bars[neck_col].iloc[peak_b])
        neckline = max(necks) if inverted else min(necks)
        depth = abs(neckline - head_p)
        if depth <= 0 or abs(left_p - right_p) > depth * SHOULDER_TOLERANCE:
            continue  # shoulders too lopsided to read as the pattern

        closes = bars["close"]
        broke = None
        for j in range(right + 1, len(bars)):
            if (closes.iloc[j] > neckline) if inverted else (closes.iloc[j] < neckline):
                broke = j
                break
        if broke is None:
            continue

        found.append(
            Pattern(
                name="inverse head-and-shoulders" if inverted else "head-and-shoulders",
                direction="long" if inverted else "short",
                breakout_index=broke,
                bars_since_breakout=last - broke,
                neckline=neckline,
            )
        )
    return found


def confluence(bars_by_timeframe: dict[str, pd.DataFrame], direction: str) -> str | None:
    """The name of a pattern arguing for `direction` that broke out recently
    and whose implied move hasn't since failed, or None. Timeframes are
    reported so the alert can say where it was seen."""
    for timeframe, bars in bars_by_timeframe.items():
        if bars is None or bars.empty:
            continue
        for detect in (inverse_head_and_shoulders, head_and_shoulders):
            for pattern in detect(bars):
                if pattern.direction != direction:
                    continue
                if pattern.bars_since_breakout > CONFLUENCE_BARS:
                    continue
                if _invalidated(bars, pattern):
                    continue
                return f"{pattern.name} on {timeframe}"
    return None


def _invalidated(bars: pd.DataFrame, pattern: Pattern) -> bool:
    """True once price has closed back through the neckline in the direction
    opposite the breakout.

    A recency window alone can't tell a pattern that still describes the
    current structure from one whose implied move already failed and
    reversed. An AEVOUSDT bearish breakout round-tripped 42% back above its
    own neckline within the same 50-bar recency window and traced out the
    opposite pattern on the way - yet was still cited as short confirmation.
    Checking every close since the breakout catches that directly instead of
    guessing at it from age alone.
    """
    closes = bars["close"].iloc[pattern.breakout_index + 1 :]
    if closes.empty:
        return False
    if pattern.direction == "long":
        return bool((closes < pattern.neckline).any())
    return bool((closes > pattern.neckline).any())
