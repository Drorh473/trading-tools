"""Chart-pattern detection, used to mark existing signals rather than to
generate its own.

That split is deliberate and measured for the pattern built and tested first,
inverse head-and-shoulders: traded standalone it showed no edge on any
timeframe tested — 1H, 4H and daily, at two pivot thresholds each. Every one
of those five configurations turned negative once its three largest winners
were removed, which is the signature of a handful of lucky trades rather than
an edge. The cheatsheet says as much: it puts the pattern at roughly a coin
flip alone, and only worth trading in combination with other confirmation.

Used as confirmation, it looks quite different. Strategy 1 longs that followed
a recent inverse H&S breakout measured +0.29R net against -0.2R for those that
did not, and the effect decays smoothly as the pattern gets older — vanishing
by 200 bars — which is the shape a real effect has rather than the erratic
jumping that noise produces. That evidence rests on 22 trades, so it is
suggestive rather than settled, and nothing here suppresses a signal: a missing
pattern costs an alert nothing.

Flags/pennants, triangles/wedges, and cup-and-handle are built to the same
"confluence only" contract but have no measurement behind them yet - the
signal log (core/storage.py's signals table) and paper_sim now record and
resolve every signal a pattern marks, so their own edge (if any) becomes
measurable in the weekly report as it accumulates, the same way inverse H&S
was in the first place.

Detection runs on 1H and 4H. Nine samples on 4H against seventeen on 1H
couldn't say which timeframe was better for H&S, so both count and either one
marks the signal - the same treatment extends to every pattern here.
"""

from dataclasses import dataclass

import pandas as pd

from notifier.strategies.indicators import atr
from notifier.strategies.structure import zigzag_pivots

ATR_PERIOD = 14
PIVOT_ATR_MULTIPLE = 4.0
# How unequal the two shoulders (or cup rims) may be, as a share of the
# pattern's own depth. Real patterns are never perfectly symmetric; too tight
# a tolerance finds nothing and too loose calls any three lows a pattern.
SHOULDER_TOLERANCE = 0.35
# How far apart the two neck points may be, as a share of the pattern's depth.
# The neckline is only a level worth breaking because the market turned at the
# SAME place twice - that is what makes it defended support (resistance, for
# the inverted pattern). Taking one neck and ignoring the other, as this used
# to, accepts two turns at completely unrelated prices and calls the line
# between them major support, which it plainly is not.
NECKLINE_TOLERANCE = 0.35
# How long after its breakout a pattern still counts as confirming a signal.
# Measured by bars rather than hours so it scales with the timeframe it was
# found on; the effect fades to nothing by roughly 200 bars either way for H&S.
CONFLUENCE_BARS = 50


@dataclass(frozen=True)
class Pattern:
    name: str
    direction: str  # the direction it argues for: "long" or "short"
    breakout_index: int
    bars_since_breakout: int
    # The price that, if closed back through in the wrong direction, means
    # the move this pattern implied has already failed - a neckline for H&S,
    # the far side of the flag/handle for those, a projected trendline value
    # for triangles/wedges. Always a fixed price, not a moving line: taking
    # the trendline's value at the moment of breakout rather than continuing
    # to extrapolate it keeps confluence()'s invalidation check the same for
    # every pattern shape.
    invalidation_level: float


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

    # Measured from the candle BODY rather than the wick. A shoulder or head
    # has to be structure price spent time at, and a single candle's
    # excursion is not that - IDOLUSDT's "head" was one lone spike whose
    # wick alone defined the level.
    body_high = bars[["open", "close"]].max(axis=1)
    body_low = bars[["open", "close"]].min(axis=1)
    shoulder_px, neck_px = (body_low, body_high) if inverted else (body_high, body_low)
    found: list[Pattern] = []
    last = len(bars) - 1

    for i in range(len(pivots) - 4):
        seq = pivots[i : i + 5]
        if [kind for _, kind in seq] != want:
            continue
        left, peak_a, head, peak_b, right = [idx for idx, _ in seq]

        left_p = shoulder_px.iloc[left]
        head_p = shoulder_px.iloc[head]
        right_p = shoulder_px.iloc[right]
        beyond = (head_p < left_p and head_p < right_p) if inverted else (head_p > left_p and head_p > right_p)
        if not beyond:
            continue  # the head must be the extreme of the three

        neck_a, neck_b = neck_px.iloc[peak_a], neck_px.iloc[peak_b]
        if peak_b == peak_a:
            continue
        neck_slope = (neck_b - neck_a) / (peak_b - peak_a)
        depth = abs((neck_a + neck_b) / 2 - head_p)
        if depth <= 0 or abs(left_p - right_p) > depth * SHOULDER_TOLERANCE:
            continue  # shoulders too lopsided to read as the pattern
        if abs(neck_a - neck_b) > depth * NECKLINE_TOLERANCE:
            continue  # the two turns are at unrelated prices - no common level

        closes = bars["close"]
        broke = None
        for j in range(right + 1, len(bars)):
            # The neckline is the line through BOTH necks, so its value moves
            # with the bar being tested rather than sitting at whichever neck
            # happened to be the more extreme.
            level = neck_a + neck_slope * (j - peak_a)
            if (closes.iloc[j] > level) if inverted else (closes.iloc[j] < level):
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
                # Frozen at the value it held when the break happened, the same
                # rule triangles and wedges follow - see Pattern's docstring.
                invalidation_level=neck_a + neck_slope * (broke - peak_a),
            )
        )
    return found


# ---- flags / pennants ----

# Flags form fast on top of a sharp move, so pivots are read at a smaller
# scale than H&S's - a large threshold would absorb the whole pole and its
# flag into one leg and never see the shape.
FLAG_PIVOT_ATR_MULTIPLE = 2.5
FLAG_POLE_ATR_MULTIPLE = 4.0  # the pole itself must still be a genuinely sharp move
FLAG_POLE_MAX_BARS = 15  # a pole is a short burst, not a slow grind
FLAG_MIN_CONSOLIDATION_BARS = 3
FLAG_MAX_CONSOLIDATION_BARS = 20
FLAG_MAX_RETRACE = 0.5  # the flag can give back at most half the pole



def flag(bars: pd.DataFrame) -> list[Pattern]:
    """Continuation: a sharp pole, then a tight consolidation giving back at
    most half of it, then a breakout continuing the pole's own direction.

    Pennants (a converging rather than parallel consolidation) read the same
    way here - the direction and the up-to-half retracement are what the
    trade cares about, not whether the two boundaries are parallel."""
    if len(bars) < ATR_PERIOD + FLAG_POLE_MAX_BARS + FLAG_MIN_CONSOLIDATION_BARS:
        return []

    atr_series = atr(bars, ATR_PERIOD)
    pivots = zigzag_pivots(bars, atr_series * FLAG_PIVOT_ATR_MULTIPLE)
    last = len(bars) - 1
    found: list[Pattern] = []

    for i in range(len(pivots) - 1):
        pole_start, _ = pivots[i]
        pole_end, end_is_high = pivots[i + 1]
        bars_in_pole = pole_end - pole_start
        if not (0 < bars_in_pole <= FLAG_POLE_MAX_BARS):
            continue

        direction = "long" if end_is_high else "short"
        pole_top = bars["high"].iloc[pole_start : pole_end + 1].max()
        pole_bottom = bars["low"].iloc[pole_start : pole_end + 1].min()
        pole_range = pole_top - pole_bottom
        if pole_range <= 0 or pole_range < FLAG_POLE_ATR_MULTIPLE * atr_series.iloc[pole_start]:
            continue  # not a sharp enough move to read as a pole

        # The floor a long pole's flag may not close below before it breaks
        # out - not just at the consolidation window checked below, but at
        # every bar between there and the eventual breakout. Without this, a
        # pole that fully reverses and only much later stages an unrelated
        # rally back through the old consolidation's high reads as a flag
        # breakout, when what actually happened is the pole failed outright.
        retrace_floor = (
            pole_top - pole_range * FLAG_MAX_RETRACE if direction == "long" else pole_bottom + pole_range * FLAG_MAX_RETRACE
        )

        consolidation_limit = min(pole_end + FLAG_MAX_CONSOLIDATION_BARS, last)
        for cons_end in range(pole_end + FLAG_MIN_CONSOLIDATION_BARS, consolidation_limit + 1):
            window = bars.iloc[pole_end : cons_end + 1]
            cons_high, cons_low = window["high"].max(), window["low"].min()

            if direction == "long":
                retrace = (pole_top - cons_low) / pole_range
                breakout_level, invalidation_level = cons_high, cons_low
            else:
                retrace = (cons_high - pole_bottom) / pole_range
                breakout_level, invalidation_level = cons_low, cons_high

            if retrace > FLAG_MAX_RETRACE:
                break  # widening further only retraces more - not a flag

            broke = None
            for j in range(cons_end + 1, len(bars)):
                low_j, high_j, close_j = bars["low"].iloc[j], bars["high"].iloc[j], bars["close"].iloc[j]
                if (direction == "long" and low_j < retrace_floor) or (direction == "short" and high_j > retrace_floor):
                    break  # broke the flag's own floor before ever breaking out - this pole failed
                if (close_j > breakout_level) if direction == "long" else (close_j < breakout_level):
                    broke = j
                    break
            if broke is not None:
                found.append(
                    Pattern(
                        name="bull flag" if direction == "long" else "bear flag",
                        direction=direction,
                        breakout_index=broke,
                        bars_since_breakout=last - broke,
                        invalidation_level=invalidation_level,
                    )
                )
                break  # this pole is claimed; the next pivot pair gets its own chance
    return found


# ---- triangles / wedges ----

TRIANGLE_PIVOT_ATR_MULTIPLE = 2.5
TRIANGLE_MAX_LOOKBACK_BARS = 100
# A line whose two points differ by less than this many ATRs counts as flat
# rather than sloped - the same normalisation reason as everywhere else here:
# a small symbol and a large one should read an equally flat line the same way.
TRIANGLE_FLAT_ATR_MULTIPLE = 1.0
# A boundary must be evidenced by at least this many pivots. Two points define
# ANY line, so fitting through exactly two asserts the shape rather than
# finding it - which is how a pair of unrelated swings became a "triangle".
# Every textbook triangle and wedge shows three or more touches a side.
TRIANGLE_MIN_TOUCHES = 3
TRIANGLE_MAX_TOUCHES = 5  # beyond this the oldest touches describe a different structure
# How far a pivot may sit off its own fitted line, in ATRs, and still count as
# respecting it. Real touches are never exactly on the line.
TRIANGLE_FIT_TOLERANCE_ATR = 1.0
# The lines must actually close on each other: width at the end over width at
# the start. Measured on live data, genuine triangles came in at 0.11-0.43
# while a descending CHANNEL masquerading as a falling wedge sat at 0.79 (its
# apex 151 bars out, against a 40-bar pattern) and a weak rising wedge at
# 0.64. Nothing measured between 0.43 and 0.64, so the cut goes in that gap.
TRIANGLE_MAX_CONTRACTION = 0.55


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) through (bar index, price) points."""
    n = len(points)
    sx = sum(i for i, _ in points)
    sy = sum(p for _, p in points)
    sxx = sum(i * i for i, _ in points)
    sxy = sum(i * p for i, p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    return slope, (sy - slope * sx) / n


def _boundary(pivots: list[tuple[int, float]], atr_now: float) -> tuple[float, float, list] | None:
    """A line through at least TRIANGLE_MIN_TOUCHES recent pivots that all
    respect it, as (slope, intercept, the pivots used).

    Starts from the most recent TRIANGLE_MAX_TOUCHES and drops the oldest
    until the remainder fits, so a boundary that only straightened out
    recently is still found - without letting an old, unrelated swing drag the
    fit through price that has nothing to do with the current structure.
    """
    points = pivots[-TRIANGLE_MAX_TOUCHES:]
    while len(points) >= TRIANGLE_MIN_TOUCHES:
        fit = _fit_line(points)
        if fit is not None:
            slope, intercept = fit
            if all(abs(p - (slope * i + intercept)) <= TRIANGLE_FIT_TOLERANCE_ATR * atr_now for i, p in points):
                return slope, intercept, points
        points = points[1:]
    return None


def triangle_or_wedge(bars: pd.DataFrame) -> list[Pattern]:
    """Two trendlines fit through the two most recent pivot highs and the two
    most recent pivot lows, classified by each line's slope:

      flat top,    rising bottom -> ascending triangle (long)
      falling top, flat bottom   -> descending triangle (short)
      falling top, rising bottom -> symmetric triangle (either direction -
                                     whichever side actually breaks first)
      both rising, bottom steeper  -> rising wedge (short - a reversal)
      both falling, top steeper    -> falling wedge (long - a reversal)

    Direction is which way the pattern argues for on a breakout, not which
    way the lines point - the wedges are reversal patterns, so their
    breakout direction is opposite their own slope.
    """
    if len(bars) < ATR_PERIOD + 10:
        return []

    atr_series = atr(bars, ATR_PERIOD)
    pivots = zigzag_pivots(bars, atr_series * TRIANGLE_PIVOT_ATR_MULTIPLE)
    highs = [(idx, bars["high"].iloc[idx]) for idx, is_high in pivots if is_high]
    lows = [(idx, bars["low"].iloc[idx]) for idx, is_high in pivots if not is_high]
    if len(highs) < TRIANGLE_MIN_TOUCHES or len(lows) < TRIANGLE_MIN_TOUCHES:
        return []

    end = len(bars) - 1
    atr_now = atr_series.iloc[end]
    upper = _boundary(highs, atr_now)
    lower = _boundary(lows, atr_now)
    if upper is None or lower is None:
        return []  # neither boundary is evidenced by enough touches
    upper_slope, h1p, upper_pts = upper
    lower_slope, l1p, lower_pts = lower
    # Both fits are already in y = slope*j + intercept form, so the "first
    # point" the rest of this function extrapolates from is bar zero.
    h1 = l1 = 0
    h2, l2 = upper_pts[-1][0], lower_pts[-1][0]

    first = min(upper_pts[0][0], lower_pts[0][0])
    if end - first > TRIANGLE_MAX_LOOKBACK_BARS:
        return []
    width_first = (upper_slope * first + h1p) - (lower_slope * first + l1p)
    width_last = (upper_slope * end + h1p) - (lower_slope * end + l1p)
    if width_first <= 0 or width_last <= 0 or width_last / width_first > TRIANGLE_MAX_CONTRACTION:
        return []  # parallel channel, or diverging - the lines never close on each other

    upper_state = _slope_state(upper_slope * first + h1p, upper_slope * end + h1p, atr_now)
    lower_state = _slope_state(lower_slope * first + l1p, lower_slope * end + l1p, atr_now)

    if upper_state == 0 and lower_state == 1:
        candidates = [("ascending triangle", "long")]
    elif upper_state == -1 and lower_state == 0:
        candidates = [("descending triangle", "short")]
    elif upper_state == -1 and lower_state == 1:
        candidates = [("symmetric triangle", "long"), ("symmetric triangle", "short")]
    elif upper_state == 1 and lower_state == 1 and lower_slope > upper_slope:
        candidates = [("rising wedge", "short")]
    elif upper_state == -1 and lower_state == -1 and upper_slope < lower_slope:
        candidates = [("falling wedge", "long")]
    else:
        return []  # parallel channel or diverging lines - not this family

    last = len(bars) - 1
    start = max(h2, l2)
    found: list[Pattern] = []
    for name, direction in candidates:
        for j in range(start + 1, len(bars)):
            close = bars["close"].iloc[j]
            upper_at_j = h1p + upper_slope * (j - h1)
            lower_at_j = l1p + lower_slope * (j - l1)
            if direction == "long" and close > upper_at_j:
                found.append(
                    Pattern(name=name, direction="long", breakout_index=j, bars_since_breakout=last - j, invalidation_level=lower_at_j)
                )
                break
            if direction == "short" and close < lower_at_j:
                found.append(
                    Pattern(name=name, direction="short", breakout_index=j, bars_since_breakout=last - j, invalidation_level=upper_at_j)
                )
                break
    return found


def _slope_state(p1: float, p2: float, atr_now: float) -> int:
    if abs(p2 - p1) < TRIANGLE_FLAT_ATR_MULTIPLE * atr_now:
        return 0
    return 1 if p2 > p1 else -1


# ---- cup and handle ----

CUP_MIN_BARS = 30
CUP_MAX_BARS = 150
CUP_RIM_TOLERANCE = 0.05  # the two rims within 5% of their average price
CUP_ROUNDING_SEGMENTS = 6  # split the cup into this many pieces to check the U-shape
HANDLE_MIN_BARS = 3
HANDLE_MAX_BARS = 30
HANDLE_MAX_RETRACE = 0.5  # the handle can give back at most half the cup's depth


def cup_and_handle(bars: pd.DataFrame) -> list[Pattern]:
    """Bullish: a rounded U-shaped base between two comparable rims (the
    cup), then a shallow pullback (the handle), then a break above the rim.

    The left rim is a confirmed zigzag pivot - the decline into the cup is
    a real swing, so that part is safe to demand. The right rim is not: it is
    simply the first bar whose high comes back within tolerance of the left
    rim's price. Requiring it to be its own confirmed pivot would mean
    requiring whatever comes after it (the handle) to already be a big enough
    reversal to register - which rules out exactly the shallow, realistic
    handles this pattern is supposed to allow.

    Rounding is checked cheaply rather than curve-fit: split the cup into
    CUP_ROUNDING_SEGMENTS equal pieces and require each segment's low to fall
    through the first half and rise through the second half. A single sharp
    spike (a V, not a U) fails this, since its neighbouring segments won't
    keep falling/rising around it the way a genuine rounded base does.
    """
    if len(bars) < ATR_PERIOD + CUP_MIN_BARS:
        return []

    pivots = zigzag_pivots(bars, atr(bars, ATR_PERIOD) * PIVOT_ATR_MULTIPLE)
    left_rims = [idx for idx, is_high in pivots if is_high]
    last = len(bars) - 1
    found: list[Pattern] = []

    for left_rim in left_rims:
        left_p = bars["high"].iloc[left_rim]
        search_limit = min(left_rim + CUP_MAX_BARS, last)

        right_rim = None
        for candidate in range(left_rim + CUP_MIN_BARS, search_limit + 1):
            if abs(bars["high"].iloc[candidate] - left_p) <= left_p * CUP_RIM_TOLERANCE:
                right_rim = candidate
                break
        if right_rim is None:
            continue

        right_p = bars["high"].iloc[right_rim]
        rim = (left_p + right_p) / 2
        cup_lows = bars["low"].iloc[left_rim : right_rim + 1]
        depth = rim - cup_lows.min()
        if depth <= 0 or not _is_rounded(cup_lows):
            continue

        handle_limit = min(right_rim + HANDLE_MAX_BARS, last)
        for handle_end in range(right_rim + HANDLE_MIN_BARS, handle_limit + 1):
            handle_low = bars["low"].iloc[right_rim : handle_end + 1].min()
            if rim - handle_low > depth * HANDLE_MAX_RETRACE:
                break  # the handle is already deeper than a handle should be

            broke = None
            for j in range(handle_end + 1, len(bars)):
                if bars["close"].iloc[j] > rim:
                    broke = j
                    break
            if broke is not None:
                found.append(
                    Pattern(
                        name="cup-and-handle",
                        direction="long",
                        breakout_index=broke,
                        bars_since_breakout=last - broke,
                        invalidation_level=handle_low,
                    )
                )
                break
    return found


def _is_rounded(lows: pd.Series) -> bool:
    n = len(lows)
    if n < CUP_ROUNDING_SEGMENTS * 2:
        return False
    edges = [round(n * k / CUP_ROUNDING_SEGMENTS) for k in range(CUP_ROUNDING_SEGMENTS + 1)]
    segment_mins = [lows.iloc[edges[k] : edges[k + 1]].min() for k in range(CUP_ROUNDING_SEGMENTS)]
    mid = CUP_ROUNDING_SEGMENTS // 2
    falling = all(segment_mins[k] >= segment_mins[k + 1] for k in range(mid))
    rising = all(segment_mins[k] <= segment_mins[k + 1] for k in range(mid, CUP_ROUNDING_SEGMENTS - 1))
    return falling and rising


# ---- confluence ----

_DETECTORS = (inverse_head_and_shoulders, head_and_shoulders, flag, triangle_or_wedge, cup_and_handle)


def confluence(bars_by_timeframe: dict[str, pd.DataFrame], direction: str) -> str | None:
    """The name of a pattern arguing for `direction` that broke out recently
    and whose implied move hasn't since failed, or None. Timeframes are
    reported so the alert can say where it was seen."""
    for timeframe, bars in bars_by_timeframe.items():
        if bars is None or bars.empty:
            continue
        for detect in _DETECTORS:
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
    """True once price has closed back through the invalidation level in the
    direction opposite the breakout.

    A recency window alone can't tell a pattern that still describes the
    current structure from one whose implied move already failed and
    reversed. An AEVOUSDT bearish H&S breakout round-tripped 42% back above
    its own neckline within the same 50-bar recency window and traced out the
    opposite pattern on the way - yet was still cited as short confirmation.
    Checking every close since the breakout catches that directly instead of
    guessing at it from age alone, for any pattern shape.
    """
    closes = bars["close"].iloc[pattern.breakout_index + 1 :]
    if closes.empty:
        return False
    if pattern.direction == "long":
        return bool((closes < pattern.invalidation_level).any())
    return bool((closes > pattern.invalidation_level).any())


# ---- pending (not yet broken) patterns ----
#
# Everything above finds patterns that have ALREADY broken out - a Pattern is
# only ever constructed once `broke is not None`. That is the right contract
# for confluence(), which asks "did something recently confirm this signal?",
# but it makes an unresolved structure invisible: the flag sitting in front of
# price right now, which is exactly what a staged entry needs to see.
#
# Hence a parallel path rather than a change to the existing one. A
# PendingPattern is the same structure caught one step earlier: found, intact,
# and NOT yet broken. It carries the level that would constitute the break
# rather than a breakout index, since by definition there isn't one.
#
# Levels are re-derived from fresh bars on every check rather than stored and
# trusted. For flags, H&S and cups that is merely tidy - their levels are
# fixed prices. For triangles and wedges it is the whole point: their boundary
# is a converging trendline sitting at a different price every bar, so a level
# frozen at alert time stops describing the pattern within hours. drift_per_bar
# is carried so an alert can say which way the level is moving instead of
# quoting a number that silently goes stale.


@dataclass(frozen=True)
class PendingPattern:
    name: str
    direction: str  # what it will argue for IF it breaks
    # A close beyond this, in `direction`, is the breakout. For sloping
    # patterns this is the line's value as of the last bar supplied.
    break_level: float
    # A close through this the other way means the structure is gone and the
    # setup should stop being watched. Distinct from Pattern.invalidation_level,
    # which asks whether an already-implied move has failed; here nothing has
    # been implied yet, so this is the level that destroys the shape itself.
    invalidation_level: float
    # How break_level moves per bar. Zero for fixed levels; non-zero for
    # triangles, wedges, and a sloping head-and-shoulders neckline.
    drift_per_bar: float = 0.0
    # The same for invalidation_level, which for a triangle or wedge is the
    # OPPOSITE trendline and slopes just as much. Carrying only the break
    # level's slope left the other boundary looking horizontal, which is wrong
    # wherever that level is used as a price - the stop the add-on moves to,
    # for one.
    invalidation_drift_per_bar: float = 0.0


def pending_inverse_head_and_shoulders(bars: pd.DataFrame) -> list[PendingPattern]:
    return _pending_head_and_shoulders(bars, inverted=True)


def pending_head_and_shoulders(bars: pd.DataFrame) -> list[PendingPattern]:
    return _pending_head_and_shoulders(bars, inverted=False)


def _pending_head_and_shoulders(bars: pd.DataFrame, inverted: bool) -> list[PendingPattern]:
    """The five-pivot shape complete but the neckline not yet taken out.

    Invalidation is the head rather than the neckline: the neckline is the
    level we're waiting to break, so it cannot also be the level that kills
    the setup. Price closing beyond the head is what destroys the shape.
    """
    if len(bars) < ATR_PERIOD + 5:
        return []

    thresholds = atr(bars, ATR_PERIOD) * PIVOT_ATR_MULTIPLE
    pivots = zigzag_pivots(bars, thresholds)
    want = [inverted is False] * 5
    want[1] = want[3] = not want[0]

    # Measured from the candle BODY rather than the wick. A shoulder or head
    # has to be structure price spent time at, and a single candle's
    # excursion is not that - IDOLUSDT's "head" was one lone spike whose
    # wick alone defined the level.
    body_high = bars[["open", "close"]].max(axis=1)
    body_low = bars[["open", "close"]].min(axis=1)
    shoulder_px, neck_px = (body_low, body_high) if inverted else (body_high, body_low)
    closes = bars["close"]
    found: list[PendingPattern] = []

    for i in range(len(pivots) - 4):
        seq = pivots[i : i + 5]
        if [kind for _, kind in seq] != want:
            continue
        left, peak_a, head, peak_b, right = [idx for idx, _ in seq]

        left_p = shoulder_px.iloc[left]
        head_p = shoulder_px.iloc[head]
        right_p = shoulder_px.iloc[right]
        beyond = (head_p < left_p and head_p < right_p) if inverted else (head_p > left_p and head_p > right_p)
        if not beyond:
            continue

        neck_a, neck_b = neck_px.iloc[peak_a], neck_px.iloc[peak_b]
        if peak_b == peak_a:
            continue
        neck_slope = (neck_b - neck_a) / (peak_b - peak_a)
        depth = abs((neck_a + neck_b) / 2 - head_p)
        if depth <= 0 or abs(left_p - right_p) > depth * SHOULDER_TOLERANCE:
            continue
        if abs(neck_a - neck_b) > depth * NECKLINE_TOLERANCE:
            continue  # the two turns are at unrelated prices - no common level

        last_bar = len(bars) - 1
        if last_bar <= right:
            continue

        def neck_at(j: int, _a=neck_a, _s=neck_slope, _p=peak_a) -> float:
            return _a + _s * (j - _p)

        broke = dead = False
        for j in range(right + 1, len(bars)):
            close_j = closes.iloc[j]
            if (close_j > neck_at(j)) if inverted else (close_j < neck_at(j)):
                broke = True
                break
            if (close_j < head_p) if inverted else (close_j > head_p):
                dead = True
                break
        if broke:
            continue  # already broken - that belongs to the Pattern path above
        if dead:
            continue  # price went through the head; the shape is gone

        found.append(
            PendingPattern(
                name="inverse head-and-shoulders" if inverted else "head-and-shoulders",
                direction="long" if inverted else "short",
                # The neckline through both necks, valued at the last bar, and
                # carrying its slope so a caller can say which way it moves -
                # the same treatment triangles and wedges get.
                break_level=float(neck_at(last_bar)),
                invalidation_level=float(head_p),
                drift_per_bar=float(neck_slope),
            )
        )
    return found


def pending_flag(bars: pd.DataFrame) -> list[PendingPattern]:
    """A pole, then a consolidation that is STILL FORMING - it runs to the
    last bar rather than to a breakout.

    The break level is the consolidation's own running high (low, for a bear
    flag), so no break can have happened by construction: a close can never
    exceed the highest high of a window that includes it.
    """
    if len(bars) < ATR_PERIOD + FLAG_POLE_MAX_BARS + FLAG_MIN_CONSOLIDATION_BARS:
        return []

    atr_series = atr(bars, ATR_PERIOD)
    pivots = zigzag_pivots(bars, atr_series * FLAG_PIVOT_ATR_MULTIPLE)
    last = len(bars) - 1
    found: list[PendingPattern] = []

    for i in range(len(pivots) - 1):
        pole_start, _ = pivots[i]
        pole_end, end_is_high = pivots[i + 1]
        bars_in_pole = pole_end - pole_start
        if not (0 < bars_in_pole <= FLAG_POLE_MAX_BARS):
            continue

        # The consolidation has to still be running right now, and be of a
        # length that reads as a flag rather than a stall.
        cons_bars = last - pole_end
        if not (FLAG_MIN_CONSOLIDATION_BARS <= cons_bars <= FLAG_MAX_CONSOLIDATION_BARS):
            continue

        direction = "long" if end_is_high else "short"
        pole_top = bars["high"].iloc[pole_start : pole_end + 1].max()
        pole_bottom = bars["low"].iloc[pole_start : pole_end + 1].min()
        pole_range = pole_top - pole_bottom
        if pole_range <= 0 or pole_range < FLAG_POLE_ATR_MULTIPLE * atr_series.iloc[pole_start]:
            continue

        window = bars.iloc[pole_end : last + 1]
        cons_high, cons_low = window["high"].max(), window["low"].min()

        if direction == "long":
            retrace = (pole_top - cons_low) / pole_range
            break_level, invalidation_level = cons_high, cons_low
        else:
            retrace = (cons_high - pole_bottom) / pole_range
            break_level, invalidation_level = cons_low, cons_high

        if retrace > FLAG_MAX_RETRACE:
            continue  # given back more than half the pole - not a flag any more

        found.append(
            PendingPattern(
                name="bull flag" if direction == "long" else "bear flag",
                direction=direction,
                break_level=float(break_level),
                invalidation_level=float(invalidation_level),
            )
        )
    return found


def pending_triangle_or_wedge(bars: pd.DataFrame) -> list[PendingPattern]:
    """The same two fitted trendlines as triangle_or_wedge, with neither yet
    taken out. break_level is the line's value AT THE LAST BAR, and
    drift_per_bar is its slope, so a caller can both test the break correctly
    now and describe which way the level is moving."""
    if len(bars) < ATR_PERIOD + 10:
        return []

    atr_series = atr(bars, ATR_PERIOD)
    pivots = zigzag_pivots(bars, atr_series * TRIANGLE_PIVOT_ATR_MULTIPLE)
    highs = [(idx, bars["high"].iloc[idx]) for idx, is_high in pivots if is_high]
    lows = [(idx, bars["low"].iloc[idx]) for idx, is_high in pivots if not is_high]
    if len(highs) < TRIANGLE_MIN_TOUCHES or len(lows) < TRIANGLE_MIN_TOUCHES:
        return []

    end = len(bars) - 1
    atr_now = atr_series.iloc[end]
    upper = _boundary(highs, atr_now)
    lower = _boundary(lows, atr_now)
    if upper is None or lower is None:
        return []  # neither boundary is evidenced by enough touches
    upper_slope, h1p, upper_pts = upper
    lower_slope, l1p, lower_pts = lower
    # Both fits are already in y = slope*j + intercept form, so the "first
    # point" the rest of this function extrapolates from is bar zero.
    h1 = l1 = 0
    h2, l2 = upper_pts[-1][0], lower_pts[-1][0]

    first = min(upper_pts[0][0], lower_pts[0][0])
    if end - first > TRIANGLE_MAX_LOOKBACK_BARS:
        return []
    width_first = (upper_slope * first + h1p) - (lower_slope * first + l1p)
    width_last = (upper_slope * end + h1p) - (lower_slope * end + l1p)
    if width_first <= 0 or width_last <= 0 or width_last / width_first > TRIANGLE_MAX_CONTRACTION:
        return []  # parallel channel, or diverging - the lines never close on each other

    upper_state = _slope_state(upper_slope * first + h1p, upper_slope * end + h1p, atr_now)
    lower_state = _slope_state(lower_slope * first + l1p, lower_slope * end + l1p, atr_now)

    if upper_state == 0 and lower_state == 1:
        candidates = [("ascending triangle", "long")]
    elif upper_state == -1 and lower_state == 0:
        candidates = [("descending triangle", "short")]
    elif upper_state == -1 and lower_state == 1:
        candidates = [("symmetric triangle", "long"), ("symmetric triangle", "short")]
    elif upper_state == 1 and lower_state == 1 and lower_slope > upper_slope:
        candidates = [("rising wedge", "short")]
    elif upper_state == -1 and lower_state == -1 and upper_slope < lower_slope:
        candidates = [("falling wedge", "long")]
    else:
        return []

    last = len(bars) - 1
    start = max(h2, l2)
    if last <= start:
        return []

    def upper_at(j: int) -> float:
        return h1p + upper_slope * (j - h1)

    def lower_at(j: int) -> float:
        return l1p + lower_slope * (j - l1)

    found: list[PendingPattern] = []
    for name, direction in candidates:
        broken = False
        for j in range(start + 1, len(bars)):
            close = bars["close"].iloc[j]
            # Either line giving way ends this candidate: through the break
            # side it is no longer pending, through the far side the shape is
            # gone.
            if close > upper_at(j) or close < lower_at(j):
                broken = True
                break
        if broken:
            continue

        if direction == "long":
            break_level, invalidation_level = upper_at(last), lower_at(last)
            drift, inv_drift = upper_slope, lower_slope
        else:
            break_level, invalidation_level = lower_at(last), upper_at(last)
            drift, inv_drift = lower_slope, upper_slope

        found.append(
            PendingPattern(
                name=name,
                direction=direction,
                break_level=float(break_level),
                invalidation_level=float(invalidation_level),
                drift_per_bar=float(drift),
                # Both boundaries of a triangle slope; neither is horizontal.
                invalidation_drift_per_bar=float(inv_drift),
            )
        )
    return found


def pending_cup_and_handle(bars: pd.DataFrame) -> list[PendingPattern]:
    """Cup complete and the handle still forming, with the rim not yet taken
    out. The handle runs to the last bar rather than to a breakout."""
    if len(bars) < ATR_PERIOD + CUP_MIN_BARS:
        return []

    pivots = zigzag_pivots(bars, atr(bars, ATR_PERIOD) * PIVOT_ATR_MULTIPLE)
    left_rims = [idx for idx, is_high in pivots if is_high]
    last = len(bars) - 1
    found: list[PendingPattern] = []

    for left_rim in left_rims:
        left_p = bars["high"].iloc[left_rim]
        search_limit = min(left_rim + CUP_MAX_BARS, last)

        right_rim = None
        for candidate in range(left_rim + CUP_MIN_BARS, search_limit + 1):
            if abs(bars["high"].iloc[candidate] - left_p) <= left_p * CUP_RIM_TOLERANCE:
                right_rim = candidate
                break
        if right_rim is None:
            continue

        right_p = bars["high"].iloc[right_rim]
        rim = (left_p + right_p) / 2
        cup_lows = bars["low"].iloc[left_rim : right_rim + 1]
        depth = rim - cup_lows.min()
        if depth <= 0 or not _is_rounded(cup_lows):
            continue

        handle_bars = last - right_rim
        if not (HANDLE_MIN_BARS <= handle_bars <= HANDLE_MAX_BARS):
            continue

        handle_low = bars["low"].iloc[right_rim : last + 1].min()
        if rim - handle_low > depth * HANDLE_MAX_RETRACE:
            continue  # deeper than a handle should be

        if (bars["close"].iloc[right_rim + 1 :] > rim).any():
            continue  # the rim already gave way

        found.append(
            PendingPattern(
                name="cup-and-handle",
                direction="long",
                break_level=float(rim),
                invalidation_level=float(handle_low),
            )
        )
    return found


_PENDING_DETECTORS = (
    pending_inverse_head_and_shoulders,
    pending_head_and_shoulders,
    pending_flag,
    pending_triangle_or_wedge,
    pending_cup_and_handle,
)


def pending(bars_by_timeframe: dict[str, pd.DataFrame], direction: str) -> tuple[PendingPattern, str] | None:
    """The unbroken pattern arguing for `direction` sitting in front of price
    right now, with the timeframe it was seen on, or None.

    Mirrors confluence()'s shape and search order deliberately, so the two
    read the same way at the call site - the only difference being that this
    one finds what hasn't happened yet.
    """
    for timeframe, bars in bars_by_timeframe.items():
        if bars is None or bars.empty:
            continue
        for detect in _PENDING_DETECTORS:
            for pattern in detect(bars):
                if pattern.direction != direction:
                    continue
                return pattern, timeframe
    return None
