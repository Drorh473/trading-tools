"""Strategy 3 from the user's cheatsheet ("volume run"): a daily consolidation
whose volume dries up, breaking out of the top of its range.

The thesis is about supply. A level is worth trading through only if the market
actually defended it, so both boundaries of the range must have been printed on
raised volume - the top on an outright spike, since that is the level the trade
breaks. Volume then falling away *inside* the range means the sellers who made
that level have stopped showing up. The break runs into whatever vacuum is left
above, which is why "no overhead resistance" is part of the setup rather than a
nicety, and why the cheatsheet notes these cluster near all-time highs: there,
by construction, nothing overhead exists at all.

The uptrend gate is deliberately only the slow structure - price above SMA200
and EMA50 above SMA200 - and NOT the four-MA stack Strategy 2 uses. Measured on
synthetic consolidations, EMA9 > EMA20 holds for every length up to ~30 days and
then starts flipping: a 90-day coil satisfies it on roughly three days in four,
essentially at random depending where the oscillation sits. Since evaluate()
only ever sees today, that would silently discard long consolidations on most
days - and the cheatsheet says the longer the consolidation, the better the
trade. The slow structure held on every length tested.

The range is the nearest pivot high ABOVE price and the nearest pivot low BELOW
it, not simply the two most recent pivots. In an uptrend price is frequently
already above its last confirmed pivot high, so taking that blindly finds a
"range" price has left; bracketing finds the one it is actually inside. That
single change took the detector from firing roughly twice a year to a workable
rate.

Two versions. BOTH read the consolidation off DAILY bars: the cheatsheet
identifies it on the daily chart in each case, and only the TRIGGER differs.
The swing version breaks out on a 1H close, takes 75% at 1:2 and runs the
rest to daily resistance or three trading days, whichever comes first. The
day version breaks out on a 5m close and closes FLAT at 1:2 - the day sheet
names one exit and there is no runner behind it.

An earlier build read the day version's entire structure off hourly bars -
range, spike, dry-up and resistance all on the 1H chart - which neither sheet
asks for. That mistake bred its own repairs: hourly ATR is inflated by the
very move that forms a range, so an absolute width ceiling was bolted on to
contain spans the ATR test waved through, and a minimum breakout penetration
was added after a graze counted as a break. With the structure back on daily
bars the width ceiling is moot; the penetration floor is kept, because a 5m
close can still graze a daily level by a hair.

What legitimately differs per version is the minimum pause. The swing sheet
is silent, so the measured 20-bar floor stands; the day sheet says outright
that the consolidation "can be just a few single days", so the day version
drops to the shortest coil the tests can actually be computed on. Everything
else - the trend gate, the volume rules, the 1:2 reward - is shared, because
the sheets share it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, sma
from notifier.strategies.structure import structure_context, zigzag_pivots

TREND_MA_PERIOD = 200
# Weeks in the fallback long-term average, used only where the 200-day SMA has
# not warmed up. Eight rather than a rounder ten: ALCHUSDT, a reference setup
# from the rebuild that follows, had exactly 8 completed weekly bars at its
# level - 10 was unreachable for it.
LONG_TERM_TREND_WEEKS = 8
# RULE 1 (uptrend before the level): how sensitive break-of-structure is to a
# swing, in ATR. Matches what Strategies 1 and 4 already use for the same
# reading.
BOS_ATR_MULTIPLE = 2.0
# RULE 1's own fallback, used ONLY when structure_context returns no verdict
# at all (too little history for any observed change of character) - never
# when it reads "down". See structural_uptrend.
FALLBACK_RALLY_ATR = 4.5
# Bars looked back, from a candidate box's own start, for the fallback rally
# above AND for the unconditional rally floor below.
RALLY_LOOKBACK_BARS = 20
# RULE 1's unconditional floor: the move into the level must be a real
# rally, whatever the uptrend read decides. Both reference setups measured
# 7.8 and 4.4 ATR into their levels; this is deliberately looser than that.
MIN_RALLY_INTO_LEVEL_ATR = 3.5
ATR_PERIOD = 14
STOP_ATR_BUFFER = 1.0  # the stop sits a full ATR below the low, never on it
REWARD_RISK_RATIO = 2.0
# THE TWO SHEETS ANCHOR THE STOP DIFFERENTLY, so this is per-instance.
#
#   swing: "stop below the last low of the breakout candle"
#   day:   "stop below the last low before the breakout"
#
# The code used the day rule for both, which was Dror's earlier call made
# before the sheets were transcribed - his reasoning then was that the breakout
# bar's low "is wherever that particular candle happened to open from", while
# the last low before the break is a level the market actually turned at. On
# the sheets being read back he chose the sheets. The day version is unchanged
# either way; only the swing instance moves.
STOP_AT_BREAKOUT_CANDLE = "breakout_candle"
STOP_AT_RECENT_LOW = "recent_low"
# The two sheets exit differently, so this is per-instance rather than a
# module constant. Swing: 75% off at 1:2, the rest runs to daily resistance or
# the three-day clock. Day: "profit at a 1:2 ratio" and nothing after it, so
# the whole position closes there - the same flat-exit idiom Strategy 4 uses.
SWING_PARTIAL_FRACTION = 0.75
DAY_PARTIAL_FRACTION = 1.0


@dataclass(frozen=True)
class ConsolidationParams:
    """What genuinely differs between the swing and day instances now that
    box-shape rules (1-5, below) are shared between them. Almost everything
    else moved to module constants: Dror's five rules make no swing/day
    distinction on shape, only on trigger speed, exit, and stop anchor - which
    VolumeRun itself carries."""

    volume_baseline_bars: int = 30  # median window each volume ratio is judged against
    pivot_atr_multiple: float = 3.0  # threshold for the resistance-pivot search (rule 2)
    zigzag_lookback: int = 300  # bars considered when searching for resistance pivots
    # How far beyond the level a close must be to count as a breakout.
    # Without it the line is the box's own high and merely grazing it
    # qualifies: TSLAUSDT triggered 0.012% past it - four cents on a $324
    # stock - and again ten minutes later at 0.006%. The swing instance's 1H
    # trigger has not needed this; the day instance's 5m trigger has.
    min_penetration_atr: float = 0.0


SWING_PARAMS = ConsolidationParams(min_penetration_atr=0.0)
DAY_PARAMS = ConsolidationParams(min_penetration_atr=0.10)

# --- Rule 3: the box itself -------------------------------------------------
MIN_BOX_BARS = 10
MAX_BOX_BARS = 60
# Widest top-to-bottom span still called a coil, in ATR. Both reference
# setups measured 1.7 and 2.2 ATR; this is deliberately looser than that, not
# tuned to the two data points.
MAX_BOX_ATR = 4.0
# How close a bar's high must sit to the box's own maximum to count as "at
# the level" - used to anchor the box's start to the bar that actually set
# the ceiling.
BOX_LEVEL_TOLERANCE_ATR = 0.5
# The ceiling must hold this many bars before it breaks, or it was a fresh
# high rather than a level the market paused at.
CEILING_HOLD_BARS = 5
# The bar that sets the ceiling must itself carry real volume, or the "level"
# is a wick nobody defended.
CEILING_MIN_VOLUME_RATIO = 0.75
# R-squared above which a rising coil is read as still trending rather than
# pausing.
MAX_COIL_UP_DRIFT_R2 = 0.5
# How much of the box's own height a downward drift may give back before it
# stops being "sideways".
MAX_COIL_DOWN_DRIFT_SHARE = 0.40
# --- Rule 4: volume declining through the box -------------------------------
# Late half of the box against its early half. <= 1.0 is "not rising".
COIL_LATE_EARLY_VOLUME_MAX = 1.0
# --- Rule 5: volume rising on the break --------------------------------------
# Measured on the ENTRY TIMEFRAME's own closed trigger bar against its own
# rolling median in evaluate() - see evaluate()'s own comment for why that
# differs from how it was validated.
BREAKOUT_VOLUME_MIN = 1.3


@dataclass(frozen=True)
class Consolidation:
    top: float
    bottom: float
    top_index: int
    bottom_index: int
    started_at: int  # the later of the two boundary pivots
    pivot_highs: tuple[int, ...]


class VolumeRun(Strategy):
    """A daily consolidation, breaking out on entry_timeframe.

    Not swept across scales the way Strategies 1 and 2 are. Both cheatsheet
    versions read the consolidation off the DAILY chart and differ only in
    trigger and exit: 1D/1H taking 75% with a runner, 1D/5m closing flat at
    1:2. trend_timeframe stays a parameter because the tests exercise the
    detector directly, not because a non-daily instance is intended.
    """

    def __init__(
        self,
        trend_timeframe: str = "1D",
        entry_timeframe: str = "1H",
        time_exit_days: int | None = 3,
        armed_only: bool = False,
        params: ConsolidationParams = SWING_PARAMS,
        session_gated: bool = False,
        partial_fraction: float = SWING_PARTIAL_FRACTION,
        stop_anchor: str = STOP_AT_BREAKOUT_CANDLE,
    ):
        self.trend_timeframe = trend_timeframe
        self.entry_timeframe = entry_timeframe
        self.tag = f"Strategy 3 {trend_timeframe}/{entry_timeframe}"
        self.timeframes = [trend_timeframe, entry_timeframe]
        self.time_exit_days = time_exit_days
        self.params = params
        # 1.0 means the whole position leaves at the 1:2 target and there is no
        # runner to manage - the day sheet's only exit. Anything less opens one.
        self.partial_fraction = partial_fraction
        if stop_anchor not in (STOP_AT_BREAKOUT_CANDLE, STOP_AT_RECENT_LOW):
            raise ValueError(f"unknown stop_anchor {stop_anchor!r}")
        self.stop_anchor = stop_anchor
        # Intraday instances read volume and structure off bars that assume
        # a market which is actually trading; a daily bar spans a whole
        # session, so the question does not arise for the swing version.
        self.session_gated = session_gated
        # The 5m version polls per-symbol instead of watchlist-wide; see
        # Strategy.armed_timeframes.
        self.armed_timeframes = (entry_timeframe,) if armed_only else ()

    def min_daily_bars(self) -> int:
        """Shortest daily history this instance can read a setup from.

        Sized for the NARROWEST box find_consolidation will try
        (MIN_BOX_BARS), not the widest - the search already narrows its own
        upper bound to whatever history is actually available
        (`max_len = min(MAX_BOX_BARS, n - baseline)`), so gating on
        MAX_BOX_BARS here would refuse a symbol that has enough history for
        every box it might actually find, just not for the widest
        conceivable one. ALCHUSDT, one of the two reference setups, has 76
        days of history at its box start and a MAX_BOX_BARS-based floor
        would have refused it outright.
        """
        return self.params.volume_baseline_bars + MIN_BOX_BARS + RALLY_LOOKBACK_BARS + 10

    def arms(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> bool:
        """Worth polling once this symbol has a live consolidation at all.

        There used to be a second condition - price within ARMING_BAND (0.10) of
        the range top, "close enough to be worth 5m polling". It was reasoned
        rather than measured, and measurement killed it: across 62,353 daily
        bars spanning 2021-2026, on 313 small-cap coins and 195 majors, it armed
        ONCE. Price never got closer than 0.79 of the way up on the small caps.

        The band assumed price creeps up to the level before clearing it. It
        does not. On the eight bars that were actually followed by a break, the
        position was:

            min 0.153   p25 0.385   median 0.612   max 0.770

        - one of them from the bottom sixth of its own range. Price jumps from a
        standing start, so position carries almost no information about whether
        a break is coming, and every candidate band either admitted nothing or
        admitted everything: 0.10 caught 0 of 8 breaks, 0.35 caught 4, and only
        "no band at all" caught all eight.

        What remains is still a real filter, because a qualifying consolidation
        is itself rare - 1.9% of small-cap symbol-days, 0.35% of majors. That is
        under two of the 100 watchlist symbols on an average day, so the 5m poll
        stays around 2,200 fetches a day against the bot's current ~3,100.
        Arming on price INSTEAD, with no band, would have armed the whole
        watchlist: 100 x 4 timeframes x 288 polls = 115,200 a day, on an API
        that already answers bursts with 429.
        """
        daily = bars_by_timeframe.get(self.trend_timeframe)
        if daily is None or len(daily) < self.min_daily_bars():
            return False
        setup = find_consolidation(daily, self.params)
        # bool(), because top and bottom are numpy floats and their comparison
        # returns np.bool_ - which is truthy but is not True, and this method
        # advertises `-> bool`.
        return setup is not None and bool(setup.top > setup.bottom)

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        daily = bars_by_timeframe.get(self.trend_timeframe)
        entry_bars = bars_by_timeframe.get(self.entry_timeframe)
        if daily is None or entry_bars is None:
            return None
        # The entry frame needs enough history for both the ATR call and
        # _volume_ratio's baseline window (rule 5) - the wider of the two.
        if len(daily) < self.min_daily_bars() or len(entry_bars) < max(
            ATR_PERIOD + 2, self.params.volume_baseline_bars + 1
        ):
            return None

        setup = find_consolidation(daily, self.params)
        if setup is None:
            return None

        # The breakout is the FIRST close above the range. Without that, every
        # later candle still sitting above the level re-fires the same trade -
        # the shape of bug that sent one stale TSLAUSDT short four times.
        closes = entry_bars["close"]
        close_now, close_prev = closes.iloc[-1], closes.iloc[-2]
        atr_now = atr(entry_bars, ATR_PERIOD).iloc[-1]

        # The breakout has to clear the level by a margin, not merely touch
        # it. The level is the pivot bar's own HIGH, so without this a close
        # a fraction of a tick above a wick counts - and then counts again
        # every time price wobbles back across it.
        threshold = setup.top + atr_now * self.params.min_penetration_atr
        if not (close_now > threshold and close_prev <= threshold):
            return None

        # RULE 5: volume must be rising on the break itself. Validated
        # against DAILY close-day volume in the backtest (BTCUSDT 1.54x,
        # ALCHUSDT 2.94x the 30-day median) - applied here to the ENTRY
        # TIMEFRAME's own closed trigger bar instead, against ITS OWN
        # rolling median, because find_consolidation runs on the STRUCTURE
        # frame, which excludes the still-forming daily bar by design: the
        # daily close for "today" does not exist yet when the 1H/5m trigger
        # fires intraday. Same idea, a different timeframe's volume
        # distribution, and not separately calibrated - watch this once live.
        entry_break_vol = _volume_ratio(
            entry_bars["base_vol"], len(entry_bars) - 1, self.params.volume_baseline_bars
        )
        if entry_break_vol < BREAKOUT_VOLUME_MIN:
            return None

        entry = close_now
        # Each sheet names its own anchor; see STOP_AT_BREAKOUT_CANDLE.
        if self.stop_anchor == STOP_AT_BREAKOUT_CANDLE:
            anchor = float(entry_bars["low"].iloc[-1])
        else:
            anchor = _recent_low_before(entry_bars["low"], len(entry_bars) - 1)
        if anchor is None:
            return None
        # "Below" the low, not on it: a stop resting exactly at the low is
        # taken out by any wick that merely matches it. Neither sheet gives a
        # distance, so this keeps the buffer the strategy has always used.
        stop = anchor - atr_now * STOP_ATR_BUFFER
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * REWARD_RISK_RATIO

        highs = daily["high"]
        # Anything the market already turned back from, sitting between the
        # break and the target, is what stops this trade reaching it.
        if any(setup.top < highs.iloc[i] < target for i in setup.pivot_highs):
            return None

        overhead = [highs.iloc[i] for i in setup.pivot_highs if highs.iloc[i] >= target]

        notes = []
        # A full-size exit at the target has no remainder to place or describe.
        # Guarding on the fraction rather than on the instance keeps the two
        # exit models in one place: whoever sets partial_fraction=1.0 gets the
        # flat exit, and there is no second flag to keep in step with it.
        runs_a_remainder = self.partial_fraction < 1.0
        if not runs_a_remainder:
            remainder_target, remainder_note = None, ""
        else:
            # The runner exits at the next level above the target; when there
            # is none the setup is at the highs and the exit is a rule, not a
            # price.
            remainder_target = min(overhead) if overhead else None
            if remainder_target is not None:
                remainder_note = "daily resistance"
            elif self.time_exit_days:
                remainder_note = f"after {self.time_exit_days} trading days"
            else:
                remainder_note = "at your discretion"
            if remainder_target is not None and self.time_exit_days:
                notes.append(
                    f"Close the runner after {self.time_exit_days} trading days if resistance is not reached first."
                )

        # Trailing applies to BOTH versions: each sheet ends on the same rule,
        # and a flat 1:2 exit still wants its stop dragged up on the way there.
        if not overhead:
            notes.append("At all-time highs: trail the stop up under each rising low.")

        return Signal(
            symbol=symbol,
            direction="long",
            entry_price=entry,
            stop_loss=stop,
            strategy_tag=self.tag,
            # The range top is what this setup IS, so the range is claimed
            # once however many times price crosses back over it.
            dedupe_key=(symbol, self.tag, "long", round(setup.top, 10)),
            reward_risk_ratio=REWARD_RISK_RATIO,
            partial_fraction=self.partial_fraction,
            remainder_target=remainder_target,
            remainder_note=remainder_note,
            extra_notes=tuple(notes),
            reason=(
                f"Daily consolidation between {setup.bottom:.8g} and {setup.top:.8g} lasting "
                f"{len(daily) - 1 - setup.started_at} days, ceiling held for at least {CEILING_HOLD_BARS} bars "
                f"on real volume, volume declining through the box. Price closed above the level on the "
                f"{self.entry_timeframe} with volume {entry_break_vol:.2f}x its median. Stop is a full ATR below "
                + ("the breakout candle's low." if self.stop_anchor == STOP_AT_BREAKOUT_CANDLE
                   else "the last low before the breakout.")
            ),
        )


def structural_uptrend(
    daily: pd.DataFrame, start: int, closes: pd.Series, levels: pd.Series, atr_series: pd.Series
) -> bool:
    """RULE 1: was the market in an uptrend before the candle at `start`?

    Two independent tests, both against history strictly BEFORE `start` - the
    market's state going into the level, not during whatever follows it.

      - price above its long-term average (levels[start], from trend_levels)
      - break-of-structure reads "up": two confirmed higher highs and higher
        lows, with an OBSERVED change of character - not merely the bootstrap
        guess from wherever the lookback window happens to start. A plain
        monotonic climb only ever produces the bootstrap reading, which
        structure_context itself refuses to credit as "up" (choch_count stays
        0), so this is not a redundant restatement of "price went up".

    Structure has its OWN fallback, used only when it returns NO verdict at
    all (too little history for any confirmed change of character): a rally
    of FALLBACK_RALLY_ATR into the level stands in. A verdict of "down" is
    never overridden - only "no verdict" is.
    """
    if start < 30 or not in_uptrend_at(closes, levels, start):
        return False

    try:
        _window, structure = structure_context(
            daily.iloc[:start].reset_index(drop=True),
            atr_multiple=BOS_ATR_MULTIPLE,
            min_lookback=min(200, start),
        )
    except Exception:
        return False

    if structure.trend == "up":
        return True
    if structure.trend == "down":
        return False

    # No verdict at all - too little history for structure to have formed.
    av = atr_series.iloc[start]
    if av <= 0:
        return False
    lookback_low = daily["low"].iloc[max(0, start - RALLY_LOOKBACK_BARS) : start]
    lookback_high = daily["high"].iloc[max(0, start - RALLY_LOOKBACK_BARS) : start]
    if lookback_low.empty:
        return False
    rally = (lookback_high.max() - lookback_low.min()) / av
    return bool(rally >= FALLBACK_RALLY_ATR)


def find_consolidation(daily: pd.DataFrame, params: ConsolidationParams = SWING_PARAMS) -> Consolidation | None:
    """The box price is CURRENTLY inside, if one qualifies under the five
    rules - or None. `daily` is the STRUCTURE frame: its last row is the most
    recently CLOSED daily bar, one behind whatever the live entry-timeframe
    trigger is doing intraday. A box is only returned if price, as of that
    last bar, is still inside it - the box must not already have broken out
    in the daily frame itself, which is what evaluate()'s own trigger check
    is for.

    Among every box length that satisfies every rule, the WIDEST one wins
    (the search runs shortest-to-longest and keeps overwriting).
    """
    closes, volumes = daily["close"], daily["base_vol"]
    highs, lows = daily["high"], daily["low"]
    n = len(daily)
    last = n - 1

    atr_series = atr(daily, ATR_PERIOD)
    levels = trend_levels(daily)

    # Resistance pivots (rule 2) are computed once, independent of which box
    # (if any) is found.
    zz_start = max(0, n - 1 - params.zigzag_lookback)
    zz_window = daily.iloc[zz_start:]
    zz_thresholds = (atr_series * params.pivot_atr_multiple).iloc[zz_start:]
    pivot_highs = tuple(
        i + zz_start for i, is_high in zigzag_pivots(zz_window, zz_thresholds) if is_high
    )

    atr_now = atr_series.iloc[last]
    if atr_now <= 0:
        return None

    baseline = params.volume_baseline_bars
    max_len = min(MAX_BOX_BARS, n - baseline)
    if max_len < MIN_BOX_BARS:
        return None

    best: Consolidation | None = None
    for box_len in range(MIN_BOX_BARS, max_len + 1):
        start = n - box_len
        window_high = highs.iloc[start:n]
        window_low = lows.iloc[start:n]
        ceiling = float(window_high.max())
        floor = float(window_low.min())
        box_height = ceiling - floor
        if box_height <= 0 or box_height / atr_now > MAX_BOX_ATR:
            continue

        # RULE 3a: the box must START where price first reaches the level it
        # later breaks - not drift into a ceiling made late inside a wider
        # window.
        if highs.iloc[start] < ceiling - BOX_LEVEL_TOLERANCE_ATR * atr_now:
            continue

        # THERE IS NO SEPARATE "PRICE STILL INSIDE THE BOX" CHECK HERE, and
        # that is deliberate, not an oversight - an earlier version of this
        # function carried one (`floor <= closes.iloc[last] <= ceiling`) and
        # it was dead code: ceiling and floor are the max/min of THIS SAME
        # window, which always includes bar `last` itself, so the close of
        # that bar is algebraically guaranteed to sit inside [floor, ceiling]
        # for every box_len tried. It could never once evaluate False.
        #
        # This module's PREDECESSOR (the impulse-candle detector this one
        # replaced) genuinely needed such a check: its range was anchored to
        # a SPECIFIC, PAST impulse candle and never re-examined against where
        # price currently stood, so a symbol that had since collapsed still
        # carried a "valid", long-stale range - which is why Strategy 3
        # produced zero signals across two live instances in its entire
        # life. That vulnerability cannot exist here: this function has no
        # persisted state, and every box is re-derived fresh, this call,
        # from a window that ends at the current bar. A collapse wide enough
        # to matter is caught by the width cap above instead (verified
        # directly: ADAUSDT's real numbers - price 5.3 range-widths below a
        # 2025-08-14 range - blow MAX_BOX_ATR on their own, before any
        # separate "still inside" check would even run).
        ceiling_index = start + int(window_high.to_numpy().argmax())

        # RULE 3b: the ceiling must HOLD before it breaks, or it is a fresh
        # high, not a level the market paused at.
        if last - ceiling_index < CEILING_HOLD_BARS:
            continue

        # RULE 4 (the level itself): the bar that sets the ceiling must carry
        # real volume, or the "level" is a wick nobody defended.
        ceiling_vol = _volume_ratio(volumes, ceiling_index, baseline)
        if ceiling_vol < CEILING_MIN_VOLUME_RATIO:
            continue

        # RULE 4 (through the box): volume must not be rising, late half
        # against early half. The ceiling bar's own elevated volume is
        # INCLUDED in the early half deliberately.
        box_volumes = volumes.iloc[start:n]
        half = box_len // 2
        if half == 0:
            continue
        early_vol = box_volumes.iloc[:half].mean()
        late_vol = box_volumes.iloc[half:].mean()
        if not early_vol or early_vol <= 0 or late_vol / early_vol > COIL_LATE_EARLY_VOLUME_MAX:
            continue

        # RULE 3c: drift. A coil may slope down (giving back some of the
        # rally is what a pause looks like) but not climb - a steady rise is
        # still the move, not a break from it.
        r_squared, slope = _coil_fit(closes.iloc[start:n])
        total_drift = slope * (box_len - 1)
        if total_drift > 0 and r_squared > MAX_COIL_UP_DRIFT_R2:
            continue
        if total_drift < 0 and abs(total_drift) / box_height > MAX_COIL_DOWN_DRIFT_SHARE:
            continue

        # RULE 1 (unconditional floor): the move into the level must be a
        # real rally, whatever the uptrend read below decides.
        rally_low_window = lows.iloc[max(0, start - RALLY_LOOKBACK_BARS) : start]
        atr_at_start = atr_series.iloc[start]
        if (
            rally_low_window.empty
            or atr_at_start <= 0
            or (ceiling - rally_low_window.min()) / atr_at_start < MIN_RALLY_INTO_LEVEL_ATR
        ):
            continue

        # RULE 1: the market must have been in a structural uptrend before
        # this box started.
        if not structural_uptrend(daily, start, closes, levels, atr_series):
            continue

        best = Consolidation(
            top=ceiling,
            bottom=floor,
            top_index=ceiling_index,
            bottom_index=start + int(window_low.to_numpy().argmin()),
            started_at=start,
            pivot_highs=pivot_highs,
        )

    return best


def _recent_low_before(lows, index: int, lookback: int = 30) -> float | None:
    """The most recent low the market actually turned at, before `index`.

    A local minimum rather than the lowest low in a window: the stop belongs
    under the last place buyers stepped in, not under whatever the deepest
    point of the last thirty bars happens to be, which on a long coil can sit
    far below anything currently relevant.
    """
    first = max(1, index - lookback)
    for i in range(index - 1, first, -1):
        if float(lows.iloc[i]) < float(lows.iloc[i - 1]) and float(lows.iloc[i]) <= float(lows.iloc[i + 1]):
            return float(lows.iloc[i])
    window = lows.iloc[first : index + 1]
    return float(window.min()) if not window.empty else None


def weekly_trend_levels(daily: pd.DataFrame, weeks: int = LONG_TERM_TREND_WEEKS) -> pd.Series:
    """For each daily bar, the mean of the last `weeks` COMPLETED weekly closes.

    Derived by resampling the daily frame rather than fetched: "1W" is not in
    TIMEFRAME_SECONDS, and adding a timeframe would mean new plumbing through
    the scanner's fetch loop and bar cache for a number that is already implied
    by the bars in hand.

    Only weeks that had CLOSED by a given daily bar count toward its level, so
    nothing here reads the future. The bar's own close is then compared against
    that level by the caller - which is the whole point, and the trap the first
    version of this fell into. Comparing the PREVIOUS WEEK'S close instead
    makes the test structurally blind to the impulse: the impulse candle is the
    move, so the week before it is the week before the move started. Measured,
    that rejected BNBUSDT (impulse closed 704, prior week 602.8), UNIUSDT (8.98
    vs 6.964) and XLMUSDT (0.252 vs 0.1512) - three real setups, and no choice
    of `weeks` fixed any of them.
    """
    ts = pd.to_datetime(daily["ts"], unit="ms")
    weekly_close = daily.assign(_ts=ts).set_index("_ts")["close"].resample("W").last().dropna()
    if weekly_close.empty:
        return pd.Series(np.nan, index=daily.index)

    means = weekly_close.rolling(weeks).mean().to_numpy()
    # Each weekly bar is labelled by the instant it closes, so a daily bar may
    # only use weeks strictly before it: searchsorted "left" minus one.
    position = np.searchsorted(weekly_close.index.to_numpy(), ts.to_numpy(), side="left") - 1
    levels = np.full(len(daily), np.nan)
    known = position >= 0
    levels[known] = means[position[known]]
    return pd.Series(levels, index=daily.index)


def trend_levels(daily: pd.DataFrame) -> pd.Series:
    """What each bar's close must beat to count as an uptrend.

    The 200-day average where it has warmed up, and the 10-week average where
    it has not. Deliberately a FALLBACK rather than a replacement: on every bar
    that has 200 days behind it the answer is exactly what it always was, so
    this cannot change any existing setup - it only supplies an answer where
    the old code had none and refused the symbol outright.

    That refusal was the binding constraint on coverage. 30 of the 100
    watchlist symbols have never had 200 daily bars, so Strategy 3 could not
    signal on them at all; the tokenized equities are mostly in that group.
    """
    daily_level = sma(daily["close"], TREND_MA_PERIOD)
    return daily_level.where(daily_level.notna(), weekly_trend_levels(daily))


def in_uptrend_at(closes: pd.Series, levels: pd.Series, index: int) -> bool:
    """Whether the market was in an uptrend at `index`: close above its trend
    level. `levels` comes from trend_levels - the 200-day average, or the
    10-week one where that has not warmed up.

    Deliberately ONLY the price-against-average test, with no second
    moving-average confirmation. This is asked about the impulse candle, and
    that candle is frequently the move that STARTS the trend, at which point a
    faster average has not caught up by construction. Requiring EMA50 above
    SMA200 rejected real setups on exactly that basis - EPICUSDT's impulse
    closed at 0.4654 over an SMA200 of 0.4126 with the EMA50 still at 0.3482.

    A bar with no level at all still returns False: the trend cannot be
    confirmed there, and an unconfirmable gate is not a passed one. With the
    weekly fallback in place that now only happens in a symbol's first ~10
    weeks, which is below min_daily_bars anyway.
    """
    if index < 0 or index >= len(levels):
        return False
    level = levels.iloc[index]
    return not pd.isna(level) and bool(closes.iloc[index] > level)


def _coil_fit(closes) -> tuple[float, float]:
    """(R-squared, slope) of a straight line through the coil's closes.

    R-squared near 1 means price walked steadily in one direction; near 0 means
    it wandered, which is what a consolidation is. The SLOPE is returned
    alongside because direction now matters: a coil is allowed to slope down
    (that is what giving some of the impulse back looks like) but not up. It
    used to be direction-blind - see the caller for why that changed.

    Too short to fit returns (1.0, 0.0): no slope, and an R-squared that fails
    the caller's cap on its own if it is ever applied.
    """
    y = closes.to_numpy(dtype=float)
    if len(y) < 3:
        return 1.0, 0.0
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = float(((y - (intercept + slope * x)) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - resid / total if total > 0 else 1.0
    return r_squared, float(slope)


def _volume_ratio(volumes: pd.Series, index: int, baseline_bars: int) -> float:
    """How far this bar's volume stood above what was normal just before it.

    Median rather than mean: one earlier spike in the baseline window would
    otherwise raise the bar for everything after it, so the more eventful a
    symbol's recent history, the harder a genuine spike would be to see.
    """
    baseline = volumes.iloc[max(0, index - baseline_bars) : index].median()
    if not baseline or baseline <= 0:
        return 0.0
    return volumes.iloc[index] / baseline
