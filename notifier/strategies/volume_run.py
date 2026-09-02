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

from notifier.risk_sizing import entry_fee_for, round_trip_fee_for
from notifier.strategies.base import FillGuard, Signal, Strategy
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
# Neither sheet gives Strategy 3 any fee-domination check, unlike Strategy 1
# (MIN_LEG_PCT) and Strategy 4 (MAX_FEE_FRACTION_OF_RISK) - a stop tight
# enough could clear the gross 2:1 target above while its NET reward:risk,
# after the round-trip fee, is far worse. Dror, 2026-08-27: "add it for the
# other [strategies]". Same floor Strategy 2.1 already runs live with.
MIN_NET_REWARD_RISK = 1.5
# ONE entry, not a split. Both sheets describe a single breakout fill -
# "price closed above the level" - with no second, better-priced leg to
# rest as a limit; entry_price is close_now, a price already in the past by
# the time an order could be placed, not a level worth waiting at. Signal's
# own market_fraction default (0.2) is the SPLIT-entry idiom Strategy 1 and
# 2 use, and Strategy 3 was inheriting it unset rather than by choice - live
# order-building masked this (scanner._build_order falls back to one full
# market order whenever limit_entry is None, ignoring market_fraction
# entirely), but the backtest engine has no such fallback: it sized the
# market leg at 20% of the position, found no limit_entry to place the
# other 80% against, and rejected nearly every signal as too small for the
# $5 floor - which is why the first-ever measurement of this strategy
# (run_s3_swing.py) found signals but zero closed trades. Dror, 2026-09-02:
# "there shouldn't be 2 entries in this strategy." 1.0 makes what already
# happens live explicit, and fixes the backtest to match it.
MARKET_FRACTION = 1.0
ENTRY_FEE_PCT = entry_fee_for(MARKET_FRACTION)
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
# R-squared above which a rising coil is read as still trending rather than
# pausing.
MAX_COIL_UP_DRIFT_R2 = 0.5
# How much of the box's own height a downward drift may give back before it
# stops being "sideways".
MAX_COIL_DOWN_DRIFT_SHARE = 0.40
# The ceiling is the highest candle in the RALLY_LOOKBACK_BARS immediately
# BEFORE the box, not a bar within the box itself - "the top on an outright
# spike", per this module's own docstring, which the box-starts-at-its-own-
# high version before this only approximated. GIGGLEUSDT, 2026-08-29: that
# version measured a real box (Aug 3-27, 42.24) while ignoring a bigger,
# more recent spike (Jul 31, 55.67, only 3 days before the box's own start)
# because 55.67 sat OUTSIDE whatever window search tried, never compared
# against it directly. Searching backward from the box's own start line, for
# the single tallest candle, fixes that: the level is always the spike that
# precedes the pause, wherever it sits, not whatever a shape-only search
# happens to also contain.
#
# This is a deliberate return to how VERSION ONE of this detector - a single
# impulse candle - picked its level, replaced on 2026-08-23 because it only
# matched 7.8% of real levels and 0% of ones touched >=3 times: a level
# defended more than once is rarely the single tallest candle nearby. That
# risk has not been re-measured here; this rebuild accepts it in exchange for
# fixing the GIGGLEUSDT failure, on Dror's explicit call.
#
# The width cap this used to carry (MAX_BOX_ATR, a hard limit on top-to-
# bottom span in ATR) was REMOVED the same day, also on Dror's explicit call:
# even with the impulse fix above, GIGGLEUSDT's real 55.67 level still could
# not qualify - its floor sits far enough below it (the crash bottom) that
# the span was 6.48 ATR wide, well past the old 4.0 cap.
#
# Removing it outright would have reopened a DIFFERENT, already-named bug:
# the original impulse-candle detector (see above) anchored its level to a
# specific past candle and never re-examined it against where price currently
# stood, so a symbol that had since collapsed still carried a "valid",
# long-stale range - the reason that detector produced zero signals across
# two live instances in its entire life. MAX_COIL_DOWN_DRIFT_SHARE does not
# catch this either: it measures NET drift, not total amplitude, and a
# crash-then-recover nets close to flat (measured on GIGGLEUSDT: down_share
# 0.03, far under the 0.40 floor) despite swinging 6+ ATR along the way.
#
# What actually distinguishes "a wide but live pause" (GIGGLEUSDT) from "a
# stale range price has abandoned" is not the box's total height, it is how
# far CURRENT price sits from the level it would need to break - so that is
# what is bounded now, instead of the box's own span. GIGGLEUSDT sits 3.0 ATR
# under its ceiling at signal time; a collapsed range like the old bug's
# would sit hundreds of ATR under its own. Not measured against the two
# reference setups or anything between GIGGLEUSDT and outright collapse; 10.0
# is a first guess, deliberately far looser than either reference case, not
# tuned to them.
MAX_DISTANCE_TO_CEILING_ATR = 10.0
# The box must not exceed its own impulse ceiling - a hard `>` with no
# tolerance until this. Checked against Dror's own confirmed real trades
# (2026-09-02): it was the SOLE reason 5 of 6 checkable ones never fired, and
# a factor in the 6th, and it is the single largest bucket in the full-
# universe rule funnel too (42.9% of every candidate box_len tried, nearly
# double the next-largest rule).
#
# THE VALUE IS 0.5, NOT THE 0.78 THAT WOULD ACTUALLY ADMIT GOOGLUSDT'S REAL
# TRADE - measured properly (minimum excess/ATR across every box_len the
# search tries, not the rough single-day-vs-prior-20-days approximation this
# started from), the real setups separate as:
#
#   GOOGLUSDT  2026-04-27 (real pause, still excluded at 0.5)  0.78 ATR
#   INTCUSDT   2026-01-12 (thin holiday-week data, unclear)    1.04 ATR
#   RIVERUSDT  2026-01-20 (still actively rallying)            2.87 ATR
#   HYPEUSDT   2026-05-21 (still actively rallying)            3.78 ATR
#   INTCUSDT   2026-04-24 (still actively rallying)            5.27 ATR
#
# 0.85 (wide enough to admit GOOGLUSDT, still short of the next case up) was
# tried first and is WRONG despite being the better-evidenced number:
# full-universe backtest, fresh account/year, drop-top-3 checked -
#
#   tolerance   signals  trades  win%  expectancy  drop-top-3   2026 alone
#   0 (none)    49       40      38%   +0.129R     -0.080R      +0.250R
#   0.5 ATR     80       57      37%   +0.096R     -0.048R      +0.453R
#   0.85 ATR    100      69      28%   -0.174R     -0.302R      -0.017R
#
# Monotonic and one real trade's evidence does not survive it: widening
# toward GOOGLUSDT specifically also widens in a population of setups that
# is, in aggregate, worse - even 2026 (robust at every other setting) turns
# net negative at 0.85. 0.5 is the only setting that is a clean improvement
# on doing nothing at all; GOOGLUSDT stays a false negative on Rule 2 (and
# separately fails Rule 4's impulse-volume floor even where Rule 2 alone
# would admit it - not chased further here). Dror's call, 2026-09-02.
RULE2_OVERSHOOT_TOLERANCE_ATR = 0.5
# --- Rule 4: volume declining through the box, real volume on the spike -----
# Late half of the box against its early half. <= 1.0 is "not rising".
COIL_LATE_EARLY_VOLUME_MAX = 1.0
# The impulse candle's own volume against the box's average - not a rolling
# baseline, because what makes a level real is that IT outprinted the quiet
# that followed it, not that it beat some generic window.
IMPULSE_MIN_VOLUME_RATIO = 2.0
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
        setup = _find_consolidation_cached(daily, self.params)
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

        setup = _find_consolidation_cached(daily, self.params)
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
            market_fraction=MARKET_FRACTION,
            fill_guard=FillGuard(
                min_net_reward_risk=MIN_NET_REWARD_RISK,
                maker_fee_pct=ENTRY_FEE_PCT,
                round_trip_fee_pct=round_trip_fee_for(MARKET_FRACTION),
            ),
            extra_notes=tuple(notes),
            reason=(
                f"Daily consolidation between {setup.bottom:.8g} and {setup.top:.8g} lasting "
                f"{len(daily) - 1 - setup.started_at} days, the level set by a volume spike beforehand, "
                f"volume declining through the box. Price closed above the level on the "
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


def find_consolidation(
    daily: pd.DataFrame, params: ConsolidationParams = SWING_PARAMS, stats: dict | None = None
) -> Consolidation | None:
    """The box price is CURRENTLY inside, if one qualifies under the five
    rules - or None. `daily` is the STRUCTURE frame: its last row is the most
    recently CLOSED daily bar, one behind whatever the live entry-timeframe
    trigger is doing intraday. A box is only returned if price, as of that
    last bar, is still inside it - the box must not already have broken out
    in the daily frame itself, which is what evaluate()'s own trigger check
    is for.

    Among every box length that satisfies every rule, the WIDEST one wins
    (the search runs shortest-to-longest and keeps overwriting).

    `stats`, when given, is incremented once per candidate box_len at the
    rule it failed on (or "passed"), so a caller can tell which of the five
    rules is actually the bottleneck instead of guessing at six different
    constants - see scripts/s3_rule_funnel.py. None by default so every
    existing caller (evaluate(), arms(), the memo cache) pays nothing for it.
    """

    def _bump(label: str) -> None:
        if stats is not None:
            stats[label] = stats.get(label, 0) + 1
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
        floor = float(window_low.min())

        # RULE 3 (the level): the ceiling is the tallest candle in the
        # RALLY_LOOKBACK_BARS immediately BEFORE this box - the spike that
        # set the level - not a bar inside the box itself. If there is no
        # room to look back this box_len cannot be evaluated.
        impulse_start = max(0, start - RALLY_LOOKBACK_BARS)
        if impulse_start >= start:
            _bump("no_impulse_lookback_room")
            continue
        impulse_highs = highs.iloc[impulse_start:start]
        impulse_index = impulse_start + int(impulse_highs.to_numpy().argmax())
        ceiling = float(highs.iloc[impulse_index])

        box_height = ceiling - floor
        if box_height <= 0:
            _bump("zero_or_negative_box_height")
            continue

        # The box itself must sit AT OR BELOW the impulse - if the pause made
        # a fresh high of its own, the impulse was not actually the peak, and
        # "the level" is not what this box would be breaking. A small ATR
        # tolerance for a graze - see RULE2_OVERSHOOT_TOLERANCE_ATR's own
        # comment for the evidence and its limits.
        if float(window_high.max()) > ceiling + atr_series.iloc[start] * RULE2_OVERSHOOT_TOLERANCE_ATR:
            _bump("rule2_box_made_a_fresh_high")
            continue

        # RULE 3 (staleness): CURRENT price must still be within reach of the
        # level - not the box's total span, see MAX_DISTANCE_TO_CEILING_ATR's
        # own comment for why this replaced a plain width cap.
        if (ceiling - float(closes.iloc[last])) / atr_now > MAX_DISTANCE_TO_CEILING_ATR:
            _bump("rule3_too_far_from_ceiling")
            continue

        # RULE 4 (the level itself): the impulse candle must carry real
        # volume against the pause that follows it, or the "spike" is not
        # one - measured against the BOX's own average, not a generic
        # rolling baseline, because what makes a level real is that it
        # outprinted the specific quiet that came after it.
        box_volumes = volumes.iloc[start:n]
        consolidation_avg_vol = float(box_volumes.mean())
        if (
            consolidation_avg_vol <= 0
            or volumes.iloc[impulse_index] < IMPULSE_MIN_VOLUME_RATIO * consolidation_avg_vol
        ):
            _bump("rule4_impulse_volume_too_low")
            continue

        # RULE 4 (through the box): volume must not be rising, late half
        # against early half.
        half = box_len // 2
        if half == 0:
            _bump("box_too_short_to_halve")
            continue
        early_vol = box_volumes.iloc[:half].mean()
        late_vol = box_volumes.iloc[half:].mean()
        if not early_vol or early_vol <= 0 or late_vol / early_vol > COIL_LATE_EARLY_VOLUME_MAX:
            _bump("rule4_volume_rising_through_box")
            continue

        # RULE 3c: drift. A coil may slope down (giving back some of the
        # rally is what a pause looks like) but not climb - a steady rise is
        # still the move, not a break from it.
        r_squared, slope = _coil_fit(closes.iloc[start:n])
        total_drift = slope * (box_len - 1)
        if total_drift > 0 and r_squared > MAX_COIL_UP_DRIFT_R2:
            _bump("rule3c_drifting_up")
            continue
        if total_drift < 0 and abs(total_drift) / box_height > MAX_COIL_DOWN_DRIFT_SHARE:
            _bump("rule3c_drifted_down_too_much")
            continue

        # RULE 1 (unconditional floor): the rally INTO the impulse must be
        # real, measured from the lowest point in the same lookback window up
        # to the impulse itself - whatever the uptrend read below decides.
        rally_low_window = lows.iloc[impulse_start : impulse_index + 1]
        atr_at_impulse = atr_series.iloc[impulse_index]
        if (
            rally_low_window.empty
            or atr_at_impulse <= 0
            or (ceiling - rally_low_window.min()) / atr_at_impulse < MIN_RALLY_INTO_LEVEL_ATR
        ):
            _bump("rule1_rally_into_level_too_small")
            continue

        # RULE 1: the market must have been in a structural uptrend before
        # this box started.
        if not structural_uptrend(daily, start, closes, levels, atr_series):
            _bump("rule1_no_structural_uptrend")
            continue

        _bump("passed")
        best = Consolidation(
            top=ceiling,
            bottom=floor,
            top_index=impulse_index,
            bottom_index=start + int(window_low.to_numpy().argmin()),
            started_at=start,
            pivot_highs=pivot_highs,
        )

    return best


# find_consolidation's input (the DAILY frame) only advances once every ~24
# calls: evaluate() runs on every 1H bar (correctly - the breakout trigger
# needs that granularity), but the daily consolidation search behind it does
# not change until a new daily bar closes. Un-memoized, that meant redoing the
# same 10-60 box-length search, structure_context call and all, ~24x more
# often than its answer could possibly change - measured at 56ms/call on
# BNBUSDT's 2,596-bar daily history, which is ~16 minutes of pure waste per
# symbol over a 2-year backtest window, and the reason a 100-symbol sample run
# hadn't finished after 2+ hours. `daily["ts"].iloc[-1]` (its last CLOSED
# bar's timestamp) plus its length is a cheap, correct fingerprint for "is
# this the same daily frame as last time" - two frames with the same length
# ending on the same day are the same frame, since daily bars are immutable
# once closed. Bounded rather than an unbounded dict: a full-universe replay
# touches many (symbol, day) pairs and this is a plain process-local cache,
# not something that should grow forever across a long-running live scanner.
_CONSOLIDATION_CACHE_MAX = 20_000
_consolidation_cache: dict[tuple, Consolidation | None] = {}


def _find_consolidation_cached(daily: pd.DataFrame, params: ConsolidationParams) -> Consolidation | None:
    if daily.empty:
        return find_consolidation(daily, params)
    # Length + last ts alone is NOT a safe fingerprint: two different symbols
    # sharing a trading calendar (or two unit-test fixtures reusing the same
    # date range) can have the same length and end date with completely
    # different prices, which silently handed one symbol's cached setup to
    # another - caught by test_arming_refuses_a_symbol_with_no_consolidation
    # returning a stale positive. Hashing the actual close/volume arrays is
    # what the box search and structural_uptrend actually read, so it is what
    # has to match - and at a few thousand floats this is still microseconds
    # against find_consolidation's ~56ms, nowhere near erasing the saving.
    fingerprint = hash(daily["close"].to_numpy().tobytes()) ^ hash(daily["base_vol"].to_numpy().tobytes())
    key = (len(daily), fingerprint, params)
    if key not in _consolidation_cache:
        if len(_consolidation_cache) >= _CONSOLIDATION_CACHE_MAX:
            _consolidation_cache.clear()
        _consolidation_cache[key] = find_consolidation(daily, params)
    return _consolidation_cache[key]


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
