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
from notifier.strategies.structure import zigzag_pivots

TREND_MA_PERIOD = 200
# Weeks in the fallback trend average, used only where the 200-day one has not
# warmed up. Ten weeks needs ~70 daily bars against the 200-day's 200, which is
# what lets the whole strategy run on a symbol listed under seven months.
WEEKLY_TREND_WEEKS = 10
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
# What makes the move INTO the level count as a move up. Measured on the whole
# candle, wick included - the opposite of the flag pole, which is body-only
# because a wicky pole is a bad pole. Here the wick is the rejection that made
# the level, so it belongs in the measurement and has to be a real share of
# the bar. See _is_upward_impulse.
IMPULSE_ATR_MULTIPLE = 2.0
IMPULSE_MIN_WICK_SHARE = 0.3
# THE RALLY INTO THE LEVEL is what "a big move up" means - not the colour of
# the candle that tops it. ADAUSDT's level was set by a RED candle and Dror
# named it the setup he wants: a 12% rally ran into it and that bar is the
# rejection at the top. SOLUSDT and DOGEUSDT were red bars too and he threw
# both out - "sol candle is big red one how it is possible that its marked" -
# because nothing had rallied into them; each was just a bar whose high
# happened to be a local high.
#
# Measured from the lowest low of the preceding RALLY_LOOKBACK bars up to the
# wick top, in ATR. The three he accepted ran 5.1-6.9; the two he rejected on
# this ground ran 2.8 and 3.8. The cut sits in the gap.
RALLY_LOOKBACK = 12
RALLY_MIN_ATR = 5.0
# THERE IS NO LIMIT ON HOW LONG THE COIL RUNS. Dror: "there is no limit to
# the width the opposite the longer the consolidation the better". A duration
# cap was tried and removed - the cheatsheet's own claim is that a longer
# consolidation is a better trade, and SOXLUSDT's 287-bar coil is refused by
# the volume rules on its own merits, not for being old.
#
# What IS limited is how far price fell back. A pullback that gives the whole
# rally back has left nothing to break out of, so the coil's lowest point must
# stay above where the rally began. No invented constant - the bound is the
# rally the setup already carries. Dror's three reference setups dipped 59%,
# 72% and 78% of their rallies.
MAX_DIP_SHARE_OF_RALLY = 1.0
# The coil must have NO TREND, up or down - Dror: "the consolidation is
# measured that there is no trend not up or down so we dont need to measure
# the bottom only to check the trend". Measured as the R-squared of a straight
# line through the coil's closes: a drift in either direction fits a line
# well, chop does not. His references scored 0.02 (SPCXUSDT), 0.25 (ETHUSDT)
# and 0.39 (ADAUSDT), so the cut sits above all three.
MAX_COIL_TREND_R2 = 0.5
# Volume across the coil against volume across the rally that made the level.
# This is what "volume dried up" actually claims, and it separates every
# accepted setup from every rejected one on its own: 0.30 / 0.36 / 0.59 for
# the good, 0.84 / 0.86 / 11.56 for the bad. The existing early-half versus
# late-half test only looks WITHIN the coil, so a coil that is quiet at both
# ends but busier than the rally still passes it.
COIL_VOLUME_MAX_SHARE = 0.7


@dataclass(frozen=True)
class ConsolidationParams:
    """Every constant find_consolidation needs. Both instances now read daily
    bars, so the two sets differ only where the sheets differ."""

    pivot_atr_multiple: float  # swing threshold defining the range boundaries
    volume_baseline_bars: int  # median window each pivot's volume is judged against
    volume_spike_multiple: float  # the range top must be printed on a real spike
    volume_increase_multiple: float  # the bottom only needs raised volume
    volume_decline_max: float  # late-half volume inside the range vs its early half
    # How long price must pause before a break counts - the ONLY value the two
    # sheets disagree on. The swing sheet is silent, so this carries Dror's
    # FIGHTUSDT call: "the breakout should be after a period of time ... not
    # instantly like fight i thinking minimum 20 candles". FIGHT's coil ran 17
    # bars; every setup he accepted ran 27 or more, so the cut sits in the gap,
    # and 20 days is also where the original calibration found real
    # consolidations living (22-25 day median). The DAY sheet overrides it
    # outright - "the consolidation can be just a few single days" - so that
    # instance sets its own floor. See DAY_PARAMS.
    min_consolidation_bars: int
    # Widest top-to-bottom span still called a coil rather than two distant
    # levels with price wandering between them. Measured in ATR, not percent:
    # percent cannot tell a quiet symbol's 10% range from a violent one's, and
    # among the setups this actually finds, 10.5% (NVDAUSDT) and 78.1%
    # (TAGUSDT) are both perfectly good coils while a single 199.9% one is not.
    # In ATR the same population is tight - median 4.9, p75 6.0 - and the bad
    # case stands alone. See SWING_PARAMS for where the cut sits and why.
    max_range_atr: float
    # And an ABSOLUTE ceiling, as a fraction of the range floor, that no amount
    # of recent volatility can argue around. ATR alone is not enough, because
    # ATR is inflated by the very move that breaks the range - and the break is
    # when the check runs. BANKUSDT measured 40.2 ATR wide five days before its
    # signal, 20.5 two days before, and 9.0 on the signal day itself, as its
    # ATR quadrupled on the way out; an ATR-only cap is at its weakest exactly
    # when it matters. In percent the same range reads 199.9% throughout.
    max_range_pct: float
    zigzag_lookback: int  # bars considered when locating the range
    # How far beyond the range top a close must be to count as a breakout.
    # Without it the line is the pivot bar's own high and merely grazing it
    # qualifies: TSLAUSDT triggered 0.012% past it - four cents on a $324
    # stock - and again ten minutes later at 0.006%.
    min_penetration_atr: float = 0.0


# The swing version: a daily consolidation broken on a 1H close. These are the
# values measured against daily bars, and with the day version moved back onto
# daily bars they are now the baseline for both.
#
# Both width caps come from replaying the detector over 70 watchlist symbols of
# daily bars, 2024-04 to 2026-08, which found 15 distinct consolidations.
#
# In ATR their spans ran 3.0, 3.3, 3.4, 4.0, 4.2, 4.2, 4.9, 4.9, 5.1, 5.4, 5.4,
# 6.6, 9.5, 9.6 - and then 20.5. In percent: 10.5 through 78.1, and then 199.9.
# Both outliers are the same setup, BANKUSDT: a 265-day, 199.9%-wide "range"
# that produced a live signal on 2026-07-19 and is not a coil at all. Each cut
# sits in its own gap - 12 ATR above every genuine span and below the bad one,
# 100% likewise.
#
# BOTH are needed, and the ATR one alone is the weaker guard. ATR is inflated
# by the breakout itself, so BANKUSDT measured 40.2 ATR wide five days out,
# 20.5 two days out and 9.0 on the signal day - it would have slipped under a
# 12-ATR cap at the only moment the cap is consulted. The percentage does not
# move: 199.9% on every one of those days.
#
# Deliberately NOT swept for a best value. With 15 setups in two years a sweep
# on outcome has nothing to measure, and what these constants honestly do is
# exclude one known-bad case with margin. A tighter 8 ATR would also drop
# XLMUSDT at 9.5 and TRXUSDT at 9.6, both ordinary-looking coils, costing a
# fifth of all setups to catch nothing extra.
SWING_PARAMS = ConsolidationParams(
    pivot_atr_multiple=3.0,
    volume_baseline_bars=30,
    volume_spike_multiple=2.0,
    volume_increase_multiple=1.0,
    volume_decline_max=0.8,
    min_consolidation_bars=20,
    max_range_atr=12.0,
    max_range_pct=1.0,  # the top may sit at most 100% above the range floor
    zigzag_lookback=300,
)

# The day version: the SAME daily consolidation, broken on a 5m close. Two
# deliberate differences from SWING_PARAMS, both traceable to a sheet.
DAY_PARAMS = ConsolidationParams(
    pivot_atr_multiple=3.0,
    volume_baseline_bars=30,
    volume_spike_multiple=2.0,
    volume_increase_multiple=1.0,
    volume_decline_max=0.8,
    # "The consolidation can be just a few single days" - the day sheet, in as
    # many words, so the 20-bar floor cannot apply here. 3 is not a taste
    # judgement about what "a few" means: it is the shortest coil the coil's
    # own tests can be computed on at all. _coil_fit needs three closes to
    # fit a line through (fewer returns 1.0 and fails the no-trend check), and
    # the early/late volume split needs two bars to have two halves. Below 3
    # the strategy cannot evaluate its own rules, so the sheet's floor and the
    # arithmetic floor happen to meet.
    min_consolidation_bars=3,
    # The same coil, so the same ceilings; see SWING_PARAMS.
    max_range_atr=12.0,
    max_range_pct=1.0,
    zigzag_lookback=300,
    # A 5-minute close can clear a daily level by a hair and still count as a
    # break without this. TSLAUSDT triggered 0.012% past the line and again
    # ten minutes later at 0.006%; 0.10 ATR rejects both while sitting far
    # below any real breakout. This is the one guard from the hourly build
    # worth keeping - it was fixing the TRIGGER being fast, not the structure
    # being on the wrong chart.
    min_penetration_atr=0.10,
)


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

        Derived from what the rules actually need rather than fixed at
        TREND_MA_PERIOD + baseline. That old floor of 230 was really just "warm
        up the 200-day average", and since trend_levels now falls back to a
        10-week one, the 200-day average is no longer the binding requirement.

        What IS required, in order: enough weeks for the fallback average to
        exist at the impulse, the volume baseline each boundary pivot is judged
        against, and the coil itself. The spare week absorbs resample
        boundaries - ten calendar weeks is not always ten complete weekly bars
        depending on where the history starts.
        """
        return (WEEKLY_TREND_WEEKS + 1) * 7 + self.params.volume_baseline_bars + self.params.min_consolidation_bars

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
        if len(daily) < self.min_daily_bars() or len(entry_bars) < ATR_PERIOD + 2:
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
                f"{len(daily) - 1 - setup.started_at} days, both boundaries formed on raised volume with a spike "
                f"at the top, volume then falling away inside the range. Price closed above the range on the "
                f"{self.entry_timeframe}. Stop is a full ATR below "
                + ("the breakout candle's low." if self.stop_anchor == STOP_AT_BREAKOUT_CANDLE
                   else "the last low before the breakout.")
            ),
        )


def find_consolidation(daily: pd.DataFrame, params: ConsolidationParams = SWING_PARAMS) -> Consolidation | None:
    """The range price is currently inside, if it qualifies as a volume-run
    consolidation. Both instances pass DAILY bars - the cheatsheets identify
    the consolidation on the daily chart in both cases - and differ only in
    how long a pause they will accept."""
    closes, volumes = daily["close"], daily["base_vol"]
    highs, lows, opens = daily["high"], daily["low"], daily["open"]

    price = closes.iloc[-1]

    # THE TREND GATE QUALIFIES THE IMPULSE CANDLE, NOT TODAY'S BAR. Dror:
    # "the sma200 and the zigzag are for finding the big candle that takes the
    # liquidity and starts the setup ... if there isnt a sma200 but the rest of
    # the condition is there the code will be able to find this trades".
    #
    # Read on the last bar it silently required the market to STILL be in an
    # uptrend at the moment of the break - but the whole setup is a pause after
    # an impulse, and price spends that pause drifting back down. ADAUSDT is the
    # case: at its 2025-03-02 impulse it was clearly in an uptrend (close 0.9567
    # against an SMA200 of 0.6702), and by the time the coil had formed on
    # 03-31 it had slipped below (0.6914 against 0.7277). Identical structure,
    # refused for what happened AFTER the level was set.
    #
    # Only the SMA200 half survives the move. "EMA50 above SMA200" is a lagging
    # confirmation that by construction cannot be true when a big candle STARTS
    # a move - the impulse is what eventually drags the EMA50 up there. It
    # rejected EPICUSDT (close 0.4654 over an SMA200 of 0.4126, but EMA50 only
    # 0.3482) and XLMUSDT on exactly that basis, both of them real setups.
    levels = trend_levels(daily)

    start = max(0, len(daily) - 1 - params.zigzag_lookback)
    window = daily.iloc[start:]
    thresholds = (atr(daily, ATR_PERIOD) * params.pivot_atr_multiple).iloc[start:]
    pivots = zigzag_pivots(window, thresholds)

    pivot_highs = tuple(i + start for i, is_high in pivots if is_high)
    pivot_lows = tuple(i + start for i, is_high in pivots if not is_high)

    # THE RANGE MUST FOLLOW A MOVE UP. This strategy only ever goes long, so a
    # consolidation is only a consolidation if there is an upward impulse to
    # consolidate FROM - rally, pause, break higher.
    #
    # Nothing checked this before, and it is what Dror rejected on BNBUSDT and
    # SOLUSDT: "this method is only for long so it must be after a big move up
    # not down". On both, price had fallen, bottomed, and the "range" was
    # simply the nearest confirmed pivot above the current price paired with
    # the nearest below it. Two pivots bracketing wherever price happens to be
    # standing is not a range price has lived in - on BNBUSDT price had spent
    # fifty of those bars trading entirely BELOW the supposed range floor.
    #
    # The impulse candle must also carry a big WICK, which is where this
    # deliberately parts company with the flag pole in patterns.py. There the
    # pole is measured on bodies and a wick-heavy candle is disqualified; here
    # the wick is the point - it is the rejection that made the level, and the
    # top of that wick is what the trade later breaks.
    above = [i for i in pivot_highs if highs.iloc[i] > price and _is_upward_impulse(
        daily, i, atr(daily, ATR_PERIOD), params) and in_uptrend_at(closes, levels, i)]
    if not above:
        return None

    atr_series = atr(daily, ATR_PERIOD)

    # The range top is picked by DOMINANCE among the confirmed pivot highs
    # above price, not by recency. "Most recently confirmed" answers "what
    # formed last" - a small candle a few bars back beats a genuinely dominant
    # one further away purely for being newer, which is what silently anchored
    # the level on a bar the market never actually treated as resistance.
    #
    # Dominance requires agreement across three signals read off the SAME bar:
    # the tallest body (real conviction drove price up to it), the longest
    # upper wick (real rejection pushed it back down - this wick IS the level),
    # and the largest volume (real supply showed up to do the rejecting). Any
    # one of the three alone can be misleading - a huge wick on thin volume is
    # a thin market's noise, not a defended level.
    for candidate in _dominant_pivots(highs, opens, closes, volumes, above, atr_series, params.volume_baseline_bars):
        if candidate is None:
            continue
        top_index = candidate

        # The level must still be UNTESTED since it was set. A dominant candle
        # that gets poked again before the market has genuinely gone quiet was
        # never actually left alone - there is no convergence to speak of, just
        # a spike and an immediate retest. Live case: INTCUSDT's dominant candle
        # set 102.37 at 13:00; by 16:00, three bars later, price had already
        # wicked to 102.69 - past the level - before anything resembling a coil
        # had time to form.
        #
        # Scoped to the FULL history since top_index formed, not to started_at -
        # deliberately not the same window as min_consolidation_bars below. An
        # old dominant candle is not disqualified for being old (that is the
        # width check's job, and age is validation there, not a penalty) - but
        # it IS disqualified if it has ever been retested since forming,
        # however long ago that was. An old level nobody has come back to test
        # is, if anything, the strategy's own "all-time high" case: nothing
        # overhead because nothing has challenged it.
        #
        # Falls through to the next candidate exactly like a width failure -
        # this is a fact about THIS bar, not a verdict on whether a setup
        # exists at all.
        since_formed = highs.iloc[top_index + 1 :]
        if not since_formed.empty and since_formed.max() >= highs.iloc[top_index]:
            continue

        # THE CONSOLIDATION STARTS AT THE FIRST LOW AFTER THE IMPULSE, and the
        # range floor is that low. Dror on SPCXUSDT: "it should count from the
        # first low so the 3 candle after the big one" - the big candle ran to
        # 141.42, price put in its first low three bars later, and the coil is
        # what follows THAT. The detector had instead paired the level with a
        # spike low thirty bars further on, which was neither the start of the
        # pause nor a floor price had respected.
        #
        # Using the nearest pivot low below price also let the floor sit
        # anywhere at all, including below ground price had long since left.
        # The first low after the impulse is a bar with a job: it is where the
        # pullback from the level stopped.
        bottom_index = _first_low_after(lows, top_index)
        if bottom_index is None or lows.iloc[bottom_index] >= highs.iloc[top_index]:
            continue

        # PRICE MUST STILL BE INSIDE THE RANGE. Nothing checked this, and it is
        # why Strategy 3 produced ZERO signals across two live instances in its
        # entire life - not a rare setup, an impossible one.
        #
        # The gates above bound the range against the IMPULSE (is it a real
        # level, was it left untested, is the coil long enough, is it narrow
        # enough) and never against where price stands NOW. `highs > price` is
        # trivially true once price has collapsed, the floor was never compared
        # to price at all, and "untested since it formed" is satisfied most
        # easily by a level price fell away from and never revisited: the level
        # is pristine precisely because it has become irrelevant.
        #
        # ADAUSDT, measured 2026-02-23: price 0.2621 against a range of
        # 0.9010-1.0204 set on 2025-08-14. Price sat 5.3 range-widths BELOW the
        # floor and the breakout would have needed a 289% single-candle move.
        # Across 4,224 daily bars, price was in the top 10% of its own range
        # exactly 0 times, and the best it ever reached was 0.773 of the way up.
        #
        # This is the BNBUSDT/SOLUSDT failure the impulse rule was added to fix,
        # returning in a form that rule cannot see: _is_upward_impulse and
        # in_uptrend_at both qualify the impulse CANDLE, and say nothing about
        # what price did in the six months after it.
        if not (lows.iloc[bottom_index] <= price <= highs.iloc[top_index]):
            continue

        started_at = bottom_index
        coil_bars = len(daily) - 1 - started_at
        if coil_bars < params.min_consolidation_bars:
            return None  # scoped to the width check below; not a candidate the fallback retries

        atr_now = atr_series.iloc[-1]
        width = highs.iloc[top_index] - lows.iloc[bottom_index]
        if atr_now <= 0:
            return None

        # THE SPAN ITSELF HAS A CEILING. Two levels far enough apart are not a
        # coil however tidily price wanders between them, and nothing else here
        # bounds the distance: the dip test below is RELATIVE to the rally, so
        # a big enough rally licenses an arbitrarily large range.
        #
        # This check existed as a documented parameter for months and was never
        # read - max_range_atr and max_range_pct were both set on every params
        # object and referenced nowhere, so BANKUSDT's 199.9%-wide, 20.5-ATR
        # "consolidation" signalled live on 2026-07-19 against a rule that was
        # supposed to have refused it. test_the_width_cap_is_actually_enforced
        # exists so that cannot recur silently.
        #
        # Falls through to the next candidate like the dip check below: this is
        # a fact about THIS top/bottom pairing, not a verdict on the symbol.
        floor = lows.iloc[bottom_index]
        if floor <= 0 or width / floor > params.max_range_pct:
            continue
        if width / atr_now > params.max_range_atr:
            continue

        # The dip, not the span: how much of the rally the pullback gave back.
        # A coil that erases the rally has nothing left to break out of.
        rally_low = lows.iloc[max(0, top_index - RALLY_LOOKBACK) : top_index].min()
        rally = highs.iloc[top_index] - rally_low
        too_wide = rally <= 0 or width / rally > MAX_DIP_SHARE_OF_RALLY
        if too_wide:
            # An old, dominant candle is not penalised for being far away - the
            # time the market has spent failing to break it is read as MORE
            # validation, not less. But this specific PAIRING with the current
            # bottom pivot does not make a coil, so try the next candidate
            # (the wick+volume co-winner with the dominant bar excluded)
            # rather than discarding the whole setup on one bad pairing.
            continue

        if not (
            _volume_ratio(volumes, top_index, params.volume_baseline_bars) >= params.volume_spike_multiple
            and _volume_ratio(volumes, bottom_index, params.volume_baseline_bars) >= params.volume_increase_multiple
        ):
            return None  # the dominance winner still has to clear the absolute spike floor

        # The window starts AFTER the boundary pivot, not at it. That bar is
        # one of the two that DEFINE the range and it was selected for having
        # raised volume - volume_increase_multiple at the bottom, an outright
        # spike at the top - so including it loads the "before" half with the
        # single highest-volume bar available and manufactures a decline out
        # of nothing.
        #
        # BNBUSDT 1H, 2026-08-11, is the case Dror caught: the bottom pivot
        # printed 2,868 against a consolidation running 293, 295, 337, 841,
        # 905, 550, 453. With that bar in, early/late read 948 -> 687 = 0.72
        # and passed. Without it the same bars read 308 -> 687 = 2.23 - volume
        # more than DOUBLED - and the setup is correctly refused.
        #
        # Same fault the flag detector had: its consolidation window began at
        # pole_end and inherited the pole's own final candle, so the pause was
        # measured against a bar belonging to the move it was supposed to be
        # pausing from.
        inside = volumes.iloc[started_at + 1 :]
        half = len(inside) // 2
        if half == 0:
            return None
        early, late = inside.iloc[:half].mean(), inside.iloc[half:].mean()
        if not early or early <= 0 or late / early > params.volume_decline_max:
            return None

        # A COIL MAY DRIFT DOWN. Dror, revising his earlier "no trend not up or
        # down": "inside the box can be a small downtrend or no trend at all".
        #
        # That is what a pause after an impulse actually looks like - price
        # gives some of the move back while the sellers who made the level stop
        # showing up. Measured across every setup the detector finds, all but
        # one drift DOWN, from -0.11% to -1.34% per bar, and the old
        # direction-blind cut was throwing out the steadiest of them (BNBUSDT
        # at R2 0.84, NEARUSDT 0.85, ADAUSDT 0.77) purely for sloping.
        #
        # An UPWARD drift is still refused: price climbing steadily into the
        # level is still running, and a leg with a box drawn round it is not a
        # pause. How far down the drift may go is already bounded by the dip
        # test above and the width caps, so there is no second constant here.
        r_squared, slope = _coil_fit(closes.iloc[started_at + 1 :])
        if slope > 0 and r_squared > MAX_COIL_TREND_R2:
            return None

        # And the coil has to be quieter than the RALLY that made the level.
        # The halves test above only compares the coil against itself, so a
        # coil that is flat but busier than the move into the level still
        # passes it - which is how SOXLUSDT coiled at eleven times the rally's
        # volume and was still called a dry-up. See COIL_VOLUME_MAX_SHARE.
        rally_vol = volumes.iloc[max(0, top_index - RALLY_LOOKBACK) : top_index + 1].mean()
        coil_vol = inside.mean()
        if not rally_vol or rally_vol <= 0 or coil_vol / rally_vol > COIL_VOLUME_MAX_SHARE:
            return None

        return Consolidation(
            top=highs.iloc[top_index],
            bottom=lows.iloc[bottom_index],
            top_index=top_index,
            bottom_index=bottom_index,
            started_at=started_at,
            pivot_highs=pivot_highs,
        )
    return None


def _is_upward_impulse(daily, index: int, atr_series, params: ConsolidationParams) -> bool:
    """Whether this bar is a big move UP that left a wick at the top.

    Three things, all read off the one bar and the ones just before it:

    - it made a NEW HIGH over the preceding bars, so price arrived here going
      up. This is what BNBUSDT and SOLUSDT failed: their levels sat above a
      market that had fallen into place, not risen into it, and a long-only
      strategy has nothing to consolidate from there.
    - its whole range, wick included, is a real move. Measured high-to-low
      rather than on the body precisely BECAUSE of the wick - the flag pole in
      patterns.py throws out wick-heavy candles, and here the wick is the
      point.
    - the upper wick is a meaningful share of that range: the move was pushed
      back. A candle that closes on its high has not been rejected and has not
      made a level yet.
    """
    a = float(atr_series.iloc[index])
    if a <= 0 or index < 1:
        return False

    high, low = float(daily["high"].iloc[index]), float(daily["low"].iloc[index])
    span = high - low
    if span < IMPULSE_ATR_MULTIPLE * a:
        return False

    body_top = max(float(daily["open"].iloc[index]), float(daily["close"].iloc[index]))
    if (high - body_top) / span < IMPULSE_MIN_WICK_SHARE:
        return False

    prior_highs = daily["high"].iloc[max(0, index - RALLY_LOOKBACK) : index]
    if not prior_highs.empty and high <= prior_highs.max():
        return False  # not a new high: price did not arrive here going up

    # And the approach has to be a real rally, not a drift that happens to
    # print one tall bar. This is the SOLUSDT / DOGEUSDT test.
    prior_lows = daily["low"].iloc[max(0, index - RALLY_LOOKBACK) : index]
    if prior_lows.empty:
        return False
    return bool((high - float(prior_lows.min())) >= RALLY_MIN_ATR * a)


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


def weekly_trend_levels(daily: pd.DataFrame, weeks: int = WEEKLY_TREND_WEEKS) -> pd.Series:
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


def _first_low_after(lows, top_index: int) -> int | None:
    """The first bar after the impulse where the pullback stopped.

    The first local minimum, not the lowest low of the whole window: the
    consolidation begins where the drop off the level ends, and anything
    deeper later belongs to the coil rather than to its floor.
    """
    for i in range(top_index + 2, len(lows) - 1):
        if float(lows.iloc[i]) < float(lows.iloc[i - 1]) and float(lows.iloc[i]) <= float(lows.iloc[i + 1]):
            return i
    return None


def _dominant_pivots(highs, opens, closes, volumes, candidates, atr_series, baseline_bars):
    """(tier-1 winner, tier-2 fallback) among the confirmed pivot highs above
    price - see find_consolidation for the reasoning. Either or both may be
    None.

    Tier 1: the same bar is the argmax of body, upper wick, AND volume, all
    three normalised by that bar's own ATR (volume is already a ratio).

    Tier 2: with the tier-1 bar excluded (if there was one - a bar that wins
    all three trivially also wins the wick+volume pair, so without excluding
    it "fall through to tier 2" would just retry the identical bar and fail
    identically), the bar that is the argmax of wick AND volume together,
    dropping body. Computed over the full set when there is no tier-1 winner
    to exclude.
    """
    def scored(pool):
        rows = []
        for i in pool:
            a = atr_series.iloc[i]
            if a <= 0:
                continue
            body = abs(closes.iloc[i] - opens.iloc[i]) / a
            wick = (highs.iloc[i] - max(opens.iloc[i], closes.iloc[i])) / a
            vol = _volume_ratio(volumes, i, baseline_bars)
            rows.append((i, body, wick, vol))
        return rows

    rows = scored(candidates)
    if not rows:
        return None, None

    body_winner = max(rows, key=lambda r: r[1])[0]
    wick_winner = max(rows, key=lambda r: r[2])[0]
    vol_winner = max(rows, key=lambda r: r[3])[0]

    tier1 = wick_winner if body_winner == wick_winner == vol_winner else None

    remaining = [i for i in candidates if i != tier1] if tier1 is not None else list(candidates)
    rows2 = scored(remaining)
    tier2 = None
    if rows2:
        w2 = max(rows2, key=lambda r: r[2])[0]
        v2 = max(rows2, key=lambda r: r[3])[0]
        tier2 = w2 if w2 == v2 else None

    return tier1, tier2


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
