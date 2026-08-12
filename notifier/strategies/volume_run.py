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

Two versions, differing in trigger, holding horizon, AND the scale the
consolidation itself is measured on - per the cheatsheet's swing/day split
rather than a scale sweep like Strategies 1 and 2 use. The swing version
reads its consolidation off daily bars with a 1H trigger and closes its
runner after three trading days if nothing else has closed it first. The day
version reads the SAME algorithm off hourly bars instead - its own range,
spike, dry-up and resistance check, all on the hourly chart - with a 5m
trigger and no time-based clock at all, since a five-minute entry paired with
a multi-day hold made no sense at that scale. The two are otherwise
unrelated: no bonus or gating between them, each stands on its own structure.

The constants that shape find_consolidation - the pivot threshold, the
baseline window, the volume ratios, how wide a range still counts as a coil -
were measured against daily bars specifically and do not transfer to hourly
ones; an hourly candle's ATR and volume behave nothing like a daily one's over
the same lookback length. Each scale therefore gets its own ConsolidationParams
rather than sharing one set of module constants, with the trend/structure
gate (SMA200 price filter, EMA50 above it) and the fixed 1:2 / 75% exit being
the only pieces the cheatsheet fixes regardless of scale.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, ema, sma
from notifier.strategies.structure import zigzag_pivots

TREND_MA_PERIOD = 200
EMA_SLOW = 50
ATR_PERIOD = 14
STOP_ATR_BUFFER = 1.0  # the stop sits a full ATR below the low, never on it
REWARD_RISK_RATIO = 2.0
PARTIAL_FRACTION = 0.75  # taken off at the 1:2 target; the rest runs
ARMING_BAND = 0.10  # top tenth of the range: close enough to be worth 5m polling
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
    """Every constant find_consolidation needs, tuned for one bar scale. See
    the module docstring for why daily and hourly need separate values."""

    pivot_atr_multiple: float  # swing threshold defining the range boundaries
    volume_baseline_bars: int  # median window each pivot's volume is judged against
    volume_spike_multiple: float  # the range top must be printed on a real spike
    volume_increase_multiple: float  # the bottom only needs raised volume
    volume_decline_max: float  # late-half volume inside the range vs its early half
    # How long price must pause before a break counts. Dror, rejecting
    # FIGHTUSDT: "the breakout should be after a period of time ... not
    # instantly like fight i thinking minimum 20 candles". FIGHT's coil ran 17
    # bars; every setup he accepted ran 27 or more, so the cut sits in the gap.
    # On the daily instance 20 bars is 20 days, which is also where the
    # original calibration found real consolidations living (22-25 day median).
    min_consolidation_bars: int
    max_range_atr: float  # widest top-to-bottom span still called a coil, not two distant levels
    zigzag_lookback: int  # bars considered when locating the range
    # ATR-relative width alone is not enough intraday: hourly ATR is inflated
    # by the very move that formed the range, so a violent move licenses an
    # absurd absolute width for the next ~14 bars. AXTIUSDT passed a
    # 28.22%-wide "consolidation" that way. This caps the span in plain
    # percentage terms as well, which no amount of recent volatility can
    # argue around.
    max_range_pct: float = 1.0  # 1.0 = no absolute cap (the daily default)
    # How far beyond the range top a close must be to count as a breakout.
    # Without it the line is the pivot bar's own high and merely grazing it
    # qualifies: TSLAUSDT triggered 0.012% past it - four cents on a $324
    # stock - and again ten minutes later at 0.006%.
    min_penetration_atr: float = 0.0


# Bracketing pivots alone say nothing about how far apart they are: on live
# daily data this happily called a 17-ATR span (price 1.245 to 3.08, a 148%
# range) a "consolidation" because price happened to sit between two distant
# levels. Candidate spans measured a median of ~6 ATR and a 90th percentile of
# ~9.7, so max_range_atr=8.0 admits roughly the tightest three-quarters while
# rejecting the ranges too wide to be a coil at all.
DAILY_PARAMS = ConsolidationParams(
    pivot_atr_multiple=3.0,
    volume_baseline_bars=30,
    volume_spike_multiple=2.0,
    volume_increase_multiple=1.0,
    volume_decline_max=0.8,
    min_consolidation_bars=20,
    max_range_atr=8.0,
    zigzag_lookback=300,
)

# Measured against ~40 symbols of real hourly bars (~999 closed bars each).
# pivot_atr_multiple needed to come down from the daily 3.0x: hourly candles
# are noisier relative to their own ATR, so 3.0x found almost nothing (median
# ~5 ATR width, 18-21 bar consolidations) while 2.0x found a comparable shape
# of setup at a pace an hourly trigger can actually use (median width ~3.4
# ATR, median length 7 bars). volume_baseline_bars was tested from 20 to 96
# bars and made no measurable difference to signal rate or quality, so it
# keeps the daily value rather than changing it without evidence. The volume
# ratio thresholds (spike/increase/decline) also transferred unchanged: at
# pivot=2.0x they land on ~16 signals/week across the sample with a ~10-hour
# median hold - meaningfully faster than the daily version's multi-day holds,
# which is the whole point of a separate hourly instance.
HOURLY_PARAMS = ConsolidationParams(
    pivot_atr_multiple=2.0,
    volume_baseline_bars=30,
    volume_spike_multiple=2.0,
    volume_increase_multiple=1.0,
    volume_decline_max=0.8,
    min_consolidation_bars=20,
    max_range_atr=8.0,
    zigzag_lookback=300,
    # Both measured against the four signals that got this instance disabled:
    # 15% rejects AXTIUSDT's 28.22% span with margin while leaving a genuine
    # hourly coil alone, and 0.10 ATR rejects TSLAUSDT's 0.02-ATR graze while
    # staying far
    # below any real breakout. Chosen to exclude the observed failures with
    # margin rather than swept - a sweep on rate is exactly what shipped this
    # broken the first time, and a sweep on quality needs more resolved
    # signals than the instance has ever produced.
    max_range_pct=0.15,
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
    """A consolidation on trend_timeframe, breaking out on entry_timeframe.
    Not swept across scales the way Strategies 1 and 2 are: the swing and day
    versions each fix their own pair (1D/1H and 1H/5m) with their own tuned
    ConsolidationParams, rather than mixing and matching."""

    def __init__(
        self,
        trend_timeframe: str = "1D",
        entry_timeframe: str = "1H",
        time_exit_days: int | None = 3,
        armed_only: bool = False,
        params: ConsolidationParams = DAILY_PARAMS,
        session_gated: bool = False,
    ):
        self.trend_timeframe = trend_timeframe
        self.entry_timeframe = entry_timeframe
        self.tag = f"Strategy 3 {trend_timeframe}/{entry_timeframe}"
        self.timeframes = [trend_timeframe, entry_timeframe]
        self.time_exit_days = time_exit_days
        self.params = params
        # Intraday instances read volume and structure off bars that assume
        # a market which is actually trading; a daily bar spans a whole
        # session, so the question does not arise for the swing version.
        self.session_gated = session_gated
        # The 5m version polls per-symbol instead of watchlist-wide; see
        # Strategy.armed_timeframes.
        self.armed_timeframes = (entry_timeframe,) if armed_only else ()

    def arms(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> bool:
        """Worth polling once price is pressing the top of a valid range.

        Expressed as a fraction of the range rather than a distance, so a tight
        coil and a loose one both arm when price is genuinely at the ceiling -
        a percentage-of-price band would mean different things across a
        watchlist spanning several orders of magnitude.
        """
        daily = bars_by_timeframe.get(self.trend_timeframe)
        if daily is None or len(daily) < TREND_MA_PERIOD + self.params.volume_baseline_bars:
            return False
        setup = find_consolidation(daily, self.params)
        if setup is None or setup.top <= setup.bottom:
            return False
        position = (daily["close"].iloc[-1] - setup.bottom) / (setup.top - setup.bottom)
        return bool(position >= 1.0 - ARMING_BAND)

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        daily = bars_by_timeframe.get(self.trend_timeframe)
        entry_bars = bars_by_timeframe.get(self.entry_timeframe)
        if daily is None or entry_bars is None:
            return None
        if len(daily) < TREND_MA_PERIOD + self.params.volume_baseline_bars or len(entry_bars) < ATR_PERIOD + 2:
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
        # THE STOP SITS UNDER THE RECENT LOW BEFORE THE BREAKOUT - Dror's call,
        # replacing a stop measured off the breakout candle's own low.
        #
        # The breakout bar is the one that ran; its low is wherever that
        # particular candle happened to open from, which says nothing about
        # where the move would be wrong. The last low before the break is a
        # level the market actually turned at, so losing it is what
        # invalidates the trade. On a long breakout bar the old rule also put
        # the stop needlessly far away, and on a short one needlessly close.
        recent_low = _recent_low_before(entry_bars["low"], len(entry_bars) - 1)
        if recent_low is None:
            return None
        stop = recent_low - atr_now * STOP_ATR_BUFFER
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * REWARD_RISK_RATIO

        highs = daily["high"]
        # Anything the market already turned back from, sitting between the
        # break and the target, is what stops this trade reaching it.
        if any(setup.top < highs.iloc[i] < target for i in setup.pivot_highs):
            return None

        # The runner exits at the next level above the target; when there is
        # none the setup is at the highs and the exit is a rule, not a price.
        overhead = [highs.iloc[i] for i in setup.pivot_highs if highs.iloc[i] >= target]
        remainder_target = min(overhead) if overhead else None

        if remainder_target is not None:
            remainder_note = "daily resistance"
        elif self.time_exit_days:
            remainder_note = f"after {self.time_exit_days} trading days"
        else:
            remainder_note = "at your discretion"

        notes = []
        if remainder_target is not None and self.time_exit_days:
            notes.append(f"Close the runner after {self.time_exit_days} trading days if resistance is not reached first.")
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
            partial_fraction=PARTIAL_FRACTION,
            remainder_target=remainder_target,
            remainder_note=remainder_note,
            extra_notes=tuple(notes),
            reason=(
                f"Daily consolidation between {setup.bottom:.8g} and {setup.top:.8g} lasting "
                f"{len(daily) - 1 - setup.started_at} days, both boundaries formed on raised volume with a spike "
                f"at the top, volume then falling away inside the range. Price closed above the range on the "
                f"{self.entry_timeframe}. Stop is a full ATR below the breakout candle's low."
            ),
        )


def find_consolidation(daily: pd.DataFrame, params: ConsolidationParams = DAILY_PARAMS) -> Consolidation | None:
    """The range price is currently inside, if it qualifies as a volume-run
    consolidation. `daily` is named for the original swing version; the day
    version calls this with hourly bars and its own ConsolidationParams -
    the algorithm itself is scale-agnostic, only the tuning differs."""
    closes, volumes = daily["close"], daily["base_vol"]
    highs, lows, opens = daily["high"], daily["low"], daily["open"]

    price = closes.iloc[-1]
    if not (price > sma(closes, TREND_MA_PERIOD).iloc[-1] and ema(closes, EMA_SLOW).iloc[-1] > sma(closes, TREND_MA_PERIOD).iloc[-1]):
        return None

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
        daily, i, atr(daily, ATR_PERIOD), params)]
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

        started_at = bottom_index
        coil_bars = len(daily) - 1 - started_at
        if coil_bars < params.min_consolidation_bars:
            return None  # scoped to the width check below; not a candidate the fallback retries

        atr_now = atr_series.iloc[-1]
        width = highs.iloc[top_index] - lows.iloc[bottom_index]
        if atr_now <= 0:
            return None
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

        # NO TREND, up or down. A coil that drifts is a leg with a box drawn
        # round it; chop does not fit a straight line. This replaces measuring
        # the bottom as a structural pivot - the low is now only used to size
        # the dip.
        if _trendiness(closes.iloc[started_at + 1 :]) > MAX_COIL_TREND_R2:
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


def _trendiness(closes) -> float:
    """R-squared of a straight line through the coil's closes.

    Near 1 means price walked steadily in one direction - a trend, whichever
    way it points. Near 0 means it wandered, which is what a consolidation is.
    Direction is deliberately ignored: Dror wants "no trend not up or down",
    and a coil sliding down is no more a coil than one climbing.
    """
    y = closes.to_numpy(dtype=float)
    if len(y) < 3:
        return 1.0
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = float(((y - (intercept + slope * x)) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - resid / total if total > 0 else 1.0


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
