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


@dataclass(frozen=True)
class ConsolidationParams:
    """Every constant find_consolidation needs, tuned for one bar scale. See
    the module docstring for why daily and hourly need separate values."""

    pivot_atr_multiple: float  # swing threshold defining the range boundaries
    volume_baseline_bars: int  # median window each pivot's volume is judged against
    volume_spike_multiple: float  # the range top must be printed on a real spike
    volume_increase_multiple: float  # the bottom only needs raised volume
    volume_decline_max: float  # late-half volume inside the range vs its early half
    min_consolidation_bars: int  # enough bars to split into halves at all
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
    min_consolidation_bars=4,
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
    min_consolidation_bars=4,
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
        stop = entry_bars["low"].iloc[-1] - atr_now * STOP_ATR_BUFFER
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
    highs, lows = daily["high"], daily["low"]

    price = closes.iloc[-1]
    if not (price > sma(closes, TREND_MA_PERIOD).iloc[-1] and ema(closes, EMA_SLOW).iloc[-1] > sma(closes, TREND_MA_PERIOD).iloc[-1]):
        return None

    start = max(0, len(daily) - 1 - params.zigzag_lookback)
    window = daily.iloc[start:]
    thresholds = (atr(daily, ATR_PERIOD) * params.pivot_atr_multiple).iloc[start:]
    pivots = zigzag_pivots(window, thresholds)

    pivot_highs = tuple(i + start for i, is_high in pivots if is_high)
    pivot_lows = tuple(i + start for i, is_high in pivots if not is_high)

    # Bracketing, not simply the latest of each: in an uptrend price is often
    # already above its last confirmed pivot high, and that is a range price
    # has left rather than one it is sitting inside.
    above = [i for i in pivot_highs if highs.iloc[i] > price]
    below = [i for i in pivot_lows if lows.iloc[i] < price]
    if not above or not below:
        return None

    top_index, bottom_index = above[-1], below[-1]
    started_at = max(top_index, bottom_index)
    if len(daily) - 1 - started_at < params.min_consolidation_bars:
        return None

    atr_now = atr(daily, ATR_PERIOD).iloc[-1]
    width = highs.iloc[top_index] - lows.iloc[bottom_index]
    if atr_now <= 0 or width > atr_now * params.max_range_atr:
        return None  # too wide to be a coil; price merely sits between two distant levels
    # ...and the same test in plain percentage terms, which a recent violent
    # move cannot inflate the way it inflates ATR.
    if lows.iloc[bottom_index] > 0 and width / lows.iloc[bottom_index] > params.max_range_pct:
        return None

    if not (
        _volume_ratio(volumes, top_index, params.volume_baseline_bars) >= params.volume_spike_multiple
        and _volume_ratio(volumes, bottom_index, params.volume_baseline_bars) >= params.volume_increase_multiple
    ):
        return None

    inside = volumes.iloc[started_at:]
    half = len(inside) // 2
    early, late = inside.iloc[:half].mean(), inside.iloc[half:].mean()
    if not early or early <= 0 or late / early > params.volume_decline_max:
        return None

    return Consolidation(
        top=highs.iloc[top_index],
        bottom=lows.iloc[bottom_index],
        top_index=top_index,
        bottom_index=bottom_index,
        started_at=started_at,
        pivot_highs=pivot_highs,
    )


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
