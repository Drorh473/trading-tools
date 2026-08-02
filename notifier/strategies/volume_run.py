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

Two versions, differing only in trigger and holding horizon, per the cheatsheet:
the swing (1H trigger) closes its runner after three trading days if nothing
else has closed it first; the day version (5m trigger) has no such clock, since
a five-minute entry and a three-day hold are three orders of magnitude apart.
Both take 75% off at 1:2 and run the remainder to chart resistance.
"""

from dataclasses import dataclass

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, ema, sma
from notifier.strategies.structure import zigzag_pivots

TREND_MA_PERIOD = 200
EMA_SLOW = 50
ATR_PERIOD = 14
PIVOT_ATR_MULTIPLE = 3.0  # daily swing threshold defining the range boundaries
VOLUME_BASELINE_BARS = 30  # median window each pivot's volume is judged against
VOLUME_SPIKE_MULTIPLE = 2.0  # the range top must be printed on a real spike
VOLUME_INCREASE_MULTIPLE = 1.0  # the bottom only needs raised volume
VOLUME_DECLINE_MAX = 0.8  # late-half volume inside the range vs its early half
MIN_CONSOLIDATION_BARS = 4  # enough bars to split into halves at all
# Bracketing pivots alone say nothing about how far apart they are: on live
# data this happily called a 17-ATR span (price 1.245 to 3.08, a 148% range) a
# "consolidation" because price happened to sit between two distant levels.
# Candidate spans measured a median of ~6 ATR and a 90th percentile of ~9.7,
# so this admits roughly the tightest three-quarters while rejecting the
# ranges too wide to be a coil at all.
MAX_RANGE_ATR = 8.0
STOP_ATR_BUFFER = 1.0  # the stop sits a full ATR below the low, never on it
REWARD_RISK_RATIO = 2.0
PARTIAL_FRACTION = 0.75  # taken off at the 1:2 target; the rest runs
ZIGZAG_LOOKBACK = 300  # daily bars considered when locating the range
ARMING_BAND = 0.10  # top tenth of the range: close enough to be worth 5m polling


@dataclass(frozen=True)
class Consolidation:
    top: float
    bottom: float
    top_index: int
    bottom_index: int
    started_at: int  # the later of the two boundary pivots
    pivot_highs: tuple[int, ...]


class VolumeRun(Strategy):
    """Daily consolidation, faster-timeframe breakout. The pairing is fixed by
    the method rather than swept across scales the way Strategies 1 and 2 are:
    the volume dry-up and the all-time-high context are daily-chart ideas, and
    only the trigger changes between the two versions."""

    def __init__(
        self,
        trend_timeframe: str = "1D",
        entry_timeframe: str = "1H",
        time_exit_days: int | None = 3,
        armed_only: bool = False,
    ):
        self.trend_timeframe = trend_timeframe
        self.entry_timeframe = entry_timeframe
        self.tag = f"Strategy 3 {trend_timeframe}/{entry_timeframe}"
        self.timeframes = [trend_timeframe, entry_timeframe]
        self.time_exit_days = time_exit_days
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
        if daily is None or len(daily) < TREND_MA_PERIOD + VOLUME_BASELINE_BARS:
            return False
        setup = find_consolidation(daily)
        if setup is None or setup.top <= setup.bottom:
            return False
        position = (daily["close"].iloc[-1] - setup.bottom) / (setup.top - setup.bottom)
        return bool(position >= 1.0 - ARMING_BAND)

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        daily = bars_by_timeframe.get(self.trend_timeframe)
        entry_bars = bars_by_timeframe.get(self.entry_timeframe)
        if daily is None or entry_bars is None:
            return None
        if len(daily) < TREND_MA_PERIOD + VOLUME_BASELINE_BARS or len(entry_bars) < ATR_PERIOD + 2:
            return None

        setup = find_consolidation(daily)
        if setup is None:
            return None

        # The breakout is the FIRST close above the range. Without that, every
        # later candle still sitting above the level re-fires the same trade -
        # the shape of bug that sent one stale TSLAUSDT short four times.
        closes = entry_bars["close"]
        close_now, close_prev = closes.iloc[-1], closes.iloc[-2]
        if not (close_now > setup.top and close_prev <= setup.top):
            return None

        atr_now = atr(entry_bars, ATR_PERIOD).iloc[-1]
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


def find_consolidation(daily: pd.DataFrame) -> Consolidation | None:
    """The range price is currently inside, if it qualifies as a volume-run
    consolidation."""
    closes, volumes = daily["close"], daily["base_vol"]
    highs, lows = daily["high"], daily["low"]

    price = closes.iloc[-1]
    if not (price > sma(closes, TREND_MA_PERIOD).iloc[-1] and ema(closes, EMA_SLOW).iloc[-1] > sma(closes, TREND_MA_PERIOD).iloc[-1]):
        return None

    start = max(0, len(daily) - 1 - ZIGZAG_LOOKBACK)
    window = daily.iloc[start:]
    thresholds = (atr(daily, ATR_PERIOD) * PIVOT_ATR_MULTIPLE).iloc[start:]
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
    if len(daily) - 1 - started_at < MIN_CONSOLIDATION_BARS:
        return None

    atr_now = atr(daily, ATR_PERIOD).iloc[-1]
    if atr_now <= 0 or (highs.iloc[top_index] - lows.iloc[bottom_index]) > atr_now * MAX_RANGE_ATR:
        return None  # too wide to be a coil; price merely sits between two distant levels

    if not (
        _volume_ratio(volumes, top_index) >= VOLUME_SPIKE_MULTIPLE
        and _volume_ratio(volumes, bottom_index) >= VOLUME_INCREASE_MULTIPLE
    ):
        return None

    inside = volumes.iloc[started_at:]
    half = len(inside) // 2
    early, late = inside.iloc[:half].mean(), inside.iloc[half:].mean()
    if not early or early <= 0 or late / early > VOLUME_DECLINE_MAX:
        return None

    return Consolidation(
        top=highs.iloc[top_index],
        bottom=lows.iloc[bottom_index],
        top_index=top_index,
        bottom_index=bottom_index,
        started_at=started_at,
        pivot_highs=pivot_highs,
    )


def _volume_ratio(volumes: pd.Series, index: int) -> float:
    """How far this bar's volume stood above what was normal just before it.

    Median rather than mean: one earlier spike in the baseline window would
    otherwise raise the bar for everything after it, so the more eventful a
    symbol's recent history, the harder a genuine spike would be to see.
    """
    baseline = volumes.iloc[max(0, index - VOLUME_BASELINE_BARS) : index].median()
    if not baseline or baseline <= 0:
        return 0.0
    return volumes.iloc[index] / baseline
