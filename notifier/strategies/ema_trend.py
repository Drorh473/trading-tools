"""Strategy 2 from the user's cheatsheet ("aggressive method"): trend
continuation via EMA9/EMA20/EMA50/SMA200 stack ordering, entering when price
is near EMA9 acting as support (long, uptrend) or resistance (short,
downtrend, with an above-average volume filter).

Each instance is self-sufficient on a SINGLE timeframe: the stack ordering
and the touch/hold/shock/chop guards are all read off the same candles - the
same complete condition, whichever timeframe it's asked to run on. A second,
larger "reference" timeframe is checked only as a bonus that raises risk; it
never gates the signal. Requiring the reference timeframe to independently
show the full touch condition too - the original two-timeframe design -
produced ZERO signals across 142 symbol-weeks of real 4H/1H data: a touch on
two different scales essentially never coincides, so gating on it would have
silenced the strategy outright. Risk tiers, checked against the reference:

  base  - the timeframe's own condition alone (defers to the scanner's
          default risk_pct)
  1.5%  - the reference timeframe's own stack agrees in direction (a
          supportive trend read, not yet an independent touch)
  2%    - the reference timeframe ALSO independently passes the full
          condition (a second, genuine signal in its own right)

The reference timeframe is display-only at the lower two tiers - at 1.5% it
only appears as a prose note ("4H trend supports"), and only earns a place in
the alert's own "Analysis timeframe" line at 2%, since that is the only tier
where it is genuinely a second confirmed timeframe rather than a supportive
read. Both mechanisms stack with pattern confluence (scanner.py takes
whichever implies the higher risk) rather than one silencing the other.

The stop is always the BASE timeframe's own EMA20, regardless of tier -
a bigger picture agreeing is a reason to risk more on the trade the base
timeframe is showing, not a reason to widen the stop onto a timeframe the
trade was never actually read from.

Entry is the EMA9 level itself (the base timeframe's own ema9_prev, the
level the touch/hold checks already confirmed), not the triggering candle's
close - measured 0.27% better on longs when it fills. There is no market
fraction: a limit that never fills here means no trade rather than a worse
one, Dror's call.

The touch/hold logic reads its EMA9 level, ATR, and hold window as of the
PRIOR candle - a wide, fast-moving candle would otherwise drag its own
average toward itself and read as testing a level it just blew through. The
reference timeframe is read the same way (closed bars only, no forming
candle) for the same reason: it now runs the exact same touch/hold check the
base timeframe does at the 2% tier, so a partial candle would corrupt it
identically. This trades a little freshness on the reference read for not
reintroducing a bug already fixed once.
"""

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, ema, sma

EMA_FAST = 9
EMA_MID = 20
EMA_SLOW = 50
TREND_MA_PERIOD = 200
VOLUME_LOOKBACK = 20
ATR_PERIOD = 14
EMA9_BAND_ATR_MULTIPLE = 0.05  # a touch is within 5% of an average candle's range
EMA9_HOLD_BARS = 10  # candles holding the level before a touch counts
STOP_ATR_BUFFER = 0.10  # the stop sits below EMA20 (above, for shorts), not on it
# A candle this much bigger than its own prior ATR likely dragged EMA9 toward
# price rather than price testing an established level.
SHOCK_CANDLE_ATR_MULTIPLE = 2.0
# A level whipsawed for hours right before the hold window still passes a
# 10-bar-only check; capping crossings over a longer lookback catches a level
# that only just settled down rather than one actually respected.
CROSSING_LOOKBACK_BARS = 30
MAX_CROSSINGS_IN_LOOKBACK = 1

# Risk tiers for the reference-timeframe bonus. Base tier is None - it
# defers to the scanner's own configured default rather than hardcoding it
# here, so risk_pct stays a scanner-level setting.
TREND_SUPPORT_RISK_PCT = 0.015
BOTH_TOUCHING_RISK_PCT = 0.02


class EmaTrendFollowing(Strategy):
    """A single self-sufficient timeframe, optionally paired with a larger
    reference timeframe that only ever raises risk. reference_timeframe=None
    (the standalone 1D instance) fires at base risk only - there is nothing
    larger to check it against, and 1W/1M support wasn't worth building for a
    tier that would rarely even move the answer."""

    def __init__(self, base_timeframe: str, reference_timeframe: str | None = None):
        self.base_timeframe = base_timeframe
        self.reference_timeframe = reference_timeframe
        self.tag = (
            f"Strategy 2 {reference_timeframe}/{base_timeframe}" if reference_timeframe else f"Strategy 2 {base_timeframe}"
        )
        self.timeframes = [base_timeframe] + ([reference_timeframe] if reference_timeframe else [])

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        base_bars = bars_by_timeframe.get(self.base_timeframe)
        if base_bars is None or len(base_bars) < TREND_MA_PERIOD + 1:
            return None

        for direction in ("up", "down"):
            result = _full_condition(base_bars, direction)
            if result is None:
                continue
            entry, stop = result

            risk_pct_override = None
            analysis_timeframes = (self.base_timeframe,)
            extra_notes: tuple[str, ...] = ()

            if self.reference_timeframe:
                ref_bars = bars_by_timeframe.get(self.reference_timeframe)
                if ref_bars is not None and _trend(ref_bars) == direction:
                    risk_pct_override = TREND_SUPPORT_RISK_PCT
                    extra_notes = (f"{self.reference_timeframe} trend supports ({direction}).",)
                    if _full_condition(ref_bars, direction) is not None:
                        risk_pct_override = BOTH_TOUCHING_RISK_PCT
                        analysis_timeframes = (self.base_timeframe, self.reference_timeframe)
                        extra_notes = ()

            direction_word = "long" if direction == "up" else "short"
            stack = "EMA9 > EMA20 > EMA50 > SMA200" if direction == "up" else "SMA200 > EMA50 > EMA20 > EMA9"
            side = "above" if direction == "up" else "below"
            return Signal(
                symbol=symbol,
                direction=direction_word,
                entry_price=entry,
                stop_loss=stop,
                strategy_tag=self.tag,
                limit_entry=entry,
                limit_note="EMA9",
                market_fraction=0.0,
                risk_pct_override=risk_pct_override,
                analysis_timeframes=analysis_timeframes,
                extra_notes=extra_notes,
                reason=(
                    f"{self.base_timeframe} {stack} confirmed; price held {side} its own EMA9 for the last "
                    f"{EMA9_HOLD_BARS} candles, then came within {EMA9_BAND_ATR_MULTIPLE:g}x ATR of it and held. "
                    f"Stop is just {'below' if direction == 'up' else 'above'} the {self.base_timeframe} EMA20."
                ),
            )

        return None


def _full_condition(bars: pd.DataFrame, direction: str) -> tuple[float, float] | None:
    """(entry_price, stop_loss) if the complete condition - stack ordered in
    `direction`, EMA9 touch/hold/shock/chop all clean - holds on these bars,
    else None. The one place the stack check and the touch guards live
    together, so a base timeframe and a reference timeframe are checked
    identically."""
    if _trend(bars) != direction:
        return None
    return _touch_and_hold(bars, direction)


def _trend(bars: pd.DataFrame) -> str | None:
    """"up" or "down" when the four MAs are fully stacked on these bars, else
    None. NaN comparisons resolve to False, so bars shorter than the 200-SMA
    needs safely return None without an explicit length guard."""
    closes = bars["close"]
    fast = ema(closes, EMA_FAST).iloc[-1]
    mid = ema(closes, EMA_MID).iloc[-1]
    slow = ema(closes, EMA_SLOW).iloc[-1]
    trend_ma = sma(closes, TREND_MA_PERIOD).iloc[-1]

    if fast > mid > slow > trend_ma:
        return "up"
    if trend_ma > slow > mid > fast:
        return "down"
    return None


def _touch_and_hold(bars: pd.DataFrame, direction: str) -> tuple[float, float] | None:
    """(ema9_prev, stop_loss) if price has held the trend side of its own
    EMA9 for EMA9_HOLD_BARS candles with no shock candle and no recent chop,
    then come within band of it and closed back on the trend side - else
    None. Direction-agnostic: "up" mirrors "down" throughout, with the short
    side additionally requiring above-average volume."""
    needed = max(ATR_PERIOD, EMA9_HOLD_BARS, CROSSING_LOOKBACK_BARS) + 2
    if len(bars) < needed:
        return None

    closes = bars["close"]
    ema9_series = ema(closes, EMA_FAST)
    ema20_series = ema(closes, EMA_MID)
    atr_series = atr(bars, ATR_PERIOD)

    ema9_prev = ema9_series.iloc[-2]
    atr_prev = atr_series.iloc[-2]
    if not atr_prev or pd.isna(atr_prev):
        return None
    band = atr_prev * EMA9_BAND_ATR_MULTIPLE
    ema9_now = ema9_series.iloc[-1]
    ema20_now = ema20_series.iloc[-1]
    stop_buffer = atr_prev * STOP_ATR_BUFFER
    close_now = closes.iloc[-1]
    high_now, low_now = bars["high"].iloc[-1], bars["low"].iloc[-1]

    prior_closes = closes.iloc[-EMA9_HOLD_BARS - 1 : -1]
    prior_ema9 = ema9_series.iloc[-EMA9_HOLD_BARS - 1 : -1]

    ranges = bars["high"] - bars["low"]
    shock_threshold = atr_series.shift(1) * SHOCK_CANDLE_ATR_MULTIPLE
    no_shock_candle = not bool(
        (ranges.iloc[-EMA9_HOLD_BARS - 1 : -1] > shock_threshold.iloc[-EMA9_HOLD_BARS - 1 : -1]).any()
    )

    above_ema9 = closes > ema9_series
    crossing_window = above_ema9.iloc[-CROSSING_LOOKBACK_BARS - 1 : -1]
    recent_crossings = int((crossing_window != crossing_window.shift()).sum() - 1)
    not_recently_choppy = recent_crossings <= MAX_CROSSINGS_IN_LOOKBACK

    if direction == "up":
        held = bool((prior_closes > prior_ema9).all())
        stop = ema20_now - stop_buffer
        ok = (
            ema9_now > ema20_now
            and held
            and no_shock_candle
            and not_recently_choppy
            and abs(low_now - ema9_prev) <= band
            and close_now > ema9_prev
            and stop < ema9_prev
        )
    else:
        volumes = bars["base_vol"]
        high_volume = volumes.iloc[-1] > volumes.rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        held = bool((prior_closes < prior_ema9).all())
        stop = ema20_now + stop_buffer
        ok = (
            high_volume
            and ema9_now < ema20_now
            and held
            and no_shock_candle
            and not_recently_choppy
            and abs(high_now - ema9_prev) <= band
            and close_now < ema9_prev
            and stop > ema9_prev
        )

    return (ema9_prev, stop) if ok else None
