"""Strategy 2 v2 - the EMA9 pullback, with the target decoupled from the stop.

v1 is a single self-sufficient timeframe whose target is a multiple of its own
stop, so its reward:risk is 1:3 by construction and cannot be anything else.
v2 splits the job for PAIRED instances:

  higher timeframe  - owns the trend and the TARGET
  lower timeframe   - owns the trigger and the STOP

Because the target no longer scales with the stop, reward:risk becomes a free
variable. Measured on 25 symbols of the cached year: median 5.4:1 against v1's
2:1, upper quartile 14.5:1.

STANDALONE instances have no higher timeframe and keep the v1 geometry - their
targets are 1:2 and 1:3 of their own stop.

WHAT THE FIRST MEASUREMENT SAYS, so nobody re-derives it from scratch: scanned
over 38 symbols this fires ~2,500 times a year and wins 10.5% of the time, and
ALL of its positive expectancy sits in trades whose stop is under 0.15% of
price - which is inside the spread and models no slippage. Every bucket with a
holdable stop measured negative. MIN_STOP_PCT exists to bound exactly that, and
is meant to be swept, not trusted at its default.

Differences from v1, all deliberate (see strategy2-v2-spec.md for the session
that decided each one):

  removed  - the 10-candle hold, the chop filter, the shock-candle filter, the
             short-side volume filter, and the 0.05 x ATR touch band
  trigger  - a limit PRE-PLACED at the last closed bar's EMA9, filling on first
             touch, with no confirming close
  trend    - strict last-3: three ascending highs AND three ascending lows, on
             the higher timeframe for a pair. "Cannot tell" now BLOCKS the
             trade, where v1 treated it as permission
  risk     - flat, no reference-timeframe tiers
  fees     - maker in, taker out at the stop: 0.08%, not 0.12%
"""

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, ema, sma
from notifier.strategies.structure import zigzag_pivots

EMA_FAST, EMA_MID, EMA_SLOW, TREND_MA_PERIOD = 9, 20, 50, 200
ATR_PERIOD = 14
STOP_ATR_BUFFER = 0.10

# THE EMA9 MUST BE ACTING AS SUPPORT (long) OR RESISTANCE (short), not merely
# be a line price happens to be crossing.
#
# A one-candle touch test - low <= EMA9 and close > EMA9 - cannot tell these
# two apart, because both satisfy it:
#
#   price above the EMA9, dips to it, closes back above   <- the setup
#   price below the EMA9, rallies UP through it, closes above  <- NOT the setup
#
# The second is a broken resistance, and the first cut of v2 took those trades.
# Dror, reading the ETHUSDT and 1000RATSUSDT charts: "if it long the ema9 is
# supposed to be support not resistence". On ETH's 4H the EMA9 had spent forty
# bars below a flat base and raced up underneath a vertical rally, so it had
# been "support" for about three candles when the trade fired.
#
# This is v1's ten-candle hold, which was removed in the redesign as a
# staleness filter without noticing it was ALSO the statement that the level is
# respected. Restored deliberately, and comparing each close to its OWN EMA9 at
# that bar rather than to today's level, because the level moves.
#
# FIVE, AND ONLY ON THE TIMEFRAME THAT SETS THE TREND. Both halves of that are
# measured (backtest/sweep_v2.py over 16,198 generated setups):
#
#   Requiring it on BOTH timeframes of a pair is fatal. Paired setups run 219
#   at no hold, 47 at one bar, 5 at three, and ZERO at ten - and paired is the
#   only place v2's target is decoupled from its stop, which is the whole
#   reason the strategy exists. Ten-on-both, which this file briefly shipped,
#   silently reduced v2 to a standalone-only strategy.
#
#   On the reference alone the paired population survives (116 setups at five)
#   and its expectancy stops being noise: +13.3R with SE 8.3 at no hold - a few
#   tiny-denominator outliers - becomes +1.36R with SE 0.55 at five. Smaller,
#   and the first paired number in this exercise distinguishable from zero.
#
# The cost, stated because it was measured and went the other way: on the
# STANDALONE population the hold curve falls monotonically, +0.470R at zero to
# +0.375R at five to +0.334R at ten, with no peak anywhere. The harness models
# no slippage, and the trades a hold removes are entries into fast moves
# through a level - precisely where slippage is worst - so that curve is read
# as "this measurement cannot price what the filter removes" rather than as
# "the filter is worthless". Dror's call, made with the table in front of him.
EMA9_HOLD_BARS = 5

# Pivot scale for the last-3 read. Same value Strategy 1 and Strategy 4 use, so
# "what counts as a swing" has one definition across the project even though
# v2 answers "what trend is this" differently from them.
STRUCTURE_ATR_MULTIPLE = 1.25
# How far back the pivot search looks. Measured: the last three highs and lows
# span a median of 27 bars on 4H and at most 76, so this is roughly 2.5x the
# worst case and is a bound, not a tuned parameter.
STRUCTURE_LOOKBACK = 200

# The higher timeframe's target, as multiples of ITS OWN stop distance. The
# prices these produce are absolute and do not move when the lower timeframe
# offers a better entry - that is the whole point of the split.
TARGET_1_RATIO, TARGET_2_RATIO = 2.0, 3.0
PARTIAL_FRACTION = 0.5

# Maker in (0.02%) + taker out at the stop (0.06%). NOT 0.12%: this strategy
# never places a market order, so its entry is always a maker fill. Same
# correction as v1's ROUND_TRIP_FEE_PCT.
ROUND_TRIP_FEE_PCT = 0.0008
MAKER_FEE_PCT = 0.0002

# v1 gated on fee-as-a-fraction-of-RISK, which only proxies for expectancy when
# reward:risk is roughly constant. Here it ranges 2:1 to 14:1, so that gate
# refused 56% of signals and refused the highest-R:R ones preferentially. This
# asks the economic question directly: after fees, is the trade still worth it.
#
# NOT 2.0, and the reason is arithmetic rather than taste. A STANDALONE
# instance targets 1:2 of its own stop, so its gross ratio is exactly 2.0 and
# its net ratio - (2r - maker) / (r + fees) - approaches 2.0 from BELOW for
# every stop width and never reaches it. A 2.0 floor silently zeroes four of
# the seven instances, which is what the first generation run did. Swept, not
# trusted.
MIN_NET_REWARD_RISK = 1.5

# THE GATE THAT MATTERS, and the one with no evidence behind its value. A stop
# tighter than the spread is not a stop, it is a coin flip with a flattering
# denominator - and because R is measured against it, such trades report
# enormous R:R and enormous expectancy while being unholdable in practice.
# The first scan found +26.9R average below 0.15% and negative everywhere a
# trade could actually be held. SWEEP THIS. Do not ship it at a guessed value.
MIN_STOP_PCT = 0.003
# The mirror case, unchanged from v1: EMA20 lagging a crash puts the stop
# absurdly far away and the setup is no longer an orderly pullback.
MAX_STOP_PCT = 0.20


class EmaTrendV2(Strategy):
    """One instance. `reference_timeframe=None` makes it standalone, which
    keeps v1's own-R target geometry; a reference makes it paired, and the
    reference then owns both the trend read and the target."""

    def __init__(self, base_timeframe: str, reference_timeframe: str | None = None):
        self.base_timeframe = base_timeframe
        self.reference_timeframe = reference_timeframe
        self.paired = reference_timeframe is not None
        self.tag = (
            f"Strategy 2.1 {reference_timeframe}/{base_timeframe}"
            if self.paired
            else f"Strategy 2.1 {base_timeframe}"
        )
        self.timeframes = [base_timeframe] + ([reference_timeframe] if self.paired else [])
        # The higher timeframe is read on its FORMING candle. Measured: when a
        # 4H touch and a 1H trigger coincide, the 1H trigger fires a median of
        # two hours BEFORE the 4H bar closes, and 44% of them land in its first
        # hour. Waiting for the close means entering after the move the whole
        # design exists to catch. Its EMA LEVELS still come from the last
        # closed bar, so the target cannot drift while the candle builds.
        self.forming_bar_timeframes = (reference_timeframe,) if self.paired else ()
        # A pair replaces its own base timeframe's standalone instance when both
        # fire on one symbol - the same entry and stop, better informed. They
        # coincide on 26% of standalone triggers, so without this the account
        # carries 2% on one idea in two correlated positions.
        self.supersedes = (f"Strategy 2.1 {base_timeframe}",) if self.paired else ()

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        base = bars_by_timeframe.get(self.base_timeframe)
        if base is None or len(base) < ATR_PERIOD + 3:
            return None

        if self.paired:
            forming = bars_by_timeframe.get(self.reference_timeframe)
            if forming is None or len(forming) < TREND_MA_PERIOD + 2:
                return None
            closed = forming.iloc[:-1]  # levels and structure from CLOSED bars only
        else:
            if len(base) < TREND_MA_PERIOD + 2:
                return None
            # The current bar plays the "forming" role and everything before it
            # is closed, mirroring the paired case exactly. Reading levels off
            # the current bar instead would let a wide candle drag its own EMA9
            # toward itself and read as testing a level it just blew through -
            # v1's prior-bar discipline, kept - and would also disagree with
            # _trigger, which sizes the trade from the PREVIOUS bar's EMA9.
            forming, closed = base, base.iloc[:-1]

        levels = _levels(closed)
        if levels is None:
            return None
        ref9, ref20 = levels

        trend = _stack(closed)
        if trend is None:
            return None

        # BOTH timeframes must show the condition - Dror's clarification: "if
        # the setup is a paired one, in both of them have to be the conditions
        # for the trade". Not the higher timeframe setting a target while the
        # lower one merely touches a line.
        #
        # The HOLD is the one part asked of the reference alone. It is the
        # statement that this EMA9 is respected, which is a claim about the
        # trend - and the reference owns the trend. Demanding it of the trigger
        # timeframe too takes paired setups to zero; see EMA9_HOLD_BARS.
        #
        # This is NOT v1's zero-signal design. That required both timeframes to
        # pass v1's full condition INCLUDING its 0.05 x ATR proximity band, and
        # a touch that tight never coincides across two scales. v2's touch is
        # an actual touch, and 85% of 4H touches have a 1H touch inside the
        # same candle - so the two agreeing is common rather than impossible.
        if not _full_condition(forming, closed, ref9, trend, require_hold=True):
            return None
        if self.paired:
            base_closed = base.iloc[:-1]
            base_levels = _levels(base_closed)
            if base_levels is None:
                return None
            if not _full_condition(base, base_closed, base_levels[0], trend, require_hold=False):
                return None

        trigger = _trigger(base, trend)
        if trigger is None:
            return None
        entry, stop = trigger

        risk = abs(entry - stop)
        stop_fraction = risk / entry
        if not MIN_STOP_PCT <= stop_fraction <= MAX_STOP_PCT:
            return None

        if self.paired:
            gap = abs(ref9 - ref20)
            sign = 1.0 if trend == "up" else -1.0
            target_1 = ref9 + sign * TARGET_1_RATIO * gap
            target_2 = ref9 + sign * TARGET_2_RATIO * gap
        else:
            sign = 1.0 if trend == "up" else -1.0
            target_1 = entry + sign * TARGET_1_RATIO * risk
            target_2 = entry + sign * TARGET_2_RATIO * risk

        reward = (target_1 - entry) if trend == "up" else (entry - target_1)
        if reward <= 0:
            return None
        # Fees on the way in are maker; the leg this gate must survive is the
        # STOP, which is taker. Both are charged against notional, so both are
        # expressed as a fraction of the entry price.
        net = (reward - MAKER_FEE_PCT * entry) / (risk + ROUND_TRIP_FEE_PCT * entry)
        if net < MIN_NET_REWARD_RISK:
            return None

        direction = "long" if trend == "up" else "short"
        return Signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            strategy_tag=self.tag,
            # Carried as a ratio because that is what the sizing path speaks,
            # but it is DERIVED from an absolute price the higher timeframe
            # set, not chosen. remainder_target carries the second tier as the
            # price it actually is.
            reward_risk_ratio=reward / risk,
            partial_fraction=PARTIAL_FRACTION,
            remainder_target=target_2,
            # Final, not a fallback. Without this the scanner puts the runner at
            # the nearest daily swing level (Strategy 3's rule) and the higher
            # timeframe's 1:3 - the price the measured 8:1 describes - is
            # discarded.
            remainder_target_is_final=True,
            remainder_note=(
                f"{self.reference_timeframe} 1:{TARGET_2_RATIO:g}" if self.paired else f"1:{TARGET_2_RATIO:g}"
            ),
            limit_entry=entry,
            limit_note="EMA9",
            # No market portion, and this is load-bearing rather than
            # incidental: measured, a limit fill at EMA9 is 7.1:1 while a
            # market entry at the same bar's close is 2.3:1. Entering at market
            # costs two thirds of the edge.
            market_fraction=0.0,
            analysis_timeframes=(
                (self.reference_timeframe, self.base_timeframe) if self.paired else (self.base_timeframe,)
            ),
            dedupe_key=(symbol, self.tag, direction, round(entry, 10)),
            reason=(
                f"{self.reference_timeframe or self.base_timeframe} stack and last-3 structure both "
                f"{trend}, price touching its EMA9; {self.base_timeframe} limit at EMA9 "
                f"{entry:.8g}, stop {stop:.8g} ({100 * stop_fraction:.2f}% of price), "
                f"targets {target_1:.8g} / {target_2:.8g} ({reward / risk:.1f}R, {net:.1f}R net)"
            ),
        )


def _levels(closed: pd.DataFrame) -> tuple[float, float] | None:
    """(EMA9, EMA20) from the last CLOSED bar, or None if not computable."""
    if len(closed) < EMA_MID + 2:
        return None
    e9 = ema(closed["close"], EMA_FAST).iloc[-1]
    e20 = ema(closed["close"], EMA_MID).iloc[-1]
    if pd.isna(e9) or pd.isna(e20) or e9 <= 0:
        return None
    return float(e9), float(e20)


def _stack(bars: pd.DataFrame) -> str | None:
    """"up"/"down" when the four MAs are fully ordered, else None. Unchanged
    from v1: NaN comparisons resolve False, so short histories return None
    without a length guard."""
    c = bars["close"]
    fast, mid = ema(c, EMA_FAST).iloc[-1], ema(c, EMA_MID).iloc[-1]
    slow, trend_ma = ema(c, EMA_SLOW).iloc[-1], sma(c, TREND_MA_PERIOD).iloc[-1]
    if fast > mid > slow > trend_ma:
        return "up"
    if trend_ma > slow > mid > fast:
        return "down"
    return None


def _last3_trend(bars: pd.DataFrame) -> str | None:
    """Strict last-3: the three most recent swing highs AND the three most
    recent swing lows must all be monotonic in the same direction.

    None is NOT permission here - the caller refuses the trade. v1's reader
    returned None in 0% of 832 sampled reads, so treating absence as
    permission was safe; this one abstains 78% of the time, and the same
    semantics would turn the gate into a no-op and reopen the MUUUSDT hole it
    exists to close.
    """
    w = bars.iloc[-STRUCTURE_LOOKBACK:].reset_index(drop=True)
    if len(w) < 30:
        return None
    thresholds = (atr(w, ATR_PERIOD) * STRUCTURE_ATR_MULTIPLE).reset_index(drop=True)
    pivots = zigzag_pivots(w, thresholds)
    highs = [float(w["high"].iloc[i]) for i, is_high in pivots if is_high][-3:]
    lows = [float(w["low"].iloc[i]) for i, is_high in pivots if not is_high][-3:]
    if len(highs) < 3 or len(lows) < 3:
        return None
    if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
        return "up"
    if highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
        return "down"
    return None


def _full_condition(
    forming: pd.DataFrame, closed: pd.DataFrame, level: float, trend: str, require_hold: bool
) -> bool:
    """The condition on ONE timeframe: structure agrees, price is touching its
    EMA9 now, and - where asked - that EMA9 has been held.

    The one place these live together, so a base timeframe and a reference
    timeframe are checked identically apart from the hold, which only the
    trend-setting timeframe must prove.
    """
    return (
        _last3_trend(closed) == trend
        and _touching(forming, level, trend)
        and (not require_hold or _holding(closed, trend))
    )


def hold_run(closed: pd.DataFrame, trend: str, cap: int = 120) -> int:
    """How many consecutive closes, counting back from the most recent one,
    sat on the trend side of their OWN EMA9.

    The measurable form of "the EMA9 has been acting as support". Returned as a
    COUNT rather than a yes/no so the threshold can be swept after the fact:
    generate once with the filter off, record the run each setup actually had,
    and every candidate value of EMA9_HOLD_BARS is then a filter over the same
    population rather than another full generation.

    Compared bar by bar against the EMA9 as it was THEN. Comparing every close
    to today's level would call a rising EMA9 "held" simply because price rose
    faster than it did, which is the ETHUSDT case this exists to refuse.
    """
    if len(closed) < 3:
        return 0
    e9 = ema(closed["close"], EMA_FAST)
    on_side = (closed["close"] > e9) if trend == "up" else (closed["close"] < e9)
    run = 0
    for value in on_side.values[::-1]:
        if not value:
            break
        run += 1
        if run >= cap:
            break
    return run


def _holding(closed: pd.DataFrame, trend: str) -> bool:
    """Whether the EMA9 has held for at least EMA9_HOLD_BARS closes."""
    if EMA9_HOLD_BARS <= 0:
        return True
    if len(closed) < EMA9_HOLD_BARS + 2:
        return False
    return hold_run(closed, trend, cap=EMA9_HOLD_BARS) >= EMA9_HOLD_BARS


def _touching(forming: pd.DataFrame, level: float, trend: str) -> bool:
    """Whether the current (possibly still forming) candle has reached the
    level and is trading back on the trend side of it.

    On its own this admits a broken resistance as readily as a respected
    support - see EMA9_HOLD_BARS. It is only ever called with _holding().
    """
    last = forming.iloc[-1]
    if trend == "up":
        return bool(last["low"] <= level and last["close"] > level)
    return bool(last["high"] >= level and last["close"] < level)


def _trigger(base: pd.DataFrame, trend: str) -> tuple[float, float] | None:
    """(entry, stop) if a limit resting at the previous bar's EMA9 would have
    been touched by the current bar, else None.

    Everything comes from the PRIOR bar. A limit cannot rest at a level
    computed from the close of the candle that fills it, and measuring it that
    way is the lookahead that made the first pass of this variant look better
    than it is.
    """
    if len(base) < max(ATR_PERIOD, EMA_MID) + 3:
        return None
    closes = base["close"]
    e9_prev = ema(closes, EMA_FAST).iloc[-2]
    e20_prev = ema(closes, EMA_MID).iloc[-2]
    atr_prev = atr(base, ATR_PERIOD).iloc[-2]
    if any(pd.isna(x) for x in (e9_prev, e20_prev, atr_prev)) or atr_prev <= 0 or e9_prev <= 0:
        return None

    last = base.iloc[-1]
    buffer = float(atr_prev) * STOP_ATR_BUFFER
    if trend == "up":
        if not last["low"] <= e9_prev:
            return None
        stop = float(e20_prev) - buffer
        if stop >= e9_prev:
            return None
    else:
        if not last["high"] >= e9_prev:
            return None
        stop = float(e20_prev) + buffer
        if stop <= e9_prev:
            return None
    return float(e9_prev), stop


# The seven instances, as agreed. Four standalone, three paired. The 1D
# standalone survives from v1's list; a 1D/1W pair was considered and dropped
# in favour of keeping the 15m end, which is where the tight stops - and the
# whole R:R thesis - actually live.
#
# NOTE the 15m pair cannot be backtested: Bitget serves ~22 days of 15m and ~2
# of 5m, and the cached year holds 1H bars only (4H and 1D resample from them,
# 15m cannot).
INSTANCES: tuple[tuple[str, str | None], ...] = (
    ("15m", None),
    ("1H", None),
    ("4H", None),
    ("1D", None),
    ("15m", "1H"),
    ("1H", "4H"),
    ("4H", "1D"),
)


def build_instances() -> list[EmaTrendV2]:
    return [EmaTrendV2(base, ref) for base, ref in INSTANCES]
