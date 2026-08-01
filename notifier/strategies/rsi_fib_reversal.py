"""Strategy 1 from the user's cheatsheet: RSI(10) mean-reversion at 30/70,
filtered by a 200-period trend MA, with entry/stop taken from Fibonacci
retracement levels off the current trend leg. Meant for 1h+ timeframes.

The Fib swing is anchored on the actual pivot that started the current
trend leg. Finding that pivot is the delicate part, and two earlier versions
got it wrong. A close-based fixed-50-bar window diverged materially from how
a chartist draws this. Replacing it with the wick extremes over a 200-bar
window fixed that, but taking the plain max/min of the window has no notion
of structure: it answers "what was the extreme?" when the strategy needs
"where did the current move begin?". Those coincide only when one continuous
trend spans the whole lookback. On a symbol that gapped — common for the
tokenized stocks in the watchlist, where earnings and corporate actions
produce sharp discrete breaks that crypto rarely shows — it drew a Fib
straddling the gap, anchoring the leg in a price regime the market had
already left and putting entry 15% from anything currently trading.

So pivots now come from a ZigZag: an extreme is only promoted to a pivot
once price reverses away from it by at least SWING_ATR_MULTIPLE x ATR(14),
which makes a gap or crash terminate the leg before it rather than be
absorbed into one giant leg spanning both regimes. The threshold scales with
each symbol's own volatility, since the watchlist mixes assets whose normal
daily range differs by an order of magnitude. If no pivot is found the
strategy declines to signal rather than falling back to the window edge —
without a visible reversal there is no honest place to anchor the
retracement.

Finding the pivot is only half of it: the leg also has to still be the
structure the market is currently in. Taking the last *high* pivot for a
short says nothing about whether that down-move is still running, so the
code would keep drawing a Fib off an anchor price had already retraced 83%,
97%, even 101% of — a leg no longer visible on the chart, giving an entry
and a stop the market had traded straight through hours earlier. So the
anchor must also be the most recent confirmed pivot of *either* kind. Once
the opposite pivot confirms, the reversal has exceeded the same ZigZag
threshold, the leg is over, and there is no retracement left to trade. That
costs no new constant and makes each symbol offer a setup in one direction
at a time, which is the only coherent reading: a market is either retracing
an up-leg or a down-leg, never both at once.

Two things the cheatsheet describes as manual judgment calls, not mechanical
gates, are deliberately left out of evaluate() and instead surfaced in the
signal's `reason` text for the human Approve/Reject step: checking the
higher-timeframe trend isn't fought too hard, and RSI divergence (the
cheatsheet calls it "optimal", not required).

Entry is the 61.8% Fib level (the limit-order portion of the cheatsheet's
20% market / 80% limit split entry — that split itself is left to the user
to execute manually, since this project doesn't place orders yet). Stop is
the 78.6% Fib level. Reward:risk is 1:2, per the cheatsheet, overriding the
scanner-wide default.
"""

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import atr, rsi, sma

TREND_MA_PERIOD = 200
RSI_PERIOD = 10
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SWING_MAX_LOOKBACK = 200  # how far back to search for the trend leg's pivot
ATR_PERIOD = 14
SWING_ATR_MULTIPLE = 6.0  # how far price must reverse to confirm a swing pivot
FIB_ENTRY = 0.618
FIB_STOP = 0.786
REWARD_RISK_RATIO = 2.0
MARKET_ENTRY_FRACTION = 0.2  # cheatsheet's split entry: ~20% at market, ~80% resting


class RsiFibReversal(Strategy):
    tag = "Strategy 1"
    timeframes = ["1H"]

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars = bars_by_timeframe["1H"]
        if len(bars) < TREND_MA_PERIOD + 1:
            return None  # not enough history for the 200-period trend filter

        closes = bars["close"]
        trend_ma = sma(closes, TREND_MA_PERIOD)
        rsi_series = rsi(closes, RSI_PERIOD)

        price = closes.iloc[-1]
        ma_now = trend_ma.iloc[-1]
        rsi_now = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2]

        crossed_below_oversold = rsi_prev >= RSI_OVERSOLD and rsi_now < RSI_OVERSOLD
        crossed_above_overbought = rsi_prev <= RSI_OVERBOUGHT and rsi_now > RSI_OVERBOUGHT

        if price > ma_now and crossed_below_oversold:
            swing = _uptrend_leg(bars)
            if swing is None:
                return None
            swing_low, swing_high = swing
            swing_range = swing_high - swing_low
            entry = swing_high - swing_range * FIB_ENTRY
            stop = swing_high - swing_range * FIB_STOP
            return Signal(
                symbol=symbol,
                direction="long",
                entry_price=entry,
                stop_loss=stop,
                strategy_tag=self.tag,
                reward_risk_ratio=REWARD_RISK_RATIO,
                limit_entry=entry,
                limit_note=f"{FIB_ENTRY:.1%} Fib",
                market_fraction=MARKET_ENTRY_FRACTION,
                reason=(
                    "RSI(10) crossed below 30 above the 200-MA trend filter. "
                    "Stop is the 78.6% Fib level. Check for RSI divergence and "
                    "higher-timeframe trend conflicts before approving."
                ),
            )

        if price < ma_now and crossed_above_overbought:
            swing = _downtrend_leg(bars)
            if swing is None:
                return None
            swing_low, swing_high = swing
            swing_range = swing_high - swing_low
            entry = swing_low + swing_range * FIB_ENTRY
            stop = swing_low + swing_range * FIB_STOP
            return Signal(
                symbol=symbol,
                direction="short",
                entry_price=entry,
                stop_loss=stop,
                strategy_tag=self.tag,
                reward_risk_ratio=REWARD_RISK_RATIO,
                limit_entry=entry,
                limit_note=f"{FIB_ENTRY:.1%} Fib",
                market_fraction=MARKET_ENTRY_FRACTION,
                reason=(
                    "RSI(10) crossed above 70 below the 200-MA trend filter. "
                    "Stop is the 78.6% Fib level. Check for RSI divergence and "
                    "higher-timeframe trend conflicts before approving."
                ),
            )

        return None


def _uptrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the up-move being retraced: the pivot low
    that started the leg, and the highest high since."""
    window, pivots = _swing_context(bars)
    anchor = _live_anchor(pivots, is_high=False)
    if anchor is None:
        return None
    swing_low = window["low"].iloc[anchor]
    swing_high = window["high"].iloc[anchor:].max()
    if swing_high <= swing_low:
        return None
    return swing_low, swing_high


def _downtrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the down-move being retraced: the pivot high
    that started the leg, and the lowest low since."""
    window, pivots = _swing_context(bars)
    anchor = _live_anchor(pivots, is_high=True)
    if anchor is None:
        return None
    swing_high = window["high"].iloc[anchor]
    swing_low = window["low"].iloc[anchor:].min()
    if swing_high <= swing_low:
        return None
    return swing_low, swing_high


def _swing_context(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, bool]]]:
    """The lookback window plus its ZigZag pivots. ATR is measured over the
    full history so the threshold is warmed up at the window's first bar."""
    thresholds = atr(bars, ATR_PERIOD) * SWING_ATR_MULTIPLE
    window = bars.iloc[-SWING_MAX_LOOKBACK:].reset_index(drop=True)
    thresholds = thresholds.iloc[-SWING_MAX_LOOKBACK:].reset_index(drop=True)
    return window, _zigzag_pivots(window, thresholds)


def _live_anchor(pivots: list[tuple[int, bool]], is_high: bool) -> int | None:
    """The leg's anchor pivot, but only while that leg is still the operative
    structure.

    Returns None once a pivot of the opposite kind has confirmed, because that
    confirmation means price reversed past the same ZigZag threshold: the leg
    is finished and its retracement is no longer something to trade. Without
    this the anchor persists until some new same-kind pivot happens to form,
    which can be dozens of bars later and hundreds of percent of the leg away.
    """
    if not pivots or pivots[-1][1] != is_high:
        return None
    return pivots[-1][0]


def _zigzag_pivots(window: pd.DataFrame, thresholds: pd.Series) -> list[tuple[int, bool]]:
    """Indices of swing pivots, oldest first, as (index, is_high).

    An extreme is only promoted to a pivot once price reverses away from it by
    at least the local ATR threshold, so a crash or gap terminates the leg
    before it instead of being absorbed into one giant leg spanning both price
    regimes. Two guards matter: the reversal must land on a *later* bar than
    the extreme (otherwise one wide candle confirms itself off its own wick),
    and the window's first bar is never a pivot (we can't see what came before
    it, so it is a boundary artifact rather than observed structure).
    """
    pivots: list[tuple[int, bool]] = []
    falling = None
    high_idx, high_price = 0, window["high"].iloc[0]
    low_idx, low_price = 0, window["low"].iloc[0]

    for i in range(1, len(window)):
        high, low = window["high"].iloc[i], window["low"].iloc[i]
        threshold = thresholds.iloc[i]

        if falling is not True:
            if high >= high_price:
                high_idx, high_price = i, high
            elif i > high_idx and high_price - low >= threshold:
                if high_idx > 0:
                    pivots.append((high_idx, True))
                falling = True
                low_idx, low_price = i, low
                continue

        if falling is not False:
            if low <= low_price:
                low_idx, low_price = i, low
            elif i > low_idx and high - low_price >= threshold:
                if low_idx > 0:
                    pivots.append((low_idx, False))
                falling = False
                high_idx, high_price = i, high

    return pivots
