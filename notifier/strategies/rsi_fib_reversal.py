"""Strategy 1 from the user's cheatsheet: RSI(10) mean-reversion at 30/70,
filtered by a 200-period trend MA, with entry/stop taken from Fibonacci
retracement levels off the current trend leg. Meant for 1h+ timeframes.

The Fib swing is anchored on the actual pivot that started the current
trend leg, not a fixed lookback window: for a long, that's the last low
before the most recent high (the impulse move being retraced); for a short,
the mirror. Swing extremes use each candle's real high/low, not its close —
a candle can wick well past its close, and that wick is the real extreme.
An earlier close-based, fixed-50-bar version diverged materially from how
a chartist actually draws this (verified directly against the user's own
chart: the pivot-based approach matched it almost exactly, the fixed-window
version put the entry noticeably off).

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
from notifier.strategies.indicators import rsi, sma

TREND_MA_PERIOD = 200
RSI_PERIOD = 10
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SWING_MAX_LOOKBACK = 200  # how far back to search for the trend leg's pivot
FIB_ENTRY = 0.618
FIB_STOP = 0.786
REWARD_RISK_RATIO = 2.0


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
                reason=(
                    "RSI(10) crossed below 30 above the 200-MA trend filter. "
                    "Enter ~20% at market, ~80% as a limit at this 61.8% Fib level; "
                    "stop is the 78.6% Fib level. Check for RSI divergence and "
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
                reason=(
                    "RSI(10) crossed above 70 below the 200-MA trend filter. "
                    "Enter ~20% at market, ~80% as a limit at this 61.8% Fib level; "
                    "stop is the 78.6% Fib level. Check for RSI divergence and "
                    "higher-timeframe trend conflicts before approving."
                ),
            )

        return None


def _uptrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the up-move being retraced: the most recent
    peak, and the lowest low before it (where that leg started)."""
    window = bars.iloc[-SWING_MAX_LOOKBACK:].reset_index(drop=True)
    peak_idx = window["high"].idxmax()
    if peak_idx == 0:
        return None  # peak is the first bar in the window; no prior low to anchor on
    swing_high = window["high"].iloc[peak_idx]
    swing_low = window["low"].iloc[: peak_idx + 1].min()
    if swing_high <= swing_low:
        return None
    return swing_low, swing_high


def _downtrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the down-move being retraced: the most
    recent trough, and the highest high before it (where that leg started)."""
    window = bars.iloc[-SWING_MAX_LOOKBACK:].reset_index(drop=True)
    trough_idx = window["low"].idxmin()
    if trough_idx == 0:
        return None
    swing_low = window["low"].iloc[trough_idx]
    swing_high = window["high"].iloc[: trough_idx + 1].max()
    if swing_high <= swing_low:
        return None
    return swing_low, swing_high
