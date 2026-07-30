"""Strategy 1 from the user's cheatsheet: RSI(10) mean-reversion at 30/70,
filtered by a 200-period trend MA, with entry/stop taken from Fibonacci
retracement levels off the recent swing high/low. Meant for 1h+ timeframes.

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
SWING_LOOKBACK = 50  # bars used to find the swing high/low for the Fib anchor
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

        window = closes.iloc[-SWING_LOOKBACK:]
        swing_high = window.max()
        swing_low = window.min()
        swing_range = swing_high - swing_low
        if swing_range == 0:
            return None

        crossed_below_oversold = rsi_prev >= RSI_OVERSOLD and rsi_now < RSI_OVERSOLD
        crossed_above_overbought = rsi_prev <= RSI_OVERBOUGHT and rsi_now > RSI_OVERBOUGHT

        if price > ma_now and crossed_below_oversold:
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
