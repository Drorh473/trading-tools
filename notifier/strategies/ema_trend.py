"""Strategy 2 from the user's cheatsheet ("aggressive method"): trend
continuation via EMA9/EMA20/SMA200 stack ordering, entering on price holding
EMA9 as support (long, uptrend) or rejecting it as resistance (short,
downtrend, with an above-average volume filter).

"Price close to resistance / far from support" for the short setup is left
as a manual judgment call at Approve/Reject time — same treatment as
Strategy 1's discretionary confirmations, since it's not reducible to a
single mechanical rule without a full support/resistance-level detector.
Likewise "exit at nearby support" is left to the user; the mechanical
target used here is the cheatsheet's 1:3 fallback (long's "at least 1:2"
is already satisfied by the scanner-wide 1:3 default, so neither side needs
a reward:risk override).
"""

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import ema, sma

EMA_FAST = 9
EMA_MID = 20
TREND_MA_PERIOD = 200
VOLUME_LOOKBACK = 20


class EmaTrendFollowing(Strategy):
    tag = "ema_trend_following"

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        if len(bars) < TREND_MA_PERIOD + 1:
            return None

        closes = bars["close"]
        highs = bars["high"]
        lows = bars["low"]
        volumes = bars["base_vol"]

        ema9_now = ema(closes, EMA_FAST).iloc[-1]
        ema20_now = ema(closes, EMA_MID).iloc[-1]
        sma200_now = sma(closes, TREND_MA_PERIOD).iloc[-1]
        close_now, high_now, low_now = closes.iloc[-1], highs.iloc[-1], lows.iloc[-1]

        uptrend = ema9_now > ema20_now > sma200_now
        downtrend = sma200_now > ema20_now > ema9_now

        if uptrend and low_now <= ema9_now < close_now:
            return Signal(
                symbol=symbol,
                direction="long",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    "Uptrend (EMA9 > EMA20 > SMA200); price wicked into EMA9 and "
                    "held as support. Stop below EMA20."
                ),
            )

        avg_volume = volumes.rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        high_volume = volumes.iloc[-1] > avg_volume

        if downtrend and high_volume and high_now >= ema9_now > close_now:
            return Signal(
                symbol=symbol,
                direction="short",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    "Downtrend (SMA200 > EMA20 > EMA9) with above-average volume; "
                    "price rejected EMA9 as resistance. Stop above EMA20. Exit at "
                    "nearby support if it comes first, otherwise the 1:3 target."
                ),
            )

        return None
