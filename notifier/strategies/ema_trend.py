"""Strategy 2 from the user's cheatsheet ("aggressive method"): trend
continuation via EMA9/EMA20/SMA200 stack ordering, entering when price is near
EMA9 acting as support (long, uptrend) or resistance (short, downtrend, with
an above-average volume filter).

Runs as a 1H + 15m confluence, per the user's own comparison of the two
timeframes: 1H confirms the broader trend (the MA-stack ordering), while 15m
provides the entry trigger and stop — its EMA20 sits much closer to price,
giving a materially tighter stop than reading the same condition off 1H
alone. A signal only fires when both agree, which is deliberately more
selective than either timeframe on its own.

The 1H trend reading is cached per symbol for the hour it was computed on
(keyed by that candle's own timestamp) rather than recomputed on every 15m
scan — avoids the reading silently drifting within an hour and saves
recomputing the same closed data four times.

The touch condition is a 0.5% proximity band around EMA9, not a strict
wick-through: an exact low<=EMA9<close check missed real setups where price
approached EMA9 closely without quite crossing it on that specific candle.

"Price close to resistance / far from support" for the short setup, and
"exit at nearby support" for either side, are left as manual judgment calls
at Approve/Reject time, same treatment as Strategy 1's discretionary
confirmations. Long's "at least 1:2" and short's "1:3" are both already
satisfied by the scanner-wide default, so neither side needs a reward:risk
override.
"""

import pandas as pd

from notifier.strategies.base import Signal, Strategy
from notifier.strategies.indicators import ema, sma

EMA_FAST = 9
EMA_MID = 20
TREND_MA_PERIOD = 200
VOLUME_LOOKBACK = 20
EMA9_PROXIMITY_PCT = 0.005  # within 0.5% of EMA9 counts as a touch


class EmaTrendFollowing(Strategy):
    tag = "Strategy 2"
    timeframes = ["1H", "15m"]

    def __init__(self):
        self._trend_cache: dict[str, tuple[str, str | None]] = {}

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars_1h = bars_by_timeframe.get("1H")
        bars_15m = bars_by_timeframe.get("15m")
        if bars_1h is None or bars_15m is None or len(bars_1h) < TREND_MA_PERIOD + 1:
            return None

        trend = self._cached_trend(symbol, bars_1h)
        if trend is None:
            return None

        closes = bars_15m["close"]
        ema9_now = ema(closes, EMA_FAST).iloc[-1]
        ema20_now = ema(closes, EMA_MID).iloc[-1]
        close_now, high_now, low_now = closes.iloc[-1], bars_15m["high"].iloc[-1], bars_15m["low"].iloc[-1]
        band = ema9_now * EMA9_PROXIMITY_PCT

        if trend == "up" and abs(low_now - ema9_now) <= band and close_now > ema9_now:
            return Signal(
                symbol=symbol,
                direction="long",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    "1H uptrend (EMA9 > EMA20 > SMA200) confirmed; price came within 0.5% of "
                    "15m EMA9 and held as support. Stop is 15m EMA20."
                ),
            )

        volumes = bars_15m["base_vol"]
        avg_volume = volumes.rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        high_volume = volumes.iloc[-1] > avg_volume

        if trend == "down" and high_volume and abs(high_now - ema9_now) <= band and close_now < ema9_now:
            return Signal(
                symbol=symbol,
                direction="short",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    "1H downtrend (SMA200 > EMA20 > EMA9) confirmed with above-average 15m volume; "
                    "price came within 0.5% of 15m EMA9 and was rejected as resistance. Stop is 15m EMA20."
                ),
            )

        return None

    def _cached_trend(self, symbol: str, bars_1h: pd.DataFrame) -> str | None:
        last_ts = str(bars_1h["ts"].iloc[-1])
        cached = self._trend_cache.get(symbol)
        if cached and cached[0] == last_ts:
            return cached[1]

        closes = bars_1h["close"]
        ema9_now = ema(closes, EMA_FAST).iloc[-1]
        ema20_now = ema(closes, EMA_MID).iloc[-1]
        sma200_now = sma(closes, TREND_MA_PERIOD).iloc[-1]

        if ema9_now > ema20_now > sma200_now:
            trend = "up"
        elif sma200_now > ema20_now > ema9_now:
            trend = "down"
        else:
            trend = None

        self._trend_cache[symbol] = (last_ts, trend)
        return trend
