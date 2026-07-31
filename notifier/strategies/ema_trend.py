"""Strategy 2 from the user's cheatsheet ("aggressive method"): trend
continuation via EMA9/EMA20/EMA50/SMA200 stack ordering, entering when price is
near EMA9 acting as support (long, uptrend) or resistance (short, downtrend,
with an above-average volume filter).

Runs as a 1H + 15m confluence, per the user's own comparison of the two
timeframes: 1H confirms the broader trend (the MA-stack ordering), while 15m
provides the entry trigger and stop — its EMA20 sits much closer to price,
giving a materially tighter stop than reading the same condition off 1H
alone. A signal only fires when both agree, which is deliberately more
selective than either timeframe on its own.

The stack is a four-MA ordering, not three. An earlier version checked only
EMA9 > EMA20 > SMA200 and skipped the 50 entirely, so it fired on symbols
whose EMA50 had already crossed below the SMA200 — a trend the user's method
does not consider up at all. On a live XAUTUSDT alert the 50 sat at 4051.93
against a 200 at 4058.71 and the signal fired anyway.

The 1H trend is recomputed on every 15m scan, from bars that include the
still-forming hourly candle (hence wants_forming_bar). Reading it only at
hourly closes meant an entry firing at :45 acted on a picture up to
45 minutes stale — long enough for price to break 1H EMA9/EMA20 while the
strategy still believed the stack was intact, which is exactly how a bad
BEATUSDT long got through. The 200-period stack barely moves across one
partial bar, so the cost of reading a forming candle here is small next to
the cost of acting on a stale one.

The touch condition is a proximity band around EMA9, not a strict
wick-through: an exact low<=EMA9<close check missed real setups where price
approached EMA9 closely without quite crossing it on that specific candle.
That band is measured in ATR, not as a percentage of price. A percentage
cannot mean the same thing across a watchlist spanning seven orders of
magnitude of price: at 0.005% the band came to 32 ticks on BTCUSDT but less
than a tenth of a tick on COTIUSDT, so the test was loose on a handful of
expensive symbols and mathematically unsatisfiable on 29 of 50 — 10 could
never fire at all. Any touch it did register on a cheap symbol was numerical
coincidence rather than price testing the level. Scaling by ATR makes one
constant mean the same thing everywhere.

The band and touch/close checks are read against EMA9 as of the PRIOR
candle, not the one closing right now. EMA9 recomputed with the current
candle's own close folded in gets pulled toward that close, so a single wide,
fast-moving candle (a breakout, not a pullback) can land its low inside the
band purely because its own close just dragged the average close to it —
that's not the candle "testing and rejecting" a level, it's the level
chasing the candle. Reading the level as of the prior candle instead means
the band reflects support/resistance that already existed before this
candle traded, so the check reflects an actual approach-then-reversal
(low nears a pre-existing EMA9, then the close moves back to the trend
side of it) rather than a coincidental overlap. ATR is read as of the prior
candle for the same reason: sizing the band with the current candle's own
range would widen the tolerance exactly when the candle is wide.

"Price close to resistance / far from support" for the short setup, and
"exit at nearby support" for either side, are left as manual judgment calls
at Approve/Reject time, same treatment as Strategy 1's discretionary
confirmations. Long's "at least 1:2" and short's "1:3" are both already
satisfied by the scanner-wide default, so neither side needs a reward:risk
override.
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


class EmaTrendFollowing(Strategy):
    tag = "Strategy 2"
    timeframes = ["1H", "15m"]
    wants_forming_bar = True  # the 1H trend must reflect the hour in progress

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars_1h = bars_by_timeframe.get("1H")
        bars_15m = bars_by_timeframe.get("15m")
        if bars_1h is None or bars_15m is None or len(bars_1h) < TREND_MA_PERIOD + 1 or len(bars_15m) < ATR_PERIOD + 2:
            return None

        trend = _trend(bars_1h)
        if trend is None:
            return None

        closes = bars_15m["close"]
        ema9_series = ema(closes, EMA_FAST)
        # The level being tested, and the tolerance around it, are both read as
        # of before this candle traded (see module docstring).
        ema9_prev = ema9_series.iloc[-2]
        band = atr(bars_15m, ATR_PERIOD).iloc[-2] * EMA9_BAND_ATR_MULTIPLE
        ema20_now = ema(closes, EMA_MID).iloc[-1]
        close_now, high_now, low_now = closes.iloc[-1], bars_15m["high"].iloc[-1], bars_15m["low"].iloc[-1]

        if trend == "up" and abs(low_now - ema9_prev) <= band and close_now > ema9_prev:
            return Signal(
                symbol=symbol,
                direction="long",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    f"1H uptrend (EMA9 > EMA20 > EMA50 > SMA200) confirmed; price came within "
                    f"{EMA9_BAND_ATR_MULTIPLE:g}x ATR of 15m EMA9 and held as support. Stop is 15m EMA20."
                ),
            )

        volumes = bars_15m["base_vol"]
        avg_volume = volumes.rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        high_volume = volumes.iloc[-1] > avg_volume

        if trend == "down" and high_volume and abs(high_now - ema9_prev) <= band and close_now < ema9_prev:
            return Signal(
                symbol=symbol,
                direction="short",
                entry_price=close_now,
                stop_loss=ema20_now,
                strategy_tag=self.tag,
                reason=(
                    f"1H downtrend (SMA200 > EMA50 > EMA20 > EMA9) confirmed with above-average 15m volume; "
                    f"price came within {EMA9_BAND_ATR_MULTIPLE:g}x ATR of 15m EMA9 and was rejected as "
                    f"resistance. Stop is 15m EMA20."
                ),
            )

        return None


def _trend(bars_1h: pd.DataFrame) -> str | None:
    """"up" or "down" when the four MAs are fully stacked, else None."""
    closes = bars_1h["close"]
    fast = ema(closes, EMA_FAST).iloc[-1]
    mid = ema(closes, EMA_MID).iloc[-1]
    slow = ema(closes, EMA_SLOW).iloc[-1]
    trend_ma = sma(closes, TREND_MA_PERIOD).iloc[-1]

    if fast > mid > slow > trend_ma:
        return "up"
    if trend_ma > slow > mid > fast:
        return "down"
    return None
