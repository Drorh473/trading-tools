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

Entry is the EMA9 level itself (ema9_prev - the level already confirmed
tested by the touch/hold checks above), not the triggering candle's close.
That was a deliberate, documented deviation for a while: the touch already
happened by the time the candle closes, so a fresh limit resting at that
same level might never fill again. Measured 0.27% better on longs when it
does fill. Unlike Strategy 1's Fib entry there is no market fraction to
guarantee partial participation - a limit that never fills means no trade,
which is the accepted tradeoff rather than entering at a worse price.

The stack is a four-MA ordering, not three. An earlier version checked only
EMA9 > EMA20 > SMA200 and skipped the 50 entirely, so it fired on symbols
whose EMA50 had already crossed below the SMA200 — a trend the user's method
does not consider up at all. On a live XAUTUSDT alert the 50 sat at 4051.93
against a 200 at 4058.71 and the signal fired anyway.

The 1H trend is recomputed on every 15m scan, from bars that include the
still-forming hourly candle. Reading it only at hourly closes meant an entry
firing at :45 acted on a picture up to 45 minutes stale — long enough for
price to break 1H EMA9/EMA20 while the strategy still believed the stack was
intact, which is exactly how a bad BEATUSDT long got through. The 200-period
stack barely moves across one partial bar, so the cost of reading a forming
candle here is small next to the cost of acting on a stale one.

That opt-in covers the 1H *only*. Applying it to 15m as well handed the entry
trigger a candle 30 seconds old — scans run just after each 15m close — whose
close, low and EMA20 were all still moving. It produced entries at prices no
candle ever closed at, stops off an EMA20 that had a partial close folded in,
and "held as support" on candles that went on to close below EMA9.

The 1H stack alone is not enough to enter on. It says nothing about the 15m,
which can be in a downtrend of its own while the hourly picture is intact —
and since the stop IS the 15m EMA20, an inverted 15m stack puts the stop on
the wrong side of the entry. Every live alert that arrived with its stop above
its entry had 15m EMA9 below EMA20; all 17 such cases over 9.4 days did. So
the 15m must agree directionally before its levels are usable.

A touch also only means support *held* if price was not already cutting
through the level. Without that, signals fired on symbols chopping back and
forth across EMA9, where the "test" was one of several crossings rather than
a rejection. Requiring price to have closed on the trend side of EMA9 for the
preceding EMA9_HOLD_BARS candles is what separates the two, and it is the
single biggest contributor to signal quality here: it takes the long side from
roughly 23 to 5 alerts a day.

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
EMA9_HOLD_BARS = 10  # 2.5h of 15m candles holding the level before a touch counts
STOP_ATR_BUFFER = 0.10  # the stop sits below EMA20 (above, for shorts), not on it
# A candle this much bigger than its own prior ATR likely dragged EMA9 toward
# price by itself rather than price testing an already-established level.
# KAITOUSDT's and LABUSDT's disqualifying candles measured 3.68x and 2.87x ATR.
SHOCK_CANDLE_ATR_MULTIPLE = 2.0
# held_above/held_below only look inside the hold window itself, so a level
# whipsawed for hours right before that window still passes as long as the
# most recent EMA9_HOLD_BARS candles happen to be clean. CLUSDT crossed EMA9
# nine times in the 30 bars before its trigger and still passed. 3x the hold
# window catches a level that only just settled down rather than one that was
# actually respected.
CROSSING_LOOKBACK_BARS = 30
MAX_CROSSINGS_IN_LOOKBACK = 1


class EmaTrendFollowing(Strategy):
    """A slower timeframe confirming the trend, a faster one triggering the
    entry. Which pair is a parameter: 1H/15m, 4H/1H and 1D/4H are the same
    method at three scales, and each is tagged separately because an edge at
    one scale says nothing about another."""

    def __init__(self, trend_timeframe: str = "1H", entry_timeframe: str = "15m"):
        self.trend_timeframe = trend_timeframe
        self.entry_timeframe = entry_timeframe
        self.tag = f"Strategy 2 {trend_timeframe}/{entry_timeframe}"
        self.timeframes = [trend_timeframe, entry_timeframe]
        # The trend reads the candle in progress; the entry trigger must not.
        self.forming_bar_timeframes = (trend_timeframe,)

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars_1h = bars_by_timeframe.get(self.trend_timeframe)
        bars_15m = bars_by_timeframe.get(self.entry_timeframe)
        needed_15m = max(ATR_PERIOD, EMA9_HOLD_BARS, CROSSING_LOOKBACK_BARS) + 2
        if bars_1h is None or bars_15m is None or len(bars_1h) < TREND_MA_PERIOD + 1 or len(bars_15m) < needed_15m:
            return None

        trend = _trend(bars_1h)
        if trend is None:
            return None

        closes = bars_15m["close"]
        ema9_series = ema(closes, EMA_FAST)
        atr_series = atr(bars_15m, ATR_PERIOD)
        # The level being tested, and the tolerance around it, are both read as
        # of before this candle traded (see module docstring).
        ema9_prev = ema9_series.iloc[-2]
        atr_prev = atr_series.iloc[-2]
        band = atr_prev * EMA9_BAND_ATR_MULTIPLE
        ema9_now = ema9_series.iloc[-1]
        ema20_now = ema(closes, EMA_MID).iloc[-1]
        # The cheatsheet places the stop below EMA20 for a long and above it for
        # a short, not on the average itself. Sitting exactly on it means any
        # wick that merely grazes the average takes the trade out: a third of
        # otherwise-valid signals had their whole stop inside one candle's
        # average range.
        stop_buffer = atr_prev * STOP_ATR_BUFFER
        close_now, high_now, low_now = closes.iloc[-1], bars_15m["high"].iloc[-1], bars_15m["low"].iloc[-1]

        # The candles before this one, and where they closed relative to the
        # EMA9 as it stood on each of them.
        prior_closes = closes.iloc[-EMA9_HOLD_BARS - 1 : -1]
        prior_ema9 = ema9_series.iloc[-EMA9_HOLD_BARS - 1 : -1]
        held_above = bool((prior_closes > prior_ema9).all())
        held_below = bool((prior_closes < prior_ema9).all())

        # A touch only means the level was actually tested if EMA9 got there on
        # its own, not because one outsized candle dragged it toward price.
        # KAITOUSDT and LABUSDT both fired on a wick landing on EMA9 only
        # because a single shock candle a few bars earlier had yanked the
        # average - the hold window was technically clean, but the level in it
        # was chased, not tested.
        ranges = bars_15m["high"] - bars_15m["low"]
        shock_threshold = atr_series.shift(1) * SHOCK_CANDLE_ATR_MULTIPLE
        no_shock_candle = not bool(
            (ranges.iloc[-EMA9_HOLD_BARS - 1 : -1] > shock_threshold.iloc[-EMA9_HOLD_BARS - 1 : -1]).any()
        )

        above_ema9 = closes > ema9_series
        crossing_window = above_ema9.iloc[-CROSSING_LOOKBACK_BARS - 1 : -1]
        recent_crossings = int((crossing_window != crossing_window.shift()).sum() - 1)
        not_recently_choppy = recent_crossings <= MAX_CROSSINGS_IN_LOOKBACK

        if (
            trend == "up"
            and ema9_now > ema20_now  # 15m agrees, so its EMA20 is a stop below entry
            and held_above  # EMA9 was support, not a level price kept cutting through
            and no_shock_candle
            and not_recently_choppy
            and abs(low_now - ema9_prev) <= band
            and close_now > ema9_prev
            and ema20_now - stop_buffer < ema9_prev  # a long's stop must sit below its entry
        ):
            return Signal(
                symbol=symbol,
                direction="long",
                # The cheatsheet's entry is the EMA9 touch itself, not wherever
                # price happened to close - measured 0.27% better on longs. A
                # limit sitting there may simply never fill again, which is the
                # accepted tradeoff (Dror's call): no market fraction bails it
                # out, unlike Strategy 1's Fib entry.
                entry_price=ema9_prev,
                stop_loss=ema20_now - stop_buffer,
                strategy_tag=self.tag,
                limit_entry=ema9_prev,
                limit_note="EMA9",
                market_fraction=0.0,
                reason=(
                    f"1H uptrend (EMA9 > EMA20 > EMA50 > SMA200) confirmed; 15m EMA9 above EMA20 and price "
                    f"held above 15m EMA9 for the last {EMA9_HOLD_BARS} candles, then came within "
                    f"{EMA9_BAND_ATR_MULTIPLE:g}x ATR of it and held as support. Stop is just below the 15m EMA20."
                ),
            )

        volumes = bars_15m["base_vol"]
        avg_volume = volumes.rolling(VOLUME_LOOKBACK).mean().iloc[-1]
        high_volume = volumes.iloc[-1] > avg_volume

        if (
            trend == "down"
            and high_volume
            and ema9_now < ema20_now  # 15m agrees, so its EMA20 is a stop above entry
            and held_below
            and no_shock_candle
            and not_recently_choppy
            and abs(high_now - ema9_prev) <= band
            and close_now < ema9_prev
            and ema20_now + stop_buffer > ema9_prev  # a short's stop must sit above its entry
        ):
            return Signal(
                symbol=symbol,
                direction="short",
                entry_price=ema9_prev,
                stop_loss=ema20_now + stop_buffer,
                strategy_tag=self.tag,
                limit_entry=ema9_prev,
                limit_note="EMA9",
                market_fraction=0.0,
                reason=(
                    f"1H downtrend (SMA200 > EMA50 > EMA20 > EMA9) confirmed with above-average 15m volume; "
                    f"15m EMA9 below EMA20 and price held below 15m EMA9 for the last {EMA9_HOLD_BARS} "
                    f"candles, then came within {EMA9_BAND_ATR_MULTIPLE:g}x ATR of it and was rejected as "
                    f"resistance. Stop is just below the 15m EMA20."
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
