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

from notifier.risk_sizing import entry_fee_for, round_trip_fee_for
from notifier.strategies.base import FillGuard, Signal, Strategy
from notifier.strategies.indicators import rsi, sma
from notifier.strategies.structure import TrendStructure, structure_context

TREND_MA_PERIOD = 200
RSI_PERIOD = 10
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
# The window starts here and grows until it contains an observed change of
# character - see _structure_context. It is a floor rather than a cap now: the
# old fixed 200 decided the answer as often as the price data did.
SWING_MIN_LOOKBACK = 200
SWING_LOOKBACK_STEP = 50  # finer than the 100-bar sweep used to measure this,
# so the accepted window is the SMALLEST containing a turn rather than
# overshooting past it and dragging in older, already-resolved structure.
ATR_PERIOD = 14
# How far price must reverse to confirm a swing. Finer than the 6.0 this used
# before, because the anchor is now chosen by structure rather than recency:
# 6.0 meant 8% of price on APTUSDT, which left exactly ONE confirmed pivot in a
# 200-bar window and nothing for a break of structure to be read from.
#
# Lowered from 2.5 after Dror read the rendered charts: at 2.5 the detector
# missed a small retrace on AAPLUSDT and lumped several APTUSDT swings into one
# leg. At 1.25 it finds them, and the AAPLUSDT anchor becomes 340.01 - the
# number he had argued for from his own chart before any of this was measured.
#
# A/B replayed over 41 days of 1H bars across the watchlist: 2.5x produced 687
# signals at +0.026R, 1.25x produced 313 at +0.058R. The difference in
# expectancy is inside the noise (SE ~0.09R) and the paired comparison on
# shared bars actually favours 2.5 slightly - so this was NOT chosen for edge.
# It was chosen because it halves the signal count for the same total R, it
# reads the structure the way Dror does, and the measurement says it costs
# nothing. Both survive removal of their best 3 trades.
STRUCTURE_ATR_MULTIPLE = 1.25

# The price paid for the finer threshold, and the fix for it. At 1.25 the rate
# of fee-dominated legs - a stop under 1%, where the 0.12% round trip eats over
# an eighth of 1R - DOUBLED from 8% to 16%. APTUSDT was Dror's example: a 4.2%
# leg giving a 0.70% stop.
#
# Swept on 1.25's own 310 signals rather than reusing the number measured under
# 2.5. Fee-dominated legs fall 16% -> 8% -> 1% at 0/5/6%, and expectancy rises
# +0.06 -> +0.13 -> +0.20R. Past the optimum as required: it peaks near 10%
# (+0.32R) and falls back by 12% (+0.27R) on a shrinking sample, so the peak is
# not trusted. 6% is where the MECHANISM resolves - stop = leg x 16.8%, so a 6%
# leg is a ~1% stop, exactly the fee-domination boundary - rather than where
# the curve happened to be highest. Survives losing its best 3 (+0.205 ->
# +0.171R).
MIN_LEG_PCT = 0.06
FIB_ENTRY = 0.618
FIB_STOP = 0.786
REWARD_RISK_RATIO = 2.0
MARKET_ENTRY_FRACTION = 0.2  # cheatsheet's split entry: ~20% at market, ~80% resting
# MIN_LEG_PCT above is a static proxy for fee-domination, calibrated once from
# a one-time sweep - it does not recompute if the fee constants it was tuned
# against ever change. This is the live equivalent Strategy 2.1 already runs
# with, computed fresh from the actual fee formula every signal. Dror,
# 2026-08-27: "add it for the other [strategies]". Kept alongside MIN_LEG_PCT
# rather than replacing it - MIN_LEG_PCT's calibration is its own measured
# story (see the comment above it) and this is a second, independent check on
# the same failure mode.
MIN_NET_REWARD_RISK = 1.5
ENTRY_FEE_PCT = entry_fee_for(MARKET_ENTRY_FRACTION)

# The market-wide trend gate, added 2026-08-27 after a long/short-by-year
# investigation: replaying the same 719/758-symbol backtests split by
# calendar year found shorts crushed during a raging bull (BTC +98% that
# year, meanR -0.179R at t=-3.09) and longs crushed during a bear (BTC -46%,
# meanR -0.098R at t=-2.57) - fighting the market's OWN dominant direction is
# the common thread, independent of which year it happens to be. Measured as
# a split (not yet as this exact gate) it held up WITHIN each year separately
# (not just across the two-bucket split it was found on) and survived
# dropping each side's top 3 trades - the two checks that killed every other
# idea tried that session (regime-straddle legs, HTF same-symbol agreement,
# RSI-divergence gaps, confirmed-rejection candles).
#
# Same convention as the per-symbol 200-period trend filter above, just
# applied to a REFERENCE symbol instead of the one being traded - one
# definition of "what counts as an uptrend" rather than inventing a second
# one for this gate. AGREE/DISAGREE were BOTH measured negative (this
# reduces losses, it does not create profit on its own) - see the
# project-btc-trend-gate memory for the numbers before trusting this beyond
# what was actually tested.
MARKET_TREND_MA_PERIOD = 200


class RsiFibReversal(Strategy):
    """The cheatsheet calls this a 1h+ method, so the timeframe is a parameter
    rather than a constant: the same logic reads 1H, 4H or 1D. The tag carries
    it so each scale's performance is measured separately - there is no reason
    to assume an edge on one transfers to another.

    `market_trend_symbol`, when set, gates every signal on whether a
    REFERENCE symbol's own trend (price vs its 200-period MA, same
    convention as the per-symbol filter above) agrees with the trade's
    direction - long only when the reference is in an uptrend, short only
    when it is in a downtrend. None (the default) preserves the original,
    ungated behaviour exactly, so every existing instance and test that does
    not pass this stays unaffected. The reference's bars arrive via a
    "SYMBOL@TIMEFRAME" compound key in bars_by_timeframe - see
    notifier/scanner.py and backtest/portfolio.py for how that key gets
    fetched and threaded through, live and in backtest.
    """

    def __init__(self, timeframe: str = "1H", market_trend_symbol: str | None = None):
        self.timeframe = timeframe
        # A gated instance is a DIFFERENT trade population from the
        # ungated one - same symbol and timeframe can produce a signal on
        # one and not the other - so it needs its own tag rather than
        # silently sharing "Strategy 1 {timeframe}" with the instance
        # already live. Sharing a tag would merge two different strategies'
        # trades under one identity everywhere a tag is the key: routing,
        # the journal, the weekly report, LIVE_TAGS/DRY_RUN_TAGS membership.
        self.tag = f"Strategy 1 {timeframe}" + (f" +{market_trend_symbol}" if market_trend_symbol else "")
        self.market_trend_symbol = market_trend_symbol
        self._market_key = f"{market_trend_symbol}@{timeframe}" if market_trend_symbol else None
        self.timeframes = [timeframe] + ([self._market_key] if self._market_key else [])

    def _market_trend_agrees(self, bars_by_timeframe: dict[str, pd.DataFrame], direction: str) -> bool:
        """True when there's no reference configured, no reference data, or
        not enough history to read one - fails OPEN, matching every other
        best-effort gate in this codebase (e.g. Scanner._session_allows): a
        missing reference is not evidence the market disagrees, and silently
        muting every signal on a transient fetch gap would be worse than the
        rare signal this gate would otherwise have caught anyway."""
        if self._market_key is None:
            return True
        ref_bars = bars_by_timeframe.get(self._market_key)
        if ref_bars is None or len(ref_bars) < MARKET_TREND_MA_PERIOD + 1:
            return True
        ref_close = ref_bars["close"]
        ref_ma = sma(ref_close, MARKET_TREND_MA_PERIOD).iloc[-1]
        ref_trend_up = ref_close.iloc[-1] > ref_ma
        return ref_trend_up == (direction == "long")

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars = bars_by_timeframe[self.timeframe]
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

        if price > ma_now and crossed_below_oversold and self._market_trend_agrees(bars_by_timeframe, "long"):
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
                fill_guard=FillGuard(
                    min_net_reward_risk=MIN_NET_REWARD_RISK,
                    maker_fee_pct=ENTRY_FEE_PCT,
                    round_trip_fee_pct=round_trip_fee_for(MARKET_ENTRY_FRACTION),
                ),
                reason=(
                    "RSI(10) crossed below 30 above the 200-MA trend filter. "
                    "Stop is the 78.6% Fib level. Check for RSI divergence and "
                    "higher-timeframe trend conflicts before approving."
                ),
            )

        if price < ma_now and crossed_above_overbought and self._market_trend_agrees(bars_by_timeframe, "short"):
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
                fill_guard=FillGuard(
                    min_net_reward_risk=MIN_NET_REWARD_RISK,
                    maker_fee_pct=ENTRY_FEE_PCT,
                    round_trip_fee_pct=round_trip_fee_for(MARKET_ENTRY_FRACTION),
                ),
                reason=(
                    "RSI(10) crossed above 70 below the 200-MA trend filter. "
                    "Stop is the 78.6% Fib level. Check for RSI divergence and "
                    "higher-timeframe trend conflicts before approving."
                ),
            )

        return None


def _uptrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the up-move being retraced: the low the
    uptrend turned up from, and the highest high since."""
    return _leg(bars, "up")


def _downtrend_leg(bars: pd.DataFrame) -> tuple[float, float] | None:
    """(swing_low, swing_high) of the down-move being retraced: the high the
    downtrend turned down from, and the lowest low since."""
    return _leg(bars, "down")


def _leg(bars: pd.DataFrame, direction: str) -> tuple[float, float] | None:
    """The leg the current trend began from, if it is still tradeable.

    The anchor is the swing the trend TURNED at, not merely the most recent
    pivot of the right kind. Dror's reading of AAPLUSDT 1H is the case that
    forced this: the code anchored on 313.36, a minor high inside the bounce,
    giving an 11-point leg and a 0.62% stop that fees ate 19% of. The high the
    downtrend actually turned from was 344.75, four swings earlier - a 44-point
    leg and a 2.26% stop. Both were "the most recent confirmed pivot high" at
    the moment they were asked for; only one started the move being retraced.

    Anchoring by structure also makes the pivot threshold largely stop
    mattering, which is what the old rule could not survive. SWING_ATR_MULTIPLE
    of 6.0 meant 2.5% of price on AAPLUSDT and 8% on APTUSDT, so the same
    constant found dozens of pivots on one symbol and exactly one on the other.
    """
    window, structure = _structure_context(bars)
    if structure.trend != direction or structure.anchor_index is None:
        return None

    anchor = structure.anchor_index
    if direction == "down":
        swing_high = window["high"].iloc[anchor]
        swing_low = window["low"].iloc[anchor:].min()
    else:
        swing_low = window["low"].iloc[anchor]
        swing_high = window["high"].iloc[anchor:].max()
    if swing_high <= swing_low:
        return None

    # Too small a leg is not a tradeable retracement, it is noise. The Fib gap
    # between the 61.8% entry and the 78.6% stop is 16.8% of the leg, so a 6%
    # leg is about a 1% stop - the point below which the round-trip fee starts
    # eating an eighth of 1R before the market has done anything. Trade #6 was
    # the live example: an 11-point leg on AAPLUSDT, a 0.62% stop, fees taking
    # 19% of R, and it lost 1.11R.
    if (swing_high - swing_low) / swing_low < MIN_LEG_PCT:
        return None

    # How far price may already have come back before the setup is no longer
    # worth taking. This replaces the old guard, which killed a leg the moment
    # an opposite pivot confirmed - too blunt: on AAPLUSDT it rejected a leg
    # retraced only 30% while the leg it had chosen instead sat at 90%, so the
    # setup vanished for being healthy and stayed for being dead.
    #
    # The bound is the MIDPOINT of the entry and the stop - "more than halfway
    # from the entry to where it fails" - so it needs no invented constant.
    # Past it the trade has under a third of its risk distance left AND the
    # resting limit is already behind the market, so the split entry the alert
    # describes cannot happen: the limit fills instantly at the market price
    # and the quoted blend is fiction. Dror caught this on two live alerts,
    # MMTUSDT at 75% and GOOGLUSDT at 74% retraced, before it was measured.
    #
    # Measured across 41 days of 1H bars: signals above this line won 9% of the
    # time at -0.73R. Cutting at FIB_ENTRY instead would be wrong - the band
    # between 61.8% and this midpoint is comfortably profitable, so only the
    # last stretch before the stop is bad.
    price = window["close"].iloc[-1]
    retraced = (price - swing_low) if direction == "down" else (swing_high - price)
    if retraced / (swing_high - swing_low) > (FIB_ENTRY + FIB_STOP) / 2:
        return None

    return swing_low, swing_high


def _structure_context(bars: pd.DataFrame) -> tuple[pd.DataFrame, TrendStructure]:
    """Strategy 1's own parameters for the shared break-of-structure read.

    The implementation moved to structure.py when Strategy 4 needed the
    identical reading - see structure_context for why the window grows until a
    CHoCH is observed. The constants stay here because they are Strategy 1's:
    STRUCTURE_ATR_MULTIPLE in particular was chosen against Dror's own reading
    of AAPLUSDT and APTUSDT charts for anchoring a Fib, and another strategy
    wanting a different swing scale must not silently move this one.
    """
    return structure_context(
        bars,
        atr_multiple=STRUCTURE_ATR_MULTIPLE,
        atr_period=ATR_PERIOD,
        min_lookback=SWING_MIN_LOOKBACK,
        lookback_step=SWING_LOOKBACK_STEP,
    )
