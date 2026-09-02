"""Strategy interface: given closed OHLCV bars for a symbol, decide whether to
fire a signal. Add one file per strategy here as they're described, plus one
registration line in whatever builds the Scanner's strategy list (currently
notifier/main.py).

Each strategy declares the timeframe(s) it needs. Single-timeframe strategies
just use one entry (e.g. ["1H"]); a strategy that wants confluence across
timeframes (e.g. 1H trend + 15m entry trigger) lists both, and evaluate()
receives a dict keyed by timeframe instead of a single dataframe. The scanner
fetches the union of every strategy's required timeframes and scans on
whichever is shortest, so adding a new timeframe combination needs no scanner
changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

# Bitget granularity token -> seconds
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "4H": 14400,
    "6H": 21600,
    "12H": 43200,
    "1D": 86400,
}


@dataclass(frozen=True)
class FillGuard:
    """The economic conditions a signal must still meet once the price it will
    actually FILL at is known.

    A strategy decides from a LEVEL. For most strategies that level is also the
    fill - Strategy 4 rests a limit at its own entry_price, Strategy 3 sets
    entry = close_now - and the distinction never arises. Strategy 2.1 breaks
    it: entry_price is the EMA9 that SELECTED the setup, while the order goes in
    at market on the candle AFTER the rejection, which by construction has
    closed back on the trend side. The fill is therefore always on the far side
    of the level, and every quantity derived from the level - stop distance,
    reward:risk - describes a trade nobody gets.

    Live consequence, from the alerts of 2026-08-18: HYPEUSDT measured a 1.50%
    stop against its EMA9 and 0.15% against the price it would have filled at.
    It passed a 0.30% floor it fails by half. Six alerts that day carried
    targets between 0.16:1 and 1.43:1 while the strategy believed it had
    refused anything under 1.5:1.

    So the thresholds travel WITH the signal and are re-applied by the scanner
    once plan_entry is known. Data rather than a callback, because signals are
    pickled by the backtest generator and a closure would not survive it.

    `atr` is the ATR the stop was buffered with, carried so a floor can be
    expressed in volatility rather than in percent. MIN_STOP_PCT cannot tell a
    wide stop from a wide market: LABUSDT's stop was 4.3% of price and 1.13 ATR
    at the same time.
    """

    atr: float | None = None
    min_stop_pct: float = 0.0
    max_stop_pct: float = 1.0
    min_stop_atr: float = 0.0
    min_net_reward_risk: float = 0.0
    maker_fee_pct: float = 0.0
    round_trip_fee_pct: float = 0.0

    def refuses(self, entry: float, stop: float, reward_risk_ratio: float) -> str | None:
        """Why this trade should not be taken at `entry`, or None to allow it.

        Returns prose rather than a bool so the refusal can be logged and read
        back in the weekly stats - a signal that vanishes silently is the same
        failure as one that never fired.
        """
        risk = abs(entry - stop)
        if entry <= 0 or risk <= 0:
            return f"entry {entry:g} and stop {stop:g} leave no risk to measure"

        stop_pct = risk / entry
        if stop_pct < self.min_stop_pct:
            return (
                f"stop is {100 * stop_pct:.2f}% of the fill at {entry:g}, under the "
                f"{100 * self.min_stop_pct:.2f}% floor - inside the spread, not a stop"
            )
        if stop_pct > self.max_stop_pct:
            return (
                f"stop is {100 * stop_pct:.2f}% of the fill at {entry:g}, over the "
                f"{100 * self.max_stop_pct:.2f}% ceiling - no longer an orderly pullback"
            )

        if self.min_stop_atr > 0:
            if not self.atr or self.atr <= 0:
                return "no ATR recorded, so the stop cannot be judged against volatility"
            stop_atr = risk / self.atr
            if stop_atr < self.min_stop_atr:
                return (
                    f"stop is {stop_atr:.2f} ATR from the fill, under the "
                    f"{self.min_stop_atr:.2f} ATR floor - inside one candle's range"
                )

        if self.min_net_reward_risk > 0 and reward_risk_ratio is not None:
            reward = reward_risk_ratio * risk
            net = (reward - self.maker_fee_pct * entry) / (risk + self.round_trip_fee_pct * entry)
            if net < self.min_net_reward_risk:
                return (
                    f"net reward:risk is {net:.2f} from the fill at {entry:g}, under the "
                    f"{self.min_net_reward_risk:.2f} floor"
                )
        return None


@dataclass
class Signal:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float  # what the plan is measured from: stop distance, size, targets
    stop_loss: float
    strategy_tag: str
    reason: str = ""  # kept for the record; deliberately not rendered in alerts
    reward_risk_ratio: float | None = None  # overrides the scanner-wide default when set
    # Set when only part of the position is meant to go in at market and the
    # rest should rest as a limit. Carried as a number rather than baked into
    # reason text so the alert can format it at the symbol's own precision.
    limit_entry: float | None = None
    limit_note: str = ""  # what that level is, e.g. "61.8% Fib"
    market_fraction: float = 0.2  # share of the position going in at market
    # Exit overrides. Setting partial_fraction means the strategy manages its
    # own two-tier exit rather than taking the scanner's 50% / 1:3 default -
    # Strategy 3 takes 75% off at its target and runs the rest to chart
    # resistance, which is a level rather than a ratio. remainder_target None
    # alongside a set partial_fraction means there is no price for the runner
    # at all (the setup is at all-time highs, nothing overhead), and
    # remainder_note carries what to do instead.
    partial_fraction: float | None = None
    remainder_target: float | None = None
    remainder_note: str = ""  # e.g. "daily resistance", "after 3 trading days"
    # Whether remainder_target is the FINAL word or merely a fallback.
    #
    # Setting partial_fraction normally hands the runner to the scanner, which
    # puts it at the nearest daily swing level and uses remainder_target only
    # when the daily offers nothing. That is right for Strategy 3, whose runner
    # genuinely goes to chart resistance.
    #
    # It is wrong for a strategy whose runner price IS the thesis. Strategy 2.1
    # takes its two targets from the higher timeframe's own 1:2 and 1:3, and
    # those prices are what its measured reward:risk of 8:1 describes. Letting
    # a daily level replace them would deploy a different strategy from the one
    # that was measured.
    remainder_target_is_final: bool = False
    extra_notes: tuple[str, ...] = ()  # standalone alert lines, e.g. trailing-stop guidance
    # Set when a strategy has its own reason to size this specific signal
    # above the scanner's default - Strategy 2's tiered risk (1%/1.5%/2%
    # depending on how much a second timeframe corroborates), for instance.
    # Stacks with pattern confluence rather than replacing it: the scanner
    # takes whichever of the two implies the higher risk, since a chart
    # pattern and a second timeframe agreeing are different kinds of evidence
    # and neither should silence the other.
    risk_pct_override: float | None = None
    # How long a resting entry may go unfilled before the trade is abandoned,
    # when this strategy has an opinion. None keeps the tracker's flat
    # ENTRY_TIMEOUT_SECONDS, which is 4 hours and was chosen for Strategy 1's
    # limit tranche.
    #
    # Strategy 4 measured its own answer - 30 candles, where the fill curve
    # flats - and then could not use it: the constant sat in order_block.py
    # marked NOT YET WIRED INTO EXECUTION while a 4-hour wall clock cancelled
    # its orders. On the 1H instance that is 4 candles against the 30 it was
    # calibrated for, so the measurement described a strategy that was not
    # running. Seconds rather than candles because the tracker counts
    # wall-clock time and only the strategy knows its own timeframe.
    unfilled_timeout_seconds: float | None = None
    # Which timeframes the alert's "Analysis timeframe" line lists, for a
    # strategy whose analysis timeframes vary signal-by-signal rather than
    # being fixed per instance - Strategy 2 only lists its reference
    # timeframe when that timeframe was a genuine second confirmation, not
    # merely a supportive trend read (which goes in extra_notes as prose
    # instead). None falls back to the strategy's own fixed Strategy.timeframes.
    analysis_timeframes: tuple[str, ...] | None = None
    # Overrides what the scanner treats as "the same trade" for de-duplication.
    # The default key includes entry_price and stop_loss, which is right when
    # those identify the setup - but a breakout strategy re-triggering against
    # ONE level produces a slightly different entry each time, so the default
    # reads every wobble across that level as a new trade. Strategy 3 fired
    # TSLAUSDT twice ten minutes apart off an identical range because the two
    # 5m closes differed by two cents. Keying on the level itself instead
    # means the range is claimed once, however often price crosses it.
    dedupe_key: tuple | None = None
    # Re-checked against the price this trade actually fills at, once the
    # scanner knows it. See FillGuard - a strategy whose entry_price is a
    # LEVEL rather than an expected fill has to state its economic conditions
    # here, because checking them against the level checks a different trade.
    fill_guard: "FillGuard | None" = None


class Strategy(ABC):
    tag: str
    # Every tag this strategy can put on a signal. Almost always just `tag`,
    # but a strategy that classifies its own signals emits more than one -
    # Strategy 4 tags each block with the version that found it, and deciding
    # "2.0 wins when both qualify" needs both detectors inside ONE instance
    # that can see both answers. The execution whitelist is checked against
    # this rather than `tag`, so a variant tag cannot be left unrouted while
    # the list looks complete.
    tags: tuple[str, ...] = ()
    timeframes: list[str] = ["1H"]
    # Strategies evaluate closed bars by default. A strategy that reads a slow
    # trend off a longer timeframe while triggering on a shorter one can opt in
    # to the forming candle for that timeframe only: waiting for the slow one
    # to close means acting on a picture that may be most of a candle out of
    # date by the time the trigger fires.
    #
    # This is deliberately per-timeframe rather than per-strategy. Applying it
    # to every timeframe a strategy declares hands its *trigger* an unfinished
    # candle, whose close, extremes and derived indicators all still move -
    # which produced entries at prices no candle ever closed at, and stops
    # computed from an EMA20 that included a partial close.
    forming_bar_timeframes: tuple[str, ...] = ()
    # Timeframes fetched ONLY for symbols this strategy has armed, rather than
    # for the whole watchlist every scan. A 5m trigger declared normally would
    # drag the entire loop to a 5m cadence and refetch 100 symbols every five
    # minutes; declared here, the regular scan keeps its own cadence and only
    # the handful of symbols with a live setup are polled that often.
    armed_timeframes: tuple[str, ...] = ()
    # Whether this strategy may only fire while the symbol's own market is
    # open. Intraday strategies must set it: their volume and structure reads
    # assume a market that is actually trading, and a tokenized stock keeps
    # printing bars long after its shares stop changing hands. Slower
    # strategies leave it False - a daily bar spans a whole session, so the
    # question does not arise. See notifier/sessions.py.
    session_gated: bool = False
    # Tags whose signal this one REPLACES when both fire on the same symbol and
    # side in the same scan. Declared here rather than decided in the scanner so
    # the scanner needs no knowledge of any particular strategy.
    #
    # Strategy 2.1's paired instances supersede their own base timeframe's
    # standalone instance. Measured, the two coincide on 26% of standalone
    # triggers, and they are the SAME trade: same symbol, same entry level, same
    # stop, differing only in where the target sits. Letting both through puts
    # 2% of equity on one idea in two positions - which is the tiered risk that
    # was deliberately removed, rebuilt by accident out of instances.
    #
    # The pair wins because it is strictly more information than the standalone:
    # everything the standalone saw, plus a higher timeframe confirming it.
    supersedes: tuple[str, ...] = ()
    # Which declared timeframe notifier.chart draws candles from for this
    # strategy's alerts. None means "the first entry in timeframes" - right
    # for every single-timeframe strategy and for one whose entry/stop/
    # indicators all live on its first declared timeframe (e.g. Strategy
    # 2.1's base_timeframe). A strategy set this explicitly when its most
    # informative frame is NOT the first one declared - Strategy 3 declares
    # [trend_timeframe, entry_timeframe] but sets this to trend_timeframe,
    # because the consolidation box that makes the setup legible lives on
    # the daily frame while entry_timeframe only supplies the breakout
    # trigger candle.
    chart_timeframe: str | None = None

    def all_tags(self) -> tuple[str, ...]:
        """Every tag this strategy may emit, for whitelist checking."""
        return self.tags or (self.tag,)

    def arms(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> bool:
        """Whether this symbol is close enough to triggering to be worth
        polling on the armed timeframe. Called on every regular scan with the
        strategy's non-armed timeframes, so arming is recomputed from scratch
        each time - a symbol whose setup dies simply stops being armed, with no
        state to go stale.
        """
        return False

    @abstractmethod
    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> "Signal | None":
        """bars_by_timeframe maps each declared timeframe to its OHLCV data
        (oldest row first, columns: ts, open, high, low, close, base_vol,
        quote_vol). The still-forming candle is excluded unless the strategy
        sets wants_forming_bar, so by default bars.iloc[-1] is always the most
        recent closed bar for that timeframe.
        """

    def chart_overlay(self, bars_by_timeframe: dict, signal: "Signal"):
        """What to draw on top of this signal's chart, beyond the entry/
        stop/target lines notifier.chart always draws - or None for candles
        alone. Returns a notifier.chart.ChartOverlay (not imported here to
        keep matplotlib out of every strategy's import chain by default).

        Called once, right after evaluate() produces `signal`, on the SAME
        bars_by_timeframe - so a strategy that needs data evaluate() only
        held in a local variable (not carried on the Signal itself) may
        stash it on self, keyed by symbol, and read it back here. See
        OrderBlockStrategy for the case that actually needs this.

        Never called for a signal nobody is about to see a chart for, so it
        is fine for this to be a little more expensive than evaluate() -
        recomputing a strategy's own indicators here rather than smuggling
        them onto Signal keeps Signal's schema (pickled by the backtest
        generator, stored via signal_to_json) free of chart-only fields.
        """
        return None


# ---------------------------------------------------------------------------
# Storing a Signal so an expired one can be offered again by its number.
# ---------------------------------------------------------------------------

_TUPLE_FIELDS = ("extra_notes", "analysis_timeframes", "dedupe_key")


def signal_to_json(signal: "Signal") -> str:
    """The whole Signal, flat enough for a TEXT column.

    The signals table's own columns describe a signal well enough to SCORE it
    and not well enough to REBUILD it - no partial_fraction, no
    remainder_target, no limit_entry, no fill guard. A trade rebuilt from those
    alone would quietly take the scanner's default exit instead of the
    strategy's, which is the difference between a 50%/1:3 runner and Strategy
    4's single flat close.

    json cannot carry a tuple or a nested dataclass, so both are converted
    explicitly here rather than left to asdict() and a round trip that returns
    lists where tuples were.
    """
    import dataclasses
    import json

    data = dataclasses.asdict(signal)
    if signal.fill_guard is not None:
        data["fill_guard"] = dataclasses.asdict(signal.fill_guard)
    return json.dumps(data)


def signal_from_json(payload: str) -> "Signal":
    """Rebuild a Signal stored by signal_to_json.

    dedupe_key comes back as a tuple deliberately: the scanner uses it for
    identity, and a list would never match the tuple a live evaluation
    produces, so a re-offered signal would dodge the de-duplication that stops
    the same setup being sent twice.
    """
    import json

    data = json.loads(payload)
    guard = data.pop("fill_guard", None)
    for name in _TUPLE_FIELDS:
        if isinstance(data.get(name), list):
            data[name] = tuple(data[name])
    signal = Signal(**data)
    if guard is not None:
        object.__setattr__(signal, "fill_guard", FillGuard(**guard))
    return signal
