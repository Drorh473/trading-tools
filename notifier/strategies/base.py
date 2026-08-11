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
    extra_notes: tuple[str, ...] = ()  # standalone alert lines, e.g. trailing-stop guidance
    # Set when a strategy has its own reason to size this specific signal
    # above the scanner's default - Strategy 2's tiered risk (1%/1.5%/2%
    # depending on how much a second timeframe corroborates), for instance.
    # Stacks with pattern confluence rather than replacing it: the scanner
    # takes whichever of the two implies the higher risk, since a chart
    # pattern and a second timeframe agreeing are different kinds of evidence
    # and neither should silence the other.
    risk_pct_override: float | None = None
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
