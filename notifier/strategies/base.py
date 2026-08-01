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


class Strategy(ABC):
    tag: str
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

    @abstractmethod
    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> "Signal | None":
        """bars_by_timeframe maps each declared timeframe to its OHLCV data
        (oldest row first, columns: ts, open, high, low, close, base_vol,
        quote_vol). The still-forming candle is excluded unless the strategy
        sets wants_forming_bar, so by default bars.iloc[-1] is always the most
        recent closed bar for that timeframe.
        """
