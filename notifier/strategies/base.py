"""Strategy interface: given recent closed OHLCV bars for a symbol, decide
whether to fire a signal. Add one file per strategy here as they're described,
plus one registration line in whatever builds the Scanner's strategy list
(currently notifier/main.py).

Each strategy declares the timeframe it wants. The scanner groups strategies
by timeframe and scans each group just after that timeframe's candle closes,
so adding a 4H or 15m strategy alongside the 1H ones needs no scanner changes.
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
    entry_price: float
    stop_loss: float
    strategy_tag: str
    reason: str = ""
    reward_risk_ratio: float | None = None  # overrides the scanner-wide default when set


class Strategy(ABC):
    tag: str
    timeframe: str = "1H"

    @abstractmethod
    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """bars is CLOSED OHLCV data, oldest row first, with columns:
        ts, open, high, low, close, base_vol, quote_vol. The still-forming
        candle is excluded, so bars.iloc[-1] is the most recent closed bar.
        """
