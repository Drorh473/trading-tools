"""Strategy interface: given recent OHLCV bars for a symbol, decide whether
to fire a signal. Add one file per strategy here as they're described, plus
one registration line in whatever builds the Scanner's strategy list
(currently notifier/main.py).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


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

    @abstractmethod
    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """bars is OHLCV data, oldest row first, with columns:
        ts, open, high, low, close, base_vol, quote_vol.
        """
