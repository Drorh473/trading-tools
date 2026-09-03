"""Fetches and caches OHLCV bars per (symbol, timeframe), refetching only
once that timeframe's candle has turned over.

Extracted from Scanner, which used to own this cache directly alongside a
dozen unrelated responsibilities. Single job: given a symbol and a
timeframe, return its bars - fresh if the candle has turned over since the
last fetch, cached otherwise. Scans run at the shortest declared
timeframe's cadence, so without this a daily series would be refetched
every few minutes just because an hourly scan asked again.
"""

from __future__ import annotations

import time

import pandas as pd

from core.bitget_client import BitgetClient
from notifier.strategies.base import TIMEFRAME_SECONDS


def bars_dataframe(candles: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df


class BarCache:
    def __init__(
        self,
        bitget: BitgetClient,
        candle_limit: int = 600,
        # Per-(symbol, timeframe) overrides above candle_limit - for a
        # reference series whose gate needs real depth (e.g. a persistent,
        # never-pruned significant-levels list, notifier.strategies.levels),
        # not the per-symbol indicator window every other fetch needs. Only
        # the keys listed here pay the deeper fetch; everything else keeps
        # the plain candle_limit default. A very large value (bigger than the
        # symbol's actual history) is fine - get_candles' own history-paging
        # loop stops once the exchange has nothing older left to page in.
        deep_history: dict[tuple[str, str], int] | None = None,
    ):
        self.bitget = bitget
        self.candle_limit = candle_limit
        self.deep_history = deep_history or {}
        # (symbol, timeframe) -> (candle this was fetched during, bars).
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}

    def get(self, symbol: str, timeframe: str, now: float | None = None) -> pd.DataFrame:
        """Bars for this symbol and timeframe, refetched only once the
        timeframe's candle has turned over.

        Includes the forming candle; callers trim it when they want closed
        bars only. Keyed on which candle is currently forming, so a 1D
        series is fetched once a day and a 15m series every scan, without
        anything having to know the scan cadence.
        """
        period = TIMEFRAME_SECONDS[timeframe]
        now = time.time() if now is None else now
        current_candle = now - (now % period)

        cached = self._cache.get((symbol, timeframe))
        if cached and cached[0] == current_candle:
            return cached[1]

        limit = self.deep_history.get((symbol, timeframe), self.candle_limit)
        candles = self.bitget.get_candles(symbol, granularity=timeframe, limit=limit + 1, closed_only=False)
        bars = bars_dataframe(candles)
        self._cache[(symbol, timeframe)] = (current_candle, bars)
        return bars
