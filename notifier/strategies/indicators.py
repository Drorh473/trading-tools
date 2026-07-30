"""Shared technical indicator calculations, reused across strategies."""

import pandas as pd


def sma(closes: pd.Series, period: int) -> pd.Series:
    return closes.rolling(period).mean()


def ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range, from full OHLC bars rather than closes."""
    prev_close = bars["close"].shift()
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (the standard convention referenced by the cheatsheet)."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
