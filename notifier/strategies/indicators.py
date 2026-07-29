"""Shared technical indicator calculations, reused across strategies."""

import pandas as pd


def sma(closes: pd.Series, period: int) -> pd.Series:
    return closes.rolling(period).mean()


def ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (the standard convention referenced by the cheatsheet)."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
