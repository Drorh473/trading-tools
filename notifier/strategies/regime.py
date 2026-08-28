"""BTC's own trend, read the way Dror reads it before every trade: direction
plus whether price sits too close to a level that could end it right now.
Standalone - this module answers whether that read is a real, persistent
market phenomenon, independent of any specific strategy's trade history. It
is not wired into any strategy's evaluate() yet.
"""
import pandas as pd

from notifier.strategies.structure import nearest_level_beyond, trend_structure


def daily_regime_read(window: pd.DataFrame, thresholds: pd.Series) -> str | None:
    """"up", "down", or None - no reading. None covers two different cases,
    both meaning "do not trust a direction here": trend_structure's own
    convention (a trend resting only on the bootstrap guess, no observed
    CHoCH, is not a read - it is an artifact of where the window starts), and
    price sitting too close to a confirmed level ahead of it in the trend's
    own direction to trust that direction right now. "Too close" reuses the
    SAME threshold `thresholds` already means for pivot confirmation, rather
    than inventing a second, unrelated distance constant.
    """
    structure = trend_structure(window, thresholds)
    if structure.trend is None or structure.choch_count == 0:
        return None
    price = float(window["close"].iloc[-1])
    direction = "long" if structure.trend == "up" else "short"
    level = nearest_level_beyond(window, thresholds, price, direction)
    if level is not None and abs(level - price) < float(thresholds.iloc[-1]):
        return None
    return structure.trend
