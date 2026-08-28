"""BTC's own trend, read the way Dror reads it before every trade: direction
plus whether price sits too close to a level that could end it right now.
Standalone - this module answers whether that read is a real, persistent
market phenomenon, independent of any specific strategy's trade history. It
is not wired into any strategy's evaluate() yet.
"""
import pandas as pd

from notifier.strategies.indicators import atr
from notifier.strategies.structure import nearest_level_beyond, structure_context, trend_structure

# Matched to STRUCTURE_ATR_MULTIPLE (rsi_fib_reversal.py) and the growing-
# window defaults every other structure_context caller uses - one definition
# of "what counts as a confirmed trend", not a second one invented here.
ATR_MULTIPLE = 1.25
ATR_PERIOD = 14


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


def daily_regime_from_bars(daily_bars: pd.DataFrame) -> str | None:
    """daily_regime_read, given raw daily bars instead of an already-grown
    window - the ONE implementation of "search a growing window for a
    confirmed trend, then apply the level-proximity check", shared between
    live strategy gates and backtest/btc_daily_regime_persistence.py rather
    than each hand-rolling the same structure_context + threshold-slice
    reconstruction and risking the two drifting apart.

    ATR is measured over daily_bars in full (matching structure_context's
    own convention - the threshold is warmed up at each window's first bar
    regardless of how far the window ends up growing) and then sliced to
    match whatever window structure_context actually settled on, since
    structure_context returns the window but not the threshold slice it used
    internally to find it.
    """
    window, _structure = structure_context(daily_bars, atr_multiple=ATR_MULTIPLE, atr_period=ATR_PERIOD)
    thresholds_full = atr(daily_bars, ATR_PERIOD) * ATR_MULTIPLE
    thresholds = thresholds_full.iloc[-len(window):].reset_index(drop=True)
    return daily_regime_read(window, thresholds)
