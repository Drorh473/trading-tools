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


def daily_regime_read(
    window: pd.DataFrame, thresholds: pd.Series, proximity_threshold: float | None = None
) -> str | None:
    """"up", "down", or None - no reading. None covers two different cases,
    both meaning "do not trust a direction here": trend_structure's own
    convention (a trend resting only on the bootstrap guess, no observed
    CHoCH, is not a read - it is an artifact of where the window starts), and
    price sitting too close to a confirmed level ahead of it in the trend's
    own direction to trust that direction right now.

    `proximity_threshold`, when None (the default), reuses `thresholds.iloc
    [-1]` as the distance cutoff - today's exact behaviour, unchanged. That
    default was never actually a deliberate choice: `thresholds` is built
    for pivot CONFIRMATION (how far price must move for a swing to count as
    real), and reusing it for "how close counts as too close to trust the
    trend" is a different job that happened to borrow the same number. Pass
    an explicit value to test whether the result is sensitive to that
    choice, rather than trusting an unexamined constant.
    """
    structure = trend_structure(window, thresholds)
    if structure.trend is None or structure.choch_count == 0:
        return None
    price = float(window["close"].iloc[-1])
    direction = "long" if structure.trend == "up" else "short"
    level = nearest_level_beyond(window, thresholds, price, direction)
    cutoff = thresholds.iloc[-1] if proximity_threshold is None else proximity_threshold
    if level is not None and abs(level - price) < float(cutoff):
        return None
    return structure.trend


def daily_regime_from_bars(
    daily_bars: pd.DataFrame, proximity_atr_multiple: float | None = None
) -> str | None:
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

    `proximity_atr_multiple`, when set, computes an independent proximity
    cutoff (ATR at the last available bar, times this multiple) instead of
    reusing the pivot-confirmation threshold - see daily_regime_read's own
    docstring for why that reuse was never actually tested. None (default)
    preserves today's exact behaviour.
    """
    window, _structure = structure_context(daily_bars, atr_multiple=ATR_MULTIPLE, atr_period=ATR_PERIOD)
    atr_full = atr(daily_bars, ATR_PERIOD)
    thresholds = (atr_full * ATR_MULTIPLE).iloc[-len(window):].reset_index(drop=True)
    proximity = None if proximity_atr_multiple is None else float(atr_full.iloc[-1]) * proximity_atr_multiple
    return daily_regime_read(window, thresholds, proximity)
