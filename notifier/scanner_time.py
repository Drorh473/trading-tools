"""When a candle closes, and how long an Approve/Reject offer should stay
live for it.

Split out of notifier.scanner as a small, dependency-free module rather
than left there, so collaborators extracted from Scanner (e.g.
PendingBreakWatcher) can use it without importing scanner.py itself and
creating a cycle.
"""

from __future__ import annotations

import time

from notifier.strategies.base import TIMEFRAME_SECONDS

CANDLE_CLOSE_DELAY = 30.0  # let Bitget settle the just-closed candle before reading it
SIGNAL_EXPIRY_FLOOR = 60.0
SIGNAL_EXPIRY_CEILING = 1800.0


def _split_reference_key(tf: str) -> tuple[str, str] | None:
    """(symbol, timeframe) if `tf` is a "SYMBOL@TIMEFRAME" cross-symbol
    reference key (see RsiFibReversal's market_trend_symbol), else None.

    A strategy that needs a REFERENCE symbol's own bars - not the one
    currently being scanned - declares it this way in its own `timeframes`,
    reusing the existing per-strategy timeframe-fetch mechanism instead of
    inventing a parallel one. The compound key travels unchanged as the dict
    key in bars_by_timeframe, so strategy.evaluate() looks it up exactly the
    way it declared it.
    """
    if "@" not in tf:
        return None
    symbol, _, timeframe = tf.partition("@")
    return symbol, timeframe


def seconds_until_next_close(timeframe: str, now: float | None = None) -> float:
    """Seconds until the next candle of this timeframe closes, plus a small
    settle delay.

    A cross-symbol reference key still closes on its own real timeframe's
    clock (an hourly reference candle closes hourly, same as any other 1H
    series), so this reads only the timeframe half of it for the lookup.
    """
    ref = _split_reference_key(timeframe)
    period = TIMEFRAME_SECONDS[ref[1] if ref else timeframe]
    now = time.time() if now is None else now
    return (period - (now % period)) + CANDLE_CLOSE_DELAY


def signal_expiry_seconds(timeframe: str, now: float | None = None) -> float:
    """The timer half of an Approve/Reject offer's expiry (see
    core.telegram_bot for the movement half it races against).

    Anchored to when this signal's OWN candle next closes - the point the
    scanner would already be re-evaluating this setup with fresh eyes, so an
    offer still unacted on past that point is stale on structural grounds
    alone, not just convention. Floored so a signal fired late in its candle
    still leaves a minute to read and tap; capped so a slow timeframe (1D
    can be most of a day away from its next close) can't leave a live-money
    offer sitting for hours just because price hasn't moved enough yet to
    trip the other cutoff.
    """
    return min(max(seconds_until_next_close(timeframe, now), SIGNAL_EXPIRY_FLOOR), SIGNAL_EXPIRY_CEILING)
