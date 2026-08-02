"""Paper-simulates dispatched signals against real candles: walks forward
from when each signal fired to see whether its stop or target was hit first,
net of round-trip taker fees. This is what lets the weekly report score a
strategy's own output separately from what you chose to approve or reject -
before this, nothing recorded a signal's outcome unless it was approved and
became a real, tracked trade.

Resolution always reads 15m candles regardless of the signal's own
timeframe, since that is fine enough to sequence stop-vs-target correctly
for every strategy instance without needing a per-timeframe candle source.
If a single 15m bar's range contains both levels, which one actually came
first inside that bar is unknowable from OHLC alone; the conservative
assumption (the stop) is used rather than guessing in the report's favor.
"""

from datetime import datetime, timezone

from core.bitget_client import BitgetClient
from core.storage import SignalRecord, Storage
from notifier.scanner import bars_dataframe

RESOLVE_TIMEFRAME = "15m"
CANDLE_FETCH_LIMIT = 1000  # ~10.4 days of 15m bars - comfortably covers a week's signals
TAKER_FEE_RATE = 0.0006  # per side; a round trip pays this twice


def resolve_pending(storage: Storage, bitget: BitgetClient) -> int:
    """Attempts to resolve every signal still waiting on an outcome. Returns
    how many resolved this pass; the rest are still running and stay
    unresolved for the next call."""
    resolved = 0
    for record in storage.unresolved_signals():
        r = _resolve_one(bitget, record)
        if r is not None:
            storage.record_paper_outcome(record.id, r)
            resolved += 1
    return resolved


def _resolve_one(bitget: BitgetClient, record: SignalRecord) -> float | None:
    dispatched = datetime.fromisoformat(record.dispatched_at).astimezone(timezone.utc).replace(tzinfo=None)
    candles = bitget.get_candles(record.symbol, granularity=RESOLVE_TIMEFRAME, limit=CANDLE_FETCH_LIMIT, closed_only=True)
    bars = bars_dataframe(candles)
    after = bars[bars["ts"] >= dispatched]
    if after.empty:
        return None  # too recent - Bitget hasn't printed a candle since dispatch yet

    long = record.direction == "long"
    for _, bar in after.iterrows():
        hit_stop = bar["low"] <= record.stop_loss if long else bar["high"] >= record.stop_loss
        hit_target = bar["high"] >= record.take_profit if long else bar["low"] <= record.take_profit
        if hit_stop:
            return _net_r(record, won=False)
        if hit_target:
            return _net_r(record, won=True)
    return None  # neither level reached yet - still running


def _net_r(record: SignalRecord, won: bool) -> float:
    risk_per_unit = abs(record.entry_price - record.stop_loss)
    reward_per_unit = abs(record.take_profit - record.entry_price)
    raw_r = reward_per_unit / risk_per_unit if won else -1.0
    fee_r = (2 * TAKER_FEE_RATE * record.entry_price) / risk_per_unit
    return raw_r - fee_r
