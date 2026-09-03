"""Confirms trades against Bitget's real futures position, then tracks them
until the position closes. Detection is account-based rather than price-based:
the bot never assumes a trade happened because price crossed a level — it
waits to actually see the position, which is also how the future
auto-execution phase will need to work.

While a trade is open the tracker keeps three things in step with reality:
  - entry price and size, so scaling in (e.g. Strategy 1's 20% market + 80%
    limit split) ends up recorded as the true average entry and final size
  - the live stop/target, which usually live on TP/SL plan orders rather than
    the position object, and may be moved by hand mid-trade
  - partial exits: any size reduction is logged as a scale-out and the
    remainder keeps running

Signal-approved trades match on symbol + direction only. Tight price/size
tolerances would fight the split-entry method — the first tranche fills at
market for a fraction of the planned size — and the real numbers get read
back from the position anyway. `/add` keeps its tolerance check, where it
still guards against typos.

The Telegram text these events get reported as lives in execution/messages.py,
not here — this module only ever reads Bitget and writes the journal, no
string formatting beyond a log line. breakeven_price() is the one exception:
it's trade-state logic exit placement depends on too, not a formatting
concern that happens to live in the wrong file.
"""

import asyncio
import logging
from typing import Callable

from core.bitget_client import BitgetClient
from core.storage import Storage, Trade

logger = logging.getLogger(__name__)

PRICE_TOLERANCE = 0.01
SIZE_TOLERANCE = 0.05
ENTRY_TIMEOUT_SECONDS = 4 * 60 * 60  # Strategy 1's limit tranche can take hours to fill
POLL_INTERVAL = 10.0
_SIZE_EPSILON = 1e-12


async def wait_for_signal_position(
    bitget: BitgetClient,
    symbol: str,
    direction: str,
    poll_interval: float = POLL_INTERVAL,
    timeout_seconds: float = ENTRY_TIMEOUT_SECONDS,
) -> dict | None:
    """Polls until a position appears on this symbol/side, or the timeout
    elapses (returns None, and the caller cancels the pending row)."""
    elapsed = 0.0
    while elapsed < timeout_seconds:
        try:
            position = bitget.get_position(symbol, direction)
            if position:
                return position
        except Exception:
            logger.exception("Position check failed for %s %s; will retry", symbol, direction)
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return None


def check_position_now(bitget: BitgetClient, symbol: str, direction: str | None = None) -> dict | None:
    """Single immediate check, used by /add where the position already exists."""
    return bitget.get_position(symbol, direction)


def matches_expected(position: dict, direction: str, entry_price: float, size: float) -> bool:
    """Tolerance check used by /add only."""
    if position["direction"] != direction:
        return False
    price_ok = abs(position["entry_price"] - entry_price) <= entry_price * PRICE_TOLERANCE
    size_ok = abs(position["size"] - size) <= size * SIZE_TOLERANCE
    return price_ok and size_ok


async def track_position(
    storage: Storage,
    bitget: BitgetClient,
    trade_id: int,
    symbol: str,
    direction: str,
    poll_interval: float = POLL_INTERVAL,
    on_close: Callable[[int, float], None] | None = None,
    on_partial: Callable[[int, float, float], None] | None = None,
    on_resize: Callable[[float], None] | None = None,
    on_scale_in: Callable[[int], None] | None = None,
) -> None:
    trade = storage.get_trade(trade_id)
    last_size = trade.גודל_פוזיציה or 0.0
    last_entry = trade.מחיר_כניסה
    last_stop, last_target = trade.סטופ_לוס_בפועל, trade.יעד_רווח_בפועל

    # A fill can arrive in pieces, and one logical fill should be one message.
    # The notification is therefore held until a poll passes with no further
    # growth. This cannot wait forever: the resting leg has a fixed size, so
    # growth necessarily stops once it is fully filled.
    scale_in_pending = False
    scale_in_size = 0.0

    while True:
        await asyncio.sleep(poll_interval)

        try:
            position = bitget.get_position(symbol, direction)
        except Exception:
            logger.exception("Poll failed for trade #%s (%s %s); will retry", trade_id, symbol, direction)
            continue

        if position is None:
            # Any held scale-in message dies with the position. The close
            # message carries the final entry, size and P&L, so it is strictly
            # more informative - and "your position grew to X" arriving after
            # "trade closed" describes something that no longer exists.
            exit_price, realized_pnl = _final_close(bitget, symbol, direction)
            storage.close_trade(trade_id, exit_price=exit_price, realized_pnl=realized_pnl)
            if on_close:
                on_close(trade_id, exit_price)
            return

        grew = position["size"] > last_size + _SIZE_EPSILON

        # Scaling in changes the average entry and total size.
        if grew or not _close(position["entry_price"], last_entry):
            storage.resync_position(trade_id, position["entry_price"], position["size"])
            last_entry = position["entry_price"]
            # A second entry leg filling means any exit order sized to the
            # first one now covers too little of the position.
            if on_resize and grew:
                on_resize(position["size"])

        if grew:
            scale_in_pending = True
            scale_in_size = position["size"]
        elif scale_in_pending:
            # Growth has settled. Fire unless the position shrank in the
            # meantime, which means a scale-out overtook the fill and the
            # figures the message would quote are already out of date.
            if on_scale_in and position["size"] >= scale_in_size - _SIZE_EPSILON:
                on_scale_in(trade_id)
            scale_in_pending = False

        # A size reduction with the position still alive is a scale-out.
        if position["size"] < last_size - _SIZE_EPSILON:
            closed_so_far = (trade.גודל_פוזיציה or last_size) - position["size"]
            storage.record_partial(trade_id, closed_so_far, position["realized_pnl"])
            if on_partial:
                on_partial(trade_id, closed_so_far, position["realized_pnl"])

        last_size = position["size"]

        try:
            stop, target = bitget.get_stop_target(symbol, direction)
        except Exception:
            logger.exception("Stop/target check failed for trade #%s; keeping previous values", trade_id)
            continue

        if (stop, target) != (last_stop, last_target):
            storage.update_actual_stop_target(trade_id, stop, target)
            last_stop, last_target = stop, target


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9


def _final_close(bitget: BitgetClient, symbol: str, direction: str) -> tuple[float, float | None]:
    """Bitget's position history is authoritative: closeAvgPrice and netProfit
    already aggregate every partial close of that position, and netProfit is
    after fees. Falls back to mark price if the record isn't there yet, letting
    storage derive P&L instead — less accurate, but never a crash."""
    try:
        record = bitget.find_closed_position(symbol, direction)
        if record:
            return record["exit_price"], record["realized_pnl"]
    except Exception:
        logger.exception("Could not read close record for %s %s; falling back to mark price", symbol, direction)
    return bitget.get_mark_price(symbol), None


def closing_exits(bitget, trade: Trade) -> list[dict]:
    """Each exit this trade actually went off at, or [] if unavailable.

    Never raises: this decorates a notification, and a trade that has closed
    has closed whether or not the fill history can be read. An empty list
    puts the message back on the single aggregate figure.
    """
    try:
        return bitget.get_closing_exits(trade.סימבול, trade.גודל_פוזיציה or 0.0)
    except Exception:
        logger.exception("Could not read the closing fills for %s; reporting the average only", trade.סימבול)
        return []


def take_profit_coverage(bitget: BitgetClient, symbol: str, direction: str, position_size: float) -> float:
    """How much of the position an existing take-profit would actually close.

    Two mechanisms have to be counted, because either can be in play and they
    are reported by different endpoints:

      - A position-level target (Bitget's own TP/SL panel, planType pos_profit)
        carries size 0, meaning "all closable" - it covers whatever the
        position happens to be, including size added later.
      - An order-level target: the bot's own partial take-profit, a resting
        reduce-only limit, and the per-leg profit presets. These are sized to
        a FIXED quantity and do NOT grow when a limit leg fills, which is the
        gap this whole notification exists to surface.
    """
    covered = 0.0
    for order in bitget.get_plan_orders(symbol, direction):
        if not order["is_target"]:
            continue
        if order["size"] == 0:
            return position_size  # all closable: covers the position whatever its size
        covered += order["size"]

    for order in bitget.get_open_orders(symbol):
        if (order.get("tradeSide") or "").lower() != "close":
            continue  # an entry leg, not an exit
        if order.get("posSide") and order.get("posSide") != direction:
            continue
        covered += float(order.get("size") or 0)

    return covered


def breakeven_price(trade: Trade) -> float | None:
    """Where this trade's stop actually goes when the partial fills, or None
    if no breakeven is armed at all.

    A scanner trade's stored breakeven is `plan_entry`: the market leg's
    EXPECTED fill blended with the limit level, computed before either had
    happened. The position's own average entry is strictly better, because
    the tracker resyncs it from Bitget on every fill - so that is what gets
    used, and a leg filling later moves it automatically.

    XAGUSDT #17 is the case that exposed it. The alert planned 63.66 assuming
    both legs filled; only the 0.17 market leg did, at 64.37. Moving that
    stop to 63.66 would not have been breakeven, it would have locked in a
    loss on the remainder - and the split-entry legs are exactly where the
    two prices diverge most, since the limit sits furthest from market by
    construction.

    A trade adopted with /manage keeps the price Dror typed. He chose it
    against the live chart, and the whole reason that command exists is that
    the bot's own idea of the trade was not good enough.
    """
    if trade.breakeven_stop is None:
        return None
    if trade.exit_managed:
        return trade.breakeven_stop
    return trade.מחיר_כניסה if trade.מחיר_כניסה is not None else trade.breakeven_stop


def resume_open_trades(
    storage: Storage,
    bitget: BitgetClient,
    poll_interval: float = POLL_INTERVAL,
    on_close: Callable[[int, float], None] | None = None,
    on_partial: Callable[[int, float, float], None] | None = None,
    on_scale_in: Callable[[int], None] | None = None,
    on_resize: Callable[[int, float], None] | None = None,
) -> list[asyncio.Task]:
    """Re-attaches trackers for trades left open across a restart. Trades still
    pending at that point aren't auto-resumed — they stay visible in
    storage.pending_trades() rather than silently resuming a stale wait.

    A leg that filled while the service was down shows up as growth on the
    first poll and does notify. That is deliberate, and differs from the
    pending-break watch which is dropped on restart: this reports the position
    as it stands right now, rather than offering an action based on a past
    event, so it cannot be acted on stale.

    on_resize takes (trade_id, size) rather than track_position's own (size),
    because this loop re-attaches many trades at once and the callback needs
    to know which one grew. DOGEUSDT and QQQUSDT both hit this gap live on
    2026-09-03: a deploy restarted the service while their limit legs were
    still resting, the legs filled afterward, and the take-profit stayed
    sized to the market leg alone because nothing re-attached this hook -
    only on_scale_in fired, which reports the shortfall but never closes it.
    """
    tasks = []
    for trade in storage.open_trades():
        trade_id = trade.מספר_עסקה
        tasks.append(
            asyncio.create_task(
                track_position(
                    storage,
                    bitget,
                    trade_id,
                    trade.סימבול,
                    trade.כיוון,
                    poll_interval=poll_interval,
                    on_close=on_close,
                    on_partial=on_partial,
                    on_scale_in=on_scale_in,
                    on_resize=(lambda size, tid=trade_id: on_resize(tid, size)) if on_resize else None,
                )
            )
        )
    return tasks
