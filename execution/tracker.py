"""Confirms trades against Bitget's real futures position, then tracks them
until the position closes. This replaces pure price-polling with
account-based detection: the bot never assumes a trade happened just because
price crossed a level — it waits to actually see the position on Bitget,
which is also how the future auto-execution phase will need to work anyway.
"""

import asyncio
from typing import Callable

from core.bitget_client import BitgetClient
from core.storage import Storage, Trade

PRICE_TOLERANCE = 0.01
SIZE_TOLERANCE = 0.05
ENTRY_TIMEOUT_SECONDS = 15 * 60
POLL_INTERVAL = 10.0


async def wait_for_signal_position(
    bitget: BitgetClient,
    symbol: str,
    direction: str,
    entry_price: float,
    size: float,
    poll_interval: float = POLL_INTERVAL,
    timeout_seconds: float = ENTRY_TIMEOUT_SECONDS,
) -> dict | None:
    """Polls until a Bitget position appears matching direction exactly and
    entry price/size within tolerance, or the timeout elapses (returns None).
    """
    elapsed = 0.0
    while elapsed < timeout_seconds:
        position = bitget.get_position(symbol)
        if position and _matches(position, direction, entry_price, size):
            return position
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return None


def check_position_now(bitget: BitgetClient, symbol: str) -> dict | None:
    """Single immediate check for /add: any live position at all is a match,
    since there's nothing proposed to compare it against."""
    return bitget.get_position(symbol)


def _matches(position: dict, direction: str, entry_price: float, size: float) -> bool:
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
    poll_interval: float = POLL_INTERVAL,
    on_close: Callable[[int, float], None] | None = None,
) -> None:
    """Polls the live position until it closes: keeps the actual stop/target
    columns in sync with whatever's really set on Bitget (in case the user
    adjusts them manually), and logs the real exit price/PnL once flat.
    """
    last_stop, last_target = None, None
    while True:
        await asyncio.sleep(poll_interval)
        position = bitget.get_position(symbol)

        if position is not None:
            stop, target = position["stop_loss"], position["take_profit"]
            if (stop, target) != (last_stop, last_target):
                storage.update_actual_stop_target(trade_id, stop, target)
                last_stop, last_target = stop, target
            continue

        exit_price, realized_pnl = _find_close(bitget, symbol)
        storage.close_trade(trade_id, exit_price=exit_price, realized_pnl=realized_pnl)
        if on_close:
            on_close(trade_id, exit_price)
        return


def _find_close(bitget: BitgetClient, symbol: str) -> tuple[float, float | None]:
    history = bitget.get_position_history(symbol, limit=5)
    if history:
        latest = history[0]
        return latest["exit_price"], latest["realized_pnl"]
    # Fallback if the history endpoint didn't have a matching record (e.g. a
    # field-name mismatch to fix once tested live): use the current mark
    # price and let storage derive PnL from entry/size instead of Bitget's
    # own realized-PnL figure.
    return bitget.get_mark_price(symbol), None


def format_close_message(trade: Trade) -> str:
    return (
        f"Trade #{trade.מספר_עסקה} closed: {trade.סימבול} {trade.כיוון}\n"
        f"Entry: {trade.מחיר_כניסה}  Exit: {trade.מחיר_יציאה}\n"
        f"P&L: {trade.רווח_הפסד:.2f}   R: {trade.מכפיל_R:.2f}"
        + ("\n(stop/target changed from the original plan)" if trade.changed_from_plan else "")
    )


def resume_open_trades(
    storage: Storage,
    bitget: BitgetClient,
    poll_interval: float = POLL_INTERVAL,
    on_close: Callable[[int, float], None] | None = None,
) -> list[asyncio.Task]:
    """Re-attaches tracker tasks for any trades left confirmed-open across a
    restart. Trades still pending (waiting for entry confirmation) when a
    crash happens are not auto-resumed — same accepted risk as a lost
    pre-approval signal; they just sit visible in storage.pending_trades().
    """
    tasks = []
    for trade in storage.open_trades():
        tasks.append(
            asyncio.create_task(
                track_position(
                    storage,
                    bitget,
                    trade.מספר_עסקה,
                    trade.סימבול,
                    poll_interval=poll_interval,
                    on_close=on_close,
                )
            )
        )
    return tasks
