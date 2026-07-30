"""Entrypoint: runs the scanner loop plus the Telegram Approve/Reject bot side
by side in one process.

Scans are aligned to candle closes, driven by each strategy's declared
timeframe (both current strategies are 1H — Strategy 1 explicitly requires
1h+, and Strategy 2 is running its longer-timeframe variant). Equity is read
live from Bitget on every scan rather than hardcoded, so position sizing tracks
the real account balance.
"""

import asyncio
import logging

from config import settings
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import NotifierBot
from execution.executor import ManualExecutor
from execution.manual_entry import make_add_conversation
from execution.tracker import format_close_message, format_partial_message, resume_open_trades
from notifier.scanner import Scanner
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.watchlist import WATCHLIST

RISK_PCT = 0.01  # 1-2% per trade, hard-capped at 2% in risk_sizing.plan_position
MAX_TOTAL_RISK_PCT = 0.06  # aggregate ceiling across all open trades
# Leverage is solved per trade to fit the margin left after other open trades;
# this only caps how high it may go.
MAX_LEVERAGE = 20.0


async def async_main() -> None:
    bitget = client_from_settings(settings)
    storage = Storage(settings.trades_db_path)
    bot = NotifierBot(settings.telegram_bot_token, settings.telegram_chat_id)
    executor = ManualExecutor()

    scanner = Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=executor,
        watchlist=WATCHLIST,
        strategies=[RsiFibReversal(), EmaTrendFollowing()],
        risk_pct=RISK_PCT,
        max_leverage=MAX_LEVERAGE,
        max_total_risk_pct=MAX_TOTAL_RISK_PCT,
    )

    bot.app.add_handler(make_add_conversation(storage, bitget))
    await bot.start_polling()

    def on_resumed_close(trade_id: int, price: float) -> None:
        asyncio.create_task(bot.send_message(format_close_message(storage.get_trade(trade_id))))

    def on_resumed_partial(trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        text = format_partial_message(storage.get_trade(trade_id), closed_size, realized_pnl)
        asyncio.create_task(bot.send_message(text))

    resume_open_trades(storage, bitget, on_close=on_resumed_close, on_partial=on_resumed_partial)

    try:
        await scanner.run_forever()
    finally:
        await bot.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
