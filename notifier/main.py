"""Entrypoint: runs the 24/7 scanner loop plus the Telegram Approve/Reject
bot side by side in one process. No concrete strategies are wired in yet —
add them to the `strategies` list below as they're described.
"""

import asyncio

from config import settings
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import NotifierBot
from execution.executor import ManualExecutor
from execution.manual_entry import make_add_conversation
from execution.tracker import format_close_message, resume_open_trades
from notifier.scanner import Scanner
from notifier.watchlist import WATCHLIST

ACCOUNT_EQUITY = 1000.0
RISK_PCT = 0.01  # 1-2% per trade, hard-capped at 2% in risk_sizing.plan_position
LEVERAGE = 1.0


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
        strategies=[],  # add strategies here as they're built
        equity=ACCOUNT_EQUITY,
        risk_pct=RISK_PCT,
        leverage=LEVERAGE,
    )

    bot.app.add_handler(make_add_conversation(storage, bitget))

    await bot.start_polling()

    def on_resumed_close(trade_id: int, price: float) -> None:
        asyncio.create_task(bot.send_message(format_close_message(storage.get_trade(trade_id))))

    resume_open_trades(storage, bitget, on_close=on_resumed_close)

    try:
        await scanner.run_forever()
    finally:
        await bot.stop()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
