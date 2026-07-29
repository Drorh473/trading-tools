"""Lets the user register a trade they took themselves (outside any notifier
signal): /add <symbol>. All trade data (direction, entry, size, current
stop/target) comes straight from the live Bitget position — the only thing
the bot can't know is the strategy/setup tag, which it asks for as a single
follow-up question with no timeout or default.
"""

import asyncio

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.tracker import check_position_now, format_close_message, track_position

ASK_STRATEGY = 1
_PENDING_TRADE_KEY = "pending_trade_id"


def make_add_conversation(storage: Storage, bitget: BitgetClient) -> ConversationHandler:
    """Builds the /add conversation, closing over storage/bitget so
    core.telegram_bot stays free of dependencies on the rest of the app.
    """

    async def handle_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not context.args:
            await update.message.reply_text("Usage: /add <symbol>")
            return ConversationHandler.END

        symbol = context.args[0].upper()

        if storage.has_open_or_pending(symbol):
            await update.message.reply_text(f"Already tracking a trade on {symbol}.")
            return ConversationHandler.END

        position = check_position_now(bitget, symbol)
        if position is None:
            await update.message.reply_text(f"No open position found for {symbol} on Bitget.")
            return ConversationHandler.END

        trade_id = storage.create_pending(symbol=symbol, direction=position["direction"])
        storage.confirm_entry(
            trade_id,
            entry_price=position["entry_price"],
            position_size=position["size"],
            actual_stop=position["stop_loss"],
            actual_target=position["take_profit"],
            leverage=position["leverage"],
        )

        context.chat_data[_PENDING_TRADE_KEY] = trade_id
        await update.message.reply_text(
            f"Found it — {symbol} {position['direction']}, entry {position['entry_price']}, "
            f"size {position['size']}. What strategy/setup was this?"
        )
        return ASK_STRATEGY

    async def handle_strategy_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        trade_id = context.chat_data.pop(_PENDING_TRADE_KEY, None)
        if trade_id is None:
            return ConversationHandler.END

        tag = update.message.text.strip()
        storage.set_strategy_tag(trade_id, tag)

        chat_id = update.effective_chat.id
        symbol = storage.get_trade(trade_id).סימבול

        def on_close(closed_id: int, price: float) -> None:
            message = format_close_message(storage.get_trade(closed_id))
            asyncio.create_task(context.bot.send_message(chat_id=chat_id, text=message))

        asyncio.create_task(track_position(storage, bitget, trade_id, symbol, on_close=on_close))

        await update.message.reply_text(f"Tagged trade #{trade_id} as '{tag}'. Tracking until it closes.")
        return ConversationHandler.END

    return ConversationHandler(
        entry_points=[CommandHandler("add", handle_add)],
        states={ASK_STRATEGY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_strategy_reply)]},
        fallbacks=[],
    )
