"""Lets the user register a trade they took themselves (outside any notifier
signal): /add <symbol> [long|short]. All trade data — direction, entry, size,
leverage, and the live stop/target — comes straight from the Bitget position,
so the only thing the bot asks for is the strategy tag, which it can't know.

The direction argument is optional and only needed in hedge mode when both a
long and a short are open on the same symbol.
"""

import asyncio
import logging

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.tracker import (
    format_close_message,
    format_partial_message,
    format_scale_in_message,
    take_profit_coverage,
    track_position,
)

logger = logging.getLogger(__name__)

ASK_STRATEGY = 1
_PENDING_TRADE_KEY = "pending_trade_id"


def make_add_conversation(storage: Storage, bitget: BitgetClient) -> ConversationHandler:
    """Builds the /add conversation, closing over storage/bitget so
    core.telegram_bot stays free of dependencies on the rest of the app."""

    async def handle_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not context.args:
            await update.message.reply_text("Usage: /add <symbol> [long|short]")
            return ConversationHandler.END

        symbol = context.args[0].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        direction = context.args[1].lower() if len(context.args) > 1 else None
        if direction is not None and direction not in ("long", "short"):
            await update.message.reply_text("Direction must be 'long' or 'short'.")
            return ConversationHandler.END

        if storage.has_open_or_pending(symbol):
            await update.message.reply_text(f"Already tracking a trade on {symbol}.")
            return ConversationHandler.END

        try:
            positions = bitget.get_positions(symbol)
        except Exception as exc:
            logger.exception("Could not read positions for %s", symbol)
            if "does not exist" in str(exc).lower() or "40034" in str(exc):
                await update.message.reply_text(f"{symbol} isn't a symbol Bitget recognizes — check the spelling.")
            else:
                await update.message.reply_text(f"Couldn't reach Bitget to check {symbol}. Try again shortly.")
            return ConversationHandler.END

        if direction is not None:
            positions = [p for p in positions if p["direction"] == direction]

        if not positions:
            await update.message.reply_text(f"No open position found for {symbol} on Bitget.")
            return ConversationHandler.END
        if len(positions) > 1:
            await update.message.reply_text(
                f"Both a long and a short are open on {symbol}. Specify which: /add {symbol} long|short"
            )
            return ConversationHandler.END

        position = positions[0]
        stop, target = _safe_stop_target(bitget, symbol, position)

        trade_id = storage.create_pending(symbol=symbol, direction=position["direction"])
        storage.confirm_entry(
            trade_id,
            entry_price=position["entry_price"],
            position_size=position["size"],
            actual_stop=stop,
            actual_target=target,
            leverage=position["leverage"],
        )

        context.chat_data[_PENDING_TRADE_KEY] = trade_id
        warning = "" if stop is not None else "\nNote: no stop-loss set on Bitget — R can't be computed until you set one."
        await update.message.reply_text(
            f"Found it — {symbol} {position['direction']}, entry {position['entry_price']:.2f}, "
            f"size {position['size']:.6f}, {position['leverage']:.0f}x."
            f"{warning}\nWhat strategy/setup was this?"
        )
        return ASK_STRATEGY

    async def handle_strategy_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        trade_id = context.chat_data.pop(_PENDING_TRADE_KEY, None)
        if trade_id is None:
            return ConversationHandler.END

        tag = update.message.text.strip()
        storage.set_strategy_tag(trade_id, tag)

        chat_id = update.effective_chat.id
        trade = storage.get_trade(trade_id)

        def on_close(closed_id: int, price: float) -> None:
            text = format_close_message(storage.get_trade(closed_id))
            asyncio.create_task(context.bot.send_message(chat_id=chat_id, text=text))

        def on_partial(closed_id: int, closed_size: float, realized_pnl: float | None) -> None:
            text = format_partial_message(storage.get_trade(closed_id), closed_size, realized_pnl)
            asyncio.create_task(context.bot.send_message(chat_id=chat_id, text=text))

        def on_scale_in(scaled_id: int) -> None:
            # Hand-placed trades get this too: the notification is about the
            # position, not about anything the bot did. A limit leg filling on
            # a trade added with /add changes the real entry and the real risk
            # exactly as it would on a bot-placed one.
            scaled = storage.get_trade(scaled_id)
            try:
                covered = take_profit_coverage(
                    bitget, scaled.סימבול, scaled.כיוון, scaled.גודל_פוזיציה or 0.0
                )
            except Exception:
                logger.exception("Could not read take-profit coverage for %s", scaled.סימבול)
                covered = None
            text = format_scale_in_message(scaled, covered)
            asyncio.create_task(context.bot.send_message(chat_id=chat_id, text=text))

        asyncio.create_task(
            track_position(
                storage,
                bitget,
                trade_id,
                trade.סימבול,
                trade.כיוון,
                on_close=on_close,
                on_partial=on_partial,
                on_scale_in=on_scale_in,
            )
        )

        await update.message.reply_text(f"Tagged trade #{trade_id} as '{tag}'. Tracking until it closes.")
        return ConversationHandler.END

    return ConversationHandler(
        entry_points=[CommandHandler("add", handle_add)],
        states={ASK_STRATEGY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_strategy_reply)]},
        fallbacks=[],
    )


def _safe_stop_target(bitget: BitgetClient, symbol: str, position: dict):
    try:
        return bitget.get_stop_target(symbol, position["direction"])
    except Exception:
        logger.exception("Could not read stop/target for %s; using position presets", symbol)
        return position["stop_loss"], position["take_profit"]
