"""Lets the user register a trade they took themselves (outside any notifier
signal): /add <symbol> [long|short]. All trade data — direction, entry, size,
leverage, and the live stop/target — comes straight from the Bitget position,
so the only thing the bot asks for is the strategy tag, which it can't know.

The direction argument is optional and only needed in hedge mode when both a
long and a short are open on the same symbol.
"""

import asyncio
import logging
from typing import Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.messages import format_close_message, format_partial_message, format_scale_in_message
from execution.tracker import closing_exits, take_profit_coverage, track_position

logger = logging.getLogger(__name__)

ASK_STRATEGY = 1
_PENDING_TRADE_KEY = "pending_trade_id"


def make_add_conversation(
    storage: Storage,
    bitget: BitgetClient,
    tag_options: list[str],
    on_partial: Callable[[int, float, float | None], None] | None = None,
    reoffer: Callable[[int], "asyncio.Future[str]"] | None = None,
) -> ConversationHandler:
    """Builds the /add conversation, closing over storage/bitget so
    core.telegram_bot stays free of dependencies on the rest of the app.

    `on_partial` lets the caller supply the scanner's own scale-out handler,
    so a hand-added trade takes the same partial-fill path as every other one
    - including honouring an exit plan armed with /manage. Left None it falls
    back to a local notification, which is all this module can do alone.

    `tag_options` is the STRICT, tappable list the strategy question offers -
    every registered strategy's own tag plus a fixed "Other / discretionary".
    Free text used to be accepted here, and every tag ever actually typed
    turned out to be a hand-spelled (often malformed) guess at a real
    strategy tag anyway - 'strategy 1', 'strategy 1 4h', 'Strategy 1 1h' -
    never a genuinely separate manual category. Dror's call, 2026-08-26:
    reuse the bot's own canonical tags so a hand-placed trade groups into the
    SAME weekly-review bucket as the bot's own trades of that shape, and make
    picking one a tap rather than a typed match, so no reply can silently
    fail to route - the exact way XAGUSDT #17 ended up managed by nobody.
    Caller supplies the list (main.py, from build_strategies()) rather than
    this module importing it, to keep this free of app-level dependencies.
    """

    async def handle_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not context.args:
            await update.message.reply_text(
                "Usage: /add <symbol> [long|short]  - register a position you opened yourself\n"
                "       /add <signal number>        - offer an expired signal again"
            )
            return ConversationHandler.END

        # A NUMBER MEANS A SIGNAL, NOT A SYMBOL. No symbol is all digits, so
        # the two readings cannot collide.
        #
        # This exists because the symbol form ends by ASKING for the strategy
        # tag, and a hand-typed tag is one character from being managed by
        # nobody: XAGUSDT #17 was entered as "Strategy 1 1h" against a
        # "Strategy 1 1H" alert and went its whole life with no breakeven, no
        # partial and no runner, because no routing set recognised it. Naming
        # the signal instead of the strategy removes the chance to mistype it.
        if reoffer is not None and context.args[0].isdigit():
            await update.message.reply_text(await reoffer(int(context.args[0])))
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
        # Two per row: 12 buttons (the 11 registered tags plus "Other /
        # discretionary") reads as six short rows rather than one long one.
        numbered = list(enumerate(tag_options))
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(t, callback_data=f"tag:{i}") for i, t in numbered[j : j + 2]]
                for j in range(0, len(numbered), 2)
            ]
        )
        await update.message.reply_text(
            f"Found it — {symbol} {position['direction']}, entry {position['entry_price']:.2f}, "
            f"size {position['size']:.6f}, {position['leverage']:.0f}x."
            f"{warning}\nWhat strategy/setup was this?",
            reply_markup=keyboard,
        )
        return ASK_STRATEGY

    async def handle_strategy_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            logger.warning("Could not answer /add's tag-selection tap; processing it anyway")

        trade_id = context.chat_data.pop(_PENDING_TRADE_KEY, None)
        if trade_id is None:
            return ConversationHandler.END

        _, _, index_text = query.data.partition(":")
        try:
            tag = tag_options[int(index_text)]
        except (ValueError, IndexError):
            logger.warning("Unrecognized /add tag selection %r for trade #%s", query.data, trade_id)
            return ConversationHandler.END
        storage.set_strategy_tag(trade_id, tag)

        chat_id = update.effective_chat.id
        trade = storage.get_trade(trade_id)

        def on_close(closed_id: int, price: float) -> None:
            closed = storage.get_trade(closed_id)
            text = format_close_message(closed, closing_exits(bitget, closed))
            asyncio.create_task(context.bot.send_message(chat_id=chat_id, text=text))

        def local_on_partial(closed_id: int, closed_size: float, realized_pnl: float | None) -> None:
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
                on_partial=on_partial or local_on_partial,
                on_scale_in=on_scale_in,
            )
        )

        try:
            # Editing rather than a fresh reply also clears the tag buttons -
            # the same convention core.telegram_bot's own Approve/Reject
            # flow uses, so a decision always reads as "already answered"
            # rather than leaving a stale, still-tappable keyboard behind.
            await query.edit_message_text(f"{query.message.text}\n\nTagged as '{tag}'. Tracking until it closes.")
        except Exception:
            logger.exception("Could not confirm the tag for trade #%s; tracking it regardless", trade_id)
        return ConversationHandler.END

    return ConversationHandler(
        entry_points=[CommandHandler("add", handle_add)],
        states={ASK_STRATEGY: [CallbackQueryHandler(handle_strategy_reply, pattern="^tag:")]},
        fallbacks=[],
    )


def _safe_stop_target(bitget: BitgetClient, symbol: str, position: dict):
    try:
        return bitget.get_stop_target(symbol, position["direction"])
    except Exception:
        logger.exception("Could not read stop/target for %s; using position presets", symbol)
        return position["stop_loss"], position["take_profit"]
