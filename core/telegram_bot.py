"""Telegram integration.

Two ways this gets used:
  - One-shot messages (journal reports, weekly review): `send_message()`.
  - The always-on notifier: `NotifierBot`, which sends a signal with inline
    Approve/Reject buttons and dispatches the button press back to whoever
    registered a handler for it (kept generic here so this module doesn't
    need to import notifier/execution code and create a circular import).
"""

import asyncio
import itertools
from dataclasses import dataclass
from typing import Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes


def send_message(token: str, chat_id: str, text: str) -> None:
    async def _send():
        from telegram import Bot

        async with Bot(token) as bot:
            await bot.send_message(chat_id=chat_id, text=text)

    asyncio.run(_send())


@dataclass
class PendingSignal:
    text: str
    on_approve: Callable[[], None]
    on_reject: Callable[[], None] | None = None


class NotifierBot:
    """Runs the long-lived polling loop for the 24/7 notifier process."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._pending: dict[str, PendingSignal] = {}
        self._ids = itertools.count()
        self.app = Application.builder().token(token).build()
        self.app.add_handler(CallbackQueryHandler(self._on_callback))

    async def send_message(self, text: str) -> None:
        """Plain push notification, e.g. when a tracked trade closes."""
        await self.app.bot.send_message(chat_id=self.chat_id, text=text)

    async def send_signal(
        self,
        text: str,
        on_approve: Callable[[], None],
        on_reject: Callable[[], None] | None = None,
    ) -> None:
        callback_id = str(next(self._ids))
        self._pending[callback_id] = PendingSignal(text, on_approve, on_reject)

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"approve:{callback_id}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject:{callback_id}"),
                ]
            ]
        )
        await self.app.bot.send_message(chat_id=self.chat_id, text=text, reply_markup=keyboard)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        action, _, callback_id = query.data.partition(":")
        pending = self._pending.pop(callback_id, None)
        if pending is None:
            await query.edit_message_text(f"{query.message.text}\n\n(already handled)")
            return

        if action == "approve":
            pending.on_approve()
            await query.edit_message_text(f"{query.message.text}\n\nApproved.")
        else:
            if pending.on_reject:
                pending.on_reject()
            await query.edit_message_text(f"{query.message.text}\n\nRejected.")

    async def start_polling(self) -> None:
        """Starts receiving button presses in the background of the current
        event loop, so a scanner loop can run alongside it (see notifier/main.py).
        """
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def stop(self) -> None:
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
