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
import logging
from dataclasses import dataclass
from typing import Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

logger = logging.getLogger(__name__)

# A signal is a snapshot of one candle's levels; acting on it much later means
# entering a trade the market has already moved past - which is how a stale
# TSLAUSDT short got taken four times over eleven hours. The caller supplies
# a per-signal ceiling (see notifier.scanner.signal_expiry_seconds), but a
# ceiling alone still let a quiet-timeframe offer sit for as long as price
# didn't move - so we also poll toward a movement cutoff and take whichever
# fires first.
SIGNAL_MOVEMENT_FRACTION = 0.15  # fraction of 1R the market may drift AWAY from the entry before the offer dies
SIGNAL_POLL_SECONDS = 15.0


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}min"


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
        *,
        expiry_seconds: float,
        entry_price: float,
        stop_loss: float,
        reference_price: float,
        price_fetcher: Callable[[], float],
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
        message = await self.app.bot.send_message(chat_id=self.chat_id, text=text, reply_markup=keyboard)
        asyncio.create_task(
            self._expire(
                callback_id, message, expiry_seconds, entry_price, stop_loss, reference_price, price_fetcher
            )
        )

    async def _expire(
        self,
        callback_id: str,
        message,
        expiry_seconds: float,
        entry_price: float,
        stop_loss: float,
        reference_price: float,
        price_fetcher: Callable[[], float],
    ) -> None:
        """Drop the offer once its levels are too old to act on.

        Two independent cutoffs, whichever fires first: the timer
        (`expiry_seconds`, the caller's per-signal ceiling), and the market
        drifting `SIGNAL_MOVEMENT_FRACTION` of 1R further AWAY from the entry
        than it was when the signal fired.

        Three prices, and they are genuinely three different things - collapsing
        any two of them has already produced a live misfire:

        `entry_price` is where the order actually rests, and with `stop_loss` it
        defines 1R. On INJUSDT (Strategy 2, 100% limit at EMA9) the plan entry
        was 4.636 against a 4.672 stop - a real 1R of 0.036 - while the market
        sat at 4.557. Measuring R from the market instead gave 0.115, over three
        times too large, so a move the alert reported as "0.17R" was really 0.54R
        of the trade's own risk.

        `reference_price` is where the market stood at dispatch, and only
        movement that increases the distance from `entry_price` beyond that
        starting gap counts. Dror's rule: for an order resting away from market,
        price travelling TOWARD it is the setup working, not decaying - a short
        whose limit sits above market needs price to rise to fill at all, and
        expiring on that would kill exactly the signals that were about to work.
        Anchoring on the raw distance instead would expire INJUSDT instantly,
        since it dispatched 2.2R away from its own resting limit by construction.

        Popping from `_pending` is what actually disables it - a button press
        after this finds nothing registered and is answered as already handled,
        so even a stale keyboard cached in the client cannot execute anything.
        """
        risk = abs(entry_price - stop_loss)
        gap_at_dispatch = abs(reference_price - entry_price)
        reason = f"not acted on within {_format_duration(expiry_seconds)}"
        elapsed = 0.0
        while elapsed < expiry_seconds:
            step = min(SIGNAL_POLL_SECONDS, expiry_seconds - elapsed)
            await asyncio.sleep(step)
            elapsed += step
            if callback_id not in self._pending:
                return  # already approved or rejected

            if risk <= 0:
                continue  # nothing to normalize movement against; let the timer decide
            try:
                price = price_fetcher()
            except Exception:
                logger.warning("Could not poll price for signal %s; retrying next tick", callback_id)
                continue
            drifted = abs(price - entry_price) - gap_at_dispatch
            if drifted <= 0:
                continue  # closer to the entry than when it fired - the setup is coming to us
            moved = drifted / risk
            if moved >= SIGNAL_MOVEMENT_FRACTION:
                reason = f"price drifted {moved:.2f}R further from the entry"
                break

        if self._pending.pop(callback_id, None) is None:
            return  # already approved or rejected

        try:
            # Editing the text without a reply_markup also clears the buttons.
            await message.edit_text(f"{message.text}\n\nExpired — {reason}.")
        except Exception:
            logger.exception("Could not expire signal message %s", callback_id)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query

        # `answer()` clears the client's loading spinner, but Telegram rejects it
        # with "Query is too old" once its own callback timeout has passed - and
        # it used to run before the offer was looked up, so that BadRequest
        # aborted the whole handler. A tap that was still perfectly valid on our
        # side was then dropped with no execution and no reply: the signal simply
        # went dead in Dror's hand. The spinner is cosmetic and the decision is
        # not, so a failure here must never stop the decision being processed.
        try:
            await query.answer()
        except Exception:
            logger.warning("Could not answer callback query %s; processing the decision anyway", query.data)

        action, _, callback_id = query.data.partition(":")
        pending = self._pending.pop(callback_id, None)
        if pending is None:
            await self._edit(query, "(already handled, or expired while the notifier was restarting)")
            return

        if action == "approve":
            pending.on_approve()
            await self._edit(query, "Approved.")
        else:
            if pending.on_reject:
                pending.on_reject()
            await self._edit(query, "Rejected.")

    async def _edit(self, query, note: str) -> None:
        """Append an outcome note to the signal message.

        Editing is the only confirmation Dror gets that a tap landed, but it is
        the LAST thing that happens - the trade is already placed by now. An edit
        that fails (message too old to edit, network blip) must not raise into
        the handler, or python-telegram-bot logs an unhandled exception for a
        signal that in fact executed correctly.
        """
        try:
            await query.edit_message_text(f"{query.message.text}\n\n{note}")
        except Exception:
            logger.exception("Could not edit signal message after '%s'", note)

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
