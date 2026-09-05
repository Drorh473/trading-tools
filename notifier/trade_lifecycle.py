"""What happens when a tracked trade's state changes: it closes, a resting
leg fills, a partial exit fires - plus /manage's own path for taking over
exits on a trade the router doesn't otherwise recognise.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities. Needs the storage journal and Bitget (every callback
reads the trade row and/or the live position), the bot (to report what
happened), and the ExitManager (a partial exit is what promotes a
scanner-default trade's runner from "at your discretion" to a real order).
"""

from __future__ import annotations

import asyncio
import logging

from core.storage import Storage
from execution.messages import format_close_message, format_partial_message, format_scale_in_message
from execution.tracker import breakeven_price, closing_exits, take_profit_coverage
from notifier.exit_manager import ExitManager
from notifier.strategies.base import Signal

logger = logging.getLogger(__name__)

# How far a hand-typed /manage breakeven may sit from the trade's recorded
# entry before it is refused as a typo. A breakeven IS the entry, so anything
# this far away is a slipped decimal rather than a judgement call - and on
# 10x that is 250% of the margin.
ADOPT_MAX_ENTRY_DISTANCE = 0.25


class TradeLifecycleHandler:
    def __init__(self, bitget, storage: Storage, bot, exits: ExitManager):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        self.exits = exits

    def manages_trade(self, trade) -> bool:
        """Whether the bot may place exits on THIS trade, as opposed to on
        this strategy.

        Two ways to qualify. A scanner-approved trade qualifies by its tag,
        which the router knows. A hand-added one never can: /add asks Dror to
        type the tag and he types "strategy 1", which will not match the
        instance tag "Strategy 1 1H" that LIVE_TAGS carries - so /add trades
        were silently unmanageable, with no log line saying so. Adopting one
        with /manage sets the permission on the row instead. Rewriting the tag
        to force a match is not an option: it is what the weekly review groups
        by, so editing it to suit the router corrupts strategy scoring.
        """
        return bool(trade.exit_managed) or self.exits.manages_exits(trade.תגית_אסטרטגיה or "")

    def exit_plan_signal(self, trade) -> Signal | None:
        """Rebuild the parts of the original Signal the exit handlers read.

        Only four fields are ever touched downstream - symbol, direction,
        strategy_tag and partial_fraction (which decides whether the runner
        aims at a daily level or at the recorded ratio target) - so the row
        carries everything needed. None means the bot doesn't own this trade's
        exits: either it never did, or the trade predates the exit plan being
        recorded at all, and in both cases the notification says to move the
        stop by hand instead.
        """
        if trade.breakeven_stop is None:
            return None
        return Signal(
            symbol=trade.סימבול,
            direction=trade.כיוון,
            entry_price=breakeven_price(trade),
            stop_loss=trade.סטופ_לוס_מקורי or trade.סטופ_לוס_בפועל or trade.breakeven_stop,
            strategy_tag=trade.תגית_אסטרטגיה or "",
            partial_fraction=trade.partial_fraction,
            remainder_target=trade.runner_target,
            # The same shape of bug as remainder_target_is_final below: set
            # on the live Signal at confirm time and used there to price the
            # partial and runner targets, but gone by the time a limit leg
            # fills later. None (a pre-migration trade, or a /manage-adopted
            # one Dror priced by hand) is what tells a later recompute to
            # leave the stale price alone rather than guess with an
            # unrelated default.
            reward_risk_ratio=trade.reward_risk_ratio,
            # WITHOUT THIS THE REBUILD LOSES THE DECISION. runner_target being
            # NULL says nothing on its own about whether that was deliberate,
            # and runner_target() reads None-plus-not-final as "no opinion, use
            # the daily level" - so it invents one, and a target turns the
            # trail off permanently.
            #
            # DOGEUSDT #29, live 2026-08-20: a Strategy 2.1 1H runner acquired
            # a 0.08586 target off the daily. Dror: "it shouldnt have tp". The
            # UNIUSDT fix of 2026-08-19 set this flag on the SIGNAL and was
            # tested there - but the Signal object is gone by the time a
            # partial fills, and this is the rebuild every partial goes through.
            remainder_target_is_final=bool(trade.runner_target_is_final),
        )

    def on_trade_closed(self, trade_id: int, price: float) -> None:
        trade = self.storage.get_trade(trade_id)
        asyncio.create_task(
            self.bot.send_message(format_close_message(trade, closing_exits(self.bitget, trade)))
        )
        # Whatever is left resting - bot-placed or placed by hand off the
        # alert - belongs to a trade that is now over. This runs from here
        # rather than after track_position's await so it also fires when a
        # trade is re-attached by resume_open_trades after a restart, where
        # there is no "after the await" to fall back on.
        self.exits.cancel_resting(trade.סימבול)

    def on_scale_in(self, trade_id: int) -> None:
        """A resting entry leg filled, so the real position has arrived.

        Purely informational - no thresholds and no flagging of the risk
        against what was planned. The recomputed risk is in the message; if
        it looks wrong, that is a sizing bug to fix at the source rather than
        something for an alert to police.
        """
        trade = self.storage.get_trade(trade_id)
        try:
            covered = take_profit_coverage(
                self.bitget, trade.סימבול, trade.כיוון, trade.גודל_פוזיציה or 0.0
            )
        except Exception:
            # Worth sending without the coverage line rather than not at all:
            # the position figures are the point, coverage is the extra.
            logger.exception("Could not read take-profit coverage for %s", trade.סימבול)
            covered = None
        asyncio.create_task(self.bot.send_message(format_scale_in_message(trade, covered)))

    def on_partial_exit(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        """The scale-out fired: report it, then honour the recorded exit plan.

        The plan comes from the trade row rather than from a closure, which is
        what lets this same callback serve a tracker re-attached after a
        restart. That path is also how the reconcile works: track_position's
        first poll compares the live position against the recorded size, so a
        partial that filled while the service was down is detected on
        re-attach and the breakeven placed immediately, rather than being
        lost with the process that was supposed to place it.
        """
        trade = self.storage.get_trade(trade_id)
        signal = self.exit_plan_signal(trade)

        if signal is None or not self.manages_trade(trade):
            # Nothing to do beyond reporting it: the message says so and tells
            # Dror to move the stop himself.
            asyncio.create_task(self.bot.send_message(format_partial_message(trade, closed_size, realized_pnl)))
            return

        async def report() -> None:
            """ONE message for one event.

            This used to be three - an announcement that said "each is confirmed
            separately", then a breakeven confirmation, then a runner
            confirmation - for a single scale-out. Dror, on the UNIUSDT partial
            of 2026-08-19: "i dont want to get it in 3 different messages".

            The steps still run independently and a failure of either is still
            named; what changed is that the report waits for both and says what
            happened, rather than announcing what is about to.
            """
            steps = await self.exits.on_partial_manage_exits(
                signal, trade.runner_target, breakeven_price(trade), managed=True, notify=False
            )
            await self.bot.send_message(
                format_partial_message(self.storage.get_trade(trade_id), closed_size, realized_pnl, steps)
            )

        # breakeven_price(), not the stored column: for a scanner trade the
        # stored value is the PLANNED blend and the position's real average
        # entry has been resynced since. It is also exactly what the message
        # prints, which is the point - the two cannot drift apart again.
        #
        # Scheduled rather than awaited: this fires synchronously from inside
        # track_position's own poll loop, and the retries would stall that
        # loop for as long as they take.
        asyncio.create_task(report())

    async def adopt_trade(self, trade_id: int, breakeven: float, runner_target: float | None = None) -> str:
        """Take over exit management of one open trade (/manage). Returns the
        reply to send, and never raises for bad input.

        Leaving partial_fraction NULL means runner_target() falls straight
        through to the fallback, so `/manage 11 0.6081` arms the stop move
        alone and adding a price arms a target too - the runner is never given
        an invented level Dror did not ask for.

        A trade whose partial has ALREADY filled is acted on immediately.
        Without that this would do nothing until the next restart: the poll
        loop compares against the size it last saw, so a scale-out that has
        already been recorded is not re-detected in a running process, and
        the trade that motivated this command (APTUSDT #11) was in exactly
        that state.
        """
        try:
            trade = self.storage.get_trade(trade_id)
        except ValueError:
            return f"No trade #{trade_id} in the journal."
        if trade.is_cancelled:
            return f"Trade #{trade_id} was cancelled — nothing to manage."
        if trade.is_pending:
            return f"Trade #{trade_id} hasn't confirmed an entry yet."
        if not trade.is_open:
            return f"Trade #{trade_id} is already closed."

        # A stop on the wrong side of the market is not a stop, it is an
        # instant market exit of the runner - and this price is typed by hand,
        # so it is the one place that mistake can enter.
        try:
            mark = self.bitget.get_mark_price(trade.סימבול)
        except Exception:
            logger.exception("Could not read the mark price for %s while adopting #%s", trade.סימבול, trade_id)
            return f"Couldn't reach Bitget to sanity-check that price against {trade.סימבול}. Nothing changed."

        wrong_side = breakeven >= mark if trade.כיוון == "long" else breakeven <= mark
        if wrong_side:
            side = "below" if trade.כיוון == "long" else "above"
            return (
                f"{breakeven:g} is on the wrong side of {trade.סימבול} at {mark:g} — a {trade.כיוון}'s stop has "
                f"to sit {side} the market or it closes the position the moment it is placed. Nothing changed."
            )

        entry = trade.מחיר_כניסה
        if entry and abs(breakeven - entry) > entry * ADOPT_MAX_ENTRY_DISTANCE:
            return (
                f"{breakeven:g} is {abs(breakeven - entry) / entry:.0%} away from #{trade_id}'s entry ({entry:g}) — "
                f"that reads like a typo rather than a breakeven. Nothing changed."
            )

        # Plan first, permission second: if the second write fails the trade
        # is unmanaged rather than managed with nothing to act on.
        self.storage.set_exit_plan(
            trade_id, breakeven_stop=breakeven, runner_target=runner_target, partial_fraction=None
        )
        self.storage.set_exit_managed(trade_id, True)

        lines = [
            f"Managing exits on #{trade_id} ({trade.סימבול} {trade.כיוון}, tagged '{trade.תגית_אסטרטגיה}').",
            f"Stop goes to {breakeven:g} when the partial fills"
            + (f", runner target {runner_target:g}." if runner_target is not None else ", no runner target."),
        ]

        if (trade.גודל_שנסגר or 0) > 0:
            lines.append("Its partial has already filled — doing both now.")
            adopted = self.storage.get_trade(trade_id)
            signal = self.exit_plan_signal(adopted)
            if signal is not None:
                await self.exits.on_partial_manage_exits(signal, runner_target, breakeven, managed=True)
        return "\n".join(lines)
