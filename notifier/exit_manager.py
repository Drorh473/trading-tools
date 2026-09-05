"""Everything that happens to a position's exits once it is open: moving the
stop to breakeven when the partial fills, finding and placing the runner's
target, and placing the partial itself.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities. Needs Bitget (every one of these is an order on the
exchange), the storage journal (for the ledger's own "did this ever work"
bookkeeping), the bot (to report what happened), a bar source (the runner's
target reads structure), and a way to ask whether a strategy tag's exits
are bot-managed at all.
"""

from __future__ import annotations

import asyncio
import logging
import time

from core import ledger
from core.storage import Storage
from notifier.bar_cache import BarCache
from notifier.strategies.base import Signal
from notifier.strategies.indicators import atr
from notifier.strategies.structure import nearest_level_beyond
from notifier.trailing_stops import RUNNER_LEVEL_ATR_PERIOD

logger = logging.getLogger(__name__)

PARTIAL_TAKE_FRACTION = 0.5
# Fixed 1:3 target for whatever's left after the partial take, regardless of
# the first tier's own ratio - replaces an open-ended "let it run" with a
# defined second exit, so nothing is left unmanaged indefinitely.
REMAINDER_TARGET_RATIO = 3.0

# The runner's take-profit, placed automatically once the partial fills.
#
# Dror's rule, after an SPCXUSDT signal whose runner read "at your discretion":
# "after the target i want the bot to make the new tp himself without my
# intervation ... the next tp under the nearest key line in the 1h graph".
#
# 1H rather than the daily the strategy's own remainder_target uses, because
# that one comes from DAILY pivot highs and is empty whenever the setup is at
# all-time highs - which is exactly when a runner most needs a target. The
# buffer sits the order in the queue ahead of the sellers defending the level:
# a limit resting exactly ON resistance frequently does not fill, since price
# stalls a tick short and turns.
# Read off the DAILY chart, though Dror reads the level on his 1H one. Both
# were tried against the SPCXUSDT trade he used to specify this. The nearest
# 1H pivot above the partial target was 136.21 - a minor swing nobody would
# aim a runner at - while the daily gives 147.24, the level he named on sight
# ("significant support with 3 touches on the 1h chart and in the daily it
# also have 2 touches"). The 1H chart is where the level is VISIBLE; the daily
# is what makes it significant enough to stop a move.
RUNNER_LEVEL_TIMEFRAME = "1D"
RUNNER_LEVEL_ATR_BUFFER = 0.25
# 2.0 on the daily. volume_run's own daily setup uses 3.0, but at 3.0 SPCXUSDT
# collapses to just three levels in 60 days (103.55 / 152.08 / 228.11) and the
# 147.24 Dror named disappears into a coarser 152.08. 2.0 resolves the swings
# a runner actually has to get through.
RUNNER_LEVEL_PIVOT_ATR_MULTIPLE = 2.0
# A level further than this from where price is now belongs to a different
# market regime, not to this trade. MUUUSDT traded 1000-1400 in June and 26.51
# today, so its nearest "level" overhead is a 803.5 low from before the
# collapse - 30x away, a target that can never fill and only looks like one.
# Measured across a live sweep the sensible levels sat at 0.004-2.47 ATR from
# price while that artifact sat at 23.7, so the cut goes in the wide gap
# between; it is a judgment call inside that gap rather than a tuned edge.
RUNNER_LEVEL_MAX_ATR = 6.0

# Bitget's position query and its order-matching book are eventually
# consistent: wait_for_signal_position already confirmed the fill, but a
# reduce-only order placed immediately after can still be rejected with
# 22002 "No position to close" for a few seconds - happened twice live on
# AAPLUSDT within one session, both times leaving a protected position with
# no target. A 400 rejection here is unambiguous (the order was never placed,
# unlike a timed-out entry leg that might have gone through), so retrying is
# safe rather than a risk of duplicating an exit - each attempt uses a fresh
# client_oid and only ever one attempt can succeed against the exchange.
PARTIAL_SETTLE_RETRY_DELAYS = (3.0, 6.0, 12.0)  # ~21s of settle time past the first attempt

_PRICE_EPSILON = 1e-9


def _tightens_stop(direction: str, current: float, candidate: float) -> bool:
    """Whether `candidate` is a strictly tighter stop than `current`.

    A long's stop sits below price, so raising it reduces risk; a short's sits
    above, so lowering it does. Strict, so re-running a breakeven against the
    stop it already placed is a no-op rather than an endless replace.
    """
    margin = _PRICE_EPSILON * max(abs(current), 1.0)
    if direction == "long":
        return candidate > current + margin
    return candidate < current - margin


class ExitManager:
    def __init__(self, bitget, storage: Storage, bot, bar_cache: BarCache, manages_exits):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        self.bar_cache = bar_cache
        # A callable (strategy_tag) -> bool, not the executor itself - this
        # class only ever needs the one question.
        self._manages_exits = manages_exits

    def manages_exits(self, strategy_tag: str) -> bool:
        """Whether the bot may place reduce-only exits on a tracked position.

        Strictly weaker than auto_executes: it cannot open or grow a
        position, only close one that already exists. Strategy 3's entries
        stay manual.
        """
        return self._manages_exits(strategy_tag)

    def runner_target(self, signal: Signal, fallback: float | None) -> tuple[float | None, str]:
        """Where the runner's take-profit goes, and what to call it.

        For a strategy managing its own two-tier exit (Strategy 3), it is
        the nearest confirmed daily swing level beyond price - high or low,
        since broken support is resistance on the way back - less a buffer.
        `fallback` is that strategy's own remainder_target, used only when
        the daily offers nothing. When neither does, the setup is at highs
        with nothing overhead and there is genuinely no price to sell into,
        so the runner keeps trailing rather than being capped at an
        invented number.

        Everything else (Strategy 1) keeps the ratio target it already
        computes and prints; this only starts PLACING it.
        """
        if signal.partial_fraction is None:
            return fallback, f"1:{REMAINDER_TARGET_RATIO:g}"

        # A strategy whose runner price is the thesis rather than a fallback.
        # Checked before the daily lookup so no level, however near, can
        # override it.
        #
        # "FINAL" INCLUDES FINAL AT None, and requiring remainder_target to be
        # set was the bug. Strategy 2.1 asks for NO runner target on purpose -
        # its remainder is handed to the trailing stop, and the trailing poll
        # only trails a position with no target - but None plus is_final=False
        # was indistinguishable from "no opinion, use the daily level". So the
        # fall-through below invented one.
        #
        # Live, 2026-08-19: UNIUSDT's partial filled, the stop went to breakeven
        # 3.395, and then a runner target of 3.48211 was placed "under the 1D
        # level at 3.536". That target silently opted the position out of the
        # trail for good - the exact outcome the Signal comment says it must not
        # have, on the trade it was written for. Dror, reading the alert: "it
        # should make the stop tighter not make a take profit."
        if signal.remainder_target_is_final:
            return signal.remainder_target, signal.remainder_note or "the strategy's own target"

        try:
            bars = self.bar_cache.get(signal.symbol, RUNNER_LEVEL_TIMEFRAME)
            # LIVE, not the daily bar's own close. The bar cache holds a 1D
            # series for the WHOLE DAY, so the forming candle's close is a
            # snapshot from whenever it was first fetched that day - hours
            # stale by the time a partial fill actually triggers this.
            #
            # DOGEUSDT and AEVOUSDT, both live: "Long position take profit
            # price please > mark price" (Bitget 40915). The target had been
            # validated as "beyond" a price hours old, on a symbol that had
            # since rallied past it - so a target that was genuinely above
            # price when computed was already behind it by the time it reached
            # Bitget, which checks against the price that actually exists now.
            #
            # Falls back to the stale close only if the live read itself fails
            # - a late target is still closer to right than none at all.
            try:
                price = float(self.bitget.get_mark_price(signal.symbol))
            except Exception:
                logger.exception("Could not read %s's live price; using the cached daily close", signal.symbol)
                price = float(bars["close"].iloc[-1])
            thresholds = atr(bars, RUNNER_LEVEL_ATR_PERIOD) * RUNNER_LEVEL_PIVOT_ATR_MULTIPLE
            level = nearest_level_beyond(bars, thresholds, price, signal.direction)
        except Exception:
            logger.exception("Could not read %s levels for %s's runner", RUNNER_LEVEL_TIMEFRAME, signal.symbol)
            return fallback, "the strategy's own resistance level"

        if level is None:
            return fallback, "the strategy's own resistance level"

        atr_now = float(atr(bars, RUNNER_LEVEL_ATR_PERIOD).iloc[-1])
        if atr_now <= 0 or abs(level - price) > RUNNER_LEVEL_MAX_ATR * atr_now:
            return fallback, "the strategy's own resistance level"  # a different regime's level

        target = level - atr_now * RUNNER_LEVEL_ATR_BUFFER if signal.direction == "long" \
            else level + atr_now * RUNNER_LEVEL_ATR_BUFFER
        # The buffer can be wider than the gap to the level when price is
        # already sitting right under it - ENAUSDT's level was 0.02% away
        # against a 1.5%-of-price buffer, which put the "take-profit" BELOW
        # the market. A reduce-only sell below market is not a target, it is
        # an instant exit at whatever is bid.
        beyond = target > price if signal.direction == "long" else target < price
        if not beyond:
            return fallback, "the strategy's own resistance level"

        return target, f"under the {RUNNER_LEVEL_TIMEFRAME} level at {level:g}"

    async def place_runner_target(
        self, signal: Signal, fallback: float | None, managed: bool | None = None, notify: bool = True
    ) -> str:
        """Place the runner's take-profit, once the partial has filled.

        Sized to whatever is actually left rather than to the plan, since
        the partial may have closed more or less than intended.

        `managed` is the caller's per-trade authorization, which is what
        the partial-fill path computes (a hand-added trade adopted with
        /manage qualifies without its tag ever matching a routing set).
        Left None it falls back to judging by tag alone, which is all a
        caller holding only a Signal can do.
        """
        if not (self.manages_exits(signal.strategy_tag) if managed is None else managed):
            return ""
        target, note = self.runner_target(signal, fallback)
        if target is None:
            # Nothing overhead, or the strategy asked for no target at all -
            # either way the runner trails, exactly as the alert said it would.
            return f"runner {note}" if note else "runner trails"

        try:
            position = self.bitget.get_position(signal.symbol, signal.direction)
        except Exception:
            logger.exception("Could not read the %s position to size its runner", signal.symbol)
            return "could not size the runner"
        if not position or position["size"] <= 0:
            return "already fully closed"

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0, *PARTIAL_SETTLE_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            if attempt > 0:
                # RECOMPUTED, not reused - keyed on this being a RETRY, not on
                # whether the delay was nonzero. A zero-length delay is still
                # a distinct attempt after a failure, and gating this on
                # `delay` instead skipped it in exactly that case.
                #
                # runner_target() now reads the LIVE mark price (see its own
                # comment - DOGEUSDT and AEVOUSDT, both live, "take profit
                # price please > mark price"), which closes the multi-hour
                # version of this. The retry delay is only seconds, but it is
                # exactly the gap the live price can move across, so a retry
                # that resubmits the SAME target is retrying the same failure.
                # fallback carries over unchanged; only the live-price-
                # dependent target needs a fresh read.
                target, note = self.runner_target(signal, fallback)
                if target is None:
                    return f"runner {note}" if note else "runner trails"
            try:
                self._place_reduce_only(signal.symbol, signal.direction, position["size"], target, "runner")
                if notify:
                    await self.bot.send_message(
                        f"Runner target set for {signal.symbol} {signal.direction} "
                        f"({signal.strategy_tag}): {target:g} — {note}."
                    )
                return f"runner target {target:g} ({note})"
            except Exception as exc:
                last_exc = exc
                logger.warning("Runner target for %s rejected on attempt %d: %s", signal.symbol, attempt + 1, exc)
                # 22002 is the position-settle race; 40915 is Bitget rejecting
                # a target that is no longer beyond the mark price - the same
                # "price moved since we computed this" shape, just caught by
                # the exchange instead of by us. Both are worth the same retry;
                # anything else is a real rejection waiting will not fix.
                if "22002" not in str(exc) and "40915" not in str(exc):
                    break

        logger.error("Could not place the runner target for %s: %s", signal.symbol, last_exc)
        await self.bot.send_message(
            f"The partial filled on {signal.symbol} {signal.direction} but the RUNNER TARGET FAILED: "
            f"{last_exc}\nThe position still has its stop — set the target by hand."
        )

    async def on_partial_manage_exits(
        self, signal: Signal, fallback: float | None, breakeven: float | None,
        managed: bool = True, notify: bool = True,
    ) -> list[str]:
        """Everything the alert told Dror to do by hand once the partial
        fills: move the stop to breakeven, then set the runner's target.

        Both are reduce-only or protective, so this needs exit management
        rather than full execution rights - Strategy 3's entries stay
        manual. Authorization is decided per TRADE by the caller and passed
        in, because judging by strategy tag alone can only ever say no to a
        hand-added trade, whose tag is free text from the /add prompt.

        The runner still gets its target when the breakeven fails: they are
        independent orders, and a failed stop move is already alerted on.

        Returns what each step actually did, so ONE message can report the
        whole event. With notify=True each step announces itself instead,
        which is what /manage still wants - it is acting on a trade out of
        band, not reporting a fill.
        """
        if not managed:
            return []
        steps: list[str] = []
        if breakeven is not None:
            steps.append(await self.move_stop_to_breakeven(signal, breakeven, notify=notify))
        runner = await self.place_runner_target(signal, fallback, managed=managed, notify=notify)
        if runner:
            steps.append(runner)
        return [s for s in steps if s]

    async def move_stop_to_breakeven(self, signal: Signal, breakeven: float, notify: bool = True) -> str:
        """Move the stop to breakeven, without ever widening it.

        The guard is what makes re-running this safe, and it has to be: a
        re-attached tracker re-detects a partial that already filled, which
        is exactly how a restart is meant to heal itself. Placing blindly
        would drag a stop Dror had since trailed forward BACK to entry,
        handing back risk on a winner, so the breakeven has to be an
        improvement on whatever is on the exchange right now or nothing
        happens.

        The new stop is placed BEFORE the old one is cancelled. Cancelling
        first would leave a 10-20x position unprotected for the width of an
        API round trip; this way the failure mode is two stops briefly on
        the book, where the tighter triggers first and closes the
        remainder anyway.

        It goes on as a POSITION-level pos_loss with size 0 - Bitget's "all
        closable" - rather than an order-level loss_plan sized to a
        quantity. Two reasons, one of them paid for:

        1. This path sent NO size until 2026-08-17 and Bitget rejected it
           with 40019 "Parameter size cannot be empty". BZUSDT #18 was the
           first partial ever to reach this handler in the service's life,
           and it failed - so the automated breakeven had a 100% failure
           rate that nothing had exercised. The bad claim was in
           place_tpsl_order's own docstring ("size omitted closes the whole
           position"), never tested.
        2. A quantity would be a SNAPSHOT. The staged confluence entry adds
           to a position after it opens, and a stop sized to the position
           as it was leaves everything added afterwards unprotected. "All
           closable" keeps covering whatever the position currently is,
           which is the only thing a stop should ever mean.

        This is also exactly what Dror sets by hand from Bitget's Position
        TP/SL panel - his BZUSDT stop after this failure was a pos_loss at
        85.27, size 0.

        ROUNDED ONCE, UP FRONT. place_tpsl_order sends the exchange
        round_price(breakeven) - APTUSDT #104 (2026-09-05) is the live trade
        that exposed this: average entry 0.579413867186, round_price gives
        0.5794, a difference of 1.4e-5. Every use of `breakeven` below used
        to be the UNROUNDED value, so _cancel_superseded_stops compared it
        against get_plan_orders' ROUNDED trigger for the order this method
        had just placed - _tightens_stop('long', 0.5794, 0.579413867186)
        reads True, since 1.4e-5 dwarfs its 1e-9 margin - and cancelled the
        breakeven's own order in the same sweep meant to clear the OLD one.
        The position sat with no stop at all from that instant. Rounding
        once here means every comparison downstream, and the order actually
        placed, agree on the same number.
        """
        placed = self.bitget.round_price(signal.symbol, breakeven)

        try:
            current_stop, _ = self.bitget.get_stop_target(signal.symbol, signal.direction)
        except Exception:
            # Better a redundant stop than none: the tighter one wins.
            logger.exception("Could not read %s's live stop; placing the breakeven anyway", signal.symbol)
            current_stop = None

        if current_stop is not None and not _tightens_stop(signal.direction, current_stop, placed):
            logger.info(
                "Breakeven for %s skipped: the live stop %g is already at or beyond %g",
                signal.symbol,
                current_stop,
                placed,
            )
            return f"stop already at {current_stop:g}"

        try:
            result = self.bitget.place_tpsl_order(
                symbol=signal.symbol,
                direction=signal.direction,
                plan_type="pos_loss",
                trigger_price=placed,
                size=0,  # all closable - see the docstring
                client_oid=f"be-{signal.symbol}-{int(time.time() * 1000)}",
            )
        except Exception:
            logger.exception("Could not move %s's stop to breakeven", signal.symbol)
            failed = f"STOP TO BREAKEVEN ({placed:g}) FAILED — move it by hand"
            if notify:
                await self.bot.send_message(
                    f"The partial filled on {signal.symbol} {signal.direction} but moving the stop to "
                    f"breakeven ({placed:g}) FAILED — move it by hand."
                )
            return failed

        # Excluded from its own supersede sweep by ID, not just by price - see
        # _cancel_superseded_stops's own docstring for why the price check
        # alone was not enough. A fake client in tests may return {} rather
        # than a real orderId; None just means this second check is a no-op,
        # which is the current behaviour for every existing caller.
        new_order_id = result.get("orderId") if isinstance(result, dict) else None
        ledger.try_record(self.storage.db_path, ledger.BREAKEVEN_STOP_MOVED)
        self._cancel_superseded_stops(signal.symbol, signal.direction, placed,
                                      exclude_order_id=new_order_id)
        if notify:
            await self.bot.send_message(
                f"Stop moved to breakeven ({placed:g}) on {signal.symbol} {signal.direction} "
                f"({signal.strategy_tag}) — the remainder is running risk-free."
            )
        return f"stop {placed:g} breakeven"

    def _cancel_superseded_stops(
        self, symbol: str, direction: str, breakeven: float, exclude_order_id: str | None = None,
    ) -> None:
        """Drop the original stop now that a tighter one is confirmed placed.

        Without this a position carries two loss_plans - the preset one
        created from the entry order's presetStopLossPrice, and the
        breakeven - and get_stop_target() reports whichever the API
        happens to list last, so the stop recorded against the trade
        becomes a coin flip. Only stops the breakeven supersedes are
        touched, which leaves the breakeven itself and anything already
        tighter alone.

        exclude_order_id is a second, independent check on top of the price
        comparison - never remove the order the caller JUST placed, whatever
        its price looks like. The rounding fix in move_stop_to_breakeven
        (comparing what round_price actually produces, not the raw value)
        should already prevent a just-placed order from reading as
        "superseded" by itself; this is what stops the same failure from
        reappearing if a future change reintroduces a price mismatch by some
        other path. APTUSDT #104 (2026-09-05) is the live trade that showed
        the price-only version of this check is not enough on its own: the
        sweep cancelled its own breakeven order 0 seconds after placing it.
        """
        try:
            for order in self.bitget.get_plan_orders(symbol, direction):
                if not order["is_stop"] or not order["order_id"]:
                    continue
                if exclude_order_id is not None and order["order_id"] == exclude_order_id:
                    continue
                trigger = order["trigger_price"]
                if trigger is None or not _tightens_stop(direction, trigger, breakeven):
                    continue
                self.bitget.cancel_plan_order(symbol, order["plan_type"], order_id=order["order_id"])
        except Exception:
            # The position is over-protected, not under-protected: safe to log.
            logger.exception("Could not cancel the superseded stop on %s", symbol)

    def _place_reduce_only(self, symbol: str, direction: str, size: float, price: float, kind: str) -> None:
        """One exit order, as a TP PLAN order for every symbol.

        THE REDUCE-ONLY LIMIT PATH HAS NEVER ONCE WORKED. Reading the whole
        service log from the day execution shipped: PEPEUSDT 2026-08-03,
        AAPLUSDT 08-04, GOOGLUSDT 08-06, ZECUSDT 08-08, WLDUSDT 08-11 -
        every attempt rejected, most with 22002 "No position to close" on
        positions that demonstrably existed (WLDUSDT's was 155 units,
        complete 709ms before the first try, and it was refused four times
        over 22 seconds). Not one automated take-profit has ever reached
        the exchange; Dror has been setting every target by hand without
        either of us realising the bot had never managed it.

        A 100% failure rate across every symbol, size and timing is not a
        race or an arithmetic edge - it is the request being wrong. The
        account is in hedge mode, where place-order needs `side` AND
        `tradeSide` to say both which position and whether to open or
        close, and the pairing this used for a close is evidently read as
        the opposite side - hence "no position to close" when there is
        plainly a position. Opens have always worked; only closes fail.

        place-tpsl-order sidesteps the ambiguity entirely: it names the
        position with `holdSide` and nothing has to be inferred from a
        side/tradeSide pair. It is also independently proven on this
        account - it is what Bitget's own "Position TP/SL" panel places,
        which is how Dror's hand-set targets have been going on, and it was
        already the RWA path for a different reason (a resting limit is
        capped to ~2% from mark on tokenized stocks; a trigger is not
        bound by that band).
        """
        self.bitget.place_tpsl_order(
            symbol=symbol,
            direction=direction,
            plan_type="profit_plan",
            trigger_price=price,
            size=size,
            client_oid=f"{kind}-{symbol}-{int(time.time() * 1000)}",
        )

    async def place_partial(self, signal: Signal, plan, position_size: float, replace: bool = False) -> None:
        """The first exit tier, at the plan's target.

        A plain reduce-only limit is capped at the exchange's own ~2% price
        band from mark on RWA (tokenized-stock) symbols - GOOGLUSDT's
        target was a perfectly ordinary ~3.7% from entry and Bitget
        rejected it outright. A TP plan order's trigger price is a
        condition rather than an order resting in the book right now, so
        it isn't bound by that band; RWA exits go through place_tpsl_order
        instead of place_order.

        Retries through PARTIAL_SETTLE_RETRY_DELAYS on Bitget's 22002 "No
        position to close" - see the constant's comment for why that's
        safe. Any other rejection is not retried: it isn't the settle
        race, so waiting longer won't fix it, and this path exists
        specifically to widen that one window rather than to retry blindly.
        """
        fraction = signal.partial_fraction if signal.partial_fraction is not None else PARTIAL_TAKE_FRACTION
        size = position_size * fraction
        if size <= 0:
            return

        specs = self.bitget.get_contract_specs(signal.symbol)

        # A partial smaller than the exchange's own minimum can NEVER be
        # placed, so attempting it is not a race to wait out - it is
        # arithmetic. ZECUSDT reported this as 22002 "No position to close",
        # the same code as the genuine settle race, which burned the whole
        # ~21s retry budget and sent the diagnosis after a transient fault
        # that did not exist. The real numbers: 0.006 x 498.41 = $2.99
        # against a $5 floor.
        #
        # This is structural on a small account rather than an edge case.
        # The first partial is market_fraction x partial_fraction = 10% of the
        # planned position, so the position must be $50+ for it to clear $5 -
        # which at 1% risk needs a stop tighter than 2% of price, and
        # Strategy 1's Fib stops run 3-6%. Dror's call was to accept the
        # limit and grow the account rather than change the exit model, so
        # nothing is placed instead: the stop still protects the position, and
        # if the resting limit leg fills, on_resize retries at a size that
        # clears the floor.
        notional = size * plan.take_profit
        min_notional = specs.get("min_notional") or 0.0
        if notional < min_notional:
            logger.info(
                "Partial take-profit for %s skipped: %s x %s = $%.2f, under the $%.2f minimum",
                signal.symbol, size, plan.take_profit, notional, min_notional,
            )
            await self.bot.send_message(
                f"{signal.symbol} {signal.direction} has its stop, but NO partial take-profit: "
                f"{size:g} at {plan.take_profit:g} is ${notional:.2f}, under Bitget's ${min_notional:.2f} "
                f"minimum for this symbol.\n"
                f"Nothing was attempted — an order this small cannot be placed. The position is protected. "
                f"If the resting limit leg fills, a target is placed automatically at the larger size; "
                f"otherwise set one by hand."
            )
            return

        if replace:
            # The position grew, so the old order covers too little of it.
            self.cancel_resting(signal.symbol, reduce_only_only=True, direction=signal.direction)

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0, *PARTIAL_SETTLE_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            try:
                # A TP plan order for EVERY symbol now, not just RWA. See
                # _place_reduce_only: the reduce-only limit path has never
                # placed a single successful take-profit since execution
                # shipped, on any symbol, and this is the path that works.
                self.bitget.place_tpsl_order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    plan_type="profit_plan",
                    trigger_price=plan.take_profit,
                    size=size,
                    client_oid=f"tp-{signal.symbol}-{int(time.time() * 1000)}",
                )
                # This is the capability that failed on every attempt from
                # 2026-08-03 to 2026-08-13 - five symbols, 100% rejection - and
                # the only reason it was ever noticed is that Dror happened to
                # read the whole log at once. Recorded so the weekly report can
                # say "never worked" out loud instead of looking quiet.
                ledger.try_record(self.storage.db_path, ledger.TAKE_PROFIT_PLACED)
                ledger.try_record(self.storage.db_path, ledger.order_placed(signal.strategy_tag))
                return
            except Exception as exc:
                last_exc = exc
                is_settle_race = "22002" in str(exc)
                logger.warning(
                    "Partial take-profit for %s rejected on attempt %d: %s", signal.symbol, attempt + 1, exc
                )
                if not is_settle_race:
                    break  # a different failure - waiting won't fix it, don't burn the retry budget

        logger.error("Could not place the partial take-profit for %s: %s", signal.symbol, last_exc)
        await self.bot.send_message(
            f"Placed {signal.symbol} {signal.direction} and its stop, but the partial take-profit "
            f"FAILED: {last_exc}\nThe position is protected but has no target — set one by hand."
        )

    def cancel_resting(self, symbol: str, reduce_only_only: bool = False, direction: str | None = None) -> None:
        try:
            for open_order in self.bitget.get_open_orders(symbol):
                if reduce_only_only and (open_order.get("tradeSide") or "").lower() != "close":
                    continue
                self.bitget.cancel_order(symbol, order_id=open_order.get("orderId"))
        except Exception:
            logger.exception("Could not cancel resting orders for %s", symbol)

        if not (reduce_only_only and direction):
            return
        # An RWA take-profit lives as a plan order (place_tpsl_order), not on
        # the regular pending-orders book the loop above just cleared - a
        # growing position that skipped this would keep the OLD, now
        # under-sized target instead of getting a replacement sized to the
        # new total, alongside whatever place_partial adds next.
        try:
            if self.bitget.get_contract_specs(symbol).get("is_rwa"):
                for plan_order in self.bitget.get_plan_orders(symbol, direction):
                    if plan_order["is_target"] and plan_order["order_id"]:
                        self.bitget.cancel_plan_order(
                            symbol, plan_order["plan_type"], order_id=plan_order["order_id"]
                        )
        except Exception:
            logger.exception("Could not cancel the resting take-profit plan order for %s", symbol)

    def safe_stop_target(self, symbol: str, direction: str, position: dict):
        try:
            return self.bitget.get_stop_target(symbol, direction)
        except Exception:
            logger.exception("Could not read stop/target for %s; using position presets", symbol)
            return position["stop_loss"], position["take_profit"]
