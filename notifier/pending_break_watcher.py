"""Watches a chart pattern that was UNBROKEN when a trade was approved, and
offers a second risk increment the moment it actually resolves.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities. Risk is staged rather than assumed: the trade goes on at
base risk when the signal fires, and the pattern only earns its second
increment once it actually breaks - a wedge that had already broken 17
hours before dispatch used to size a trade at 2% on the strength of
evidence that was, by dispatch time, already stale.
"""

from __future__ import annotations

import asyncio
import logging

from core.storage import Storage
from execution.executor import Executor, OrderLeg, TradeOrder
from execution.tracker import check_position_now
from notifier.bar_cache import BarCache
from notifier.risk_sizing import plan_position, round_trip_fee_for
from notifier.scanner_time import signal_expiry_seconds
from notifier.strategies import patterns

logger = logging.getLogger(__name__)


class PendingBreakWatcher:
    def __init__(
        self,
        bitget,
        storage: Storage,
        bot,
        executor: Executor,
        bar_cache: BarCache,
        max_total_risk_pct: float,
        reward_risk_ratio: float,
        auto_executes,
        symbol_max_leverage,
    ):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        self.executor = executor
        self.bar_cache = bar_cache
        self.max_total_risk_pct = max_total_risk_pct
        self.reward_risk_ratio = reward_risk_ratio
        # Callables rather than the owning objects directly - this class only
        # ever needs these two questions answered, and taking the callables
        # keeps it from depending on Scanner's or Executor's full interface.
        self._auto_executes = auto_executes
        self._symbol_max_leverage = symbol_max_leverage
        # symbol -> the unbroken pattern a held position is waiting on, so the
        # second risk increment can be offered when it breaks. Cannot be
        # rebuilt from current bars - it records what was true when the trade
        # was approved - so it is held rather than recomputed, and is
        # deliberately lost on restart: re-offering an add-on for a break
        # that happened while the process was down would be acting on stale
        # news.
        self.awaiting_break: dict[str, dict] = {}

    def register(self, symbol: str, watch: dict) -> None:
        """Start watching a pattern for its trade. Called on approval, so a
        signal that was never taken cannot produce an add-on for a position
        that does not exist."""
        self.awaiting_break[symbol] = watch

    async def poll(self) -> None:
        """Offer the second risk increment once a held position's pattern
        actually breaks.

        Ends a watch when the position closes or the pattern dies - there is
        deliberately no clock, since both of those bound it naturally and an
        arbitrary bar count would be a constant nobody measured.
        """
        if not self.awaiting_break:
            return
        try:
            equity = self.bitget.get_account_equity()
        except Exception:
            logger.exception("Could not read equity; skipping the pending-break poll")
            return

        for symbol, watch in list(self.awaiting_break.items()):
            try:
                trade = self.storage.get_trade(watch["trade_id"])
            except Exception:
                self.awaiting_break.pop(symbol, None)
                continue
            if not trade.is_open:
                self.awaiting_break.pop(symbol, None)  # nothing left to add to
                continue

            try:
                bars = self.bar_cache.get(symbol, watch["timeframe"]).iloc[:-1]
            except Exception:
                logger.exception("Could not fetch bars for the pending-break watch on %s", symbol)
                continue
            if len(bars) < 2:
                continue

            direction = watch["direction"]
            # Re-derive the level from fresh bars BEFORE testing it: a
            # triangle or wedge sits on a converging line at a different price
            # every bar, so the level stored at approval time is already out
            # of date by the time this runs.
            try:
                refreshed = patterns.pending({watch["timeframe"]: bars}, direction)
            except Exception:
                logger.exception("Pending-pattern refresh failed for %s", symbol)
                refreshed = None
            if refreshed is not None and refreshed[0].name == watch["name"]:
                watch["break_level"] = refreshed[0].break_level
                watch["invalidation_level"] = refreshed[0].invalidation_level

            close = float(bars["close"].iloc[-1])
            broke = close > watch["break_level"] if direction == "long" else close < watch["break_level"]
            died = close < watch["invalidation_level"] if direction == "long" else close > watch["invalidation_level"]

            if broke:
                self.awaiting_break.pop(symbol, None)
                await self._offer_add_on(symbol, watch, trade, equity, close)
            elif died:
                self.awaiting_break.pop(symbol, None)
                # No exit action: the stop already defines where this trade
                # ends, and overriding a defined stop with a discretionary
                # exit is a different decision than the one being made here.
                await self.bot.send_message(
                    f"{symbol}: the {watch['name']} on {watch['timeframe']} broke the WRONG way "
                    f"(closed {close:g} through {watch['invalidation_level']:g}). No add-on. "
                    f"Your stop still governs the position — nothing was changed."
                )
            elif refreshed is None:
                self.awaiting_break.pop(symbol, None)
                logger.info("Pending %s on %s no longer reads as itself; watch dropped", watch["name"], symbol)

    async def _offer_add_on(self, symbol: str, watch: dict, trade, equity: float, break_close: float) -> None:
        direction = watch["direction"]
        existing_stop = trade.סטופ_לוס_בפועל or trade.סטופ_לוס_מקורי
        flag_stop = watch["invalidation_level"]
        # The break is a reason to risk more behind a tighter stop, never to
        # give the original leg more room - so take whichever is tighter.
        if existing_stop is None:
            new_stop = flag_stop
        else:
            new_stop = max(existing_stop, flag_stop) if direction == "long" else min(existing_stop, flag_stop)

        try:
            market_price = self.bitget.get_mark_price(symbol)
        except Exception:
            logger.exception("Could not read mark price for the %s add-on", symbol)
            market_price = break_close

        try:
            plan = plan_position(
                equity=equity,
                risk_pct=watch["risk_pct"],
                entry_price=market_price,
                stop_loss=new_stop,
                direction=direction,
                reward_risk_ratio=self.reward_risk_ratio,
                available_budget=equity - self.storage.committed_margin(),
                max_leverage=self._symbol_max_leverage(symbol),
                # The add-on order below always places at market (see the
                # "market" order_type a few lines down) - never a resting
                # limit - so its true fee is taker both legs.
                round_trip_fee_pct=round_trip_fee_for(1.0),
            )
        except ValueError as exc:
            logger.info("No add-on for %s: %s", symbol, exc)
            return

        # The same aggregate ceiling every other trade obeys - a pattern
        # breaking is not a licence to exceed it.
        risk_cap = equity * self.max_total_risk_pct
        if self.storage.total_open_risk() + plan.risk_amount > risk_cap:
            logger.info("No add-on for %s: it would exceed the %.0f%% cap", symbol, self.max_total_risk_pct * 100)
            return

        specs = self.bitget.get_contract_specs(symbol)
        if self.bitget.round_size(symbol, plan.position_size) <= 0 or plan.notional_value < specs["min_notional"]:
            logger.info("No add-on for %s: below the exchange minimum", symbol)
            return

        def px(v: float) -> str:
            return f"{v:.{specs['price_place']}f}"

        def qty(v: float) -> str:
            return f"{v:.{specs['volume_place']}f}"

        text = "\n".join(
            [
                f"ADD-ON: {symbol} {direction.upper()} ({watch['strategy_tag']})",
                f"The {watch['name']} on {watch['timeframe']} broke — closed {px(break_close)} "
                f"through {px(watch['break_level'])}.",
                f"Add: ${plan.notional_value:,.0f} ({qty(plan.position_size)} @ {plan.leverage:.1f}x) "
                f"at market {px(market_price)}  risk {watch['risk_pct']:.0%}",
                f"Move the stop on the WHOLE position to {px(new_stop)} — the {watch['name']}'s own level, "
                f"which is where the break is proven wrong.",
            ]
        )

        order = TradeOrder(
            symbol=symbol,
            direction=direction,
            legs=[OrderLeg(size=plan.position_size, order_type="market")],
            stop_loss=new_stop,
            leverage=plan.leverage,
            strategy_tag=watch["strategy_tag"],
        )

        def on_approve() -> None:
            if not self._auto_executes(watch["strategy_tag"]):
                return  # alert-only strategy: placed by hand, same as its entry
            result = self.executor.execute(order)
            if not result.ok:
                asyncio.create_task(
                    self.bot.send_message(
                        f"ADD-ON FAILED for {symbol} {direction} ({watch['strategy_tag']}): {result.error}\n"
                        f"The original position is untouched and nothing was retried."
                    )
                )
                return

            # TELL THE JOURNAL THE POSITION GREW. Until this was here the
            # add-on wrote nothing back, so the row kept its original size,
            # entry and risk while the exchange held twice the position.
            # total_open_risk() enforces the aggregate cap off that column, so
            # the cap was undercounting exactly the trades carrying the most
            # risk - and committed_margin(), multiplying stale size by stale
            # entry, was wrong the same way.
            #
            # Read back from BITGET rather than adding the plan's numbers on:
            # the plan knew an intended size at an expected price, the exchange
            # knows what actually filled and at what average. Same rule
            # breakeven_price() follows, learned on XAGUSDT #17.
            try:
                position = check_position_now(self.bitget, symbol, direction)
                if position:
                    self.storage.resync_position(
                        trade.מספר_עסקה,
                        entry_price=position["entry_price"],
                        position_size=position["size"],
                        stop=new_stop,
                        leverage=position.get("leverage"),
                    )
            except Exception:
                # The add-on is already placed; failing to record it must not
                # raise into the button handler. Logged loudly because a silent
                # miss here is the bug this block exists to fix.
                logger.exception(
                    "Add-on for %s executed but the journal could not be updated - "
                    "total_open_risk and committed_margin now understate this trade",
                    symbol,
                )

        await self.bot.send_signal(
            text,
            on_approve,
            expiry_seconds=signal_expiry_seconds(watch["timeframe"]),
            # An add-on goes in at market, so entry and reference are the same
            # price here - the starting gap is zero and any drift counts.
            entry_price=market_price,
            stop_loss=new_stop,
            reference_price=market_price,
            price_fetcher=lambda: self.bitget.get_mark_price(symbol),
        )
