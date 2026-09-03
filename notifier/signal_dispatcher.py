"""Turns a fresh Signal into a sized, gated Approve/Reject alert - and, once
approved, into real orders and a tracked trade.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities. This is the largest and most load-bearing single piece
of the split: every gate a trade has to clear (the fill guard re-asked at
the real fill price, the swing-slot cap, the aggregate risk cap, the
exchange's minimum notional on each leg) lives here, and so does the
handoff to order placement and position tracking once Dror taps Approve.

Depends on ExitManager, PendingBreakWatcher and TradeLifecycleHandler
directly (not just a callable each) because dispatch and confirmation
genuinely call several of their methods, not one yes/no question - see
each parameter's own docstring for which.
"""

from __future__ import annotations

import asyncio
import logging

from core import ledger
from core.storage import Storage
from execution.executor import Executor, OrderLeg, TradeOrder
from execution.tracker import track_position, wait_for_signal_position
from notifier import chart
from notifier.exit_manager import ExitManager, PARTIAL_TAKE_FRACTION, REMAINDER_TARGET_RATIO
from notifier.pending_break_watcher import PendingBreakWatcher
from notifier.scanner_time import signal_expiry_seconds
from notifier.strategies.base import Signal, Strategy, signal_to_json
from notifier.trade_lifecycle import TradeLifecycleHandler

logger = logging.getLogger(__name__)

# Relative tolerance for deciding two price levels are the same one. A
# strategy whose own reward:risk already equals REMAINDER_TARGET_RATIO puts
# both exit tiers on the identical price, and describing that as a partial
# take plus a stop-to-breakeven is describing steps that cannot happen.
_PRICE_EPSILON = 1e-9
# A "remainder" this small is float noise from position_size x 1.0, not a
# tranche anyone can close.
_SIZE_EPSILON = 1e-12


def _reward_target(entry_price: float, stop_loss: float, direction: str, ratio: float) -> float:
    risk_per_unit = abs(entry_price - stop_loss)
    return entry_price + risk_per_unit * ratio if direction == "long" else entry_price - risk_per_unit * ratio


def _build_order(signal: Signal, plan, market_price: float) -> TradeOrder:
    """The trade as orders: the legs the alert describes, plus the stop.

    A split entry is two legs at two prices, so it is placed as two orders -
    not one averaged order, which would fill the whole position at market and
    discard the better price the limit leg exists to get.
    """
    if signal.limit_entry is None:
        legs = [OrderLeg(size=plan.position_size, order_type="market")]
    else:
        market_size = plan.position_size * signal.market_fraction
        limit_size = plan.position_size - market_size
        legs = []
        if market_size > 0:
            legs.append(OrderLeg(size=market_size, order_type="market"))
        legs.append(
            OrderLeg(size=limit_size, order_type="limit", price=signal.limit_entry, note=signal.limit_note)
        )

    return TradeOrder(
        symbol=signal.symbol,
        direction=signal.direction,
        legs=legs,
        stop_loss=signal.stop_loss,
        leverage=plan.leverage,
        strategy_tag=signal.strategy_tag,
    )


class SignalDispatcher:
    def __init__(
        self,
        bitget,
        storage: Storage,
        bot,
        executor: Executor,
        exits: ExitManager,
        pending_breaks: PendingBreakWatcher,
        lifecycle: TradeLifecycleHandler,
        *,
        risk_pct: float,
        reward_risk_ratio: float,
        max_total_risk_pct: float,
        swing_tags: frozenset[str],
        max_swing_slots: int,
        send_chart_images: bool,
        already_exposed,
        symbol_max_leverage,
        auto_executes,
        mark_alerted,
        plan_position,
        round_trip_fee_for,
    ):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        self.executor = executor
        self.exits = exits
        self.pending_breaks = pending_breaks
        self.lifecycle = lifecycle
        self.risk_pct = risk_pct
        self.reward_risk_ratio = reward_risk_ratio
        self.max_total_risk_pct = max_total_risk_pct
        self.swing_tags = swing_tags
        self.max_swing_slots = max_swing_slots
        self.send_chart_images = send_chart_images
        # Callables into Scanner's own remaining state/logic, rather than
        # Scanner itself - this class only ever needs these specific
        # questions answered.
        self._already_exposed = already_exposed
        self._symbol_max_leverage = symbol_max_leverage
        self._auto_executes = auto_executes
        self._mark_alerted = mark_alerted
        self._plan_position = plan_position
        self._round_trip_fee_for = round_trip_fee_for

    async def dispatch(
        self,
        signal: Signal,
        equity: float,
        timeframes: list[str],
        confluence: str | None = None,
        pending_pattern: tuple | None = None,
        *,
        strategy: Strategy | None = None,
        bars_by_tf: dict | None = None,
    ) -> None:
        if self._already_exposed(signal.symbol):
            return  # already in this symbol; one at a time

        reward_risk_ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        # A chart pattern no longer raises risk merely by existing. It used to:
        # confluence() accepts a breakout up to CONFLUENCE_BARS old, so TRXUSDT
        # was sized at 2% citing a wedge that had broken 17 hours earlier while
        # the structure actually in front of price was an unresolved flag that
        # could still have broken down. Risk is staged instead - the trade goes
        # on at base risk here, and the pattern earns its second increment only
        # when it actually breaks. A strategy's OWN tier (Strategy 2 reading a
        # second timeframe's stack) is untouched: that is a different kind of
        # evidence and it is confirmed at signal time, not pending.
        risk_pct = signal.risk_pct_override if signal.risk_pct_override is not None else self.risk_pct
        available_budget = equity - self.storage.committed_margin()

        # The headline Entry is where the market is right now, so the alert can
        # be read against the chart at a glance. It is also needed *before*
        # sizing, because a split entry's real cost basis includes the market
        # leg (see plan_entry below).
        try:
            market_price = self.bitget.get_mark_price(signal.symbol)
        except Exception:
            logger.exception("Could not read mark price for %s; showing the planned entry", signal.symbol)
            market_price = signal.entry_price

        # A split entry holds both legs, so the position's actual cost basis is
        # their weighted average - not the limit level alone. Sizing off the
        # limit understated risk on 91% of replayed Strategy 1 signals, by a
        # median of 29% and up to 60%, which quietly pushed a 2%-risk trade to
        # over 3% of equity. This is also the only price at which "move the
        # stop to breakeven" is actually breakeven.
        if signal.limit_entry is not None and signal.market_fraction > 0:
            plan_entry = signal.market_fraction * market_price + (1 - signal.market_fraction) * signal.limit_entry
        elif signal.limit_entry is None and signal.market_fraction >= 1.0:
            # A pure market entry fills at MARKET, not at the level that
            # selected it. The branch above covers a split entry; this one was
            # missing, so a strategy whose entry_price is a REFERENCE rather
            # than an expected fill was sized against a price it never gets.
            #
            # Strategy 2.1 is exactly that: entry_price is e9_prev, the EMA9,
            # while ENTRY_MODE="next_open" enters at market on the candle AFTER
            # the rejection - which by construction has closed back on the trend
            # side, so the fill is always on the far side of the EMA9. Measured
            # over 591 setups on WLD/SOL/PEPE, the real distance to the stop is
            # 1.89x the distance sized against (median 1.44x, p90 2.97x), and
            # worse on 100% of them. A trade meant to risk 1% risked ~1.9%, and
            # 25% of them breached the 2% cap outright.
            #
            # The drift guard does not catch it: gap_at_dispatch is subtracted
            # before drift is counted, which is right for a resting limit - the
            # gap is distance the order waits to cross - and wrong for a market
            # order, where the same gap is simply the wrong sizing basis.
            #
            # Strategy 1 blends both legs above; Strategy 4 is pure limit and
            # its entry_price IS the limit; Strategy 3 sets entry = close_now.
            # 2.1 is the only instance that lands here.
            plan_entry = market_price
        else:
            plan_entry = signal.entry_price

        try:
            plan = self._plan_position(
                equity=equity,
                risk_pct=risk_pct,
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                direction=signal.direction,
                reward_risk_ratio=reward_risk_ratio,
                available_budget=available_budget,
                max_leverage=self._symbol_max_leverage(signal.symbol),
                # The fee sizing is meant to absorb depends on how much of THIS
                # signal's entry is market vs limit - a flat assumption is
                # right for Strategy 4 (all limit) and close enough for a 20%
                # split, but Strategy 2.1 enters 100% at market under
                # ENTRY_MODE="next_open" and its true round-trip fee is taker
                # both legs, 50% more than the flat default. See
                # round_trip_fee_for's own docstring.
                round_trip_fee_pct=self._round_trip_fee_for(signal.market_fraction),
            )
        except ValueError as exc:
            logger.info("Skipping %s/%s: %s", signal.symbol, signal.strategy_tag, exc)
            return

        # THE GATES, RE-ASKED AGAINST THE PRICE THE TRADE ACTUALLY FILLS AT.
        #
        # A strategy decides from a LEVEL, and for most of them that level is
        # also the fill, so the two questions have one answer. Strategy 2.1
        # breaks that: its entry_price is the EMA9 that selected the setup while
        # the order goes in at market on the candle after the rejection, which
        # has closed back on the trend side by construction - so the fill is
        # always on the far side of the level.
        #
        # HYPEUSDT, 2026-08-18: 1.50% stop measured against its EMA9, 0.15%
        # against plan_entry. It passed the strategy's own 0.30% floor while
        # failing it by half. Commit 60aa796 fixed the SIZING to use this price;
        # the gates that decide whether to trade at all were still asking about
        # the EMA9.
        #
        # Logged under its own decision rather than dropped, so a refusal is
        # visible in the weekly stats. A signal that vanishes silently is the
        # same failure mode as one that never fired.
        if signal.fill_guard is not None:
            refusal = signal.fill_guard.refuses(plan_entry, signal.stop_loss, reward_risk_ratio)
            if refusal is not None:
                logger.info(
                    "Skipping %s/%s at the fill: %s", signal.symbol, signal.strategy_tag, refusal
                )
                signal_id = self.storage.log_signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=plan_entry,
                    stop_loss=signal.stop_loss,
                    take_profit=plan.take_profit,
                    strategy_tag=signal.strategy_tag,
                    confluence=confluence,
                )
                self.storage.mark_signal_decision(signal_id, "refused_at_fill")
                return

        # The swing pool's own hard slot cap, enforced independently of and in
        # addition to the aggregate dollar cap below - two swing trades can
        # each be well under the dollar cap and still be the two the pool
        # allows, at which point a third swing signal is suppressed outright
        # regardless of how much risk budget remains.
        if signal.strategy_tag in self.swing_tags:
            swing_open = sum(
                1
                for t in (*self.storage.pending_trades(), *self.storage.open_trades())
                if t.תגית_אסטרטגיה in self.swing_tags
            )
            if swing_open >= self.max_swing_slots:
                logger.info(
                    "Skipping %s/%s: swing slots full (%d/%d)",
                    signal.symbol,
                    signal.strategy_tag,
                    swing_open,
                    self.max_swing_slots,
                )
                signal_id = self.storage.log_signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=plan_entry,
                    stop_loss=signal.stop_loss,
                    take_profit=plan.take_profit,
                    strategy_tag=signal.strategy_tag,
                    confluence=confluence,
                )
                self.storage.mark_signal_decision(signal_id, "swing_slots_full")
                return

        risk_cap = equity * self.max_total_risk_pct
        open_risk = self.storage.total_open_risk()
        if open_risk + plan.risk_amount > risk_cap:
            logger.info(
                "Skipping %s/%s: total open risk %.2f + %.2f would exceed the %.0f%% cap (%.2f)",
                signal.symbol,
                signal.strategy_tag,
                open_risk,
                plan.risk_amount,
                self.max_total_risk_pct * 100,
                risk_cap,
            )
            return

        specs = self.bitget.get_contract_specs(signal.symbol)

        # Bitget enforces its minimum notional on EACH order it receives, not
        # on the position as a whole. A split entry is two separate orders, so
        # the market leg - always the smaller one, since market_fraction is
        # never above 0.5 in a strategy that splits - is what actually binds.
        # ADAUSDT cleared $6.35 total and was still rejected live, because its
        # 20% market leg alone was $1.05, under the $5 floor.
        #
        # Per Dror's call: don't fall back to a single non-split order, and
        # don't send the alert at all when the trade genuinely isn't
        # placeable at this size - but log it under its own decision so it
        # stays visible in the weekly stats rather than silently vanishing,
        # which is exactly what "the signals table only ever gains a row on
        # approval" used to do to rejected/ignored signals.
        too_small = plan.position_size < specs["min_size"] or plan.notional_value < specs["min_notional"]
        if not too_small and signal.limit_entry is not None and signal.market_fraction > 0:
            market_leg_notional = plan.position_size * signal.market_fraction * market_price
            too_small = market_leg_notional < specs["min_notional"]

        # The checks above compare dollars against the size the PLAN wants,
        # not the size that actually gets sent. round_size floors to the
        # exchange's own step, so a leg clears the minimum on paper and is then
        # rejected as the smaller quantity that leaves.
        #
        # Two live failures, one from each half of that:
        #   AAVEUSDT trades in units of 0.1. A 0.06-unit market leg was worth
        #   $6, over the $5 minimum, but rounding floored it to 0.
        #   CLUSDT rounds to 2dp. Its market leg of 0.064938 was worth $5.30,
        #   over the same minimum, but rounded to 0.06 it was worth $4.896 and
        #   Bitget refused it with "less than the minimum amount 5 USDT" -
        #   after the signal had been approved and the entry attempted.
        #
        # So each leg is now valued at the quantity that will really be sent,
        # AND at its own price: the market leg fills at market while the limit
        # leg rests at signal.limit_entry, and on a split entry those differ by
        # design - valuing both at one price is what makes a leg look bigger
        # than it is.
        if not too_small:
            legs = [(plan.position_size, market_price)]
            if signal.limit_entry is not None and signal.market_fraction > 0:
                market_size = plan.position_size * signal.market_fraction
                legs = [
                    (market_size, market_price),
                    (plan.position_size - market_size, signal.limit_entry),
                ]
            for size, price in legs:
                rounded = self.bitget.round_size(signal.symbol, size)
                if rounded <= 0 or rounded * price < specs["min_notional"]:
                    too_small = True
                    break

        if too_small:
            logger.info(
                "Not tradeable at this size: %s/%s (notional %.2f, exchange minimum %.2f)",
                signal.symbol,
                signal.strategy_tag,
                plan.notional_value,
                specs["min_notional"],
            )
            signal_id = self.storage.log_signal(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                take_profit=plan.take_profit,
                strategy_tag=signal.strategy_tag,
                confluence=confluence,
            )
            self.storage.mark_signal_decision(signal_id, "too_small")
            return

        # A strategy that sets partial_fraction manages its own two-tier exit;
        # everything else takes the scanner's 50% / 1:3 default.
        strategy_manages_exit = signal.partial_fraction is not None
        partial_fraction = signal.partial_fraction if strategy_manages_exit else PARTIAL_TAKE_FRACTION
        partial_size = plan.position_size * partial_fraction
        remainder_size = plan.position_size - partial_size
        remainder_target = _reward_target(plan_entry, signal.stop_loss, signal.direction, REMAINDER_TARGET_RATIO)

        # A pending pattern is only worth naming if this trade would live long
        # enough to see it break. Dror's rule: the break must fall between the
        # stop and the final target, because outside that window the trade has
        # already resolved - past the stop it is closed, past the target it is
        # closed - and the +1% the alert promises could never be offered.
        #
        # Deliberately NO threshold constant. The bound comes from levels the
        # trade already defines, which is why this could be settled without the
        # proximity number nobody had measured: a real run found 27 of 50
        # symbol-directions carrying a pending pattern, with break levels
        # ranging from +0.76% (BTCUSDT) to -13.11% (SOLUSDT). The far ones are
        # not slightly worse, they are unreachable, and they cost attention at
        # approval time to read and dismiss.
        #
        # Bounded by the FINAL target rather than the partial: the position is
        # still open through it, and the add-on's never-loosen stop rule
        # already handles a partial having pulled the stop to breakeven first.
        #
        # Applied here rather than inside pending() because R is only defined
        # once a pattern is attached to a signal that has an entry and a stop.
        if pending_pattern is not None:
            level = pending_pattern[0].break_level
            lo, hi = sorted((signal.stop_loss, remainder_target))
            if not lo < level < hi:
                logger.info(
                    "Pending %s on %s not reachable inside this trade (%s vs %s..%s); not quoting it",
                    pending_pattern[0].name, signal.symbol, level, lo, hi,
                )
                pending_pattern = None

        # Prices and sizes at the precision the exchange actually quotes, so
        # every level in the alert is a value that can be entered as an order.
        def px(value: float) -> str:
            return f"{value:.{specs['price_place']}f}"

        def qty(value: float) -> str:
            return f"{value:.{specs['volume_place']}f}"

        def usd(value: float) -> str:
            return f"${value:,.0f}" if abs(value) >= 10 else f"${value:,.2f}"

        # Logged before Approve/Reject is even seen, so a rejected or ignored
        # signal is still measurable - the trades table only ever gains a row
        # once a signal is both approved AND confirmed on Bitget, which left
        # rejected/ignored signals with no record anywhere. Moved ahead of
        # `lines` (it used to run after) so the id it returns can go in the
        # header instead of a separate trailing line - Dror: "make the signal
        # to be in this format is smaller then the corrent one".
        signal_id = self.storage.log_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=plan_entry,
            stop_loss=signal.stop_loss,
            take_profit=plan.take_profit,
            strategy_tag=signal.strategy_tag,
            confluence=confluence,
            # The whole Signal, so this alert can be offered again later by its
            # number if it expires on a setup Dror still likes. See
            # signal_to_json: the columns beside it cannot rebuild an exit.
            signal_json=signal_to_json(signal),
        )

        lines = [f"Signal #{signal_id}  {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})"]
        if signal.analysis_timeframes is not None:
            # Only worth a line when it says something the tag doesn't
            # already: analysis_timeframes is None for the common case (the
            # strategy's own fixed timeframes, which the tag already names),
            # and set only when a signal-specific read genuinely differs -
            # e.g. Strategy 2's second timeframe, listed only when it was a
            # real confirmation rather than a supportive trend read.
            lines.append(f"Analysis timeframe: {', '.join(timeframes)}")
        lines.extend([
            # "Entry" means the market price for a SPLIT entry, where neither
            # leg alone is the cost basis and the blend is what sizing uses.
            # A strategy with one leg at one price has a real entry, and
            # printing the market instead reads as a second, wrong number:
            # Dror on a Strategy 4 alert - "why the entry is different from the
            # limit there is only one entry in this strategy".
            (
                f"Entry: {px(plan_entry)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}"
                if signal.limit_entry is not None and signal.market_fraction == 0
                else f"Entry: {px(market_price)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}"
            ),
            f"Size: {usd(plan.notional_value)} ({qty(plan.position_size)} @ {plan.leverage:.1f}x)"
            + (f"  risk {risk_pct:.0%}" if risk_pct > self.risk_pct else ""),
        ])
        if confluence:
            lines.append(f"Confirmed by {confluence}")

        # The unbroken structure in front of price, stated with the level that
        # would confirm it and how far off that is, so the setup can be
        # validated against the chart at approval time rather than taken on
        # trust. For triangles and wedges the level sits on a converging line
        # and moves every bar, so its drift is stated too - quoting a number
        # that silently goes stale is exactly what made the old confluence
        # bump misleading.
        if pending_pattern is not None:
            pat, pat_tf = pending_pattern
            side = "above" if pat.direction == "long" else "below"
            gap = (pat.break_level - market_price) / market_price * 100 if market_price else 0.0
            drift = ""
            if pat.drift_per_bar:
                drift = f", {'rising' if pat.drift_per_bar > 0 else 'falling'} {abs(pat.drift_per_bar):.{specs['price_place']}f}/bar"
            lines.append(
                f"Pending {pat.name} on {pat_tf}: breaks {side} {px(pat.break_level)} "
                f"({gap:+.2f}% away{drift}). Risk stays {risk_pct:.0%} until it breaks."
            )

        # A split entry is two orders at two prices, so the alert states how
        # much goes into each rather than leaving the arithmetic to be done at
        # the moment of placing them. Each leg's dollar value is its own
        # quantity at its own price, not a share of the total notional.
        if signal.limit_entry is not None:
            market_size = plan.position_size * signal.market_fraction
            limit_size = plan.position_size - market_size
            note = f" ({signal.limit_note})" if signal.limit_note else ""

            if market_size <= 0:
                # No market fraction: the whole position rests on the limit,
                # same as Strategy 2 entering at the EMA9 touch rather than the
                # candle close. There is no partial fill to give a fallback
                # target for here - if it never fills, nothing is open.
                lines.append(f"Enter: {usd(limit_size * signal.limit_entry)} ({qty(limit_size)}) limit {px(signal.limit_entry)}{note}")
            else:
                lines.append(
                    f"Enter: {usd(market_size * market_price)} ({qty(market_size)}) at market {px(market_price)}"
                    f"  ·  {usd(limit_size * signal.limit_entry)} ({qty(limit_size)}) limit {px(signal.limit_entry)}{note}"
                )
                # Every level above is measured from the blended entry both legs
                # would produce. If the limit never fills, the position is only
                # the market fraction, bought at a worse price, and those levels
                # no longer describe it.
                #
                # Re-anchored on the RATIO, not the distance. Carrying the same
                # absolute reward across was the first version and it silently
                # halved the trade: the market fill is further from the stop
                # than the blended entry would have been, so the same dollar
                # gain is a smaller multiple of a larger risk. ZECUSDT trade #13
                # is the worked example - planned entry 483.774 against a 467.97
                # stop is 15.80 of risk and a 515.377 target, but only the market
                # leg filled at 498.41, making the real risk 30.44. The
                # distance-based fallback of 530.01 came out at 1.04R, and Dror
                # closed it at 529.00 for 0.99R on a setup meant to pay 2R.
                #
                # Measuring from the market fill's own risk keeps the 1:2 the
                # strategy is built around. It does ask for a bigger move -
                # 559.29 rather than 530.01 on ZECUSDT - which is the honest
                # price of the worse entry rather than a target being moved for
                # convenience.
                fallback_target = _reward_target(
                    market_price, signal.stop_loss, signal.direction, reward_risk_ratio
                )
                lines.append(
                    f"If the limit leg never fills: exit the market-only {qty(market_size)} at {px(fallback_target)}."
                )

        # Only a real two-tier exit is worth describing. When the strategy's own
        # reward:risk already equals the remainder ratio both tiers land on the
        # same price, so the partial and the stop-to-breakeven step do nothing.
        if strategy_manages_exit and remainder_size <= _SIZE_EPSILON:
            # A strategy can manage its own exit by having only ONE tier.
            # Strategy 4 closes the whole position at its gap target, so there
            # is no partial to describe and no runner to hand over - the
            # headline Target line above already states it. Printing the
            # two-tier sentence here would say "close the remaining 0".
            pass
        elif strategy_manages_exit:
            if signal.remainder_target is not None:
                note = f" ({signal.remainder_note})" if signal.remainder_note else ""
                tail = f"at {px(signal.remainder_target)}{note}"
            else:
                # No price for the runner - the setup is at all-time highs with
                # nothing overhead, so the exit is a rule rather than a level.
                tail = signal.remainder_note or "at your discretion"
            lines.append(
                f"Partial: close {qty(partial_size)} ({partial_fraction:.0%}) at {px(plan.take_profit)}, "
                f"move stop to {px(plan_entry)}, then close the remaining {qty(remainder_size)} {tail}."
            )
        elif abs(plan.take_profit - remainder_target) > _PRICE_EPSILON * max(abs(remainder_target), 1.0):
            lines.append(
                f"Partial: close {qty(partial_size)} ({partial_fraction:.0%}) at {px(plan.take_profit)}, "
                f"move stop to {px(plan_entry)}, then close the remaining "
                f"{qty(remainder_size)} at {px(remainder_target)} (1:{REMAINDER_TARGET_RATIO:g})."
            )

        lines.extend(signal.extra_notes)

        # The id in the header IS the /add hint now - it used to be a
        # separate trailing line, spelling out "/add {signal_id}" so there
        # was something to type. Without the number showing SOMEWHERE, the
        # alternative is naming the strategy by hand, which is how XAGUSDT
        # #17 came to be tagged "Strategy 1 1h" against a "Strategy 1 1H"
        # alert and went its whole life unmanaged - the header now carries
        # that same number instead.
        text = "\n".join(lines)

        def on_approve() -> None:
            self.storage.mark_signal_decision(signal_id, "approved")
            # ROUNDED TO THE EXCHANGE'S OWN PRICE PRECISION before it is
            # recorded as "the plan". Left as the strategy's raw float, this
            # was never equal to what actually lands on Bitget - every stop
            # and target placed there is rounded to the symbol's price_place
            # first (round_price, called inside place_order/place_tpsl_order),
            # while the value stored here was not.
            #
            # changed_from_plan compares the two at a tolerance of 1e-9, so
            # that mismatch reads as a genuine deviation on almost every trade.
            # AIOUSDT (Strategy 2.1 15m, 2026-08-19) is typical: original
            # 0.04359480847113895, actual 0.04359 - a difference of 4.8e-6 that
            # is pure price-tick rounding, four orders of magnitude past the
            # tolerance, and the close message told Dror the stop had been
            # changed when nothing had touched it.
            #
            # Rounding here instead of loosening the tolerance keeps
            # changed_from_plan meaning what it says: a real difference, in
            # units the exchange can actually place, rather than "did the
            # strategy happen to compute a round number". get_contract_specs
            # is cached for the process lifetime and was already read above
            # for the min-notional check, so this costs no extra request.
            try:
                proposed_stop = self.bitget.round_price(signal.symbol, signal.stop_loss)
                proposed_target = self.bitget.round_price(signal.symbol, plan.take_profit)
            except Exception:
                logger.exception("Could not round %s's plan to price precision; recording it raw", signal.symbol)
                proposed_stop, proposed_target = signal.stop_loss, plan.take_profit
            trade_id = self.storage.create_pending(
                symbol=signal.symbol,
                direction=signal.direction,
                proposed_stop=proposed_stop,
                proposed_target=proposed_target,
                strategy_tag=signal.strategy_tag,
            )
            self.storage.link_signal_trade(signal_id, trade_id)

            # Start waiting for the pattern to resolve. Registered on approval
            # rather than at dispatch, so a signal that was never taken cannot
            # produce an add-on for a position that does not exist.
            if pending_pattern is not None:
                pat, pat_tf = pending_pattern
                self.pending_breaks.register(signal.symbol, {
                    "direction": signal.direction,
                    "name": pat.name,
                    "timeframe": pat_tf,
                    "break_level": pat.break_level,
                    "invalidation_level": pat.invalidation_level,
                    "trade_id": trade_id,
                    "strategy_tag": signal.strategy_tag,
                    "risk_pct": risk_pct,
                })

            order = _build_order(signal, plan, market_price)

            async def _execute_and_track() -> None:
                if self._auto_executes(signal.strategy_tag):
                    # Off the event loop, on a worker thread: BitgetClient is
                    # synchronous (requests, not async httpx), so calling
                    # execute() straight from on_approve used to freeze the
                    # WHOLE bot - the scan loop and every other Telegram tap -
                    # for as long as leverage-set plus each order leg took to
                    # answer. on_approve has already returned by the time this
                    # runs, so the "Approved." edit Dror sees no longer waits
                    # on it either - it was already just an ack that the tap
                    # was received, never proof the order went through; a
                    # separate EXECUTION FAILED / confirmation message always
                    # followed on its own regardless.
                    result = await asyncio.to_thread(self.executor.execute, order)
                    if not result.ok:
                        # Fail-safe: no retry. The account is the only truth
                        # about what exists after an ambiguous failure, so say
                        # what was attempted and stop rather than guessing.
                        self.storage.cancel_pending(trade_id, f"execution failed: {result.error}")
                        await self.bot.send_message(
                            f"EXECUTION FAILED for {signal.symbol} {signal.direction} "
                            f"({signal.strategy_tag}): {result.error}\n"
                            f"{len(result.placed)} of {len(order.legs)} leg(s) were placed before it stopped. "
                            f"Check the account before acting — nothing was retried."
                        )
                        return
                    ledger.try_record(self.storage.db_path, ledger.ENTRY_ORDER_PLACED)

                await self.confirm_and_track(
                    trade_id,
                    signal,
                    plan,
                    order,
                    # The runner's fallback level, carried through so the
                    # partial-fill handler places exactly what the alert
                    # promised. The breakeven is NOT carried: plan_entry is an
                    # estimate made before anything filled, and the confirmed
                    # position knows better - see set_exit_plan below.
                    remainder_target=signal.remainder_target
                    if signal.partial_fraction is not None
                    else remainder_target,
                )

            asyncio.create_task(_execute_and_track())

        def on_reject() -> None:
            self.storage.mark_signal_decision(signal_id, "rejected")

        # Best-effort, never a reason to withhold the alert - see chart.build's
        # own docstring. Only attempted when both the setting is on AND the
        # caller actually has bars to draw from (poll_pending_breaks/
        # _offer_add_on don't pass bars_by_tf, and that's fine: those alerts
        # still go out text-only, same as ever).
        photo = None
        if self.send_chart_images and strategy is not None and bars_by_tf is not None:
            photo = chart.build(bars_by_tf, strategy, signal, entry=plan_entry, target=plan.take_profit)

        # The prompt is the ONLY thing that spends the day's allowance, and it
        # is spent after the send rather than before it - see _throttled.
        try:
            await self.bot.send_signal(
                text,
                on_approve,
                on_reject,
                expiry_seconds=signal_expiry_seconds(timeframes[0]),
                # plan_entry defines 1R with the stop, since that is where the
                # order actually rests; market_price is only the starting point
                # drift is measured FROM. Passing market_price as both (the
                # first attempt at the QQQUSDT fix) made 1R three times too
                # large on INJUSDT, whose limit sits far from market by
                # construction. See NotifierBot._expire for why all three
                # prices are distinct.
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                reference_price=market_price,
                price_fetcher=lambda: self.bitget.get_mark_price(signal.symbol),
                photo=photo,
            )
        except Exception:
            # Deliberately NOT retried. A Telegram timeout is ambiguous - the
            # message often did arrive - so resending is as likely to double-post
            # as to deliver. The throttle simply goes unspent, and _seen keeps
            # this setup from re-prompting for the life of the process.
            logger.exception("Could not send the %s %s alert", signal.symbol, signal.strategy_tag)
            self.storage.mark_signal_decision(signal_id, "send_failed")
            return

        self._mark_alerted((signal.symbol, signal.strategy_tag))

    async def confirm_and_track(
        self,
        trade_id: int,
        signal: Signal,
        plan=None,
        order: TradeOrder | None = None,
        remainder_target: float | None = None,
    ) -> None:
        # Only a strategy whose orders really reach the exchange should have
        # exit orders placed for it; a dry-run strategy must stay dry all the
        # way through, not just at entry.
        executed = (
            order is not None
            and self._auto_executes(signal.strategy_tag)
            and self.executor.handles_live(signal.strategy_tag)
        )
        # A strategy that measured its own unfilled window gets it; everything
        # else keeps the tracker's flat default.
        position = await wait_for_signal_position(
            self.bitget, signal.symbol, signal.direction,
            **({"timeout_seconds": signal.unfilled_timeout_seconds}
               if signal.unfilled_timeout_seconds else {}),
        )
        if position is None:
            self.storage.cancel_pending(trade_id)
            # Nothing filled, so any resting leg - bot-placed or placed by
            # hand off the alert - is an order with no trade behind it. Left
            # alone it could open a position hours later against a setup that
            # no longer exists, so it's cancelled regardless of who placed it.
            self.exits.cancel_resting(signal.symbol)
            await self.bot.send_message(
                f"No position detected for trade #{trade_id} ({signal.symbol} {signal.direction}) "
                f"within the timeout — marked cancelled, and {signal.symbol} is free to signal again."
            )
            return

        stop, target = self.exits.safe_stop_target(signal.symbol, signal.direction, position)
        self.storage.confirm_entry(
            trade_id,
            position["entry_price"],
            position["size"],
            stop,
            target,
            leverage=position["leverage"],
        )

        if stop is None:
            await self.bot.send_message(
                f"Trade #{trade_id} ({signal.symbol} {signal.direction}) is being tracked, but no stop-loss "
                f"is set on Bitget — R-multiple can't be computed until you set one."
            )

        # The exit plan goes to the DB rather than staying in this coroutine's
        # closure. It used to live only here, so a restart - and every deploy
        # is one - left the re-attached tracker able to SEE the partial fill
        # but with no idea that a breakeven was owed. Recorded only when the
        # bot really manages this trade's exits, so a set breakeven_stop is a
        # commitment rather than a note.
        #
        # The breakeven recorded is the CONFIRMED entry, not the alert's
        # plan_entry. plan_entry blends the market leg's expected fill with
        # the limit level before either has happened, so it is an estimate
        # twice over; what actually filled is known here. breakeven_price()
        # re-derives it from the row at partial time anyway, which is what
        # picks up a limit leg that fills later - this only keeps the stored
        # value from being a number nothing would ever use.
        # BOTH TARGETS RE-ANCHORED ON THE REAL FILL, for the same reason the
        # breakeven above is: plan_entry blends the market leg's EXPECTED fill
        # with the limit level before either has happened, so it is an
        # estimate, and position["entry_price"] is what actually filled.
        #
        # Until this, plan.take_profit and remainder_target stayed frozen at
        # their plan_entry-derived values all the way through - the exact
        # comment three lines above ("plan_entry is an estimate... the
        # confirmed position knows better") was already true of them and
        # applied only to the stop. Measured across every closed Strategy 1
        # trade, plan_entry differs from the real fill by more than 1% on a
        # third of them - EULUSDT +4.6%, DEXEUSDT -5.5%, ZECUSDT #13 +3.0%,
        # SNDKUSDT #54 -1.4% - and each point of that gap is a point the
        # realized reward:risk drifts from the 1:2 / 1:3 the strategy is
        # built around, in either direction. SNDKUSDT #54 closed both legs
        # almost exactly on its (wrong) targets and still read 1.17R instead
        # of a number nearer 2, because the risk it was measured against
        # (from the real, worse entry) was 60% wider than the risk the
        # targets were priced for.
        #
        # Safe for a pure-limit strategy (Strategy 4, market_fraction=0):
        # position["entry_price"] there equals signal.entry_price exactly, so
        # this is a no-op. It only moves anything when a market leg pulled the
        # blended entry away from what was planned - which is precisely when
        # the old prices were wrong.
        ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        real_entry = position["entry_price"]
        if plan is not None:
            plan.take_profit = _reward_target(real_entry, signal.stop_loss, signal.direction, ratio)
        # remainder_target is only ever this ratio-derived fallback when the
        # strategy did NOT supply its own (partial_fraction is None) - the
        # call site above already resolves signal.remainder_target through
        # unchanged for a self-managing strategy, and that absolute price (or
        # None, meaning "trail") is the strategy's own decision, not an
        # estimate to correct.
        if signal.partial_fraction is None:
            remainder_target = _reward_target(
                real_entry, signal.stop_loss, signal.direction, REMAINDER_TARGET_RATIO
            )

        if self.exits.manages_exits(signal.strategy_tag):
            self.storage.set_exit_plan(
                trade_id,
                breakeven_stop=position["entry_price"],
                runner_target=remainder_target,
                partial_fraction=signal.partial_fraction,
                # Recorded because the partial handler rebuilds the signal from
                # this row - see exit_plan_signal. Without it, a strategy that
                # deliberately asks for NO runner target is indistinguishable
                # from one that simply has not said.
                runner_target_is_final=signal.remainder_target_is_final,
            )

        # The partial can't ride on the entry the way the stop does: a preset
        # carries one target for the whole position, and this closes only part
        # of it. Sized to what actually filled rather than to the intended
        # position - on a split entry the market leg confirms first, and the
        # limit leg may fill later or never.
        if executed and plan is not None:
            await self.exits.place_partial(signal, plan, position["size"])

        await track_position(
            self.storage,
            self.bitget,
            trade_id,
            signal.symbol,
            signal.direction,
            on_close=self.lifecycle.on_trade_closed,
            # The partial filling is what promotes the runner from "at your
            # discretion" to a real order: the stop goes to the breakeven the
            # alert already printed, and the runner gets a target. That now
            # happens inside on_partial_exit off the stored plan, so this is
            # the same callback resume_open_trades hands to a re-attached
            # tracker and there is exactly one path to a breakeven.
            on_partial=self.lifecycle.on_partial_exit,
            on_scale_in=self.lifecycle.on_scale_in,
            # on_resize fires synchronously from inside track_position's poll
            # loop, so the retryable coroutine has to be scheduled rather than
            # awaited here - awaiting would stall that loop's own polling for
            # as long as the retry takes.
            on_resize=(lambda size: asyncio.create_task(self.exits.place_partial(signal, plan, size, replace=True)))
            if (executed and plan is not None)
            else None,
        )
