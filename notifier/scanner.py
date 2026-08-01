"""Main scanning loop: pulls closed bars for each watchlist symbol, runs the
active strategies, and — on a fresh signal — sizes the position and dispatches
an Approve/Reject alert to Telegram.

Scans are aligned to candle closes rather than run on a fixed interval:
strategies evaluate closed candles, so between one close and the next the data
is identical and re-scanning only burns API quota. Each strategy declares the
timeframe(s) it needs; the scanner fetches the union of all of them and scans
at the cadence of the shortest one, so a strategy needing 1H+15m confluence
runs alongside a 1H-only strategy without special-casing either.

Approval doesn't log the trade immediately — it waits to see a matching
position actually appear on Bitget (execution.tracker.wait_for_signal_position),
the same account-based detection the `/add` flow and the future auto-execution
phase use. If none appears within the timeout the row is cancelled, which also
frees the symbol for future signals.
"""

import asyncio
import logging
import time

import pandas as pd

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.executor import Executor
from execution.tracker import (
    format_close_message,
    format_partial_message,
    track_position,
    wait_for_signal_position,
)
from notifier.risk_sizing import DEFAULT_MAX_LEVERAGE, DEFAULT_REWARD_RISK_RATIO, plan_position
from notifier.strategies.base import TIMEFRAME_SECONDS, Signal, Strategy

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOTAL_RISK_PCT = 0.06
CANDLE_CLOSE_DELAY = 30.0  # let Bitget settle the just-closed candle before reading it
PARTIAL_TAKE_FRACTION = 0.5
# Fixed 1:3 target for whatever's left after the partial take, regardless of
# the first tier's own ratio — replaces an open-ended "let it run" with a
# defined second exit, so nothing is left unmanaged indefinitely.
REMAINDER_TARGET_RATIO = 3.0
# Relative tolerance for deciding two price levels are the same one. A strategy
# whose own reward:risk already equals REMAINDER_TARGET_RATIO puts both exit
# tiers on the identical price, and describing that as a partial take plus a
# stop-to-breakeven is describing steps that cannot happen.
_PRICE_EPSILON = 1e-9


def _reward_target(entry_price: float, stop_loss: float, direction: str, ratio: float) -> float:
    risk_per_unit = abs(entry_price - stop_loss)
    return entry_price + risk_per_unit * ratio if direction == "long" else entry_price - risk_per_unit * ratio


def bars_dataframe(candles: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df


def seconds_until_next_close(timeframe: str, now: float | None = None) -> float:
    """Seconds until the next candle of this timeframe closes, plus a small
    settle delay."""
    period = TIMEFRAME_SECONDS[timeframe]
    now = time.time() if now is None else now
    return (period - (now % period)) + CANDLE_CLOSE_DELAY


class Scanner:
    def __init__(
        self,
        bitget: BitgetClient,
        bot,
        storage: Storage,
        executor: Executor,
        watchlist: list[str],
        strategies: list[Strategy],
        risk_pct: float = 0.01,
        reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
        max_leverage: float = DEFAULT_MAX_LEVERAGE,
        max_total_risk_pct: float = DEFAULT_MAX_TOTAL_RISK_PCT,
        candle_limit: int = 250,
    ):
        self.bitget = bitget
        self.bot = bot
        self.storage = storage
        self.executor = executor
        self.watchlist = watchlist
        self.strategies = strategies
        self.risk_pct = risk_pct
        self.reward_risk_ratio = reward_risk_ratio
        self.max_leverage = max_leverage
        self.max_total_risk_pct = max_total_risk_pct
        self.candle_limit = candle_limit
        self._seen: set[tuple] = set()

    def required_timeframes(self) -> set[str]:
        timeframes: set[str] = set()
        for strategy in self.strategies:
            timeframes.update(strategy.timeframes)
        return timeframes

    async def run_forever(self) -> None:
        timeframes = self.required_timeframes()
        if not timeframes:
            logger.warning("No strategies registered; scanner has nothing to do")
            return

        while True:
            scan_tf = min(timeframes, key=seconds_until_next_close)
            delay = seconds_until_next_close(scan_tf)
            logger.info("Next scan (driven by %s) in %.0fs", scan_tf, delay)
            await asyncio.sleep(delay)
            await self.tick()

    async def tick(self, timeframes: set[str] | None = None) -> None:
        timeframes = timeframes if timeframes is not None else self.required_timeframes()

        try:
            equity = self.bitget.get_account_equity()
        except Exception:
            # Sizing off a stale or guessed equity silently corrupts every
            # downstream number, so skip the scan instead.
            logger.exception("Could not read account equity; skipping this scan")
            return

        logger.info("Scanning %d symbols on %s (equity %.2f)", len(self.watchlist), ",".join(sorted(timeframes)), equity)

        for symbol in self.watchlist:
            # Fetched with the forming candle included, then trimmed per
            # strategy: most want closed bars only, but one reading a slow
            # trend off a longer timeframe needs the hour in progress rather
            # than a picture up to a full candle stale. One fetch serves both.
            bars_by_tf: dict[str, pd.DataFrame] = {}
            for tf in timeframes:
                try:
                    candles = self.bitget.get_candles(
                        symbol, granularity=tf, limit=self.candle_limit + 1, closed_only=False
                    )
                    bars_by_tf[tf] = bars_dataframe(candles)
                except Exception:
                    logger.exception("Skipping %s this scan: failed to fetch/parse %s candles", symbol, tf)
                    bars_by_tf = None
                    break

            if not bars_by_tf or any(len(b) < 2 for b in bars_by_tf.values()):
                continue

            for strategy in self.strategies:
                strategy_bars = {
                    tf: (bars_by_tf[tf] if tf in strategy.forming_bar_timeframes else bars_by_tf[tf].iloc[:-1])
                    for tf in strategy.timeframes
                    if tf in bars_by_tf
                }
                if len(strategy_bars) < len(strategy.timeframes):
                    continue  # one of this strategy's timeframes failed to fetch this scan

                try:
                    signal = strategy.evaluate(symbol, strategy_bars)
                except Exception:
                    logger.exception("Skipping %s/%s this scan: strategy raised", symbol, strategy.tag)
                    continue

                if signal is None:
                    continue

                # Keyed on the trade being proposed, not on the candle that
                # produced it. A per-candle key still re-alerts every time the
                # trigger re-fires against an unchanged leg: one stale TSLAUSDT
                # short went out four times over eleven hours, same entry, same
                # stop, while price walked 5 points past that stop. Identical
                # levels mean it is the same trade, however often it retriggers.
                dedupe_key = (
                    signal.symbol,
                    signal.strategy_tag,
                    signal.direction,
                    signal.entry_price,
                    signal.stop_loss,
                )
                if dedupe_key in self._seen:
                    continue
                self._seen.add(dedupe_key)
                self._prune_seen()

                await self._dispatch(signal, equity, strategy.timeframes)

    def _prune_seen(self, max_entries: int = 5000) -> None:
        """Bounded so a long-running process can't leak memory on a big watchlist."""
        if len(self._seen) > max_entries:
            self._seen = set(list(self._seen)[-max_entries // 2 :])

    async def _dispatch(self, signal: Signal, equity: float, timeframes: list[str]) -> None:
        if self.storage.has_open_or_pending(signal.symbol):
            return  # already tracking a trade on this symbol; one at a time

        reward_risk_ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        available_budget = equity - self.storage.committed_margin()

        try:
            plan = plan_position(
                equity=equity,
                risk_pct=self.risk_pct,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                direction=signal.direction,
                reward_risk_ratio=reward_risk_ratio,
                available_budget=available_budget,
                max_leverage=self.max_leverage,
            )
        except ValueError as exc:
            logger.info("Skipping %s/%s: %s", signal.symbol, signal.strategy_tag, exc)
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
        if plan.position_size < specs["min_size"] or plan.notional_value < specs["min_notional"]:
            logger.info(
                "Skipping %s/%s: size %.8f (notional %.2f) below exchange minimum %.8f / %.2f",
                signal.symbol,
                signal.strategy_tag,
                plan.position_size,
                plan.notional_value,
                specs["min_size"],
                specs["min_notional"],
            )
            return

        partial_size = plan.position_size * PARTIAL_TAKE_FRACTION
        remainder_size = plan.position_size - partial_size
        remainder_target = _reward_target(signal.entry_price, signal.stop_loss, signal.direction, REMAINDER_TARGET_RATIO)

        # Prices and sizes at the precision the exchange actually quotes, so
        # every level in the alert is a value that can be entered as an order.
        def px(value: float) -> str:
            return f"{value:.{specs['price_place']}f}"

        def qty(value: float) -> str:
            return f"{value:.{specs['volume_place']}f}"

        def usd(value: float) -> str:
            return f"${value:,.0f}" if abs(value) >= 10 else f"${value:,.2f}"

        # The headline Entry is where the market is right now, so the alert can
        # be read against the chart at a glance. The plan itself stays measured
        # from signal.entry_price — for a strategy entering on a limit those are
        # different prices, and it is the plan's one that sets risk and size.
        try:
            market_price = self.bitget.get_mark_price(signal.symbol)
        except Exception:
            logger.exception("Could not read mark price for %s; showing the planned entry", signal.symbol)
            market_price = signal.entry_price

        lines = [
            f"Signal: {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})",
            f"Analysis timeframe: {', '.join(timeframes)}",
            f"Entry: {px(market_price)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}",
            f"Size: {usd(plan.notional_value)} ({qty(plan.position_size)} @ {plan.leverage:.1f}x)",
        ]

        # A split entry is two orders at two prices, so the alert states how
        # much goes into each rather than leaving the arithmetic to be done at
        # the moment of placing them. Each leg's dollar value is its own
        # quantity at its own price, not a share of the total notional.
        if signal.limit_entry is not None:
            market_size = plan.position_size * signal.market_fraction
            limit_size = plan.position_size - market_size
            note = f" ({signal.limit_note})" if signal.limit_note else ""
            lines.append(
                f"Enter: {usd(market_size * market_price)} ({qty(market_size)}) at market {px(market_price)}"
                f"  ·  {usd(limit_size * signal.limit_entry)} ({qty(limit_size)}) limit {px(signal.limit_entry)}{note}"
            )

        # Only a real two-tier exit is worth describing. When the strategy's own
        # reward:risk already equals the remainder ratio both tiers land on the
        # same price, so the partial and the stop-to-breakeven step do nothing.
        instructions = []
        if abs(plan.take_profit - remainder_target) > _PRICE_EPSILON * max(abs(remainder_target), 1.0):
            instructions.append(
                f"Partial: close {qty(partial_size)} ({PARTIAL_TAKE_FRACTION:.0%}) at {px(plan.take_profit)}, "
                f"move stop to {px(signal.entry_price)}, then close the remaining "
                f"{qty(remainder_size)} at {px(remainder_target)} (1:{REMAINDER_TARGET_RATIO:g})."
            )
        if instructions:
            lines.append(" ".join(instructions))

        text = "\n".join(lines)

        def on_approve() -> None:
            trade_id = self.storage.create_pending(
                symbol=signal.symbol,
                direction=signal.direction,
                proposed_stop=signal.stop_loss,
                proposed_target=plan.take_profit,
                strategy_tag=signal.strategy_tag,
            )
            self.executor.execute(signal.symbol, signal.direction, plan.position_size, signal.entry_price)
            asyncio.create_task(self._confirm_and_track(trade_id, signal))

        await self.bot.send_signal(text, on_approve)

    async def _confirm_and_track(self, trade_id: int, signal: Signal) -> None:
        position = await wait_for_signal_position(self.bitget, signal.symbol, signal.direction)
        if position is None:
            self.storage.cancel_pending(trade_id)
            await self.bot.send_message(
                f"No position detected for trade #{trade_id} ({signal.symbol} {signal.direction}) "
                f"within the timeout — marked cancelled, and {signal.symbol} is free to signal again."
            )
            return

        stop, target = self._safe_stop_target(signal.symbol, signal.direction, position)
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

        await track_position(
            self.storage,
            self.bitget,
            trade_id,
            signal.symbol,
            signal.direction,
            on_close=self._on_trade_closed,
            on_partial=self._on_partial_exit,
        )

    def _safe_stop_target(self, symbol: str, direction: str, position: dict):
        try:
            return self.bitget.get_stop_target(symbol, direction)
        except Exception:
            logger.exception("Could not read stop/target for %s; using position presets", symbol)
            return position["stop_loss"], position["take_profit"]

    def _on_trade_closed(self, trade_id: int, price: float) -> None:
        message = format_close_message(self.storage.get_trade(trade_id))
        asyncio.create_task(self.bot.send_message(message))

    def _on_partial_exit(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        message = format_partial_message(self.storage.get_trade(trade_id), closed_size, realized_pnl)
        asyncio.create_task(self.bot.send_message(message))
