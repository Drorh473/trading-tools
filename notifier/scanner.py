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
            bars_by_tf: dict[str, pd.DataFrame] = {}
            for tf in timeframes:
                try:
                    candles = self.bitget.get_candles(symbol, granularity=tf, limit=self.candle_limit)
                    bars_by_tf[tf] = bars_dataframe(candles)
                except Exception:
                    logger.exception("Skipping %s this scan: failed to fetch/parse %s candles", symbol, tf)
                    bars_by_tf = None
                    break

            if not bars_by_tf or any(b.empty for b in bars_by_tf.values()):
                continue

            for strategy in self.strategies:
                strategy_bars = {tf: bars_by_tf[tf] for tf in strategy.timeframes if tf in bars_by_tf}
                if len(strategy_bars) < len(strategy.timeframes):
                    continue  # one of this strategy's timeframes failed to fetch this scan

                try:
                    signal = strategy.evaluate(symbol, strategy_bars)
                except Exception:
                    logger.exception("Skipping %s/%s this scan: strategy raised", symbol, strategy.tag)
                    continue

                if signal is None:
                    continue

                dedupe_key = (
                    signal.symbol,
                    signal.strategy_tag,
                    tuple(str(strategy_bars[tf]["ts"].iloc[-1]) for tf in strategy.timeframes),
                )
                if dedupe_key in self._seen:
                    continue
                self._seen.add(dedupe_key)
                self._prune_seen()

                await self._dispatch(signal, equity)

    def _prune_seen(self, max_entries: int = 5000) -> None:
        """Bounded so a long-running process can't leak memory on a big watchlist."""
        if len(self._seen) > max_entries:
            self._seen = set(list(self._seen)[-max_entries // 2 :])

    async def _dispatch(self, signal: Signal, equity: float) -> None:
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
        text = (
            f"Signal: {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})\n"
            f"Entry: {signal.entry_price:.2f}  Stop: {signal.stop_loss:.2f}  Target: {plan.take_profit:.2f}\n"
            f"Size: {plan.position_size:.2f}  Notional: {plan.notional_value:.2f}  "
            f"Margin needed ({plan.leverage:.2f}x): {plan.required_margin:.2f}\n"
            f"Risk: {plan.risk_amount:.2f} ({self.risk_pct:.1%} of {equity:.2f})\n"
            f"Partial: close {partial_size:.2f} ({PARTIAL_TAKE_FRACTION:.0%}) at {plan.take_profit:.2f}, "
            f"move stop to entry {signal.entry_price:.2f}, then close the remaining "
            f"{remainder_size:.2f} at {remainder_target:.2f} (1:{REMAINDER_TARGET_RATIO:g})\n"
            f"{signal.reason}"
        )

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
