"""Main scanning loop: pulls bars for each watchlist symbol, runs the active
strategies against them, and — on a fresh signal — computes position sizing
and dispatches an Approve/Reject alert to Telegram. Approval doesn't log the
trade immediately: it waits to see a matching position actually appear on
Bitget first (see execution.tracker.wait_for_signal_position), same as the
`/add` flow and the same account-based detection the future auto-execution
phase will need anyway.
"""

import asyncio
import logging

import pandas as pd

logger = logging.getLogger(__name__)

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.executor import Executor
from execution.tracker import format_close_message, track_position, wait_for_signal_position
from notifier.risk_sizing import DEFAULT_MAX_LEVERAGE, DEFAULT_REWARD_RISK_RATIO, plan_position
from notifier.strategies.base import Signal, Strategy


def bars_dataframe(candles: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df


class Scanner:
    def __init__(
        self,
        bitget: BitgetClient,
        bot,
        storage: Storage,
        executor: Executor,
        watchlist: list[str],
        strategies: list[Strategy],
        equity: float,
        risk_pct: float = 0.01,
        reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
        max_leverage: float = DEFAULT_MAX_LEVERAGE,
        granularity: str = "5m",
        poll_interval: float = 60.0,
        candle_limit: int = 250,
    ):
        self.bitget = bitget
        self.bot = bot
        self.storage = storage
        self.executor = executor
        self.watchlist = watchlist
        self.strategies = strategies
        self.equity = equity
        self.risk_pct = risk_pct
        self.reward_risk_ratio = reward_risk_ratio
        self.max_leverage = max_leverage
        self.granularity = granularity
        self.poll_interval = poll_interval
        self.candle_limit = candle_limit
        self._seen: set[tuple] = set()

    async def run_forever(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self.poll_interval)

    async def tick(self) -> None:
        for symbol in self.watchlist:
            try:
                candles = self.bitget.get_candles(symbol, granularity=self.granularity, limit=self.candle_limit)
                bars = bars_dataframe(candles)
            except Exception:
                logger.exception("Skipping %s this tick: failed to fetch/parse candles", symbol)
                continue

            last_ts = str(bars["ts"].iloc[-1])

            for strategy in self.strategies:
                try:
                    signal = strategy.evaluate(symbol, bars)
                except Exception:
                    logger.exception("Skipping %s/%s this tick: strategy raised", symbol, strategy.tag)
                    continue

                if signal is None:
                    continue

                dedupe_key = (signal.symbol, signal.strategy_tag, last_ts)
                if dedupe_key in self._seen:
                    continue
                self._seen.add(dedupe_key)

                await self._dispatch(signal)

    async def _dispatch(self, signal: Signal) -> None:
        if self.storage.has_open_or_pending(signal.symbol):
            return  # already tracking a trade on this symbol; one at a time

        reward_risk_ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        available_budget = self.equity - self.storage.committed_margin()
        plan = plan_position(
            equity=self.equity,
            risk_pct=self.risk_pct,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            direction=signal.direction,
            reward_risk_ratio=reward_risk_ratio,
            available_budget=available_budget,
            max_leverage=self.max_leverage,
        )

        text = (
            f"Signal: {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})\n"
            f"Entry: {signal.entry_price:.2f}  Stop: {signal.stop_loss:.2f}  Target: {plan.take_profit:.2f}\n"
            f"Size: {plan.position_size:.2f}  Notional: {plan.notional_value:.2f}  "
            f"Margin needed ({plan.leverage:.2f}x): {plan.required_margin:.2f}\n"
            f"Risk: {plan.risk_amount:.2f}\n"
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
            asyncio.create_task(self._confirm_and_track(trade_id, signal, plan))

        await self.bot.send_signal(text, on_approve)

    async def _confirm_and_track(self, trade_id: int, signal: Signal, plan) -> None:
        position = await wait_for_signal_position(
            self.bitget, signal.symbol, signal.direction, signal.entry_price, plan.position_size
        )
        if position is None:
            await self.bot.send_message(
                f"No position detected for trade #{trade_id} ({signal.symbol} {signal.direction}) "
                f"within the timeout window — still shown as pending."
            )
            return

        self.storage.confirm_entry(
            trade_id,
            position["entry_price"],
            position["size"],
            position["stop_loss"],
            position["take_profit"],
            leverage=position["leverage"],
        )
        await track_position(self.storage, self.bitget, trade_id, signal.symbol, on_close=self._on_trade_closed)

    def _on_trade_closed(self, trade_id: int, price: float) -> None:
        message = format_close_message(self.storage.get_trade(trade_id))
        asyncio.create_task(self.bot.send_message(message))
