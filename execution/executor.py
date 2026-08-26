"""Order execution.

Three implementations, all satisfying the same interface so the scanner never
learns which one it holds:

  ManualExecutor  - does nothing; the user places orders themselves.
  DryRunExecutor  - builds the exact payload and reports it without sending.
  LiveExecutor    - actually places orders on Bitget.

DryRunExecutor exists because the failure modes here are binary rather than
degraded, and unit tests cannot catch them. The account is in hedge mode,
where a wrong `side`/`tradeSide` pairing opens the *opposite* direction
instead of erroring; size is in base coins, where a unit error is a factor of
a hundred; and leverage is per symbol and side, so a stale value from an
earlier trade silently decides how much margin this one consumes. Reading a
real payload against a trade you would have placed by hand checks all three
at once, for nothing.

A trade is placed as its legs plus a stop, never as a bare entry. The stop
rides on the order itself (`presetStopLossPrice`) rather than following as a
second call, so there is no window - not even the width of one API round trip
- in which a filled position sits unprotected on 10-20x leverage. The partial
take-profit cannot be attached that way, since a preset carries one target for
the whole position and the partial closes only part of it; the scanner places
it separately once a position is actually confirmed, sized to what really
filled.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.bitget_client import BitgetClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderLeg:
    size: float
    order_type: str  # "market" or "limit"
    price: float | None = None  # required for a limit
    note: str = ""  # e.g. "61.8% Fib", for the dry-run report


@dataclass
class TradeOrder:
    """Everything needed to open one trade, independent of who places it."""

    symbol: str
    direction: str
    legs: list[OrderLeg]
    stop_loss: float
    leverage: float
    strategy_tag: str = ""
    client_prefix: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def client_oid(self, index: int) -> str:
        """Stable per leg, so a retry after an ambiguous failure is rejected
        by the exchange as a duplicate rather than doubling the position."""
        return f"{self.client_prefix}-{index}"


@dataclass
class ExecutionResult:
    placed: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Executor(ABC):
    @abstractmethod
    def execute(self, order: TradeOrder) -> ExecutionResult:
        ...

    def handles_live(self, strategy_tag: str) -> bool:
        """Whether orders for this strategy really reach the exchange.

        The scanner places the partial take-profit and cancels leftover legs
        by calling the exchange directly - those cannot go through execute(),
        which only opens a trade. Without this, a strategy routed to a dry run
        would still have real exit orders placed on its behalf, which is a dry
        run in name only.
        """
        return False

    def manages_exits(self, strategy_tag: str) -> bool:
        """Whether the bot may place REDUCE-ONLY exit orders on a position of
        this strategy's that it is already tracking.

        Deliberately separate from handles_live, and strictly weaker. A
        reduce-only order can only close exposure that already exists - it
        cannot open a trade or increase one - so this can be granted to a
        strategy whose entries stay manual. Strategy 3 is exactly that case:
        Dror places its entry, stop and partial by hand and has kept it
        non-executing on purpose, but wants the runner's take-profit placed
        for him once the partial fills.

        Defaults to handles_live so nothing changes for a strategy that
        already executes, and a dry run stays dry.
        """
        return self.handles_live(strategy_tag)

    def describe(self, order: TradeOrder) -> str:
        """The payload in the terms the alert already uses, for reporting."""
        lines = [
            f"{order.symbol} {order.direction.upper()} @ {order.leverage:.1f}x  stop {order.stop_loss:g}",
        ]
        for i, leg in enumerate(order.legs):
            where = f"limit {leg.price:g}" if leg.order_type == "limit" else "market"
            note = f" ({leg.note})" if leg.note else ""
            lines.append(f"  leg {i + 1}: {leg.size:g} {where}{note}  oid={order.client_oid(i)}")
        return "\n".join(lines)


class ManualExecutor(Executor):
    """The user places orders themselves; the tracker does the watching."""

    def execute(self, order: TradeOrder) -> ExecutionResult:
        return ExecutionResult()


class DryRunExecutor(Executor):
    """Builds and reports the real payload without sending it."""

    def __init__(self, report=None):
        self.report = report
        self.orders: list[TradeOrder] = []

    def execute(self, order: TradeOrder) -> ExecutionResult:
        self.orders.append(order)
        text = f"DRY RUN — would place:\n{self.describe(order)}"
        logger.info("DRY RUN order for %s/%s:\n%s", order.symbol, order.strategy_tag, self.describe(order))
        if self.report:
            self.report(text)
        return ExecutionResult()


class RoutingExecutor(Executor):
    """Sends each strategy's orders to its own executor.

    Strategies graduate to live trading one at a time - Strategy 1 goes live
    once its payloads have been read, while Strategy 2 keeps reporting dry-run
    payloads until it has been checked the same way. Routing lives here rather
    than in the Scanner because `TradeOrder` already carries the tag, so the
    scanning loop never has to know that two modes exist.
    """

    def _for(self, strategy_tag: str) -> Executor:
        return self.by_tag.get(strategy_tag, self.default)

    def __init__(self, by_tag: dict[str, Executor], default: Executor | None = None, exit_managed_tags=()):
        self.by_tag = by_tag
        self.default = default or ManualExecutor()
        # Tags whose EXITS the bot may manage even though their entries are
        # not routed live here - see Executor.manages_exits.
        self.exit_managed_tags = set(exit_managed_tags)

    def handles_live(self, strategy_tag: str) -> bool:
        return self._for(strategy_tag).handles_live(strategy_tag)

    def manages_exits(self, strategy_tag: str) -> bool:
        if strategy_tag in self.exit_managed_tags:
            return True
        return self._for(strategy_tag).manages_exits(strategy_tag)

    def execute(self, order: TradeOrder) -> ExecutionResult:
        return self._for(order.strategy_tag).execute(order)


class LiveExecutor(Executor):
    """Places real orders on the real account.

    Fail-safe by construction: legs are placed in sequence and the first
    failure stops the rest. Nothing is retried, because the dangerous case is
    not a clean rejection but an ambiguous one - a timeout after Bitget
    received the order - where retrying is how a position silently doubles.
    The caller reconciles against the account and alerts instead.
    """

    def __init__(self, bitget: BitgetClient):
        self.bitget = bitget
        # (symbol, direction) -> the leverage this process last successfully
        # set. Skips a full network round trip that used to run before EVERY
        # order regardless of whether the account's leverage on that side was
        # already correct - the common case, since a symbol/direction rarely
        # changes leverage between one trade and the next.
        #
        # Per-process only, on purpose: a leverage change made outside the
        # bot (the Bitget app, by hand) between trades would go unnoticed
        # until a restart clears this cache. Accepted cost of not
        # re-verifying it over the network on every single order.
        self._last_leverage: dict[tuple[str, str], float] = {}

    def handles_live(self, strategy_tag: str) -> bool:
        return True

    def execute(self, order: TradeOrder) -> ExecutionResult:
        result = ExecutionResult()

        leverage_key = (order.symbol, order.direction)
        if self._last_leverage.get(leverage_key) != order.leverage:
            try:
                self.bitget.set_leverage(order.symbol, order.direction, order.leverage)
            except Exception as exc:
                # Placing without this would size margin off whatever leverage a
                # previous trade on this symbol happened to leave behind.
                logger.exception("Could not set leverage for %s", order.symbol)
                result.error = f"leverage not set: {exc}"
                return result
            self._last_leverage[leverage_key] = order.leverage

        for index, leg in enumerate(order.legs):
            try:
                placed = self.bitget.place_order(
                    symbol=order.symbol,
                    direction=order.direction,
                    size=leg.size,
                    order_type=leg.order_type,
                    price=leg.price,
                    stop_loss=order.stop_loss,
                    client_oid=order.client_oid(index),
                )
                result.placed.append(placed or {})
            except Exception as exc:
                logger.exception("Order leg %d failed for %s", index + 1, order.symbol)
                result.error = f"leg {index + 1} of {len(order.legs)} failed: {exc}"
                return result

        return result
