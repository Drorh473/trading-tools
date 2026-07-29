"""Seam for order execution.

Today this is the "manual watch" mode: the user executes the trade
themselves on Bitget, so there's nothing to do here but acknowledge. Later,
this becomes the place that calls Bitget's order-placement endpoint for
auto-execution — nothing else in the codebase (scanner, tracker, storage)
needs to change when that switch happens; only which Executor gets
constructed in notifier/main.py changes.
"""

from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def execute(self, symbol: str, direction: str, size: float, entry_price: float) -> None:
        ...


class ManualExecutor(Executor):
    """The user executes manually on Bitget; the tracker does the watching."""

    def execute(self, symbol: str, direction: str, size: float, entry_price: float) -> None:
        return None
