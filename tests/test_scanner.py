import asyncio

import pytest

from core.storage import Storage
from execution.executor import ManualExecutor
from notifier.scanner import Scanner
from notifier.strategies.base import Signal, Strategy


class AlwaysFireStrategy(Strategy):
    """Trivial test-only strategy: always signals long at the last close."""

    tag = "always_fire"

    def evaluate(self, symbol, bars):
        last_close = bars["close"].iloc[-1]
        return Signal(
            symbol=symbol,
            direction="long",
            entry_price=last_close,
            stop_loss=last_close * 0.95,
            strategy_tag=self.tag,
            reason="test signal",
        )


class FakeBitget:
    def __init__(self, position=None, failing_symbols=()):
        self._position = position
        self._failing_symbols = set(failing_symbols)

    def get_candles(self, symbol, granularity="5m", limit=100):
        if symbol in self._failing_symbols:
            raise RuntimeError(f"simulated API failure for {symbol}")
        return [
            ["1000", "100", "101", "99", "100", "1", "1"],
            ["2000", "100", "101", "99", "100", "1", "1"],
            ["3000", "100", "101", "99", "100", "1", "1"],
        ]

    def get_position(self, symbol):
        return self._position


def make_position(direction="long", entry_price=100.0, size=20.0, stop=95.0, target=None):
    return {
        "symbol": "BTCUSDT",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "stop_loss": stop,
        "take_profit": target,
        "unrealized_pnl": 0.0,
        "leverage": 1.0,
        "raw": {},
    }


class FakeBot:
    def __init__(self):
        self.sent = []
        self.messages = []

    async def send_signal(self, text, on_approve, on_reject=None):
        self.sent.append(text)
        on_approve()  # simulate the user approving immediately

    async def send_message(self, text):
        self.messages.append(text)


async def test_scanner_dispatches_signal_and_confirms_entry(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    # matches the AlwaysFireStrategy signal (entry=100, stop=95) within tolerance
    bitget = FakeBitget(position=make_position(entry_price=100.0, size=20.0, stop=95.0, target=110.0))
    bot = FakeBot()

    scanner = Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        equity=10_000,
        risk_pct=0.01,
    )

    await scanner.tick()

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]

    # on_approve only creates a pending row synchronously
    assert len(storage.pending_trades()) == 1

    # let the background confirmation task run its first (non-blocking) step
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    open_trades = storage.open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].סימבול == "BTCUSDT"
    assert open_trades[0].תגית_אסטרטגיה == "always_fire"
    assert open_trades[0].מחיר_כניסה == 100.0
    assert open_trades[0].גודל_פוזיציה == 20.0

    # Same last-candle timestamp on the next tick should be deduped, not resent.
    await scanner.tick()
    assert len(bot.sent) == 1


async def test_scanner_skips_symbol_already_tracked(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="BTCUSDT", direction="long")
    bitget = FakeBitget(position=make_position())
    bot = FakeBot()

    scanner = Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        equity=10_000,
        risk_pct=0.01,
    )

    await scanner.tick()

    assert bot.sent == []  # already tracking BTCUSDT; signal should be skipped


async def test_scanner_skips_failing_symbol_and_continues(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position(), failing_symbols={"BADUSDT"})
    bot = FakeBot()

    scanner = Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BADUSDT", "BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        equity=10_000,
        risk_pct=0.01,
    )

    await scanner.tick()  # must not raise, and must still process BTCUSDT after BADUSDT fails

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]
