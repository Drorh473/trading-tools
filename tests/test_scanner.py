import asyncio

import pytest

from core.storage import Storage
from execution.executor import ManualExecutor
from notifier.scanner import Scanner, seconds_until_next_close
from notifier.strategies.base import Signal, Strategy


class AlwaysFireStrategy(Strategy):
    """Trivial test-only strategy: always signals long at the last close."""

    tag = "always_fire"
    timeframe = "1H"

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


def make_position(direction="long", entry_price=100.0, size=20.0, stop=95.0, target=None):
    return {
        "symbol": "BTCUSDT",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "stop_loss": stop,
        "take_profit": target,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "leverage": 1.0,
        "raw": {},
    }


class FakeBitget:
    def __init__(self, position=None, failing_symbols=(), equity=10_000.0, min_size=0.0, min_notional=0.0):
        self._position = position
        self._failing_symbols = set(failing_symbols)
        self._equity = equity
        self._specs = {"min_size": min_size, "min_notional": min_notional}

    def get_account_equity(self):
        return self._equity

    def get_candles(self, symbol, granularity="1H", limit=100):
        if symbol in self._failing_symbols:
            raise RuntimeError(f"simulated API failure for {symbol}")
        return [
            ["1000", "100", "101", "99", "100", "1", "1"],
            ["2000", "100", "101", "99", "100", "1", "1"],
            ["3000", "100", "101", "99", "100", "1", "1"],
        ]

    def get_position(self, symbol, direction=None):
        return self._position

    def get_stop_target(self, symbol, direction):
        return 95.0, 110.0

    def get_contract_specs(self, symbol):
        return self._specs

    def find_closed_position(self, symbol, direction):
        return None

    def get_mark_price(self, symbol):
        return 100.0


class FakeBot:
    def __init__(self):
        self.sent = []
        self.messages = []

    async def send_signal(self, text, on_approve, on_reject=None):
        self.sent.append(text)
        on_approve()  # simulate the user approving immediately

    async def send_message(self, text):
        self.messages.append(text)


def build_scanner(storage, bitget, bot, watchlist=("BTCUSDT",), **kwargs):
    return Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=list(watchlist),
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
        **kwargs,
    )


def test_seconds_until_next_close_aligns_to_period():
    # 100s past the hour -> 3500s left, plus the settle delay
    assert seconds_until_next_close("1H", now=3600 * 5 + 100) == pytest.approx(3500 + 30)
    assert seconds_until_next_close("15m", now=900 * 3 + 60) == pytest.approx(840 + 30)


def test_strategies_grouped_by_timeframe():
    class FourHour(AlwaysFireStrategy):
        tag = "four_hour"
        timeframe = "4H"

    scanner = Scanner(
        bitget=None, bot=None, storage=None, executor=None,
        watchlist=[], strategies=[AlwaysFireStrategy(), FourHour()],
    )
    grouped = scanner.strategies_by_timeframe()
    assert set(grouped) == {"1H", "4H"}


async def test_scanner_dispatches_signal_and_confirms_entry(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position())
    bot = FakeBot()
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick("1H")

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]
    assert "Partial:" in bot.sent[0]  # partial-take guidance included
    assert len(storage.pending_trades()) == 1

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    open_trades = storage.open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].סימבול == "BTCUSDT"
    assert open_trades[0].תגית_אסטרטגיה == "always_fire"
    assert open_trades[0].סטופ_לוס_בפועל == 95.0  # read via get_stop_target, not the position preset

    # same closed candle on the next scan -> deduped
    await scanner.tick("1H")
    assert len(bot.sent) == 1


async def test_scanner_skips_symbol_already_tracked(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="BTCUSDT", direction="long")
    scanner = build_scanner(storage, FakeBitget(position=make_position()), FakeBot())

    await scanner.tick("1H")
    assert scanner.bot.sent == []


async def test_cancelled_trade_frees_the_symbol(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.cancel_pending(trade_id)

    scanner = build_scanner(storage, FakeBitget(position=make_position()), FakeBot())
    await scanner.tick("1H")

    assert len(scanner.bot.sent) == 1  # symbol is signalable again


async def test_scanner_skips_failing_symbol_and_continues(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position(), failing_symbols={"BADUSDT"})
    bot = FakeBot()
    scanner = build_scanner(storage, bitget, bot, watchlist=("BADUSDT", "BTCUSDT"))

    await scanner.tick("1H")

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]


async def test_scanner_skips_scan_when_equity_unavailable(tmp_path):
    class NoEquity(FakeBitget):
        def get_account_equity(self):
            raise RuntimeError("bitget down")

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, NoEquity(position=make_position()), bot)

    await scanner.tick("1H")

    assert bot.sent == []  # no sizing off a guessed equity
    assert storage.read_all() == []


async def test_scanner_enforces_total_risk_cap(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    # existing open trade already risking 6% of a 10k account
    existing = storage.create_pending(symbol="ETHUSDT", direction="long")
    storage.confirm_entry(existing, entry_price=100, position_size=60, actual_stop=90, actual_target=130, leverage=1.0)
    assert storage.total_open_risk() == pytest.approx(600)

    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick("1H")

    assert bot.sent == []  # at the 6% cap, new signals are skipped


async def test_scanner_skips_below_exchange_minimum(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(
        storage, FakeBitget(position=make_position(), min_notional=1_000_000), bot
    )

    await scanner.tick("1H")

    assert bot.sent == []
