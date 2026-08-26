import asyncio
import logging
import math
import sqlite3

import pytest

from core.storage import Storage
from execution.executor import ManualExecutor
from notifier import scanner
from notifier.scanner import (
    CONFLUENCE_TIMEFRAMES,
    RUNNER_LEVEL_TIMEFRAME,
    SIGNAL_EXPIRY_CEILING,
    SIGNAL_EXPIRY_FLOOR,
    Scanner,
    seconds_until_next_close,
    signal_expiry_seconds,
)
from notifier.strategies.base import FillGuard, Signal, Strategy


class AlwaysFireStrategy(Strategy):
    """Trivial test-only strategy: always signals long at the last close."""

    tag = "always_fire"
    timeframes = ["1H"]

    def evaluate(self, symbol, bars_by_timeframe):
        last_close = bars_by_timeframe["1H"]["close"].iloc[-1]
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
    def __init__(
        self,
        position=None,
        failing_symbols=(),
        equity=10_000.0,
        min_size=0.0,
        min_notional=0.0,
        volume_place=2,
        price_place=2,
        is_rwa=False,
        max_leverage=125.0,
        account_positions=(),
        open_orders=(),
    ):
        self.account_positions = list(account_positions)
        self._open_orders = list(open_orders)
        self._position = position
        self._failing_symbols = set(failing_symbols)
        self._equity = equity
        self._specs = {
            "min_size": min_size,
            "min_notional": min_notional,
            "price_place": price_place,
            "volume_place": volume_place,
            "is_rwa": is_rwa,
            # Bitget's own per-symbol ceiling. Effectively uncapped by default
            # so existing tests keep sizing the way they always did.
            "max_leverage": max_leverage,
        }

    def get_account_equity(self):
        return self._equity

    def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
        if symbol in self._failing_symbols:
            raise RuntimeError(f"simulated API failure for {symbol}")
        candles = [
            ["1000", "100", "101", "99", "100", "1", "1"],
            ["2000", "100", "101", "99", "100", "1", "1"],
            ["3000", "100", "101", "99", "100", "1", "1"],
            ["4000", "100", "101", "99", "100", "1", "1"],
        ]
        return candles[:-1] if closed_only else candles

    def get_position(self, symbol, direction=None):
        return self._position

    def get_positions(self, symbol):
        """What the ACCOUNT holds, independently of the trades DB. Empty by
        default: most tests are about a symbol with nothing on it, and
        `_position` stands for the position a just-approved signal produces
        rather than one that predates it."""
        return [p for p in self.account_positions if p.get("symbol") == symbol]

    def get_all_positions(self):
        return list(self.account_positions)

    def get_stop_target(self, symbol, direction):
        return 95.0, 110.0

    def get_contract_specs(self, symbol):
        return self._specs

    def round_price(self, symbol, price):
        return round(price, self.get_contract_specs(symbol)["price_place"])

    def round_size(self, symbol, size):
        specs = self.get_contract_specs(symbol)
        step = 10 ** -specs["volume_place"]
        size = math.floor(size / step) * step
        min_size = specs["min_size"]
        if min_size >= 1:
            size = math.floor(size / min_size) * min_size
        return round(size, specs["volume_place"])

    def find_closed_position(self, symbol, direction):
        return None

    def get_open_orders(self, symbol=None):
        return list(self._open_orders)

    def cancel_order(self, symbol, order_id=None, **kw):
        return {}

    def get_mark_price(self, symbol):
        return 100.0

    def place_tpsl_order(self, *a, **kw):
        return {}

    def get_plan_orders(self, symbol, direction):
        return []

    def cancel_plan_order(self, symbol, plan_type, order_id=None, **kw):
        return {}


class FakeBot:
    def __init__(self):
        self.sent = []
        self.messages = []
        self.expiry_kwargs = []

    async def send_signal(self, text, on_approve, on_reject=None, **expiry_kwargs):
        self.sent.append(text)
        self.expiry_kwargs.append(expiry_kwargs)
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


async def test_a_scan_that_raises_does_not_end_the_loop(tmp_path, monkeypatch):
    """2026-08-18: the service restarted every 14-16 minutes for hours because
    an exception escaping tick() (telegram.error.TimedOut, in practice) took
    down the whole process - run_forever gathers _scan_loop with the other
    loops, so nothing else was catching it, and systemd's Restart=always
    brought it back with every piece of in-memory state gone (including the
    dedupe set and alert throttle, which is what produced a duplicate
    MMTUSDT alert on top of the crash loop itself).

    _scan_loop's own try/except around tick() is the fix; this is what
    proves it still holds rather than trusting the comment above it.
    """
    import notifier.scanner as scanner_module

    async def fast_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr(scanner_module.asyncio, "sleep", fast_sleep)

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    calls = []

    class _StopTheLoop(BaseException):
        """Raised on the SECOND tick, so reaching it proves the first
        failure did not end the loop. Deliberately a BaseException, not an
        Exception - _scan_loop's own guard is `except Exception:`, so an
        Exception raised here would just get logged and swallowed like the
        first one, spinning the loop forever instead of ending the test.
        BaseException is the one thing that clause is NOT supposed to
        catch, and it doesn't - which is itself worth confirming: too
        broad a catch here would silently swallow this too."""

    async def flaky_tick():
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("simulated scan failure")
        raise _StopTheLoop()

    scanner.tick = flaky_tick

    with pytest.raises(_StopTheLoop):
        await scanner._scan_loop({"15m"})

    assert len(calls) == 2, "a failed scan must not stop the next cycle from running"


def test_seconds_until_next_close_aligns_to_period():
    # 100s past the hour -> 3500s left, plus the settle delay
    assert seconds_until_next_close("1H", now=3600 * 5 + 100) == pytest.approx(3500 + 30)
    assert seconds_until_next_close("15m", now=900 * 3 + 60) == pytest.approx(840 + 30)


def test_signal_expiry_tracks_the_uncapped_middle_of_the_range():
    # 1H, fired 50 minutes into the candle -> 10 minutes to next close, well
    # inside [60s, 1800s], so the raw next-close value should pass through
    # untouched rather than getting clamped to either edge.
    raw = seconds_until_next_close("1H", now=3000)
    assert SIGNAL_EXPIRY_FLOOR < raw < SIGNAL_EXPIRY_CEILING
    assert signal_expiry_seconds("1H", now=3000) == pytest.approx(raw)


def test_signal_expiry_floors_a_signal_fired_late_in_its_candle():
    # 5s before a 1m candle closes -> next close is ~5s + settle delay away,
    # far under the 60s floor. Without the floor a signal fired here would
    # give almost no time to read the alert, let alone tap a button.
    assert signal_expiry_seconds("1m", now=55) == pytest.approx(SIGNAL_EXPIRY_FLOOR)


def test_signal_expiry_caps_a_slow_timeframe():
    # 1D fired right after its own candle opens -> next close is ~24h away.
    # Without the cap a quiet 1D signal could sit live for most of a day.
    assert signal_expiry_seconds("1D", now=10) == pytest.approx(SIGNAL_EXPIRY_CEILING)


def test_required_timeframes_is_union_across_strategies():
    class Confluence(AlwaysFireStrategy):
        tag = "confluence"
        timeframes = ["1H", "15m"]

    scanner = Scanner(
        bitget=None, bot=None, storage=None, executor=None,
        watchlist=[], strategies=[AlwaysFireStrategy(), Confluence()],
    )
    # The confluence timeframes are always fetched on top of whatever the
    # strategies ask for, since pattern confirmation is read independently of
    # what any individual strategy looks at. Asserted against the constant
    # rather than a literal set, so adding a pattern timeframe (1D was added
    # after the flag-pole rework) does not read as a regression here.
    assert scanner.required_timeframes() == {"1H", "15m"} | set(CONFLUENCE_TIMEFRAMES)
    assert "1D" in scanner.required_timeframes()  # daily patterns are scanned


async def test_scanner_dispatches_signal_and_confirms_entry(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position())
    bot = FakeBot()
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]
    # AlwaysFireStrategy takes the scanner-wide 1:3, which is also the remainder
    # ratio, so both exit tiers land on 115.00. Describing that as "take half
    # here, move the stop, then take the rest there" describes steps that cannot
    # happen, so the partial guidance is omitted entirely.
    assert "Partial:" not in bot.sent[0]
    assert "115.00" in bot.sent[0]  # still shown as the target
    assert len(storage.pending_trades()) == 1

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    open_trades = storage.open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].סימבול == "BTCUSDT"
    assert open_trades[0].תגית_אסטרטגיה == "always_fire"
    assert open_trades[0].סטופ_לוס_בפועל == 95.0  # read via get_stop_target, not the position preset

    # same closed candle on the next scan -> deduped
    await scanner.tick()
    assert len(bot.sent) == 1


async def test_signal_logged_and_linked_to_its_trade_on_approve(tmp_path):
    # The trades table only ever gains a row once a signal is approved AND
    # confirmed on Bitget - the signal log exists precisely so an approved
    # signal is still traceable back to what fired, and vice versa.
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position())
    bot = FakeBot()
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].symbol == "BTCUSDT"
    assert signals[0].decision == "approved"
    assert signals[0].trade_id == storage.pending_trades()[0].מספר_עסקה


async def test_signal_logged_as_rejected_and_no_trade_created(tmp_path):
    class RejectingBot(FakeBot):
        async def send_signal(self, text, on_approve, on_reject=None, **_expiry_kwargs):
            self.sent.append(text)
            on_reject()

    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position())
    bot = RejectingBot()
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "rejected"
    assert signals[0].trade_id is None
    assert len(storage.pending_trades()) == 0


async def test_scanner_skips_symbol_already_tracked(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="BTCUSDT", direction="long")
    scanner = build_scanner(storage, FakeBitget(position=make_position()), FakeBot())

    await scanner.tick()
    assert scanner.bot.sent == []


async def test_cancelled_trade_frees_the_symbol(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.cancel_pending(trade_id)

    scanner = build_scanner(storage, FakeBitget(position=make_position()), FakeBot())
    await scanner.tick()

    assert len(scanner.bot.sent) == 1  # symbol is signalable again


async def test_scanner_skips_failing_symbol_and_continues(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(position=make_position(), failing_symbols={"BADUSDT"})
    bot = FakeBot()
    scanner = build_scanner(storage, bitget, bot, watchlist=("BADUSDT", "BTCUSDT"))

    await scanner.tick()

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]


async def test_scanner_skips_scan_when_equity_unavailable(tmp_path):
    class NoEquity(FakeBitget):
        def get_account_equity(self):
            raise RuntimeError("bitget down")

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, NoEquity(position=make_position()), bot)

    await scanner.tick()

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

    await scanner.tick()

    assert bot.sent == []  # at the 6% cap, new signals are skipped


class SwingStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 1 1D / Strategy 2 1D: a signal classified as
    swing by its own actionable timeframe, not merely by mentioning "1D"."""

    tag = "Strategy 1 1D"


class OtherSwingStrategy(AlwaysFireStrategy):
    tag = "Strategy 2 1D"


async def test_swing_slot_cap_blocks_a_third_swing_signal(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    # Two swing slots already occupied - one pending, one open, combined
    # across BOTH swing tags, matching "pending + open together, global
    # across symbols, combined across both swing instances."
    storage.create_pending(symbol="ETHUSDT", direction="long", strategy_tag="Strategy 1 1D")
    open_trade = storage.create_pending(symbol="SOLUSDT", direction="long", strategy_tag="Strategy 2 1D")
    storage.confirm_entry(open_trade, entry_price=100, position_size=1, actual_stop=90, actual_target=130, leverage=1.0)

    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[SwingStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert bot.sent == []  # both swing slots taken; the third is suppressed outright
    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "swing_slots_full"


async def test_swing_slot_cap_does_not_touch_the_day_pool(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="ETHUSDT", direction="long", strategy_tag="Strategy 1 1D")
    other_open = storage.create_pending(symbol="SOLUSDT", direction="long", strategy_tag="Strategy 2 1D")
    storage.confirm_entry(other_open, entry_price=100, position_size=1, actual_stop=90, actual_target=130, leverage=1.0)

    bot = FakeBot()
    # A day-pool strategy (not one of the swing tags) must fire normally even
    # though both swing slots are full - the cap is a swing-pool reservation,
    # not a global "any 1D-mentioning tag" throttle.
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    assert len(bot.sent) == 1


async def test_swing_slot_cap_allows_the_second_slot(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="ETHUSDT", direction="long", strategy_tag="Strategy 1 1D")  # one slot taken

    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[OtherSwingStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert len(bot.sent) == 1  # the second swing slot is still open


async def test_scanner_skips_below_exchange_minimum(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(
        storage, FakeBitget(position=make_position(), min_notional=1_000_000), bot
    )

    await scanner.tick()

    assert bot.sent == []
    # Not just silently dropped: logged under its own decision so the weekly
    # report can still show it, the way rejected/ignored signals already are.
    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "too_small"


async def test_scanner_skips_when_only_the_market_leg_is_below_minimum(tmp_path):
    # ADAUSDT, live: total position $6.35 clears a $5 minimum, but a split
    # entry is placed as two SEPARATE orders and Bitget enforces the minimum
    # on each. The 20% market leg alone was $1.05 and was rejected -
    # "less than the minimum amount 5 USDT" - after the trade was approved.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    # LimitEntryStrategy's fixture position (entry 100, stop 95, equity
    # 10,000, risk 1%) sizes a $2,000 total position - a 20% market leg of
    # $400. min_notional=500 clears the total comfortably but not that leg.
    scanner = Scanner(
        bitget=FakeBitget(position=make_position(), min_notional=500),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert bot.sent == []  # the $400 market leg (20% of $2,000) doesn't clear $500
    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "too_small"
    assert signals[0].symbol == "BTCUSDT"


async def test_scanner_skips_when_a_leg_rounds_to_zero_at_the_exchange_step(tmp_path):
    # AAVEUSDT, live: the 20% market leg was worth $6, comfortably clearing the
    # $5 minimum notional - but AAVEUSDT trades in steps of 0.1, and the leg's
    # raw size (0.06) floors to zero at that step. The notional check above
    # has no idea the exchange rounds by quantity, not by dollars; it passed,
    # the alert went out, the user approved it, and place_order's own
    # "size rounds to zero" guard rejected the order after the fact.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    # entry 100, stop 95, risk 1% of a $150 equity -> a $30 position (0.3
    # units), 20% market leg = 0.06 units. floor(0.06 / 0.1) * 0.1 == 0.
    scanner = Scanner(
        bitget=FakeBitget(
            position=make_position(), equity=150.0, min_size=0.1, min_notional=5, volume_place=1
        ),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert bot.sent == []  # the 0.06-unit market leg floors to zero, unplaceable
    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "too_small"
    assert signals[0].symbol == "BTCUSDT"


async def test_scanner_allows_a_split_entry_whose_market_leg_clears_minimum(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position(), min_notional=5),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert len(bot.sent) == 1


async def test_identical_trade_is_not_alerted_twice(tmp_path):
    # A per-candle dedupe key re-alerted every time the trigger re-fired
    # against an unchanged leg: one stale TSLAUSDT short went out four times
    # over eleven hours with identical levels while price walked past the stop.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()
    await scanner.tick()  # same symbol, same levels, a candle later

    assert len(bot.sent) == 1


async def test_alert_uses_exchange_price_precision(tmp_path):
    # A fixed 2dp collapsed every level of a cheap symbol to the same string:
    # a real DOGEUSDT alert read "Entry 0.07 Stop 0.07 Target 0.07", which is
    # neither actionable nor auditable.
    class PennyBitget(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            rows = [["1000", "0.07", "0.07", "0.07", "0.069442", "1", "1"]] * 4
            return rows[:-1] if closed_only else rows

        def get_contract_specs(self, symbol):
            return {**self._specs, "price_place": 6, "volume_place": 0}

        def get_mark_price(self, symbol):
            return 0.069442  # the Entry line shows live market price now

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, PennyBitget(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "0.069442" in text  # entry at the precision Bitget actually quotes
    levels = next(line for line in text.split("\n") if line.startswith("Entry:"))
    entry, stop, target = (part.split()[-1] for part in levels.split("  ") if part.strip())
    assert len({entry, stop, target}) == 3  # three distinct levels, not "0.07" three times


async def test_the_signal_id_is_folded_into_the_header_not_a_trailing_line(tmp_path):
    """Dror: "make the signal to be in this format is smaller then the
    corrent one" - the id used to be a separate "Signal #N - /add N..." line
    appended after log_signal(), which only ran once the rest of the alert
    was already built. Moved earlier so the header can carry it instead."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert text.startswith("Signal #1  BTCUSDT LONG (always_fire)")
    assert "/add" not in text


async def test_the_analysis_timeframe_line_is_dropped_when_it_only_repeats_the_tag(tmp_path):
    """Every real strategy's tag already names its own timeframe(s)
    (ema_trend_v2, rsi_fib_reversal, volume_run, order_block all build it
    into the tag string), so analysis_timeframes is None - the fallback to
    Strategy.timeframes - whenever the line would say nothing the tag
    doesn't already say."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    assert "Analysis timeframe" not in bot.sent[0]


async def test_the_analysis_timeframe_line_survives_a_real_signal_specific_override(tmp_path):
    """The line still earns its place when it says something the tag does
    not - e.g. Strategy 2's second timeframe, listed only when it was a
    genuine confirmation. analysis_timeframes being explicitly set (not None)
    is exactly that signal."""
    class OverriddenTimeframeStrategy(AlwaysFireStrategy):
        tag = "custom_tf"

        def evaluate(self, symbol, bars_by_timeframe):
            signal = super().evaluate(symbol, bars_by_timeframe)
            signal.analysis_timeframes = ("1H", "4H")
            return signal

    from execution.executor import ManualExecutor

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[OverriddenTimeframeStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    assert "Analysis timeframe: 1H, 4H" in bot.sent[0]


class LimitEntryStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 1: enters part at market and rests the remainder
    as a limit, and its own 1:2 differs from the 1:3 remainder ratio."""

    tag = "limit_entry"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.reward_risk_ratio = 2.0
        signal.limit_entry = signal.entry_price
        signal.limit_note = "61.8% Fib"
        return signal


async def test_alert_plans_from_the_blended_cost_basis_of_both_legs(tmp_path):
    # A split entry holds BOTH legs, so the position's real cost basis is their
    # weighted average - not the limit alone. Planning off the limit understated
    # risk on 91% of replayed Strategy 1 signals (median 29%, worst 60%), which
    # silently pushed a 2%-risk trade past 3% of equity.
    #
    # Here: 20% at the 101.00 market, 80% at the 100.00 limit -> 100.20 basis.
    # Risk is 5.20 (not 5.00), so 1:2 is 110.60 and the 1:3 remainder 115.80,
    # and breakeven - the price the stop moves to - is 100.20.
    class MarketAt101(FakeBitget):
        def get_mark_price(self, symbol):
            return 101.0

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt101(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    assert "Entry: 101.00" in text  # live market, for reading against the chart
    assert "Target: 110.60" in text  # 1:2 measured from the 100.20 blended basis
    assert "Partial:" in text  # tiers differ, so the guidance is real
    assert "at 115.80 (1:3)" in text
    assert "move stop to 100.20" in text  # breakeven IS the blended basis
    # A split entry is two orders at two prices, so each leg is stated in full.
    # Each leg's dollars are its own quantity at its own price, so they do not
    # simply split the total notional.
    assert "Enter: $383 (3.79) at market 101.00" in text
    assert "$1,515 (15.15) limit 100.00 (61.8% Fib)" in text
    # If the resting limit never fills, the market-only fragment needs its own
    # target - and it is a true 1:2 against the risk THAT fill actually takes,
    # not the same dollar distance the blended plan would have paid.
    # Market 101.00 against a 95.00 stop is 6.00 of risk, so 1:2 is 113.00.
    # Carrying the blended plan's 10.40 distance across instead gives 111.40,
    # which is only 1.73R - the trade quietly stops being the 1:2 it was sized
    # and approved as. ZECUSDT trade #13 lost half its intended reward exactly
    # this way.
    assert "If the limit leg never fills: exit the market-only 3.79 at 113.00." in text


async def test_signal_expiry_measures_drift_from_market_but_risk_from_the_plan(tmp_path):
    """The two halves of the expiry calculation come from different prices,
    and conflating them has misfired live in BOTH directions.

    QQQUSDT: drift was measured from plan_entry - the BLENDED cost basis that
    assumes the resting limit leg has already filled. The market hadn't moved
    off 713 while that blend sat at 693.39, so a real $1 of drift read as
    1.25R and expired an alert that was not stale at all. Drift must be
    measured from where the MARKET was: reference_price.

    INJUSDT: the first fix then passed market_price as both, so 1R became
    |market - stop| = 0.115 against a real |plan_entry - stop| of 0.036 -
    three times too large. 1R must come from where the ORDER rests:
    entry_price.

    Here: 20% at the 101.00 market, 80% at the 100.00 limit -> 100.20 blended
    entry, market 101.00.
    """
    class MarketAt101(FakeBitget):
        def get_mark_price(self, symbol):
            return 101.0

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt101(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    kwargs = bot.expiry_kwargs[0]
    assert kwargs["reference_price"] == 101.0  # where the market was: the QQQUSDT half
    assert kwargs["entry_price"] == pytest.approx(100.20)  # where the order rests: the INJUSDT half


class HighConvictionStrategy(AlwaysFireStrategy):
    """A strategy with its own reason to rate a signal above average, with no
    chart pattern involved."""

    tag = "conviction"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.risk_pct_override = 0.02
        signal.extra_notes = ("both timeframes aligned",)
        return signal


async def test_strategy_conviction_raises_risk_like_pattern_confluence(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()), bot=bot, storage=storage,
        executor=ManualExecutor(), watchlist=["BTCUSDT"],
        strategies=[HighConvictionStrategy()], risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    assert "risk 2%" in text  # raised to the same ceiling a pattern would earn
    assert "both timeframes aligned" in text


class PureLimitStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 2's EMA9 entry: the whole position rests on the
    limit, no market fraction at all."""

    tag = "pure_limit"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.limit_entry = signal.entry_price
        signal.limit_note = "EMA9"
        signal.market_fraction = 0.0
        return signal


class RecordingExecutor:
    def __init__(self, fail=False):
        self.orders = []
        self.fail = fail

    def execute(self, order):
        from execution.executor import ExecutionResult

        self.orders.append(order)
        return ExecutionResult(error="boom" if self.fail else None)


def _exec_scanner(tmp_path, executor, tags, **kwargs):
    return Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=kwargs.pop("bot", FakeBot()),
        storage=Storage(str(tmp_path / "trades.db")),
        executor=executor,
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
        auto_execute_tags=tags,
        **kwargs,
    )


async def test_only_whitelisted_strategies_place_orders(tmp_path):
    # Registering a strategy must never be enough to make it spend money.
    executor = RecordingExecutor()
    scanner = _exec_scanner(tmp_path, executor, tags={"some other strategy"})

    await scanner.tick()

    assert executor.orders == []
    assert scanner.auto_executes("always_fire") is False


async def test_whitelisted_strategy_places_the_alert_as_orders(tmp_path):
    executor = RecordingExecutor()
    scanner = _exec_scanner(tmp_path, executor, tags={"always_fire"})

    await scanner.tick()
    # execute() now runs on a worker thread (asyncio.to_thread) so it can
    # never block the event loop - see on_approve. A real sleep, not
    # sleep(0), is what lets the actual OS thread finish and post its result
    # back before the assertion runs.
    await asyncio.sleep(0.05)

    assert len(executor.orders) == 1
    order = executor.orders[0]
    assert order.symbol == "BTCUSDT"
    assert order.direction == "long"
    assert order.stop_loss == 95.0  # the stop travels with the order
    assert [leg.order_type for leg in order.legs] == ["market"]


async def test_pause_stops_execution_but_not_signals(tmp_path):
    executor = RecordingExecutor()
    bot = FakeBot()
    scanner = _exec_scanner(tmp_path, executor, tags={"always_fire"}, bot=bot)
    scanner.execution_paused = True

    await scanner.tick()

    assert executor.orders == []  # nothing placed
    assert len(bot.sent) == 1  # but the signal still arrived to place by hand


async def test_execution_failure_cancels_the_trade_and_alerts(tmp_path):
    # Fail-safe: no retry, and the row must not sit "pending" forever holding
    # the symbol hostage.
    executor = RecordingExecutor(fail=True)
    bot = FakeBot()
    scanner = _exec_scanner(tmp_path, executor, tags={"always_fire"}, bot=bot)

    await scanner.tick()
    await asyncio.sleep(0.05)  # execute() runs on a worker thread now

    assert scanner.storage.pending_trades() == []
    assert any("EXECUTION FAILED" in m for m in bot.messages)


async def test_split_entry_is_placed_as_two_legs(tmp_path):
    executor = RecordingExecutor()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=FakeBot(),
        storage=Storage(str(tmp_path / "trades.db")),
        executor=executor,
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
        auto_execute_tags={"limit_entry"},
    )

    await scanner.tick()
    await asyncio.sleep(0.05)  # execute() runs on a worker thread now

    legs = executor.orders[0].legs
    assert [leg.order_type for leg in legs] == ["market", "limit"]
    assert legs[1].price == 100.0  # the resting leg keeps its own better price
    # 20/80 split of the position, not one averaged market order
    assert legs[0].size == pytest.approx(legs[1].size / 4)


async def test_a_dry_run_strategy_gets_no_real_exit_orders(tmp_path):
    # The scanner places the partial take-profit and cancels leftover legs by
    # calling the exchange directly, not through execute(). Gating those on
    # whitelist membership alone meant a dry-run strategy still had REAL exit
    # orders placed for it - a dry run in name only.
    from execution.executor import DryRunExecutor

    class CountingBitget(FakeBitget):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.orders_placed = 0

        def place_order(self, *a, **kw):
            self.orders_placed += 1
            return {}

        def get_open_orders(self, symbol=None):
            return []

        def cancel_order(self, *a, **kw):
            return {}

    bitget = CountingBitget(position=make_position())
    scanner = Scanner(
        bitget=bitget, bot=FakeBot(), storage=Storage(str(tmp_path / "trades.db")),
        executor=DryRunExecutor(), watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()], risk_pct=0.01,
        auto_execute_tags={"always_fire"},
    )

    await scanner.tick()
    for _ in range(4):
        await asyncio.sleep(0)

    assert bitget.orders_placed == 0


async def test_trade_close_cancels_resting_orders_even_when_placed_by_hand(tmp_path):
    # PEPEUSDT, live: a split-entry limit leg placed BY HAND (a strategy not
    # on the auto-execute whitelist) was still resting on the exchange after
    # the position's take-profit closed the trade - nothing had ever been
    # wired to cancel it, because cleanup used to be gated on the bot having
    # placed the order itself. It's no longer gated: whoever placed a leg,
    # once the trade it belongs to is over, it's cancelled.
    class RecordingBitget(FakeBitget):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.cancelled = []

        def get_open_orders(self, symbol=None):
            return [{"orderId": "resting-1", "symbol": symbol}]

        def cancel_order(self, symbol, order_id=None, **kw):
            self.cancelled.append((symbol, order_id))
            return {}

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="PEPEUSDT", direction="long", strategy_tag="manual_only")
    storage.confirm_entry(trade_id, entry_price=100, position_size=2, actual_stop=95, actual_target=115, leverage=1.0)
    # Mirrors what track_position does before calling on_close: the trade is
    # already marked closed by the time this callback runs.
    storage.close_trade(trade_id, exit_price=115.0, realized_pnl=10.0)

    bitget = RecordingBitget(position=make_position())
    scanner = Scanner(
        bitget=bitget,
        bot=FakeBot(),
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["PEPEUSDT"],
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
        auto_execute_tags=set(),  # "manual_only" is not on the whitelist - placed by hand
    )

    scanner._on_trade_closed(trade_id, 115.0)
    await asyncio.sleep(0)  # let the close-message task it schedules run

    assert bitget.cancelled == [("PEPEUSDT", "resting-1")]


class PartialRaceBitget(FakeBitget):
    """Bitget's position query and order-matching book are eventually
    consistent: the position is confirmed, but an exit order placed right
    after can still be rejected for a few seconds. `fail_times` controls how
    many 22002 rejections come back before an order is finally accepted;
    `other_error` makes every attempt fail with something that ISN'T the
    settle race, to prove that case is never retried.

    Counts attempts on place_tpsl_order, since that is what every exit goes
    through now. It used to count reduce-only place_order calls - a path that,
    it turned out, had never once succeeded against the real exchange while
    these tests passed against the fake.
    """

    def __init__(self, fail_times=0, other_error=False, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.other_error = other_error
        self.attempts = 0
        self.leverage_calls = 0

    def set_leverage(self, *a, **kw):
        self.leverage_calls += 1

    def place_order(self, *a, reduce_only=False, **kw):
        return {}  # entries always succeed; only the exit races

    def place_tpsl_order(self, **kw):
        self.attempts += 1
        if self.other_error:
            raise RuntimeError('Bitget 400 on /api/v2/mix/order/place-tpsl-order: {"code":"40762","msg":"price deviates too much"}')
        if self.attempts <= self.fail_times:
            raise RuntimeError('Bitget 400 on /api/v2/mix/order/place-tpsl-order: {"code":"22002","msg":"No position to close"}')
        return {}


def _live_partial_scanner(tmp_path, bitget, bot=None):
    from execution.executor import LiveExecutor

    return Scanner(
        bitget=bitget,
        bot=bot or FakeBot(),
        storage=Storage(str(tmp_path / "trades.db")),
        executor=LiveExecutor(bitget),
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
        auto_execute_tags={"always_fire"},
    )


async def test_partial_retries_through_the_settle_race_and_succeeds(tmp_path, monkeypatch):
    import notifier.scanner as scanner_module

    monkeypatch.setattr(scanner_module, "PARTIAL_SETTLE_RETRY_DELAYS", (0.0, 0.0))
    bitget = PartialRaceBitget(position=make_position(), fail_times=2)
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert bitget.attempts == 3  # failed twice, succeeded on the third


async def test_partial_alerts_after_exhausting_retries(tmp_path, monkeypatch):
    import notifier.scanner as scanner_module

    monkeypatch.setattr(scanner_module, "PARTIAL_SETTLE_RETRY_DELAYS", (0.0, 0.0))
    bitget = PartialRaceBitget(position=make_position(), fail_times=99)  # never recovers
    bot = FakeBot()
    scanner = _live_partial_scanner(tmp_path, bitget, bot=bot)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert bitget.attempts == 3  # first attempt + both retries, then gives up
    assert any("partial take-profit" in m and "FAILED" in m for m in bot.messages)
    assert any("set one by hand" in m for m in bot.messages)


async def test_partial_does_not_retry_a_different_rejection(tmp_path, monkeypatch):
    """22002 is a settle race worth waiting out. Anything else means retrying
    won't help, so it must fail fast rather than spend the whole retry budget."""
    import notifier.scanner as scanner_module

    monkeypatch.setattr(scanner_module, "PARTIAL_SETTLE_RETRY_DELAYS", (0.0, 0.0))
    bitget = PartialRaceBitget(position=make_position(), other_error=True)
    bot = FakeBot()
    scanner = _live_partial_scanner(tmp_path, bitget, bot=bot)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert bitget.attempts == 1  # gave up immediately, no retries burned
    assert any("FAILED" in m for m in bot.messages)


class RwaTpslBitget(PartialRaceBitget):
    """Records place_tpsl_order / place_order calls separately, and a
    resting profit_plan order the resize path should cancel."""

    def __init__(self, resting_plan_order_id=None, **kw):
        super().__init__(**kw)
        self.tpsl_calls = []
        self.cancelled_plan_orders = []
        self._resting_plan_order_id = resting_plan_order_id

    def place_order(self, *a, reduce_only=False, **kw):
        if reduce_only:
            raise AssertionError("an RWA take-profit must not go through a plain reduce-only limit")
        return {}

    def place_tpsl_order(self, **kw):
        self.tpsl_calls.append(kw)
        return {}

    def get_plan_orders(self, symbol, direction):
        if self._resting_plan_order_id is None:
            return []
        return [
            {
                "plan_type": "profit_plan",
                "is_stop": False,
                "is_target": True,
                "trigger_price": 110.0,
                "size": 10.0,
                "order_id": self._resting_plan_order_id,
            }
        ]

    def cancel_plan_order(self, symbol, plan_type, order_id=None, **kw):
        self.cancelled_plan_orders.append((symbol, plan_type, order_id))
        return {}


async def test_an_rwa_symbols_take_profit_goes_through_a_plan_order(tmp_path):
    """GOOGLUSDT's actual failure: a plain reduce-only limit take-profit is
    capped at the exchange's own ~2% price band on tokenized-stock symbols,
    and a completely ordinary ~3.7%-from-entry target was rejected outright.
    A TP plan order's trigger price isn't a resting order in the book, so
    it isn't bound by that band."""
    bitget = RwaTpslBitget(position=make_position(), is_rwa=True)
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert len(bitget.tpsl_calls) == 1
    call = bitget.tpsl_calls[0]
    assert call["plan_type"] == "profit_plan"
    assert call["direction"] == "long"


async def test_every_symbol_takes_its_exit_through_a_plan_order(tmp_path):
    """Crypto symbols go through place_tpsl_order too, not just RWA ones.

    The reduce-only limit path they used to take never placed a single
    successful take-profit against the live account - PEPEUSDT, AAPLUSDT,
    GOOGLUSDT, ZECUSDT and WLDUSDT all rejected between 2026-08-03 and
    08-11, mostly 22002 "No position to close" against positions that plainly
    existed. Plan orders name the position with holdSide instead of leaving it
    to a side/tradeSide pairing, and are what Bitget's own TP/SL panel places.
    """
    bitget = PartialRaceBitget(position=make_position(), is_rwa=False)
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert bitget.attempts == 1  # the plan-order path ran
    assert not any(kw.get("reduce_only") for kw in getattr(bitget, "placed", []))


async def test_no_exit_is_ever_sent_as_a_reduce_only_limit(tmp_path):
    """A standing guard on the primitive, not on one call site.

    place_order(reduce_only=True) has a 100% live failure rate and is now
    unused; this fails if any exit path reaches for it again. The old tests
    passed against a fake that happily accepted it, which is precisely why
    nobody noticed the real exchange never had.
    """
    calls = []

    class Recording(PartialRaceBitget):
        def place_order(self, *a, reduce_only=False, **kw):
            calls.append(reduce_only)
            return {}

    bitget = Recording(position=make_position(), is_rwa=False)
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert calls, "the entry legs should still have gone through place_order"
    assert not any(calls), "an exit was sent as a reduce-only limit"


async def test_a_growing_rwa_position_cancels_the_old_plan_order_before_replacing_it(tmp_path):
    """The resize path (a resting limit leg filling and growing the position)
    already cancelled stale REGULAR resting orders before replacing the
    take-profit. An RWA take-profit lives on the plan-orders book instead,
    which the old cancel loop never looked at - left alone, a growing
    position would keep an old, now under-sized target instead of getting
    one replaced with the new total."""
    bitget = RwaTpslBitget(position=make_position(), is_rwa=True, resting_plan_order_id="plan-1")
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner._place_partial(
        Signal(symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=95.0, strategy_tag="t"),
        plan=type("Plan", (), {"take_profit": 110.0})(),
        position_size=10.0,
        replace=True,
    )

    assert bitget.cancelled_plan_orders == [("BTCUSDT", "profit_plan", "plan-1")]
    assert len(bitget.tpsl_calls) == 1


class ArmedStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 3's day version: its trigger lives on a
    timeframe fetched only for symbols it has armed."""

    tag = "armed"
    timeframes = ["1H", "5m"]
    armed_timeframes = ("5m",)

    def __init__(self, should_arm=True):
        self.should_arm = should_arm

    def arms(self, symbol, bars_by_timeframe):
        return self.should_arm


def test_armed_timeframes_do_not_drive_the_scan_cadence():
    # Declaring 5m normally would drag the whole watchlist to a 5m loop, which
    # is exactly what arming exists to avoid.
    scanner = Scanner(
        bitget=None, bot=None, storage=None, executor=None,
        watchlist=[], strategies=[ArmedStrategy()],
    )

    assert "5m" not in scanner.required_timeframes()
    assert scanner.armed_timeframes() == {"5m"}


async def test_arming_is_recomputed_every_scan(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    strategy = ArmedStrategy(should_arm=True)
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()), bot=FakeBot(), storage=storage,
        executor=ManualExecutor(), watchlist=["BTCUSDT"], strategies=[strategy], risk_pct=0.01,
    )

    await scanner.tick()
    assert scanner._armed == {"armed": {"BTCUSDT"}}

    # The setup dies: the next scan must drop it, with no disarm step and no
    # way for a stale flag to keep it armed.
    strategy.should_arm = False
    await scanner.tick()
    assert scanner._armed == {}


async def test_armed_strategy_does_not_dispatch_from_the_regular_scan(tmp_path):
    # Its trigger timeframe isn't fetched watchlist-wide, so evaluating it here
    # would be reading data the scan never gathered.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()), bot=bot, storage=storage,
        executor=ManualExecutor(), watchlist=["BTCUSDT"], strategies=[ArmedStrategy()], risk_pct=0.01,
    )

    await scanner.tick()

    assert bot.sent == []


async def test_poll_armed_dispatches_for_armed_symbols_only(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()), bot=bot, storage=storage,
        executor=ManualExecutor(), watchlist=["BTCUSDT"], strategies=[ArmedStrategy()], risk_pct=0.01,
    )

    await scanner.poll_armed()
    assert bot.sent == []  # nothing armed yet

    await scanner.tick()
    await scanner.poll_armed()

    assert len(bot.sent) == 1
    assert "BTCUSDT" in bot.sent[0]


class TieredExitStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 3: 75% off at its own target, the rest run to a
    chart level rather than a fixed ratio."""

    tag = "tiered"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.reward_risk_ratio = 2.0
        signal.partial_fraction = 0.75
        signal.remainder_target = 130.0
        signal.remainder_note = "daily resistance"
        signal.extra_notes = ("At all-time highs: trail the stop up under each rising low.",)
        return signal


async def test_alert_renders_a_strategy_owned_two_tier_exit(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()), bot=bot, storage=storage,
        executor=ManualExecutor(), watchlist=["BTCUSDT"], strategies=[TieredExitStrategy()], risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    # 20 units total: 75% off at the 1:2 target, the remaining 25% to the
    # named level - not the scanner's fixed 1:3.
    assert "close 14.76 (75%) at 110.00" in text
    assert "close the remaining 4.92 at 130.00 (daily resistance)" in text
    assert "(1:3)" not in text
    assert "trail the stop up" in text


async def test_alert_shows_a_single_leg_when_theres_no_market_fraction(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[PureLimitStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    # The whole 20-unit position rests on the one limit, no "at market" leg,
    # and no fallback-target line - there's no partial fill to give one for.
    assert "Enter: $1,969 (19.69) limit 100.00 (EMA9)" in text
    assert "at market" not in text
    assert "If the limit leg never fills" not in text


async def test_size_line_shows_dollars_quantity_and_leverage(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    # risk 1% of 10k = 100, stop 5% away, sized against price risk + the
    # round-trip fee -> notional 1,969, 19.69 units at 100
    assert "Size: $1,969 (19.69 @ 10.0x)" in bot.sent[0]
    assert "Notional" not in bot.sent[0]  # the old long-form line is gone
    assert "Margin needed" not in bot.sent[0]
    assert "Risk:" not in bot.sent[0]


async def test_confluence_marks_the_alert_but_no_longer_raises_the_risk(tmp_path):
    # A signal confirmed by a pattern is sized at 2% instead of 1% and says so.
    # Nothing is suppressed for lacking confirmation - the evidence behind the
    # size-up is 22 trades, so it earns extra size but never a veto.
    from tests.test_patterns import IHS, _bars

    pattern_bars = _bars(IHS)

    class PatternBitget(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            if granularity == "1H":
                rows = [
                    [str(i * 1000), "100", f"{h}", f"{lo}", f"{c}", "1", "1"]
                    for i, (h, lo, c) in enumerate(
                        zip(pattern_bars["high"], pattern_bars["low"], pattern_bars["close"])
                    )
                ]
                return rows[:-1] if closed_only else rows
            return super().get_candles(symbol, granularity, limit, closed_only)

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, PatternBitget(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "Confirmed by inverse head-and-shoulders on 1H" in text
    # It marks the alert but no longer sizes it up. confluence() accepts a
    # breakout up to CONFLUENCE_BARS old, so this used to pay 2% on evidence
    # that could be two days stale - TRXUSDT was sized off a wedge that broke
    # 17 hours earlier. Risk is staged on the pattern actually breaking now.
    assert "risk 2%" not in text
    # 1% of 10k = 100 risk over a 5% stop -> 1,969 notional, the SAME as the
    # unconfirmed case. The second increment is earned by the break, not by
    # the pattern being present.
    assert "$1,969" in text


async def test_no_confluence_leaves_risk_and_message_alone(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "Confirmed by" not in text
    assert "risk" not in text
    assert "$1,969" in text  # 1% risk, unchanged


def test_bars_are_refetched_only_when_the_candle_turns_over(tmp_path):
    # Scans run every 15m, so without this a daily series would be refetched 96
    # times a day to discover it had not changed.
    calls = []

    class CountingBitget(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            calls.append(granularity)
            return super().get_candles(symbol, granularity, limit, closed_only)

    scanner = build_scanner(Storage(str(tmp_path / "t.db")), CountingBitget(), FakeBot())

    hour = 3_600_000.0  # exactly on a 1D and 1H boundary
    scanner._bars("BTCUSDT", "1D", now=hour)
    scanner._bars("BTCUSDT", "1D", now=hour + 900)  # a scan later, same daily candle
    scanner._bars("BTCUSDT", "1D", now=hour + 3600)  # still the same daily candle
    assert calls.count("1D") == 1

    scanner._bars("BTCUSDT", "1D", now=hour + 86_400)  # next daily candle
    assert calls.count("1D") == 2

    # A 15m series turns over every scan, so it is fetched every time.
    scanner._bars("BTCUSDT", "15m", now=hour)
    scanner._bars("BTCUSDT", "15m", now=hour + 900)
    assert calls.count("15m") == 2


def _leg(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


# A pole is now a short (<= FLAG_POLE_MAX_BARS), uninterrupted run of
# same-direction candles (notifier.strategies.patterns._clean_poles) rather
# than any confirmed zigzag swing - see tests/test_patterns.py for the full
# story (QQQUSDT's real "pole" was a 10-bar grind Dror rejected on sight).
# That means the pole portion needs a REAL candle body, so _coiling_bitget
# takes independent opens/closes rather than one "closes" list that would
# make every bar open == close (a doji _clean_poles reads as neither up nor
# down, never as part of a pole).
_POLE_OPENS = [100.0] * 30 + [100.0, 120.0]
_POLE_CLOSES = [100.0] * 30 + [120.0, 140.0]
_CONSOLIDATION = [136.0, 130.0, 133.0, 128.0, 131.0, 129.0, 130.5]
COILING_OPENS = _POLE_OPENS + _CONSOLIDATION
COILING_CLOSES = _POLE_CLOSES + _CONSOLIDATION


def _coiling_bitget(opens, closes, timeframe="1H"):
    class PendingBitget(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            if granularity == timeframe:
                rows = [
                    [str(i * 1000), f"{o}", f"{max(o, c) + 1}", f"{min(o, c) - 1}", f"{c}", "1", "1"]
                    for i, (o, c) in enumerate(zip(opens, closes))
                ]
                return rows[:-1] if closed_only else rows
            return super().get_candles(symbol, granularity, limit, closed_only)

    return PendingBitget


async def test_pending_pattern_is_named_with_its_break_price_and_does_not_raise_risk(tmp_path):
    # A pole then a still-running consolidation: a bull flag that has NOT
    # broken. The alert should say so and quote the level, so the setup can be
    # checked against the chart at approval - but the size must stay at base
    # risk, because the flag can still break down from here.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, _coiling_bitget(COILING_OPENS, COILING_CLOSES)(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "Pending bull flag on 1H" in text
    assert "breaks above" in text
    assert "% away" in text
    assert "Risk stays 1% until it breaks." in text
    # Present but unbroken must never size the trade up on its own.
    assert "risk 2%" not in text


def _watching(scanner, trade_id, *, break_level, invalidation, direction="long"):
    scanner._awaiting_break["BTCUSDT"] = {
        "direction": direction,
        "name": "bull flag",
        "timeframe": "1H",
        "break_level": break_level,
        "invalidation_level": invalidation,
        "trade_id": trade_id,
        "strategy_tag": "always_fire",
        "risk_pct": 0.01,
    }


def _open_trade(storage, stop=90.0):
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag="always_fire")
    storage.confirm_entry(trade_id, entry_price=100, position_size=1, actual_stop=stop, actual_target=130, leverage=1.0)
    return trade_id


async def test_pattern_break_offers_the_add_on_and_tightens_the_stop(tmp_path):
    # FakeBitget's bars close at 100, so a break level of 99 has been taken out.
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage, stop=90.0)
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()

    assert len(bot.sent) == 1
    text = bot.sent[0]
    assert "ADD-ON: BTCUSDT LONG" in text
    assert "bull flag" in text
    assert "risk 1%" in text  # the SECOND 1%, not a jump straight to 2%
    # The flag's low (95) is tighter than the trade's own 90 stop, so it wins.
    assert "WHOLE position to 95.00" in text
    assert scanner._awaiting_break == {}  # offered once, then the watch ends


async def test_the_add_on_never_loosens_an_already_tighter_stop(tmp_path):
    # Flag low at 85 is LOOSER than the trade's 90 stop. Moving there would
    # widen risk on the original leg, so the existing stop has to survive.
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage, stop=90.0)
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)
    _watching(scanner, trade_id, break_level=99.0, invalidation=85.0)

    await scanner.poll_pending_breaks()

    assert "WHOLE position to 90.00" in bot.sent[0]


async def test_add_on_approval_executes_and_resyncs_the_journal(tmp_path):
    """Zero coverage before this: _offer_add_on's on_approve places the order
    AND resyncs the trade row from Bitget's real fill. Until that resync
    existed, an approved add-on doubled the position on the exchange while
    the journal kept its ORIGINAL size/entry/risk - the XAGUSDT #17 failure
    mode, since total_open_risk() and committed_margin() both read off that
    row and both went stale the same way."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage, stop=90.0)
    bot = FakeBot()
    executor = RecordingExecutor()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position(entry_price=100.5, size=2.0)),
        bot=bot, storage=storage, executor=executor,
        watchlist=["BTCUSDT"], strategies=[AlwaysFireStrategy()], risk_pct=0.01,
        auto_execute_tags={"always_fire"},
    )
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()

    assert len(executor.orders) == 1, "the alert-only early return must not have fired"
    order = executor.orders[0]
    assert order.symbol == "BTCUSDT" and order.direction == "long"

    # Read back from Bitget, not the plan's guessed numbers - matches
    # make_position's real fill, not the trade's original entry=100/size=1.
    trade = storage.get_trade(trade_id)
    assert trade.מחיר_כניסה == 100.5
    assert trade.גודל_פוזיציה == 2.0


async def test_add_on_execution_failure_sends_a_message_and_never_resyncs(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage, stop=90.0)
    bot = FakeBot()
    executor = RecordingExecutor(fail=True)
    scanner = Scanner(
        bitget=FakeBitget(position=make_position(entry_price=100.5, size=2.0)),
        bot=bot, storage=storage, executor=executor,
        watchlist=["BTCUSDT"], strategies=[AlwaysFireStrategy()], risk_pct=0.01,
        auto_execute_tags={"always_fire"},
    )
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()
    for _ in range(3):
        await asyncio.sleep(0)  # the failure message is sent via asyncio.create_task

    assert len(executor.orders) == 1, "it was attempted"
    assert any("ADD-ON FAILED" in m for m in bot.messages)
    # Untouched - resync must never run for an order that never landed.
    trade = storage.get_trade(trade_id)
    assert trade.מחיר_כניסה == 100.0
    assert trade.גודל_פוזיציה == 1.0


async def test_add_on_resync_failure_is_logged_not_raised(tmp_path, caplog):
    """The add-on is already placed by the time this can fail - raising into
    the button handler here would be worse than a stale journal row, since
    NotifierBot's on_callback has no idea an order already went out and
    would log an unhandled exception for a trade that in fact executed.

    Asserts on the SPECIFIC logged exception rather than just "did not
    raise" - a first version of this test passed even with
    check_position_now missing from scanner.py's own imports entirely
    (NameError, not this test's crafted RuntimeError), because both are
    silently swallowed by the same except Exception and the test's only
    assertions were "no raise" and "no resync", true either way. Checking
    the logged message is what would have caught that the import was
    missing, instead of accidentally passing while testing nothing real.
    """
    caplog.set_level(logging.ERROR, logger="notifier.scanner")
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage, stop=90.0)
    bot = FakeBot()
    executor = RecordingExecutor()

    class BrokenPositionRead(FakeBitget):
        def get_position(self, symbol, direction=None):
            raise RuntimeError("simulated Bitget read failure")

    scanner = Scanner(
        bitget=BrokenPositionRead(position=make_position()), bot=bot, storage=storage, executor=executor,
        watchlist=["BTCUSDT"], strategies=[AlwaysFireStrategy()], risk_pct=0.01,
        auto_execute_tags={"always_fire"},
    )
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()  # must not raise

    assert len(executor.orders) == 1, "the order still goes out before the resync is attempted"
    trade = storage.get_trade(trade_id)
    assert trade.מחיר_כניסה == 100.0, "resync never landed, but nothing crashed either"
    assert "simulated Bitget read failure" in caplog.text, (
        "must be THIS test's own crafted failure, not some other exception swallowed by the same except clause"
    )


async def test_wrong_way_break_sends_a_note_and_no_add_on(tmp_path):
    # Close 100 is through a 105 invalidation and nowhere near the 200 break.
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage)
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)
    _watching(scanner, trade_id, break_level=200.0, invalidation=105.0)

    await scanner.poll_pending_breaks()

    assert bot.sent == []  # no Approve/Reject - nothing is being offered
    assert len(bot.messages) == 1
    assert "broke the WRONG way" in bot.messages[0]
    # No exit action: the stop already defines where the trade ends.
    assert "Your stop still governs" in bot.messages[0]
    assert scanner._awaiting_break == {}


async def test_a_signal_that_never_fills_cancels_the_pending_row_and_resting_orders(tmp_path, monkeypatch):
    """_confirm_and_track's other branch, zero coverage before this:
    wait_for_signal_position returning None (the fill timeout elapsed) must
    not leave a pending trade row stuck, or a resting order with no trade
    behind it - left alone it could open a position hours later against a
    setup that no longer exists, bot-placed or placed by hand off the
    alert.

    wait_for_signal_position's OWN timeout/poll mechanics are already
    covered directly in test_tracker.py (including the timeout case) - this
    monkeypatches it to return None immediately rather than re-deriving that
    with a real signal.unfilled_timeout_seconds, which would need a real
    ~10s sleep here since the scanner's call site never threads poll_interval
    through, only timeout_seconds.
    """
    import notifier.scanner as scanner_module

    async def never_fills(*_a, **_kw):
        return None

    monkeypatch.setattr(scanner_module, "wait_for_signal_position", never_fills)

    class TrackedBitget(FakeBitget):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.cancelled = []

        def cancel_order(self, symbol, order_id=None, **kw):
            self.cancelled.append((symbol, order_id))
            return {}

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = TrackedBitget(position=None, open_orders=[{"orderId": "abc123"}])
    scanner = Scanner(
        bitget=bitget, bot=bot, storage=storage, executor=ManualExecutor(),
        watchlist=["BTCUSDT"], strategies=[AlwaysFireStrategy()], risk_pct=0.01,
    )

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert storage.pending_trades() == []
    assert bitget.cancelled == [("BTCUSDT", "abc123")]
    assert any("marked cancelled" in m for m in bot.messages)


def _be_signal(symbol="BTCUSDT", direction="long"):
    return Signal(symbol=symbol, direction=direction, entry_price=100.0, stop_loss=95.0, strategy_tag="t")


async def test_breakeven_is_placed_anyway_when_the_live_stop_cannot_be_read(tmp_path):
    """Better a redundant stop than none, per the method's own comment: a
    failure reading the CURRENT stop must not block placing the new one -
    the guard against loosening only applies when there's something real to
    compare against."""
    class UnreadableStop(FakeBitget):
        def get_stop_target(self, symbol, direction):
            raise RuntimeError("simulated read failure")

    placed = []

    class RecordingBitget(UnreadableStop):
        def place_tpsl_order(self, **kw):
            placed.append(kw)
            return {}

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, RecordingBitget(position=make_position()), bot)

    result = await scanner._move_stop_to_breakeven(_be_signal(), breakeven=100.0)

    assert len(placed) == 1
    assert placed[0]["trigger_price"] == 100.0
    assert "FAILED" not in result


async def test_breakeven_never_loosens_a_stop_already_tighter(tmp_path):
    """The guard this whole method exists for: a re-attached tracker
    re-detecting an already-filled partial must not drag a stop Dror had
    since trailed forward BACK to breakeven, handing back risk on a
    winner."""
    class AlreadyTighter(FakeBitget):
        def get_stop_target(self, symbol, direction):
            return 105.0, 130.0  # already past the 100.0 breakeven, for a long

    placed = []

    class RecordingBitget(AlreadyTighter):
        def place_tpsl_order(self, **kw):
            placed.append(kw)
            return {}

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, RecordingBitget(position=make_position()), bot)

    result = await scanner._move_stop_to_breakeven(_be_signal(), breakeven=100.0)

    assert placed == [], "the tighter, already-live stop must never be replaced with a looser one"
    assert "already at" in result


async def test_breakeven_placement_failure_notifies_and_returns_failed(tmp_path):
    class RejectsBreakeven(FakeBitget):
        def place_tpsl_order(self, **kw):
            raise RuntimeError("simulated exchange rejection")

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, RejectsBreakeven(position=make_position()), bot)

    result = await scanner._move_stop_to_breakeven(_be_signal(), breakeven=100.0)

    assert "FAILED" in result
    assert any("FAILED" in m and "move it by hand" in m for m in bot.messages)


async def test_breakeven_placement_failure_can_suppress_the_message(tmp_path):
    class RejectsBreakeven(FakeBitget):
        def place_tpsl_order(self, **kw):
            raise RuntimeError("simulated exchange rejection")

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, RejectsBreakeven(position=make_position()), bot)

    result = await scanner._move_stop_to_breakeven(_be_signal(), breakeven=100.0, notify=False)

    assert "FAILED" in result
    assert bot.messages == []


async def test_watch_ends_when_the_position_is_gone(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage)
    storage.close_trade(trade_id, exit_price=130.0)
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()

    assert bot.sent == [] and bot.messages == []  # nothing left to add to
    assert scanner._awaiting_break == {}


async def test_add_on_respects_the_aggregate_risk_cap(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_trade(storage)
    # A second position already at the 6% ceiling on a 10k account.
    hog = storage.create_pending(symbol="ETHUSDT", direction="long", strategy_tag="other")
    storage.confirm_entry(hog, entry_price=100, position_size=60, actual_stop=90, actual_target=130, leverage=1.0)

    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)
    _watching(scanner, trade_id, break_level=99.0, invalidation=95.0)

    await scanner.poll_pending_breaks()

    assert bot.sent == []  # a pattern breaking is not a licence to exceed the cap


async def test_approving_a_pending_pattern_signal_arms_the_break_watch(tmp_path):
    # The link between the two halves: the entry alert names the pattern, and
    # approving it must leave the watch armed so the break can be spotted.
    # Registered on APPROVAL, not dispatch - a signal never taken must not
    # produce an add-on for a position that does not exist.
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, _coiling_bitget(COILING_OPENS, COILING_CLOSES)(position=make_position()), bot)

    await scanner.tick()

    watch = scanner._awaiting_break.get("BTCUSDT")
    assert watch is not None, "approving a signal with a pending pattern should arm the watch"
    assert watch["name"] == "bull flag"
    assert watch["timeframe"] == "1H"
    assert watch["risk_pct"] == 0.01  # the first increment; the break earns the second
    assert watch["break_level"] > 130.5  # the consolidation's high, still overhead



# A daily pole/consolidation scaled to sit inside the AlwaysFireStrategy
# signal's own stop-to-target band (entry 100, stop 95, target 115), so the
# reachability filter cannot be what admits or rejects it - only whether 1D
# is scanned at all.
_DAILY_POLE_OPENS = [90.0] * 30 + [90.0, 97.0]
_DAILY_POLE_CLOSES = [90.0] * 30 + [97.0, 104.0]
_DAILY_CONSOLIDATION = [102.0, 99.0, 101.0, 98.5, 100.0, 99.5, 100.5]


async def test_a_daily_pattern_is_scanned_and_named(tmp_path):
    """1D used to be invisible to pattern detection.

    Dror kept reading rendered matches as "a flag, but one timeframe up" -
    SNDKUSDT, COTIUSDT, AMZNUSDT - and the daily frame those comments pointed
    at was never scanned, so the pattern he was describing could not be found
    on any timeframe. Real daily shapes exist: a watchlist run found a live
    MSFTUSDT bull flag plus seven pending daily patterns the moment 1D was
    added.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = _coiling_bitget(
        _DAILY_POLE_OPENS + _DAILY_CONSOLIDATION,
        _DAILY_POLE_CLOSES + _DAILY_CONSOLIDATION,
        timeframe="1D",
    )(position=make_position())
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    assert "Pending bull flag on 1D" in bot.sent[0], bot.sent[0]


async def test_the_longest_timeframe_wins_when_the_same_shape_appears_on_several(tmp_path):
    """CONFLUENCE_TIMEFRAMES order IS the precedence - both confluence() and
    pending() return the first match.

    The identical series is served on 1H and 1D here, so the pattern, its
    levels and its reachability are the same on both and ONLY the ordering can
    decide which is named. Longest first by Dror's call: a daily pattern is
    stronger evidence than an hourly one. The old 1H-first order was inherited
    from when 1H and 4H were the only options, and meant the weakest available
    reading won by default.
    """

    class BothFrames(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            if granularity in ("1H", "1D"):
                rows = [
                    [str(i * 1000), f"{o}", f"{max(o, c) + 1}", f"{min(o, c) - 1}", f"{c}", "1", "1"]
                    for i, (o, c) in enumerate(zip(COILING_OPENS, COILING_CLOSES))
                ]
                return rows[:-1] if closed_only else rows
            return super().get_candles(symbol, granularity, limit, closed_only)

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, BothFrames(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "Pending bull flag on 1D" in text, text
    assert "on 1H" not in text.split("Pending bull flag")[1].split("\n")[0]


# ---- the runner's automatic take-profit ----
#
# Dror, after an SPCXUSDT signal whose runner tail read "close the remaining
# 0.13 at your discretion": "after the target i want the bot to make the new tp
# himself without my intervation ... the next tp under the nearest key line in
# the 1h graph".


class RunnerBitget(FakeBitget):
    """Daily bars carrying clean swings above price, so nearest_level_beyond
    has real levels to find. Served on RUNNER_LEVEL_TIMEFRAME, which is the
    daily: on SPCXUSDT the nearest 1H pivot above the target was a 136.21
    blip while the daily gave the 147.24 Dror named on sight."""

    def __init__(self, closes=None, **kw):
        super().__init__(**kw)
        self.placed = []
        self.tpsl = []
        self._closes = closes

    def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
        if granularity == RUNNER_LEVEL_TIMEFRAME and self._closes is not None:
            rows = [
                [str(i * 1000), f"{c}", f"{c + 1}", f"{c - 1}", f"{c}", "1", "1"]
                for i, c in enumerate(self._closes)
            ]
            return rows[:-1] if closed_only else rows
        return super().get_candles(symbol, granularity, limit, closed_only)

    def get_mark_price(self, symbol):
        # Consistent with the daily fixture by default - FakeBitget's own
        # fixed 100.0 is unrelated to `closes` and would silently make every
        # runner_target() call here about whatever level happens to sit near
        # 100, not the one the test actually built. A test modelling a live
        # price that has since diverged from the daily close overrides this.
        return self._closes[-1] if self._closes is not None else super().get_mark_price(symbol)

    def place_order(self, *a, **kw):
        self.placed.append(kw)
        return {}

    def place_tpsl_order(self, **kw):
        self.tpsl.append(kw)
        return {}


def _swinging_closes():
    """Up to 130, back to 110, up to 150, back to 120 - two confirmed swing
    highs (130 and 150) sitting above a final price of ~120."""
    return (
        [100.0] * 20
        + _leg(100, 130, 10)
        + _leg(130, 110, 8)
        + _leg(110, 150, 12)
        + _leg(150, 120, 10)
    )


class RunnerStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 3: manages its own two-tier exit, so the runner
    level is read off the 1H chart rather than a reward ratio."""

    tag = "runner"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.partial_fraction = 0.75
        signal.remainder_target = 400.0  # the daily fallback, deliberately far away
        return signal


def _runner_scanner(tmp_path, bitget, bot=None, tags=("runner",)):
    from execution.executor import ManualExecutor, RoutingExecutor

    return Scanner(
        bitget=bitget,
        bot=bot or FakeBot(),
        storage=Storage(str(tmp_path / "trades.db")),
        executor=RoutingExecutor({}, default=ManualExecutor(), exit_managed_tags=set(tags)),
        watchlist=["BTCUSDT"],
        strategies=[RunnerStrategy()],
        risk_pct=0.01,
    )


def test_exit_management_is_weaker_than_execution(tmp_path):
    """Strategy 3 may have its exits managed while its ENTRIES stay manual -
    that separation is the whole point, since a reduce-only order cannot open
    or grow a position."""
    scanner = _runner_scanner(tmp_path, RunnerBitget(position=make_position()))

    assert scanner.manages_exits("runner")
    assert not scanner.auto_executes("runner"), "exit management must not imply execution"
    assert not scanner.manages_exits("some_other_strategy")


def test_the_runner_target_sits_under_the_nearest_level(tmp_path):
    bitget = RunnerBitget(position=make_position(), closes=_swinging_closes())
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, note = scanner.runner_target(signal, fallback=400.0)

    # 131 is the nearest confirmed swing above the ~120 close (the 130 close
    # plus its 1-point wick); 151 is the further one and must not win.
    assert 125.0 < target < 131.0, f"expected just under the 131 level, got {target}"
    assert f"{RUNNER_LEVEL_TIMEFRAME} level" in note
    assert target != 400.0, "the daily fallback must not be used when 1H has a level"


def test_runner_target_uses_the_live_price_not_a_stale_daily_close(tmp_path):
    """DOGEUSDT and AEVOUSDT, both live: "Long position take profit price
    please > mark price" (Bitget 40915).

    _bars() caches a 1D series for the whole day - its own docstring says so
    - so the forming candle's close it hands to runner_target() is a snapshot
    from whenever that series was first fetched, not the price right now. A
    long that has since rallied past the level runner_target() picked makes
    the "target is beyond price" check pass against a price that no longer
    exists.

    Built so the stale daily close (126) is close enough to the only level in
    this data (~141, a confirmed swing high) to pass RUNNER_LEVEL_MAX_ATR, but
    the live mark price (142) has already moved past where that level's
    target lands (~140.2): reading the stale close returns that ~140.2 target
    without ever knowing price is no longer under it. Reading the live price
    instead must either find nothing beyond 142 in this data, or produce a
    target that is genuinely beyond it - never hand back that same ~140.2.
    """
    closes = [100.0] * 20 + _leg(100, 140, 10) + _leg(140, 126, 8)  # confirmed high ~141, stale close 126

    class RalliedPastTheLevel(RunnerBitget):
        def get_mark_price(self, symbol):
            return 142.0  # above the ~140.2 target the stale close would compute

    bitget = RalliedPastTheLevel(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, note = scanner.runner_target(signal, fallback=400.0)

    assert target is None or target >= 142.0, (
        f"target {target} is below the live mark price 142.0 - Bitget would reject this "
        f"exactly like DOGEUSDT and AEVOUSDT were rejected live"
    )


def test_broken_support_counts_as_a_level_overhead(tmp_path):
    """SPCXUSDT, and the defect that made this whole rule wrong first time.

    The level Dror named for the runner - 147.24 - is a daily LOW, tested
    repeatedly in June and July, that price later fell through. Coming back up
    it is resistance. The first version only considered pivot HIGHS for a long
    and therefore could not see it at all, reporting "no level, keep trailing"
    on a trade where he could name the target on sight.

    Built so the two rules give clearly different answers rather than answers
    that differ only by the buffer: price ends near 90 with a swing LOW at
    ~104 and swing HIGHS at ~126 and ~131 above it. A highs-only rule can only
    reach 126; including lows finds 104 first, which is much nearer.
    """
    closes = (
        [100.0] * 20
        + _leg(100, 130, 10)      # high ~131
        + _leg(130, 105, 8)       # pulls back to a LOW ~104
        + _leg(105, 125, 8)       # high ~126
        + _leg(125, 60, 20)       # collapses through the 104 shelf
        + _leg(60, 90, 12)        # recovers to ~90, now BELOW that old support
    )
    bitget = RunnerBitget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, _ = scanner.runner_target(signal, fallback=400.0)

    assert target is not None and target != 400.0, "a level built from lows must still count"
    assert target < 115.0, (
        f"expected the ~104 low-built shelf, not the 126 high - got {target}. "
        "A highs-only rule cannot see broken support overhead."
    )


def test_a_level_from_a_dead_regime_is_not_a_target(tmp_path):
    """MUUUSDT traded 1000-1400 in June and 26.51 in August.

    Its nearest level overhead is an 803.5 low from before that collapse - 30x
    away, 23.7 daily ATRs, a price this trade will never see. Without the cap
    the runner gets a "target" that is pure fiction while reading as a real
    one. Sensible levels in the same live sweep sat at 0.004-2.47 ATR.
    """
    closes = [1000.0] * 25 + _leg(1000, 30, 20) + [30.0, 31.0, 29.0, 30.5] * 3
    bitget = RunnerBitget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, _ = scanner.runner_target(signal, fallback=None)

    assert target is None, "a level from before a 97% collapse is not this trade's target"


def test_the_buffer_never_pushes_the_target_below_the_market(tmp_path):
    """ENAUSDT's level sat 0.02% above price against a buffer worth 1.5% of
    price, so target = level - buffer landed BELOW the market.

    A reduce-only sell below the market is not a take-profit, it is an instant
    exit at whatever is bid - it would have dumped the runner the moment it
    was placed. Better to place nothing and let the runner trail.
    """
    # Price finishes 0.4 under a 141 level while one buffer is worth ~1.05, so
    # level - buffer = 139.95 against a 140.40 market. Verified against the
    # reverted guard, which returns exactly that below-market price; an
    # earlier fixture ending at 139.9 left a 1.10 gap that the buffer did not
    # cross, so it passed either way and proved nothing.
    closes = [100.0] * 20 + _leg(100, 140, 12) + _leg(140, 100, 12) + _leg(100, 140.4, 12)
    bitget = RunnerBitget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, _ = scanner.runner_target(signal, fallback=None)
    price = float(scanner._bars("BTCUSDT", RUNNER_LEVEL_TIMEFRAME)["close"].iloc[-1])

    assert target is None or target > price, (
        f"a long's take-profit must sit ABOVE the market, got {target} against {price}"
    )


def test_the_trail_timeframe_comes_from_the_strategys_own_structure(tmp_path):
    scanner = _runner_scanner(tmp_path, RunnerBitget(position=make_position()))

    assert scanner.trail_timeframe("Strategy 3 1D/1H") == "1D"
    # The FIRST half of the pair, not the trigger: this used to assert on
    # "Strategy 3 1H/5m", a tag no strategy produces any more now that both
    # Strategy 3 instances read daily structure. Strategy 2's pair keeps the
    # discriminating power - a different structural frame from the line above.
    assert scanner.trail_timeframe("Strategy 2 1H/15m") == "1H"
    assert scanner.trail_timeframe("Strategy 1 4H") == "4H"


def test_the_stop_trails_the_last_low_while_highs_keep_rising(tmp_path):
    """Dror's fallback for the MUUUSDT case: "as long there is rising highs
    change the stoploss only to the last low (in long)".

    Rising highs at 130 then 160, with the most recent confirmed low at ~119.
    """
    closes = (
        [100.0] * 20
        + _leg(100, 130, 10) + _leg(130, 110, 8)
        + _leg(110, 160, 12) + _leg(160, 120, 10) + _leg(120, 150, 8)
    )
    bitget = RunnerBitget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)

    new_stop = scanner.trailing_stop("BTCUSDT", "long", "Strategy 3 1D/1H", current_stop=95.0)

    assert new_stop is not None and new_stop > 95.0, "the stop should ratchet up to the last swing low"
    assert new_stop < 150.0, "it is a swing low, not the current price"


def test_the_stop_never_loosens(tmp_path):
    """A stop only ever improves. Returning a LOWER stop for a long would give
    back protection already banked, which is the one thing a trailing rule
    must never do."""
    closes = (
        [100.0] * 20
        + _leg(100, 130, 10) + _leg(130, 110, 8)
        + _leg(110, 160, 12) + _leg(160, 120, 10) + _leg(120, 150, 8)
    )
    scanner = _runner_scanner(tmp_path, RunnerBitget(position=make_position(), closes=closes))

    assert scanner.trailing_stop("BTCUSDT", "long", "Strategy 3 1D/1H", current_stop=145.0) is None


def test_the_stop_stops_trailing_once_the_highs_stop_rising(tmp_path):
    """The "as long there is rising highs" half. A lower high means the move
    is topping, and ratcheting a stop into that is how a runner gets stopped
    out at the worst moment."""
    closes = (
        [100.0] * 20
        + _leg(100, 160, 12) + _leg(160, 120, 10)
        + _leg(120, 140, 8) + _leg(140, 110, 8) + _leg(110, 130, 8)  # 140 < 160: lower high
    )
    scanner = _runner_scanner(tmp_path, RunnerBitget(position=make_position(), closes=closes))

    assert scanner.trailing_stop("BTCUSDT", "long", "Strategy 3 1D/1H", current_stop=95.0) is None


async def test_a_position_that_already_has_a_target_is_not_trailed(tmp_path):
    """Trailing is the fallback for having NO target. A position with a
    defined exit is left alone."""
    closes = [100.0] * 20 + _leg(100, 130, 10) + _leg(130, 110, 8) + _leg(110, 160, 12) + _leg(160, 130, 8)

    class HasTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, 175.0  # a real target already set

    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag="runner")
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95.0, actual_target=175.0, leverage=1.0)
    bitget = HasTarget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget)
    scanner.storage = storage

    await scanner.poll_trailing_stops()

    assert bitget.tpsl == [], "a position with a target must not be trailed"


async def test_a_scanner_default_target_that_never_reached_the_exchange_is_not_trailed(tmp_path):
    """SPCXUSDT #37, live on 2026-08-20: Strategy 1's partial take-profit was
    skipped for being under Bitget's $5 minimum notional, so the exchange
    never got a target order - and poll_trailing_stops, seeing no live
    target, trailed the stop as if that were the plan. It never is for a
    scanner-default exit: runner_target() always returns a real price when
    signal.partial_fraction is None (Strategy 1's shape - see its own
    `if signal.partial_fraction is None: return fallback, "1:3"`), so a
    missing LIVE target on one of these trades can only mean the placement
    failed, not that trailing was ever intended. partial_fraction is None on
    the trade row is exactly that signal, stored by set_exit_plan straight
    off the same field on the signal at confirm time."""
    closes = [100.0] * 20 + _leg(100, 130, 10) + _leg(130, 110, 8) + _leg(110, 160, 12) + _leg(160, 130, 8)

    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, None

    tag = "Strategy 1 1D"  # trails the 1D, which is what RunnerBitget serves
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag=tag)
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95.0, actual_target=None, leverage=1.0)
    storage.set_exit_plan(tid, breakeven_stop=None, runner_target=120.0, partial_fraction=None)
    bitget = NoTarget(position=make_position(), closes=closes)
    scanner = _runner_scanner(tmp_path, bitget, tags=(tag,))
    scanner.storage = storage

    assert scanner.trailing_stop("BTCUSDT", "long", tag, 95.0) is not None, (
        "the fixture must give the trail a real swing, or this tests nothing"
    )

    await scanner.poll_trailing_stops()

    assert bitget.tpsl == [], "a scanner-default (partial_fraction is None) trade must never be trailed"
    assert storage.open_trades()[0].סטופ_לוס_בפועל == pytest.approx(95.0), "the stop must be left untouched"


def _naked_trade(tmp_path, tag="Strategy 1 1D"):
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag=tag)
    storage.confirm_entry(tid, entry_price=100, position_size=1, actual_stop=95.0, actual_target=None, leverage=1.0)
    storage.set_exit_plan(tid, breakeven_stop=None, runner_target=120.0, partial_fraction=None)
    return storage


async def test_a_naked_scanner_default_position_is_reminded_about(tmp_path):
    """SPCXUSDT #37: _place_partial's skip message fired once, at entry, and
    nothing ever followed up in the six days the position sat with no target
    - poll_trailing_stops just silently left it alone once trailing stopped
    being the (wrong) fallback. It must keep telling Dror instead."""
    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, None

    tag = "Strategy 1 1D"
    storage = _naked_trade(tmp_path, tag)
    bot = FakeBot()
    bitget = NoTarget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, bot=bot, tags=(tag,))
    scanner.storage = storage

    await scanner.poll_trailing_stops()

    assert len(bot.messages) == 1
    assert "BTCUSDT" in bot.messages[0] and "no take-profit" in bot.messages[0].lower()


async def test_the_naked_reminder_is_throttled_within_the_rolling_day(tmp_path):
    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, None

    tag = "Strategy 1 1D"
    storage = _naked_trade(tmp_path, tag)
    bot = FakeBot()
    bitget = NoTarget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, bot=bot, tags=(tag,))
    scanner.storage = storage

    await scanner.poll_trailing_stops()
    await scanner.poll_trailing_stops()
    await scanner.poll_trailing_stops()

    assert len(bot.messages) == 1, "three polls inside one day must produce exactly one reminder"


async def test_the_naked_reminder_fires_again_after_the_throttle_window(tmp_path, monkeypatch):
    import time as time_module

    import notifier.scanner as scanner_module

    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, None

    tag = "Strategy 1 1D"
    storage = _naked_trade(tmp_path, tag)
    bot = FakeBot()
    bitget = NoTarget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, bot=bot, tags=(tag,))
    scanner.storage = storage

    await scanner.poll_trailing_stops()
    assert len(bot.messages) == 1

    real_now = time_module.time()
    monkeypatch.setattr(
        scanner_module.time, "time",
        lambda: real_now + scanner_module.ALERT_THROTTLE_SECONDS + 1,
    )
    await scanner.poll_trailing_stops()

    assert len(bot.messages) == 2, "a day later, the position is still naked and must be mentioned again"


async def test_a_naked_position_that_gets_its_target_is_not_reminded(tmp_path):
    """The other side: once a target actually reaches the exchange (the
    resting limit leg fills and the self-heal places one at the larger size,
    or Dror sets one by hand), the nagging must stop."""
    class HasTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 95.0, 130.0

    tag = "Strategy 1 1D"
    storage = _naked_trade(tmp_path, tag)
    bot = FakeBot()
    bitget = HasTarget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, bot=bot, tags=(tag,))
    scanner.storage = storage

    await scanner.poll_trailing_stops()

    assert bot.messages == []


def test_the_runner_falls_back_when_no_level_is_found(tmp_path):
    # A monotonic ramp ends at its own extreme, so nothing sits above it.
    bitget = RunnerBitget(position=make_position(), closes=[100.0] * 20 + _leg(100, 200, 40))
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, note = scanner.runner_target(signal, fallback=400.0)

    assert target == 400.0
    assert "resistance" in note


def test_no_runner_target_at_all_time_highs_leaves_the_runner_trailing(tmp_path):
    """The SPCXUSDT case exactly: nothing overhead on either timeframe, so
    there is no price to sell into and capping the runner would be inventing
    one. The alert's "trail the stop up under each rising low" still governs.
    """
    bitget = RunnerBitget(position=make_position(), closes=[100.0] * 20 + _leg(100, 200, 40))
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, _ = scanner.runner_target(signal, fallback=None)

    assert target is None


async def test_a_strategy_that_does_not_manage_its_own_exit_keeps_its_ratio_target(tmp_path):
    """Strategy 1 is unchanged: its runner is the 1:3 price it already computes
    and prints. This only starts PLACING it."""
    bitget = RunnerBitget(position=make_position(), closes=_swinging_closes())
    scanner = _runner_scanner(tmp_path, bitget)
    signal = AlwaysFireStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    target, note = scanner.runner_target(signal, fallback=175.0)

    assert target == 175.0, "no partial_fraction means the scanner's own ratio target"
    assert "1:3" in note


async def test_the_runner_order_is_placed_as_a_plan_order_for_what_is_left(tmp_path):
    bitget = RunnerBitget(position=make_position(size=5.0), closes=_swinging_closes())
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    await scanner.place_runner_target(signal, fallback=400.0)

    assert len(bitget.tpsl) == 1, "one TP plan order for the remaining size"
    order = bitget.tpsl[0]
    assert order["plan_type"] == "profit_plan"
    assert order["size"] == 5.0, "sized to what is actually left, not to the plan"
    assert any("Runner target set" in m for m in bot.messages)


async def test_a_strategy_without_exit_management_gets_no_runner_order(tmp_path):
    bitget = RunnerBitget(position=make_position(), closes=_swinging_closes())
    scanner = _runner_scanner(tmp_path, bitget, tags=())  # nothing exit-managed
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    await scanner.place_runner_target(signal, fallback=400.0)

    assert bitget.placed == [] and bitget.tpsl == []


async def test_a_price_rejected_runner_target_retries_with_a_fresh_price(tmp_path, monkeypatch):
    """Bitget's 40915 ("take profit price please > mark price") is the same
    "price moved since we computed this" shape as the 22002 settle race - just
    caught by the exchange instead of by runner_target()'s own check. Retrying
    with the SAME target would just fail the same way again; the retry has to
    read price again and recompute.

    Built so the two reads land on different levels entirely: mark price 120
    on the first read finds the ~131 swing (attempt 1's target, which the
    fake exchange rejects); mark price 140 on the retry's fresh read means 131
    is no longer beyond price at all, so the recompute must land on the
    further ~151 swing instead. A stub that just resubmitted the same number
    would either keep failing or - worse - eventually place an order Bitget
    would have refused live, and this proves neither happened.
    """
    import notifier.scanner as scanner_module

    monkeypatch.setattr(scanner_module, "PARTIAL_SETTLE_RETRY_DELAYS", (0.0,))

    class RepricesOnRetry(RunnerBitget):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._mark_calls = 0
            self._tpsl_calls = 0

        def get_mark_price(self, symbol):
            self._mark_calls += 1
            return 120.0 if self._mark_calls == 1 else 140.0

        def place_tpsl_order(self, **kw):
            self._tpsl_calls += 1
            if self._tpsl_calls == 1:
                raise RuntimeError(
                    'Bitget 400 on /api/v2/mix/order/place-tpsl-order: '
                    '{"code":"40915","msg":"Long position take profit price please > mark price"}'
                )
            self.tpsl.append(kw)
            return {}

    bitget = RepricesOnRetry(position=make_position(), closes=_swinging_closes())
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})

    await scanner.place_runner_target(signal, fallback=400.0)

    assert len(bitget.tpsl) == 1, "must succeed on the retry rather than give up after the first rejection"
    placed_price = bitget.tpsl[0]["trigger_price"]
    assert placed_price > 140.0, (
        f"placed at {placed_price}, which is the ~131 level from the FIRST (rejected) read - "
        f"the retry must use the ~151 level a fresh read of the moved price finds"
    )
    assert not any("FAILED" in m for m in bot.messages)


async def test_a_position_the_db_lost_track_of_still_suppresses_the_signal(tmp_path):
    """Live on 2026-08-08: a real APTUSDT short of 9.035 @ 0.592, open since
    the 5th, was recorded in the trades DB as closed. Nothing suppressed a
    fresh Strategy 1 LONG on the same symbol.

    On a hedge-mode account that does not add to the position, it opens an
    OPPOSING one - so trusting our own bookkeeping does not risk a duplicate
    trade, it risks an accidental hedge nobody chose. The exchange is the only
    thing that actually knows.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = FakeBitget(
        position=make_position(),
        account_positions=[make_position(direction="short", entry_price=0.592, size=9.035)],
    )
    scanner = build_scanner(storage, bitget, bot)

    assert not storage.has_open_or_pending("BTCUSDT"), "the DB must be the thing that is wrong here"
    await scanner.tick()

    assert bot.sent == [], "a symbol already held on the account must not produce a signal"


async def test_a_resting_entry_order_the_db_lost_track_of_also_suppresses(tmp_path):
    """An unfilled entry limit is a trade in flight - exactly the state the DB
    calls "pending" and can lose the same way a position can."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = FakeBitget(
        position=make_position(),
        open_orders=[{"orderId": "1", "tradeSide": "open", "symbol": "BTCUSDT"}],
    )
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    assert bot.sent == []


async def test_a_resting_EXIT_order_does_not_suppress(tmp_path):
    """The bot's own take-profit is a reduce-only close, not a trade in
    flight. Counting it would mute a symbol for as long as any exit rests."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = FakeBitget(
        position=make_position(),
        open_orders=[{"orderId": "1", "tradeSide": "close", "symbol": "BTCUSDT"}],
    )
    scanner = build_scanner(storage, bitget, bot)

    await scanner.tick()

    assert len(bot.sent) == 1, "a resting exit is not exposure that blocks a new signal"


async def test_an_account_read_failure_falls_back_to_the_db(tmp_path):
    """Same call _may_signal_now makes about session data: muting the whole
    watchlist on one bad response is worse than the occasional signal it would
    have prevented. No worse than the behaviour this replaced."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()

    class Unreadable(FakeBitget):
        def get_positions(self, symbol):
            raise RuntimeError("Bitget 500")

    scanner = build_scanner(storage, Unreadable(position=make_position()), bot)

    await scanner.tick()  # must not raise

    assert len(bot.sent) == 1


def _untracked(symbol="APTUSDT", direction="short", size=9.035, entry=0.592, ctime="1785942354805"):
    p = make_position(direction=direction, entry_price=entry, size=size)
    p["symbol"] = symbol
    p["raw"] = {"cTime": ctime}
    return p


async def test_an_untracked_position_is_reported(tmp_path):
    """The real APTUSDT short: opened by hand 2026-08-05 15:05:54 UTC, 57
    seconds after trade #9 closed, and never registered. Nothing was wrong
    with the records - the bot was simply never told - and it surfaced three
    days later only because a signal fired on the same symbol.

    already_exposed() now suppresses those signals, which is correct but
    invisible: without this, a forgotten position quietly mutes its own
    symbol forever.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = FakeBitget(account_positions=[_untracked()])
    scanner = build_scanner(storage, bitget, bot)

    await scanner.poll_untracked_positions()

    assert len(bot.messages) == 1
    msg = bot.messages[0]
    assert "UNTRACKED position: APTUSDT short" in msg
    assert "blocking new APTUSDT signals" in msg
    assert "/add" in msg


async def test_an_untracked_position_is_reported_only_once(tmp_path):
    """A position deliberately left alone must not nag every hour."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(account_positions=[_untracked()]), bot)

    await scanner.poll_untracked_positions()
    await scanner.poll_untracked_positions()
    await scanner.poll_untracked_positions()

    assert len(bot.messages) == 1


async def test_a_new_position_on_the_same_symbol_is_reported_again(tmp_path):
    """Keyed on when it opened, so closing one and opening another is a
    different position and gets its own alert."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    bitget = FakeBitget(account_positions=[_untracked(ctime="111")])
    scanner = build_scanner(storage, bitget, bot)

    await scanner.poll_untracked_positions()
    bitget.account_positions = [_untracked(ctime="222")]  # closed and reopened
    await scanner.poll_untracked_positions()

    assert len(bot.messages) == 2


async def test_a_tracked_position_is_not_reported(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="APTUSDT", direction="short", strategy_tag="Strategy 1 1H")
    storage.confirm_entry(tid, entry_price=0.592, position_size=9.035, actual_stop=0.62, actual_target=0.55, leverage=10.0)
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(account_positions=[_untracked()]), bot)
    scanner.storage = storage

    await scanner.poll_untracked_positions()

    assert bot.messages == []


async def test_the_report_names_a_missing_stop(tmp_path):
    """The APT short sat for three days with no stop and no target. That is
    the part worth saying out loud."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()

    class NoProtection(FakeBitget):
        def get_stop_target(self, symbol, direction):
            return None, None

    scanner = build_scanner(storage, NoProtection(account_positions=[_untracked()]), bot)

    await scanner.poll_untracked_positions()

    assert "no stop and no target" in bot.messages[0]


class _Plan:
    def __init__(self, take_profit):
        self.take_profit = take_profit


async def test_a_partial_under_the_exchange_minimum_is_not_even_attempted(tmp_path):
    """ZECUSDT: 0.006 x 498.41 = $2.99 against a $5 floor.

    Bitget reported it as 22002 "No position to close" - the same code as the
    genuine settle race - so the retry loop spent its full ~21s budget waiting
    out a transient fault that did not exist, and the alert blamed a race. An
    order below the minimum can never be placed, so attempting it is
    arithmetic, not a race.
    """
    bitget = RunnerBitget(position=make_position(), min_notional=5.0)
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})
    signal.partial_fraction = 0.5

    await scanner._place_partial(signal, _Plan(take_profit=498.41), position_size=0.012)

    assert bitget.placed == [] and bitget.tpsl == [], "nothing may be sent"
    msg = bot.messages[0]
    assert "$2.99" in msg and "$5.00" in msg
    assert "NO partial take-profit" in msg
    assert "Nothing was attempted" in msg


async def test_a_partial_above_the_minimum_is_still_placed(tmp_path):
    """The control: the guard must only skip orders that cannot be placed."""
    bitget = RunnerBitget(position=make_position(), min_notional=5.0)
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})
    signal.partial_fraction = 0.5

    # 0.5 x 498.41 = $249, comfortably above the floor
    await scanner._place_partial(signal, _Plan(take_profit=498.41), position_size=1.0)

    assert len(bitget.tpsl) == 1
    assert not any("NO partial take-profit" in m for m in bot.messages)


async def test_a_rejected_signal_arms_no_watch(tmp_path):
    class RejectingBot(FakeBot):
        async def send_signal(self, text, on_approve, on_reject=None, **_expiry_kwargs):
            self.sent.append(text)
            if on_reject:
                on_reject()

    storage = Storage(str(tmp_path / "trades.db"))
    scanner = build_scanner(
        storage, _coiling_bitget(COILING_OPENS, COILING_CLOSES)(position=make_position()), RejectingBot()
    )

    await scanner.tick()

    assert scanner._awaiting_break == {}


async def test_scale_in_notification_reaches_telegram(tmp_path):
    """The scanner's handler wiring, end to end.

    Guards the three-call-site wiring: track_position detects the growth, the
    scanner formats it, and it reaches the bot. FakeBitget has no
    get_plan_orders, so this also exercises the path where coverage cannot be
    read - the message must still go out, since the position figures are the
    point and coverage is the extra.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0)
    storage.confirm_entry(
        trade_id, entry_price=100.0, position_size=0.1, actual_stop=95.0, actual_target=115.0, leverage=10.0
    )
    storage.resync_position(trade_id, entry_price=99.0, position_size=0.51)

    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    scanner._on_scale_in(trade_id)
    await asyncio.sleep(0)

    assert any("limit leg filled" in m for m in bot.messages)
    assert any("0.51 @ 99" in m for m in bot.messages)


class TightStopStrategy(AlwaysFireStrategy):
    """Same signal, but a stop close enough that the 1:3 target lands below the
    pending pattern's break level - so the trade resolves before it could break."""

    tag = "tight_stop"

    def evaluate(self, symbol, bars_by_timeframe):
        last_close = bars_by_timeframe["1H"]["close"].iloc[-1]
        return Signal(
            symbol=symbol,
            direction="long",
            entry_price=last_close,
            stop_loss=last_close * 0.999,
            strategy_tag=self.tag,
            reason="test signal",
        )


async def test_a_break_the_trade_would_never_live_to_see_is_not_quoted(tmp_path):
    """Dror's rule: the break must fall between the stop and the final target.

    Outside that window the trade has already resolved - past the stop it is
    closed, past the target it is closed - so the +1% the alert promises could
    never be offered, and the line costs attention at approval to read and
    dismiss. A real run found break levels from +0.76% to -13.11% away; the far
    ones are not slightly worse, they are unreachable.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=_coiling_bitget(COILING_OPENS, COILING_CLOSES)(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[TightStopStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    assert "Pending" not in text, f"an unreachable break should not be quoted:\n{text}"


async def test_a_reachable_break_is_still_quoted(tmp_path):
    """The control: same pattern, a normal stop, so the break sits inside the
    trade's life and must survive the filter."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, _coiling_bitget(COILING_OPENS, COILING_CLOSES)(position=make_position()), bot)

    await scanner.tick()

    assert "Pending bull flag on 1H" in bot.sent[0]


def test_pruning_the_seen_set_drops_the_oldest_not_an_arbitrary_half(tmp_path):
    """The whole point of _seen is to stop a signal re-alerting.

    This used to call list() on a SET, which has no order, so "keep the last
    half" kept an arbitrary half - and could discard keys added seconds ago
    while retaining ones from days back. Dropping the newest entries defeats
    de-duplication at exactly the moment it matters.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    scanner = build_scanner(storage, FakeBitget(), FakeBot())

    for i in range(120):
        scanner._seen[("SYM", "tag", "long", float(i), 0.0)] = None
    scanner._prune_seen(max_entries=100)

    assert len(scanner._seen) == 50
    kept = [k[3] for k in scanner._seen]
    assert kept == [float(i) for i in range(70, 120)], "the 50 most recent must survive"
    # The newest key specifically - the one most likely to re-fire next scan.
    assert ("SYM", "tag", "long", 119.0, 0.0) in scanner._seen


def test_pruning_leaves_a_small_seen_set_alone(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    scanner = build_scanner(storage, FakeBitget(), FakeBot())
    for i in range(10):
        scanner._seen[("SYM", "tag", "long", float(i), 0.0)] = None

    scanner._prune_seen(max_entries=100)

    assert len(scanner._seen) == 10


async def test_the_market_only_fallback_holds_its_reward_ratio(tmp_path):
    """ZECUSDT trade #13, in miniature.

    Its plan was a 483.774 blended entry against a 467.97 stop - 15.80 of risk,
    target 515.377 at 1:2. Only the market leg filled, at 498.41, so the real
    risk was 30.44. Re-anchoring the plan's ABSOLUTE reward distance gave
    530.01, which is 1.04R; Dror closed at 529.00 for 0.99R on a setup meant to
    pay 2R. The fallback has to preserve the RATIO, so a worse entry asks for a
    bigger move rather than quietly paying half.
    """
    class MarketAt101(FakeBitget):
        def get_mark_price(self, symbol):
            return 101.0

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt101(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()
    text = bot.sent[0]

    line = next(ln for ln in text.split("\n") if "never fills" in ln)
    fallback = float(line.rsplit(" at ", 1)[1].rstrip("."))

    market, stop = 101.0, 95.0
    achieved_r = (fallback - market) / (market - stop)
    assert achieved_r == pytest.approx(2.0), f"the fallback must still be 1:2, got {achieved_r:.2f}R"


async def test_a_leg_that_rounds_below_the_minimum_notional_is_refused(tmp_path):
    """The real CLUSDT rejection, reproduced to the cent.

    Its market leg came to 0.064938 at 81.60 - $5.30, comfortably over the $5
    minimum - but CLUSDT rounds to 2dp, so what actually left was 0.06, worth
    $4.896. Bitget answered "less than the minimum amount 5 USDT" AFTER the
    signal was approved and the entry attempted, cancelling the trade.

    Both existing guards missed it, from opposite sides: one valued the
    UNROUNDED size (which passes at $5.30), the other only checked the rounded
    size was non-zero (0.06 is not zero). The gap is a leg that rounds DOWN to
    something still positive but no longer worth the minimum.

    An earlier version of this test used a tiny risk_pct that made the leg
    round to zero instead - which the OLD check already caught, so it passed
    against the reverted code and proved nothing.
    """
    class MarketAt81(FakeBitget):
        def get_mark_price(self, symbol):
            return 81.6

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt81(position=make_position(), min_notional=5.0, volume_place=2),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[LimitEntryStrategy()],
        # Sized so the market leg is 0.0650: $5.30 unrounded, $4.90 once
        # floored to 0.06 - exactly CLUSDT's geometry.
        risk_pct=0.0000429,
    )

    await scanner.tick()

    assert bot.sent == [], "a leg that cannot be placed must not be alerted"
    signals = storage.read_signals()
    assert signals and signals[0].decision == "too_small"


async def test_an_overdue_weekly_report_is_reported(tmp_path):
    """The report crashing now alerts on its own way out, but nothing inside
    the job can notice a job that never runs. Two weeks of missing reports
    surfaced only because Dror asked where they had gone."""
    from datetime import datetime, timedelta, timezone

    from weekly_review import heartbeat

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(), bot)
    heartbeat.record_success(storage.db_path, now=datetime.now(timezone.utc) - timedelta(days=11))

    await scanner.poll_weekly_report_overdue()

    assert len(bot.messages) == 1
    assert "WEEKLY REPORT OVERDUE" in bot.messages[0]


async def test_an_overdue_weekly_report_is_reported_once_a_day(tmp_path):
    """It is a "look at this when you can" fact; repeating it hourly would
    make it noise and train the alert to be ignored."""
    from datetime import datetime, timedelta, timezone

    from weekly_review import heartbeat

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(), bot)
    heartbeat.record_success(storage.db_path, now=datetime.now(timezone.utc) - timedelta(days=11))

    await scanner.poll_weekly_report_overdue()
    await scanner.poll_weekly_report_overdue()

    assert len(bot.messages) == 1


async def test_a_healthy_weekly_report_says_nothing(tmp_path):
    from datetime import datetime, timedelta, timezone

    from weekly_review import heartbeat

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(), bot)
    heartbeat.record_success(storage.db_path, now=datetime.now(timezone.utc) - timedelta(days=3))

    await scanner.poll_weekly_report_overdue()

    assert bot.messages == []


# ---- the exit plan survives a restart (APTUSDT #11, 2026-08-13) ----


class BreakevenBitget(RunnerBitget):
    """Records the exact order of stop placements and cancellations, since
    placing the breakeven BEFORE cancelling the old stop is the whole point -
    the other order leaves a 10-20x position momentarily naked."""

    def __init__(self, live_stop=95.0, resting_stops=(), **kw):
        super().__init__(**kw)
        self._live_stop = live_stop
        self._resting_stops = list(resting_stops)
        self.calls = []

    def get_stop_target(self, symbol, direction):
        return self._live_stop, None

    def get_plan_orders(self, symbol, direction):
        return [
            {
                "plan_type": "loss_plan",
                "is_stop": True,
                "is_target": False,
                "trigger_price": trigger,
                "size": 0.0,
                "order_id": f"sl-{i}",
            }
            for i, trigger in enumerate(self._resting_stops)
        ]

    def place_tpsl_order(self, **kw):
        self.calls.append(("place", kw["plan_type"], kw["trigger_price"]))
        return super().place_tpsl_order(**kw)

    def cancel_plan_order(self, symbol, plan_type, order_id=None, **kw):
        self.calls.append(("cancel", plan_type, order_id))
        return {}


def _tracked_trade(scanner, breakeven=100.0, direction="long", tag="runner", entry=None):
    """An open trade as a re-attached tracker would find it in the DB.

    The entry defaults to the breakeven because that IS the invariant for a
    scanner trade - _confirm_and_track records the confirmed entry as the
    plan, and breakeven_price() re-derives from the entry anyway. Pass them
    separately only to test what happens when they diverge.
    """
    entry = (100.0 if breakeven is None else breakeven) if entry is None else entry
    trade_id = scanner.storage.create_pending("BTCUSDT", direction, strategy_tag=tag)
    scanner.storage.confirm_entry(
        trade_id, entry_price=entry, position_size=20.0,
        actual_stop=95.0, actual_target=None, leverage=1.0,
    )
    if breakeven is not None:
        scanner.storage.set_exit_plan(
            trade_id, breakeven_stop=breakeven, runner_target=400.0, partial_fraction=0.75
        )
    return trade_id


async def _settle():
    for _ in range(6):
        await asyncio.sleep(0)


def _stops_placed(bitget):
    """Triggers of every STOP placed, whichever plan family it used.

    Matched on the substring the way get_plan_orders does, rather than on one
    exact string: stops moved to breakeven and trailed go on as position-level
    `pos_loss` (size 0, "all closable") while an order-level `loss_plan` is
    still what an entry's presetStopLossPrice creates. Both are stops, and a
    test asking "what stops were placed" should not care which."""
    return [
        trigger
        for kind, plan_type, trigger in bitget.calls
        if kind == "place" and "loss" in plan_type
    ]


async def test_a_partial_after_a_restart_still_moves_the_stop_to_breakeven(tmp_path):
    """THE regression. The breakeven used to live only in a closure inside
    _confirm_and_track, so a tracker re-attached by resume_open_trades saw the
    partial fill and moved nothing. APTUSDT #11 took 50% off and rode the
    remainder on its original stop while the alert said the stop "should
    already be at entry". The plan is read from the trade row now, so the
    handler a restart re-attaches is the same one."""
    bitget = BreakevenBitget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner)

    # Exactly what resume_open_trades hands a re-attached tracker.
    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.66)
    await _settle()

    assert _stops_placed(bitget) == [100.0]


async def test_a_trade_the_bot_does_not_manage_gets_no_stop_moved(tmp_path):
    """A /add trade, or one that predates the exit plan being recorded: the
    message says to move it by hand and the bot touches nothing."""
    bitget = BreakevenBitget(position=make_position())
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    trade_id = _tracked_trade(scanner, breakeven=None)

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.66)
    await _settle()

    assert _stops_placed(bitget) == []
    assert "by hand" in bot.messages[0]


async def test_the_breakeven_never_drags_a_trailed_stop_backwards(tmp_path):
    """Re-detecting an old partial is how a restart heals, so this can run
    twice on one trade. If Dror has since trailed the stop past breakeven,
    re-placing it would hand risk back on a winner."""
    bitget = BreakevenBitget(position=make_position(), live_stop=101.0)  # already past breakeven
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner, breakeven=100.2)

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.66)
    await _settle()

    assert _stops_placed(bitget) == []


async def test_the_breakeven_is_placed_before_the_old_stop_is_cancelled(tmp_path):
    """Two stops briefly on the book is safe - the tighter triggers first.
    Cancelling first is not: it leaves the position naked for a round trip.

    The breakeven goes on as a position-level `pos_loss` while the stop being
    superseded is the order-level `loss_plan` the entry's presetStopLossPrice
    created, so the two calls name different plan families - the ORDER is what
    this pins, not the strings."""
    bitget = BreakevenBitget(position=make_position(), resting_stops=(95.0,))
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner)

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.66)
    await _settle()

    stop_calls = [c for c in bitget.calls if "loss" in c[1]]
    assert stop_calls == [("place", "pos_loss", 100.0), ("cancel", "loss_plan", "sl-0")]


async def test_a_stop_already_tighter_than_breakeven_is_not_cancelled(tmp_path):
    """Only stops the breakeven supersedes are cleared. A hand-trailed one
    sitting tighter has to survive, or the cleanup undoes the protection."""
    bitget = BreakevenBitget(position=make_position(), resting_stops=(95.0, 100.9))
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner)

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.66)
    await _settle()

    cancelled = [order_id for kind, _, order_id in bitget.calls if kind == "cancel"]
    assert cancelled == ["sl-0"]  # the 95.0 only; the 100.9 stays


async def test_a_short_moves_its_breakeven_down_not_up(tmp_path):
    """APTUSDT was a short: its stop sits ABOVE price, so tightening means
    lowering it. Reusing the long's comparison would have skipped every one."""
    bitget = BreakevenBitget(position=make_position(direction="short"), live_stop=0.6312)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner, breakeven=0.6134, direction="short")

    scanner._on_partial_exit(trade_id, closed_size=35.01, realized_pnl=1.66)
    await _settle()

    assert _stops_placed(bitget) == [0.6134]


async def _run_confirm(scanner, signal, **kw):
    """_confirm_and_track hands off to track_position, which polls forever on
    a live position. Only the confirmation half is under test here."""
    task = asyncio.create_task(scanner._confirm_and_track(**kw, signal=signal))
    await _settle()
    task.cancel()
    return task


async def test_confirming_a_trade_records_the_exit_plan_for_later(tmp_path):
    """The plan has to reach the DB at entry, because the process that holds
    it in memory may not be the one alive when the partial fills."""
    bitget = BreakevenBitget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget)
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})
    trade_id = scanner.storage.create_pending("BTCUSDT", "long", strategy_tag="runner")

    await _run_confirm(scanner, signal, trade_id=trade_id, remainder_target=400.0)

    trade = scanner.storage.get_trade(trade_id)
    assert trade.breakeven_stop == 100.0, "the CONFIRMED entry, not the alert's blended plan_entry"
    assert trade.runner_target == 400.0
    assert trade.partial_fraction == 0.75


async def test_a_trade_the_bot_does_not_manage_records_no_exit_plan(tmp_path):
    """breakeven_stop set means "the bot WILL move the stop here" - so a
    strategy whose exits stay manual must leave it NULL, which is what the
    partial notification keys on to say "move it by hand"."""
    bitget = BreakevenBitget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, tags=())  # nothing exit-managed
    signal = RunnerStrategy().evaluate("BTCUSDT", {"1H": scanner._bars("BTCUSDT", "1H")})
    trade_id = scanner.storage.create_pending("BTCUSDT", "long", strategy_tag="runner")

    await _run_confirm(scanner, signal, trade_id=trade_id, remainder_target=400.0)

    assert scanner.storage.get_trade(trade_id).breakeven_stop is None


async def test_an_untracked_position_is_not_re_reported_after_a_restart(tmp_path):
    """Dror: "every time the bot restart i get a message about apt".

    The APT short is deliberately left untracked, and the dedupe set that stops
    it nagging lived only in memory - so every restart forgot and told him
    again. Six deploys in one afternoon meant six identical alerts.
    """
    db = str(tmp_path / "trades.db")
    first = build_scanner(Storage(db), FakeBitget(account_positions=[_untracked()]), FakeBot())
    await first.poll_untracked_positions()
    assert len(first.bot.messages) == 1

    # a fresh Scanner over the same data directory is what a restart looks like
    restarted_bot = FakeBot()
    restarted = build_scanner(Storage(db), FakeBitget(account_positions=[_untracked()]), restarted_bot)
    await restarted.poll_untracked_positions()

    assert restarted_bot.messages == [], "the restart re-announced a position already reported"


# ---- adopting a hand-added trade into exit management (/manage) ----


class AdoptBitget(BreakevenBitget):
    """Mark price is configurable, because /manage validates the typed stop
    against where the market actually is."""

    def __init__(self, mark=105.0, **kw):
        super().__init__(**kw)
        self._mark = mark

    def get_mark_price(self, symbol):
        return self._mark


def _hand_added_trade(scanner, direction="long", entry=100.0, closed=0.0, tag="strategy 1"):
    """A trade as /add leaves it: a tag Dror typed, and no bot plan."""
    trade_id = scanner.storage.create_pending("BTCUSDT", direction, strategy_tag=tag)
    scanner.storage.confirm_entry(
        trade_id, entry_price=entry, position_size=20.0,
        actual_stop=95.0, actual_target=None, leverage=1.0,
    )
    if closed:
        scanner.storage.record_partial(trade_id, closed, 1.0)
    return trade_id


def _targets_placed(bitget):
    return [t for kind, plan_type, t in bitget.calls if kind == "place" and plan_type == "profit_plan"]


async def test_a_hand_added_trade_is_not_managed_until_it_is_adopted(tmp_path):
    """The silent gap: /add asks for a tag and Dror types "strategy 1", which
    never matches the instance tag "Strategy 1 1H" in LIVE_TAGS, so the bot
    managed nothing and said nothing about it."""
    bitget = AdoptBitget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner)

    assert not scanner._manages_trade(scanner.storage.get_trade(trade_id))

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.0)
    await _settle()
    assert _stops_placed(bitget) == []


async def test_manage_adopts_the_trade_without_touching_its_tag(tmp_path):
    """The tag is what the weekly review groups by, so the permission goes on
    the row instead. Rewriting it to match a routing set would buy automation
    by corrupting strategy scoring."""
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, tag="strategy 1")

    reply = await scanner.adopt_trade(trade_id, breakeven=100.0)

    trade = scanner.storage.get_trade(trade_id)
    assert trade.תגית_אסטרטגיה == "strategy 1", "the journal tag must be left exactly as typed"
    assert trade.exit_managed == 1
    assert trade.breakeven_stop == 100.0
    assert scanner._manages_trade(trade)
    assert "Managing exits on" in reply

    # and it now behaves like any other managed trade
    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.0)
    await _settle()
    assert _stops_placed(bitget) == [100.0]


async def test_managing_a_trade_whose_partial_already_filled_acts_immediately(tmp_path):
    """APTUSDT #11's exact state. The poll loop compares against the size it
    last saw, so an already-recorded scale-out is never re-detected in a
    running process - without acting here, /manage would do nothing at all
    until the next restart."""
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, closed=10.0)

    reply = await scanner.adopt_trade(trade_id, breakeven=100.0)

    assert _stops_placed(bitget) == [100.0]
    assert "already filled" in reply


async def test_manage_refuses_a_stop_on_the_wrong_side_of_the_market(tmp_path):
    """A long's stop above the market closes the runner the instant it is
    placed. This price is typed by hand, so it is where that gets caught."""
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner)

    reply = await scanner.adopt_trade(trade_id, breakeven=110.0)

    assert "wrong side" in reply
    assert bitget.calls == []
    trade = scanner.storage.get_trade(trade_id)
    assert trade.exit_managed == 0 and trade.breakeven_stop is None, "nothing may be written"


async def test_manage_refuses_a_price_that_reads_like_a_slipped_decimal(tmp_path):
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, entry=100.0)

    reply = await scanner.adopt_trade(trade_id, breakeven=10.0)  # right side, wrong magnitude

    assert "typo" in reply
    assert scanner.storage.get_trade(trade_id).breakeven_stop is None


async def test_manage_reports_trades_it_cannot_adopt(tmp_path):
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)

    assert "No trade #999" in await scanner.adopt_trade(999, breakeven=100.0)

    closed_id = _hand_added_trade(scanner)
    scanner.storage.close_trade(closed_id, exit_price=104.0)
    assert "already closed" in await scanner.adopt_trade(closed_id, breakeven=100.0)

    pending_id = scanner.storage.create_pending("BTCUSDT", "long", strategy_tag="strategy 1")
    assert "hasn't confirmed" in await scanner.adopt_trade(pending_id, breakeven=100.0)


async def test_managing_without_a_runner_target_arms_only_the_stop(tmp_path):
    """No invented level for the runner - the alert's own rule."""
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, closed=10.0)

    await scanner.adopt_trade(trade_id, breakeven=100.0)

    assert _stops_placed(bitget) == [100.0]
    assert _targets_placed(bitget) == []


async def test_managing_with_a_runner_target_arms_that_too(tmp_path):
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, closed=10.0)

    await scanner.adopt_trade(trade_id, breakeven=100.0, runner_target=130.0)

    assert _stops_placed(bitget) == [100.0]
    assert _targets_placed(bitget) == [130.0]


async def test_a_recorded_plan_alone_does_not_authorise_exits(tmp_path):
    """Defence in depth: the permission is checked separately from the plan,
    so a strategy demoted out of the routing sets stops being managed even
    though its open trades still carry a plan."""
    bitget = AdoptBitget(position=make_position())
    scanner = _runner_scanner(tmp_path, bitget, tags=())  # nothing exit-managed
    trade_id = _hand_added_trade(scanner, tag="runner")
    scanner.storage.set_exit_plan(trade_id, breakeven_stop=100.0, runner_target=None, partial_fraction=None)

    scanner._on_partial_exit(trade_id, closed_size=10.0, realized_pnl=1.0)
    await _settle()

    assert _stops_placed(bitget) == []


# ---- the breakeven follows the real entry, not the planned blend (XAGUSDT #17) ----


async def test_the_breakeven_follows_the_real_entry_when_only_one_leg_filled(tmp_path):
    """XAGUSDT #17: the alert planned 63.66 by blending the market leg's
    expected fill with the 63.42 limit, but only the 0.17 market leg filled,
    at 64.37. Moving that stop to 63.66 would have locked in a loss on the
    remainder rather than protecting it."""
    bitget = AdoptBitget(position=make_position(), live_stop=62.46)
    scanner = _runner_scanner(tmp_path, bitget)
    # a plan recorded before the entry was resynced: the two disagree
    trade_id = _tracked_trade(scanner, breakeven=63.66, entry=64.37)

    scanner._on_partial_exit(trade_id, closed_size=0.08, realized_pnl=0.2)
    await _settle()

    assert _stops_placed(bitget) == [64.37]


async def test_a_later_leg_fill_moves_the_breakeven_with_it(tmp_path):
    """The whole reason this is derived rather than stored: resync_position
    updates the average entry when the limit leg fills, and the breakeven has
    to follow it without anything else being told."""
    bitget = AdoptBitget(position=make_position(), live_stop=62.46)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _tracked_trade(scanner, breakeven=64.37, entry=64.37)

    # the resting limit fills; the tracker resyncs the true blended average
    scanner.storage.resync_position(trade_id, entry_price=63.60, position_size=0.87)

    scanner._on_partial_exit(trade_id, closed_size=0.44, realized_pnl=0.5)
    await _settle()

    assert _stops_placed(bitget) == [63.60]


async def test_an_adopted_trade_keeps_the_price_that_was_typed(tmp_path):
    """/manage exists because the bot's own idea of the trade was not good
    enough, so his number wins over the recorded entry."""
    bitget = AdoptBitget(position=make_position(), mark=105.0)
    scanner = _runner_scanner(tmp_path, bitget)
    trade_id = _hand_added_trade(scanner, entry=99.0, closed=10.0)

    await scanner.adopt_trade(trade_id, breakeven=100.0)  # deliberately not the entry

    assert _stops_placed(bitget) == [100.0]


async def test_the_message_quotes_the_same_breakeven_the_bot_places(tmp_path):
    """The failure this session opened with was a message describing an
    action nothing took. The two must not be allowed to drift apart again."""
    bitget = AdoptBitget(position=make_position(), live_stop=62.46)
    bot = FakeBot()
    scanner = _runner_scanner(tmp_path, bitget, bot=bot)
    trade_id = _tracked_trade(scanner, breakeven=63.66, entry=64.37)

    scanner._on_partial_exit(trade_id, closed_size=0.08, realized_pnl=0.2)
    await _settle()

    assert "64.37 breakeven" in bot.messages[0], "one message, and it quotes the placed price"
    assert "63.66" not in bot.messages[0]
    assert _stops_placed(bitget) == [64.37]


# ---- superseding: two instances describing one trade ----


class _Strat:
    def __init__(self, tag, supersedes=()):
        self.tag = tag
        self.supersedes = tuple(supersedes)


def _sig(symbol, tag, direction="long"):
    return Signal(
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        stop_loss=99.0,
        strategy_tag=tag,
    )


def test_a_superseding_signal_removes_the_one_it_replaces():
    pair, alone = _Strat("Strategy 2.1 4H/1H", ("Strategy 2.1 1H",)), _Strat("Strategy 2.1 1H")
    produced = [
        (alone, _sig("BTCUSDT", "Strategy 2.1 1H")),
        (pair, _sig("BTCUSDT", "Strategy 2.1 4H/1H")),
    ]
    kept = scanner._drop_superseded(produced)
    assert [s.strategy_tag for _, s in kept] == ["Strategy 2.1 4H/1H"]


def test_superseding_is_per_symbol():
    """The pair firing on BTC says nothing about the standalone on ETH."""
    pair, alone = _Strat("Strategy 2.1 4H/1H", ("Strategy 2.1 1H",)), _Strat("Strategy 2.1 1H")
    produced = [
        (alone, _sig("ETHUSDT", "Strategy 2.1 1H")),
        (pair, _sig("BTCUSDT", "Strategy 2.1 4H/1H")),
    ]
    assert len(scanner._drop_superseded(produced)) == 2


def test_superseding_is_per_direction():
    """A long and a short on one symbol are not the same trade. If both fire,
    that is a contradiction worth seeing rather than one silently hiding."""
    pair, alone = _Strat("Strategy 2.1 4H/1H", ("Strategy 2.1 1H",)), _Strat("Strategy 2.1 1H")
    produced = [
        (alone, _sig("BTCUSDT", "Strategy 2.1 1H", "short")),
        (pair, _sig("BTCUSDT", "Strategy 2.1 4H/1H", "long")),
    ]
    assert len(scanner._drop_superseded(produced)) == 2


def test_nothing_is_dropped_when_no_strategy_supersedes():
    a, b = _Strat("Strategy 1 1H"), _Strat("Strategy 2.1 1H")
    produced = [(a, _sig("BTCUSDT", "Strategy 1 1H")), (b, _sig("BTCUSDT", "Strategy 2.1 1H"))]
    assert scanner._drop_superseded(produced) == produced


# ---- the alert throttle: one prompt per symbol per instance per rolling day ----


class _FakeThrottleStore:
    """Stands in for the alert_throttle table. Same three calls, held in a dict
    so the behavioural tests below stay fast; test_the_throttle_survives_a_restart
    exercises the real table."""

    def __init__(self, open_symbols=()):
        self.rows: dict[tuple[str, str], float] = {}
        self._open = list(open_symbols)

    def open_trades(self):
        return [type("T", (), {"סימבול": sym})() for sym in self._open]

    def last_alerted(self, symbol, strategy_tag):
        return self.rows.get((symbol, strategy_tag))

    def record_alerted(self, symbol, strategy_tag, at):
        self.rows[(symbol, strategy_tag)] = at

    def clear_alert_throttle(self, symbol):
        for key in [k for k in self.rows if k[0] == symbol]:
            del self.rows[key]


class _Throttle:
    """The throttle in isolation - it touches only _alerted and storage."""

    def __init__(self, open_symbols=()):
        self._alerted = {}
        self._open_symbols = set()
        self.storage = _FakeThrottleStore(open_symbols)

    _throttled = scanner.Scanner._throttled
    _mark_alerted = scanner.Scanner._mark_alerted
    release_closed_symbols = scanner.Scanner.release_closed_symbols


def _sig2(symbol="BTCUSDT", tag="Strategy 2.1 1H", direction="long"):
    return Signal(symbol=symbol, direction=direction, entry_price=100.0,
                  stop_loss=99.0, strategy_tag=tag)


def test_the_first_prompt_on_a_symbol_goes_through():
    t = _Throttle()
    assert t._throttled(_sig2()) is False


def test_a_second_prompt_for_the_same_instance_is_suppressed():
    """2.1 fires ~166 times a day against Strategy 1's ~17, and nothing executes
    without Dror pressing Approve - so the scarce resource is his attention. An
    alert stream too noisy to read is a silent failure."""
    t = _Throttle()
    assert t._throttled(_sig2()) is False
    t._mark_alerted(("BTCUSDT", "Strategy 2.1 1H"))     # the prompt went out
    assert t._throttled(_sig2()) is True


def test_a_different_instance_on_the_same_symbol_still_prompts():
    """The same pullback on 15m and on 4H are different trades with different
    stops; collapsing them would hide which timeframe found it."""
    t = _Throttle()
    t._mark_alerted(("BTCUSDT", "Strategy 2.1 1H"))
    assert t._throttled(_sig2(tag="Strategy 2.1 4H")) is False


def test_a_different_symbol_still_prompts():
    t = _Throttle()
    t._mark_alerted(("BTCUSDT", "Strategy 2.1 1H"))
    assert t._throttled(_sig2(symbol="ETHUSDT")) is False


def test_the_throttle_expires_after_a_rolling_day():
    t = _Throttle()
    key = ("BTCUSDT", "Strategy 2.1 1H")
    t._mark_alerted(key)
    t._alerted[key] -= scanner.ALERT_THROTTLE_SECONDS + 1
    t.storage.rows[key] -= scanner.ALERT_THROTTLE_SECONDS + 1
    assert t._throttled(_sig2()) is False


def test_going_flat_releases_the_throttle():
    """One position per symbol means the throttle and the position overlap -
    once a symbol is tradeable again it should be able to ask. The trades table
    has no close timestamp, so the transition is observed rather than queried."""
    t = _Throttle(open_symbols=["BTCUSDT"])
    t._mark_alerted(("BTCUSDT", "Strategy 2.1 1H"))
    t.release_closed_symbols()                 # still open: nothing released
    assert t._throttled(_sig2()) is True

    t.storage._open = []
    t.release_closed_symbols()                 # gone flat
    assert t._throttled(_sig2()) is False


class PureMarketStrategy(AlwaysFireStrategy):
    """Stands in for Strategy 2.1 under ENTRY_MODE="next_open": the whole
    position goes in at market, and entry_price is the EMA9 that SELECTED the
    setup rather than a price the order will fill at."""

    tag = "pure_market"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.reward_risk_ratio = 2.0
        signal.limit_entry = None
        signal.market_fraction = 1.0
        return signal


async def test_a_pure_market_entry_is_sized_from_the_market_not_from_its_own_reference(tmp_path):
    """The branch that was missing, and it was sizing live money.

    A split entry blends both legs; a pure LIMIT entry fills at its own
    entry_price. Neither describes Strategy 2.1, whose entry_price is the EMA9
    while the order goes in at market on the candle AFTER the rejection - and
    that candle has closed back on the trend side by construction, so the fill
    is always on the far side of the level. Sizing off the EMA9 therefore
    always understates the distance to the stop, never overstates it.

    Measured over 591 setups on WLD/SOL/PEPE: real risk 1.89x the risk sized
    against, median 1.44x, p90 2.97x, and worse on 100% of them. A 1% trade
    risked ~1.9%, and a quarter of them breached the 2% cap.

    Here: entry_price (the "EMA9") is 100.00 with a 95.00 stop, so the old code
    sized 5.00 of risk. The market is at 102.00, making the real risk 7.00 -
    1.4x - and every level has to be measured from there.
    """
    class MarketAt102(FakeBitget):
        def get_mark_price(self, symbol):
            return 102.0

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt102(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[PureMarketStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()

    text = bot.sent[0]
    assert "Entry: 102.00" in text
    # 1:2 measured from the 102.00 fill against the 95.00 stop, not from 100.00.
    # The old basis gave 110.00, which is only 1.14R on the risk actually taken.
    assert "Target: 116.00" in text
    assert "move stop to 102.00" in text, "breakeven is the price it filled at"

    # And the size must fall, because the real risk per unit is larger. 1% of
    # the 10,000 equity is 100.00; at 7.00 of risk per unit (plus the
    # round-trip fee) that is 14.12 units, where the old 5.00 basis bought
    # 19.69 - still an oversize carrying more than the intended risk, which
    # is the whole defect stated in units.
    assert "(14.12 @" in text


class GuardedMarketStrategy(Strategy):
    """Strategy 2.1 as it actually ships: the whole position goes in at market,
    entry_price is the LEVEL that selected the setup rather than a fill, and the
    economic gates ride along on the signal to be re-asked once the scanner
    knows what the fill will be."""

    tag = "guarded_market"
    timeframes = ["1H"]

    def __init__(self, entry, stop, guard):
        self.entry, self.stop, self.guard = entry, stop, guard

    def evaluate(self, symbol, bars_by_timeframe):
        return Signal(
            symbol=symbol,
            direction="long",
            entry_price=self.entry,
            stop_loss=self.stop,
            strategy_tag=self.tag,
            reward_risk_ratio=2.0,
            limit_entry=None,
            market_fraction=1.0,
            fill_guard=self.guard,
        )


def _guarded(tmp_path, name, market, entry, stop, guard):
    class MarketAt(FakeBitget):
        def get_mark_price(self, symbol):
            return market

    storage = Storage(str(tmp_path / f"{name}.db"))
    bot = FakeBot()
    scanner = Scanner(
        bitget=MarketAt(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[GuardedMarketStrategy(entry, stop, guard)],
        risk_pct=0.01,
    )
    return scanner, bot, storage


async def test_a_stop_that_only_clears_the_floor_at_the_selecting_level_is_refused(tmp_path):
    """HYPEUSDT, 2026-08-18, and it is the whole reason FillGuard exists.

    The alert went out with entry 58.393 and stop 58.305 - a stop 0.15% of
    price, half of MIN_STOP_PCT. It passed because the strategy measured that
    stop against e9_prev = 59.191, where the same distance is 1.50%. The gate
    and the trade were talking about two different prices, and the gate lost.

    Reproduced at the same shape: the selecting level is 100.00 with a 99.00
    stop, so the strategy sees a 1.00% stop and allows it. The market is at
    99.20, making the real stop 0.20% - under the 0.30% floor.
    """
    guard = FillGuard(min_stop_pct=0.003, max_stop_pct=0.20)
    scanner, bot, storage = _guarded(tmp_path, "refused", 99.20, 100.0, 99.0, guard)

    await scanner.tick()

    assert bot.sent == [], "a 0.20% stop at the fill must not reach the alert"
    # Refused, not vanished: a signal that disappears silently is the same
    # failure as one that never fired, so the weekly report has to see it.
    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "refused_at_fill"


async def test_the_guard_lets_through_a_stop_still_wide_enough_at_the_fill(tmp_path):
    """The other half, so the guard is not merely "refuse everything".

    Same geometry, market moved the other way: a fill at 100.80 puts the 99.00
    stop 1.79% away, inside both bounds, and the trade goes out measured from
    that fill.
    """
    guard = FillGuard(min_stop_pct=0.003, max_stop_pct=0.20)
    scanner, bot, _ = _guarded(tmp_path, "allowed", 100.80, 100.0, 99.0, guard)

    await scanner.tick()

    assert bot.sent, "a 1.79% stop clears the floor and must be alerted"
    assert "Entry: 100.80" in bot.sent[0]


async def test_an_atr_floor_refuses_a_stop_inside_one_candle(tmp_path):
    """Dror on LABUSDT: "the stop is too close". It was 4.3% of price - and
    1.13 ATR at the same time, because that symbol's ATR was 5.4% of price.
    MIN_STOP_PCT is scale-free in percent and cannot express the difference.

    Here the stop is 1.00 wide against an ATR of 2.00: 0.50 ATR. It clears every
    percent bound and must still be refused by a 1.0 ATR floor - and must NOT be
    refused when the floor is off, which is what ships until the sweep sets it.
    """
    off = FillGuard(atr=2.0, min_stop_pct=0.003, max_stop_pct=0.20, min_stop_atr=0.0)
    scanner, bot, _ = _guarded(tmp_path, "atr_off", 100.0, 100.0, 99.0, off)
    await scanner.tick()
    assert bot.sent, "with the floor off this is the shipping behaviour"

    on = FillGuard(atr=2.0, min_stop_pct=0.003, max_stop_pct=0.20, min_stop_atr=1.0)
    scanner, bot, _ = _guarded(tmp_path, "atr_on", 100.0, 100.0, 99.0, on)
    await scanner.tick()
    assert bot.sent == [], "0.50 ATR is inside one candle and must be refused"


def test_the_throttle_survives_a_restart(tmp_path):
    """MMTUSDT, 2026-08-18: the SAME 4H short prompted twice inside one candle.

    Both alerts reconstruct to identical state - EMA9 bar 08-18 08:00, stop
    0.1747, target 0.1611, breakeven 0.1701 - and differ only in the market
    price quoted, so they came from two scans of one 4H bar. Dedupe should have
    collapsed them and the once-a-day throttle should have caught what dedupe
    missed. Neither did, because both lived in plain dicts on the Scanner and a
    restart between the two scans emptied them.

    A fresh Scanner object is exactly that restart.
    """
    storage = Storage(str(tmp_path / "trades.db"))

    first = _Throttle()
    first.storage = storage
    first._open_symbols = set()
    assert first._throttled(_sig2(symbol="MMTUSDT", tag="Strategy 2.1 4H")) is False
    first._mark_alerted(("MMTUSDT", "Strategy 2.1 4H"))    # the prompt went out

    restarted = _Throttle()
    restarted.storage = storage
    assert restarted._throttled(_sig2(symbol="MMTUSDT", tag="Strategy 2.1 4H")) is True, (
        "a restart must not release the throttle"
    )
    # and a different instance on the same symbol is still free to ask
    assert restarted._throttled(_sig2(symbol="MMTUSDT", tag="Strategy 2.1 15m")) is False


def test_an_unreadable_throttle_does_not_silence_the_symbol(tmp_path):
    """A throttle that cannot be read must fail open. Silencing a symbol because
    a query raised would be a worse failure than one extra prompt - the same
    call _may_signal_now makes about session data."""
    class Broken(_FakeThrottleStore):
        def last_alerted(self, symbol, strategy_tag):
            raise sqlite3.OperationalError("database is locked")

    t = _Throttle()
    t.storage = Broken()
    assert t._throttled(_sig2()) is False


def test_a_prompt_that_never_went_out_does_not_spend_the_day(tmp_path):
    """_throttled asks; only a delivered prompt records.

    It used to record on the intention to ask, and _dispatch has a dozen ways to
    return after that point - the fill guard, the risk cap, the swing slots, an
    exchange minimum, and the Telegram send itself. Every one of those silenced
    a symbol Dror was never actually asked about.

    Harmless while the throttle died with the process. Not harmless now that it
    is a table: it would hold for a full day.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    t = _Throttle()
    t.storage = storage

    assert t._throttled(_sig2(symbol="LABUSDT")) is False     # asked, not sent
    assert storage.last_alerted("LABUSDT", "Strategy 2.1 1H") is None

    restarted = _Throttle()
    restarted.storage = storage
    assert restarted._throttled(_sig2(symbol="LABUSDT")) is False, (
        "a signal refused before the send must leave the symbol free to ask"
    )


async def test_a_failed_telegram_send_does_not_kill_the_scan(tmp_path):
    """2026-08-18: the service restarted every 14-16 minutes for hours, restart
    counter 11 by 18:35, every time on telegram.error.TimedOut raised out of
    send_signal on an e2-micro's network. It escaped tick(), escaped the gather
    in run_forever, and took the process down - which is what cleared the dedupe
    set and the alert throttle between the two MMTUSDT prompts.
    """
    class TimingOutBot(FakeBot):
        async def send_signal(self, *a, **kw):
            raise TimeoutError("Timed out")

    storage = Storage(str(tmp_path / "trades.db"))
    bot = TimingOutBot()
    scanner = Scanner(
        bitget=FakeBitget(position=make_position()),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()          # must not raise

    signals = storage.read_signals()
    assert len(signals) == 1
    assert signals[0].decision == "send_failed", "the failure is recorded, not swallowed"
    # and the day's allowance is intact, because nothing was ever asked
    assert storage.last_alerted("BTCUSDT", "always_fire") is None


class _Upkeep:
    """The upkeep cadence in isolation - it reads open trades and nothing else."""

    def __init__(self, tags=()):
        self.storage = type("S", (), {
            "open_trades": lambda _s: [
                type("T", (), {"תגית_אסטרטגיה": tag})() for tag in tags
            ]
        })()

    upkeep_timeframe = scanner.Scanner.upkeep_timeframe
    trail_timeframe = scanner.Scanner.trail_timeframe
    manages_exits = staticmethod(lambda tag: True)


def test_the_upkeep_loop_wakes_on_the_fastest_frame_it_actually_trails():
    """A stop that follows 15m structure on an hourly clock ratchets once per
    four bars, so a runner gives back up to three bars of move before the stop
    follows. Strategy 2.1's 15m instance is the only thing this trails today.
    """
    assert _Upkeep(["Strategy 2.1 15m"]).upkeep_timeframe() == "15m"
    assert _Upkeep(["Strategy 1 1H"]).upkeep_timeframe() == "1H"
    # the FASTEST wins, so one 15m runner speeds the loop up for as long as it
    # is open and nothing else has to pay for it
    assert _Upkeep(["Strategy 1 1D", "Strategy 2.1 15m"]).upkeep_timeframe() == "15m"


def test_an_idle_account_does_not_get_the_fast_cadence():
    """The cost is bounded by the fast cadence existing only while a fast trade
    does - otherwise this would query an empty account every 15 minutes."""
    assert _Upkeep([]).upkeep_timeframe() == scanner.TRAILING_POLL_TIMEFRAME


def test_the_cadence_falls_back_rather_than_stalling_the_loop():
    """Same call every other poll in this loop makes: a failed read must not be
    able to stop the loop that trails live stops."""
    broken = _Upkeep([])
    broken.storage = type("S", (), {"open_trades": lambda _s: (_ for _ in ()).throw(
        sqlite3.OperationalError("database is locked"))})()
    assert broken.upkeep_timeframe() == scanner.TRAILING_POLL_TIMEFRAME


def test_the_trail_uses_its_own_pivot_scale_not_the_daily_level_one():
    """2.0 is calibrated for DAILY runner TARGET levels (the SPCXUSDT case) and
    was being reused as the trail's pivot scale on any timeframe.

    It also diverged from what was measured: the "+0.045R vs a fixed target"
    claim comes from score.simulate(runner="choch"), whose pivots are
    confirmed_pivots(..., multiple=1.25). On the two live 2.1 trades of
    2026-08-19 the difference was 0.88% vs 3.74% from price on UNIUSDT.
    """
    from backtest.score import confirmed_pivots
    import inspect

    assert scanner.TRAIL_PIVOT_ATR_MULTIPLE == 1.25
    assert scanner.RUNNER_LEVEL_PIVOT_ATR_MULTIPLE == 2.0, "the daily level scale is unchanged"
    # the live trail and the scorer that measured it must agree
    modelled = inspect.signature(confirmed_pivots).parameters["multiple"].default
    assert scanner.TRAIL_PIVOT_ATR_MULTIPLE == modelled, (
        "the live trail and the backtest that justified it must use one pivot scale"
    )


async def test_a_partial_fill_sends_ONE_message_not_three(tmp_path):
    """Dror, on the UNIUSDT partial of 2026-08-19: "i dont want to get it in 3
    different messages".

    A single scale-out produced an announcement that said "each is confirmed
    separately", then a breakeven confirmation, then a runner confirmation.
    The steps still run independently and either failing is still named - what
    changed is that the report waits for both and says what happened, instead
    of announcing what is about to.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner_ = build_scanner(storage, FakeBitget(position=make_position()), bot)

    trade_id = storage.create_pending(
        symbol="BTCUSDT", direction="long", proposed_stop=95.0,
        proposed_target=110.0, strategy_tag="Strategy 2.1 1H",
    )
    storage.confirm_entry(trade_id, entry_price=100.0, position_size=20.0,
                          actual_stop=95.0, actual_target=110.0, leverage=10.0)
    storage.set_exit_plan(trade_id, breakeven_stop=100.0, runner_target=None,
                          partial_fraction=0.5)

    scanner_._on_partial_exit(trade_id, 10.0, 2.0)
    for _ in range(12):
        await asyncio.sleep(0)

    assert len(bot.messages) == 1, f"one event, one message; got {bot.messages}"
    text = bot.messages[0]
    assert "BTCUSDT long" in text and "partial filled" in text
    assert "Closed 10 of 20 (50%)" in text, "sizes without six trailing zeros"
    assert "each is confirmed separately" not in text


async def test_a_plan_stored_at_full_float_precision_does_not_report_as_changed(tmp_path):
    """AIOUSDT, Strategy 2.1 15m, 2026-08-19. The strategy's raw stop was
    0.04359480847113895; Bitget only quotes 5 decimals for that symbol, so the
    order that actually reached the exchange sat at 0.04359 - a difference of
    4.8e-6, pure price-tick rounding.

    create_pending stored the RAW float as the plan and confirm_entry stored
    what Bitget read back, already rounded. changed_from_plan compares the two
    at a tolerance of 1e-9, so that gap - four orders of magnitude past the
    tolerance - read as a genuine deviation on almost every trade whose price
    is not already exchange-round. Dror's close message said the stop had
    changed when nothing had touched it.

    The fix rounds the plan to the same precision BEFORE it is stored, so the
    two are compared on the basis they actually share.
    """
    entry, raw_stop, ratio = 0.04294, 0.04359480847113895, 2.0
    # The same arithmetic plan_position uses, so the mocked exchange target
    # (what get_stop_target reports back) and the strategy's own plan agree -
    # anything else would test a mismatch this fix was never meant to paper
    # over.
    risk = abs(entry - raw_stop)
    raw_target = entry - ratio * risk
    exchange_stop, exchange_target = round(raw_stop, 5), round(raw_target, 5)

    class FivePlaceBitget(FakeBitget):
        def get_contract_specs(self, symbol):
            return {**self._specs, "price_place": 5}

        def get_stop_target(self, symbol, direction):
            return exchange_stop, exchange_target

    class AlwaysFiresAIO(AlwaysFireStrategy):
        def evaluate(self, symbol, bars_by_timeframe):
            signal = super().evaluate(symbol, bars_by_timeframe)
            signal.direction = "short"
            signal.entry_price = entry
            signal.stop_loss = raw_stop
            signal.reward_risk_ratio = ratio
            signal.strategy_tag = "Strategy 2.1 15m"
            return signal

    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    position = make_position(direction="short", entry_price=entry, stop=exchange_stop, target=exchange_target)
    scanner = Scanner(
        bitget=FivePlaceBitget(position=position),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["AIOUSDT"],
        strategies=[AlwaysFiresAIO()],
        risk_pct=0.01,
    )

    await scanner.tick()  # FakeBot.send_signal approves immediately
    for _ in range(6):
        await asyncio.sleep(0)

    trades = storage.open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.סטופ_לוס_מקורי == exchange_stop, "the plan is stored at the exchange's own precision"
    assert trade.יעד_רווח_מקורי == exchange_target
    assert trade.changed_from_plan is False, "price-tick rounding must not read as a deviation"


async def test_changed_from_plan_still_catches_a_real_deviation(tmp_path):
    """The fix must not blunt the flag it is protecting - a stop genuinely
    replaced (a manual move, a re-attach that finds a different exchange
    value) is still a full tick or more away and must still be flagged."""
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    position = make_position(stop=97.0)  # differs from the 95.0 the signal proposes

    class MovedStopBitget(FakeBitget):
        def get_stop_target(self, symbol, direction):
            return 97.0, 110.0

    scanner = Scanner(
        bitget=MovedStopBitget(position=position),
        bot=bot,
        storage=storage,
        executor=ManualExecutor(),
        watchlist=["BTCUSDT"],
        strategies=[AlwaysFireStrategy()],
        risk_pct=0.01,
    )

    await scanner.tick()  # FakeBot.send_signal approves immediately
    for _ in range(6):
        await asyncio.sleep(0)

    trade = storage.open_trades()[0]
    assert trade.changed_from_plan is True, "a real difference, a full point wide, must still be caught"


def test_a_stop_left_far_behind_is_pulled_halfway_to_the_price(tmp_path):
    """Dror's stall rule. The trail only ratchets to CONFIRMED swing lows, so a
    vertical run leaves no swing to move to and the whole open gain rides on a
    stop far below it. DOGEUSDT sat at +4.98R open with its stop at +0.24R
    before he closed half of it by hand on 2026-08-20.

    Entry 100, original stop 95 - one R is 5. The stop has trailed to 101 and
    price is 122, so the gap is (122 - 101) / 5 = 4.2R, past STALL_TIGHTEN_R.
    Halving it puts the stop at 101 + (122 - 101) / 2 = 111.5.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0, strategy_tag="runner")
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=101.0,
                          actual_target=None, leverage=1.0)
    scanner_ = _runner_scanner(tmp_path, RunnerBitget(position=make_position()))
    scanner_.storage = storage

    assert scanner_.stall_tighten(storage.open_trades()[0], price=122.0) == pytest.approx(111.5)


def test_a_stop_that_is_keeping_up_is_left_alone(tmp_path):
    """The other half of the rule, and why it is a GAP rather than an
    open-profit test: a runner can be far in front and still fully protected,
    because the trail found swings to move to. Same +22 open gain as above,
    but the stop has followed to 119 - a 0.6R gap, nothing to do."""
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0, strategy_tag="runner")
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=119.0,
                          actual_target=None, leverage=1.0)
    scanner_ = _runner_scanner(tmp_path, RunnerBitget(position=make_position()))
    scanner_.storage = storage

    assert scanner_.stall_tighten(storage.open_trades()[0], price=122.0) is None


def test_a_short_stop_left_behind_is_pulled_down_toward_the_price(tmp_path):
    """The mirror. Entry 100, original stop 105, so one R is still 5. The stop
    sits at 99 with price at 78: the gap is (99 - 78) / 5 = 4.2R, and halving
    it brings the stop DOWN to 88.5 rather than up."""
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="short", proposed_stop=105.0, strategy_tag="runner")
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=99.0,
                          actual_target=None, leverage=1.0)
    scanner_ = _runner_scanner(tmp_path, RunnerBitget(position=make_position()))
    scanner_.storage = storage

    assert scanner_.stall_tighten(storage.open_trades()[0], price=78.0) == pytest.approx(88.5)


async def test_a_runner_the_trail_cannot_move_has_its_stop_pulled_in_instead(tmp_path):
    """The wiring. A vertical run makes no confirmed swing, so trailing_stop()
    has nothing to return and the stop would sit still while the whole gain
    rides on top of it. The stall rule is what moves it - measured to take 409
    weak runners down to 333 (see STALL_TIGHTEN_R).

    A monotonic ramp is exactly that case: no pullback, so no swing low.
    Entry 100, original stop 95 (one R = 5), stop still at 101, price 122 -
    a 4.2R gap, halved to 111.5.
    """
    tag = "Strategy 3 1D/1H"  # trails the 1D, which is what RunnerBitget serves

    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 101.0, None

        def get_mark_price(self, symbol):
            return 122.0

    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0, strategy_tag=tag)
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=101.0,
                          actual_target=None, leverage=1.0)
    # A self-managing runner: set_exit_plan is how partial_fraction actually
    # gets onto the trade row in production, and poll_trailing_stops now
    # reads it to tell a self-managed "no target, trail instead" from a
    # scanner-default one whose target placement simply failed.
    storage.set_exit_plan(tid, breakeven_stop=100.0, runner_target=None, partial_fraction=0.5)
    bitget = NoTarget(position=make_position(), closes=[100.0] * 20 + _leg(100, 122, 40))
    scanner_ = _runner_scanner(tmp_path, bitget, tags=(tag,))
    scanner_.storage = storage

    assert scanner_.trailing_stop("BTCUSDT", "long", tag, 101.0) is None, (
        "the ramp must give the trail nothing, or this tests the wrong path"
    )

    await scanner_.poll_trailing_stops()

    assert len(bitget.tpsl) == 1, "the stalled stop should still have been moved"
    assert bitget.tpsl[0]["trigger_price"] == pytest.approx(111.5)
    assert bitget.tpsl[0]["plan_type"] == "pos_loss"
    assert storage.open_trades()[0].סטופ_לוס_בפועל == pytest.approx(111.5)


async def test_a_trade_the_trail_just_protected_is_not_also_tightened(tmp_path):
    """The ordering matters. If the gap were measured from the stop as it stood
    BEFORE the trail moved, a trade the trail had just pulled up behind price
    would still read as stalled and be tightened again on top - firing on
    exactly the trades this rule is not for.

    Entry 100, original stop 95 (one R = 5), stop recorded at 101, price 122.
    Measured from 101 that is a 4.2R gap; measured from the swing low the trail
    finds, it is far less - so the trail's own level must be what gets placed.
    """
    tag = "Strategy 3 1D/1H"
    closes = (
        [100.0] * 20
        + _leg(100, 130, 10) + _leg(130, 110, 8)
        + _leg(110, 160, 12) + _leg(160, 120, 10) + _leg(120, 150, 8)
    )

    class NoTarget(RunnerBitget):
        def get_stop_target(self, symbol, direction):
            return 101.0, None

        def get_mark_price(self, symbol):
            return 122.0

    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0, strategy_tag=tag)
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=101.0,
                          actual_target=None, leverage=1.0)
    storage.set_exit_plan(tid, breakeven_stop=100.0, runner_target=None, partial_fraction=0.5)
    bitget = NoTarget(position=make_position(), closes=closes)
    scanner_ = _runner_scanner(tmp_path, bitget, tags=(tag,))
    scanner_.storage = storage

    trail_level = scanner_.trailing_stop("BTCUSDT", "long", tag, 101.0)
    assert trail_level is not None, "the trail must have something here, or this tests nothing"
    assert (122.0 - 101.0) / 5.0 >= scanner.STALL_TIGHTEN_R, (
        "measured from the OLD stop this must look stalled, or the test cannot catch the bug"
    )

    await scanner_.poll_trailing_stops()

    assert len(bitget.tpsl) == 1
    assert bitget.tpsl[0]["trigger_price"] == pytest.approx(trail_level), (
        "the trail found a swing, so its level stands - the stall rule must not fire on top"
    )


def test_a_rebuilt_exit_plan_still_means_no_runner_target(tmp_path):
    """DOGEUSDT #29, live on 2026-08-20: a Strategy 2.1 1H runner that must
    have NO target got one at 0.08586, off the daily level - which silently
    opted it out of the trail for good. Dror: "it shouldnt have tp".

    This is the UNIUSDT bug of 2026-08-19 coming back through a door its fix
    never covered. ema_trend_v2 sets remainder_target_is_final=True on the
    Signal, and the only test for it asserts on that Signal - but the Signal
    object is long gone by the time a partial fills. _exit_plan_signal rebuilds
    one from the trade row, and the flag was not among the fields it carried,
    so it defaulted to False: None plus not-final reads as "no opinion, use
    the daily level", which is exactly the fall-through the flag exists to stop.

    So the plan has to record that "no runner target" was a DECISION, and the
    rebuilt signal has to carry it.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0,
                                 strategy_tag="Strategy 2.1 1H")
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=95.0,
                          actual_target=None, leverage=1.0)
    storage.set_exit_plan(tid, breakeven_stop=100.0, runner_target=None, partial_fraction=0.5,
                          runner_target_is_final=True)
    scanner_ = _runner_scanner(tmp_path, RunnerBitget(position=make_position(), closes=_swinging_closes()))
    scanner_.storage = storage

    signal = scanner_._exit_plan_signal(storage.get_trade(tid))

    assert signal.remainder_target_is_final is True, "the decision must survive the rebuild"
    target, _note = scanner_.runner_target(signal, fallback=None)
    assert target is None, "a 2.1 runner trails; no daily level may be invented for it"


def test_a_plan_recorded_without_the_decision_still_invents_a_target(tmp_path):
    """The other side of the same coin, kept so the fix cannot be quietly
    reverted to "always final". A trade whose plan does NOT record the
    decision - Strategy 1, which really does want the daily level - must keep
    getting one. If this ever starts returning None, the flag has stopped
    meaning anything and 2.1's protection is accidental."""
    storage = Storage(str(tmp_path / "trades.db"))
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", proposed_stop=95.0,
                                 strategy_tag="Strategy 1 1H")
    storage.confirm_entry(tid, entry_price=100.0, position_size=2.0, actual_stop=95.0,
                          actual_target=None, leverage=1.0)
    storage.set_exit_plan(tid, breakeven_stop=100.0, runner_target=None, partial_fraction=0.5)
    scanner_ = _runner_scanner(tmp_path, RunnerBitget(position=make_position(), closes=_swinging_closes()))
    scanner_.storage = storage

    signal = scanner_._exit_plan_signal(storage.get_trade(tid))

    assert signal.remainder_target_is_final is False
    target, _note = scanner_.runner_target(signal, fallback=None)
    assert target is not None, "a strategy that never asked to trail still aims at the daily level"


async def test_a_symbol_capped_below_10x_is_sized_at_its_own_ceiling(tmp_path):
    """BTWUSDT, live 2026-08-21: EXECUTION FAILED, "Exceeded the maximum
    settable leverage" (Bitget 40797). Its maxLever is 5; the bot asked for 10.

    risk_sizing.MIN_LEVERAGE is a FLOOR of 10, so on a symbol capped below it
    the plan could only ever request something the exchange refuses - the trade
    could NEVER be placed, whatever the account looked like. 17 of Bitget's 759
    contracts sit under 10x: mostly tokenized stocks (XIAOMI, MEITUAN, NETEASE,
    KUAISHOU, SMIC, GIGADEVICE, QNTSTOCK) plus HUSDT at 4x.

    plan_position already resolves the cap correctly when it is GIVEN one -
    min(max(n/budget, 10), 5) is 5 - so nothing there was broken. What was
    missing is that get_contract_specs never read maxLever and the scanner
    passed one global ceiling for every symbol.
    """
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(
        storage, FakeBitget(position=make_position(), max_leverage=5.0), bot
    )

    await scanner.tick()

    assert bot.sent, "the signal should still be dispatched, just sized differently"
    assert "@ 5.0x" in bot.sent[0], (
        "the alert must quote the leverage the exchange will actually accept: %s"
        % bot.sent[0]
    )
    assert "@ 10.0x" not in bot.sent[0]


# ---------------------------------------------------------------------------
# Exit targets re-anchored on the real fill, not the pre-fill plan_entry.
# ---------------------------------------------------------------------------

class ReanchorSplitStrategy(AlwaysFireStrategy):
    """A 1:2 split-entry strategy, for testing that its targets follow the
    REAL fill rather than staying frozen at the pre-fill blended estimate."""

    tag = "reanchor_split"

    def evaluate(self, symbol, bars_by_timeframe):
        signal = super().evaluate(symbol, bars_by_timeframe)
        signal.reward_risk_ratio = 2.0
        signal.limit_entry = signal.entry_price  # 100.0
        signal.limit_note = "test limit"
        return signal


class RecordingTpslBitget(PartialRaceBitget):
    """PartialRaceBitget's place_order/place_tpsl_order (entries always
    succeed; place_tpsl_order counts attempts), plus a record of every
    trigger_price actually placed, keyed by plan_type."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tpsl_calls = []

    def place_tpsl_order(self, **kw):
        self.tpsl_calls.append(kw)
        return super().place_tpsl_order(**kw)


def _reanchor_scanner(tmp_path, bitget, strategy, bot=None):
    from execution.executor import LiveExecutor

    return Scanner(
        bitget=bitget,
        bot=bot or FakeBot(),
        storage=Storage(str(tmp_path / "trades.db")),
        executor=LiveExecutor(bitget),
        watchlist=["BTCUSDT"],
        strategies=[strategy],
        risk_pct=0.01,
        auto_execute_tags={strategy.tag},
    )


async def test_both_exit_targets_follow_the_real_fill_not_the_pre_fill_plan(tmp_path):
    """SNDKUSDT #54: entry 100 (Fib), stop 95, planned blend 100.20 (20% at a
    101 market + 80% at the 100 limit) - but the REAL average fill came back at
    103, because the market leg filled worse than the mark price used to plan
    it. plan.take_profit and the runner target were computed ONCE at dispatch,
    from the 100.20 estimate, and never touched again even though
    position["entry_price"] is read and correctly used for the BREAKEVEN three
    lines above in the same function - the exact case its own comment already
    names ("plan_entry is an estimate... the confirmed position knows better")
    and had simply not been applied to the two prices that matter most.

    Old (buggy): target1 = 100.20 + 2*(100.20-95) = 110.60
                 runner  = 100.20 + 3*(100.20-95) = 115.80
    Fixed:       target1 = 103.00 + 2*(103.00-95) = 119.00
                 runner  = 103.00 + 3*(103.00-95) = 127.00
    """
    class MarketAt101(RecordingTpslBitget):
        def get_mark_price(self, symbol):
            return 101.0

    bitget = MarketAt101(position=make_position(entry_price=103.0, stop=95.0))
    live_scanner = _reanchor_scanner(tmp_path, bitget, ReanchorSplitStrategy())

    await live_scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    profit_calls = [c for c in bitget.tpsl_calls if c.get("plan_type") == "profit_plan"]
    assert profit_calls, "the partial take-profit must have been placed"
    assert profit_calls[0]["trigger_price"] == pytest.approx(119.0), (
        "target1 must be re-derived from the real 103 fill, not the 100.20 plan"
    )

    trade = live_scanner.storage.get_trade(1)
    assert trade.runner_target == pytest.approx(127.0), (
        "the stored runner fallback must also follow the real fill"
    )


async def test_a_fill_that_matches_the_plan_leaves_targets_unchanged(tmp_path):
    """When the real average fill happens to land exactly on plan_entry - the
    ordinary case, no gap between planning and filling - re-deriving against
    it must be a no-op. market_fraction defaults to 0.2, get_mark_price
    defaults to 100.0, and limit_entry is 100.0, so plan_entry is 100.0
    exactly, matching the real fill given here."""
    bitget = RecordingTpslBitget(position=make_position(entry_price=100.0, stop=95.0))
    live_scanner = _reanchor_scanner(tmp_path, bitget, ReanchorSplitStrategy())

    await live_scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    profit_calls = [c for c in bitget.tpsl_calls if c.get("plan_type") == "profit_plan"]
    assert profit_calls, "the partial take-profit must have been placed"
    assert profit_calls[0]["trigger_price"] == pytest.approx(110.0)


async def test_a_self_managed_remainder_target_is_never_re_derived(tmp_path):
    """A strategy that sets its own partial_fraction owns its runner price -
    an absolute level (or None, meaning trail) - and the real-fill re-anchor
    must not touch it. Only the scanner's OWN ratio-derived fallback
    (partial_fraction is None) is corrected."""
    class SelfManagedStrategy(AlwaysFireStrategy):
        tag = "self_managed"

        def evaluate(self, symbol, bars_by_timeframe):
            signal = super().evaluate(symbol, bars_by_timeframe)
            signal.reward_risk_ratio = 2.0
            signal.limit_entry = signal.entry_price
            signal.partial_fraction = 0.5
            signal.remainder_target = None  # "trail", the strategy's own choice
            signal.remainder_target_is_final = False
            return signal

    bitget = RecordingTpslBitget(position=make_position(entry_price=103.0, stop=95.0))
    live_scanner = _reanchor_scanner(tmp_path, bitget, SelfManagedStrategy())

    await live_scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    # The partial itself still gets placed (partial_fraction owns SIZE, not
    # target) and must ALSO follow the real fill - target1 is never the
    # strategy's own price, only the runner (remainder_target) can be.
    profit_calls = [c for c in bitget.tpsl_calls if c.get("plan_type") == "profit_plan"]
    assert profit_calls, "the partial take-profit must have been placed"
    assert profit_calls[0]["trigger_price"] == pytest.approx(119.0)  # 103 + 2*(103-95)

    trade = live_scanner.storage.get_trade(1)
    assert trade.runner_target is None, "the strategy's own None (trail) must survive untouched"
