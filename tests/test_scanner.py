import asyncio
import math

import pytest

from core.storage import Storage
from execution.executor import ManualExecutor
from notifier.scanner import (
    SIGNAL_EXPIRY_CEILING,
    SIGNAL_EXPIRY_FLOOR,
    Scanner,
    seconds_until_next_close,
    signal_expiry_seconds,
)
from notifier.strategies.base import Signal, Strategy


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
    ):
        self._position = position
        self._failing_symbols = set(failing_symbols)
        self._equity = equity
        self._specs = {
            "min_size": min_size,
            "min_notional": min_notional,
            "price_place": price_place,
            "volume_place": volume_place,
            "is_rwa": is_rwa,
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

    def get_stop_target(self, symbol, direction):
        return 95.0, 110.0

    def get_contract_specs(self, symbol):
        return self._specs

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
        return []

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
    # what any individual strategy looks at.
    assert scanner.required_timeframes() == {"1H", "15m", "4H"}


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


async def test_alert_uses_exchange_price_precision_and_states_timeframe(tmp_path):
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
    levels = text.split("\n")[2]  # "Entry: ...  Stop: ...  Target: ..."
    entry, stop, target = (part.split()[-1] for part in levels.split("  ") if part.strip())
    assert len({entry, stop, target}) == 3  # three distinct levels, not "0.07" three times
    assert "Analysis timeframe: 1H" in text


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
    assert "Enter: $388 (3.85) at market 101.00" in text
    assert "$1,538 (15.38) limit 100.00 (61.8% Fib)" in text
    # If the resting limit never fills, the market-only fragment needs its own
    # target: the same 10.40 reward distance, re-anchored onto the market fill.
    assert "If the limit leg never fills: exit the market-only 3.85 at 111.40." in text


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
    await asyncio.sleep(0)

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
    consistent: the position is confirmed, but a reduce-only order placed
    right after can still be rejected for a few seconds. `fail_times` controls
    how many 22002 rejections come back before an order is finally accepted;
    `other_error` makes every attempt fail with something that ISN'T the
    settle race, to prove that case is never retried.
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
        if not reduce_only:
            return {}  # the entry legs always succeed; only the partial races
        self.attempts += 1
        if self.other_error:
            raise RuntimeError('Bitget 400 on /api/v2/mix/order/place-order: {"code":"40762","msg":"price deviates too much"}')
        if self.attempts <= self.fail_times:
            raise RuntimeError('Bitget 400 on /api/v2/mix/order/place-order: {"code":"22002","msg":"No position to close"}')
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


async def test_a_non_rwa_symbol_still_uses_the_plain_reduce_only_limit(tmp_path):
    """The RWA branch must not swallow the ordinary path - a crypto symbol's
    take-profit keeps going through place_order exactly as before."""
    bitget = PartialRaceBitget(position=make_position(), is_rwa=False)
    scanner = _live_partial_scanner(tmp_path, bitget)

    await scanner.tick()
    for _ in range(6):
        await asyncio.sleep(0)

    assert bitget.attempts == 1  # the reduce-only place_order path ran


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
    assert "close 15.00 (75%) at 110.00" in text
    assert "close the remaining 5.00 at 130.00 (daily resistance)" in text
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
    assert "Enter: $2,000 (20.00) limit 100.00 (EMA9)" in text
    assert "at market" not in text
    assert "If the limit leg never fills" not in text


async def test_size_line_shows_dollars_quantity_and_leverage(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    # risk 1% of 10k = 100, stop 5% away -> notional 2000, 20 units at 100
    assert "Size: $2,000 (20.00 @ 10.0x)" in bot.sent[0]
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
    # 1% of 10k = 100 risk over a 5% stop -> 2000 notional, the SAME as the
    # unconfirmed case. The second increment is earned by the break, not by
    # the pattern being present.
    assert "$2,000" in text


async def test_no_confluence_leaves_risk_and_message_alone(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bot = FakeBot()
    scanner = build_scanner(storage, FakeBitget(position=make_position()), bot)

    await scanner.tick()

    text = bot.sent[0]
    assert "Confirmed by" not in text
    assert "risk" not in text
    assert "$2,000" in text  # 1% risk, unchanged


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
_POLE_OPENS = [100.0] * 30 + [100.0, 110.0, 118.0, 128.0]
_POLE_CLOSES = [100.0] * 30 + [110.0, 118.0, 128.0, 140.0]
_CONSOLIDATION = [136.0, 130.0, 133.0, 128.0, 131.0, 129.0, 130.5]
COILING_OPENS = _POLE_OPENS + _CONSOLIDATION
COILING_CLOSES = _POLE_CLOSES + _CONSOLIDATION


def _coiling_bitget(opens, closes):
    class PendingBitget(FakeBitget):
        def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
            if granularity == "1H":
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
