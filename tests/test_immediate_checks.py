"""The two checks split out of the monthly report to fire immediately.

Both exist because a monthly cadence is the wrong latency for them: a dead
instance and an unexplained balance are things you want on day 3, not day 31.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from core import balance_check, ledger
from core.storage import Storage
from notifier.scanner import CAPABILITY_REALERT_SECONDS


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, text, **kwargs):
        self.messages.append(text)


class FakeBitget:
    def __init__(self, equity=100.0, fees=0.0, funding=0.0):
        self.equity, self.fees, self.funding = equity, fees, funding

    def get_account_equity(self):
        return self.equity

    def get_fees_paid(self, a, b):
        return self.fees

    def get_funding_paid(self, a, b):
        return self.funding


class Harness:
    """The two polls only touch storage, bitget and bot, so the real Scanner's
    heavy constructor is not what is under test here."""

    def __init__(self, storage, bitget, bot, expectations=None):
        self.storage, self.bitget, self.bot = storage, bitget, bot
        self.ledger_expectations = expectations or {}

    poll_capability_silence = __import__(
        "notifier.scanner", fromlist=["Scanner"]
    ).Scanner.poll_capability_silence
    poll_balance_divergence = __import__(
        "notifier.scanner", fromlist=["Scanner"]
    ).Scanner.poll_balance_divergence


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "trades.db"))


def _closed(storage, pnl, entry=100.0):
    exit_price = entry + pnl
    tid = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag="Strategy 1 1H")
    storage.confirm_entry(
        tid, entry_price=entry, position_size=1.0, actual_stop=95.0,
        actual_target=110.0, leverage=1.0,
    )
    storage.close_trade(tid, exit_price=exit_price, realized_pnl=pnl)
    return tid


# ---- capability silence ----


@pytest.mark.asyncio
async def test_a_capability_that_never_worked_is_reported_without_waiting_for_sunday(storage):
    """Strategy 3 shipped live and produced zero signals. Until now that fact
    only reached Dror inside the weekly report - and only when the weekly
    report managed to send."""
    db = storage.db_path
    ledger.began_watching(db, datetime.now(timezone.utc) - timedelta(days=30))
    bot = FakeBot()
    h = Harness(storage, FakeBitget(), bot, {ledger.signal_seen("Strategy 3 1D/1H"): 4.0})

    await h.poll_capability_silence()

    assert len(bot.messages) == 1
    assert "CAPABILITY SILENT" in bot.messages[0]
    assert "NEVER worked" in bot.messages[0]


@pytest.mark.asyncio
async def test_a_still_broken_capability_does_not_nag_every_cycle(storage):
    """The untracked-position lesson: six alerts in one afternoon of deploys.
    Reported on the transition, then weekly."""
    db = storage.db_path
    ledger.began_watching(db, datetime.now(timezone.utc) - timedelta(days=30))
    bot = FakeBot()
    h = Harness(storage, FakeBitget(), bot, {ledger.signal_seen("Strategy 3 1D/1H"): 4.0})

    for _ in range(5):
        await h.poll_capability_silence()

    assert len(bot.messages) == 1, "one transition, one alert"


@pytest.mark.asyncio
async def test_it_speaks_again_once_the_re_alert_window_has_passed(storage):
    db = storage.db_path
    ledger.began_watching(db, datetime.now(timezone.utc) - timedelta(days=30))
    bot = FakeBot()
    capability = ledger.signal_seen("Strategy 3 1D/1H")
    h = Harness(storage, FakeBitget(), bot, {capability: 4.0})

    await h.poll_capability_silence()
    storage.record_alerted(
        "__capability__", capability, time.time() - CAPABILITY_REALERT_SECONDS - 1
    )
    await h.poll_capability_silence()

    assert len(bot.messages) == 2


@pytest.mark.asyncio
async def test_recovery_clears_the_throttle_so_a_relapse_alerts_at_once(storage):
    """A capability that breaks, is fixed, then breaks again must not be
    swallowed by the timestamp from the first break."""
    db = storage.db_path
    ledger.began_watching(db, datetime.now(timezone.utc) - timedelta(days=30))
    bot = FakeBot()
    capability = ledger.signal_seen("Strategy 3 1D/1H")
    h = Harness(storage, FakeBitget(), bot, {capability: 4.0})

    await h.poll_capability_silence()          # breaks, alerts
    ledger.record(db, capability)              # recovers
    await h.poll_capability_silence()          # quiet, throttle cleared
    assert storage.last_alerted("__capability__", capability) is None

    # relapse: last success ages out past the threshold
    ledger.record(db, capability, now=datetime.now(timezone.utc) - timedelta(days=10))
    await h.poll_capability_silence()

    assert len(bot.messages) == 2, "a relapse must alert immediately"


@pytest.mark.asyncio
async def test_one_capability_recovering_does_not_re_arm_the_others(storage):
    """The reason clear_alert_throttle_for exists: every capability throttle
    is parked under one synthetic symbol, so clearing by symbol would release
    all of them at once."""
    db = storage.db_path
    ledger.began_watching(db, datetime.now(timezone.utc) - timedelta(days=30))
    bot = FakeBot()
    a, b = ledger.signal_seen("Strategy 3 1D/1H"), ledger.signal_seen("Strategy 4 4H")
    h = Harness(storage, FakeBitget(), bot, {a: 4.0, b: 4.0})

    await h.poll_capability_silence()
    assert len(bot.messages) == 2

    ledger.record(db, a)                       # only A recovers
    await h.poll_capability_silence()

    assert len(bot.messages) == 2, "B was still broken and already reported"


# ---- balance divergence ----


@pytest.mark.asyncio
async def test_nothing_happens_while_a_position_is_open(storage):
    """The whole premise is that flat-to-flat has no unrealized P&L in it."""
    storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag="Strategy 1 1H")
    h = Harness(storage, FakeBitget(), FakeBot())

    await h.poll_balance_divergence()

    assert balance_check.load(storage.db_path) is None


@pytest.mark.asyncio
async def test_the_first_flat_moment_only_records_a_checkpoint(storage):
    h = Harness(storage, FakeBitget(equity=100.0), FakeBot())

    await h.poll_balance_divergence()

    checkpoint = balance_check.load(storage.db_path)
    assert checkpoint is not None and checkpoint.equity == 100.0
    assert h.bot.messages == []


@pytest.mark.asyncio
async def test_a_matching_account_says_nothing(storage):
    """Won $10, paid $1.34 in fees and $0.11 funding, so equity should be up
    $8.55. It is. Silence is correct."""
    db = storage.db_path
    balance_check.save(db, balance_check.Checkpoint(at_ms=0, equity=100.0, cumulative_realized=0.0))
    _closed(storage, pnl=10.0)
    h = Harness(storage, FakeBitget(equity=108.55, fees=1.34, funding=0.11), FakeBot())

    await h.poll_balance_divergence()

    assert h.bot.messages == []


@pytest.mark.asyncio
async def test_money_that_moved_for_an_unmodelled_reason_is_reported(storage):
    db = storage.db_path
    balance_check.save(db, balance_check.Checkpoint(at_ms=0, equity=100.0, cumulative_realized=0.0))
    _closed(storage, pnl=10.0)
    # $3 short of where it should be, with flat books at both ends.
    h = Harness(storage, FakeBitget(equity=105.55, fees=1.34, funding=0.11), FakeBot())

    await h.poll_balance_divergence()

    assert len(h.bot.messages) == 1
    assert "BALANCE DIVERGENCE" in h.bot.messages[0]
    assert "-3.00" in h.bot.messages[0]


@pytest.mark.asyncio
async def test_a_reported_divergence_is_absorbed_into_the_new_baseline(storage):
    """Otherwise one unexplained dollar is re-reported against every future
    window, forever."""
    db = storage.db_path
    balance_check.save(db, balance_check.Checkpoint(at_ms=0, equity=100.0, cumulative_realized=0.0))
    _closed(storage, pnl=10.0)
    h = Harness(storage, FakeBitget(equity=105.55, fees=1.34, funding=0.11), FakeBot())

    await h.poll_balance_divergence()
    balance_check.save(
        db,
        balance_check.Checkpoint(
            at_ms=0,  # backdate so MIN_INTERVAL does not suppress the second run
            equity=balance_check.load(db).equity,
            cumulative_realized=balance_check.load(db).cumulative_realized,
        ),
    )
    h.bitget.fees = h.bitget.funding = 0.0
    await h.poll_balance_divergence()

    assert len(h.bot.messages) == 1, "the same divergence must not be re-reported"


@pytest.mark.asyncio
async def test_it_does_not_re_reconcile_an_empty_window(storage):
    """A flat account would otherwise re-run the fee and funding calls on
    every upkeep cycle to reconcile a window containing nothing."""
    db = storage.db_path
    balance_check.save(
        db,
        balance_check.Checkpoint(
            at_ms=int(time.time() * 1000), equity=100.0, cumulative_realized=0.0
        ),
    )
    calls = []

    class Counting(FakeBitget):
        def get_account_equity(self):
            calls.append(1)
            return 100.0

    h = Harness(storage, Counting(), FakeBot())
    await h.poll_balance_divergence()

    assert calls == [], "inside MIN_INTERVAL, it must not call the exchange at all"
