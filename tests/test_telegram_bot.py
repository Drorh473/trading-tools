import asyncio

from core import telegram_bot
from core.telegram_bot import NotifierBot


class FakeMessage:
    def __init__(self, text):
        self.text = text
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)
        self.text = text


class FakeBotApi:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        message = FakeMessage(text)
        self.sent.append((message, reply_markup))
        return message


def _bot(monkeypatch):
    monkeypatch.setattr(telegram_bot.Application, "builder", _stub_builder)
    return NotifierBot("token", "chat")


class _StubApp:
    def __init__(self):
        self.bot = FakeBotApi()
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)


class _StubBuilder:
    def token(self, _):
        return self

    def build(self):
        return _StubApp()


def _stub_builder():
    return _StubBuilder()


# Every test below sends against a 100 -> 90 long-style setup (entry 100,
# stop 90, so 1R = 10) unless it cares specifically about the movement
# cutoff, in which case it says so.


async def test_offer_expires_on_the_timer_and_can_no_longer_be_approved(monkeypatch):
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: approved.append(True),
        expiry_seconds=0.01,
        entry_price=100.0,
        stop_loss=90.0,
        price_fetcher=lambda: 100.0,  # never moves; only the timer can end this
    )
    assert len(bot._pending) == 1

    await asyncio.sleep(0.05)

    # The offer is gone, so a stale button press finds nothing to run.
    assert bot._pending == {}
    assert approved == []
    message, _ = bot.app.bot.sent[0]
    assert "Expired — not acted on within" in message.text


async def test_offer_expires_early_when_price_moves_past_the_movement_threshold(monkeypatch):
    """The whole point of this redesign: a slow timeframe's timer ceiling is
    generous (up to 30 minutes), so on its own it would leave this offer live
    for `expiry_seconds` below. The movement cutoff is what actually ends it
    in practice, well before the timer would. Poll interval is monkeypatched
    down so the test doesn't need to wait on real 15s polling.
    """
    monkeypatch.setattr(telegram_bot, "SIGNAL_POLL_SECONDS", 0.01)
    bot = _bot(monkeypatch)
    approved = []

    # entry 100, stop 90 -> 1R = 10. Price at 98.4 is 1.6/10 = 0.16R moved,
    # just over the 0.15R threshold.
    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: approved.append(True),
        expiry_seconds=5.0,  # generous timer; movement should end this first
        entry_price=100.0,
        stop_loss=90.0,
        price_fetcher=lambda: 98.4,
    )

    await asyncio.sleep(0.05)

    assert bot._pending == {}
    assert approved == []
    message, _ = bot.app.bot.sent[0]
    assert "Expired — price moved 0.16R since the signal." in message.text


async def test_ordinary_movement_under_the_threshold_does_not_expire_early(monkeypatch):
    """Verified against the pre-fix behaviour directly: reverting the
    threshold check to fire on ANY movement (not just >= 0.15R) makes this
    fail, since 99.2 is only 0.08R off entry - comfortably under the cutoff.
    """
    monkeypatch.setattr(telegram_bot, "SIGNAL_POLL_SECONDS", 0.01)
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: approved.append(True),
        expiry_seconds=0.05,  # timer is the only thing that should end this
        entry_price=100.0,
        stop_loss=90.0,
        price_fetcher=lambda: 99.2,  # 0.08R - under the 0.15R threshold
    )

    await asyncio.sleep(0.1)

    assert bot._pending == {}
    message, _ = bot.app.bot.sent[0]
    assert "Expired — not acted on within" in message.text  # timer, not movement


async def test_a_price_fetcher_error_is_skipped_not_treated_as_expiry(monkeypatch):
    """A failed poll must not kill a live offer - it should just be retried
    on the next tick, same spirit as every other best-effort Telegram call
    in this module."""
    monkeypatch.setattr(telegram_bot, "SIGNAL_POLL_SECONDS", 0.01)
    bot = _bot(monkeypatch)
    approved = []

    def _boom():
        raise RuntimeError("mark price fetch failed")

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: approved.append(True),
        expiry_seconds=0.03,
        entry_price=100.0,
        stop_loss=90.0,
        price_fetcher=_boom,
    )

    await asyncio.sleep(0.06)

    # Still expires eventually (on the timer), just never crashes along the way.
    assert bot._pending == {}
    message, _ = bot.app.bot.sent[0]
    assert "Expired — not acted on within" in message.text


class FakeQuery:
    """A button press. `answer_error` / `edit_error` make Telegram reject the
    two cosmetic calls the handler makes around the decision itself.
    """

    def __init__(self, data, text, answer_error=None, edit_error=None):
        self.data = data
        self.message = FakeMessage(text)
        self.answer_error = answer_error
        self.edit_error = edit_error
        self.edits = []

    async def answer(self):
        if self.answer_error:
            raise self.answer_error

    async def edit_message_text(self, text, **kwargs):
        if self.edit_error:
            raise self.edit_error
        self.edits.append(text)


class FakeUpdate:
    def __init__(self, query):
        self.callback_query = query


async def _send(bot, on_approve=None, on_reject=None, **overrides):
    kwargs = dict(
        expiry_seconds=5.0,
        entry_price=100.0,
        stop_loss=90.0,
        price_fetcher=lambda: 100.0,
    )
    kwargs.update(overrides)
    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        on_approve or (lambda: None),
        on_reject,
        **kwargs,
    )


async def test_a_late_tap_still_executes_when_telegram_rejects_answer(monkeypatch):
    """Telegram's "Query is too old" must not swallow a valid approval.

    The offer is still in `_pending` - the expiry window has not passed -
    so the decision is live even though Telegram will not take the spinner ack.
    Before the fix, `answer()` raised first and the trade was never placed.
    """
    bot = _bot(monkeypatch)
    approved = []

    await _send(bot, on_approve=lambda: approved.append(True))
    query = FakeQuery("approve:0", "Signal: BTCUSDT LONG", answer_error=Exception("Query is too old"))

    await bot._on_callback(FakeUpdate(query), None)

    assert approved == [True]
    assert bot._pending == {}
    assert "Approved." in query.edits[0]


async def test_a_failed_edit_does_not_raise_after_the_trade_is_placed(monkeypatch):
    """The confirmation edit is the last step and the least important one."""
    bot = _bot(monkeypatch)
    approved = []

    await _send(bot, on_approve=lambda: approved.append(True))
    query = FakeQuery("approve:0", "Signal: BTCUSDT LONG", edit_error=Exception("message to edit not found"))

    await bot._on_callback(FakeUpdate(query), None)  # must not raise

    assert approved == [True]


async def test_rejecting_still_runs_its_handler_when_answer_fails(monkeypatch):
    bot = _bot(monkeypatch)
    rejected = []

    await _send(bot, on_reject=lambda: rejected.append(True))
    query = FakeQuery("reject:0", "Signal: BTCUSDT LONG", answer_error=Exception("Query is too old"))

    await bot._on_callback(FakeUpdate(query), None)

    assert rejected == [True]


async def test_acting_in_time_prevents_expiry(monkeypatch):
    bot = _bot(monkeypatch)
    approved = []

    await _send(bot, on_approve=lambda: approved.append(True), expiry_seconds=0.05)
    # Simulate the user pressing Approve before the timer fires.
    pending = bot._pending.pop("0")
    pending.on_approve()

    await asyncio.sleep(0.1)

    message, _ = bot.app.bot.sent[0]
    assert approved == [True]
    assert "Expired" not in message.text  # nothing to expire; it was handled
