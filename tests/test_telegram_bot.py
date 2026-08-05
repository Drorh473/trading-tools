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


async def test_offer_expires_and_can_no_longer_be_approved(monkeypatch):
    monkeypatch.setattr(telegram_bot, "SIGNAL_EXPIRY_SECONDS", 0.01)
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal("Signal: BTCUSDT LONG", lambda: approved.append(True))
    assert len(bot._pending) == 1

    await asyncio.sleep(0.05)

    # The offer is gone, so a stale button press finds nothing to run.
    assert bot._pending == {}
    assert approved == []
    message, _ = bot.app.bot.sent[0]
    assert "Expired" in message.text


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


async def test_a_late_tap_still_executes_when_telegram_rejects_answer(monkeypatch):
    """Telegram's "Query is too old" must not swallow a valid approval.

    The offer is still in `_pending` - our own 5-minute window has not passed -
    so the decision is live even though Telegram will not take the spinner ack.
    Before the fix, `answer()` raised first and the trade was never placed.
    """
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal("Signal: BTCUSDT LONG", lambda: approved.append(True))
    query = FakeQuery("approve:0", "Signal: BTCUSDT LONG", answer_error=Exception("Query is too old"))

    await bot._on_callback(FakeUpdate(query), None)

    assert approved == [True]
    assert bot._pending == {}
    assert "Approved." in query.edits[0]


async def test_a_failed_edit_does_not_raise_after_the_trade_is_placed(monkeypatch):
    """The confirmation edit is the last step and the least important one."""
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal("Signal: BTCUSDT LONG", lambda: approved.append(True))
    query = FakeQuery("approve:0", "Signal: BTCUSDT LONG", edit_error=Exception("message to edit not found"))

    await bot._on_callback(FakeUpdate(query), None)  # must not raise

    assert approved == [True]


async def test_rejecting_still_runs_its_handler_when_answer_fails(monkeypatch):
    bot = _bot(monkeypatch)
    rejected = []

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        on_approve=lambda: None,
        on_reject=lambda: rejected.append(True),
    )
    query = FakeQuery("reject:0", "Signal: BTCUSDT LONG", answer_error=Exception("Query is too old"))

    await bot._on_callback(FakeUpdate(query), None)

    assert rejected == [True]


async def test_acting_in_time_prevents_expiry(monkeypatch):
    monkeypatch.setattr(telegram_bot, "SIGNAL_EXPIRY_SECONDS", 0.05)
    bot = _bot(monkeypatch)
    approved = []

    await bot.send_signal("Signal: BTCUSDT LONG", lambda: approved.append(True))
    # Simulate the user pressing Approve before the timer fires.
    pending = bot._pending.pop("0")
    pending.on_approve()

    await asyncio.sleep(0.1)

    message, _ = bot.app.bot.sent[0]
    assert approved == [True]
    assert "Expired" not in message.text  # nothing to expire; it was handled
