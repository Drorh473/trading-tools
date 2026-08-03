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
