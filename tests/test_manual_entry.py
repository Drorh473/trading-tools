from types import SimpleNamespace

import pytest

from core.storage import Storage
from execution.manual_entry import ASK_STRATEGY, make_add_conversation
from telegram.ext import ConversationHandler


class FakeBitget:
    def __init__(self, position=None):
        self._position = position

    def get_position(self, symbol):
        return self._position


def make_position(direction="long", entry_price=63000.0, size=0.05, stop=61000.0, target=67000.0):
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


class FakeMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


def make_update(text=""):
    return SimpleNamespace(message=FakeMessage(text), effective_chat=SimpleNamespace(id=42))


def make_context(args=None):
    return SimpleNamespace(args=args or [], chat_data={}, bot=FakeBot())


def get_handlers(storage, bitget):
    conv = make_add_conversation(storage, bitget)
    handle_add = conv.entry_points[0].callback
    handle_strategy_reply = conv.states[ASK_STRATEGY][0].callback
    return handle_add, handle_strategy_reply


async def test_add_rejects_when_no_position(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    handle_add, _ = get_handlers(storage, FakeBitget(position=None))

    update = make_update()
    result = await handle_add(update, make_context(args=["BTCUSDT"]))

    assert result == ConversationHandler.END
    assert "No open position" in update.message.replies[0]
    assert storage.read_all() == []


async def test_add_rejects_when_already_tracking(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="BTCUSDT", direction="long")
    handle_add, _ = get_handlers(storage, FakeBitget(position=make_position()))

    update = make_update()
    result = await handle_add(update, make_context(args=["BTCUSDT"]))

    assert result == ConversationHandler.END
    assert "Already tracking" in update.message.replies[0]


async def test_add_confirms_position_and_asks_for_strategy(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    handle_add, _ = get_handlers(storage, FakeBitget(position=make_position()))

    update = make_update()
    context = make_context(args=["btcusdt"])
    result = await handle_add(update, context)

    assert result == ASK_STRATEGY
    assert "strategy" in update.message.replies[0].lower()

    [trade] = storage.read_all()
    assert trade.סימבול == "BTCUSDT"
    assert trade.מחיר_כניסה == 63000.0
    assert trade.גודל_פוזיציה == 0.05
    assert trade.תגית_אסטרטגיה is None  # not asked yet
    assert context.chat_data["pending_trade_id"] == trade.מספר_עסקה


async def test_strategy_reply_tags_trade(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(trade_id, entry_price=63000, position_size=0.05, actual_stop=61000, actual_target=67000)

    _, handle_strategy_reply = get_handlers(storage, FakeBitget(position=make_position()))

    update = make_update(text="breakout")
    context = make_context()
    context.chat_data["pending_trade_id"] = trade_id

    result = await handle_strategy_reply(update, context)

    assert result == ConversationHandler.END
    trade = storage.get_trade(trade_id)
    assert trade.תגית_אסטרטגיה == "breakout"
    assert "pending_trade_id" not in context.chat_data
