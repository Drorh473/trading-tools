from types import SimpleNamespace

import pytest
from telegram.ext import ConversationHandler

from core.storage import Storage
from execution.manual_entry import ASK_STRATEGY, make_add_conversation


class FakeBitget:
    def __init__(self, positions=(), stop_target=(61000.0, 67000.0), error=None):
        self._positions = list(positions)
        self._stop_target = stop_target
        self._error = error
        self.last_symbol_queried = None

    def get_positions(self, symbol):
        self.last_symbol_queried = symbol
        if self._error is not None:
            raise self._error
        return list(self._positions)

    def get_position(self, symbol, direction=None):
        matches = [p for p in self._positions if direction is None or p["direction"] == direction]
        return matches[0] if matches else None

    def get_stop_target(self, symbol, direction):
        return self._stop_target

    def find_closed_position(self, symbol, direction):
        return None

    def get_mark_price(self, symbol):
        return 63000.0


def make_position(direction="long", entry_price=63000.0, size=0.05):
    return {
        "symbol": "BTCUSDT",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "stop_loss": None,
        "take_profit": None,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "leverage": 10.0,
        "raw": {},
    }


class FakeMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []
        self.reply_markups = []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append(text)
        self.reply_markups.append(reply_markup)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


def make_update(text=""):
    return SimpleNamespace(message=FakeMessage(text), effective_chat=SimpleNamespace(id=42))


def make_context(args=None):
    return SimpleNamespace(args=args or [], chat_data={}, bot=FakeBot())


DEFAULT_TAG_OPTIONS = ["Strategy 1 1H", "breakout", "Other / discretionary"]


def get_handlers(storage, bitget, tag_options=DEFAULT_TAG_OPTIONS):
    conv = make_add_conversation(storage, bitget, tag_options)
    return conv.entry_points[0].callback, conv.states[ASK_STRATEGY][0].callback


class FakeQuery:
    """A tap on one of the /add tag buttons."""

    def __init__(self, data, text="", answer_error=None, edit_error=None):
        self.data = data
        self.message = FakeMessage(text)
        self.answer_error = answer_error
        self.edit_error = edit_error

    async def answer(self):
        if self.answer_error:
            raise self.answer_error

    async def edit_message_text(self, text, **kwargs):
        if self.edit_error:
            raise self.edit_error
        self.message.text = text


def make_callback_update(query):
    return SimpleNamespace(callback_query=query, effective_chat=SimpleNamespace(id=42))


async def test_add_rejects_when_no_position(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    handle_add, _ = get_handlers(storage, FakeBitget(positions=[]))

    update = make_update()
    result = await handle_add(update, make_context(args=["BTCUSDT"]))

    assert result == ConversationHandler.END
    assert "No open position" in update.message.replies[0]
    assert storage.read_all() == []


async def test_add_rejects_when_already_tracking(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    storage.create_pending(symbol="BTCUSDT", direction="long")
    handle_add, _ = get_handlers(storage, FakeBitget(positions=[make_position()]))

    update = make_update()
    result = await handle_add(update, make_context(args=["BTCUSDT"]))

    assert result == ConversationHandler.END
    assert "Already tracking" in update.message.replies[0]


async def test_add_appends_usdt_suffix_automatically(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(positions=[make_position()])
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    result = await handle_add(update, make_context(args=["uai"]))

    assert bitget.last_symbol_queried == "UAIUSDT"
    assert result == ASK_STRATEGY
    assert storage.read_all()[0].סימבול == "UAIUSDT"


async def test_add_reports_invalid_symbol_distinctly(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(error=RuntimeError("Bitget API error 40034: Parameter UAI does not exist"))
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    result = await handle_add(update, make_context(args=["uai"]))

    assert result == ConversationHandler.END
    assert "isn't a symbol Bitget recognizes" in update.message.replies[0]
    assert storage.read_all() == []


async def test_add_reports_generic_connectivity_failure(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(error=ConnectionError("timed out"))
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    await handle_add(update, make_context(args=["BTCUSDT"]))

    assert "Couldn't reach Bitget" in update.message.replies[0]


async def test_add_asks_for_direction_when_hedged(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(positions=[make_position("long"), make_position("short")])
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    result = await handle_add(update, make_context(args=["BTCUSDT"]))

    assert result == ConversationHandler.END
    assert "long|short" in update.message.replies[0]
    assert storage.read_all() == []


async def test_add_disambiguates_with_direction_argument(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(positions=[make_position("long", 63000.0), make_position("short", 64000.0)])
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    result = await handle_add(update, make_context(args=["btcusdt", "short"]))

    assert result == ASK_STRATEGY
    [trade] = storage.read_all()
    assert trade.כיוון == "short"
    assert trade.מחיר_כניסה == 64000.0


async def test_add_confirms_position_and_pulls_stop_from_plan_orders(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    handle_add, _ = get_handlers(storage, FakeBitget(positions=[make_position()]))

    update = make_update()
    context = make_context(args=["btcusdt"])
    result = await handle_add(update, context)

    assert result == ASK_STRATEGY
    assert "strategy" in update.message.replies[0].lower()

    [trade] = storage.read_all()
    assert trade.סימבול == "BTCUSDT"
    assert trade.מחיר_כניסה == 63000.0
    assert trade.מינוף == 10.0
    # position presets were empty; values came from the plan orders
    assert trade.סטופ_לוס_בפועל == 61000.0
    assert trade.יעד_רווח_בפועל == 67000.0
    assert trade.סכום_סיכון == pytest.approx(abs(63000 - 61000) * 0.05)
    assert trade.תגית_אסטרטגיה is None
    assert context.chat_data["pending_trade_id"] == trade.מספר_עסקה


async def test_add_warns_when_no_stop_anywhere(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    bitget = FakeBitget(positions=[make_position()], stop_target=(None, None))
    handle_add, _ = get_handlers(storage, bitget)

    update = make_update()
    await handle_add(update, make_context(args=["BTCUSDT"]))

    assert "no stop-loss" in update.message.replies[0].lower()
    [trade] = storage.read_all()
    assert trade.סכום_סיכון is None


async def test_strategy_reply_tags_trade(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=63000, position_size=0.05, actual_stop=61000, actual_target=67000, leverage=10.0
    )

    _, handle_strategy_reply = get_handlers(storage, FakeBitget(positions=[make_position()]))

    # "breakout" is index 1 in DEFAULT_TAG_OPTIONS.
    update = make_callback_update(FakeQuery("tag:1", text="Found it — ..."))
    context = make_context()
    context.chat_data["pending_trade_id"] = trade_id

    result = await handle_strategy_reply(update, context)

    assert result == ConversationHandler.END
    assert storage.get_trade(trade_id).תגית_אסטרטגיה == "breakout"
    assert "pending_trade_id" not in context.chat_data
    assert "Tagged as 'breakout'" in update.callback_query.message.text


async def test_strategy_reply_rejects_a_selection_outside_the_list(tmp_path):
    """A stray or stale callback_data must never silently tag a trade with
    something that isn't one of the offered buttons."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=63000, position_size=0.05, actual_stop=61000, actual_target=67000, leverage=10.0
    )

    _, handle_strategy_reply = get_handlers(storage, FakeBitget(positions=[make_position()]))

    update = make_callback_update(FakeQuery("tag:99"))  # out of range for DEFAULT_TAG_OPTIONS
    context = make_context()
    context.chat_data["pending_trade_id"] = trade_id

    result = await handle_strategy_reply(update, context)

    assert result == ConversationHandler.END
    assert storage.get_trade(trade_id).תגית_אסטרטגיה is None


async def test_strategy_reply_offers_a_button_per_tag_option(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    handle_add, _ = get_handlers(
        storage, FakeBitget(positions=[make_position()]), tag_options=DEFAULT_TAG_OPTIONS,
    )

    update = make_update()
    await handle_add(update, make_context(args=["BTCUSDT"]))

    keyboard = update.message.reply_markups[0]
    buttons = [b for row in keyboard.inline_keyboard for b in row]
    assert [b.text for b in buttons] == DEFAULT_TAG_OPTIONS
    assert [b.callback_data for b in buttons] == [f"tag:{i}" for i in range(len(DEFAULT_TAG_OPTIONS))]


async def test_add_routes_partials_through_the_supplied_handler(tmp_path, monkeypatch):
    """A hand-added trade has to take the same scale-out path as every other
    one, or an exit plan armed with /manage is never acted on: this module's
    own callback only sends a message and has no exit handling at all."""
    captured = {}

    def fake_track(*args, **kwargs):
        captured.update(kwargs)

        async def noop():
            return None

        return noop()

    monkeypatch.setattr("execution.manual_entry.track_position", fake_track)

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=63000, position_size=0.05, actual_stop=61000, actual_target=67000, leverage=10.0
    )

    scanner_handler = object()  # stands in for Scanner._on_partial_exit
    conv = make_add_conversation(
        storage, FakeBitget(positions=[make_position()]), DEFAULT_TAG_OPTIONS, on_partial=scanner_handler,
    )
    handle_strategy_reply = conv.states[ASK_STRATEGY][0].callback

    context = make_context()
    context.chat_data["pending_trade_id"] = trade_id
    await handle_strategy_reply(make_callback_update(FakeQuery("tag:0")), context)

    assert captured["on_partial"] is scanner_handler


async def test_add_still_notifies_on_its_own_when_no_handler_is_given(tmp_path, monkeypatch):
    """The fallback keeps this module usable standalone, as its tests do."""
    captured = {}

    def fake_track(*args, **kwargs):
        captured.update(kwargs)

        async def noop():
            return None

        return noop()

    monkeypatch.setattr("execution.manual_entry.track_position", fake_track)

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long")
    storage.confirm_entry(
        trade_id, entry_price=63000, position_size=0.05, actual_stop=61000, actual_target=67000, leverage=10.0
    )

    _, handle_strategy_reply = get_handlers(storage, FakeBitget(positions=[make_position()]))
    context = make_context()
    context.chat_data["pending_trade_id"] = trade_id
    await handle_strategy_reply(make_callback_update(FakeQuery("tag:0")), context)

    assert callable(captured["on_partial"])
