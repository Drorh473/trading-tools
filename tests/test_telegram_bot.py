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


async def _settle(bot):
    """Wait for the offer's own expiry task, however long the machine takes.

    This used to be `await asyncio.sleep(0.1)` - a guess at how long the
    background task would need. The expiry loop does not read a clock; it
    counts nominal steps, sleeping SIGNAL_POLL_SECONDS at a time. So under CPU
    load its five wakeups take far longer in real time than the test's single
    one, the assertion runs before the task has finished, and a different
    subset of these tests fails on every run.

    Waiting on the task itself is exact: it cannot return early and it cannot
    time out, whatever else the machine is doing. A decision cancels the
    timer, so a settled offer resolves here immediately rather than ticking
    out its whole ceiling.
    """
    tasks = list(bot._expiry_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


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
        reference_price=100.0,
        price_fetcher=lambda: 100.0,  # never moves; only the timer can end this
    )
    assert len(bot._pending) == 1

    await _settle(bot)

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

    # entry 100, stop 90 -> 1R = 10. Market started AT the entry, so the
    # starting gap is 0 and price at 98.4 has drifted 1.6/10 = 0.16R away,
    # just over the 0.15R threshold.
    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: approved.append(True),
        expiry_seconds=5.0,  # generous timer; movement should end this first
        entry_price=100.0,
        stop_loss=90.0,
        reference_price=100.0,
        price_fetcher=lambda: 98.4,
    )

    await _settle(bot)

    assert bot._pending == {}
    assert approved == []
    message, _ = bot.app.bot.sent[0]
    assert "Expired — price drifted 0.16R further from the entry." in message.text


async def test_price_moving_toward_a_resting_entry_never_expires_the_offer(monkeypatch):
    """Dror's rule, from INJUSDT: for an order resting away from market,
    price travelling TOWARD it is the setup working, not decaying.

    Strategy 2 enters 100% on a limit at EMA9. Here that limit is 100 with
    the market down at 94 - it must RISE to fill at all. Price climbing to
    97.5 more than halves the gap, yet still sits 0.25R from the entry in
    absolute terms - so the pre-fix "distance from entry" rule would have
    killed this offer even though it was moving the right way. The timer is
    what ends it instead. (Verified: reverting the drift logic makes this
    fail, not just the standing-gap test below.)
    """
    monkeypatch.setattr(telegram_bot, "SIGNAL_POLL_SECONDS", 0.01)
    bot = _bot(monkeypatch)

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: None,
        expiry_seconds=0.05,
        entry_price=100.0,
        stop_loss=90.0,
        reference_price=94.0,  # market started 6 points (0.6R) BELOW the resting entry
        price_fetcher=lambda: 97.5,  # closed to 2.5 points (0.25R) - moving toward it
    )

    await _settle(bot)

    message, _ = bot.app.bot.sent[0]
    assert "not acted on within" in message.text, "movement toward the entry must not expire the offer"


async def test_the_starting_gap_does_not_itself_count_as_drift(monkeypatch):
    """A resting limit is far from market BY CONSTRUCTION, so measuring raw
    distance would expire such a signal the instant it fired.

    INJUSDT dispatched with the market 2.2R away from its own limit entry.
    Only distance ADDED after dispatch counts; here price hasn't moved at
    all, so nothing has drifted despite the large standing gap.
    """
    monkeypatch.setattr(telegram_bot, "SIGNAL_POLL_SECONDS", 0.01)
    bot = _bot(monkeypatch)

    await bot.send_signal(
        "Signal: BTCUSDT LONG",
        lambda: None,
        expiry_seconds=0.05,
        entry_price=100.0,
        stop_loss=90.0,
        reference_price=78.0,  # 22 points = 2.2R away at dispatch
        price_fetcher=lambda: 78.0,  # and it has not moved since
    )

    await _settle(bot)

    message, _ = bot.app.bot.sent[0]
    assert "not acted on within" in message.text, "the standing gap is not drift"


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
        reference_price=100.0,
        price_fetcher=lambda: 99.2,  # 0.08R - under the 0.15R threshold
    )

    await _settle(bot)

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
        reference_price=100.0,
        price_fetcher=_boom,
    )

    await _settle(bot)

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
        reference_price=100.0,
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

    await _settle(bot)

    message, _ = bot.app.bot.sent[0]
    assert approved == [True]
    assert "Expired" not in message.text  # nothing to expire; it was handled


async def test_cancel_all_pending_marks_every_open_offer_dead(monkeypatch):
    """SNXXUSDT #1463: a signal dispatched 33 seconds before a deploy sat
    looking exactly as live as when it was sent for the better part of an
    hour, because nothing edited its message when the restart wiped _pending
    out from under it. This is the fix - called from a SIGTERM handler
    before the process actually exits."""
    bot = _bot(monkeypatch)
    await _send(bot, expiry_seconds=999.0)

    await bot.cancel_all_pending()

    assert bot._pending == {}
    message, _ = bot.app.bot.sent[0]
    assert "Bot restarting" in message.text


async def test_cancel_all_pending_clears_every_offer_not_just_the_first(monkeypatch):
    bot = _bot(monkeypatch)
    await _send(bot, expiry_seconds=999.0)
    await _send(bot, expiry_seconds=999.0)
    await _send(bot, expiry_seconds=999.0)
    assert len(bot._pending) == 3

    await bot.cancel_all_pending()

    assert bot._pending == {}
    assert all("Bot restarting" in msg.text for msg, _ in bot.app.bot.sent)


async def test_cancel_all_pending_cancels_the_expiry_timer_too(monkeypatch):
    """Otherwise a still-sleeping _expire task wakes up later, finds nothing
    in _pending (already cleared) and no-ops - harmless, but a task the
    process never waits on and never explicitly ends is exactly the leak
    _expiry_tasks was introduced to prevent."""
    bot = _bot(monkeypatch)
    await _send(bot, expiry_seconds=999.0)
    task = bot._expiry_tasks["0"]

    await bot.cancel_all_pending()
    await asyncio.sleep(0)  # let the cancellation actually land

    assert task.cancelled() or task.done()


async def test_a_tap_after_cancel_all_pending_is_answered_as_already_handled(monkeypatch):
    """The exact safety net that made this fix low-stakes in the first place:
    a tap on a restart-orphaned offer was always safe, just silent about it.
    This closes the silence, not the safety - confirming the safety net is
    still there is part of the same fix."""
    bot = _bot(monkeypatch)
    await _send(bot, expiry_seconds=999.0)
    await bot.cancel_all_pending()

    query = FakeQuery("approve:0", "Signal: BTCUSDT LONG")
    await bot._on_callback(FakeUpdate(query), None)

    assert "already handled" in query.edits[0]


async def test_cancel_all_pending_keeps_going_after_one_edit_fails(monkeypatch):
    bot = _bot(monkeypatch)
    await _send(bot, expiry_seconds=999.0)
    await _send(bot, expiry_seconds=999.0)
    first_message, _ = bot.app.bot.sent[0]

    async def _broken_edit(text, **kwargs):
        raise Exception("message to edit not found")

    first_message.edit_text = _broken_edit

    await bot.cancel_all_pending()  # must not raise

    assert bot._pending == {}
    second_message, _ = bot.app.bot.sent[1]
    assert "Bot restarting" in second_message.text
