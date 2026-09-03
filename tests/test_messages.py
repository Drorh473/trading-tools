from core.storage import Storage
from execution.messages import format_close_message, format_partial_message, format_scale_in_message


def test_scale_in_message_states_the_new_position_without_deltas(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="AAPLUSDT", direction="short", proposed_stop=310.94)
    storage.confirm_entry(
        trade_id, entry_price=309.03, position_size=0.51, actual_stop=310.94, actual_target=None, leverage=10.0
    )

    text = format_scale_in_message(storage.get_trade(trade_id), covered=0.51)

    assert "limit leg filled" in text
    assert "0.51 @ 309.03" in text
    assert "stop 310.94" in text
    assert "Breakeven is now 309.03" in text
    assert "Take-profit covers 0.51 of 0.51" in text
    # New state only - Dror's standing preference, same call he made for the
    # weekly report's real-trades section.
    assert "→" not in text and "->" not in text


def test_scale_in_message_survives_a_sub_penny_symbol(tmp_path):
    """The older formatters use .2f, which prints PEPEUSDT's entry as 0.00."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="PEPEUSDT", direction="long", proposed_stop=2.7267e-06)
    storage.confirm_entry(
        trade_id, entry_price=2.8377e-06, position_size=3262000, actual_stop=2.7267e-06,
        actual_target=None, leverage=10.0,
    )

    text = format_scale_in_message(storage.get_trade(trade_id), covered=None)

    assert "2.8377e-06" in text
    assert "0.00 " not in text
    assert "Take-profit covers" not in text  # omitted when coverage is unknown


def _short_with_partial(tmp_path, breakeven=None):
    """The APTUSDT short of 2026-08-13, at the point its partial filled."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short", proposed_stop=0.6312)
    storage.confirm_entry(
        trade_id, entry_price=0.6134, position_size=70.019,
        actual_stop=0.6312, actual_target=None, leverage=10.0,
    )
    if breakeven is not None:
        storage.set_exit_plan(trade_id, breakeven_stop=breakeven, runner_target=0.58, partial_fraction=0.5)
    return storage.get_trade(trade_id)


def test_the_partial_message_no_longer_claims_a_stop_move_nobody_made(tmp_path):
    """The line it used to end on - "stop should already be at entry (0.61)" -
    was unconditional, so it printed on trades that had no breakeven handler
    attached at all. That is how APTUSDT #11 rode its remainder on the
    original stop while the message said otherwise."""
    trade = _short_with_partial(tmp_path)  # no exit plan recorded

    text = format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)

    assert "should already be at entry" not in text
    assert "does NOT manage this trade" in text
    assert "by hand" in text


def test_the_partial_message_states_the_move_when_the_bot_owns_the_exits(tmp_path):
    trade = _short_with_partial(tmp_path, breakeven=0.6134)

    text = format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)

    assert "0.6134 breakeven" in text
    assert "by hand" not in text


def test_the_partial_message_prints_the_breakeven_at_full_precision(tmp_path):
    """.2f rounded this 0.6134 short's breakeven to 0.61 - 0.8% of the price,
    on a stop, which is the difference between breakeven and a small loss."""
    trade = _short_with_partial(tmp_path, breakeven=0.6134)

    assert "0.61)" not in format_partial_message(trade, closed_size=35.01, realized_pnl=1.6564)


def _closed_apt(tmp_path):
    """#11 as it actually closed."""
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(symbol="APTUSDT", direction="short")
    storage.confirm_entry(
        trade_id, entry_price=0.608112826518, position_size=70.019,
        actual_stop=0.6285, actual_target=None, leverage=10.0,
    )
    storage.close_trade(trade_id, exit_price=0.5485, realized_pnl=4.17908288)
    return storage.get_trade(trade_id)


def test_the_close_message_breaks_out_each_exit(tmp_path):
    """"Exit: 0.55" described a price nothing traded at: it is the average of
    a 0.5608 partial and a 0.5360 runner."""
    exits = [
        {"size": 35.010, "price": 0.5608, "profit": 1.65642205, "at": 1},
        {"size": 35.009, "price": 0.5360, "profit": 2.52459794, "at": 2},
    ]

    text = format_close_message(_closed_apt(tmp_path), exits)

    assert "35.01 @ 0.5608" in text
    assert "35.009 @ 0.536" in text
    assert "avg of the closes below" in text


def test_the_close_message_does_not_call_a_single_exit_an_average(tmp_path):
    """With one close the average IS the fill, so the breakdown is noise."""
    text = format_close_message(_closed_apt(tmp_path), [{"size": 70.019, "price": 0.5485, "profit": 4.0, "at": 1}])

    assert "avg of the closes below" not in text
    assert "@ 0.5485" not in text


def test_the_close_message_prints_prices_at_full_precision(tmp_path):
    """.2f rendered this entry as 0.61 and this exit as 0.55."""
    text = format_close_message(_closed_apt(tmp_path))

    assert "0.608113" in text and "0.5485" in text
    assert "Entry: 0.61 " not in text
