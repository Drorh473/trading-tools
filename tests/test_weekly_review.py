import sqlite3
from datetime import date, timedelta

import pytest

from core.storage import Storage
from weekly_review.analyze import analyze, start_of_week


def _backdate(db_path, trade_id, when: date):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE trades SET תאריך = ? WHERE מספר_עסקה = ?", (when.isoformat(), trade_id))
    conn.commit()
    conn.close()


def _open_and_close(storage, symbol, direction, entry_price, size, stop, target, exit_price, strategy_tag):
    trade_id = storage.create_pending(symbol=symbol, direction=direction, strategy_tag=strategy_tag)
    storage.confirm_entry(trade_id, entry_price=entry_price, position_size=size, actual_stop=stop, actual_target=target)
    storage.close_trade(trade_id, exit_price=exit_price)
    return trade_id


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)

    week_start = start_of_week()
    last_week = week_start - timedelta(days=1)

    # closed trade from before this week: a loss on strategy A
    old = _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=90, strategy_tag="A")
    _backdate(db_path, old, last_week)

    # this week: two wins on strategy B
    _open_and_close(s, "ETHUSDT", "long", 50, 2, 45, 60, exit_price=60, strategy_tag="B")
    _open_and_close(s, "SOLUSDT", "short", 20, 5, 22, 14, exit_price=14, strategy_tag="B")

    return s


def test_weekly_comparison(storage):
    comparison = analyze(storage)

    assert comparison.all_time.total_closed == 3
    assert comparison.this_week.total_closed == 2
    assert comparison.this_week.win_rate == 1.0
    assert comparison.best_strategy_this_week == "B"
    assert comparison.current_streak_len == 2
    assert comparison.current_streak_type == "win"
    assert comparison.win_rate_delta == pytest.approx(1.0 - 2 / 3)


def test_weekly_comparison_no_trades_this_week(tmp_path):
    db_path = str(tmp_path / "trades.db")
    s = Storage(db_path)
    week_start = start_of_week()
    old = _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=120, strategy_tag="A")
    _backdate(db_path, old, week_start - timedelta(days=1))

    comparison = analyze(s)
    assert comparison.this_week.total_closed == 0
    assert comparison.best_strategy_this_week is None
