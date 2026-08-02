import sqlite3
from datetime import datetime, timezone

import pytest

from core.storage import Storage
from journal.paper_sim import resolve_pending

DISPATCHED = "2020-01-01T00:00:00+00:00"
BASE_MS = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


class FakeBitget:
    def __init__(self, candles):
        self._candles = candles

    def get_candles(self, symbol, granularity="15m", limit=1000, closed_only=True):
        return self._candles


def _candle(ts_ms, o, h, l, c):
    return [str(ts_ms), str(o), str(h), str(l), str(c), "1", "1"]


def _log_signal_at(storage, dispatched_at, direction="long", **levels):
    signal_id = storage.log_signal(
        symbol="BTCUSDT",
        direction=direction,
        entry_price=levels["entry_price"],
        stop_loss=levels["stop_loss"],
        take_profit=levels["take_profit"],
        strategy_tag="test",
    )
    # log_signal always stamps the real current time; tests need a fixed,
    # known dispatch time to build deterministic candles around.
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("UPDATE signals SET dispatched_at = ? WHERE id = ?", (dispatched_at, signal_id))
    return signal_id


def test_resolves_a_long_that_hits_target(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
    bitget = FakeBitget([_candle(BASE_MS + 900_000 * i, 100, 111, 99, 105) for i in range(3)])

    resolved = resolve_pending(storage, bitget)

    assert resolved == 1
    # raw R = (110-100)/(100-95) = 2.0, fee_r = (2*0.0006*100)/5 = 0.024
    assert storage.read_signals()[0].paper_r == pytest.approx(2.0 - 0.024)


def test_resolves_a_long_that_hits_stop(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
    bitget = FakeBitget([_candle(BASE_MS, 100, 101, 94, 96)])

    resolve_pending(storage, bitget)

    assert storage.read_signals()[0].paper_r == pytest.approx(-1.0 - 0.024)


def test_leaves_a_signal_unresolved_when_neither_level_is_reached(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
    bitget = FakeBitget([_candle(BASE_MS, 100, 102, 98, 101)])

    resolved = resolve_pending(storage, bitget)

    assert resolved == 0
    assert storage.read_signals()[0].paper_r is None


def test_leaves_unresolved_when_no_candles_exist_since_dispatch(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
    bitget = FakeBitget([])

    assert resolve_pending(storage, bitget) == 0


def test_short_direction_resolves_against_the_mirrored_levels(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0)
    bitget = FakeBitget([_candle(BASE_MS, 100, 101, 89, 90)])

    resolve_pending(storage, bitget)

    # raw R = (100-90)/(105-100) = 2.0, fee_r = (2*0.0006*100)/5 = 0.024
    assert storage.read_signals()[0].paper_r == pytest.approx(2.0 - 0.024)


def test_a_bar_hitting_both_levels_conservatively_counts_as_the_stop(tmp_path):
    # OHLC alone can't say which level came first inside one bar; the report
    # should never be flattered by an ambiguous case, so the stop wins.
    storage = Storage(str(tmp_path / "trades.db"))
    _log_signal_at(storage, DISPATCHED, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
    bitget = FakeBitget([_candle(BASE_MS, 100, 112, 94, 105)])

    resolve_pending(storage, bitget)

    assert storage.read_signals()[0].paper_r == pytest.approx(-1.0 - 0.024)
