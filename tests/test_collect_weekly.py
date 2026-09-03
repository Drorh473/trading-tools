import pandas as pd

from backtest.collect_weekly import TOPUP_BUFFER_BARS, TOPUP_FALLBACK_HOURS, top_up_symbol

COLUMNS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]
HOUR_MS = 3_600_000


def _frame(rows: list[list]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.astype({c: float for c in COLUMNS})
    frame["ts"] = frame["ts"].astype("int64")
    return frame


def _row(ts: int, price: float = 1.0) -> list:
    return [ts, price, price, price, price, 10.0, 10.0]


class FakeClient:
    def __init__(self, rows: list[list]):
        self.rows = rows
        self.calls: list[dict] = []

    def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
        self.calls.append({"symbol": symbol, "granularity": granularity, "limit": limit})
        return self.rows


def test_top_up_symbol_appends_only_bars_newer_than_the_cache():
    last_ts = 10 * HOUR_MS
    existing = _frame([_row(last_ts)])
    now_ms = 13 * HOUR_MS
    # One stale bar (already cached) plus two genuinely new ones.
    fetched = [_row(last_ts), _row(11 * HOUR_MS), _row(12 * HOUR_MS)]
    client = FakeClient(fetched)

    result = top_up_symbol(client, "BTCUSDT", existing, now_ms=now_ms)

    assert list(result["ts"]) == [last_ts, 11 * HOUR_MS, 12 * HOUR_MS]


def test_top_up_symbol_returns_existing_unchanged_when_nothing_new():
    last_ts = 10 * HOUR_MS
    existing = _frame([_row(last_ts)])
    client = FakeClient([_row(last_ts)])  # API has nothing past the cache yet

    result = top_up_symbol(client, "BTCUSDT", existing, now_ms=10 * HOUR_MS + 1)

    assert list(result["ts"]) == [last_ts]


def test_top_up_symbol_dedupes_a_bar_returned_by_both_sides():
    last_ts = 5 * HOUR_MS
    existing = _frame([_row(4 * HOUR_MS), _row(last_ts)])
    # The API's own oldest row overlaps the cache's last row, as a real
    # page boundary would.
    client = FakeClient([_row(last_ts), _row(6 * HOUR_MS)])

    result = top_up_symbol(client, "BTCUSDT", existing, now_ms=6 * HOUR_MS + 1)

    assert list(result["ts"]) == [4 * HOUR_MS, last_ts, 6 * HOUR_MS]
    assert len(result) == 3


def test_top_up_symbol_requests_a_limit_sized_to_the_actual_gap():
    """A fixed limit either over-fetches every single week forever, or
    under-fetches the one week a run was skipped. Size it to the real gap
    instead, plus a fixed safety buffer."""
    last_ts = 0
    existing = _frame([_row(last_ts)])
    now_ms = 50 * HOUR_MS
    client = FakeClient([_row(last_ts)])

    top_up_symbol(client, "BTCUSDT", existing, now_ms=now_ms)

    assert client.calls[0]["limit"] == 50 + TOPUP_BUFFER_BARS


def test_top_up_symbol_falls_back_to_full_refetch_past_the_fallback_gap(monkeypatch):
    """get_candles's own history-candles top-up loop has no attempt cap and
    skips fetch_symbol's 429 backoff - fine for a normal weekly gap, not
    safe to trust for months of missed runs."""
    import backtest.collect_weekly as collect_weekly

    calls = []
    monkeypatch.setattr(collect_weekly, "fetch_symbol_full", lambda client, symbol: calls.append(symbol) or "FULL")

    existing = _frame([_row(0)])
    now_ms = (TOPUP_FALLBACK_HOURS + 1) * HOUR_MS
    client = FakeClient([_row(0)])

    result = top_up_symbol(client, "BTCUSDT", existing, now_ms=now_ms)

    assert result == "FULL"
    assert calls == ["BTCUSDT"]
    assert client.calls == []  # never asked get_candles for a gap this large


def test_top_up_symbol_bootstraps_a_symbol_with_no_cache_at_all(monkeypatch):
    import backtest.collect_weekly as collect_weekly

    monkeypatch.setattr(collect_weekly, "fetch_symbol_full", lambda client, symbol: "FULL")

    result = top_up_symbol(FakeClient([]), "NEWUSDT", pd.DataFrame(columns=COLUMNS))

    assert result == "FULL"
