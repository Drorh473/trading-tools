import pandas as pd
import pytest

from core.storage import SignalRecord, Storage
from notifier.strategies.base import Signal, signal_to_json
from tools.reconcile import find_start_index, is_eligible, reconcile_trade, resolve_remainder_target


def _signal(**overrides):
    fields = dict(
        symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=90.0,
        strategy_tag="Strategy 1 1H", reward_risk_ratio=3.0,
    )
    fields.update(overrides)
    return Signal(**fields)


def _signal_record(signal, trade_id=1, dispatched_at="2026-01-01T05:00:00+00:00"):
    return SignalRecord(
        id=1, dispatched_at=dispatched_at, symbol=signal.symbol, direction=signal.direction,
        entry_price=signal.entry_price, stop_loss=signal.stop_loss, take_profit=110.0,
        strategy_tag=signal.strategy_tag, confluence=None, decision="approved", trade_id=trade_id,
        paper_r=None, paper_resolved_at=None, signal_json=signal_to_json(signal),
    )


def _open_and_close(storage, **kw):
    kw.setdefault("symbol", "BTCUSDT")
    kw.setdefault("direction", "long")
    kw.setdefault("strategy_tag", "Strategy 1 1H")
    trade_id = storage.create_pending(symbol=kw["symbol"], direction=kw["direction"], strategy_tag=kw["strategy_tag"])
    storage.confirm_entry(
        trade_id, entry_price=kw["entry_price"], position_size=1.0,
        actual_stop=kw["stop"], actual_target=kw.get("target"), leverage=1.0,
    )
    storage.close_trade(trade_id, exit_price=kw["exit_price"])
    return trade_id


def _bars(n=250, start_price=100.0):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    close = [start_price] * n
    return pd.DataFrame({
        "ts": ts, "open": close, "high": [c + 1 for c in close], "low": [c - 1 for c in close],
        "close": close, "base_vol": [1.0] * n, "quote_vol": [1.0] * n,
    })


# ---------------------------------------------------------------------------
# is_eligible
# ---------------------------------------------------------------------------


def test_a_signal_with_no_partial_fraction_override_is_eligible():
    eligible, _reason = is_eligible(_signal(partial_fraction=None))
    assert eligible is True


def test_a_signal_with_the_scanner_default_50pct_is_eligible():
    eligible, _reason = is_eligible(_signal(partial_fraction=0.5))
    assert eligible is True


def test_a_signal_with_a_different_partial_fraction_is_not_eligible():
    eligible, reason = is_eligible(_signal(partial_fraction=0.75))
    assert eligible is False
    assert "0.75" in reason


def test_a_full_close_signal_is_not_eligible():
    eligible, reason = is_eligible(_signal(partial_fraction=1.0))
    assert eligible is False
    assert "1.0" in reason


# ---------------------------------------------------------------------------
# resolve_remainder_target
# ---------------------------------------------------------------------------


class _FakeTrade:
    def __init__(self, runner_target=None):
        self.runner_target = runner_target


def test_resolve_remainder_target_prefers_the_trades_own_tracked_target():
    signal = _signal(remainder_target=999.0)
    trade = _FakeTrade(runner_target=130.0)

    assert resolve_remainder_target(signal, trade) == 130.0


def test_resolve_remainder_target_falls_back_to_the_signals_own_when_the_trade_has_none():
    signal = _signal(remainder_target=140.0)
    trade = _FakeTrade(runner_target=None)

    assert resolve_remainder_target(signal, trade) == 140.0


# ---------------------------------------------------------------------------
# find_start_index
# ---------------------------------------------------------------------------


def test_find_start_index_returns_the_last_bar_at_or_before_the_timestamp():
    bars = _bars(n=10)
    at = pd.Timestamp("2026-01-01 05:30:00")  # between bar 5 (05:00) and bar 6 (06:00)

    assert find_start_index(bars, at) == 5


def test_find_start_index_returns_none_when_every_bar_is_after_the_timestamp():
    bars = _bars(n=10)
    at = pd.Timestamp("2025-01-01")

    assert find_start_index(bars, at) is None


# ---------------------------------------------------------------------------
# reconcile_trade: real score.simulate() underneath, synthetic bars
# ---------------------------------------------------------------------------


def test_reconcile_trade_reports_a_delta_between_real_and_backtest_r(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    # +3R real trade: entry 100, stop 90 -> 1R = 10; exit at 130 = +3R.
    trade_id = _open_and_close(storage, entry_price=100.0, stop=90.0, exit_price=130.0)
    trade = storage.get_trade(trade_id)

    signal = _signal(entry_price=100.0, stop_loss=90.0, reward_risk_ratio=1.0, remainder_target=None)
    record = _signal_record(signal, trade_id=trade_id, dispatched_at="2026-01-01T04:00:00")

    # Bars that never move - the backtest's own simulated trade should be
    # "unresolved" (never hits stop or target1), giving a deterministic,
    # checkable 0.0 backtest R.
    bars = _bars(n=250, start_price=100.0)

    result = reconcile_trade(trade, record, bars)

    assert result.skipped is False
    assert result.real_r == pytest.approx(trade.מכפיל_R)
    assert result.backtest_r is not None
    assert result.delta == pytest.approx(result.real_r - result.backtest_r)


def test_reconcile_trade_skips_an_ineligible_partial_fraction(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_and_close(storage, entry_price=100.0, stop=90.0, exit_price=130.0)
    trade = storage.get_trade(trade_id)

    signal = _signal(partial_fraction=0.75)
    record = _signal_record(signal, trade_id=trade_id)

    result = reconcile_trade(trade, record, _bars())

    assert result.skipped is True
    assert "0.75" in result.skip_reason


def test_reconcile_trade_skips_when_the_trade_predates_every_fetched_bar(tmp_path):
    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = _open_and_close(storage, entry_price=100.0, stop=90.0, exit_price=130.0)
    trade = storage.get_trade(trade_id)

    signal = _signal()
    record = _signal_record(signal, trade_id=trade_id, dispatched_at="2000-01-01T00:00:00")

    result = reconcile_trade(trade, record, _bars())

    assert result.skipped is True
    assert "no bar" in result.skip_reason.lower()


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_shows_the_mean_delta_and_each_scored_trade():
    from tools.reconcile import ReconcileResult, format_report

    results = [
        ReconcileResult(
            trade_id=1, symbol="BTCUSDT", strategy_tag="Strategy 1 1H",
            skipped=False, real_r=2.0, backtest_r=1.5, delta=0.5,
        ),
    ]
    text = format_report(results)

    assert "1 trade(s) reconciled" in text
    assert "BTCUSDT" in text
    assert "+0.50" in text


def test_format_report_lists_skip_reasons_separately():
    from tools.reconcile import ReconcileResult, format_report

    results = [
        ReconcileResult(trade_id=2, symbol="ETHUSDT", strategy_tag="Strategy 3 1D/1H",
                         skipped=True, skip_reason="partial_fraction mismatch"),
    ]
    text = format_report(results)

    assert "skipped" in text.lower()
    assert "ETHUSDT" in text
    assert "partial_fraction mismatch" in text


def test_format_report_handles_no_results():
    from tools.reconcile import format_report

    assert "0 trade(s) reconciled" in format_report([])
