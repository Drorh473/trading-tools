import pytest

from core.storage import SignalRecord, Storage
from journal.stats import compute_paper_stats, compute_stats


def _open_and_close(storage, symbol, direction, entry_price, size, stop, target, exit_price, strategy_tag):
    trade_id = storage.create_pending(symbol=symbol, direction=direction, strategy_tag=strategy_tag)
    storage.confirm_entry(
        trade_id, entry_price=entry_price, position_size=size, actual_stop=stop, actual_target=target, leverage=1.0
    )
    storage.close_trade(trade_id, exit_price=exit_price)
    return trade_id


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "trades.db"))

    # +2R win on strategy A
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=120, strategy_tag="A")

    # -1R loss on strategy A
    _open_and_close(s, "ETHUSDT", "long", 50, 2, 45, 60, exit_price=45, strategy_tag="A")

    # +3R win on strategy B
    _open_and_close(s, "SOLUSDT", "short", 20, 5, 22, 14, exit_price=14, strategy_tag="B")

    # still open, should be excluded
    trade_id = s.create_pending(symbol="XRPUSDT", direction="long", strategy_tag="B")
    s.confirm_entry(trade_id, entry_price=1, position_size=100, actual_stop=0.9, actual_target=1.2, leverage=1.0)

    return s


def test_compute_stats_overall(storage):
    stats = compute_stats(storage.read_all())

    assert stats.total_closed == 3
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.expectancy == pytest.approx((2 - 1 + 3) / 3)
    assert stats.total_pnl == pytest.approx(20 - 10 + 30)
    assert stats.equity_curve == pytest.approx([20, 10, 40])
    assert stats.max_drawdown == pytest.approx(10)  # peak 20 -> dip to 10
    assert stats.best_trade.סימבול == "SOLUSDT"
    assert stats.worst_trade.סימבול == "ETHUSDT"


def test_compute_stats_by_strategy(storage):
    stats = compute_stats(storage.read_all())

    a = stats.by_strategy["A"]
    assert a.count == 2
    assert a.win_rate == pytest.approx(0.5)
    assert a.expectancy == pytest.approx((2 - 1) / 2)
    assert a.total_pnl == pytest.approx(10)

    b = stats.by_strategy["B"]
    assert b.count == 1
    assert b.win_rate == 1.0
    assert b.total_pnl == pytest.approx(30)


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats.total_closed == 0
    assert stats.best_trade is None


def _signal(strategy_tag, direction, paper_r, decision=None):
    return SignalRecord(
        id=1,
        dispatched_at="2020-01-01T00:00:00+00:00",
        symbol="BTCUSDT",
        direction=direction,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        strategy_tag=strategy_tag,
        confluence=None,
        decision=decision,
        trade_id=None,
        paper_r=paper_r,
        paper_resolved_at="2020-01-01T01:00:00+00:00" if paper_r is not None else None,
    )


def test_compute_paper_stats_overall_and_by_strategy_direction():
    signals = [
        _signal("Strategy 1 1H", "long", 2.0, decision="approved"),
        _signal("Strategy 1 1H", "long", -1.0, decision="rejected"),
        _signal("Strategy 1 1H", "short", 1.5, decision=None),  # never acted on -> "ignored"
        _signal("Strategy 2 1H/15m", "long", -1.0, decision="approved"),
        _signal("Strategy 1 1H", "long", None),  # still running, excluded entirely
    ]

    stats = compute_paper_stats(signals)

    assert stats.total_resolved == 4
    assert stats.win_rate == pytest.approx(2 / 4)
    assert stats.expectancy == pytest.approx((2.0 - 1.0 + 1.5 - 1.0) / 4)

    long_1h = stats.by_strategy_direction["Strategy 1 1H long"]
    assert long_1h.count == 2
    assert long_1h.expectancy == pytest.approx((2.0 - 1.0) / 2)

    short_1h = stats.by_strategy_direction["Strategy 1 1H short"]
    assert short_1h.count == 1
    assert short_1h.expectancy == pytest.approx(1.5)


def test_compute_paper_stats_by_decision_treats_unset_as_ignored():
    signals = [
        _signal("Strategy 1 1H", "long", 2.0, decision="approved"),
        _signal("Strategy 1 1H", "long", -1.0, decision="rejected"),
        _signal("Strategy 1 1H", "long", 1.0, decision=None),
    ]

    stats = compute_paper_stats(signals)

    assert stats.by_decision["approved"].count == 1
    assert stats.by_decision["rejected"].count == 1
    assert stats.by_decision["ignored"].count == 1
    assert stats.by_decision["ignored"].expectancy == pytest.approx(1.0)


def test_compute_paper_stats_empty():
    stats = compute_paper_stats([])
    assert stats.total_resolved == 0
    assert stats.by_strategy_direction == {}
    assert stats.by_decision == {}


def test_too_small_is_excluded_from_the_headline_and_per_strategy_numbers():
    # A too_small signal was never a placeable trade - it's the account
    # saying a trade wasn't possible, not a judgment call about a candidate
    # one. Blending it into "how is the strategy doing" would read a sizing
    # artifact as a strategy-quality signal.
    signals = [
        _signal("Strategy 1 1D", "short", 2.0, decision="approved"),
        _signal("Strategy 1 1D", "short", -5.0, decision="too_small"),
    ]

    stats = compute_paper_stats(signals)

    assert stats.total_resolved == 1  # the too_small one doesn't count
    assert stats.expectancy == pytest.approx(2.0)  # unpolluted by the -5.0
    assert stats.by_strategy_direction["Strategy 1 1D short"].count == 1


def test_too_small_still_shows_up_in_by_decision():
    signals = [_signal("Strategy 1 1D", "short", -5.0, decision="too_small")]

    stats = compute_paper_stats(signals)

    assert stats.by_decision["too_small"].count == 1
    assert stats.by_decision["too_small"].expectancy == pytest.approx(-5.0)


def test_all_too_small_leaves_the_headline_at_zero_but_by_decision_populated():
    signals = [_signal("Strategy 1 1D", "short", -5.0, decision="too_small")]

    stats = compute_paper_stats(signals)

    assert stats.total_resolved == 0
    assert stats.by_decision["too_small"].count == 1


def test_a_signal_the_code_refused_is_not_scored_as_a_strategy_trade():
    """refused_at_fill and send_failed never reached Dror, and both write a
    signals row - so paper_sim resolves them and they would land in
    by_strategy_direction unless excluded.

    That is the number Strategy 2.1's 1H hold is being watched on. It ships at
    10 rather than the 20 the backtest prefers, specifically to accumulate a
    live sample faster; a sample blended with trades the strategy itself refused
    would answer a different question than the one being asked of it.

    Still reachable through by_decision, which is where "is the guard refusing
    winners" gets answered.
    """
    signals = [
        _signal(strategy_tag="Strategy 2.1 1H", direction="long", decision="approved", paper_r=1.0),
        _signal(strategy_tag="Strategy 2.1 1H", direction="long", decision="ignored", paper_r=-1.0),
        # never asked for, and hugely profitable - it must not flatter the strategy
        _signal(strategy_tag="Strategy 2.1 1H", direction="long", decision="refused_at_fill", paper_r=9.0),
        _signal(strategy_tag="Strategy 2.1 1H", direction="long", decision="send_failed", paper_r=9.0),
    ]

    stats = compute_paper_stats(signals)

    assert stats.total_resolved == 2, "only the two that were actually proposed"
    assert stats.expectancy == 0.0
    assert stats.by_strategy_direction["Strategy 2.1 1H long"].count == 2
    # but the refusals stay inspectable, which is how the guard gets judged
    assert stats.by_decision["refused_at_fill"].count == 1
    assert stats.by_decision["refused_at_fill"].expectancy == 9.0
    assert stats.by_decision["send_failed"].count == 1


# ---------------------------------------------------------------------------
# by_symbol: which watchlist symbols carry the edge vs. drag, with the same
# drop-top-3 discipline applied per strategy elsewhere in this project (see
# feedback-sweep-past-the-optimum: an edge that dies without its three best
# trades is not an edge).
# ---------------------------------------------------------------------------


def test_compute_stats_by_symbol_counts_and_expectancy(storage):
    stats = compute_stats(storage.read_all())

    btc = stats.by_symbol["BTCUSDT"]
    assert btc.count == 1
    assert btc.expectancy == pytest.approx(2.0)
    assert btc.total_pnl == pytest.approx(20)

    eth = stats.by_symbol["ETHUSDT"]
    assert eth.count == 1
    assert eth.expectancy == pytest.approx(-1.0)


def test_by_symbol_drop_top3_is_none_with_three_or_fewer_trades(storage):
    """Dropping the top 3 of a 3-trade (or smaller) sample leaves nothing to
    average - reporting a number there would look measured and not be."""
    stats = compute_stats(storage.read_all())

    assert stats.by_symbol["BTCUSDT"].expectancy_drop_top3 is None  # 1 trade


def test_by_symbol_drop_top3_excludes_the_three_best_r_multiples(tmp_path):
    from core.storage import Storage

    s = Storage(str(tmp_path / "trades.db"))
    # Four trades on one symbol: R = +5, +3, +2, -1. Dropping the top 3
    # (+5, +3, +2) should leave only the -1 loss.
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 150, exit_price=150, strategy_tag="A")  # +5R
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 130, exit_price=130, strategy_tag="A")  # +3R
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=120, strategy_tag="A")  # +2R
    _open_and_close(s, "BTCUSDT", "long", 100, 1, 90, 120, exit_price=90, strategy_tag="A")   # -1R

    stats = compute_stats(s.read_all())

    btc = stats.by_symbol["BTCUSDT"]
    assert btc.count == 4
    assert btc.expectancy == pytest.approx((5 + 3 + 2 - 1) / 4)
    assert btc.expectancy_drop_top3 == pytest.approx(-1.0)


def test_by_symbol_empty_when_no_closed_trades():
    assert compute_stats([]).by_symbol == {}
