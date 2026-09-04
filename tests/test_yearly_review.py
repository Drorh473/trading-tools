"""End-to-end: does the yearly report cover the right calendar year, and does
it refuse to invent numbers it does not have?
"""

import sqlite3
from datetime import date, datetime

import pytest

from core.storage import Storage
from yearly_review import snapshot
from yearly_review.analyze import analyze
from yearly_review.render import render
from yearly_review.window import last_full_year


def _backdate_trade(db_path, trade_id, when: date):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE trades SET תאריך = ? WHERE מספר_עסקה = ?", (when.isoformat(), trade_id))
    conn.commit()
    conn.close()


def _closed_trade(storage, db_path, when: date, tag, entry=100.0, exit_price=110.0):
    trade_id = storage.create_pending(symbol="BTCUSDT", direction="long", strategy_tag=tag)
    storage.confirm_entry(
        trade_id, entry_price=entry, position_size=1.0, actual_stop=95.0,
        actual_target=110.0, leverage=1.0,
    )
    storage.close_trade(trade_id, exit_price=exit_price)
    _backdate_trade(db_path, trade_id, when)
    return trade_id


class FakeBitget:
    def __init__(self, equity=97.0, fees=1.20, funding=0.05):
        self._equity, self._fees, self._funding = equity, fees, funding

    def get_account_equity(self):
        return self._equity

    def get_fees_paid(self, start_ms, end_ms):
        return self._fees

    def get_funding_paid(self, start_ms, end_ms):
        return self._funding


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "trades.db"))


def test_the_window_is_the_calendar_year_that_just_ended():
    """The job fires on Jan 1. Resolving 'this year' would report on a few
    hours - the same bug the weekly and monthly reports both had at their
    own boundary."""
    start, end = last_full_year(date(2027, 1, 1))

    assert start == date(2026, 1, 1)
    assert end == date(2027, 1, 1)


def test_a_mid_year_run_still_reports_the_prior_calendar_year():
    start, end = last_full_year(date(2026, 6, 15))
    assert (start, end) == (date(2025, 1, 1), date(2026, 1, 1))


def test_the_last_day_of_the_year_is_included_and_the_next_first_is_not(storage):
    """The boundary bug monthly_review nearly shipped with, at the year's own
    edge: trades.תאריך is a plain date, so an inclusive `end` handed the
    window's exclusive boundary would pull Jan 1 of the next year in."""
    db = storage.db_path
    _closed_trade(storage, db, date(2026, 12, 31), "Strategy 1 1H")
    _closed_trade(storage, db, date(2027, 1, 1), "Strategy 1 1H")

    report = analyze(storage, today=date(2027, 1, 1))

    assert len(report.trades) == 1, "Jan 1 trade leaked into the prior year's report"


def test_no_prior_snapshot_says_so_rather_than_showing_a_zero_change(storage):
    """A balance line that reads $0.00 change and one that could not be
    computed look identical to a reader. Only one of them is true."""
    report = analyze(storage, today=date(2027, 1, 1), bitget=FakeBitget())
    text = render(report)

    assert report.reconciliation.actual is None
    assert "not available" in text
    assert "no snapshot from a previous run" in text


def test_the_residual_is_computed_even_though_the_report_no_longer_prints_it(storage):
    """Realized P&L minus fees minus funding must equal the change in equity.
    render() dropped the detailed Balance section (redundant with the verdict
    line - Dror, 2026-09-04), but analyze() still computes the reconciliation
    in full; a future section could use it without re-deriving anything."""
    db = storage.db_path
    snapshot.record(db, 100.0, now=datetime(2026, 1, 1))
    _closed_trade(storage, db, date(2026, 6, 10), "Strategy 1 1H", entry=100.0, exit_price=110.0)

    report = analyze(
        storage, today=date(2027, 1, 1),
        bitget=FakeBitget(equity=97.0, fees=1.2, funding=0.05),
    )

    assert report.reconciliation.residual is not None
    assert not report.reconciliation.residual_is_explainable
    assert "## Balance" not in render(report)


def test_a_snapshot_from_the_year_boundary_is_used(storage):
    """What the cron actually produces: the previous run fired on Jan 1 at
    09:00 local, so the snapshot sits a few hours into the year."""
    snapshot.record(storage.db_path, 100.0, now=datetime(2026, 1, 1, 9, 0))

    report = analyze(storage, today=date(2027, 1, 1), bitget=FakeBitget(equity=97.0))

    assert report.reconciliation.equity_start == 100.0
    assert report.reconciliation.actual == pytest.approx(-3.0)


def test_a_snapshot_from_mid_year_is_refused_rather_than_used(storage):
    """Running the report by hand mid-year would re-baseline the snapshot to
    a date that does not match the fees/funding window - a wrong number is
    worse than an absent one."""
    snapshot.record(storage.db_path, 100.0, now=datetime(2026, 7, 4, 14, 0))

    report = analyze(storage, today=date(2027, 1, 1), bitget=FakeBitget())
    text = render(report)

    assert report.reconciliation.equity_start is None
    assert "does not line up" in text


def test_render_survives_a_completely_empty_year(storage):
    """The first year after deployment, before a full calendar year has
    passed. It must still send something rather than crash on an empty
    stats object."""
    report = analyze(storage, today=date(2027, 1, 1))
    text = render(report)

    assert "Yearly Review" in text
    assert "No closed trades this year" in text


def test_performance_reflects_real_closed_trades(storage):
    """The per-strategy breakdown itself (win rate, expectancy, count) lives
    only in report.stats.by_strategy and the yearly_review.chart.strategy_pnl
    bar chart now - render() dropped its own text section for it
    (Dror, 2026-09-04) since the chart already carries P&L and count."""
    db = storage.db_path
    _closed_trade(storage, db, date(2026, 3, 1), "Strategy 1 1H", entry=100.0, exit_price=110.0)
    _closed_trade(storage, db, date(2026, 5, 1), "Strategy 1 1H", entry=100.0, exit_price=95.0)
    _closed_trade(storage, db, date(2026, 7, 1), "Strategy 4", entry=100.0, exit_price=120.0)

    report = analyze(storage, today=date(2027, 1, 1))
    text = render(report)

    assert report.stats.total_closed == 3
    assert report.stats.by_strategy["Strategy 1 1H"].count == 2
    assert report.stats.by_strategy["Strategy 4"].count == 1
    assert "Closed trades: 3" in text
    assert "## Strategy performance" not in text
    assert "win rate," not in text  # the per-strategy bullet format is gone entirely


def test_a_negative_strategy_is_flagged(storage):
    db = storage.db_path
    for _ in range(6):
        _closed_trade(storage, db, date(2026, 3, 1), "Strategy 1 1H", entry=100.0, exit_price=90.0)

    text = render(analyze(storage, today=date(2027, 1, 1)))

    assert "Strategy 1 1H is negative this year" in text


def test_the_verdict_carries_a_flagged_count_not_the_itemised_reasons(storage):
    """The itemised text belongs to '## Flagged strategies' alone - the
    verdict block only needs to say how many, the same split monthly_review
    draws between its verdict counts and its Failures/Tuning sections
    (Dror, 2026-09-04)."""
    db = storage.db_path
    for _ in range(6):
        _closed_trade(storage, db, date(2026, 3, 1), "Strategy 1 1H", entry=100.0, exit_price=90.0)

    text = render(analyze(storage, today=date(2027, 1, 1)))
    verdict = text.split("## Performance")[0]

    assert "Flagged: 1 strategy" in verdict
    assert "is negative this year" not in verdict


def test_a_strategy_with_too_few_trades_is_flagged_even_if_profitable(storage):
    db = storage.db_path
    _closed_trade(storage, db, date(2026, 3, 1), "Strategy 4", entry=100.0, exit_price=110.0)

    text = render(analyze(storage, today=date(2027, 1, 1)))

    assert "Strategy 4 only closed 1 trade(s)" in text


def test_a_strategy_that_is_positive_with_enough_trades_is_not_flagged(storage):
    db = storage.db_path
    for _ in range(6):
        _closed_trade(storage, db, date(2026, 3, 1), "Strategy 1 1H", entry=100.0, exit_price=110.0)

    text = render(analyze(storage, today=date(2027, 1, 1)))

    assert "Nothing negative and nothing too thin to read." in text


def test_monthly_equity_points_come_from_the_monthly_snapshot_history_within_the_year(storage):
    """Bitget has no historical-equity endpoint, so the only source for a
    monthly curve is whatever monthly_review's own boundary runs recorded -
    see monthly_review.snapshot.history. Points outside the reported year
    (a snapshot from the year before, or from the year that follows) must
    not leak in."""
    from monthly_review import snapshot as monthly_snapshot

    db = storage.db_path
    monthly_snapshot.record(db, 90.0, now=datetime(2025, 12, 1, 9, 0))  # prior year
    monthly_snapshot.record(db, 100.0, now=datetime(2026, 2, 1, 9, 0))
    monthly_snapshot.record(db, 105.0, now=datetime(2026, 6, 1, 9, 0))
    monthly_snapshot.record(db, 98.0, now=datetime(2027, 2, 1, 9, 0))  # next year

    report = analyze(storage, today=date(2027, 1, 1))

    assert [equity for _d, equity in report.monthly_equity] == [100.0, 105.0]


def test_the_year_boundary_snapshot_closes_the_curve_rather_than_starting_the_next_one(storage):
    """The Jan 1 snapshot is dated into the NEXT year but is the closing
    equity of the year that just ended - the same value the verdict's
    balance line reads via bitget.get_account_equity() at that same moment.
    Dropping it would leave the chart stopping a month short of the number
    printed above it."""
    from monthly_review import snapshot as monthly_snapshot

    db = storage.db_path
    monthly_snapshot.record(db, 105.0, now=datetime(2026, 6, 1, 9, 0))
    monthly_snapshot.record(db, 120.0, now=datetime(2027, 1, 1, 9, 0))  # closes 2026, opens 2027

    report = analyze(storage, today=date(2027, 1, 1))

    assert [equity for _d, equity in report.monthly_equity] == [105.0, 120.0]
