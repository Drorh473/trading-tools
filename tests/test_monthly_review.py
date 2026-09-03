"""End-to-end: does the monthly report cover the right month, and does it
refuse to invent numbers it does not have?
"""

import sqlite3
from datetime import date, datetime

import pytest

from core.storage import Storage
from monthly_review import snapshot
from monthly_review.analyze import analyze
from monthly_review.render import render
from monthly_review.window import last_full_month


def _silence_days(tag: str) -> float:
    return 4.0


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


def _signal(storage, db_path, at: str, tag, trade_id=None, entry=100.0, decision=None):
    signal_id = storage.log_signal(
        symbol="BTCUSDT", direction="long", entry_price=entry,
        stop_loss=95.0, take_profit=110.0, strategy_tag=tag,
    )
    if trade_id:
        storage.link_signal_trade(signal_id, trade_id)
    if decision:
        storage.mark_signal_decision(signal_id, decision)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE signals SET dispatched_at = ? WHERE id = ?", (at, signal_id))
    conn.commit()
    conn.close()
    return signal_id


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


def test_the_window_is_the_month_that_just_ended():
    """The job fires on the 1st. Resolving 'this month' would report on a few
    hours - the same bug the weekly report had, where a Sunday run produced a
    20-hour 'week'."""
    start, end = last_full_month(date(2026, 9, 1))

    assert start == date(2026, 8, 1)
    assert end == date(2026, 9, 1)


def test_january_reports_on_december():
    start, end = last_full_month(date(2026, 1, 14))
    assert (start, end) == (date(2025, 12, 1), date(2026, 1, 1))


def test_the_last_day_of_the_month_is_included_and_the_next_first_is_not(storage):
    """The boundary bug this nearly shipped with: Storage's `end` is INCLUSIVE
    and trades store a plain date while signals store a full timestamp, so one
    shared bound is off by a day for one of the two tables - silently."""
    db = storage.db_path
    _closed_trade(storage, db, date(2026, 8, 31), "Strategy 1 1H")
    _closed_trade(storage, db, date(2026, 9, 1), "Strategy 1 1H")
    _signal(storage, db, "2026-08-31T22:14:05", "Strategy 1 1H")
    _signal(storage, db, "2026-09-01T08:00:00", "Strategy 1 1H")

    report = analyze(storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1))

    assert len(report.trades) == 1, "Sept 1 trade leaked into the August report"
    assert len(report.signals) == 1, "the last day of August was dropped"


def test_a_silent_instance_is_a_failure_not_a_clean_month(storage):
    """Strategy 3 is live and has produced zero signals in the entire signals
    table. A report that scores no-signals as healthy would never have caught
    it."""
    _signal(storage, storage.db_path, "2026-08-05T10:00:00", "Strategy 1 1H")

    report = analyze(
        storage, {"Strategy 1 1H", "Strategy 3 1D/1H"}, _silence_days, today=date(2026, 9, 1)
    )

    silent = [c for c in report.failures if c.tag == "Strategy 3 1D/1H"]
    assert silent and silent[0].silent_all_month
    assert "SILENT ALL MONTH" in render(report)


def test_no_prior_snapshot_says_so_rather_than_showing_a_zero_change(storage):
    """A balance line that reads $0.00 change and one that could not be
    computed look identical to a reader. Only one of them is true."""
    report = analyze(
        storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1), bitget=FakeBitget()
    )
    text = render(report)

    assert report.reconciliation.actual is None
    assert "not available" in text
    assert "no snapshot from a previous run" in text


def test_the_residual_is_called_a_defect_when_the_books_were_flat(storage):
    """Realized P&L minus fees minus funding must equal the change in equity.
    When it does not, and nothing was open to explain it, that is a bug report."""
    db = storage.db_path
    snapshot.record(db, 100.0, now=datetime(2026, 8, 1))
    _closed_trade(storage, db, date(2026, 8, 10), "Strategy 1 1H", entry=100.0, exit_price=110.0)

    # equity fell to 97 while the bot believes it made money: unexplained.
    report = analyze(
        storage, {"Strategy 1 1H"}, _silence_days,
        today=date(2026, 9, 1), bitget=FakeBitget(equity=97.0, fees=1.2, funding=0.05),
    )
    text = render(report)

    assert report.reconciliation.residual is not None
    assert not report.reconciliation.residual_is_explainable
    assert "defect, not a statistic" in text


def test_slippage_measured_but_not_flagged_reports_what_it_could_have_seen(storage):
    """The rule Dror set: fire only outside the band. Eleven fills scattered
    either side of plan must come back quiet, and must say how big a cost
    would have had to be before it showed."""
    db = storage.db_path
    scatter_bp = [10, -10, 12, -8, 9, -11, 10, -9, 11, -10, 6]  # mean +0.9bp, sd ~10
    for day, bp in enumerate(scatter_bp, start=1):
        fill = 100.0 * (1 + bp / 10_000)
        trade_id = _closed_trade(storage, db, date(2026, 8, day), "Strategy 1 1H", entry=fill)
        _signal(storage, db, f"2026-08-{day:02d}T10:00:00", "Strategy 1 1H",
                trade_id=trade_id, entry=100.0)

    report = analyze(storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1))
    finding = report.slippage["Strategy 1 1H"]

    assert finding.n == 11
    assert not finding.fires, "a sub-1bp mean under 10bp of scatter is not a finding"
    assert finding.detectable > 4.0, "11 noisy fills cannot resolve a small cost"
    assert "would have shown" in render(report)


def test_an_identical_offset_on_every_single_fill_does_fire(storage):
    """The degenerate case, pinned deliberately. Real fills scatter; a fill
    that lands exactly 5bp adverse EVERY time is not a sample of a noisy
    process, it is a systematic offset - a stale reference price, or a spread
    being crossed the same way each time. Firing is the right call, but it is
    the one path where the band is zero, so it must be a decision rather than
    an accident of the sd==0 branch.
    """
    db = storage.db_path
    for day in range(1, 12):
        trade_id = _closed_trade(storage, db, date(2026, 8, day), "Strategy 1 1H", entry=100.05)
        _signal(storage, db, f"2026-08-{day:02d}T10:00:00", "Strategy 1 1H",
                trade_id=trade_id, entry=100.0)

    report = analyze(storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1))
    finding = report.slippage["Strategy 1 1H"]

    assert finding.live == pytest.approx(5.0)
    assert finding.fires and finding.note == "zero spread in sample"


def test_render_survives_a_completely_empty_month(storage):
    """The first month after deployment, and any month the bot was off. It
    must still send something rather than crash on an empty stats object."""
    report = analyze(storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1))
    text = render(report)

    assert "Monthly Review" in text
    assert "No signals dispatched" in text


def test_a_snapshot_from_mid_month_is_refused_rather_than_used(storage):
    """Running the report by hand after a deploy re-baselines the snapshot to
    the middle of a month. The balance line would then cover a window that does
    not match the fees and funding beside it - and nothing on the page would
    say so. A wrong number here is worse than an absent one."""
    snapshot.record(storage.db_path, 100.0, now=datetime(2026, 8, 17, 14, 0))

    report = analyze(
        storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1), bitget=FakeBitget()
    )
    text = render(report)

    assert report.reconciliation.equity_start is None
    assert report.reconciliation.actual is None
    assert "does not line up" in text
    assert "2026-08-17" in text


def test_a_snapshot_from_the_month_boundary_is_used(storage):
    """What the cron actually produces: the previous run fired on the 1st at
    09:00 local, so the snapshot sits a few hours into the month."""
    snapshot.record(storage.db_path, 100.0, now=datetime(2026, 8, 1, 9, 0))

    report = analyze(
        storage, {"Strategy 1 1H"}, _silence_days,
        today=date(2026, 9, 1), bitget=FakeBitget(equity=97.0),
    )

    assert report.reconciliation.equity_start == 100.0
    assert report.reconciliation.actual == pytest.approx(-3.0)


def test_a_naive_snapshot_timestamp_is_read_as_local(storage):
    """snapshot.record defaults to UTC while monthly_review.main passes local
    time, and a hand-written file could be either. Comparing a naive stamp
    against a local month start without a zone would drift by hours - enough to
    push a boundary snapshot outside the tolerance."""
    snapshot.record(storage.db_path, 100.0, now=datetime(2026, 8, 1, 9, 0))

    report = analyze(
        storage, {"Strategy 1 1H"}, _silence_days, today=date(2026, 9, 1), bitget=FakeBitget()
    )

    assert report.reconciliation.equity_start == 100.0
