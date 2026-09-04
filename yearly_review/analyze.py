"""Builds the yearly report.

WHAT THIS IS FOR, AND WHY IT IS NOT THE MONTHLY REPORT WITH A WIDER FILTER

weekly_review deliberately never scores real trades against a baseline - the
sample is too thin, so a week-vs-all-time delta on real trades would mostly
be noise (see weekly_review.analyze's own module docstring). monthly_review
does not report real performance at all - its job is failures and tuning,
not "how did the account do".

Nobody has ever answered "how did the account actually perform" with real
trades, because nothing has run long enough for that answer to mean
anything. A year is the first window where the real-trade sample is large
enough to say something about win rate, expectancy, drawdown and
per-strategy P&L without every number being swamped by noise - which is what
journal.stats.compute_stats already computes, this just points it at real
trades over a year instead of a week.

This is deliberately NOT a re-run of monthly_review's failure/tuning checks
at a wider window - twelve monthly reports already covered that ground, and
repeating it here would show old news, not new signal.

Like monthly_review.analyze, `bitget` is optional and duck-typed: without one
the money sections report "not available" rather than guessing.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from core.storage import Storage, Trade
from journal.stats import Stats, compute_stats
from monthly_review import snapshot as monthly_snapshot
from monthly_review.analyze import Reconciliation
from yearly_review import snapshot
from yearly_review.window import LOCAL_TZ, last_full_year, to_ms, year_name


@dataclass
class YearlyReport:
    year_start: date
    year_end: date
    label: str
    trades: list[Trade]
    stats: Stats
    reconciliation: Reconciliation
    # One (date, equity) point per monthly report that ran inside this year -
    # see monthly_review.snapshot.history for why that is the finest curve
    # that can honestly be drawn, and yearly_review.chart for why it renders
    # nothing below two points.
    monthly_equity: list[tuple[date, float]]


# Same reasoning as monthly_review.analyze.SNAPSHOT_TOLERANCE_DAYS: the
# opening snapshot has to line up with the cron's own boundary run, or the
# balance line covers a different window than the fees and funding beside
# it.
SNAPSHOT_TOLERANCE_DAYS = 1.0


def _opening_equity(db_path: str, start: date, label: str) -> tuple[float | None, str | None]:
    """The equity to measure this year FROM, or None and the reason why not.

    A wrong balance line is worse than an absent one - see
    monthly_review.analyze._opening_equity, which this mirrors.
    """
    prior = snapshot.previous(db_path)
    if prior is None:
        return None, "no snapshot from a previous run"

    taken_at, equity = prior
    if taken_at.tzinfo is None:
        taken_at = taken_at.replace(tzinfo=LOCAL_TZ)

    year_start = datetime(start.year, start.month, start.day, tzinfo=LOCAL_TZ)
    drift_days = abs((taken_at - year_start).total_seconds()) / 86400.0
    if drift_days <= SNAPSHOT_TOLERANCE_DAYS:
        return equity, None

    return None, (
        f"the last snapshot was taken {taken_at:%Y-%m-%d %H:%M}, "
        f"{drift_days:.1f} days from the start of {label} — "
        f"it does not line up with the window fees and funding were read over"
    )


def analyze(
    storage: Storage,
    today: date | None = None,
    bitget=None,
) -> YearlyReport:
    start, end = last_full_year(today)

    # Same off-by-one-day trap as monthly_review.analyze: trades.תאריך is a
    # plain date, so the window's exclusive end would pull Jan 1 of the next
    # year into this report if handed straight to Storage's inclusive `end`.
    trades = storage.read_all(start=start, end=end - timedelta(days=1))
    closed = [t for t in trades if t.is_closed]

    equity_end = bitget.get_account_equity() if bitget is not None else None
    equity_start, equity_start_note = _opening_equity(storage.db_path, start, year_name(start))
    fees = funding = None
    if bitget is not None:
        start_ms, end_ms = to_ms(start), to_ms(end)
        fees = bitget.get_fees_paid(start_ms, end_ms)
        funding = bitget.get_funding_paid(start_ms, end_ms)

    reconciliation = Reconciliation(
        equity_start=equity_start,
        equity_end=equity_end,
        realized_pnl=sum(t.רווח_הפסד or 0.0 for t in closed),
        fees=fees,
        funding=funding,
        open_at_end=len(storage.open_trades()),
        equity_start_note=equity_start_note,
    )

    # Inclusive of `end` deliberately, unlike every other window in this
    # module: the snapshot recorded AT the year boundary (Jan 1) is the
    # closing equity of the year that just ended, even though it is dated
    # into the next one - the same value monthly_review's own January report
    # uses as ITS opening balance. Excluding it would leave this curve
    # stopping one month short of the balance line above it, which reads
    # from bitget.get_account_equity() at that same boundary.
    monthly_equity = [
        (at.date(), equity)
        for at, equity in monthly_snapshot.history(storage.db_path)
        if start <= at.date() <= end
    ]

    return YearlyReport(
        year_start=start,
        year_end=end,
        label=year_name(start),
        trades=closed,
        stats=compute_stats(trades),
        reconciliation=reconciliation,
        monthly_equity=monthly_equity,
    )
