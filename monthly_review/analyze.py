"""Builds the monthly report.

WHAT THIS IS FOR, AND WHY IT IS NOT THE WEEKLY REPORT WITH A WIDER FILTER

The weekly report answers "is the bot running, and what did it signal". This
one answers the two questions a week cannot reach:

  did anything FAIL   - the failures worth catching happen a few times a
                        month, so four clean weekly reports and one broken
                        month look identical
  does anything need  - the leaks worth catching accumulate slower than a
  TUNING                week's noise, so a week can only ever show the noise

Every section here had to pass that test. Anything that would show real signal
at weekly cadence belongs in weekly_review, not here.

THE FLAG RULE (Dror, 2026-09-03): a tuning item fires only when it is outside
its noise band. Everything else prints its number alongside the smallest
difference the month could have detected, and stays silent. See
monthly_review.noise for why that band is wider than a naive 95%.

Like weekly_review.analyze, `bitget` is optional and duck-typed: without one
the money sections report "not available" rather than guessing.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from core.storage import SignalRecord, Storage, Trade
from journal.stats import Stats, compute_stats
from monthly_review import snapshot
from monthly_review.cadence import Cadence, cadence_for
from monthly_review.noise import Finding
from monthly_review.tuning import FeeModelCheck, slippage_findings
from monthly_review.window import LOCAL_TZ, last_full_month, month_name, to_ms


@dataclass
class Reconciliation:
    """Where the account's money went, checked against the exchange.

    The residual is the point. Realized P&L, fees and funding are the three
    things the bot knows about; if they do not add up to the actual change in
    equity, something moved the balance for a reason nothing in this codebase
    modelled - which is a defect report, not a statistic.
    """

    equity_start: float | None
    equity_end: float | None
    realized_pnl: float
    fees: float | None
    funding: float | None
    open_at_end: int
    # Why there is no opening equity, when there is none. None means there is.
    equity_start_note: str | None = None

    @property
    def explained(self) -> float | None:
        if self.fees is None or self.funding is None:
            return None
        return self.realized_pnl - self.fees - self.funding

    @property
    def actual(self) -> float | None:
        if self.equity_start is None or self.equity_end is None:
            return None
        return self.equity_end - self.equity_start

    @property
    def residual(self) -> float | None:
        actual, explained = self.actual, self.explained
        return None if actual is None or explained is None else actual - explained

    @property
    def residual_is_explainable(self) -> bool:
        """Positions open at the closing snapshot carry unrealized P&L that
        has not been realized into any trade row, so it lands in the residual
        legitimately. A residual with flat books has no such excuse and is
        worth chasing.
        """
        return bool(self.open_at_end)


@dataclass
class Autonomy:
    """How much the month needed Dror's hands - the metric that decides
    whether the bot can be left alone, tracked rather than assumed."""

    approved: int
    rejected: int
    never_acted_on: int
    changed_from_plan: int

    @property
    def total_signals(self) -> int:
        return self.approved + self.rejected + self.never_acted_on

    @property
    def intervention_rate(self) -> float | None:
        """Share of signals that needed a human decision at all. Approving
        counts: a bot that runs alone does not wait to be told."""
        total = self.total_signals
        return (self.approved + self.rejected) / total if total else None


@dataclass
class MonthlyReport:
    month_start: date
    month_end: date
    label: str
    trades: list[Trade]
    signals: list[SignalRecord]
    stats: Stats
    reconciliation: Reconciliation
    autonomy: Autonomy
    cadence: list[Cadence]
    slippage: dict[str, Finding] = field(default_factory=dict)
    fee_model: FeeModelCheck | None = None

    @property
    def failures(self) -> list[Cadence]:
        return [c for c in self.cadence if c.alarming]

    @property
    def fired(self) -> list[Finding]:
        """Only findings outside their band. The rule lives in one place so no
        renderer can quietly loosen it."""
        return [f for f in self.slippage.values() if f.fires]


# How far the opening snapshot may sit from the start of the reported month.
#
# The balance line subtracts an equity recorded by the PREVIOUS run from equity
# read now, while fees and funding are queried over the calendar month. Those
# two windows agree only if the previous run happened at the month boundary -
# which is what the cron does, and what a manual run does not.
#
# Running the report by hand to see what it looks like is the obvious thing to
# do after a deploy, and it would silently re-baseline the snapshot to the
# middle of a month. The next report would then show a balance change over a
# window that does not match the month it claims to describe, with nothing on
# the page saying so. A day of slack covers the run's own 09:00 offset and a
# retry, and nothing else.
SNAPSHOT_TOLERANCE_DAYS = 1.0


def _opening_equity(db_path: str, start: date, label: str) -> tuple[float | None, str | None]:
    """The equity to measure this month FROM, or None and the reason why not.

    A wrong balance line is worse than an absent one: absent is visibly absent,
    while wrong is a number Dror would reasonably act on.
    """
    prior = snapshot.previous(db_path)
    if prior is None:
        return None, "no snapshot from a previous run"

    taken_at, equity = prior
    if taken_at.tzinfo is None:
        # Snapshots written before this carried an offset, and any hand-made
        # one. Local is the only sane reading - it is the zone everything else
        # in this report is dated in.
        taken_at = taken_at.replace(tzinfo=LOCAL_TZ)

    month_start = datetime(start.year, start.month, start.day, tzinfo=LOCAL_TZ)
    drift_days = abs((taken_at - month_start).total_seconds()) / 86400.0
    if drift_days <= SNAPSHOT_TOLERANCE_DAYS:
        return equity, None

    return None, (
        f"the last snapshot was taken {taken_at:%Y-%m-%d %H:%M}, "
        f"{drift_days:.1f} days from the start of {label} — "
        f"it does not line up with the window fees and funding were read over"
    )


def analyze(
    storage: Storage,
    live_tags: set[str],
    silence_days,
    today: date | None = None,
    bitget=None,
    modelled_fees: float | None = None,
) -> MonthlyReport:
    """`silence_days` is notifier.main.signal_silence_days, passed in rather
    than imported: it keeps this module off the import path of the scanner it
    reports on, and lets tests set thresholds without monkeypatching.
    """
    start, end = last_full_month(today)

    # Storage's `end` is INCLUSIVE (<=), and the two tables store their dates
    # in different shapes, so the same bound cannot be handed to both:
    #
    #   trades.תאריך      a plain date, "2026-08-31". Passing the window's
    #                     exclusive end ("2026-09-01") would compare equal and
    #                     pull the next month's first day into this report.
    #   signals.dispatched_at
    #                     a full timestamp, "2026-08-31T22:14:05". Passing the
    #                     LAST DAY ("2026-08-31") drops the entire final day,
    #                     since "2026-08-31T22:14:05" <= "2026-08-31" is false
    #                     as a string compare.
    #
    # So trades take the last day and signals take the exclusive end. Each is
    # wrong for the other table, silently and by exactly one day.
    trades = storage.read_all(start=start, end=end - timedelta(days=1))
    signals = storage.read_signals(start=start, end=end)
    closed = [t for t in trades if t.is_closed]

    # NUMBER OF TESTS, fixed before any result is looked at: one slippage test
    # per instance that dispatched anything. Deciding it afterwards - or
    # counting only the tests that happened to look interesting - would defeat
    # the family-wise correction entirely. See monthly_review.noise.
    trades_by_id = {t.מספר_עסקה: t for t in trades}
    num_tests = max(1, len({s.strategy_tag for s in signals}))
    slippage = slippage_findings(signals, trades_by_id, num_tests=num_tests)

    equity_end = bitget.get_account_equity() if bitget is not None else None
    equity_start, equity_start_note = _opening_equity(storage.db_path, start, month_name(start))
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

    autonomy = Autonomy(
        approved=sum(1 for s in signals if s.decision == "approved"),
        rejected=sum(1 for s in signals if s.decision == "rejected"),
        never_acted_on=sum(1 for s in signals if not s.decision),
        changed_from_plan=sum(1 for t in closed if t.changed_from_plan),
    )

    cadence = [cadence_for(tag, signals, start, end, silence_days(tag)) for tag in sorted(live_tags)]

    fee_model = (
        FeeModelCheck(fees, modelled_fees)
        if fees is not None and modelled_fees is not None
        else None
    )

    return MonthlyReport(
        month_start=start,
        month_end=end,
        label=month_name(start),
        trades=closed,
        signals=signals,
        stats=compute_stats(trades),
        reconciliation=reconciliation,
        autonomy=autonomy,
        cadence=cadence,
        slippage=slippage,
        fee_model=fee_model,
    )
