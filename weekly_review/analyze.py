"""Builds the weekly performance report.

Real trades get a plain listing rather than a week-vs-all-time comparison:
the sample is still tiny (the trades table only ever gains a row once a
signal is both approved and confirmed on Bitget), so a delta against
all-time would mostly be noise. Every dispatched signal gets a paper
outcome regardless of what you did with it though, so that side of the
report can support a real week-vs-all-time comparison, broken down by
strategy+direction and by whether you approved, rejected, or never acted on
it. See journal/paper_sim.py for how a signal gets its paper_r - this module
only ever reads what's already been resolved.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from core import clock
from core.storage import Storage, Trade
from journal.stats import PaperStats, Stats, compute_paper_stats, compute_stats

# Dror trades out of Israel; the week he means is Sunday-Saturday there, not
# the ISO Monday-start week the stdlib defaults to. core.clock is the same zone
# the trade rows are now DATED in - they used to be written in the VM's UTC
# while this filtered them by a Jerusalem week, so a trade opened after
# midnight local fell into the previous report.
JERUSALEM = clock.LOCAL_TZ


def start_of_week(today: date | None = None) -> date:
    today = today or datetime.now(JERUSALEM).date()
    days_since_sunday = (today.weekday() + 1) % 7  # date.weekday(): Monday=0 ... Sunday=6
    return today - timedelta(days=days_since_sunday)


# A STRATEGY UNDER REVIEW, and the count that ends the review.
#
# Strategy 1 1H measures no edge in any configuration tested: +0.002R at the 6%
# cap and -0.019R at 10% over 686 and 788 replayed trades, ending the year down
# 16.5% and 34.2%; and +0.284R but MINUS 43 cents across its 12 real closed
# trades. Dror's call on 2026-08-20 was to leave it live and unchanged and
# revisit at 100 closed trades.
#
# The counter exists because that is the only part of the decision with no
# mechanism behind it. "Revisit later" with nothing counting becomes never.
#
# 100 is honest about what it can settle: at the live SD of 1.82 it can detect
# a +0.5R edge and nothing smaller. A true +0.1R - or a true -0.1R - will not
# show up at any sample this account can produce in years. The review is a
# prompt to look again, not a significance test.
REVIEW_THRESHOLD = 100
UNDER_REVIEW = ("Strategy 1 1H",)


@dataclass
class WeeklyReport:
    week_trades: list[Trade]
    real_this_week: Stats
    real_all_time: Stats
    paper_this_week: PaperStats
    paper_all_time: PaperStats
    paper_win_rate_delta: float
    paper_expectancy_delta: float
    best_strategy_this_week: str | None
    worst_strategy_this_week: str | None
    current_streak_len: int
    current_streak_type: str  # "win", "loss", or "none"
    review_progress: dict[str, int]  # tag -> closed trades so far
    downtime: list[tuple[float, float, float]]  # (last_seen, back_at, seconds)
    watched_seconds: float  # how long the heartbeat actually covered
    restarts: list[float]  # service starts inside the window


def analyze(storage: Storage, today: date | None = None) -> WeeklyReport:
    week_start = start_of_week(today)

    all_trades = storage.read_all()
    week_trades_all = storage.read_all(start=week_start)
    week_trades_closed = [t for t in week_trades_all if t.is_closed]

    all_signals = storage.read_signals()
    week_signals = storage.read_signals(start=week_start)

    # AVAILABILITY. Measured from the scan heartbeat, whose rows say when each
    # cycle began and when the next was due, so a gap is judged against what
    # the bot itself expected rather than an assumed cadence.
    week_start_ts = datetime(
        week_start.year, week_start.month, week_start.day, tzinfo=clock.LOCAL_TZ
    ).timestamp()
    downtime = storage.downtime_gaps(since=week_start_ts)
    span = storage.heartbeat_span(since=week_start_ts)
    watched_seconds = (span[1] - span[0]) if span else 0.0
    restarts = storage.service_starts(since=week_start_ts)

    review_progress = {
        tag: sum(1 for t in all_trades if t.is_closed and t.תגית_אסטרטגיה == tag)
        for tag in UNDER_REVIEW
    }

    real_all_time = compute_stats(all_trades)
    real_this_week = compute_stats(week_trades_all)

    paper_all_time = compute_paper_stats(all_signals)
    paper_this_week = compute_paper_stats(week_signals)

    paper_win_rate_delta = (
        paper_this_week.win_rate - paper_all_time.win_rate if paper_this_week.total_resolved else 0.0
    )
    paper_expectancy_delta = (
        paper_this_week.expectancy - paper_all_time.expectancy if paper_this_week.total_resolved else 0.0
    )

    best_tag, worst_tag = _best_worst_strategy(real_this_week)
    streak_len, streak_type = _current_streak(all_trades)

    return WeeklyReport(
        week_trades=week_trades_closed,
        real_this_week=real_this_week,
        real_all_time=real_all_time,
        paper_this_week=paper_this_week,
        paper_all_time=paper_all_time,
        paper_win_rate_delta=paper_win_rate_delta,
        paper_expectancy_delta=paper_expectancy_delta,
        best_strategy_this_week=best_tag,
        worst_strategy_this_week=worst_tag,
        current_streak_len=streak_len,
        current_streak_type=streak_type,
        review_progress=review_progress,
        downtime=downtime,
        watched_seconds=watched_seconds,
        restarts=restarts,
    )


def prune_stale_heartbeats(storage: Storage, now: float | None = None) -> None:
    """Drop heartbeats older than the retention window.

    Storage.prune_heartbeats existed and nothing called it, so the table grew
    at ~96 rows a day forever, for a report that never reads further back than
    one week. A retention policy nothing applies is not a retention policy.

    Called from the weekly run rather than on every scan: the deletion is
    housekeeping, and doing it inside the trading loop would put a write in the
    hot path to save nothing.
    """
    import time

    from core.storage import HEARTBEAT_PRUNE_DAYS

    cutoff = (now if now is not None else time.time()) - HEARTBEAT_PRUNE_DAYS * 86400
    storage.prune_heartbeats(before=cutoff)


def render(report: WeeklyReport) -> str:
    lines = ["# Weekly Performance Review", ""]
    lines += _render_real_section(report)
    lines.append("")
    lines += _render_paper_section(report)
    lines.append("")
    lines += _render_too_small_section(report)
    lines.append("")
    lines += _render_swing_slots_full_section(report)
    lines.append("")
    lines += _render_review_section(report)
    lines.append("")
    lines += _render_availability_section(report)
    return "\n".join(lines)


def _render_availability_section(report: WeeklyReport) -> list[str]:
    """Every gap, however small - Dror asked for "even a small time".

    A restart shows up here, and that is deliberate: a deploy IS a window where
    nothing was watching the market. The duration is printed so a 40-second
    bounce reads differently from a two-hour outage.

    Percentages are against the period the heartbeat actually COVERED, not
    against the whole week. A table that only began recording on Tuesday cannot
    honestly claim a clean Sunday, and saying "100% up" from a week with no
    heartbeat at all would be the same failure as the ledger's silent all-clear.
    """
    lines = ["## Bot availability"]
    if report.watched_seconds <= 0:
        lines.append("No heartbeat recorded this week - availability is UNKNOWN, not perfect.")
        return lines

    # Restarts are reported SEPARATELY from late scans, because they are a
    # different fact. A process that dies and returns inside its own sleep
    # window misses no scan, so it shows no gap and the market was genuinely
    # never unwatched - but the bot was still down, which is what was asked
    # for. Collapsing the two would either hide restarts or invent outages.
    if report.restarts:
        when = ", ".join(
            f"{datetime.fromtimestamp(t, clock.LOCAL_TZ):%a %H:%M}" for t in report.restarts
        )
        lines.append(f"Service started {len(report.restarts)}x this week: {when}.")

    total = sum(g[2] for g in report.downtime)
    hours = report.watched_seconds / 3600
    if not report.downtime:
        lines.append(f"No scan was missed or late. Watched continuously for {hours:.1f}h.")
        return lines

    pct = total / report.watched_seconds * 100
    lines.append(
        f"**{len(report.downtime)} gap(s), {_duration(total)} missing** "
        f"out of {hours:.1f}h watched ({100 - pct:.2f}% up)."
    )
    for last_seen, back_at, seconds in report.downtime:
        started = datetime.fromtimestamp(last_seen, clock.LOCAL_TZ)
        back = datetime.fromtimestamp(back_at, clock.LOCAL_TZ)
        lines.append(
            f"- {started:%a %d %b %H:%M} to {back:%H:%M} - {_duration(seconds)} unwatched"
        )
    return lines


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}min"
    return f"{seconds / 3600:.1f}h"


def _render_review_section(report: WeeklyReport) -> list[str]:
    """Say how far a strategy under review has to go, every week, unprompted.

    Reaching the threshold is stated loudly rather than as another progress
    line: the point is that the review happens without either party having to
    remember it was agreed to.
    """
    if not report.review_progress:
        return []
    lines = ["## Strategies under review"]
    for tag, n in sorted(report.review_progress.items()):
        if n >= REVIEW_THRESHOLD:
            lines.append(
                f"- **{tag}: {n} closed trades - THE REVIEW IS DUE.** Left live and "
                f"unchanged on 2026-08-20 pending {REVIEW_THRESHOLD}; that point has arrived."
            )
        else:
            lines.append(f"- {tag}: {n} of {REVIEW_THRESHOLD} closed trades toward review")
    return lines


def _render_real_section(report: WeeklyReport) -> list[str]:
    lines = ["## Real trades this week"]

    if not report.week_trades:
        lines.append("None closed this week.")
    else:
        for t in report.week_trades:
            tag = f" [{t.תגית_אסטרטגיה}]" if t.תגית_אסטרטגיה else ""
            lines.append(f"- {t.סימבול} {t.כיוון}: {t.מכפיל_R:+.2f}R (${t.רווח_הפסד:+,.2f}){tag}")
        w = report.real_this_week
        lines.append(
            f"- Total: {w.total_closed} trades, {w.win_rate:.0%} win rate, "
            f"{w.expectancy:+.2f}R expectancy, ${w.total_pnl:+,.2f} P&L"
        )

    a = report.real_all_time
    if a.total_closed:
        lines.append(f"- All-time: {a.total_closed} trades, {a.win_rate:.0%} win rate, {a.expectancy:+.2f}R expectancy")

    if report.best_strategy_this_week:
        lines.append(f"- Best setup this week: {report.best_strategy_this_week}")
    if report.worst_strategy_this_week and report.worst_strategy_this_week != report.best_strategy_this_week:
        lines.append(f"- Worst setup this week: {report.worst_strategy_this_week}")
    if report.current_streak_len:
        lines.append(f"- Current streak: {report.current_streak_len} {report.current_streak_type}s")

    return lines


def _render_paper_section(report: WeeklyReport) -> list[str]:
    w, a = report.paper_this_week, report.paper_all_time
    lines = ["## Paper-simulated signals this week"]

    if not w.total_resolved:
        lines.append("None resolved this week.")
        return lines

    lines.append(f"- Resolved: {w.total_resolved}")
    lines.append(f"- Win rate: {w.win_rate:.0%} ({report.paper_win_rate_delta:+.0%} vs all-time)")
    lines.append(f"- Expectancy: {w.expectancy:+.2f}R ({report.paper_expectancy_delta:+.2f}R vs all-time)")

    lines.append("")
    lines.append("### By strategy")
    for key in sorted(w.by_strategy_direction):
        b = w.by_strategy_direction[key]
        lines.append(f"- {key}: {b.count} signals, {b.win_rate:.0%} win rate, {b.expectancy:+.2f}R")

    lines.append("")
    lines.append("### By decision")
    for key in ("approved", "rejected", "ignored"):
        b = w.by_decision.get(key)
        if b:
            lines.append(f"- {key}: {b.count} signals, {b.win_rate:.0%} win rate, {b.expectancy:+.2f}R")

    lines.append("")
    lines.append("### All-time baseline (paper)")
    lines.append(f"- Resolved: {a.total_resolved}, {a.win_rate:.0%} win rate, {a.expectancy:+.2f}R expectancy")

    return lines


def _render_too_small_section(report: WeeklyReport) -> list[str]:
    """Strategy 1 split entries whose market leg couldn't clear Bitget's
    per-order minimum, so no alert was sent and no trade was attempted. Kept
    separate from "By decision" above deliberately: those are judgment calls
    on a signal you could have taken, this is the account telling you a trade
    wasn't possible at all - a sizing constraint, not a decision.
    """
    lines = ["## Too small to execute this week"]
    blocked = report.paper_this_week.by_decision.get("too_small")
    if not blocked:
        lines.append("None this week.")
    else:
        lines.append(
            f"- {blocked.count} signal(s) couldn't be split-entered at current equity, "
            f"net {blocked.expectancy:+.2f}R had they been taken"
        )
    return lines


def _render_swing_slots_full_section(report: WeeklyReport) -> list[str]:
    """Strategy 1 1D / Strategy 2 1D signals suppressed because both swing
    slots were already occupied (pending + open, combined across both
    instances). Kept separate from "By decision" for the same reason as Too
    small above: this is the swing pool's own hard cap saying a trade wasn't
    possible, not a judgment call on a signal you could have taken.
    """
    lines = ["## Swing slots full this week"]
    blocked = report.paper_this_week.by_decision.get("swing_slots_full")
    if not blocked:
        lines.append("None this week.")
    else:
        lines.append(
            f"- {blocked.count} signal(s) suppressed because both swing slots were taken, "
            f"net {blocked.expectancy:+.2f}R had they been taken"
        )
    return lines


def _best_worst_strategy(stats: Stats) -> tuple[str | None, str | None]:
    if not stats.by_strategy:
        return None, None
    best = max(stats.by_strategy.items(), key=lambda kv: kv[1].expectancy)[0]
    worst = min(stats.by_strategy.items(), key=lambda kv: kv[1].expectancy)[0]
    return best, worst


def _current_streak(trades: list[Trade]) -> tuple[int, str]:
    closed = sorted((t for t in trades if t.מכפיל_R is not None), key=lambda t: t.מספר_עסקה)
    if not closed:
        return 0, "none"

    streak_type = "win" if closed[-1].רווח_הפסד > 0 else "loss"
    streak_len = 0
    for trade in reversed(closed):
        is_win = trade.רווח_הפסד > 0
        if (is_win and streak_type == "win") or (not is_win and streak_type == "loss"):
            streak_len += 1
        else:
            break
    return streak_len, streak_type
