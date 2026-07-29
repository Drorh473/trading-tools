"""Builds a week-vs-all-time performance comparison by calling
journal.stats.compute_stats() twice — once over the full trade history and
once over just the current week — and diffing the two.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from core.storage import Storage, Trade
from journal.stats import Stats, compute_stats


@dataclass
class WeeklyComparison:
    all_time: Stats
    this_week: Stats
    win_rate_delta: float
    expectancy_delta: float
    best_strategy_this_week: str | None
    worst_strategy_this_week: str | None
    current_streak_len: int
    current_streak_type: str  # "win", "loss", or "none"


def start_of_week(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=today.weekday())  # Monday


def analyze(storage: Storage, today: date | None = None) -> WeeklyComparison:
    all_trades = storage.read_all()
    week_trades = storage.read_all(start=start_of_week(today))

    all_time = compute_stats(all_trades)
    this_week = compute_stats(week_trades)

    win_rate_delta = this_week.win_rate - all_time.win_rate if this_week.total_closed else 0.0
    expectancy_delta = this_week.expectancy - all_time.expectancy if this_week.total_closed else 0.0

    best_tag, worst_tag = _best_worst_strategy(this_week)
    streak_len, streak_type = _current_streak(all_trades)

    return WeeklyComparison(
        all_time=all_time,
        this_week=this_week,
        win_rate_delta=win_rate_delta,
        expectancy_delta=expectancy_delta,
        best_strategy_this_week=best_tag,
        worst_strategy_this_week=worst_tag,
        current_streak_len=streak_len,
        current_streak_type=streak_type,
    )


def render_comparison(comparison: WeeklyComparison) -> str:
    a, w = comparison.all_time, comparison.this_week

    if w.total_closed == 0:
        return "# Weekly Performance Review\n\nNo trades closed this week."

    lines = [
        "# Weekly Performance Review",
        "",
        "## This week",
        f"- Closed trades: {w.total_closed}",
        f"- Win rate: {w.win_rate:.1%} ({comparison.win_rate_delta:+.1%} vs all-time)",
        f"- Expectancy: {w.expectancy:.2f}R ({comparison.expectancy_delta:+.2f}R vs all-time)",
        f"- P&L: {w.total_pnl:.2f}",
        "",
        "## All-time baseline",
        f"- Closed trades: {a.total_closed}",
        f"- Win rate: {a.win_rate:.1%}",
        f"- Expectancy: {a.expectancy:.2f}R",
        f"- P&L: {a.total_pnl:.2f}",
        "",
    ]
    if comparison.best_strategy_this_week:
        lines.append(f"- Best setup this week: {comparison.best_strategy_this_week}")
    if comparison.worst_strategy_this_week and comparison.worst_strategy_this_week != comparison.best_strategy_this_week:
        lines.append(f"- Worst setup this week: {comparison.worst_strategy_this_week}")
    if comparison.current_streak_len:
        lines.append(f"- Current streak: {comparison.current_streak_len} {comparison.current_streak_type}s")

    return "\n".join(lines)


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
