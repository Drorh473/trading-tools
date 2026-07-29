"""Generic trade statistics, computed over whatever list of trades is given.
Used both for the on-demand all-time report and, twice over, for the weekly
review's all-time-vs-this-week comparison — the date-range filtering happens
one level up via Storage.read_all(start, end).
"""

from dataclasses import dataclass, field

from core.storage import Trade


@dataclass
class StrategyBreakdown:
    count: int
    win_rate: float
    expectancy: float
    total_pnl: float


@dataclass
class Stats:
    total_closed: int
    win_rate: float
    expectancy: float
    total_pnl: float
    max_drawdown: float
    r_multiples: list[float]
    equity_curve: list[float]
    best_trade: Trade | None
    worst_trade: Trade | None
    changed_from_plan_count: int = 0
    by_strategy: dict[str, StrategyBreakdown] = field(default_factory=dict)


def compute_stats(trades: list[Trade]) -> Stats:
    closed = [t for t in trades if t.מכפיל_R is not None]

    if not closed:
        return Stats(
            total_closed=0,
            win_rate=0.0,
            expectancy=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            r_multiples=[],
            equity_curve=[],
            best_trade=None,
            worst_trade=None,
        )

    r_multiples = [t.מכפיל_R for t in closed]
    pnls = [t.רווח_הפסד for t in closed]
    wins = [t for t in closed if t.רווח_הפסד > 0]

    equity_curve = _cumulative(pnls)

    return Stats(
        total_closed=len(closed),
        win_rate=len(wins) / len(closed),
        expectancy=sum(r_multiples) / len(r_multiples),
        total_pnl=sum(pnls),
        max_drawdown=_max_drawdown(equity_curve),
        r_multiples=r_multiples,
        equity_curve=equity_curve,
        best_trade=max(closed, key=lambda t: t.רווח_הפסד),
        worst_trade=min(closed, key=lambda t: t.רווח_הפסד),
        changed_from_plan_count=sum(1 for t in closed if t.changed_from_plan),
        by_strategy=_breakdown_by_strategy(closed),
    )


def _cumulative(pnls: list[float]) -> list[float]:
    curve = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        curve.append(running)
    return curve


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def _breakdown_by_strategy(closed: list[Trade]) -> dict[str, StrategyBreakdown]:
    tags = {t.תגית_אסטרטגיה or "(untagged)" for t in closed}
    breakdown = {}
    for tag in tags:
        group = [t for t in closed if (t.תגית_אסטרטגיה or "(untagged)") == tag]
        wins = [t for t in group if t.רווח_הפסד > 0]
        breakdown[tag] = StrategyBreakdown(
            count=len(group),
            win_rate=len(wins) / len(group),
            expectancy=sum(t.מכפיל_R for t in group) / len(group),
            total_pnl=sum(t.רווח_הפסד for t in group),
        )
    return breakdown
