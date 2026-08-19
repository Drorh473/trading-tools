"""Generic trade statistics, computed over whatever list of trades is given.
Used both for the on-demand all-time report and, twice over, for the weekly
review's all-time-vs-this-week comparison — the date-range filtering happens
one level up via Storage.read_all(start, end).
"""

from dataclasses import dataclass, field

from core.storage import SignalRecord, Trade


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


@dataclass
class PaperBreakdown:
    count: int
    win_rate: float
    expectancy: float


@dataclass
class PaperStats:
    total_resolved: int
    win_rate: float
    expectancy: float
    # Keyed "Strategy 1 1H long" etc. - strategy tag and direction together,
    # since the same instance can behave very differently long vs short (the
    # short side of Strategy 2 carries an extra volume filter the long side
    # doesn't).
    by_strategy_direction: dict[str, PaperBreakdown] = field(default_factory=dict)
    # Keyed "approved" / "rejected" / "ignored" - what the signal alone would
    # have done versus what you chose to act on.
    by_decision: dict[str, PaperBreakdown] = field(default_factory=dict)


def compute_paper_stats(signals: list[SignalRecord]) -> PaperStats:
    resolved_all = [s for s in signals if s.paper_r is not None]

    def _breakdown(group: list[SignalRecord]) -> PaperBreakdown:
        wins = [s for s in group if s.paper_r > 0]
        return PaperBreakdown(
            count=len(group),
            win_rate=len(wins) / len(group),
            expectancy=sum(s.paper_r for s in group) / len(group),
        )

    by_decision = {
        key: _breakdown([s for s in resolved_all if (s.decision or "ignored") == key])
        for key in {(s.decision or "ignored") for s in resolved_all}
    }

    # A "too_small" or "swing_slots_full" signal was never a placeable trade -
    # it's the account (or the swing pool's own hard cap) telling you a trade
    # wasn't possible, not a judgment call about a candidate one. Blending
    # either into the headline win rate / expectancy or the per-strategy
    # breakdown would read a sizing/capacity artifact as a strategy-quality
    # signal, so both are excluded and only visible through by_decision.
    #
    # "refused_at_fill" and "send_failed" join them, for the same reason and one
    # that now matters more. Neither ever reached Dror:
    #
    #   refused_at_fill  the strategy's own gates, re-asked against the price
    #                    the trade would fill at, declined it. Scoring it under
    #                    the strategy's name measures trades that strategy
    #                    explicitly refuses to take.
    #   send_failed      Telegram did not deliver. That is an infrastructure
    #                    outcome and says nothing about the setup.
    #
    # Both were added on 2026-08-19, and both write a signals row - so without
    # this they would flow straight into by_strategy_direction. That is the
    # number Strategy 2.1's 1H hold is deliberately being watched on: it ships
    # at 10 rather than the 20 the backtest prefers precisely to accumulate a
    # live sample faster, and a live sample blended with refusals would answer
    # a different question than the one being asked of it.
    #
    # Still visible through by_decision, which is where "would the guard have
    # refused winners" gets answered.
    NOT_A_CANDIDATE = ("too_small", "swing_slots_full", "refused_at_fill", "send_failed")
    resolved = [s for s in resolved_all if s.decision not in NOT_A_CANDIDATE]

    if not resolved:
        return PaperStats(total_resolved=0, win_rate=0.0, expectancy=0.0, by_decision=by_decision)

    by_strategy_direction = {
        key: _breakdown([s for s in resolved if f"{s.strategy_tag} {s.direction}" == key])
        for key in {f"{s.strategy_tag} {s.direction}" for s in resolved}
    }

    return PaperStats(
        total_resolved=len(resolved),
        win_rate=len([s for s in resolved if s.paper_r > 0]) / len(resolved),
        expectancy=sum(s.paper_r for s in resolved) / len(resolved),
        by_strategy_direction=by_strategy_direction,
        by_decision=by_decision,
    )
