"""One shared summary over a list of closed trades, so win rate / total R /
expectancy / drop-top-3 / per-tag / exit-reason stats are computed the same
way everywhere a script reports a backtest result.

drop-top-3 in particular (memory: sweep-past-the-optimum) is easy to forget
in a new report script and easy to get subtly wrong (dropping the three
SMALLEST losers instead of the three biggest winners would flatter every
result) - one implementation means it can only be wrong once.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TagSummary:
    n: int = 0
    wins: int = 0
    total_r: float = 0.0
    pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def expectancy(self) -> float:
        return self.total_r / self.n if self.n else float("nan")


@dataclass
class Summary:
    n: int = 0
    wins: int = 0
    total_r: float = 0.0
    drop_top3_n: int | None = None
    drop_top3_total_r: float | None = None
    by_tag: dict[str, TagSummary] = field(default_factory=dict)
    exit_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def expectancy(self) -> float:
        return self.total_r / self.n if self.n else float("nan")

    @property
    def drop_top3_expectancy(self) -> float | None:
        if not self.drop_top3_n:
            return None
        return self.drop_top3_total_r / self.drop_top3_n


def summarize(trades) -> Summary:
    """trades: anything with .r, .pnl, .tag, .reason - engine.Closed, or a
    slice of one account's .closed."""
    trades = list(trades)
    n = len(trades)
    if n == 0:
        return Summary()

    wins = sum(1 for t in trades if t.pnl > 0)
    total_r = sum(t.r for t in trades)

    drop_top3_n = drop_top3_total_r = None
    if n > 3:
        # Drop the three BIGGEST winners by R, not the three worst losers -
        # an edge that only survives because of a few outsized wins is not
        # an edge, and dropping losers instead would flatter every result.
        kept = sorted(trades, key=lambda t: -t.r)[3:]
        drop_top3_n = len(kept)
        drop_top3_total_r = sum(t.r for t in kept)

    by_tag: dict[str, TagSummary] = {}
    for t in trades:
        row = by_tag.setdefault(t.tag, TagSummary())
        row.n += 1
        row.wins += 1 if t.pnl > 0 else 0
        row.total_r += t.r
        row.pnl += t.pnl

    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.reason] = exit_reasons.get(t.reason, 0) + 1

    return Summary(n=n, wins=wins, total_r=total_r,
                    drop_top3_n=drop_top3_n, drop_top3_total_r=drop_top3_total_r,
                    by_tag=by_tag, exit_reasons=exit_reasons)
