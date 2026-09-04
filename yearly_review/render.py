"""Renders the yearly report as Telegram-friendly markdown.

Same shape rule as monthly_review.render: a verdict at the top, evidence
below it. Numbers that could not be produced say so, rather than printing a
zero that would look identical to "not available".
"""

from journal.stats import Stats
from yearly_review.analyze import YearlyReport


def render(report: YearlyReport) -> str:
    lines = [f"# Yearly Review — {report.label}", ""]
    lines += _verdict(report)
    lines.append("")
    lines += _performance(report.stats)
    lines.append("")
    lines += _flagged(report)
    return "\n".join(lines)


def _verdict(report: YearlyReport) -> list[str]:
    """The whole report in a few lines. Read these; open the rest only if
    something below looks worth a closer read."""
    rec = report.reconciliation
    stats = report.stats

    if rec.actual is None:
        note = rec.equity_start_note or "unknown reason"
        balance = f"Balance:  not available ({note})"
    else:
        pct = (rec.actual / rec.equity_start * 100) if rec.equity_start else 0.0
        balance = (
            f"Balance:  ${rec.equity_start:.2f} → ${rec.equity_end:.2f} "
            f"({pct:+.1f}%, {rec.actual:+.2f})"
        )

    block = [
        "```",
        balance,
        f"Closed trades: {stats.total_closed}",
    ]
    if stats.total_closed:
        block.append(f"Win rate: {stats.win_rate:.0%}  Expectancy: {stats.expectancy:+.2f}R")

    # A count here, not the concerns themselves - see the "Flagged strategies"
    # section below for the itemised version. Same split monthly_review draws
    # between its verdict counts and its own Failures/Tuning sections.
    concerns = _concerns(report)
    block.append(f"Flagged: {len(concerns)} strateg{'y' if len(concerns) == 1 else 'ies'}")
    block.append("```")
    return block


# Dror's rule (2026-09-04): flag a strategy that is NEGATIVE over the year,
# or that has too FEW closed trades to say anything from - the two ways a
# by_strategy row can be worth a second look without needing a noise-band
# model the way monthly_review's slippage check does (that one exists
# because a few basis points of drift needs real statistics to separate from
# scatter; a strategy sitting on negative P&L or five trades a year does not
# need a test to be worth a glance).
#
# MIN_TRADES_TO_JUDGE is a starting number, not a measured one - adjust it
# once a year's worth of real counts makes clear what "too few" should mean
# for this account.
MIN_TRADES_TO_JUDGE = 5


def _concerns(report: YearlyReport) -> list[str]:
    lines = []
    for tag in sorted(report.stats.by_strategy):
        b = report.stats.by_strategy[tag]
        if b.total_pnl < 0:
            lines.append(f"{tag} is negative this year ({b.total_pnl:+.2f} over {b.count} trades)")
        elif b.count < MIN_TRADES_TO_JUDGE:
            lines.append(f"{tag} only closed {b.count} trade(s) — too few to read anything from")
    return lines


def _performance(stats: Stats) -> list[str]:
    lines = ["## Performance", ""]
    if not stats.total_closed:
        lines.append("No closed trades this year.")
        return lines

    lines += [
        f"- Closed trades: {stats.total_closed}",
        f"- Win rate: {stats.win_rate:.0%}",
        f"- Expectancy: {stats.expectancy:+.2f}R",
        f"- Total realized P&L: {stats.total_pnl:+.2f}",
        f"- Max drawdown (realized equity curve): {stats.max_drawdown:.2f}",
    ]
    if stats.best_trade is not None:
        lines.append(f"- Best trade: {stats.best_trade.רווח_הפסד:+.2f}")
    if stats.worst_trade is not None:
        lines.append(f"- Worst trade: {stats.worst_trade.רווח_הפסד:+.2f}")
    if stats.changed_from_plan_count:
        lines.append(f"- Trades whose stop or target was changed after the plan: {stats.changed_from_plan_count}")
    return lines


def _flagged(report: YearlyReport) -> list[str]:
    lines = ["## Flagged strategies", ""]
    concerns = _concerns(report)
    if not concerns:
        lines.append("Nothing negative and nothing too thin to read.")
        return lines
    for c in concerns:
        lines.append(f"- {c}")
    return lines
