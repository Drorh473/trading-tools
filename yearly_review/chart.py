"""Renders the year's monthly equity curve and per-strategy P&L as PNGs.

Same rendering approach as notifier.chart (Agg backend, PNG bytes via
BytesIO) - a second matplotlib dependency would be redundant, a second style
would look like a different product.
"""

from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from journal.stats import StrategyBreakdown

UP_COLOR = "#1a7f4b"
DOWN_COLOR = "#b3261e"
GRID_COLOR = "#e2e0dc"
AXIS_COLOR = "#9a9690"


def equity_curve(points: list[tuple[date, float]], year_label: str) -> bytes | None:
    """One point per monthly report that successfully read equity - see
    monthly_review.snapshot.history for why that, and not a daily curve, is
    what actually exists to plot: Bitget has no endpoint for equity on a past
    date, so anything finer than "whenever the monthly job happened to run"
    would have to be invented rather than read.

    None rather than a one-point or empty chart - see build()'s docstring in
    notifier/chart.py for the same "fails soft, never blocks the report"
    reasoning. A single dot has no shape to read; two points at least draw a
    line.
    """
    if len(points) < 2:
        return None

    ordered = sorted(points)
    xs = list(range(len(ordered)))
    # The closing snapshot is dated into the next year (see analyze.py's own
    # note on why it's included), so a plain "%b" would print two "Jan"
    # labels back to back. Only that one point ever needs the year appended.
    first_year = ordered[0][0].year
    labels = [d.strftime("%b") if d.year == first_year else d.strftime("%b ’%y") for d, _ in ordered]
    ys = [equity for _, equity in ordered]
    color = UP_COLOR if ys[-1] >= ys[0] else DOWN_COLOR

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(xs, ys, min(ys) - (max(ys) - min(ys) or max(ys) or 1) * 0.08, color=color, alpha=0.08, zorder=1)
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.6, markeredgecolor=color, zorder=3)

    # The starting value, held as a faint reference so a reader can see the
    # curve's excursions above and below where the year began without doing
    # the arithmetic themselves.
    ax.axhline(ys[0], color=AXIS_COLOR, linewidth=0.8, linestyle=":", zorder=1)

    # The one point worth a reader's eye more than the rest: where the year
    # actually ended up. A plain line of monthly dots reads as a shape, not a
    # number, until one point on it is called out.
    ax.scatter([xs[-1]], [ys[-1]], color=color, s=55, zorder=4, edgecolor="white", linewidth=1.2)
    ax.annotate(
        f"${ys[-1]:,.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
        fontsize=10, fontweight="bold", color=color, va="center", ha="left", zorder=5,
    )

    span = max(ys) - min(ys) or max(ys) or 1
    ax.set_xlim(-0.6, len(xs) - 1 + (span * 0 + 1.4))
    ax.set_ylim(min(ys) - span * 0.15, max(ys) + span * 0.22)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5, color=AXIS_COLOR)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"${v:,.0f}"))
    ax.tick_params(axis="y", labelsize=9.5, colors=AXIS_COLOR, length=0)
    ax.tick_params(axis="x", length=0)

    ax.set_title(f"Equity — {year_label}", fontsize=12, fontweight="bold", loc="left", color="#1b2430")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def strategy_pnl(by_strategy: dict[str, StrategyBreakdown], year_label: str) -> bytes | None:
    """Each strategy's realized P&L this year, one horizontal bar per
    strategy, longest-to-shortest.

    A pie can't draw a negative slice, which is exactly the number a strategy
    review needs to see - Strategy 1 1H losing money this year (see
    render._concerns) has to show up as plainly as a strategy that made
    money. A bar can cross zero; a pie cannot. Green/red follow the same
    up/down convention as equity_curve above rather than a per-strategy
    rotation, because P&L only ever has the one meaning here.

    The trade count rides on the axis LABEL rather than a separate legend or
    text section - see render._by_strategy's removal (Dror, 2026-09-04): this
    chart is now the only place that count lives.
    """
    strategies = {tag: b for tag, b in by_strategy.items() if b.count > 0}
    if not strategies:
        return None

    tags = sorted(strategies, key=lambda t: strategies[t].total_pnl, reverse=True)
    pnls = [strategies[t].total_pnl for t in tags]
    labels = [f"{t}  ({strategies[t].count} trade{'s' if strategies[t].count != 1 else ''})" for t in tags]
    colors = [UP_COLOR if p >= 0 else DOWN_COLOR for p in pnls]

    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.85 * len(tags) + 1.1)), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ys = range(len(tags))
    ax.barh(list(ys), pnls, color=colors, height=0.55, zorder=3)
    ax.axvline(0, color=AXIS_COLOR, linewidth=0.9, zorder=2)

    span = max(abs(p) for p in pnls) or 1
    for y, p in zip(ys, pnls):
        offset = span * 0.03
        ax.text(
            p + (offset if p >= 0 else -offset), y, f"{p:+,.2f}",
            va="center", ha="left" if p >= 0 else "right",
            fontsize=10, fontweight="bold", color=UP_COLOR if p >= 0 else DOWN_COLOR, zorder=4,
        )

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=10.5, color="#1b2430")
    ax.invert_yaxis()  # best strategy on top, matching the sort above
    ax.set_xlim(-span * 1.35, span * 1.35)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"${v:,.0f}"))
    ax.tick_params(axis="x", labelsize=9.5, colors=AXIS_COLOR, length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_title(f"P&L by strategy — {year_label}", fontsize=12, fontweight="bold", loc="left", color="#1b2430")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
