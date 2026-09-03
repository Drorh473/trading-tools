"""Renders the weekly review's two charts as PNGs - per-strategy performance
and daily $ PnL - sent alongside the text report rather than replacing any of
it (see weekly_review/main.py).

Real trades, not paper-simulated signals: the text report dropped the real
trades section on 2026-08-27 because the sample was too small for a
week-vs-all-time COMPARISON, but a chart isn't making that claim - it's just
showing what actually happened this week, which is worth seeing regardless
of sample size.
"""

from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from journal.stats import StrategyBreakdown

UP_COLOR = "#1a7f4b"
DOWN_COLOR = "#b3261e"

# A qualitative palette for the pie's slices - distinct from UP_COLOR/
# DOWN_COLOR, which mean "profit" / "loss" everywhere else in this module and
# would misread as a win/loss signal here, where color means nothing but
# "which strategy". Cycles if there are ever more strategies than colors.
PIE_COLORS = ["#3b5bdb", "#f59f00", "#1a7f4b", "#b3261e", "#7048e8", "#0c8599", "#e64980", "#495057"]


def strategy_breakdown_chart(by_strategy: dict[str, StrategyBreakdown]) -> bytes | None:
    """Two panels sharing the same strategy rows: each strategy's SHARE of
    this week's trades as a pie, and its average R (expectancy) as a bar -
    a pie fits "how the week divided up between strategies" but not R, which
    can be negative and isn't a part-of-a-whole.

    None when nothing closed this week - nothing to draw, matching
    notifier/chart.py's own fail-soft convention rather than sending an
    empty picture.
    """
    if not by_strategy:
        return None

    tags = sorted(by_strategy, key=lambda t: by_strategy[t].count)
    counts = [by_strategy[t].count for t in tags]
    expectancies = [by_strategy[t].expectancy for t in tags]

    fig, (ax_count, ax_r) = plt.subplots(1, 2, figsize=(10, max(2.0, 0.5 * len(tags) + 1.2)))

    pie_colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(tags))]
    ax_count.pie(
        counts,
        labels=[f"{t} ({c})" for t, c in zip(tags, counts)],
        autopct="%1.0f%%",
        colors=pie_colors,
        startangle=90,
    )
    ax_count.set_title("Share of trades this week")

    colors = [UP_COLOR if r >= 0 else DOWN_COLOR for r in expectancies]
    ax_r.barh(tags, expectancies, color=colors)
    ax_r.axvline(0, color="black", linewidth=0.8)
    ax_r.set_title("Avg R this week")
    ax_r.set_xlabel("expectancy (R)")
    # A bar near zero (e.g. -0.12R) otherwise puts its label right where the
    # y-axis strategy names sit - a fixed-magnitude pad, not a fraction of the
    # data range, since a range near zero would pad by almost nothing.
    lo, hi = min(expectancies + [0.0]), max(expectancies + [0.0])
    pad = max((hi - lo) * 0.3, 0.3)
    ax_r.set_xlim(lo - pad, hi + pad)
    for y, r in enumerate(expectancies):
        _label_bar(ax_r, r, y, f"{r:+.2f}")

    fig.tight_layout()
    return _to_png(fig)


def _label_bar(ax, value: float, y: int, text: str) -> None:
    """A bar-tip label offset in PIXELS, not data coordinates, so it clears
    the bar by a constant distance regardless of how small the value is -
    the fix for a near-zero bar's label landing on top of the axis labels."""
    ax.annotate(
        text, xy=(value, y), xytext=(4 if value >= 0 else -4, 0),
        textcoords="offset points", va="center", ha="left" if value >= 0 else "right",
    )


def daily_pnl_chart(daily_pnl: dict[date, float]) -> bytes | None:
    """One bar per day of the week (Sun-Sat), green for a profitable day, red
    for a loss. Renders even an all-zero week - a clean week is itself worth
    seeing, not worth hiding - but None for a genuinely empty dict, the same
    "nothing to draw" case strategy_breakdown_chart guards against.
    """
    if not daily_pnl:
        return None

    days = sorted(daily_pnl)
    values = [daily_pnl[d] for d in days]
    labels = [f"{d:%a} {d.day}" for d in days]
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily PnL this week")
    ax.set_ylabel("$")
    for x, v in enumerate(values):
        ax.text(x, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top")

    fig.tight_layout()
    return _to_png(fig)


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
