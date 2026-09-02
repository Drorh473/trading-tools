"""Renders a written report from a Stats result, plus equity-curve and
R-multiple histogram charts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from journal.stats import Stats


def render_markdown(stats: Stats, title: str = "Trade Report") -> str:
    if stats.total_closed == 0:
        return f"# {title}\n\nNo closed trades yet."

    lines = [
        f"# {title}",
        "",
        f"- Closed trades: {stats.total_closed}",
        f"- Win rate: {stats.win_rate:.1%}",
        f"- Expectancy (avg R): {stats.expectancy:.2f}R",
        f"- Total P&L: {stats.total_pnl:.2f}",
        f"- Max drawdown: {stats.max_drawdown:.2f}",
        f"- Best trade: {stats.best_trade.סימבול} ({stats.best_trade.רווח_הפסד:.2f})",
        f"- Worst trade: {stats.worst_trade.סימבול} ({stats.worst_trade.רווח_הפסד:.2f})",
        f"- Changed from plan: {stats.changed_from_plan_count}/{stats.total_closed} trades "
        f"(stop/target adjusted after the bot's original proposal)",
        "",
        "## By strategy",
        "",
    ]
    for tag, breakdown in sorted(stats.by_strategy.items()):
        lines.append(
            f"- **{tag}**: {breakdown.count} trades, "
            f"{breakdown.win_rate:.1%} win rate, "
            f"{breakdown.expectancy:.2f}R expectancy, "
            f"{breakdown.total_pnl:.2f} total P&L"
        )

    lines += ["", "## By symbol", ""]
    # Sorted by plain expectancy, not drop-top-3 - which symbols LOOK best is
    # the interesting ordering; whether that survives is the next column, not
    # a second silent ranking criterion.
    for symbol, breakdown in sorted(stats.by_symbol.items(), key=lambda kv: kv[1].expectancy, reverse=True):
        drop3 = f"{breakdown.expectancy_drop_top3:.2f}R" if breakdown.expectancy_drop_top3 is not None else "too few trades"
        lines.append(
            f"- **{symbol}**: {breakdown.count} trades, "
            f"{breakdown.win_rate:.1%} win rate, "
            f"{breakdown.expectancy:.2f}R expectancy ({drop3} without its best 3), "
            f"{breakdown.total_pnl:.2f} total P&L"
        )
    return "\n".join(lines)


def save_charts(stats: Stats, output_dir: str) -> tuple[str, str] | None:
    if stats.total_closed == 0:
        return None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    equity_path = output / "equity_curve.png"
    fig, ax = plt.subplots()
    ax.plot(stats.equity_curve)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L")
    fig.savefig(equity_path)
    plt.close(fig)

    hist_path = output / "r_distribution.png"
    fig, ax = plt.subplots()
    ax.hist(stats.r_multiples, bins=20)
    ax.set_title("R-Multiple Distribution")
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Count")
    fig.savefig(hist_path)
    plt.close(fig)

    return str(equity_path), str(hist_path)
