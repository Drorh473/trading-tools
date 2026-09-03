from datetime import date, timedelta

from journal.stats import StrategyBreakdown
from weekly_review.charts import daily_pnl_chart, strategy_breakdown_chart

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_strategy_breakdown_chart_returns_png_bytes():
    by_strategy = {
        "Strategy 1 1H": StrategyBreakdown(count=5, win_rate=0.6, expectancy=0.42, total_pnl=12.3),
        "Strategy 4 1H": StrategyBreakdown(count=2, win_rate=0.5, expectancy=-0.1, total_pnl=-4.0),
    }

    png = strategy_breakdown_chart(by_strategy)

    assert png[:8] == PNG_MAGIC


def test_strategy_breakdown_chart_is_none_when_nothing_closed():
    """Nothing to draw - matching notifier/chart.py's own fail-soft
    convention rather than sending an empty picture."""
    assert strategy_breakdown_chart({}) is None


def test_strategy_breakdown_chart_handles_a_single_strategy():
    by_strategy = {"Strategy 3": StrategyBreakdown(count=1, win_rate=1.0, expectancy=1.5, total_pnl=8.0)}

    png = strategy_breakdown_chart(by_strategy)

    assert png[:8] == PNG_MAGIC


def _a_week_of_days(start=date(2026, 8, 30)):
    return {start + timedelta(days=i): 0.0 for i in range(7)}


def test_daily_pnl_chart_returns_png_bytes():
    days = _a_week_of_days()
    days[date(2026, 9, 1)] = 12.5
    days[date(2026, 9, 2)] = -6.75

    png = daily_pnl_chart(days)

    assert png[:8] == PNG_MAGIC


def test_daily_pnl_chart_still_renders_an_all_zero_week():
    """A clean week with nothing closed is itself worth seeing, not worth
    hiding - unlike the per-strategy chart, this one is never empty since
    analyze() always fills all seven days."""
    png = daily_pnl_chart(_a_week_of_days())

    assert png[:8] == PNG_MAGIC


def test_daily_pnl_chart_is_none_for_an_empty_dict():
    assert daily_pnl_chart({}) is None
