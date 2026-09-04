"""yearly_review.chart: fails soft below the minimum shape each chart needs,
otherwise renders a PNG.
"""

from datetime import date

from journal.stats import StrategyBreakdown
from yearly_review import chart


def test_equity_curve_needs_at_least_two_points():
    assert chart.equity_curve([], "2026") is None
    assert chart.equity_curve([(date(2026, 3, 1), 100.0)], "2026") is None


def test_equity_curve_renders_a_png_for_two_or_more_points():
    points = [(date(2026, 2, 1), 100.0), (date(2026, 6, 1), 105.0), (date(2026, 10, 1), 98.0)]
    image = chart.equity_curve(points, "2026")

    assert image is not None
    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_strategy_pnl_needs_at_least_one_strategy_with_trades():
    assert chart.strategy_pnl({}, "2026") is None


def test_strategy_pnl_renders_a_single_bar():
    """A single strategy is still a legible bar chart - unlike a pie, which
    would just be a full circle with nothing to compare it against."""
    one = {"Strategy 4": StrategyBreakdown(count=5, win_rate=0.6, expectancy=0.3, total_pnl=50.0)}
    image = chart.strategy_pnl(one, "2026")

    assert image is not None
    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_strategy_pnl_renders_a_png_with_a_negative_bar():
    """The whole reason this replaced the pie chart: a strategy that lost
    money this year has to be drawable, not excluded for having no honest
    slice."""
    by_strategy = {
        "Strategy 1 1H": StrategyBreakdown(count=3, win_rate=0.3, expectancy=-0.4, total_pnl=-30.0),
        "Strategy 4": StrategyBreakdown(count=6, win_rate=0.7, expectancy=0.5, total_pnl=120.0),
    }
    image = chart.strategy_pnl(by_strategy, "2026")

    assert image is not None
    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_strategy_pnl_ignores_a_strategy_with_zero_trades():
    """A strategy present in by_strategy with zero closed trades has no P&L
    worth a bar and no trade count worth labelling."""
    by_strategy = {
        "Strategy 1 1H": StrategyBreakdown(count=5, win_rate=0.4, expectancy=-0.2, total_pnl=-30.0),
        "Strategy 3": StrategyBreakdown(count=0, win_rate=0.0, expectancy=0.0, total_pnl=0.0),
    }
    image = chart.strategy_pnl(by_strategy, "2026")

    assert image is not None  # the one real strategy still draws
