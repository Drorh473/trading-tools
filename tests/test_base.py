from notifier.strategies.base import Signal, Strategy


class _ConcreteStrategy(Strategy):
    """The minimal subclass needed to exercise Strategy's own defaults -
    evaluate() is abstract and irrelevant to what's under test here."""

    tag = "Test Strategy"

    def evaluate(self, symbol, bars_by_timeframe):
        return None


def test_default_chart_timeframe_is_none():
    assert _ConcreteStrategy().chart_timeframe is None


def test_default_chart_overlay_returns_none():
    strategy = _ConcreteStrategy()
    signal = Signal(symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=90.0, strategy_tag="Test Strategy")
    assert strategy.chart_overlay({}, signal) is None
