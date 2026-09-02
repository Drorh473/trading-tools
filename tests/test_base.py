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


# ---------------------------------------------------------------------------
# explain(): the default fallback every strategy gets for free, wrapping
# evaluate() itself - a strategy earns a richer ladder by overriding this,
# same pattern as chart_overlay.
# ---------------------------------------------------------------------------


class _FiresStrategy(_ConcreteStrategy):
    def evaluate(self, symbol, bars_by_timeframe):
        return Signal(symbol=symbol, direction="long", entry_price=100.0, stop_loss=90.0, strategy_tag=self.tag)


def test_default_explain_reports_a_fired_signal():
    result = _FiresStrategy().explain("BTCUSDT", {})

    assert result.fired is True
    assert result.signal is not None
    assert len(result.checks) == 1
    assert result.checks[0].passed is True


def test_default_explain_reports_no_signal():
    result = _ConcreteStrategy().explain("BTCUSDT", {})

    assert result.fired is False
    assert result.signal is None
    assert len(result.checks) == 1
    assert result.checks[0].passed is False


def test_default_explain_has_an_empty_funnel():
    """The funnel is for strategies that search an internal candidate space
    (Strategy 3's box-length sweep, say) - nothing to report by default."""
    result = _ConcreteStrategy().explain("BTCUSDT", {})

    assert result.funnel == {}
