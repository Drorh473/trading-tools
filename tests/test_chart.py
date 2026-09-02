"""Regression tests for notifier/chart.py.

Written AFTER render()/build() (built and visually approved before the TDD
requirement was adopted partway through this feature) - these lock in the
contract the four strategy overlays are about to build on top of, not a
red-then-green history for this file specifically.
"""

import numpy as np
import pandas as pd
import pytest

from notifier.chart import ChartOverlay, build, render


def _bars(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.random(n)
    low = np.minimum(open_, close) - rng.random(n)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=n, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "base_vol": rng.random(n) * 100,
            "quote_vol": rng.random(n) * 1000,
        }
    )


def test_render_returns_png_bytes():
    bars = _bars()
    png = render(
        bars, symbol="BTCUSDT", strategy_tag="Strategy 1 1H", direction="long",
        entry=bars["close"].iloc[-1], stop=bars["close"].iloc[-1] * 0.97,
        target=bars["close"].iloc[-1] * 1.06,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_with_no_overlay_still_succeeds():
    bars = _bars()
    png = render(
        bars, symbol="ETHUSDT", strategy_tag="Strategy 4 1H OB1.0", direction="short",
        entry=100.0, stop=105.0, target=90.0, overlay=None,
    )
    assert len(png) > 0


def test_render_with_every_overlay_primitive_does_not_raise():
    bars = _bars()
    overlay = ChartOverlay(
        levels=[(bars["close"].iloc[-1] * 1.05, "box top", "#9a6a00")],
        series=[("ema9", bars["close"].ewm(span=9).mean(), "#3b5bdb")],
        markers=[(len(bars) - 30, bars["high"].iloc[-30], "impulse")],
        zones=[(len(bars) - 40, len(bars) - 10, bars["low"].iloc[-40:-10].min(), bars["high"].iloc[-40:-10].max(), "box")],
    )
    png = render(
        bars, symbol="TESTUSDT", strategy_tag="Strategy 3 1D/1H", direction="long",
        entry=bars["close"].iloc[-1], stop=bars["close"].iloc[-1] * 0.97,
        target=bars["close"].iloc[-1] * 1.08, overlay=overlay,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_handles_a_zone_extending_past_the_last_candle():
    """Strategy 3's box is still 'live' as of the most recent close - its
    zone's end_position can equal or exceed the last plotted bar."""
    bars = _bars()
    overlay = ChartOverlay(zones=[(len(bars) - 20, len(bars) + 5, 90.0, 110.0, "box")])
    png = render(
        bars, symbol="TESTUSDT", strategy_tag="Strategy 3 1D/1H", direction="long",
        entry=100.0, stop=95.0, target=115.0, overlay=overlay,
    )
    assert len(png) > 0


def test_render_handles_a_marker_outside_the_visible_window():
    """An overlay built against the full bars frame may reference a pivot
    older than candles_shown - it should be silently clipped, not raise."""
    bars = _bars(n=200)
    overlay = ChartOverlay(markers=[(5, bars["low"].iloc[5], "old pivot")])
    png = render(
        bars, symbol="TESTUSDT", strategy_tag="Strategy 1 1H", direction="long",
        entry=bars["close"].iloc[-1], stop=90.0, target=110.0, overlay=overlay,
        candles_shown=80,
    )
    assert len(png) > 0


def test_render_respects_candles_shown():
    """Fewer candles requested should still produce a valid chart - this is
    how Strategy 3's daily chart_timeframe gets a tighter window than an
    intraday strategy's."""
    bars = _bars(n=200)
    png = render(
        bars, symbol="TESTUSDT", strategy_tag="Strategy 3 1D/1H", direction="long",
        entry=100.0, stop=95.0, target=110.0, candles_shown=30,
    )
    assert len(png) > 0


class _FakeStrategy:
    tag = "Fake Strategy 1H"
    timeframes = ["1H"]
    chart_timeframe = None

    def chart_overlay(self, bars_by_timeframe, signal):
        return ChartOverlay(levels=[(signal.entry_price * 1.1, "resistance", "#9a6a00")])


class _FakeSignal:
    symbol = "BTCUSDT"
    strategy_tag = "Fake Strategy 1H"
    direction = "long"
    stop_loss = 90.0


def test_build_uses_chart_timeframe_when_set():
    strategy = _FakeStrategy()
    strategy.chart_timeframe = "1D"
    bars_by_tf = {"1H": _bars(n=10), "1D": _bars(n=100)}
    signal = _FakeSignal()

    png = build(bars_by_tf, strategy, signal, entry=100.0, target=110.0)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_falls_back_to_first_declared_timeframe_when_chart_timeframe_unset():
    strategy = _FakeStrategy()
    bars_by_tf = {"1H": _bars(n=100)}
    signal = _FakeSignal()

    png = build(bars_by_tf, strategy, signal, entry=100.0, target=110.0)

    assert png is not None


def test_build_returns_none_when_the_chart_timeframe_bars_are_missing():
    strategy = _FakeStrategy()
    bars_by_tf = {"4H": _bars(n=100)}  # strategy wants "1H", never fetched
    signal = _FakeSignal()

    assert build(bars_by_tf, strategy, signal, entry=100.0, target=110.0) is None


def test_build_returns_none_rather_than_raising_when_chart_overlay_misbehaves():
    """A broken chart_overlay() must cost the chart, never the alert."""

    class _BrokenStrategy(_FakeStrategy):
        def chart_overlay(self, bars_by_timeframe, signal):
            raise RuntimeError("boom")

    bars_by_tf = {"1H": _bars(n=100)}
    signal = _FakeSignal()

    png = build(bars_by_tf, _BrokenStrategy(), signal, entry=100.0, target=110.0)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # candles alone, overlay dropped


def test_build_returns_none_when_there_are_too_few_bars():
    strategy = _FakeStrategy()
    bars_by_tf = {"1H": _bars(n=3)}
    signal = _FakeSignal()

    assert build(bars_by_tf, strategy, signal, entry=100.0, target=110.0) is None
