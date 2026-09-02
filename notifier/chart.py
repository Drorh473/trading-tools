"""Renders a candlestick PNG for one signal, with per-strategy overlays.

WHY THIS EXISTS
  Dror decides Approve/Reject by reading a chart, and the alert has always
  been text only - entry, stop, target as numbers. Every real decision costs
  a tab-switch to TradingView, re-finding the exact symbol and re-drawing by
  eye the same levels the strategy already computed, inside the same
  expiry window (see telegram_bot.SIGNAL_MOVEMENT_FRACTION) that is ticking
  the whole time. A picture answers "does this look like the setup" in the
  time it takes to glance at a phone.

WHY OVERLAYS ARE PER-STRATEGY, NOT GENERIC
  Entry/stop/target alone would still be an improvement, but the more useful
  question is "did the strategy read the chart correctly" - which needs the
  EVIDENCE it fired on, not just its conclusion. Strategy 1's Fib swing,
  Strategy 3's consolidation box, Strategy 2.1's EMA stack, Strategy 4's
  order block, are each a different shape and none of them is derivable from
  the other three - so each strategy contributes its own overlay through
  Strategy.chart_overlay() (see notifier/strategies/base.py) rather than this
  module trying to infer one.

  ChartOverlay is a plain-data vocabulary (horizontal levels, indicator
  series, single markers, shaded zones) precisely so a strategy module never
  has to import matplotlib and this module never has to know what a Fib leg
  or an order block IS - it only knows how to draw a level, a line, a point,
  and a box.

FAILS SOFT. A signal must still reach Telegram if the chart can't be built -
see build()'s own docstring - matching how confluence/pending_pattern
detection in the scanner already treat their own failures as "send the alert
anyway", not as a reason to drop it.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Enough candles to see the swing, box or block a strategy read - not the
# whole fetched history, which would squash the actual setup into a handful
# of pixels. 80 was chosen against Strategy 3's box (its consolidations run
# 10-60 daily bars per find_consolidation) and Strategy 1's Fib leg, both of
# which still read clearly at this width; a strategy needing more of its own
# history back can ask for it via chart_overlay's own data if it truly must,
# but nothing here has needed to yet.
CANDLES_SHOWN = 80

UP_COLOR = "#1a7f4b"
DOWN_COLOR = "#b3261e"
ENTRY_COLOR = "#3b5bdb"
STOP_COLOR = "#b3261e"
TARGET_COLOR = "#1a7f4b"
LEVEL_COLOR = "#9a6a00"
ZONE_COLOR = "#3b5bdb"
MARKER_COLOR = "#9a6a00"


@dataclass
class ChartOverlay:
    """What a strategy wants drawn on top of its own candles, beyond the
    entry/stop/target lines render() always draws.

    Every coordinate that references a bar is a POSITION in the same bars
    DataFrame chart_overlay() was handed (0 = its first row), never a
    timestamp - render() slices that frame down to the tail it actually
    plots and re-bases positions onto it, so a strategy never needs to know
    how many candles will end up on screen.
    """

    # Horizontal price levels, drawn edge-to-edge: (price, label, color).
    levels: list[tuple[float, str, str]] = field(default_factory=list)
    # Indicator lines over the SAME bars the candles come from - (name,
    # series, color). The series is aligned by pandas index against `bars`,
    # so pass the indicator computed over the strategy's own full frame
    # (e.g. ema(bars["close"], 9)) rather than a pre-sliced tail.
    series: list[tuple[str, pd.Series, str]] = field(default_factory=list)
    # A single labelled point: (bar_position, price, label) - the pivot or
    # impulse candle a setup is anchored on.
    markers: list[tuple[int, float, str]] = field(default_factory=list)
    # A shaded rectangle in (bar_position, price) space: (start_position,
    # end_position, low_price, high_price, label). end_position may exceed
    # the last plotted bar - Strategy 3's box, for one, is still "live" as of
    # the most recent close, and the zone should visibly run up to the
    # candles rather than stopping short of them.
    zones: list[tuple[int, int, float, float, str]] = field(default_factory=list)


def render(
    bars: pd.DataFrame,
    *,
    symbol: str,
    strategy_tag: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    overlay: ChartOverlay | None = None,
    candles_shown: int = CANDLES_SHOWN,
) -> bytes:
    """One PNG: candles, entry/stop/target, and whatever `overlay` supplies.

    `bars` is oldest-first OHLCV with a contiguous 0-based index (what
    scanner._bars() returns) and may include the still-forming candle - it is
    drawn like any other but is not closed, which is why the x-axis label
    below it reads "now" rather than a bar count.
    """
    tail = bars.tail(candles_shown).reset_index(drop=True)
    offset = len(bars) - len(tail)  # positions in `bars` -> positions in `tail`
    n = len(tail)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if overlay is not None:
        for start, end, y0, y1, label in overlay.zones:
            x0, x1 = start - offset, end - offset
            ax.add_patch(
                Rectangle(
                    (x0 - 0.5, min(y0, y1)),
                    max(x1 - x0 + 1, 0.5),
                    abs(y1 - y0),
                    facecolor=ZONE_COLOR,
                    alpha=0.10,
                    edgecolor=ZONE_COLOR,
                    linewidth=0.8,
                    zorder=1,
                )
            )
            if label:
                ax.text(
                    max(x1, x0) + 0.6, max(y0, y1), label, fontsize=7.5,
                    color=ZONE_COLOR, va="bottom", zorder=4,
                )

    # Candles. A plain up/down wick-and-body plot, not mplfinance - the whole
    # display is entry/stop/target plus a handful of overlay primitives, not
    # worth a dependency for.
    width = 0.6
    for i, row in tail.iterrows():
        color = UP_COLOR if row["close"] >= row["open"] else DOWN_COLOR
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1, zorder=2)
        body_low, body_high = sorted((row["open"], row["close"]))
        ax.add_patch(
            Rectangle(
                (i - width / 2, body_low), width, max(body_high - body_low, (body_high * 0.0005) or 1e-9),
                facecolor=color, edgecolor=color, zorder=3,
            )
        )

    if overlay is not None:
        for name, series, color in overlay.series:
            aligned = series.reindex(bars.index).iloc[offset:].to_numpy()
            ax.plot(range(n), aligned, color=color, linewidth=1.3, label=name, zorder=5)

        for position, price, label in overlay.markers:
            x = position - offset
            if -1 <= x <= n:
                ax.scatter([x], [price], color=MARKER_COLOR, s=28, zorder=6, marker="D")
                ax.annotate(
                    label, (x, price), textcoords="offset points", xytext=(0, 8),
                    fontsize=7.5, color=MARKER_COLOR, ha="center", zorder=6,
                )

        for price, label, color in overlay.levels:
            ax.axhline(price, color=color, linewidth=0.9, linestyle=":", zorder=4)
            if label:
                ax.text(
                    n - 0.5, price, f" {label}", fontsize=7.5, color=color, va="center", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                )

    # Entry/stop/target always drawn last, so they read as the headline
    # levels even on a busy strategy overlay. Labelled on the RIGHT, past the
    # overlay's own level labels - not on the left, where a bold multi-digit
    # price collided with matplotlib's own y-axis tick numbers (every price
    # under 1000 still overlapped "100"/"200"/etc: the text is sized in
    # points, not data units, so at this figure's scale it routinely spanned
    # several data-x-units past where it was anchored, well past the left
    # spine and into the tick-label margin).
    label_x = n + 8
    for price, color, label in ((entry, ENTRY_COLOR, "Entry"), (stop, STOP_COLOR, "Stop"), (target, TARGET_COLOR, "Target")):
        ax.axhline(price, color=color, linewidth=1.4, linestyle="--", zorder=7)
        ax.text(
            label_x, price, f"{label} {price:g}", fontsize=8, color=color, va="center", ha="left",
            fontweight="bold", zorder=7, bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )

    ax.set_xlim(-2.5, label_x + 14)
    lo = min([tail["low"].min(), stop, entry, target] + [p for p, _, _ in (overlay.levels if overlay else [])])
    hi = max([tail["high"].max(), stop, entry, target] + [p for p, _, _ in (overlay.levels if overlay else [])])
    pad = (hi - lo) * 0.06 or hi * 0.01 or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([])
    ax.set_title(f"{symbol} {direction.upper()}  ·  {strategy_tag}", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e2e0dc", linewidth=0.6, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    if overlay is not None and overlay.series:
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def build(
    bars_by_timeframe: dict[str, pd.DataFrame],
    strategy,
    signal,
    entry: float,
    target: float,
) -> bytes | None:
    """The chart for one dispatched signal, or None if it can't be built.

    Never raises - called from the dispatch path right before the alert goes
    out, and a chart is an enhancement to that alert, not a precondition for
    it. A strategy whose chart_overlay() misbehaves should lose its picture,
    not its trade.
    """
    try:
        timeframe = strategy.chart_timeframe or strategy.timeframes[0]
        bars = bars_by_timeframe.get(timeframe)
        if bars is None or len(bars) < 5:
            return None
        try:
            overlay = strategy.chart_overlay(bars_by_timeframe, signal)
        except Exception:
            logger.exception(
                "chart_overlay failed for %s/%s; charting candles alone", signal.symbol, signal.strategy_tag
            )
            overlay = None
        return render(
            bars,
            symbol=signal.symbol,
            strategy_tag=signal.strategy_tag,
            direction=signal.direction,
            entry=entry,
            stop=signal.stop_loss,
            target=target,
            overlay=overlay,
        )
    except Exception:
        logger.exception("Could not render a chart for %s/%s; sending text only", signal.symbol, signal.strategy_tag)
        return None
