"""Standalone premise test: is Dror's own pre-trade BTC read (trend structure
plus proximity to a level that could end it - notifier/strategies/regime.py's
daily_regime_read) a real, persistent market phenomenon, independent of any
specific strategy's trade history?

Not a gate, not wired to any strategy. Answers one question: does today's
regime label say anything real about where BTC goes next.

Data: BTCUSDT 1H from data/bars_1h_deep.pkl, resampled to daily. Bitget's own
BTCUSDT perp lists from 2019-07-10 (~7 years) - not the ~2014 spot history
that exists elsewhere, by Dror's own call: "go only up to 2019 its enough
data".

No look-ahead: day i's label is read from bars_so_far = daily.iloc[:i] -
everything strictly BEFORE day i's own bar - matching "what you'd know at
today's open, before today's candle exists".

    python -m backtest.btc_daily_regime_persistence [horizons...]
"""
import pickle
import sys

import numpy as np
import pandas as pd

from notifier.strategies.regime import daily_regime_from_bars

BARS_1H = "data/bars_1h_deep.pkl"
MIN_LOOKBACK = 200  # structure_context's own DEFAULT_MIN_LOOKBACK - daily_regime_from_bars uses it too
DEFAULT_HORIZONS = (5, 20)  # ~1 week, ~1 month of daily bars


def _daily_bars() -> pd.DataFrame:
    with open(BARS_1H, "rb") as f:
        h1 = pickle.load(f)["BTCUSDT"]
    d = h1.copy()
    d["t"] = pd.to_datetime(d["ts"], unit="ms")
    daily = (
        d.set_index("t")
        .resample("1D")
        .agg({"ts": "first", "open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index(drop=True)
    )
    return daily


def label_series(daily: pd.DataFrame) -> pd.Series:
    """One label per day, read from everything strictly before it.

    Delegates to daily_regime_from_bars - the same function a live strategy
    gate calls - rather than a second, hand-rolled copy of the growing-window
    search that could silently drift from what actually runs live.
    """
    labels = pd.Series([None] * len(daily), dtype=object)
    for i in range(MIN_LOOKBACK, len(daily)):
        labels.iloc[i] = daily_regime_from_bars(daily.iloc[:i])
    return labels


def _runs(s: pd.Series) -> list[tuple[object, int]]:
    out, cur, n = [], object(), 0
    for v in s:
        if v != cur:
            if n:
                out.append((cur, n))
            cur, n = v, 1
        else:
            n += 1
    if n:
        out.append((cur, n))
    return out


def _permute_preserving_runs(s: pd.Series, rng: np.random.Generator) -> pd.Series:
    vals = []
    for v, n in _runs(s):
        nv = None if v is None else rng.choice(["up", "down"])
        vals.extend([nv] * n)
    return pd.Series(vals, index=s.index)


def _t_stat(x: np.ndarray) -> float:
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def measure(daily: pd.DataFrame, labels: pd.Series, horizons=DEFAULT_HORIZONS, n_perm: int = 200) -> None:
    close = daily["close"].values
    counts = labels.value_counts(dropna=False)
    print("label counts:", dict(counts))
    real = [v for v in labels if v is not None]
    runs = _runs(labels)
    real_runs = [r for r in runs if r[0] is not None]
    print("regime episodes (runs of up/down, ignoring None gaps): %d" % len(real_runs))
    if real_runs:
        print("  run lengths:", [n for _v, n in real_runs])

    for h in horizons:
        fwd = np.full(len(daily), np.nan)
        fwd[: len(daily) - h] = close[h:] / close[: len(daily) - h] - 1.0
        up = np.array([fwd[i] for i in range(len(daily)) if labels.iloc[i] == "up" and not np.isnan(fwd[i])])
        down = np.array([fwd[i] for i in range(len(daily)) if labels.iloc[i] == "down" and not np.isnan(fwd[i])])
        base = fwd[~np.isnan(fwd)]
        sep = (up.mean() if len(up) else float("nan")) - (down.mean() if len(down) else float("nan"))
        print(
            "\n[horizon=%dd] up: n=%d mean=%+.4f t=%+.2f | down: n=%d mean=%+.4f t=%+.2f | "
            "baseline: n=%d mean=%+.4f | separation=%+.4f"
            % (h, len(up), up.mean() if len(up) else float("nan"), _t_stat(up),
               len(down), down.mean() if len(down) else float("nan"), _t_stat(down),
               len(base), base.mean(), sep)
        )

        rng = np.random.default_rng(12345)
        perm_seps = []
        for _ in range(n_perm):
            pl = _permute_preserving_runs(labels, rng)
            pu = np.array([fwd[i] for i in range(len(daily)) if pl.iloc[i] == "up" and not np.isnan(fwd[i])])
            pd_ = np.array([fwd[i] for i in range(len(daily)) if pl.iloc[i] == "down" and not np.isnan(fwd[i])])
            if len(pu) and len(pd_):
                perm_seps.append(pu.mean() - pd_.mean())
        perm_seps = np.array(perm_seps)
        pctile = 100 * (perm_seps < sep).mean() if len(perm_seps) else float("nan")
        p_ge = (perm_seps >= sep).mean() if len(perm_seps) else float("nan")
        print(
            "  permutation (run-structure preserved, n=%d): perm mean=%+.4f sd=%.4f | "
            "real at %.0f pctile | P(perm>=real)=%.2f"
            % (len(perm_seps), perm_seps.mean() if len(perm_seps) else float("nan"),
               perm_seps.std(ddof=1) if len(perm_seps) > 1 else float("nan"), pctile, p_ge)
        )


if __name__ == "__main__":
    horizons = tuple(int(a) for a in sys.argv[1:]) or DEFAULT_HORIZONS
    daily = _daily_bars()
    span_lo = pd.to_datetime(daily["ts"].iloc[0], unit="ms")
    span_hi = pd.to_datetime(daily["ts"].iloc[-1], unit="ms")
    print("BTC daily bars: %d, %s -> %s" % (len(daily), span_lo.date(), span_hi.date()))
    labels = label_series(daily)
    measure(daily, labels, horizons=horizons)
