"""Replay Strategy 4's deep signal set, one fresh account per calendar year.

    python -m backtest.replay_s4_deep

WHY PER YEAR, NOT ONE CONTINUOUS ACCOUNT. A single compounding $100 account
that hits a bad stretch shrinks below Bitget's flat $5 minimum notional and
then cannot place anything for the rest of the run, so trade count collapses
and every later year is measured on an account that is effectively dead. That
is a real property of the live account worth knowing about separately, but it
makes a strategy look consistent when what actually happened is that it
stopped trading. Each year therefore starts fresh at $100.

WHAT THIS CANNOT TELL YOU. Strategy 4 is one instance among five live ones and
is in DRY_RUN_TAGS - it places nothing today. Replayed alone it never competes
for margin or the one-position-per-symbol slot with Strategy 1 or 3, so these
numbers are Strategy 4's own edge, not the account's. That is the right first
question for a strategy with no measurement at all, but it is not "what would
the account have done".

Every headline is printed next to the same number with its best three trades
removed. A result that only survives with them is a result about three trades.

THE BINDING CONSTRAINT IS FILLS, NOT SIGNALS. Strategy 4 rests a limit at the
order block midpoint and the live bot cancels every unfilled entry at a flat
four hours (tracker.ENTRY_TIMEOUT_SECONDS = 4 bars on 1H). Measured on the 57
pre-existing signals, that window fills 1 of 56 - 2%. Widening it is the only
thing that moves the count, and it moves it a long way:

    cancel   taken  filled  fill%   equity  exits
         4      56       1     2%    99.00  1 stop
        10      56       3     5%    97.05  3 stops
        30      56      13    23%    87.73  13 stops, 0 targets
        96      55      25    45%    91.44  3 targets, 22 stops
       240      55      33    60%    89.55  4 targets, 29 stops

So a signal count is not a sample size here - divide it by about fifty before
believing anything. Note also that every window above loses money, and that
the instance's own configured window (30) took thirteen trades and won none.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import pandas as pd

from backtest import engine as bt
from backtest import stats
from backtest.portfolio import replay

BARS_DEFAULT = "data/bars_1h_deep_np.pkl"
SIGNALS_DEFAULT = "data/s4_signals_deep.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")

# Bars are kept this far past the year's end so a position opened in December
# can still resolve. Signals are NOT taken from the tail - it exists only to
# let the year's own trades finish.
TAIL_DAYS = 90
MIN_TRADES = 30  # below this an arm is not a result, it is an anecdote


def _frames(bars: dict, lo_ms: int, hi_ms: int) -> tuple[dict, dict]:
    """Year-sliced frames, plus the offset each slice was cut at.

    The offset is not bookkeeping - it is load-bearing. engine.try_open stores
    `pending_until = bar_index + cancel_after` from the index the SIGNAL
    carries, while engine.step_position tests `bar_index >= pos.pending_until`
    using the row index of the frame it is walking. Slicing a year out of the
    history restarts the row count near zero while the signal's index stays
    absolute (tens of thousands), so the test can never fire: no resting entry
    is ever cancelled, every one eventually fills, and the fill rate - the one
    number this whole exercise is about - comes out badly inflated. Callers
    must rebase signal indices onto the slice with these offsets.
    """
    out, offsets = {}, {}
    for sym, cols in bars.items():
        ts = cols["ts"]
        first, last = ts.searchsorted(lo_ms, "left"), ts.searchsorted(hi_ms, "right")
        if last - first < 2:
            continue
        out[sym] = pd.DataFrame(
            {c: cols[c][first:last] for c in COLUMNS if c in cols}
        ).reset_index(drop=True)
        offsets[sym] = int(first)
    return out, offsets


def _summarise(acct, label: str, note: str = "") -> dict:
    closed = acct.closed
    n = len(closed)
    # acct.taken counts positions CREATED, not trades filled. Strategy 4 rests a
    # limit at the order block midpoint, and engine.step_position deletes a leg
    # that never gets touched (`cancelled unfilled`) without ever appending it
    # to acct.closed. For a market-entry strategy taken and closed nearly
    # coincide, so taken reads like a trade count; here it is roughly 50x the
    # real one. The fill column exists so that gap cannot be misread.
    row = {"label": label, "taken": acct.taken, "closed": n,
           "fill": n / acct.taken if acct.taken else 0.0,
           "equity": acct.equity, "dd": acct.max_dd,
           "too_small": acct.declined_too_small, "note": note}
    if n:
        s = stats.summarize(closed)
        row |= {"win": s.win_rate, "totR": s.total_r, "expR": s.expectancy,
                "expR_drop3": s.drop_top3_expectancy if s.drop_top3_n else float("nan")}
    return row


def _print(rows: list[dict]) -> None:
    print(f"\n{'year':<8}{'signals':>9}{'taken':>7}{'filled':>8}{'fill%':>7}{'win':>6}"
          f"{'expR':>8}{'drop3':>8}{'equity':>9}{'maxDD':>7}{'too_small':>11}")
    for r in rows:
        if not r["closed"]:
            print(f"{r['label']:<8}{r.get('signals',0):>9}{r['taken']:>7}{0:>8}"
                  f"{r['fill']*100:>6.0f}%{'-':>6}{'-':>8}{'-':>8}"
                  f"{r['equity']:>8.0f}{'-':>7}{r['too_small']:>11}")
            continue
        thin = "" if r["closed"] >= MIN_TRADES else " *"
        d3 = r["expR_drop3"]
        d3s = f"{d3:>+8.2f}" if d3 == d3 else f"{'-':>8}"  # nan when <=3 trades
        print(f"{r['label']:<8}{r.get('signals',0):>9}{r['taken']:>7}{r['closed']:>8}"
              f"{r['fill']*100:>6.0f}%{r['win']*100:>5.0f}%{r['expR']:>+8.2f}{d3s}"
              f"{r['equity']:>8.0f}{r['dd']*100:>6.0f}%{r['too_small']:>10}{thin}")
    if any(0 < r["closed"] < MIN_TRADES for r in rows):
        print(f"  * fewer than {MIN_TRADES} closed trades - not a result")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=BARS_DEFAULT)
    ap.add_argument("--signals", default=SIGNALS_DEFAULT)
    ap.add_argument("--cancel", type=int, default=4,
                    help="unfilled-entry window in bars; 4 models the live flat 4h timeout")
    args = ap.parse_args()

    with open(args.bars, "rb") as fh:
        bars = pickle.load(fh)
    with open(args.signals, "rb") as fh:
        signals = pickle.load(fh)

    flat = [(sym, e) for sym, v in signals.items() for e in v]
    print(f"{len(flat)} signals across {sum(1 for v in signals.values() if v)} symbols")
    if not flat:
        print("nothing to replay")
        return
    stamps = sorted(e[0] for _s, e in flat)
    print(f"span {pd.Timestamp(stamps[0], unit='ms'):%Y-%m-%d} .. "
          f"{pd.Timestamp(stamps[-1], unit='ms'):%Y-%m-%d}")

    by_year = defaultdict(list)
    for sym, e in flat:
        by_year[pd.Timestamp(e[0], unit="ms").year].append((sym, e))
    print("signals per year: " + "  ".join(f"{y}:{len(v)}" for y, v in sorted(by_year.items())))

    rows = []
    for year in sorted(by_year):
        lo = int(pd.Timestamp(f"{year}-01-01").value // 10**6)
        hi = int(pd.Timestamp(f"{year+1}-01-01").value // 10**6)
        hi_tail = hi + TAIL_DAYS * 86_400_000
        year_signals = defaultdict(list)
        for sym, e in by_year[year]:
            year_signals[sym].append(e)
        frames, offsets = _frames({s: bars[s] for s in year_signals if s in bars}, lo, hi_tail)
        year_signals = {s: v for s, v in year_signals.items() if s in frames}
        if not year_signals:
            continue
        # Rebase onto the slice - see _frames. Guard rather than trust: a
        # negative index would silently mean the signal predates its own frame.
        year_signals = {
            s: [(ts, i - offsets[s], close, pos_i, sig)
                for (ts, i, close, pos_i, sig) in v
                if i - offsets[s] >= 0]
            for s, v in year_signals.items()
        }
        year_signals = {s: v for s, v in year_signals.items() if v}
        acct = replay(frames, year_signals, cancel_override=args.cancel)
        row = _summarise(acct, str(year))
        row["signals"] = sum(len(v) for v in year_signals.values())
        rows.append(row)
        print(f"  {year}: {row['closed']} closed", flush=True)

    _print(rows)

    closed_total = sum(r["closed"] for r in rows)
    print(f"\npooled across years: {closed_total} closed trades")
    if closed_total < MIN_TRADES:
        print("NOT ENOUGH TO MEASURE ANYTHING. That is the finding, not a preliminary.")


if __name__ == "__main__":
    main()
