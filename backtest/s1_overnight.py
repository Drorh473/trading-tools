"""Strategy 1 1H: can it be made positive? A blind fit-year1 / confirm-year2 run.

WHY THIS SHAPE
  Strategy 1 1H measures -0.061R (748 closed, 49.9% win, $100 -> $60.05) and
  four separate attempts to fix it have failed. The handoff's reading is that
  the deficit lives on the entry/stop side, not the exit. But nothing has ever
  separated the two halves of that number:

      fee_R = round_trip_fee_pct / stop_pct

  which the handoff measured at 0.10-0.44 R per trade depending on how wide the
  symbol's Fib stop is. Against a -0.061R expectancy, fees are plausibly LARGER
  than the whole deficit - i.e. the strategy may be positive gross and negative
  net. Every arm here therefore reports gross expectancy and fee cost as well as
  net, so the first question the morning can answer is "cost problem or signal
  problem", not "which arm won".

  Four entry-QUALITY filters have already been tested null (divergence, HTF
  agreement, confirmed rejection as a gate, the BTC 200MA regime gate). Adding a
  fifth is the lowest-prior action available, so nothing here is a new filter.
  What is swept instead is geometry (where entry and stop sit on the same
  swing), exit policy (the partial and the breakeven move), and cost (a floor on
  stop width, which is the direct fee-drag lever).

THE PROCEDURE, AND WHY IT IS NOT A FIT
  Every sweep is scored on YEAR 1 ONLY. The best arm on each axis is then
  replayed, untouched, on YEAR 2 ONLY. Selection and confirmation never see the
  same bars. This is the procedure that conclusively killed the divergence/HTF
  gate, and it is what stops the morning from picking a winner post-hoc out of a
  full-population table - which is how the ATR-buffer and RR sweeps produced
  numbers nobody could trust.

  Confirmation runs on TWO universes, because they answer different questions:
    LIVE2Y  - the 35 watchlist symbols with 2 full years. What the bot actually
              trades. Honest, but thin.
    DEEP2Y  - every symbol in the deep cache with 2 full years (~191). 5.5x the
              statistical power, but includes symbols the bot does not trade, so
              a win here still needs the LIVE2Y column to agree before shipping.
  Agreement between them is real evidence. Disagreement means the effect is
  watchlist-specific and should not be shipped on the deep number alone.

  A third universe, LIVE100 (all 100 watchlist symbols, whatever history each
  has), carries the descriptive baseline and the live-realism sweeps. It is NOT
  used for fit or confirm: 49 of those 100 symbols hold less than a year of
  bars, so a year1/year2 split across it compares two different universes - the
  same class of error as the BARS-cap bug, in a different disguise.

TRAPS THIS FILE IS DELIBERATELY BUILT AROUND
  - portfolio.generate()'s BARS cap silently ignores `hours`. Sidestepped by
    building the bars cache from data/bars_1h_deep_np.pkl and passing it as
    `cache=`.
  - The per-instance signal cache cannot tell that the BARS underneath it
    changed. This run writes its own file and never touches the shared one.
  - Windows spawn re-execs this module in every worker, so the INSTANCES
    restriction is at module level, not inside main().
  - Frames are TRIMMED to warmup+hours before generation. scan_symbol only
    scans the last `hours` bars either way, but every evaluate() call sees the
    whole frame it was handed: measured 152s -> 83s (ICX) and 106s -> 39s (LTC)
    for byte-identical signals, and 552s for BNBUSDT's untrimmed 62k bars.
  - confirmed_pivots is computed ONCE per universe and shared across arms.
  - This box's system python has a different pandas major than .venv. Run this
    with .venv/Scripts/python.exe or the pickles will not load.

Writes s1_overnight_results.txt (human) and s1_overnight_results.json
(structured, for the morning's report) into the repo root.
"""
import dataclasses
import json
import os
import pickle
import time
import traceback

import numpy as np
import pandas as pd

from backtest import engine as bt
from backtest import portfolio as pf
from backtest.score import confirmed_pivots
from notifier.strategies.ema_trend_v2 import EmaTrendV2
from notifier.strategies.indicators import atr
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.watchlist import WATCHLIST

# Module level, NOT inside main(): spawn re-execs this whole file in each
# worker, where the __main__ guard's body does not run but everything above it
# does. A restriction placed only in main() never reaches the workers.
S1, S21 = 0, 1
pf.INSTANCES = [
    (RsiFibReversal("1H"), ["1H"], 24 * 4),
    (EmaTrendV2("1H"), ["1H"], 24 * 4),
]

# S1_SMOKE=<n> caps every universe to n symbols and S1_HOURS shortens the
# window, so the exact code path that runs unattended tonight can be dry-run in
# minutes first. The previous version of this job was scheduled without ever
# being run start to finish; this is what makes that unnecessary.
SMOKE = int(os.getenv("S1_SMOKE", "0"))
HOURS_2Y = int(os.getenv("S1_HOURS", "17520"))
MAX_WAIT = 96
DEEP_BARS = os.path.join("data", "bars_1h_deep_np.pkl")
INSTANCE_CACHE = os.path.join("data", "instance_signals_overnight.pkl")
RESULTS_TXT = "s1_overnight_results.txt"
RESULTS_JSON = "s1_overnight_results.json"

# Strategy 1's shipped geometry. Entry and stop are both read off the SAME
# swing, so a signal's (entry, stop) pair determines that swing exactly - which
# is what makes these replay-time sweeps rather than regeneration-time ones.
# Verified against 12,792 real signals: worst relative round-trip error 4.1e-16.
FIB_ENTRY, FIB_STOP = 0.618, 0.786

RESULTS: list = []


def keep_awake():
    """Ask Windows not to sleep while this runs, and release the request on exit.

    This box is a laptop with Modern Standby, and the handoff records TWO
    previous overnight runs killed by the machine sleeping partway through -
    a multi-hour generation lost each time, with a half-written cache left
    behind. ES_SYSTEM_REQUIRED keeps the system up without keeping the display
    on; ES_CONTINUOUS makes it hold until reset rather than being a one-shot
    nudge. Best-effort: a failure here must never stop the measurement.
    """
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return bool(ok)
    except Exception:
        return False


def release_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

def load_deep():
    with open(DEEP_BARS, "rb") as f:
        return pickle.load(f)


def frames_for(raw, symbols, hours=HOURS_2Y, require_full=False):
    """Trimmed 1H frames, and the symbols that survived.

    require_full keeps only symbols holding the WHOLE window, so a year1/year2
    split sees the same universe in both halves instead of a year 2 populated
    by symbols that had not listed yet in year 1.
    """
    need = hours + pf.WARMUP["1H"] + 5
    cache, usable = {}, []
    for sym in symbols:
        d = raw.get(sym)
        if d is None:
            continue
        n = len(d["ts"])
        if require_full and n < need:
            continue
        if n < pf.WARMUP["1H"] + 100:
            continue
        keep = min(need, n)
        frame = pd.DataFrame(
            {k: d[k][n - keep:] for k in ("ts", "open", "high", "low", "close")}
        ).reset_index(drop=True)
        # data/bars_1h_deep_np.pkl keeps ts as the raw int64 epoch ms the
        # exchange returned, while bt.load_bars' own frames hold datetime64
        # (bars_dataframe converts on fetch). portfolio._ts_ms tolerates both,
        # but anything doing calendar arithmetic on ts does not: the previously
        # scheduled version of this job carried the same `max(ts) -
        # pd.Timedelta(...)` line and would have raised TypeError there AFTER
        # paying for the whole generation. Normalise once, here.
        if not np.issubdtype(frame["ts"].dtype, np.datetime64):
            frame["ts"] = pd.to_datetime(frame["ts"], unit="ms")
        cache[(sym, "1H")] = frame
        usable.append(sym)
    return cache, usable


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def drop_top3(rs):
    """Expectancy with the three best trades removed. None, not NaN, when there
    is nothing left to average - NaN would serialise as invalid JSON and, worse,
    silently lose every comparison it takes part in, so an arm with three trades
    would quietly rank as "not the winner" rather than "not eligible"."""
    rs = sorted(rs, reverse=True)[3:]
    return sum(rs) / len(rs) if rs else None


def score(label, acct, universe, block):
    """Every arm reports the same columns, and n/refused alongside them.

    n and declined_too_small are not decoration. The ATR-buffer sweep looked
    like an improvement mostly because a wider stop needs a smaller position,
    which pushed more setups under the $5 leg floor and halved the sample - a
    selection artifact reading as a quality gain. Any arm whose n moves a long
    way from the baseline's is suspect for that reason alone.
    """
    closed = acct.closed
    row = {"block": block, "universe": universe, "label": label,
           "n": len(closed), "declined_too_small": acct.declined_too_small,
           "declined_exposed": acct.declined_exposed,
           "declined_risk_cap": acct.declined_risk_cap}
    if closed:
        rs = [c.r for c in closed]
        tot = sum(rs)
        # r is already net of fees, so gross is r + fee/risk. risk_amount is not
        # on Closed, but pnl/r recovers it exactly (both are the same trade).
        fee_r = []
        for c in closed:
            risk = c.pnl / c.r if c.r else 0.0
            fee_r.append(c.fees / risk if risk else 0.0)
        reasons = {}
        for c in closed:
            reasons[c.reason] = reasons.get(c.reason, 0) + 1
        row.update({
            "win_pct": round(100 * sum(1 for c in closed if c.pnl > 0) / len(closed), 2),
            "tot_r": round(tot, 2),
            "exp_r": round(tot / len(closed), 4),
            "exp_r_drop_top3": (lambda v: round(v, 4) if v is not None else None)(drop_top3(rs)),
            "fee_r_median": round(float(np.median(fee_r)), 4) if fee_r else None,
            "exp_r_gross": round((tot + sum(fee_r)) / len(closed), 4),
            "equity": round(acct.equity, 2),
            "max_dd_pct": round(100 * acct.max_dd, 2),
            "exits": reasons,
        })
    RESULTS.append(row)
    return row


def line(row):
    if not row.get("n"):
        return f"  {row['label']:<30} no closed trades"
    # drop-top-3 is None for an arm with three or fewer trades - there is
    # nothing left to average. Render it as a dash rather than formatting None.
    t3 = row.get("exp_r_drop_top3")
    t3s = f"{t3:>+7.3f}" if t3 is not None else "      -"
    return (f"  {row['label']:<30} n={row['n']:<5} win={row['win_pct']:>5.1f}% "
            f"expR={row['exp_r']:>+7.3f} (gross {row['exp_r_gross']:>+7.3f}, "
            f"fee {row['fee_r_median']:>5.3f}) top3={t3s} "
            f"eq=${row['equity']:>7.2f} dd={row['max_dd_pct']:>5.1f}% "
            f"refused={row['declined_too_small']}")


# ---------------------------------------------------------------------------
# Signal mutations - all post-hoc, none needs regeneration
# ---------------------------------------------------------------------------

def swing_of(sig):
    """(low, high) of the leg this signal was measured from.

    long : entry = hi - rng*FE, stop = hi - rng*FS  ->  entry-stop = rng*(FS-FE)
    short: entry = lo + rng*FE, stop = lo + rng*FS  ->  stop-entry = rng*(FS-FE)
    """
    e, s = float(sig.entry_price), float(sig.stop_loss)
    if sig.direction == "long":
        rng = (e - s) / (FIB_STOP - FIB_ENTRY)
        hi = e + rng * FIB_ENTRY
        return hi - rng, hi
    rng = (s - e) / (FIB_STOP - FIB_ENTRY)
    lo = e - rng * FIB_ENTRY
    return lo, lo + rng


def map_s1(signals, fn):
    """Apply fn to Strategy 1's rows only; every other instance passes through.

    fn returns a replacement row, or None to drop the signal entirely.
    """
    out = {}
    for sym, rows in signals.items():
        new = []
        for row in rows:
            if row[3] != S1:
                new.append(row)
                continue
            r = fn(row)
            if r is not None:
                new.append(r)
        out[sym] = new
    return out


def refib(signals, fib_entry, fib_stop):
    """Move entry and stop to different Fib levels on the SAME swing."""
    def f(row):
        ts, i, close, pos, sig = row
        lo, hi = swing_of(sig)
        rng = hi - lo
        if sig.direction == "long":
            e, s = hi - rng * fib_entry, hi - rng * fib_stop
        else:
            e, s = lo + rng * fib_entry, lo + rng * fib_stop
        return (ts, i, close, pos,
                dataclasses.replace(sig, entry_price=e, stop_loss=s, limit_entry=e,
                                    limit_note=f"{fib_entry:.1%} Fib"))
    return map_s1(signals, f)


def min_stop_width(signals, min_pct):
    """Drop signals whose stop is too tight to survive its own fees.

    fee_R = fee_pct / stop_pct, so a 0.3%-wide stop pays ~0.4R in fees before
    the trade has done anything. This is the DIRECT lever on that, and it is the
    exact inverse of what the $5 floor selects: the floor refuses WIDE stops
    (small notional), this refuses TIGHT ones. Both counts are reported so the
    two are never confused.
    """
    def f(row):
        sig = row[4]
        e = float(sig.entry_price)
        if abs(e - float(sig.stop_loss)) / e < min_pct:
            return None
        return row
    return map_s1(signals, f)


def set_rr(signals, rr):
    return map_s1(signals, lambda r: (r[0], r[1], r[2], r[3],
                                      dataclasses.replace(r[4], reward_risk_ratio=rr)))


def rejection_entry(signals, bars_1h, max_wait=MAX_WAIT):
    """Wait for price to touch the entry zone and CLOSE back out of it, then
    take the trade at market on that close, instead of resting a limit.

    Measured +0.005R full-population last session - the only positive number
    found - but drop-top-3 flipped it negative and the year split that would
    have settled it was invalid. This is the run that settles it.
    """
    out, triggers, confirmed = {}, 0, 0
    for sym, rows in signals.items():
        frame = bars_1h[sym]
        lows, highs, closes = frame["low"].values, frame["high"].values, frame["close"].values
        ts_vals = frame["ts"].values
        new = []
        for ts, i, close, pos, sig in rows:
            if pos != S1:
                new.append((ts, i, close, pos, sig))
                continue
            triggers += 1
            entry, stop = float(sig.entry_price), float(sig.stop_loss)
            long = sig.direction == "long"
            found = None
            for j in range(i + 1, min(i + 1 + max_wait, len(frame))):
                lo, hi, c = lows[j], highs[j], closes[j]
                if long:
                    if lo <= stop:
                        break
                    if lo <= entry and c > entry:
                        found = j
                        break
                else:
                    if hi >= stop:
                        break
                    if hi >= entry and c < entry:
                        found = j
                        break
            if found is None:
                continue
            confirmed += 1
            new.append((ts_vals[found], found, float(closes[found]), pos,
                        dataclasses.replace(sig, entry_price=float(closes[found]),
                                            limit_entry=None, market_fraction=1.0)))
        out[sym] = new
    return out, triggers, confirmed


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run(bars, sigs, pivots, out, label, universe, block,
        cancel=4, start=None, end=None, partial=None, be=0.0):
    """One replay arm. partial/be temporarily override the engine's exit policy
    the same way replay() already overrides MAX_TOTAL_RISK_PCT."""
    other = {i for i in range(len(pf.INSTANCES)) if i != S1}
    old_p, old_b = bt.PARTIAL_DEFAULT, bt.BREAKEVEN_R_ON_PARTIAL
    if partial is not None:
        bt.PARTIAL_DEFAULT = partial
    bt.BREAKEVEN_R_ON_PARTIAL = be
    try:
        acct = pf.replay(bars, sigs, skip_pos=other, cancel_override=cancel,
                         max_total_risk=0.15, start_ts=start, end_ts=end,
                         pivots_cache=pivots)
    finally:
        bt.PARTIAL_DEFAULT, bt.BREAKEVEN_R_ON_PARTIAL = old_p, old_b
    row = score(label, acct, universe, block)
    print(line(row), file=out, flush=True)
    return row


def arms(bars, sigs):
    """Every arm this run scores, as (block, label, build, extra_kwargs).

    `build` is a zero-argument callable returning that arm's signal set, NOT the
    set itself: eagerly materialising all ~35 would hold tens of thousands of
    freshly-constructed Signal objects per arm alive at once, and this has to
    survive a night unattended. Building lazily keeps one mutated set live at a
    time, and a picked arm is rebuilt for its year-2 confirmation from the SAME
    definition - so an arm cannot silently differ between the two halves.
    """
    a = [("baseline", "baseline (as shipped)", lambda: sigs, {})]

    for fe in (0.500, 0.618, 0.705, 0.786):
        for fs in (0.786, 0.886, 1.000, 1.130):
            if fs <= fe:
                continue
            a.append(("fib_grid", f"entry {fe:.3f} / stop {fs:.3f}",
                      lambda fe=fe, fs=fs: refib(sigs, fe, fs), {}))

    for part in (0.0, 0.25, 0.5, 0.75, 1.0):
        for be in (0.0, None, 0.5):
            tag = {0.0: "BE", None: "no-BE", 0.5: "BE+0.5R"}[be]
            a.append(("exit_policy", f"partial {part:.2f} / {tag}", lambda: sigs,
                      {"partial": part, "be": be}))

    for m in (0.005, 0.0075, 0.010, 0.015, 0.020, 0.030):
        a.append(("stop_width", f"min stop {m:.2%}",
                  lambda m=m: min_stop_width(sigs, m), {}))

    for rr in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        a.append(("reward_risk", f"RR {rr:.2f}", lambda rr=rr: set_rr(sigs, rr), {}))

    for c in (4, 8, 12, 24, 48, 72, 96):
        a.append(("cancel_window", f"cancel {c}h", lambda: sigs, {"cancel": c}))

    # Scanned once here to label the arm with its confirmation rate; the result
    # is reused rather than rescanned when the arm actually runs.
    rej, trig, conf = rejection_entry(sigs, bars)
    a.append(("entry_mechanism",
              f"rejection-confirmed ({conf}/{trig} = "
              f"{100*conf/trig if trig else 0:.0f}%)", lambda: rej, {}))
    return a


def fit_and_confirm(bars, sigs, pivots, out, universe, mid, end):
    """Score every arm on year 1, pick each block's winner, confirm on year 2."""
    print(f"\n{'='*78}\n=== {universe}: YEAR 1 (fit) ===", file=out, flush=True)
    defs = arms(bars, sigs)
    blocks: dict = {}
    for block, label, build, kw in defs:
        blocks.setdefault(block, [])
        row = run(bars, build(), pivots, out, label, universe, f"{block}/year1",
                  end=mid, **kw)
        blocks[block].append((row, label, build, kw))

    print(f"\n=== {universe}: YEAR 2 (blind confirmation of year 1's picks) ===",
          file=out, flush=True)
    picks = []
    base_n = next((r[0]["n"] for r in blocks.get("baseline", []) if r[0].get("n")), 0)
    # An arm only competes to represent its block if it still has a real sample.
    # Without this the winner is routinely an arm that kept 2 trades out of 748
    # and posted a huge expectancy on them - the same sample-shrinkage artifact
    # that made the ATR-buffer sweep look like an improvement. A block whose
    # every arm is too thin has no pick, and says so, rather than promoting the
    # thinnest one.
    floor = max(20, int(0.25 * base_n))
    for block, rows in blocks.items():
        scored = [r for r in rows
                  if r[0].get("n", 0) >= floor and r[0].get("exp_r_drop_top3") is not None]
        if not scored:
            print(f"  [{block}] no arm kept >= {floor} trades - no pick to confirm",
                  file=out, flush=True)
            continue
        # Rank on drop-top-3, not raw expectancy: a winner carried by three
        # outliers is exactly what this procedure exists to refuse.
        best = max(scored, key=lambda r: (r[0]["exp_r_drop_top3"], r[0]["exp_r"]))
        picks.append((block, best))
    for block, (row, label, build, kw) in picks:
        print(f"  [{block}] year-1 pick: {label}", file=out, flush=True)
        run(bars, build(), pivots, out, label, universe, f"{block}/year2",
            start=mid, end=end, **kw)
    return picks


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    awake = keep_awake()
    out = open(RESULTS_TXT, "w", encoding="utf-8")
    try:
        print(f"sleep suppression: {'on' if awake else 'UNAVAILABLE'}", file=out, flush=True)
        raw = load_deep()
        live100 = list(WATCHLIST)
        deep2y = sorted(s for s, v in raw.items() if len(v["ts"]) >= HOURS_2Y + pf.WARMUP["1H"] + 5)
        live2y = [s for s in live100 if s in set(deep2y)]
        if SMOKE:
            live100, deep2y, live2y = live100[:SMOKE], deep2y[:SMOKE], live2y[:SMOKE]
            print(f"SMOKE RUN: universes capped to {SMOKE}, hours={HOURS_2Y}. "
                  f"Numbers here are NOT results.", file=out, flush=True)
        every = sorted(set(live100) | set(deep2y))
        print(f"universes: LIVE100={len(live100)}  LIVE2Y={len(live2y)}  "
              f"DEEP2Y={len(deep2y)}  generate={len(every)}", file=out, flush=True)

        cache, usable = frames_for(raw, every)
        print(f"{len(usable)}/{len(every)} symbols usable", file=out, flush=True)

        t = time.time()
        # only_pos=[S1]: Strategy 2.1's baseline was a nice-to-have on the
        # earlier version of this job, but generation is the one phase that
        # costs hours and it scales with the number of instances. Tonight's
        # question is Strategy 1, and halving the generation is what buys the
        # margin for ~35 sweep arms across three universes. S21 stays in
        # INSTANCES so position indices - and therefore skip_pos - are unchanged.
        bars, sigs = pf.generate(usable, HOURS_2Y, workers=10, cache=cache,
                                 instance_cache_path=INSTANCE_CACHE, only_pos=[S1])
        print(f"generation: {time.time()-t:.0f}s", file=out, flush=True)

        t = time.time()
        pivots = {s: confirmed_pivots(f, atr(f)) for s, f in bars.items()}
        print(f"pivots: {time.time()-t:.0f}s (shared across every arm)", file=out, flush=True)

        n_s1 = sum(1 for rows in sigs.values() for r in rows if r[3] == S1)
        print(f"Strategy 1 signals: {n_s1}", file=out, flush=True)

        def subset(symbols):
            keep = set(symbols) & set(bars)
            return ({s: bars[s] for s in keep}, {s: sigs.get(s, []) for s in keep},
                    {s: pivots[s] for s in keep})

        # --- the two 2-year universes: blind fit / confirm --------------------
        for name, syms in (("LIVE2Y", live2y), ("DEEP2Y", deep2y)):
            b, g, p = subset(syms)
            if not b:
                continue
            top = max(f["ts"].max() for f in b.values())
            mid = top - pd.Timedelta(hours=HOURS_2Y // 2)
            end = top + pd.Timedelta(seconds=1)
            print(f"\n{name} window: ..{mid} | {mid}..{top}", file=out, flush=True)
            fit_and_confirm(b, g, p, out, name, mid, end)

        # --- LIVE100: descriptive only, never fit or confirmed ---------------
        b, g, p = subset(live100)
        print(f"\n{'='*78}\n=== LIVE100 (all {len(b)} watchlist symbols, whatever "
              f"history each has) ===", file=out, flush=True)
        print("Descriptive only - 49 of these hold under a year of bars, so a "
              "year split here\ncompares two different universes. Do not fit on "
              "this table.", file=out, flush=True)
        for block, label, build, kw in arms(b, g):
            run(b, build(), p, out, label, "LIVE100", f"{block}/full", **kw)

        print(f"\nTOTAL WALL TIME {time.time()-t0:.0f}s", file=out, flush=True)
    except Exception:
        print("CRASHED:", file=out, flush=True)
        traceback.print_exc(file=out)
        raise
    finally:
        release_awake()
        out.close()
        with open(RESULTS_JSON, "w", encoding="utf-8") as jf:
            json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "rows": RESULTS}, jf, indent=1)
        # Render the human report here rather than leaving it to a morning
        # session: the run finishes unattended, and results that only exist
        # inside a chat transcript are lost if that session is gone. Guarded so
        # a formatting bug can never destroy the measurement it is formatting -
        # by this point both result files are already safely on disk.
        try:
            from backtest import s1_report
            s1_report.main()
        except Exception:
            print("report generation failed (results files are still intact):")
            traceback.print_exc()


if __name__ == "__main__":
    main()
