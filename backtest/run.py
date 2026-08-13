"""Generate signals once, then replay the portfolio under each entry rule.

  python -m backtest.run [n_symbols] [hours] [workers]

Signals are cached to signals.pkl, so re-running to try another rule variant
costs seconds rather than another generation pass.

The three arms differ ONLY in what happens when a split entry's per-leg
notional falls under Bitget's $5 floor:

  baseline  the whole trade is refused          (today's live behaviour)
  limit     collapse to one leg at the limit    (better price, may never fill)
  market    collapse to one leg at market       (always fills, worse price)

Risk is identical in all three - the same 1% of equity, the same stop. Only
the number of legs changes, and with it whether the floor is cleared.
"""
import os
import pickle
import sys
import time

from backtest import engine as bt
from backtest import portfolio as pf

SIG_CACHE = pf.SIGNALS


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] * 100 if xs else float("nan")


def report(acct, label):
    print(f"\n================ {label} ================")
    print(f"end equity       ${acct.equity:.2f}   ({(acct.equity/bt.START_EQUITY-1)*100:+.1f}%)")
    print(f"max drawdown     {acct.max_dd*100:.1f}%")
    print(f"trades taken     {acct.taken}   closed {len(acct.closed)}   rescued {acct.rescued}")
    print(f"declined: too_small {acct.declined_too_small}  risk_cap {acct.declined_risk_cap}  "
          f"already_in_symbol {acct.declined_exposed}  swing_slots {acct.declined_swing}")
    if not acct.closed:
        return
    wins = [c for c in acct.closed if c.pnl > 0]
    tot_r = sum(c.r for c in acct.closed)
    print(f"win rate         {len(wins)/len(acct.closed)*100:.0f}%")
    print(f"total R          {tot_r:+.1f}   expectancy {tot_r/len(acct.closed):+.2f}R")
    print(f"median stop      taken {med(acct.taken_stop_pct):.2f}%  "
          f"rescued {med(acct.rescued_stop_pct):.2f}%  refused {med(acct.refused_stop_pct):.2f}%")

    by = {}
    for c in acct.closed:
        d = by.setdefault(c.tag, [0, 0, 0.0, 0.0])
        d[0] += 1
        d[1] += 1 if c.pnl > 0 else 0
        d[2] += c.r
        d[3] += c.pnl
    print(f"\n{'strategy':<24} {'n':>4} {'win':>5} {'totR':>8} {'expR':>7} {'$':>9}")
    for tag, (n, w, r, pnl) in sorted(by.items(), key=lambda kv: -kv[1][3]):
        print(f"{tag:<24} {n:>4} {w/n*100:>4.0f}% {r:>+8.1f} {r/n:>+7.2f} {pnl:>+9.2f}")
    print("exit reasons:", {r: sum(1 for c in acct.closed if c.reason == r)
                            for r in ("stop", "target", "runner")})

    scored = getattr(acct, "scored_too_small", [])
    if scored:
        tot = sum(r for _, r in scored)
        wins = sum(1 for _, r in scored if r > 0)
        print(f"\nstill refused by the $5 floor: {acct.declined_too_small}, "
              f"{len(scored)} resolved · win {wins/len(scored)*100:.0f}% · "
              f"{tot/len(scored):+.2f}R each")
        print("  (scored in isolation - guaranteed full fill at the blended price,")
        print("   no margin or slot competition, so this OVERSTATES them)")


def main():
    n_symbols = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 8760
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    key = (n_symbols, hours)
    if os.path.exists(SIG_CACHE):
        cached_key, bars_1h, signals = pickle.load(open(SIG_CACHE, "rb"))
        if cached_key != key:
            bars_1h = None
    else:
        cached_key, bars_1h, signals = None, None, None

    if bars_1h is None:
        t0 = time.time()
        bars_1h, signals = pf.generate(pf.WATCHLIST[:n_symbols], hours, workers)
        pickle.dump((key, bars_1h, signals), open(SIG_CACHE, "wb"))
        print(f"generation done in {time.time()-t0:.0f}s", flush=True)
    else:
        print("reusing cached signals", flush=True)

    total = sum(len(v) for v in signals.values())
    print(f"\n{total} signals across {len(bars_1h)} symbols")
    per_tag = {}
    for found in signals.values():
        for _ts, _i, _c, pos_i, _sig in found:
            tag = pf.INSTANCES[pos_i][0].tag
            per_tag[tag] = per_tag.get(tag, 0) + 1
    for tag, n in sorted(per_tag.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:<24} {n:>6}")

    # Strategy 4 is dry-run live, so the arms that exclude it are the ones that
    # describe the account as it actually trades today.
    s4 = {i for i, (s, _n, _c) in enumerate(pf.INSTANCES)
          if s.tag.startswith("Strategy 4")}

    # cancel=None uses the per-instance windows (up to 96 bars); cancel=4 is
    # the flat 4-hour timeout live actually applies to every unfilled entry.
    # Both are run because the gap between them prices what wiring
    # UNFILLED_CANDLES into execution would actually be worth.
    arms = [
        ("", (), None, "BASELINE - refuse the trade (live behaviour), with S4"),
        ("", s4, None, "BASELINE - refuse the trade, S4 EXCLUDED (today's live set)"),
        ("", s4, 4, "BASELINE, S4 excluded, LIVE 4h cancel"),
        ("limit", s4, None, "FALLBACK single leg at the LIMIT, S4 excluded, per-instance cancel"),
        ("limit", s4, 4, "FALLBACK single leg at the LIMIT, S4 excluded, LIVE 4h cancel"),
        ("market", s4, None, "FALLBACK single leg at MARKET, S4 excluded"),
        ("market", s4, 4, "FALLBACK single leg at MARKET, S4 excluded, LIVE 4h cancel"),
        ("limit", (), None, "FALLBACK single leg at the LIMIT, with S4"),
    ]
    for mode, skip, cancel, label in arms:
        bt.SPLIT_FALLBACK = mode
        t0 = time.time()
        acct = pf.replay(bars_1h, signals, skip_pos=skip, cancel_override=cancel)
        report(acct, label)
        print(f"[replay {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
