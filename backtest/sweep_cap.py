"""Sweep the aggregate risk cap across the cached one-year signal set.

Deliberately runs PAST the range anyone is considering, in both directions:
an interior optimum that turns out to be the edge of the swept range is not an
optimum, it is a boundary, and the two look identical if you only sweep where
you already expect the answer to be.

Also reports what the result looks like WITHOUT its three best trades. A
near-zero-edge system's equity curve is dominated by a handful of outcomes,
and a setting that only wins because of three trades has not been shown to be
better than one that does not.
"""
import os
import pickle
import sys

sys.path.insert(0, r"C:\Users\dror\study\projects\trading-tools")

from backtest import engine as bt
from backtest import portfolio as pf

SIG = os.environ["BACKTEST_SIGNALS"]

key, bars_1h, signals = pickle.load(open(SIG, "rb"))
s4 = {i for i, (s, _n, _c) in enumerate(pf.INSTANCES) if s.tag.startswith("Strategy 4")}
print(f"{sum(len(v) for v in signals.values())} signals, {len(bars_1h)} symbols\n")

print(f"{'cap':>6} {'end $':>9} {'maxDD':>7} {'taken':>6} {'declRisk':>9} "
      f"{'expR':>7} {'end $ less top-3':>17}")
for cap in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
    bt.SPLIT_FALLBACK = ""
    acct = pf.replay(bars_1h, signals, skip_pos=s4, cancel_override=None, max_total_risk=cap)
    closed = acct.closed
    exp_r = sum(c.r for c in closed) / len(closed) if closed else float("nan")
    # Strip the three biggest dollar winners and re-add their P&L out of the
    # final equity. Crude - it ignores the compounding they carried - but it
    # answers "does this setting survive losing its luckiest trades".
    top3 = sorted(closed, key=lambda c: -c.pnl)[:3]
    less_top3 = acct.equity - sum(c.pnl for c in top3)
    print(f"{cap:>5.0%} {acct.equity:>9.2f} {acct.max_dd*100:>6.1f}% {acct.taken:>6} "
          f"{acct.declined_risk_cap:>9} {exp_r:>+7.2f} {less_top3:>17.2f}")
