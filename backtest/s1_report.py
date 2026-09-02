"""Turn s1_overnight_results.json into a report a human reads in the morning.

WHY A SCRIPT AND NOT A CHAT MESSAGE
  The run finishes in the middle of the night with nobody attached. If the only
  rendering of the result lives in an assistant session, a crashed or closed
  session loses the whole night. This writes a standalone HTML file to disk as
  the last step of the run itself, so the results exist regardless.

WHAT IT IS OPINIONATED ABOUT
  It does not just tabulate. It applies the reading rules that the last three
  sessions had to learn the hard way, and says them out loud next to the number:

  - A year-1 pick that does not survive year 2 is REFUTED, not "mixed". The
    whole point of splitting was to make that call in advance.
  - An arm whose closed-trade count moved a long way from the baseline's is
    flagged, because the ATR-buffer sweep's apparent win was mostly the $5 leg
    floor shrinking the sample, not a better trade.
  - Net expectancy is shown against GROSS and the median fee in R, so "the
    signal is bad" and "the trade cannot carry its own costs" stay separable.
  - drop-top-3 is shown beside every expectancy. An edge that dies without its
    three best trades is not an edge.
"""
import html
import json
import os
import sys

RESULTS_JSON = "s1_overnight_results.json"
REPORT_HTML = "s1_overnight_report.html"

CSS = """
:root{--bg:#fbfaf9;--fg:#1a1a18;--muted:#6b6a66;--line:#e2e0dc;--card:#fff;
--good:#1a7f4b;--bad:#b3261e;--warn:#9a6a00;--warnbg:#fdf6e3;--accent:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#16161a;--fg:#ecebe8;--muted:#9a988f;--line:#2e2e34;--card:#1e1e24;
--good:#4ade80;--bad:#f87171;--warn:#fbbf24;--warnbg:#2a2313;--accent:#8da2fb}}
:root[data-theme="dark"]{--bg:#16161a;--fg:#ecebe8;--muted:#9a988f;--line:#2e2e34;
--card:#1e1e24;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;--warnbg:#2a2313;--accent:#8da2fb}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem 5rem;
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
h2{font-size:1.15rem;margin:2.5rem 0 .5rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:.95rem;margin:1.5rem 0 .4rem;color:var(--muted);font-weight:600;
letter-spacing:.04em;text-transform:uppercase}
.sub{color:var(--muted);margin:0 0 2rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem 1.15rem;margin:1rem 0}
.note{background:var(--warnbg);border-left:3px solid var(--warn);border-radius:0 6px 6px 0;
padding:.7rem .9rem;margin:.8rem 0;font-size:.9rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.86rem;
font-variant-numeric:tabular-nums;min-width:760px}
th,td{padding:.4rem .55rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
tr.base td{background:color-mix(in srgb,var(--accent) 8%,transparent);font-weight:600}
.pos{color:var(--good)}.neg{color:var(--bad)}
.tag{display:inline-block;padding:.1rem .45rem;border-radius:99px;font-size:.72rem;
font-weight:600;letter-spacing:.02em}
.t-ok{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.t-no{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.t-w{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.hl{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0}
.hl>div{flex:1 1 150px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:.8rem .9rem}
.hl .k{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.04em}
.hl .v{font-size:1.35rem;font-weight:650;font-variant-numeric:tabular-nums}
code{background:color-mix(in srgb,var(--fg) 8%,transparent);padding:.1rem .3rem;border-radius:4px;font-size:.85em}
"""

BLOCK_TITLES = {
    "baseline": "Baseline",
    "fib_grid": "Fib geometry - where entry and stop sit on the same swing",
    "exit_policy": "Exit policy - the partial size and the breakeven move",
    "stop_width": "Minimum stop width - the direct fee-drag lever",
    "reward_risk": "Reward:risk ratio",
    "cancel_window": "Entry-cancel window",
    "entry_mechanism": "Entry mechanism",
}


def esc(x):
    return html.escape(str(x))


def num(v, digits=3, sign=True, cls=True):
    if v is None:
        return "<td class='muted'>-</td>"
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    k = "" if not cls else (" class='pos'" if v > 0 else (" class='neg'" if v < 0 else ""))
    return f"<td{k}>{s}</td>"


def table(rows, baseline=None):
    """One block's arms. baseline, when given, is the row every other is read against."""
    head = ("<div class='scroll'><table><thead><tr><th>arm</th><th>n</th><th>win%</th>"
            "<th>exp R</th><th>drop-top-3</th><th>gross R</th><th>fee R</th>"
            "<th>equity</th><th>max DD</th><th>refused</th><th></th></tr></thead><tbody>")
    body = []
    bn = baseline.get("n") if baseline else None
    for r in rows:
        if not r.get("n"):
            body.append(f"<tr><td>{esc(r['label'])}</td><td colspan='10' class='muted'>"
                        "no closed trades</td></tr>")
            continue
        flag = ""
        if bn and abs(r["n"] - bn) / bn > 0.30:
            d = "shrank" if r["n"] < bn else "grew"
            flag = (f"<span class='tag t-w' title='Sample {d} {abs(r['n']-bn)/bn:.0%} vs "
                    f"baseline - check this is not the $5 leg floor selecting on stop "
                    f"width rather than a real effect'>sample {d} {abs(r['n']-bn)/bn:.0%}</span>")
        cls = " class='base'" if baseline is r else ""
        body.append(
            f"<tr{cls}><td>{esc(r['label'])}</td><td>{r['n']}</td>"
            f"<td>{r['win_pct']:.1f}</td>"
            + num(r["exp_r"]) + num(r["exp_r_drop_top3"]) + num(r["exp_r_gross"])
            + num(r.get("fee_r_median"), 3, sign=False, cls=False)
            + f"<td>${r['equity']:.2f}</td><td>{r['max_dd_pct']:.1f}%</td>"
            f"<td>{r['declined_too_small']}</td><td>{flag}</td></tr>")
    return head + "".join(body) + "</tbody></table></div>"


def verdict(y1, y2):
    """The whole point of the split: year 2 decides, and it decides in advance."""
    if not y2 or not y2.get("n"):
        return ("<span class='tag t-w'>no year-2 trades</span>",
                "Year 2 produced nothing to score, so this arm is untested, not confirmed.")
    a, b = y1.get("exp_r_drop_top3"), y2.get("exp_r_drop_top3")
    if a is None or b is None:
        return ("<span class='tag t-w'>too few trades</span>",
                "One half has three or fewer closed trades, so drop-top-3 has nothing "
                "left to average. Untested, not confirmed.")
    if b > 0 and a > 0:
        return ("<span class='tag t-ok'>CONFIRMED</span>",
                f"Positive on both halves after dropping its three best trades "
                f"({a:+.3f} then {b:+.3f}). This is the only shape that justifies shipping.")
    if b > 0 >= a:
        return ("<span class='tag t-w'>year 2 only</span>",
                f"Year 2 is positive ({b:+.3f}) but year 1 was not ({a:+.3f}). One good "
                f"half is what a coin does; do not ship on this.")
    return ("<span class='tag t-no'>REFUTED</span>",
            f"Best on year 1 ({a:+.3f}) and does not hold on year 2 ({b:+.3f}). "
            f"The year-1 ranking was fitting noise. This is the answer the split exists to give.")


def build(data):
    rows = data["rows"]
    universes, blocks = [], []
    for r in rows:
        if r["universe"] not in universes:
            universes.append(r["universe"])
    out = [f"<title>Strategy 1 Overnight</title><style>{CSS}</style><div class='wrap'>",
           "<h1>Strategy 1 1H - overnight sweep</h1>",
           f"<p class='sub'>Generated {esc(data.get('generated_at',''))}. "
           "Every sweep was scored on year 1 alone; each block's winner was then "
           "replayed untouched on year 2. Selection and confirmation never saw the "
           "same bars.</p>"]

    # --- headline: is this a cost problem or a signal problem? ---------------
    base = next((r for r in rows if r["block"].startswith("baseline")
                 and r["universe"] == "LIVE100" and r.get("n")), None)
    if base:
        out.append("<h2>The first question: cost or signal?</h2>")
        out.append("<div class='hl'>"
                   f"<div><div class='k'>net expectancy</div><div class='v "
                   f"{'pos' if base['exp_r']>0 else 'neg'}'>{base['exp_r']:+.3f}R</div></div>"
                   f"<div><div class='k'>gross of fees</div><div class='v "
                   f"{'pos' if base['exp_r_gross']>0 else 'neg'}'>{base['exp_r_gross']:+.3f}R</div></div>"
                   f"<div><div class='k'>median fee</div><div class='v'>"
                   f"{base['fee_r_median']:.3f}R</div></div>"
                   f"<div><div class='k'>closed trades</div><div class='v'>{base['n']}</div></div>"
                   "</div>")
        if base["exp_r_gross"] > 0 >= base["exp_r"]:
            out.append("<div class='note'><strong>Positive gross, negative net.</strong> "
                       "The signal makes money and the execution gives it back. That points "
                       "at stop width, fee structure and the partial - not at another entry "
                       "filter. Four entry filters have already tested null.</div>")
        elif base["exp_r_gross"] <= 0:
            out.append("<div class='note'><strong>Negative even before fees.</strong> "
                       "Cost reduction cannot rescue this on its own - the entry/stop "
                       "geometry has to change, or the strategy does not have an edge on "
                       "this population.</div>")
        if base.get("exits"):
            tot = sum(base["exits"].values())
            bits = ", ".join(f"{k} {v} ({v/tot:.0%})" for k, v in sorted(base["exits"].items()))
            out.append(f"<div class='card'><h3>How trades actually end</h3><p>{esc(bits)}</p>"
                       "<p class='sub' style='margin:0'>A small <code>target</code> share means "
                       "most trades never reach the first partial at all, which caps how much "
                       "any exit-policy change can be worth.</p></div>")

    # --- per universe --------------------------------------------------------
    for uni in universes:
        urows = [r for r in rows if r["universe"] == uni]
        out.append(f"<h2>{esc(uni)}</h2>")
        if uni == "LIVE100":
            out.append("<div class='note'>Descriptive only. 49 of these 100 symbols hold "
                       "under a year of bars, so a year split here would compare two "
                       "different universes. Nothing was fitted or confirmed on this table."
                       "</div>")
        # confirmation summary first - it is the answer
        y1s = {r["block"].split("/")[0]: r for r in urows if r["block"].endswith("/year1")}
        y2s = {r["block"].split("/")[0]: r for r in urows if r["block"].endswith("/year2")}
        if y2s:
            out.append("<h3>Year-1 picks, judged on year 2</h3><div class='card'>")
            for blk, y2 in y2s.items():
                y1 = next((r for r in urows if r["block"] == f"{blk}/year1"
                           and r["label"] == y2["label"]), y1s.get(blk))
                tag, why = verdict(y1, y2)
                out.append(f"<p><strong>{esc(BLOCK_TITLES.get(blk, blk))}</strong> {tag}<br>"
                           f"<span class='sub'>{esc(y2['label'])} - {esc(why)}</span></p>")
            out.append("</div>")
        for blk in BLOCK_TITLES:
            brows = [r for r in urows if r["block"].split("/")[0] == blk]
            if not brows:
                continue
            out.append(f"<h3>{esc(BLOCK_TITLES[blk])}</h3>")
            for phase in ("year1", "year2", "full"):
                sel = [r for r in brows if r["block"].endswith("/" + phase)]
                if not sel:
                    continue
                bl = next((r for r in urows if r["block"] == f"baseline/{phase}"), None)
                if phase != "full":
                    out.append(f"<p class='sub' style='margin:.3rem 0'>{phase}</p>")
                out.append(table(sel, bl))
    out.append("</div>")
    return "\n".join(out)


def main(path=RESULTS_JSON, dest=REPORT_HTML):
    if not os.path.exists(path):
        print(f"no {path} - nothing to report", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(build(data))
    print(f"wrote {dest} ({len(data['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
