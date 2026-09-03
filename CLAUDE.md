# trading-tools

A live crypto/RWA trading bot on Bitget, running against Dror's real money
(a small, real account — currently on the order of $100). This is not a
simulator with a live mode bolted on; every current strategy auto-executes
on Telegram approval. Treat code changes here with the caution that implies.

**The full project history and context lives outside every repo, in
`Documents/trading-bot/` on Dror's machine.** This file is a map for
orienting quickly inside THIS repo, not a replacement for that handoff — if
something here seems to contradict it, the handoff is more likely current.

## What this is, in one pass

1. **`notifier/`** scans the watchlist, evaluates every live strategy
   (`notifier/strategies/`), and sends a Telegram Approve/Reject alert. On
   approval, `execution/` places the order and `core/` talks to Bitget and
   persists everything to a local SQLite journal.
2. **`backtest/`** is the research side: signal generation, scoring
   (`backtest/score.py` is the ONE trade scorer — see its own docstring),
   sweeps, and experiment tracking. Nothing here executes anything.
3. **`journal/` and `weekly_review/`** turn the SQLite journal into reports —
   an on-demand stats report and a Sunday-night Telegram summary.
4. **`tools/`** are one-off-turned-reusable developer utilities: `why.py`
   (why did/didn't a strategy fire), `data.py` (what's in `data/` and what
   produced it), `reconcile.py` (real trades vs. backtest), `experiments.py`
   (has this already been tried), `verify_order_path.py` (read-only exchange
   state check).
5. **`tests/`** mirrors the source tree file-for-file — `notifier/scanner.py`
   has `tests/test_scanner.py`, etc. It is large (900+ tests) and taken
   seriously; a change without a matching test change is the exception, not
   the norm.

## Non-negotiable invariants

These come from real incidents, not preferences. Do not silently revert or
"simplify" past them.

- **Isolated margin only, never cross.** Cross backs a losing position with
  the whole account; it breaks the 2%-per-trade risk model outright.
- **Root-cause fixes, not patches.** If a bad signal comes out of a
  calculation, fix the calculation — don't filter the symptom downstream.
- **Every strategy auto-executes once whitelisted.** `AUTO_EXECUTE_TAGS` in
  `notifier/main.py` (`LIVE_TAGS | DRY_RUN_TAGS`) is the single gate between
  a strategy and real money. It's a whitelist on purpose: a new or edited
  strategy must be deliberately added, so nothing starts trading by being
  merely registered. Read the comment block above `LIVE_TAGS` before
  changing it — it's a live decision log, not boilerplate.
- **Blind fit/confirm splits for anything claiming an edge.** Select on one
  year, confirm untouched on a second, unseen year. This is what killed the
  divergence/HTF gate and the (previously shipped, since reverted) BTC
  200MA regime gate — the reverted gate's own postmortem is `git log
  --grep="BTC.*gate"` worth reading before re-proposing anything similar.
- **An edge that dies without its best 3 trades is not an edge.** Every
  expectancy number in this project is read next to its drop-top-3 figure
  (see `backtest/experiments.py`'s required arm fields, and
  `journal/stats.py`'s `by_symbol`/`by_strategy` breakdowns).
- **A capability that never fires looks exactly like a quiet market.**
  `core/ledger.py` exists because three real features (a partial take-
  profit, the weekly report, untracked-position dedupe) silently never
  worked for weeks and nothing said so. If you add something that runs
  unattended, give it a way to prove it ran.

## Working conventions

- **Docstrings carry evidence, not descriptions.** A comment on a magic
  number here usually cites the sweep, the incident, or the live trade that
  produced it — see `MAX_TOTAL_RISK_PCT` in `notifier/main.py` for the
  shape. Match this when you add a constant; "why" belongs next to the
  value, not in a commit message nobody will `git blame` their way back to.
- **Test-driven development is the norm**, with descriptive test names that
  read as sentences and docstrings that explain the failure mode being
  guarded against, not just the mechanics. `tests/test_telegram_bot.py` is
  a good reference for the house style.
- **Strategy contract** (`notifier/strategies/base.py`): every strategy
  implements `evaluate()`; may override `chart_overlay()` (what to draw on
  its Telegram alert chart — see `notifier/chart.py`) and `explain()` (why
  it did/didn't fire — see `tools/why.py`) for a richer picture than the
  default fallback gives for free.
- **Generated data is provenance-tracked, going forward.** A pickle under
  `data/` written by a generator that calls `backtest/manifest.py`'s
  `write_manifest()` carries a `.manifest.json` sidecar (git SHA, row
  counts, whatever else the generator recorded). Not every generator does
  this yet — `python -m tools.data ls` shows you which files have one and
  which don't. `signals_v2_ship.pkl` silently bundling 3 instances and
  shipping a wrong number (2026-08-21) is why this exists; don't add a new
  generator without wiring it in.
- **`data/*.pkl` and `*.hash` sidecars are gitignored** (rebuildable,
  tens of MB). `.manifest.json` sidecars are NOT — they're small and meant
  to be permanent.

## Running things

```bash
python -m pytest                    # full suite
python -m notifier.main             # the live scanner + Telegram bot
python -m journal.main              # on-demand stats report
python -m backtest.run_all          # every backtest generation/measurement step
python -m tools.why SYMBOL --strategy TAG [--at TIMESTAMP]
python -m tools.data ls
python -m tools.reconcile [--strategy TAG] [--since DATE]
python -m tools.experiments ls [--name SUBSTRING]
```

Deployment (GCP VM, systemd) is not in this repo — see the external handoff.

## Where things are still rough

- `notifier/scanner.py` is ~3,000 lines doing scanning, dispatch, order
  placement, trailing stops, partials, breakeven, and upkeep in one class.
  Heavily tested, but a natural place for a future regression to hide.
- `chart_overlay()` and `explain()` are only implemented for some
  strategies (see each method's own docstring in `base.py` for which).
  Others fall back to the generic default rather than a strategy-specific
  picture or ladder.
- The data manifest (above) is only wired into `backtest/generate_v2.py` so
  far, not the other generators under `backtest/`.
