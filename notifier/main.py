"""Entrypoint: runs the scanner loop plus the Telegram Approve/Reject bot side
by side in one process.

Scans are aligned to candle closes, driven by each strategy's declared
timeframe (both current strategies are 1H — Strategy 1 explicitly requires
1h+, and Strategy 2 is running its longer-timeframe variant). Equity is read
live from Bitget on every scan rather than hardcoded, so position sizing tracks
the real account balance.
"""

import asyncio
import logging
import signal

from telegram.ext import CommandHandler

from config import settings
from core import ledger
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import NotifierBot
from execution.executor import DryRunExecutor, LiveExecutor, RoutingExecutor
from execution.manual_entry import make_add_conversation
from execution.tracker import resume_open_trades
from notifier.scanner import Scanner
from notifier.strategies.base import signal_from_json
from notifier.strategies.ema_trend_v2 import EmaTrendV2, INSTANCES as V21_INSTANCES
from notifier.strategies.order_block import OrderBlockStrategy
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.strategies.volume_run import (
    DAY_PARAMS,
    DAY_PARTIAL_FRACTION,
    STOP_AT_RECENT_LOW,
    VolumeRun,
)
from notifier.watchlist import WATCHLIST

logger = logging.getLogger(__name__)

RISK_PCT = 0.01  # 1-2% per trade, hard-capped at 2% in risk_sizing.plan_position
# Aggregate ceiling across all open trades. 6% -> 15% -> 10% on 2026-08-14,
# Dror's call each time; 10% is his middle ground after the first backtest that
# could price the question at all. Per-trade risk is unchanged and still hard
# capped at 2% in risk_sizing.plan_position, so what this governs is how many
# trades may run AT ONCE - a drawdown decision, not a per-trade one.
#
# Swept over the same 7,130 signals, only this constant moving - and swept past
# the optimum in both directions, because an optimum at the edge of the swept
# range is a boundary wearing a disguise:
#
#     cap    end $    maxDD   taken    expR   end $ less its top 3 trades
#      2%    97.91     9.2%     132   +0.03    85.52
#      4%   143.81    12.5%     266   +0.17   125.06
#      6%   136.41    17.3%     441   +0.11   117.95
#      8%   116.03    23.0%     627   +0.06    97.60
#     10%   136.41    25.7%     779   +0.08   108.74   <- here
#     12%    69.70    47.8%    1006   +0.00    51.09
#     15%    51.80    59.4%     952   -0.03    34.61
#     20%    46.85    64.8%     834   -0.05    30.95
#     30%    45.41    65.0%     823   -0.06    29.80
#
# The mechanism is visible rather than inferred. At 6% the cap refused 4,488
# signals; at 15% it refused 163, and those extra trades are collectively
# negative - Strategy 1 1H alone goes from 368 trades at 57% win and +0.12R to
# 796 at 52% and -0.02R. Same strategy, same year, diluted by what the cap had
# been keeping out. Fifteen concurrent crypto longs is one correlated bet.
#
# READ THE COLUMNS SEPARATELY. Return is noisy - 8% lands below 10%, which is
# path luck and not a trend. Max drawdown is monotone across all nine points
# with no exceptions, and expectancy nearly so. Within 4-10% the return
# differences are noise and the drawdown differences are not.
#
# 10% IS DROR'S SETTLED CHOICE, made with the table above in front of him. It is
# not an oversight and it is not waiting to be revisited - do not re-propose a
# lower cap. The numbers are kept here because they are evidence worth having,
# not because the decision is open.
#
# The caveat that matters: expectancy at 6% is +0.11R over 424 closed trades,
# standard error about 0.06, so t is roughly 1.7 - NOT significant. The honest
# reading is that a lower cap loses less, not that this system has proven edge.
# When expectancy is near zero, added variance is pure cost, which is why this
# constant moves the result so much more than it looks like it should.
MAX_TOTAL_RISK_PCT = 0.10
# Leverage is solved per trade to fit the margin left after other open trades;
# this only caps how high it may go.
MAX_LEVERAGE = 20.0
# Only these place orders on their own. A whitelist rather than a flag: a new
# strategy has to be named here before it can spend money, so registering one
# can never start it executing by accident. That property still holds even
# though every CURRENT strategy is listed - the next one added will not be.
#
# EVERY strategy now places real orders when Dror approves the alert. He asked
# for this directly - "first make all the strategys work when i approve them" -
# after Strategy 1 had run live for several days.
#
# Two things were fixed first rather than graduating them as they stood:
#
#   Strategy 2's 1H/15m stops were fee-dominated. Its stop sits at EMA20 and
#   its entry at EMA9, and on a 15m base those are barely apart - median stop
#   0.145% of price against a 0.12% round trip, so fees cost 0.83R at the
#   median and 1.71R on the tightest live signals. It also sized positions at
#   10-14x account equity. MAX_FEE_FRACTION_OF_RISK now declines those; the
#   instance emits ~1 signal per 2 days across the watchlist instead of 14,
#   and the worst survivor costs 0.18R.
#
#   Strategy 2 had no break-of-structure gate at all, so it shorted MUUUSDT
#   into rising highs AND rising lows. 36% of its raw signals were
#   counter-trend.
#
# STRATEGY 3 MOVED BACK TO DRY RUN on 2026-08-23, the day its ENTIRE detection
# algorithm was rebuilt. It had sat in LIVE_TAGS for weeks producing zero
# signals in the whole signals table - not a slow week, nothing at all - which
# is why it was safe to leave there: an algorithm that never fires cannot lose
# money. That is no longer true. The rebuild replaces the impulse-candle
# detector with a horizontal-level one built from five rules given directly
# and checked chart by chart against two real reference setups (BTCUSDT
# 2024-02, ALCHUSDT 2025-03) and five rejected ones - but that is SHAPE
# validation, not P&L validation. Nobody has measured whether the setups it
# now finds make money; the rebuild fixes what the strategy IS, not whether
# it WORKS. The session gate that bounded the old, silent instance's risk was
# never the point - the point is that this is the first time either instance
# has had a real chance to fire at all, and it should do that on paper first.
#
# Both instances stay wired into evaluate() and will alert on Telegram
# exactly as before; only the auto-execute permission moved. Promote back to
# LIVE_TAGS once real signals have accumulated and been reviewed - not on a
# calendar, on evidence.
#
# The day instance is "Strategy 3 1D/5m", not "1H/5m": its consolidation moved
# back onto daily bars to match the cheatsheet. The tag is built from the
# timeframe pair, so it changed with the instance - and a tag missing from
# this set silently loses auto-execution rather than failing loudly, which is
# why test_main_wiring asserts every registered tag is routed.
#
# STRATEGY 2.1 GOES LIVE WITH ITS MEASUREMENTS AGAINST IT, and that is Dror's
# call made with the table in front of him. Every honest entry construction
# measured negative over 2025-07 to 2026-08 on 98 symbols:
#
#     rejection + next candle open   -0.020R   <- what ships
#     pre-placed, no rejection       -0.029R
#     band touched                   -0.078R
#     rejection + retest at EMA9     -0.086R
#     rejection + band edge retest   -0.106R
#
# What DID get established is faithfulness: against setups Dror marked on blind
# charts - no signals drawn, nothing showing what the code thought - it finds
# 18 of 20, including 4 of 4 on symbols it had never seen, three of those on the
# exact candle. Earlier versions found 1 in 7. So a negative number is now a
# statement about the strategy on this data rather than about a broken
# implementation, which it was not before.
#
# Nothing here executes on its own: LIVE_TAGS decides whether pressing Approve
# places the order or leaves it to be done by hand. The gate is Dror.
#
# Strategy 2.1's BTC gate (daily_regime_read, see regime.py) was measured
# and then dropped 2026-08-29: agreeing/disagreeing reads both stay near
# the same ~99%+ drawdown wipeout on both the 1H and 15m instances -
# smaller losses in R terms, but no real capital protection - while
# Strategy 1's own levels-based gate DID show a real improvement (see
# build_strategies() below). Every V21 instance, including "1H", is
# therefore back to the plain, generically-derived tag.
V21_TAGS = {f"Strategy 2.1 {base}" for base, _ref in V21_INSTANCES}
# The "1H"/"4H" entries are gated (see build_strategies() below) - their
# real tags are "Strategy 1 1H +BTCUSDT(levels)" / "Strategy 1 4H +BTCUSDT
# (levels)", not the plain strings.
LIVE_TAGS = {
    "Strategy 1 1H +BTCUSDT(levels)", "Strategy 1 4H +BTCUSDT(levels)", "Strategy 1 1D",
} | V21_TAGS
# Strategy 4 ships here, NOT live, and should stay here for a while.
#
# One round of Dror's chart review has happened and produced five corrections
# (gap direction, gap floor 0.25 -> 1.0 ATR, expansion trimming, an OB 1.0
# displacement floor, and dropping the opposing-candle rule). It cost the
# signal rate heavily - the last full scan found 2 setups across 100 symbols
# over 40 candles - so the rate itself is now an open question.
#
# Still unmeasured: the expansion steepness floor, the stop's ATR buffer and
# the tolerated counter-candle retrace are all numbers with no evidence behind
# them, and the strategy has no backtest at all. Dry run reports the exact
# payload it WOULD have sent, which is what makes further review possible
# without money moving.
#
# Strategy 3 joined here 2026-08-23 - see the LIVE_TAGS comment above for why.
#
# "Strategy 1 1H +BTCUSDT" joined here 2026-08-27: a SEPARATE instance from
# the live "Strategy 1 1H", gating every signal on whether BTCUSDT's own
# trend (price vs its 200MA) agrees with the trade's direction. Built and
# backtested (see project-btc-trend-gate memory) after a long/short-by-year
# investigation found shorts crushed in a raging bull and longs crushed in a
# bear - the one idea that session which held up in both years independently
# and survived dropping each side's top 3 trades. Explicitly NOT a source of
# profit on its own - both AGREE and DISAGREE measured negative in every
# window tested - it reduces losses by skipping the worse half. Dry run
# alongside the live, ungated instance for now: this is a new, separate
# thing to observe before any deployment decision, not a replacement.
DRY_RUN_TAGS: set[str] = {
    f"Strategy 4 {tf} {variant}"
    for tf in ("15m", "1H")
    for variant in ("OB1.0", "OB2.0")
} | {"Strategy 3 1D/1H", "Strategy 3 1D/5m"}
AUTO_EXECUTE_TAGS = LIVE_TAGS | DRY_RUN_TAGS
# Strategies whose EXITS the bot may manage on a position it is already
# tracking, even though it never opens one for them. Strictly weaker than
# LIVE_TAGS: only reduce-only take-profits and protective stop moves, which
# cannot create or increase exposure.
#
# Deliberately just LIVE_TAGS, not AUTO_EXECUTE_TAGS: a dry-run strategy
# (Strategy 3, Strategy 4) has never opened a position for the bot to be
# managing exits ON, so there is nothing for the weaker permission to cover
# yet. If one of them is ever promoted to LIVE_TAGS and later demoted again
# WITH an open position on the book, that tag belongs in LEGACY_EXIT_TAGS
# below instead - the mechanism this distinction exists for.
#
# LEGACY_EXIT_TAGS is for tags that no longer PRODUCE signals but may still have
# open positions the bot must keep managing. A strategy replaced by a new
# version stops being in LIVE_TAGS the moment it is unregistered - and every
# position it opened, still on the book with a stop to move and a target to
# place, instantly stops being managed. That is the same silent orphaning as a
# hand-typed /add tag, arriving all at once instead of one trade at a time.
#
# Empty today. Replacing Strategy 2 with Strategy 2.1 is what fills it, and its
# entries come out again once those positions are closed.
#
# Strategy 2 was RETIRED on 2026-08-16 - see the handoff. Its four tags live
# here because trade 12 (ZHIPUHKDUSDT long, "Strategy 2 4H/1H") was open on the
# account when it was removed. Unregistering a strategy takes its tags out of
# LIVE_TAGS instantly, and with them the bot's permission to move that trade's
# stop to breakeven or place its take-profit. These entries come out once that
# position is closed.
#
# Strategy 2.1's 4H and 1D instances were RETIRED on 2026-08-19 on their own
# measurement - 4H at -0.058R over 2,030 setups and getting worse when its three
# best trades are discarded, 1D at -0.271R on 104. Nothing was open under either
# tag when they were unregistered, so these two entries cover only the window
# between the decision and the deploy: a signal approved in those minutes would
# otherwise have opened a position whose stop the bot instantly lost permission
# to move. They come out once nothing can be holding them.
#
# "Strategy 1 1H" and "Strategy 1 4H" were briefly listed here on 2026-08-27,
# while the BTC-gated instances replaced the ungated ones - the gate changes
# the tag (RsiFibReversal appends " +BTCUSDT"), so the old tags needed exit
# cover across that deploy. Both entries came out when the gate was reverted
# the same day (see build_strategies), and nothing was ever deployed under the
# "+BTCUSDT" tags, so no position can exist under them. Recorded because the
# next person to enable that gate needs to redo the same two-sided move: new
# tag into LIVE_TAGS, old tag into here until the book is clear.
#
# "Strategy 2.1 1H" briefly joined here 2026-08-28 for the same reason as
# the entries above it - its 1H instance was gated on daily_regime_from_bars
# for about a day. That gate was DROPPED 2026-08-29 (see build_strategies)
# before ever reaching a deploy, so no position ever opened under the
# "+BTCUSDT(1D)" tag and this entry came back out again the same session it
# went in - never actually needed.
#
# "Strategy 1 1H"/"Strategy 1 4H" join here 2026-08-29, for the SAME reason
# as their 08-27 entry above: btc_levels_symbol changes the tag
# (RsiFibReversal appends " +BTCUSDT(levels)"), so any position the plain,
# ungated instances already opened needs this cover across the deploy. Comes
# out once nothing can be holding either old tag.
#
# "Strategy 2.1 15m" joined here 2026-08-30, RETIRED on its own measurement -
# see ema_trend_v2.INSTANCES for the full table. Unlike the 08-19 4H/1D
# retirement this one found a real, held-out-confirmed gross edge (+0.12R in
# its best cell), just smaller than the fee drag 15m's own proportionally
# tighter ATR imposes on it - not a population with no edge at all, a
# population whose edge cannot currently clear fees. Dror confirmed neither
# Bitget lever closes that gap (BGB discount is spot-only, VIP tier not
# relevant to this account), which is what made this a retirement rather
# than a fee-tier fix. 1H is unaffected and stays the only live V21 instance.
LEGACY_EXIT_TAGS: set[str] = {
    "Strategy 2 1H/15m", "Strategy 2 4H/1H", "Strategy 2 1D/4H", "Strategy 2 1D",
    "Strategy 2.1 4H", "Strategy 2.1 1D", "Strategy 2.1 15m",
    "Strategy 1 1H", "Strategy 1 4H",
}
EXIT_MANAGED_TAGS = LIVE_TAGS | LEGACY_EXIT_TAGS
# How many days a capability may stay silent before the weekly report says so.
#
# These are judgements about CADENCE, not technical constants, and they are the
# one part of the ledger that needs Dror's eye rather than a measurement. Too
# tight and the report cries wolf every Sunday until it stops being read -
# which is its own silent failure, and the same shape as the five-minute
# Telegram expiry. Too loose and it takes a month to notice a dead capability.
#
# "Never worked at all" is reported regardless of these numbers, because that
# is the case that hid the partial take-profit for five months. The thresholds
# only govern the has-it-worked-LATELY question.
#
# Set against the OBSERVED gaps in the signals table, not guessed. Over 549
# logged signals, the longest a live instance has ever gone quiet:
#
#     Strategy 1 1H    117 signals / 16.1 days   longest silence 1.97 days
#     Strategy 1 4H     86 signals /  7.5 days   longest silence 1.50 days
#     Strategy 1 1D     11 signals / 14.7 days   longest silence 8.15 days
#     Strategy 2.1 15m 101 signals /  0.7 days   longest silence 0.05 days
#
# which is why one blanket 21 days was wrong in both directions at once: a dead
# 1H instance would have sat unnoticed for three weeks against a two-day normal,
# while 1D genuinely idles over a week and needs the room. Split by the
# instance's own base timeframe, with roughly 2x headroom over the worst silence
# seen - enough that ordinary quiet never speaks up.
#
#   entry orders    3 days - Dror's number. A whole system placing nothing for
#                   three days is either a dead market or a dead bot, and he
#                   would rather be asked.
#   take-profit     3 days - it is placed WITH the entry, right after the
#                   position confirms, so it shares the entry's cadence. It was
#                   14 on the assumption that it waits for a winner, which is
#                   not what the code does.
#   breakeven /     7 days - Dror's number, down from 14. Both need a winner to
#   trailing        reach its first target.
#   weekly report   8 days - matches WEEKLY_REPORT_MAX_AGE_DAYS.
#
# STRATEGY 2.1 KEEPS 2 DAYS whatever its base timeframe says. It prompts many
# times a day, so a single quiet day is already wrong. Strategy 3 is the
# standing lesson: shipped live, and its two instances have produced ZERO
# signals in the entire signals table - not a slow week, nothing at all.
SIGNAL_SILENCE_DAYS: dict[str, float] = {"5m": 2.0, "15m": 2.0, "1H": 4.0, "4H": 7.0, "1D": 14.0}
V21_SILENCE_DAYS = 2.0


def signal_silence_days(tag: str) -> float:
    """How long this instance may go quiet before the report says so.

    Keyed on the instance's own BASE timeframe, which is the last one named in
    the tag - "Strategy 3 1D/1H" reads its trend off the daily and acts on the
    hour, so it is an hourly instance. Same convention SWING_TAGS uses.
    """
    if tag in V21_TAGS:
        return V21_SILENCE_DAYS
    base = next((part for part in reversed(tag.replace("/", " ").split())
                 if part in SIGNAL_SILENCE_DAYS), None)
    return SIGNAL_SILENCE_DAYS.get(base, 14.0)


LEDGER_EXPECTATIONS: dict[str, float] = {
    ledger.TAKE_PROFIT_PLACED: 3.0,
    ledger.BREAKEVEN_STOP_MOVED: 7.0,
    ledger.TRAILING_STOP_MOVED: 7.0,
    ledger.ENTRY_ORDER_PLACED: 3.0,
    ledger.WEEKLY_REPORT: 8.0,
    # 35 days: the longest legitimate gap between two monthly runs is 31 (Oct 1
    # to Nov 1), plus four days of slack, mirroring the weekly's 8-for-7. It
    # rides poll_capability_silence like every other row here, so a monthly job
    # that quietly stops running is reported within days rather than waiting
    # for the second missed report - which would be two months.
    ledger.MONTHLY_REPORT: 35.0,
    **{ledger.signal_seen(tag): signal_silence_days(tag) for tag in sorted(LIVE_TAGS)},
}
# Every instance whose OWN actionable timeframe is 1D or slower - not every
# tag that happens to mention "1D" (Strategy 2 1D/4H trades off its 4H base
# and stays a day-pool signal even with 1D as its reference). These two share
# a hard cap of MAX_SWING_SLOTS concurrently pending+open positions, enforced
# independently of the aggregate dollar cap; everything else shares whatever
# headroom is left under that dollar cap with no separate reservation.
SWING_TAGS = {"Strategy 1 1D", "Strategy 2 1D"}
MAX_SWING_SLOTS = 2


_MANAGE_USAGE = (
    "Usage: /manage <trade_id> <breakeven> [runner_target]\n"
    "Arms the stop-to-breakeven (and optionally a runner target) for an open trade the bot "
    "isn't already managing — a trade added with /add, whose tag no routing set knows."
)


def parse_manage_args(args: list[str]) -> tuple[int, float, float | None] | str:
    """(trade_id, breakeven, runner_target) or the message to reply with.

    Split out from the handler because this is where a typed command goes
    wrong, and a handler that needs a running Telegram application to reach
    is a handler nothing tests. The prices themselves are sanity-checked
    against the live market by Scanner.adopt_trade, not here.
    """
    if len(args) not in (2, 3):
        return _MANAGE_USAGE
    try:
        trade_id = int(args[0])
    except ValueError:
        return f"'{args[0]}' isn't a trade id — it has to be the whole number from the trade's own alert."
    try:
        breakeven = float(args[1])
        runner_target = float(args[2]) if len(args) == 3 else None
    except ValueError:
        return "The breakeven and runner target have to be prices."
    return trade_id, breakeven, runner_target


def build_strategies() -> list:
    """Every strategy instance the notifier runs.

    The same two methods at three scales. The cheatsheet calls Strategy 1 a
    "1h+" method and describes chart structure as clearest on higher
    timeframes, and each pairing keeps a slower trend confirming a faster
    entry. Every instance is tagged with its own timeframe so their
    performance is measured separately - an edge at one scale is not evidence
    of one at another.

    A function rather than an inline list so the execution whitelist can be
    checked against it: a tag that no strategy produces, or a strategy no tag
    routes, is dead config that reads as intent. Both are silent failures -
    the second one means a strategy simply never trades.
    """
    return [
        # EVERY INSTANCE IS UNGATED, and that is a measured decision rather
        # than the absence of one. `market_trend_symbol` exists and works (see
        # rsi_fib_reversal.py and its tests); what did not survive measurement
        # is the case for switching it on.
        #
        # The gate was enabled on 1H and 4H on 2026-08-27 and reverted the same
        # day, when all three instances were finally put through ONE replay
        # path instead of comparing arms measured different ways. Separation
        # (agree meanR - disagree meanR), 735 symbols, this cap, each year a
        # fresh $100:
        #
        #     instance   year 1     year 2
        #     1H         +0.035     -0.002
        #     4H         +0.067     +0.128
        #     1D          n=0       -0.013
        #
        # 1H was shipped on "smaller losses in BOTH years independently". Year
        # 1 reproduces (+0.035 vs the +0.031 on file). YEAR 2 DOES NOT: -0.051
        # agree vs -0.050 disagree is nothing, against the +0.053 claimed. The
        # agree arm matches what was published (-0.051 vs -0.057); the disagree
        # arm does not (-0.050 vs -0.110), and those two published arms appear
        # to have come from different runs. Worse, in year 2 BOTH halves beat
        # baseline (-0.073) - the signature of a capital-allocation effect from
        # halving the book, not of selection.
        #
        # 4H looked like the stronger case (+0.067/+0.128) until the $5-per-leg
        # floor was removed: separation falls to -0.020 and +0.024. So most of
        # it was the gate correlating with which trades clear the minimum
        # notional - the floor selects on STOP WIDTH - rather than with whether
        # a trade wins. Its year 1 also failed drop-top-3 (agree -0.187 ->
        # -0.385, disagree -0.253 -> -0.388 on n=39/55).
        #
        # 1D cannot be tested at all: n=0 closed trades in year 1 and 5 in year
        # 2, because 230 warmup bars plus a 200-day MA eat ~430 of the 730 days
        # available, and the floor declines nearly all the rest.
        #
        # market_trend_symbol (above) is retired - the story stops there.
        #
        # A DIFFERENT gate, btc_levels_symbol (mtf_regime_read_timing, built
        # 2026-08-28/29 from a rule-by-rule review of Dror's own chart-reading
        # habit), is enabled on 1H and 4H below. Daily structure_trend sets
        # direction; BTC's OWN 1H significant-levels list (persisted over its
        # full available history, never pruned - see deep_history on the
        # Scanner construction below) sets timing. Measured through ONE
        # consistent replay path, 2026-08-29, meanR/drop3/eq/dd all next to
        # each other:
        #
        #     instance  year  baseline eq/dd      gated eq/dd         meanR base->gated   drop3 base->gated
        #     1H        Y1    $29.5 / 72.7%        $30.6 / 70.3%       -0.178 -> -0.168     -0.210 -> -0.199
        #     1H        Y2    $31.4 / 71.9%        $70.2 / 45.5%       -0.073 -> -0.019     -0.092 -> -0.037
        #     4H        Y1    $85.7 / 18.3%        $89.7 / 15.2%       -0.189 -> -0.156     -0.291 -> -0.272
        #     4H        Y2    $56.1 / 43.9%        $69.5 / 33.1%       -0.280 -> -0.183     -0.326 -> -0.229
        #
        # 1D not gated: too few closed trades (n<=5 either year) to read.
        #
        # NOT YET DONE before this was wired: a statistical significance test
        # (a fast permutation-test proxy was attempted and its own sanity
        # check failed badly - +1.14 meanR vs the real replay's -0.073 - so no
        # trustworthy p-value exists yet). Shipped on the eq/dd/meanR/drop3
        # table above at Dror's explicit direction, with that gap still open.
        RsiFibReversal("1H", btc_levels_symbol="BTCUSDT"),
        RsiFibReversal("4H", btc_levels_symbol="BTCUSDT"),
        RsiFibReversal("1D"),
        # EVERY V21 INSTANCE IS UNGATED, and that is a measured decision.
        # daily_regime_from_bars (structure + level-proximity on BTC's OWN
        # daily chart, regime.py) was gated onto the 1H instance 2026-08-28
        # ("agreeing reads -0.331R/-0.447R, fighting reads -0.474R/-0.824R -
        # smaller losses in both years"), then DROPPED 2026-08-29 once
        # measured against Strategy 1's own newer levels-based gate as the
        # bar to clear: on Strategy 2.1's 1H AND 15m, ungated vs gated
        # equity/drawdown barely move (1H: $0.92/$0.58 -> $0.94/$0.63; 15m:
        # $0.59/$0.56 -> $0.60/$0.57, both still ~99%+ max drawdown either
        # way) - smaller per-trade R losses, but no real capital protection,
        # unlike Strategy 1's 1H/4H below where the same kind of gate
        # measurably changed the outcome. market_regime_symbol still works
        # (see ema_trend_v2.py and its tests); what did not survive this
        # comparison is the case for using it on Strategy 2.1 specifically.
        #
        # "Strategy 2.1 15m" RETIRED 2026-08-30 - see ema_trend_v2.INSTANCES
        # for the measurement. Not instantiated here at all now, so it never
        # evaluates and never alerts, unlike a DRY_RUN_TAGS demotion. Tag
        # moved to LEGACY_EXIT_TAGS below so any position already open under
        # it keeps being managed.
        EmaTrendV2("1H"),
        # Strategy 3's swing version: the consolidation read off daily
        # bars, triggered on 1H, 75% at 1:2 and the runner closed at daily
        # resistance or after 3 trading days, whichever comes first.
        VolumeRun("1D", "1H", time_exit_days=3),
        # The day version. SAME daily consolidation - the cheatsheet
        # identifies it on the daily chart for both - with a 5m trigger and
        # a flat exit at 1:2, no runner and no time clock behind it.
        #
        # This previously read its whole structure off HOURLY bars, which
        # neither sheet asks for; the width and volume rules were then
        # retuned to compensate, and the instance still shipped broken once
        # on rate-only calibration. Kept from that episode: the breakout
        # penetration floor (a 5m close can graze a daily level), the
        # range-keyed de-duplication, and session gating so a tokenized
        # stock cannot signal while its market is shut.
        VolumeRun(
            "1D", "5m",
            time_exit_days=None,
            armed_only=True,
            params=DAY_PARAMS,
            session_gated=True,
            partial_fraction=DAY_PARTIAL_FRACTION,
            # The day sheet stops below the last low BEFORE the break; the
            # swing sheet uses the breakout candle's own low. Only instance
            # difference in the stop.
            stop_anchor=STOP_AT_RECENT_LOW,
        ),
        # Strategy 4, order blocks - single timeframe each, not a slow/fast
        # pair: the block, its sweep, the dealing range and the trigger are all
        # read off one chart. Order matters, since the scanner takes one
        # position per symbol and the first instance to produce a signal wins
        # it: slowest first, matching the pattern-precedence rule.
        #
        # 5m was measured and dropped. Its gap targets essentially do not
        # exist - only 0.6% of otherwise-valid setups had an unclosed gap to
        # aim at, because a 5m imbalance is filled almost as soon as it forms.
        OrderBlockStrategy("1H"),
        OrderBlockStrategy("15m"),
    ]


async def async_main() -> None:
    bitget = client_from_settings(settings)
    storage = Storage(settings.trades_db_path)
    bot = NotifierBot(settings.telegram_bot_token, settings.telegram_chat_id)

    # Dry run reports the exact payload instead of sending it, so the hedge-mode
    # side pairing, the size units and the leverage can be read against a real
    # trade before any money moves. AUTO_EXECUTE is the master switch: with it
    # off, every strategy reports rather than places, whatever LIVE_TAGS says -
    # so a restart or a fresh deploy can never come up trading by itself.
    dry_run = DryRunExecutor(report=lambda text: asyncio.create_task(bot.send_message(text)))
    if settings.auto_execute:
        live = LiveExecutor(bitget)
        executor = RoutingExecutor(
            {tag: live for tag in LIVE_TAGS} | {tag: dry_run for tag in DRY_RUN_TAGS},
            exit_managed_tags=EXIT_MANAGED_TAGS,
        )
    else:
        executor = dry_run

    scanner = Scanner(
        bitget=bitget,
        bot=bot,
        storage=storage,
        executor=executor,
        watchlist=WATCHLIST,
        strategies=build_strategies(),
        risk_pct=RISK_PCT,
        max_leverage=MAX_LEVERAGE,
        max_total_risk_pct=MAX_TOTAL_RISK_PCT,
        auto_execute_tags=AUTO_EXECUTE_TAGS,
        # Same table the weekly report surveys, now also watched
        # continuously so a dead capability surfaces the day it dies
        # rather than on the next Sunday that a report actually sends.
        ledger_expectations=LEDGER_EXPECTATIONS,
        swing_tags=SWING_TAGS,
        max_swing_slots=MAX_SWING_SLOTS,
        # btc_levels_symbol (RsiFibReversal, see build_strategies) needs the
        # "BTCUSDT@1H" reference series deep enough for build_levels'
        # persistent, never-pruned significant-levels list to mean what was
        # actually measured - the plain 600-bar default is ~25 days, and
        # ~17% of the levels that mattered in that measurement were formed
        # MORE than 2 years before they were used. 100,000 is bigger than
        # BTCUSDT's entire 1H history (~62,000 bars back to its 2019-07-10
        # listing) - get_candles' own history-paging stops once the exchange
        # has nothing older left, so this asks for everything there is
        # rather than an arbitrary cutoff. Refetched once per closed hourly
        # candle (Scanner._bars' own cache), not once per evaluate() call.
        deep_history={("BTCUSDT", "1H"): 100_000},
        send_chart_images=settings.send_chart_images,
    )

    async def pause(update, _context) -> None:
        scanner.execution_paused = True
        await update.message.reply_text("Execution paused. Signals keep arriving; place them by hand.")

    async def resume(update, _context) -> None:
        scanner.execution_paused = False
        mode = "live" if settings.auto_execute else "dry run"
        await update.message.reply_text(f"Execution resumed ({mode}).")

    async def status(update, _context) -> None:
        mode = "LIVE" if settings.auto_execute else "dry run"
        state = "PAUSED" if scanner.execution_paused else "active"
        await update.message.reply_text(f"Execution: {state} ({mode}).")

    async def manage(update, context) -> None:
        """Adopt a hand-added trade into exit management.

        Needed because a trade registered with /add carries whatever tag was
        typed at the prompt, which never matches an instance tag in LIVE_TAGS
        - so the bot would silently manage nothing about it. The permission
        goes on the trade row rather than on the tag, since the tag is what
        the weekly review groups by.
        """
        parsed = parse_manage_args(context.args or [])
        if isinstance(parsed, str):
            await update.message.reply_text(parsed)
            return
        await update.message.reply_text(await scanner.adopt_trade(*parsed))

    async def reoffer_signal(signal_id: int) -> str:
        """/add <n> - put an expired signal back in front of Dror.

        Dror, 2026-08-20: "add a number to each signal so i can add them after
        they expire if i think the setup is still alive".

        It goes back through Scanner._dispatch rather than straight to an
        order, which is the whole point: sizing, the fill guard, the aggregate
        risk cap and the exchange minimums are all recomputed at the CURRENT
        price, and it still arrives as an Approve/Reject prompt. An expired
        signal's stored entry is stale by definition - placing at it would
        repeat exactly the defect found in Strategy 2.1 this morning, where a
        trade was sized against a price it could never fill at.

        The tag comes from the stored Signal, so there is no prompt to mistype.
        """
        payload = storage.signal_payload(signal_id)
        if payload is None:
            return (
                f"No stored signal #{signal_id}. Either the number is wrong, or the alert "
                f"predates signal storage, or it was logged without one (a too-small refusal)."
            )
        try:
            signal = signal_from_json(payload)
        except Exception:
            logger.exception("Could not rebuild signal %s", signal_id)
            return f"Signal #{signal_id} is stored but could not be rebuilt. Nothing was sent."

        if storage.has_open_or_pending(signal.symbol):
            return f"Already in {signal.symbol} - one position per symbol. Nothing was sent."
        try:
            equity = bitget.get_account_equity()
        except Exception:
            logger.exception("Could not read equity while re-offering signal %s", signal_id)
            return "Could not read the account equity, so nothing was sized. Nothing was sent."

        before = len(bot.app.bot.sent) if hasattr(bot.app.bot, "sent") else None
        await scanner._dispatch(signal, equity, list(signal.analysis_timeframes or ()))
        # _dispatch refuses silently in several places - the throttle, the risk
        # cap, the exchange minimum - and a command that answers nothing at all
        # is indistinguishable from one that crashed.
        if before is not None and len(bot.app.bot.sent) == before:
            return (
                f"Signal #{signal_id} ({signal.symbol} {signal.direction}, {signal.strategy_tag}) "
                f"was not re-sent: it is refused at today's price, size or risk cap. "
                f"The log says which."
            )
        return f"Re-offered signal #{signal_id}: {signal.symbol} {signal.direction} ({signal.strategy_tag})."

    bot.app.add_handler(CommandHandler("pause", pause))
    bot.app.add_handler(CommandHandler("resume", resume))
    bot.app.add_handler(CommandHandler("status", status))
    bot.app.add_handler(CommandHandler("manage", manage))
    # The partial-fill callback is the scanner's, so an /add trade takes the
    # same path as every other: one place decides what a scale-out means, and
    # an adopted trade gets its exits managed from there. Close and scale-in
    # stay local - _on_trade_closed also cancels resting orders on the
    # symbol, which is not something to start doing to hand-placed trades
    # without asking.
    #
    # Strict, tappable list rather than free text - see manual_entry's own
    # docstring for why. Built from the SAME strategies the scanner is
    # already running (scanner.strategies), not a fresh build_strategies()
    # call, so a hand-added trade can never offer a tag the running process
    # doesn't actually route somewhere.
    manual_tag_options = sorted({t for s in scanner.strategies for t in s.all_tags()}) + ["Other / discretionary"]
    bot.app.add_handler(make_add_conversation(storage, bitget, manual_tag_options,
                                              on_partial=scanner._on_partial_exit,
                                              reoffer=reoffer_signal))
    await bot.start_polling()

    # Reuses the scanner's own callbacks rather than duplicating them: a trade
    # re-attached after a restart needs the same close/partial handling -
    # including cancelling whatever's left resting once it closes - as one
    # tracked without interruption.
    resume_open_trades(
        storage,
        bitget,
        on_close=scanner._on_trade_closed,
        on_partial=scanner._on_partial_exit,
        on_scale_in=scanner._on_scale_in,
    )

    # `systemctl restart` sends SIGTERM. python-telegram-bot only installs a
    # handler for it inside its own run_polling()/run_webhook() convenience
    # wrappers - this process calls the lower-level start_polling() instead
    # (so the scanner's loops can run in the same event loop), which means
    # nothing here catches SIGTERM by default and Python's own default
    # disposition is immediate termination: no `finally` block runs, nothing
    # gets cleaned up. Every deploy this evening silently orphaned whatever
    # signal offers were mid-flight - see NotifierBot.cancel_all_pending's
    # docstring for what that cost. Handled explicitly instead of assumed:
    # cancel every pending offer's message, then cancel this task so the
    # `finally` below (and asyncio.run's own unwind) still runs normally.
    # Systemd's default stop grace (~90s) is far more than the handful of
    # Telegram edits this needs.
    main_task = asyncio.current_task()

    async def _on_sigterm() -> None:
        logger.info("SIGTERM received; clearing pending signal offers before exit")
        try:
            await bot.cancel_all_pending()
        except Exception:
            logger.exception("Could not clear pending signal offers before shutdown")
        main_task.cancel()

    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, lambda: asyncio.create_task(_on_sigterm())
    )

    try:
        await scanner.run_forever()
    except asyncio.CancelledError:
        # Only ever raised here by _on_sigterm cancelling main_task above -
        # a deliberate, self-triggered shutdown, not an external cancellation
        # to propagate. Left uncaught, it escapes asyncio.run() and crashes
        # the process with exit code 1 even though everything actually
        # shut down cleanly (confirmed live on the VM, 2026-08-26: SIGTERM
        # logged, cancel_all_pending() ran, bot.stop() completed - then a
        # CancelledError traceback and "Failed with result 'exit-code'"
        # anyway). Swallowed here so a SIGTERM restart reads as the plain
        # stop it is.
        logger.info("Shutting down after SIGTERM")
    finally:
        await bot.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs at INFO, which writes the Telegram bot
    # token into the system journal in plaintext.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
