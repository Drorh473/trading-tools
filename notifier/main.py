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
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.order_block import OrderBlockStrategy
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.strategies.volume_run import (
    DAY_PARAMS,
    DAY_PARTIAL_FRACTION,
    STOP_AT_RECENT_LOW,
    VolumeRun,
)
from notifier.watchlist import WATCHLIST

RISK_PCT = 0.01  # 1-2% per trade, hard-capped at 2% in risk_sizing.plan_position
MAX_TOTAL_RISK_PCT = 0.06  # aggregate ceiling across all open trades
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
# STRATEGY 3 IS STILL UNMEASURED - no backtest, and its day instance shipped
# broken once on rate-only calibration. It goes live on Dror's explicit call
# with that stated, not because the evidence changed. Its session gate and
# alert-only history are what bound the risk; watch its first live fills.
#
# The day instance is "Strategy 3 1D/5m", not "1H/5m": its consolidation moved
# back onto daily bars to match the cheatsheet. The tag is built from the
# timeframe pair, so it changed with the instance - and a tag missing from
# this set silently loses auto-execution rather than failing loudly, which is
# why test_main_wiring asserts every registered tag is routed.
LIVE_TAGS = {
    "Strategy 1 1H", "Strategy 1 4H", "Strategy 1 1D",
    "Strategy 2 1H/15m", "Strategy 2 4H/1H", "Strategy 2 1D/4H", "Strategy 2 1D",
    "Strategy 3 1D/1H", "Strategy 3 1D/5m",
}
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
DRY_RUN_TAGS: set[str] = {
    f"Strategy 4 {tf} {variant}"
    for tf in ("15m", "1H")
    for variant in ("OB1.0", "OB2.0")
}
AUTO_EXECUTE_TAGS = LIVE_TAGS | DRY_RUN_TAGS
# Strategies whose EXITS the bot may manage on a position it is already
# tracking, even though it never opens one for them. Strictly weaker than
# LIVE_TAGS: only reduce-only take-profits and protective stop moves, which
# cannot create or increase exposure.
#
# Now that every strategy is in LIVE_TAGS this is the same set, and
# manages_exits() already defaults to handles_live(). It is kept as its own
# name because the DISTINCTION still matters: exit management is the weaker
# permission, and if any strategy is ever demoted back to dry run it should
# lose the right to open trades without necessarily losing the right to
# manage exits on positions already placed by hand. That is exactly the state
# Strategy 3 was in until this commit.
EXIT_MANAGED_TAGS = LIVE_TAGS
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
# Starting points, deliberately generous:
#   take-profit / breakeven  14 days - they need a winner to reach its target
#                            first, and at ~1 trade a week that is not quick
#   entry orders              7 days - if a week passes with nothing placed,
#                            either the market is dead or the bot is
#   weekly report             8 days - matches WEEKLY_REPORT_MAX_AGE_DAYS
#   per-instance signals     21 days - a 1D instance can idle for weeks and be
#                            perfectly healthy; this only catches the truly
#                            inert, which is what it is for
LEDGER_EXPECTATIONS: dict[str, float] = {
    ledger.TAKE_PROFIT_PLACED: 14.0,
    ledger.BREAKEVEN_STOP_MOVED: 14.0,
    ledger.ENTRY_ORDER_PLACED: 7.0,
    ledger.WEEKLY_REPORT: 8.0,
    **{ledger.signal_seen(tag): 21.0 for tag in sorted(LIVE_TAGS)},
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
        RsiFibReversal("1H"),
        RsiFibReversal("4H"),
        RsiFibReversal("1D"),
        EmaTrendFollowing("15m", "1H"),
        EmaTrendFollowing("1H", "4H"),
        EmaTrendFollowing("4H", "1D"),
        EmaTrendFollowing("1D"),
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
            # DRY_RUN_TAGS is empty now that every strategy is live, but the
            # routing stays: demoting one back to dry run is a single move
            # between the two sets, with no other change needed here.
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
        swing_tags=SWING_TAGS,
        max_swing_slots=MAX_SWING_SLOTS,
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
    bot.app.add_handler(make_add_conversation(storage, bitget, on_partial=scanner._on_partial_exit))
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

    try:
        await scanner.run_forever()
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
