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
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import NotifierBot
from execution.executor import DryRunExecutor, LiveExecutor, RoutingExecutor
from execution.manual_entry import make_add_conversation
from execution.tracker import resume_open_trades
from notifier.scanner import Scanner
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.strategies.volume_run import HOURLY_PARAMS, VolumeRun
from notifier.watchlist import WATCHLIST

RISK_PCT = 0.01  # 1-2% per trade, hard-capped at 2% in risk_sizing.plan_position
MAX_TOTAL_RISK_PCT = 0.06  # aggregate ceiling across all open trades
# Leverage is solved per trade to fit the margin left after other open trades;
# this only caps how high it may go.
MAX_LEVERAGE = 20.0
# Only these place orders on their own. A whitelist rather than a flag: a new
# strategy has to be named here before it can spend money, so registering one
# can never start it executing by accident. Strategy 3 is deliberately absent -
# it has no measured signals yet, and its 5m instance would fire unattended on
# the timeframe where fees measured worst.
#
# Strategies graduate to live trading one at a time, once their dry-run
# payloads have been read against a real chart. Strategy 1 has been; Strategy 2
# has not, so it keeps reporting what it would have placed. Strategy 3 is in
# neither set - it has no measured signals yet, and its 5m instance would fire
# unattended on the timeframe where fees measured worst.
LIVE_TAGS = {"Strategy 1 1H", "Strategy 1 4H", "Strategy 1 1D"}
DRY_RUN_TAGS = {"Strategy 2 1H/15m", "Strategy 2 4H/1H", "Strategy 2 1D/4H", "Strategy 2 1D"}
AUTO_EXECUTE_TAGS = LIVE_TAGS | DRY_RUN_TAGS
# Strategies whose EXITS the bot may manage on a position it is already
# tracking, even though it never opens one for them. Strictly weaker than
# LIVE_TAGS: only reduce-only take-profits and protective stop moves, which
# cannot create or increase exposure.
#
# Strategy 3 is here and NOT in AUTO_EXECUTE_TAGS on purpose. Dror places its
# entry, stop and partial by hand and that has not changed; what he asked for
# is the runner's take-profit set for him once the partial fills, because the
# alert's own tail read "close the remaining 0.13 at your discretion" and the
# level to use is a chart read the bot can do. Strategy 1's tags are included
# so its runner - a 1:3 price it already computes and prints, and has never
# actually placed - gets an order too.
EXIT_MANAGED_TAGS = LIVE_TAGS | {"Strategy 3 1D/1H", "Strategy 3 1H/5m"}
# Every instance whose OWN actionable timeframe is 1D or slower - not every
# tag that happens to mention "1D" (Strategy 2 1D/4H trades off its 4H base
# and stays a day-pool signal even with 1D as its reference). These two share
# a hard cap of MAX_SWING_SLOTS concurrently pending+open positions, enforced
# independently of the aggregate dollar cap; everything else shares whatever
# headroom is left under that dollar cap with no separate reservation.
SWING_TAGS = {"Strategy 1 1D", "Strategy 2 1D"}
MAX_SWING_SLOTS = 2


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
            # Strategy 3 falls through to the default ManualExecutor for
            # ENTRIES - nothing is placed - while still qualifying for exit
            # management via exit_managed_tags.
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
        # The same two methods at three scales. The cheatsheet calls
        # Strategy 1 a "1h+" method and describes chart structure as clearest
        # on higher timeframes, and each pairing keeps a slower trend
        # confirming a faster entry. Every instance is tagged with its own
        # timeframe so their performance is measured separately - an edge at
        # one scale is not evidence of one at another.
        strategies=[
            RsiFibReversal("1H"),
            RsiFibReversal("4H"),
            RsiFibReversal("1D"),
            EmaTrendFollowing("15m", "1H"),
            EmaTrendFollowing("1H", "4H"),
            EmaTrendFollowing("4H", "1D"),
            EmaTrendFollowing("1D"),
            # Strategy 3's swing version: the consolidation read off daily
            # bars, triggered on 1H, runner closed after 3 trading days.
            VolumeRun("1D", "1H", time_exit_days=3),
            # The intraday version: the same consolidation algorithm read off
            # HOURLY bars with a 5m trigger, its own tuned params, and no time
            # exit. Re-enabled after a rebuild - it was originally calibrated
            # on signal rate alone, met that, and then fired four live signals
            # that were not the setup at all. What changed: a minimum breakout
            # penetration (TSLAUSDT triggered 0.012% past the line), an
            # absolute width ceiling on top of the ATR one (AXTIUSDT passed a
            # 28.22%-wide "consolidation" because a violent move had inflated
            # hourly ATR), de-duplication keyed on the range instead of the
            # entry price (TSLAUSDT re-fired ten minutes later off the same
            # range), and session gating so a tokenized stock cannot signal
            # while its market is shut.
            VolumeRun("1H", "5m", time_exit_days=None, armed_only=True, params=HOURLY_PARAMS, session_gated=True),
        ],
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

    bot.app.add_handler(CommandHandler("pause", pause))
    bot.app.add_handler(CommandHandler("resume", resume))
    bot.app.add_handler(CommandHandler("status", status))
    bot.app.add_handler(make_add_conversation(storage, bitget))
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
