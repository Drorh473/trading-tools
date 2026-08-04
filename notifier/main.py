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
from notifier.strategies.volume_run import VolumeRun
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
        executor = RoutingExecutor({tag: live for tag in LIVE_TAGS} | {tag: dry_run for tag in DRY_RUN_TAGS})
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
            # The 1H/5m instance is DISABLED pending a rebuild. It was
            # calibrated on signal RATE alone, which it met - and then fired
            # four signals live that were not the setup at all:
            #
            #  - breakout line is the pivot bar's HIGH with no minimum
            #    penetration, so TSLAUSDT triggered 0.012% past it (4 cents on
            #    $324), then AGAIN 10 minutes later at 0.006% when price
            #    wobbled back under the line and re-armed the "first close
            #    above" guard. Dedupe missed it: the 5m closes differed by 2c.
            #  - max_range_atr is ATR-relative only, and hourly ATR is
            #    inflated by the very move that formed the range - so AXTIUSDT
            #    passed a 28.22%-wide "consolidation", INTCUSDT a 6.15% one
            #    that had held all of 5 hourly bars.
            #  - worst, the volume thesis is satisfied by the trading session
            #    rather than by supply. INTCUSDT (tokenized Intel) has a 28x
            #    median-volume swing between 13:00 UTC (US open) and 21:00
            #    against a 2.0x spike threshold: its range bottom scored a
            #    136x "spike" because that bar IS the opening bell, and the
            #    "volume dried up inside the range" reading of 0.10 was just
            #    the session winding down. Every tokenized equity reproduces
            #    this every single day.
            #
            # HOURLY_PARAMS and the params plumbing stay in volume_run.py -
            # the fix is a real redesign (session-normalized volume, an
            # absolute width ceiling, a minimum penetration, stateful re-fire
            # suppression per range rather than per price), not new constants,
            # and it needs measuring for quality and not just for rate.
        ],
        risk_pct=RISK_PCT,
        # TEMPORARY, 2026-08-04: equal to risk_pct, so pattern confluence
        # cannot raise risk at all right now. confluence() accepts a breakout
        # up to CONFLUENCE_BARS=50 bars old - 2.1 days on 1H - and TRXUSDT was
        # sized at 2% citing a falling wedge that had broken 17 HOURS earlier,
        # while the structure actually in front of price was an unresolved
        # flag that could still have broken down. Strategy 1 is live and
        # auto-executes, so rather than leave real money sized off two-day-old
        # evidence, confluence stops paying until the staleness window is
        # measured properly (a guessed constant would be no better than the
        # one it replaced). Restore to CONFLUENCE_RISK_PCT once that lands -
        # and note the staged 1% -> +1% design replaces this bump entirely.
        confluence_risk_pct=RISK_PCT,
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
    resume_open_trades(storage, bitget, on_close=scanner._on_trade_closed, on_partial=scanner._on_partial_exit)

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
