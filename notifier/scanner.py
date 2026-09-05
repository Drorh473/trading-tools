"""Main scanning loop: pulls closed bars for each watchlist symbol, runs the
active strategies, and — on a fresh signal — sizes the position and dispatches
an Approve/Reject alert to Telegram.

Scans are aligned to candle closes rather than run on a fixed interval:
strategies evaluate closed candles, so between one close and the next the data
is identical and re-scanning only burns API quota. Each strategy declares the
timeframe(s) it needs; the scanner fetches the union of all of them and scans
at the cadence of the shortest one, so a strategy needing 1H+15m confluence
runs alongside a 1H-only strategy without special-casing either.

Approval doesn't log the trade immediately — it waits to see a matching
position actually appear on Bitget (execution.tracker.wait_for_signal_position),
the same account-based detection the `/add` flow and the future auto-execution
phase use. If none appears within the timeout the row is cancelled, which also
frees the symbol for future signals.
"""

import asyncio
import logging
import time
from types import SimpleNamespace

import pandas as pd

from core import ledger
from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.executor import Executor
from notifier import sessions
from notifier.bar_cache import BarCache, bars_dataframe
from notifier.exit_manager import (
    ExitManager,
    PARTIAL_TAKE_FRACTION,
    REMAINDER_TARGET_RATIO,
    RUNNER_LEVEL_PIVOT_ATR_MULTIPLE,
    RUNNER_LEVEL_TIMEFRAME,
)
from notifier.pending_break_watcher import PendingBreakWatcher
from notifier.position_health import PositionHealthMonitor
from notifier.scanner_time import (
    SIGNAL_EXPIRY_CEILING,
    SIGNAL_EXPIRY_FLOOR,
    _split_reference_key,
    seconds_until_next_close,
    signal_expiry_seconds,
)
from notifier.trailing_stops import (
    RUNNER_LEVEL_ATR_PERIOD,
    STALL_TIGHTEN_FRACTION,
    STALL_TIGHTEN_R,
    TRAILING_POLL_TIMEFRAME,
    TRAIL_PIVOT_ATR_MULTIPLE,
    TrailingStopManager,
)
from notifier.signal_dispatcher import SignalDispatcher, _reward_target
from notifier.trade_lifecycle import TradeLifecycleHandler
from notifier.risk_sizing import (
    DEFAULT_MAX_LEVERAGE,
    DEFAULT_REWARD_RISK_RATIO,
    plan_position,
    round_trip_fee_for,
)
from notifier.strategies import patterns
from notifier.strategies.indicators import atr
from notifier.strategies.structure import nearest_level_beyond, zigzag_pivots
from notifier.strategies.base import TIMEFRAME_SECONDS, Signal, Strategy, signal_to_json

logger = logging.getLogger(__name__)

# Deliberately left at the conservative 6% even though production now runs
# 15%: notifier.main always passes max_total_risk_pct explicitly and is the
# authority, so this is only reached by a Scanner constructed without one. A
# fallback that errs low cannot spend money that was never asked for.
DEFAULT_MAX_TOTAL_RISK_PCT = 0.06
# Timeframes scanned for chart patterns that confirm a signal. Patterns never
# generate an alert of their own — measured standalone they had no edge on any
# timeframe — but a recent one alongside a signal measured +0.29R against
# -0.2R without. 1H and 4H are both kept because nine samples on 4H against
# seventeen on 1H cannot say which is the better confirmation.
#
# 1D was added after the flag-pole rework. Reviewing rendered matches Dror kept
# saying the same thing about shapes on 1H and 4H - SNDKUSDT, COTIUSDT and
# AMZNUSDT were each "a flag, but one timeframe up" - and the daily frame those
# comments pointed at was simply never scanned, so the pattern he was
# describing could not be found anywhere. Costs no extra fetching: 1D is
# already in required_timeframes() because Strategy 1 1D and Strategy 2 1D
# declare it.
#
# ORDER IS THE PRECEDENCE. confluence() and pending() both return the FIRST
# match, so this tuple decides which timeframe an alert cites when a symbol
# carries the same shape on more than one. Longest first, by Dror's call: a
# daily head-and-shoulders is stronger evidence than an hourly one, and its
# levels are the ones price actually respects. The previous 1H-first order was
# inherited from when 1H and 4H were the only options and nothing had chosen
# between them - it meant the weakest available reading won by default.
CONFLUENCE_TIMEFRAMES = ("1D", "4H", "1H")
# Cadence for the pending-break watch. The break itself is a CLOSE beyond the
# level, so on a 1H pattern it can only happen hourly - polling at 5m bounds
# how long after that close the add-on is offered, rather than making the
# break detectable more often. The regular scan's 15m cadence would leave the
# quoted level up to 15 minutes stale, by which point price may be well past
# the entry the add-on was sized for.
PENDING_BREAK_TIMEFRAME = "5m"
# How long one symbol/instance pair stays quiet after prompting. A day, rolling.
ALERT_THROTTLE_SECONDS = 24 * 3600


def _drop_superseded(
    produced: list[tuple[Strategy, Signal]],
) -> list[tuple[Strategy, Signal]]:
    """The signals worth acting on, once instances describing the SAME trade
    have been collapsed to the best-informed one.

    A strategy declares `supersedes` - tags whose signal it replaces on the same
    symbol and side. Strategy 2.1's paired instances supersede their own base
    timeframe's standalone instance: measured, the two coincide on 26% of
    standalone triggers with the same entry level and the same stop, differing
    only in where the target sits. Acting on both puts 2% of equity on one idea
    in two correlated positions.

    Direction is part of the match. A long and a short on one symbol are not the
    same trade, and if both somehow fire neither should silence the other -
    that is a contradiction worth seeing, not hiding.
    """
    claimed = {
        (signal.symbol, signal.direction, tag)
        for strategy, signal in produced
        for tag in strategy.supersedes
    }
    if not claimed:
        return produced
    kept = []
    for strategy, signal in produced:
        if (signal.symbol, signal.direction, signal.strategy_tag) in claimed:
            logger.info(
                "%s %s superseded on %s: a better-informed instance fired the same trade",
                signal.strategy_tag,
                signal.direction,
                signal.symbol,
            )
            continue
        kept.append((strategy, signal))
    return kept


class Scanner:
    def __init__(
        self,
        bitget: BitgetClient,
        bot,
        storage: Storage,
        executor: Executor,
        watchlist: list[str],
        strategies: list[Strategy],
        risk_pct: float = 0.01,
        reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
        max_leverage: float = DEFAULT_MAX_LEVERAGE,
        max_total_risk_pct: float = DEFAULT_MAX_TOTAL_RISK_PCT,
        # Enough history for a 200-period MA on every timeframe, plus room
        # for pattern detection which needs several swings in the window.
        candle_limit: int = 600,
        # Per-(symbol, timeframe) overrides above candle_limit - for a
        # reference series whose gate needs real depth (e.g. a persistent,
        # never-pruned significant-levels list, notifier.strategies.levels),
        # not the per-symbol indicator window every other fetch needs. Only
        # the keys listed here pay the deeper fetch; everything else keeps
        # the plain 600-bar default. A very large value (bigger than the
        # symbol's actual history) is fine - get_candles' own history-paging
        # loop stops once the exchange has nothing older left to page in.
        deep_history: dict[tuple[str, str], int] | None = None,
        # Which strategy tags may place orders automatically. Deliberately a
        # whitelist rather than a flag: a newly added strategy has to be named
        # here before it can spend money, so it cannot start executing merely
        # by being registered. Everything not listed still alerts normally and
        # is placed by hand.
        auto_execute_tags: set[str] | None = None,
        # Tags that were once in auto_execute_tags and no longer are, but may
        # still have an open position on the book - notifier.main's own
        # LEGACY_EXIT_TAGS, deliberately never used for a tag that has NEVER
        # been live (see its comment there: promoted-then-demoted only). This
        # is what _on_resize needs and auto_executes/handles_live cannot give
        # it: whether the bot ITSELF placed this trade's resting limit leg,
        # which is a fact about the trade's history and does not change when
        # the tag is later retired from new entries.
        legacy_exit_tags: set[str] | None = None,
        # capability -> days it may go quiet, handed straight to
        # PositionHealthMonitor, which owns the silence check. notifier.main
        # owns the table (LEDGER_EXPECTATIONS); importing it here would be a
        # cycle. Left out, the check is off.
        ledger_expectations: dict[str, float] | None = None,
        # A signal on one of these tags counts against the swing pool's own
        # hard slot cap (pending + open, combined across every swing tag)
        # rather than only the aggregate dollar cap - classified by each
        # instance's own actionable timeframe: 1D or slower is a swing, not
        # every alert whose tag string happens to mention "1D" (Strategy 2
        # 1D/4H trades off its 4H base and stays a day-pool signal even
        # though 1D appears as its reference).
        swing_tags: frozenset[str] = frozenset({"Strategy 1 1D", "Strategy 2 1D"}),
        max_swing_slots: int = 2,
        # Off by default, matching Settings.send_chart_images - see its own
        # docstring. When True, _dispatch attaches a candlestick chart (see
        # notifier.chart) to every alert it can build one for; a symbol with
        # too little fetched history still gets the plain text alert.
        send_chart_images: bool = False,
    ):
        self.bitget = bitget
        self.bot = bot
        self.storage = storage
        self.executor = executor
        self.watchlist = watchlist
        self.strategies = strategies
        self.risk_pct = risk_pct
        # A signal confirmed by a chart pattern is sized at a higher risk. The
        # evidence for that is 22 trades, so it is deliberately capped at the
        # same 2% ceiling every other trade obeys rather than treated as a
        # licence to size beyond the usual rules.
        self.reward_risk_ratio = reward_risk_ratio
        self.max_leverage = max_leverage
        self.max_total_risk_pct = max_total_risk_pct
        self.candle_limit = candle_limit
        self.deep_history = deep_history or {}
        self.swing_tags = swing_tags
        self.max_swing_slots = max_swing_slots
        self._bar_cache = BarCache(bitget, candle_limit, deep_history)
        # A lambda, not `executor.manages_exits` directly - the latter binds
        # the method eagerly at construction time, which means a test double
        # that never implements manages_exits (because no path it exercises
        # ever calls it) would break Scanner's OWN constructor instead of
        # only the specific call that actually needed it. This keeps the
        # lookup exactly as lazy as it was when it lived inline.
        self._exits = ExitManager(bitget, storage, bot, self._bar_cache, lambda tag: self.executor.manages_exits(tag))
        self._lifecycle = TradeLifecycleHandler(bitget, storage, bot, self._exits)
        self._trailing = TrailingStopManager(
            bitget, storage, bot, self._bar_cache, lambda tag: self.executor.manages_exits(tag)
        )
        # Insertion-ordered so _prune_seen can drop the OLDEST rather than an
        # arbitrary half - see there. Values are unused; this is a set that
        # remembers order.
        self._seen: dict[tuple, None] = {}
        # strategy tag -> symbols worth polling on its armed timeframe. Rebuilt
        # wholesale by every scan rather than mutated, so nothing can stay
        # armed after its setup stops qualifying.
        self._armed: dict[str, set[str]] = {}
        # (symbol, strategy_tag) -> when it last produced an Approve/Reject
        # prompt. See _throttled.
        self._alerted: dict[tuple[str, str], float] = {}
        # Symbols seen holding an open trade last time upkeep looked. Anything
        # that drops out of this set has gone flat, which is what releases its
        # alert throttle - the DB stores no close timestamp to ask for.
        self._open_symbols: set[str] = set()
        self._pending_breaks = PendingBreakWatcher(
            bitget, storage, bot, executor, self._bar_cache,
            max_total_risk_pct, reward_risk_ratio,
            lambda tag: self.auto_executes(tag), lambda symbol: self._symbol_max_leverage(symbol),
        )
        self._health = PositionHealthMonitor(bitget, storage, bot, ledger_expectations)
        self._dispatcher = SignalDispatcher(
            bitget, storage, bot, executor,
            self._exits, self._pending_breaks, self._lifecycle,
            risk_pct=risk_pct,
            reward_risk_ratio=reward_risk_ratio,
            max_total_risk_pct=max_total_risk_pct,
            swing_tags=swing_tags,
            max_swing_slots=max_swing_slots,
            send_chart_images=send_chart_images,
            already_exposed=lambda symbol: self.already_exposed(symbol),
            symbol_max_leverage=lambda symbol: self._symbol_max_leverage(symbol),
            auto_executes=lambda tag: self.auto_executes(tag),
            mark_alerted=lambda key: self._mark_alerted(key),
            plan_position=plan_position,
            round_trip_fee_for=round_trip_fee_for,
        )
        self.auto_execute_tags = auto_execute_tags or set()
        self.legacy_exit_tags = legacy_exit_tags or set()
        self.send_chart_images = send_chart_images
        # Runtime kill switch, flipped by /pause and /resume. Separate from the
        # whitelist so stopping execution never means losing signals: alerts
        # keep arriving and can still be placed by hand.
        self.execution_paused = False

    def auto_executes(self, strategy_tag: str) -> bool:
        return not self.execution_paused and strategy_tag in self.auto_execute_tags

    def required_timeframes(self) -> set[str]:
        """What the regular scan fetches for every symbol.

        Armed timeframes are excluded deliberately: they would otherwise set
        the scan cadence for the whole watchlist, which is exactly what the
        armed mechanism exists to avoid.
        """
        timeframes: set[str] = set(CONFLUENCE_TIMEFRAMES)
        for strategy in self.strategies:
            timeframes.update(tf for tf in strategy.timeframes if tf not in strategy.armed_timeframes)
        return timeframes

    def armed_timeframes(self) -> set[str]:
        return {tf for strategy in self.strategies for tf in strategy.armed_timeframes}

    async def run_forever(self) -> None:
        # THE PROCESS CAME UP. The scan heartbeat cannot record this: a restart
        # inside a sleep window misses no scan and correctly shows no gap, so
        # without its own row a bounce would be invisible in a report whose job
        # is to say the bot was down.
        try:
            self.storage.record_service_start(time.time())
        except Exception:
            logger.exception("Could not record the service start; running anyway")

        timeframes = self.required_timeframes()
        if not timeframes:
            logger.warning("No strategies registered; scanner has nothing to do")
            return

        loops = [self._scan_loop(timeframes), self._pending_break_loop(), self._position_upkeep_loop()]
        if self.armed_timeframes():
            loops.append(self._armed_loop())
        await asyncio.gather(*loops)

    async def _scan_loop(self, timeframes: set[str]) -> None:
        while True:
            scan_tf = min(timeframes, key=seconds_until_next_close)
            delay = seconds_until_next_close(scan_tf)
            logger.info("Next scan (driven by %s) in %.0fs", scan_tf, delay)
            # A HEARTBEAT, carrying when the bot expects to be back. The
            # capability ledger stores each capability's LATEST success, which
            # can never answer "was it down at any point" - a gap that has
            # since recovered leaves no trace in a last-seen timestamp. This
            # writes a row per cycle so the weekly review can find gaps after
            # the fact.
            #
            # due_at is recorded rather than assumed because the cadence is not
            # fixed: it is whichever timeframe closes next, so an ordinary wait
            # for a 4H close is hours long. A gap is only a gap against what
            # the bot itself said it would do.
            try:
                self.storage.record_heartbeat(time.time(), time.time() + delay)
            except Exception:
                # Availability bookkeeping must never end a scan. The whole
                # point of the try/except below is that one bad cycle does not
                # take down the process; this must not become the exception it
                # was built to prevent.
                logger.exception("Could not record the scan heartbeat; scanning anyway")
            await asyncio.sleep(delay)
            try:
                await self.tick()
            except Exception:
                # ONE BAD SCAN MUST NOT END THE PROCESS. It did: run_forever
                # gathers this loop with the others, so anything escaping tick()
                # took down the whole bot, and systemd's Restart=always brought
                # it back with every piece of in-memory state gone.
                #
                # Measured on 2026-08-18: the service restarted every 14-16
                # minutes for hours - restart counter 11 by 18:35 - every time
                # on `telegram.error.TimedOut` raised by send_signal on an
                # e2-micro's network. That is the mechanism behind the duplicate
                # MMTUSDT alert: the dedupe set and the alert throttle were both
                # process-local, and the process kept dying between scans.
                #
                # The other two loops have always guarded their bodies this way;
                # this one was simply missed.
                logger.exception("Scan failed; continuing to the next cycle")

    def _session_allows(self, symbol: str, strategy: Strategy) -> bool:
        """Whether this strategy may fire on this symbol right now.

        Only intraday strategies opt in. Failing open on a specs lookup error
        is deliberate: the gate exists to avoid trading a shut market, and a
        transient API failure is not evidence the market is shut - silently
        muting the whole watchlist on one bad response would be worse than the
        occasional out-of-hours signal it prevents.
        """
        if not strategy.session_gated:
            return True
        try:
            is_rwa = bool(self.bitget.get_contract_specs(symbol).get("is_rwa"))
        except Exception:
            logger.exception("Could not read contract specs for %s; not session-gating it", symbol)
            return True
        return sessions.may_signal_now(symbol, is_rwa)

    async def _position_upkeep_loop(self) -> None:
        """Position hygiene: trail what the bot manages, and report what it does
        not. Both ask "what is actually open right now", and a swing low only
        changes when a bar closes, so they share a cadence.

        THE CADENCE IS THE FASTEST TIMEFRAME ACTUALLY BEING TRAILED, not a fixed
        hour. A stop that follows 15m structure on an hourly clock can only
        ratchet once per four bars, so a runner gives back up to three bars of
        move before the stop follows it - and Strategy 2.1's 15m instance is the
        only thing this trails today.

        Falls back to TRAILING_POLL_TIMEFRAME when nothing is open, which is
        also what bounds the cost: the fast cadence exists only while a fast
        trade does, rather than querying an idle account every 15 minutes.

        The untracked check runs FIRST and on its own try/except: it is the
        one that tells Dror something is wrong, so a failure in trailing must
        not be able to silence it.
        """
        while True:
            try:
                await self.poll_untracked_positions()
            except Exception:
                logger.exception("Untracked-position check failed; continuing")
            try:
                await self.poll_weekly_report_overdue()
            except Exception:
                logger.exception("Weekly-report staleness check failed; continuing")
            try:
                await self.poll_capability_silence()
            except Exception:
                logger.exception("Capability-silence check failed; continuing")
            try:
                await self.poll_balance_divergence()
            except Exception:
                logger.exception("Balance-divergence check failed; continuing")
            try:
                await self.poll_trailing_stops()
            except Exception:
                logger.exception("Trailing-stop poll failed; continuing")
            try:
                self.release_closed_symbols()
            except Exception:
                logger.exception("Releasing alert throttles failed; continuing")
            await asyncio.sleep(seconds_until_next_close(self.upkeep_timeframe()))

    async def _pending_break_loop(self) -> None:
        while True:
            await asyncio.sleep(seconds_until_next_close(PENDING_BREAK_TIMEFRAME))
            try:
                await self.poll_pending_breaks()
            except Exception:
                logger.exception("Pending-break poll failed; continuing")

    async def poll_pending_breaks(self) -> None:
        await self._pending_breaks.poll()

    async def _armed_loop(self) -> None:
        """Polls only the symbols the regular scan armed, at the armed
        timeframe's own cadence."""
        while True:
            tf = min(self.armed_timeframes(), key=seconds_until_next_close)
            await asyncio.sleep(seconds_until_next_close(tf))
            try:
                await self.poll_armed()
            except Exception:
                logger.exception("Armed poll failed; continuing")

    async def tick(self, timeframes: set[str] | None = None) -> None:
        timeframes = timeframes if timeframes is not None else self.required_timeframes()

        try:
            equity = self.bitget.get_account_equity()
        except Exception:
            # Sizing off a stale or guessed equity silently corrupts every
            # downstream number, so skip the scan instead.
            logger.exception("Could not read account equity; skipping this scan")
            return

        logger.info("Scanning %d symbols on %s (equity %.2f)", len(self.watchlist), ",".join(sorted(timeframes)), equity)

        armed: dict[str, set[str]] = {}

        for symbol in self.watchlist:
            # Fetched with the forming candle included, then trimmed per
            # strategy: most want closed bars only, but one reading a slow
            # trend off a longer timeframe needs the hour in progress rather
            # than a picture up to a full candle stale. One fetch serves both.
            bars_by_tf: dict[str, pd.DataFrame] = {}
            for tf in timeframes:
                try:
                    ref = _split_reference_key(tf)
                    bars_by_tf[tf] = self._bars(*ref) if ref else self._bars(symbol, tf)
                except Exception:
                    logger.exception("Skipping %s this scan: failed to fetch/parse %s candles", symbol, tf)
                    bars_by_tf = None
                    break

            if not bars_by_tf or any(len(b) < 2 for b in bars_by_tf.values()):
                continue

            produced: list[tuple[Strategy, Signal]] = []
            for strategy in self.strategies:
                strategy_bars = {
                    tf: (bars_by_tf[tf] if tf in strategy.forming_bar_timeframes else bars_by_tf[tf].iloc[:-1])
                    for tf in strategy.timeframes
                    if tf in bars_by_tf
                }

                # A strategy whose trigger lives on an armed timeframe can't be
                # evaluated here - that data isn't fetched for the watchlist at
                # large. All this scan decides is whether the symbol is close
                # enough to be worth polling, recomputed from scratch every
                # time so a dead setup simply stops being armed.
                if not self._session_allows(symbol, strategy):
                    continue  # market shut: don't arm it and don't evaluate it

                if strategy.armed_timeframes:
                    try:
                        if strategy.arms(symbol, strategy_bars):
                            armed.setdefault(strategy.tag, set()).add(symbol)
                    except Exception:
                        logger.exception("Arming check failed for %s/%s", symbol, strategy.tag)
                    continue

                if len(strategy_bars) < len(strategy.timeframes):
                    continue  # one of this strategy's timeframes failed to fetch this scan

                try:
                    signal = strategy.evaluate(symbol, strategy_bars)
                except Exception:
                    logger.exception("Skipping %s/%s this scan: strategy raised", symbol, strategy.tag)
                    continue

                if signal is None:
                    continue

                produced.append((strategy, signal))

            # Resolved per SYMBOL, after every strategy has been asked, because
            # a signal cannot be compared with one that has not been produced
            # yet and strategy order is not something to depend on.
            for strategy, signal in _drop_superseded(produced):
                await self._handle_signal(signal, strategy, equity, bars_by_tf)

        self._armed = armed

    async def _handle_signal(self, signal: Signal, strategy: Strategy, equity: float, bars_by_tf: dict) -> None:
        # Before dedupe, because the question this answers is "is this instance
        # producing setups at all". Two of the nine live instances are still
        # unbacktested - they need 15m/5m bars no cache holds yet. That is a
        # missing fetch, NOT a missing exchange: "Bitget serves 22 days of 15m,
        # ~2 days of 5m" was one get_candles result written down as a property
        # of the exchange, and history-candles pages back to a symbol's listing
        # date (BTCUSDT 5m reaches 2019-07-10, measured 2026-08-17). Strategy
        # 3's 5m instance also arms only when the daily close sits in the top
        # 10% of its range - last measured, never. An instance that can never
        # fire looks exactly like a market with no setups, which is the same
        # blindness that hid the take-profit for five months.
        ledger.try_record(self.storage.db_path, ledger.signal_seen(signal.strategy_tag))

        # Keyed on the trade being proposed, not on the candle that produced
        # it. A per-candle key still re-alerts every time the trigger re-fires
        # against an unchanged leg: one stale TSLAUSDT short went out four
        # times over eleven hours, same entry, same stop, while price walked 5
        # points past that stop. Identical levels mean it is the same trade,
        # however often it retriggers.
        # A strategy whose setup is a LEVEL rather than a price supplies its
        # own key: keying on entry/stop makes every re-cross of that level
        # look like a fresh trade.
        dedupe_key = signal.dedupe_key or (
            signal.symbol,
            signal.strategy_tag,
            signal.direction,
            signal.entry_price,
            signal.stop_loss,
        )
        if dedupe_key in self._seen:
            return
        self._seen[dedupe_key] = None

        if self._throttled(signal):
            return
        self._prune_seen()

        # Confirmation is read from closed bars on every confluence
        # timeframe, independently of what this strategy itself looks at.
        confirming = {tf: bars_by_tf[tf].iloc[:-1] for tf in CONFLUENCE_TIMEFRAMES if tf in bars_by_tf}
        try:
            confluence = patterns.confluence(confirming, signal.direction)
        except Exception:
            logger.exception("Pattern detection failed for %s; treating as no confluence", signal.symbol)
            confluence = None

        # The structure sitting UNBROKEN in front of price, as opposed to the
        # one that already broke. This is what the staged entry waits on: the
        # trade goes on at base risk now, and the pattern only earns more once
        # it actually resolves. Detected separately and never gates the signal.
        try:
            pending_pattern = patterns.pending(confirming, signal.direction)
        except Exception:
            logger.exception("Pending-pattern detection failed for %s; continuing without it", signal.symbol)
            pending_pattern = None

        await self._dispatch(
            signal, equity, signal.analysis_timeframes or strategy.timeframes, confluence, pending_pattern,
            strategy=strategy, bars_by_tf=bars_by_tf,
        )

    async def poll_armed(self) -> None:
        """Evaluate armed strategies against their armed timeframe, for the
        symbols the last regular scan marked - a few, not the watchlist."""
        if not self._armed:
            return

        try:
            equity = self.bitget.get_account_equity()
        except Exception:
            logger.exception("Could not read account equity; skipping armed poll")
            return

        by_tag = {strategy.tag: strategy for strategy in self.strategies}
        logger.info("Polling %d armed symbol(s)", sum(len(s) for s in self._armed.values()))

        for tag, symbols in list(self._armed.items()):
            strategy = by_tag.get(tag)
            if strategy is None:
                continue
            for symbol in sorted(symbols):
                bars_by_tf: dict[str, pd.DataFrame] = {}
                try:
                    for tf in (*strategy.timeframes, *CONFLUENCE_TIMEFRAMES):
                        if tf not in bars_by_tf:
                            ref = _split_reference_key(tf)
                            bars_by_tf[tf] = self._bars(*ref) if ref else self._bars(symbol, tf)
                except Exception:
                    logger.exception("Skipping armed %s this poll: failed to fetch candles", symbol)
                    continue

                if any(len(bars_by_tf[tf]) < 2 for tf in strategy.timeframes):
                    continue

                # Re-checked here as well as at arming: a symbol armed during
                # the session is polled every 5 minutes, and would otherwise
                # keep triggering for hours after its market closed.
                if not self._session_allows(symbol, strategy):
                    continue

                strategy_bars = {
                    tf: (bars_by_tf[tf] if tf in strategy.forming_bar_timeframes else bars_by_tf[tf].iloc[:-1])
                    for tf in strategy.timeframes
                }
                try:
                    signal = strategy.evaluate(symbol, strategy_bars)
                except Exception:
                    logger.exception("Skipping %s/%s this poll: strategy raised", symbol, tag)
                    continue

                if signal is not None:
                    await self._handle_signal(signal, strategy, equity, bars_by_tf)

    def _symbol_max_leverage(self, symbol: str) -> float:
        """The account ceiling, or the symbol's own if the exchange caps it lower.

        Bitget's maxLever is per contract and 17 of 759 sit below the 10x
        MIN_LEVERAGE floor in risk_sizing, so a single global ceiling makes
        those symbols unplaceable: the plan asks for 10x, the exchange answers
        40797 and the executor stops before placing any leg. BTWUSDT (5x) did
        exactly that on 2026-08-21.

        Read here rather than at execution time on purpose - capping leverage
        RAISES the margin a position needs, so it has to be known while the
        trade is being sized. Capping it after the fact would commit less
        margin than the position actually consumes.
        """
        try:
            cap = float(self.bitget.get_contract_specs(symbol).get("max_leverage") or 0)
        except Exception:
            logger.exception("Could not read %s's leverage ceiling; using the account default", symbol)
            return self.max_leverage
        return min(self.max_leverage, cap) if cap > 0 else self.max_leverage

    def _bars(self, symbol: str, timeframe: str, now: float | None = None) -> pd.DataFrame:
        """Bars for this symbol and timeframe - see notifier.bar_cache.BarCache."""
        return self._bar_cache.get(symbol, timeframe, now)

    def _prune_seen(self, max_entries: int = 5000) -> None:
        """Bounded so a long-running process can't leak memory on a big watchlist.

        Drops the OLDEST half. This used to call list() on a set, which has no
        order, so it kept an arbitrary half - and could throw away the keys
        added seconds earlier while retaining ones from days back. The whole
        point of the set is to stop a signal re-alerting, so discarding the
        newest entries defeats it exactly when it matters. `_seen` is an
        insertion-ordered dict now, which makes "oldest" mean something.
        """
        if len(self._seen) > max_entries:
            keep = list(self._seen)[-(max_entries // 2):]
            self._seen = dict.fromkeys(keep)

    async def _dispatch(
        self,
        signal: Signal,
        equity: float,
        timeframes: list[str],
        confluence: str | None = None,
        pending_pattern: tuple | None = None,
        *,
        strategy: Strategy | None = None,
        bars_by_tf: dict | None = None,
    ) -> None:
        await self._dispatcher.dispatch(
            signal, equity, timeframes, confluence, pending_pattern,
            strategy=strategy, bars_by_tf=bars_by_tf,
        )

    async def _confirm_and_track(
        self,
        trade_id: int,
        signal: Signal,
        plan=None,
        order=None,
        remainder_target: float | None = None,
    ) -> None:
        await self._dispatcher.confirm_and_track(trade_id, signal, plan, order, remainder_target)

    async def poll_untracked_positions(self) -> None:
        await self._health.poll_untracked_positions()

    async def poll_weekly_report_overdue(self) -> None:
        await self._health.poll_weekly_report_overdue()

    async def poll_capability_silence(self) -> None:
        await self._health.poll_capability_silence()

    async def poll_balance_divergence(self) -> None:
        await self._health.poll_balance_divergence()

    def already_exposed(self, symbol: str) -> bool:
        return self._health.already_exposed(symbol)

    def upkeep_timeframe(self) -> str:
        return self._trailing.upkeep_timeframe()

    def trail_timeframe(self, strategy_tag: str) -> str:
        return self._trailing.trail_timeframe(strategy_tag)

    def trailing_stop(self, symbol: str, direction: str, strategy_tag: str, current_stop: float | None):
        return self._trailing.trailing_stop(symbol, direction, strategy_tag, current_stop)

    def stall_tighten(self, trade, price: float, stop: float | None = None) -> float | None:
        return self._trailing.stall_tighten(trade, price, stop)

    async def poll_trailing_stops(self) -> None:
        await self._trailing.poll()

    def release_closed_symbols(self) -> None:
        """Clear the alert throttle for any symbol that has gone flat.

        Observed rather than queried: the trades table has no close timestamp,
        so the transition is caught by comparing which symbols hold open trades
        now against which did last time. Run from the upkeep loop, which
        already asks "what is actually open right now".
        """
        now_open = {t.סימבול for t in self.storage.open_trades()}
        for symbol in self._open_symbols - now_open:
            for key in [k for k in self._alerted if k[0] == symbol]:
                del self._alerted[key]
            self.storage.clear_alert_throttle(symbol)
            logger.info("%s went flat; its alert throttle is released", symbol)
        self._open_symbols = now_open

    def _throttled(self, signal: Signal) -> bool:
        """Whether this alert is suppressed as a repeat of one already sent.

        AT MOST ONE PROMPT PER SYMBOL PER INSTANCE PER ROLLING DAY.

        Dedupe already collapses one unbroken EMA9 run into a single trade, but
        a symbol can produce run after run: Strategy 2.1 fires ~166 times a day
        across the watchlist against Strategy 1's ~17, and every one of those is
        an Approve/Reject prompt needing an answer. Nothing executes without
        Dror pressing Approve, so the scarce resource is his attention, and an
        alert stream too noisy to read is a silent failure - the same shape as
        the thirteen NEVER lines on a fresh ledger and the five-minute Telegram
        expiry.

        Per INSTANCE, not per symbol: the same pullback seen on 15m and on 4H
        are different trades with different stops, and collapsing them would
        hide which timeframe found it.

        ROLLING, not midnight: a boundary would let a 23:50 signal silence the
        whole of the next day.

        RESET WHEN THE POSITION CLOSES, because one-position-per-symbol means
        the throttle and the position overlap - once a symbol is tradeable
        again it should be able to ask. The trades table records no close TIME
        (תאריך/שעת_כניסה are the entry's), so the release is observed rather
        than queried: see release_closed_symbols.
        """
        key = (signal.symbol, signal.strategy_tag)
        # The in-memory map is a cache in front of the table, not the record.
        # It was the record until 2026-08-19, and being a plain dict on this
        # object it emptied on every restart - so a deploy or a crash silently
        # released every throttle at once. MMTUSDT went out twice inside one 4H
        # candle that way: identical stop, identical target, identical
        # breakeven, only the quoted market price differing.
        last = self._alerted.get(key)
        if last is None:
            try:
                last = self.storage.last_alerted(*key)
            except Exception:
                # A throttle that cannot be read must not silence the symbol -
                # the same call _may_signal_now makes about session data.
                logger.exception("Could not read %s's alert throttle; allowing the prompt", key)
                last = None
            if last is not None:
                self._alerted[key] = last

        if last is None or time.time() - last >= ALERT_THROTTLE_SECONDS:
            # NOT recorded here. A prompt that is never sent must not spend the
            # day's allowance: _dispatch still has a dozen ways to return - the
            # fill guard, the risk cap, the swing slots, an exchange minimum -
            # and the Telegram send itself can fail. Recording on the intention
            # to ask rather than on the asking silenced symbols nobody had been
            # asked about, and now that the throttle is durable that would last
            # a full day rather than until the next restart.
            #
            # _seen still holds this setup for the life of the process, so
            # nothing re-prompts on the very next scan; only a genuinely new
            # setup can ask again.
            return False

        logger.info(
            "%s %s on %s throttled: already prompted %.1f h ago",
            signal.strategy_tag, signal.direction, signal.symbol,
            (time.time() - last) / 3600,
        )
        return True

    def _mark_alerted(self, key: tuple[str, str]) -> None:
        """Record a prompt in both the cache and the table.

        A failed write leaves the in-memory entry standing, so the throttle
        still holds for this process and only a restart can lose it - strictly
        better than the behaviour this replaces, where every restart did.
        """
        now = time.time()
        self._alerted[key] = now
        try:
            self.storage.record_alerted(*key, now)
        except Exception:
            logger.exception("Could not persist %s's alert throttle", key)

    def manages_exits(self, strategy_tag: str) -> bool:
        return self._exits.manages_exits(strategy_tag)

    def runner_target(self, signal: Signal, fallback: float | None) -> tuple[float | None, str]:
        return self._exits.runner_target(signal, fallback)

    async def place_runner_target(
        self, signal: Signal, fallback: float | None, managed: bool | None = None, notify: bool = True
    ) -> str:
        return await self._exits.place_runner_target(signal, fallback, managed, notify)

    async def _on_partial_manage_exits(
        self, signal: Signal, fallback: float | None, breakeven: float | None,
        managed: bool = True, notify: bool = True,
    ) -> list[str]:
        return await self._exits.on_partial_manage_exits(signal, fallback, breakeven, managed, notify)

    async def _move_stop_to_breakeven(self, signal: Signal, breakeven: float, notify: bool = True) -> str:
        return await self._exits.move_stop_to_breakeven(signal, breakeven, notify)

    def _cancel_superseded_stops(self, symbol: str, direction: str, breakeven: float) -> None:
        self._exits._cancel_superseded_stops(symbol, direction, breakeven)

    def _place_reduce_only(self, symbol: str, direction: str, size: float, price: float, kind: str) -> None:
        self._exits._place_reduce_only(symbol, direction, size, price, kind)

    async def _place_partial(self, signal: Signal, plan, position_size: float, replace: bool = False) -> None:
        await self._exits.place_partial(signal, plan, position_size, replace)

    def _cancel_resting(self, symbol: str, reduce_only_only: bool = False, direction: str | None = None) -> None:
        self._exits.cancel_resting(symbol, reduce_only_only, direction)

    def _safe_stop_target(self, symbol: str, direction: str, position: dict):
        return self._exits.safe_stop_target(symbol, direction, position)

    def _on_trade_closed(self, trade_id: int, price: float) -> None:
        self._lifecycle.on_trade_closed(trade_id, price)

    def _on_scale_in(self, trade_id: int) -> None:
        self._lifecycle.on_scale_in(trade_id)

    def _on_resize(self, trade_id: int, size: float) -> None:
        """Grows a resumed trade's take-profit to match a limit leg that just
        filled - the resize a freshly-tracked trade gets for free from the
        on_resize closure at SignalDispatcher's own track_position call,
        which only exists in that coroutine and dies with it. A tracker
        re-attached by resume_open_trades after a restart has no signal/plan
        to call it with, so without this it does nothing: _on_scale_in still
        reports "take-profit covers X of Y", but nothing ever closes the gap.

        DOGEUSDT and QQQUSDT both hit exactly this on 2026-09-03 - a deploy
        restarted the service while their limit legs were still resting, the
        legs filled after the restart, and the TP order stayed sized to the
        market leg alone.

        Gated the same way the ORIGINAL partial was: only a trade whose
        entries the bot actually executes gets its take-profit replaced here.
        _manages_trade alone is not enough - Strategy 3 manages exits while
        entering (and placing its own first partial) by hand, and this must
        not cancel and replace an order Dror placed himself.

        THE SAME GATE STILL MISSED A CASE: trade #98, 1000RATSUSDT, live
        2026-09-05. Its tag was live when the position opened, then retired
        into LEGACY_EXIT_TAGS by the 2026-09-03 market-entry switch - and its
        resting limit leg filled two days later, under the RETIRED tag.
        auto_executes(tag) and executor.handles_live(tag) both read the tag's
        CURRENT routing, which says nothing about whether the BOT placed this
        specific trade's resting limit leg back when it opened - a fact about
        history that does not change when a tag is later retired from new
        entries. legacy_exit_tags is that fact, made explicit: a tag only
        lands there by having been promoted to live and later demoted (see
        notifier.main's own comment on LEGACY_EXIT_TAGS) - never for a tag
        that has merely been GRANTED exit rights without ever going live,
        which is exactly the Strategy 3 shape the check above still has to
        keep excluding.
        """
        trade = self.storage.get_trade(trade_id)
        tag = trade.תגית_אסטרטגיה or ""
        if not (
            (self.auto_executes(tag) and self.executor.handles_live(tag))
            or tag in self.legacy_exit_tags
        ):
            return
        signal = self._exit_plan_signal(trade)
        if signal is None:
            return

        # Recompute the price too, not just resize the quantity - trade #98,
        # 1000RATSUSDT, live 2026-09-05: its partial was correctly priced at
        # 2.0R off the MARKET LEG's own fill (all that was known at confirm
        # time), and the resting limit leg later filled at a WORSE price,
        # pulling the true blended entry further from the stop - the exact
        # same drift notifier.signal_dispatcher's own real_entry recompute
        # already fixes ONCE, at confirm time (see its docstring: EULUSDT
        # +4.6%, DEXEUSDT -5.5%, ...). That fix never covered the case where
        # the entry keeps moving AFTER confirmation, because a limit leg can
        # fill days later - which is exactly when on_resize fires.
        #
        # Only when the ratio is actually known (reward_risk_ratio survived
        # the rebuild - see exit_plan_signal) and the entry has too. Neither
        # holds for a trade that predates the reward_risk_ratio column or was
        # adopted via /manage with a hand-typed level - those fall all the
        # way back to reusing whatever price was already stored, exactly as
        # before this recompute existed, rather than guess with an unrelated
        # default (Scanner's OWN reward_risk_ratio, 3.0, is not the 2.0
        # Strategy 1's first tier actually uses - using it here would have
        # replaced one wrong price with a different wrong one).
        real_entry = signal.entry_price
        ratio = signal.reward_risk_ratio
        if real_entry is not None and ratio is not None:
            target = _reward_target(real_entry, signal.stop_loss, signal.direction, ratio)
            self.storage.update_actual_stop_target(trade_id, signal.stop_loss, target)
            # The runner's OWN fallback target shares the identical drift -
            # same confirm-time recompute, same market-leg-only entry - and
            # if only the resting partial order is corrected here,
            # place_runner_target() would later read this UNCORRECTED value
            # back out of the row and place the runner wrong too. Only for a
            # strategy that does not manage its own remainder
            # (partial_fraction is None) - a self-managing strategy's own
            # absolute level (Strategy 3's daily line) is a decision, not an
            # estimate to correct, the same gate the confirm-time code uses.
            if signal.partial_fraction is None:
                remainder = _reward_target(real_entry, signal.stop_loss, signal.direction,
                                           REMAINDER_TARGET_RATIO)
                self.storage.set_exit_plan(
                    trade_id, trade.breakeven_stop, remainder, trade.partial_fraction,
                    bool(trade.runner_target_is_final), reward_risk_ratio=ratio,
                )
        else:
            target = trade.יעד_רווח_בפועל or trade.יעד_רווח_מקורי

        if target is None:
            return
        asyncio.create_task(
            self._place_partial(signal, SimpleNamespace(take_profit=target), size, replace=True)
        )

    def _on_partial_exit(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        self._lifecycle.on_partial_exit(trade_id, closed_size, realized_pnl)

    def _manages_trade(self, trade) -> bool:
        return self._lifecycle.manages_trade(trade)

    async def adopt_trade(self, trade_id: int, breakeven: float, runner_target: float | None = None) -> str:
        return await self._lifecycle.adopt_trade(trade_id, breakeven, runner_target)

    def _exit_plan_signal(self, trade) -> Signal | None:
        return self._lifecycle.exit_plan_signal(trade)
