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

import pandas as pd

from core import ledger
from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.executor import Executor, OrderLeg, TradeOrder
from execution.tracker import (
    breakeven_price,
    check_position_now,
    closing_exits,
    format_close_message,
    format_partial_message,
    format_scale_in_message,
    take_profit_coverage,
    track_position,
    wait_for_signal_position,
)
from notifier import chart, sessions
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
# Relative tolerance for deciding two price levels are the same one. A strategy
# whose own reward:risk already equals REMAINDER_TARGET_RATIO puts both exit
# tiers on the identical price, and describing that as a partial take plus a
# stop-to-breakeven is describing steps that cannot happen.
_PRICE_EPSILON = 1e-9
# A "remainder" this small is float noise from position_size x 1.0, not a
# tranche anyone can close.
_SIZE_EPSILON = 1e-12


def _reward_target(entry_price: float, stop_loss: float, direction: str, ratio: float) -> float:
    risk_per_unit = abs(entry_price - stop_loss)
    return entry_price + risk_per_unit * ratio if direction == "long" else entry_price - risk_per_unit * ratio


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


def _build_order(signal: Signal, plan, market_price: float) -> TradeOrder:
    """The trade as orders: the legs the alert describes, plus the stop.

    A split entry is two legs at two prices, so it is placed as two orders -
    not one averaged order, which would fill the whole position at market and
    discard the better price the limit leg exists to get.
    """
    if signal.limit_entry is None:
        legs = [OrderLeg(size=plan.position_size, order_type="market")]
    else:
        market_size = plan.position_size * signal.market_fraction
        limit_size = plan.position_size - market_size
        legs = []
        if market_size > 0:
            legs.append(OrderLeg(size=market_size, order_type="market"))
        legs.append(
            OrderLeg(size=limit_size, order_type="limit", price=signal.limit_entry, note=signal.limit_note)
        )

    return TradeOrder(
        symbol=signal.symbol,
        direction=signal.direction,
        legs=legs,
        stop_loss=signal.stop_loss,
        leverage=plan.leverage,
        strategy_tag=signal.strategy_tag,
    )




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
        self._health = PositionHealthMonitor(bitget, storage, bot)
        self.auto_execute_tags = auto_execute_tags or set()
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
        if self.already_exposed(signal.symbol):
            return  # already in this symbol; one at a time

        reward_risk_ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        # A chart pattern no longer raises risk merely by existing. It used to:
        # confluence() accepts a breakout up to CONFLUENCE_BARS old, so TRXUSDT
        # was sized at 2% citing a wedge that had broken 17 hours earlier while
        # the structure actually in front of price was an unresolved flag that
        # could still have broken down. Risk is staged instead - the trade goes
        # on at base risk here, and the pattern earns its second increment only
        # when it actually breaks. A strategy's OWN tier (Strategy 2 reading a
        # second timeframe's stack) is untouched: that is a different kind of
        # evidence and it is confirmed at signal time, not pending.
        risk_pct = signal.risk_pct_override if signal.risk_pct_override is not None else self.risk_pct
        available_budget = equity - self.storage.committed_margin()

        # The headline Entry is where the market is right now, so the alert can
        # be read against the chart at a glance. It is also needed *before*
        # sizing, because a split entry's real cost basis includes the market
        # leg (see plan_entry below).
        try:
            market_price = self.bitget.get_mark_price(signal.symbol)
        except Exception:
            logger.exception("Could not read mark price for %s; showing the planned entry", signal.symbol)
            market_price = signal.entry_price

        # A split entry holds both legs, so the position's actual cost basis is
        # their weighted average - not the limit level alone. Sizing off the
        # limit understated risk on 91% of replayed Strategy 1 signals, by a
        # median of 29% and up to 60%, which quietly pushed a 2%-risk trade to
        # over 3% of equity. This is also the only price at which "move the
        # stop to breakeven" is actually breakeven.
        if signal.limit_entry is not None and signal.market_fraction > 0:
            plan_entry = signal.market_fraction * market_price + (1 - signal.market_fraction) * signal.limit_entry
        elif signal.limit_entry is None and signal.market_fraction >= 1.0:
            # A pure market entry fills at MARKET, not at the level that
            # selected it. The branch above covers a split entry; this one was
            # missing, so a strategy whose entry_price is a REFERENCE rather
            # than an expected fill was sized against a price it never gets.
            #
            # Strategy 2.1 is exactly that: entry_price is e9_prev, the EMA9,
            # while ENTRY_MODE="next_open" enters at market on the candle AFTER
            # the rejection - which by construction has closed back on the trend
            # side, so the fill is always on the far side of the EMA9. Measured
            # over 591 setups on WLD/SOL/PEPE, the real distance to the stop is
            # 1.89x the distance sized against (median 1.44x, p90 2.97x), and
            # worse on 100% of them. A trade meant to risk 1% risked ~1.9%, and
            # 25% of them breached the 2% cap outright.
            #
            # The drift guard does not catch it: gap_at_dispatch is subtracted
            # before drift is counted, which is right for a resting limit - the
            # gap is distance the order waits to cross - and wrong for a market
            # order, where the same gap is simply the wrong sizing basis.
            #
            # Strategy 1 blends both legs above; Strategy 4 is pure limit and
            # its entry_price IS the limit; Strategy 3 sets entry = close_now.
            # 2.1 is the only instance that lands here.
            plan_entry = market_price
        else:
            plan_entry = signal.entry_price

        try:
            plan = plan_position(
                equity=equity,
                risk_pct=risk_pct,
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                direction=signal.direction,
                reward_risk_ratio=reward_risk_ratio,
                available_budget=available_budget,
                max_leverage=self._symbol_max_leverage(signal.symbol),
                # The fee sizing is meant to absorb depends on how much of THIS
                # signal's entry is market vs limit - a flat assumption is
                # right for Strategy 4 (all limit) and close enough for a 20%
                # split, but Strategy 2.1 enters 100% at market under
                # ENTRY_MODE="next_open" and its true round-trip fee is taker
                # both legs, 50% more than the flat default. See
                # round_trip_fee_for's own docstring.
                round_trip_fee_pct=round_trip_fee_for(signal.market_fraction),
            )
        except ValueError as exc:
            logger.info("Skipping %s/%s: %s", signal.symbol, signal.strategy_tag, exc)
            return

        # THE GATES, RE-ASKED AGAINST THE PRICE THE TRADE ACTUALLY FILLS AT.
        #
        # A strategy decides from a LEVEL, and for most of them that level is
        # also the fill, so the two questions have one answer. Strategy 2.1
        # breaks that: its entry_price is the EMA9 that selected the setup while
        # the order goes in at market on the candle after the rejection, which
        # has closed back on the trend side by construction - so the fill is
        # always on the far side of the level.
        #
        # HYPEUSDT, 2026-08-18: 1.50% stop measured against its EMA9, 0.15%
        # against plan_entry. It passed the strategy's own 0.30% floor while
        # failing it by half. Commit 60aa796 fixed the SIZING to use this price;
        # the gates that decide whether to trade at all were still asking about
        # the EMA9.
        #
        # Logged under its own decision rather than dropped, so a refusal is
        # visible in the weekly stats. A signal that vanishes silently is the
        # same failure mode as one that never fired.
        if signal.fill_guard is not None:
            refusal = signal.fill_guard.refuses(plan_entry, signal.stop_loss, reward_risk_ratio)
            if refusal is not None:
                logger.info(
                    "Skipping %s/%s at the fill: %s", signal.symbol, signal.strategy_tag, refusal
                )
                signal_id = self.storage.log_signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=plan_entry,
                    stop_loss=signal.stop_loss,
                    take_profit=plan.take_profit,
                    strategy_tag=signal.strategy_tag,
                    confluence=confluence,
                )
                self.storage.mark_signal_decision(signal_id, "refused_at_fill")
                return

        # The swing pool's own hard slot cap, enforced independently of and in
        # addition to the aggregate dollar cap below - two swing trades can
        # each be well under the dollar cap and still be the two the pool
        # allows, at which point a third swing signal is suppressed outright
        # regardless of how much risk budget remains.
        if signal.strategy_tag in self.swing_tags:
            swing_open = sum(
                1
                for t in (*self.storage.pending_trades(), *self.storage.open_trades())
                if t.תגית_אסטרטגיה in self.swing_tags
            )
            if swing_open >= self.max_swing_slots:
                logger.info(
                    "Skipping %s/%s: swing slots full (%d/%d)",
                    signal.symbol,
                    signal.strategy_tag,
                    swing_open,
                    self.max_swing_slots,
                )
                signal_id = self.storage.log_signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=plan_entry,
                    stop_loss=signal.stop_loss,
                    take_profit=plan.take_profit,
                    strategy_tag=signal.strategy_tag,
                    confluence=confluence,
                )
                self.storage.mark_signal_decision(signal_id, "swing_slots_full")
                return

        risk_cap = equity * self.max_total_risk_pct
        open_risk = self.storage.total_open_risk()
        if open_risk + plan.risk_amount > risk_cap:
            logger.info(
                "Skipping %s/%s: total open risk %.2f + %.2f would exceed the %.0f%% cap (%.2f)",
                signal.symbol,
                signal.strategy_tag,
                open_risk,
                plan.risk_amount,
                self.max_total_risk_pct * 100,
                risk_cap,
            )
            return

        specs = self.bitget.get_contract_specs(signal.symbol)

        # Bitget enforces its minimum notional on EACH order it receives, not
        # on the position as a whole. A split entry is two separate orders, so
        # the market leg - always the smaller one, since market_fraction is
        # never above 0.5 in a strategy that splits - is what actually binds.
        # ADAUSDT cleared $6.35 total and was still rejected live, because its
        # 20% market leg alone was $1.05, under the $5 floor.
        #
        # Per Dror's call: don't fall back to a single non-split order, and
        # don't send the alert at all when the trade genuinely isn't
        # placeable at this size - but log it under its own decision so it
        # stays visible in the weekly stats rather than silently vanishing,
        # which is exactly what "the signals table only ever gains a row on
        # approval" used to do to rejected/ignored signals.
        too_small = plan.position_size < specs["min_size"] or plan.notional_value < specs["min_notional"]
        if not too_small and signal.limit_entry is not None and signal.market_fraction > 0:
            market_leg_notional = plan.position_size * signal.market_fraction * market_price
            too_small = market_leg_notional < specs["min_notional"]

        # The checks above compare dollars against the size the PLAN wants,
        # not the size that actually gets sent. round_size floors to the
        # exchange's own step, so a leg clears the minimum on paper and is then
        # rejected as the smaller quantity that leaves.
        #
        # Two live failures, one from each half of that:
        #   AAVEUSDT trades in units of 0.1. A 0.06-unit market leg was worth
        #   $6, over the $5 minimum, but rounding floored it to 0.
        #   CLUSDT rounds to 2dp. Its market leg of 0.064938 was worth $5.30,
        #   over the same minimum, but rounded to 0.06 it was worth $4.896 and
        #   Bitget refused it with "less than the minimum amount 5 USDT" -
        #   after the signal had been approved and the entry attempted.
        #
        # So each leg is now valued at the quantity that will really be sent,
        # AND at its own price: the market leg fills at market while the limit
        # leg rests at signal.limit_entry, and on a split entry those differ by
        # design - valuing both at one price is what makes a leg look bigger
        # than it is.
        if not too_small:
            legs = [(plan.position_size, market_price)]
            if signal.limit_entry is not None and signal.market_fraction > 0:
                market_size = plan.position_size * signal.market_fraction
                legs = [
                    (market_size, market_price),
                    (plan.position_size - market_size, signal.limit_entry),
                ]
            for size, price in legs:
                rounded = self.bitget.round_size(signal.symbol, size)
                if rounded <= 0 or rounded * price < specs["min_notional"]:
                    too_small = True
                    break

        if too_small:
            logger.info(
                "Not tradeable at this size: %s/%s (notional %.2f, exchange minimum %.2f)",
                signal.symbol,
                signal.strategy_tag,
                plan.notional_value,
                specs["min_notional"],
            )
            signal_id = self.storage.log_signal(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                take_profit=plan.take_profit,
                strategy_tag=signal.strategy_tag,
                confluence=confluence,
            )
            self.storage.mark_signal_decision(signal_id, "too_small")
            return

        # A strategy that sets partial_fraction manages its own two-tier exit;
        # everything else takes the scanner's 50% / 1:3 default.
        strategy_manages_exit = signal.partial_fraction is not None
        partial_fraction = signal.partial_fraction if strategy_manages_exit else PARTIAL_TAKE_FRACTION
        partial_size = plan.position_size * partial_fraction
        remainder_size = plan.position_size - partial_size
        remainder_target = _reward_target(plan_entry, signal.stop_loss, signal.direction, REMAINDER_TARGET_RATIO)

        # A pending pattern is only worth naming if this trade would live long
        # enough to see it break. Dror's rule: the break must fall between the
        # stop and the final target, because outside that window the trade has
        # already resolved - past the stop it is closed, past the target it is
        # closed - and the +1% the alert promises could never be offered.
        #
        # Deliberately NO threshold constant. The bound comes from levels the
        # trade already defines, which is why this could be settled without the
        # proximity number nobody had measured: a real run found 27 of 50
        # symbol-directions carrying a pending pattern, with break levels
        # ranging from +0.76% (BTCUSDT) to -13.11% (SOLUSDT). The far ones are
        # not slightly worse, they are unreachable, and they cost attention at
        # approval time to read and dismiss.
        #
        # Bounded by the FINAL target rather than the partial: the position is
        # still open through it, and the add-on's never-loosen stop rule
        # already handles a partial having pulled the stop to breakeven first.
        #
        # Applied here rather than inside pending() because R is only defined
        # once a pattern is attached to a signal that has an entry and a stop.
        if pending_pattern is not None:
            level = pending_pattern[0].break_level
            lo, hi = sorted((signal.stop_loss, remainder_target))
            if not lo < level < hi:
                logger.info(
                    "Pending %s on %s not reachable inside this trade (%s vs %s..%s); not quoting it",
                    pending_pattern[0].name, signal.symbol, level, lo, hi,
                )
                pending_pattern = None

        # Prices and sizes at the precision the exchange actually quotes, so
        # every level in the alert is a value that can be entered as an order.
        def px(value: float) -> str:
            return f"{value:.{specs['price_place']}f}"

        def qty(value: float) -> str:
            return f"{value:.{specs['volume_place']}f}"

        def usd(value: float) -> str:
            return f"${value:,.0f}" if abs(value) >= 10 else f"${value:,.2f}"

        # Logged before Approve/Reject is even seen, so a rejected or ignored
        # signal is still measurable - the trades table only ever gains a row
        # once a signal is both approved AND confirmed on Bitget, which left
        # rejected/ignored signals with no record anywhere. Moved ahead of
        # `lines` (it used to run after) so the id it returns can go in the
        # header instead of a separate trailing line - Dror: "make the signal
        # to be in this format is smaller then the corrent one".
        signal_id = self.storage.log_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=plan_entry,
            stop_loss=signal.stop_loss,
            take_profit=plan.take_profit,
            strategy_tag=signal.strategy_tag,
            confluence=confluence,
            # The whole Signal, so this alert can be offered again later by its
            # number if it expires on a setup Dror still likes. See
            # signal_to_json: the columns beside it cannot rebuild an exit.
            signal_json=signal_to_json(signal),
        )

        lines = [f"Signal #{signal_id}  {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})"]
        if signal.analysis_timeframes is not None:
            # Only worth a line when it says something the tag doesn't
            # already: analysis_timeframes is None for the common case (the
            # strategy's own fixed timeframes, which the tag already names),
            # and set only when a signal-specific read genuinely differs -
            # e.g. Strategy 2's second timeframe, listed only when it was a
            # real confirmation rather than a supportive trend read.
            lines.append(f"Analysis timeframe: {', '.join(timeframes)}")
        lines.extend([
            # "Entry" means the market price for a SPLIT entry, where neither
            # leg alone is the cost basis and the blend is what sizing uses.
            # A strategy with one leg at one price has a real entry, and
            # printing the market instead reads as a second, wrong number:
            # Dror on a Strategy 4 alert - "why the entry is different from the
            # limit there is only one entry in this strategy".
            (
                f"Entry: {px(plan_entry)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}"
                if signal.limit_entry is not None and signal.market_fraction == 0
                else f"Entry: {px(market_price)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}"
            ),
            f"Size: {usd(plan.notional_value)} ({qty(plan.position_size)} @ {plan.leverage:.1f}x)"
            + (f"  risk {risk_pct:.0%}" if risk_pct > self.risk_pct else ""),
        ])
        if confluence:
            lines.append(f"Confirmed by {confluence}")

        # The unbroken structure in front of price, stated with the level that
        # would confirm it and how far off that is, so the setup can be
        # validated against the chart at approval time rather than taken on
        # trust. For triangles and wedges the level sits on a converging line
        # and moves every bar, so its drift is stated too - quoting a number
        # that silently goes stale is exactly what made the old confluence
        # bump misleading.
        if pending_pattern is not None:
            pat, pat_tf = pending_pattern
            side = "above" if pat.direction == "long" else "below"
            gap = (pat.break_level - market_price) / market_price * 100 if market_price else 0.0
            drift = ""
            if pat.drift_per_bar:
                drift = f", {'rising' if pat.drift_per_bar > 0 else 'falling'} {abs(pat.drift_per_bar):.{specs['price_place']}f}/bar"
            lines.append(
                f"Pending {pat.name} on {pat_tf}: breaks {side} {px(pat.break_level)} "
                f"({gap:+.2f}% away{drift}). Risk stays {risk_pct:.0%} until it breaks."
            )

        # A split entry is two orders at two prices, so the alert states how
        # much goes into each rather than leaving the arithmetic to be done at
        # the moment of placing them. Each leg's dollar value is its own
        # quantity at its own price, not a share of the total notional.
        if signal.limit_entry is not None:
            market_size = plan.position_size * signal.market_fraction
            limit_size = plan.position_size - market_size
            note = f" ({signal.limit_note})" if signal.limit_note else ""

            if market_size <= 0:
                # No market fraction: the whole position rests on the limit,
                # same as Strategy 2 entering at the EMA9 touch rather than the
                # candle close. There is no partial fill to give a fallback
                # target for here - if it never fills, nothing is open.
                lines.append(f"Enter: {usd(limit_size * signal.limit_entry)} ({qty(limit_size)}) limit {px(signal.limit_entry)}{note}")
            else:
                lines.append(
                    f"Enter: {usd(market_size * market_price)} ({qty(market_size)}) at market {px(market_price)}"
                    f"  ·  {usd(limit_size * signal.limit_entry)} ({qty(limit_size)}) limit {px(signal.limit_entry)}{note}"
                )
                # Every level above is measured from the blended entry both legs
                # would produce. If the limit never fills, the position is only
                # the market fraction, bought at a worse price, and those levels
                # no longer describe it.
                #
                # Re-anchored on the RATIO, not the distance. Carrying the same
                # absolute reward across was the first version and it silently
                # halved the trade: the market fill is further from the stop
                # than the blended entry would have been, so the same dollar
                # gain is a smaller multiple of a larger risk. ZECUSDT trade #13
                # is the worked example - planned entry 483.774 against a 467.97
                # stop is 15.80 of risk and a 515.377 target, but only the market
                # leg filled at 498.41, making the real risk 30.44. The
                # distance-based fallback of 530.01 came out at 1.04R, and Dror
                # closed it at 529.00 for 0.99R on a setup meant to pay 2R.
                #
                # Measuring from the market fill's own risk keeps the 1:2 the
                # strategy is built around. It does ask for a bigger move -
                # 559.29 rather than 530.01 on ZECUSDT - which is the honest
                # price of the worse entry rather than a target being moved for
                # convenience.
                fallback_target = _reward_target(
                    market_price, signal.stop_loss, signal.direction, reward_risk_ratio
                )
                lines.append(
                    f"If the limit leg never fills: exit the market-only {qty(market_size)} at {px(fallback_target)}."
                )

        # Only a real two-tier exit is worth describing. When the strategy's own
        # reward:risk already equals the remainder ratio both tiers land on the
        # same price, so the partial and the stop-to-breakeven step do nothing.
        if strategy_manages_exit and remainder_size <= _SIZE_EPSILON:
            # A strategy can manage its own exit by having only ONE tier.
            # Strategy 4 closes the whole position at its gap target, so there
            # is no partial to describe and no runner to hand over - the
            # headline Target line above already states it. Printing the
            # two-tier sentence here would say "close the remaining 0".
            pass
        elif strategy_manages_exit:
            if signal.remainder_target is not None:
                note = f" ({signal.remainder_note})" if signal.remainder_note else ""
                tail = f"at {px(signal.remainder_target)}{note}"
            else:
                # No price for the runner - the setup is at all-time highs with
                # nothing overhead, so the exit is a rule rather than a level.
                tail = signal.remainder_note or "at your discretion"
            lines.append(
                f"Partial: close {qty(partial_size)} ({partial_fraction:.0%}) at {px(plan.take_profit)}, "
                f"move stop to {px(plan_entry)}, then close the remaining {qty(remainder_size)} {tail}."
            )
        elif abs(plan.take_profit - remainder_target) > _PRICE_EPSILON * max(abs(remainder_target), 1.0):
            lines.append(
                f"Partial: close {qty(partial_size)} ({partial_fraction:.0%}) at {px(plan.take_profit)}, "
                f"move stop to {px(plan_entry)}, then close the remaining "
                f"{qty(remainder_size)} at {px(remainder_target)} (1:{REMAINDER_TARGET_RATIO:g})."
            )

        lines.extend(signal.extra_notes)

        # The id in the header IS the /add hint now - it used to be a
        # separate trailing line, spelling out "/add {signal_id}" so there
        # was something to type. Without the number showing SOMEWHERE, the
        # alternative is naming the strategy by hand, which is how XAGUSDT
        # #17 came to be tagged "Strategy 1 1h" against a "Strategy 1 1H"
        # alert and went its whole life unmanaged - the header now carries
        # that same number instead.
        text = "\n".join(lines)

        def on_approve() -> None:
            self.storage.mark_signal_decision(signal_id, "approved")
            # ROUNDED TO THE EXCHANGE'S OWN PRICE PRECISION before it is
            # recorded as "the plan". Left as the strategy's raw float, this
            # was never equal to what actually lands on Bitget - every stop
            # and target placed there is rounded to the symbol's price_place
            # first (round_price, called inside place_order/place_tpsl_order),
            # while the value stored here was not.
            #
            # changed_from_plan compares the two at a tolerance of 1e-9, so
            # that mismatch reads as a genuine deviation on almost every trade.
            # AIOUSDT (Strategy 2.1 15m, 2026-08-19) is typical: original
            # 0.04359480847113895, actual 0.04359 - a difference of 4.8e-6 that
            # is pure price-tick rounding, four orders of magnitude past the
            # tolerance, and the close message told Dror the stop had been
            # changed when nothing had touched it.
            #
            # Rounding here instead of loosening the tolerance keeps
            # changed_from_plan meaning what it says: a real difference, in
            # units the exchange can actually place, rather than "did the
            # strategy happen to compute a round number". get_contract_specs
            # is cached for the process lifetime and was already read above
            # for the min-notional check, so this costs no extra request.
            try:
                proposed_stop = self.bitget.round_price(signal.symbol, signal.stop_loss)
                proposed_target = self.bitget.round_price(signal.symbol, plan.take_profit)
            except Exception:
                logger.exception("Could not round %s's plan to price precision; recording it raw", signal.symbol)
                proposed_stop, proposed_target = signal.stop_loss, plan.take_profit
            trade_id = self.storage.create_pending(
                symbol=signal.symbol,
                direction=signal.direction,
                proposed_stop=proposed_stop,
                proposed_target=proposed_target,
                strategy_tag=signal.strategy_tag,
            )
            self.storage.link_signal_trade(signal_id, trade_id)

            # Start waiting for the pattern to resolve. Registered on approval
            # rather than at dispatch, so a signal that was never taken cannot
            # produce an add-on for a position that does not exist.
            if pending_pattern is not None:
                pat, pat_tf = pending_pattern
                self._pending_breaks.register(signal.symbol, {
                    "direction": signal.direction,
                    "name": pat.name,
                    "timeframe": pat_tf,
                    "break_level": pat.break_level,
                    "invalidation_level": pat.invalidation_level,
                    "trade_id": trade_id,
                    "strategy_tag": signal.strategy_tag,
                    "risk_pct": risk_pct,
                })

            order = _build_order(signal, plan, market_price)

            async def _execute_and_track() -> None:
                if self.auto_executes(signal.strategy_tag):
                    # Off the event loop, on a worker thread: BitgetClient is
                    # synchronous (requests, not async httpx), so calling
                    # execute() straight from on_approve used to freeze the
                    # WHOLE bot - the scan loop and every other Telegram tap -
                    # for as long as leverage-set plus each order leg took to
                    # answer. on_approve has already returned by the time this
                    # runs, so the "Approved." edit Dror sees no longer waits
                    # on it either - it was already just an ack that the tap
                    # was received, never proof the order went through; a
                    # separate EXECUTION FAILED / confirmation message always
                    # followed on its own regardless.
                    result = await asyncio.to_thread(self.executor.execute, order)
                    if not result.ok:
                        # Fail-safe: no retry. The account is the only truth
                        # about what exists after an ambiguous failure, so say
                        # what was attempted and stop rather than guessing.
                        self.storage.cancel_pending(trade_id, f"execution failed: {result.error}")
                        await self.bot.send_message(
                            f"EXECUTION FAILED for {signal.symbol} {signal.direction} "
                            f"({signal.strategy_tag}): {result.error}\n"
                            f"{len(result.placed)} of {len(order.legs)} leg(s) were placed before it stopped. "
                            f"Check the account before acting — nothing was retried."
                        )
                        return
                    ledger.try_record(self.storage.db_path, ledger.ENTRY_ORDER_PLACED)

                await self._confirm_and_track(
                    trade_id,
                    signal,
                    plan,
                    order,
                    # The runner's fallback level, carried through so the
                    # partial-fill handler places exactly what the alert
                    # promised. The breakeven is NOT carried: plan_entry is an
                    # estimate made before anything filled, and the confirmed
                    # position knows better - see set_exit_plan below.
                    remainder_target=signal.remainder_target
                    if signal.partial_fraction is not None
                    else remainder_target,
                )

            asyncio.create_task(_execute_and_track())

        def on_reject() -> None:
            self.storage.mark_signal_decision(signal_id, "rejected")

        # Best-effort, never a reason to withhold the alert - see chart.build's
        # own docstring. Only attempted when both the setting is on AND the
        # caller actually has bars to draw from (poll_pending_breaks/
        # _offer_add_on don't pass bars_by_tf, and that's fine: those alerts
        # still go out text-only, same as ever).
        photo = None
        if self.send_chart_images and strategy is not None and bars_by_tf is not None:
            photo = chart.build(bars_by_tf, strategy, signal, entry=plan_entry, target=plan.take_profit)

        # The prompt is the ONLY thing that spends the day's allowance, and it
        # is spent after the send rather than before it - see _throttled.
        try:
            await self.bot.send_signal(
                text,
                on_approve,
                on_reject,
                expiry_seconds=signal_expiry_seconds(timeframes[0]),
                # plan_entry defines 1R with the stop, since that is where the
                # order actually rests; market_price is only the starting point
                # drift is measured FROM. Passing market_price as both (the
                # first attempt at the QQQUSDT fix) made 1R three times too
                # large on INJUSDT, whose limit sits far from market by
                # construction. See NotifierBot._expire for why all three
                # prices are distinct.
                entry_price=plan_entry,
                stop_loss=signal.stop_loss,
                reference_price=market_price,
                price_fetcher=lambda: self.bitget.get_mark_price(signal.symbol),
                photo=photo,
            )
        except Exception:
            # Deliberately NOT retried. A Telegram timeout is ambiguous - the
            # message often did arrive - so resending is as likely to double-post
            # as to deliver. The throttle simply goes unspent, and _seen keeps
            # this setup from re-prompting for the life of the process.
            logger.exception("Could not send the %s %s alert", signal.symbol, signal.strategy_tag)
            self.storage.mark_signal_decision(signal_id, "send_failed")
            return

        self._mark_alerted((signal.symbol, signal.strategy_tag))

    async def _confirm_and_track(
        self,
        trade_id: int,
        signal: Signal,
        plan=None,
        order: TradeOrder | None = None,
        remainder_target: float | None = None,
    ) -> None:
        # Only a strategy whose orders really reach the exchange should have
        # exit orders placed for it; a dry-run strategy must stay dry all the
        # way through, not just at entry.
        executed = (
            order is not None
            and self.auto_executes(signal.strategy_tag)
            and self.executor.handles_live(signal.strategy_tag)
        )
        # A strategy that measured its own unfilled window gets it; everything
        # else keeps the tracker's flat default.
        position = await wait_for_signal_position(
            self.bitget, signal.symbol, signal.direction,
            **({"timeout_seconds": signal.unfilled_timeout_seconds}
               if signal.unfilled_timeout_seconds else {}),
        )
        if position is None:
            self.storage.cancel_pending(trade_id)
            # Nothing filled, so any resting leg - bot-placed or placed by
            # hand off the alert - is an order with no trade behind it. Left
            # alone it could open a position hours later against a setup that
            # no longer exists, so it's cancelled regardless of who placed it.
            self._cancel_resting(signal.symbol)
            await self.bot.send_message(
                f"No position detected for trade #{trade_id} ({signal.symbol} {signal.direction}) "
                f"within the timeout — marked cancelled, and {signal.symbol} is free to signal again."
            )
            return

        stop, target = self._safe_stop_target(signal.symbol, signal.direction, position)
        self.storage.confirm_entry(
            trade_id,
            position["entry_price"],
            position["size"],
            stop,
            target,
            leverage=position["leverage"],
        )

        if stop is None:
            await self.bot.send_message(
                f"Trade #{trade_id} ({signal.symbol} {signal.direction}) is being tracked, but no stop-loss "
                f"is set on Bitget — R-multiple can't be computed until you set one."
            )

        # The exit plan goes to the DB rather than staying in this coroutine's
        # closure. It used to live only here, so a restart - and every deploy
        # is one - left the re-attached tracker able to SEE the partial fill
        # but with no idea that a breakeven was owed. Recorded only when the
        # bot really manages this trade's exits, so a set breakeven_stop is a
        # commitment rather than a note.
        #
        # The breakeven recorded is the CONFIRMED entry, not the alert's
        # plan_entry. plan_entry blends the market leg's expected fill with
        # the limit level before either has happened, so it is an estimate
        # twice over; what actually filled is known here. breakeven_price()
        # re-derives it from the row at partial time anyway, which is what
        # picks up a limit leg that fills later - this only keeps the stored
        # value from being a number nothing would ever use.
        # BOTH TARGETS RE-ANCHORED ON THE REAL FILL, for the same reason the
        # breakeven above is: plan_entry blends the market leg's EXPECTED fill
        # with the limit level before either has happened, so it is an
        # estimate, and position["entry_price"] is what actually filled.
        #
        # Until this, plan.take_profit and remainder_target stayed frozen at
        # their plan_entry-derived values all the way through - the exact
        # comment three lines above ("plan_entry is an estimate... the
        # confirmed position knows better") was already true of them and
        # applied only to the stop. Measured across every closed Strategy 1
        # trade, plan_entry differs from the real fill by more than 1% on a
        # third of them - EULUSDT +4.6%, DEXEUSDT -5.5%, ZECUSDT #13 +3.0%,
        # SNDKUSDT #54 -1.4% - and each point of that gap is a point the
        # realized reward:risk drifts from the 1:2 / 1:3 the strategy is
        # built around, in either direction. SNDKUSDT #54 closed both legs
        # almost exactly on its (wrong) targets and still read 1.17R instead
        # of a number nearer 2, because the risk it was measured against
        # (from the real, worse entry) was 60% wider than the risk the
        # targets were priced for.
        #
        # Safe for a pure-limit strategy (Strategy 4, market_fraction=0):
        # position["entry_price"] there equals signal.entry_price exactly, so
        # this is a no-op. It only moves anything when a market leg pulled the
        # blended entry away from what was planned - which is precisely when
        # the old prices were wrong.
        ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else self.reward_risk_ratio
        real_entry = position["entry_price"]
        if plan is not None:
            plan.take_profit = _reward_target(real_entry, signal.stop_loss, signal.direction, ratio)
        # remainder_target is only ever this ratio-derived fallback when the
        # strategy did NOT supply its own (partial_fraction is None) - the
        # call site above already resolves signal.remainder_target through
        # unchanged for a self-managing strategy, and that absolute price (or
        # None, meaning "trail") is the strategy's own decision, not an
        # estimate to correct.
        if signal.partial_fraction is None:
            remainder_target = _reward_target(
                real_entry, signal.stop_loss, signal.direction, REMAINDER_TARGET_RATIO
            )

        if self.manages_exits(signal.strategy_tag):
            self.storage.set_exit_plan(
                trade_id,
                breakeven_stop=position["entry_price"],
                runner_target=remainder_target,
                partial_fraction=signal.partial_fraction,
                # Recorded because the partial handler rebuilds the signal from
                # this row - see _exit_plan_signal. Without it, a strategy that
                # deliberately asks for NO runner target is indistinguishable
                # from one that simply has not said.
                runner_target_is_final=signal.remainder_target_is_final,
            )

        # The partial can't ride on the entry the way the stop does: a preset
        # carries one target for the whole position, and this closes only part
        # of it. Sized to what actually filled rather than to the intended
        # position - on a split entry the market leg confirms first, and the
        # limit leg may fill later or never.
        if executed and plan is not None:
            await self._place_partial(signal, plan, position["size"])

        await track_position(
            self.storage,
            self.bitget,
            trade_id,
            signal.symbol,
            signal.direction,
            on_close=self._on_trade_closed,
            # The partial filling is what promotes the runner from "at your
            # discretion" to a real order: the stop goes to the breakeven the
            # alert already printed, and the runner gets a target. That now
            # happens inside _on_partial_exit off the stored plan, so this is
            # the same callback resume_open_trades hands to a re-attached
            # tracker and there is exactly one path to a breakeven.
            on_partial=self._on_partial_exit,
            on_scale_in=self._on_scale_in,
            # on_resize fires synchronously from inside track_position's poll
            # loop, so the retryable coroutine has to be scheduled rather than
            # awaited here - awaiting would stall that loop's own polling for
            # as long as the retry takes.
            on_resize=(lambda size: asyncio.create_task(self._place_partial(signal, plan, size, replace=True)))
            if (executed and plan is not None)
            else None,
        )

    async def poll_untracked_positions(self) -> None:
        await self._health.poll_untracked_positions()

    async def poll_weekly_report_overdue(self) -> None:
        await self._health.poll_weekly_report_overdue()

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

    def _on_partial_exit(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        self._lifecycle.on_partial_exit(trade_id, closed_size, realized_pnl)

    def _manages_trade(self, trade) -> bool:
        return self._lifecycle.manages_trade(trade)

    async def adopt_trade(self, trade_id: int, breakeven: float, runner_target: float | None = None) -> str:
        return await self._lifecycle.adopt_trade(trade_id, breakeven, runner_target)

    def _exit_plan_signal(self, trade) -> Signal | None:
        return self._lifecycle.exit_plan_signal(trade)
