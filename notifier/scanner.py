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

from core.bitget_client import BitgetClient
from core.storage import Storage
from execution.executor import Executor, OrderLeg, TradeOrder
from execution.tracker import (
    format_close_message,
    format_partial_message,
    format_scale_in_message,
    take_profit_coverage,
    track_position,
    wait_for_signal_position,
)
from notifier import sessions
from notifier.risk_sizing import DEFAULT_MAX_LEVERAGE, DEFAULT_REWARD_RISK_RATIO, plan_position
from notifier.strategies import patterns
from notifier.strategies.indicators import atr
from notifier.strategies.structure import nearest_level_beyond, zigzag_pivots
from notifier.strategies.base import TIMEFRAME_SECONDS, Signal, Strategy

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOTAL_RISK_PCT = 0.06
CANDLE_CLOSE_DELAY = 30.0  # let Bitget settle the just-closed candle before reading it
PARTIAL_TAKE_FRACTION = 0.5
# Fixed 1:3 target for whatever's left after the partial take, regardless of
# the first tier's own ratio — replaces an open-ended "let it run" with a
# defined second exit, so nothing is left unmanaged indefinitely.
REMAINDER_TARGET_RATIO = 3.0
# The runner's take-profit, placed automatically once the partial fills.
#
# Dror's rule, after an SPCXUSDT signal whose runner read "at your discretion":
# "after the target i want the bot to make the new tp himself without my
# intervation ... the next tp under the nearest key line in the 1h graph".
#
# 1H rather than the daily the strategy's own remainder_target uses, because
# that one comes from DAILY pivot highs and is empty whenever the setup is at
# all-time highs - which is exactly when a runner most needs a target. The
# buffer sits the order in the queue ahead of the sellers defending the level:
# a limit resting exactly ON resistance frequently does not fill, since price
# stalls a tick short and turns.
# Read off the DAILY chart, though Dror reads the level on his 1H one. Both
# were tried against the SPCXUSDT trade he used to specify this. The nearest
# 1H pivot above the partial target was 136.21 - a minor swing nobody would
# aim a runner at - while the daily gives 147.24, the level he named on sight
# ("significant support with 3 touches on the 1h chart and in the daily it
# also have 2 touches"). The 1H chart is where the level is VISIBLE; the daily
# is what makes it significant enough to stop a move.
RUNNER_LEVEL_TIMEFRAME = "1D"
RUNNER_LEVEL_ATR_BUFFER = 0.25
RUNNER_LEVEL_ATR_PERIOD = 14
# 2.0 on the daily. volume_run's own daily setup uses 3.0, but at 3.0 SPCXUSDT
# collapses to just three levels in 60 days (103.55 / 152.08 / 228.11) and the
# 147.24 Dror named disappears into a coarser 152.08. 2.0 resolves the swings
# a runner actually has to get through.
RUNNER_LEVEL_PIVOT_ATR_MULTIPLE = 2.0
# A level further than this from where price is now belongs to a different
# market regime, not to this trade. MUUUSDT traded 1000-1400 in June and 26.51
# today, so its nearest "level" overhead is a 803.5 low from before the
# collapse - 30x away, a target that can never fill and only looks like one.
# Measured across a live sweep the sensible levels sat at 0.004-2.47 ATR from
# price while that artifact sat at 23.7, so the cut goes in the wide gap
# between; it is a judgment call inside that gap rather than a tuned edge.
RUNNER_LEVEL_MAX_ATR = 6.0
# When no level can be found at all, the runner is not left unmanaged: the
# stop trails instead. Dror's rule - "if not support or resistence can be
# found like muu change to the all time high method we already have, as long
# there is rising highs change the stoploss only to the last low (in long)".
#
# This is the "At all-time highs: trail the stop up under each rising low"
# line the alert has always printed as an instruction, now actually carried
# out. STOP ONLY: no target is invented, and the stop is never loosened - it
# moves up for a long, down for a short, or not at all.
TRAILING_POLL_TIMEFRAME = "1H"
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
# Bitget's position query and its order-matching book are eventually
# consistent: wait_for_signal_position already confirmed the fill, but a
# reduce-only order placed immediately after can still be rejected with
# 22002 "No position to close" for a few seconds - happened twice live on
# AAPLUSDT within one session, both times leaving a protected position with
# no target. A 400 rejection here is unambiguous (the order was never placed,
# unlike a timed-out entry leg that might have gone through), so retrying is
# safe rather than a risk of duplicating an exit - each attempt uses a fresh
# client_oid and only ever one attempt can succeed against the exchange.
PARTIAL_SETTLE_RETRY_DELAYS = (3.0, 6.0, 12.0)  # ~21s of settle time past the first attempt
# Relative tolerance for deciding two price levels are the same one. A strategy
# whose own reward:risk already equals REMAINDER_TARGET_RATIO puts both exit
# tiers on the identical price, and describing that as a partial take plus a
# stop-to-breakeven is describing steps that cannot happen.
_PRICE_EPSILON = 1e-9


def _reward_target(entry_price: float, stop_loss: float, direction: str, ratio: float) -> float:
    risk_per_unit = abs(entry_price - stop_loss)
    return entry_price + risk_per_unit * ratio if direction == "long" else entry_price - risk_per_unit * ratio


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


def bars_dataframe(candles: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df


def seconds_until_next_close(timeframe: str, now: float | None = None) -> float:
    """Seconds until the next candle of this timeframe closes, plus a small
    settle delay."""
    period = TIMEFRAME_SECONDS[timeframe]
    now = time.time() if now is None else now
    return (period - (now % period)) + CANDLE_CLOSE_DELAY


SIGNAL_EXPIRY_FLOOR = 60.0
SIGNAL_EXPIRY_CEILING = 1800.0


def signal_expiry_seconds(timeframe: str, now: float | None = None) -> float:
    """The timer half of an Approve/Reject offer's expiry (see
    core.telegram_bot for the movement half it races against).

    Anchored to when this signal's OWN candle next closes - the point the
    scanner would already be re-evaluating this setup with fresh eyes, so an
    offer still unacted on past that point is stale on structural grounds
    alone, not just convention. Floored so a signal fired late in its candle
    still leaves a minute to read and tap; capped so a slow timeframe (1D
    can be most of a day away from its next close) can't leave a live-money
    offer sitting for hours just because price hasn't moved enough yet to
    trip the other cutoff.
    """
    return min(max(seconds_until_next_close(timeframe, now), SIGNAL_EXPIRY_FLOOR), SIGNAL_EXPIRY_CEILING)


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
        self.swing_tags = swing_tags
        self.max_swing_slots = max_swing_slots
        # Insertion-ordered so _prune_seen can drop the OLDEST rather than an
        # arbitrary half - see there. Values are unused; this is a set that
        # remembers order.
        self._seen: dict[tuple, None] = {}
        # (symbol, timeframe) -> (candle this was fetched during, bars). Scans
        # run at the shortest timeframe's cadence, so without this a daily
        # candle would be refetched 96 times a day to learn it had not changed.
        # Caching until the candle actually turns over cuts the load roughly
        # fourfold even while adding two timeframes.
        self._bars_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        # strategy tag -> symbols worth polling on its armed timeframe. Rebuilt
        # wholesale by every scan rather than mutated, so nothing can stay
        # armed after its setup stops qualifying.
        self._armed: dict[str, set[str]] = {}
        # symbol -> the unbroken pattern a held position is waiting on, so the
        # second risk increment can be offered when it breaks. Unlike _armed
        # this cannot be rebuilt from current bars - it records what was true
        # when the trade was approved - so it is held rather than recomputed,
        # and is deliberately lost on restart: re-offering an add-on for a
        # break that happened while the process was down would be acting on
        # stale news.
        self._awaiting_break: dict[str, dict] = {}
        self.auto_execute_tags = auto_execute_tags or set()
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
        timeframes = self.required_timeframes()
        if not timeframes:
            logger.warning("No strategies registered; scanner has nothing to do")
            return

        loops = [self._scan_loop(timeframes), self._pending_break_loop(), self._trailing_loop()]
        if self.armed_timeframes():
            loops.append(self._armed_loop())
        await asyncio.gather(*loops)

    async def _scan_loop(self, timeframes: set[str]) -> None:
        while True:
            scan_tf = min(timeframes, key=seconds_until_next_close)
            delay = seconds_until_next_close(scan_tf)
            logger.info("Next scan (driven by %s) in %.0fs", scan_tf, delay)
            await asyncio.sleep(delay)
            await self.tick()

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

    async def _trailing_loop(self) -> None:
        """Hourly, since a swing low only changes when a bar closes."""
        while True:
            await asyncio.sleep(seconds_until_next_close(TRAILING_POLL_TIMEFRAME))
            try:
                await self.poll_trailing_stops()
            except Exception:
                logger.exception("Trailing-stop poll failed; continuing")

    async def _pending_break_loop(self) -> None:
        while True:
            await asyncio.sleep(seconds_until_next_close(PENDING_BREAK_TIMEFRAME))
            try:
                await self.poll_pending_breaks()
            except Exception:
                logger.exception("Pending-break poll failed; continuing")

    async def poll_pending_breaks(self) -> None:
        """Offer the second risk increment once a held position's pattern
        actually breaks.

        Ends a watch when the position closes or the pattern dies - there is
        deliberately no clock, since both of those bound it naturally and an
        arbitrary bar count would be a constant nobody measured.
        """
        if not self._awaiting_break:
            return
        try:
            equity = self.bitget.get_account_equity()
        except Exception:
            logger.exception("Could not read equity; skipping the pending-break poll")
            return

        for symbol, watch in list(self._awaiting_break.items()):
            try:
                trade = self.storage.get_trade(watch["trade_id"])
            except Exception:
                self._awaiting_break.pop(symbol, None)
                continue
            if not trade.is_open:
                self._awaiting_break.pop(symbol, None)  # nothing left to add to
                continue

            try:
                bars = self._bars(symbol, watch["timeframe"]).iloc[:-1]
            except Exception:
                logger.exception("Could not fetch bars for the pending-break watch on %s", symbol)
                continue
            if len(bars) < 2:
                continue

            direction = watch["direction"]
            # Re-derive the level from fresh bars BEFORE testing it: a
            # triangle or wedge sits on a converging line at a different price
            # every bar, so the level stored at approval time is already out
            # of date by the time this runs.
            try:
                refreshed = patterns.pending({watch["timeframe"]: bars}, direction)
            except Exception:
                logger.exception("Pending-pattern refresh failed for %s", symbol)
                refreshed = None
            if refreshed is not None and refreshed[0].name == watch["name"]:
                watch["break_level"] = refreshed[0].break_level
                watch["invalidation_level"] = refreshed[0].invalidation_level

            close = float(bars["close"].iloc[-1])
            broke = close > watch["break_level"] if direction == "long" else close < watch["break_level"]
            died = close < watch["invalidation_level"] if direction == "long" else close > watch["invalidation_level"]

            if broke:
                self._awaiting_break.pop(symbol, None)
                await self._offer_add_on(symbol, watch, trade, equity, close)
            elif died:
                self._awaiting_break.pop(symbol, None)
                # No exit action: the stop already defines where this trade
                # ends, and overriding a defined stop with a discretionary
                # exit is a different decision than the one being made here.
                await self.bot.send_message(
                    f"{symbol}: the {watch['name']} on {watch['timeframe']} broke the WRONG way "
                    f"(closed {close:g} through {watch['invalidation_level']:g}). No add-on. "
                    f"Your stop still governs the position — nothing was changed."
                )
            elif refreshed is None:
                self._awaiting_break.pop(symbol, None)
                logger.info("Pending %s on %s no longer reads as itself; watch dropped", watch["name"], symbol)

    async def _offer_add_on(self, symbol: str, watch: dict, trade, equity: float, break_close: float) -> None:
        direction = watch["direction"]
        existing_stop = trade.סטופ_לוס_בפועל or trade.סטופ_לוס_מקורי
        flag_stop = watch["invalidation_level"]
        # The break is a reason to risk more behind a tighter stop, never to
        # give the original leg more room - so take whichever is tighter.
        if existing_stop is None:
            new_stop = flag_stop
        else:
            new_stop = max(existing_stop, flag_stop) if direction == "long" else min(existing_stop, flag_stop)

        try:
            market_price = self.bitget.get_mark_price(symbol)
        except Exception:
            logger.exception("Could not read mark price for the %s add-on", symbol)
            market_price = break_close

        try:
            plan = plan_position(
                equity=equity,
                risk_pct=watch["risk_pct"],
                entry_price=market_price,
                stop_loss=new_stop,
                direction=direction,
                reward_risk_ratio=self.reward_risk_ratio,
                available_budget=equity - self.storage.committed_margin(),
                max_leverage=self.max_leverage,
            )
        except ValueError as exc:
            logger.info("No add-on for %s: %s", symbol, exc)
            return

        # The same aggregate ceiling every other trade obeys - a pattern
        # breaking is not a licence to exceed it.
        risk_cap = equity * self.max_total_risk_pct
        if self.storage.total_open_risk() + plan.risk_amount > risk_cap:
            logger.info("No add-on for %s: it would exceed the %.0f%% cap", symbol, self.max_total_risk_pct * 100)
            return

        specs = self.bitget.get_contract_specs(symbol)
        if self.bitget.round_size(symbol, plan.position_size) <= 0 or plan.notional_value < specs["min_notional"]:
            logger.info("No add-on for %s: below the exchange minimum", symbol)
            return

        def px(v: float) -> str:
            return f"{v:.{specs['price_place']}f}"

        def qty(v: float) -> str:
            return f"{v:.{specs['volume_place']}f}"

        text = "\n".join(
            [
                f"ADD-ON: {symbol} {direction.upper()} ({watch['strategy_tag']})",
                f"The {watch['name']} on {watch['timeframe']} broke — closed {px(break_close)} "
                f"through {px(watch['break_level'])}.",
                f"Add: ${plan.notional_value:,.0f} ({qty(plan.position_size)} @ {plan.leverage:.1f}x) "
                f"at market {px(market_price)}  risk {watch['risk_pct']:.0%}",
                f"Move the stop on the WHOLE position to {px(new_stop)} — the {watch['name']}'s own level, "
                f"which is where the break is proven wrong.",
            ]
        )

        order = TradeOrder(
            symbol=symbol,
            direction=direction,
            legs=[OrderLeg(size=plan.position_size, order_type="market")],
            stop_loss=new_stop,
            leverage=plan.leverage,
            strategy_tag=watch["strategy_tag"],
        )

        def on_approve() -> None:
            if not self.auto_executes(watch["strategy_tag"]):
                return  # alert-only strategy: placed by hand, same as its entry
            result = self.executor.execute(order)
            if not result.ok:
                asyncio.create_task(
                    self.bot.send_message(
                        f"ADD-ON FAILED for {symbol} {direction} ({watch['strategy_tag']}): {result.error}\n"
                        f"The original position is untouched and nothing was retried."
                    )
                )

        await self.bot.send_signal(
            text,
            on_approve,
            expiry_seconds=signal_expiry_seconds(watch["timeframe"]),
            # An add-on goes in at market, so entry and reference are the same
            # price here - the starting gap is zero and any drift counts.
            entry_price=market_price,
            stop_loss=new_stop,
            reference_price=market_price,
            price_fetcher=lambda: self.bitget.get_mark_price(symbol),
        )

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
                    bars_by_tf[tf] = self._bars(symbol, tf)
                except Exception:
                    logger.exception("Skipping %s this scan: failed to fetch/parse %s candles", symbol, tf)
                    bars_by_tf = None
                    break

            if not bars_by_tf or any(len(b) < 2 for b in bars_by_tf.values()):
                continue

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

                await self._handle_signal(signal, strategy, equity, bars_by_tf)

        self._armed = armed

    async def _handle_signal(self, signal: Signal, strategy: Strategy, equity: float, bars_by_tf: dict) -> None:
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
            signal, equity, signal.analysis_timeframes or strategy.timeframes, confluence, pending_pattern
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
                            bars_by_tf[tf] = self._bars(symbol, tf)
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

    def _bars(self, symbol: str, timeframe: str, now: float | None = None) -> pd.DataFrame:
        """Bars for this symbol and timeframe, refetched only once the
        timeframe's candle has turned over.

        Includes the forming candle; callers trim it when they want closed bars
        only. Keyed on which candle is currently forming, so a 1D series is
        fetched once a day and a 15m series every scan, without anything having
        to know the scan cadence.
        """
        period = TIMEFRAME_SECONDS[timeframe]
        now = time.time() if now is None else now
        current_candle = now - (now % period)

        cached = self._bars_cache.get((symbol, timeframe))
        if cached and cached[0] == current_candle:
            return cached[1]

        candles = self.bitget.get_candles(symbol, granularity=timeframe, limit=self.candle_limit + 1, closed_only=False)
        bars = bars_dataframe(candles)
        self._bars_cache[(symbol, timeframe)] = (current_candle, bars)
        return bars

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
    ) -> None:
        if self.storage.has_open_or_pending(signal.symbol):
            return  # already tracking a trade on this symbol; one at a time

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
                max_leverage=self.max_leverage,
            )
        except ValueError as exc:
            logger.info("Skipping %s/%s: %s", signal.symbol, signal.strategy_tag, exc)
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

        # The checks above compare dollars, not the quantity that actually gets
        # sent - a leg can clear both minimums on paper and still floor to
        # exactly zero once rounded to the exchange's own step. AAVEUSDT trades
        # in units of 0.1: a 0.06-unit market leg was worth $6, comfortably over
        # the $5 minimum notional, but place_order's own rounding floored it to
        # 0 and Bitget rejected the order live - after the signal was already
        # approved. round_size is the same rounding place_order applies, so
        # this catches it before the alert ever goes out.
        if not too_small:
            leg_sizes = [plan.position_size]
            if signal.limit_entry is not None and signal.market_fraction > 0:
                market_size = plan.position_size * signal.market_fraction
                leg_sizes = [market_size, plan.position_size - market_size]
            too_small = any(self.bitget.round_size(signal.symbol, size) <= 0 for size in leg_sizes)

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

        lines = [
            f"Signal: {signal.symbol} {signal.direction.upper()} ({signal.strategy_tag})",
            f"Analysis timeframe: {', '.join(timeframes)}",
            f"Entry: {px(market_price)}  Stop: {px(signal.stop_loss)}  Target: {px(plan.take_profit)}",
            f"Size: {usd(plan.notional_value)} ({qty(plan.position_size)} @ {plan.leverage:.1f}x)"
            + (f"  risk {risk_pct:.0%}" if risk_pct > self.risk_pct else ""),
        ]
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
                # Every level above is measured from the resting limit price. If
                # it never fills, the position is only the market fraction, and
                # those levels sit on the wrong side of that fill to serve as
                # its target. Re-anchor the same reward distance onto the
                # market price instead.
                reward_distance = abs(plan_entry - plan.take_profit)
                fallback_target = (
                    market_price - reward_distance if signal.direction == "short" else market_price + reward_distance
                )
                lines.append(
                    f"If the limit leg never fills: exit the market-only {qty(market_size)} at {px(fallback_target)}."
                )

        # Only a real two-tier exit is worth describing. When the strategy's own
        # reward:risk already equals the remainder ratio both tiers land on the
        # same price, so the partial and the stop-to-breakeven step do nothing.
        if strategy_manages_exit:
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

        text = "\n".join(lines)

        # Logged before Approve/Reject is even seen, so a rejected or ignored
        # signal is still measurable - the trades table only ever gains a row
        # once a signal is both approved AND confirmed on Bitget, which left
        # rejected/ignored signals with no record anywhere.
        signal_id = self.storage.log_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=plan_entry,
            stop_loss=signal.stop_loss,
            take_profit=plan.take_profit,
            strategy_tag=signal.strategy_tag,
            confluence=confluence,
        )

        def on_approve() -> None:
            self.storage.mark_signal_decision(signal_id, "approved")
            trade_id = self.storage.create_pending(
                symbol=signal.symbol,
                direction=signal.direction,
                proposed_stop=signal.stop_loss,
                proposed_target=plan.take_profit,
                strategy_tag=signal.strategy_tag,
            )
            self.storage.link_signal_trade(signal_id, trade_id)

            # Start waiting for the pattern to resolve. Registered on approval
            # rather than at dispatch, so a signal that was never taken cannot
            # produce an add-on for a position that does not exist.
            if pending_pattern is not None:
                pat, pat_tf = pending_pattern
                self._awaiting_break[signal.symbol] = {
                    "direction": signal.direction,
                    "name": pat.name,
                    "timeframe": pat_tf,
                    "break_level": pat.break_level,
                    "invalidation_level": pat.invalidation_level,
                    "trade_id": trade_id,
                    "strategy_tag": signal.strategy_tag,
                    "risk_pct": risk_pct,
                }

            order = _build_order(signal, plan, market_price)
            if self.auto_executes(signal.strategy_tag):
                result = self.executor.execute(order)
                if not result.ok:
                    # Fail-safe: no retry. The account is the only truth about
                    # what exists after an ambiguous failure, so say what was
                    # attempted and stop rather than guessing.
                    self.storage.cancel_pending(trade_id, f"execution failed: {result.error}")
                    asyncio.create_task(
                        self.bot.send_message(
                            f"EXECUTION FAILED for {signal.symbol} {signal.direction} "
                            f"({signal.strategy_tag}): {result.error}\n"
                            f"{len(result.placed)} of {len(order.legs)} leg(s) were placed before it stopped. "
                            f"Check the account before acting — nothing was retried."
                        )
                    )
                    return

            asyncio.create_task(
                self._confirm_and_track(
                    trade_id,
                    signal,
                    plan,
                    order,
                    # The runner's fallback level and the breakeven the alert
                    # already printed, carried through so the partial-fill
                    # handler places exactly what the alert promised.
                    remainder_target=signal.remainder_target
                    if signal.partial_fraction is not None
                    else remainder_target,
                    breakeven_stop=plan_entry,
                )
            )

        def on_reject() -> None:
            self.storage.mark_signal_decision(signal_id, "rejected")

        await self.bot.send_signal(
            text,
            on_approve,
            on_reject,
            expiry_seconds=signal_expiry_seconds(timeframes[0]),
            # plan_entry defines 1R with the stop, since that is where the
            # order actually rests; market_price is only the starting point
            # drift is measured FROM. Passing market_price as both (the first
            # attempt at the QQQUSDT fix) made 1R three times too large on
            # INJUSDT, whose limit sits far from market by construction. See
            # NotifierBot._expire for why all three prices are distinct.
            entry_price=plan_entry,
            stop_loss=signal.stop_loss,
            reference_price=market_price,
            price_fetcher=lambda: self.bitget.get_mark_price(signal.symbol),
        )

    async def _confirm_and_track(
        self,
        trade_id: int,
        signal: Signal,
        plan=None,
        order: TradeOrder | None = None,
        remainder_target: float | None = None,
        breakeven_stop: float | None = None,
    ) -> None:
        # Only a strategy whose orders really reach the exchange should have
        # exit orders placed for it; a dry-run strategy must stay dry all the
        # way through, not just at entry.
        executed = (
            order is not None
            and self.auto_executes(signal.strategy_tag)
            and self.executor.handles_live(signal.strategy_tag)
        )
        position = await wait_for_signal_position(self.bitget, signal.symbol, signal.direction)
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
            # alert already printed, and the runner gets a target. Scheduled
            # rather than awaited for the same reason on_resize is - this
            # fires from inside the tracker's own poll loop.
            on_partial=lambda tid, closed, pnl: (
                self._on_partial_exit(tid, closed, pnl),
                asyncio.create_task(
                    self._on_partial_manage_exits(signal, plan, remainder_target, breakeven_stop)
                ),
            )[0],
            on_scale_in=self._on_scale_in,
            # on_resize fires synchronously from inside track_position's poll
            # loop, so the retryable coroutine has to be scheduled rather than
            # awaited here - awaiting would stall that loop's own polling for
            # as long as the retry takes.
            on_resize=(lambda size: asyncio.create_task(self._place_partial(signal, plan, size, replace=True)))
            if (executed and plan is not None)
            else None,
        )

    def trail_timeframe(self, strategy_tag: str) -> str:
        """The frame whose swings a trade's stop should trail.

        A strategy's own structural timeframe, read off its tag - "Strategy 3
        1D/1H" trails daily swings, "Strategy 3 1H/5m" hourly ones. Trailing a
        multi-day setup on 5m lows would ratchet the stop into the first bit
        of noise, and trailing an intraday one on daily lows would never move
        at all.
        """
        for part in strategy_tag.split():
            if "/" in part:
                candidate = part.split("/")[0]
                return candidate if candidate in TIMEFRAME_SECONDS else TRAILING_POLL_TIMEFRAME
            if part in TIMEFRAME_SECONDS:
                return part
        return TRAILING_POLL_TIMEFRAME

    def trailing_stop(self, symbol: str, direction: str, strategy_tag: str, current_stop: float | None):
        """Where a trailing stop belongs right now, or None to leave it alone.

        Only while structure is still making higher highs (lower lows for a
        short) - that is the "as long there is rising highs" half of the rule,
        and without it a stop would keep ratcheting into a topping market. The
        new stop is the last CONFIRMED swing low, never an unconfirmed one,
        and it is only ever returned when it improves on the current stop.
        """
        bars = self._bars(symbol, self.trail_timeframe(strategy_tag))
        thresholds = atr(bars, RUNNER_LEVEL_ATR_PERIOD) * RUNNER_LEVEL_PIVOT_ATR_MULTIPLE
        pivots = zigzag_pivots(bars, thresholds)

        highs = [bars["high"].iloc[i] for i, is_high in pivots if is_high]
        lows = [bars["low"].iloc[i] for i, is_high in pivots if not is_high]
        if len(highs) < 2 or not lows:
            return None

        if direction == "long":
            if not highs[-1] > highs[-2]:
                return None  # no longer making higher highs
            new_stop = float(lows[-1])
            return new_stop if current_stop is None or new_stop > current_stop else None

        if not lows[-1] < lows[-2]:
            return None
        new_stop = float(highs[-1])
        return new_stop if current_stop is None or new_stop < current_stop else None

    async def poll_trailing_stops(self) -> None:
        """Trail the stop on any managed position that has no target.

        Having no target is precisely the case this exists for: the runner
        could not be given one because nothing was found overhead, so the
        stop is the only thing left to manage. A position WITH a target is
        left alone - it already has a defined exit.
        """
        for trade in self.storage.open_trades():
            tag = trade.תגית_אסטרטגיה or ""
            if not self.manages_exits(tag):
                continue
            symbol, direction = trade.סימבול, trade.כיוון
            try:
                _, target = self.bitget.get_stop_target(symbol, direction)
                if target is not None:
                    continue  # it has a defined exit; nothing to trail toward
                new_stop = self.trailing_stop(symbol, direction, tag, trade.סטופ_לוס_בפועל)
                if new_stop is None:
                    continue
                self.bitget.place_tpsl_order(
                    symbol=symbol,
                    direction=direction,
                    plan_type="loss_plan",
                    trigger_price=new_stop,
                    client_oid=f"trail-{symbol}-{int(time.time() * 1000)}",
                )
            except Exception:
                logger.exception("Could not trail the stop on %s", symbol)
                continue

            self.storage.update_actual_stop_target(trade.מספר_עסקה, new_stop, None)
            await self.bot.send_message(
                f"Trailed the stop on {symbol} {direction} ({tag}) up to {new_stop:g} — "
                f"the last confirmed {self.trail_timeframe(tag)} swing "
                f"{'low' if direction == 'long' else 'high'}, while structure keeps making "
                f"{'higher highs' if direction == 'long' else 'lower lows'}."
            )

    def manages_exits(self, strategy_tag: str) -> bool:
        """Whether the bot may place reduce-only exits on a tracked position.

        Strictly weaker than auto_executes: it cannot open or grow a position,
        only close one that already exists. Strategy 3's entries stay manual.
        """
        return self.executor.manages_exits(strategy_tag)

    def runner_target(self, signal: Signal, fallback: float | None) -> tuple[float | None, str]:
        """Where the runner's take-profit goes, and what to call it.

        For a strategy managing its own two-tier exit (Strategy 3), it is the
        nearest confirmed daily swing level beyond price - high or low, since
        broken support is resistance on the way back - less a buffer.
        `fallback` is that strategy's own remainder_target, used only when the
        daily offers nothing. When neither does, the setup is at highs with
        nothing overhead and there is genuinely no price to sell into, so the
        runner keeps trailing rather than being capped at an invented number.

        Everything else (Strategy 1) keeps the ratio target it already
        computes and prints; this only starts PLACING it.
        """
        if signal.partial_fraction is None:
            return fallback, f"1:{REMAINDER_TARGET_RATIO:g}"

        try:
            bars = self._bars(signal.symbol, RUNNER_LEVEL_TIMEFRAME)
            price = float(bars["close"].iloc[-1])
            thresholds = atr(bars, RUNNER_LEVEL_ATR_PERIOD) * RUNNER_LEVEL_PIVOT_ATR_MULTIPLE
            level = nearest_level_beyond(bars, thresholds, price, signal.direction)
        except Exception:
            logger.exception("Could not read %s levels for %s's runner", RUNNER_LEVEL_TIMEFRAME, signal.symbol)
            return fallback, "the strategy's own resistance level"

        if level is None:
            return fallback, "the strategy's own resistance level"

        atr_now = float(atr(bars, RUNNER_LEVEL_ATR_PERIOD).iloc[-1])
        if atr_now <= 0 or abs(level - price) > RUNNER_LEVEL_MAX_ATR * atr_now:
            return fallback, "the strategy's own resistance level"  # a different regime's level

        target = level - atr_now * RUNNER_LEVEL_ATR_BUFFER if signal.direction == "long" \
            else level + atr_now * RUNNER_LEVEL_ATR_BUFFER
        # The buffer can be wider than the gap to the level when price is
        # already sitting right under it - ENAUSDT's level was 0.02% away
        # against a 1.5%-of-price buffer, which put the "take-profit" BELOW
        # the market. A reduce-only sell below market is not a target, it is
        # an instant exit at whatever is bid.
        beyond = target > price if signal.direction == "long" else target < price
        if not beyond:
            return fallback, "the strategy's own resistance level"

        return target, f"under the {RUNNER_LEVEL_TIMEFRAME} level at {level:g}"

    async def place_runner_target(self, signal: Signal, plan, fallback: float | None) -> None:
        """Place the runner's take-profit, once the partial has filled.

        Sized to whatever is actually left rather than to the plan, since the
        partial may have closed more or less than intended, and placed only
        for a strategy whose exits the bot manages.
        """
        if not self.manages_exits(signal.strategy_tag):
            return
        target, note = self.runner_target(signal, fallback)
        if target is None:
            return  # nothing overhead - the runner trails instead, as the alert said

        try:
            position = self.bitget.get_position(signal.symbol, signal.direction)
        except Exception:
            logger.exception("Could not read the %s position to size its runner", signal.symbol)
            return
        if not position or position["size"] <= 0:
            return  # already fully closed

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0, *PARTIAL_SETTLE_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            try:
                self._place_reduce_only(signal.symbol, signal.direction, position["size"], target, "runner")
                await self.bot.send_message(
                    f"Runner target set for {signal.symbol} {signal.direction} "
                    f"({signal.strategy_tag}): {target:g} — {note}."
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("Runner target for %s rejected on attempt %d: %s", signal.symbol, attempt + 1, exc)
                if "22002" not in str(exc):
                    break  # not the settle race; waiting will not fix it

        logger.error("Could not place the runner target for %s: %s", signal.symbol, last_exc)
        await self.bot.send_message(
            f"The partial filled on {signal.symbol} {signal.direction} but the RUNNER TARGET FAILED: "
            f"{last_exc}\nThe position still has its stop — set the target by hand."
        )

    async def _on_partial_manage_exits(self, signal: Signal, plan, fallback: float | None, breakeven: float | None) -> None:
        """Everything the alert told Dror to do by hand once the partial fills:
        move the stop to breakeven, then set the runner's target.

        Both are reduce-only or protective, so this needs manages_exits rather
        than full execution rights - Strategy 3's entries stay manual.
        """
        if not self.manages_exits(signal.strategy_tag):
            return
        if breakeven is not None:
            try:
                self.bitget.place_tpsl_order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    plan_type="loss_plan",
                    trigger_price=breakeven,
                    client_oid=f"be-{signal.symbol}-{int(time.time() * 1000)}",
                )
            except Exception:
                logger.exception("Could not move %s's stop to breakeven", signal.symbol)
                await self.bot.send_message(
                    f"The partial filled on {signal.symbol} {signal.direction} but moving the stop to "
                    f"breakeven ({breakeven:g}) FAILED — move it by hand."
                )
        await self.place_runner_target(signal, plan, fallback)

    def _place_reduce_only(self, symbol: str, direction: str, size: float, price: float, kind: str) -> None:
        """One reduce-only exit, routed the way the symbol requires.

        RWA (tokenized-stock) symbols cap a resting limit to ~2% from mark, so
        an ordinary target is rejected outright - see _place_partial. A plan
        order's trigger is a condition rather than a resting price and is not
        bound by that band.
        """
        client_oid = f"{kind}-{symbol}-{int(time.time() * 1000)}"
        if self.bitget.get_contract_specs(symbol).get("is_rwa"):
            self.bitget.place_tpsl_order(
                symbol=symbol,
                direction=direction,
                plan_type="profit_plan",
                trigger_price=price,
                size=size,
                client_oid=client_oid,
            )
        else:
            self.bitget.place_order(
                symbol=symbol,
                direction=direction,
                size=size,
                order_type="limit",
                price=price,
                client_oid=client_oid,
                reduce_only=True,
            )

    async def _place_partial(self, signal: Signal, plan, position_size: float, replace: bool = False) -> None:
        """The first exit tier, at the plan's target.

        A plain reduce-only limit is capped at the exchange's own ~2% price
        band from mark on RWA (tokenized-stock) symbols - GOOGLUSDT's target
        was a perfectly ordinary ~3.7% from entry and Bitget rejected it
        outright. A TP plan order's trigger price is a condition rather than
        an order resting in the book right now, so it isn't bound by that
        band; RWA exits go through place_tpsl_order instead of place_order.

        Retries through PARTIAL_SETTLE_RETRY_DELAYS on Bitget's 22002 "No
        position to close" - see the constant's comment for why that's safe.
        Any other rejection is not retried: it isn't the settle race, so
        waiting longer won't fix it, and this path exists specifically to
        widen that one window rather than to retry blindly.
        """
        fraction = signal.partial_fraction if signal.partial_fraction is not None else PARTIAL_TAKE_FRACTION
        size = position_size * fraction
        if size <= 0:
            return

        if replace:
            # The position grew, so the old order covers too little of it.
            self._cancel_resting(signal.symbol, reduce_only_only=True, direction=signal.direction)

        is_rwa = bool(self.bitget.get_contract_specs(signal.symbol).get("is_rwa"))

        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0, *PARTIAL_SETTLE_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            try:
                if is_rwa:
                    self.bitget.place_tpsl_order(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        plan_type="profit_plan",
                        trigger_price=plan.take_profit,
                        size=size,
                        client_oid=f"tp-{signal.symbol}-{int(time.time() * 1000)}",
                    )
                else:
                    self.bitget.place_order(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        size=size,
                        order_type="limit",
                        price=plan.take_profit,
                        client_oid=f"tp-{signal.symbol}-{int(time.time() * 1000)}",
                        reduce_only=True,
                    )
                return
            except Exception as exc:
                last_exc = exc
                is_settle_race = "22002" in str(exc)
                logger.warning(
                    "Partial take-profit for %s rejected on attempt %d: %s", signal.symbol, attempt + 1, exc
                )
                if not is_settle_race:
                    break  # a different failure - waiting won't fix it, don't burn the retry budget

        logger.error("Could not place the partial take-profit for %s: %s", signal.symbol, last_exc)
        await self.bot.send_message(
            f"Placed {signal.symbol} {signal.direction} and its stop, but the partial take-profit "
            f"FAILED: {last_exc}\nThe position is protected but has no target — set one by hand."
        )

    def _cancel_resting(self, symbol: str, reduce_only_only: bool = False, direction: str | None = None) -> None:
        try:
            for open_order in self.bitget.get_open_orders(symbol):
                if reduce_only_only and (open_order.get("tradeSide") or "").lower() != "close":
                    continue
                self.bitget.cancel_order(symbol, order_id=open_order.get("orderId"))
        except Exception:
            logger.exception("Could not cancel resting orders for %s", symbol)

        if not (reduce_only_only and direction):
            return
        # An RWA take-profit lives as a plan order (place_tpsl_order), not on
        # the regular pending-orders book the loop above just cleared - a
        # growing position that skipped this would keep the OLD, now
        # under-sized target instead of getting a replacement sized to the
        # new total, alongside whatever _place_partial adds next.
        try:
            if self.bitget.get_contract_specs(symbol).get("is_rwa"):
                for plan_order in self.bitget.get_plan_orders(symbol, direction):
                    if plan_order["is_target"] and plan_order["order_id"]:
                        self.bitget.cancel_plan_order(
                            symbol, plan_order["plan_type"], order_id=plan_order["order_id"]
                        )
        except Exception:
            logger.exception("Could not cancel the resting take-profit plan order for %s", symbol)

    def _safe_stop_target(self, symbol: str, direction: str, position: dict):
        try:
            return self.bitget.get_stop_target(symbol, direction)
        except Exception:
            logger.exception("Could not read stop/target for %s; using position presets", symbol)
            return position["stop_loss"], position["take_profit"]

    def _on_trade_closed(self, trade_id: int, price: float) -> None:
        trade = self.storage.get_trade(trade_id)
        asyncio.create_task(self.bot.send_message(format_close_message(trade)))
        # Whatever is left resting - bot-placed or placed by hand off the
        # alert - belongs to a trade that is now over. This runs from here
        # rather than after track_position's await so it also fires when a
        # trade is re-attached by resume_open_trades after a restart, where
        # there is no "after the await" to fall back on.
        self._cancel_resting(trade.סימבול)

    def _on_scale_in(self, trade_id: int) -> None:
        """A resting entry leg filled, so the real position has arrived.

        Purely informational - no thresholds and no flagging of the risk
        against what was planned. The recomputed risk is in the message; if
        it looks wrong, that is a sizing bug to fix at the source rather than
        something for an alert to police.
        """
        trade = self.storage.get_trade(trade_id)
        try:
            covered = take_profit_coverage(
                self.bitget, trade.סימבול, trade.כיוון, trade.גודל_פוזיציה or 0.0
            )
        except Exception:
            # Worth sending without the coverage line rather than not at all:
            # the position figures are the point, coverage is the extra.
            logger.exception("Could not read take-profit coverage for %s", trade.סימבול)
            covered = None
        asyncio.create_task(self.bot.send_message(format_scale_in_message(trade, covered)))

    def _on_partial_exit(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        message = format_partial_message(self.storage.get_trade(trade_id), closed_size, realized_pnl)
        asyncio.create_task(self.bot.send_message(message))
