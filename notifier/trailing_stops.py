"""Trails the stop on a managed position that has no take-profit - the
runner's ONLY protection once it has neither a fixed target nor anything
overhead to aim at.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities. Needs a bar source (for reading structure to trail
against), the storage journal (which trades are open and what manages
them), Bitget (to place the trail and read the live target), and a way to
ask whether a strategy tag's exits are bot-managed at all.
"""

from __future__ import annotations

import logging
import time

from core import ledger
from core.storage import Storage
from notifier.bar_cache import BarCache
from notifier.strategies.base import TIMEFRAME_SECONDS
from notifier.strategies.indicators import atr
from notifier.strategies.structure import zigzag_pivots

logger = logging.getLogger(__name__)

TRAILING_POLL_TIMEFRAME = "1H"

# THE TRAIL'S OWN PIVOT SCALE.
#
# 2.0 is calibrated for finding DAILY runner TARGET levels (see
# notifier.scanner's RUNNER_LEVEL_PIVOT_ATR_MULTIPLE) and it was being reused
# as the pivot scale for the trailing stop on whatever timeframe the trade
# was taken on. Those are different questions: one asks "which levels are big
# enough to stop a move", the other "where did structure last turn".
#
# It also diverged from what was measured. The claim that trailing beats a
# fixed second target by ~0.045R per trade comes from
# score.simulate(runner="choch"), whose pivots are confirmed_pivots(...,
# multiple=1.25). The live trail ran at 2.0, so the measurement described a
# trail the bot does not have - the same shape of defect as scoring an entry
# at a price the order never fills at.
#
# Measured on the two live Strategy 2.1 trades of 2026-08-19, distance from
# price to the candidate stop:
#
#     UNIUSDT long    1.25x -> 0.88%     2.0x -> 3.74%
#     AIOUSDT short   1.25x -> 2.35%     2.0x -> 2.61%
#
# Four times looser on UNIUSDT. 1.25 is also ema_trend_v2.STRUCTURE_ATR_MULTIPLE
# and what Strategy 1 and Strategy 4 use, so "what counts as a swing" goes back
# to having one definition across the project.
TRAIL_PIVOT_ATR_MULTIPLE = 1.25
RUNNER_LEVEL_ATR_PERIOD = 14

# WHEN THE TRAIL HAS NOTHING TO MOVE TO, TAKE SOME OFF ANYWAY.
#
# The trail above ratchets to CONFIRMED swing lows only, which is right - an
# unconfirmed one is not a level yet. But a vertical run makes no swing at
# all, so the stop stays where it was while the open gain grows on top of
# it, and a single pullback gives all of it back. DOGEUSDT sat at +4.98R
# open with its stop at +0.24R on 2026-08-20; Dror closed half of it by hand.
#
# Measured as the GAP between the stop and price, in R, which states both of
# his conditions at once: if the trail were finding swings the gap would
# stay small by construction, so a wide gap IS "price ran and there was no
# low to move the stop to". Swept over 5,978 runner-phase 1H trades - the
# whole population rather than the 245 the shipped filters leave, since this
# rule is independent of the entry gates:
#
#     X     fires on   runner std   under +0.75R   cost
#    off        -         2.144          409         -
#    1.0       83%        1.273          191       0.040R
#    2.0       60%        1.331          252       0.032R
#    3.0       38%        1.435          301       0.027R
#    4.0       25%        1.524          333       0.021R   <- here
#    5.0       17%        1.606          357       0.014R
#
# THERE IS NO OPTIMUM HERE and the table should not be read as if there
# were. Dror chose 4.0 on 2026-08-20 with the table above in front of him.
# Do not re-derive it from a ratio - see notifier.scanner's git history for
# the fuller reasoning this constant was moved out of.
STALL_TIGHTEN_R = 4.0
STALL_TIGHTEN_FRACTION = 0.5

ALERT_THROTTLE_SECONDS = 24 * 3600


class TrailingStopManager:
    def __init__(self, bitget, storage: Storage, bot, bar_cache: BarCache, manages_exits):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        self.bar_cache = bar_cache
        # A callable (strategy_tag) -> bool, not the executor itself - this
        # class only ever needs the one question, and taking the callable
        # keeps it from depending on execution.Executor's full interface.
        self._manages_exits = manages_exits
        # trade_id -> when a reminder about a naked (no live take-profit)
        # scanner-default position last went out. In memory only: a restart
        # re-reminding early is the safe side of this failure - the opposite
        # of a case durable state was built for - a naked position that stops
        # being mentioned is the actual danger.
        self._naked_reminded: dict[int, float] = {}

    def upkeep_timeframe(self) -> str:
        """The frame the upkeep loop should wake on: the fastest one any
        trade it manages is actually trailed against.

        Read from open trades each pass rather than fixed at startup, so the
        cadence follows what is on the book - a 15m runner speeds it up
        while it exists and nothing else pays for it. A failed read falls
        back rather than stalling the loop, on the same reasoning as every
        other poll in this project.
        """
        try:
            frames = {
                self.trail_timeframe(t.תגית_אסטרטגיה or "")
                for t in self.storage.open_trades()
                if self._manages_exits(t.תגית_אסטרטגיה or "")
            }
        except Exception:
            logger.exception("Could not read open trades for the upkeep cadence")
            frames = set()
        frames.discard(None)
        if not frames:
            return TRAILING_POLL_TIMEFRAME
        # The shortest period, not the soonest close - the loop must keep up
        # with the fastest structure, not merely with whatever closes next.
        return min(frames, key=lambda tf: TIMEFRAME_SECONDS.get(tf, 10**9))

    def trail_timeframe(self, strategy_tag: str) -> str:
        """The frame whose swings a trade's stop should trail.

        A strategy's own structural timeframe, read off its tag - "Strategy
        3 1D/1H" trails daily swings, "Strategy 2 1H/15m" hourly ones.
        Trailing a multi-day setup on 5m lows would ratchet the stop into
        the first bit of noise, and trailing an intraday one on daily lows
        would never move at all.

        Note this reads the STRUCTURE half of the pair, not the trigger:
        both Strategy 3 instances are 1D/... and both therefore trail daily
        swings, including the one that enters on a 5m close.
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
        short) - that is the "as long there is rising highs" half of the
        rule, and without it a stop would keep ratcheting into a topping
        market. The new stop is the last CONFIRMED swing low, never an
        unconfirmed one, and it is only ever returned when it improves on
        the current stop.
        """
        bars = self.bar_cache.get(symbol, self.trail_timeframe(strategy_tag))
        thresholds = atr(bars, RUNNER_LEVEL_ATR_PERIOD) * TRAIL_PIVOT_ATR_MULTIPLE
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

    def stall_tighten(self, trade, price: float, stop: float | None = None) -> float | None:
        """Where the stop belongs when the trail has nothing to move to, else None.

        Measured as the stop-to-price gap in R against the trade's ORIGINAL
        risk, so R means the same thing here as everywhere else. That single
        number states both of Dror's conditions at once: a trail that was
        finding swings would keep the gap small, so a wide gap IS "price ran
        and there was no low to move the stop to". See STALL_TIGHTEN_R.
        """
        entry, original_stop = trade.מחיר_כניסה, trade.סטופ_לוס_מקורי
        # `stop` is passed by the poll so the gap is measured from where the
        # stop is AFTER the trail has had its turn - measuring from the
        # older, looser stop would fire on a trade the trail had just
        # protected.
        stop = trade.סטופ_לוס_בפועל if stop is None else stop
        if entry is None or original_stop is None or stop is None:
            return None
        risk = abs(float(entry) - float(original_stop))
        if risk <= 0:
            return None
        stop, price = float(stop), float(price)
        sign = 1.0 if trade.כיוון == "long" else -1.0
        if (price - stop) * sign / risk < STALL_TIGHTEN_R:
            return None
        # Correct for both directions: on a short (price - stop) is negative,
        # so the stop comes DOWN toward price.
        return stop + (price - stop) * STALL_TIGHTEN_FRACTION

    async def poll(self) -> None:
        """Trail the stop on any managed position that has no target.

        Having no target is precisely the case this exists for: the runner
        could not be given one because nothing was found overhead, so the
        stop is the only thing left to manage. A position WITH a target is
        left alone - it already has a defined exit.

        Placed as a position-level pos_loss with size 0 ("all closable") for
        the same two reasons as the breakeven move (see ExitManager): a
        specific size gets a 40019 rejection, and position-level means each
        trail REPLACES the previous stop rather than stacking another
        loss_plan on the book.
        """
        for trade in self.storage.open_trades():
            tag = trade.תגית_אסטרטגיה or ""
            if not self._manages_exits(tag):
                continue
            if trade.partial_fraction is None:
                # A scanner-default exit (Strategy 1's shape) NEVER wants a
                # trail: runner_target() always hands it a real ratio-derived
                # price when the signal carried no partial_fraction of its
                # own, so a live target that is missing here can only mean
                # placement failed - e.g. SPCXUSDT #37's partial fell under
                # Bitget's $5 minimum notional and was skipped outright. That
                # is a "no exit is protecting this position" problem, and
                # trailing the stop as if it were the plan papered over it
                # instead of fixing it. Leave it alone, but keep reminding
                # Dror it needs a hand-placed target.
                await self._remind_if_naked(trade)
                continue
            symbol, direction = trade.סימבול, trade.כיוון
            try:
                _, target = self.bitget.get_stop_target(symbol, direction)
                if target is not None:
                    continue  # it has a defined exit; nothing to trail toward
                new_stop = self.trailing_stop(symbol, direction, tag, trade.סטופ_לוס_בפועל)
                # The trail gets its turn first; only then is the remaining gap
                # measured, so this fires on the runs that left it with nothing
                # to move to rather than on the ones it just protected.
                effective = new_stop if new_stop is not None else trade.סטופ_לוס_בפועל
                tightened = None
                if effective is not None:
                    tightened = self.stall_tighten(
                        trade, float(self.bitget.get_mark_price(symbol)), stop=effective
                    )
                stalled = tightened is not None
                new_stop = tightened if stalled else new_stop
                if new_stop is None:
                    continue
                self.bitget.place_tpsl_order(
                    symbol=symbol,
                    direction=direction,
                    plan_type="pos_loss",
                    trigger_price=new_stop,
                    size=0,  # all closable - see the docstring
                    client_oid=f"trail-{symbol}-{int(time.time() * 1000)}",
                )
            except Exception:
                logger.exception("Could not trail the stop on %s", symbol)
                continue

            self.storage.update_actual_stop_target(trade.מספר_עסקה, new_stop, None)
            ledger.try_record(self.storage.db_path, ledger.TRAILING_STOP_MOVED)
            if stalled:
                await self.bot.send_message(
                    f"Pulled the stop in on {symbol} {direction} ({tag}) to {new_stop:g} — "
                    f"there was no new {self.trail_timeframe(tag)} swing to trail to and the "
                    f"stop had fallen more than {STALL_TIGHTEN_R:g}R behind the price, so the "
                    f"gap was halved rather than left riding."
                )
            else:
                await self.bot.send_message(
                    f"Trailed the stop on {symbol} {direction} ({tag}) up to {new_stop:g} — "
                    f"the last confirmed {self.trail_timeframe(tag)} swing "
                    f"{'low' if direction == 'long' else 'high'}, while structure keeps making "
                    f"{'higher highs' if direction == 'long' else 'lower lows'}."
                )

    async def _remind_if_naked(self, trade) -> None:
        """Nag, at most once a rolling day, about a scanner-default position
        with no live take-profit - the exact state poll() used to silently
        convert into an unintended trail. _place_partial's own skip message
        fires once, at entry; without this, a position stuck naked past that
        first message is never mentioned again.
        """
        try:
            _, target = self.bitget.get_stop_target(trade.סימבול, trade.כיוון)
        except Exception:
            logger.exception("Could not check %s's target for the naked-position reminder", trade.סימבול)
            return
        if target is not None:
            self._naked_reminded.pop(trade.מספר_עסקה, None)
            return
        last = self._naked_reminded.get(trade.מספר_עסקה)
        if last is not None and time.time() - last < ALERT_THROTTLE_SECONDS:
            return
        self._naked_reminded[trade.מספר_עסקה] = time.time()
        await self.bot.send_message(
            f"{trade.סימבול} {trade.כיוון} ({trade.תגית_אסטרטגיה}) still has NO take-profit on "
            f"the exchange - it was skipped as too small for Bitget's minimum and nothing has "
            f"replaced it. Only the stop protects this position; set a target by hand if you want one."
        )
