"""One year of the WHOLE ACCOUNT, not one strategy at a time.

paper_sim resolves each signal in isolation: no equity, no position limits, no
fees compounding, every signal taken regardless of whether the account could
have afforded it. That answers "was this signal any good"; it cannot answer
"what would the account have done", which is a portfolio question. This does
the portfolio.

WHAT IS MODELLED
  - equity compounding from a starting balance, every fill and exit applied
  - one position per symbol (Scanner.already_exposed)
  - the 6% aggregate open-risk cap, and the 2-slot swing pool
  - per-trade sizing through the real risk_sizing.plan_position, including the
    leverage solve against margin actually free
  - Bitget's $5 minimum notional PER LEG, valued at the rounded size - the
    constraint that declined 9 signals in one real week
  - taker 0.06% / maker 0.02% on every leg, entry and exit
  - split entries: market leg at the signal bar's close, limit leg only if
    price actually trades through it
  - two-tier exits where the strategy sets one (partial, stop to breakeven,
    remainder), single exit where it does not

ASSUMPTIONS, all of which flatter the result and none of which are avoidable
  - EVERY SIGNAL IS APPROVED. The bot only trades on Dror's tap, and there is
    no way to know which he would have taken. A year where he rejected the bad
    ones would look better; one where he missed the good ones, worse.
  - The runner is mechanical here. In reality "close at resistance or after 3
    days" is his judgement, and the weekly report says his real trades beat the
    paper simulation of the same signals.
  - A bar that touches both stop and target counts as a STOP. Nothing in OHLC
    says which came first and the optimistic reading is how backtests lie.
  - No slippage beyond the fee model, and limit fills assume the whole size
    fills the moment price touches - generous on thin alts.
"""
import os
import pickle
from dataclasses import dataclass, field

import pandas as pd

from core.bitget_client import BitgetClient
from notifier.risk_sizing import plan_position
from notifier.scanner import bars_dataframe
from notifier.strategies.base import TIMEFRAME_SECONDS

# Beside the trades DB and gitignored with it: tens of MB of downloaded bars
# are a cache, not source. BACKTEST_BARS points a run at a cache built
# elsewhere without having to copy it.
CACHE = os.getenv("BACKTEST_BARS", os.path.join("data", "backtest_bars.pkl"))

TAKER, MAKER = 0.0006, 0.0002
START_EQUITY = 100.0
RISK_PCT = 0.01
MAX_TOTAL_RISK_PCT = 0.06
MAX_LEVERAGE = 20.0
MIN_NOTIONAL = 5.0
SWING_TAGS = {"Strategy 1 1D", "Strategy 2 1D"}
MAX_SWING_SLOTS = 2
REMAINDER_RATIO = 3.0
PARTIAL_DEFAULT = 0.5
# "" (baseline, today's behaviour) | "limit" | "market". See try_open.
SPLIT_FALLBACK = os.getenv("SPLIT_FALLBACK", "").strip().lower()


@dataclass
class Position:
    symbol: str
    tag: str
    direction: str
    entry: float           # blended cost basis of what has filled
    size: float            # base units filled so far
    stop: float
    target: float
    remainder_target: float | None
    partial_fraction: float
    opened_at: int
    risk_amount: float     # dollars at risk when opened, for the 6% cap
    margin: float
    pending_size: float = 0.0   # unfilled limit leg
    pending_price: float = 0.0
    pending_until: int = 0      # bar index the resting leg is cancelled at
    took_partial: bool = False
    # P&L already BANKED by the partial. Without this a trade that took its
    # partial and then stopped at breakeven recorded only the final leg - i.e.
    # ~0R and a "loss" - while the money sat in equity. It made every
    # partial-then-breakeven trade invisible in the R stats and understated
    # the taken side against the refused one.
    realised_pnl: float = 0.0
    # Structure trailing for a remainder with no stated target. Live, such a
    # remainder is trailed by scanner.poll_trailing_stops onto the last
    # confirmed swing; it is not aimed at a fixed multiple. Pivots are
    # [(confirmed_at_1h_bar, price, is_high)], each dated by the 1H bar on which
    # it could first have been KNOWN - see score.confirmed_pivots.
    pivots: list = field(default_factory=list)
    pivot_cursor: int = 0
    trailing: bool = False
    limit_filled: bool = False


@dataclass
class Closed:
    symbol: str
    tag: str
    direction: str
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    r: float
    pnl: float
    reason: str
    # Did the resting entry leg ever fill? A split entry that fills both legs
    # is price COMING BACK to the level; one whose limit expires unfilled is
    # price running away from it. Those are different market states, and the
    # live record hints they have different outcomes - Strategy 1's two
    # market-only trades both won while its two largest both-legs trades both
    # lost. n=12 cannot settle that; this is what lets the replay try.
    limit_filled: bool = False


@dataclass
class Account:
    equity: float = START_EQUITY
    open_positions: dict = field(default_factory=dict)   # symbol -> Position
    closed: list = field(default_factory=list)
    declined_too_small: int = 0
    declined_risk_cap: int = 0
    declined_exposed: int = 0
    declined_swing: int = 0
    taken: int = 0
    peak: float = START_EQUITY
    max_dd: float = 0.0
    # (symbol, direction, entry, stop, target, tag, bar_index) for every signal
    # the $5-per-leg floor refused. Scored afterwards against the same bars.
    too_small: list = field(default_factory=list)
    # Stop distance as a share of entry, for taken vs refused. The $5 floor
    # bites when notional is small, and notional = (risk_pct / stop_pct) x
    # equity - so it selects on stop WIDTH, not on quality. These two lists
    # are what turns that from a story into a measurement.
    taken_stop_pct: list = field(default_factory=list)
    refused_stop_pct: list = field(default_factory=list)
    # Trades that would have been refused under the split and were taken as a
    # single leg instead. Their stop widths are the interesting half: if the
    # fallback is working the way the arithmetic says, these are much wider
    # than the ordinary taken set.
    rescued: int = 0
    rescued_stop_pct: list = field(default_factory=list)

    def committed_margin(self) -> float:
        return sum(p.margin for p in self.open_positions.values())

    def open_risk(self) -> float:
        return sum(p.risk_amount for p in self.open_positions.values())

    def mark(self) -> None:
        self.peak = max(self.peak, self.equity)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - self.equity) / self.peak)


def load_bars(symbols, timeframes, bars_per_tf):
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    client = BitgetClient()
    fetched = 0
    for i, symbol in enumerate(symbols, 1):
        for tf in timeframes:
            if (symbol, tf) in cache:
                continue
            try:
                cache[(symbol, tf)] = bars_dataframe(
                    client.get_candles(symbol, granularity=tf, limit=bars_per_tf[tf], closed_only=True)
                )
            except Exception:
                cache[(symbol, tf)] = None
            fetched += 1
        if i % 10 == 0:
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"  fetched {i}/{len(symbols)} symbols", flush=True)
    pickle.dump(cache, open(CACHE, "wb"))
    return cache


def round_size(size: float, step: float) -> float:
    if step <= 0:
        return size
    return int(size / step) * step


def _fee(notional: float, maker: bool) -> float:
    return notional * (MAKER if maker else TAKER)


def try_open(acct: Account, signal, bar_close: float, bar_index: int, specs, cancel_after: int,
             pivots: list | None = None) -> bool:
    """Everything the Scanner checks before an order is placed, in the same order.

    `pivots` are the confirmed swings this position would trail on if its
    remainder has no stated target, in the 1H bar indexing this engine steps on.
    Omitting them leaves the remainder pinned at breakeven, which understates a
    trailing runner - so a caller replaying a remainder_target=None strategy
    should pass them.
    """
    if signal.symbol in acct.open_positions:
        acct.declined_exposed += 1
        return False

    if signal.strategy_tag in SWING_TAGS:
        swing_open = sum(1 for p in acct.open_positions.values() if p.tag in SWING_TAGS)
        if swing_open >= MAX_SWING_SLOTS:
            acct.declined_swing += 1
            return False

    market_fraction = signal.market_fraction
    limit_entry = signal.limit_entry
    split = limit_entry is not None and market_fraction > 0
    plan_entry = (
        market_fraction * bar_close + (1 - market_fraction) * limit_entry
        if split
        else signal.entry_price
    )
    ratio = signal.reward_risk_ratio if signal.reward_risk_ratio is not None else REMAINDER_RATIO
    risk_pct = signal.risk_pct_override if signal.risk_pct_override is not None else RISK_PCT
    risk_pct = min(risk_pct, 0.02)

    budget = acct.equity - acct.committed_margin()
    if budget <= 0:
        acct.declined_risk_cap += 1
        return False

    # The $5 floor applies PER LEG, so a split entry can be refused for a trade
    # whose TOTAL notional clears it comfortably - and since notional is
    # (risk% / stop%) x equity, the ones cut are the WIDE-stop trades. With
    # SPLIT_FALLBACK set, such a trade collapses onto a single leg at identical
    # risk, which is what lets the refused population be measured through the
    # same fill mechanism the taken ones go through instead of scored in
    # isolation by score_too_small (which grants a guaranteed full fill at the
    # blended price and so flatters them).
    candidates = [(market_fraction, limit_entry, plan_entry)]
    if split and SPLIT_FALLBACK == "limit":
        candidates.append((0.0, limit_entry, limit_entry))
    elif split and SPLIT_FALLBACK == "market":
        candidates.append((1.0, None, bar_close))

    chosen = base_plan = None
    for cand_frac, cand_limit, cand_entry in candidates:
        try:
            plan = plan_position(
                equity=acct.equity, risk_pct=risk_pct, entry_price=cand_entry,
                stop_loss=signal.stop_loss, direction=signal.direction,
                reward_risk_ratio=ratio, available_budget=budget, max_leverage=MAX_LEVERAGE,
            )
        except ValueError:
            acct.declined_risk_cap += 1
            return False
        if base_plan is None:
            base_plan = plan

        if acct.open_risk() + plan.risk_amount > acct.equity * MAX_TOTAL_RISK_PCT:
            acct.declined_risk_cap += 1
            return False

        step = specs["step"]
        market_size = round_size(plan.position_size * cand_frac, step)
        limit_size = round_size(plan.position_size * (1 - cand_frac), step)
        legs = []
        if cand_frac > 0:
            legs.append(market_size * bar_close)
        if cand_limit is not None and (1 - cand_frac) > 0:
            legs.append(limit_size * cand_limit)
        if legs and all(v >= MIN_NOTIONAL for v in legs):
            chosen = (cand_frac, cand_limit, cand_entry, plan, market_size, limit_size)
            break

    if chosen is not None:
        if chosen[0] != market_fraction:
            acct.rescued += 1
            acct.rescued_stop_pct.append(abs(chosen[2] - signal.stop_loss) / chosen[2])
        market_fraction, limit_entry, plan_entry, plan, market_size, limit_size = chosen
    else:
        plan = base_plan
        acct.declined_too_small += 1
        # Recorded so the cost of the $5 floor can be measured rather than
        # assumed. These are the signals a bigger account would have taken;
        # scoring them says whether the constraint is protecting Dror or
        # quietly costing him the year's edge. Resolved on the same convention
        # as journal/paper_sim so the number is comparable to the weekly
        # report's "too small to execute" line.
        acct.refused_stop_pct.append(abs(plan_entry - signal.stop_loss) / plan_entry)
        acct.too_small.append((signal.symbol, signal.direction, plan_entry,
                               signal.stop_loss, plan.take_profit, signal.strategy_tag, bar_index,
                               signal.partial_fraction if signal.partial_fraction is not None else PARTIAL_DEFAULT,
                               signal.remainder_target))
        return False

    filled_size = market_size if market_fraction > 0 else 0.0
    entry_basis = bar_close
    acct.equity -= _fee(filled_size * bar_close, maker=False)

    pos = Position(
        symbol=signal.symbol, tag=signal.strategy_tag, direction=signal.direction,
        entry=entry_basis if filled_size else limit_entry,
        size=filled_size, stop=signal.stop_loss, target=plan.take_profit,
        remainder_target=signal.remainder_target,
        partial_fraction=signal.partial_fraction if signal.partial_fraction is not None else PARTIAL_DEFAULT,
        opened_at=bar_index, risk_amount=plan.risk_amount, margin=plan.required_margin,
        pending_size=limit_size if limit_entry is not None and (1 - market_fraction) > 0 else 0.0,
        pending_price=limit_entry or 0.0,
        pending_until=bar_index + cancel_after,
        pivots=pivots or [],
        # Start PAST every swing already confirmed when the trade opened.
        # Without this the ratchet takes max() over the symbol's whole history
        # of swing lows, which is routinely far above the entry: the stop lands
        # above the market, step_position closes there, and books the gap as
        # profit. It read +27565% on a 30-day window with a single +164R trade
        # before this line existed. score.simulate has always skipped them -
        # `while pivots[cursor][0] <= start: cursor += 1` - and the two must
        # agree or the portfolio and the scorer describe different trades.
        pivot_cursor=sum(1 for at, _p, _h in (pivots or []) if at <= bar_index),
    )
    acct.open_positions[signal.symbol] = pos
    acct.taken_stop_pct.append(abs(plan_entry - signal.stop_loss) / plan_entry)
    acct.taken += 1
    return True


def step_position(acct: Account, pos: Position, bar, bar_index: int, ts) -> bool:
    """Advance one position by one bar. Returns True when it closed."""
    hi, lo = float(bar["high"]), float(bar["low"])
    long = pos.direction == "long"

    # A resting entry leg fills when price trades through it.
    if pos.pending_size > 0:
        touched = lo <= pos.pending_price if long else hi >= pos.pending_price
        if touched:
            new_size = pos.size + pos.pending_size
            pos.entry = ((pos.entry * pos.size) + (pos.pending_price * pos.pending_size)) / new_size if pos.size else pos.pending_price
            pos.size = new_size
            acct.equity -= _fee(pos.pending_size * pos.pending_price, maker=True)
            pos.pending_size = 0.0
            pos.limit_filled = True
        elif bar_index >= pos.pending_until:
            pos.pending_size = 0.0  # cancelled unfilled
            if pos.size == 0:
                del acct.open_positions[pos.symbol]
                return True

    if pos.size == 0:
        return False

    # Ratchet onto any swing CONFIRMED by this bar, before the stop is checked -
    # the level has to be knowable before it can protect anything. max/min never
    # loosens the stop, so it can only travel away from the entry, and the
    # breakeven floor set when the partial filled always holds.
    if pos.trailing:
        while pos.pivot_cursor < len(pos.pivots) and pos.pivots[pos.pivot_cursor][0] <= bar_index:
            _at, price, is_high = pos.pivots[pos.pivot_cursor]
            pos.pivot_cursor += 1
            if long and not is_high:
                pos.stop = max(pos.stop, price)
            elif not long and is_high:
                pos.stop = min(pos.stop, price)

    hit_stop = lo <= pos.stop if long else hi >= pos.stop
    hit_target = hi >= pos.target if long else lo <= pos.target

    if hit_stop:  # checked first: a bar spanning both is scored a loss
        pnl = (pos.stop - pos.entry) * pos.size * (1 if long else -1)
        acct.equity += pnl - _fee(pos.size * pos.stop, maker=False)
        _close(acct, pos, ts, pos.realised_pnl + pnl, "stop")
        return True

    if hit_target and not pos.took_partial:
        part = pos.size * pos.partial_fraction
        pnl = (pos.target - pos.entry) * part * (1 if long else -1)
        acct.equity += pnl - _fee(part * pos.target, maker=True)
        pos.realised_pnl += pnl
        pos.size -= part
        pos.took_partial = True
        pos.stop = pos.entry  # to breakeven, as the alert instructs
        if pos.size <= 1e-12:
            _close(acct, pos, ts, pos.realised_pnl, "target")
            return True
        if pos.remainder_target is None:
            # No stated runner target means the live bot TRAILS this remainder
            # on structure (scanner.poll_trailing_stops only trails positions
            # with no target), so there is no second price to aim at. Modelling
            # it as a fixed multiple was wrong twice over: wrong mechanism, and
            # the multiple was miscomputed. `risk` read
            #     abs(pos.entry - pos.stop) or abs(pos.entry - pos.target) / 1.0
            # three lines after pos.stop was set to pos.entry, so the first term
            # was ALWAYS 0 and the fallback always fired - and the fallback
            # measures the distance to target 1, i.e. r1 x risk. The runner was
            # therefore sent to 3R x r1: at the shipping r1 of 2.0, 6R instead
            # of 3R. That biased every remainder_target=None strategy
            # PESSIMISTIC, because a runner aimed twice as far mostly came back
            # to the breakeven stop instead of paying.
            pos.trailing = True
            pos.target = float("inf") if long else float("-inf")
        else:
            pos.target = pos.remainder_target
        return False

    if hit_target and pos.took_partial:
        pnl = (pos.target - pos.entry) * pos.size * (1 if long else -1)
        acct.equity += pnl - _fee(pos.size * pos.target, maker=True)
        _close(acct, pos, ts, pos.realised_pnl + pnl, "runner")
        return True
    return False


def _close(acct: Account, pos: Position, ts, pnl: float, reason: str) -> None:
    risk = pos.risk_amount or 1e-9
    acct.closed.append(Closed(pos.symbol, pos.tag, pos.direction, pos.opened_at, ts,
                              pnl / risk, pnl, reason, pos.limit_filled))
    acct.open_positions.pop(pos.symbol, None)
    acct.mark()


def score_too_small(entries, bars):
    """What the signals the $5 floor refused would have returned.

    Scored with the SAME two-tier exit the taken trades get - partial at the
    target, stop to breakeven, remainder to the runner - because otherwise the
    comparison is rigged. A simple stop-vs-full-target model pays every winner
    full R while the real trades give half of it back at the partial, and the
    first version of this reported +1.06R against the portfolio's -0.13R
    largely for that reason alone.

    No portfolio, though: these never competed for margin or slots. This is
    what a big enough account would have collected, not what THIS one could.
    """
    out = []
    for symbol, direction, entry, stop, target, tag, i, partial_fraction, remainder_target in entries:
        long = direction == "long"
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        fee_r = (2 * TAKER * entry) / risk
        realised, remaining, took_partial = 0.0, 1.0, False
        cur_stop, cur_target = stop, target
        for j in range(i + 1, len(bars)):
            hi, lo = float(bars["high"].iloc[j]), float(bars["low"].iloc[j])
            hit_stop = lo <= cur_stop if long else hi >= cur_stop
            hit_target = hi >= cur_target if long else lo <= cur_target
            if hit_stop:
                realised += remaining * (cur_stop - entry) / risk * (1 if long else -1)
                remaining = 0.0
                break
            if hit_target:
                part = remaining if took_partial else remaining * partial_fraction
                realised += part * abs(cur_target - entry) / risk
                remaining -= part
                if remaining <= 1e-12:
                    break
                took_partial = True
                cur_stop = entry  # to breakeven, exactly as the alert says
                cur_target = (remainder_target if remainder_target is not None
                              else entry + risk * REMAINDER_RATIO * (1 if long else -1))
        if remaining < 1.0:  # something resolved
            out.append((tag, realised - fee_r))
    return out
