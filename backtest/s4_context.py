"""The raw material of a Strategy 4 setup, separated from the trade built out of it.

WHY THIS EXISTS
  Entry, stop and target are all computed inside _signal_for at GENERATION
  time, and the R:R gates decide whether a signal exists at all. So naively,
  sweeping ENTRY_FRACTION or GAP_TARGET_FRACTION costs a full ~6.5h detector
  run per arm - one arm per night.

  Nearly all of that 6.5h is the detector: structure_context's adaptive window,
  zigzag_pivots, find_expansions, _find_blocks. None of it depends on the
  parameters being swept. What does depend on them is pure arithmetic over a
  block and a list of gaps.

  So: run the detector ONCE, record (blocks, gaps, price, atr, equilibrium)
  per setup, and rebuild the trade per arm from that. Sweeps become minutes.
  This is the technique that made Strategy 1's Fib entry/stop sweepable.

WHAT IS RECORDED AND WHAT IS DEFERRED
  Recorded: everything non-parametric - which blocks exist, which gaps are
  still open, how big each gap is in ATR at its own end bar, whether the block
  formed in the Asia session.
  Deferred to the arm: ENTRY_FRACTION, GAP_TARGET_FRACTION, MIN_GAP_ATR,
  STOP_ATR_BUFFER, MIN/MAX_REWARD_RISK, MAX_STOP_PCT, the fee gate and the
  session gate.

  ALL candidate blocks are recorded, not just the one that signalled. _signal_for
  takes the most recent block that passes every gate, so an arm whose R:R band
  rejects the newest block must be free to fall through to an older one -
  recording only the winner would silently change that semantics.

DRIFT IS THE RISK. build_signal() below duplicates _signal_for's arithmetic,
so the two can disagree. tests/test_s4_context.py pins them together: at
baseline parameters, build_signal() must reproduce _signal_for's Signal on
real bars, field for field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from notifier.strategies import order_block as ob
from notifier.strategies.base import TIMEFRAME_SECONDS, Signal


@dataclass(slots=True)
class GapCtx:
    """A target candidate. `atr_at_end` is ATR at the gap's own end bar, which
    is what MIN_GAP_ATR is measured against - not ATR now."""
    direction: str
    low: float
    high: float
    size: float
    atr_at_end: float
    start_index: int
    end_index: int


@dataclass(slots=True)
class BlockCtx:
    low: float
    high: float
    direction: str
    variant: str
    index: int
    displacement_end: int
    sweep_level: float
    sweep_extreme: float
    in_asia: bool


@dataclass(slots=True)
class SetupCtx:
    """One bar on one symbol at which at least one block was live."""
    symbol: str
    ts: int
    bar_index: int
    close: float
    price: float
    atr_now: float
    equilibrium: float
    blocks: list = field(default_factory=list)   # most-recent-first
    gaps: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Params:
    """One sweep arm. Defaults are exactly what ships today."""
    entry_fraction: float = ob.ENTRY_FRACTION
    gap_target_fraction: float = ob.GAP_TARGET_FRACTION
    min_gap_atr: float = ob.MIN_GAP_ATR
    stop_atr_buffer: float = ob.STOP_ATR_BUFFER
    min_reward_risk: float = ob.MIN_REWARD_RISK
    max_reward_risk: float = ob.MAX_REWARD_RISK
    max_stop_pct: float = ob.MAX_STOP_PCT
    asia_gated: bool = False

    def label(self) -> str:
        return (f"entry={self.entry_fraction:g} tgt={self.gap_target_fraction:g} "
                f"gap>={self.min_gap_atr:g} stop={self.stop_atr_buffer:g} "
                f"R:R[{self.min_reward_risk:g},{self.max_reward_risk:g}]"
                + (" noAsia" if self.asia_gated else ""))


def _entry(block: BlockCtx, f: float) -> float:
    """f=0.5 is the block midpoint (what ships). f=0 is the NEAR edge - the
    side price reaches first, so it fills earlier and at a worse price."""
    near, far = (block.high, block.low) if block.direction == "short" else (block.low, block.high)
    return near + (far - near) * f


def _target_price(gap: GapCtx, f: float, direction: str) -> float:
    """f=0.5 is the gap midpoint (what ships). f=0 is the near edge - the
    "start" of the zone, reached first and therefore hit far more often."""
    near, far = (gap.high, gap.low) if direction == "short" else (gap.low, gap.high)
    return near + (far - near) * f


def pick_target(ctx: SetupCtx, block: BlockCtx, entry: float, p: Params) -> float | None:
    """_target's rule, with MIN_GAP_ATR and the fraction as parameters.

    Unchanged from the shipped rule: opposite-direction gaps only, never the
    setup's own displacement gap, never one already closed, and the gap must
    sit entirely beyond current price on the far side from the entry - a gap
    between price and the entry is filled on the way in and is not a target.
    Nearest survivor wins.
    """
    want = "down" if block.direction == "long" else "up"
    best = None
    for gap in ctx.gaps:
        if gap.direction != want:
            continue
        if block.index <= gap.start_index <= block.displacement_end:
            continue
        if gap.size < p.min_gap_atr * gap.atr_at_end:
            continue
        price = ctx.price
        mid = _target_price(gap, p.gap_target_fraction, block.direction)
        if block.direction == "long":
            if gap.low > price and mid > entry:
                best = mid if best is None else min(best, mid)
        else:
            if gap.high < price and mid < entry:
                best = mid if best is None else max(best, mid)
    return best


def build_signal(ctx: SetupCtx, p: Params, timeframe: str, tag_prefix: str = "Strategy 4"):
    """The trade this setup produces under these parameters, or None.

    Mirrors _signal_for's gate ORDER exactly, and walks the blocks
    most-recent-first taking the first that survives - which is what
    evaluate() does.
    """
    for block in ctx.blocks:
        if p.asia_gated and block.in_asia:
            continue
        entry = _entry(block, p.entry_fraction)
        # Premium/discount, measured at the price actually transacted.
        if block.direction == "long" and entry >= ctx.equilibrium:
            continue
        if block.direction == "short" and entry <= ctx.equilibrium:
            continue
        # Price must still be outside the block - it is untested, so it is.
        near = block.high if block.direction == "short" else block.low
        if block.direction == "long" and ctx.price <= near:
            continue
        if block.direction == "short" and ctx.price >= near:
            continue

        buffer = ctx.atr_now * p.stop_atr_buffer
        stop = (block.sweep_extreme - buffer if block.direction == "long"
                else block.sweep_extreme + buffer)
        risk = entry - stop if block.direction == "long" else stop - entry
        if risk <= 0:
            continue
        if abs(entry - stop) / entry > p.max_stop_pct:
            continue
        if ob._fee_fraction_of_risk(entry, stop) > ob.MAX_FEE_FRACTION_OF_RISK:
            continue

        target = pick_target(ctx, block, entry, p)
        if target is None:
            continue
        reward = target - entry if block.direction == "long" else entry - target
        ratio = reward / risk
        if not (p.min_reward_risk <= ratio <= p.max_reward_risk):
            continue

        return Signal(
            symbol=ctx.symbol,
            direction=block.direction,
            entry_price=entry,
            stop_loss=stop,
            strategy_tag=f"{tag_prefix} {timeframe} {block.variant}",
            reward_risk_ratio=ratio,
            limit_entry=entry,
            limit_note=f"order block {p.entry_fraction:g}",
            market_fraction=0.0,
            partial_fraction=1.0,
            unfilled_timeout_seconds=ob.UNFILLED_CANDLES * TIMEFRAME_SECONDS[timeframe],
            reason=(f"{block.variant} on the {timeframe}: block "
                    f"{block.low:g}-{block.high:g}, entry at its "
                    f"{p.entry_fraction:g}, target {target:.8g} ({ratio:.1f}R)."),
        )
    return None
