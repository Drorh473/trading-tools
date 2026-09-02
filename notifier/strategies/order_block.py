"""Strategy 4 from the user's cheatsheet: order blocks, entered on a
retracement back into the block, targeting the next gap that has not yet been
filled.

The thesis is about unfilled orders. A gap ("גאפ") is a zone the market moved
through so fast that orders resting inside it never got filled, so price tends
to be drawn back to it - which makes it both a place price returns to and a
place to take profit. An order block is the candle that was holding those
orders just before the market expanded away: the last candle against the move,
sitting where the imbalance began.

TWO VERSIONS, DELIBERATELY DIFFERENT CANDLES. Dror's cheatsheet describes them
separately and says 2.0 works better statistically, so both are built and each
signal is tagged with the version that found it - the weekly report settles
which actually pays rather than the belief doing it.

  OB 1.0 is anchored on the GAP: "זהו הנר שלפני הגאפ" - the block is the candle
  immediately before a gap, i.e. the first candle of the three-candle imbalance.
  It additionally wants a structure break, a reversal candle, and (preferably,
  not mandatorily) not to have formed in the Asia session.

  OB 2.0 is anchored on the EXPANSION: "זהו הנר שלפני מהלך שמרחיב את השוק" - the
  block is the candle before the move that expands the market. No gap is
  required, so a displacement that left no imbalance is still a 2.0 and never a
  1.0. The two are genuinely different sets, not nested.

Where they agree: liquidity must have been taken in the direction the market
came from (the sweep that fuels the reversal), the block must be untested since
it formed, and it must sit in discount for a long or premium for a short.

WHAT MAKES THE EXPANSION AN EXPANSION. Deliberately close to the flag pole in
patterns.py, which Dror reviewed over four passes and corrected: measured on
candle BODIES, not wicks ("most of the pole is wick and we dont want a big wick
in the pole"), sized at 4x ATR but capped at 35% of price so a symbol whose ATR
is 12% of its own price is not asked for a move no candle in its history could
make. One difference, and it is his: an order block's expansion may run several
candles where a flag pole is 1-2, so the bar cap is gone. Removing it puts back
what the pole could delete - a steepness floor. Over 2 bars a 4x ATR move is at
worst 2 ATR/bar and no floor could ever bind; over 6 bars the same move is 0.67
ATR/bar, which is a grind, and a grind reading as displacement is exactly the
QQQUSDT failure the pole rule was rebuilt to stop. So size alone cannot carry
the definition once length is uncapped.

THE ENTRY IS A RETRACEMENT, NOT A BREAKOUT. The alert waits until price has
come at least halfway back from the expansion's extreme toward the block, so a
resting limit is quoted when it has some chance of filling rather than the
moment the block forms with price still far away. Halfway is not an invented
constant: it is the same 0.5 that already places the entry (the block's own
midpoint), the target (the gap's midpoint) and equilibrium (premium vs
discount). The trigger has to stay strictly OUTSIDE the block, because a block
price has re-entered is by definition no longer untested - a threshold reaching
into the block would disqualify every setup at the moment it fired.

TARGET IS A LEVEL, NOT A RATIO, which no other strategy here does. It is the
midpoint of the nearest gap that is not yet fully closed, skipping any gap the
setup's own displacement leg created - that one is part of the entry structure
price is currently retracing through, not an objective ahead of the trade.
Note "not fully CLOSED" rather than "never touched": Dror's own wording
separates a block that is "טרם נבדק" (untested) from a gap that is "טרם נסגר"
(unclosed), and his rule that a gap tested once will probably fill completely
next time makes a partially-filled gap a MORE reliable target than a virgin
one, not a worse one. The midpoint is of the ORIGINAL gap, his call.

Because the target is a level, reward:risk falls out of geometry instead of
being chosen, and it can land anywhere. Anything under 1:2 is declined - the
floor Strategy 1 and Strategy 3 both use - and a setup that fails it is
declined outright rather than having the search walk further out for a gap that
pays. Walking on would be picking the target that produces the wanted R rather
than the one structure offers, and every surviving signal would then look like
it passed a bar it had been fitted to.
"""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from notifier.risk_sizing import ROUND_TRIP_FEE_PCT
from notifier.strategies.base import TIMEFRAME_SECONDS, Signal, Strategy
from notifier.strategies.indicators import atr
from notifier.strategies.structure import structure_context, zigzag_pivots

ATR_PERIOD = 14
# The swing scale for pivots, sweeps and the trend read. Same value Strategy 1
# settled on against Dror's charts; a strategy wanting a different scale should
# say so rather than move his.
STRUCTURE_ATR_MULTIPLE = 1.25

# Displacement, per the flag pole (patterns.py) with the bar cap removed.
EXPANSION_ATR_MULTIPLE = 4.0
EXPANSION_MAX_DEMAND_PCT = 0.35
# Body ATRs per bar. PROVISIONAL - it is the one number here with no
# measurement behind it, and it exists only because uncapping the run length
# made size-alone insufficient (see the module docstring). 1.0 is the floor at
# which a 4x ATR move takes at most 4 bars; the review session should sweep it
# against rendered charts before this strategy graduates out of dry run.
EXPANSION_MIN_STEEPNESS = 1.0
# One counter-direction candle is allowed inside the run - Dror's call, since a
# four-candle expansion with one small red bar in it still reads as one move -
# but it must stay small relative to what the run has already travelled, or a
# genuine two-way fight passes as displacement.
EXPANSION_MAX_COUNTER_CANDLES = 1
EXPANSION_COUNTER_MAX_RETRACE = 0.5
# A bar belongs to the burst only if its body is at least this share of the
# largest body in it. Relative on purpose - "part of the same move" is a
# statement about the move, not about an absolute size, and CRCLUSDT's drift
# (1.32, 0.38, 0.09 ATR) sits far below its displacement (3.51, 5.92) while
# both of the latter clearly belong together.
EXPANSION_BAR_MIN_SHARE = 0.5
# How far back the looser sweep reading looks for the candle that took the
# liquidity. Only used when sweep_on_block_only is off.
SWEEP_LEG_LOOKBACK = 5

# Entry mechanics.
# How many candles an unfilled entry limit may rest before it is cancelled.
#
# A resting limit is not free: it blocks its symbol through already_exposed()
# and holds risk against the 6% aggregate cap for as long as it waits, which at
# ~$100 equity is roughly one of three available slots. Dror's constraint: "i
# want the time the limit catch from the risk budjet to be minimal".
#
# Measured on 1H with no proximity gate, 7 setups over 11,650 symbol-bars:
#
#     rest  10 (10h)  14% filled
#     rest  20 (20h)  29%
#     rest  30 (30h)  43%   <- the elbow
#     rest  50 (2.1d) 43%
#     rest  80 (3.3d) 43%
#     rest 120 (5.0d) 57%
#     rest 200 (8.3d) 86%
#
# 30 is where the curve flats: 50 and 80 candles buy nothing at all over 30,
# and reaching 57% costs five days of budget occupancy for +14 points. Chosen
# for that flat spot rather than for the highest fill, which would be 200
# candles and over a week of holding a slot per pending order.
#
# WIRED INTO EXECUTION as Signal.unfilled_timeout_seconds. It was not, for
# long enough to matter: the tracker applied one flat 4-hour wall clock to
# every strategy (ENTRY_TIMEOUT_SECONDS, sized for Strategy 1's limit tranche),
# so the 1H instance got 4 candles of the 30 this was calibrated for and the
# 15m instance got 16. Without the proximity gate the limit is EXPECTED to
# wait, which made the gap load-bearing rather than cosmetic.
UNFILLED_CANDLES = 30
# How far back the premium/discount range is measured, ending at the block.
# It has to be independent of the block: deriving it from a leg the block
# anchors makes the block sit at that leg's own extreme by construction, which
# is how a WIFUSDT short in cheap territory was passed as premium.
DEALING_RANGE_LOOKBACK = 60
ENTRY_FRACTION = 0.5  # "בנקודת ה-0.5 (אמצע) של טווח האורדר בלוק"
GAP_TARGET_FRACTION = 0.5  # midpoint of the original gap
MIN_REWARD_RISK = 2.0
# Above this the target is not ambitious, it is unreachable. The first replay
# planned trades at 13R, 19R, 30R and 49R - all of them cases where the nearest
# qualifying gap simply sat a very long way off, so the number describes the
# distance to the next imbalance rather than anything about the trade. Dror's
# pick after seeing them.
MAX_REWARD_RISK = 10.0
# Reference numbers for how far a limit can be from the market and still be
# reached, kept because they are what settled the design even though no gate
# uses them now. The largest excursion over 10 bars, in ATR(14) of the starting
# bar, is the SAME on every timeframe: p50 1.20/1.26/1.26 and p75 2.19/2.38/
# 2.34 across 1D/4H/1H over ~300k windows. Directionally (only a move against
# the trade reaches its limit) the fill odds are ~78% at 0.5 ATR, ~58% at 1.0,
# ~24% at 2.3.
#
# Blocks sit a median 4-7 ATR from price when they qualify, so no cap in that
# range admits them - which is why the gate was removed rather than tuned. See
# _signal_for.
ENTRY_DISTANCE_REFERENCE = {"p50_travel_10_bars_atr": 1.25, "p75_travel_10_bars_atr": 2.3}
# A gap has to be big enough to be a zone worth trading to, and this has been
# raised twice on Dror's chart reads.
#
# First at 0.25, after "in nvda there isnt a gap there" - that level was a
# 0.1-wide sliver on a ~3.0 ATR, about 0.03 ATR, invisible at chart scale.
# Then at 1.0, after FIGHTUSDT: "ther target isnt a gap the candle its too
# small for it to be a gap", and his instruction "so we need to make the demand
# of the gap to be larger". That gap measured 0.95 ATR and 1.46% of price -
# comfortably past the old floor and around the 90th percentile of all gaps -
# so nothing short of a real tightening excludes it.
#
# Measured across 9,239 real gaps on 45 symbols over 1D/4H/1H: median 0.26 ATR,
# p75 0.525, p90 0.915. A 1.0 floor keeps roughly the largest 8%. That is a
# severe cut and it will show up directly in the signal rate - a setup with no
# qualifying gap ahead is declined outright, it does not fall back to a ratio.
MIN_GAP_ATR = 1.0
# The stop sits below the low where liquidity was taken, plus a buffer. Never
# exactly ON the level: that defect has been found five separate times in this
# project, because risk shrinks toward zero while the round-trip fee stays
# fixed against notional.
#
# Widened from 0.10 (Strategy 2's borrowed value) on Dror's decision,
# 2026-08-26. Swept on the existing signals in the handoff's day3 §67: 0/13
# wins at 0.10, monotonically improving through 1.00 ATR on both win rate and
# max drawdown, though mean R never turns positive at any tested value - the
# strategy's real bottleneck is signal volume (13 fills in ~150 symbol-years),
# not stop placement. Shown against three real recent signals (COTIUSDT,
# LTCUSDT, TAGUSDT) with their own stop recomputed at each buffer: 0.50 was
# picked over 1.00 as "the safer middle ground" - 1.00 already fails the
# MIN_REWARD_RISK floor on two of those three setups outright, where 0.50
# still clears it on one and comes closer on the others without giving up as
# much of the strategy's edge (its far gap targets, 5-9R at the original
# stop). Strategy 4 stays in dry run either way - see main.py's DRY_RUN_TAGS.
STOP_ATR_BUFFER = 0.50

# Both adopted from Strategy 2's measured values rather than re-derived. The
# failure modes are identical: a stop too tight is eaten by fees before the
# market moves, and a stop widened by a crash is a stop that means nothing.
# Measured in percent of PRICE, not of ATR - LABUSDT's ATR grew to ~88% of its
# own price during its crash, so stop/ATR looked ordinary at 1.4 while
# stop/price was 127%. A denominator that expands with the thing it measures
# cannot bound it.
MAX_FEE_FRACTION_OF_RISK = 0.25
MAX_STOP_PCT = 0.20
# Maker in, taker out - the same correction made in Strategy 2, and for the
# same reason: this strategy also sets market_fraction = 0.0, so the whole
# entry rests as a limit and fills as a MAKER at 0.02%. The exit a RISK gate
# should price is the stop, taker at 0.06%. Total 0.08%, not the 0.12% a taker
# entry would cost. (partial_fraction = 1.0 means the winning exit is a single
# maker limit at the target, 0.04% - cheaper still, and not the case to
# calibrate a risk gate on.)
#
# backtest/engine.py charges pending limit fills maker and stops taker, so the
# old constant had the harness and the strategy pricing the same trade
# differently. The minimum stop this gate admits moves 0.48% -> 0.32% of price,
# which LOOSENS Strategy 4's signal rate - already noted as an open question
# after the chart review cut it to 2 setups across 100 symbols. Strategy 4 is
# in DRY_RUN_TAGS, so the effect is observable before it costs anything.
#
# Was a local hardcoded 0.0008 duplicating notifier.risk_sizing's constant -
# numerically correct for THIS strategy (market_fraction=0.0 matches the
# maker-in-taker-out case exactly) but a second source of truth Dror's "one
# shared fee constant, use it everywhere" (2026-08-26) was meant to prevent.
# Deduped 2026-08-27 while auditing every strategy's fee basis for the same
# assumption that was silently wrong for Strategy 2.1 - this one just
# happened to be right by coincidence, not by being tied to the source.
ROUND_TRIP_FEE = ROUND_TRIP_FEE_PCT

# Named to match the other three strategies' own MARKET_FRACTION /
# MARKET_ENTRY_FRACTION constants (each strategy's fee basis now reads FROM
# one of these, e.g. the weekly report's fee-by-strategy breakdown), rather
# than leaving this as an inline 0.0 literal on the Signal below with nothing
# to import.
MARKET_FRACTION = 0.0

# "עדיף שלא נוצר בסשן אסיה" - PREFERRED, not required. So it is stated on the
# alert and left to judgment rather than silently discarding setups, which is
# what writing it as a filter would have done.
ASIA_TZ = ZoneInfo("Asia/Tokyo")
ASIA_SESSION_HOURS = range(0, 9)


@dataclass(frozen=True)
class Gap:
    """A three-candle imbalance. `start_index` is the candle BEFORE the gap and
    `end_index` the one after it - Dror's marking rule exactly: "לוקחים את קצה
    הפתיל של הנר שלפני הגאפ ואת קצה הנר שאחריו", wick ends of both, so the zone
    is bounded by wicks rather than bodies."""

    direction: str  # "up" if the imbalance was left by an up-move
    low: float
    high: float
    start_index: int
    end_index: int

    @property
    def midpoint(self) -> float:
        return self.low + (self.high - self.low) * GAP_TARGET_FRACTION

    @property
    def size(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Expansion:
    start: int
    end: int
    direction: str  # "up" or "down"
    extreme: float  # furthest price the move reached


@dataclass(frozen=True)
class OrderBlock:
    index: int
    low: float
    high: float
    direction: str  # the trade this block sets up
    variant: str  # "OB1.0" or "OB2.0"
    displacement_end: int
    extreme: float
    sweep_level: float  # the pivot whose liquidity was taken
    # The price the sweep actually REACHED - the wick's own extreme. This is
    # what the stop sits beyond, not the level: "מתחת לנמוך שבו נלקחה הנזילות",
    # the low AT WHICH liquidity was taken. Anchoring on the level instead
    # puts the stop inside the block whenever the wick ran well past it, so
    # the candle that created the setup would have taken the trade out.
    sweep_extreme: float

    @property
    def entry(self) -> float:
        return self.low + (self.high - self.low) * ENTRY_FRACTION

    @property
    def near_edge(self) -> float:
        """The edge price retraces back to: the top of a bullish block."""
        return self.high if self.direction == "long" else self.low


def find_gaps(bars: pd.DataFrame) -> list[Gap]:
    """Every three-candle imbalance in the window, oldest first."""
    highs, lows = bars["high"], bars["low"]
    gaps: list[Gap] = []
    for i in range(len(bars) - 2):
        before_high, before_low = highs.iloc[i], lows.iloc[i]
        after_high, after_low = highs.iloc[i + 2], lows.iloc[i + 2]
        if after_low > before_high:
            gaps.append(Gap("up", float(before_high), float(after_low), i, i + 2))
        elif after_high < before_low:
            gaps.append(Gap("down", float(after_high), float(before_low), i, i + 2))
    return gaps


def gap_is_closed(gap: Gap, bars: pd.DataFrame) -> bool:
    """Whether price has traded all the way back through the zone.

    "טרם נסגר" is about the gap being FILLED, which is a weaker condition than
    a block's "טרם נבדק" - price may have poked into the zone repeatedly and
    the gap is still open until it is fully traversed. That is deliberate:
    Dror's own rule is that a gap tested once will probably break and fill
    completely on the next test, so a partially-filled gap is a target price is
    more likely to reach, not less.
    """
    after = bars.iloc[gap.end_index + 1 :]
    if after.empty:
        return False
    if gap.direction == "up":
        return bool(after["low"].min() <= gap.low)
    return bool(after["high"].max() >= gap.high)


def find_expansions(bars: pd.DataFrame, atr_series: pd.Series) -> list[Expansion]:
    """Maximal directional runs that genuinely expand the market.

    A run is extended by same-direction candles, and tolerates at most
    EXPANSION_MAX_COUNTER_CANDLES counter-direction candle whose own body stays
    under EXPANSION_COUNTER_MAX_RETRACE of the distance travelled so far. It is
    then trimmed so it never ENDS on a counter candle - a run ending against
    itself has simply already finished.
    """
    opens, closes = bars["open"], bars["close"]
    highs, lows = bars["high"], bars["low"]
    found: list[Expansion] = []

    i = 0
    while i < len(bars) - 1:
        direction = "up" if closes.iloc[i] > opens.iloc[i] else "down"
        if closes.iloc[i] == opens.iloc[i]:
            i += 1
            continue

        end = i
        counters = 0
        last_aligned = i
        j = i + 1
        while j < len(bars):
            candle_up = closes.iloc[j] > opens.iloc[j]
            aligned = (direction == "up") == candle_up and closes.iloc[j] != opens.iloc[j]
            if aligned:
                end, last_aligned, j = j, j, j + 1
                continue
            progress = abs(closes.iloc[end] - opens.iloc[i])
            body = abs(closes.iloc[j] - opens.iloc[j])
            if counters < EXPANSION_MAX_COUNTER_CANDLES and progress > 0 and body <= EXPANSION_COUNTER_MAX_RETRACE * progress:
                counters += 1
                end, j = j, j + 1
                continue
            break

        end = last_aligned  # never end on the tolerated counter candle

        # TRIM THE WEAK LEADING BARS. Dror, on CRCLUSDT: "the candle before the
        # big 2 is the order block not what the code found". That run was five
        # bars - 1.32, 0.38 (against), 0.09, then the real move at 3.51 and
        # 5.92 ATR. Taken whole it qualified on total size and put the block
        # before the drift rather than before the displacement, three bars too
        # early and at a different price entirely.
        #
        # Trimming to the SHORTEST qualifying burst was tried first and goes
        # too far the other way: 5.92 clears the bar alone, so the run became
        # just that candle and the block landed on 3.51 - one of the two
        # candles that ARE the move. "The big 2" means both of them.
        #
        # So a bar joins the burst when it is at least half the largest body in
        # it. That is relative rather than another absolute constant, and it
        # says what a chartist means by one move: 3.51 against 5.92 belongs,
        # 1.32 and 0.09 do not.
        bodies = [abs(closes.iloc[k] - opens.iloc[k]) for k in range(i, end + 1)]
        biggest = max(bodies) if bodies else 0.0
        start = i
        while start < end and abs(closes.iloc[start] - opens.iloc[start]) < biggest * EXPANSION_BAR_MIN_SHARE:
            start += 1
        run = _qualifies(bars, atr_series, start, end, direction, highs, lows, opens, closes)
        if run is not None:
            found.append(run)
        i = max(end + 1, i + 1)

    return found


def _qualifies(bars, atr_series, start, end, direction, highs, lows, opens, closes) -> Expansion | None:
    if end <= start - 1 or end < start:
        return None
    body = abs(closes.iloc[end] - opens.iloc[start])
    # ATR from BEFORE the run, not at its first bar. ATR(14) at the opening
    # candle of a violent move already contains that candle's own true range,
    # so the move inflates the threshold it is being asked to clear and the
    # sharpest displacements disqualify themselves. Same shape as the AXTIUSDT
    # range (hourly ATR inflated by the move that formed it) and LABUSDT's
    # crash stop: a reference that moves with the thing it measures cannot
    # measure it.
    atr_at = atr_series.iloc[max(0, start - 1)]
    if atr_at <= 0 or body <= 0:
        return None

    # 4x ATR, but never asking for more than 35% of price. NBISUSDT's daily ATR
    # is ~12% of its own price, so the ATR test alone demanded a 49% candle -
    # 44 of 95 watchlist symbols demanded over 30% on 1D. A flat cap can only
    # ever LOOSEN this, so nothing that already qualified can stop qualifying.
    demanded = min(EXPANSION_ATR_MULTIPLE * atr_at, EXPANSION_MAX_DEMAND_PCT * opens.iloc[start])
    if body < demanded:
        return None

    bars_used = end - start + 1
    if body / atr_at / bars_used < EXPANSION_MIN_STEEPNESS:
        return None

    extreme = float(highs.iloc[start : end + 1].max() if direction == "up" else lows.iloc[start : end + 1].min())
    return Expansion(start, end, direction, extreme)


def _nearest_prior_pivot(pivots: list[tuple[int, bool]], bars: pd.DataFrame, before: int, want_high: bool) -> float | None:
    """The level whose stops the sweep runs, i.e. the most recent confirmed
    swing of the right kind sitting before this candle."""
    for idx, is_high in reversed(pivots):
        if idx >= before or is_high != want_high:
            continue
        return float(bars["high"].iloc[idx] if is_high else bars["low"].iloc[idx])
    return None


def _swept(bars: pd.DataFrame, index: int, level: float, direction: str) -> bool:
    """Wick through the level, close back on the safe side.

    Merely trading through is not a sweep - that is the market leaving. What
    marks stops being taken and the move rejected is the wick going and the
    close coming back, which is the whole reason an order block is expected to
    hold on the retest.
    """
    if direction == "long":
        return bool(bars["low"].iloc[index] < level <= bars["close"].iloc[index])
    return bool(bars["high"].iloc[index] > level >= bars["close"].iloc[index])


def _untested(bars: pd.DataFrame, block_low: float, block_high: float, after: int) -> bool:
    """Whether price has stayed out of the block since the move left it.

    Scoped from the END of the displacement, not from the block itself. The
    displacement candle opens inside the block it is leaving almost by
    construction, so measuring from the block would mark every block tested by
    its own expansion and the strategy would never fire.
    """
    rest = bars.iloc[after + 1 :]
    if rest.empty:
        return True
    return not bool(((rest["low"] <= block_high) & (rest["high"] >= block_low)).any())


def _fee_fraction_of_risk(entry: float, stop: float) -> float:
    risk_pct = abs(entry - stop) / entry
    return ROUND_TRIP_FEE / risk_pct if risk_pct > 0 else float("inf")


class OrderBlockStrategy(Strategy):
    """A block found on the structure timeframe, triggered by a retracement
    seen on the entry timeframe.

    Both versions run inside one instance rather than one instance each,
    because "2.0 wins when the same block qualifies as both" is a decision
    about one block and cannot be made by two strategies that cannot see each
    other. The instance therefore declares BOTH tags; see Strategy.tags.
    """

    def __init__(
        self,
        timeframe: str = "1H",
        session_gated: bool = True,
        # Whether the block candle must itself be the one that swept, or the
        # sweep may sit anywhere in the leg before the displacement (with the
        # stop then anchored on that lower low). Dror deferred this to the
        # chart review - "be decided later when the setup will be seen
        # visually" - so both readings are implemented and rendered.
        sweep_on_block_only: bool = True,
    ):
        # ONE timeframe, not a structure/entry pair. Dror: "here there is no
        # meaning for combination of 2 timeframes" - the block, the sweep, the
        # dealing range and the trigger are all read off the same chart, unlike
        # Strategies 2 and 3 where a slower frame genuinely confirms a faster
        # one. The paired version also made the resting window incoherent: 10
        # candles meant 10 four-hour bars for a block living on a daily chart.
        self.timeframe = timeframe
        self.tag = f"Strategy 4 {timeframe} OB2.0"
        self.tags = (f"Strategy 4 {timeframe} OB1.0", f"Strategy 4 {timeframe} OB2.0")
        self.timeframes = [timeframe]
        # Both instances are intraday and the sweep is a wick event. A wick
        # printed while the underlying market is shut is not liquidity being
        # taken from anyone, so tokenized stocks are gated out of hours.
        self.session_gated = session_gated
        self.sweep_on_block_only = sweep_on_block_only
        # symbol -> (window, block) for the block the most recent evaluate()
        # call actually signalled on, for chart_overlay to read back. Block-
        # finding lives deep in _find_blocks/_signal_for (structure_context,
        # zigzag_pivots, find_gaps all feed it) and isn't cheap or safe to
        # re-derive standalone without risking a second, drifting copy of
        # that logic - see this module's own "one implementation" reasoning
        # elsewhere in the codebase (backtest/score.py). Safe as instance
        # state because the scan loop is single-threaded and chart_overlay is
        # only ever called immediately after evaluate() in the same tick, on
        # the symbol that just fired - never independently, never stale.
        self._chart_context: dict[str, tuple] = {}

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        bars = bars_by_timeframe.get(self.timeframe)
        if bars is None or len(bars) < 60:
            return None

        # The trend gate. Strategy 2 shorted MUUUSDT into rising highs AND
        # rising lows because nothing in it consulted structure; 36% of its raw
        # signals were counter-trend on replay. OB 2.0 has no structure
        # requirement of its own, so without this it would do the same.
        window, structure = structure_context(bars, atr_multiple=STRUCTURE_ATR_MULTIPLE, atr_period=ATR_PERIOD)
        if structure.trend is None or structure.anchor_index is None:
            return None
        direction = "long" if structure.trend == "up" else "short"

        # Equilibrium of the dealing range: the leg the current trend began
        # from. Only its 0.5 is used - this is the premium/discount line, not
        # an anchor for anything else.
        anchor = structure.anchor_index
        # THE DEALING RANGE MUST NOT BE ANCHORED ON THE BLOCK. It used to be
        # trend_structure's current leg, anchor to extreme - and on WIFUSDT the
        # CHoCH anchor WAS the block's own candle, so the "range" was the 11
        # bars the block itself opened, with the block sitting at the top of
        # it. Premium was then guaranteed for every short of that shape and the
        # test could never fail. Dror: "if it ob 2.0 and it for short it should
        # be in the expensive area not in the cheap one as this block" - read
        # against the real structure, with the 0.146 zone overhead, 0.1404 is
        # cheap.
        #
        # Measured over a window ENDING AT the block instead, so it describes
        # the range price had established BEFORE the block existed and nothing
        # the block does can move it.
        range_start = max(0, anchor - DEALING_RANGE_LOOKBACK)
        leg_high = float(window["high"].iloc[range_start : anchor + 1].max())
        leg_low = float(window["low"].iloc[range_start : anchor + 1].min())
        if leg_high <= leg_low:
            return None
        equilibrium = (leg_high + leg_low) / 2

        atr_series = atr(window, ATR_PERIOD)
        thresholds = atr_series * STRUCTURE_ATR_MULTIPLE
        pivots = zigzag_pivots(window, thresholds)
        gaps = find_gaps(window)

        blocks = self._find_blocks(window, atr_series, pivots, gaps, direction)
        if not blocks:
            return None

        price = float(window["close"].iloc[-1])
        atr_now = float(atr_series.iloc[-1])

        # Most recent first: an older block is not disqualified for being old,
        # but when several are live the one price is actually working back into
        # is the one being traded.
        for block in sorted(blocks, key=lambda b: b.index, reverse=True):
            signal = self._signal_for(symbol, window, block, gaps, price, atr_now, equilibrium)
            if signal is not None:
                self._chart_context[symbol] = (window, block)
                return signal
        return None

    def chart_overlay(self, bars_by_timeframe: dict, signal):
        """The block itself and the level it swept - see OrderBlock's own
        field docs for why the stop sits beyond sweep_extreme, not the level.
        """
        from notifier.chart import ChartOverlay

        context = self._chart_context.get(signal.symbol)
        bars = bars_by_timeframe.get(self.timeframe)
        if context is None or bars is None:
            return None
        window, block = context

        # block.index is relative to structure_context's own WINDOW (a tail
        # slice reset to a 0-based index, see its docstring) - not to `bars`,
        # which is what chart.render() actually draws against.
        offset = len(bars) - len(window)
        position = block.index + offset

        return ChartOverlay(
            zones=[(position, position, block.low, block.high, block.variant)],
            markers=[(position, block.sweep_extreme, "sweep")],
            levels=[(block.sweep_level, "swept level", "#9a6a00")],
        )

    def _find_blocks(self, window, atr_series, pivots, gaps, direction) -> list[OrderBlock]:
        """Both versions, with 2.0 taking precedence on a shared candle."""
        opens, closes = window["open"], window["close"]
        highs, lows = window["high"], window["low"]
        expansion_dir = "up" if direction == "long" else "down"

        by_index: dict[int, OrderBlock] = {}

        def consider(block_index: int, displacement_end: int, extreme: float, variant: str) -> None:
            if block_index <= 0 or block_index >= len(window):
                return
            # THE BLOCK IS THE CANDLE BEFORE THE MOVE, whatever colour it is.
            #
            # This used to also require it to close AGAINST the move - the ICT
            # "last opposing candle". That condition is mine, not the
            # cheatsheet's, which says only "זהו הנר שלפני מהלך שמרחיב את השוק".
            # Dror struck it out on CRCLUSDT, where the candle he named sits in
            # the direction of the move and my rule discarded it: "the candle
            # before the big 2 is the order block not what the code found",
            # then "drop the rule".
            #
            # The chart also shows why it was a poor filter regardless: that
            # candle's body is 0.09 ATR. On a doji the colour is a coin flip,
            # so the rule was rejecting setups on noise.

            swept = self._sweep(window, pivots, block_index, displacement_end, direction)
            if swept is None:
                return
            sweep_level, sweep_extreme = swept
            if not _untested(window, float(lows.iloc[block_index]), float(highs.iloc[block_index]), displacement_end):
                return

            existing = by_index.get(block_index)
            if existing is not None and existing.variant == "OB2.0":
                return  # 2.0 wins when the same candle qualifies as both
            by_index[block_index] = OrderBlock(
                index=block_index,
                low=float(lows.iloc[block_index]),
                high=float(highs.iloc[block_index]),
                direction=direction,
                variant=variant,
                displacement_end=displacement_end,
                extreme=extreme,
                sweep_level=sweep_level,
                sweep_extreme=sweep_extreme,
            )

        # OB 2.0 - anchored on the expansion.
        for exp in find_expansions(window, atr_series):
            if exp.direction != expansion_dir:
                continue
            consider(exp.start - 1, exp.end, exp.extreme, "OB2.0")

        # OB 1.0 - anchored on the gap, and additionally requiring a structure
        # break and a reversal candle.
        for gap in gaps:
            if gap.direction != expansion_dir:
                continue
            block_index = gap.start_index
            displacement = block_index + 1
            if displacement >= len(window):
                continue

            # "נר היפוך": the move away from the block closes beyond the
            # block's far edge, so the reversal is confirmed by the
            # displacement itself rather than asserted about one candle.
            if direction == "long" and closes.iloc[displacement] <= highs.iloc[block_index]:
                continue
            if direction == "short" and closes.iloc[displacement] >= lows.iloc[block_index]:
                continue

            # The move away has to be a real expansion, not merely a candle
            # that left a gap. Dror on SYNUSDT: "the expanding of the market is
            # too small so it dont count realy" - that displacement measured
            # 2.85 ATR where OB 2.0 demands 4.0, and nothing in the 1.0 path
            # was checking size at all. His cheatsheet does not state a size
            # for 1.0, but a gap can form on a quiet stagger and that is not
            # what "created a gap" is describing.
            #
            # Same floor as the expansion test so the two versions cannot
            # disagree about what counts as displacement: 4x ATR, capped at
            # 35% of price for symbols whose ATR is a large share of price.
            atr_at = atr_series.iloc[max(0, block_index - 1)]
            move = abs(closes.iloc[gap.end_index] - opens.iloc[displacement])
            if atr_at <= 0 or move < min(
                EXPANSION_ATR_MULTIPLE * atr_at, EXPANSION_MAX_DEMAND_PCT * opens.iloc[displacement]
            ):
                continue

            # "שבר או המשיך את מבנה השוק" - broke OR continued, so any close
            # beyond the last confirmed pivot in the trade's direction counts;
            # a continuation break is as good as a reversal here.
            level = _nearest_prior_pivot(pivots, window, block_index, want_high=(direction == "long"))
            if level is None:
                continue
            broke = closes.iloc[gap.end_index] > level if direction == "long" else closes.iloc[gap.end_index] < level
            if not broke:
                continue

            extreme = float(
                highs.iloc[block_index : gap.end_index + 1].max()
                if direction == "long"
                else lows.iloc[block_index : gap.end_index + 1].min()
            )
            consider(block_index, gap.end_index, extreme, "OB1.0")

        return list(by_index.values())

    def _sweep(self, window, pivots, block_index, displacement_end, direction) -> tuple[float, float] | None:
        """(level taken, price the sweep reached), or None if none was taken.

        For a long that is a prior swing LOW wicked through and closed back
        above - "לקח נזילות" from the side the market came from, which is both
        the reversal's fuel and what the stop then sits under.
        """
        level = _nearest_prior_pivot(pivots, window, block_index, want_high=(direction == "short"))
        if level is None:
            return None

        def extreme_at(i: int) -> float:
            return float(window["low"].iloc[i] if direction == "long" else window["high"].iloc[i])

        if _swept(window, block_index, level, direction):
            return level, extreme_at(block_index)
        if self.sweep_on_block_only:
            return None
        # The looser reading, pending Dror's chart review: any candle in the
        # leg running into the displacement may have done the sweeping, and
        # the stop then anchors on THAT candle's extreme rather than the
        # block's - which is the whole reason the two readings differ.
        for i in range(max(0, block_index - SWEEP_LEG_LOOKBACK), displacement_end + 1):
            if _swept(window, i, level, direction):
                return level, extreme_at(i)
        return None

    def _signal_for(self, symbol, window, block, gaps, price, atr_now, equilibrium) -> Signal | None:
        direction = block.direction

        # Premium/discount. "האורדר בלוק חייב להימצא באזור דיסקאונט" - measured
        # at the price actually transacted, the block's own midpoint.
        entry = block.entry
        if direction == "long" and entry >= equilibrium:
            return None
        if direction == "short" and entry <= equilibrium:
            return None

        # Price must still be outside the block - it is untested, so it is.
        #
        # NOTHING ELSE GATES THE ALERT. There is deliberately no "wait until
        # price is near the block" condition, and that is the single most
        # consequential thing measured about this strategy.
        #
        # Two versions of such a wait were built and both were wrong. First
        # "price has retraced at least halfway from the displacement's
        # extreme", which bounds a FRACTION while filling is a question of
        # DISTANCE - on a 60% displacement even 85% of the way back is still
        # 9% away, and 24 of 26 replayed signals never filled. Then a proper
        # distance test, price within N ATR of the limit. Measured across
        # ~10,000 bars per timeframe, THAT REMOVED 99.1% of otherwise complete
        # setups on 1H (110 valid setups became 1) and 94.7% on 15m.
        #
        # The reason is structural rather than a bad threshold: a block is
        # about half an ATR tall, and when price comes back it crosses from
        # "1.5 ATR away" to "inside the zone" inside a candle or two. There is
        # no lingering near-but-outside state to detect, so any proximity rule
        # is really a rule about catching a single bar, and it mostly misses.
        #
        # A resting limit does not need price to be nearby when it is placed -
        # waiting at the level is what the order is for. So the limit goes out
        # when the setup is complete, and the cost of waiting is bounded by
        # UNFILLED_CANDLES instead.
        near = block.near_edge
        if direction == "long" and price <= near:
            return None
        if direction == "short" and price >= near:
            return None

        buffer = atr_now * STOP_ATR_BUFFER
        stop = block.sweep_extreme - buffer if direction == "long" else block.sweep_extreme + buffer
        risk = entry - stop if direction == "long" else stop - entry
        if risk <= 0:
            return None

        if abs(entry - stop) / entry > MAX_STOP_PCT:
            return None
        if _fee_fraction_of_risk(entry, stop) > MAX_FEE_FRACTION_OF_RISK:
            return None

        target = self._target(window, gaps, block, entry, price)
        if target is None:
            return None
        reward = target - entry if direction == "long" else entry - target
        ratio = reward / risk
        if not (MIN_REWARD_RISK <= ratio <= MAX_REWARD_RISK):
            return None

        notes = []
        if self._in_asia_session(window, block.index):
            notes.append("Formed during the Asia session - the cheatsheet prefers blocks that did not.")

        return Signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            strategy_tag=f"Strategy 4 {self.timeframe} {block.variant}",
            # A level target, expressed as the ratio that reproduces it. Exact
            # because market_fraction is 0: the scanner then sizes from
            # signal.entry_price itself, so entry + risk x ratio lands on the
            # gap midpoint to the cent. If a market portion is ever added here,
            # the scanner blends the entry and this stops being exact.
            reward_risk_ratio=ratio,
            limit_entry=entry,
            limit_note="order block 0.5",
            market_fraction=MARKET_FRACTION,
            # This strategy's own measured window, in the unit the tracker
            # counts. 30 candles of whichever timeframe this instance runs on.
            unfilled_timeout_seconds=UNFILLED_CANDLES * TIMEFRAME_SECONDS[self.timeframe],
            # Full close at the gap - no partial, no runner. The cheatsheet
            # names one exit and this strategy has one.
            partial_fraction=1.0,
            dedupe_key=(symbol, self.tag, direction, round(block.low, 10), round(block.high, 10)),
            extra_notes=tuple(notes),
            reason=(
                f"{block.variant} on the {self.timeframe}: the candle before a "
                f"{'bullish' if direction == 'long' else 'bearish'} expansion took liquidity at "
                f"{block.sweep_level:.8g} and has not been tested since. Block {block.low:.8g}-{block.high:.8g}, "
                f"entry at its 0.5, stop beyond the swept level, target the midpoint of the next unclosed gap "
                f"at {target:.8g} ({ratio:.1f}R)."
            ),
        )

    def _target(self, window, gaps, block, entry, price) -> float | None:
        """Midpoint of the nearest gap the trade can actually still reach.

        A GAP BETWEEN PRICE AND THE ENTRY IS NOT A TARGET. Dror, reading the
        first rendered batch: "if the price will go to the ob it will fill the
        gap so this gap is irrelevant". Getting to the entry means travelling
        through everything in between, so by the time the position exists those
        gaps are closed and the trade would be aiming at a level that no longer
        holds any unfilled orders. A long's limit sits BELOW the market, so the
        descent to it closes every up-gap above the entry that price passes
        through - which is all of them up to where price stands now.

        What survives is a gap entirely beyond the current price, on the far
        side from the entry: the approach moves away from it, not through it.
        That also makes the choice stable - a target picked at signal time is
        still unfilled when the limit fills, so nothing has to be re-derived
        later.

        The setup's own displacement gap is skipped for the same reason plus
        one more: for OB 1.0 it is the very gap that identified the block, and
        it sits immediately against it.
        """
        direction = block.direction
        atr_series = atr(window, ATR_PERIOD)
        candidates = []
        for gap in gaps:
            # OPPOSITE direction, deliberately. An up-gap is left behind by an
            # up-move, so price stands above it and it fills when price comes
            # back DOWN - meaning an unfilled up-gap always sits BELOW the
            # market. Those are precisely the gaps a long fills on its way down
            # to the block. The one still open ahead of a long is the gap left
            # by a DOWN move overhead, which the descent to the entry moves
            # away from rather than through.
            if gap.direction != ("down" if direction == "long" else "up"):
                continue
            if block.index <= gap.start_index <= block.displacement_end:
                continue  # the setup's own gap; structurally redundant now
            if gap_is_closed(gap, window):
                continue
            if gap.size < MIN_GAP_ATR * float(atr_series.iloc[gap.end_index]):
                continue  # a sliver is not an imbalance - see MIN_GAP_ATR
            if direction == "long" and gap.low > price and gap.midpoint > entry:
                candidates.append(gap.midpoint)
            elif direction == "short" and gap.high < price and gap.midpoint < entry:
                candidates.append(gap.midpoint)
        if not candidates:
            return None
        return min(candidates) if direction == "long" else max(candidates)

    @staticmethod
    def _atr_at(window, index: int) -> float:
        return float(atr(window, ATR_PERIOD).iloc[index])

    def _in_asia_session(self, window, index) -> bool:
        ts = window["ts"].iloc[index]
        if pd.isna(ts):
            return False
        return pd.Timestamp(ts).tz_localize("UTC").astimezone(ASIA_TZ).hour in ASIA_SESSION_HOURS
