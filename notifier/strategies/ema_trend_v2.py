"""Strategy 2 v2 - the EMA9 pullback, with the target decoupled from the stop.

v1 is a single self-sufficient timeframe whose target is a multiple of its own
stop, so its reward:risk is 1:3 by construction and cannot be anything else.
v2 splits the job for PAIRED instances:

  higher timeframe  - owns the trend and the TARGET
  lower timeframe   - owns the trigger and the STOP

Because the target no longer scales with the stop, reward:risk becomes a free
variable. Measured on 25 symbols of the cached year: median 5.4:1 against v1's
2:1, upper quartile 14.5:1.

STANDALONE instances have no higher timeframe and keep the v1 geometry - their
targets are 1:2 and 1:3 of their own stop.

WHAT THE FIRST MEASUREMENT SAYS, so nobody re-derives it from scratch: scanned
over 38 symbols this fires ~2,500 times a year and wins 10.5% of the time, and
ALL of its positive expectancy sits in trades whose stop is under 0.15% of
price - which is inside the spread and models no slippage. Every bucket with a
holdable stop measured negative. MIN_STOP_PCT exists to bound exactly that, and
is meant to be swept, not trusted at its default.

Differences from v1, all deliberate (see strategy2-v2-spec.md for the session
that decided each one):

  removed  - the 10-candle hold, the chop filter, the shock-candle filter, the
             short-side volume filter, and the 0.05 x ATR touch band
  trigger  - a limit PRE-PLACED at the last closed bar's EMA9, filling on first
             touch, with no confirming close
  trend    - strict last-3: three ascending highs AND three ascending lows, on
             the higher timeframe for a pair. "Cannot tell" now BLOCKS the
             trade, where v1 treated it as permission
  risk     - flat, no reference-timeframe tiers
  fees     - maker in, taker out at the stop: 0.08%, not 0.12%
"""

import pandas as pd

from notifier.strategies.base import FillGuard, Signal, Strategy
from notifier.strategies.indicators import atr, ema, sma
from notifier.strategies.structure import zigzag_pivots

EMA_FAST, EMA_MID, EMA_SLOW, TREND_MA_PERIOD = 9, 20, 50, 200
ATR_PERIOD = 14
STOP_ATR_BUFFER = 0.10

# THE EMA9 IS A ZONE, NOT A LINE.
#
# Requiring price to reach the EMA9 exactly refused six of sixteen setups Dror
# marked by eye, by gaps of 0.16% to 6.2% of price - and the 0.16% one is a
# rounding difference, not a difference of opinion. v1 had this: a touch counted
# within EMA9_BAND_ATR_MULTIPLE = 0.05 of an ATR. v2 dropped it in decision 5
# and these six are the cost.
#
# Measured in ATR rather than percent so it means the same thing on a quiet
# symbol and a violent one - the same reason v1's crash guard is in percent and
# its touch band is not.
#
# Swept against sixteen setups Dror marked on blind charts:
#
#     band (ATR)   recall
#        0.00       9/16     exact touch
#        0.10      12/16
#        0.20      13/16
#        0.35      14/16     <- here
#        0.50      14/16
#        0.75      15/16
#
# 0.35 is the last point that buys anything: 0.50 adds nothing and 0.75 costs
# more than double the tolerance for one more mark. v1's 0.05 recovers none of
# them, so that value was far too tight rather than roughly right.
#
# The one mark this cannot reach, 2C315, needed 6.2% of price - about 2 ATR.
# At that distance it is not a tolerance question and probably a different
# entry anchor.
EMA9_TOUCH_ATR = 0.35

# WHERE THE ORDER RESTS, and whether it waits for the rejection.
#
# "rejection"  the signal bar must close back on the trend side, the order rests
#              afterwards and fills on a retest. Faithful to the setup Dror
#              reads on a chart - and measured at -0.02R to -0.11R depending on
#              where the limit sits, because price returning to a level it just
#              rejected selects the rejections that FAILED.
#
# "band"       a limit rests at EMA9 +/- EMA9_TOUCH_ATR x ATR, both knowable at
#              the previous close, and fills on any touch. No rejection, so it
#              takes the breakdowns Dror rejected on the AAVE and PAXG charts.
#              Not the strategy he learned; chosen with that stated.
#
# The distinction is worth 0.2R per trade, and every earlier attempt to have
# BOTH - the rejection to select and an intrabar fill to enter - produced a
# lookahead, four separate times.
# "next_open"  the rejection is required, and the trade is entered at MARKET on
#              the candle after it. 100% fill, no retest, no lookahead - the
#              rejection candle has closed and the next price is the first one
#              actually available. Measured best of the faithful constructions
#              at -0.020R, against -0.086R for a limit at the EMA9.
ENTRY_MODE = "next_open"

# THE EMA9 MUST BE ACTING AS SUPPORT (long) OR RESISTANCE (short), not merely
# be a line price happens to be crossing.
#
# A one-candle touch test - low <= EMA9 and close > EMA9 - cannot tell these
# two apart, because both satisfy it:
#
#   price above the EMA9, dips to it, closes back above   <- the setup
#   price below the EMA9, rallies UP through it, closes above  <- NOT the setup
#
# The second is a broken resistance, and the first cut of v2 took those trades.
# Dror, reading the ETHUSDT and 1000RATSUSDT charts: "if it long the ema9 is
# supposed to be support not resistence". On ETH's 4H the EMA9 had spent forty
# bars below a flat base and raced up underneath a vertical rally, so it had
# been "support" for about three candles when the trade fired.
#
# This is v1's ten-candle hold, which was removed in the redesign as a
# staleness filter without noticing it was ALSO the statement that the level is
# respected. Restored deliberately, and comparing each close to its OWN EMA9 at
# that bar rather than to today's level, because the level moves.
#
# FIVE, AND ONLY ON THE TIMEFRAME THAT SETS THE TREND. Both halves of that are
# measured (backtest/sweep_v2.py over 16,198 generated setups):
#
#   Requiring it on BOTH timeframes of a pair is fatal. Paired setups run 219
#   at no hold, 47 at one bar, 5 at three, and ZERO at ten - and paired is the
#   only place v2's target is decoupled from its stop, which is the whole
#   reason the strategy exists. Ten-on-both, which this file briefly shipped,
#   silently reduced v2 to a standalone-only strategy.
#
#   On the reference alone the paired population survives (116 setups at five)
#   and its expectancy stops being noise: +13.3R with SE 8.3 at no hold - a few
#   tiny-denominator outliers - becomes +1.36R with SE 0.55 at five. Smaller,
#   and the first paired number in this exercise distinguishable from zero.
#
# TWO, not five. Measured against setups Dror marked by eye on blind charts:
# B210 passed every other gate - structure agreeing, rejection true, stop 3.16%,
# net R:R 1.94 - and was refused for a hold run of 2. He had called it the
# entry. Lowering the hold from 5 to 2 takes recall from 4/7 to 6/7 with the
# structure gate off.
#
# The cost, stated because it was measured and went the other way: on the
# STANDALONE population the hold curve falls monotonically, +0.470R at zero to
# +0.375R at five to +0.334R at ten, with no peak anywhere. The harness models
# no slippage, and the trades a hold removes are entries into fast moves
# through a level - precisely where slippage is worst - so that curve is read
# as "this measurement cannot price what the filter removes" rather than as
# "the filter is worthless". Dror's call, made with the table in front of him.
EMA9_HOLD_BARS = 2
# THE HOLD IS PER INSTANCE TOO, and 1H is the reason.
#
# Swept behind the stop floors that ship, over the population 1H actually
# trades. Net R, and the same value after discarding the three best trades:
#
#     hold >=     n      netR      t     drop top 3
#        2      1478    +0.002   0.06      -0.014
#        5      1220    +0.018   0.41      -0.001
#        8      1031    +0.045   0.96      +0.031
#       10       893    +0.068   1.33      +0.051   <- here
#       15       533    +0.048   0.73      +0.023
#       20       280    +0.179   1.90      +0.134
#       25       127    +0.283   2.03      +0.199
#
# The gradient matters more than any single cell. Eight grid points rising
# almost monotonically is far harder to get by chance than one significant t,
# and this same sweep produced no such gradient on 4H or 1D at all.
#
# 10 IS NOT THE VALUE THE DATA PREFERS, and that is deliberate. Splitting the
# two years and choosing on the first half only:
#
#     hold >=      first half        second half (held out)
#        2       +0.031 [ 702]        -0.023 [ 776]
#       10       +0.106 [ 455]        +0.029 [ 438]   <- ships
#       20       +0.210 [ 151]        +0.142 [ 129]
#
# 20 is positive in BOTH halves at every stop floor - +0.080/+0.155 at no
# floor, +0.096/+0.131 at 1.0, +0.210/+0.142 at 1.5 - which is six positive
# cells across data it was not chosen on. 10's held-out half is +0.029 at
# t 0.41, near zero rather than negative: the trades between hold 10 and hold
# 20 contribute approximately nothing.
#
# Dror was shown that and chose 10 anyway, on 2026-08-19: "keep 10 and watch it
# live". The reasoning is about what the backtest CANNOT supply. 20 is ~10
# signals a week on 1H, so a quarter is ~130 trades and he sees almost none of
# them; 10 is ~32 a week, which accumulates a live sample - and his own read of
# the alerts - fast enough to be worth having while the expectancy is merely
# not-negative. He is buying observations, knowingly, at a measured cost of
# about 0.11R per trade against the held-out numbers.
#
# So this is a value to REVISIT with live evidence, not one to re-derive from
# this table. If the live 1H sample turns out negative, 20 is where it goes.
#
# WHAT THIS COSTS, and it is not visible in the table. Dror lowered the hold
# from 5 to 2 deliberately, with recall in front of him: on setups he had
# marked by eye on blind charts it took 4/7 to 6/7, and B210 was specifically
# his entry. Raising 1H to 10 refuses most of what he would mark there. The
# harness cannot price what a filter removes - the same caveat this file
# already carries about slippage - so this is a trade of his eye for the
# measurement, made knowingly on 2026-08-19.
#
# 15m keeps 2, and no longer because it is unswept - see INSTANCES. Its own
# population says the hold barely matters there: hold 20 is its least-bad value
# at -0.072R and still negative, so there is no value to move it to.
EMA9_HOLD_BARS_BY_TIMEFRAME: dict[str, int] = {"1H": 10}


def _hold_bars(timeframe: str | None) -> int:
    """How many closes this timeframe must have held its EMA9.

    A generator that zeroes EMA9_HOLD_BARS to build the widest population must
    not be silently re-filtered by a per-timeframe override, so zero wins over
    the map. Without that, generation would pre-filter 1H at 10 and every sweep
    over the result would compare a subset against a whole.
    """
    if EMA9_HOLD_BARS <= 0:
        return 0
    return EMA9_HOLD_BARS_BY_TIMEFRAME.get(timeframe or "", EMA9_HOLD_BARS)

# The TRIGGER timeframe's own version of the same requirement, and the reason
# it is separate and much smaller.
#
# Dror, on an INJUSDT short: "the strategy core idea is that the ema9 is a
# resistance or support, so once it broke like that it is no longer valid."
# Three of five reviewed losers were that: LABUSDT entered long after the 1H had
# closed BELOW its EMA9 twice; 1000RATSUSDT filled a short as price rallied
# through the level and closed above it; INJUSDT and WLDUSDT broke the level
# after the order was placed.
#
# All of them passed because decision 15 put the hold on the trend-setting
# timeframe alone, leaving the trigger timeframe with no EMA9 condition at all.
#
# ONE bar, not five. Requiring the full hold on both timeframes was measured
# and is fatal - paired setups run 219 at no hold, 47 at one bar and ZERO at
# ten. One bar is the minimal statement of "this level is not currently
# broken", which is all Dror's rule actually claims.
BASE_HOLD_BARS = 1

# Pivot scale for the last-3 read. Same value Strategy 1 and Strategy 4 use, so
# "what counts as a swing" has one definition across the project even though
# v2 answers "what trend is this" differently from them.
STRUCTURE_ATR_MULTIPLE = 1.25
# How far back the pivot search looks. Measured: the last three highs and lows
# span a median of 27 bars on 4H and at most 76, so this is roughly 2.5x the
# worst case and is a bound, not a tuned parameter.
STRUCTURE_LOOKBACK = 200
EMA9_CROSSING_LOOKBACK = 30

# THREE SCALE CONDITIONS ON A RULE THAT ONLY EVER TESTED SHAPE.
#
# The last-3 read asks that three highs and three lows are each monotonic. It
# never asks how far they travelled, over how long, or how cleanly - so it
# returns a trend for things nobody would call one. Every setup Dror rejected
# on charts had the same signature:
#
#   DOGEUSDT   1D  "there isn't really a trend, the graph is going sideways"
#                  highs drifted 0.58 ATR across 26 daily bars
#   AVAXUSDT   1H  "sideways and a big dump" - all SIX pivots inside 15 hours,
#                  and low 3 was the October crash wick at 8.43 from a market
#                  trading at 28. Highs drifted 1.08 ATR, lows 14.05.
#   SOXSUSDT   4H  "broke 3 times the ema9 before the setup" - actually TEN
#                  crossings in 30 bars
#   LABUSDT    1H  read "up" on drift alone, but its six pivots spanned ten
#                  hours: one leg with wobble in it, not a structure
#
# Dror: "the found trend is too gentle."
#
# All three DEFAULT TO OFF and are recorded per signal by generate_v2, so they
# are swept over one population rather than picked - the same discipline as
# EMA9_HOLD_BARS, whose guessed 10 turned out to be worth measuring away from.
MIN_PIVOT_SPAN_BARS = 0
MIN_SWING_DRIFT_ATR = 0.0
MAX_EMA9_CROSSINGS = 999

# THE STRUCTURE GATE IS ON, and this is the most consequential rule in the file.
#
# IT WENT BACK ON 2026-08-21, on expectancy rather than recall. The exact
# structure_metrics path, pivot scale chosen on the FIRST half of each
# population and scored on the half it was not chosen on:
#
#     population   keep      drop      gap      t      n(keep)   gate OFF
#        15m      -0.120    -0.245    +0.125   2.95     2,720     -0.223
#        1H       +0.100    -0.066    +0.166   1.34       214     -0.017
#
# The trades it refuses really are the worse ones, and on 15m that is
# SIGNIFICANT rather than suggestive - the first t over 2 this rule has ever
# produced, on the instance carrying most of the alert volume.
#
# WHY THIS REVERSES THE 2026-08-19 READING. That measurement used the structure
# metrics RECORDED at generation time, which disagree with a fresh computation
# on about 5% of setups - a discrepancy that was noticed, flagged, and then not
# acted on. Recomputed exactly, the gate is positive out of sample in BOTH
# populations at this scale, where the recorded metrics said it did nothing.
#
# WHAT IT DOES NOT DO: rescue 15m. Gated, 15m goes -0.223 -> -0.120 and stays
# firmly negative, with drop-best-3 at -0.140. It halves the bleed. 1H goes
# -0.017 -> +0.100, which is the only cell here that reaches positive.
#
# THE PIVOT SCALE IS NOT TUNED, DELIBERATELY. Held-out gap by scale disagrees
# across populations - 1H peaks at 1.50 and is WORST at 1.00; 15m peaks at 1.00
# and rates 1.50 mid-table; 18 charts Dror hand-marked point at 1.75-2.5, which
# is weak-to-inverting on both. Three methods, three answers. 1.25 is positive
# out of sample in both (+0.079 1H, +0.049 15m) and is left alone rather than
# fitted to whichever population is being looked at. See
# [[project-swing-threshold-fit]] for why marking more charts will not settle
# it: the fitted optimum tracks marking density at r = -0.78.
#
# THE COST, unchanged and still real. It keeps ~35% of 1H signals and ~18% of
# 15m, so the alert stream drops by two thirds to four fifths. It is the same
# gate that blocked 5 of 7 setups Dror marked by eye - the difference now is
# that the recall it costs buys something measurable rather than nothing.
#
# The recall case that switched it OFF, kept because it is the reason to watch
# this live rather than trust the table above:
#
# It required the last N swing highs and lows to be monotonic in the trade's
# direction. Measured against seven setups Dror marked by eye on blind charts -
# no signals drawn, nothing showing what the code thought - it blocked five of
# them, and recall was 1/7.
#
# It fails for a reason no threshold fixes. During a PULLBACK the most recent
# swings run AGAINST the trend: chart A was a clean downtrend by the four-MA
# stack, and the last two swings read "up" at both marked entries. The gate asks
# recent structure to agree with the trade at exactly the moment it cannot,
# because the pullback is the setup.
#
# Recall by rule set over those seven marks:
#
#     last-3, hold 5   (as shipped)   1/7
#     last-2, hold 5                  3/7
#     last-2, hold 2                  4/7
#     no structure, hold 5            4/7
#     no structure, hold 2            6/7   <- here
#
# The four-MA stack already states the trend and was correct on all seven. The
# structure gate came in to stop one counter-trend short on MUUUSDT, which the
# stack itself also refuses; it has been asking the same question with an
# instrument that cannot tell "against the trend" from "inside a pullback".
#
# span, drift and crossings are still MEASURED and recorded per signal, so the
# gate can be swept back on if a wider sample disagrees with these seven.
# A wider sample did: 30,000 15m setups and 1,450 1H ones, above.
REQUIRE_STRUCTURE_TREND = True
# STRUCTURE_PIVOTS lived here claiming the read was last-2. It was DEAD CODE -
# defined once, referenced nowhere, while structure_metrics and _last3_trend
# both hardcode [-3:]. Removed rather than wired, because last-2 was measured
# on 2026-08-21 and is the wrong direction: last-3 agreeing always implies
# last-2 agrees, so it can only ADD setups, and the ones it adds are the worst
# cohort in the population (-0.136R over 4,508 of them, against -0.104R
# ungated).

# The higher timeframe's target, as multiples of ITS OWN stop distance. The
# prices these produce are absolute and do not move when the lower timeframe
# offers a better entry - that is the whole point of the split.
TARGET_1_RATIO, TARGET_2_RATIO = 2.0, 3.0
PARTIAL_FRACTION = 0.5

# Maker in (0.02%) + taker out at the stop (0.06%). NOT 0.12%: this strategy
# never places a market order, so its entry is always a maker fill. Same
# correction as v1's ROUND_TRIP_FEE_PCT.
ROUND_TRIP_FEE_PCT = 0.0008
MAKER_FEE_PCT = 0.0002

# v1 gated on fee-as-a-fraction-of-RISK, which only proxies for expectancy when
# reward:risk is roughly constant. Here it ranges 2:1 to 14:1, so that gate
# refused 56% of signals and refused the highest-R:R ones preferentially. This
# asks the economic question directly: after fees, is the trade still worth it.
#
# NOT 2.0, and the reason is arithmetic rather than taste. A STANDALONE
# instance targets 1:2 of its own stop, so its gross ratio is exactly 2.0 and
# its net ratio - (2r - maker) / (r + fees) - approaches 2.0 from BELOW for
# every stop width and never reaches it. A 2.0 floor silently zeroes four of
# the seven instances, which is what the first generation run did. Swept, not
# trusted.
MIN_NET_REWARD_RISK = 1.5

# THE GATE THAT MATTERS, and the one with no evidence behind its value. A stop
# tighter than the spread is not a stop, it is a coin flip with a flattering
# denominator - and because R is measured against it, such trades report
# enormous R:R and enormous expectancy while being unholdable in practice.
# The first scan found +26.9R average below 0.15% and negative everywhere a
# trade could actually be held. SWEEP THIS. Do not ship it at a guessed value.
MIN_STOP_PCT = 0.003
# The mirror case, unchanged from v1: EMA20 lagging a crash puts the stop
# absurdly far away and the setup is no longer an orderly pullback.
MAX_STOP_PCT = 0.20

# THE SAME GATE, IN THE UNIT THAT CAN ACTUALLY EXPRESS IT.
#
# MIN_STOP_PCT is scale-free in percent, so it cannot tell a wide stop from a
# wide market. LABUSDT's 4H stop was 4.3% of price - fourteen times the 0.3%
# floor - and 1.13 ATR, because LABUSDT's ATR was itself 5.4% of price. Dror,
# on that alert: "the stop is too close".
#
# It is not one symbol. Across the eight alerts of 2026-08-18, the stop
# measured from the price the trade would fill at:
#
#     HYPE 0.10   AVAX 0.58   MMT 0.80   WIF 0.85
#     BTW  0.96   LAB  1.13   SOL 1.63           (ATR, median 0.85)
#
# Every one inside one candle's normal range. The cause is structural rather
# than incidental: the stop is EMA20 +/- STOP_ATR_BUFFER x ATR, so its distance
# from the entry is just however wide the EMA9-EMA20 gap happens to be, and
# nothing in this file ever asked that gap to be big enough to survive noise.
#
# SWEPT, not guessed - over 17,981 setups on 27 symbols and two years, scored
# at the price the trade actually fills at:
#
#     floor (ATR)      n      netR      t
#        0.00       17981    -0.103   -6.83
#        0.25       17377    -0.095   -6.66
#        0.50       14667    -0.074   -4.97
#        0.75       10660    -0.076   -4.64
#        1.00        6867    -0.067   -3.37
#        1.25        3967    -0.048   -1.98
#        1.50        2163    -0.034   -1.04   <- here
#        2.00         593    -0.040   -0.67
#        2.50         163    -0.081   -0.74
#        3.00          67    +0.104   +0.58
#
# Monotonic to 1.50 and then noise: 2.00 and 2.50 are WORSE on a tenth of the
# sample, and the positive 3.00 cell is 67 trades. Swept past the optimum
# deliberately, because a peak with nothing beyond it is not a peak. A 20-symbol
# cut of the same population put the same shape at the same place, which is the
# only out-of-sample comfort available here.
#
# WHAT THIS DOES AND DOES NOT BUY. It takes the strategy from clearly negative
# to indistinguishable from zero; it does NOT make it profitable, and nothing
# else in this sweep does either. It keeps 12% of setups, which also takes the
# alert stream from ~166 a day to something readable.
#
# The distribution it is cutting into: stop-in-ATR runs p10 0.38, p25 0.57,
# median 0.85, p75 1.20, p90 1.57. So this refuses about seven setups in eight,
# and the median setup is refused - which IS the finding, not a side effect of
# it. The stop is EMA20 +/- 0.10 x ATR, so its distance is just however wide
# the EMA9-EMA20 gap happens to be, and that gap is usually under one ATR.
#
# AND THE VALUE IS PER TIMEFRAME TOO, because one number is wrong on two of the
# three instances that were measured. Net R by floor, 27 symbols:
#
#     floor        1H                4H                1D
#      0.00   -0.115 [14694]   -0.039 [ 3111]   -0.176 [  176]
#      0.50   -0.078 [12238]   -0.046 [ 2308]   -0.122 [  121]
#      0.75   -0.075 [ 9162]   -0.081 [ 1448]   -0.060 [   50]
#      1.00   -0.062 [ 6094]   -0.094 [  755]        -  [   18]
#      1.25   -0.031 [ 3659]   -0.251 [  302]        -  [    6]
#      1.50   -0.025 [ 2063]   -0.220 [  100]        -  [    0]
#      2.00   -0.043 [  585]        -  [    8]        -  [    0]
#
# 1H improves monotonically to 1.50 and turns down at 2.00. 4H goes the OTHER
# WAY - every floor makes it worse, and 1.25 costs it 0.2R. And 1D's widest stop
# in two years is 1.41 ATR, so a 1.50 floor there is not a filter, it is a
# silent shutdown of the instance: 176 setups in, zero out.
#
# Re-checked with slippage charged on the stop leg at 0.02%, 0.05% and 0.10% of
# price - the obvious explanation, since slippage in R is slippage/stop_fraction
# and so falls hardest on the tight stops a floor removes. It does not invert
# anything; 4H prefers no floor at every level tested. The hypothesis was wrong
# and the table stands.
#
# WHERE THIS DISAGREES WITH THE CHART, and how that was settled. Dror read
# LABUSDT's 4H stop (1.13 ATR) as "too close", and that reading is the reason
# this constant exists at all. On 4H the measurement says the opposite: setups
# with stops under ~1 ATR are the better half of that population, and refusing
# them removes the part worth keeping. 1H agrees with him; 4H does not.
#
# SETTLED 2026-08-19: "keep the 4h floor off, trust the measurement." His call,
# made with the table above in front of him and knowing it means LABUSDT- and
# MMTUSDT-shaped 4H alerts keep arriving. Do not re-raise it, and do not
# reintroduce a 4H floor as a side effect of tuning something else - the test
# below pins the value so that cannot happen quietly.
#
# PER TIMEFRAME ALSO because the sweep could only see three of the four.
#
# The cache generate_v2 walks holds 1H bars; 4H and 1D resample from them and
# 15m cannot be derived at any price. So every number in the table above is
# 1H/4H/1D, and 15m has NO evidence for this floor in either direction.
#
# That is not a technicality. Applied to 15m as well, this floor refuses seven
# of the eight alerts Dror reviewed on 2026-08-18 - including BTWUSDT at 0.96
# ATR, which ran +36R and is the one he called great - while ALLOWING SOLUSDT
# at 1.63 ATR, the one he said had the wrong entry. Both of those are 15m. A
# filter measured on three timeframes and applied to a fourth is a guess with
# a table next to it, and this particular guess reads the two 15m cases exactly
# backwards.
#
# 15m stayed at 0.0 pending its own population. That population now exists
# (115,179 setups, 102 symbols) and it does not change the answer: no floor
# brings 15m to zero, the best being ~-0.09R at 2.0 ATR. So 0.0 here is no
# longer "unmeasured", it is "measured, and nothing to gain". See INSTANCES.
#
# Applied at the FILL and not here - see FillGuard. Checking it against
# e9_prev would measure the same trade nobody gets.
MIN_STOP_ATR: dict[str, float] = {"15m": 0.0, "1H": 1.5, "4H": 0.0, "1D": 0.0}


class EmaTrendV2(Strategy):
    """One instance. `reference_timeframe=None` makes it standalone, which
    keeps v1's own-R target geometry; a reference makes it paired, and the
    reference then owns both the trend read and the target."""

    def __init__(self, base_timeframe: str, reference_timeframe: str | None = None):
        self.base_timeframe = base_timeframe
        self.reference_timeframe = reference_timeframe
        self.paired = reference_timeframe is not None
        self.tag = (
            f"Strategy 2.1 {reference_timeframe}/{base_timeframe}"
            if self.paired
            else f"Strategy 2.1 {base_timeframe}"
        )
        self.timeframes = [base_timeframe] + ([reference_timeframe] if self.paired else [])
        # The higher timeframe is read on its FORMING candle. Measured: when a
        # 4H touch and a 1H trigger coincide, the 1H trigger fires a median of
        # two hours BEFORE the 4H bar closes, and 44% of them land in its first
        # hour. Waiting for the close means entering after the move the whole
        # design exists to catch. Its EMA LEVELS still come from the last
        # closed bar, so the target cannot drift while the candle builds.
        self.forming_bar_timeframes = (reference_timeframe,) if self.paired else ()
        # A pair replaces its own base timeframe's standalone instance when both
        # fire on one symbol - the same entry and stop, better informed. They
        # coincide on 26% of standalone triggers, so without this the account
        # carries 2% on one idea in two correlated positions.
        self.supersedes = (f"Strategy 2.1 {base_timeframe}",) if self.paired else ()

    def evaluate(self, symbol: str, bars_by_timeframe: dict[str, pd.DataFrame]) -> Signal | None:
        base = bars_by_timeframe.get(self.base_timeframe)
        if base is None or len(base) < ATR_PERIOD + 3:
            return None

        if self.paired:
            forming = bars_by_timeframe.get(self.reference_timeframe)
            if forming is None or len(forming) < TREND_MA_PERIOD + 2:
                return None
            closed = forming.iloc[:-1]  # levels and structure from CLOSED bars only
        else:
            if len(base) < TREND_MA_PERIOD + 2:
                return None
            # The current bar plays the "forming" role and everything before it
            # is closed, mirroring the paired case exactly. Reading levels off
            # the current bar instead would let a wide candle drag its own EMA9
            # toward itself and read as testing a level it just blew through -
            # v1's prior-bar discipline, kept - and would also disagree with
            # _trigger, which sizes the trade from the PREVIOUS bar's EMA9.
            forming, closed = base, base.iloc[:-1]

        levels = _levels(closed)
        if levels is None:
            return None
        ref9, ref20 = levels

        trend = _stack(closed)
        if trend is None:
            return None

        # BOTH timeframes must show the condition - Dror's clarification: "if
        # the setup is a paired one, in both of them have to be the conditions
        # for the trade". Not the higher timeframe setting a target while the
        # lower one merely touches a line.
        #
        # The HOLD is the one part asked of the reference alone. It is the
        # statement that this EMA9 is respected, which is a claim about the
        # trend - and the reference owns the trend. Demanding it of the trigger
        # timeframe too takes paired setups to zero; see EMA9_HOLD_BARS.
        #
        # This is NOT v1's zero-signal design. That required both timeframes to
        # pass v1's full condition INCLUDING its 0.05 x ATR proximity band, and
        # a touch that tight never coincides across two scales. v2's touch is
        # an actual touch, and 85% of 4H touches have a 1H touch inside the
        # same candle - so the two agreeing is common rather than impossible.
        # The trend-setting timeframe. A PAIR's reference must be touching its
        # own EMA9 - that is a state, read from the partial candle as it stood
        # at the previous close, and it is not this trade's fill. A STANDALONE
        # has no separate reference: its touch IS the fill, tested by _trigger
        # against the entry bar, so checking it here as well would be the
        # lookahead described in _full_condition.
        # For a PAIR the reference's touch is a state - it must still be at its
        # own EMA9 - and stays required in both modes. For a STANDALONE this is
        # the trigger timeframe itself, so in band mode the resting order is the
        # trigger and no rejection is asked for.
        if not _full_condition(
            forming, closed, ref9, trend, require_hold=True,
            require_touch=True if self.paired else (ENTRY_MODE != "band"),  # noqa: E501
            band=_touch_band(closed),
            hold_bars=_hold_bars(self.reference_timeframe or self.base_timeframe),
        ):
            return None
        base_closed = base.iloc[:-1]
        # In band mode the trigger timeframe is not asked for a rejection: the
        # resting order IS the trigger, and it fills before any close is known.
        base_touch = ENTRY_MODE != "band"
        if self.paired:
            base_levels = _levels(base_closed)
            if base_levels is None:
                return None
            # The trigger timeframe must show its own stack. Decision 3 asked
            # for it on both timeframes and only the reference ever got checked
            # - `trend = _stack(closed)` reads the reference, and nothing read
            # the base. Dror spotted it on a FIGHTUSDT short where the 1H EMA9
            # and EMA20 were 0.3% apart and visibly crossing.
            if _stack(base_closed) != trend:
                return None
            # The base's touch is now REQUIRED and is the rejection candle:
            # price reached its EMA9 and closed back on the trend side. The
            # order rests after that candle closes and fills on a retest, so
            # nothing here uses information the order would not have had.
            if not _full_condition(
                base, base_closed, base_levels[0], trend, require_hold=False,
                require_touch=base_touch, band=_touch_band(base_closed),
            ):
                return None

        # THE LEVEL MUST NOT ALREADY BE BROKEN on the timeframe being traded.
        # See BASE_HOLD_BARS. A standalone proves this through EMA9_HOLD_BARS
        # already, but checking both keeps the rule in one place.
        base_run = hold_run(base_closed, trend, cap=max(BASE_HOLD_BARS, 1))
        if base_run < BASE_HOLD_BARS:
            return None

        trigger = _trigger(base, trend)
        if trigger is None:
            return None
        entry, stop = trigger
        # The ATR the stop was buffered with, carried on the signal so the
        # scanner can judge the stop against volatility once it knows the fill.
        # Same bar and same period _trigger uses, so the floor and the buffer
        # speak about one number.
        atr_prev = atr(base, ATR_PERIOD).iloc[-2]
        atr_prev = float(atr_prev) if not pd.isna(atr_prev) else None

        risk = abs(entry - stop)
        stop_fraction = risk / entry
        if not MIN_STOP_PCT <= stop_fraction <= MAX_STOP_PCT:
            return None

        if self.paired:
            gap = abs(ref9 - ref20)
            sign = 1.0 if trend == "up" else -1.0
            target_1 = ref9 + sign * TARGET_1_RATIO * gap
            target_2 = ref9 + sign * TARGET_2_RATIO * gap
        else:
            sign = 1.0 if trend == "up" else -1.0
            target_1 = entry + sign * TARGET_1_RATIO * risk
            target_2 = entry + sign * TARGET_2_RATIO * risk

        reward = (target_1 - entry) if trend == "up" else (entry - target_1)
        if reward <= 0:
            return None
        # Fees on the way in are maker; the leg this gate must survive is the
        # STOP, which is taker. Both are charged against notional, so both are
        # expressed as a fraction of the entry price.
        net = (reward - MAKER_FEE_PCT * entry) / (risk + ROUND_TRIP_FEE_PCT * entry)
        if net < MIN_NET_REWARD_RISK:
            return None

        direction = "long" if trend == "up" else "short"
        # Read from the timeframe the reason names, so the message and the
        # measurement cannot describe different bars.
        structure_read = structure_metrics(
            (closed if self.paired else base_closed)
        )["trend"]
        return Signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            strategy_tag=self.tag,
            # Carried as a ratio because that is what the sizing path speaks,
            # but it is DERIVED from an absolute price the higher timeframe
            # set, not chosen. remainder_target carries the second tier as the
            # price it actually is.
            reward_risk_ratio=reward / risk,
            partial_fraction=PARTIAL_FRACTION,
            # NO RUNNER TARGET. The remainder is managed by the scanner's
            # trailing stop, which ratchets to the last CONFIRMED swing low
            # (high for a short) while structure still makes higher highs -
            # exactly the CHoCH exit Dror specified, and already built.
            #
            # poll_trailing_stops only trails positions with NO target ("having
            # no target is precisely the case this exists for"), so setting one
            # here would silently opt out of the trail and leave the runner on a
            # fixed 1:3 instead. Measured, the fixed target is ~0.045R per trade
            # worse than trailing the structure.
            remainder_target=None,
            # FINAL, AND FINAL AT None. remainder_target above is deliberately
            # unset: this runner belongs to the trailing stop, and
            # poll_trailing_stops only trails a position that has no target.
            #
            # This said False, and the comment above it described the bug it was
            # causing. Without the flag the scanner falls through to Strategy 3's
            # rule and puts the runner at the nearest daily swing level - which
            # is a target, which turns the trail off permanently.
            #
            # It did exactly that on UNIUSDT on 2026-08-19: partial filled, stop
            # to breakeven 3.395, then a runner target at 3.48211 "under the 1D
            # level at 3.536", and the trail could never run again on that
            # position.
            remainder_target_is_final=True,
            remainder_note=(
                f"trailed to the last confirmed {self.base_timeframe} swing, "
                f"while structure keeps going our way"
            ),
            limit_entry=None if ENTRY_MODE == "next_open" else entry,
            limit_note="" if ENTRY_MODE == "next_open" else "EMA9",
            # No market portion, and this is load-bearing rather than
            # incidental: measured, a limit fill at EMA9 is 7.1:1 while a
            # market entry at the same bar's close is 2.3:1. Entering at market
            # costs two thirds of the edge.
            market_fraction=1.0 if ENTRY_MODE == "next_open" else 0.0,
            analysis_timeframes=(
                (self.reference_timeframe, self.base_timeframe) if self.paired else (self.base_timeframe,)
            ),
            # ONE TRADE PER UNBROKEN EMA9 RUN, not one per bar.
            #
            # The default key includes entry_price, and this entry is
            # `ema9_prev` - a level that drifts a little every bar - so every
            # bar reads as a new trade. WLDUSDT fired the SAME 4H long on eight
            # consecutive bars (0.3600, 0.3603, 0.3605, 0.3614, 0.3589, 0.3553,
            # 0.3484, 0.3411), each one stopping out and each one counted
            # separately. base.py documents the identical failure for Strategy
            # 3's breakout re-triggering against one level.
            #
            # It is not only a noisy alert. It inflates n, shrinks every
            # standard error and overstates every t-statistic, because eight
            # copies of one setup are not eight observations.
            #
            # The run's START identifies the episode: it is fixed for as long
            # as price holds the level and moves only when the level breaks,
            # which is exactly when a genuinely new setup may form.
            dedupe_key=(symbol, self.tag, direction, _run_start(base_closed, trend)),
            # EVERY ECONOMIC GATE ABOVE JUDGED THE WRONG PRICE, and this is the
            # correction. `entry` is e9_prev - the EMA9 that selected the setup
            # - while ENTRY_MODE="next_open" opens at market on the following
            # candle, which has closed back on the trend side by construction.
            # The fill is therefore always on the far side of the level.
            #
            # HYPEUSDT on 2026-08-18 measured a 1.50% stop here and 0.15% at
            # the price it would have filled at: it passed this file's own
            # 0.30% floor by failing it twice over. The scanner re-applies the
            # same thresholds against plan_entry, where they mean what they say.
            fill_guard=FillGuard(
                atr=atr_prev,
                min_stop_pct=MIN_STOP_PCT,
                max_stop_pct=MAX_STOP_PCT,
                min_stop_atr=MIN_STOP_ATR.get(self.base_timeframe, 0.0),
                min_net_reward_risk=MIN_NET_REWARD_RISK,
                maker_fee_pct=MAKER_FEE_PCT,
                round_trip_fee_pct=ROUND_TRIP_FEE_PCT,
            ),
            # SAY WHAT WAS CHECKED, NOT WHAT WOULD BE REASSURING.
            #
            # This read "stack and last-3 structure both {trend}" - but `trend`
            # comes from _stack() alone, and REQUIRE_STRUCTURE_TREND is False,
            # so the structure half was never tested. Every signal asserted
            # agreement regardless of what structure_metrics actually returned.
            #
            # DRAMUSDT #1061, 2026-08-21: "15m stack and last-3 structure both
            # up" on a setup whose structure read was None - lows 58.04 / 58.80
            # / 58.55, the last one lower, and seven EMA9 crossings in thirty
            # bars. Dror, reading the chart: "there was a lower high recently
            # that broke the rising highs". The alert had told him otherwise.
            #
            # The stack is the gate and is named as such; the structure read is
            # reported as the observation it is.
            reason=(
                f"{self.reference_timeframe or self.base_timeframe} stack {trend} "
                f"(last-3 structure {structure_read or 'unreadable'}"
                f"{'' if structure_read == trend else ', which does NOT confirm it'}), "
                f"price touching its EMA9; {self.base_timeframe} limit at EMA9 "
                f"{entry:.8g}, stop {stop:.8g} ({100 * stop_fraction:.2f}% of price), "
                f"targets {target_1:.8g} / {target_2:.8g} ({reward / risk:.1f}R, {net:.1f}R net)"
            ),
        )


def _touch_band(closed: pd.DataFrame) -> float:
    """EMA9_TOUCH_ATR x the last closed bar's ATR, as a price distance."""
    if EMA9_TOUCH_ATR <= 0:
        return 0.0
    a = atr(closed, ATR_PERIOD).iloc[-1]
    return 0.0 if pd.isna(a) or a <= 0 else float(a) * EMA9_TOUCH_ATR


def _levels(closed: pd.DataFrame) -> tuple[float, float] | None:
    """(EMA9, EMA20) from the last CLOSED bar, or None if not computable."""
    if len(closed) < EMA_MID + 2:
        return None
    e9 = ema(closed["close"], EMA_FAST).iloc[-1]
    e20 = ema(closed["close"], EMA_MID).iloc[-1]
    if pd.isna(e9) or pd.isna(e20) or e9 <= 0:
        return None
    return float(e9), float(e20)


def _stack(bars: pd.DataFrame) -> str | None:
    """"up"/"down" when the four MAs are fully ordered, else None. Unchanged
    from v1: NaN comparisons resolve False, so short histories return None
    without a length guard."""
    c = bars["close"]
    fast, mid = ema(c, EMA_FAST).iloc[-1], ema(c, EMA_MID).iloc[-1]
    slow, trend_ma = ema(c, EMA_SLOW).iloc[-1], sma(c, TREND_MA_PERIOD).iloc[-1]
    if fast > mid > slow > trend_ma:
        return "up"
    if trend_ma > slow > mid > fast:
        return "down"
    return None


def structure_metrics(bars: pd.DataFrame) -> dict:
    """The last-3 read plus the three scale measures it never took.

    trend  - "up"/"down"/None, the monotonic-swings test
    span   - bars covered by the six pivots. Ten hours of 1H bars is one leg
             with wobble, not a structure.
    drift_high / drift_low - how far each series travelled, in ATR, SIGNED IN
             THE TREND'S DIRECTION so both are positive for a genuine trend.
             Reported separately because their DISAGREEMENT is the tell: AVAX
             ran 1.08 against 14.05, a range plus one crash wick.
    crossings - closes through the EMA9 in the last EMA9_CROSSING_LOOKBACK
             bars. v1 refused more than one; v2 dropped the filter entirely.
    """
    out = {"trend": None, "span": 0, "drift_high": 0.0, "drift_low": 0.0, "crossings": 0}
    w = bars.iloc[-STRUCTURE_LOOKBACK:].reset_index(drop=True)
    if len(w) < 30:
        return out
    atr_series = atr(w, ATR_PERIOD)
    thresholds = (atr_series * STRUCTURE_ATR_MULTIPLE).reset_index(drop=True)
    pivots = zigzag_pivots(w, thresholds)
    hi = [i for i, is_high in pivots if is_high][-3:]
    lo = [i for i, is_high in pivots if not is_high][-3:]
    if len(hi) < 3 or len(lo) < 3:
        return out
    highs = [float(w["high"].iloc[i]) for i in hi]
    lows = [float(w["low"].iloc[i]) for i in lo]

    if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
        out["trend"] = "up"
    elif highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
        out["trend"] = "down"

    a = float(atr_series.iloc[-1])
    if a > 0:
        sign = 1.0 if out["trend"] == "up" else -1.0
        out["drift_high"] = sign * (highs[2] - highs[0]) / a
        out["drift_low"] = sign * (lows[2] - lows[0]) / a
    out["span"] = int(max(hi + lo) - min(hi + lo))

    e9 = ema(w["close"], EMA_FAST)
    above = (w["close"] > e9).iloc[-EMA9_CROSSING_LOOKBACK:]
    out["crossings"] = int((above != above.shift()).sum() - 1)
    return out


def _last3_trend(bars: pd.DataFrame) -> str | None:
    """Strict last-3: the three most recent swing highs AND the three most
    recent swing lows must all be monotonic in the same direction.

    None is NOT permission here - the caller refuses the trade. v1's reader
    returned None in 0% of 832 sampled reads, so treating absence as
    permission was safe; this one abstains 78% of the time, and the same
    semantics would turn the gate into a no-op and reopen the MUUUSDT hole it
    exists to close.
    """
    w = bars.iloc[-STRUCTURE_LOOKBACK:].reset_index(drop=True)
    if len(w) < 30:
        return None
    thresholds = (atr(w, ATR_PERIOD) * STRUCTURE_ATR_MULTIPLE).reset_index(drop=True)
    pivots = zigzag_pivots(w, thresholds)
    highs = [float(w["high"].iloc[i]) for i, is_high in pivots if is_high][-3:]
    lows = [float(w["low"].iloc[i]) for i, is_high in pivots if not is_high][-3:]
    if len(highs) < 3 or len(lows) < 3:
        return None
    if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
        return "up"
    if highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
        return "down"
    return None


def _full_condition(
    forming: pd.DataFrame,
    closed: pd.DataFrame,
    level: float,
    trend: str,
    require_hold: bool,
    require_touch: bool,
    band: float = 0.0,
    hold_bars: int = 0,
) -> bool:
    """The condition on ONE timeframe: structure agrees, the EMA9 has been held
    where that is asked, and the timeframe is touching its EMA9 where that is
    asked.

    EVERYTHING HERE IS READ FROM DATA AVAILABLE BEFORE THE ENTRY BAR OPENS.
    That is not a detail - it is the correctness condition for this whole
    strategy. The entry is a limit PRE-PLACED at the previous bar's EMA9, so
    the decision to have an order resting there is made at the previous close.
    A condition that peeks at the entry bar decides with information the order
    could not have had, and the trade is then filled at a price chosen before
    that information existed.

    That defect shipped once here and was worth 3,614x a year in the backtest:
    `_touching` demanded the entry bar CLOSE back above its EMA9, which
    silently discarded the 54% of touches that close below - every one of them
    a loser a resting limit would have caught - while still filling at the
    level as though the order had been there all along.

    `require_touch` is therefore False for any timeframe whose touch IS the
    fill. The base timeframe never checks it: reaching the level is what
    executes the order, and _trigger tests that against the entry bar because
    that is a FILL, not a decision.
    """
    m = structure_metrics(closed)
    if REQUIRE_STRUCTURE_TREND and m["trend"] != trend:
        return False
    # The scale conditions, applied to EVERY timeframe that has to show the
    # condition - so a pair proves them on both, which is what Dror asked for:
    # "if it is a paired trade all the conditions should met in both, include
    # the 3h and 3l".
    # Each scale condition applies only when it is actually SET. Comparing an
    # unset threshold is not harmless here: structure_metrics signs the drift by
    # the trend it read, and signs by -1 when it read none - so an unreadable
    # structure yields a NEGATIVE drift, and `drift < 0.0` refused it. That
    # silently re-implemented the structure gate through the drift check, and
    # kept refusing 7 of 16 marked setups after REQUIRE_STRUCTURE_TREND was
    # switched off.
    if MIN_PIVOT_SPAN_BARS > 0 and m["span"] < MIN_PIVOT_SPAN_BARS:
        return False
    if MIN_SWING_DRIFT_ATR > 0 and min(m["drift_high"], m["drift_low"]) < MIN_SWING_DRIFT_ATR:
        return False
    if MAX_EMA9_CROSSINGS < 999 and m["crossings"] > MAX_EMA9_CROSSINGS:
        return False
    if require_hold and not _holding(closed, trend, hold_bars):
        return False
    if require_touch and not _touching(forming, level, trend, band):
        return False
    return True


def hold_run(closed: pd.DataFrame, trend: str, cap: int = 120) -> int:
    """How many consecutive closes, counting back from the most recent one,
    sat on the trend side of their OWN EMA9.

    The measurable form of "the EMA9 has been acting as support". Returned as a
    COUNT rather than a yes/no so the threshold can be swept after the fact:
    generate once with the filter off, record the run each setup actually had,
    and every candidate value of EMA9_HOLD_BARS is then a filter over the same
    population rather than another full generation.

    Compared bar by bar against the EMA9 as it was THEN. Comparing every close
    to today's level would call a rising EMA9 "held" simply because price rose
    faster than it did, which is the ETHUSDT case this exists to refuse.
    """
    if len(closed) < 3:
        return 0
    e9 = ema(closed["close"], EMA_FAST)
    on_side = (closed["close"] > e9) if trend == "up" else (closed["close"] < e9)
    run = 0
    for value in on_side.values[::-1]:
        if not value:
            break
        run += 1
        if run >= cap:
            break
    return run


def _run_start(closed: pd.DataFrame, trend: str):
    """When the current unbroken EMA9 run began - the episode's identity.

    Fixed for as long as price holds the level, moving only when the level
    breaks, which is precisely when a new setup may legitimately form. Used as
    the dedupe key so one run produces one trade rather than one per bar.
    """
    run = hold_run(closed, trend)
    if run <= 0 or closed.empty:
        return None
    row = closed.iloc[-run]
    stamp = row["ts"] if "ts" in closed.columns else closed.index[-run]
    return str(stamp)


def _holding(closed: pd.DataFrame, trend: str, hold_bars: int) -> bool:
    """Whether the EMA9 has held for at least `hold_bars` closes.

    Taken as an argument rather than read from the module, because the
    requirement is per instance now - see EMA9_HOLD_BARS_BY_TIMEFRAME.
    """
    if hold_bars <= 0:
        return True
    if len(closed) < hold_bars + 2:
        return False
    return hold_run(closed, trend, cap=hold_bars) >= hold_bars


def _touching(forming: pd.DataFrame, level: float, trend: str, band: float = 0.0) -> bool:
    """Whether the current (possibly still forming) candle came within `band` of
    the level and is trading back on the trend side of it.

    `band` is a price distance - EMA9_TOUCH_ATR x the prior ATR. At 0.0 the
    candle must reach the level exactly.

    On its own this admits a broken resistance as readily as a respected
    support - see EMA9_HOLD_BARS. It is only ever called with _holding().
    """
    last = forming.iloc[-1]
    if trend == "up":
        return bool(last["low"] <= level + band and last["close"] > level)
    return bool(last["high"] >= level - band and last["close"] < level)


def _trigger(base: pd.DataFrame, trend: str) -> tuple[float, float] | None:
    """(entry, stop) if a limit resting at the previous bar's EMA9 would have
    been touched by the current bar, else None.

    Everything comes from the PRIOR bar. A limit cannot rest at a level
    computed from the close of the candle that fills it, and measuring it that
    way is the lookahead that made the first pass of this variant look better
    than it is.
    """
    if len(base) < max(ATR_PERIOD, EMA_MID) + 3:
        return None
    closes = base["close"]
    e9_prev = ema(closes, EMA_FAST).iloc[-2]
    e20_prev = ema(closes, EMA_MID).iloc[-2]
    atr_prev = atr(base, ATR_PERIOD).iloc[-2]
    if any(pd.isna(x) for x in (e9_prev, e20_prev, atr_prev)) or atr_prev <= 0 or e9_prev <= 0:
        return None

    last = base.iloc[-1]
    buffer = float(atr_prev) * STOP_ATR_BUFFER
    # The same band _touching uses. Without it here the band did nothing: the
    # condition passed and then the trigger refused on an exact touch, which is
    # why sweeping the band from 0 to 1.0 ATR moved recall not at all.
    band = float(atr_prev) * EMA9_TOUCH_ATR if EMA9_TOUCH_ATR > 0 else 0.0
    entry = e9_prev + band if ENTRY_MODE == "band" else e9_prev
    if trend == "up":
        if not last["low"] <= entry:
            return None
        stop = float(e20_prev) - buffer
        if stop >= entry:
            return None
    else:
        entry = e9_prev - band if ENTRY_MODE == "band" else e9_prev
        if not last["high"] >= entry:
            return None
        stop = float(e20_prev) + buffer
        if stop <= entry:
            return None
    return float(entry), stop


# The seven instances, as agreed. Four standalone, three paired. The 1D
# standalone survives from v1's list; a 1D/1W pair was considered and dropped
# in favour of keeping the 15m end, which was believed to be where the tight
# stops - and the whole R:R thesis - actually live. MEASURED AND FALSE, see
# INSTANCES: 15m is the worst-measuring instance of the four.
#
# NOTE the 15m pair is not backtested YET, which is not the same as the
# "cannot be backtested" this said before: the cached year holds 1H bars only
# (4H and 1D resample from them, 15m cannot). The exchange is NOT short of 15m
# - "Bitget serves ~22 days of 15m and ~2 of 5m" was one get_candles result
# mistaken for a property of the exchange. history-candles pages back to the
# listing date: 249,112 15m bars on BTCUSDT to 2019-07-10, measured
# 2026-08-17. What is missing is a 15m fetch, not the data.
# PAIRED INSTANCES REMOVED. The decoupled target was v2's whole thesis and it
# measured worst of everything: -0.171R at t -5.79 for 4H/1H over 5,094 trades.
# The mechanism is understood - pairing tightens the stop by entering on the
# lower timeframe, and a tighter stop is hit more often while fees and noise
# stay fixed.
#
# 15m added. It was called permanently unmeasurable on the grounds that Bitget
# serves ~22 days of it; that was false, and history-candles pages back 418 days.
#
# 15m IS NOW MEASURED, and it is the worst of the four. Over 115,179 setups on
# 102 symbols, scored at the price the trade fills at:
#
#     filter                     n        netR       t     held out
#     none                   115179     -0.283   -46.05     -0.247
#     stop >= 1.5 ATR         20294     -0.121   -10.66     -0.086
#     hold >= 20               3144     -0.072    -2.47     -0.075
#     stop >= 1.5, hold >= 20     -     -0.075        -      -0.049
#
# Nothing reaches zero. The best cell in the whole grid is about -0.067R, the
# drop-top-three column is uniformly worse, and the held-out half agrees
# throughout. It is not a distribution artifact either: 15m's stop-in-ATR
# profile is nearly identical to 1H's (median 0.94 against 0.89), so the same
# floors are expressible there and simply do not rescue it.
#
# That refutes the reason the instance exists - see the note above about the
# 15m end being "where the whole R:R thesis actually lives". It is not.
#
# KEPT ANYWAY, and deliberately: Dror was shown the table on 2026-08-19 and
# chose "keep it as is" - hold 2, no stop floor, auto-executing on approval, at
# roughly 1,937 signals a week before his own filtering. BTWUSDT came from this
# instance and ran +36R, which is one observation out of 115,179 and is also
# the kind of trade a population average cannot represent to him. His call,
# made against the numbers rather than in ignorance of them. Do not re-raise it;
# revisit only with LIVE evidence, the same way the 1H hold is being revisited.
#
# 5m is NOT here, and the reason is operational rather than about the setup. The
# scanner runs at `min(timeframes, key=seconds_until_next_close)`, so any
# timeframe not listed in armed_timeframes sets the cadence for EVERY strategy:
# a 5m instance would take the whole loop to 288 scans a day and roughly 28,800
# symbol-fetches against today's 3,100, on an API already returning 429s. It
# needs the armed-polling machinery Strategy 3 uses, and it has never been
# backtested either.
# 4H AND 1D RETIRED 2026-08-19, on their own measurement rather than on taste.
#
# Scored at the fill, behind the floors each of them measured best with:
#
#     instance   signals/wk     netR       t     drop top 3
#       1H          52.6      +0.002     0.06      -0.014
#       4H          72.3      -0.058    -1.32      -0.090
#       1D           3.7      -0.271    -2.09      -0.421
#
# 4H was 56% of the alert volume and the bulk of the loss, and unlike 1H nothing
# rescued it: negative at every hold and every stop floor, and DISCARDING ITS
# THREE BEST TRADES MAKES IT WORSE - so it is not a few bad trades inside a
# sound population, it is the population. 1D was worse per trade than anything
# else measured, on too few trades to tune.
#
# Their tags are in main.LEGACY_EXIT_TAGS so any position opened under them
# before this deploy keeps being managed. Nothing was open when it shipped.
INSTANCES: tuple[tuple[str, str | None], ...] = (
    ("15m", None),
    ("1H", None),
)


def build_instances() -> list[EmaTrendV2]:
    return [EmaTrendV2(base, ref) for base, ref in INSTANCES]
