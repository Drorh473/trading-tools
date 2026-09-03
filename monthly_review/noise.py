"""When a month's number is different enough from its baseline to act on.

Dror's rule, 2026-09-03: a tuning flag fires only when the difference is
OUTSIDE the noise band. Everything else is printed with its number and an
explicit "cannot distinguish", never as a suggestion.

That rule needs one thing added to it to actually work, and it is the reason
this is a module rather than an `if abs(delta) > x` at each call site.

THE MULTIPLE-COMPARISONS PROBLEM

A 95% band means a 5% chance of firing on pure noise - PER TEST. The monthly
report runs one test per instance per metric: nine live instances against
expectancy, win rate, fill rate, slippage and fee rate is 45 tests. At 5%
each, the expected number of false flags in a clean month is 2.3. Two
made-up tuning jobs every single month is exactly how a section stops being
read, which is worse than not having the section.

So the band is widened for the number of tests the report actually runs, via
Šidák: per_test_alpha = 1 - (1 - family_alpha)^(1/k). With family_alpha at
0.05 and k=45 that is a per-test alpha of 0.00114, z = 3.25 rather than 1.96 -
and the expected number of false flags across the WHOLE report returns to
0.05. A flag that survives that is worth opening an editor for.

The cost is real and is stated rather than hidden: widening the band raises
the smallest difference the month can detect. Every finding therefore carries
`detectable`, the half-width of its own band, so a quiet result reads "no
difference larger than 0.41R would have shown up" instead of the much weaker
"no difference".
"""

import math
from dataclasses import dataclass

# The chance this report contains ONE OR MORE false flags, across every test
# it runs. Not the per-test rate - see the module docstring.
FAMILY_ALPHA = 0.05

# Below this many observations, no test runs at all. A standard error from
# three trades is arithmetic, not evidence: the sample SD is itself so noisy
# that the band it produces is meaningless in either direction. Such metrics
# are reported as numbers with "n too small", which is the honest state and
# the one most instances will be in most months at this account size.
MIN_N = 8


@dataclass(frozen=True)
class Finding:
    """One metric, this month, against its baseline."""

    metric: str
    live: float
    baseline: float
    n: int
    detectable: float  # smallest |difference| this month could have shown
    fires: bool
    note: str  # why it did not fire, when it did not

    @property
    def delta(self) -> float:
        return self.live - self.baseline


def z_for(num_tests: int, family_alpha: float = FAMILY_ALPHA) -> float:
    """The z the band uses, widened for how many tests the report runs.

    num_tests is the count for the WHOLE report, not for this metric, and it
    must be decided before any test is read - picking it afterwards, or
    counting only the tests that happened to look interesting, is the same
    mistake as running the sweep and then choosing the window.
    """
    num_tests = max(1, num_tests)
    per_test = 1.0 - (1.0 - family_alpha) ** (1.0 / num_tests)
    return _z_from_two_sided_alpha(per_test)


def _z_from_two_sided_alpha(alpha: float) -> float:
    """Inverse normal CDF at 1 - alpha/2, via the stdlib's erf.

    Bisection rather than a closed-form approximation: it is called a handful
    of times per report, and a bisection cannot be subtly wrong in the tails
    the way a fitted rational approximation can - which is where every value
    this module uses actually lives.
    """
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def check_mean(
    metric: str,
    values: list[float],
    baseline: float,
    num_tests: int,
    family_alpha: float = FAMILY_ALPHA,
) -> Finding:
    """A per-trade average - expectancy in R, slippage in bp - against what it
    is supposed to be.

    The band is built from the sample's OWN spread, not an assumed one. R
    multiples on this account have run an SD near 1.8, so an instance with 20
    closed trades carries a standard error of 0.40R before any correction: at
    that sample the honest finding is almost always "cannot distinguish", and
    the report should say so rather than reach.
    """
    n = len(values)
    if n < MIN_N:
        return Finding(metric, _mean(values), baseline, n, float("inf"), False, f"n={n}, too small")

    mean = _mean(values)
    sd = _sample_sd(values, mean)
    if sd == 0.0:
        # Every value identical. Real for a rate that never moved; for a mean
        # it means the sample cannot speak to spread at all.
        return Finding(metric, mean, baseline, n, 0.0, mean != baseline, "zero spread in sample")

    detectable = z_for(num_tests, family_alpha) * sd / math.sqrt(n)
    fires = abs(mean - baseline) > detectable
    note = "" if fires else f"difference under {detectable:.3f}, inside the band"
    return Finding(metric, mean, baseline, n, detectable, fires, note)


def check_rate(
    metric: str,
    hits: int,
    n: int,
    baseline: float,
    num_tests: int,
    family_alpha: float = FAMILY_ALPHA,
) -> Finding:
    """A proportion - win rate, fill rate - against what it is supposed to be.

    Uses the binomial standard error rather than treating the 0/1 outcomes as
    a sample of a continuous variable. The distinction matters most where this
    report needs it most: at n=20 and p=0.35 the difference is small, but the
    binomial form stays honest as p approaches 0, which is precisely where an
    instance like Strategy 4 lives (its fill rate is about 2%).
    """
    if n < MIN_N:
        rate = hits / n if n else 0.0
        return Finding(metric, rate, baseline, n, float("inf"), False, f"n={n}, too small")

    rate = hits / n
    # The band is built around the BASELINE's spread, not the sample's: the
    # question is "could a process whose true rate is the baseline have
    # produced this?", so the null's variance is the right one. It also keeps
    # the band from collapsing to zero when a small sample happens to come
    # back all wins or all losses.
    var = baseline * (1.0 - baseline)
    if var <= 0.0:
        return Finding(metric, rate, baseline, n, 0.0, rate != baseline, "baseline rate is 0 or 1")

    detectable = z_for(num_tests, family_alpha) * math.sqrt(var / n)
    fires = abs(rate - baseline) > detectable
    note = "" if fires else f"difference under {detectable:.1%}, inside the band"
    return Finding(metric, rate, baseline, n, detectable, fires, note)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_sd(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
