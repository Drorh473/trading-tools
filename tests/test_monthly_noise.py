"""The gate that decides whether a monthly number becomes a tuning job."""

import random

import pytest

from monthly_review.noise import MIN_N, check_mean, check_rate, z_for


def test_a_small_sample_never_fires_however_far_off_it_looks():
    """Three trades at -3R each look catastrophic and mean nothing. This is
    the common case at this account size, and the report must not manufacture
    work out of it."""
    finding = check_mean("S1 expectancy", [-3.0, -3.0, -3.0], baseline=0.3, num_tests=45)

    assert not finding.fires
    assert "too small" in finding.note
    assert finding.n < MIN_N


def test_a_real_shift_fires_once_the_sample_supports_it():
    values = [-1.0] * 30 + [-0.9] * 30  # tight, and far below a +0.3R baseline
    finding = check_mean("S1 expectancy", values, baseline=0.3, num_tests=45)

    assert finding.fires
    assert finding.delta < 0


def test_a_wide_sample_around_the_baseline_stays_quiet():
    """R multiples spread near 1.8 SD: 20 trades cannot resolve a small
    difference, and the finding must say so rather than round it to 'fine'."""
    rng = random.Random(7)
    values = [rng.gauss(0.35, 1.8) for _ in range(20)]
    finding = check_mean("S1 expectancy", values, baseline=0.3, num_tests=45)

    assert not finding.fires
    assert finding.detectable > 0.5, "20 trades at SD 1.8 resolve nothing smaller"


def test_the_band_widens_with_the_number_of_tests_in_the_report():
    """One test at 95% is z=1.96. Forty-five tests at the same FAMILY error
    rate is much wider - that widening is the whole point."""
    assert z_for(1) == pytest.approx(1.96, abs=0.01)
    assert z_for(45) > 3.0
    assert z_for(45) > z_for(9) > z_for(1)


def test_a_clean_month_produces_about_no_false_flags_across_the_whole_report():
    """The property that matters: run a full report's worth of tests against
    data generated FROM the baseline, many times, and count reports containing
    at least one flag. Uncorrected this sits near 90%; corrected it must sit
    near the 5% family rate.

    This is the test that would catch someone 'simplifying' z_for back to a
    flat 1.96.
    """
    rng = random.Random(11)
    tests_per_report = 45
    reports_with_a_false_flag = 0
    runs = 200

    for _ in range(runs):
        fired = False
        for _ in range(tests_per_report):
            values = [rng.gauss(0.3, 1.8) for _ in range(40)]
            if check_mean("m", values, baseline=0.3, num_tests=tests_per_report).fires:
                fired = True
        reports_with_a_false_flag += fired

    rate = reports_with_a_false_flag / runs
    assert rate < 0.15, f"{rate:.0%} of clean months raised a false flag"


def test_rate_uses_the_baseline_spread_so_a_tiny_rate_stays_testable():
    """Strategy 4 fills about 2% of its signals. A sample that happens to
    return zero fills has zero sample variance; building the band from the
    sample would declare that a certainty."""
    finding = check_rate("S4 fill rate", hits=0, n=200, baseline=0.02, num_tests=45)

    assert finding.detectable > 0
    assert finding.detectable < 0.05


def test_rate_fires_on_a_genuine_collapse():
    """An instance that filled 40 of 200 last year and 0 of 200 this month is
    not noise - it is usually something broken upstream."""
    finding = check_rate("S4 fill rate", hits=0, n=200, baseline=0.20, num_tests=45)

    assert finding.fires


def test_rate_with_too_few_signals_stays_quiet():
    finding = check_rate("S4 fill rate", hits=0, n=5, baseline=0.20, num_tests=45)

    assert not finding.fires
    assert "too small" in finding.note


def test_every_quiet_finding_says_what_it_could_have_detected():
    """A quiet month must never read as 'no difference' - only as 'no
    difference bigger than this'."""
    rng = random.Random(3)
    values = [rng.gauss(0.3, 1.8) for _ in range(30)]
    finding = check_mean("S1 expectancy", values, baseline=0.3, num_tests=45)

    assert not finding.fires
    assert finding.detectable not in (0.0, float("inf"))
    assert "inside the band" in finding.note
