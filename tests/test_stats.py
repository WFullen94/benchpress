"""Tests for the stats module — no ML backend required."""

import pytest
import numpy as np
from benchpress.stats import (
    bootstrap_ci, mannwhitney_p, effect_size_cohen_d, summarize,
    wilcoxon_p, holm_bonferroni, sig_stars, thermal_trend,
)


def test_bootstrap_ci_contains_mean():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    lo, hi = bootstrap_ci(samples)
    mean = np.mean(samples)
    assert lo <= mean <= hi


def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci([3.14])
    assert lo == hi == pytest.approx(3.14)


def test_bootstrap_ci_wider_for_high_variance():
    stable = [1.0] * 50
    noisy = list(np.random.default_rng(0).normal(1, 5, 50))
    lo_s, hi_s = bootstrap_ci(stable)
    lo_n, hi_n = bootstrap_ci(noisy)
    assert (hi_n - lo_n) > (hi_s - lo_s)


def test_mannwhitney_identical_returns_high_p():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = mannwhitney_p(a, a)
    assert p > 0.05


def test_mannwhitney_different_returns_low_p():
    a = [1.0, 1.1, 1.2, 0.9, 1.0]
    b = [10.0, 10.1, 9.9, 10.2, 10.0]
    p = mannwhitney_p(a, b)
    assert p < 0.05


def test_cohen_d_same_distribution():
    a = [1.0, 2.0, 3.0]
    assert abs(effect_size_cohen_d(a, a)) < 1e-9


def test_cohen_d_large_effect():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0, 1, 30))
    b = list(rng.normal(10, 1, 30))  # means 10 apart, std ≈ 1 → d ≈ 10
    d = effect_size_cohen_d(a, b)
    assert abs(d) > 5.0


def test_summarize_keys():
    s = summarize([1, 2, 3, 4, 5])
    assert set(s.keys()) == {"mean", "median", "p95", "p99", "std", "min", "max"}
    assert s["min"] == 1.0
    assert s["max"] == 5.0


# ---------------------------------------------------------------------------
# wilcoxon_p
# ---------------------------------------------------------------------------

def test_wilcoxon_identical_high_p():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert wilcoxon_p(a, a) > 0.05


def test_wilcoxon_different_low_p():
    # n=5 pairs: minimum Wilcoxon p is 0.0625, need n≥6 for p<0.05
    a = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]
    b = [10.0, 10.1, 9.9, 10.2, 10.0, 10.05]
    assert wilcoxon_p(a, b) < 0.05


def test_wilcoxon_falls_back_to_mannwhitney_on_unequal_length():
    a = [1.0, 2.0, 3.0]
    b = [10.0, 11.0]
    p = wilcoxon_p(a, b)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_single():
    assert holm_bonferroni([0.03]) == pytest.approx([0.03])


def test_holm_bonferroni_caps_at_one():
    adjusted = holm_bonferroni([0.9, 0.8])
    assert all(p <= 1.0 for p in adjusted)


def test_holm_bonferroni_increases_p_values():
    raw = [0.01, 0.04]
    adj = holm_bonferroni(raw)
    for r, a in zip(raw, adj):
        assert a >= r


# ---------------------------------------------------------------------------
# sig_stars
# ---------------------------------------------------------------------------

def test_sig_stars():
    assert sig_stars(0.0005) == "***"
    assert sig_stars(0.005) == "**"
    assert sig_stars(0.03) == "*"
    assert sig_stars(0.1) == "ns"


# ---------------------------------------------------------------------------
# thermal_trend
# ---------------------------------------------------------------------------

def test_thermal_trend_flat_no_throttle():
    flat = [100.0] * 10
    result = thermal_trend(flat)
    assert result["throttling"] is False


def test_thermal_trend_strong_decline_detected():
    declining = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
    result = thermal_trend(declining)
    assert result["throttling"] is True
    assert result["tau"] < 0


def test_thermal_trend_too_few_samples():
    result = thermal_trend([100.0, 90.0])
    assert result["throttling"] is False


def test_thermal_trend_increasing_no_throttle():
    increasing = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0]
    result = thermal_trend(increasing)
    assert result["throttling"] is False
