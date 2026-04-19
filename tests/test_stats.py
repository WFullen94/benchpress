"""Tests for the stats module — no ML backend required."""

import pytest
import numpy as np
from benchpress.stats import bootstrap_ci, mannwhitney_p, effect_size_cohen_d, summarize


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
