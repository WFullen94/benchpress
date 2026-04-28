"""Statistical utilities: bootstrap CIs and pairwise significance testing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bootstrap_ci(
    samples: list[float],
    statistic: callable = np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for a statistic.

    Uses the percentile method (BCa would be more accurate but overkill here).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(samples)
    if len(arr) < 2:
        val = float(statistic(arr))
        return (val, val)

    boot_stats = [
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def mannwhitney_p(a: list[float], b: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value for comparing two sample sets."""
    from scipy.stats import mannwhitneyu

    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def effect_size_cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def summarize(samples: list[float]) -> dict[str, float]:
    """Return mean, median, p95, p99, std for a sample list."""
    arr = np.asarray(samples)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples.

    Use this instead of Mann-Whitney when both runs used the same prompts in
    the same order — it has more statistical power for paired data.
    Falls back to Mann-Whitney if lengths differ.
    """
    from scipy.stats import wilcoxon

    if len(a) != len(b):
        return mannwhitney_p(a, b)
    if len(a) < 2:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    _, p = wilcoxon(a, b, alternative="two-sided")
    return float(p)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values in the same order as the input.
    Use when running multiple pairwise comparisons to control family-wise
    error rate without being as conservative as plain Bonferroni.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def sig_stars(p: float) -> str:
    """Return significance stars for a p-value: *** / ** / * / ns."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def thermal_trend(tps_samples: list[float]) -> dict:
    """Run a Mann-Kendall trend test on ordered tokens/sec samples.

    Returns a dict with:
        tau     — Kendall's τ (negative = downward trend = throttling)
        p_value — two-sided p-value
        throttling — True if p < 0.05 and τ < 0

    A significant negative τ means throughput declined monotonically across
    runs, which is the signature of thermal throttling on M-series chips.
    """
    n = len(tps_samples)
    if n < 4:
        return {"tau": 0.0, "p_value": 1.0, "throttling": False}

    arr = np.asarray(tps_samples)
    # Compute Kendall's S statistic manually (no extra dependency)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = arr[j] - arr[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S under H0 (no ties case — good enough for small n)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if var_s == 0:
        return {"tau": 0.0, "p_value": 1.0, "throttling": False}

    # Continuity-corrected z
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    from scipy.stats import norm
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    tau = s / (0.5 * n * (n - 1))

    return {
        "tau": round(float(tau), 3),
        "p_value": round(p_value, 4),
        "throttling": bool(p_value < 0.05 and tau < 0),
    }
