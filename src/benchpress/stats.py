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
