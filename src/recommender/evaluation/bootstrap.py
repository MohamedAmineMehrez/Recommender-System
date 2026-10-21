"""User-level bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float]:
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype="float64")
    n = values.size
    for index in range(n_samples):
        sample = values[rng.integers(0, n, size=n)]
        estimates[index] = statistic(sample)
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_ratio_ci(
    numerators: np.ndarray,
    denominators: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    numerators = np.asarray(numerators, dtype="float64")
    denominators = np.asarray(denominators, dtype="float64")
    if numerators.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype="float64")
    n = numerators.size
    for index in range(n_samples):
        sample = rng.integers(0, n, size=n)
        denominator = denominators[sample].sum()
        estimates[index] = numerators[sample].sum() / denominator if denominator else 0.0
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_bootstrap_difference(
    numerators_a: np.ndarray,
    denominators_a: np.ndarray,
    numerators_b: np.ndarray,
    denominators_b: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired user-level bootstrap for a micro-averaged metric difference."""
    numerators_a = np.asarray(numerators_a, dtype="float64")
    denominators_a = np.asarray(denominators_a, dtype="float64")
    numerators_b = np.asarray(numerators_b, dtype="float64")
    denominators_b = np.asarray(denominators_b, dtype="float64")
    if not (
        numerators_a.size == numerators_b.size
        and denominators_a.size == denominators_b.size
    ):
        raise ValueError("paired arrays must have the same length")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype="float64")
    n = numerators_a.size
    for index in range(n_samples):
        sample = rng.integers(0, n, size=n)
        denominator_a = denominators_a[sample].sum()
        denominator_b = denominators_b[sample].sum()
        value_a = numerators_a[sample].sum() / denominator_a if denominator_a else 0.0
        value_b = numerators_b[sample].sum() / denominator_b if denominator_b else 0.0
        estimates[index] = value_a - value_b
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    observed_a = numerators_a.sum() / denominators_a.sum() if denominators_a.sum() else 0.0
    observed_b = numerators_b.sum() / denominators_b.sum() if denominators_b.sum() else 0.0
    return {
        "difference": float(observed_a - observed_b),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value_two_sided": float(min(1.0, 2 * min((estimates <= 0).mean(), (estimates >= 0).mean()))),
    }
