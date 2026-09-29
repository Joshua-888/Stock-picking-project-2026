"""WP5 dependence-aware inference over the monthly IC series.

Adjacent monthly IC values are NOT independent: each 12-month forward label is
observed by roughly twelve consecutive monthly predictions (WP4 measured ~11.8x
overlap). Treating the IC series as IID would understate the variance of its mean
and manufacture significance. This module therefore offers two dependence-aware
instruments and uses them as the primary evidence:

* a Newey-West / HAC standard error on the IC time series (Bartlett kernel);
* a circular block bootstrap of the IC series.

A naive IID ``t`` test is exposed only for contrast and is explicitly labelled
non-primary.
"""

from __future__ import annotations

import numpy as np

from .catalog import DiscoveryConfig


class InferenceError(ValueError):
    """Raised when an inference request is ill-formed."""


def _clean(values):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    return arr


def hac_standard_error(values, lags=12):
    """Newey-West (Bartlett) HAC standard error of the sample mean."""
    arr = _clean(values)
    n = len(arr)
    if n < 2:
        return None
    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / n)
    variance = gamma0
    max_lag = min(int(lags), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        variance += 2.0 * weight * covariance
    if variance <= 0.0:
        variance = gamma0 if gamma0 > 0.0 else 0.0
    if variance <= 0.0:
        return 0.0
    return float(np.sqrt(variance / n))


def hac_mean_test(values, lags=12):
    """HAC test that the mean IC differs from zero (the PRIMARY test)."""
    arr = _clean(values)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": None, "hac_se": None, "t_stat": None, "p_value": None}
    mean = float(arr.mean())
    se = hac_standard_error(arr, lags=lags)
    if not se:
        return {"n": n, "mean": mean, "hac_se": se, "t_stat": None, "p_value": None}
    t_stat = mean / se
    p_value = float(2.0 * _normal_sf(abs(t_stat)))
    return {"n": n, "mean": mean, "hac_se": float(se), "t_stat": float(t_stat), "p_value": p_value}


def _normal_sf(z):
    """Upper-tail standard-normal probability via the error function."""
    from math import erfc, sqrt
    return 0.5 * erfc(float(z) / sqrt(2.0))


def iid_mean_test(values):
    """Naive IID t test. NON-PRIMARY; reported only for contrast."""
    arr = _clean(values)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": None, "iid_se": None, "t_stat": None, "p_value": None}
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return {"n": n, "mean": mean, "iid_se": 0.0, "t_stat": None, "p_value": None}
    t_stat = mean / se
    return {"n": n, "mean": mean, "iid_se": se, "t_stat": float(t_stat),
            "p_value": float(2.0 * _normal_sf(abs(t_stat)))}


def block_bootstrap_mean(values, block=6, iterations=1000, seed=20260926):
    """Circular moving-block bootstrap CI for the mean monthly IC.

    Adjacent monthly IC values are NOT independent (a 12-month forward label is
    observed by roughly twelve consecutive monthly predictions), so resampling
    single observations as if IID would understate the variance of the mean and
    manufacture significance. This routine resamples *blocks* of consecutive
    observations: for every replicate it draws ``ceil(n / block)`` INDEPENDENT
    circular starting positions, concatenates the ``block``-long runs and truncates
    the result to ``n`` observations.

    That is a genuine dependence-preserving resample of overlapping windows, NOT a
    full-series rotation (a rotation would leave every replicate equal to the
    observed mean and force a zero bootstrap variance). ``mean_std`` reports the
    standard deviation of the bootstrap means so the non-degeneracy is observable,
    and the returned p-value is DESCRIPTIVE evidence only - it is never used to
    force a significance claim (HAC/Newey-West on the same series remains the
    primary inference instrument).
    """
    arr = _clean(values)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": None, "ci_low": None, "ci_high": None, "p_value": None,
                "iterations": 0, "block": int(block), "mean_std": None, "n_blocks": 0}
    block = max(1, int(block))
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    span = np.arange(block)
    means = np.empty(int(iterations), dtype="float64")
    for index in range(int(iterations)):
        starts = rng.integers(0, n, size=n_blocks)
        picks = ((starts[:, None] + span[None, :]) % n).reshape(-1)[:n]
        means[index] = arr[picks].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    mean = float(arr.mean())
    std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    # Two-sided bootstrap p-value around zero, centred by the observed mean.
    centred = means - mean
    p_value = float(np.mean(np.abs(centred) >= abs(mean)))
    if p_value == 0.0:
        p_value = float(1.0 / (int(iterations) + 1))
    return {"n": n, "mean": mean, "ci_low": float(low), "ci_high": float(high),
            "p_value": p_value, "iterations": int(iterations), "block": block,
            "mean_std": std, "n_blocks": n_blocks}


def dependence_diagnostics(values, config=None):
    """Lag-1..lag-k autocorrelation of the IC series (dependence evidence)."""
    config = config or DiscoveryConfig()
    arr = _clean(values)
    if len(arr) < 3:
        return {"autocorrelation": {}, "overlap_note": "series too short"}
    centred = arr - arr.mean()
    denominator = float(np.dot(centred, centred))
    autocorrelation = {}
    if denominator > 0.0:
        for lag in range(1, min(config.hac_lags, len(arr) - 1) + 1):
            autocorrelation[lag] = float(np.dot(centred[lag:], centred[:-lag]) / denominator)
    return {
        "autocorrelation": autocorrelation,
        "lag1": autocorrelation.get(1),
        "overlap_note": "12-month labels overlap; monthly IC autocorrelation is expected",
    }
