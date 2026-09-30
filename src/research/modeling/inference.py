"""WP6 inference over the MONTHLY metric series.

This module reuses the WP5 dependence-aware instruments
(:func:`src.research.discovery.inference.hac_mean_test` and
:func:`~src.research.discovery.inference.block_bootstrap_mean`) and adds paired
monthly model comparisons. HAC/Newey-West is the PRIMARY instrument; the circular
block bootstrap is SECONDARY. No pooled-row IID significance claim is ever made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.inference import block_bootstrap_mean, hac_mean_test, iid_mean_test
from .contract import DEFAULT_CONFIG


class InferenceRequestError(RuntimeError):
    """Raised when an inference request is ill-formed."""


def _clean(values):
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def monthly_significance(series, config=None, column="rank_ic"):
    """HAC (primary) plus block bootstrap (secondary) on a monthly metric series."""
    config = config or DEFAULT_CONFIG
    if series is None or len(series) == 0:
        return {"n": 0, "mean": None, "hac": None, "bootstrap": None, "iid_contrast": None}
    values = _clean(series[column].to_numpy(dtype="float64"))
    if len(values) < 2:
        return {"n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
                "hac": None, "bootstrap": None, "iid_contrast": None}
    hac = hac_mean_test(values, lags=int(config.hac_lags))
    bootstrap = block_bootstrap_mean(values, block=int(config.bootstrap_block),
                                     iterations=int(config.bootstrap_iterations),
                                     seed=int(config.bootstrap_seed))
    return {"n": int(len(values)), "mean": float(values.mean()), "hac": hac,
            "bootstrap": bootstrap,
            "iid_contrast": iid_mean_test(values)}


def _aligned_series(left, right, column="rank_ic"):
    if left is None or right is None:
        return np.array([]), np.array([]), []
    lframe = left.loc[:, ["month", column]].rename(columns={column: "left"})
    rframe = right.loc[:, ["month", column]].rename(columns={column: "right"})
    merged = lframe.merge(rframe, on="month", how="inner").sort_values("month", kind="mergesort")
    merged = merged.loc[np.isfinite(merged["left"]) & np.isfinite(merged["right"])]
    return (merged["left"].to_numpy(dtype="float64"),
            merged["right"].to_numpy(dtype="float64"),
            merged["month"].tolist())


def paired_monthly_comparison(left_series, right_series, config=None, column="rank_ic",
                              left_name="left", right_name="right"):
    """Paired monthly difference test between two models on the SAME months.

    Returns the mean difference, its HAC p-value (primary), a bootstrap p-value
    (secondary) and the fraction of shared months where ``left`` beats ``right``.
    """
    config = config or DEFAULT_CONFIG
    left, right, months = _aligned_series(left_series, right_series, column=column)
    if len(months) < 2:
        return {"months": int(len(months)), "left": left_name, "right": right_name,
                "mean_difference": None, "hac": None, "bootstrap": None,
                "left_better_fraction": None}
    difference = left - right
    hac = hac_mean_test(difference, lags=int(config.hac_lags))
    bootstrap = block_bootstrap_mean(difference, block=int(config.bootstrap_block),
                                     iterations=int(config.bootstrap_iterations),
                                     seed=int(config.bootstrap_seed))
    return {
        "months": int(len(months)),
        "left": left_name,
        "right": right_name,
        "mean_difference": float(difference.mean()),
        "hac": hac,
        "bootstrap": bootstrap,
        "left_better_fraction": float(np.mean(difference > 0.0)),
        "mean_left": float(left.mean()),
        "mean_right": float(right.mean()),
    }


def paired_across_folds(per_fold_series, left_name, right_name, config=None):
    """Run the paired monthly comparison inside every fold that has both models."""
    config = config or DEFAULT_CONFIG
    records = []
    differences = []
    for fold in sorted(per_fold_series):
        left = per_fold_series[fold].get(left_name)
        right = per_fold_series[fold].get(right_name)
        if left is None or right is None:
            continue
        comparison = paired_monthly_comparison(left, right, config=config,
                                               left_name=left_name, right_name=right_name)
        comparison["fold"] = fold
        records.append(comparison)
        if comparison["mean_difference"] is not None:
            differences.append(comparison["mean_difference"])
    pooled = None
    if len(differences) >= 2:
        pooled = hac_mean_test(np.asarray(differences, dtype="float64"), lags=min(int(config.hac_lags), len(differences) - 1))
    return {"folds": records, "fold_level_difference_hac": pooled}
