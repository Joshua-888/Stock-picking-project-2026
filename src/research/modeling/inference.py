"""WP6 inference over the MONTHLY metric series with correct metric nulls.

Corrective defect #2. The core metrics have DIFFERENT null hypotheses and each
metric therefore declares an explicit comparator:

* cross-sectional Spearman IC (regression): ``H0: mean(IC_t) = 0``.
* ROC-AUC (classification): ``H0: mean(AUC_t - 0.5) = 0``. Inference is performed
  on the AUC SKILL series ``AUC_t - 0.5`` (the v1 code tested AUC against 0.0,
  which is meaningless because AUC is ~0.5 under no signal).
* PR-AUC: comparator = frozen base prevalence (never zero).
* Accuracy: comparator = base rate (never zero).
* Brier: comparator = frozen base-rate Brier benchmark ``mean((b - y)^2)``.
* Log loss: comparator = frozen base-rate log loss ``-mean(y ln b + (1-y) ln(1-b))``.

HAC/Newey-West is the PRIMARY instrument (with an explicit null mean); the circular
block bootstrap is SECONDARY. No pooled-row IID significance claim is ever made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.inference import _normal_sf, block_bootstrap_mean, hac_standard_error, iid_mean_test
from .contract import DEFAULT_CONFIG


class InferenceRequestError(RuntimeError):
    """Raised when an inference request is ill-formed."""


# ── Explicit comparator / null definition per metric (defect #2) ──────────────

METRIC_DEFINITIONS = {
    "rank_ic": {"null_mean": 0.0, "skill_transform": "identity",
                "null": "H0: mean(IC_t) = 0",
                "comparator": "zero"},
    "auc": {"null_mean": 0.5, "skill_transform": "subtract_0.5",
            "null": "H0: mean(AUC_t - 0.5) = 0",
            "comparator": "0.5 (chance)"},
    "pr_auc": {"null_mean": None, "skill_transform": "subtract_comparator",
               "null": "H0: mean(PR-AUC_t - base_prevalence_t) = 0",
               "comparator": "frozen base prevalence (per month)"},
    "accuracy": {"null_mean": None, "skill_transform": "subtract_comparator",
                 "null": "H0: mean(Accuracy_t - base_rate_t) = 0",
                 "comparator": "base rate / frozen benchmark"},
    "brier": {"null_mean": None, "skill_transform": "subtract_comparator",
              "null": "H0: mean(Brier_t - base_rate_brier_t) = 0",
              "comparator": "frozen base-rate Brier mean((b - y)^2)"},
    "log_loss": {"null_mean": None, "skill_transform": "subtract_comparator",
                 "null": "H0: mean(logloss_t - base_rate_logloss_t) = 0",
                 "comparator": "frozen base-rate log loss -mean(y ln b + (1-y) ln(1-b))"},
}


def metric_null(metric):
    """Return the explicit comparator/null definition for a metric."""
    definition = METRIC_DEFINITIONS.get(str(metric))
    if definition is None:
        raise InferenceRequestError("unknown metric %r" % (metric,))
    return definition


def skill_series(values, metric, comparator=None):
    """Transform a monthly metric series into its zero-null skill series.

    * ``rank_ic`` : unchanged (null 0).
    * ``auc``     : ``AUC_t - 0.5``.
    * other metrics: subtract the per-observation comparator (caller supplies it).
    """
    arr = np.asarray(values, dtype="float64")
    definition = metric_null(metric)
    transform = definition["skill_transform"]
    if transform == "identity":
        return arr.copy()
    if transform == "subtract_0.5":
        return arr - 0.5
    if comparator is None:
        raise InferenceRequestError(
            "metric %r requires an explicit per-observation comparator" % (metric,))
    return arr - np.asarray(comparator, dtype="float64")


def _clean(values):
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def hac_mean_test_against(values, null_mean=0.0, lags=12):
    """HAC/Newey-West test that ``mean(values) == null_mean`` (PRIMARY)."""
    arr = _clean(values)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": None, "null_mean": float(null_mean), "hac_se": None,
                "t_stat": None, "p_value": None}
    centred = arr - float(null_mean)
    mean = float(centred.mean())
    se = hac_standard_error(centred, lags=lags)
    if not se:
        return {"n": n, "mean": float(arr.mean()), "null_mean": float(null_mean),
                "hac_se": se, "t_stat": None, "p_value": None}
    t_stat = mean / se
    p_value = float(2.0 * _normal_sf(abs(t_stat)))
    return {"n": n, "mean": float(arr.mean()), "null_mean": float(null_mean),
            "skill_mean": mean, "hac_se": float(se), "t_stat": float(t_stat),
            "p_value": p_value}


def monthly_significance(series, config=None, column="rank_ic", metric=None, comparator=None):
    """HAC (primary) plus block bootstrap (secondary) with the correct null.

    ``metric`` selects the declared null/comparator. When omitted it is inferred
    from ``column`` (``rank_ic`` -> IC null, ``auc`` -> AUC-skill null). Inference
    is ALWAYS run on the skill series, whose null mean is exactly zero.
    """
    config = config or DEFAULT_CONFIG
    metric = metric or ("auc" if column == "auc" else "rank_ic")
    definition = metric_null(metric)
    if series is None or len(series) == 0:
        return {"n": 0, "mean": None, "skill_mean": None, "null_mean": 0.0,
                "metric": metric, "null": definition["null"], "comparator": definition["comparator"],
                "hac": None, "bootstrap": None, "iid_contrast": None}
    raw = _clean(series[column].to_numpy(dtype="float64"))
    if len(raw) < 2:
        return {"n": int(len(raw)), "mean": float(raw.mean()) if len(raw) else None,
                "skill_mean": None, "null_mean": 0.0, "metric": metric,
                "null": definition["null"], "comparator": definition["comparator"],
                "hac": None, "bootstrap": None, "iid_contrast": None}
    skill = skill_series(raw, metric, comparator=comparator)
    hac = hac_mean_test_against(skill, null_mean=0.0, lags=int(config.hac_lags))
    bootstrap = block_bootstrap_mean(skill, block=int(config.bootstrap_block),
                                     iterations=int(config.bootstrap_iterations),
                                     seed=int(config.bootstrap_seed))
    return {"n": int(len(raw)), "mean": float(raw.mean()), "skill_mean": float(skill.mean()),
            "metric": metric, "null": definition["null"], "comparator": definition["comparator"],
            "null_mean": 0.0, "hac": hac, "bootstrap": bootstrap,
            "iid_contrast": iid_mean_test(skill)}


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

    For AUC the shared ``-0.5`` shift cancels in the difference, so the null of the
    difference is zero regardless of the metric.
    """
    config = config or DEFAULT_CONFIG
    left, right, months = _aligned_series(left_series, right_series, column=column)
    if len(months) < 2:
        return {"months": int(len(months)), "left": left_name, "right": right_name,
                "mean_difference": None, "hac": None, "bootstrap": None,
                "left_better_fraction": None}
    difference = left - right
    hac = hac_mean_test_against(difference, null_mean=0.0, lags=int(config.hac_lags))
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
