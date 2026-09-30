"""WP6 metrics.

The CORE predictive metrics are month-by-month and rank-based, so overlapping
12-month labels are never treated as independent rows:

* regression: cross-sectional monthly Spearman rank IC, its mean, its ICIR, the
  positive-IC-month fraction, MAE and RMSE;
* classification: ROC-AUC, PR-AUC, log loss, Brier and calibration diagnostics.

LEVEL-RETURN portfolio diagnostics (top/bottom quintile and decile mean observed
excess return, top-minus-bottom spread) are reported SEPARATELY because they are
not predictive metrics: they depend on the realised level of returns and are
dominated by a few months. They must never be read as evidence of predictive
skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.cross_sectional import spearman


class MetricError(RuntimeError):
    """Raised when a metric request is ill-formed."""


def _finite(values):
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


# ── Core cross-sectional rank metrics ────────────────────────────────────────

def monthly_ic_series(frame, prediction, target, asof_col="modeling_month",
                      min_observations=30):
    """Per-month cross-sectional Spearman rank IC of a prediction vs the target."""
    for column in (prediction, target, asof_col):
        if column not in frame.columns:
            raise MetricError("frame lacks %r" % (column,))
    working = frame.loc[:, [asof_col, prediction, target]].copy()
    working[prediction] = pd.to_numeric(working[prediction], errors="coerce")
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[asof_col, prediction, target])
    records, skipped = [], []
    for asof, chunk in working.groupby(asof_col, sort=True):
        paired = int(len(chunk))
        if paired < int(min_observations):
            skipped.append({"month": str(asof), "paired_obs": paired,
                            "reason": "below_min_observations"})
            continue
        ic = spearman(chunk[prediction].to_numpy(), chunk[target].to_numpy())
        if ic is None or not np.isfinite(ic):
            skipped.append({"month": str(asof), "paired_obs": paired,
                            "reason": "no_rank_variance"})
            continue
        records.append({"month": str(asof), "paired_obs": paired, "rank_ic": float(ic)})
    series = pd.DataFrame(records, columns=["month", "paired_obs", "rank_ic"])
    if not series.empty:
        series = series.sort_values("month", kind="mergesort").reset_index(drop=True)
    return series, {"months": int(len(series)), "skipped": skipped,
                    "min_observations": int(min_observations)}


def summarize_ic(series):
    """Mean IC, IC std, ICIR and positive-IC-month fraction from a monthly series."""
    if series is None or len(series) == 0:
        return {"months": 0, "mean_ic": None, "ic_std": None, "icir": None,
                "positive_ic_fraction": None, "mean_paired_obs": None}
    values = series["rank_ic"].to_numpy(dtype="float64")
    months = int(len(values))
    mean_ic = float(np.mean(values))
    std_ic = float(np.std(values, ddof=1)) if months > 1 else 0.0
    return {
        "months": months,
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "icir": float(mean_ic / std_ic) if std_ic > 0.0 else None,
        "positive_ic_fraction": float(np.mean(values > 0.0)),
        "mean_paired_obs": float(series["paired_obs"].mean()),
    }


def regression_errors(frame, prediction, target):
    """MAE and RMSE over finite prediction/target pairs."""
    if prediction not in frame.columns or target not in frame.columns:
        raise MetricError("frame lacks %r/%r" % (prediction, target))
    subset = frame.loc[:, [prediction, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        return {"n": 0, "mae": None, "rmse": None}
    error = subset[prediction].to_numpy() - subset[target].to_numpy()
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


# ── Classification metrics ───────────────────────────────────────────────────

def classification_metrics(frame, score, target, threshold=0.5):
    """ROC-AUC, PR-AUC, log loss and Brier score for a probability score."""
    for column in (score, target):
        if column not in frame.columns:
            raise MetricError("frame lacks %r" % (column,))
    subset = frame.loc[:, [score, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        return {"n": 0, "roc_auc": None, "pr_auc": None, "log_loss": None,
                "brier": None, "base_rate": None}
    y = subset[target].to_numpy(dtype="float64")
    p = np.clip(subset[score].to_numpy(dtype="float64"), 1e-9, 1 - 1e-9)
    base_rate = float(np.mean(y))
    roc_auc = pr_auc = None
    if 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score, roc_auc_score
        roc_auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    return {"n": int(len(y)), "roc_auc": roc_auc, "pr_auc": pr_auc,
            "log_loss": log_loss, "brier": brier, "base_rate": base_rate}


def calibration_table(frame, score, target, buckets=10):
    """Reliability bins plus a logistic intercept/slope calibration fit."""
    subset = frame.loc[:, [score, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        return {"bins": [], "intercept": None, "slope": None}
    y = subset[target].to_numpy(dtype="float64")
    p = subset[score].to_numpy(dtype="float64")
    order = np.argsort(p, kind="mergesort")
    positions = np.array_split(order, int(buckets))
    bins = []
    for index, chunk in enumerate(positions, start=1):
        if len(chunk) == 0:
            continue
        bins.append({"bin": index, "count": int(len(chunk)),
                     "mean_predicted": float(np.mean(p[chunk])),
                     "observed_rate": float(np.mean(y[chunk]))})
    intercept = slope = None
    if 0 < y.sum() < len(y):
        from sklearn.linear_model import LogisticRegression
        logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(logit.reshape(-1, 1), y.astype(int))
        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0][0])
    return {"bins": bins, "intercept": intercept, "slope": slope}


# ── Level-return portfolio diagnostics (kept separate) ───────────────────────

def _bucket_returns(frame, prediction, target, buckets):
    subset = frame.loc[:, ["modeling_month", prediction, target]].copy()
    subset[prediction] = pd.to_numeric(subset[prediction], errors="coerce")
    subset[target] = pd.to_numeric(subset[target], errors="coerce")
    subset = subset.dropna(subset=[prediction, target])
    if subset.empty:
        return {}
    means = {}
    for _month, chunk in subset.groupby("modeling_month", sort=True):
        values = chunk[prediction].to_numpy(dtype="float64")
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype="float64")
        ranks[order] = np.arange(len(values), dtype="float64")
        labels = np.floor(ranks * buckets / len(values)).astype(int)
        labels[labels >= buckets] = buckets - 1
        target_values = chunk[target].to_numpy(dtype="float64")
        for bucket in range(buckets):
            selection = target_values[labels == bucket]
            if len(selection):
                means.setdefault(bucket, []).append(float(np.mean(selection)))
    return {bucket: float(np.mean(values)) for bucket, values in means.items()}


def quantile_spread(frame, prediction, target, buckets=5):
    """LEVEL-RETURN diagnostic: mean top-minus-bottom bucket excess return."""
    means = _bucket_returns(frame, prediction, target, buckets)
    bottom, top = means.get(0), means.get(buckets - 1)
    return {"bucket_means": means,
            "top_mean": top, "bottom_mean": bottom,
            "spread": None if bottom is None or top is None else float(top - bottom)}


def decile_spread(frame, prediction, target):
    """LEVEL-RETURN diagnostic at decile resolution."""
    return quantile_spread(frame, prediction, target, buckets=10)


# ── Monthly classification series (parallel to the monthly IC series) ─────────

def monthly_auc_series(frame, score, target, asof_col="modeling_month", min_observations=30):
    """Per-month cross-sectional ROC-AUC of a probability score vs a binary label."""
    for column in (score, target, asof_col):
        if column not in frame.columns:
            raise MetricError("frame lacks %r" % (column,))
    working = frame.loc[:, [asof_col, score, target]].copy()
    working[score] = pd.to_numeric(working[score], errors="coerce")
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[asof_col, score, target])
    from sklearn.metrics import roc_auc_score
    records, skipped = [], []
    for asof, chunk in working.groupby(asof_col, sort=True):
        paired = int(len(chunk))
        labels = chunk[target].to_numpy(dtype="float64")
        if paired < int(min_observations) or 0 == labels.sum() or labels.sum() == paired:
            skipped.append({"month": str(asof), "paired_obs": paired,
                            "reason": "below_min_observations_or_single_class"})
            continue
        auc = float(roc_auc_score(labels, chunk[score].to_numpy(dtype="float64")))
        if not np.isfinite(auc):
            skipped.append({"month": str(asof), "paired_obs": paired, "reason": "undefined_auc"})
            continue
        records.append({"month": str(asof), "paired_obs": paired, "auc": auc})
    series = pd.DataFrame(records, columns=["month", "paired_obs", "auc"])
    if not series.empty:
        series = series.sort_values("month", kind="mergesort").reset_index(drop=True)
    return series, {"months": int(len(series)), "skipped": skipped,
                    "min_observations": int(min_observations)}


def monthly_brier_series(frame, score, target, asof_col="modeling_month", min_observations=30):
    """Per-month cross-sectional Brier score of a probability score."""
    working = frame.loc[:, [asof_col, score, target]].copy()
    working[score] = pd.to_numeric(working[score], errors="coerce")
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[asof_col, score, target])
    records = []
    for asof, chunk in working.groupby(asof_col, sort=True):
        if int(len(chunk)) < int(min_observations):
            continue
        error = chunk[score].to_numpy(dtype="float64") - chunk[target].to_numpy(dtype="float64")
        records.append({"month": str(asof), "paired_obs": int(len(chunk)),
                        "brier": float(np.mean(error ** 2))})
    series = pd.DataFrame(records, columns=["month", "paired_obs", "brier"])
    return series, {"months": int(len(series))}
