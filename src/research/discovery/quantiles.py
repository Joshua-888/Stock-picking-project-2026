"""WP5 fixed cross-sectional quantile analysis.

For each ``feature_asof`` date the feature is split into a FIXED number of
equal-count buckets with deterministic tie handling. The module reports the
bottom/top bucket mean excess return, the top-minus-bottom spread, the
outperformance-rate spread and monotonicity.

No cutoff is ever optimised: the bucket count is an a-priori constant, the
direction of the spread is only reported (never chosen), and quantiles are
computed WITHIN each date so no cross-date leakage can occur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class QuantileError(ValueError):
    """Raised when a quantile request is ill-formed."""


def _bucket_labels(values, buckets):
    """Deterministic quantile bucket per value: ties share a bucket by rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(len(values), dtype="float64")
    # Equal-width rank bins; the final bucket absorbs the remainder.
    labels = np.floor(ranks * buckets / len(values)).astype(int)
    labels[labels >= buckets] = buckets - 1
    return labels


def quantile_profile(frame, feature, target="future_12m_excess_return",
                     classification="outperform_12m", asof_col="feature_asof",
                     buckets=5, min_observations=30):
    """Return ``(profile, summary)`` for a fixed cross-sectional quantile split."""
    if feature not in frame.columns:
        raise QuantileError("feature %r absent from frame" % (feature,))
    if target not in frame.columns:
        raise QuantileError("target %r absent from frame" % (target,))
    buckets = int(buckets)
    if buckets < 2:
        raise QuantileError("buckets must be >= 2")

    columns = [asof_col, feature, target]
    if classification in frame.columns:
        columns.append(classification)
    working = frame.loc[:, columns].copy()
    working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[asof_col, feature, target])

    per_date_rows = []
    for asof, chunk in working.groupby(asof_col, sort=True):
        if len(chunk) < int(min_observations):
            continue
        labels = _bucket_labels(chunk[feature].to_numpy(dtype="float64"), buckets)
        for bucket in range(buckets):
            selection = chunk.iloc[np.where(labels == bucket)[0]]
            if selection.empty:
                continue
            record = {
                "feature_asof": str(asof)[:10],
                "bucket": bucket,
                "count": int(len(selection)),
                "mean_target": float(selection[target].mean()),
            }
            if classification in selection.columns:
                record["outperform_rate"] = float(
                    pd.to_numeric(selection[classification], errors="coerce").mean())
            per_date_rows.append(record)

    profile = pd.DataFrame(per_date_rows)
    summary = _summarize_quantiles(profile, buckets)
    summary["feature"] = feature
    return profile, summary


def _summarize_quantiles(profile, buckets):
    if profile.empty:
        return {"months": 0, "bottom_mean": None, "top_mean": None, "spread": None,
                "bottom_outperform": None, "top_outperform": None, "outperform_spread": None,
                "monotonic": None, "bucket_means": None}
    grouped = profile.groupby("bucket")["mean_target"].mean()
    bucket_means = {int(k): float(v) for k, v in grouped.items()}
    bottom = bucket_means.get(0)
    top = bucket_means.get(buckets - 1)
    bottom_out = top_out = None
    if "outperform_rate" in profile.columns:
        rates = profile.groupby("bucket")["outperform_rate"].mean()
        bottom_out = rates.get(0)
        top_out = rates.get(buckets - 1)
        bottom_out = None if bottom_out is None else float(bottom_out)
        top_out = None if top_out is None else float(top_out)
    ordered = [bucket_means.get(index) for index in range(buckets)]
    monotonic = None
    if all(value is not None for value in ordered):
        differences = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
        monotonic = bool(all(diff > 0 for diff in differences)) or bool(all(diff < 0 for diff in differences))
    return {
        "months": int(profile["feature_asof"].nunique()),
        "bottom_mean": bottom,
        "top_mean": top,
        "spread": None if bottom is None or top is None else float(top - bottom),
        "bottom_outperform": bottom_out,
        "top_outperform": top_out,
        "outperform_spread": None if bottom_out is None or top_out is None else float(top_out - bottom_out),
        "monotonic": monotonic,
        "bucket_means": bucket_means,
    }
