"""WP5 feature quality: factual, per-candidate diagnostics.

For every candidate feature this reports coverage, first/last observation, the
missing share, cross-sectional and temporal variation, stale / near-constant /
duplicate-value counts and a temporal-safety verdict. It never silently drops a
feature: a candidate with no usable values is reported as ``NO_COVERAGE`` with
the factual reason attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_STATUSES = (
    "OK",
    "NO_COVERAGE",
    "LOW_COVERAGE",
    "NEAR_CONSTANT",
    "TEMPORALLY_UNSAFE",
)

FORBIDDEN_FEATURE_INPUTS = (
    "future_12m_excess_return", "future_12m_stock_return", "future_12m_benchmark_return",
    "outperform_12m", "target_known_at", "target_end", "target_censor_reason",
)


class FeatureQualityError(ValueError):
    """Raised when a feature-quality request is ill-formed."""


def _series(frame, name):
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce")


def _stale_share(frame, name, asof_col="feature_asof", max_repeats=13):
    """Share of observations whose value repeats more than ``max_repeats`` times.

    A quarterly filing seen monthly legitimately repeats, so the threshold is set
    above a full year of monthly snapshots; only structurally stale series (a
    constant carried for years) trip it.
    """
    if name not in frame.columns or asof_col not in frame.columns:
        return None
    working = frame.loc[:, ["security_id", asof_col, name]].copy()
    working[asof_col] = pd.to_datetime(working[asof_col], errors="coerce", utc=True)
    working = working.dropna(subset=[asof_col])
    working = working.sort_values(["security_id", asof_col], kind="mergesort")
    stale, total = 0, 0
    for _security, chunk in working.groupby("security_id", sort=False):
        values = chunk[name].to_numpy()
        run = 0
        previous = object()
        for value in values:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                previous = object()
                run = 0
                continue
            total += 1
            if previous is not object() and isinstance(previous, float) and not np.isnan(previous) \
                    and float(value) == float(previous):
                run += 1
                if run >= max_repeats:
                    stale += 1
            else:
                run = 0
            previous = value
    if total == 0:
        return None
    return stale / total


def feature_quality(frame, name, low_coverage_threshold=0.35, asof_col="feature_asof"):
    """Return the factual quality record for one candidate feature."""
    if name in FORBIDDEN_FEATURE_INPUTS:
        raise FeatureQualityError("%s is a forward-looking label, not a feature" % name)
    series = _series(frame, name)
    row_count = int(len(frame))
    if series is None:
        return {
            "feature": name, "status": "NO_COVERAGE", "rows": row_count, "present": 0,
            "coverage": 0.0, "missing_share": 1.0, "first_asof": None, "last_asof": None,
            "cross_sectional_std_median": None, "temporal_std_median": None,
            "distinct_values": 0, "duplicate_value_share": None, "stale_value_share": None,
            "temporal_safety": "OK", "notes": "feature column absent from the panel",
        }
    present_mask = series.notna()
    present = int(present_mask.sum())
    coverage = (present / row_count) if row_count else 0.0
    if present == 0:
        first_asof = last_asof = None
    else:
        stamps = pd.to_datetime(frame.loc[present_mask, asof_col], errors="coerce", utc=True)
        first_asof = str(stamps.min().date()) if len(stamps.dropna()) else None
        last_asof = str(stamps.max().date()) if len(stamps.dropna()) else None

    observed = series.dropna()
    distinct_values = int(observed.nunique()) if present else 0

    cross_sectional_std = None
    if present and asof_col in frame.columns:
        grouped = frame.assign(__v=series).dropna(subset=["__v"]).groupby(asof_col)["__v"]
        stds = grouped.std(ddof=0).dropna()
        counts = grouped.count()
        stds = stds[counts >= 2]
        if len(stds):
            cross_sectional_std = float(stds.median())

    temporal_std = None
    if present and "security_id" in frame.columns:
        per_security = frame.assign(__v=series).dropna(subset=["__v"]).groupby("security_id")["__v"]
        stds = per_security.std(ddof=0).dropna()
        if len(stds):
            temporal_std = float(stds.median())

    duplicate_value_share = (1.0 - distinct_values / present) if present else None
    stale_value_share = _stale_share(frame, name, asof_col=asof_col)

    temporal_safety = "OK"
    if "available_at" in frame.columns and "feature_asof" in frame.columns:
        available = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
        asof = pd.to_datetime(frame["feature_asof"], errors="coerce", utc=True)
        violations = int(((available > asof) & present_mask).sum())
        if violations:
            temporal_safety = "VIOLATION"

    status = "OK"
    notes = ""
    if coverage < low_coverage_threshold:
        status = "LOW_COVERAGE"
        notes = "coverage %.3f below %.2f" % (coverage, low_coverage_threshold)
    if present and cross_sectional_std is not None and cross_sectional_std <= 0.0 and temporal_std in (None, 0.0):
        status = "NEAR_CONSTANT"
        notes = "no cross-sectional or temporal variation"
    if temporal_safety == "VIOLATION":
        status = "TEMPORALLY_UNSAFE"
        notes = "availability after the prediction instant"

    return {
        "feature": name,
        "status": status,
        "rows": row_count,
        "present": present,
        "coverage": float(coverage),
        "missing_share": float(1.0 - coverage),
        "first_asof": first_asof,
        "last_asof": last_asof,
        "cross_sectional_std_median": cross_sectional_std,
        "temporal_std_median": temporal_std,
        "distinct_values": distinct_values,
        "duplicate_value_share": duplicate_value_share,
        "stale_value_share": stale_value_share,
        "temporal_safety": temporal_safety,
        "notes": notes,
    }


def feature_quality_table(frame, features, low_coverage_threshold=0.35, asof_col="feature_asof"):
    """Quality records for every candidate, in declaration order."""
    return [feature_quality(frame, name, low_coverage_threshold=low_coverage_threshold, asof_col=asof_col)
            for name in features]
