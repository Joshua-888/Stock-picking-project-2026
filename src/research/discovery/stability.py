"""WP5 temporal stability of the monthly cross-sectional IC.

A feature that only works in one short window is not a candidate. This module
slices the monthly IC series by calendar year and into early/middle/recent
thirds, and reports direction consistency, sign flips and coverage drift across
periods. Nothing is selected here; a short-period-only signal is FLAGGED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StabilityError(ValueError):
    """Raised when a stability request is ill-formed."""


def _sign(value):
    if value is None:
        return 0
    return 1 if value > 0 else (-1 if value < 0 else 0)


def yearly_ic(series):
    """Per-calendar-year mean IC and month count from a monthly IC series."""
    if series is None or len(series) == 0:
        return {}
    stamps = pd.to_datetime(series["feature_asof"], errors="coerce", utc=True)
    working = series.assign(__year=stamps.dt.year)
    result = {}
    for year, chunk in working.groupby("__year", sort=True):
        if pd.isna(year):
            continue
        values = chunk["rank_ic"].to_numpy(dtype="float64")
        result[int(year)] = {
            "months": int(len(values)),
            "mean_ic": float(np.mean(values)),
            "positive_share": float(np.mean(values > 0.0)),
        }
    return result


def period_slices(series, count=3):
    """Split the chronological IC series into ``count`` equal contiguous parts."""
    if series is None or len(series) == 0:
        return []
    ordered = series.sort_values("feature_asof", kind="mergesort").reset_index(drop=True)
    count = max(2, int(count))
    slices = np.array_split(np.arange(len(ordered)), count)
    labels = ["early", "middle", "recent"] if count == 3 else ["part_%d" % i for i in range(count)]
    results = []
    for label, index in zip(labels, slices):
        chunk = ordered.iloc[index]
        if chunk.empty:
            continue
        values = chunk["rank_ic"].to_numpy(dtype="float64")
        results.append({
            "label": label,
            "months": int(len(values)),
            "mean_ic": float(np.mean(values)),
            "start": str(chunk["feature_asof"].iloc[0])[:10],
            "end": str(chunk["feature_asof"].iloc[-1])[:10],
        })
    return results


def coverage_by_year(frame, feature, asof_col="feature_asof"):
    """Per-year present/total coverage for one feature (coverage drift)."""
    if feature not in frame.columns or asof_col not in frame.columns:
        return {}
    stamps = pd.to_datetime(frame[asof_col], errors="coerce", utc=True)
    values = pd.to_numeric(frame[feature], errors="coerce")
    working = pd.DataFrame({"__year": stamps.dt.year, "__present": values.notna()})
    working = working.dropna(subset=["__year"])
    result = {}
    for year, chunk in working.groupby("__year", sort=True):
        total = int(len(chunk))
        present = int(chunk["__present"].sum())
        result[int(year)] = {
            "total": total,
            "present": present,
            "coverage": float(present / total) if total else 0.0,
        }
    return result


def stability_report(series, frame=None, feature=None, min_months=24,
                     stable_positive_share=0.6, stable_negative_share=0.6):
    """Summarise direction and coverage stability across time."""
    months = 0 if series is None else int(len(series))
    years = yearly_ic(series)
    slices = period_slices(series)
    yearly_means = [record["mean_ic"] for record in years.values()]
    signs = {_sign(value) for value in yearly_means}
    direction_consistent = len([s for s in signs if s != 0]) <= 1 and 0 not in signs

    recent = slices[-1]["mean_ic"] if slices else None
    early = slices[0]["mean_ic"] if slices else None
    sign_flip = (early is not None and recent is not None and _sign(early) != 0
                 and _sign(recent) != 0 and _sign(early) != _sign(recent))

    overall_mean = float(np.mean(series["rank_ic"].to_numpy(dtype="float64"))) if months else None
    positive_share = None
    if months:
        values = series["rank_ic"].to_numpy(dtype="float64")
        positive_share = float(np.mean(values > 0.0))

    short_period_only = False
    if months and len(slices) >= 2:
        magnitudes = [abs(record["mean_ic"]) for record in slices]
        total = sum(magnitudes)
        if total > 0.0 and max(magnitudes) / total >= 0.8:
            short_period_only = True

    coverage_drift = None
    coverage = None
    if frame is not None and feature is not None:
        coverage = coverage_by_year(frame, feature)
        values = [record["coverage"] for record in coverage.values()]
        if len(values) >= 2:
            coverage_drift = float(max(values) - min(values))

    return {
        "months": months,
        "overall_mean_ic": overall_mean,
        "positive_share": positive_share,
        "yearly": years,
        "slices": slices,
        "direction_consistent": bool(direction_consistent),
        "sign_flip": bool(sign_flip),
        "short_period_only": bool(short_period_only),
        "coverage_by_year": coverage,
        "coverage_drift": coverage_drift,
        "sufficient_months": months >= int(min_months),
        "stable_positive_share": float(stable_positive_share),
        "stable_negative_share": float(stable_negative_share),
    }
