"""WP5 placebo battery and temporal guards.

A discovery pipeline that produces an impressive IC for a shuffled target is
broken. This module runs four independent falsification checks against the SAME
cross-sectional IC machinery used for a real feature:

* a SHUFFLED target, preserving each date's cross-section;
* a RANDOM-NOISE feature with no relationship to the outcome;
* a SIGN-RANDOMIZED feature (real values, randomly flipped sign per date);
* a FUTURE-SHIFT guard that verifies a deliberately forward-shifted feature is
  rejected by the point-in-time machinery.

If a placebo reproduces the strength of a real candidate the pipeline must stop
and be investigated, so ``placebo_verdict`` returns an explicit STOP signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PlaceboError(ValueError):
    """Raised when a placebo request is ill-formed."""


def shuffled_target(frame, target="future_12m_excess_return", seed=20260926, asof_col="feature_asof"):
    """A copy of ``frame`` whose target is shuffled WITHIN each date."""
    rng = np.random.default_rng(int(seed))
    working = frame.copy()
    shuffled = np.full(len(working), np.nan, dtype="float64")
    values = pd.to_numeric(working[target], errors="coerce").to_numpy()
    for _asof, index in working.groupby(asof_col, sort=True).groups.items():
        positions = working.index.get_indexer(index)
        local = values[positions]
        mask = np.isfinite(local)
        permuted = local.copy()
        permuted[mask] = rng.permutation(local[mask])
        shuffled[positions] = permuted
    working[target] = shuffled
    return working


def noise_feature(frame, name="__noise", seed=20260927):
    """A copy of ``frame`` with a deterministic standard-normal noise column."""
    rng = np.random.default_rng(int(seed))
    working = frame.copy()
    working[name] = rng.standard_normal(len(working))
    return working, name


def sign_randomized_feature(frame, feature, name=None, seed=20260928):
    """A copy of ``frame`` where each date's feature signs are flipped randomly."""
    rng = np.random.default_rng(int(seed))
    name = name or ("__sign_randomized_" + feature)
    working = frame.copy()
    values = pd.to_numeric(working[feature], errors="coerce").to_numpy(dtype="float64")
    flipped = values.copy()
    for _asof, index in working.groupby("feature_asof", sort=True).groups.items():
        positions = working.index.get_indexer(index)
        flips = rng.integers(0, 2, size=len(positions))
        flipped[positions] = np.where(flips == 1, -values[positions], values[positions])
    working[name] = flipped
    return working, name


def future_shift_guard(frame, feature, days=365, asof_col="feature_asof"):
    """Return a DATE rule proving a forward-shifted feature is not PIT-safe.

    The guard does not mutate the panel: it returns, for each observation, the
    instant the shifted value would have needed to be known and a boolean that
    the shift is future information. A PIT pipeline must drop every shifted row.
    """
    stamps = pd.to_datetime(frame[asof_col], errors="coerce", utc=True)
    required = stamps + pd.Timedelta(days=int(days))
    record = pd.DataFrame({
        "feature_asof": stamps.dt.strftime("%Y-%m-%d"),
        "required_available_at": required.dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "violates_point_in_time": required > stamps,
    })
    return record


def placebo_verdict(real_ic, placebo_ics, floor=0.05):
    """Decide whether a placebo reproduces the real IC.

    ``real_ic`` is the mean cross-sectional IC of the real feature; ``placebo_ics``
    maps placebo name -> mean IC. A placebo whose absolute mean IC reaches the
    real magnitude (or the ``floor``) is a STOP signal.
    """
    real_magnitude = abs(float(real_ic)) if real_ic is not None else 0.0
    verdicts = {}
    stop = False
    for name, value in (placebo_ics or {}).items():
        magnitude = abs(float(value)) if value is not None else 0.0
        threshold = max(real_magnitude, float(floor))
        suspicious = magnitude >= threshold and magnitude > 0.0
        verdicts[name] = {
            "mean_ic": None if value is None else float(value),
            "abs_mean_ic": magnitude,
            "suspicious": bool(suspicious),
        }
        stop = stop or suspicious
    return {"stop": bool(stop), "real_abs_mean_ic": real_magnitude,
            "floor": float(floor), "verdicts": verdicts}
