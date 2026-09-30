"""WP6 falsification battery (a model that reproduces placebo strength is broken).

WP6 reuses the WP5 placebo primitives where they apply and adds the MODEL-LEVEL
checks the frozen contract names:

* a SHUFFLED-target model placebo (does a model trained on a within-date shuffled
  target reproduce the real months' IC?);
* a noise-feature control (a pure-noise feature must not beat the real features);
* a label-shift / future guard (a deliberately forward-shifted feature must be
  rejected by the point-in-time machinery);
* a reduced-feature comparison (the frozen family-strategy subset vs all-eligible);
* a subperiod temporal-stability split (first vs second half of development months);
* a complexity comparison (nonlinear best vs linear Ridge/Logistic).

The module computes a single explicit ``stop`` flag. It issues NO verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.placebo import future_shift_guard as _future_shift_guard
from ..discovery.placebo import noise_feature as _noise_feature
from ..discovery.placebo import shuffled_target as _shuffled_target
from .contract import DEFAULT_CONFIG


class PlaceboRequestError(RuntimeError):
    """Raised when a placebo request is ill-formed."""


# Re-export the WP5 primitives so WP6 has a single placebo surface.
shuffled_target = _shuffled_target
noise_feature = _noise_feature
future_shift_guard = _future_shift_guard


def shuffled_target_placebo(real_mean_ic, placebo_mean_ic, floor=0.05):
    """STOP when a shuffled-target model reproduces the real months' magnitude."""
    real_magnitude = abs(float(real_mean_ic)) if real_mean_ic is not None else 0.0
    placebo_magnitude = abs(float(placebo_mean_ic)) if placebo_mean_ic is not None else 0.0
    threshold = max(real_magnitude, float(floor))
    suspicious = placebo_magnitude > 0.0 and placebo_magnitude >= threshold
    return {"real_abs_mean_ic": real_magnitude, "placebo_abs_mean_ic": placebo_magnitude,
            "floor": float(floor), "suspicious": bool(suspicious)}


def noise_feature_placebo(real_mean_ic, noise_mean_ic):
    """STOP when a pure-noise feature matches or beats the real features."""
    real_magnitude = abs(float(real_mean_ic)) if real_mean_ic is not None else 0.0
    noise_magnitude = abs(float(noise_mean_ic)) if noise_mean_ic is not None else 0.0
    suspicious = noise_magnitude > 0.0 and noise_magnitude >= real_magnitude
    return {"real_abs_mean_ic": real_magnitude, "noise_abs_mean_ic": noise_magnitude,
            "suspicious": bool(suspicious)}


def future_guard_report(frame, feature="twelve_month_momentum", days=365):
    """Confirm the PIT machinery flags a forward-shifted feature as unavailable."""
    record = future_shift_guard(frame, feature, days=days)
    violations = int(record["violates_point_in_time"].sum())
    return {"feature": feature, "rows": int(len(record)), "violations": violations,
            "all_violate": bool(violations == len(record) and len(record) > 0),
            "stop": bool(violations == 0),  # zero violations would mean no PIT enforcement
            "sample": record.head(3).to_dict(orient="records")}


def reduced_feature_comparison(full_metrics, reduced_metrics, metric="mean_ic"):
    """Compare the frozen family subset against all-eligible on a primary metric."""
    full_value = full_metrics.get(metric) if full_metrics else None
    reduced_value = reduced_metrics.get(metric) if reduced_metrics else None
    difference = None
    if full_value is not None and reduced_value is not None:
        difference = float(reduced_value - full_value)
    return {"metric": metric, "full": full_value, "reduced": reduced_value,
            "difference": difference}


def subperiod_stability(series, split_fraction=0.5):
    """Split a monthly metric series in half and report both halves' means."""
    if series is None or len(series) == 0:
        return {"months": 0, "first_half_mean": None, "second_half_mean": None,
                "sign_flip": None}
    ordered = series.sort_values("month", kind="mergesort")
    values = ordered["rank_ic"].to_numpy(dtype="float64")
    cut = max(1, int(len(values) * float(split_fraction)))
    first = values[:cut]
    second = values[cut:]
    first_mean = float(np.mean(first)) if len(first) else None
    second_mean = float(np.mean(second)) if len(second) else None
    sign_flip = None
    if first_mean is not None and second_mean is not None:
        sign_flip = bool(np.sign(first_mean) != np.sign(second_mean))
    return {"months": int(len(values)), "first_half_mean": first_mean,
            "second_half_mean": second_mean, "sign_flip": sign_flip}


def complexity_comparison(nonlinear_metrics, linear_metrics, logistic_metrics=None,
                         metric="mean_ic"):
    """Compare nonlinear best vs linear Ridge / Logistic on a primary metric."""
    nonlinear = nonlinear_metrics.get(metric) if nonlinear_metrics else None
    linear = linear_metrics.get(metric) if linear_metrics else None
    logistic = logistic_metrics.get(metric) if logistic_metrics else None
    return {
        "metric": metric,
        "nonlinear": nonlinear,
        "linear": linear,
        "logistic": logistic,
        "nonlinear_minus_linear": None if (nonlinear is None or linear is None) else float(nonlinear - linear),
        "nonlinear_minus_logistic": None if (nonlinear is None or logistic is None) else float(nonlinear - logistic),
        "complexity_justified": None if (nonlinear is None or linear is None) else bool(nonlinear > linear),
    }


def overall_stop_flag(checks):
    """STOP when any predeclared placebo check is suspicious."""
    suspicious = {}
    for name, payload in (checks or {}).items():
        suspicious[name] = bool(payload.get("suspicious") or payload.get("stop"))
    return {"stop": any(suspicious.values()), "checks": suspicious}
