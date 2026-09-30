"""WP6 negative controls with explicit PASS / FAIL / STOP semantics.

Corrective defect #3. The v1 battery returned tautological flags
(``future_guard_report.stop = bool(violations == 0)``) and a single opaque
``overall_stop_flag`` driven by ``suspicious`` booleans. That made the STOP rule
inert and unreviewable. Every mandatory control now returns an explicit control
object:

    control_id, control_type, expected_behavior, observed_behavior,
    matched_real_config_id, real_metric, control_metric, failure_threshold,
    failure_condition, passed, stop_required, reason

Semantics (frozen in ``wp6_corrective_contract_v2.md``):

* PASS = the control behaves as expected under no signal / no leakage;
* FAIL = the control behaves like genuine predictive signal beyond the frozen
  tolerance;
* STOP = at least one MANDATORY control FAILs a frozen materiality rule.

``overall_stop`` is derived transparently from the control objects. PASS, FAIL and
STOP are all reachable (proven by fixtures). No control decides a score.

The negative control names a PREDECLARED matched real configuration by deterministic
identity (:mod:`src.research.modeling.identity`); no ``max(values, key=abs)`` or
best-observed reference is used anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..discovery.placebo import noise_feature as _noise_feature
from .contract import DEFAULT_CONFIG


class PlaceboRequestError(RuntimeError):
    """Raised when a placebo/control request is ill-formed."""


# ── Control-object factory (defect #3) ───────────────────────────────────────

CONTROL_FIELDS = (
    "control_id", "control_type", "expected_behavior", "observed_behavior",
    "matched_real_config_id", "real_metric", "control_metric", "failure_threshold",
    "failure_condition", "passed", "stop_required", "reason", "mandatory",
    "affected_tasks", "extra",
)


def control_object(control_id, control_type, expected_behavior, observed_behavior,
                   matched_real_config_id, real_metric, control_metric,
                   failure_threshold, failure_condition, passed, stop_required, reason,
                   mandatory=True, affected_tasks=(), extra=None):
    """Build one explicit, serialisable control-evaluation object."""
    return {
        "control_id": str(control_id),
        "control_type": str(control_type),
        "expected_behavior": str(expected_behavior),
        "observed_behavior": str(observed_behavior),
        "matched_real_config_id": (None if matched_real_config_id is None
                                   else str(matched_real_config_id)),
        "real_metric": (None if real_metric is None else float(real_metric)),
        "control_metric": (None if control_metric is None else float(control_metric)),
        "failure_threshold": (None if failure_threshold is None else float(failure_threshold)),
        "failure_condition": str(failure_condition),
        "passed": bool(passed),
        "stop_required": bool(stop_required),
        "reason": str(reason),
        "mandatory": bool(mandatory),
        "affected_tasks": list(affected_tasks or ()),
        "extra": dict(extra or {}),
    }


def _tolerance(real_metric, floor):
    """Frozen materiality tolerance: max(matched real magnitude, floor)."""
    real_magnitude = abs(float(real_metric)) if real_metric is not None else 0.0
    return max(real_magnitude, abs(float(floor or 0.0)))


def evaluate_metric_control(control_id, control_type, real_metric, control_metric,
                           matched_real_config_id, floor, expected_behavior,
                           observed_template, mandatory=True, affected_tasks=()):
    """Evaluate a metric-based control with explicit PASS/FAIL/STOP semantics.

    FAIL (=> STOP when mandatory) when the control's metric magnitude reaches the
    frozen tolerance ``max(|matched real|, floor)`` while the control should have
    produced none. A ``None`` control metric is treated as FAIL (the control could
    not be evaluated honestly).
    """
    threshold = _tolerance(real_metric, floor)
    if control_metric is None:
        observed = "control metric unavailable"
        passed = False
    else:
        magnitude = abs(float(control_metric))
        observed = observed_template % magnitude
        passed = bool(magnitude < threshold)
    failed = not passed
    reason = ("control within frozen tolerance under the null"
              if passed else
              "control metric reached the matched real magnitude / frozen tolerance")
    return control_object(
        control_id=control_id, control_type=control_type,
        expected_behavior=expected_behavior, observed_behavior=observed,
        matched_real_config_id=matched_real_config_id, real_metric=real_metric,
        control_metric=control_metric, failure_threshold=threshold,
        failure_condition="abs(control_metric) >= max(abs(real_metric), floor)",
        passed=passed, stop_required=failed, reason=reason,
        mandatory=mandatory, affected_tasks=affected_tasks,
    )


def overall_stop(controls):
    """Derive the STOP flag transparently from mandatory controls (defect #3)."""
    items = list(controls or [])
    failing = [item["control_id"] for item in items
               if item.get("mandatory") and item.get("stop_required")]
    failed_any = [item["control_id"] for item in items if not item.get("passed")]
    return {
        "stop": bool(failing),
        "mandatory_failing_controls": failing,
        "failed_controls": failed_any,
        "controls_evaluated": [item["control_id"] for item in items],
        "rule": "STOP iff any MANDATORY control has stop_required == True",
    }


# ── Frozen shuffle methodology (defect #5) ───────────────────────────────────


def shuffle_training_target(train_frame, target="future_12m_excess_return",
                            seed=20260926, asof_col="modeling_month"):
    """Within-date CROSS-SECTIONAL permutation of the TRAINING target ONLY.

    Frozen before the corrective rerun (contract defect #5) and chosen on
    METHODOLOGY, not on outcome. Rationale: a within-date permutation destroys the
    cross-sectional relationship the model is supposed to exploit while preserving
    (a) each date's marginal label distribution and (b) the panel's date structure,
    so the null the control tests is meaningful. The permutation is deterministic
    given ``seed`` and never reads a validation or holdout row.
    """
    rng = np.random.default_rng(int(seed))
    working = train_frame.copy()
    if target not in working.columns:
        raise PlaceboRequestError("training frame lacks target %r" % (target,))
    if asof_col not in working.columns:
        raise PlaceboRequestError("training frame lacks date column %r" % (asof_col,))
    shuffled = np.full(len(working), np.nan, dtype="float64")
    values = pd.to_numeric(working[target], errors="coerce").to_numpy(dtype="float64")
    for _asof, index in working.groupby(asof_col, sort=True).groups.items():
        positions = working.index.get_indexer(index)
        local = values[positions]
        mask = np.isfinite(local)
        permuted = local.copy()
        permuted[mask] = rng.permutation(local[mask])
        shuffled[positions] = permuted
    working[target] = shuffled
    return working


# Backwards-compatible alias: the discovery primitive is the SAME within-date
# permutation and is reused here after being documented as training-only.
shuffled_target = shuffle_training_target
noise_feature = _noise_feature


# ── Robustness checks (defect #8: EXECUTED, not inert) ───────────────────────


def subperiod_stability(series, split_fraction=0.5, column="rank_ic"):
    """Frozen half-split of the monthly series (first vs second half).

    No favorable-subperiod selection: the split point is the frozen fraction of the
    chronological series and both halves are always reported.
    """
    if series is None or len(series) == 0:
        return {"months": 0, "first_half_mean": None, "second_half_mean": None,
                "sign_flip": None, "split_fraction": float(split_fraction)}
    ordered = series.sort_values("month", kind="mergesort")
    values = ordered[column].to_numpy(dtype="float64")
    cut = max(1, int(len(values) * float(split_fraction)))
    first, second = values[:cut], values[cut:]
    first_mean = float(np.mean(first)) if len(first) else None
    second_mean = float(np.mean(second)) if len(second) else None
    sign_flip = None
    if first_mean is not None and second_mean is not None:
        sign_flip = bool(np.sign(first_mean) != np.sign(second_mean))
    return {"months": int(len(values)), "first_half_months": int(len(first)),
            "second_half_months": int(len(second)), "first_half_mean": first_mean,
            "second_half_mean": second_mean, "sign_flip": sign_flip,
            "split_fraction": float(split_fraction)}


def reduced_feature_comparison(full_metrics, reduced_metrics, metric="mean_metric",
                               hac=None, fdr_rejected=None, label=None):
    """Reduced strategies (F family subset; E ic_top_k) vs A_all_eligible.

    A reduced strategy is reported as justified ONLY when it improves the primary
    metric with an HAC-significant difference that ALSO survives FDR.
    """
    full_value = (full_metrics or {}).get(metric)
    reduced_value = (reduced_metrics or {}).get(metric)
    difference = None
    if full_value is not None and reduced_value is not None:
        difference = float(reduced_value - full_value)
    hac_p = (hac or {}).get("p_value")
    improves = difference is not None and difference > 0.0
    significant = hac_p is not None and hac_p < 0.05
    justified = bool(improves and significant and bool(fdr_rejected))
    return {"label": label, "metric": metric, "full": full_value, "reduced": reduced_value,
            "difference": difference, "hac_p_value": hac_p, "fdr_rejected": fdr_rejected,
            "improves": improves, "hac_significant": significant,
            "reduced_justified": justified}


def complexity_comparison(nonlinear_metrics, linear_metrics, metric="mean_metric",
                          hac=None, fdr_rejected=None, logistic_metrics=None):
    """Nonlinear vs simple linear on the primary metric.

    Complexity is justified ONLY with an HAC-significant improvement that survives
    FDR; a raw positive difference is never sufficient.
    """
    nonlinear = (nonlinear_metrics or {}).get(metric)
    linear = (linear_metrics or {}).get(metric)
    logistic = (logistic_metrics or {}).get(metric)
    difference = None
    if nonlinear is not None and linear is not None:
        difference = float(nonlinear - linear)
    hac_p = (hac or {}).get("p_value")
    improves = difference is not None and difference > 0.0
    significant = hac_p is not None and hac_p < 0.05
    justified = bool(improves and significant and bool(fdr_rejected))
    return {"metric": metric, "nonlinear": nonlinear, "linear": linear, "logistic": logistic,
            "nonlinear_minus_linear": difference, "hac_p_value": hac_p,
            "fdr_rejected": fdr_rejected, "improves": improves,
            "hac_significant": significant, "complexity_justified": justified}
