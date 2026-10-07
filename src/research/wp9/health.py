"""WP9 operational health and descriptive drift monitoring.

This module reports per-snapshot operational health without ever changing the
frozen champion. It computes only descriptive diagnostics; drift comparison is
optional, descriptive, and emits ``SHADOW_DATA_DRIFT_ALERT`` flags. No
retraining, recalibration, feature selection, or model mutation is attempted.

Operational health may return *blocking* reasons for a snapshot (for example,
zero scoreable securities or a failed artifact-hash verification). Blocking is
about snapshot validity only; it never changes the model.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..fingerprints import fingerprint_obj
from .champion import FROZEN_FEATURES

HEALTH_SCHEMA_VERSION = "wp9_operational_health_v1"

CONTRACT_MONITORING_FIELDS = (
    "universe_size",
    "scoreable_count",
    "coverage_rate",
    "feature_missingness",
    "identity_failures",
    "provider_failures",
    "artifact_hash_verification",
    "score_distribution",
    "rank_concentration",
)


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _distribution(series: Sequence[float]) -> Dict[str, float | None]:
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std(ddof=0)),
        "min": _safe_float(values.min()),
        "p01": _safe_float(values.quantile(0.01) if len(values) > 1 else values.iloc[0]),
        "p05": _safe_float(values.quantile(0.05) if len(values) > 1 else values.iloc[0]),
        "p25": _safe_float(values.quantile(0.25)),
        "p50": _safe_float(values.quantile(0.50)),
        "p75": _safe_float(values.quantile(0.75)),
        "p95": _safe_float(values.quantile(0.95) if len(values) > 1 else values.iloc[0]),
        "p99": _safe_float(values.quantile(0.99) if len(values) > 1 else values.iloc[0]),
        "max": _safe_float(values.max()),
    }


def _rank_concentration(ranks: Sequence[int]) -> Dict[str, float]:
    values = np.asarray([int(value) for value in ranks])
    count = int(len(values))
    if count == 0:
        return {"rank_count": 0, "top_decile_share_of_rank_sum": None}
    unique = len(set(values.tolist()))
    # Concentration of top rank mass; descriptive only. Top 10% of ordered
    # ranks should own ~5.5% of the rank-sum under a perfectly uniform
    # 1..n ranking. A materially larger share is a descriptive red flag.
    top_n = max(1, int(round(count * 0.10)))
    top_share = float(values[values <= top_n].sum() / values.sum()) if values.sum() else None
    return {
        "rank_count": count,
        "unique_ranks": unique,
        "top_decile_rank_mass_share": top_share,
        "perfect_uniform_reference": float((top_n * (top_n + 1) / 2.0) / (count * (count + 1) / 2.0))
        if count > 1
        else 1.0,
    }


def compare_to_reference(
    current: Sequence[float],
    reference: Sequence[float],
    *,
    label: str = "score",
    relative_std_shift: float = 2.0,
    relative_mean_shift: float = 2.0,
) -> Dict[str, Any]:
    """Descriptive shift of ``current`` versus a frozen reference distribution.

    Alerts only. No adaptive action is available or attempted.
    """
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna()
    result: Dict[str, Any] = {
        "label": label,
        "reference_n": int(len(ref)),
        "current_n": int(len(cur)),
    }
    if ref.empty or cur.empty:
        result["alert"] = "SKIPPED"
        result["reason"] = "missing reference or current values"
        return result
    ref_std = float(ref.std(ddof=0))
    cur_std = float(cur.std(ddof=0))
    mean_shift = abs(float(cur.mean()) - float(ref.mean()))
    std_shift_ratio = abs(cur_std / ref_std) if ref_std > 0 else (1.0 if cur_std > 0 else 0.0)
    alerts = []
    if ref_std > 0 and std_shift_ratio > relative_std_shift:
        alerts.append("SHADOW_DATA_DRIFT_ALERT:std_shift")
    if ref_std > 0 and mean_shift > relative_mean_shift * ref_std:
        alerts.append("SHADOW_DATA_DRIFT_ALERT:mean_shift")
    result["reference_mean"] = _safe_float(ref.mean())
    result["reference_std"] = _safe_float(ref_std)
    result["current_mean"] = _safe_float(cur.mean())
    result["current_std"] = _safe_float(cur_std)
    result["std_shift_ratio"] = _safe_float(std_shift_ratio)
    result["mean_shift_in_reference_std_units"] = _safe_float(
        mean_shift / ref_std if ref_std > 0 else None
    )
    result["alerts"] = alerts if alerts else ["SHADOW_DATA_DRIFT_ALERT:none"]
    result["action"] = "none; no retrain/recalibrate"
    return result


def compute_operational_health(
    score_frame: pd.DataFrame,
    *,
    raw_scores: Sequence[float],
    calibrated_scores: Sequence[float],
    ranks: Sequence[int],
    universe: Mapping[str, Any],
    feature_summary: Mapping[str, Any],
    artifact_verification: Mapping[str, bool],
    identity_failures: Sequence[Mapping[str, Any]] = (),
    provider_failures: Sequence[Mapping[str, Any]] = (),
    reference_distributions: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Compute the per-snapshot operational health record.

    Returns a mapping whose keys cover the frozen
    ``operational_monitoring`` fields. ``artifact_verification`` is a mapping of
    label -> boolean verification result. Any ``False`` value is a blocking
    operational failure.
    """
    universe_size = int(universe.get("universe_count", 0))
    scoreable = int(len(score_frame))
    coverage = (scoreable / universe_size) if universe_size else 0.0

    missingness = {}
    for name in FROZEN_FEATURES:
        if name in score_frame.columns:
            series = pd.to_numeric(score_frame[name], errors="coerce")
            missingness[name] = {
                "present": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "coverage": float(series.notna().mean()) if len(series) else 0.0,
            }
        else:
            missingness[name] = {"present": 0, "missing": scoreable, "coverage": 0.0}

    score_distribution = {
        "raw_model_score": _distribution(raw_scores),
        "frozen_calibrated_score": _distribution(calibrated_scores),
    }
    drift = {}
    if reference_distributions:
        drift["raw_model_score"] = compare_to_reference(
            raw_scores, reference_distributions.get("raw_model_score", []), label="raw_model_score"
        )
        drift["frozen_calibrated_score"] = compare_to_reference(
            calibrated_scores,
            reference_distributions.get("frozen_calibrated_score", []),
            label="frozen_calibrated_score",
        )

    health = {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "snapshot_asof": universe.get("snapshot_asof"),
        "universe_size": universe_size,
        "scoreable_count": scoreable,
        "excluded_count": max(0, universe_size - scoreable),
        "coverage_rate": coverage,
        "feature_missingness": missingness,
        "identity_failures": [dict(failure) for failure in identity_failures],
        "provider_failures": [dict(failure) for failure in provider_failures],
        "artifact_hash_verification": dict(artifact_verification),
        "score_distribution": score_distribution,
        "rank_concentration": _rank_concentration(ranks),
        "drift": drift,
        "drift_mode": "descriptive_non_adaptive",
        "drift_auto_action": "none; no retrain/recalibrate",
        "data_drift_alerts": [],
    }
    if drift:
        alerts = []
        for item in drift.values():
            for alert in item.get("alerts", []):
                if alert.startswith("SHADOW_DATA_DRIFT_ALERT"):
                    alerts.append(alert)
        health["data_drift_alerts"] = sorted(set(alerts))
    return health


def operational_blockers(health: Mapping[str, Any]) -> Sequence[str]:
    """Return exact blocking reasons for an operational health record.

    Blocking reasons invalidate a snapshot as non-evidentiary; they never
    mutate the model. Currently: any failed artifact hash verification, a
    zero-scoreable snapshot, and a negative coverage (impossible) are blockers.
    """
    blockers = []
    if not health.get("artifact_hash_verification"):
        blockers.append("OPERATIONAL_BLOCK:artifact_hash_verification_failed")
    for label, ok in (health.get("artifact_hash_verification") or {}).items():
        if not bool(ok):
            blockers.append("OPERATIONAL_BLOCK:artifact_hash_mismatch:%s" % label)
    if int(health.get("scoreable_count", 0)) <= 0:
        blockers.append("OPERATIONAL_BLOCK:no_scoreable_securities")
    if int(health.get("universe_size", 0)) <= 0:
        blockers.append("OPERATIONAL_BLOCK:empty_universe")
    return sorted(set(blockers))


def health_digest(health: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the operational health record."""
    return fingerprint_obj(dict(health))


__all__ = [
    "CONTRACT_MONITORING_FIELDS",
    "HEALTH_SCHEMA_VERSION",
    "compare_to_reference",
    "compute_operational_health",
    "health_digest",
    "operational_blockers",
]
