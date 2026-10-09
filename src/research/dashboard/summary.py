"""Read-only research summary and model definition from certified provenance.

Every numeric value carries exact ``{path, field}`` lineage. This module never
reads target, holdout-label, future-return, or target-observable data and never
recomputes holdout metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from . import schemas
from .identity import attach_identity

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

# All paths are relative to the repository root and are read-only.
WP5_CERT = "provenance/certifications/wp5_experiment_f985287c1315.json"
WP6_CERT = "provenance/certifications/wp6_experiment_ee434a07a25d.json"
WP7_CERT = "provenance/certifications/wp7_generation_de0f9bbd0bec.json"
WP7_METRICS = "artifacts/research/wp7/generation_de0f9bbd0bec/metrics.json"
WP8_FREEZE = "provenance/wp8/final_candidate_freeze.json"
WP8_HOLDOUT = "provenance/wp8/holdout_evaluation_summary.json"
WP9_INDEX = "provenance/wp9/index.json"
WP9_CONTRACT = "provenance/wp9/forward_validation_contract_v1.json"


def _load(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError("certified provenance file missing: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def _get(obj: dict, *keys: str) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("missing provenance field %s in %s" % (" -> ".join(keys), " -> ".join(keys)))
        current = current[key]
    return current


def _lineage(rel: str, field: str, value: Any) -> dict:
    return {"value": value, "source": {"path": rel, "field": field}}


def research_summary(root: Path | None = None) -> dict:
    """Assemble the WP5..WP9 machine-readable research summary."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    wp5 = _load(root, WP5_CERT)
    wp6 = _load(root, WP6_CERT)
    wp7_metrics = _load(root, WP7_METRICS)
    wp8 = _load(root, WP8_FREEZE)
    holdout = _load(root, WP8_HOLDOUT)
    wp9_idx = _load(root, WP9_INDEX)

    fold_skills = [
        float(fold["outer_metrics"]["mean_auc_skill"])
        for fold in _get(wp7_metrics, "outer_folds")
    ]
    mean_auc_skill = float(sum(fold_skills) / len(fold_skills)) if fold_skills else 0.0

    result = {
        **schemas.schema_basis("research_summary"),
        "generated_by": "src.research.dashboard.summary.research_summary",
        "wp5": {
            "candidate_count": _lineage(WP5_CERT, "candidate_count", int(_get(wp5, "candidate_count"))),
            "robust_candidate_count": _lineage(
                WP5_CERT, "robust_candidate_count", int(_get(wp5, "robust_candidate_count"))),
            "fdr": {
                "method": _lineage(WP5_CERT, "fdr.method", str(_get(wp5, "fdr", "method"))),
                "alpha": _lineage(WP5_CERT, "fdr.alpha", float(_get(wp5, "fdr", "alpha"))),
                "hypotheses": _lineage(WP5_CERT, "fdr.hypotheses", int(_get(wp5, "fdr", "hypotheses"))),
                "rejected": _lineage(WP5_CERT, "fdr.rejected", int(_get(wp5, "fdr", "rejected"))),
            },
            "status": _lineage(WP5_CERT, "status", str(_get(wp5, "status"))),
        },
        "wp6": {
            "configuration_count": _lineage(
                WP6_CERT, "configuration_count", int(_get(wp6, "configuration_count"))),
            "status": _lineage(WP6_CERT, "status", str(_get(wp6, "status"))),
            "experiment_id": _lineage(WP6_CERT, "experiment_id", str(_get(wp6, "experiment_id"))),
        },
        "wp7": {
            "outer_fold_mean_auc_skill": [
                _lineage(WP7_METRICS, "outer_folds[%d].outer_metrics.mean_auc_skill" % i, value)
                for i, value in enumerate(fold_skills)
            ],
            "mean_mean_auc_skill": _lineage(
                WP7_METRICS, "outer_folds[*].outer_metrics.mean_auc_skill (mean)", mean_auc_skill),
        },
        "wp8": {
            "holdout_primary_metrics": {
                "mean_auc": _lineage(
                    WP8_HOLDOUT, "primary_metrics.mean_auc",
                    float(_get(holdout, "primary_metrics", "mean_auc"))),
                "mean_auc_skill": _lineage(
                    WP8_HOLDOUT, "primary_metrics.mean_auc_skill",
                    float(_get(holdout, "primary_metrics", "mean_auc_skill"))),
                "one_sided_hac_p": _lineage(
                    WP8_HOLDOUT, "primary_metrics.one_sided_hac_p",
                    float(_get(holdout, "primary_metrics", "one_sided_hac_p"))),
                "t_stat": _lineage(
                    WP8_HOLDOUT, "primary_metrics.t_stat",
                    float(_get(holdout, "primary_metrics", "t_stat"))),
                "hac_lag": _lineage(
                    WP8_HOLDOUT, "primary_metrics.hac_lag",
                    int(_get(holdout, "primary_metrics", "hac_lag"))),
                "months": _lineage(
                    WP8_HOLDOUT, "primary_metrics.months",
                    int(_get(holdout, "primary_metrics", "months"))),
                "positive_month_fraction": _lineage(
                    WP8_HOLDOUT, "primary_metrics.positive_month_fraction",
                    float(_get(holdout, "primary_metrics", "positive_month_fraction"))),
                "ci": [
                    _lineage(
                        WP8_HOLDOUT, "primary_metrics.ci_lower",
                        float(_get(holdout, "primary_metrics", "ci_lower"))),
                    _lineage(
                        WP8_HOLDOUT, "primary_metrics.ci_upper",
                        float(_get(holdout, "primary_metrics", "ci_upper"))),
                ],
                "verdict": _lineage(
                    WP8_HOLDOUT, "primary_metrics.verdict",
                    str(_get(holdout, "primary_metrics", "verdict"))),
            },
            "training": {
                "data_fingerprint": _lineage(
                    WP8_FREEZE, "training_data_fingerprint",
                    str(_get(wp8, "training_data_fingerprint"))),
                "start": _lineage(WP8_FREEZE, "training_period.start", str(_get(wp8, "training_period", "start"))),
                "end": _lineage(WP8_FREEZE, "training_period.end", str(_get(wp8, "training_period", "end"))),
                "rows": _lineage(WP8_FREEZE, "training_row_count", int(_get(wp8, "training_row_count"))),
                "securities": _lineage(
                    WP8_FREEZE, "training_data_fingerprint_payload.securities",
                    int(_get(wp8, "training_data_fingerprint_payload", "securities"))),
            },
        },
        "limitations": [
            "holdout metrics are read-only lineage from the certified WP8 evaluation; never recomputed here",
            "WP7 outer-fold mean AUC skill values are read from the certified metrics file, not recomputed",
            "no target, holdout-label, future-return, or target-observable column is read by this module",
        ],
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("research_summary", result)
    return result


def model_definition(root: Path | None = None) -> dict:
    """Assemble the frozen champion model definition from read-only provenance."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    wp8 = _load(root, WP8_FREEZE)
    wp9_contract = _load(root, WP9_CONTRACT)

    features = list(_get(wp8, "final_selected_features"))
    schemas.validate_feature_order(features)

    hyperparameters = dict(_get(wp8, "hyperparameters"))
    seeds = dict(_get(wp8, "seeds"))
    preprocessing = dict(_get(wp8, "preprocessing"))
    artifact_hashes = _get(wp8, "artifact_hashes")

    result = {
        **schemas.schema_basis("model_definition"),
        "freeze_id": str(_get(wp8, "freeze_id")),
        "model_type": str(_get(wp8, "model")),
        "calibration": str(_get(wp8, "final_calibration")),
        "feature_strategy": str(_get(wp8, "feature_strategy")),
        "config_id": str(_get(wp8, "final_config_id")),
        "hyperparameters": hyperparameters,
        "seeds": seeds,
        "training_period": {
            "start": str(_get(wp8, "training_period", "start")),
            "end": str(_get(wp8, "training_period", "end")),
        },
        "features": features,
        "preprocessing_spec": {
            "frozen_spec": preprocessing,
            "documented_spec": str(_get(wp9_contract, "champion", "preprocessing_spec")),
        },
        "transformed_column_order": list(schemas.TRANSFORMED_COLUMN_ORDER),
        "feature_families": {feature: schemas.FEATURE_FAMILY_MAP[feature] for feature in schemas.FEATURE_ORDER},
        "artifact_hashes": {
            name: {
                "sha256": str(spec.get("sha256")),
                "path": str(spec.get("path")),
            }
            for name, spec in artifact_hashes.items()
            if name in ("model", "preprocessor", "calibrator")
        },
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("model_definition", result)
    return result


__all__ = ["model_definition", "research_summary", "_load", "_get", "_lineage"]
