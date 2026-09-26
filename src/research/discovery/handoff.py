"""WP5.18 machine-readable handoff to WP6.

The handoff is a CANDIDATE universe, not a final feature set. It carries, for every
candidate, its definition, family/category, allowed transformations, missingness
policy, redundancy cluster, stability metadata, hypothesized direction, Keyes
benchmark reference and limitations - plus the train-window-selection
requirements and the locked-holdout policy that bound how WP6 may use it.
"""

from __future__ import annotations

from .catalog import FEATURE_CATALOG, HANDOFF_VERSION
from .scorecard import classification_counts


class HandoffError(ValueError):
    """Raised when a handoff request is ill-formed."""


TRAIN_WINDOW_REQUIREMENTS = (
    "select features INSIDE each training window only; never on the full sample",
    "fit any transform (z-score/percentile clip) on training rows and apply to validation rows",
    "respect the 12-month label-overlap purge band around every validation fold",
    "use dependence-aware inference (HAC/block bootstrap) on the monthly IC series, not pooled row p-values",
    "apply Benjamini-Hochberg FDR within each selection scan",
    "do not elect a single representative per redundancy cluster on the full sample",
)

LOCKED_HOLDOUT_POLICY = (
    "WP5 discovery consumed development rows only (pre-embargo, observable labels)",
    "the locked holdout (2022-01-01..2025-08-31) was never read during discovery",
    "WP6 must keep the locked holdout evaluation-only and use trainable_mask for training",
)


def candidate_entry(spec, scorecard_row, cluster_name):
    """One machine-readable candidate entry for the WP6 handoff."""
    return {
        "feature": spec.feature_name,
        "registry_feature_id": spec.registry_feature_id,
        "definition": spec.definition,
        "source": spec.source,
        "category": spec.category,
        "cluster_family": spec.cluster_family,
        "hypothesized_direction": spec.hypothesized_direction,
        "pit_status": spec.pit_status,
        "missing_policy": spec.missing_policy,
        "allowed_transformations": list(spec.allowed_transformations),
        "is_keyes": bool(spec.is_keyes),
        "redundancy_cluster": cluster_name,
        "classification": scorecard_row.get("classification"),
        "coverage": scorecard_row.get("coverage"),
        "mean_ic": scorecard_row.get("mean_ic"),
        "median_ic": scorecard_row.get("median_ic"),
        "icir": scorecard_row.get("icir"),
        "positive_month_share": scorecard_row.get("positive_month_share"),
        "months": scorecard_row.get("months"),
        "hac_p_value": scorecard_row.get("hac_p_value"),
        "fdr_q_value": scorecard_row.get("fdr_q_value"),
        "fdr_rejected": scorecard_row.get("fdr_rejected"),
        "direction_consistent": scorecard_row.get("direction_consistent"),
        "short_period_only": scorecard_row.get("short_period_only"),
        "coverage_drift": scorecard_row.get("coverage_drift"),
        "limitations": scorecard_row.get("limitations", []),
    }


def build_handoff(scorecards, clusters, dataset_id, target_id, feature_set_id,
                  experiment, keyes_reference=None, notes=None):
    """Assemble the full WP6 handoff document (candidate universe + policies)."""
    by_name = {row["feature"]: row for row in scorecards}
    members = (clusters or {}).get("member_to_cluster", {})
    candidates = []
    for spec in FEATURE_CATALOG:
        row = by_name.get(spec.feature_name)
        if row is None:
            continue
        candidates.append(candidate_entry(spec, row, members.get(spec.feature_name)))
    return {
        "handoff_version": HANDOFF_VERSION,
        "dataset_id": dataset_id,
        "target_id": target_id,
        "feature_set_id": feature_set_id,
        "discovery_experiment": experiment,
        "candidate_universe": candidates,
        "candidate_count": len(candidates),
        "classification_counts": classification_counts(scorecards),
        "redundancy_clusters": (clusters or {}).get("clusters", {}),
        "keyes_reference": keyes_reference or {},
        "train_window_selection_requirements": list(TRAIN_WINDOW_REQUIREMENTS),
        "locked_holdout_policy": list(LOCKED_HOLDOUT_POLICY),
        "known_limitations": notes or [
            "WP5 is discovery only: no final feature set and no weighted score is produced",
            "classifications are research categories, not a ranking",
            "EDGAR companyfacts begin around 2009, so early fundamental coverage is bounded",
            "abnormal_volume is UNAVAILABLE because the certified PIT price table has no volume",
        ],
    }
