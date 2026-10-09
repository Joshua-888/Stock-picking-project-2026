"""Versioned dashboard data contracts for the WP9 apply-only truth layer.

This module owns only constants, schema declarations and schema validation for
the nine dashboard contracts. It imports no target, holdout-label, price, or
fundamental data and performs no model or preprocessor fitting.

Contract kinds
--------------
research_summary, model_definition, global_feature_importance,
feature_family_attribution, stock_explanations, latest_shadow_ranking,
freshness_completeness, official_snapshot_history, integrity_audit_status

Every emitted artefact carries ``schema_version``. Validation is intentionally
strict: unexpected fields are rejected so the dashboard can never silently
start promising more than this layer verifies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

SCHEMA_VERSION = "wp9_dashboard_contracts_v1"

# The 13 frozen champion predictors in the exact frozen model order. This is the
# single source of truth for dashboard ordering; it is enforced against
# src.research.wp9.champion.FROZEN_FEATURES by callers, never duplicated there.
FEATURE_ORDER: tuple[str, ...] = (
    "earnings_yield",
    "current_pe",
    "price_to_book",
    "free_cash_flow_yield",
    "dividend_yield",
    "roa",
    "roe",
    "eps_growth_acceleration",
    "six_month_momentum",
    "twelve_month_momentum",
    "five_year_price_gain",
    "debt_to_equity",
    "market_cap",
)

# The preprocessor emits each base followed immediately by its ``__missing``
# indicator. This order is frozen by the champion preprocessor and is asserted
# at runtime before any attribution input is accepted.
TRANSFORMED_COLUMN_ORDER: tuple[str, ...] = tuple(
    column
    for feature in FEATURE_ORDER
    for column in (feature, "%s__missing" % feature)
)

# Economic-family mapping. This is a presentation grouping, not a feature
# selection or scientific significance claim.
FEATURE_FAMILY_MAP: dict[str, str] = {
    "earnings_yield": "VALUATION",
    "current_pe": "VALUATION",
    "price_to_book": "VALUATION",
    "free_cash_flow_yield": "VALUATION",
    "dividend_yield": "VALUATION",
    "roa": "PROFITABILITY",
    "roe": "PROFITABILITY",
    "eps_growth_acceleration": "GROWTH",
    "six_month_momentum": "MOMENTUM",
    "twelve_month_momentum": "MOMENTUM",
    "five_year_price_gain": "MOMENTUM",
    "debt_to_equity": "BALANCE_SHEET",
    "market_cap": "SIZE",
}

FAMILY_ORDER: tuple[str, ...] = (
    "VALUATION",
    "PROFITABILITY",
    "GROWTH",
    "MOMENTUM",
    "BALANCE_SHEET",
    "SIZE",
)

CONTRACT_KINDS: tuple[str, ...] = (
    "research_summary",
    "model_definition",
    "global_feature_importance",
    "feature_family_attribution",
    "stock_explanations",
    "latest_shadow_ranking",
    "freshness_completeness",
    "official_snapshot_history",
    "integrity_audit_status",
)

EXPECTED_FREEZE_ID = "freeze_e62eac30df40"

# Every dashboard contract carries the same frozen-identity block so the emitted
# numbers can be bound to the exact champion, feature order, contract bytes, and
# producing commit without research-internals access.
IDENTITY_KEYS: tuple[str, ...] = (
    "freeze_id",
    "model_hash",
    "preprocessor_hash",
    "calibrator_hash",
    "feature_order",
    "contract_digest",
    "producing_commit",
    "dataset_id",
    "feature_set_id",
    "target_set_id",
)

_IDENTITY_SPEC: dict[str, Any] = {
    "freeze_id": str,
    "model_hash": str,
    "preprocessor_hash": str,
    "calibrator_hash": str,
    "feature_order": list,
    "contract_digest": str,
    "producing_commit": str,
    "dataset_id": str,
    "feature_set_id": str,
    "target_set_id": str,
}

# Explicit language constraints. The dashboard truth layer must never state an
# unsupported probability, causal, significance, or SHAP identity claim.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "probability of outperformance",
    "probable outperformer",
    "will outperform",
    "predicts outperformance",
    "guaranteed",
    "guarantee",
    "causal effect",
    "statistically significant",
    "significant predictor",
    "tree shap",
    "shap value",
)


def schema_basis(kind: str) -> dict[str, Any]:
    """Return common fields shared by every dashboard contract."""
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": kind,
    }


# Required paths for each contract. ``field`` values are the exact JSON paths
# documented in the emitted records; they are checked by validation but are not
# used to read any source file here.
_SCHEMAS: dict[str, dict[str, Any]] = {
    "research_summary": {
        "schema_version": str,
        "contract_kind": str,
        "generated_by": str,
        "wp5": dict,
        "wp6": dict,
        "wp7": dict,
        "wp8": dict,
        "limitations": list,
    },
    "model_definition": {
        "schema_version": str,
        "contract_kind": str,
        "freeze_id": str,
        "model_type": str,
        "calibration": str,
        "feature_strategy": str,
        "config_id": str,
        "hyperparameters": dict,
        "seeds": dict,
        "training_period": dict,
        "features": list,
        "preprocessing_spec": dict,
        "transformed_column_order": list,
        "feature_families": dict,
        "artifact_hashes": dict,
    },
    "global_feature_importance": {
        "schema_version": str,
        "contract_kind": str,
        "freeze_id": str,
        "importance_kind": str,
        "basis": str,
        "features": list,
        "shares": dict,
        "limitations": list,
    },
    "feature_family_attribution": {
        "schema_version": str,
        "contract_kind": str,
        "freeze_id": str,
        "attribution_kind": str,
        "attribution_basis": str,
        "sample": dict,
        "by_feature": dict,
        "by_family": dict,
        "limitations": list,
    },
    "stock_explanations": {
        "schema_version": str,
        "contract_kind": str,
        "ticker": str,
        "security_id": str,
        "as_of_date": str,
        "raw_model_score": float,
        "frozen_calibrated_score": float,
        "rank": object,
        "percentile": object,
        "feature_values": dict,
        "missing_indicators": dict,
        "attribution": dict,
        "top_positive_contributors": list,
        "top_negative_contributors": list,
        "attribution_residual": float,
        "attribution_tolerance": float,
        "limitations": list,
    },
    "latest_shadow_ranking": {
        "schema_version": str,
        "contract_kind": str,
        "registry_path": str,
        "official_snapshot_ids": list,
        "dry_run_rankings": list,
        "limitations": list,
    },
    "freshness_completeness": {
        "schema_version": str,
        "contract_kind": str,
        "current_prospective_stage": str,
        "official_snapshot_count": int,
        "dry_run_count": int,
        "matured_evaluation_count": int,
        "invalidated_snapshot_count": int,
        "latest_dry_run": object,
        "limitations": list,
    },
    "official_snapshot_history": {
        "schema_version": str,
        "contract_kind": str,
        "registry_path": str,
        "official_snapshot_ids": list,
        "history": list,
    },
    "integrity_audit_status": {
        "schema_version": str,
        "contract_kind": str,
        "frozen_artifacts_verified": bool,
        "no_refit_guard": bool,
        "metric_lineage_complete": bool,
        "no_wp8_holdout_reevaluation": bool,
        "read_only_registry_access": bool,
        "no_unsupported_probability_claims": bool,
        "apply_only_check": bool,
        "limitations": list,
    },
}


for _dashboard_schema in _SCHEMAS.values():
    _dashboard_schema.setdefault("artifact_identity", _IDENTITY_SPEC)


class SchemaError(ValueError):
    """Raised when a dashboard object does not satisfy its contract."""


def validate_dashboard_artifact(kind: str, obj: Mapping[str, Any]) -> str:
    """Validate ``obj`` against ``kind`` and return the normalized kind.

    Unknown kinds, missing required fields, and unexpected top-level fields are
    all rejected. Container value shapes are checked recursively where declared.
    """
    if kind not in _SCHEMAS:
        raise SchemaError("unknown dashboard contract kind %r" % kind)
    if not isinstance(obj, Mapping):
        raise SchemaError("dashboard artefact for %r must be a mapping" % kind)

    contract = _SCHEMAS[kind]
    required = set(contract)
    present = set(obj)
    missing = sorted(required - present)
    if missing:
        raise SchemaError("dashboard artefact %r is missing field(s): %s" % (kind, ", ".join(missing)))
    unexpected = sorted(present - required)
    if unexpected:
        raise SchemaError(
            "dashboard artefact %r has unexpected field(s): %s" % (kind, ", ".join(unexpected))
        )

    if obj.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            "dashboard artefact %r has wrong schema_version %r; expected %r"
            % (kind, obj.get("schema_version"), SCHEMA_VERSION)
        )
    if obj.get("contract_kind") != kind:
        raise SchemaError(
            "dashboard artefact %r has wrong contract_kind %r" % (kind, obj.get("contract_kind"))
        )

    _validate_container(kind, "", obj, contract)
    validate_artifact_identity(obj.get("artifact_identity"))
    return kind


def validate_artifact_identity(obj: Mapping[str, Any]) -> dict:
    """Validate the common frozen artifact-identity block for every contract.

    Returns the identity dict unchanged when valid.
    """
    if not isinstance(obj, Mapping):
        raise SchemaError("artifact_identity must be a mapping")
    missing = [name for name in IDENTITY_KEYS if name not in obj]
    if missing:
        raise SchemaError("artifact_identity is missing field(s): %s" % ", ".join(missing))
    unexpected = [name for name in obj if name not in IDENTITY_KEYS]
    if unexpected:
        raise SchemaError("artifact_identity has unexpected field(s): %s" % ", ".join(unexpected))
    _validate_container("artifact_identity", "", dict(obj), _IDENTITY_SPEC)
    if obj.get("freeze_id") != EXPECTED_FREEZE_ID:
        raise SchemaError("artifact_identity has wrong freeze_id %r" % obj.get("freeze_id"))
    if tuple(obj.get("feature_order")) != FEATURE_ORDER:
        raise SchemaError("artifact_identity feature_order does not match frozen order")
    for key in ("model_hash", "preprocessor_hash", "calibrator_hash", "contract_digest"):
        value = str(obj.get(key))
        if len(value) != 64 or not all(ch in "0123456789abcdef" for ch in value.lower()):
            raise SchemaError("artifact_identity %s is not a 64-character sha256 hex digest" % key)
    return dict(obj)


def _validate_container(kind: str, path: str, value: Any, spec: Any) -> None:
    if isinstance(spec, type):
        if not isinstance(value, spec):
            raise SchemaError(
                "dashboard artefact %r field %s must be %s; got %s"
                % (kind, path or "<root>", spec.__name__, type(value).__name__)
            )
        return
    if isinstance(spec, dict):
        if not isinstance(value, Mapping):
            raise SchemaError(
                "dashboard artefact %r field %s must be a mapping; got %s"
                % (kind, path or "<root>", type(value).__name__)
            )
        missing = [name for name in spec if name not in value]
        if missing:
            raise SchemaError(
                "dashboard artefact %r field %s is missing subfield(s): %s"
                % (kind, path or "<root>", ", ".join(missing))
            )
        unexpected = [name for name in value if name not in spec]
        if unexpected:
            raise SchemaError(
                "dashboard artefact %r field %s has unexpected subfield(s): %s"
                % (kind, path or "<root>", ", ".join(unexpected))
            )
        for name, child_spec in spec.items():
            _validate_container(kind, "%s.%s" % (path, name) if path else name, value[name], child_spec)
        return
    if isinstance(spec, list):
        if not isinstance(value, list):
            raise SchemaError(
                "dashboard artefact %r field %s must be a list; got %s"
                % (kind, path or "<root>", type(value).__name__)
            )
        for position, item in enumerate(value):
            _validate_container(kind, "%s[%d]" % (path or "<root>", position), item, spec[0])
        return
    raise SchemaError("invalid schema spec for dashboard artefact %r field %s" % (kind, path or "<root>"))


def lineage(path: str, field: str, value: Any) -> dict[str, Any]:
    """Attach exact metric lineage to one number or record."""
    return {
        "value": value,
        "source": {"path": path, "field": field},
    }


def validate_feature_order(features: Sequence[str]) -> None:
    """Refuse any feature ordering other than the frozen 13-feature order."""
    if tuple(features) != FEATURE_ORDER:
        raise SchemaError("feature order does not match the frozen champion order")


__all__ = [
    "CONTRACT_KINDS",
    "FAMILY_ORDER",
    "FEATURE_FAMILY_MAP",
    "FEATURE_ORDER",
    "FORBIDDEN_PHRASES",
    "SCHEMA_VERSION",
    "TRANSFORMED_COLUMN_ORDER",
    "SchemaError",
    "lineage",
    "schema_basis",
    "validate_dashboard_artifact",
    "validate_feature_order",
]
