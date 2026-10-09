"""WP9A.2 apply-only dashboard explainability and truth layer.

This package is intentionally research-internals-free for UI consumers: it never
imports or reads target, holdout-label, future-return, or target-observable
columns, and it never fits the frozen model/preprocessor/calibrator.
"""

from . import attribution, contract, explain, identity, integrity, schemas, summary
from .identity import artifact_identity, attach_identity
from .attribution import (
    ATTRIBUTION_KIND,
    IMPORTANCE_KIND,
    ATTRIBUTION_TOLERANCE,
    ATTRIBUTION_LIMITATIONS,
    AttributionError,
    GlobalAttribution,
    LocalAttribution,
    attribute_local_rows,
    contributor_shape,
    expected_value,
    family_attribution,
    feature_importance,
    global_attribution_by_feature,
    tree_path_local,
)
from .explain import explain_security
from .contract import (
    freshness_completeness,
    latest_shadow_ranking,
    official_snapshot_history,
)
from .integrity import audit_status
from .schemas import (
    CONTRACT_KINDS,
    FAMILY_ORDER,
    FEATURE_FAMILY_MAP,
    FEATURE_ORDER,
    FORBIDDEN_PHRASES,
    SCHEMA_VERSION,
    TRANSFORMED_COLUMN_ORDER,
    SchemaError,
    lineage,
    validate_dashboard_artifact,
)
from .summary import model_definition, research_summary

__all__ = [
    "ATTRIBUTION_KIND",
    "ATTRIBUTION_LIMITATIONS",
    "ATTRIBUTION_TOLERANCE",
    "AttributionError",
    "CONTRACT_KINDS",
    "FAMILY_ORDER",
    "FEATURE_FAMILY_MAP",
    "FEATURE_ORDER",
    "FORBIDDEN_PHRASES",
    "GlobalAttribution",
    "IMPORTANCE_KIND",
    "LocalAttribution",
    "SCHEMA_VERSION",
    "TRANSFORMED_COLUMN_ORDER",
    "SchemaError",
    "artifact_identity",
    "attach_identity",
    "attribute_local_rows",
    "attribution",
    "audit_status",
    "contract",
    "contributor_shape",
    "expected_value",
    "explain_security",
    "family_attribution",
    "feature_importance",
    "freshness_completeness",
    "global_attribution_by_feature",
    "integrity",
    "latest_shadow_ranking",
    "lineage",
    "model_definition",
    "official_snapshot_history",
    "research_summary",
    "schemas",
    "summary",
    "tree_path_local",
    "validate_dashboard_artifact",
]
