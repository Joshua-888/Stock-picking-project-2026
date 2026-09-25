"""WP3 feature subpackage: registry, initial set, transforms, missingness, return metrics.

This package defines WHAT may be a research feature and UNDER WHAT integrity
rules. It computes no model and no score. Macro features are declared but
disabled until their vintages exist.
"""

from .registry import (
    CATEGORIES,
    PIT_STATUSES,
    REGISTRY_VERSION,
    FeatureDefinition,
    FeatureRegistry,
    FeatureRegistryError,
)
from .initial_set import DISPOSITIONS, build_initial_registry, v1_disposition
from .return_metrics import (
    LEGACY_ROA_ROIC_NOTE,
    V2_ROA_DEFINITION,
    V2_ROIC_DEFINITION,
    compute_v2_return_metrics,
    compute_v2_roa,
)
from .keyes import (
    KEYES_VARIABLES,
    TRACK_HISTORICAL,
    TRACK_MODERN,
    KeyesCriterion,
    KeyesQualification,
    KeyesRuleSet,
    proxy_variable,
    qualify,
)
from .missingness import Imputer, apply_imputer, fit_imputer
from .transforms import FittedTransform, apply_transform, fit_transform
from .featureset import build_feature_set

__all__ = [
    "CATEGORIES",
    "PIT_STATUSES",
    "REGISTRY_VERSION",
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureRegistryError",
    "DISPOSITIONS",
    "build_initial_registry",
    "v1_disposition",
    "LEGACY_ROA_ROIC_NOTE",
    "V2_ROA_DEFINITION",
    "V2_ROIC_DEFINITION",
    "compute_v2_return_metrics",
    "compute_v2_roa",
    "KEYES_VARIABLES",
    "TRACK_HISTORICAL",
    "TRACK_MODERN",
    "KeyesCriterion",
    "KeyesQualification",
    "KeyesRuleSet",
    "proxy_variable",
    "qualify",
    "Imputer",
    "apply_imputer",
    "fit_imputer",
    "FittedTransform",
    "apply_transform",
    "fit_transform",
    "build_feature_set",
]
