"""WP5 Variable Discovery.

WP5 materialises a point-in-time candidate feature panel and characterises each
candidate feature with dependence-aware cross-sectional diagnostics. It emits a
``CANDIDATE_FEATURE_UNIVERSE`` of research categories; it never produces a final,
full-sample-selected feature set or a single weighted score. Target-driven
selection happens later, inside training windows (WP6/WP7).

Nothing in this package touches the locked holdout: the panel builder restricts
discovery to development rows (strictly before the frozen embargo cutoff).
"""

from .catalog import (
    CLASSIFICATIONS,
    CLUSTER_FAMILIES,
    DISCOVERY_VERSION,
    FEATURE_CATALOG,
    FEATURE_NAMES,
    FEATURE_PANEL_VERSION,
    CatalogError,
    DiscoveryConfig,
    FeatureSpec,
    catalog_payload,
)

__all__ = [
    "CLASSIFICATIONS",
    "CLUSTER_FAMILIES",
    "DISCOVERY_VERSION",
    "FEATURE_CATALOG",
    "FEATURE_NAMES",
    "FEATURE_PANEL_VERSION",
    "CatalogError",
    "DiscoveryConfig",
    "FeatureSpec",
    "catalog_payload",
]
