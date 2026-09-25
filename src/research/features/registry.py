"""WP3 feature registry: a formal, versioned contract for every candidate feature.

A feature is not "whatever a script computed"; it is a declared artifact with an
explicit availability rule, transformation rule, missing-data rule and lineage.
The registry is the single source of truth WP4/WP5/WP6 consume, so a feature
cannot be silently changed or silently borrowed from future information.

Definitions only live here. No feature VALUE is computed in this module, and no
model is fitted. Macro features are declared but DISABLED (enabled=False) until
their vintages are wired; they are never silently switched on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REGISTRY_VERSION = "v2_wp3_feature_registry_v1"

CATEGORIES = (
    "VALUATION", "PROFITABILITY", "QUALITY", "GROWTH", "MOMENTUM", "REVISION",
    "BALANCE_SHEET", "CASH_FLOW", "CAPITAL_EFFICIENCY", "RISK", "LIQUIDITY",
    "SIZE", "MACRO", "MARKET_REGIME", "OTHER",
)

PIT_STATUSES = ("POINT_IN_TIME", "PARTIAL", "LEGACY_ONLY", "PROXY", "UNKNOWN")

# Missing-value semantics. Filling with zero is deliberately absent.
MISSING_RULES = (
    "EXCLUDE_ROW",
    "MODEL_NATIVE",
    "INDICATOR_PLUS_TRAIN_MEDIAN",
    "INDICATOR_PLUS_TRAIN_SECTOR_MEDIAN",
    "NOT_APPLICABLE",
)

# A fitted transform must fit on training data only.
NORMALIZATION_RULES = ("NONE", "TRAIN_ZSCORE", "TRAIN_RANK", "TRAIN_ROBUST_SCALE", "LOG")
WINSORIZATION_RULES = ("NONE", "TRAIN_PERCENTILE_CLIP")

AVAILABILITY_RULES = (
    "SEC_FILING_AVAILABILITY",  # acceptance datetime, else filing_date + 1 day
    "PRICE_CLOSE_AVAILABILITY",  # trade date + conservative close in UTC
    "MACRO_VINTAGE_AVAILABILITY",
    "UNIVERSE_MEMBERSHIP_AVAILABILITY",
)


class FeatureRegistryError(ValueError):
    """Raised when a feature definition is incomplete or inconsistent."""


@dataclass(frozen=True)
class FeatureDefinition:
    """One declared research feature with its integrity contract."""

    feature_id: str
    feature_name: str
    category: str
    definition: str
    source: str
    raw_dependencies: tuple
    transformation: str
    hypothesized_direction: str
    available_at_rule: str
    lookback_window: str
    missing_rule: str
    winsorization_rule: str
    normalization_rule: str
    pit_status: str
    legacy_mapping: str
    enabled: bool = True
    definition_version: str = REGISTRY_VERSION
    notes: str = ""

    def validate(self):
        problems = []
        if not self.feature_id or not str(self.feature_id).strip():
            problems.append("feature_id is required")
        if not self.feature_name or not str(self.feature_name).strip():
            problems.append("feature_name is required")
        if self.category not in CATEGORIES:
            problems.append("category %r not in %s" % (self.category, CATEGORIES))
        if self.pit_status not in PIT_STATUSES:
            problems.append("pit_status %r not in %s" % (self.pit_status, PIT_STATUSES))
        if self.missing_rule not in MISSING_RULES:
            problems.append("missing_rule %r not in %s" % (self.missing_rule, MISSING_RULES))
        if self.normalization_rule not in NORMALIZATION_RULES:
            problems.append("normalization_rule %r not in %s" % (self.normalization_rule, NORMALIZATION_RULES))
        if self.winsorization_rule not in WINSORIZATION_RULES:
            problems.append("winsorization_rule %r not in %s" % (self.winsorization_rule, WINSORIZATION_RULES))
        if self.available_at_rule not in AVAILABILITY_RULES:
            problems.append("available_at_rule %r not in %s" % (self.available_at_rule, AVAILABILITY_RULES))
        if not self.raw_dependencies:
            problems.append("raw_dependencies must not be empty")
        if self.hypothesized_direction not in ("POSITIVE", "NEGATIVE", "UNKNOWN"):
            problems.append("hypothesized_direction %r invalid" % (self.hypothesized_direction,))
        if problems:
            raise FeatureRegistryError("%s invalid: %s" % (self.feature_id, "; ".join(problems)))
        return True

    def to_dict(self):
        payload = dict(self.__dict__)
        payload["raw_dependencies"] = list(self.raw_dependencies)
        return payload


@dataclass
class FeatureRegistry:
    """An immutable-once-sealed collection of :class:`FeatureDefinition`."""

    features: dict = field(default_factory=dict)
    version: str = REGISTRY_VERSION
    sealed: bool = False

    def add(self, definition):
        if self.sealed:
            raise FeatureRegistryError("registry is sealed; create a new version instead")
        definition.validate()
        if definition.feature_id in self.features:
            raise FeatureRegistryError("duplicate feature_id %r" % definition.feature_id)
        self.features[definition.feature_id] = definition
        return definition

    def seal(self):
        self.sealed = True
        return self

    def enabled_features(self):
        return [definition for definition in self.sorted_features() if definition.enabled]

    def sorted_features(self):
        return [self.features[key] for key in sorted(self.features)]

    def by_category(self):
        counts = {category: 0 for category in CATEGORIES}
        for definition in self.features.values():
            counts[definition.category] += 1
        return {category: count for category, count in counts.items() if count}

    def definition_versions(self):
        return {definition.feature_name: definition.definition_version for definition in self.features.values()}

    def validate(self):
        for definition in self.features.values():
            definition.validate()
        return True

    def fingerprint_payload(self):
        """Canonical payload hashed into the FeatureSetManifest fingerprint."""
        return {
            "version": self.version,
            "features": [definition.to_dict() for definition in self.sorted_features()],
        }
