"""Immutable manifest schemas for V2 research artefacts.

Each manifest is a frozen dataclass with ``to_dict`` / ``from_dict`` and a
``validate`` that raises ``ManifestValidationError`` for missing required fields
or malformed artefact identifiers. Validating does not mutate state.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from .ids import is_valid_id


class ManifestValidationError(ValueError):
    """Raised when a manifest is incomplete or references a malformed ID."""


ALLOWED_PIT_STATUS = ("point_in_time", "partially_point_in_time", "not_point_in_time", "unknown")
ALLOWED_SYNTHETIC_STATUS = ("none", "synthetic", "mixed", "unknown")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value):
    """True when ``value`` is a lower-case hex SHA-256 digest."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def _field_names(cls):
    return {item.name for item in dataclasses.fields(cls)}


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


class _Manifest:
    """Shared behaviour for the V2 manifest dataclasses."""

    REQUIRED = ()
    ID_FIELDS = {}

    def to_dict(self):
        """Plain JSON-serialisable mapping of every declared field."""
        return {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a manifest, rejecting unknown keys."""
        if not isinstance(data, dict):
            raise ManifestValidationError("manifest payload must be a mapping, got %s" % type(data).__name__)
        known = _field_names(cls)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ManifestValidationError("unknown %s field(s): %s" % (cls.__name__, ", ".join(unknown)))
        return cls(**data)

    def validate(self):
        """Raise :class:`ManifestValidationError` unless the manifest is complete."""
        problems = []
        for name in self.REQUIRED:
            if _missing(getattr(self, name)):
                problems.append("missing required field %r" % name)
        for name, kind in self.ID_FIELDS.items():
            value = getattr(self, name)
            if not _missing(value) and not is_valid_id(value, kind):
                problems.append("field %r is not a valid %s identifier: %r" % (name, kind, value))
        problems.extend(self._extra_problems())
        if problems:
            raise ManifestValidationError("%s invalid: %s" % (type(self).__name__, "; ".join(problems)))
        return True

    def _extra_problems(self):
        return []


@dataclass(frozen=True)
class DatasetManifest(_Manifest):
    """Provenance record for one immutable dataset version."""

    dataset_id: str
    created_at: str
    git_commit: str
    branch: str
    sources: list
    universe_definition: str
    period_start: str
    period_end: str
    row_count: int
    schema_version: str
    config_fingerprint: str
    source_fingerprints: dict
    dataset_fingerprint: str
    pit_status: str
    synthetic_data_status: str
    known_limitations: list
    notes: str = ""
    # WP2C Phase E: explicit source/security-master/universe versions and the
    # censoring statistics of the label space. All optional (defaulted) so older
    # manifests remain valid; populated by the WP2B/WP2C build finalizer.
    source_versions: dict = None
    security_master_version: str = None
    universe_version: str = None
    censoring_statistics: dict = None

    REQUIRED = (
        "dataset_id", "created_at", "git_commit", "branch", "sources", "universe_definition",
        "period_start", "period_end", "row_count", "schema_version", "config_fingerprint",
        "source_fingerprints", "dataset_fingerprint", "pit_status", "synthetic_data_status",
        "known_limitations",
    )
    ID_FIELDS = {"dataset_id": "dataset"}

    def _extra_problems(self):
        problems = []
        if self.pit_status not in ALLOWED_PIT_STATUS:
            problems.append("pit_status must be one of %s" % ", ".join(ALLOWED_PIT_STATUS))
        if self.synthetic_data_status not in ALLOWED_SYNTHETIC_STATUS:
            problems.append("synthetic_data_status must be one of %s" % ", ".join(ALLOWED_SYNTHETIC_STATUS))
        if not self.sources:
            problems.append("sources must list at least one provider")
        if not self.source_fingerprints:
            problems.append("source_fingerprints must record at least one source")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 0:
            problems.append("row_count must be a non-negative integer")
        for name in ("config_fingerprint", "dataset_fingerprint"):
            if not _missing(getattr(self, name)) and not is_sha256(getattr(self, name)):
                problems.append("field %r must be a lower-case hex SHA-256 digest" % name)
        if isinstance(self.source_fingerprints, dict):
            for source, digest in sorted(self.source_fingerprints.items()):
                if not is_sha256(digest):
                    problems.append("source_fingerprints[%r] must be a lower-case hex SHA-256 digest" % source)
        for name in ("source_versions", "censoring_statistics"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                problems.append("field %r must be a mapping when supplied" % name)
        return problems


@dataclass(frozen=True)
class FeatureSetManifest(_Manifest):
    """Provenance record for one immutable feature-set version."""

    feature_set_id: str
    dataset_id: str
    features: list
    feature_definition_versions: dict
    transformations: list
    availability_policy_ref: str
    missingness_policy: str
    fingerprint: str
    git_commit: str
    notes: str = ""

    REQUIRED = (
        "feature_set_id", "dataset_id", "features", "feature_definition_versions",
        "transformations", "availability_policy_ref", "missingness_policy", "fingerprint", "git_commit",
    )
    ID_FIELDS = {"feature_set_id": "feature_set", "dataset_id": "dataset"}

    def _extra_problems(self):
        problems = []
        if not self.features:
            problems.append("features must not be empty")
        if not isinstance(self.feature_definition_versions, dict):
            problems.append("feature_definition_versions must map feature name -> version")
        else:
            for feature_name in self.features:
                if feature_name not in self.feature_definition_versions:
                    problems.append("feature %r has no recorded definition version" % feature_name)
        if not _missing(self.fingerprint) and not is_sha256(self.fingerprint):
            problems.append("field 'fingerprint' must be a lower-case hex SHA-256 digest")
        return problems


@dataclass(frozen=True)
class TargetManifest(_Manifest):
    """Provenance record for one immutable target-set version."""

    target_set_id: str
    target_definitions: dict
    horizon: str
    benchmark: str
    observability_semantics: object
    construction_version: str
    fingerprint: str
    dataset_id: str = None
    notes: str = ""

    REQUIRED = (
        "target_set_id", "target_definitions", "horizon", "benchmark",
        "observability_semantics", "construction_version", "fingerprint",
    )
    ID_FIELDS = {"target_set_id": "target_set", "dataset_id": "dataset"}

    def _extra_problems(self):
        problems = []
        if not self.target_definitions:
            problems.append("target_definitions must not be empty")
        semantics = self.observability_semantics
        if not isinstance(semantics, (dict, str)):
            problems.append("observability_semantics must be a mapping or string")
        elif isinstance(semantics, str) and not semantics.strip():
            problems.append("observability_semantics must not be empty")
        elif isinstance(semantics, dict) and not semantics:
            problems.append("observability_semantics must not be empty")
        if not _missing(self.fingerprint) and not is_sha256(self.fingerprint):
            problems.append("field 'fingerprint' must be a lower-case hex SHA-256 digest")
        return problems


MANIFEST_TYPES = {
    "dataset": DatasetManifest,
    "feature_set": FeatureSetManifest,
    "target_set": TargetManifest,
}
