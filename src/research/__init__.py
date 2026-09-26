"""V2 research infrastructure: identifiers, fingerprints, manifests, registry, lineage.

This package is infrastructure only. It performs no data selection, no feature
engineering, no model fitting and no scoring; it records and protects research
provenance so that those steps stay reproducible and auditable.
"""

from .fingerprints import fingerprint_dataframe, fingerprint_file, fingerprint_obj
from .holdout import (
    LockedHoldout,
    HoldoutError,
    HoldoutLeakageError,
    assert_trainable,
    build_holdout,
    embargo_cutoff,
    embargo_mask,
    holdout_mask,
    holdout_payload,
    locked_holdout,
)
from .ids import (
    KINDS,
    audit_id,
    canonical_json,
    dataset_id,
    experiment_id,
    feature_set_id,
    holdout_id,
    is_valid_id,
    make_id,
    model_id,
    parse_id,
    prediction_id,
    promotion_id,
    target_set_id,
    validation_id,
)
from .immutability import ImmutabilityError, load_immutable, save_immutable, write_json_atomic
from .lineage import (
    KIND_ORDER,
    LineageError,
    LineageGraph,
    LineageNode,
    build_lineage,
    validate_lineage,
)
from .manifests import (
    MANIFEST_TYPES,
    DatasetManifest,
    FeatureSetManifest,
    ManifestValidationError,
    TargetManifest,
)
from .modes import (
    ResearchMode,
    SyntheticDataError,
    assert_no_synthetic_in_research,
    current_branch,
    current_git_commit,
)
from .registry import (
    ALLOWED_STATUSES,
    ExperimentRegistry,
    RegistryError,
    UnknownExperimentError,
)

__all__ = [
    "KINDS",
    "ALLOWED_STATUSES",
    "KIND_ORDER",
    "MANIFEST_TYPES",
    "DatasetManifest",
    "ExperimentRegistry",
    "FeatureSetManifest",
    "HoldoutError",
    "HoldoutLeakageError",
    "ImmutabilityError",
    "LineageError",
    "LockedHoldout",
    "LineageGraph",
    "LineageNode",
    "ManifestValidationError",
    "RegistryError",
    "ResearchMode",
    "SyntheticDataError",
    "TargetManifest",
    "UnknownExperimentError",
    "assert_no_synthetic_in_research",
    "assert_trainable",
    "audit_id",
    "build_holdout",
    "build_lineage",
    "canonical_json",
    "current_branch",
    "current_git_commit",
    "dataset_id",
    "embargo_cutoff",
    "embargo_mask",
    "experiment_id",
    "feature_set_id",
    "fingerprint_dataframe",
    "fingerprint_file",
    "fingerprint_obj",
    "holdout_id",
    "holdout_mask",
    "holdout_payload",
    "is_valid_id",
    "load_immutable",
    "locked_holdout",
    "make_id",
    "model_id",
    "parse_id",
    "prediction_id",
    "promotion_id",
    "save_immutable",
    "target_set_id",
    "validate_lineage",
    "validation_id",
    "write_json_atomic",
]
