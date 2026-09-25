"""WP3 target construction subpackage.

Public surface: the versioned target contract, deterministic date matching, the
benchmark-relative label builder and the training-time observability metadata.
This package defines WHAT the 12-month forward labels are and WHEN they become
usable; it never defines features and never fits a model.
"""

from .contract import (
    CLASSIFICATION_TARGET,
    CONTINUOUS_TARGET,
    TARGET_VERSION,
    TargetContract,
)
from .matching import (
    intended_horizon_date,
    intended_horizon_key,
    match_observation,
    match_target_window,
    match_target_window_keys,
)
from .build import TARGET_COLUMNS, TargetBuildError, build_targets
from .observability import (
    OBSERVABILITY_COLUMNS,
    ObservabilityError,
    annotate_observability,
    target_known_at,
    trainable_mask,
)

__all__ = [
    "CLASSIFICATION_TARGET",
    "CONTINUOUS_TARGET",
    "TARGET_VERSION",
    "TargetContract",
    "intended_horizon_date",
    "match_observation",
    "match_target_window",
    "TARGET_COLUMNS",
    "TargetBuildError",
    "build_targets",
    "OBSERVABILITY_COLUMNS",
    "ObservabilityError",
    "annotate_observability",
    "target_known_at",
    "trainable_mask",
]
