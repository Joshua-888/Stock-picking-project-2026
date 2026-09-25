"""WP4 Keyes value engine, signals and diagnostics.

This package turns the WP3 Keyes CONTRACT into computed, point-in-time feature
VALUES. It is deliberately separate from ``src.research.features.keyes`` (the
contract) so the contract can never be edited by a value-engine change.

Two tracks, never merged:

* :mod:`src.research.keyes.signals` builds the ``HISTORICAL_KEYES_REPLICATION``
  rule set (explicit simultaneous thresholds, no quota) and the
  ``MODERN_KEYES_INSPIRED`` equal-weight rank composite.
* :mod:`src.research.keyes.variables` computes X5, X6, X8, X9 and the disclosed
  ``X12_PROXY`` under strict point-in-time availability.

:mod:`src.research.keyes.diagnostics` measures what the signals did against the
observable benchmark-relative target only, and runs the placebo/leakage checks.
"""

from __future__ import annotations

from .diagnostics import (
    DIAGNOSTIC_VERSION,
    DiagnosticError,
    component_diagnostics,
    composite_diagnostics,
    concentration_and_turnover,
    fidelity_table,
    observable_slice,
    placebo_checks,
    qualification_summary,
    rank_ic,
    replication_status,
    sector_component_stability,
    yearly_component_stability,
)
from .signals import (
    COMPONENT_SIGNS,
    HISTORICAL_COMPONENTS,
    HISTORICAL_RULE_SETS,
    PRIMARY_THRESHOLDS,
    REPLICATION_FULL,
    REPLICATION_PARTIAL,
    REPLICATION_UNAVAILABLE,
    SIGNAL_VERSION,
    CompositeConfig,
    KeyesSignalError,
    ThresholdSpec,
    build_historical_rule_set,
    direction_payload,
    fit_train_only_weights,
    historical_signals,
    historical_threshold_payload,
    modern_composite,
)
from .variables import (
    DEFINITION_VERSION,
    FIDELITY_CLOSE_EQUIVALENT,
    FIDELITY_EXACT,
    FIDELITY_PROXY,
    FIDELITY_UNAVAILABLE,
    FORBIDDEN_INPUT_COLUMNS,
    KEYES_DIRECTIONS,
    VARIABLE_SPECS,
    X5,
    X6,
    X8,
    X9,
    X12_PROXY,
    ComputationConfig,
    FundamentalHistory,
    KeyesVariableError,
    PriceHistory,
    VariableEngine,
    VariableSpec,
    beta_against_benchmark,
    compute_keyes_variables,
    wide_variables,
)

__all__ = [
    # variables
    "DEFINITION_VERSION",
    "FIDELITY_CLOSE_EQUIVALENT",
    "FIDELITY_EXACT",
    "FIDELITY_PROXY",
    "FIDELITY_UNAVAILABLE",
    "FORBIDDEN_INPUT_COLUMNS",
    "KEYES_DIRECTIONS",
    "VARIABLE_SPECS",
    "X5",
    "X6",
    "X8",
    "X9",
    "X12_PROXY",
    "ComputationConfig",
    "FundamentalHistory",
    "KeyesVariableError",
    "PriceHistory",
    "VariableEngine",
    "VariableSpec",
    "beta_against_benchmark",
    "compute_keyes_variables",
    "wide_variables",
    # signals
    "COMPONENT_SIGNS",
    "HISTORICAL_COMPONENTS",
    "HISTORICAL_RULE_SETS",
    "PRIMARY_THRESHOLDS",
    "REPLICATION_FULL",
    "REPLICATION_PARTIAL",
    "REPLICATION_UNAVAILABLE",
    "SIGNAL_VERSION",
    "CompositeConfig",
    "KeyesSignalError",
    "ThresholdSpec",
    "build_historical_rule_set",
    "direction_payload",
    "fit_train_only_weights",
    "historical_signals",
    "historical_threshold_payload",
    "modern_composite",
    # diagnostics
    "DIAGNOSTIC_VERSION",
    "DiagnosticError",
    "component_diagnostics",
    "composite_diagnostics",
    "concentration_and_turnover",
    "fidelity_table",
    "observable_slice",
    "placebo_checks",
    "qualification_summary",
    "rank_ic",
    "replication_status",
    "sector_component_stability",
    "yearly_component_stability",
]
