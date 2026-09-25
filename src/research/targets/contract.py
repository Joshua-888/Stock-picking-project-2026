"""WP3 target contract: versioned, explicit, point-in-time-safe target definitions.

This module declares WHAT the research targets are and the policy under which
they may be computed. It computes nothing from raw data: construction lives in
``src.research.targets.build`` and date policy in ``src.research.targets.matching``.

The primary target is benchmark-relative (vs SPY) because absolute returns in a
bull market would reward a model that simply predicts "up". Future returns are
labels, never features, and are only computed where WP2C marked the observation
``target_observable``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ids import target_set_id

TARGET_VERSION = "v2_wp3_targets_v1"
DEFAULT_HORIZON_MONTHS = 12
DEFAULT_BENCHMARK = "SPY"

CONTINUOUS_TARGET = "future_12m_excess_return"
CLASSIFICATION_TARGET = "outperform_12m"

TARGET_DEFINITIONS = {
    CONTINUOUS_TARGET: "stock_total_return(T -> T+12m) - spy_total_return(T -> T+12m)",
    CLASSIFICATION_TARGET: "1 if future_12m_excess_return > 0 else 0",
}

# How each target return is formed from raw closes and corporate actions known
# through the horizon endpoint. No provider-adjusted close is ever used.
CALCULATION_METHODOLOGY = {
    "stock_total_return": (
        "ratio of point-in-time back-adjusted closes at target_end and target_start, "
        "where the adjustment applies only corporate actions with effective_date in "
        "(target_start, target_end]; raw close reconstructed, never provider-adjusted"
    ),
    "benchmark_total_return": (
        "same point-in-time reconstruction applied to the benchmark series (SPY), "
        "including its dividends and splits"
    ),
    "excess_return": "stock_total_return - benchmark_total_return",
}

CENSORING_POLICY = {
    "compute_only_when": "target_observable == true",
    "consume": "WP2C per-row target_observable / target_censored / target_censor_reason",
    "censored_rows": (
        "retained in the panel but never treated as 0, as a negative label, or as a "
        "silent drop; excluded from supervised labels requiring a known 12m outcome"
    ),
    "delisting_returns": "not manufactured; the source publishes none",
}

AVAILABILITY_METHODOLOGY = {
    "feature_price": "trade date + conservative market close in UTC (availability.PRICE)",
    "feature_fundamental": (
        "SEC acceptance datetime, else filing_date + 1 day; never a fiscal period end "
        "(availability.FUNDAMENTAL)"
    ),
    "target_known_at": (
        "availability instant of the terminal benchmark/stock observation used to form "
        "the label; a row is trainable at T_model only when target_known_at <= T_model"
    ),
}


@dataclass(frozen=True)
class TargetContract:
    """Immutable, content-addressed description of the WP3 target set."""

    target_version: str = TARGET_VERSION
    horizon_months: int = DEFAULT_HORIZON_MONTHS
    benchmark: str = DEFAULT_BENCHMARK
    definitions: dict = None
    calculation_methodology: dict = None
    censoring_policy: dict = None
    availability_methodology: dict = None
    dataset_id: str = None

    def __post_init__(self):
        # frozen dataclass: assign via object.__setattr__ so defaults are deep copies
        if self.definitions is None:
            object.__setattr__(self, "definitions", dict(TARGET_DEFINITIONS))
        if self.calculation_methodology is None:
            object.__setattr__(self, "calculation_methodology", dict(CALCULATION_METHODOLOGY))
        if self.censoring_policy is None:
            object.__setattr__(self, "censoring_policy", dict(CENSORING_POLICY))
        if self.availability_methodology is None:
            object.__setattr__(self, "availability_methodology", dict(AVAILABILITY_METHODOLOGY))

    def payload(self):
        """Canonical payload used for the deterministic target-set identifier."""
        return {
            "target_version": self.target_version,
            "horizon_months": self.horizon_months,
            "benchmark": self.benchmark,
            "definitions": dict(self.definitions),
            "calculation_methodology": dict(self.calculation_methodology),
            "censoring_policy": dict(self.censoring_policy),
            "availability_methodology": dict(self.availability_methodology),
            "dataset_id": self.dataset_id,
        }

    @property
    def target_id(self):
        """Deterministic ``target_set_<hex>`` identifier for this contract."""
        return target_set_id(self.payload())

    def to_dict(self):
        payload = self.payload()
        payload["target_id"] = self.target_id
        return payload