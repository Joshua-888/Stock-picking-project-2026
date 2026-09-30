"""WP6 model-research CONTRACT (frozen BEFORE any model result was read).

This module declares, in code, the a-priori policy that WP6 must obey. It is
written and reviewed before a single model metric is computed, so no threshold,
feature list, fold boundary or category rule can be reverse-engineered from a
result. Nothing here fits a model or reads a label.

Scope guard: WP6 is DEVELOPMENT-PERIOD model research. The locked holdout
(``feature_asof >= 2022-01-01``) and its labels are NEVER read, scored or
mentioned in any metric. WP7 (nested walk-forward validation) is NOT started
here and no final production model is fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..discovery.catalog import CATALOG_BY_NAME, FEATURE_NAMES

# ── Version stamps (content-addressed into the experiment binding) ───────────
WP6_CONTRACT_VERSION = "v2_wp6_model_research_v1"
WP6_PANEL_VERSION = "v2_wp6_modeling_panel_v1"
WP6_FOLDS_VERSION = "v2_wp6_walk_forward_folds_v1"
WP6_PREPROCESSING_VERSION = "v2_wp6_preprocessing_v1"
WP6_FEATURES_VERSION = "v2_wp6_feature_strategies_v1"
WP6_MODELS_VERSION = "v2_wp6_model_registry_v1"
WP6_METRICS_VERSION = "v2_wp6_metrics_v1"
WP6_INFERENCE_VERSION = "v2_wp6_inference_v1"
WP6_PLACEBO_VERSION = "v2_wp6_placebo_v1"
WP6_RUNNER_VERSION = "v2_wp6_runner_v1"

# ── Explicit, frozen feature exclusions ──────────────────────────────────────
# These three candidates are REMOVED from the WP6 feature universe and must never
# be resurrected by a later strategy: nothing may undo an a-priori exclusion.
EXCLUDED_FEATURES = ("abnormal_volume", "beta", "x12_appreciation_proxy")
EXCLUSION_REASONS = {
    "abnormal_volume": "UNAVAILABLE: certified PIT price table has no volume column (WP5 coverage 0.0)",
    "beta": "UNAVAILABLE: zero coverage in the certified WP5 panel",
    "x12_appreciation_proxy": "X12_PROXY: no point-in-time forward-appreciation source; never treated as X12",
}

# ── Research categories (NOT a ranking) ──────────────────────────────────────
RESEARCH_CATEGORIES = ("PROMISING", "INCONCLUSIVE", "UNSTABLE", "NO_EVIDENCE", "REJECTED")

# ── Declared baselines (transparent, no arbitrary composite score) ────────────
# The primary-metric baseline has cross-sectional variation so a rank metric is
# defined: an EQUAL-WEIGHT composite of the eligible features' within-date ranks.
# The loss baselines are the train mean (regression) / train base rate (classification).
PRIMARY_BASELINE_MODEL = "baseline_ew_composite"
REGRESSION_LOSS_BASELINE = "baseline_mean"
CLASSIFICATION_LOSS_BASELINE = "baseline_base_rate"


@dataclass(frozen=True)
class ModelingConfig:
    """Frozen WP6 design constants. Not tuned to any result."""

    # targets
    regression_target: str = "future_12m_excess_return"
    classification_target: str = "outperform_12m"

    # eligibility / temporal policy
    horizon_months: int = 12
    validation_window_months: int = 24
    min_train_months: int = 60
    min_folds: int = 4

    # cross-sectional metric floor
    min_cross_section_obs: int = 30

    # feature-strategy thresholds (all measured in TRAIN windows only)
    coverage_threshold: float = 0.35
    correlation_threshold: float = 0.70
    ic_top_k: int = 5

    # preprocessing
    clip_low: float = 0.01
    clip_high: float = 0.99

    # inference
    hac_lags: int = 12
    bootstrap_block: int = 6
    bootstrap_iterations: int = 1000

    # category rule constants (predeclared, not tuned)
    alpha: float = 0.05
    fold_positive_fraction: float = 0.75
    no_evidence_ic_floor: float = 0.01

    # seeds
    model_seed: int = 20260930
    bootstrap_seed: int = 20260926
    placebo_seed: int = 20260926

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = ModelingConfig()


def model_feature_universe():
    """Eligible WP6 features: the 23 candidates minus the frozen exclusions."""
    return tuple(name for name in FEATURE_NAMES if name not in EXCLUDED_FEATURES)


def economic_family(feature):
    """Declared economic family of a feature (from the WP3/WP5 catalogue)."""
    return CATALOG_BY_NAME[feature].cluster_family


def contract_payload(config=None):
    """Canonical, deterministic contract payload used in the experiment id."""
    config = config or DEFAULT_CONFIG
    universe = model_feature_universe()
    return {
        "contract_version": WP6_CONTRACT_VERSION,
        "panel_version": WP6_PANEL_VERSION,
        "folds_version": WP6_FOLDS_VERSION,
        "preprocessing_version": WP6_PREPROCESSING_VERSION,
        "features_version": WP6_FEATURES_VERSION,
        "models_version": WP6_MODELS_VERSION,
        "metrics_version": WP6_METRICS_VERSION,
        "inference_version": WP6_INFERENCE_VERSION,
        "placebo_version": WP6_PLACEBO_VERSION,
        "runner_version": WP6_RUNNER_VERSION,
        "config": config.to_dict(),
        "feature_universe": list(universe),
        "excluded_features": {name: EXCLUSION_REASONS[name] for name in EXCLUDED_FEATURES},
        "research_categories": list(RESEARCH_CATEGORIES),
        "baselines": {
            "primary": PRIMARY_BASELINE_MODEL,
            "regression_loss": REGRESSION_LOSS_BASELINE,
            "classification_loss": CLASSIFICATION_LOSS_BASELINE,
        },
    }
