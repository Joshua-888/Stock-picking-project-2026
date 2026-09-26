"""WP5 discovery catalogue: versions, configuration and the candidate feature universe.

WP5 is VARIABLE DISCOVERY. It materialises a point-in-time feature panel and
characterises every candidate feature honestly. It deliberately produces NO
final, full-sample-selected feature set and NO single weighted score: WP6/WP7 do
target-driven selection inside training windows. What leaves WP5 is a
``CANDIDATE_FEATURE_UNIVERSE`` with factual research classifications.

Nothing here computes a value or fits anything; it declares WHAT is researched,
under which a-priori parameters, and how a candidate is categorised.
"""

from __future__ import annotations

from dataclasses import dataclass

DISCOVERY_VERSION = "v2_wp5_discovery_v1"
FEATURE_PANEL_VERSION = "v2_wp5_feature_panel_v1"
QUALITY_VERSION = "v2_wp5_feature_quality_v1"
IC_VERSION = "v2_wp5_cross_sectional_ic_v1"
INFERENCE_VERSION = "v2_wp5_dependence_inference_v1"
QUANTILE_VERSION = "v2_wp5_quantile_v1"
STABILITY_VERSION = "v2_wp5_temporal_stability_v1"
MISSINGNESS_VERSION = "v2_wp5_missingness_v1"
FDR_VERSION = "v2_wp5_multiple_testing_v1"
REDUNDANCY_VERSION = "v2_wp5_redundancy_v1"
PLACEBO_VERSION = "v2_wp5_placebo_v1"
SCORECARD_VERSION = "v2_wp5_scorecard_v1"
HANDOFF_VERSION = "v2_wp5_wp6_handoff_v1"

# Research clusters used for the redundancy analysis. These group features by the
# economic quantity they describe; a cluster is NEVER used to pick a "winner".
CLUSTER_FAMILIES = (
    "VALUATION",
    "PROFITABILITY",
    "QUALITY",
    "CAPITAL_EFFICIENCY",
    "GROWTH",
    "MOMENTUM",
    "RISK",
    "SIZE",
)

class CatalogError(ValueError):
    """Raised when a discovery-catalog request is ill-formed."""


# Research categories (NOT a ranking).
CLASSIFICATIONS = (
    "ROBUST_CANDIDATE",
    "PROMISING_BUT_UNSTABLE",
    "REDUNDANT",
    "LOW_COVERAGE",
    "NO_CLEAR_SIGNAL",
    "DIRECTION_UNSTABLE",
    "TEMPORALLY_UNSAFE",
    "DEFER",
)


@dataclass(frozen=True)
class DiscoveryConfig:
    """A-priori discovery parameters. Chosen before any result was seen.

    These are fixed design constants, not tuned knobs. ``hac_lags`` and the
    bootstrap block length are set from the overlap structure of the monthly IC
    series, not from any performance value.
    """

    min_ic_observations: int = 30
    min_rank_variance: float = 1e-12
    hac_lags: int = 12
    bootstrap_block: int = 6
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 20260926
    quantile_buckets: int = 5
    min_quantile_observations: int = 30
    fdr_alpha: float = 0.05
    redundancy_rank_corr: float = 0.7
    redundancy_ic_corr: float = 0.7
    min_months_for_stability: int = 24
    low_coverage_threshold: float = 0.35
    stable_positive_share: float = 0.6
    stable_negative_share: float = 0.6
    keyes_benchmark: tuple = ("X5", "X6", "X8", "X9", "X12_PROXY")
    placeholder_ic_floor: float = 0.05

    def to_dict(self):
        payload = dict(self.__dict__)
        payload["keyes_benchmark"] = list(self.keyes_benchmark)
        return payload


@dataclass(frozen=True)
class FeatureSpec:
    """Declared identity and integrity contract of one candidate feature."""

    feature_name: str
    registry_feature_id: str
    category: str
    cluster_family: str
    hypothesized_direction: str
    source: str
    pit_status: str
    missing_policy: str
    allowed_transformations: tuple
    definition: str
    is_keyes: bool = False

    def to_dict(self):
        payload = dict(self.__dict__)
        payload["allowed_transformations"] = list(self.allowed_transformations)
        return payload


_TRANSFORM = ("TRAIN_ZSCORE", "TRAIN_PERCENTILE_CLIP")
_POS = "POSITIVE"
_NEG = "NEGATIVE"
_UNK = "UNKNOWN"
_PIT = "POINT_IN_TIME"


def _spec(name, registry_id, category, family, direction, source, definition,
          pit_status=_PIT, missing_policy="EXCLUDE_ROW", is_keyes=False):
    return FeatureSpec(
        feature_name=name,
        registry_feature_id=registry_id,
        category=category,
        cluster_family=family,
        hypothesized_direction=direction,
        source=source,
        pit_status=pit_status,
        missing_policy=missing_policy,
        allowed_transformations=_TRANSFORM,
        definition=definition,
        is_keyes=is_keyes,
    )


FEATURE_CATALOG = (
    # ── VALUATION ────────────────────────────────────────────────────────────
    _spec("earnings_yield", "val_earnings_yield", "VALUATION", "VALUATION", _POS,
          "SEC EDGAR + prices", "trailing-twelve-month net income / market cap"),
    _spec("current_pe", "val_current_pe", "VALUATION", "VALUATION", _NEG,
          "SEC EDGAR + prices", "prediction-date price / trailing-twelve-month filed EPS (Keyes X8); undefined for non-positive EPS",
          is_keyes=True),
    _spec("price_to_book", "val_price_to_book", "VALUATION", "VALUATION", _NEG,
          "SEC EDGAR + prices", "market cap / latest filed total equity"),
    _spec("price_to_sales", "val_price_to_sales", "VALUATION", "VALUATION", _NEG,
          "SEC EDGAR + prices", "market cap / trailing-twelve-month revenue"),
    _spec("free_cash_flow_yield", "val_fcf_yield", "VALUATION", "VALUATION", _POS,
          "SEC EDGAR + prices", "trailing-twelve-month free cash flow / market cap"),
    _spec("dividend_yield", "val_dividend_yield", "VALUATION", "VALUATION", _POS,
          "corporate actions + prices", "declared cash dividends over the trailing twelve months / price (no estimation)"),
    # ── PROFITABILITY / QUALITY / CAPITAL EFFICIENCY ─────────────────────────
    _spec("roa", "prof_roa", "PROFITABILITY", "PROFITABILITY", _POS,
          "SEC EDGAR", "net income / average total assets (return_metrics V2 definition) ", "POINT_IN_TIME", "EXCLUDE_ROW"),
    _spec("roe", "prof_roe", "PROFITABILITY", "PROFITABILITY", _POS,
          "SEC EDGAR", "trailing-twelve-month net income / latest filed total equity"),
    _spec("gross_margin", "prof_gross_margin", "PROFITABILITY", "QUALITY", _POS,
          "SEC EDGAR", "trailing-twelve-month gross profit / revenue"),
    _spec("operating_margin", "prof_operating_margin", "PROFITABILITY", "QUALITY", _POS,
          "SEC EDGAR", "trailing-twelve-month operating income / revenue"),
    _spec("roic", "prof_roic", "PROFITABILITY", "CAPITAL_EFFICIENCY", _POS,
          "SEC EDGAR", "NOPAT / invested capital; invested capital = total debt + equity - cash"),
    # ── GROWTH ───────────────────────────────────────────────────────────────
    _spec("five_year_eps_growth", "growth_eps_5y", "GROWTH", "GROWTH", _POS,
          "SEC EDGAR", "compound annual growth rate of trailing-twelve-month EPS over five years (Keyes X5)",
          is_keyes=True),
    _spec("five_year_revenue_growth", "growth_revenue_5y", "GROWTH", "GROWTH", _POS,
          "SEC EDGAR", "compound annual growth rate of trailing-twelve-month revenue over five years (NOT a Keyes X12 substitute)"),
    _spec("eps_growth_acceleration", "growth_eps_accel", "GROWTH", "GROWTH", _POS,
          "SEC EDGAR", "recent one-year TTM EPS growth minus prior one-year TTM EPS growth"),
    _spec("revenue_growth_acceleration", "growth_revenue_accel", "GROWTH", "GROWTH", _POS,
          "SEC EDGAR", "recent one-year TTM revenue growth minus prior one-year TTM revenue growth"),
    _spec("x12_appreciation_proxy", "keyes_x12_proxy", "REVISION", "GROWTH", _POS,
          "unavailable", "X12_PROXY: explicit stand-in for Keyes X12 forward expected appreciation; never reported as X12",
          pit_status="PROXY", is_keyes=True),
    # ── MOMENTUM / LIQUIDITY / RISK / SIZE ───────────────────────────────────
    _spec("six_month_momentum", "mom_6m", "MOMENTUM", "MOMENTUM", _POS,
          "prices + corporate actions", "point-in-time total return over the trailing six months"),
    _spec("twelve_month_momentum", "mom_12m", "MOMENTUM", "MOMENTUM", _POS,
          "prices + corporate actions", "point-in-time total return over the trailing twelve months"),
    _spec("five_year_price_gain", "mom_price_5y", "MOMENTUM", "MOMENTUM", _POS,
          "prices + corporate actions", "point-in-time total return over the trailing five years (Keyes X6)", is_keyes=True),
    _spec("abnormal_volume", "liq_abnormal_volume", "LIQUIDITY", "RISK", _UNK,
          "prices", "current volume / trailing twelve-month average volume (volume source absent from the certified PIT price table)",
          pit_status="UNKNOWN"),
    _spec("beta", "risk_beta", "RISK", "RISK", _NEG,
          "prices + benchmark", "trailing-window OLS slope of stock returns on the SPY benchmark, fitted on the trailing window only"),
    _spec("debt_to_equity", "bs_debt_to_equity", "BALANCE_SHEET", "RISK", _NEG,
          "SEC EDGAR", "latest filed total debt / total equity"),
    _spec("market_cap", "size_log_market_cap", "SIZE", "SIZE", _NEG,
          "SEC EDGAR + prices", "prediction-date price times latest filed shares outstanding (ranked; log applied downstream on training data only)"),
)

FEATURE_NAMES = tuple(spec.feature_name for spec in FEATURE_CATALOG)
CATALOG_BY_NAME = {spec.feature_name: spec for spec in FEATURE_CATALOG}


def catalog_payload():
    """Deterministic catalogue payload used in the discovery experiment binding."""
    return {
        "discovery_version": DISCOVERY_VERSION,
        "feature_panel_version": FEATURE_PANEL_VERSION,
        "features": [spec.to_dict() for spec in FEATURE_CATALOG],
    }
