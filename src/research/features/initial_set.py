"""WP3 initial V2 feature set and the V1 disposition table.

Two deliverables:

1. ``v1_disposition()`` - every legacy V1 feature labelled KEEP / REDEFINE /
   LEGACY_ONLY / DROP / DEFER with a reason. This is a persisted research
   artifact, not a comment: a reviewer can see exactly what was carried over and
   why.
2. ``build_initial_registry()`` - the first V2 candidate registry, containing
   only features whose inputs and availability are defensible. Macro features
   are declared but DISABLED, so no macro vintage can silently enter a model.

Feature VALUES are not computed here; the registry is a contract that WP4/WP5
implement against.
"""

from __future__ import annotations

from .registry import (
    FeatureDefinition,
    FeatureRegistry,
)

# Disposition codes.
KEEP = "KEEP"
REDEFINE = "REDEFINE"
LEGACY_ONLY = "LEGACY_ONLY"
DROP = "DROP"
DEFER = "DEFER"

DISPOSITIONS = (KEEP, REDEFINE, LEGACY_ONLY, DROP, DEFER)

# Legacy V1 features (from src/features/feature_engineering.py, read-only) and
# the V2 decision for each. Reasons are short but load-bearing.
V1_DISPOSITION = {
    "current_pe_ratio": (KEEP, "PIT-able from price and TTM EPS available by T"),
    "pe_vs_historical_median": (REDEFINE, "V1 median window unspecified; V2 uses a fixed PIT lookback"),
    "five_year_eps_growth": (KEEP, "Keyes X5; computable from filed EPS history"),
    "five_year_revenue_growth": (REDEFINE, "kept, but explicitly NOT a stand-in for Keyes X12"),
    "roe": (KEEP, "net_income/equity, PIT from filings"),
    "roic": (REDEFINE, "V2 distinct NOPAT / invested_capital definition (return_metrics)"),
    "roa": (REDEFINE, "V2 average-total-assets definition; V1 conflated with ROIC semantics"),
    "debt_to_equity": (KEEP, "PIT from filings"),
    "free_cash_flow_yield": (KEEP, "TTM FCF / market cap, both available by T"),
    "gross_margin": (KEEP, "PIT from filings"),
    "operating_margin": (KEEP, "PIT from filings"),
    "dividend_yield": (REDEFINE, "V1 estimated dividends; V2 uses actual declared dividends known by T"),
    "market_cap": (KEEP, "price x shares outstanding, both PIT"),
    "price_to_sales": (KEEP, "market cap / TTM revenue, PIT"),
    "price_to_book": (KEEP, "market cap / equity, PIT"),
    "six_month_momentum": (KEEP, "price-only, available by T"),
    "twelve_month_momentum": (KEEP, "price-only, available by T"),
    "five_year_price_gain": (KEEP, "Keyes X6; price-only"),
    "abnormal_volume": (KEEP, "volume-only, available by T"),
    "beta": (REDEFINE, "V2 beta fitted on a training window only; never full-sample"),
    "eps_growth_acceleration": (KEEP, "from filed EPS history"),
    "revenue_growth_acceleration": (KEEP, "from filed revenue history"),
    "sector_relative_pe": (REDEFINE, "V2 sector medians computed cross-sectionally at T only"),
    "sector_relative_momentum": (REDEFINE, "V2 sector medians computed cross-sectionally at T only"),
    "sector_relative_ps": (REDEFINE, "V2 sector medians computed cross-sectionally at T only"),
    "sector_relative_fcf_yield": (REDEFINE, "V2 sector medians computed cross-sectionally at T only"),
    "data_quality_score": (DROP, "not a market input; would leak an arbitrary quality judgement into features"),
    "keyes_qualified_flag": (DROP, "V1 forced ~top 30% quota; V2 uses explicit logical thresholds only"),
    "analyst_forward_appreciation": (DEFER, "Keyes X12; no PIT source yet - materialise as X12_PROXY, never as X12"),
}


class InitialSetError(ValueError):
    """Raised when the initial set cannot be assembled."""


def v1_disposition():
    """Return the disposition table as ``{feature: {disposition, reason}}``."""
    return {
        name: {"disposition": code, "reason": reason}
        for name, (code, reason) in sorted(V1_DISPOSITION.items())
    }


def _definition(feature_id, name, category, definition, source, deps, transformation,
                direction, available_at_rule, lookback, missing_rule, legacy, enabled=True, notes=""):
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=name,
        category=category,
        definition=definition,
        source=source,
        raw_dependencies=tuple(deps),
        transformation=transformation,
        hypothesized_direction=direction,
        available_at_rule=available_at_rule,
        lookback_window=lookback,
        missing_rule=missing_rule,
        winsorization_rule="TRAIN_PERCENTILE_CLIP",
        normalization_rule="TRAIN_ZSCORE",
        pit_status="POINT_IN_TIME",
        legacy_mapping=legacy,
        enabled=enabled,
        notes=notes,
    )


def build_initial_registry(include_macro=False):
    """Assemble the initial V2 candidate registry.

    Fundamental features use the SEC filing availability rule; price features use
    the conservative close rule. Macro features are declared DISABLED unless
    ``include_macro`` is explicitly true, so they never enter a model by
    accident.
    """
    registry = FeatureRegistry()
    filing = "SEC_FILING_AVAILABILITY"
    close = "PRICE_CLOSE_AVAILABILITY"

    defs = [
        # VALUATION
        _definition("val_earnings_yield", "earnings_yield", "VALUATION",
                    "TTM net income / market cap", "SEC EDGAR + prices",
                    ("net_income", "shares_outstanding", "raw_close"), "inverse_pe", "POSITIVE",
                    filing, "4 quarters", "INDICATOR_PLUS_TRAIN_MEDIAN", "current_pe_ratio"),
        _definition("val_price_to_book", "price_to_book", "VALUATION",
                    "market cap / total equity", "SEC EDGAR + prices",
                    ("equity", "shares_outstanding", "raw_close"), "ratio", "NEGATIVE",
                    filing, "latest", "INDICATOR_PLUS_TRAIN_MEDIAN", "price_to_book"),
        _definition("val_price_to_sales", "price_to_sales", "VALUATION",
                    "market cap / TTM revenue", "SEC EDGAR + prices",
                    ("revenue", "shares_outstanding", "raw_close"), "ratio", "NEGATIVE",
                    filing, "4 quarters", "INDICATOR_PLUS_TRAIN_MEDIAN", "price_to_sales"),
        _definition("val_fcf_yield", "free_cash_flow_yield", "VALUATION",
                    "TTM free cash flow / market cap", "SEC EDGAR + prices",
                    ("free_cash_flow", "shares_outstanding", "raw_close"), "ratio", "POSITIVE",
                    filing, "4 quarters", "INDICATOR_PLUS_TRAIN_MEDIAN", "free_cash_flow_yield"),
        _definition("val_dividend_yield", "dividend_yield", "VALUATION",
                    "actual declared dividends over TTM / price (no estimation)", "EODHD + prices",
                    ("dividends", "raw_close"), "ratio", "POSITIVE",
                    close, "12 months", "INDICATOR_PLUS_TRAIN_MEDIAN", "dividend_yield"),
        # PROFITABILITY
        _definition("prof_roa", "roa", "PROFITABILITY",
                    "net_income / average total assets (return_metrics V2)", "SEC EDGAR",
                    ("net_income", "total_assets", "total_assets_begin"), "ratio", "POSITIVE",
                    filing, "1 period", "INDICATOR_PLUS_TRAIN_MEDIAN", "roa"),
        _definition("prof_roic", "roic", "PROFITABILITY",
                    "NOPAT / invested_capital; invested_capital = total_debt + equity - cash", "SEC EDGAR",
                    ("operating_income", "total_debt", "equity", "cash"), "ratio", "POSITIVE",
                    filing, "1 period", "INDICATOR_PLUS_TRAIN_MEDIAN", "roic"),
        _definition("prof_gross_margin", "gross_margin", "PROFITABILITY",
                    "gross profit / revenue", "SEC EDGAR",
                    ("gross_profit", "revenue"), "ratio", "POSITIVE",
                    filing, "4 quarters", "INDICATOR_PLUS_TRAIN_MEDIAN", "gross_margin"),
        _definition("prof_operating_margin", "operating_margin", "PROFITABILITY",
                    "operating income / revenue", "SEC EDGAR",
                    ("operating_income", "revenue"), "ratio", "POSITIVE",
                    filing, "4 quarters", "INDICATOR_PLUS_TRAIN_MEDIAN", "operating_margin"),
        _definition("prof_roe", "roe", "PROFITABILITY",
                    "net income / total equity", "SEC EDGAR",
                    ("net_income", "equity"), "ratio", "POSITIVE",
                    filing, "1 period", "INDICATOR_PLUS_TRAIN_MEDIAN", "roe"),
        # GROWTH
        _definition("growth_eps_5y", "five_year_eps_growth", "GROWTH",
                    "CAGR of TTM EPS over five years (Keyes X5)", "SEC EDGAR",
                    ("eps",), "cagr", "POSITIVE",
                    filing, "5 years", "EXCLUDE_ROW", "five_year_eps_growth",
                    notes="Keyes variable X5 - HISTORICAL_KEYES_REPLICATION track"),
        _definition("growth_revenue_5y", "five_year_revenue_growth", "GROWTH",
                    "CAGR of TTM revenue over five years", "SEC EDGAR",
                    ("revenue",), "cagr", "POSITIVE",
                    filing, "5 years", "EXCLUDE_ROW", "five_year_revenue_growth",
                    notes="NOT a stand-in for Keyes X12 - see X12_PROXY"),
        _definition("growth_eps_accel", "eps_growth_acceleration", "GROWTH",
                    "recent TTM EPS growth minus prior TTM EPS growth", "SEC EDGAR",
                    ("eps",), "difference", "POSITIVE",
                    filing, "2 years", "EXCLUDE_ROW", "eps_growth_acceleration"),
        _definition("growth_revenue_accel", "revenue_growth_acceleration", "GROWTH",
                    "recent TTM revenue growth minus prior TTM revenue growth", "SEC EDGAR",
                    ("revenue",), "difference", "POSITIVE",
                    filing, "2 years", "EXCLUDE_ROW", "revenue_growth_acceleration"),
        # MOMENTUM
        _definition("mom_6m", "six_month_momentum", "MOMENTUM",
                    "PIT-adjusted total return over trailing 6 months", "EODHD prices",
                    ("raw_close", "splits", "dividends"), "pit_return", "POSITIVE",
                    close, "6 months", "EXCLUDE_ROW", "six_month_momentum"),
        _definition("mom_12m", "twelve_month_momentum", "MOMENTUM",
                    "PIT-adjusted total return over trailing 12 months", "EODHD prices",
                    ("raw_close", "splits", "dividends"), "pit_return", "POSITIVE",
                    close, "12 months", "EXCLUDE_ROW", "twelve_month_momentum"),
        _definition("mom_price_5y", "five_year_price_gain", "MOMENTUM",
                    "PIT-adjusted total return over trailing 5 years (Keyes X6)", "EODHD prices",
                    ("raw_close", "splits", "dividends"), "pit_return", "POSITIVE",
                    close, "5 years", "EXCLUDE_ROW", "five_year_price_gain",
                    notes="Keyes variable X6 - HISTORICAL_KEYES_REPLICATION track"),
        # LIQUIDITY
        _definition("liq_abnormal_volume", "abnormal_volume", "LIQUIDITY",
                    "current volume / trailing 12-month average volume", "EODHD prices",
                    ("volume",), "ratio", "UNKNOWN",
                    close, "12 months", "INDICATOR_PLUS_TRAIN_MEDIAN", "abnormal_volume"),
        # RISK
        _definition("risk_beta", "beta", "RISK",
                    "return beta vs SPY fitted on a TRAINING window only", "EODHD prices",
                    ("raw_close", "benchmark_close"), "ols", "NEGATIVE",
                    close, "training window", "EXCLUDE_ROW", "beta",
                    notes="V1 fitted full-sample beta; V2 must fit on training data only"),
        # SIZE
        _definition("size_log_market_cap", "market_cap", "SIZE",
                    "log(price x shares outstanding), both PIT", "SEC EDGAR + prices",
                    ("shares_outstanding", "raw_close"), "log", "NEGATIVE",
                    filing, "latest", "INDICATOR_PLUS_TRAIN_MEDIAN", "market_cap"),
        # BALANCE_SHEET
        _definition("bs_debt_to_equity", "debt_to_equity", "BALANCE_SHEET",
                    "total debt / total equity", "SEC EDGAR",
                    ("total_debt", "equity"), "ratio", "NEGATIVE",
                    filing, "latest", "INDICATOR_PLUS_TRAIN_MEDIAN", "debt_to_equity"),
    ]
    for definition in defs:
        registry.add(definition)

    registry.add(_definition(
        "keyes_x12_proxy", "x12_appreciation_proxy", "REVISION",
        "PROXY for Keyes X12 forward expected appreciation; NO true PIT X12 source exists",
        "unavailable", ("proxy_series",), "identity", "POSITIVE",
        filing, "unspecified", "EXCLUDE_ROW", "analyst_forward_appreciation",
        enabled=False,
        notes="named X12_PROXY explicitly; never reported as Keyes X12; disabled until a proxy is defined",
    ))

    macro_defs = [
        _definition("macro_term_spread", "term_spread", "MACRO",
                    "10y minus 2y treasury spread, ALFRED vintage", "FRED/ALFRED",
                    ("dgs10", "dgs2"), "difference", "UNKNOWN",
                    "MACRO_VINTAGE_AVAILABILITY", "latest", "EXCLUDE_ROW", "",
                    enabled=bool(include_macro), notes="macro features OFF until vintages are wired"),
        _definition("macro_regime_proxy", "market_regime", "MARKET_REGIME",
                    "benchmark trailing regime descriptor known by T", "EODHD prices",
                    ("benchmark_close",), "regime", "UNKNOWN",
                    close, "12 months", "EXCLUDE_ROW", "",
                    enabled=bool(include_macro), notes="regime feature OFF pending design review"),
    ]
    for definition in macro_defs:
        registry.add(definition)

    return registry.seal()