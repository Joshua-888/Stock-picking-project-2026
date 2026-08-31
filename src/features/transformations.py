"""
src/features/transformations.py

Feature-level transformations applied after raw feature calculation.

Three operations are available:
  1. Winsorisation  – clips extreme outliers at configurable percentiles
  2. Standardisation – z-score normalises each feature to mean=0, std=1
  3. Min-max scaling – scales each feature to [0, 1]

The pipeline stores BOTH raw and standardised feature values:
  • Raw values    → stored in monthly_features table, shown in dashboard
  • Standardised  → used as model input (features_std_json in snapshots)

This separation is important: raw values are interpretable by humans;
standardised values are needed because regression coefficients are only
comparable when features are on the same scale.

Usage
-----
    from src.features.transformations import winsorize_features, standardize_features

    raw_df = compute_monthly_features(...)
    clean_df = winsorize_features(raw_df)        # clip outliers
    std_df   = standardize_features(clean_df)    # z-score for models
"""

from __future__ import annotations

import pandas as pd

from src.utils.math_utils import winsorize, zscore, minmax_scale

# Feature columns subject to winsorisation and standardisation.
# Metadata columns (ticker, feature_date, data_quality_score) are excluded.
FEATURE_COLS = [
    # ── Keyes-inspired core ───────────────────────────────────────────────────
    "five_year_price_gain",
    "five_year_eps_growth",
    "five_year_revenue_growth",
    "current_pe_ratio",
    "pe_vs_historical_median",
    "dividend_yield",
    # ── Fundamental quality ───────────────────────────────────────────────────
    "roe",
    "roic",
    "debt_to_equity",
    "free_cash_flow_yield",
    "gross_margin",
    "operating_margin",
    "price_to_book",
    "price_to_sales",
    # ── Momentum & risk (original) ────────────────────────────────────────────
    "beta",
    "six_month_momentum",
    "twelve_month_momentum",
    "volatility_12m",
    "market_cap",
    # ── Growth acceleration ───────────────────────────────────────────────────
    "revenue_growth_acceleration",
    "eps_growth_acceleration",
    # ── Sector-relative (original) ────────────────────────────────────────────
    "sector_relative_pe",
    "sector_relative_momentum",
    # ── NEW: Momentum at additional horizons ──────────────────────────────────
    "return_1m",           # short-term reversal signal
    "return_3m",           # intermediate momentum gap-filler
    # ── NEW: Volatility & downside risk ───────────────────────────────────────
    "volatility_3m",
    "downside_volatility_12m",
    # ── NEW: Technical / price level ─────────────────────────────────────────
    "drawdown_from_52w_high",
    # ── NEW: Volume / liquidity ───────────────────────────────────────────────
    "abnormal_volume",
    # ── NEW: Enhanced sector-relative ────────────────────────────────────────
    "sector_relative_ps",
    "sector_relative_fcf_yield",
    # ── NEW: Cross-sectional peer z-scores ────────────────────────────────────
    "peer_momentum_zscore",
    "peer_valuation_zscore",
]


def winsorize_features(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clip extreme outliers at the given quantile bounds.

    Operates cross-sectionally on the full DataFrame passed in —
    quantiles are computed across all rows (all tickers for a given month).

    Args:
        df:    Feature DataFrame (one row per ticker).
        lower: Lower quantile bound (default 1st percentile).
        upper: Upper quantile bound (default 99th percentile).
        cols:  Columns to winsorise (defaults to FEATURE_COLS).

    Returns:
        DataFrame with outlier-clipped numeric columns.
    """
    result = df.copy()
    target_cols = [c for c in (cols or FEATURE_COLS) if c in result.columns]

    for col in target_cols:
        series = result[col]
        if series.dtype in ("float64", "float32", "int64", "int32"):
            result[col] = winsorize(series, lower, upper)

    return result


def standardize_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Z-score standardise feature columns (mean=0, std=1).

    Used to produce the model-input feature matrix.
    Operates cross-sectionally: standardises across all tickers in the DataFrame.

    Args:
        df:   Feature DataFrame (one row per ticker, ideally post-winsorisation).
        cols: Columns to standardise (defaults to FEATURE_COLS).

    Returns:
        DataFrame with standardised numeric columns.
        Column names are unchanged — caller is responsible for labelling.
    """
    result = df.copy()
    target_cols = [c for c in (cols or FEATURE_COLS) if c in result.columns]

    for col in target_cols:
        series = result[col]
        if series.dtype in ("float64", "float32", "int64", "int32"):
            result[col] = zscore(series)

    return result


def minmax_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Scale feature columns to [0, 1] using min-max scaling.

    Used for score component calculations that require values in [0, 1].

    Args:
        df:   Feature DataFrame.
        cols: Columns to scale (defaults to FEATURE_COLS).

    Returns:
        DataFrame with min-max scaled columns.
    """
    result = df.copy()
    target_cols = [c for c in (cols or FEATURE_COLS) if c in result.columns]

    for col in target_cols:
        series = result[col]
        if series.dtype in ("float64", "float32", "int64", "int32"):
            result[col] = minmax_scale(series)

    return result


def get_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary statistics table for all feature columns.

    Useful for the Model Diagnostics dashboard page.

    Returns:
        DataFrame with columns: feature, count, mean, std, min, p25, p50, p75, max, pct_missing
    """
    target_cols = [c for c in FEATURE_COLS if c in df.columns]
    rows = []
    for col in target_cols:
        s = df[col]
        n_missing = s.isna().sum()
        desc = s.describe(percentiles=[0.25, 0.5, 0.75])
        rows.append({
            "feature":     col,
            "count":       int(desc.get("count", 0)),
            "mean":        round(float(desc.get("mean", float("nan"))), 4),
            "std":         round(float(desc.get("std",  float("nan"))), 4),
            "min":         round(float(desc.get("min",  float("nan"))), 4),
            "p25":         round(float(desc.get("25%",  float("nan"))), 4),
            "p50":         round(float(desc.get("50%",  float("nan"))), 4),
            "p75":         round(float(desc.get("75%",  float("nan"))), 4),
            "max":         round(float(desc.get("max",  float("nan"))), 4),
            "pct_missing": round(n_missing / max(len(s), 1) * 100, 1),
        })
    return pd.DataFrame(rows)
