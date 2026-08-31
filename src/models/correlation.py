"""
src/models/correlation.py

Step 9: Correlation analysis between every feature and the target variable.

WHAT THIS MODULE COMPUTES
--------------------------
For each of the 35 features we compute:

1. Pearson r (pooled)
   Classic linear correlation on all (feature, target) pairs at once.
   Fast but treats every stock-month as independent — ignores time structure.

2. Spearman IC (monthly Information Coefficient)
   Each month: rank-correlate feature values with future excess returns
   across the cross-section of ~30 stocks.  Average these monthly ICs.
   This is the standard quant-finance measure of signal quality.
   IC > 0.05 is considered "useful"; IC > 0.10 is considered "strong".

3. IC t-statistic and ICIR
   IC t-stat = mean_IC / (std_IC / sqrt(n_months)) — tests whether IC
   is distinguishable from zero.
   ICIR (IC Information Ratio) = mean_IC / std_IC — higher means more
   consistent signal month-to-month.

4. Multiple-testing corrections
   When testing 35 features at α = 0.05, we expect ~1-2 false positives
   by chance.  Two corrections are applied:
   - Bonferroni: strict (p × n_tests ≤ 0.05)
   - Benjamini-Hochberg FDR: less strict, controls false discovery rate

5. Quintile analysis
   Each month: sort stocks into 5 groups by feature value.
   Report average future excess return per quintile.
   A "good" signal shows a monotone pattern: Q1 worst, Q5 best (or reversed).

6. Rolling IC stability
   Compute IC in rolling 36-month windows.
   A signal whose IC sign flips frequently is unreliable.
   Stability = 1 - (sign_flips / n_windows)

IMPORTANT LIMITATIONS
---------------------
- With 30 stocks we have ~30 observations per cross-section.
  Monthly ICs are noisy individual data points.
- Time-series autocorrelation in returns inflates t-statistics.
  We do NOT apply Newey-West correction here; treat t-stats as indicative.
- Sample data ICs are synthetic.  Real data ICs will differ.
- Finding a significant correlation does NOT prove it will persist
  out-of-sample.  That test happens in the backtesting module.

Usage
-----
    from src.models.correlation import run_correlation_analysis

    results = run_correlation_analysis(features_df, targets_df)
    print(results["summary"])          # per-feature table
    print(results["quintile_analysis"]) # top feature quintile breakdown
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.features.transformations import FEATURE_COLS
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL    = "future_12m_excess_return"
MIN_CROSS_SEC = 5    # min stocks per cross-section to compute IC
MIN_IC_OBS    = 12   # min months of IC to report mean/t-stat
ROLLING_WIN   = 36   # months for rolling IC stability window


# ── Multiple-testing corrections ──────────────────────────────────────────────

def _bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni correction: multiply each p by the number of tests, cap at 1."""
    return np.minimum(p_values * len(p_values), 1.0)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values (not a binary accept/reject).
    """
    n    = len(p_values)
    rank = np.argsort(p_values)          # indices that would sort ascending
    adj  = np.empty(n)

    # Adjusted p = p_i * n / rank_i, then enforce monotonicity right-to-left
    sorted_p = p_values[rank]
    sorted_adj = sorted_p * n / (np.arange(1, n + 1))

    # Cumulative minimum from the right ensures monotonicity
    min_so_far = 1.0
    for i in range(n - 1, -1, -1):
        min_so_far    = min(min_so_far, sorted_adj[i])
        sorted_adj[i] = min_so_far

    adj[rank] = np.minimum(sorted_adj, 1.0)
    return adj


# ── Monthly IC computation ────────────────────────────────────────────────────

def compute_monthly_ic(
    merged_df: pd.DataFrame,
    feature:   str,
) -> pd.Series:
    """
    Compute the monthly Information Coefficient (Spearman rank correlation)
    for one feature.

    Args:
        merged_df: DataFrame with columns feature_date, feature, future_12m_excess_return.
        feature:   Name of the feature column.

    Returns:
        pd.Series of monthly IC values indexed by feature_date.
        Empty if insufficient data.
    """
    monthly_ics: dict[str, float] = {}

    for date, grp in merged_df.groupby("feature_date"):
        valid = grp[[feature, TARGET_COL]].dropna()
        if len(valid) < MIN_CROSS_SEC:
            continue
        ic, _ = stats.spearmanr(valid[feature].values, valid[TARGET_COL].values)
        if not math.isnan(ic):
            monthly_ics[str(date)] = float(ic)

    return pd.Series(monthly_ics, name=feature)


def compute_rolling_ic_stability(
    ic_series: pd.Series,
    window:    int = ROLLING_WIN,
) -> dict:
    """
    Measure IC stability in rolling windows.

    Returns:
        dict with keys:
          rolling_ic_mean_min   – worst rolling-window mean IC
          rolling_ic_mean_max   – best rolling-window mean IC
          pct_positive_windows  – fraction of windows where mean IC > 0
          stability_score       – 0-1; 1 = IC sign never flips across windows
    """
    if len(ic_series) < window:
        return {
            "rolling_ic_mean_min":  None,
            "rolling_ic_mean_max":  None,
            "pct_positive_windows": None,
            "stability_score":      None,
        }

    window_means = ic_series.rolling(window).mean().dropna()
    positive     = (window_means > 0).sum()

    return {
        "rolling_ic_mean_min":  round(float(window_means.min()), 4),
        "rolling_ic_mean_max":  round(float(window_means.max()), 4),
        "pct_positive_windows": round(float(positive / len(window_means)), 4),
        "stability_score":      round(float(positive / len(window_means)), 4),
    }


# ── Quintile analysis ─────────────────────────────────────────────────────────

def compute_quintile_analysis(
    merged_df:   pd.DataFrame,
    feature:     str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Each month: assign stocks to quintiles by feature value.
    Report average future excess return per quintile.

    A signal with genuine predictive power shows a monotone pattern:
      low-feature stocks → low return, high-feature stocks → high return
      (or the reverse, for negatively correlated features).

    Returns:
        DataFrame with columns:
          quantile, mean_excess_return, n_stock_months, hit_rate_vs_q1
    """
    rows: list[dict] = []

    for date, grp in merged_df.groupby("feature_date"):
        valid = grp[[feature, TARGET_COL]].dropna()
        if len(valid) < n_quantiles * 2:
            continue
        try:
            valid = valid.copy()
            valid["q"] = pd.qcut(valid[feature], n_quantiles, labels=False, duplicates="drop") + 1
            for q, qg in valid.groupby("q"):
                rows.append({
                    "quantile":    int(q),
                    "excess_ret":  float(qg[TARGET_COL].mean()),
                    "n":           len(qg),
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("quantile")
        .agg(
            mean_excess_return = ("excess_ret", "mean"),
            n_stock_months     = ("n", "sum"),
        )
        .reset_index()
        .round(4)
    )

    # Long-short spread: Q5 - Q1
    if len(summary) >= 2:
        q1 = summary.loc[summary["quantile"] == summary["quantile"].min(), "mean_excess_return"].values[0]
        summary["spread_vs_q1"] = round(summary["mean_excess_return"] - q1, 4)

    return summary


# ── Per-feature correlation summary ──────────────────────────────────────────

def _interpret(ic: Optional[float], significant: bool) -> str:
    if ic is None or math.isnan(ic):
        return "Insufficient data"
    abs_ic = abs(ic)
    direction = "positive" if ic > 0 else "negative"
    if not significant:
        strength = "Not significant after multiple-testing correction"
    elif abs_ic < 0.02:
        strength = "Very weak signal — likely noise"
    elif abs_ic < 0.05:
        strength = "Weak signal — modest value"
    elif abs_ic < 0.10:
        strength = "Moderate signal — worth including"
    else:
        strength = "Strong signal — high predictive value"
    return f"{direction.capitalize()} relationship. {strength}."


def _compute_single_feature(
    merged_df: pd.DataFrame,
    feature:   str,
) -> dict:
    """Compute all statistics for one feature."""
    valid = merged_df[[feature, TARGET_COL]].dropna()
    n     = len(valid)

    result: dict = {
        "feature":      feature,
        "n_obs":        n,
        "pearson_r":    None,
        "pearson_p":    None,
        "spearman_ic":  None,   # mean monthly IC
        "ic_tstat":     None,
        "ic_ir":        None,   # ICIR = mean_IC / std_IC
        "ic_std":       None,
        "p_bonferroni": None,
        "p_bh_fdr":     None,
        "bh_significant": False,
        "direction":    "unknown",
        "stability_score": None,
        "pct_positive_windows": None,
        "interpretation": "Insufficient data",
    }

    if n < 10:
        return result

    # Pooled Pearson correlation
    r, p = stats.pearsonr(valid[feature].values, valid[TARGET_COL].values)
    result["pearson_r"] = round(float(r), 4)
    result["pearson_p"] = round(float(p), 6)
    result["direction"] = "positive" if r > 0 else "negative"

    # Monthly IC (Spearman per cross-section, then averaged)
    ic_series = compute_monthly_ic(merged_df, feature)
    if len(ic_series) >= MIN_IC_OBS:
        mean_ic = float(ic_series.mean())
        std_ic  = float(ic_series.std())
        n_ic    = len(ic_series)

        result["spearman_ic"] = round(mean_ic, 4)
        result["ic_std"]      = round(std_ic, 4)

        if std_ic > 0:
            result["ic_tstat"] = round(mean_ic / (std_ic / math.sqrt(n_ic)), 3)
            result["ic_ir"]    = round(mean_ic / std_ic, 3)

        stability = compute_rolling_ic_stability(ic_series)
        result["stability_score"]      = stability["stability_score"]
        result["pct_positive_windows"] = stability["pct_positive_windows"]

    return result


# ── Main analysis function ────────────────────────────────────────────────────

def run_correlation_analysis(
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Run the full correlation analysis for all features.

    Args:
        features_df:  monthly_features DataFrame (with sector column if available).
        targets_df:   targets DataFrame with future_12m_excess_return column.
        feature_cols: Subset of features to analyse (default: all FEATURE_COLS).

    Returns:
        dict with keys:
          "summary"          – per-feature correlation table (pd.DataFrame)
          "ic_series"        – dict of {feature: pd.Series of monthly ICs}
          "quintile_analysis" – quintile table for the best IC feature
          "top_features"     – list of feature names passing BH correction
          "n_obs"            – total labelled stock-months analysed
          "n_features_tested" – number of features tested
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in features_df.columns]
    if not cols:
        log.warning("No valid feature columns found for correlation analysis.")
        return {}

    # Merge features with labelled targets
    merged = features_df.merge(
        targets_df[["feature_date", "ticker", TARGET_COL]].dropna(subset=[TARGET_COL]),
        on=["feature_date", "ticker"],
        how="inner",
    )

    n_total = len(merged)
    log.info(
        "Correlation analysis: %d features, %d labelled stock-months",
        len(cols), n_total,
    )

    if n_total < 30:
        log.warning("Too few observations (%d) for reliable correlation analysis.", n_total)
        return {}

    # ── Per-feature statistics ──────────────────────────────────────────────
    rows = [_compute_single_feature(merged, f) for f in cols]
    summary_df = pd.DataFrame(rows)

    # ── Apply multiple-testing corrections ────────────────────────────────
    valid_p = summary_df["pearson_p"].dropna().values
    if len(valid_p) > 0:
        p_arr = summary_df["pearson_p"].fillna(1.0).values

        summary_df["p_bonferroni"] = np.round(_bonferroni(p_arr), 6)
        summary_df["p_bh_fdr"]     = np.round(_bh_fdr(p_arr),    6)
        summary_df["bh_significant"] = summary_df["p_bh_fdr"] < 0.05

    # ── Interpretation strings ────────────────────────────────────────────
    summary_df["interpretation"] = summary_df.apply(
        lambda r: _interpret(r["spearman_ic"], r["bh_significant"]), axis=1
    )

    # ── Sort by |IC| descending ───────────────────────────────────────────
    summary_df["abs_ic"] = pd.to_numeric(summary_df["spearman_ic"], errors="coerce").fillna(0).abs()
    summary_df = summary_df.sort_values("abs_ic", ascending=False).drop(columns=["abs_ic"])
    summary_df = summary_df.reset_index(drop=True)

    # ── Monthly IC series for all features ───────────────────────────────
    ic_series = {f: compute_monthly_ic(merged, f) for f in cols}

    # ── Quintile analysis for the best IC feature ────────────────────────
    best_feature = summary_df["feature"].iloc[0] if not summary_df.empty else None
    quintile_df  = pd.DataFrame()
    if best_feature:
        quintile_df = compute_quintile_analysis(merged, best_feature)

    top_features = summary_df.loc[
        summary_df["bh_significant"], "feature"
    ].tolist()

    log.info(
        "Correlation analysis complete. BH-significant features: %d / %d",
        len(top_features), len(cols),
    )

    return {
        "summary":            summary_df,
        "ic_series":          ic_series,
        "quintile_analysis":  quintile_df,
        "top_features":       top_features,
        "n_obs":              n_total,
        "n_features_tested":  len(cols),
        "merged_df":          merged,  # for ad-hoc quintile queries on other features
    }


def get_quintile_for_feature(
    correlation_results: dict,
    feature: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Compute quintile analysis for any feature using cached merged data.
    Avoids re-merging features and targets.
    """
    merged = correlation_results.get("merged_df")
    if merged is None or merged.empty:
        return pd.DataFrame()
    return compute_quintile_analysis(merged, feature, n_quantiles)


def get_ic_heatmap_data(ic_series_dict: dict) -> pd.DataFrame:
    """
    Build a features × months heatmap DataFrame of monthly ICs.
    Useful for the Model Diagnostics page IC chart.

    Returns:
        DataFrame with features as rows, dates as columns, IC values as cells.
    """
    frames = {f: s for f, s in ic_series_dict.items() if not s.empty}
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).T
