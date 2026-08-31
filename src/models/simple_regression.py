"""
src/models/simple_regression.py

Step 10: Simple linear regression — one OLS model per feature.

For each of the 33 features this module fits:

    future_12m_excess_return = α + β × feature + ε

using Ordinary Least Squares (statsmodels OLS).

WHY SIMPLE REGRESSION AFTER CORRELATION?
-----------------------------------------
Pearson correlation tells us the *direction* and *strength* of a linear
relationship but not the *magnitude*.  Simple regression adds:
  • β (coefficient): "a one-unit increase in this feature is associated with
    a β change in predicted 12-month excess return."
  • Standardised β: "a one standard-deviation increase predicts a [X]%
    change in excess return" — comparable across features with different scales.
  • R²: fraction of return variance explained by this feature alone.
  • t-statistic and p-value: whether the coefficient is distinguishable from zero.
  • 95% confidence interval: the range of plausible coefficient values.
  • Residuals: what the model cannot explain.

This single-feature analysis is the foundation for multiple regression (Step 11),
where we ask: is a feature still predictive after controlling for others?

TRAIN / TEST SPLIT
------------------
We use a TIME-BASED split, never a random split.  A random split would
allow the model to learn from future data leaking into the training set via
cross-sectional correlations.

Default split: train on the earlier 70 % of dates, test on the remaining 30 %.

IMPORTANT LIMITATIONS
---------------------
• Pooled OLS ignores the panel structure (multiple stocks, many dates).
  Standard errors are likely understated because returns are correlated
  across stocks in the same month.  Treat p-values as indicative.
• Low R² is expected and normal in financial return prediction.
  Even R² = 0.01–0.05 can represent meaningful signal at scale.
• OLS assumes a LINEAR relationship.  Non-linear effects are missed.
  The quintile analysis in Step 9 shows whether the real relationship is linear.

Usage
-----
    from src.models.simple_regression import run_all_simple_regressions

    summary_df = run_all_simple_regressions(features_and_targets_df)
    # sorted by R² — highest explanatory power first
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features.transformations import FEATURE_COLS
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "future_12m_excess_return"
DATE_COL   = "feature_date"

# ── Keyes (1972) variable mapping ─────────────────────────────────────────────
# Maps Keyes' original X-labels to our feature column names.
# X12 (Value Line 3-5yr appreciation) is proprietary — we use five_year_revenue_growth
# as the closest available proxy for long-term company-level appreciation potential.
KEYES_VARIABLE_MAP: dict[str, str] = {
    "X5":  "five_year_eps_growth",      # 5-year EPS growth
    "X6":  "five_year_price_gain",      # 5-year stock price gain
    "X8":  "current_pe_ratio",          # Price-to-earnings ratio
    "X9":  "pe_vs_historical_median",   # Current P/E minus historical median P/E
    "X12": "five_year_revenue_growth",  # Proxy: Value Line 3-5yr price appreciation
}


# ── Train / test split ────────────────────────────────────────────────────────

def time_split(
    df:          pd.DataFrame,
    train_frac:  float = 0.70,
    date_col:    str   = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a merged features+targets DataFrame by date.

    The earliest train_frac of unique dates form the training set.
    The remaining dates form the test set.

    This preserves the temporal order and prevents any form of look-ahead.
    """
    dates      = sorted(df[date_col].unique())
    cutoff_idx = int(len(dates) * train_frac)
    cutoff     = dates[cutoff_idx]

    train = df[df[date_col] <  cutoff].copy()
    test  = df[df[date_col] >= cutoff].copy()

    return train, test


# ── Single-feature OLS ────────────────────────────────────────────────────────

def run_simple_regression(
    df:          pd.DataFrame,
    feature:     str,
    target_col:  str   = TARGET_COL,
    train_frac:  float = 0.70,
) -> dict:
    """
    Fit OLS simple linear regression for one feature.

    Returns a dict with all statistics for the summary table.
    """
    valid = df[[feature, target_col, DATE_COL]].dropna()
    n     = len(valid)

    base: dict = {
        "feature":        feature,
        "n_obs":          n,
        "intercept":      None,
        "coefficient":    None,
        "std_error":      None,
        "t_stat":         None,
        "p_value":        None,
        "r_squared":      None,
        "adj_r_squared":  None,
        "f_stat":         None,
        "f_p_value":      None,
        "coef_ci_lower":  None,
        "coef_ci_upper":  None,
        "coef_std":       None,    # standardised coefficient (β × std_X / std_Y)
        "r_squared_test": None,    # out-of-sample R² on held-out test period
        "train_cutoff":   None,
        "direction":      "unknown",
        "significant":    False,
        "interpretation": "Insufficient data",
    }

    if n < 20:
        return base

    # ── Full-sample OLS ───────────────────────────────────────────────────────
    X_full = sm.add_constant(valid[feature].values, has_constant="add")
    y_full = valid[target_col].values

    try:
        result = sm.OLS(y_full, X_full).fit()
    except Exception as exc:
        log.warning("OLS failed for %s: %s", feature, exc)
        return base

    intercept   = float(result.params[0])
    coefficient = float(result.params[1])
    std_error   = float(result.bse[1])
    t_stat      = float(result.tvalues[1])
    p_value     = float(result.pvalues[1])
    r2          = float(result.rsquared)
    adj_r2      = float(result.rsquared_adj)
    f_stat      = float(result.fvalue) if not math.isnan(result.fvalue) else None
    f_p_value   = float(result.f_pvalue) if not math.isnan(result.f_pvalue) else None
    ci_raw      = result.conf_int(alpha=0.05)
    ci_arr      = ci_raw.values if hasattr(ci_raw, "values") else np.asarray(ci_raw)
    ci_lower    = float(ci_arr[1, 0])   # row 1 = coefficient (row 0 = intercept)
    ci_upper    = float(ci_arr[1, 1])

    # Standardised coefficient: β × (std_X / std_Y)
    std_x   = float(valid[feature].std())
    std_y   = float(valid[target_col].std())
    coef_std = (coefficient * std_x / std_y) if std_y > 0 else None

    # ── Time-based train/test split ───────────────────────────────────────────
    train, test = time_split(valid, train_frac)
    r2_test     = None
    train_cutoff = None

    if len(train) >= 20 and len(test) >= 10:
        X_train = sm.add_constant(train[feature].values, has_constant="add")
        y_train = train[target_col].values
        X_test  = sm.add_constant(test[feature].values,  has_constant="add")
        y_test  = test[target_col].values

        try:
            train_result = sm.OLS(y_train, X_train).fit()
            y_pred       = train_result.predict(X_test)
            ss_res       = ((y_test - y_pred) ** 2).sum()
            ss_tot       = ((y_test - y_test.mean()) ** 2).sum()
            r2_test      = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
            train_cutoff = sorted(train[DATE_COL].unique())[-1]
        except Exception:
            pass

    direction = "positive" if coefficient > 0 else "negative"
    significant = p_value < 0.05

    # Interpretation string
    pct_change   = round(coefficient * 100, 2)
    std_pct      = round((coef_std or 0) * 100, 2)
    sig_label    = "significant" if significant else "not statistically significant"
    interp = (
        f"{'Positive' if coefficient > 0 else 'Negative'} relationship ({sig_label}). "
        f"A 1-unit increase predicts {pct_change:+.2f}pp excess return. "
        f"A 1 SD increase predicts {std_pct:+.2f}pp. "
        f"R² = {r2:.4f} ({r2*100:.2f}% of variance explained)."
    )

    return {
        "feature":        feature,
        "n_obs":          n,
        "intercept":      round(intercept,   6),
        "coefficient":    round(coefficient, 6),
        "std_error":      round(std_error,   6),
        "t_stat":         round(t_stat,      3),
        "p_value":        round(p_value,     6),
        "r_squared":      round(r2,          4),
        "adj_r_squared":  round(adj_r2,      4),
        "f_stat":         round(f_stat, 3) if f_stat is not None else None,
        "f_p_value":      round(f_p_value, 6) if f_p_value is not None else None,
        "coef_ci_lower":  round(ci_lower, 6),
        "coef_ci_upper":  round(ci_upper, 6),
        "coef_std":       round(coef_std, 4) if coef_std is not None else None,
        "r_squared_test": round(r2_test, 4) if r2_test is not None else None,
        "train_cutoff":   train_cutoff,
        "direction":      direction,
        "significant":    significant,
        "interpretation": interp,
    }


# ── Fitted values and residuals ───────────────────────────────────────────────

def get_regression_plot_data(
    df:         pd.DataFrame,
    feature:    str,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Return a DataFrame with feature value, actual target, fitted value,
    and residual for every observation.  Used for scatter / residual plots.
    """
    valid = df[[feature, target_col, DATE_COL, "ticker"]].dropna()
    if len(valid) < 10:
        return pd.DataFrame()

    X      = sm.add_constant(valid[feature].values, has_constant="add")
    y      = valid[target_col].values
    result = sm.OLS(y, X).fit()

    out = valid.copy().reset_index(drop=True)
    out["fitted"]   = result.fittedvalues
    out["residual"] = result.resid
    return out


def get_residual_stats(plot_df: pd.DataFrame) -> dict:
    """Return summary statistics for the residuals column."""
    r = plot_df["residual"].dropna()
    return {
        "mean":     round(float(r.mean()), 4),
        "std":      round(float(r.std()),  4),
        "min":      round(float(r.min()),  4),
        "max":      round(float(r.max()),  4),
        "skew":     round(float(r.skew()), 3),
        "kurtosis": round(float(r.kurtosis()), 3),
        "pct_positive": round(float((r > 0).mean()), 4),
    }


# ── Full summary across all features ─────────────────────────────────────────

def run_all_simple_regressions(
    df:           pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col:   str                 = TARGET_COL,
    train_frac:   float               = 0.70,
) -> pd.DataFrame:
    """
    Run OLS simple linear regression for every feature.

    Args:
        df:           Merged features + targets DataFrame (from load_features_and_targets).
        feature_cols: Subset of features to test (default: all FEATURE_COLS).
        target_col:   Regression target column name.
        train_frac:   Fraction of dates used for training in the train/test split.

    Returns:
        DataFrame with one row per feature, sorted by R² descending.
        Includes in-sample AND out-of-sample R² for honest comparison.
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    if not cols:
        log.warning("No valid feature columns found for simple regression.")
        return pd.DataFrame()

    n_obs = df[[TARGET_COL]].dropna().shape[0]
    log.info("Running simple OLS for %d features on %d observations", len(cols), n_obs)

    rows = [run_simple_regression(df, f, target_col, train_frac) for f in cols]
    summary = pd.DataFrame(rows)

    # Sort by in-sample R² descending (best explanatory power first)
    summary = summary.sort_values("r_squared", ascending=False).reset_index(drop=True)

    n_sig = summary["significant"].sum()
    log.info(
        "Simple regression complete: %d features, %d significant at p<0.05, "
        "best R²=%.4f (%s)",
        len(cols), n_sig,
        summary["r_squared"].iloc[0],
        summary["feature"].iloc[0],
    )
    return summary


# ── Coefficient stability over time ──────────────────────────────────────────

def compute_rolling_coefficients(
    df:          pd.DataFrame,
    feature:     str,
    window:      int  = 36,
    target_col:  str  = TARGET_COL,
) -> pd.DataFrame:
    """
    Compute OLS coefficient in rolling windows.

    A feature whose coefficient changes sign frequently is unstable and
    should be treated with caution even if the full-sample p-value is small.

    Returns:
        DataFrame with columns: end_date, coefficient, p_value, r_squared
    """
    valid = df[[feature, target_col, DATE_COL]].dropna().sort_values(DATE_COL)
    dates = sorted(valid[DATE_COL].unique())

    if len(dates) < window + 1:
        return pd.DataFrame()

    rows = []
    for i in range(window, len(dates)):
        window_dates = dates[i - window: i]
        subset = valid[valid[DATE_COL].isin(window_dates)]

        if len(subset) < 15:
            continue

        X = sm.add_constant(subset[feature].values, has_constant="add")
        y = subset[target_col].values

        try:
            res = sm.OLS(y, X).fit()
            rows.append({
                "end_date":    dates[i],
                "coefficient": round(float(res.params[1]), 6),
                "p_value":     round(float(res.pvalues[1]), 6),
                "r_squared":   round(float(res.rsquared), 4),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    # Stability metrics
    sign_changes = (df_out["coefficient"] > 0).diff().abs().sum()
    df_out.attrs["pct_positive_windows"] = round(
        float((df_out["coefficient"] > 0).mean()), 4
    )
    df_out.attrs["sign_changes"] = int(sign_changes)
    return df_out


# ── Keyes (1972) OLS models ───────────────────────────────────────────────────
# Canonical implementations now live in src/models/keyes_ols.py (dedicated file,
# no stale-cache risk). Kept here as forwards for backward compatibility.

def train_keyes_ols_models(
    ft_df:      pd.DataFrame,
    target_col: str = TARGET_COL,
) -> dict[str, dict]:
    """
    Train 5 single-variable OLS models replicating Keyes (1972) Steps 10–11.

        Y = a + b·X   where Y = future_12m_excess_return

    One model per Keyes variable (X5, X6, X8, X9, X12).
    All available historical data is used so coefficients are as accurate
    as possible at prediction time.

    Returns:
        {feature_name: {keyes_key, intercept, coefficient, r_squared, p_value, n_obs}}
    """
    models: dict[str, dict] = {}

    for keyes_key, feature in KEYES_VARIABLE_MAP.items():
        if feature not in ft_df.columns:
            log.debug("Keyes %s (%s) not in training data — skipping", keyes_key, feature)
            continue

        valid = ft_df[[feature, target_col, DATE_COL]].dropna()
        if len(valid) < 20:
            log.debug("Keyes %s: only %d obs — skipping", keyes_key, len(valid))
            continue

        X = sm.add_constant(valid[feature].values, has_constant="add")
        y = valid[target_col].values

        try:
            result = sm.OLS(y, X).fit()
            models[feature] = {
                "keyes_key":   keyes_key,
                "feature":     feature,
                "intercept":   float(result.params[0]),
                "coefficient": float(result.params[1]),
                "r_squared":   float(result.rsquared),
                "p_value":     float(result.pvalues[1]),
                "n_obs":       len(valid),
            }
            log.debug(
                "Keyes %s (%s): a=%.4f  b=%.6f  R²=%.4f  p=%.4f  n=%d",
                keyes_key, feature,
                models[feature]["intercept"], models[feature]["coefficient"],
                models[feature]["r_squared"], models[feature]["p_value"],
                len(valid),
            )
        except Exception as exc:
            log.warning("Keyes OLS failed for %s (%s): %s", keyes_key, feature, exc)

    log.info(
        "Keyes OLS models trained: %d/%d variables available",
        len(models), len(KEYES_VARIABLE_MAP),
    )
    return models


def apply_keyes_ols_models(
    feats_df: pd.DataFrame,
    models:   dict[str, dict],
) -> pd.DataFrame:
    """
    Apply trained Keyes OLS models to the current month's feature values.

    For each model:  predicted_excess_return = intercept + coefficient × X_value

    Missing feature values are filled with the cross-sectional median so every
    stock gets a prediction (conservative — median predicts near-zero excess).

    Args:
        feats_df: Features DataFrame indexed by ticker (current month).
        models:   Output of train_keyes_ols_models().

    Returns:
        DataFrame indexed by ticker.  One column per Keyes model:
        "keyes_X5_pred", "keyes_X6_pred", "keyes_X8_pred", "keyes_X9_pred",
        "keyes_X12_pred"  (values in decimal, e.g. 0.12 = +12% excess return).
    """
    if feats_df.empty or not models:
        return pd.DataFrame()

    preds: dict[str, pd.Series] = {}

    for feature, model in models.items():
        if feature not in feats_df.columns:
            log.debug("Keyes apply: feature %s missing from current features", feature)
            continue

        x = feats_df[feature].copy()
        median_val = x.median()
        x = x.fillna(median_val if not pd.isna(median_val) else 0.0)

        keyes_key = model["keyes_key"]
        pred = model["intercept"] + model["coefficient"] * x
        preds[f"keyes_{keyes_key}_pred"] = pred

    if not preds:
        return pd.DataFrame()

    result = pd.DataFrame(preds, index=feats_df.index).astype(float)
    log.info(
        "Keyes OLS predictions computed for %d stocks (%d models)",
        len(result), len(preds),
    )
    return result
