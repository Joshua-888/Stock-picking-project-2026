"""
src/models/multiple_regression.py

Step 11: Multiple linear regression with VIF multicollinearity checks.

The model:
    excess_return = α + β1·X1 + β2·X2 + … + βn·Xn + ε

WHY MULTIPLE REGRESSION AFTER SIMPLE?
--------------------------------------
Simple regression tests each feature in isolation.  Multiple regression asks:
"Is this feature still predictive AFTER controlling for all the others?"

This is a much harder and more honest test.  A feature that looks strong
individually may be redundant once we add related features (e.g., ROE and ROIC
both measure profitability — together they may add little beyond either alone).

Conversely, a feature that looks weak alone may become significant in
combination with others.

MULTICOLLINEARITY AND VIF
--------------------------
When two or more features are highly correlated, the regression cannot cleanly
separate their individual effects.  The coefficients become unstable: a small
change in the data can flip the sign of a coefficient entirely.

The Variance Inflation Factor (VIF) quantifies this:

    VIF_i = 1 / (1 − R²_i)

where R²_i is the R² from regressing feature i on all other features.

Rules of thumb:
    VIF < 5   → acceptable
    VIF 5–10  → moderate concern; coefficients may be unstable
    VIF > 10  → high concern; coefficient interpretation is unreliable

Expected high-VIF pairs in our feature set:
    ROE / ROIC                       — both measure profitability
    six_month_momentum / twelve_month — momentum at different horizons
    sector_relative_pe / current_pe  — same underlying variable, different scale
    peer_valuation_zscore / current_pe

VIF FILTERING
-------------
We run two models:
    1. Full model — all features with valid data
    2. VIF-filtered model — iteratively remove the highest-VIF feature
       until all remaining features have VIF ≤ threshold (default 10)

ADJUSTED R²
-----------
Unlike simple R², adjusted R² penalises for the number of features:

    Adj-R² = 1 − (1 − R²) × (n − 1) / (n − k − 1)

where n = observations, k = number of features.
A model with more features must earn a meaningfully higher R² to
improve its adjusted R².  If adding a feature reduces adjusted R²,
it is not helping.

TRAIN / TEST SPLIT
------------------
Same time-based split as Step 10 (70 % train, 30 % test).
Out-of-sample R² on the test period is the honest evaluation metric.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.transformations import FEATURE_COLS
from src.models.simple_regression import time_split
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "future_12m_excess_return"
DATE_COL   = "feature_date"


# ── VIF computation ────────────────────────────────────────────────────────────

def compute_vif(
    df:           pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.

    Uses listwise deletion: only rows with no NaN across ALL feature_cols
    are included.  This matches what the actual regression will use.

    Returns:
        DataFrame with columns: feature, vif, concern
        Sorted by VIF descending (worst collinearity first).
    """
    cols  = [c for c in feature_cols if c in df.columns]
    valid = df[cols].dropna()

    if len(valid) < len(cols) + 5:
        log.warning("Too few complete rows for VIF calculation (%d rows, %d features)",
                    len(valid), len(cols))
        return pd.DataFrame({"feature": cols, "vif": [None] * len(cols), "concern": ["—"] * len(cols)})

    X = valid.values.astype(float)
    rows = []
    for i, col in enumerate(cols):
        try:
            v = float(variance_inflation_factor(X, i))
        except Exception:
            v = float("nan")

        if v > 10:    concern = "High — coefficient unreliable"
        elif v > 5:   concern = "Moderate — monitor"
        else:         concern = "Acceptable"

        rows.append({"feature": col, "vif": round(v, 2), "concern": concern})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def select_features_by_vif(
    df:            pd.DataFrame,
    feature_cols:  list[str],
    vif_threshold: float = 10.0,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Iteratively remove the feature with the highest VIF until all remaining
    features are below vif_threshold.

    This greedy approach is the standard practice.  It removes one feature
    at a time rather than all violators at once, because removing one
    high-VIF feature often brings others below the threshold.

    Returns:
        (selected_features, removed_features)
        where removed_features = [(feature_name, vif_at_removal), ...]
    """
    selected = [c for c in feature_cols if c in df.columns]
    removed: list[tuple[str, float]] = []

    while len(selected) >= 2:
        valid = df[selected].dropna()
        if len(valid) < len(selected) + 5:
            break

        X    = valid.values.astype(float)
        vifs = []
        for i in range(X.shape[1]):
            try:
                vifs.append(float(variance_inflation_factor(X, i)))
            except Exception:
                vifs.append(float("nan"))

        max_vif = max((v for v in vifs if not math.isnan(v)), default=0.0)
        if max_vif <= vif_threshold:
            break

        worst_idx = vifs.index(max(vifs))
        removed_name = selected.pop(worst_idx)
        removed.append((removed_name, round(max_vif, 2)))
        log.debug("VIF filter: removed %s (VIF=%.1f)", removed_name, max_vif)

    log.info(
        "VIF filtering: %d features kept, %d removed (threshold=%.0f)",
        len(selected), len(removed), vif_threshold,
    )
    return selected, removed


# ── OLS multiple regression ────────────────────────────────────────────────────

def _fit_ols(X_train: np.ndarray, y_train: np.ndarray,
             X_test: Optional[np.ndarray], y_test: Optional[np.ndarray],
             ) -> dict:
    """Fit OLS and return a result dict with all standard statistics."""
    result = sm.OLS(y_train, X_train).fit()

    r2_test = None
    if X_test is not None and y_test is not None and len(y_test) >= 5:
        y_pred  = result.predict(X_test)
        ss_res  = ((y_test - y_pred) ** 2).sum()
        ss_tot  = ((y_test - y_test.mean()) ** 2).sum()
        r2_test = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    def _safe_float(v):
        try:
            f = float(v)
            return None if math.isnan(f) or math.isinf(f) else f
        except Exception:
            return None

    return {
        "sm_result":      result,
        "r_squared":      float(result.rsquared),
        "adj_r_squared":  float(result.rsquared_adj),
        "r_squared_test": round(r2_test, 4) if r2_test is not None else None,
        "f_stat":         _safe_float(result.fvalue),
        "f_p_value":      _safe_float(result.f_pvalue),
        "aic":            _safe_float(result.aic),
        "bic":            _safe_float(result.bic),
        "n_obs":          int(result.nobs),
        "n_features":     int(result.df_model),
    }


def _build_coefficient_table(
    result,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract per-feature stats from a statsmodels OLS result."""
    # Normalise to plain numpy arrays — statsmodels returns ndarray or Series
    # depending on version; np.asarray() handles both safely.
    params   = np.asarray(result.params)
    bse      = np.asarray(result.bse)
    tvalues  = np.asarray(result.tvalues)
    pvalues  = np.asarray(result.pvalues)
    ci_raw   = result.conf_int(alpha=0.05)
    ci_arr   = ci_raw.values if hasattr(ci_raw, "values") else np.asarray(ci_raw)

    rows = []
    for i, name in enumerate(["intercept"] + feature_names):
        rows.append({
            "feature":     name,
            "coefficient": round(float(params[i]),       6),
            "std_error":   round(float(bse[i]),          6),
            "t_stat":      round(float(tvalues[i]),      3),
            "p_value":     round(float(pvalues[i]),      6),
            "ci_lower":    round(float(ci_arr[i, 0]),    6),
            "ci_upper":    round(float(ci_arr[i, 1]),    6),
            "significant": bool(pvalues[i] < 0.05),
        })
    df = pd.DataFrame(rows)
    # Exclude intercept row — it's not a feature signal
    return df[df["feature"] != "intercept"].reset_index(drop=True)


# ── Main multiple regression runner ───────────────────────────────────────────

def run_multiple_regression(
    df:            pd.DataFrame,
    feature_cols:  Optional[list[str]] = None,
    target_col:    str                 = TARGET_COL,
    train_frac:    float               = 0.70,
    vif_threshold: float               = 10.0,
) -> dict:
    """
    Run OLS multiple regression with VIF filtering.

    Fits two models:
      1. Full model — all features with complete data
      2. VIF-filtered model — features with VIF ≤ vif_threshold

    Args:
        df:            Merged features + targets DataFrame.
        feature_cols:  Features to include (default: all FEATURE_COLS).
        target_col:    Regression target.
        train_frac:    Fraction of dates used for training.
        vif_threshold: Maximum acceptable VIF.

    Returns dict with keys:
        full_model         — stats for the all-features model
        filtered_model     — stats for the VIF-filtered model
        coefficient_table  — per-feature coefficient table (filtered model)
        vif_table_full     — VIF scores for all features
        vif_table_filtered — VIF scores after filtering
        features_full      — list of features in full model
        features_filtered  — list of features in filtered model
        features_removed   — list of (feature, vif) tuples removed by filter
        train_cutoff       — date of train/test boundary
        n_obs_train        — training observations
        n_obs_test         — test observations
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    if not cols:
        log.warning("No valid feature columns for multiple regression.")
        return {}

    # ── Complete-cases dataset ─────────────────────────────────────────────
    needed = cols + [target_col, DATE_COL]
    valid  = df[needed].dropna().copy()
    n_full = len(valid)

    if n_full < 50:
        log.warning("Too few complete observations (%d) for multiple regression.", n_full)
        return {}

    log.info(
        "Multiple regression: %d features, %d complete obs (dropped %d with NaN)",
        len(cols), n_full, len(df) - n_full,
    )

    # ── Time-based train/test split ────────────────────────────────────────
    train, test   = time_split(valid, train_frac, DATE_COL)
    train_cutoff  = sorted(train[DATE_COL].unique())[-1]
    n_obs_train   = len(train)
    n_obs_test    = len(test)

    X_train = sm.add_constant(train[cols].values.astype(float), has_constant="add")
    y_train = train[target_col].values.astype(float)
    X_test  = sm.add_constant(test[cols].values.astype(float),  has_constant="add")
    y_test  = test[target_col].values.astype(float)

    # ── Model 1: Full model ────────────────────────────────────────────────
    try:
        full_stats = _fit_ols(X_train, y_train, X_test, y_test)
    except Exception as exc:
        log.error("Full model OLS failed: %s", exc)
        return {}

    # VIF on the full feature set (using full valid dataset for stability)
    vif_full = compute_vif(valid, cols)

    # ── VIF filtering ──────────────────────────────────────────────────────
    selected, removed = select_features_by_vif(valid, cols.copy(), vif_threshold)

    # ── Model 2: VIF-filtered model ────────────────────────────────────────
    valid_filtered = valid[selected + [target_col, DATE_COL]].dropna()
    train_f, test_f = time_split(valid_filtered, train_frac, DATE_COL)

    X_train_f = sm.add_constant(train_f[selected].values.astype(float), has_constant="add")
    y_train_f = train_f[target_col].values.astype(float)
    X_test_f  = sm.add_constant(test_f[selected].values.astype(float),  has_constant="add")
    y_test_f  = test_f[target_col].values.astype(float)

    try:
        filt_stats = _fit_ols(X_train_f, y_train_f, X_test_f, y_test_f)
    except Exception as exc:
        log.error("Filtered model OLS failed: %s", exc)
        filt_stats = {}

    # Coefficient table from filtered model
    coef_table = pd.DataFrame()
    if filt_stats and "sm_result" in filt_stats:
        coef_table = _build_coefficient_table(filt_stats["sm_result"], selected)

    vif_filtered = compute_vif(valid_filtered, selected) if selected else pd.DataFrame()

    log.info(
        "Multiple regression complete. "
        "Full: R²=%.4f adj-R²=%.4f test-R²=%s | "
        "Filtered (%d features): R²=%.4f adj-R²=%.4f test-R²=%s",
        full_stats["r_squared"], full_stats["adj_r_squared"],
        full_stats["r_squared_test"],
        len(selected),
        filt_stats.get("r_squared", float("nan")),
        filt_stats.get("adj_r_squared", float("nan")),
        filt_stats.get("r_squared_test"),
    )

    return {
        "full_model":         full_stats,
        "filtered_model":     filt_stats,
        "coefficient_table":  coef_table,
        "vif_table_full":     vif_full,
        "vif_table_filtered": vif_filtered,
        "features_full":      cols,
        "features_filtered":  selected,
        "features_removed":   removed,
        "train_cutoff":       train_cutoff,
        "n_obs_train":        n_obs_train,
        "n_obs_test":         n_obs_test,
    }


# ── Actual vs predicted ────────────────────────────────────────────────────────

def get_actual_vs_predicted(
    df:           pd.DataFrame,
    feature_cols: list[str],
    target_col:   str = TARGET_COL,
) -> pd.DataFrame:
    """
    Return actual and predicted values for scatter / residual plots.
    Fitted using the full (not split) dataset.
    """
    valid = df[feature_cols + [target_col, DATE_COL, "ticker"]].dropna().copy()
    if len(valid) < len(feature_cols) + 10:
        return pd.DataFrame()

    X = sm.add_constant(valid[feature_cols].values.astype(float), has_constant="add")
    y = valid[target_col].values.astype(float)

    result = sm.OLS(y, X).fit()
    out = valid[[DATE_COL, "ticker", target_col]].copy().reset_index(drop=True)
    out["predicted"] = result.fittedvalues
    out["residual"]  = result.resid
    return out


# ── Rolling multi-regression ───────────────────────────────────────────────────

def compute_rolling_r_squared(
    df:           pd.DataFrame,
    feature_cols: list[str],
    window:       int  = 36,
    target_col:   str  = TARGET_COL,
) -> pd.DataFrame:
    """
    Compute adjusted R² in rolling windows.

    Shows whether the model's explanatory power is stable over time
    or concentrated in specific regimes.

    Returns DataFrame with columns: end_date, r_squared, adj_r_squared, n_obs
    """
    valid = df[feature_cols + [target_col, DATE_COL]].dropna().sort_values(DATE_COL)
    dates = sorted(valid[DATE_COL].unique())

    if len(dates) < window + 1:
        return pd.DataFrame()

    rows = []
    for i in range(window, len(dates)):
        window_dates = dates[i - window: i]
        subset = valid[valid[DATE_COL].isin(window_dates)]

        if len(subset) < len(feature_cols) + 10:
            continue

        X = sm.add_constant(subset[feature_cols].values.astype(float), has_constant="add")
        y = subset[target_col].values.astype(float)

        try:
            res = sm.OLS(y, X).fit()
            rows.append({
                "end_date":     dates[i],
                "r_squared":    round(float(res.rsquared),     4),
                "adj_r_squared": round(float(res.rsquared_adj), 4),
                "n_obs":        int(res.nobs),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)
