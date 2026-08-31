"""
src/models/keyes_ols.py

Dedicated module for the Keyes (1972) OLS replication — kept separate so
Streamlit never loads a stale cached version of simple_regression.py.

train_keyes_ols_models  — train 5 single-variable OLS models
apply_keyes_ols_models  — apply trained models to current features
KEYES_VARIABLE_MAP      — mapping of Keyes X-labels to feature columns
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "future_12m_excess_return"
DATE_COL   = "feature_date"

KEYES_VARIABLE_MAP: dict[str, str] = {
    "X5":  "five_year_eps_growth",
    "X6":  "five_year_price_gain",
    "X8":  "current_pe_ratio",
    "X9":  "pe_vs_historical_median",
    "X12": "five_year_revenue_growth",
}


def train_keyes_ols_models(
    ft_df:      pd.DataFrame,
    target_col: str = TARGET_COL,
) -> dict[str, dict]:
    """
    Train 5 single-variable OLS models replicating Keyes (1972) Steps 10–11.
    Returns {feature_name: {keyes_key, intercept, coefficient, r_squared, p_value, n_obs}}
    """
    import statsmodels.api as sm

    models: dict[str, dict] = {}

    for keyes_key, feature in KEYES_VARIABLE_MAP.items():
        if feature not in ft_df.columns:
            continue
        valid = ft_df[[feature, target_col, DATE_COL]].dropna()
        if len(valid) < 20:
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
        except Exception as exc:
            log.warning("Keyes OLS failed for %s (%s): %s", keyes_key, feature, exc)

    log.info("Keyes OLS models trained: %d/%d variables", len(models), len(KEYES_VARIABLE_MAP))
    return models


def apply_keyes_ols_models(
    feats_df: pd.DataFrame,
    models:   dict[str, dict],
) -> pd.DataFrame:
    """
    Apply trained Keyes OLS models to current-month features.
    Returns DataFrame indexed by ticker with columns keyes_X5_pred … keyes_X12_pred.
    """
    if feats_df.empty or not models:
        return pd.DataFrame()

    preds: dict[str, pd.Series] = {}

    for feature, model in models.items():
        if feature not in feats_df.columns:
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
    log.info("Keyes OLS predictions computed for %d stocks (%d models)", len(result), len(preds))
    return result
