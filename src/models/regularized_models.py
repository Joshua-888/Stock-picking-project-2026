"""
src/models/regularized_models.py

Step 13: Ridge and Lasso regularised regression.

WHY REGULARISATION?
--------------------
Steps 10 and 11 showed that OLS multiple regression overfits badly on 30 stocks:
  • Full OLS: in-sample R² = 0.15, out-of-sample R² = -0.45
  • This means OLS memorises the training data but fails on new data.

Regularisation fixes this by adding a penalty for large coefficients:

  Ridge (L2): minimises  Σ(y - Xβ)² + α·Σβ²
  Lasso (L1): minimises  Σ(y - Xβ)² + α·Σ|β|

The penalty α controls how strongly we shrink coefficients toward zero.
Larger α = more shrinkage = simpler model = less overfitting.

RIDGE vs LASSO
--------------
Ridge (L2 penalty):
  • Shrinks ALL coefficients toward zero but never to exactly zero.
  • Handles multicollinearity well — spreads the effect among correlated features.
  • Good when all features are expected to have some relevance.

Lasso (L1 penalty):
  • Shrinks SOME coefficients to exactly zero — built-in feature selection.
  • Produces a sparse model: only the most important features survive.
  • Better when many features are expected to be irrelevant noise.

In practice we run BOTH and compare:
  • Features that survive Lasso are the most robustly predictive.
  • Ridge coefficients show the relative importance of all features.

ALPHA SELECTION
---------------
We select the best α using cross-validation — specifically, time-series
cross-validation (no random shuffling) to prevent look-ahead bias.

sklearn's RidgeCV and LassoCV implement this automatically.

COEFFICIENT INTERPRETATION
--------------------------
Features are standardised before fitting (mean=0, std=1).
Coefficients represent: "a 1 standard-deviation increase in this feature
predicts X percentage points of additional excess return."

This makes Ridge and Lasso coefficients directly comparable to each other
and to the standardised OLS coefficients from Step 10.

Usage
-----
    from src.models.regularized_models import run_regularized_models

    result = run_regularized_models(features_and_targets_df)
    ridge = result["ridge"]
    lasso = result["lasso"]
    print(lasso["selected_features"])   # features with non-zero Lasso coefficient
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.features.transformations import FEATURE_COLS
from src.models.simple_regression import time_split
from src.models.multiple_regression import select_features_by_vif
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "future_12m_excess_return"
DATE_COL   = "feature_date"

# Alpha grid: logarithmically spaced.
# Upper bound = 2 (not 1000) — very high alpha collapses all coefficients to zero
# which is unhelpful for feature selection and interpretation.
ALPHA_GRID = np.logspace(-4, 2, 50)


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², MAE, RMSE, and hit rate (correct sign prediction)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae    = float(mean_absolute_error(y_true, y_pred))
    rmse   = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    hit    = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {
        "r_squared": round(r2,   4),
        "mae":       round(mae,  4),
        "rmse":      round(rmse, 4),
        "hit_rate":  round(hit,  4),
        "n_obs":     int(len(y_true)),
    }


# ── Time-series cross-validation ──────────────────────────────────────────────

def _tscv(n_splits: int = 5) -> TimeSeriesSplit:
    """Time-series cross-validator: train on past, test on future."""
    return TimeSeriesSplit(n_splits=n_splits)


# ── Ridge regression ──────────────────────────────────────────────────────────

def run_ridge(
    df:           pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col:   str                 = TARGET_COL,
    train_frac:   float               = 0.70,
    alphas:       np.ndarray          = ALPHA_GRID,
    n_cv_splits:  int                 = 5,
) -> dict:
    """
    Fit Ridge regression with cross-validated alpha selection.

    Cross-validation uses time-series splits (no data leakage).
    Features are standardised before fitting.

    Returns dict with:
        best_alpha         — selected regularisation strength
        coefficient_table  — all features with Ridge coefficients
        metrics_train      — R², MAE, RMSE, hit rate on training set
        metrics_test       — same on held-out test set
        alpha_path         — DataFrame of alpha vs CV score (for plot)
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    if not cols:
        return {}

    valid = df[cols + [target_col, DATE_COL]].dropna()
    if len(valid) < 30:
        return {}

    # Ridge handles multicollinearity via regularization — VIF filtering is
    # not needed and removes our highest-IC features (e.g. price_to_book).
    removed_vif: list = []
    valid = valid[cols + [target_col, DATE_COL]].dropna()

    train, test = time_split(valid, train_frac, DATE_COL)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[cols].values.astype(float))
    X_test  = scaler.transform(test[cols].values.astype(float))
    y_train = train[target_col].values.astype(float)
    y_test  = test[target_col].values.astype(float)

    # RidgeCV with time-series cross-validation
    cv_model = RidgeCV(alphas=alphas, cv=_tscv(n_cv_splits), scoring="r2")
    cv_model.fit(X_train, y_train)
    best_alpha = float(cv_model.alpha_)

    # Refit with the selected alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)

    # Coefficient table
    coef_df = pd.DataFrame({
        "feature":     cols,
        "coefficient": np.round(model.coef_, 6),
        "abs_coef":    np.abs(np.round(model.coef_, 6)),
    }).sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    coef_df["rank"] = range(1, len(coef_df) + 1)

    # CV alpha path (Ridge's CV scores per alpha)
    cv_scores = cv_model.cv_values_.mean(axis=0) if hasattr(cv_model, "cv_values_") else None
    alpha_path = pd.DataFrame({
        "alpha":    alphas,
        "cv_score": cv_scores if cv_scores is not None else [None] * len(alphas),
    })

    log.info(
        "Ridge: best_alpha=%.4f  train_R2=%.4f  test_R2=%.4f  features=%d",
        best_alpha,
        _regression_metrics(y_train, model.predict(X_train))["r_squared"],
        _regression_metrics(y_test,  model.predict(X_test)) ["r_squared"],
        len(cols),
    )

    return {
        "model":              model,
        "scaler":             scaler,
        "best_alpha":         best_alpha,
        "features_used":      cols,
        "features_removed_vif": removed_vif,
        "coefficient_table":  coef_df,
        "metrics_train":      _regression_metrics(y_train, model.predict(X_train)),
        "metrics_test":       _regression_metrics(y_test,  model.predict(X_test)),
        "alpha_path":         alpha_path,
        "train_cutoff":       sorted(train[DATE_COL].unique())[-1],
        "n_obs_train":        len(train),
        "n_obs_test":         len(test),
    }


# ── Lasso regression ──────────────────────────────────────────────────────────

def run_lasso(
    df:           pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col:   str                 = TARGET_COL,
    train_frac:   float               = 0.70,
    n_cv_splits:  int                 = 5,
    max_iter:     int                 = 5000,
) -> dict:
    """
    Fit Lasso regression with cross-validated alpha selection.

    Lasso automatically zeroes out irrelevant features — the surviving
    non-zero features are the model's best predictors.

    Returns dict with:
        best_alpha         — selected regularisation strength
        coefficient_table  — all features, zero-coefficient = excluded by Lasso
        selected_features  — features with non-zero coefficient
        n_selected         — number of selected features
        metrics_train, metrics_test
        alpha_path         — DataFrame of alpha vs CV MSE
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    if not cols:
        return {}

    valid = df[cols + [target_col, DATE_COL]].dropna()
    if len(valid) < 30:
        return {}

    # Lasso handles multicollinearity via L1 sparsity — no VIF filter needed.
    removed_vif: list = []
    valid = valid[cols + [target_col, DATE_COL]].dropna()

    train, test = time_split(valid, train_frac, DATE_COL)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[cols].values.astype(float))
    X_test  = scaler.transform(test[cols].values.astype(float))
    y_train = train[target_col].values.astype(float)
    y_test  = test[target_col].values.astype(float)

    # LassoCV with time-series cross-validation
    cv_model = LassoCV(
        alphas=ALPHA_GRID,
        cv=_tscv(n_cv_splits),
        max_iter=max_iter,
        random_state=42,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_model.fit(X_train, y_train)

    best_alpha = float(cv_model.alpha_)

    # Refit with selected alpha
    model = Lasso(alpha=best_alpha, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    # Coefficient table — sort by abs value, mark zeros
    coef_df = pd.DataFrame({
        "feature":     cols,
        "coefficient": np.round(model.coef_, 6),
        "selected":    model.coef_ != 0,
        "abs_coef":    np.abs(np.round(model.coef_, 6)),
    }).sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    coef_df["rank"] = range(1, len(coef_df) + 1)

    selected_features = coef_df.loc[coef_df["selected"], "feature"].tolist()

    # Alpha path
    mse_path = cv_model.mse_path_.mean(axis=1) if hasattr(cv_model, "mse_path_") else None
    alpha_path = pd.DataFrame({
        "alpha":    cv_model.alphas_ if hasattr(cv_model, "alphas_") else ALPHA_GRID,
        "cv_mse":   mse_path if mse_path is not None else [None] * len(ALPHA_GRID),
    })

    log.info(
        "Lasso: best_alpha=%.4f  selected=%d/%d  train_R2=%.4f  test_R2=%.4f",
        best_alpha,
        len(selected_features), len(cols),
        _regression_metrics(y_train, model.predict(X_train))["r_squared"],
        _regression_metrics(y_test,  model.predict(X_test)) ["r_squared"],
    )

    return {
        "model":               model,
        "scaler":              scaler,
        "best_alpha":          best_alpha,
        "features_used":       cols,
        "features_removed_vif": removed_vif,
        "coefficient_table":   coef_df,
        "selected_features":   selected_features,
        "n_selected":          len(selected_features),
        "metrics_train":       _regression_metrics(y_train, model.predict(X_train)),
        "metrics_test":        _regression_metrics(y_test,  model.predict(X_test)),
        "alpha_path":          alpha_path,
        "train_cutoff":        sorted(train[DATE_COL].unique())[-1],
        "n_obs_train":         len(train),
        "n_obs_test":          len(test),
    }


# ── Combined runner ────────────────────────────────────────────────────────────

def run_regularized_models(
    df:           pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col:   str                 = TARGET_COL,
    train_frac:   float               = 0.70,
) -> dict:
    """
    Run both Ridge and Lasso, return combined results.

    Also produces a coefficient comparison table showing how each feature's
    coefficient differs between OLS, Ridge, and Lasso.

    Returns dict with keys: ridge, lasso, comparison_table
    """
    log.info("Running regularised models (Ridge + Lasso)...")

    ridge = run_ridge(df, feature_cols, target_col, train_frac)
    lasso = run_lasso(df, feature_cols, target_col, train_frac)

    # Build comparison table
    comparison = _build_comparison_table(ridge, lasso)

    log.info(
        "Regularised models complete. "
        "Ridge test R2=%.4f  Lasso test R2=%.4f  Lasso selected=%d features",
        ridge.get("metrics_test", {}).get("r_squared", float("nan")),
        lasso.get("metrics_test", {}).get("r_squared", float("nan")),
        lasso.get("n_selected", 0),
    )

    return {
        "ridge":             ridge,
        "lasso":             lasso,
        "comparison_table":  comparison,
    }


def _build_comparison_table(ridge: dict, lasso: dict) -> pd.DataFrame:
    """
    Side-by-side coefficient comparison: Ridge vs Lasso.
    Sorted by |Ridge coefficient| descending.
    """
    if not ridge or not lasso:
        return pd.DataFrame()

    r_df = ridge["coefficient_table"][["feature", "coefficient"]].rename(
        columns={"coefficient": "ridge_coef"}
    )
    l_df = lasso["coefficient_table"][["feature", "coefficient", "selected"]].rename(
        columns={"coefficient": "lasso_coef"}
    )

    comp = r_df.merge(l_df, on="feature", how="outer").fillna({"lasso_coef": 0, "selected": False})
    comp["abs_ridge"] = comp["ridge_coef"].abs()
    comp = comp.sort_values("abs_ridge", ascending=False).drop(columns=["abs_ridge"])

    # Direction agreement
    def _agree(r, l):
        if l == 0: return "Lasso zeroed"
        if r > 0 and l > 0: return "Both positive"
        if r < 0 and l < 0: return "Both negative"
        return "Sign disagree"

    comp["agreement"] = comp.apply(lambda row: _agree(row.ridge_coef, row.lasso_coef), axis=1)
    return comp.reset_index(drop=True)


# ── Regularisation path (Lasso) ────────────────────────────────────────────────

def compute_lasso_path(
    df:           pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col:   str                 = TARGET_COL,
    n_alphas:     int                 = 40,
) -> pd.DataFrame:
    """
    Compute Lasso coefficient paths across a range of alpha values.

    Shows how features enter / leave the model as regularisation strength
    increases.  Used for the regularisation path plot in the dashboard.

    Returns:
        DataFrame: alpha (rows) × feature (columns), values = coefficients.
    """
    from sklearn.linear_model import lasso_path

    cols  = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    valid = df[cols + [target_col]].dropna()
    if len(valid) < 30 or not cols:
        return pd.DataFrame()

    valid   = valid[cols + [target_col]].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(valid[cols].values.astype(float))
    y = valid[target_col].values.astype(float)

    alphas = np.logspace(-3, 1, n_alphas)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas_out, coefs, _ = lasso_path(X, y, alphas=alphas, max_iter=5000)

    # coefs shape: (n_features, n_alphas) — transpose so rows = alphas
    path_df = pd.DataFrame(coefs.T, columns=cols)
    path_df.insert(0, "alpha", alphas_out)
    return path_df


# ── Predict with fitted model ─────────────────────────────────────────────────

def predict_regularized(
    df:          pd.DataFrame,
    result_dict: dict,
    model_key:   str = "ridge",
) -> pd.Series:
    """
    Apply a fitted Ridge or Lasso model to new observations.

    Used by the scoring model (Step 14).

    Args:
        df:          DataFrame with the same features as the fitted model.
        result_dict: Output from run_regularized_models().
        model_key:   'ridge' or 'lasso'.

    Returns:
        pd.Series of predicted excess returns.
    """
    m = result_dict.get(model_key, {})
    model  = m.get("model")
    scaler = m.get("scaler")
    cols   = m.get("features_used", [])

    if model is None or not cols:
        return pd.Series(dtype=float)

    X = df[cols].values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    # Impute NaN with 0 (= feature mean after standardisation)
    X = np.nan_to_num(X, nan=0.0)

    return pd.Series(model.predict(X), index=df.index, name=f"{model_key}_predicted")
