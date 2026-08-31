"""
src/models/logistic_regression.py

Step 12: Logistic regression — probability of outperforming the benchmark.

THE MODEL
---------
    P(winner = 1) = σ(α + β1·X1 + β2·X2 + … + βn·Xn)

where σ is the logistic (sigmoid) function: σ(z) = 1 / (1 + e^−z).

Unlike OLS which predicts a continuous return, logistic regression outputs a
PROBABILITY that a stock beats the benchmark over the next 12 months.

This probability is the core input for the scoring model in Step 14.

WHY TWO IMPLEMENTATIONS?
------------------------
We use both statsmodels and scikit-learn because each provides something
the other does not:

  statsmodels Logit
  → p-values for each coefficient (z-test)
  → confidence intervals for odds ratios
  → model fit statistics (log-likelihood, pseudo-R², AIC, BIC)
  → required for honest research reporting

  scikit-learn LogisticRegression
  → calibrated probabilities for ranking stocks
  → ROC AUC, precision-recall, confusion matrix
  → L2 regularisation (C parameter) to reduce overfitting
  → standardised feature scaling

ODDS RATIOS
-----------
An odds ratio (OR) is the exponentiated coefficient:

    OR_i = exp(β_i)

Interpretation:
  OR > 1  → feature increases the odds of beating the benchmark
  OR < 1  → feature decreases the odds of beating the benchmark
  OR = 1.5 → a 1-unit increase in the feature multiplies the odds of
             winning by 1.5 (i.e., 50% higher odds)

CALIBRATION
-----------
A probability model should be calibrated: when it says "70% probability
of winning", roughly 70% of those stocks should actually win.

We check this with a reliability diagram:
  Sort predictions into 10 equal-frequency buckets by predicted probability.
  For each bucket: compare mean predicted probability vs actual win rate.
  A perfectly calibrated model falls on the diagonal.

ROC AUC
-------
The ROC curve plots True Positive Rate vs False Positive Rate at every
possible classification threshold. AUC = area under the ROC curve.

  0.50 = random guessing (no skill)
  0.55 = modest predictive power
  0.60 = meaningful for financial markets
  0.70+ = strong (unusual in practice)

In financial return prediction, AUC of 0.55–0.62 is a realistic target.

IMPORTANT LIMITATIONS
---------------------
• Monthly returns are correlated across stocks and over time.
  Standard errors from statsmodels are likely understated.
• With 30 stocks, cross-sectional sample size per month is small (~30 obs).
  The model learns primarily from the time series dimension.
• A high in-sample AUC does not guarantee out-of-sample performance.
  The rolling AUC shows whether the model holds up across time.
• Winner classification ignores the MAGNITUDE of excess returns.
  A stock beating the benchmark by 0.01% counts the same as beating by 20%.

Usage
-----
    from src.models.logistic_regression import run_logistic_regression

    result = run_logistic_regression(features_and_targets_df)
    proba_df = result["predictions"]    # ticker, date, probability, actual winner
    auc      = result["roc_auc_test"]   # out-of-sample AUC
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, log_loss, brier_score_loss,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from src.features.transformations import FEATURE_COLS
from src.models.simple_regression import time_split
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "winner"
EXCESS_COL = "future_12m_excess_return"
DATE_COL   = "feature_date"


# ── Classification metrics helper ─────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute a complete set of classification metrics.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Predicted binary labels (threshold = 0.5).
        y_prob: Predicted probabilities for class 1.

    Returns:
        dict with accuracy, precision, recall, F1, ROC AUC, log-loss, Brier score.
    """
    has_both_classes = len(np.unique(y_true)) == 2

    metrics: dict = {
        "n_obs":       int(len(y_true)),
        "n_winners":   int(y_true.sum()),
        "actual_win_rate": round(float(y_true.mean()), 4),
        "pred_win_rate":   round(float((y_prob >= 0.5).mean()), 4),
        "accuracy":    round(float(accuracy_score(y_true, y_pred)),        4),
        "precision":   round(float(precision_score(y_true, y_pred,
                                    zero_division=0)),                    4),
        "recall":      round(float(recall_score(y_true, y_pred,
                                    zero_division=0)),                    4),
        "f1":          round(float(f1_score(y_true, y_pred,
                                    zero_division=0)),                    4),
        "roc_auc":     round(float(roc_auc_score(y_true, y_prob)), 4)
                       if has_both_classes else None,
        "log_loss":    round(float(log_loss(y_true, y_prob)), 4)
                       if has_both_classes else None,
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }
    return metrics


# ── Calibration curve ─────────────────────────────────────────────────────────

def compute_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a reliability diagram: predicted probability vs actual win rate.

    Stocks are sorted into n_bins equal-frequency buckets by predicted
    probability.  Each row shows the mean prediction and actual outcome
    for that bucket.

    Returns:
        DataFrame with columns:
          bin, mean_predicted_prob, actual_win_rate, n_obs, calibration_error
    """
    df = pd.DataFrame({"prob": y_prob, "actual": y_true})
    df["bin"] = pd.qcut(df["prob"], n_bins, labels=False, duplicates="drop") + 1

    rows = []
    for b, grp in df.groupby("bin"):
        rows.append({
            "bin":                  int(b),
            "mean_predicted_prob":  round(float(grp["prob"].mean()),   4),
            "actual_win_rate":      round(float(grp["actual"].mean()), 4),
            "n_obs":                len(grp),
            "calibration_error":    round(abs(float(grp["prob"].mean()) -
                                              float(grp["actual"].mean())), 4),
        })

    cal_df = pd.DataFrame(rows)
    if not cal_df.empty:
        cal_df.attrs["mean_calibration_error"] = round(
            float(cal_df["calibration_error"].mean()), 4
        )
    return cal_df


# ── ROC curve ─────────────────────────────────────────────────────────────────

def compute_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Return fpr, tpr, thresholds for drawing the ROC curve.
    """
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame()
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


# ── Statsmodels Logit (for inference) ─────────────────────────────────────────

def _run_statsmodels_logit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Fit statsmodels Logit and extract coefficient table with odds ratios.

    Returns empty dict if the model fails to converge.
    """
    X_sm = sm.add_constant(X_train, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = sm.Logit(y_train, X_sm).fit(
                disp=False, maxiter=200, method="bfgs"
            )
        except Exception as exc:
            log.warning("Statsmodels Logit failed to converge: %s", exc)
            return {}

    # Coefficient table
    ci_raw  = result.conf_int(alpha=0.05)
    ci_arr  = ci_raw.values if hasattr(ci_raw, "values") else np.asarray(ci_raw)
    params  = np.asarray(result.params)
    pvalues = np.asarray(result.pvalues)
    bse     = np.asarray(result.bse)
    tvalues = np.asarray(result.tvalues)   # statsmodels stores z-stats as tvalues

    def _safe_exp(x: float) -> float | None:
        try:
            v = math.exp(float(x))
            return v if not (math.isnan(v) or math.isinf(v)) else None
        except (OverflowError, ValueError):
            return None

    rows = []
    for i, name in enumerate(["intercept"] + feature_names):
        coef  = float(params[i])
        pval  = float(pvalues[i])
        se    = float(bse[i])
        zstat = float(tvalues[i])
        ci_lo = float(ci_arr[i, 0])
        ci_hi = float(ci_arr[i, 1])

        rows.append({
            "feature":     name,
            "coefficient": round(coef,  6),
            "std_error":   round(se,    6),
            "z_stat":      round(zstat, 3),
            "p_value":     round(pval,  6),
            "ci_lower":    round(ci_lo, 6),
            "ci_upper":    round(ci_hi, 6),
            "odds_ratio":  _safe_exp(coef),
            "or_ci_lower": _safe_exp(ci_lo),
            "or_ci_upper": _safe_exp(ci_hi),
            "significant": bool(pval < 0.05),
        })

    coef_df = pd.DataFrame(rows)
    coef_df = coef_df[coef_df["feature"] != "intercept"].reset_index(drop=True)

    # Pseudo-R² (McFadden): 1 - (log-likelihood / null log-likelihood)
    ll      = float(result.llf)
    ll_null = float(result.llnull)
    pseudo_r2 = round(1 - ll / ll_null, 4) if ll_null != 0 else None

    return {
        "coefficient_table": coef_df,
        "log_likelihood":    round(ll, 3),
        "aic":               round(float(result.aic), 2),
        "bic":               round(float(result.bic), 2),
        "pseudo_r2":         pseudo_r2,
        "converged":         bool(result.mle_retvals.get("converged", True)),
    }


# ── Main logistic regression runner ──────────────────────────────────────────

def run_logistic_regression(
    df:              pd.DataFrame,
    feature_cols:    Optional[list[str]] = None,
    target_col:      str                 = TARGET_COL,
    train_frac:      float               = 0.70,
    C:               float               = 1.0,
    scale_features:  bool                = True,
) -> dict:
    """
    Run logistic regression: statsmodels for inference, sklearn for probabilities.

    Args:
        df:             Merged features + targets DataFrame.
        feature_cols:   Features to use (default: all FEATURE_COLS present in df).
        target_col:     Binary target column (winner = 0 or 1).
        train_frac:     Fraction of dates for training.
        C:              Inverse regularisation strength (sklearn). Larger C = less
                        regularisation.  Default 1.0 is a mild L2 penalty.
        scale_features: Standardise features before fitting (recommended for logistic).

    Returns dict with keys:
        coefficient_table  — log-odds + odds ratios + p-values (statsmodels)
        metrics_train      — accuracy, precision, recall, F1, AUC on training set
        metrics_test       — same metrics on held-out test set
        confusion_matrix   — 2×2 array [[TN,FP],[FN,TP]] on test set
        roc_curve_data     — fpr, tpr, threshold DataFrame for ROC plot
        roc_auc_test       — scalar AUC on test set
        calibration_data   — reliability diagram DataFrame
        predictions        — full DataFrame: date, ticker, actual, predicted, probability
        features_used      — list of feature names actually used
        n_obs_train        — training observations
        n_obs_test         — test observations
        train_cutoff       — last training date
        pseudo_r2          — McFadden pseudo-R²
        aic, bic           — statsmodels model selection criteria
        rolling_auc        — rolling AUC DataFrame (36-month windows)
    """
    cols = [c for c in (feature_cols or FEATURE_COLS) if c in df.columns]
    if not cols:
        log.warning("No valid feature columns for logistic regression.")
        return {}

    # Keep only rows with winner = 0 or 1 (drop NaN targets)
    needed = cols + [target_col, DATE_COL] + (
        ["ticker"] if "ticker" in df.columns else []
    )
    valid = df[needed].dropna(subset=[target_col]).copy()
    valid = valid[valid[target_col].isin([0, 1, 0.0, 1.0])].dropna(subset=cols)
    valid[target_col] = valid[target_col].astype(int)

    if len(valid) < 50:
        log.warning("Too few observations (%d) for logistic regression.", len(valid))
        return {}

    if valid[target_col].nunique() < 2:
        log.warning("Only one class present in target — cannot fit logistic regression.")
        return {}

    # ── VIF filtering ─────────────────────────────────────────────────────────
    # Threshold raised to 20 (was 10) so high-IC features like price_to_book
    # (VIF ~12-15) are retained. Only extreme collinearity (VIF > 20) is removed.
    from src.models.multiple_regression import select_features_by_vif
    cols, vif_removed = select_features_by_vif(valid, cols, vif_threshold=20.0)
    if vif_removed:
        log.info("Logistic VIF filter removed %d features: %s",
                 len(vif_removed), [n for n, _ in vif_removed])

    valid = valid[cols + [target_col, DATE_COL] +
                  (["ticker"] if "ticker" in valid.columns else [])].dropna(subset=cols)

    if len(valid) < 50 or valid[target_col].nunique() < 2:
        log.warning("Too few observations after VIF filter.")
        return {}

    log.info(
        "Logistic regression: %d features (after VIF), %d obs (%.1f%% winners)",
        len(cols), len(valid), valid[target_col].mean() * 100,
    )

    # ── Time-based train/test split ───────────────────────────────────────────
    train, test   = time_split(valid, train_frac, DATE_COL)
    train_cutoff  = sorted(train[DATE_COL].unique())[-1]

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_raw = train[cols].values.astype(float)
    X_test_raw  = test[cols].values.astype(float)

    if scale_features:
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    y_train = train[target_col].values.astype(int)
    y_test  = test[target_col].values.astype(int)

    # ── Statsmodels Logit (scaled features for convergence) ──────────────────
    # Coefficients represent per-1-SD-change in each feature.
    sm_result = _run_statsmodels_logit(X_train, y_train, cols)

    # ── Scikit-learn LogisticRegression ───────────────────────────────────────
    # penalty="l2" is deprecated in sklearn 1.8+ — omit it (L2 remains the default)
    clf = LogisticRegression(
        C=C, solver="lbfgs",
        max_iter=500, random_state=42,
    )
    clf.fit(X_train, y_train)

    # Training metrics
    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_train_pred = clf.predict(X_train)
    metrics_train = compute_classification_metrics(y_train, y_train_pred, y_train_prob)

    # Test metrics
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    y_test_pred = clf.predict(X_test)
    metrics_test = compute_classification_metrics(y_test, y_test_pred, y_test_prob)

    # Confusion matrix (test set)
    cm = confusion_matrix(y_test, y_test_pred)

    # ROC curve (test set)
    roc_df  = compute_roc_curve_data(y_test, y_test_prob)
    roc_auc = metrics_test.get("roc_auc")

    # Calibration — use full valid set for more stable calibration
    X_all = scaler.transform(valid[cols].values.astype(float)) if scale_features \
            else valid[cols].values.astype(float)
    y_all     = valid[target_col].values.astype(int)
    y_all_prob = clf.predict_proba(X_all)[:, 1]
    cal_df = compute_calibration_data(y_all, y_all_prob)

    # Predictions DataFrame
    ticker_col = valid["ticker"].values if "ticker" in valid.columns else np.arange(len(valid))
    pred_df = pd.DataFrame({
        DATE_COL:    valid[DATE_COL].values,
        "ticker":    ticker_col,
        "actual_winner":   y_all,
        "predicted_winner": (y_all_prob >= 0.5).astype(int),
        "probability":      np.round(y_all_prob, 4),
        "correct":         (y_all == (y_all_prob >= 0.5).astype(int)).astype(int),
    })

    # Rolling AUC — use the VIF-filtered feature set
    rolling_auc = compute_rolling_auc(valid, cols, scale_features=scale_features)

    log.info(
        "Logistic regression complete. "
        "Train AUC=%.4f  Test AUC=%s  pseudo-R2=%s",
        metrics_train.get("roc_auc", 0),
        roc_auc,
        sm_result.get("pseudo_r2"),
    )

    return {
        "coefficient_table": sm_result.get("coefficient_table", pd.DataFrame()),
        "pseudo_r2":         sm_result.get("pseudo_r2"),
        "log_likelihood":    sm_result.get("log_likelihood"),
        "aic":               sm_result.get("aic"),
        "bic":               sm_result.get("bic"),
        "converged":         sm_result.get("converged", False),
        "metrics_train":     metrics_train,
        "metrics_test":      metrics_test,
        "confusion_matrix":  cm,
        "roc_curve_data":    roc_df,
        "roc_auc_test":      roc_auc,
        "calibration_data":  cal_df,
        "predictions":       pred_df,
        "features_used":     cols,
        "n_obs_train":       len(train),
        "n_obs_test":        len(test),
        "train_cutoff":      train_cutoff,
        "sklearn_model":     clf,
        "scaler":            scaler,
        "rolling_auc":       rolling_auc,
    }


# ── Get probabilities for new observations ─────────────────────────────────────

def predict_probabilities(
    df:          pd.DataFrame,
    result_dict: dict,
) -> pd.Series:
    """
    Apply a fitted logistic regression model to new observations.

    Used by the scoring model (Step 14) to generate current-month probabilities.

    Args:
        df:          DataFrame with the same feature columns as the fitted model.
        result_dict: Output from run_logistic_regression().

    Returns:
        pd.Series of probabilities indexed by df's index.
    """
    clf    = result_dict.get("sklearn_model")
    scaler = result_dict.get("scaler")
    cols   = result_dict.get("features_used", [])

    if clf is None or not cols:
        return pd.Series(dtype=float)

    X = df[cols].values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    # Impute NaN with 0 (= feature mean after standardisation)
    X = np.nan_to_num(X, nan=0.0)

    proba = clf.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="probability_of_outperformance")


# ── Rolling AUC ───────────────────────────────────────────────────────────────

def compute_rolling_auc(
    df:            pd.DataFrame,
    feature_cols:  list[str],
    window:        int  = 36,
    target_col:    str  = TARGET_COL,
    scale_features: bool = True,
) -> pd.DataFrame:
    """
    Compute ROC AUC in rolling 36-month windows.

    Each window uses a 70/30 train/test split internally so the AUC is
    genuinely out-of-sample.

    Returns DataFrame: end_date, auc, n_train, n_test
    """
    valid = df[feature_cols + [target_col, DATE_COL]].dropna().copy()
    valid = valid[valid[target_col].isin([0, 1, 0.0, 1.0])].copy()
    valid[target_col] = valid[target_col].astype(int)
    dates = sorted(valid[DATE_COL].unique())

    if len(dates) < window + 1:
        return pd.DataFrame()

    rows = []
    for i in range(window, len(dates)):
        win_dates = dates[i - window: i]
        subset    = valid[valid[DATE_COL].isin(win_dates)].copy()

        if subset[target_col].nunique() < 2 or len(subset) < 30:
            continue

        # Inner 70/30 split within this window
        cutoff_idx  = int(len(win_dates) * 0.70)
        cutoff_date = win_dates[cutoff_idx]
        tr = subset[subset[DATE_COL] <  cutoff_date]
        te = subset[subset[DATE_COL] >= cutoff_date]

        if te[target_col].nunique() < 2 or len(tr) < 15 or len(te) < 5:
            continue

        try:
            sc = StandardScaler()
            Xtr = sc.fit_transform(tr[feature_cols].values.astype(float)) \
                  if scale_features else tr[feature_cols].values.astype(float)
            Xte = sc.transform(te[feature_cols].values.astype(float)) \
                  if scale_features else te[feature_cols].values.astype(float)

            clf = LogisticRegression(C=1.0, max_iter=300, random_state=42, solver="lbfgs")  # L2 default
            clf.fit(Xtr, tr[target_col].values.astype(int))
            proba = clf.predict_proba(Xte)[:, 1]
            auc   = roc_auc_score(te[target_col].values, proba)

            rows.append({
                "end_date": dates[i],
                "auc":      round(float(auc), 4),
                "n_train":  len(tr),
                "n_test":   len(te),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)
