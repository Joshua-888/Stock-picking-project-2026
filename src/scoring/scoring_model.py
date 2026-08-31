"""
src/scoring/scoring_model.py

Step 14: Scoring model — combines all model outputs into a single ranked score.

DESIGN PHILOSOPHY
-----------------
The scoring model never produces a "buy" or "sell" signal.
It produces a probability-weighted, risk-adjusted research ranking.

Every component score is computed on a 0–100 scale.
The final_score is a weighted average of component scores.
Weights are configurable in config.yaml (scoring.weights).

DEFAULT WEIGHTS
---------------
    35%  probability_of_outperformance  (logistic regression)
    25%  expected_excess_return_score   (Ridge regression, rank-normalised)
    15%  model_agreement_score          (fraction of models agree on direction)
    10%  risk_score                     (inverse of volatility + leverage)
    10%  data_quality_score             (from feature engineering)
     5%  valuation_score                (sector-relative cheapness)

COMPONENT SCORES — HOW EACH IS CALCULATED
------------------------------------------
probability_of_outperformance_score
    Directly from logistic regression: P(stock beats benchmark) × 100.
    A probability of 0.65 → score of 65.
    If logistic unavailable: neutral 50.

expected_excess_return_score
    Ridge-predicted excess return, rank-normalised within the cross-section.
    Best-predicted stock gets 100; worst gets 0.
    Rank normalisation prevents a single stock with huge predicted return
    from dominating the score calculation.

model_agreement_score
    Fraction of the following that point positively:
      • Logistic: P(winner) ≥ 0.50
      • Ridge:    predicted_excess_return ≥ 0
      • Lasso:    predicted_excess_return ≥ 0  (if available)
    Score = fraction × 100.  Range: 0–100.

risk_score
    Composite of inverse-normalised risk metrics:
      volatility_12m, downside_volatility_12m, beta (if available)
    High risk → low score.  Low risk → high score.
    Missing risk features → neutral 50.

data_quality_score
    Directly from feature engineering (already 0–100).

valuation_score
    Based on sector_relative_pe and pe_vs_historical_median.
    Cheaper-than-sector → higher score.
    Computed as (100 - percentile_rank_of_pe_within_universe).

KEYES-STYLE AGREEMENT FLAG
--------------------------
A stock receives this flag (1) if ALL of:
  • Logistic probability ≥ 0.50
  • Ridge predicted excess return ≥ 0
  • data_quality_score ≥ 60

This modernises Keyes's original principle: only select a stock when
MULTIPLE independent statistical methods agree it should outperform.

Usage
-----
    from src.scoring.scoring_model import run_scoring_pipeline

    scores = run_scoring_pipeline(
        feature_date = "2024-01-31",
        prices_df    = prices_clean_df,
        benchmark_df = benchmark_prices_df,
        funds_df     = fundamentals_clean_df,
        stocks_df    = stocks_df,
    )
    # scores is a DataFrame sorted by final_score descending
    # with all component scores and candidate_classification
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.features.transformations import FEATURE_COLS
from src.utils.logging import get_logger

log = get_logger(__name__)

TARGET_COL = "future_12m_excess_return"

# ── Normalisation helpers ──────────────────────────────────────────────────────

def _rank_score(series: pd.Series) -> pd.Series:
    """
    Convert a Series to 0–100 percentile ranks (higher value → higher score).
    NaN values receive a neutral score of 50.
    """
    ranked = series.rank(pct=True, na_option="keep") * 100
    return ranked.fillna(50.0)


def _inverse_rank_score(series: pd.Series) -> pd.Series:
    """
    Convert a Series to 0–100 percentile ranks (LOWER value → higher score).
    Used for risk metrics: lower volatility → higher risk score.
    """
    return 100.0 - _rank_score(series)


def _clamp(series: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


# ── Component score functions ──────────────────────────────────────────────────

def _probability_score(logit_proba: Optional[pd.Series], n: int) -> pd.Series:
    """
    0–100 score from logistic regression probability.
    If unavailable, return neutral 50 for all stocks.
    """
    if logit_proba is None or logit_proba.empty:
        return pd.Series(50.0, index=range(n), name="probability_score")
    return _clamp(logit_proba * 100, 0, 100).rename("probability_score")


def _excess_return_score(ridge_pred: Optional[pd.Series], n: int) -> pd.Series:
    """
    0–100 rank-normalised score from Ridge predicted excess return.
    Rank normalisation: best predicted return = 100, worst = 0.
    """
    if ridge_pred is None or ridge_pred.empty:
        return pd.Series(50.0, index=range(n), name="excess_return_score")
    return _rank_score(ridge_pred).rename("excess_return_score")


def _model_agreement_score(
    logit_proba:  Optional[pd.Series],
    ridge_pred:   Optional[pd.Series],
    lasso_pred:   Optional[pd.Series],
    n:            int,
    index,
) -> pd.Series:
    """
    0–100 score = fraction of model signals pointing positively × 100.
    Each model contributes one vote.
    """
    votes = pd.DataFrame(index=index)

    if logit_proba is not None and not logit_proba.empty:
        votes["logit"]  = (logit_proba >= 0.50).astype(float)
    if ridge_pred is not None and not ridge_pred.empty:
        votes["ridge"]  = (ridge_pred  >= 0.0).astype(float)
    if lasso_pred is not None and not lasso_pred.empty:
        votes["lasso"]  = (lasso_pred  >= 0.0).astype(float)

    if votes.empty:
        return pd.Series(50.0, index=index, name="model_agreement_score")

    return (votes.mean(axis=1) * 100).rename("model_agreement_score")


def _risk_score(features_df: pd.DataFrame) -> pd.Series:
    """
    0–100 score inversely proportional to risk.
    Lower volatility / lower leverage → higher score.
    """
    risk_components = []

    for col in ["volatility_12m", "downside_volatility_12m", "beta"]:
        if col in features_df.columns:
            s = features_df[col]
            if s.notna().sum() >= 3:
                risk_components.append(_rank_score(s))   # high value = high risk

    if not risk_components:
        return pd.Series(50.0, index=features_df.index, name="risk_score")

    risk_composite = pd.concat(risk_components, axis=1).mean(axis=1)
    return _clamp(100.0 - risk_composite, 0, 100).rename("risk_score")


def _valuation_score(features_df: pd.DataFrame) -> pd.Series:
    """
    0–100 score: cheaper relative valuation → higher score.
    Uses sector_relative_pe and pe_vs_historical_median.
    More negative sector_relative_pe → cheaper → higher score.
    """
    val_components = []

    for col in ["sector_relative_pe", "pe_vs_historical_median"]:
        if col in features_df.columns:
            s = features_df[col]
            if s.notna().sum() >= 3:
                # These are "expensive = positive" features, so invert
                val_components.append(_inverse_rank_score(s))

    # current_pe_ratio: lower PE = cheaper = better
    if "current_pe_ratio" in features_df.columns:
        s = features_df["current_pe_ratio"]
        if s.notna().sum() >= 3:
            val_components.append(_inverse_rank_score(s))

    if not val_components:
        return pd.Series(50.0, index=features_df.index, name="valuation_score")

    return _clamp(
        pd.concat(val_components, axis=1).mean(axis=1), 0, 100
    ).rename("valuation_score")


def _momentum_score(features_df: pd.DataFrame) -> pd.Series:
    """
    0–100 score from price momentum signals.
    Higher 12-month momentum → higher score.
    """
    mom_components = []
    for col in ["twelve_month_momentum", "six_month_momentum", "return_3m"]:
        if col in features_df.columns:
            s = features_df[col]
            if s.notna().sum() >= 3:
                mom_components.append(_rank_score(s))

    if not mom_components:
        return pd.Series(50.0, index=features_df.index, name="momentum_score")

    return _clamp(
        pd.concat(mom_components, axis=1).mean(axis=1), 0, 100
    ).rename("momentum_score")


def _fundamental_score(features_df: pd.DataFrame) -> pd.Series:
    """
    0–100 score from fundamental quality signals.
    Higher profitability / lower leverage → higher score.
    """
    qual_pos = []   # higher = better
    qual_neg = []   # lower = better

    for col in ["five_year_eps_growth", "five_year_revenue_growth", "roe", "roic",
                "free_cash_flow_yield", "gross_margin", "operating_margin"]:
        if col in features_df.columns and features_df[col].notna().sum() >= 3:
            qual_pos.append(_rank_score(features_df[col]))

    for col in ["debt_to_equity"]:
        if col in features_df.columns and features_df[col].notna().sum() >= 3:
            qual_neg.append(_inverse_rank_score(features_df[col]))

    all_components = qual_pos + qual_neg
    if not all_components:
        return pd.Series(50.0, index=features_df.index, name="fundamental_score")

    return _clamp(
        pd.concat(all_components, axis=1).mean(axis=1), 0, 100
    ).rename("fundamental_score")


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_scores(
    features_df:  pd.DataFrame,
    logit_proba:  Optional[pd.Series] = None,
    ridge_pred:   Optional[pd.Series] = None,
    lasso_pred:   Optional[pd.Series] = None,
    weights:      Optional[dict]       = None,
) -> pd.DataFrame:
    """
    Compute all component scores and final_score for a set of stocks.

    Args:
        features_df:  Feature DataFrame — one row per ticker.
                      Must contain columns: ticker, data_quality_score,
                      plus any feature columns available.
        logit_proba:  Series of predicted win probabilities (index = ticker).
        ridge_pred:   Series of predicted excess returns from Ridge (index = ticker).
        lasso_pred:   Series of predicted excess returns from Lasso (optional).
        weights:      Dict overriding the default component weights.

    Returns:
        DataFrame with one row per ticker containing all scores, final_score,
        and candidate_classification.  Sorted by final_score descending.
    """
    if features_df.empty:
        return pd.DataFrame()

    # Align all prediction series to the features_df index
    idx = features_df.index

    def _align(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None or s.empty:
            return None
        # Try aligning by ticker if features has a ticker column
        if "ticker" in features_df.columns and s.index.dtype == object:
            aligned = features_df["ticker"].map(s)
            return aligned if aligned.notna().sum() > 0 else s.reindex(idx)
        return s.reindex(idx)

    logit_aligned = _align(logit_proba)
    ridge_aligned = _align(ridge_pred)
    lasso_aligned = _align(lasso_pred)

    n = len(features_df)

    # ── Compute all component scores ─────────────────────────────────────────
    prob_score  = _probability_score(logit_aligned, n).values
    ret_score   = _excess_return_score(ridge_aligned, n).values
    agree_score = _model_agreement_score(
        logit_aligned, ridge_aligned, lasso_aligned, n, idx
    ).values
    risk_sc     = _risk_score(features_df).values
    dq_sc       = features_df["data_quality_score"].fillna(50.0).values
    val_sc      = _valuation_score(features_df).values
    mom_sc      = _momentum_score(features_df).values
    fund_sc     = _fundamental_score(features_df).values

    # ── Weights ───────────────────────────────────────────────────────────────
    w = {
        "probability_of_outperformance": 0.35,
        "expected_excess_return":        0.25,
        "model_agreement":               0.15,
        "risk_score":                    0.10,
        "data_quality":                  0.10,
        "valuation":                     0.05,
    }
    if weights:
        w.update(weights)

    # ── Final score ───────────────────────────────────────────────────────────
    final = (
        w["probability_of_outperformance"] * prob_score  +
        w["expected_excess_return"]        * ret_score   +
        w["model_agreement"]               * agree_score +
        w["risk_score"]                    * risk_sc     +
        w["data_quality"]                  * dq_sc       +
        w["valuation"]                     * val_sc
    )

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    out = features_df[["ticker"]].copy() if "ticker" in features_df.columns \
          else pd.DataFrame(index=idx)
    out = out.reset_index(drop=True)

    out["probability_of_outperformance"] = (
        (logit_aligned.values if logit_aligned is not None else np.full(n, 0.50))
    )
    out["predicted_12m_excess_return"]   = (
        ridge_aligned.values if ridge_aligned is not None else np.full(n, 0.0)
    )
    out["probability_score"]    = np.round(prob_score,  2)
    out["excess_return_score"]  = np.round(ret_score,   2)
    out["model_agreement_score"] = np.round(agree_score, 2)
    out["risk_score"]           = np.round(risk_sc,     2)
    out["data_quality_score"]   = np.round(dq_sc,       2)
    out["valuation_score"]      = np.round(val_sc,      2)
    out["momentum_score"]       = np.round(mom_sc,      2)
    out["fundamental_score"]    = np.round(fund_sc,     2)
    out["final_score"]          = np.round(final,       2)

    return out.sort_values("final_score", ascending=False).reset_index(drop=True)


# ── Full scoring pipeline ─────────────────────────────────────────────────────

def run_scoring_pipeline(
    feature_date: str,
    prices_df:    pd.DataFrame,
    benchmark_df: pd.DataFrame,
    funds_df:     pd.DataFrame,
    stocks_df:    pd.DataFrame,
    train_start:  str  = "2015-01-31",
    weights:      Optional[dict] = None,
) -> pd.DataFrame:
    """
    End-to-end scoring for a single feature_date.

    1. Compute features for feature_date.
    2. Train models on historical data up to feature_date.
    3. Apply models to generate predictions.
    4. Compute component scores and final_score.
    5. Assign candidate_classification and Keyes flag.

    Returns a scored and classified DataFrame sorted by final_score.

    Args:
        feature_date: The prediction month-end date (YYYY-MM-DD).
        prices_df:    prices_clean DataFrame.
        benchmark_df: benchmark_prices DataFrame.
        funds_df:     fundamentals_clean DataFrame.
        stocks_df:    stocks DataFrame.
        train_start:  Earliest date to use for model training.
        weights:      Optional weight overrides.
    """
    from src.features.feature_engineering import compute_features_for_date
    from src.features.target_creation import compute_all_targets
    from src.models.logistic_regression import run_logistic_regression, predict_probabilities
    from src.models.regularized_models import run_regularized_models, predict_regularized
    from src.scoring.classifications import classify_stocks, compute_keyes_flag

    log.info("Running scoring pipeline for %s", feature_date)

    # ── Step 1: Compute current features (use full DB fundamentals history) ──
    # funds_df may only have the latest quarters; load full history from DB
    # so 5-year growth features (eps, revenue) can be computed correctly.
    try:
        from src.database.queries import load_fundamentals_clean
        full_funds = load_fundamentals_clean()
        if not full_funds.empty:
            funds_df = full_funds
    except Exception:
        pass  # fall back to passed funds_df

    feats = compute_features_for_date(
        feature_date, prices_df, benchmark_df, funds_df, stocks_df
    )
    if feats.empty:
        log.warning("No features computed for %s", feature_date)
        return pd.DataFrame()

    # ── Step 2: Build training dataset (history before feature_date) ──────────
    hist_prices = prices_df[prices_df["date"] < feature_date]
    hist_bench  = benchmark_df[benchmark_df["date"] < feature_date]

    from src.database.queries import load_features_and_targets
    ft_df = load_features_and_targets(
        start_date=train_start,
        end_date=feature_date,
    )

    if ft_df.empty or len(ft_df) < 50:
        log.warning("Insufficient training data for %s", feature_date)
        return _score_without_models(feats, stocks_df, weights)

    # ── Step 3: Train models ──────────────────────────────────────────────────
    from src.models.simple_regression import train_keyes_ols_models, apply_keyes_ols_models
    logit_result  = run_logistic_regression(ft_df)
    reg_result    = run_regularized_models(ft_df)
    keyes_models  = train_keyes_ols_models(ft_df)

    # ── Step 4: Generate predictions for current features ────────────────────
    logit_proba = pd.Series(dtype=float)
    ridge_pred  = pd.Series(dtype=float)
    lasso_pred  = pd.Series(dtype=float)

    feats_for_pred = feats.set_index("ticker") if "ticker" in feats.columns else feats

    if logit_result:
        try:
            logit_proba = predict_probabilities(feats_for_pred, logit_result)
            logit_proba.index = feats_for_pred.index
        except Exception as exc:
            log.warning("Logistic prediction failed: %s", exc)

    if reg_result.get("ridge"):
        try:
            ridge_pred = predict_regularized(feats_for_pred, reg_result, "ridge")
        except Exception as exc:
            log.warning("Ridge prediction failed: %s", exc)

    if reg_result.get("lasso") and reg_result["lasso"].get("n_selected", 0) > 0:
        try:
            lasso_pred = predict_regularized(feats_for_pred, reg_result, "lasso")
        except Exception as exc:
            log.warning("Lasso prediction failed: %s", exc)

    # Keyes (1972) 5-variable OLS predictions
    keyes_ols_preds = apply_keyes_ols_models(feats_for_pred, keyes_models)
    for col in keyes_ols_preds.columns:
        feats_for_pred[col] = keyes_ols_preds[col]

    # ── Step 5: Compute scores ────────────────────────────────────────────────
    feats_indexed = feats_for_pred
    scores_df = compute_scores(
        feats_indexed, logit_proba, ridge_pred, lasso_pred, weights
    )

    # Re-attach ticker from index
    if "ticker" not in scores_df.columns:
        scores_df.insert(0, "ticker", feats_indexed.index)

    # Join sector / company info
    meta = stocks_df[["ticker", "company_name", "sector", "industry"]].copy()
    scores_df = scores_df.merge(meta, on="ticker", how="left")

    # ── Step 6: Classify ──────────────────────────────────────────────────────
    scores_df = classify_stocks(scores_df)
    scores_df["keyes_agreement_flag"] = compute_keyes_flag(
        scores_df, keyes_ols_preds=keyes_ols_preds
    )
    scores_df["feature_date"]         = feature_date

    log.info(
        "Scoring complete for %s: %d stocks  strong=%d  watchlist=%d",
        feature_date,
        len(scores_df),
        (scores_df["candidate_classification"] == "Strong candidate").sum(),
        (scores_df["candidate_classification"] == "Watchlist candidate").sum(),
    )

    return scores_df


def _score_without_models(
    feats:    pd.DataFrame,
    stocks_df: pd.DataFrame,
    weights:  Optional[dict],
) -> pd.DataFrame:
    """Score using only feature-based signals (no regression models)."""
    feats_indexed = feats.set_index("ticker") if "ticker" in feats.columns else feats
    scores_df = compute_scores(feats_indexed, weights=weights)
    if "ticker" not in scores_df.columns:
        scores_df.insert(0, "ticker", feats_indexed.index)
    meta = stocks_df[["ticker", "company_name", "sector", "industry"]].copy()
    scores_df = scores_df.merge(meta, on="ticker", how="left")

    from src.scoring.classifications import classify_stocks, compute_keyes_flag
    scores_df = classify_stocks(scores_df)
    scores_df["keyes_agreement_flag"] = compute_keyes_flag(scores_df)
    return scores_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — PREDICTION SNAPSHOT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def _generate_warnings(row: pd.Series) -> list[str]:
    """
    Generate human-readable warnings for a single stock's score row.
    Warnings are stored in the snapshot and shown on the dashboard.
    """
    warnings: list[str] = []

    dq = row.get("data_quality_score", 100)
    if dq < 40:
        warnings.append(f"Critical: data quality score is very low ({dq:.0f}/100)")
    elif dq < 60:
        warnings.append(f"Low data quality ({dq:.0f}/100) — prediction less reliable")

    prob = row.get("probability_of_outperformance", 0.5)
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        warnings.append("Logistic regression probability unavailable — using neutral 50%")

    agree = row.get("model_agreement_score", 50)
    if agree < 40:
        warnings.append(f"Low model agreement ({agree:.0f}/100) — models disagree on direction")

    risk = row.get("risk_score", 50)
    if risk < 30:
        warnings.append("High risk profile — elevated volatility or leverage")

    return warnings


def build_snapshot_records(
    scores_df:       pd.DataFrame,
    features_df:     pd.DataFrame,
    prediction_date: str,
    model_version:   str,
    data_provider:   str = "sample",
) -> list[dict]:
    """
    Build a list of snapshot dicts ready for database insertion.

    Each dict corresponds to one row in the prediction_snapshots table.
    The features are serialised to JSON at the time of prediction —
    once written to the DB they must never be changed.

    Args:
        scores_df:        Output from compute_scores() + classify_stocks().
        features_df:      Features DataFrame for this prediction_date (indexed by ticker).
        prediction_date:  The prediction month-end date (YYYY-MM-DD).
        model_version:    Version string, e.g. "v2024-01-31".
        data_provider:    Name of the data source used.

    Returns:
        List of dicts — one per ticker.
    """
    import json
    import uuid
    from src.features.transformations import FEATURE_COLS, standardize_features

    # Standardise features for model-input reference (stored alongside raw)
    feats_for_std = features_df.copy()
    std_df = standardize_features(feats_for_std)

    records = []
    for _, row in scores_df.iterrows():
        ticker = row.get("ticker")
        if not ticker:
            continue

        # Raw feature values + Keyes OLS predictions for this ticker
        if ticker in features_df.index:
            feat_row = features_df.loc[ticker]
            raw_feats = {
                col: (None if (isinstance(v, float) and math.isnan(v)) else v)
                for col, v in feat_row[FEATURE_COLS].items()
                if col in feat_row.index
            }
            # Append Keyes OLS predictions so they are visible in Stock Detail
            keyes_preds = {
                col: (None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 6))
                for col, v in feat_row.items()
                if str(col).startswith("keyes_") and col in feat_row.index
            }
            raw_feats.update(keyes_preds)

            std_feats = {
                col: (None if (isinstance(v, float) and math.isnan(v)) else v)
                for col, v in std_df.loc[ticker, [c for c in FEATURE_COLS if c in std_df.columns]].items()
            } if ticker in std_df.index else {}
        else:
            raw_feats, std_feats = {}, {}

        warnings = _generate_warnings(row)

        records.append({
            "snapshot_id":                   str(uuid.uuid4()),
            "prediction_date":               prediction_date,
            "ticker":                        ticker,
            "company_name":                  row.get("company_name"),
            "sector":                        row.get("sector"),
            "industry":                      row.get("industry"),
            "model_version":                 model_version,
            "data_provider":                 data_provider,
            "data_update_timestamp":         pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_json":                 json.dumps(raw_feats),
            "features_std_json":             json.dumps(std_feats),
            "predicted_12m_return":          row.get("predicted_12m_excess_return"),   # best proxy available
            "predicted_12m_excess_return":   row.get("predicted_12m_excess_return"),
            "probability_of_outperformance": row.get("probability_of_outperformance"),
            "model_agreement_score":         row.get("model_agreement_score"),
            "risk_score":                    row.get("risk_score"),
            "valuation_score":               row.get("valuation_score"),
            "momentum_score":                row.get("momentum_score"),
            "fundamental_score":             row.get("fundamental_score"),
            "data_quality_score":            row.get("data_quality_score"),
            "final_score":                   row.get("final_score"),
            "candidate_classification":      row.get("candidate_classification"),
            "keyes_agreement_flag":          int(row.get("keyes_agreement_flag", 0)),
            "category":                      row.get("category", "Core"),
            "warnings_json":                 json.dumps(warnings),
            "notes":                         None,
        })

    return records


def save_scoring_snapshots(
    scores_df:       pd.DataFrame,
    features_df:     pd.DataFrame,
    prediction_date: str,
    model_version:   Optional[str] = None,
    data_provider:   str           = "sample",
) -> int:
    """
    Save one immutable prediction snapshot per stock to the database.

    This is the most important function in the pipeline for honest evaluation.
    Once a snapshot is saved it is NEVER updated — it represents exactly
    what the model predicted at the time.

    Safe to call multiple times: INSERT OR IGNORE prevents duplicates.

    Args:
        scores_df:       Scored and classified DataFrame from compute_scores().
        features_df:     Feature DataFrame for this date (indexed by ticker).
        prediction_date: The prediction month-end date.
        model_version:   Auto-generated from prediction_date if not provided.
        data_provider:   Name of the data source.

    Returns:
        Number of new snapshots saved (0 if all already exist).
    """
    from src.database.db import save_prediction_snapshots

    if model_version is None:
        model_version = f"v{prediction_date}"

    records = build_snapshot_records(
        scores_df, features_df, prediction_date, model_version, data_provider
    )

    if not records:
        log.warning("No snapshot records to save for %s", prediction_date)
        return 0

    snapshot_df = pd.DataFrame(records)
    n = save_prediction_snapshots(snapshot_df)

    log.info(
        "Saved %d prediction snapshots for %s (model_version=%s)",
        n, prediction_date, model_version,
    )
    return n
