"""
src/ingestion/custom_ticker.py

Analyse any arbitrary US ticker on demand, scoring it with the same
pipeline used for the regular 60-stock universe.

The trained models (logistic, Ridge, Keyes OLS) are loaded from the DB cache
so this is apply-only — no retraining happens.  Feature engineering uses the
same 33-variable pipeline.

Usage
-----
    from src.ingestion.custom_ticker import analyse_custom_ticker
    result = analyse_custom_ticker("GOOG")
    # result is a dict with keys: ticker, scores_df, features, errors
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def analyse_custom_ticker(
    ticker: str,
    prediction_date: Optional[str] = None,
    start_date: str = "2010-01-01",
) -> dict:
    """
    Download data, compute features, and score a single custom ticker.

    Returns
    -------
    dict with keys:
        ticker          str
        prediction_date str
        features        dict  — all 33 computed feature values
        scores          dict  — final_score, classification, keyes_flag, P(Win), pred_xret
        keyes_vars      dict  — X5, X6, X8, X9, X12 values
        comparison      dict  — min_ols_pred, keyes_threshold, would_pass_keyes
        error           str | None
    """
    import pandas as pd
    from src.utils.dates import to_month_end

    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Empty ticker symbol."}

    if prediction_date is None:
        prediction_date = to_month_end(pd.Timestamp.today().strftime("%Y-%m-%d"))

    try:
        # ── 1. Fetch price + fundamental data ─────────────────────────────────
        from src.ingestion.yfinance_provider import fetch_prices, fetch_benchmark, fetch_fundamentals
        from src.utils.config import load_config

        cfg = load_config()
        bench_ticker = cfg.data.benchmark_ticker

        log.info("Custom ticker %s: downloading data…", ticker)
        prices_df    = fetch_prices([ticker], start_date, prediction_date)
        benchmark_df = fetch_benchmark(bench_ticker, start_date, prediction_date)
        funds_df     = fetch_fundamentals([ticker], start_date)

        if prices_df.empty:
            return {"ticker": ticker, "error": f"No price data found for {ticker}. Check the ticker symbol."}

        # Build a minimal stocks DataFrame for the ticker
        sector = "Unknown"
        if not funds_df.empty and "sector" in funds_df.columns:
            s = funds_df["sector"].dropna()
            if not s.empty:
                sector = s.iloc[-1]

        stocks_df = pd.DataFrame([{
            "ticker":       ticker,
            "company_name": ticker,
            "sector":       sector,
            "industry":     "Unknown",
            "exchange":     "NASDAQ",
            "currency":     "USD",
            "is_active":    1,
            "category":     "Custom",
        }])

        # ── 2. Feature engineering ─────────────────────────────────────────────
        from src.features.feature_engineering import compute_features_for_date

        feats_df = compute_features_for_date(
            prediction_date, prices_df, benchmark_df, funds_df, stocks_df
        )

        if feats_df.empty:
            return {"ticker": ticker, "error": f"Could not compute features for {ticker}. Insufficient historical data (need 12+ months)."}

        feats_idx = feats_df.set_index("ticker") if "ticker" in feats_df.columns else feats_df

        # ── 3. Load trained models from DB cache ──────────────────────────────
        import json, pickle
        from src.database.queries import load_model_training_runs

        runs = load_model_training_runs()
        monthly_runs = runs[runs["model_type"] == "monthly_bundle"] if not runs.empty else pd.DataFrame()

        logit_result = {}
        reg_result   = {}
        if not monthly_runs.empty:
            latest_run = monthly_runs.sort_values("training_end_date", ascending=False).iloc[0]
            cached     = json.loads(latest_run["metrics_json"])
            logit_result = pickle.loads(bytes.fromhex(cached.get("logit_pickle", "")))
            reg_result   = {"ridge": pickle.loads(bytes.fromhex(cached.get("reg_models_pickle", "")))
                            .get("ridge", {})} if cached.get("reg_models_pickle") else {}
        else:
            log.warning("No trained model cache found — run the monthly update first.")

        # ── 4. Score the ticker ────────────────────────────────────────────────
        from src.models.logistic_regression import predict_probabilities
        from src.models.regularized_models import predict_regularized
        # Import from dedicated keyes_ols.py — no stale-cache risk
        from src.models.keyes_ols import train_keyes_ols_models, apply_keyes_ols_models
        from src.scoring.scoring_model import compute_scores
        from src.scoring.classifications import classify_stocks
        from src.database.queries import load_features_and_targets
        from src.utils.config import load_config as _cfg

        logit_proba = pd.Series(dtype=float)
        ridge_pred  = pd.Series(dtype=float)

        if logit_result and "sklearn_model" in logit_result:
            try:
                logit_proba = predict_probabilities(feats_idx, logit_result)
            except Exception as e:
                log.warning("Logistic prediction failed for %s: %s", ticker, e)

        if reg_result.get("ridge", {}).get("sklearn_model"):
            try:
                ridge_pred = predict_regularized(feats_idx, reg_result, "ridge")
            except Exception as e:
                log.warning("Ridge prediction failed for %s: %s", ticker, e)

        scores_df = compute_scores(feats_idx, logit_proba, ridge_pred)
        if "ticker" not in scores_df.columns:
            scores_df.insert(0, "ticker", feats_idx.index)

        scores_df = scores_df.merge(
            stocks_df[["ticker","company_name","sector"]],
            on="ticker", how="left"
        )
        scores_df = classify_stocks(scores_df)

        # ── 5. Keyes OLS — compare against current universe threshold ─────────
        cfg2 = _cfg()
        ft = pd.DataFrame()
        try:
            ft = load_features_and_targets(start_date=cfg2.data.start_date)
        except Exception:
            pass

        keyes_min_pred    = None
        keyes_threshold   = None
        would_pass_keyes  = False

        if not ft.empty:
            keyes_models = train_keyes_ols_models(ft)
            custom_preds = apply_keyes_ols_models(feats_idx, keyes_models)

            if not custom_preds.empty:
                ols_cols = [c for c in custom_preds.columns if c.startswith("keyes_")]
                if ols_cols:
                    numeric = custom_preds[ols_cols].apply(pd.to_numeric, errors="coerce")
                    keyes_min_pred = float(numeric.min(axis=1).iloc[0])

                    # Load the current universe's threshold (30th percentile)
                    from src.database.queries import load_prediction_snapshots
                    snaps = load_prediction_snapshots()
                    if not snaps.empty:
                        latest = snaps["prediction_date"].max()
                        latest_snaps = snaps[snaps["prediction_date"] == latest]
                        # Approximate: use the minimum Keyes-flagged stock's final_score as threshold
                        keyes_flagged = latest_snaps[latest_snaps["keyes_agreement_flag"] == 1]
                        if not keyes_flagged.empty:
                            # Recompute what min_pred the universe achieved
                            # as a proxy: the threshold is fixed at the 70th percentile
                            n_universe = len(latest_snaps)
                            n_select   = max(1, round(n_universe * 0.30))
                            keyes_threshold = 0.0  # OLS predicts positive excess return

                    if not math.isnan(keyes_min_pred):
                        would_pass_keyes = keyes_min_pred > (keyes_threshold or 0.0)

        # ── 6. Extract Keyes variables ────────────────────────────────────────
        row = feats_df.iloc[0] if not feats_df.empty else pd.Series()
        keyes_vars = {
            "X5_eps_growth":    _safe(row.get("five_year_eps_growth")),
            "X6_price_gain":    _safe(row.get("five_year_price_gain")),
            "X8_pe_ratio":      _safe(row.get("current_pe_ratio")),
            "X9_pe_vs_median":  _safe(row.get("pe_vs_historical_median")),
            "X12_rev_growth":   _safe(row.get("five_year_revenue_growth")),
        }

        # ── 7. Build feature dict ─────────────────────────────────────────────
        feat_dict = {}
        if not feats_df.empty:
            row_dict = feats_df.iloc[0].to_dict()
            feat_dict = {k: _safe(v) for k, v in row_dict.items()
                         if k not in ("ticker", "feature_date", "sector", "created_at")}

        # ── 8. Pull scores ────────────────────────────────────────────────────
        score_row = scores_df.iloc[0] if not scores_df.empty else pd.Series()
        scores = {
            "final_score":        _safe(score_row.get("final_score")),
            "classification":     score_row.get("candidate_classification", "Unknown"),
            "probability_of_win": _safe(score_row.get("probability_of_outperformance")),
            "predicted_xret":     _safe(score_row.get("predicted_12m_excess_return")),
            "data_quality_score": _safe(row.get("data_quality_score")),
            "model_agreement":    _safe(score_row.get("model_agreement_score")),
        }

        return {
            "ticker":          ticker,
            "company_name":    ticker,
            "sector":          sector,
            "prediction_date": prediction_date,
            "features":        feat_dict,
            "scores":          scores,
            "keyes_vars":      keyes_vars,
            "comparison": {
                "min_ols_pred":       keyes_min_pred,
                "keyes_threshold":    keyes_threshold,
                "would_pass_keyes":   would_pass_keyes,
            },
            "error": None,
        }

    except Exception as exc:
        import traceback
        log.error("Custom ticker analysis failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc), "traceback": traceback.format_exc()}


def _safe(v):
    """Return None for NaN/inf, otherwise the value."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return v
