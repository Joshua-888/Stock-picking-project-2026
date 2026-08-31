"""
src/workflows/monthly_update.py

Step 18: Monthly update workflow — the master orchestration function.

This is the single function that runs the complete pipeline end-to-end.
Call it once per month (manually for now; schedulable via cron / GitHub Actions later).

WHAT IT DOES IN ORDER
---------------------
 1. Load config and initialise the database.
 2. Pull data from the configured provider (sample or Nasdaq Data Link).
 3. Validate data quality and log any issues.
 4. Save raw + cleaned data to the database.
 5. Recompute monthly features for any new dates (skip-existing flag).
 6. Backfill target variables now that new prices are available.
 7. Retrain all models (logistic, Ridge, Lasso, correlation).
    Uses a version cache so retraining is skipped if already done this month.
 8. Score every stock for the current prediction date.
 9. Save an immutable prediction snapshot to the database.
10. Update realised performance for snapshots whose horizon has elapsed.
11. Update the backtesting results.
12. Return a human-readable summary report.

IDEMPOTENCY
-----------
Every step is safe to run multiple times:
  • INSERT OR IGNORE prevents duplicate DB rows.
  • skip_existing=True in feature engineering skips computed dates.
  • Model cache key is based on the latest feature date.
  • Snapshots use INSERT OR IGNORE (never overwrite).

Running the update twice in the same month produces the same result as
running it once.

Usage
-----
    from src.workflows.monthly_update import run_monthly_update

    report = run_monthly_update()
    print(report["summary"])

CLI:
    python run_monthly_update.py
    python run_monthly_update.py --date 2024-01-31
    python run_monthly_update.py --provider nasdaq_data_link --force-retrain
"""

from __future__ import annotations

import time
import traceback
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger, setup_logging
from src.utils.dates import to_month_end

log = get_logger(__name__)


# ── Step helpers ───────────────────────────────────────────────────────────────

def _step(name: str, report: dict) -> None:
    """Log the start of a workflow step."""
    log.info("━━━ %s ━━━", name)
    report["steps_run"].append(name)


def _warn(msg: str, report: dict) -> None:
    log.warning(msg)
    report["warnings"].append(msg)


def _err(msg: str, report: dict) -> None:
    log.error(msg)
    report["errors"].append(msg)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def run_monthly_update(
    prediction_date:  Optional[str]  = None,
    data_provider:    Optional[str]  = None,
    force_retrain:    bool           = False,
    top_n_portfolio:  int            = 10,
    verbose:          bool           = True,
    include_gems:     bool           = True,
    gems_n_override:  Optional[int]  = None,
) -> dict:
    """
    Run the complete monthly update pipeline.

    Args:
        prediction_date: Month-end date to predict for.
                         Defaults to the most recent month-end with available data.
        data_provider:   Override the config data provider.
                         Options: "sample" | "nasdaq_data_link".
        force_retrain:   Force model retraining even if a cached version exists.
        top_n_portfolio: Number of top stocks for the backtest portfolio.
        verbose:         Print progress to console.

    Returns:
        Summary report dict with keys:
          prediction_date, provider, status, n_tickers_scored,
          n_strong_candidates, n_watchlist, n_snapshots_saved,
          n_realized_updated, model_version, total_time_seconds,
          warnings, errors, steps_run, score_table (DataFrame).
    """
    t_start = time.time()
    report: dict = {
        "prediction_date":   prediction_date,
        "provider":          data_provider,
        "status":            "running",
        "steps_run":         [],
        "warnings":          [],
        "errors":            [],
        "n_tickers_scored":  0,
        "n_strong_candidates": 0,
        "n_watchlist":       0,
        "n_snapshots_saved": 0,
        "n_realized_updated":0,
        "n_new_features":    0,
        "n_new_targets":     0,
        "model_version":     None,
        "total_time_seconds":0,
        "score_table":       pd.DataFrame(),
    }

    # ── STEP 1: Setup ──────────────────────────────────────────────────────────
    _step("1. Setup — config, logging, database", report)
    try:
        from src.utils.config import load_config
        from src.database.db import initialize_db
        from src.database.migrations import apply_migrations

        cfg = load_config()
        if verbose:
            setup_logging(
                level    = cfg.logging.level,
                log_file = cfg.logging.file,
            )

        provider = data_provider or cfg.data.provider
        report["provider"] = provider

        initialize_db()
        apply_migrations()
        log.info("Provider: %s | Prediction date: %s", provider, prediction_date or "auto")
    except Exception as exc:
        _err(f"Setup failed: {exc}", report)
        report["status"] = "failed"
        return _finalise(report, t_start)

    # ── STEP 2: Data ingestion (Core + Hidden Gems) ────────────────────────────
    _step("2. Data ingestion", report)
    try:
        from src.ingestion.provider_router import get_all_data
        from src.ingestion.hidden_gems import get_all_tickers, get_monthly_gems
        from src.database.db import (save_stocks, save_prices_clean,
                                      save_benchmark_prices, save_fundamentals_clean)

        # Determine the combined ticker universe (Core 30 + Hidden Gems).
        # First pass: use all pool tickers so we download data for as many as possible,
        # then substitute failed downloads with available alternatives.
        prov_date = prediction_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        if include_gems:
            combined_initial = get_all_tickers(prov_date, cfg)
        else:
            from src.ingestion.hidden_gems import get_core_tickers
            combined_initial = get_core_tickers(cfg)

        # Fetch data for the initial universe
        all_tickers_initial = [t["ticker"] for t in combined_initial]
        data = get_all_data(cfg, ticker_override=all_tickers_initial)

        # Determine which tickers actually have sufficient price data
        price_counts = data["prices_clean"].groupby("ticker").size()
        available_tickers = set(price_counts[price_counts >= 12].index)
        failed = set(all_tickers_initial) - available_tickers
        if failed:
            log.warning("Insufficient data for %d tickers — will substitute: %s",
                        len(failed), sorted(failed))

        # Re-build universe substituting failed tickers with available alternatives
        if include_gems:
            combined = get_all_tickers(prov_date, cfg, available_tickers=available_tickers)
            gem_count = sum(1 for t in combined if t.get("category") == "Hidden Gem")
            log.info("Universe: %d total tickers (%d core + %d hidden gems)",
                     len(combined), len(combined) - gem_count, gem_count)
        else:
            combined = [t for t in combined_initial if t["ticker"] in available_tickers]
            gem_count = 0
            log.info("Universe: %d core tickers only (gems disabled)", len(combined))

        report["n_gems_selected"] = gem_count
        report["n_core_tickers"]  = len(combined) - gem_count

        # Attach category metadata — only keep tickers that are in the confirmed universe
        cat_map = {t["ticker"]: t.get("category", "Core") for t in combined}
        data["stocks"] = data["stocks"][data["stocks"]["ticker"].isin(cat_map)].copy()
        data["stocks"]["category"] = data["stocks"]["ticker"].map(cat_map)

        save_stocks(data["stocks"])
        save_prices_clean(data["prices_clean"])
        save_benchmark_prices(data["benchmark_prices"])
        save_fundamentals_clean(data["fundamentals_clean"])

        log.info(
            "Data loaded: %d tickers  %d price rows  %d fundamental rows",
            data["stocks"]["ticker"].nunique(),
            len(data["prices_clean"]),
            len(data["fundamentals_clean"]),
        )
    except Exception as exc:
        _err(f"Data ingestion failed: {exc}", report)
        report["status"] = "failed"
        return _finalise(report, t_start)

    prices_df    = data["prices_clean"]
    benchmark_df = data["benchmark_prices"]
    funds_df     = data["fundamentals_clean"]
    stocks_df    = data["stocks"]

    # ── STEP 3: Determine prediction date ──────────────────────────────────────
    _step("3. Determine prediction date", report)
    if prediction_date is None:
        latest = prices_df["date"].max()
        prediction_date = to_month_end(latest)
    report["prediction_date"] = prediction_date
    log.info("Prediction date: %s", prediction_date)

    # ── STEP 4: Feature engineering ────────────────────────────────────────────
    _step("4. Feature engineering", report)
    try:
        from src.features.feature_engineering import compute_features_for_all_dates
        from src.database.db import save_monthly_features

        new_feats = compute_features_for_all_dates(
            prices_df, benchmark_df, funds_df, stocks_df,
            start_date=cfg.data.start_date,
            skip_existing=True,
        )
        if not new_feats.empty:
            save_monthly_features(new_feats.drop(columns=["sector"], errors="ignore"))
            report["n_new_features"] = len(new_feats)
            log.info("New feature rows computed: %d", len(new_feats))
        else:
            log.info("All feature dates already up to date.")
    except Exception as exc:
        _warn(f"Feature engineering error: {exc}", report)

    # ── STEP 4b: Backfill historical features for new tickers ─────────────────
    # Runs when Hidden Gems (or any ticker) enter the universe for the first time.
    # Without this, the model has no training data for new tickers → 50% P(Win).
    _step("4b. Backfill features for new tickers", report)
    try:
        from src.features.feature_engineering import backfill_new_tickers
        from src.database.queries import load_computed_feature_pairs

        existing_pairs   = load_computed_feature_pairs()
        tickers_in_db    = {t for _, t in existing_pairs}
        universe_tickers = list(stocks_df["ticker"].unique())
        new_tickers      = [t for t in universe_tickers if t not in tickers_in_db]

        if new_tickers:
            log.info("New tickers with no feature history: %s", new_tickers)
            backfill_df = backfill_new_tickers(
                new_tickers, prices_df, benchmark_df, funds_df, stocks_df,
                start_date=cfg.data.start_date,
            )
            if not backfill_df.empty:
                save_monthly_features(backfill_df.drop(columns=["sector"], errors="ignore"))
                report["n_new_features"] = report.get("n_new_features", 0) + len(backfill_df)
                log.info("Backfilled %d feature rows for %d new tickers.",
                         len(backfill_df), len(new_tickers))
                force_retrain = True  # Training data changed — must retrain
                log.info("force_retrain set to True — new ticker history requires model retraining.")
        else:
            log.info("No new tickers — backfill not needed.")
    except Exception as exc:
        _warn(f"Feature backfill error: {exc}\n{traceback.format_exc()}", report)

    # ── STEP 5: Target variable backfill ───────────────────────────────────────
    _step("5. Target variable backfill", report)
    try:
        from src.features.target_creation import compute_all_targets
        from src.database.db import save_targets

        targets = compute_all_targets(prices_df, benchmark_df, only_available=True)
        if not targets.empty:
            save_targets(targets)
            report["n_new_targets"] = len(targets)
            log.info("Targets saved: %d rows (winner_rate=%.1f%%)",
                     len(targets), targets["winner"].mean() * 100)
    except Exception as exc:
        _warn(f"Target backfill error: {exc}", report)

    # ── STEP 6: Model training ─────────────────────────────────────────────────
    _step("6. Model training (with cache)", report)
    logit_result = {}
    reg_result   = {}
    try:
        import hashlib, json, pickle
        from io import StringIO
        from src.database.queries import load_features_and_targets, load_model_training_runs
        from src.database.db import save_model_training_run
        from src.models.logistic_regression import run_logistic_regression
        from src.models.regularized_models import run_regularized_models

        ft = load_features_and_targets(start_date=cfg.data.start_date)
        if ft.empty or len(ft) < cfg.models.min_observations:
            _warn("Insufficient training data — skipping model training.", report)
        else:
            latest_feat  = ft["feature_date"].max()
            n_tickers    = ft["ticker"].nunique()
            cache_input  = f"{latest_feat}|{len(ft)}|{n_tickers}"
            cache_key    = hashlib.md5(cache_input.encode()).hexdigest()[:8]
            model_version = f"monthly_{prediction_date}_{cache_key}"
            report["model_version"] = model_version

            existing = load_model_training_runs()

            REQUIRED = {"logit_pickle", "reg_models_pickle"}
            already_cached = (
                not existing.empty and
                model_version in existing["model_version"].values and
                not force_retrain
            )
            if already_cached:
                row    = existing[existing["model_version"] == model_version].iloc[0]
                cached = json.loads(row["metrics_json"])
                if REQUIRED.issubset(cached.keys()):
                    logit_result = pickle.loads(bytes.fromhex(cached["logit_pickle"]))
                    reg_result   = pickle.loads(bytes.fromhex(cached["reg_models_pickle"]))
                    log.info("Models loaded from cache (version=%s)", model_version)
                else:
                    already_cached = False

            if not already_cached:
                t_train = time.time()
                logit_result = run_logistic_regression(ft)
                reg_result   = run_regularized_models(ft)
                t_elapsed    = time.time() - t_train
                log.info("Models trained in %.1fs", t_elapsed)

                # Store full model objects (sklearn_model, scaler etc.) so cache
                # hits in subsequent runs can make real predictions.
                # pickle handles sklearn objects natively — no stripping needed.
                save_model_training_run({
                    "model_version":       model_version,
                    "training_start_date": ft["feature_date"].min(),
                    "training_end_date":   latest_feat,
                    "model_type":          "monthly_bundle",
                    "features_used":       json.dumps(ft.columns.tolist()),
                    "metrics_json":        json.dumps({
                        "logit_pickle":      pickle.dumps(logit_result).hex(),
                        "reg_models_pickle": pickle.dumps({
                            "ridge": reg_result.get("ridge", {}),
                            "lasso": reg_result.get("lasso", {}),
                        }).hex(),
                    }),
                })
    except Exception as exc:
        _warn(f"Model training error: {exc}\n{traceback.format_exc()}", report)

    # ── STEP 7: Score current month ────────────────────────────────────────────
    _step("7. Score stocks for current month", report)
    scores_df  = pd.DataFrame()
    feats_idx  = pd.DataFrame()
    try:
        from src.features.feature_engineering import compute_features_for_date
        from src.models.logistic_regression import predict_probabilities
        from src.models.regularized_models import predict_regularized
        from src.models.simple_regression import train_keyes_ols_models, apply_keyes_ols_models
        from src.scoring.scoring_model import compute_scores
        from src.scoring.classifications import classify_stocks, compute_keyes_flag

        # Compute current-month features with FULL DB fundamentals history.
        # This gives highest DQ (84+ vs 68 with fresh-download only).
        # Save results back to monthly_features so future runs have the best quality.
        from src.database.queries import load_fundamentals_clean
        full_funds_df = load_fundamentals_clean()
        if full_funds_df.empty:
            full_funds_df = funds_df  # fallback to fresh download

        feats = compute_features_for_date(
            prediction_date, prices_df, benchmark_df, full_funds_df, stocks_df
        )

        # Persist these high-quality features to monthly_features (INSERT OR REPLACE)
        if not feats.empty:
            try:
                from src.database.db import save_monthly_features
                save_monthly_features(
                    feats.drop(columns=["sector"], errors="ignore"),
                )
                log.info("Saved high-quality features for %s to monthly_features (%d tickers)",
                         prediction_date, len(feats))
            except Exception as exc:
                _warn(f"Could not save features to monthly_features: {exc}", report)

        if feats.empty:
            _warn("No features for prediction date — skipping scoring.", report)
        else:
            feats_idx = feats.set_index("ticker") if "ticker" in feats.columns else feats

            # ── Winsorise features at training p1/p99 before model prediction ──
            # This prevents extreme feature values (e.g. AAPL P/B=58 vs median=3.7)
            # from causing Ridge extrapolation or logistic saturation.
            # Original feature values are preserved in feats_idx for display/storage.
            from src.features.transformations import FEATURE_COLS
            feats_for_pred = feats_idx.copy()
            if ft is not None and not ft.empty:
                for col in FEATURE_COLS:
                    if col in feats_for_pred.columns and col in ft.columns:
                        p01 = ft[col].quantile(0.01)
                        p99 = ft[col].quantile(0.99)
                        if pd.notna(p01) and pd.notna(p99) and p99 > p01:
                            feats_for_pred[col] = feats_for_pred[col].clip(p01, p99)
            else:
                feats_for_pred = feats_idx

            logit_p = pd.Series(dtype=float)
            ridge_p = pd.Series(dtype=float)

            if logit_result:
                try:
                    logit_p = predict_probabilities(feats_for_pred, logit_result)
                except Exception as exc:
                    _warn(f"Logistic prediction failed: {exc}", report)

            if reg_result.get("ridge"):
                try:
                    ridge_p = predict_regularized(feats_for_pred, reg_result, "ridge")
                except Exception as exc:
                    _warn(f"Ridge prediction failed: {exc}", report)

            # ── Keyes (1972) 5-variable OLS models ────────────────────────────
            # Train one OLS per Keyes variable on ALL historical data, then apply
            # to this month's features. A stock earns the Keyes Agreement flag
            # only when ALL 5 models predict positive excess return (Step 14).
            keyes_ols_preds = pd.DataFrame()
            try:
                keyes_models    = train_keyes_ols_models(ft)
                # Keyes OLS also uses winsorized features for consistency
                keyes_ols_preds = apply_keyes_ols_models(feats_for_pred, keyes_models)
                # Embed predictions into feats_idx so they are stored in features_json
                for col in keyes_ols_preds.columns:
                    feats_idx[col] = keyes_ols_preds[col]
            except Exception as exc:
                _warn(f"Keyes OLS training/application failed: {exc}", report)

            scores_df = compute_scores(feats_idx, logit_p, ridge_p)
            if "ticker" not in scores_df.columns:
                scores_df.insert(0, "ticker", feats_idx.index)

            meta_cols = ["ticker", "company_name", "sector", "industry"]
            if "category" in stocks_df.columns:
                meta_cols.append("category")
            meta = stocks_df[meta_cols].copy()
            scores_df = scores_df.merge(meta, on="ticker", how="left")
            if "category" not in scores_df.columns:
                scores_df["category"] = "Core"
            scores_df = classify_stocks(scores_df)
            scores_df["keyes_agreement_flag"] = compute_keyes_flag(
                scores_df, keyes_ols_preds=keyes_ols_preds
            )

            report["n_tickers_scored"]   = len(scores_df)
            report["n_strong_candidates"] = (scores_df["candidate_classification"] == "Strong candidate").sum()
            report["n_watchlist"]        = (scores_df["candidate_classification"] == "Watchlist candidate").sum()
            report["score_table"]        = scores_df

            log.info(
                "Scored %d tickers: strong=%d  watchlist=%d  keyes=%d",
                len(scores_df),
                report["n_strong_candidates"],
                report["n_watchlist"],
                scores_df["keyes_agreement_flag"].sum(),
            )
    except Exception as exc:
        _err(f"Scoring failed: {exc}\n{traceback.format_exc()}", report)

    # ── STEP 8: Save prediction snapshot ──────────────────────────────────────
    _step("8. Save prediction snapshot", report)
    if not scores_df.empty and not feats_idx.empty:
        try:
            from src.scoring.scoring_model import save_scoring_snapshots
            n = save_scoring_snapshots(
                scores_df, feats_idx, prediction_date,
                model_version=report.get("model_version", f"v{prediction_date}"),
                data_provider=provider,
            )
            report["n_snapshots_saved"] = n
            log.info("Saved %d prediction snapshots.", n)
        except Exception as exc:
            _warn(f"Snapshot save failed: {exc}", report)

    # ── STEP 9: Update realised performance ────────────────────────────────────
    _step("9. Update realised performance", report)
    try:
        from src.scoring.realized_performance import update_all_realized_performance
        n = update_all_realized_performance(prices_df, benchmark_df)
        report["n_realized_updated"] = n
        log.info("Updated %d realised performance rows.", n)
    except Exception as exc:
        _warn(f"Realised performance update failed: {exc}", report)

    # ── STEP 10: Update backtest ───────────────────────────────────────────────
    _step("10. Update backtesting results", report)
    try:
        from src.database.queries import load_monthly_features, load_stocks
        from src.backtesting.backtest import run_backtest

        feats_all = load_monthly_features(start_date=cfg.data.start_date)
        if not feats_all.empty:
            run_backtest(
                prices_df=prices_df, benchmark_df=benchmark_df,
                stocks_df=stocks_df, features_df=feats_all,
                top_n=top_n_portfolio, save_to_db=True,
            )
            log.info("Backtest results updated.")
    except Exception as exc:
        _warn(f"Backtest update failed: {exc}", report)

    # ── STEP 11: Finalise ──────────────────────────────────────────────────────
    report["status"] = "partial" if report["errors"] else "success"
    return _finalise(report, t_start)


def _finalise(report: dict, t_start: float) -> dict:
    """Add timing and print the summary."""
    report["total_time_seconds"] = round(time.time() - t_start, 1)

    lines = [
        "",
        "=" * 60,
        "  MONTHLY UPDATE SUMMARY",
        "=" * 60,
        f"  Status          : {report['status'].upper()}",
        f"  Prediction date : {report['prediction_date']}",
        f"  Provider        : {report['provider']}",
        f"  Tickers scored  : {report['n_tickers_scored']}",
        f"  Strong cands    : {report['n_strong_candidates']}",
        f"  Watchlist       : {report['n_watchlist']}",
        f"  Snapshots saved : {report['n_snapshots_saved']}",
        f"  Realised updated: {report['n_realized_updated']}",
        f"  New features    : {report['n_new_features']}",
        f"  New targets     : {report['n_new_targets']}",
        f"  Model version   : {report['model_version']}",
        f"  Total time      : {report['total_time_seconds']}s",
        f"  Steps run       : {len(report['steps_run'])}",
    ]
    if report["warnings"]:
        lines.append(f"  Warnings        : {len(report['warnings'])}")
        for w in report["warnings"]:
            lines.append(f"    - {w[:100]}")
    if report["errors"]:
        lines.append(f"  ERRORS          : {len(report['errors'])}")
        for e in report["errors"]:
            lines.append(f"    ! {e[:100]}")
    lines.append("=" * 60)

    for line in lines:
        log.info(line)

    return report
