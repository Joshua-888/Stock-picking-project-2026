"""
src/scoring/realized_performance.py

Step 16: Realised performance tracking.

PURPOSE
-------
For every prediction snapshot where the 12-month horizon has elapsed,
this module looks up what actually happened to the stock and computes:

  • Actual 12-month stock return
  • Actual 12-month benchmark return
  • Actual 12-month excess return
  • winner_actual  = 1 if stock beat benchmark, 0 otherwise
  • winner_predicted = 1 if P(outperform) ≥ 0.50 at prediction time
  • prediction_correct = 1 if winner_actual == winner_predicted
  • return_error = predicted_excess_return − actual_excess_return

WHY THIS MATTERS
----------------
Backtesting reconstructs performance after the fact, which is subject
to look-ahead bias and in-sample optimism.

Realised performance tracking records predictions BEFORE outcomes are
known (via the snapshot system in Step 15), then fills in the actual
outcomes after the fact.

This is the ONLY honest measure of whether the model actually works.

PROBABILITY BUCKETS
-------------------
Stocks are grouped by their predicted probability into buckets:
    < 0.40     "Low"
    0.40–0.50  "Below neutral"
    0.50–0.60  "Slight positive"
    0.60–0.70  "Moderate positive"
    0.70–0.80  "High positive"
    ≥ 0.80     "Very high positive"

A calibrated model should show actual win rates that match these buckets.
E.g. stocks in the "0.60–0.70" bucket should win ~65% of the time.

KEYES CONNECTION
----------------
The original Keyes +10% rule was an early version of this:
only select stocks where all models agreed on ≥ 10% appreciation,
then measure actual outcomes.

Our realised tracking modernises this: we record the full probability
and all component scores, then measure what actually happened.

Usage
-----
    from src.scoring.realized_performance import update_all_realized_performance

    n = update_all_realized_performance(prices_df, benchmark_df)
    print(f"Updated {n} realised performance rows")
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.features.target_creation import (
    is_target_available, get_target_date, _lookup_price
)
from src.utils.logging import get_logger

log = get_logger(__name__)

HORIZON_MONTHS = 12


# ── Probability bucket helper ──────────────────────────────────────────────────

def probability_bucket(prob: Optional[float]) -> str:
    """Assign a probability bucket label for grouping / calibration analysis."""
    if prob is None or math.isnan(prob):
        return "Unknown"
    if prob < 0.40: return "< 0.40 (Low)"
    if prob < 0.50: return "0.40–0.50"
    if prob < 0.60: return "0.50–0.60"
    if prob < 0.70: return "0.60–0.70"
    if prob < 0.80: return "0.70–0.80"
    return "≥ 0.80 (High)"


# ── Per-date realised computation ─────────────────────────────────────────────

def compute_realized_for_date(
    prediction_date: str,
    snapshots_df:    pd.DataFrame,
    prices_df:       pd.DataFrame,
    benchmark_df:    pd.DataFrame,
    horizon_months:  int = HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Compute realised performance for all snapshots on a given prediction_date.

    Args:
        prediction_date: The original prediction month-end date.
        snapshots_df:    All prediction snapshots (all dates, all tickers).
        prices_df:       prices_clean DataFrame.
        benchmark_df:    benchmark_prices DataFrame.
        horizon_months:  Forward horizon (default 12).

    Returns:
        DataFrame with one row per ticker, ready for realized_performance table.
        Returns empty DataFrame if horizon has not yet elapsed.
    """
    if not is_target_available(prediction_date, horizon_months):
        return pd.DataFrame()

    evaluation_date = get_target_date(prediction_date, horizon_months)

    # Snapshots for this prediction date
    snaps = snapshots_df[snapshots_df["prediction_date"] == prediction_date].copy()
    if snaps.empty:
        return pd.DataFrame()

    # Benchmark return over the horizon
    bench_idx    = benchmark_df.set_index("date")["adjusted_close"]
    bench_start  = _lookup_price(bench_idx, prediction_date)
    bench_end    = _lookup_price(bench_idx, evaluation_date)

    if bench_start and bench_end and bench_start > 0:
        bench_return_actual = bench_end / bench_start - 1.0
    else:
        bench_return_actual = None
        log.warning("Benchmark prices unavailable for %s → %s",
                    prediction_date, evaluation_date)

    # Per-ticker realised returns
    rows: list[dict] = []
    for _, snap in snaps.iterrows():
        ticker = snap["ticker"]

        ticker_prices = prices_df[prices_df["ticker"] == ticker].set_index("date")["adjusted_close"]
        price_start   = _lookup_price(ticker_prices, prediction_date)
        price_end     = _lookup_price(ticker_prices, evaluation_date)

        if price_start and price_end and price_start > 0:
            stock_return_actual  = price_end / price_start - 1.0
        else:
            stock_return_actual  = None

        if stock_return_actual is not None and bench_return_actual is not None:
            excess_return_actual = stock_return_actual - bench_return_actual
            winner_actual        = int(excess_return_actual > 0)
        else:
            excess_return_actual = None
            winner_actual        = None
        # Note: r_excess (derived from rounded values) is set later in the row dict

        # Predicted winner: P(outperform) ≥ 0.50
        prob_pred = snap.get("probability_of_outperformance")
        pred_xret = snap.get("predicted_12m_excess_return")

        if prob_pred is not None and not math.isnan(float(prob_pred or 0)):
            winner_predicted = int(float(prob_pred) >= 0.50)
        else:
            winner_predicted = None

        if winner_actual is not None and winner_predicted is not None:
            prediction_correct = int(winner_actual == winner_predicted)
        else:
            prediction_correct = None

        # Return error: predicted minus actual
        if pred_xret is not None and excess_return_actual is not None:
            try:
                return_error = float(pred_xret) - excess_return_actual
            except (TypeError, ValueError):
                return_error = None
        else:
            return_error = None

        # Round stock and benchmark first; derive excess FROM those rounded values
        # so that stored_excess == stored_stock - stored_bench exactly.
        r_stock = round(stock_return_actual, 8) if stock_return_actual is not None else None
        r_bench = round(bench_return_actual, 8) if bench_return_actual is not None else None
        r_excess = (r_stock - r_bench) if (r_stock is not None and r_bench is not None) else None

        rows.append({
            "prediction_date":               prediction_date,
            "evaluation_date":               evaluation_date,
            "ticker":                        ticker,
            "realized_12m_stock_return":     r_stock,
            "realized_12m_benchmark_return": r_bench,
            "realized_12m_excess_return":    r_excess,
            "winner_actual":                 winner_actual,
            "winner_predicted":              winner_predicted,
            "prediction_correct":            prediction_correct,
            "return_error":                  round(return_error, 8) if return_error is not None else None,
            "probability_bucket":            probability_bucket(float(prob_pred) if prob_pred is not None else None),
            "model_version":                 snap.get("model_version"),
        })

    df = pd.DataFrame(rows)
    n_valid = df["winner_actual"].notna().sum()
    log.info(
        "Realised performance for %s: %d tickers, %d with valid outcomes",
        prediction_date, len(df), n_valid,
    )
    return df


# ── Full update ────────────────────────────────────────────────────────────────

def update_all_realized_performance(
    prices_df:      pd.DataFrame,
    benchmark_df:   pd.DataFrame,
    horizon_months: int = HORIZON_MONTHS,
) -> int:
    """
    Check all prediction snapshots and fill in realised performance for
    any prediction_date where the horizon has elapsed and no record yet exists.

    Safe to call on every monthly update — already-computed rows are skipped
    (unique constraint on prediction_date + ticker).

    Returns:
        Total number of new realised performance rows saved.
    """
    from src.database.queries import load_prediction_snapshots
    from src.database.db import save_realized_performance

    all_snaps    = load_prediction_snapshots()
    if all_snaps.empty:
        log.info("No prediction snapshots found.")
        return 0

    snap_dates   = sorted(all_snaps["prediction_date"].unique())
    eligible     = [d for d in snap_dates if is_target_available(d, horizon_months)]

    if not eligible:
        log.info("No prediction dates have elapsed the %d-month horizon yet.", horizon_months)
        return 0

    log.info(
        "%d / %d snapshot dates are eligible for realised performance computation.",
        len(eligible), len(snap_dates),
    )

    total_saved = 0
    for pred_date in eligible:
        realized_df = compute_realized_for_date(
            pred_date, all_snaps, prices_df, benchmark_df, horizon_months
        )
        if realized_df.empty:
            continue
        n = save_realized_performance(realized_df)
        total_saved += n

    log.info("Realised performance update complete. %d total rows saved.", total_saved)
    return total_saved


# ── Analysis functions ────────────────────────────────────────────────────────

def compute_accuracy_summary(realized_df: pd.DataFrame) -> dict:
    """
    Overall accuracy statistics from the realised performance table.

    Returns dict with:
        n_predictions, n_correct, overall_hit_rate,
        mean_actual_excess_return, mean_return_error,
        mean_predicted_prob, actual_winner_rate,
        mean_actual_stock_return, mean_actual_benchmark_return
    """
    valid = realized_df[realized_df["winner_actual"].notna()].copy()
    if valid.empty:
        return {"status": "no_realized_data"}

    n_correct   = valid["prediction_correct"].sum()
    n_total     = len(valid)
    hit_rate    = float(n_correct / n_total) if n_total > 0 else None

    return {
        "n_predictions":              n_total,
        "n_correct":                  int(n_correct),
        "overall_hit_rate":           round(hit_rate, 4) if hit_rate else None,
        "actual_winner_rate":         round(float(valid["winner_actual"].mean()), 4),
        "mean_actual_excess_return":  round(float(valid["realized_12m_excess_return"].dropna().mean()), 4),
        "mean_return_error":          round(float(valid["return_error"].dropna().mean()), 4),
        "n_prediction_dates":         valid["prediction_date"].nunique(),
        "n_tickers":                  valid["ticker"].nunique(),
        "date_range":                 f"{valid['prediction_date'].min()} to {valid['prediction_date'].max()}",
    }


def compute_accuracy_by_classification(
    realized_df:   pd.DataFrame,
    snapshots_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Hit rate and mean excess return broken down by candidate_classification.

    Shows whether Strong candidates actually outperform Neutral / Weak stocks.
    This is the key test of the scoring model's discriminating power.
    """
    # Join classification from snapshots (not stored in realized_performance table)
    merged = realized_df.merge(
        snapshots_df[["prediction_date", "ticker", "candidate_classification",
                      "final_score", "probability_of_outperformance"]],
        on=["prediction_date", "ticker"],
        how="left",
    )

    valid = merged[merged["winner_actual"].notna()]
    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("candidate_classification")
        .agg(
            n                    = ("winner_actual", "count"),
            hit_rate             = ("prediction_correct", "mean"),
            actual_winner_rate   = ("winner_actual", "mean"),
            mean_actual_xret     = ("realized_12m_excess_return", "mean"),
            mean_predicted_prob  = ("probability_of_outperformance", "mean"),
            mean_final_score     = ("final_score", "mean"),
        )
        .reset_index()
        .round(4)
        .sort_values("mean_actual_xret", ascending=False)
    )
    return summary


def compute_calibration_by_bucket(realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Actual win rate by predicted probability bucket.

    A well-calibrated model should show actual win rates that match
    the predicted probability bucket.
    E.g. stocks predicted at 60–70% should win ~65% of the time.
    """
    valid = realized_df[realized_df["winner_actual"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("probability_bucket")
        .agg(
            n                  = ("winner_actual", "count"),
            actual_win_rate    = ("winner_actual", "mean"),
            mean_actual_xret   = ("realized_12m_excess_return", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return summary


def compute_rolling_hit_rate(
    realized_df: pd.DataFrame,
    window:      int = 6,
) -> pd.DataFrame:
    """
    Rolling hit rate over consecutive prediction dates.

    Shows whether model accuracy is improving, declining, or stable.
    """
    valid = realized_df[realized_df["prediction_correct"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    monthly = (
        valid.groupby("prediction_date")["prediction_correct"]
        .mean()
        .reset_index()
        .sort_values("prediction_date")
        .rename(columns={"prediction_correct": "hit_rate"})
    )
    monthly["rolling_hit_rate"] = monthly["hit_rate"].rolling(window, min_periods=1).mean()
    return monthly
