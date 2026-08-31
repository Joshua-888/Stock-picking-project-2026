"""
src/features/target_creation.py

Creates the two target variables used for model training and evaluation.

TARGET VARIABLES
----------------
1. Regression target (continuous)
   future_12m_excess_return = future_12m_stock_return - future_12m_benchmark_return

2. Classification target (binary)
   winner = 1  if future_12m_stock_return > future_12m_benchmark_return
   winner = 0  otherwise

WHY BENCHMARK-RELATIVE?
-----------------------
In a bull market, most absolute stock returns are positive.
A model that predicts "everything will rise" would have high accuracy
but zero practical value.  Benchmark-relative returns answer the only
question that matters: does this stock beat the market?

LOOK-AHEAD SAFETY
-----------------
Targets are computed entirely from FUTURE prices — prices that occur
AFTER the feature_date.  They must NEVER be used as inputs to features.
The calling code is responsible for this separation, but this module
exposes is_target_available() to make the boundary explicit.

Timeline for one prediction:
  feature_date ──────────────────── target_date (feature_date + 12 months)
       │                                   │
  Features computed here              Outcome measured here
  (all past data, lag-safe)           (future price known)

HOW TARGETS ARE COMPUTED
------------------------
  future_12m_stock_return    = price(target_date) / price(feature_date) - 1
  future_12m_benchmark_return = bench(target_date) / bench(feature_date) - 1
  future_12m_excess_return   = stock_return - benchmark_return
  winner                     = 1 if excess_return > 0 else 0

If the exact target_date is not in the price data, the nearest
available month-end within ±15 days is used.

Usage
-----
    from src.features.target_creation import compute_all_targets

    targets_df = compute_all_targets(prices_df, benchmark_df)
    # Returns one row per (feature_date, ticker) where target is computable.
    # Rows where target_date has not yet passed have winner = NULL.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.dates import months_ago, to_month_end
from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_HORIZON_MONTHS = 12


# ── Date helpers ──────────────────────────────────────────────────────────────

def get_target_date(feature_date: str, horizon_months: int = DEFAULT_HORIZON_MONTHS) -> str:
    """
    Return the evaluation date for a given feature date.

    Example: get_target_date("2022-01-31", 12) → "2023-01-31"
    """
    ts = pd.Timestamp(feature_date) + pd.DateOffset(months=horizon_months)
    return to_month_end(ts.strftime("%Y-%m-%d"))


def is_target_available(
    feature_date:   str,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    today:          Optional[str] = None,
) -> bool:
    """
    Return True if enough time has elapsed to compute the target.

    A target for feature_date is available only after target_date has passed.
    Used to separate historical (labelled) rows from current predictions.

    Args:
        feature_date:   The prediction month-end.
        horizon_months: Forward horizon in months (default 12).
        today:          Override 'today' for testing (default: actual today).
    """
    target_dt = pd.Timestamp(get_target_date(feature_date, horizon_months))
    as_of     = pd.Timestamp(today) if today else pd.Timestamp.today()
    return as_of >= target_dt


# ── Price lookup ──────────────────────────────────────────────────────────────

def _lookup_price(
    price_index:    pd.Series,
    date:           str,
    tolerance_days: int = 15,
) -> Optional[float]:
    """
    Look up the price at a given date using O(log n) binary search.

    Month-end dates may be weekends or holidays in real data.
    Returns the nearest available price within tolerance_days, or None.

    Args:
        price_index:    Series indexed by ISO date strings, values = adjusted_close.
        date:           Target date string (YYYY-MM-DD).
        tolerance_days: Maximum calendar-day gap allowed.
    """
    idx = price_index.index  # sorted string index

    # Exact match — fast path
    if date in idx:
        return float(price_index[date])

    # Binary search for insertion point
    pos = idx.searchsorted(date)

    # Check the neighbour just before and just after the target date
    candidates: list[str] = []
    if pos < len(idx):
        candidates.append(idx[pos])
    if pos > 0:
        candidates.append(idx[pos - 1])

    target_ts = pd.Timestamp(date)
    best_price: Optional[float] = None
    best_gap   = tolerance_days + 1

    for c in candidates:
        gap = abs((target_ts - pd.Timestamp(c)).days)
        if gap < best_gap:
            best_gap   = gap
            best_price = float(price_index[c])

    return best_price if best_gap <= tolerance_days else None


# ── Single-date target computation ───────────────────────────────────────────

def compute_targets_for_date(
    feature_date:   str,
    prices_df:      pd.DataFrame,
    benchmark_df:   pd.DataFrame,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Compute target variables for all tickers at one feature_date.

    Returns a DataFrame with one row per ticker.
    Rows where target_date prices are not yet available have NaN targets.

    Args:
        feature_date:   The prediction month-end (YYYY-MM-DD).
        prices_df:      prices_clean DataFrame (all tickers, all dates).
        benchmark_df:   benchmark_prices DataFrame.
        horizon_months: Forward horizon (default 12).

    Returns:
        DataFrame with columns:
            feature_date, ticker,
            future_12m_stock_return, future_12m_benchmark_return,
            future_12m_excess_return, winner
    """
    target_date = get_target_date(feature_date, horizon_months)

    # ── Benchmark return ──────────────────────────────────────────────────────
    bench_idx    = benchmark_df.set_index("date")["adjusted_close"]
    bench_start  = _lookup_price(bench_idx, feature_date)
    bench_end    = _lookup_price(bench_idx, target_date)

    if bench_start and bench_end and bench_start > 0:
        bench_return = bench_end / bench_start - 1.0
    else:
        bench_return = None

    # ── Stock returns ─────────────────────────────────────────────────────────
    rows: list[dict] = []
    for ticker, grp in prices_df.groupby("ticker"):
        price_idx   = grp.set_index("date")["adjusted_close"]
        start_price = _lookup_price(price_idx, feature_date)
        end_price   = _lookup_price(price_idx, target_date)

        if start_price and end_price and start_price > 0:
            stock_return  = end_price / start_price - 1.0
            excess_return = (stock_return - bench_return) if bench_return is not None else None
            winner        = int(excess_return > 0) if excess_return is not None else None
        else:
            stock_return  = None
            excess_return = None
            winner        = None

        rows.append({
            "feature_date":               feature_date,
            "ticker":                     ticker,
            "future_12m_stock_return":    stock_return,
            "future_12m_benchmark_return": bench_return,
            "future_12m_excess_return":   excess_return,
            "winner":                     winner,
        })

    return pd.DataFrame(rows)


# ── Bulk target computation ───────────────────────────────────────────────────

def compute_all_targets(
    prices_df:       pd.DataFrame,
    benchmark_df:    pd.DataFrame,
    feature_dates:   Optional[list[str]] = None,
    horizon_months:  int = DEFAULT_HORIZON_MONTHS,
    only_available:  bool = False,
) -> pd.DataFrame:
    """
    Compute targets for every feature_date in the dataset.

    Args:
        prices_df:      prices_clean DataFrame.
        benchmark_df:   benchmark_prices DataFrame.
        feature_dates:  Specific dates to compute (default: all dates in prices_df).
        horizon_months: Forward horizon in months.
        only_available: If True, skip feature_dates where target_date has
                        not yet passed (i.e., only return labelled rows).

    Returns:
        DataFrame with all target rows, sorted by feature_date, ticker.
        Rows where prices are missing at target_date have NaN targets.
    """
    if feature_dates is None:
        feature_dates = sorted(prices_df["date"].unique())

    if only_available:
        feature_dates = [
            d for d in feature_dates
            if is_target_available(d, horizon_months)
        ]

    if not feature_dates:
        log.warning("No feature dates available for target computation.")
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    n_with_targets = 0

    for date in feature_dates:
        df = compute_targets_for_date(date, prices_df, benchmark_df, horizon_months)
        if df["winner"].notna().any():
            n_with_targets += 1
        chunks.append(df)

    # Filter out empty/all-NA chunks before concat to avoid pandas FutureWarning
    chunks = [c for c in chunks if not c.empty and c["winner"].notna().any()]
    all_targets = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    n_labelled = all_targets["winner"].notna().sum()
    n_total    = len(all_targets)
    winner_rate = all_targets["winner"].mean()

    log.info(
        "Targets computed: %d total rows, %d labelled (%.1f%%), winner rate = %.1f%%",
        n_total,
        n_labelled,
        100 * n_labelled / max(n_total, 1),
        100 * (winner_rate or 0),
    )

    return all_targets.sort_values(["feature_date", "ticker"]).reset_index(drop=True)


# ── Incremental update ────────────────────────────────────────────────────────

def update_targets_for_new_prices(
    existing_targets_df: pd.DataFrame,
    prices_df:           pd.DataFrame,
    benchmark_df:        pd.DataFrame,
    horizon_months:      int = DEFAULT_HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Fill in previously NULL targets now that new prices are available.

    Called during the monthly update workflow to backfill realized targets
    for predictions made ~12 months ago.

    Args:
        existing_targets_df: Current targets table (may have NULL winners).
        prices_df:           Updated prices_clean with new data.
        benchmark_df:        Updated benchmark prices.
        horizon_months:      Forward horizon.

    Returns:
        Updated targets DataFrame with previously NULL rows filled where possible.
    """
    null_mask = existing_targets_df["winner"].isna()
    if not null_mask.any():
        log.info("No NULL targets to update.")
        return existing_targets_df

    dates_to_fill = existing_targets_df.loc[null_mask, "feature_date"].unique()
    log.info("Attempting to fill targets for %d dates", len(dates_to_fill))

    result = existing_targets_df.copy()

    for date in dates_to_fill:
        if not is_target_available(date, horizon_months):
            continue
        new_targets = compute_targets_for_date(date, prices_df, benchmark_df, horizon_months)

        for _, new_row in new_targets.iterrows():
            if new_row["winner"] is None:
                continue
            mask = (result["feature_date"] == date) & (result["ticker"] == new_row["ticker"])
            result.loc[mask, "future_12m_stock_return"]     = new_row["future_12m_stock_return"]
            result.loc[mask, "future_12m_benchmark_return"] = new_row["future_12m_benchmark_return"]
            result.loc[mask, "future_12m_excess_return"]    = new_row["future_12m_excess_return"]
            result.loc[mask, "winner"]                      = new_row["winner"]

    n_filled = result["winner"].notna().sum() - existing_targets_df["winner"].notna().sum()
    log.info("Filled %d previously NULL target rows.", n_filled)
    return result


# ── Diagnostics ───────────────────────────────────────────────────────────────

def target_summary(targets_df: pd.DataFrame) -> dict:
    """
    Return summary statistics for a targets DataFrame.

    Useful for diagnostics and the Model Diagnostics dashboard page.

    Returns dict with keys:
        total_rows, labelled_rows, pct_labelled,
        winner_rate, mean_excess_return, std_excess_return,
        min_excess_return, max_excess_return,
        n_feature_dates, n_tickers, date_range
    """
    labelled = targets_df[targets_df["winner"].notna()]

    return {
        "total_rows":          len(targets_df),
        "labelled_rows":       len(labelled),
        "pct_labelled":        round(100 * len(labelled) / max(len(targets_df), 1), 1),
        "winner_rate":         round(float(labelled["winner"].mean()), 4) if len(labelled) > 0 else None,
        "mean_excess_return":  round(float(labelled["future_12m_excess_return"].mean()), 4) if len(labelled) > 0 else None,
        "std_excess_return":   round(float(labelled["future_12m_excess_return"].std()), 4) if len(labelled) > 0 else None,
        "min_excess_return":   round(float(labelled["future_12m_excess_return"].min()), 4) if len(labelled) > 0 else None,
        "max_excess_return":   round(float(labelled["future_12m_excess_return"].max()), 4) if len(labelled) > 0 else None,
        "n_feature_dates":     targets_df["feature_date"].nunique(),
        "n_tickers":           targets_df["ticker"].nunique(),
        "date_range":          f"{targets_df['feature_date'].min()} to {targets_df['feature_date'].max()}",
    }


def winner_rate_by_sector(
    targets_df: pd.DataFrame,
    stocks_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Return winner rate and mean excess return broken down by sector.

    Args:
        targets_df: Targets DataFrame with winner column.
        stocks_df:  Stocks metadata DataFrame with sector column.

    Returns:
        DataFrame: sector, n, winner_rate, mean_excess_return
    """
    merged = targets_df.merge(
        stocks_df[["ticker", "sector"]], on="ticker", how="left"
    )
    labelled = merged[merged["winner"].notna()]

    summary = (
        labelled.groupby("sector")
        .agg(
            n              = ("winner", "count"),
            winner_rate    = ("winner", "mean"),
            mean_excess    = ("future_12m_excess_return", "mean"),
            std_excess     = ("future_12m_excess_return", "std"),
        )
        .reset_index()
        .round(4)
        .sort_values("winner_rate", ascending=False)
    )
    return summary
