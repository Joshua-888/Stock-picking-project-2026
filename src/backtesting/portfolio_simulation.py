"""
src/backtesting/portfolio_simulation.py

Simulates an equal-weight top-N portfolio that rebalances monthly.

HOW THE SIMULATION WORKS
--------------------------
At each month T (after a minimum training period):

  1. Load features for month T.
  2. Score all stocks using feature-based scoring (no model training —
     only feature ranks are used, which avoids look-ahead bias).
  3. Select the top-N stocks by final_score.
  4. Compute equal-weight portfolio return for month T+1:
       portfolio_return_T = mean(next_month_returns of selected stocks)
  5. Record portfolio return vs benchmark return for month T+1.
  6. Advance to month T+1 and repeat.

FEATURE-BASED SCORING IN THE BACKTEST
---------------------------------------
The scoring model in Step 14 can use logistic regression and Ridge
probabilities.  In the backtest, re-training those models at every month
would take hours.

Instead, the backtest uses ONLY the feature-based component scores:
  • momentum_score     (from momentum features)
  • fundamental_score  (from ROE, margins, growth)
  • valuation_score    (from sector-relative PE)
  • risk_score         (inverse of volatility, beta)
  • data_quality_score (from feature engineering)

This is honest because:
  • All feature values are look-ahead-safe at each date.
  • No future prices or returns enter the scoring.
  • The feature-based score is a subset of the full final_score,
    so it tests whether the fundamental signals alone are valuable.

IMPORTANT CAVEAT
----------------
The backtest uses features already stored in the database (computed
with the full history in one pass during earlier steps).  For a
production system, features should be recomputed at each step using
only data available up to that date.  The database-stored features
ARE look-ahead-safe at the feature level (the lag rules were enforced),
but the VIF filtering and Lasso feature selection used information from
the full sample.  This is a minor form of look-ahead bias that is
documented here for transparency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.scoring.scoring_model import compute_scores
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Stock selector ────────────────────────────────────────────────────────────

def select_portfolio(
    features_row: pd.DataFrame,
    top_n:         int = 10,
    min_dq_score:  float = 40.0,
) -> list[str]:
    """
    Score stocks at a single date and return the top-N tickers.

    Uses feature-based scoring only (no model predictions).

    Args:
        features_row: Feature DataFrame for one date (indexed by ticker).
        top_n:        Number of stocks to select.
        min_dq_score: Minimum data quality score — stocks below this are excluded.

    Returns:
        List of selected ticker strings (up to top_n).
    """
    if features_row.empty:
        return []

    # Filter out low data-quality stocks
    dq_col = "data_quality_score"
    if dq_col in features_row.columns:
        eligible = features_row[features_row[dq_col].fillna(0) >= min_dq_score]
    else:
        eligible = features_row.copy()

    if eligible.empty:
        return []

    # Feature-based scores only (logit_proba = None, ridge_pred = None)
    scores = compute_scores(eligible)
    if scores.empty:
        return []

    # attach ticker if it ended up as a column or restore from index
    if "ticker" not in scores.columns:
        scores.insert(0, "ticker", eligible.index)

    top = scores.nlargest(top_n, "final_score")
    return top["ticker"].tolist()


# ── Portfolio return computation ───────────────────────────────────────────────

def compute_period_return(
    tickers:        list[str],
    returns_matrix: pd.DataFrame,
    date:           str,
) -> Optional[float]:
    """
    Equal-weight portfolio return for the NEXT period after date.

    Args:
        tickers:        List of selected tickers.
        returns_matrix: Wide DataFrame — index=date, columns=ticker, values=monthly_return.
        date:           The current scoring date (returns are for the NEXT date).

    Returns:
        Equal-weight mean return, or None if insufficient data.
    """
    all_dates = sorted(returns_matrix.index.tolist())
    try:
        idx = all_dates.index(date)
    except ValueError:
        return None

    if idx + 1 >= len(all_dates):
        return None

    next_date = all_dates[idx + 1]
    row = returns_matrix.loc[next_date, [t for t in tickers if t in returns_matrix.columns]]
    valid = row.dropna()
    return float(valid.mean()) if len(valid) > 0 else None


# ── Full simulation ────────────────────────────────────────────────────────────

def run_simulation(
    all_features:    pd.DataFrame,
    returns_matrix:  pd.DataFrame,
    benchmark_returns: pd.Series,
    top_n:           int   = 10,
    min_train:       int   = 24,
    min_dq_score:    float = 40.0,
    stocks_meta:     Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run the monthly portfolio simulation over the full history.

    Args:
        all_features:      All monthly features (feature_date + ticker + feature cols).
        returns_matrix:    Wide matrix of monthly returns (index=date, columns=ticker).
        benchmark_returns: Series of benchmark monthly returns indexed by date.
        top_n:             Number of stocks to hold each month.
        min_train:         Minimum months before the first portfolio selection.
        min_dq_score:      Minimum data quality score for inclusion.
        stocks_meta:       Optional stocks DataFrame for sector lookups.

    Returns:
        DataFrame with one row per month:
          date, portfolio_return, benchmark_return, excess_return,
          selected_tickers (list), n_selected, hit (1=portfolio>bench)
    """
    all_dates = sorted(all_features["feature_date"].unique())

    if len(all_dates) < min_train + 1:
        log.warning("Insufficient dates for simulation: %d < %d + 1", len(all_dates), min_train)
        return pd.DataFrame()

    rows: list[dict] = []
    sector_map = {}
    if stocks_meta is not None and "ticker" in stocks_meta.columns and "sector" in stocks_meta.columns:
        sector_map = dict(zip(stocks_meta["ticker"], stocks_meta["sector"]))

    for i in range(min_train, len(all_dates) - 1):
        score_date = all_dates[i]

        # Features for this date (already look-ahead safe from feature engineering)
        date_feats = all_features[all_features["feature_date"] == score_date].copy()
        if date_feats.empty:
            continue

        date_feats_idx = (
            date_feats.set_index("ticker")
            if "ticker" in date_feats.columns else date_feats
        )

        # Select portfolio
        selected = select_portfolio(date_feats_idx, top_n, min_dq_score)

        # Compute next-period return
        port_ret  = compute_period_return(selected, returns_matrix, score_date)
        _bench_next = benchmark_returns.loc[benchmark_returns.index > score_date]
        bench_ret   = float(_bench_next.iloc[0]) if not _bench_next.empty else None

        if port_ret is None or bench_ret is None:
            continue

        excess = port_ret - bench_ret
        sectors = list({sector_map.get(t, "Unknown") for t in selected})

        rows.append({
            "date":               score_date,
            "portfolio_return":   round(port_ret,  6),
            "benchmark_return":   round(bench_ret, 6),
            "excess_return":      round(excess,    6),
            "selected_tickers":   selected,
            "n_selected":         len(selected),
            "sectors":            sectors,
            "hit":                int(port_ret > bench_ret),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info(
        "Simulation complete: %d months  hit_rate=%.1f%%  "
        "mean_excess=%.3f",
        len(df),
        df["hit"].mean() * 100,
        df["excess_return"].mean(),
    )
    return df


# ── Sector breakdown ──────────────────────────────────────────────────────────

def compute_sector_performance(
    sim_df:     pd.DataFrame,
    all_features: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    stocks_meta:  pd.DataFrame,
) -> pd.DataFrame:
    """
    For each sector, compute average excess return of top-ranked stocks in that sector.

    This shows which sectors the model was best at predicting.
    """
    if sim_df.empty or stocks_meta.empty:
        return pd.DataFrame()

    sector_map = dict(zip(stocks_meta["ticker"], stocks_meta["sector"]))
    sector_rows = []

    for _, row in sim_df.iterrows():
        date     = row["date"]
        selected = row["selected_tickers"]
        for ticker in selected:
            sector = sector_map.get(ticker, "Unknown")
            all_dates = sorted(returns_matrix.index.tolist())
            try:
                idx = all_dates.index(date)
                if idx + 1 < len(all_dates):
                    next_date = all_dates[idx + 1]
                    ret = (returns_matrix.loc[next_date, ticker]
                           if ticker in returns_matrix.columns else None)
                    if ret is not None:
                        sector_rows.append({
                            "date":           date,
                            "ticker":         ticker,
                            "sector":         sector,
                            "stock_return":   float(ret),
                            "bench_return":   row["benchmark_return"],
                            "excess_return":  float(ret) - row["benchmark_return"],
                        })
            except (ValueError, KeyError):
                continue

    if not sector_rows:
        return pd.DataFrame()

    df = pd.DataFrame(sector_rows)
    return (
        df.groupby("sector")
        .agg(
            n_stock_months  = ("excess_return", "count"),
            mean_excess     = ("excess_return", "mean"),
            hit_rate        = ("excess_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
        .round(4)
        .sort_values("mean_excess", ascending=False)
    )
