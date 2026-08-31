"""
src/backtesting/backtest.py

Main backtesting orchestration — ties together simulation, metrics, and storage.

Usage
-----
    from src.backtesting.backtest import run_backtest

    result = run_backtest(top_n=10)
    print(result["metrics"])
    print(result["monthly_returns"].head())
"""

from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd

from src.backtesting.metrics import compute_all_metrics, rolling_sharpe, rolling_hit_rate
from src.backtesting.portfolio_simulation import (
    run_simulation, compute_sector_performance,
)
from src.database.db import save_backtest_result
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Main backtest function ────────────────────────────────────────────────────

def run_backtest(
    prices_df:      Optional[pd.DataFrame] = None,
    benchmark_df:   Optional[pd.DataFrame] = None,
    stocks_df:      Optional[pd.DataFrame] = None,
    features_df:    Optional[pd.DataFrame] = None,
    top_n:          int   = 10,
    min_train:      int   = 24,
    min_dq_score:   float = 40.0,
    strategy_name:  str   = "top_N_feature_score",
    save_to_db:     bool  = True,
) -> dict:
    """
    Run the full backtesting pipeline and return results.

    If DataFrames are not provided, loads from the database automatically.

    Args:
        prices_df:     prices_clean DataFrame.
        benchmark_df:  benchmark_prices DataFrame.
        stocks_df:     stocks metadata DataFrame.
        features_df:   monthly_features DataFrame (all dates).
        top_n:         Number of stocks in the equal-weight portfolio.
        min_train:     Minimum months of history before starting.
        min_dq_score:  Minimum data quality score for stock inclusion.
        strategy_name: Label for this backtest run.
        save_to_db:    Whether to save results to backtest_results table.

    Returns:
        dict with keys:
          monthly_returns    — per-month portfolio/benchmark/excess returns
          cumulative_returns — cumulative growth of $1 invested
          metrics            — summary statistics dict
          rolling_metrics    — rolling Sharpe and hit rate
          sector_performance — per-sector breakdown
          backtest_id        — UUID for this run
    """
    # ── Load data from DB if not provided ─────────────────────────────────────
    if prices_df is None or benchmark_df is None or features_df is None:
        from src.database.queries import (
            load_prices_clean, load_benchmark_prices,
            load_monthly_features, load_stocks,
        )
        if prices_df   is None: prices_df   = load_prices_clean()
        if benchmark_df is None: benchmark_df = load_benchmark_prices()
        if features_df  is None: features_df  = load_monthly_features(start_date="2015-01-31")
        if stocks_df    is None: stocks_df    = load_stocks()

    if prices_df.empty or benchmark_df.empty or features_df.empty:
        log.warning("Insufficient data for backtest.")
        return {}

    # ── Build returns matrix ───────────────────────────────────────────────────
    returns_matrix = prices_df.pivot(
        index="date", columns="ticker", values="monthly_return"
    ).sort_index()

    benchmark_returns = (
        benchmark_df
        .set_index("date")["monthly_return"]
        .sort_index()
        .dropna()
    )

    # ── Run simulation ─────────────────────────────────────────────────────────
    log.info(
        "Starting backtest: top_n=%d  min_train=%d  strategy=%s",
        top_n, min_train, strategy_name,
    )

    sim_df = run_simulation(
        all_features      = features_df,
        returns_matrix    = returns_matrix,
        benchmark_returns = benchmark_returns,
        top_n             = top_n,
        min_train         = min_train,
        min_dq_score      = min_dq_score,
        stocks_meta       = stocks_df,
    )

    if sim_df.empty:
        log.warning("Simulation returned no data.")
        return {}

    # ── Compute cumulative returns ─────────────────────────────────────────────
    port_ret  = sim_df.set_index("date")["portfolio_return"]
    bench_ret = sim_df.set_index("date")["benchmark_return"]

    cum_port  = (1 + port_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()
    cum_excess = cum_port / cum_bench   # growth relative to benchmark

    cumulative = pd.DataFrame({
        "date":               cum_port.index,
        "portfolio_cumret":   cum_port.values,
        "benchmark_cumret":   cum_bench.values,
        "relative_cumret":    cum_excess.values,
    })

    # ── Compute all metrics ────────────────────────────────────────────────────
    metrics = compute_all_metrics(port_ret, bench_ret)
    metrics["top_n"]          = top_n
    metrics["strategy_name"]  = strategy_name
    metrics["start_date"]     = sim_df["date"].min()
    metrics["end_date"]       = sim_df["date"].max()

    # ── Rolling metrics ────────────────────────────────────────────────────────
    roll_sharpe  = rolling_sharpe(port_ret, window=12)
    roll_hit     = rolling_hit_rate(port_ret, bench_ret, window=12)

    rolling_df = pd.DataFrame({
        "date":          roll_sharpe.index,
        "rolling_sharpe": roll_sharpe.values,
        "rolling_hit":   roll_hit.values,
    })

    # ── Sector performance ─────────────────────────────────────────────────────
    sector_perf = compute_sector_performance(
        sim_df, features_df, returns_matrix, stocks_df
    ) if stocks_df is not None else pd.DataFrame()

    # ── Save to database ───────────────────────────────────────────────────────
    backtest_id = str(uuid.uuid4())
    if save_to_db:
        import json
        save_backtest_result({
            "backtest_id":     backtest_id,
            "model_version":   strategy_name,
            "start_date":      metrics["start_date"],
            "end_date":        metrics["end_date"],
            "strategy_name":   strategy_name,
            "benchmark_return": metrics["total_return_benchmark"],
            "strategy_return":  metrics["total_return_portfolio"],
            "excess_return":    metrics["total_excess_return"],
            "hit_rate":         metrics["hit_rate"],
            "drawdown":         metrics["max_drawdown_portfolio"],
            "volatility":       metrics["ann_volatility_portfolio"],
            "sharpe_ratio":     metrics["sharpe_ratio"],
            "sortino_ratio":    metrics["sortino_ratio"],
            "metrics_json":     json.dumps({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, type(None)))
            }),
        })

    log.info(
        "Backtest complete: %d months  port=%.1f%%  bench=%.1f%%  "
        "excess=%.1f%%  Sharpe=%.2f  hitrate=%.1f%%",
        metrics["n_periods"],
        metrics["total_return_portfolio"] * 100,
        metrics["total_return_benchmark"] * 100,
        metrics["total_excess_return"] * 100,
        metrics.get("sharpe_ratio") or 0,
        metrics["hit_rate"] * 100,
    )

    return {
        "monthly_returns":    sim_df,
        "cumulative_returns": cumulative,
        "metrics":            metrics,
        "rolling_metrics":    rolling_df,
        "sector_performance": sector_perf,
        "backtest_id":        backtest_id,
    }
