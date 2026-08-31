"""
tests/test_backtest.py

Tests for the backtesting engine and performance metrics.
"""

import math
import pytest
import numpy as np
import pandas as pd


# ── Metrics: mathematical properties ─────────────────────────────────────────

def test_total_return_compounds_correctly():
    """Total return of +10% then -10% should be -1% (not 0%)."""
    from src.backtesting.metrics import total_return
    returns  = pd.Series([0.10, -0.10])
    expected = (1.10 * 0.90) - 1   # = -0.01
    assert abs(total_return(returns) - expected) < 1e-9


def test_annualised_return_12_periods():
    """Monthly return of +1% compounded 12 times should be ≈ +12.68% annually."""
    from src.backtesting.metrics import annualised_return
    returns  = pd.Series([0.01] * 12)
    expected = (1.01 ** 12) - 1
    assert abs(annualised_return(returns, 12) - expected) < 1e-9


def test_max_drawdown_is_nonpositive():
    """Max drawdown must always be ≤ 0."""
    from src.backtesting.metrics import max_drawdown
    for returns in [
        pd.Series([0.05, -0.10, 0.03, -0.15, 0.08]),
        pd.Series([0.01, 0.02, 0.03]),   # all positive — drawdown should still be ≤ 0
        pd.Series([-0.10, -0.05, -0.02]),
    ]:
        mdd = max_drawdown(returns)
        assert mdd <= 0, f"Max drawdown {mdd} is positive"


def test_max_drawdown_all_positive_returns():
    """A series of all positive returns should have max drawdown = 0."""
    from src.backtesting.metrics import max_drawdown
    returns = pd.Series([0.01, 0.02, 0.005, 0.015])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_hit_rate_range():
    """Hit rate must be in [0, 1]."""
    from src.backtesting.metrics import hit_rate
    rng     = np.random.default_rng(42)
    port    = pd.Series(rng.normal(0, 0.05, 50))
    bench   = pd.Series(rng.normal(0, 0.04, 50))
    hr      = hit_rate(port, bench)
    assert 0.0 <= hr <= 1.0


def test_sharpe_positive_for_positive_excess():
    """Portfolio with strictly positive excess returns should have positive Sharpe."""
    from src.backtesting.metrics import sharpe_ratio
    returns = pd.Series([0.02, 0.025, 0.015, 0.022, 0.018, 0.020] * 10)
    sr = sharpe_ratio(returns)
    assert sr is not None and sr > 0


def test_compute_all_metrics_keys(tiny_prices, tiny_benchmark):
    """compute_all_metrics must return all expected keys."""
    from src.backtesting.metrics import compute_all_metrics
    rng      = np.random.default_rng(42)
    n        = 30
    port_ret = pd.Series(rng.normal(0.01, 0.05, n))
    bench_ret = pd.Series(rng.normal(0.008, 0.04, n))
    metrics  = compute_all_metrics(port_ret, bench_ret)
    required = ["total_return_portfolio","total_return_benchmark","total_excess_return",
                "ann_return_portfolio","ann_volatility_portfolio","sharpe_ratio",
                "sortino_ratio","max_drawdown_portfolio","hit_rate","n_periods"]
    for key in required:
        assert key in metrics, f"Missing key: {key}"


# ── Walk-forward splits ────────────────────────────────────────────────────────

def test_expanding_splits_no_overlap():
    """In expanding splits, train dates must not overlap with the test date."""
    from src.backtesting.walk_forward import expanding_splits
    dates  = [f"202{y}-0{m}-28" for y in range(0,5) for m in range(1,7)]
    splits = list(expanding_splits(dates, min_train=12))
    for train, test in splits:
        assert test not in train, "Test date found in train set"


def test_expanding_splits_min_train():
    """Expanding splits must not start until min_train dates are available."""
    from src.backtesting.walk_forward import expanding_splits
    dates  = [f"2020-{m:02d}-28" for m in range(1, 25)]
    splits = list(expanding_splits(dates, min_train=12))
    first_train, first_test = splits[0]
    assert len(first_train) >= 12


def test_rolling_splits_fixed_window():
    """Rolling splits must have exactly train_size training dates in each window."""
    from src.backtesting.walk_forward import rolling_splits
    dates  = [f"2020-{m:02d}-28" for m in range(1, 25)]
    splits = list(rolling_splits(dates, train_size=12))
    for train, test in splits:
        assert len(train) == 12, f"Expected 12 training dates, got {len(train)}"


# ── Portfolio simulation ───────────────────────────────────────────────────────

def test_portfolio_return_none_for_last_date(tiny_prices):
    """compute_period_return returns None for the last date (no next period)."""
    from src.backtesting.portfolio_simulation import compute_period_return
    matrix    = tiny_prices.pivot(index="date", columns="ticker", values="monthly_return")
    last_date = sorted(matrix.index)[-1]
    result    = compute_period_return(["A","B"], matrix, last_date)
    assert result is None


def test_portfolio_return_equals_mean_of_selected(tiny_prices):
    """Portfolio return must equal the equal-weight mean of selected stocks."""
    from src.backtesting.portfolio_simulation import compute_period_return
    matrix    = tiny_prices.pivot(index="date", columns="ticker", values="monthly_return")
    all_dates = sorted(matrix.index)
    if len(all_dates) < 2:
        pytest.skip("Insufficient dates")
    date      = all_dates[-2]
    next_date = all_dates[-1]
    tickers   = ["A", "B", "C"]
    result    = compute_period_return(tickers, matrix, date)
    expected  = matrix.loc[next_date, tickers].dropna().mean()
    assert abs(result - expected) < 1e-9
