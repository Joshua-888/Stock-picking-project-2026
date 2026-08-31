"""
src/backtesting/metrics.py

Performance metrics for the backtesting engine.

All functions operate on pd.Series of periodic returns (e.g. monthly).
Annualisation assumes 12 periods per year (monthly data).

METRICS IMPLEMENTED
-------------------
Sharpe ratio        = annualised_excess_return / annualised_vol
                      (excess over risk-free = 0, which is conservative)
Sortino ratio       = annualised_excess_return / annualised_downside_vol
                      (only penalises negative returns, unlike Sharpe)
Max drawdown        = (peak − trough) / peak — largest loss from any high point
Hit rate            = fraction of periods portfolio beat the benchmark
Information ratio   = annualised_excess_return / annualised_tracking_error
Calmar ratio        = annualised_return / |max_drawdown|
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12   # monthly data


# ── Core helpers ──────────────────────────────────────────────────────────────

def _annualise(periodic_return: float, n: int = PERIODS_PER_YEAR) -> float:
    """Compound a periodic return to an annual rate."""
    return (1 + periodic_return) ** n - 1


def _annualise_vol(periodic_vol: float, n: int = PERIODS_PER_YEAR) -> float:
    """Scale periodic std-dev to annual: σ_annual = σ_monthly × √12"""
    return periodic_vol * math.sqrt(n)


# ── Individual metric functions ───────────────────────────────────────────────

def total_return(returns: pd.Series) -> float:
    """
    Compound total return over the full series.
    E.g. [0.02, -0.01, 0.03] → (1.02)(0.99)(1.03) − 1
    """
    return float((1 + returns.dropna()).prod() - 1)


def annualised_return(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Geometric annualised return."""
    n = len(returns.dropna())
    if n == 0:
        return float("nan")
    tr = total_return(returns)
    return float((1 + tr) ** (periods / n) - 1)


def annualised_volatility(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualised standard deviation of returns."""
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std() * math.sqrt(periods))


def downside_volatility(returns: pd.Series, periods: int = PERIODS_PER_YEAR,
                        threshold: float = 0.0) -> float:
    """
    Annualised standard deviation of returns below the threshold.
    Used in the Sortino ratio denominator.
    """
    clean = returns.dropna()
    neg   = clean[clean < threshold]
    if len(neg) < 2:
        return float("nan")
    return float(neg.std() * math.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    """
    Annualised Sharpe ratio.

    Risk-free rate is set to 0 (conservative for research purposes).
    """
    ann_ret = annualised_return(returns, periods)
    ann_vol = annualised_volatility(returns, periods)
    if ann_vol == 0 or math.isnan(ann_vol):
        return None
    return round(float((ann_ret - risk_free) / ann_vol), 4)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    """
    Annualised Sortino ratio (uses downside deviation, not total std).
    Better measure for asymmetric return distributions.
    """
    ann_ret   = annualised_return(returns, periods)
    down_vol  = downside_volatility(returns, periods)
    if down_vol is None or down_vol == 0 or math.isnan(down_vol):
        return None
    return round(float((ann_ret - risk_free) / down_vol), 4)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in cumulative returns.
    Returns a negative decimal, e.g. -0.25 = 25% drawdown.
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    cum = (1 + clean).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def hit_rate(portfolio_returns: pd.Series,
             benchmark_returns: pd.Series) -> float:
    """
    Fraction of periods where the portfolio beat the benchmark.
    A random portfolio has hit rate ≈ 0.50.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    p_col, b_col = aligned.columns[0], aligned.columns[1]
    return float((aligned[p_col] > aligned[b_col]).mean())


def information_ratio(portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series,
                      periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    """
    Annualised Information Ratio = annualised_excess_return / tracking_error.
    Measures the consistency of outperformance.
    """
    excess = portfolio_returns - benchmark_returns
    ann_excess = annualised_return(excess, periods)
    tracking   = annualised_volatility(excess, periods)
    if tracking == 0 or math.isnan(tracking):
        return None
    return round(float(ann_excess / tracking), 4)


def calmar_ratio(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    """
    Calmar ratio = annualised_return / |max_drawdown|.
    Higher is better; >1 means annualised return exceeds max drawdown.
    """
    ann_ret = annualised_return(returns, periods)
    mdd     = max_drawdown(returns)
    if mdd == 0 or math.isnan(mdd):
        return None
    return round(float(ann_ret / abs(mdd)), 4)


# ── Rolling metrics ────────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling annualised Sharpe ratio over a window of months."""
    def _sr(x):
        if len(x.dropna()) < 3:
            return float("nan")
        v = annualised_volatility(x)
        if not v or math.isnan(v):
            return float("nan")
        return annualised_return(x) / v
    return returns.rolling(window).apply(_sr, raw=False)


def rolling_hit_rate(portfolio_returns: pd.Series,
                     benchmark_returns: pd.Series,
                     window: int = 12) -> pd.Series:
    """Rolling hit rate over a window of months."""
    excess = (portfolio_returns > benchmark_returns).astype(float)
    return excess.rolling(window).mean()


# ── Full summary ──────────────────────────────────────────────────────────────

def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Compute all performance metrics for a portfolio vs benchmark.

    Args:
        portfolio_returns: Monthly portfolio returns (equal-weight top-N).
        benchmark_returns: Monthly benchmark returns aligned to same dates.

    Returns:
        dict of named metrics.
    """
    excess = portfolio_returns - benchmark_returns

    return {
        "total_return_portfolio":    round(total_return(portfolio_returns), 4),
        "total_return_benchmark":    round(total_return(benchmark_returns), 4),
        "total_excess_return":       round(total_return(excess), 4),
        "ann_return_portfolio":      round(annualised_return(portfolio_returns), 4),
        "ann_return_benchmark":      round(annualised_return(benchmark_returns), 4),
        "ann_excess_return":         round(annualised_return(excess), 4),
        "ann_volatility_portfolio":  round(annualised_volatility(portfolio_returns), 4),
        "ann_volatility_benchmark":  round(annualised_volatility(benchmark_returns), 4),
        "sharpe_ratio":              sharpe_ratio(portfolio_returns),
        "sortino_ratio":             sortino_ratio(portfolio_returns),
        "information_ratio":         information_ratio(portfolio_returns, benchmark_returns),
        "calmar_ratio":              calmar_ratio(portfolio_returns),
        "max_drawdown_portfolio":    round(max_drawdown(portfolio_returns), 4),
        "max_drawdown_benchmark":    round(max_drawdown(benchmark_returns), 4),
        "hit_rate":                  round(hit_rate(portfolio_returns, benchmark_returns), 4),
        "n_periods":                 int(portfolio_returns.dropna().shape[0]),
    }
