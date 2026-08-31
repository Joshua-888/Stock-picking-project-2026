"""
src/utils/math_utils.py

Shared mathematical helpers for feature engineering and model evaluation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def cagr(start_value: float, end_value: float, years: float) -> Optional[float]:
    """
    Compound Annual Growth Rate.

    Returns None for undefined cases (zero/negative start, zero years).

    Example: cagr(100, 200, 5) → 0.1487 (≈ 14.9% per year)
    """
    if years <= 0 or start_value <= 0:
        return None
    if end_value <= 0:
        return None
    return (end_value / start_value) ** (1.0 / years) - 1.0


def annualize_vol(monthly_returns: pd.Series) -> Optional[float]:
    """
    Annualise standard deviation of monthly returns.

    Multiplies monthly std by √12.  Returns None if fewer than 3 observations.
    """
    clean = monthly_returns.dropna()
    if len(clean) < 3:
        return None
    return float(clean.std() * math.sqrt(12))


def rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    min_obs: int = 12,
) -> Optional[float]:
    """
    Calculate beta (market sensitivity) from aligned return series.

    beta = Cov(stock, market) / Var(market)

    Returns None if insufficient overlapping observations.
    """
    merged = pd.DataFrame({"s": stock_returns, "m": market_returns}).dropna()
    if len(merged) < min_obs:
        return None
    cov_matrix = np.cov(merged["s"].values, merged["m"].values)
    var_market = cov_matrix[1, 1]
    if var_market == 0:
        return None
    return float(cov_matrix[0, 1] / var_market)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip values at the given quantiles.

    Values below the lower quantile are set to the lower quantile value.
    Values above the upper quantile are set to the upper quantile value.
    NaN values are preserved.

    Args:
        series: Numeric pd.Series.
        lower:  Lower quantile (e.g. 0.01 = 1st percentile).
        upper:  Upper quantile (e.g. 0.99 = 99th percentile).
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def zscore(series: pd.Series) -> pd.Series:
    """
    Standardise a series to zero mean and unit variance (z-score).

    Returns a series of NaN if std is zero.
    """
    mu  = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / std


def minmax_scale(series: pd.Series) -> pd.Series:
    """
    Scale a series to the [0, 1] range.

    Returns 0.5 everywhere if min == max.
    """
    lo = series.min()
    hi = series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Divide two floats, returning None for zero denominator or NaN inputs."""
    if denominator is None or numerator is None:
        return None
    try:
        if math.isnan(numerator) or math.isnan(denominator):
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    except (TypeError, ValueError):
        return None


def pct_change(new: float, old: float) -> Optional[float]:
    """
    Percentage change: (new - old) / abs(old).

    Returns None if old is zero, None, or NaN.
    """
    return safe_divide(new - old, abs(old)) if old else None


def clip_pe(pe: Optional[float], max_pe: float = 500.0) -> Optional[float]:
    """
    Return None for negative or implausibly large P/E ratios.

    A negative P/E (negative earnings) is valid data but not useful as a
    predictive feature in its raw form — it gets flagged and excluded.
    Cap raised to 500 to accommodate high-growth companies (e.g. Tesla ~225).
    Feature winsorization at training p99 handles extreme values during scoring.
    """
    if pe is None or math.isnan(pe):
        return None
    if pe <= 0 or pe > max_pe:
        return None
    return pe


def trailing_sum(series: pd.Series, n: int = 4) -> Optional[float]:
    """
    Sum the last n non-NaN values of a series.
    Returns None if fewer than n values are available.
    """
    clean = series.dropna()
    if len(clean) < n:
        return None
    return float(clean.iloc[-n:].sum())
