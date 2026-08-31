"""
tests/test_feature_engineering.py

Tests for feature engineering: correctness of calculations,
look-ahead bias prevention, and data quality scoring.
"""

import math
import pytest
import numpy as np
import pandas as pd


# ── Momentum skip-month convention ────────────────────────────────────────────

def test_momentum_12m_skip_uses_price_minus_2(tiny_prices, tiny_stocks, tiny_fundamentals):
    """
    12-month momentum should use price[-2] / price[-14] - 1 (skip last month).
    This implements the standard 12-1 momentum convention.
    """
    from src.features.feature_engineering import compute_features_for_date

    bench   = pd.DataFrame({"benchmark_ticker": "SPY", "date": tiny_prices["date"].unique(),
                             "adjusted_close": 100.0, "monthly_return": 0.0})
    date    = sorted(tiny_prices["date"].unique())[-1]
    feats   = compute_features_for_date(date, tiny_prices, bench, tiny_fundamentals, tiny_stocks)
    row     = feats[feats["ticker"] == "A"].iloc[0]

    # Compute expected value manually
    a_prices = tiny_prices[tiny_prices["ticker"] == "A"].sort_values("date")["adjusted_close"]
    prices_to_date = a_prices[a_prices.index[:a_prices.index.get_loc(
        tiny_prices[tiny_prices["ticker"]=="A"].sort_values("date").index[-1]) + 1]]

    expected = float(prices_to_date.iloc[-2]) / float(prices_to_date.iloc[-14]) - 1
    actual   = float(row["twelve_month_momentum"])
    assert abs(expected - actual) < 1e-6, f"12m momentum: expected {expected:.6f} got {actual:.6f}"


def test_momentum_1m_uses_most_recent(tiny_prices, tiny_stocks, tiny_fundamentals):
    """
    1-month return should use price[-1] / price[-2] - 1 (NO skip — reversal signal).
    """
    from src.features.feature_engineering import compute_features_for_date

    bench = pd.DataFrame({"benchmark_ticker": "SPY", "date": tiny_prices["date"].unique(),
                          "adjusted_close": 100.0, "monthly_return": 0.0})
    date  = sorted(tiny_prices["date"].unique())[-1]
    feats = compute_features_for_date(date, tiny_prices, bench, tiny_fundamentals, tiny_stocks)
    row   = feats[feats["ticker"] == "A"].iloc[0]

    a_prices = tiny_prices[tiny_prices["ticker"] == "A"].sort_values("date")["adjusted_close"].values
    expected = a_prices[-1] / a_prices[-2] - 1
    actual   = float(row["return_1m"])
    assert abs(expected - actual) < 1e-6


# ── Drawdown must be non-positive ─────────────────────────────────────────────

def test_drawdown_from_52w_high_nonpositive(tiny_prices, tiny_stocks, tiny_fundamentals):
    """drawdown_from_52w_high is always ≤ 0 (stock cannot be above its own 52w high)."""
    from src.features.feature_engineering import compute_features_for_date

    bench = pd.DataFrame({"benchmark_ticker": "SPY", "date": tiny_prices["date"].unique(),
                          "adjusted_close": 100.0, "monthly_return": 0.0})
    date  = sorted(tiny_prices["date"].unique())[-1]
    feats = compute_features_for_date(date, tiny_prices, bench, tiny_fundamentals, tiny_stocks)

    dd = feats["drawdown_from_52w_high"].dropna()
    assert (dd <= 0.001).all(), f"drawdown_from_52w_high has positive values: {dd[dd>0.001]}"


# ── Market cap uses shares × price ────────────────────────────────────────────

def test_market_cap_equals_price_times_shares(tiny_prices, tiny_stocks, tiny_fundamentals):
    """market_cap (in $M) = adjusted_close × shares_outstanding_m."""
    from src.features.feature_engineering import compute_features_for_date

    bench = pd.DataFrame({"benchmark_ticker": "SPY", "date": tiny_prices["date"].unique(),
                          "adjusted_close": 100.0, "monthly_return": 0.0})
    date  = sorted(tiny_prices["date"].unique())[-1]
    feats = compute_features_for_date(date, tiny_prices, bench, tiny_fundamentals, tiny_stocks)

    for ticker in tiny_stocks["ticker"].tolist():
        row     = feats[feats["ticker"] == ticker]
        if row.empty or pd.isna(row.iloc[0].get("market_cap")):
            continue
        shares  = float(tiny_stocks[tiny_stocks["ticker"] == ticker]["shares_outstanding_m"].iloc[0])
        price   = float(tiny_prices[(tiny_prices["ticker"] == ticker) &
                                    (tiny_prices["date"] == date)]["adjusted_close"].iloc[0])
        expected_mcap = price * shares
        actual_mcap   = float(row.iloc[0]["market_cap"])
        # Allow 2% tolerance — floating-point and mid-month price differences
        assert abs(expected_mcap - actual_mcap) / expected_mcap < 0.02, (
            f"{ticker}: market_cap {actual_mcap:.0f} != price×shares {expected_mcap:.0f}"
        )


# ── DQ scores in valid range ───────────────────────────────────────────────────

def test_dq_scores_in_range(tiny_prices, tiny_stocks, tiny_fundamentals):
    """Data quality scores must be in [0, 100]."""
    from src.features.feature_engineering import compute_features_for_date

    bench = pd.DataFrame({"benchmark_ticker": "SPY", "date": tiny_prices["date"].unique(),
                          "adjusted_close": 100.0, "monthly_return": 0.0})
    date  = sorted(tiny_prices["date"].unique())[-1]
    feats = compute_features_for_date(date, tiny_prices, bench, tiny_fundamentals, tiny_stocks)

    dq = feats["data_quality_score"].dropna()
    assert (dq >= 0).all() and (dq <= 100).all(), "DQ score out of [0, 100] range"


# ── Lag rules ─────────────────────────────────────────────────────────────────

def test_lag_filter_excludes_future_reports(tiny_fundamentals):
    """filter_fundamentals_lag_safe must exclude rows where report_date > as_of_date."""
    from src.features.lag_rules import filter_fundamentals_lag_safe

    as_of    = "2018-06-30"
    filtered = filter_fundamentals_lag_safe(tiny_fundamentals, as_of)
    if not filtered.empty:
        assert (filtered["report_date"] <= as_of).all(), (
            "Lag filter allowed rows with future report_date"
        )


def test_lag_filter_returns_less_than_full(tiny_fundamentals):
    """Lag-safe filter with an early date returns fewer rows than the full dataset."""
    from src.features.lag_rules import filter_fundamentals_lag_safe

    early   = filter_fundamentals_lag_safe(tiny_fundamentals, "2017-01-01")
    full    = filter_fundamentals_lag_safe(tiny_fundamentals, "2030-01-01")
    assert len(early) < len(full)


# ── Math helpers ───────────────────────────────────────────────────────────────

def test_cagr_correctness():
    """CAGR(100, 200, 5) should be approximately 14.87%."""
    from src.utils.math_utils import cagr
    result = cagr(100.0, 200.0, 5.0)
    assert result is not None
    assert abs(result - 0.14870) < 0.0001, f"CAGR({100},{200},{5}) = {result:.5f}"


def test_cagr_zero_start_returns_none():
    """CAGR with zero start value should return None (undefined)."""
    from src.utils.math_utils import cagr
    assert cagr(0.0, 100.0, 5.0) is None


def test_annualise_vol():
    """Annualised vol = monthly std × sqrt(12)."""
    from src.utils.math_utils import annualize_vol
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.00])
    result  = annualize_vol(returns)
    expected = returns.std() * math.sqrt(12)
    assert abs(result - expected) < 1e-9
