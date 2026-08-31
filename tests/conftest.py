"""
tests/conftest.py

Shared pytest fixtures for the stock analysis test suite.

All tests that need a database receive a fresh temporary SQLite database
via the `test_db` fixture — no test ever touches the production database.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ── Tiny price dataset (5 tickers × 60 months) ────────────────────────────────

@pytest.fixture(scope="session")
def tiny_prices():
    """
    Minimal monthly price data for 5 tickers from 2018-01 to 2022-12.
    Prices follow a simple random walk with positive drift.
    """
    rng     = np.random.default_rng(42)
    tickers = ["A", "B", "C", "D", "E"]
    dates   = pd.date_range("2018-01-31", periods=60, freq="ME").strftime("%Y-%m-%d").tolist()
    rows    = []
    for ticker in tickers:
        price = 100.0
        for date in dates:
            ret   = rng.normal(0.008, 0.05)
            price = max(price * (1 + ret), 1.0)
            rows.append({
                "ticker":            ticker,
                "date":              date,
                "adjusted_close":    round(price, 4),
                "monthly_return":    round(ret, 6),
                "volume":            rng.integers(1_000_000, 10_000_000),
                "data_quality_flag": "ok",
            })
    df               = pd.DataFrame(rows)
    # First row per ticker has no return
    df.loc[df.groupby("ticker")["date"].idxmin(), "monthly_return"] = float("nan")
    return df


@pytest.fixture(scope="session")
def tiny_benchmark():
    """SPY-like benchmark monthly prices, same date range."""
    rng   = np.random.default_rng(99)
    dates = pd.date_range("2018-01-31", periods=60, freq="ME").strftime("%Y-%m-%d").tolist()
    rows  = []
    price = 200.0
    for i, date in enumerate(dates):
        ret   = rng.normal(0.007, 0.04)
        price = max(price * (1 + ret), 1.0)
        rows.append({
            "benchmark_ticker": "SPY",
            "date":             date,
            "adjusted_close":   round(price, 4),
            "monthly_return":   round(ret, 6) if i > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_fundamentals():
    """Minimal quarterly fundamentals for 5 tickers."""
    rng     = np.random.default_rng(7)
    tickers = ["A", "B", "C", "D", "E"]
    q_dates = pd.date_range("2015-03-31", periods=32, freq="QE")
    rows    = []
    for ticker in tickers:
        eps_q = rng.uniform(0.5, 2.0)
        for qend in q_dates:
            q_label  = f"Q{(qend.month - 1) // 3 + 1}"
            lag_days = 90 if q_label == "Q4" else 45
            eps_q    = max(eps_q * rng.uniform(0.97, 1.05), 0.01)
            rows.append({
                "ticker":           ticker,
                "fiscal_period":    q_label,
                "fiscal_date":      qend.strftime("%Y-%m-%d"),
                "report_date":      (qend + pd.Timedelta(days=lag_days)).strftime("%Y-%m-%d"),
                "revenue":          round(eps_q * rng.uniform(8, 15), 4),
                "eps":              round(eps_q, 4),
                "net_income":       round(eps_q * rng.uniform(0.5, 1.0), 4),
                "equity":           round(rng.uniform(5, 50), 4),
                "total_debt":       round(rng.uniform(2, 20), 4),
                "free_cash_flow":   round(eps_q * rng.uniform(0.4, 0.9), 4),
                "roe":              round(rng.uniform(0.10, 0.35), 4),
                "roic":             round(rng.uniform(0.08, 0.30), 4),
                "debt_to_equity":   round(rng.uniform(0.2, 1.5), 4),
                "gross_margin":     round(rng.uniform(0.30, 0.70), 4),
                "operating_margin": round(rng.uniform(0.10, 0.35), 4),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_stocks():
    """Stocks metadata for 5 tickers."""
    return pd.DataFrame([
        {"ticker": "A", "company_name": "Alpha Corp",   "sector": "Technology",        "industry": "Software", "exchange": "NASDAQ", "currency": "USD", "is_active": 1, "shares_outstanding_m": 1000.0},
        {"ticker": "B", "company_name": "Beta Inc",     "sector": "Healthcare",         "industry": "Biotech",  "exchange": "NYSE",   "currency": "USD", "is_active": 1, "shares_outstanding_m": 500.0},
        {"ticker": "C", "company_name": "Gamma Ltd",    "sector": "Consumer Staples",   "industry": "Food",     "exchange": "NYSE",   "currency": "USD", "is_active": 1, "shares_outstanding_m": 2000.0},
        {"ticker": "D", "company_name": "Delta Group",  "sector": "Financials",         "industry": "Banks",    "exchange": "NASDAQ", "currency": "USD", "is_active": 1, "shares_outstanding_m": 800.0},
        {"ticker": "E", "company_name": "Epsilon Co",   "sector": "Energy",             "industry": "Oil",      "exchange": "NYSE",   "currency": "USD", "is_active": 1, "shares_outstanding_m": 600.0},
    ])


# ── Temporary database fixture ─────────────────────────────────────────────────

@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """
    Yields a path to a fresh temporary SQLite database.

    Resets the SQLAlchemy engine cache before and after so the module-level
    cached engine never points at the wrong database.
    """
    import src.database.db as db_module

    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    # Reset any cached engine so _resolve_db_path picks up the new env var
    db_module.reset_engine()

    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    initialize_db(db_path)
    apply_migrations()

    yield db_path

    db_module.reset_engine()


# ── Tiny features+targets dataset ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_ft():
    """
    Minimal features-and-targets DataFrame (3 tickers × 30 labelled months).
    All values are synthetic but structurally valid.
    """
    rng    = np.random.default_rng(11)
    tickers = ["A", "B", "C"]
    dates   = pd.date_range("2019-01-31", periods=30, freq="ME").strftime("%Y-%m-%d").tolist()
    rows    = []
    for date in dates:
        for ticker in tickers:
            excess = rng.normal(0.02, 0.15)
            rows.append({
                "feature_date":               date,
                "ticker":                     ticker,
                "twelve_month_momentum":      rng.uniform(-0.3, 0.5),
                "six_month_momentum":         rng.uniform(-0.2, 0.4),
                "current_pe_ratio":           rng.uniform(10, 50),
                "five_year_eps_growth":       rng.uniform(-0.1, 0.3),
                "roe":                        rng.uniform(0.05, 0.35),
                "volatility_12m":             rng.uniform(0.1, 0.5),
                "data_quality_score":         rng.uniform(60, 100),
                "future_12m_excess_return":   round(excess, 6),
                "winner":                     int(excess > 0),
            })
    return pd.DataFrame(rows)
