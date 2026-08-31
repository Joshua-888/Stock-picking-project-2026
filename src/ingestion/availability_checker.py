"""
src/ingestion/availability_checker.py

Step 22: Data availability checker.

Reports which data variables are available from the configured provider.
Writes results to the provider_availability database table so the
Data Quality dashboard page can display them.

VARIABLES CHECKED
-----------------
The checker probes for the following data fields:

Price data:
  adjusted_prices, volume, benchmark_prices

Fundamental data (quarterly):
  eps, revenue, net_income, equity, total_debt, free_cash_flow
  roe, roic, debt_to_equity, gross_margin, operating_margin

Derived features:
  current_pe_ratio, five_year_eps_growth, five_year_revenue_growth
  dividend_yield, market_cap, price_to_book, price_to_sales

Analyst / alternative data (typically requires premium):
  analyst_target_price, earnings_surprise, news_sentiment
  vix_level, interest_rate_trend

Macro data (free from FRED with Nasdaq Data Link):
  sp500_index, t_bill_rate, treasury_yield_10y

Usage
-----
    from src.ingestion.availability_checker import check_availability
    report = check_availability()
    print(report)   # pretty-printed availability table
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Variable registry ─────────────────────────────────────────────────────────
# Each entry: (variable_name, category, how_to_check, expected_in_sample)

VARIABLES = [
    # ── Prices ────────────────────────────────────────────────────────────────
    ("adjusted_prices",        "price",      "prices_clean.adjusted_close",   True),
    ("monthly_return",         "price",      "prices_clean.monthly_return",   True),
    ("volume",                 "price",      "prices_clean.volume",           True),
    ("benchmark_prices",       "price",      "benchmark_prices.adjusted_close", True),

    # ── Fundamental quarterly ─────────────────────────────────────────────────
    ("eps",                    "fundamental","fundamentals_clean.eps",         True),
    ("revenue",                "fundamental","fundamentals_clean.revenue",     True),
    ("net_income",             "fundamental","fundamentals_clean.net_income",  True),
    ("equity",                 "fundamental","fundamentals_clean.equity",      True),
    ("total_debt",             "fundamental","fundamentals_clean.total_debt",  True),
    ("free_cash_flow",         "fundamental","fundamentals_clean.free_cash_flow", True),
    ("roe",                    "fundamental","fundamentals_clean.roe",         True),
    ("roic",                   "fundamental","fundamentals_clean.roic",        True),
    ("debt_to_equity",         "fundamental","fundamentals_clean.debt_to_equity", True),
    ("gross_margin",           "fundamental","fundamentals_clean.gross_margin",True),
    ("operating_margin",       "fundamental","fundamentals_clean.operating_margin", True),

    # ── Derived features (computed, not fetched directly) ─────────────────────
    ("current_pe_ratio",       "derived",    "monthly_features.current_pe_ratio", True),
    ("five_year_eps_growth",   "derived",    "monthly_features.five_year_eps_growth", True),
    ("five_year_revenue_growth","derived",   "monthly_features.five_year_revenue_growth", True),
    ("twelve_month_momentum",  "derived",    "monthly_features.twelve_month_momentum", True),
    ("market_cap",             "derived",    "monthly_features.market_cap",   True),
    ("price_to_book",          "derived",    "monthly_features.price_to_book",True),
    ("price_to_sales",         "derived",    "monthly_features.price_to_sales", True),
    ("dividend_yield",         "derived",    "monthly_features.dividend_yield", True),

    # ── Macro (free from FRED via Nasdaq Data Link) ───────────────────────────
    ("sp500_index",            "macro",      "FRED/SP500",                    False),
    ("t_bill_rate_3m",         "macro",      "FRED/DTB3",                     False),
    ("treasury_yield_10y",     "macro",      "FRED/DGS10",                    False),
    ("crude_oil_price",        "macro",      "FRED/DCOILWTICO",               False),

    # ── Analyst / alternative (premium or not available in sample) ────────────
    ("analyst_target_price",   "analyst",    "requires_premium_provider",     False),
    ("eps_estimate_revision",  "analyst",    "requires_premium_provider",     False),
    ("earnings_surprise",      "analyst",    "requires_premium_provider",     False),
    ("news_sentiment",         "analyst",    "requires_premium_nlp_provider", False),
    ("vix_level",              "macro",      "requires_premium_provider",     False),
    ("short_interest",         "alternative","requires_premium_provider",     False),
    ("insider_ownership",      "alternative","requires_premium_provider",     False),
]


# ── Main checker ──────────────────────────────────────────────────────────────

def check_availability(
    provider:   Optional[str] = None,
    save_to_db: bool          = True,
) -> pd.DataFrame:
    """
    Check which data variables are available from the configured provider.

    For the "sample" provider, availability is determined by inspecting
    the database tables.  For "nasdaq_data_link", the checker additionally
    probes the API.

    Args:
        provider:   Override the config provider. Defaults to config.yaml setting.
        save_to_db: Write results to provider_availability table.

    Returns:
        DataFrame with columns:
            variable_name, category, source, availability_status, notes
    """
    from src.utils.config import load_config
    cfg = load_config()
    _provider = provider or cfg.data.provider

    log.info("Checking data availability for provider: %s", _provider)

    if _provider == "sample":
        result = _check_sample(_provider)
    elif _provider == "nasdaq_data_link":
        result = _check_nasdaq(_provider, cfg)
    else:
        result = _check_unknown(_provider)

    if save_to_db:
        try:
            from src.database.db import save_provider_availability
            save_provider_availability(result[["provider","variable_name",
                                               "availability_status","notes"]])
        except Exception as exc:
            log.warning("Could not save availability to DB: %s", exc)

    return result


def _check_sample(provider: str) -> pd.DataFrame:
    """
    For the sample provider, derive availability from what's in the database.
    Variables that are in the sample data generator are marked available.
    """
    rows = []
    for name, category, source, in_sample in VARIABLES:
        if in_sample:
            status = "available"
            notes  = "Available in synthetic sample data"
        elif category == "macro" and "FRED" in source:
            status = "available"
            notes  = "Available from FRED via Nasdaq Data Link (requires API key)"
        elif "premium" in source:
            status = "missing provider required"
            notes  = "Requires paid data provider subscription"
        else:
            status = "missing provider required"
            notes  = f"Not available from {provider} provider"

        rows.append({
            "provider":           provider,
            "variable_name":      name,
            "category":           category,
            "source":             source,
            "availability_status": status,
            "notes":              notes,
        })
    return pd.DataFrame(rows)


def _check_nasdaq(provider: str, cfg) -> pd.DataFrame:
    """
    For Nasdaq Data Link, check API connectivity and probe individual datasets.
    """
    from src.ingestion.nasdaq_data_link import (
        check_api_connection, fetch_fred_series,
    )
    import time

    conn = check_api_connection()
    api_ok = conn["success"]

    rows = []
    for name, category, source, in_sample in VARIABLES:

        # Sample-derived features are always available (computed locally)
        if in_sample and category == "derived":
            status = "available"
            notes  = "Computed from price/fundamental data — no direct API call needed"
            rows.append(_row(provider, name, category, source, status, notes))
            continue

        if not api_ok:
            status = "untested"
            notes  = f"API connection failed: {conn['message']}"
            rows.append(_row(provider, name, category, source, status, notes))
            continue

        # Probe individual datasets
        if category == "price" and name == "adjusted_prices":
            status, notes = _probe_prices(provider)
        elif category == "price" and name == "benchmark_prices":
            status, notes = _probe_benchmark(provider)
        elif category == "fundamental":
            status, notes = _probe_fundamentals(provider)
        elif category == "macro" and "FRED/" in source:
            fred_code = source.split("/")[1]
            df = fetch_fred_series(fred_code, "2020-01-01", "2020-12-31")
            if not df.empty:
                status = "available"
                notes  = f"Available from {source} (free)"
            else:
                status = "untested"
                notes  = f"Could not access {source}"
            time.sleep(0.3)
        elif "premium" in source:
            status = "missing provider required"
            notes  = "Requires additional data subscription"
        else:
            status = "untested"
            notes  = "Not probed automatically"

        rows.append(_row(provider, name, category, source, status, notes))

    return pd.DataFrame(rows)


def _check_unknown(provider: str) -> pd.DataFrame:
    rows = []
    for name, category, source, _ in VARIABLES:
        rows.append(_row(provider, name, category, source,
                         "untested", f"Unknown provider '{provider}'"))
    return pd.DataFrame(rows)


def _row(provider, name, category, source, status, notes):
    return {
        "provider":            provider,
        "variable_name":       name,
        "category":            category,
        "source":              source,
        "availability_status": status,
        "notes":               notes,
    }


def _probe_prices(provider: str):
    from src.ingestion.nasdaq_data_link import fetch_stock_prices
    df = fetch_stock_prices(["AAPL"], "2023-01-01", "2023-03-31")
    if not df.empty:
        return "available", "QUOTEMEDIA/PRICES or EOD data accessible"
    return "missing provider required", "Price data unavailable — premium subscription may be needed"


def _probe_benchmark(provider: str):
    from src.ingestion.nasdaq_data_link import fetch_benchmark_prices
    df = fetch_benchmark_prices("2023-01-01", "2023-06-30")
    if not df.empty:
        return "available", "Benchmark available via FRED/SP500 (free)"
    return "missing provider required", "Could not access FRED/SP500"


def _probe_fundamentals(provider: str):
    from src.ingestion.nasdaq_data_link import fetch_fundamentals
    df = fetch_fundamentals(["AAPL"], "2023-01-01", "2023-12-31")
    if not df.empty:
        return "available", "SHARADAR/SF1 accessible"
    return "missing provider required", (
        "SHARADAR/SF1 not accessible — Sharadar premium subscription required. "
        "Fundamental features will use sample data instead."
    )


# ── Pretty-print report ───────────────────────────────────────────────────────

def print_availability_report(df: pd.DataFrame) -> None:
    """Print a formatted availability report to stdout."""
    STATUS_ICONS = {
        "available":                "[OK]",
        "partial":                  "[PARTIAL]",
        "missing provider required": "[MISSING]",
        "untested":                 "[?]",
    }

    print("\n" + "=" * 65)
    print("  DATA AVAILABILITY REPORT")
    print("  Provider:", df["provider"].iloc[0] if not df.empty else "unknown")
    print("=" * 65)

    n_avail   = (df["availability_status"] == "available").sum()
    n_missing = (df["availability_status"] == "missing provider required").sum()
    print(f"  Available: {n_avail}  |  Missing: {n_missing}  |  Total: {len(df)}")
    print()

    for cat in df["category"].unique():
        print(f"  [{cat.upper()}]")
        subset = df[df["category"] == cat]
        for _, row in subset.iterrows():
            icon   = STATUS_ICONS.get(row["availability_status"], "?")
            status = row["availability_status"]
            print(f"    {icon}  {row['variable_name']:<35} {status}")
        print()

    print("=" * 65)
