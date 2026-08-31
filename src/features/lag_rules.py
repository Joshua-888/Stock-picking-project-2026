"""
src/features/lag_rules.py

Reporting-lag enforcement for look-ahead bias prevention.

Rule
----
At each prediction date (feature_date), the model may ONLY use information
that would realistically have been available on or before that date.

Financial statements are not published immediately after the period ends:
  • Quarterly reports (10-Q): available ~45 days after quarter end
  • Annual reports  (10-K):  available ~90 days after fiscal year end

These lags are configured in config.yaml (reporting_lags section) and
applied here when filtering fundamental data before feature calculation.

Usage
-----
    from src.features.lag_rules import filter_fundamentals_lag_safe

    # Only get fundamentals whose report_date is on or before feature_date
    available = filter_fundamentals_lag_safe(all_fundamentals, "2024-01-31")
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def filter_fundamentals_lag_safe(
    fundamentals_df: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Return only fundamental rows that would have been published by as_of_date.

    Filters on the report_date column, which represents when the filing
    was actually (or estimated to be) available to the public.

    Args:
        fundamentals_df: fundamentals_clean DataFrame (all tickers, all dates).
        as_of_date:      The prediction date (YYYY-MM-DD).

    Returns:
        Filtered DataFrame containing only rows where report_date <= as_of_date.
    """
    if "report_date" not in fundamentals_df.columns:
        log.warning("fundamentals_df missing 'report_date' column — no lag filtering applied")
        return fundamentals_df.copy()

    mask = fundamentals_df["report_date"] <= as_of_date
    filtered = fundamentals_df[mask].copy()

    n_total    = len(fundamentals_df)
    n_filtered = len(filtered)
    log.debug(
        "Lag filter as_of=%s: kept %d / %d fundamental rows",
        as_of_date, n_filtered, n_total,
    )
    return filtered


def get_ttm_fundamentals(
    fundamentals_df: pd.DataFrame,
    ticker: str,
    as_of_date: str,
    n_quarters: int = 4,
) -> pd.DataFrame:
    """
    Return the most recent n_quarters of fundamentals for a ticker,
    already filtered to be lag-safe as of as_of_date.

    "TTM" = Trailing Twelve Months = last 4 quarters.

    Args:
        fundamentals_df: Lag-safe filtered fundamentals (already filtered).
        ticker:          Stock ticker.
        as_of_date:      The prediction date (used for log messages).
        n_quarters:      Number of quarters to return (default 4 = TTM).

    Returns:
        DataFrame with up to n_quarters rows, sorted by fiscal_date ascending.
        Empty DataFrame if no data available.
    """
    ticker_data = (
        fundamentals_df[fundamentals_df["ticker"] == ticker]
        .sort_values("fiscal_date")
    )
    return ticker_data.tail(n_quarters)


def get_historical_fundamentals(
    fundamentals_df: pd.DataFrame,
    ticker: str,
    n_years_ago: int,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Return the TTM fundamentals from approximately n_years_ago.

    Used for multi-year growth rate calculations (e.g., 5-year EPS CAGR).
    Looks for quarterly data within a ±6-month window around the target date.

    Args:
        fundamentals_df: Lag-safe filtered fundamentals.
        ticker:          Stock ticker.
        n_years_ago:     How many years back to look (e.g., 5).
        as_of_date:      The current prediction date.

    Returns:
        DataFrame with up to 4 rows representing TTM fundamentals from that period.
    """
    target = pd.Timestamp(as_of_date) - pd.DateOffset(years=n_years_ago)
    window_start = (target - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    window_end   = (target + pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    ticker_data = fundamentals_df[fundamentals_df["ticker"] == ticker]
    window_data = ticker_data[
        (ticker_data["fiscal_date"] >= window_start) &
        (ticker_data["fiscal_date"] <= window_end)
    ].sort_values("fiscal_date")

    return window_data.tail(4)


def staleness_days(
    fundamentals_df: pd.DataFrame,
    ticker: str,
    as_of_date: str,
) -> int | None:
    """
    Return how many days old the most recent available fundamental report is.

    A large value (e.g., >180 days) indicates stale data and should
    lower the data_quality_score.

    Returns None if no fundamental data exists for the ticker.
    """
    ticker_data = fundamentals_df[fundamentals_df["ticker"] == ticker]
    if ticker_data.empty:
        return None

    latest_report = ticker_data["report_date"].max()
    delta = (pd.Timestamp(as_of_date) - pd.Timestamp(latest_report)).days
    return max(delta, 0)
