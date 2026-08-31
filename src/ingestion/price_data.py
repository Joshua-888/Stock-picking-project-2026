"""
src/ingestion/price_data.py

Normalises raw price data from any provider into the standard
prices_clean DataFrame format used by the rest of the pipeline.

Every provider module returns prices in its own format.  This module
converts them to a single canonical schema:

    ticker | date | adjusted_close | monthly_return | volume | data_quality_flag

Usage
-----
    from src.ingestion.price_data import standardize_prices, compute_monthly_returns

    clean = standardize_prices(raw_df, ticker="AAPL", date_col="Date",
                                price_col="Adj. Close")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# Flags used in data_quality_flag column
FLAG_OK              = "ok"
FLAG_SPLIT_ADJUSTED  = "split_adjusted"
FLAG_ESTIMATED       = "estimated"
FLAG_SUSPECT         = "suspect"          # e.g. price spike or zero volume
FLAG_MISSING         = "missing"


def standardize_prices(
    df: pd.DataFrame,
    *,
    ticker: str,
    date_col: str           = "date",
    price_col: str          = "adjusted_close",
    volume_col: str | None  = "volume",
) -> pd.DataFrame:
    """
    Convert a raw price DataFrame into the canonical prices_clean format.

    Steps
    -----
    1. Rename and select the required columns.
    2. Parse dates to ISO-8601 strings (YYYY-MM-DD).
    3. Coerce prices to float; flag negative or zero prices.
    4. Resample to month-end frequency (last observation per month).
    5. Compute month-over-month returns.
    6. Assign data_quality_flag.

    Args:
        df:         Raw DataFrame from any provider.
        ticker:     Ticker symbol to stamp on every row.
        date_col:   Name of the date column in df.
        price_col:  Name of the adjusted close price column in df.
        volume_col: Name of the volume column (None to skip).

    Returns:
        DataFrame with columns:
          ticker, date, adjusted_close, monthly_return, volume, data_quality_flag
    """
    work = df.copy()
    work.rename(columns={date_col: "date", price_col: "adjusted_close"}, inplace=True)
    if volume_col and volume_col in work.columns:
        work.rename(columns={volume_col: "volume"}, inplace=True)
    else:
        work["volume"] = float("nan")

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work.dropna(subset=["date"], inplace=True)
    work.set_index("date", inplace=True)
    work.sort_index(inplace=True)

    work["adjusted_close"] = pd.to_numeric(work["adjusted_close"], errors="coerce")

    # Resample to month-end (take last valid price of each month)
    monthly = work[["adjusted_close", "volume"]].resample("ME").last()
    monthly["ticker"] = ticker
    monthly["data_quality_flag"] = FLAG_OK

    # Flag suspect prices
    monthly.loc[monthly["adjusted_close"] <= 0, "data_quality_flag"] = FLAG_SUSPECT
    monthly.loc[monthly["adjusted_close"].isna(), "data_quality_flag"] = FLAG_MISSING

    # Monthly return: (price_t / price_{t-1}) - 1
    monthly["monthly_return"] = monthly["adjusted_close"].pct_change()

    monthly.reset_index(inplace=True)
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")

    cols = ["ticker", "date", "adjusted_close", "monthly_return", "volume", "data_quality_flag"]
    return monthly[cols].reset_index(drop=True)


def compute_monthly_returns(prices: pd.Series) -> pd.Series:
    """
    Compute month-over-month percentage returns from a price series.

    Args:
        prices: pd.Series of adjusted close prices indexed by date.

    Returns:
        pd.Series of monthly returns (first value is NaN).
    """
    return prices.pct_change()


def validate_prices(df: pd.DataFrame, ticker: str) -> list[str]:
    """
    Run sanity checks on a prices_clean DataFrame for one ticker.

    Returns a list of warning strings (empty list = clean).
    """
    warnings: list[str] = []

    if df.empty:
        warnings.append(f"{ticker}: no price data")
        return warnings

    n_missing = df["adjusted_close"].isna().sum()
    if n_missing > 0:
        warnings.append(f"{ticker}: {n_missing} missing adjusted_close values")

    n_zero = (df["adjusted_close"] <= 0).sum()
    if n_zero > 0:
        warnings.append(f"{ticker}: {n_zero} zero-or-negative prices")

    # Flag extreme monthly returns (>80% gain or loss in one month)
    extreme = df["monthly_return"].abs() > 0.80
    if extreme.any():
        n = extreme.sum()
        warnings.append(f"{ticker}: {n} extreme monthly return(s) — possible split or data error")

    # Check for duplicate dates
    n_dupes = df["date"].duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"{ticker}: {n_dupes} duplicate date rows")

    return warnings


def prices_to_returns_matrix(
    prices_df: pd.DataFrame,
    min_history_months: int = 12,
) -> pd.DataFrame:
    """
    Pivot a long-format prices_clean DataFrame into a wide returns matrix.

    Rows = dates, columns = tickers.
    Tickers with fewer than min_history_months observations are excluded.

    Useful for cross-sectional feature calculations like sector momentum.

    Args:
        prices_df:            Long-format prices_clean DataFrame.
        min_history_months:   Minimum non-null months required per ticker.

    Returns:
        Wide DataFrame: index=date, columns=ticker, values=monthly_return.
    """
    pivot = prices_df.pivot(index="date", columns="ticker", values="monthly_return")
    pivot = pivot.sort_index()

    # Drop tickers with insufficient history
    sufficient = pivot.count() >= min_history_months
    dropped = (~sufficient).sum()
    if dropped > 0:
        log.debug("Dropping %d tickers with < %d months of return history",
                  dropped, min_history_months)
    return pivot.loc[:, sufficient]
