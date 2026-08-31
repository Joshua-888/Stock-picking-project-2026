"""
src/ingestion/benchmark.py

Provides benchmark price data (default: SPY / S&P 500).

Routes to the correct data source based on the active provider,
then normalises the output to the benchmark_prices table schema:

    benchmark_ticker | date | adjusted_close | monthly_return

Usage
-----
    from src.ingestion.benchmark import get_benchmark_data
    bench_df = get_benchmark_data(cfg)
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def get_benchmark_data(
    cfg=None,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Fetch benchmark price data for the configured benchmark ticker.

    Dispatches to the appropriate provider based on cfg.data.provider:
      - "sample"             → synthetic SPY data from sample_data.py
      - "nasdaq_data_link"   → real data via nasdaq_data_link.py (Step 21)

    Returns a DataFrame with columns:
        benchmark_ticker, date, adjusted_close, monthly_return

    Args:
        cfg:        AppConfig object (loaded from config if None).
        start_date: Override start date (YYYY-MM-DD).
        end_date:   Override end date (YYYY-MM-DD).
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    provider         = cfg.data.provider
    benchmark_ticker = cfg.data.benchmark_ticker
    _start = start_date or cfg.data.start_date
    _end   = end_date   or pd.Timestamp.today().strftime("%Y-%m-%d")

    log.info("Fetching benchmark data: ticker=%s provider=%s", benchmark_ticker, provider)

    if provider == "sample":
        return _get_sample_benchmark(_start, _end)

    if provider == "nasdaq_data_link":
        # Implemented in Step 21
        from src.ingestion.nasdaq_data_link import fetch_benchmark_prices
        return fetch_benchmark_prices(benchmark_ticker, _start, _end)

    raise ValueError(
        f"Unknown data provider '{provider}'. "
        "Set DATA_PROVIDER=sample or DATA_PROVIDER=nasdaq_data_link in .env"
    )


def _get_sample_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    from src.ingestion.sample_data import generate_benchmark_prices
    return generate_benchmark_prices(start_date=start_date, end_date=end_date)


def compute_benchmark_returns(bench_df: pd.DataFrame) -> pd.Series:
    """
    Return a monthly_return Series indexed by date string.

    Convenience wrapper for use in feature engineering.

    Args:
        bench_df: benchmark_prices DataFrame.

    Returns:
        pd.Series indexed by date (str), values = monthly returns.
    """
    s = bench_df.set_index("date")["monthly_return"].copy()
    s.name = "benchmark_return"
    return s


def get_benchmark_cumulative_return(
    bench_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> float | None:
    """
    Calculate the total benchmark return between two dates (inclusive).

    Returns None if there is insufficient data for either date.

    Args:
        bench_df:   benchmark_prices DataFrame.
        start_date: YYYY-MM-DD
        end_date:   YYYY-MM-DD

    Returns:
        Total return as a decimal (e.g. 0.15 = 15%) or None.
    """
    prices = (
        bench_df
        .set_index("date")["adjusted_close"]
        .sort_index()
    )

    dates_available = prices.index.tolist()

    # Find nearest available date on or after start_date
    start_candidates = [d for d in dates_available if d >= start_date]
    end_candidates   = [d for d in dates_available if d <= end_date]

    if not start_candidates or not end_candidates:
        return None

    p_start = prices[start_candidates[0]]
    p_end   = prices[end_candidates[-1]]

    if p_start == 0 or pd.isna(p_start) or pd.isna(p_end):
        return None

    return float(p_end / p_start - 1.0)
