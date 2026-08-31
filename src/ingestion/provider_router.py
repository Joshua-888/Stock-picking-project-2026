"""
src/ingestion/provider_router.py

Central dispatcher: routes data requests to the correct provider module.

Adding a new provider
---------------------
1. Create src/ingestion/<provider_name>.py
2. Add a branch to each _get_* function below
3. Add the provider name to SUPPORTED_PROVIDERS

Usage
-----
    from src.ingestion.provider_router import get_all_data
    data = get_all_data(cfg)
    prices       = data["prices_clean"]
    benchmark    = data["benchmark_prices"]
    fundamentals = data["fundamentals_clean"]
    stocks       = data["stocks"]
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED_PROVIDERS = {"sample", "nasdaq_data_link", "yfinance"}


def get_all_data(
    cfg=None,
    start_date:      str | None       = None,
    end_date:        str | None       = None,
    ticker_override: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all data needed by the pipeline from the configured provider.

    Returns a dict with keys:
        "stocks"              pd.DataFrame  → stocks table
        "prices_clean"        pd.DataFrame  → prices_clean table
        "benchmark_prices"    pd.DataFrame  → benchmark_prices table
        "fundamentals_clean"  pd.DataFrame  → fundamentals_clean table

    Args:
        cfg:        AppConfig (loaded automatically if None).
        start_date: Override start date (YYYY-MM-DD).
        end_date:   Override end date (YYYY-MM-DD).
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    provider = cfg.data.provider
    _validate_provider(provider)

    log.info("Provider router: using '%s'", provider)

    if provider == "sample":
        return _get_sample_data(cfg, start_date, end_date)

    if provider == "yfinance":
        return _get_yfinance_data(cfg, start_date, end_date, ticker_override)

    if provider == "nasdaq_data_link":
        return _get_nasdaq_data(cfg, start_date, end_date)

    raise ValueError(f"Provider '{provider}' is not implemented.")


def get_price_data(
    tickers: list[str],
    cfg=None,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Fetch prices_clean DataFrame for a list of tickers.
    """
    data = get_all_data(cfg, start_date, end_date)
    prices = data["prices_clean"]
    return prices[prices["ticker"].isin(tickers)].copy()


def get_fundamental_data(
    tickers: list[str],
    cfg=None,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Fetch fundamentals_clean DataFrame for a list of tickers.
    """
    data = get_all_data(cfg, start_date, end_date)
    funds = data["fundamentals_clean"]
    return funds[funds["ticker"].isin(tickers)].copy()


def get_benchmark_data(
    cfg=None,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Fetch benchmark_prices DataFrame.
    """
    from src.ingestion.benchmark import get_benchmark_data as _get_bench
    return _get_bench(cfg, start_date, end_date)


# ── Provider implementations ─────────────────────────────────────────────────

def _get_yfinance_data(cfg, start_date, end_date, ticker_override=None) -> dict[str, pd.DataFrame]:
    """Fetch real market data from Yahoo Finance — free, no API key needed."""
    from src.ingestion.yfinance_provider import fetch_prices, fetch_benchmark, fetch_fundamentals
    from src.ingestion.sample_data import generate_stocks_metadata, _get_tickers_config

    _start = start_date or cfg.data.start_date
    _end   = end_date   or pd.Timestamp.today().strftime("%Y-%m-%d")

    # Use override ticker list (e.g. Core + Hidden Gems) if provided
    if ticker_override:
        tickers = ticker_override
    else:
        tickers = [t.ticker for t in cfg.tickers.universe]

    # Build stocks metadata from config; add any extra tickers from the hidden gems pool
    config_tickers = _get_tickers_config(cfg)
    config_set = {t["ticker"] for t in config_tickers}

    # Load the hidden gems pool so we have metadata for extra tickers
    try:
        import yaml
        from pathlib import Path as _Path
        _ty = _Path(__file__).resolve().parents[2] / "config" / "tickers.yaml"
        _pool = (yaml.safe_load(_ty.read_text(encoding="utf-8")) or {}).get("hidden_gems_pool", [])
        pool_meta = {t["ticker"]: t for t in _pool}
    except Exception:
        pool_meta = {}

    all_meta = list(config_tickers)
    for t in tickers:
        if t not in config_set and t in pool_meta:
            all_meta.append(pool_meta[t])

    stocks_df = generate_stocks_metadata(all_meta, _start, _end)
    prices_df = fetch_prices(tickers, _start, _end)
    bench_df  = fetch_benchmark(cfg.data.benchmark_ticker, _start, _end)
    funds_df  = fetch_fundamentals(tickers, _start)

    if prices_df.empty:
        raise RuntimeError("Yahoo Finance returned no price data. Check your internet connection.")

    return {
        "stocks":             stocks_df,
        "prices_clean":       prices_df,
        "benchmark_prices":   bench_df,
        "fundamentals_clean": funds_df,
    }


def _get_sample_data(cfg, start_date, end_date) -> dict[str, pd.DataFrame]:
    from src.ingestion.sample_data import load_sample_data
    return load_sample_data(cfg=cfg, end_date=end_date)


def _get_nasdaq_data(cfg, start_date, end_date) -> dict[str, pd.DataFrame]:
    """
    Fetch all data from Nasdaq Data Link.

    Prices and benchmark are fetched via the API.
    Fundamentals fall back to sample data if SHARADAR/SF1 is unavailable
    (premium subscription required).
    """
    from src.ingestion.nasdaq_data_link import (
        fetch_stock_prices, fetch_benchmark_prices, fetch_fundamentals,
    )
    from src.ingestion.sample_data import load_sample_data

    _start = start_date or cfg.data.start_date
    _end   = end_date   or pd.Timestamp.today().strftime("%Y-%m-%d")

    tickers = [t.ticker for t in cfg.tickers.universe]

    # Stocks metadata (always from config — not provider-dependent)
    from src.ingestion.sample_data import generate_stocks_metadata, _get_tickers_config
    stocks_df = generate_stocks_metadata(_get_tickers_config(cfg), _start, _end)

    # Prices
    prices_df = fetch_stock_prices(tickers, _start, _end)
    if prices_df.empty:
        log.warning("No prices from Nasdaq Data Link — falling back to sample prices")
        sample    = load_sample_data(cfg)
        prices_df = sample["prices_clean"]

    # Benchmark
    bench_df = fetch_benchmark_prices(_start, _end, cfg.data.benchmark_ticker)
    if bench_df.empty:
        log.warning("No benchmark from Nasdaq Data Link — falling back to sample benchmark")
        sample   = load_sample_data(cfg)
        bench_df = sample["benchmark_prices"]

    # Fundamentals (premium — fall back gracefully)
    funds_df = fetch_fundamentals(tickers, _start, _end)
    if funds_df.empty:
        log.warning(
            "No fundamentals from Nasdaq Data Link (SHARADAR/SF1 requires premium). "
            "Using synthetic fundamentals from sample data."
        )
        sample   = load_sample_data(cfg)
        funds_df = sample["fundamentals_clean"]

    return {
        "stocks":             stocks_df,
        "prices_clean":       prices_df,
        "benchmark_prices":   bench_df,
        "fundamentals_clean": funds_df,
    }


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_provider(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}. "
            "Set DATA_PROVIDER in your .env file."
        )


def list_providers() -> list[str]:
    """Return the list of currently supported provider names."""
    return sorted(SUPPORTED_PROVIDERS)
