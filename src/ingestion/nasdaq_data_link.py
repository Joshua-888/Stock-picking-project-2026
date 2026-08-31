"""
src/ingestion/nasdaq_data_link.py

Step 21: Nasdaq Data Link data provider.

Provides standardised DataFrames matching the database schemas for
prices_clean, benchmark_prices, and fundamentals_clean.

API KEY
-------
The key is read from the environment variable NASDAQ_DATA_LINK_API_KEY.
Never hard-code it.  Set it in your .env file:

    NASDAQ_DATA_LINK_API_KEY=your_key_here

Get a free key at: https://data.nasdaq.com/sign-up

FREE vs PREMIUM DATASETS
-------------------------
Free tier (works with any valid API key):
  • FRED/SP500        — S&P 500 index level (monthly)
  • FRED/DTB3         — 3-month T-bill rate
  • FRED/DGS10        — 10-year Treasury yield
  • EOD/{ticker}      — End-of-day prices (some historical data, limited tickers)

Premium datasets (require paid Nasdaq Data Link subscription):
  • QUOTEMEDIA/PRICES — Full adjusted price history for all tickers
  • SHARADAR/SF1      — Fundamental data (EPS, revenue, margins, etc.)
  • SHARADAR/SEP      — Adjusted prices from Sharadar
  • SHARADAR/DAILY    — Daily metrics

This module tries free datasets first and clearly flags when premium
access is needed.  Stocks with no price data are excluded with a
data quality warning.

RATE LIMITS
-----------
Free tier: 50 API calls per day.  The module caches responses to
data/raw/ so the same data is never fetched twice.

Usage
-----
    from src.ingestion.nasdaq_data_link import (
        fetch_stock_prices, fetch_benchmark_prices,
        fetch_fundamentals, check_api_connection,
    )

    prices = fetch_stock_prices(["AAPL","MSFT"], "2020-01-01", "2024-12-31")
    bench  = fetch_benchmark_prices("2020-01-01", "2024-12-31")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.db import save_data_quality_log
from src.utils.logging import get_logger

log = get_logger(__name__)

# Raw response cache directory
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Rate-limit guard: seconds to wait between API calls
_CALL_DELAY = 0.3


# ── API setup ─────────────────────────────────────────────────────────────────

def _get_ndl():
    """
    Return the nasdaqdatalink module, configured with the API key.
    Raises ImportError if the package is not installed.
    Raises ValueError if the API key is not set.
    """
    try:
        import nasdaqdatalink as ndl
    except ImportError:
        raise ImportError(
            "nasdaqdatalink package not installed. "
            "Run: pip install Nasdaq-Data-Link"
        )

    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "NASDAQ_DATA_LINK_API_KEY is not set. "
            "Add it to your .env file:\n"
            "  NASDAQ_DATA_LINK_API_KEY=your_key_here\n"
            "Get a free key at https://data.nasdaq.com/sign-up"
        )
    ndl.ApiConfig.api_key = api_key
    return ndl


def check_api_connection() -> dict:
    """
    Test whether the API key is valid and the connection works.

    Returns a dict with keys: success (bool), message (str), api_key_set (bool).
    """
    api_key_set = bool(os.getenv("NASDAQ_DATA_LINK_API_KEY"))
    if not api_key_set:
        return {"success": False, "api_key_set": False,
                "message": "NASDAQ_DATA_LINK_API_KEY not set in environment"}
    try:
        ndl = _get_ndl()
        # Test with a tiny free FRED dataset
        df = ndl.get("FRED/SP500", rows=1)
        return {"success": True, "api_key_set": True,
                "message": "Connection successful (FRED/SP500 accessible)"}
    except Exception as exc:
        return {"success": False, "api_key_set": api_key_set,
                "message": f"Connection failed: {exc}"}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = key.replace("/", "_").replace(":", "_")
    return _RAW_DIR / f"ndl_{safe}.json"


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if path.exists():
        try:
            return pd.read_json(path)
        except Exception:
            pass
    return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    try:
        _cache_path(key).write_text(df.to_json(), encoding="utf-8")
    except Exception as exc:
        log.debug("Cache write failed for %s: %s", key, exc)


# ── Benchmark prices (free — FRED S&P 500) ────────────────────────────────────

def fetch_benchmark_prices(
    start_date: str,
    end_date:   str,
    ticker:     str = "SPY",
) -> pd.DataFrame:
    """
    Fetch benchmark (S&P 500) monthly prices from FRED.

    Uses FRED/SP500 (monthly S&P 500 index level), which is freely available.
    Converts index levels to adjusted_close and computes monthly_return.

    Returns DataFrame matching benchmark_prices table schema:
        benchmark_ticker, date, adjusted_close, monthly_return
    """
    cache_key = f"FRED_SP500_{start_date}_{end_date}"
    cached    = _load_cache(cache_key)
    if cached is not None:
        log.debug("Benchmark prices loaded from cache.")
        return cached

    try:
        ndl = _get_ndl()
        time.sleep(_CALL_DELAY)

        df = ndl.get(
            "FRED/SP500",
            start_date=start_date,
            end_date=end_date,
            collapse="monthly",
        )
        df = df.reset_index()
        df.columns = ["date", "adjusted_close"]
        df["date"]            = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["benchmark_ticker"] = ticker
        df["adjusted_close"]  = df["adjusted_close"].astype(float)
        df["monthly_return"]  = df["adjusted_close"].pct_change()

        df = df[["benchmark_ticker", "date", "adjusted_close", "monthly_return"]]
        df = df.sort_values("date").reset_index(drop=True)

        _save_cache(cache_key, df)
        log.info("Fetched benchmark prices: %d rows from FRED/SP500", len(df))
        return df

    except Exception as exc:
        log.error("Failed to fetch benchmark prices: %s", exc)
        save_data_quality_log(
            issue_type="api_error", severity="error",
            provider="nasdaq_data_link",
            message=f"Benchmark fetch failed: {exc}",
        )
        return pd.DataFrame()


# ── Stock prices ──────────────────────────────────────────────────────────────

def fetch_stock_prices(
    tickers:    list[str],
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetch adjusted monthly stock prices for a list of tickers.

    Strategy:
      1. Try QUOTEMEDIA/PRICES table (requires auth, may be free-tier accessible)
      2. Fall back to EOD/{ticker} time-series (older dataset, limited coverage)
      3. Log a warning for any ticker where data is unavailable

    Returns DataFrame matching prices_clean table schema:
        ticker, date, adjusted_close, monthly_return, volume, data_quality_flag
    """
    all_rows: list[pd.DataFrame] = []

    for ticker in tickers:
        df = _fetch_single_ticker_prices(ticker, start_date, end_date)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        log.warning("No price data fetched for any ticker.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    log.info("Fetched prices: %d rows for %d tickers", len(combined), len(all_rows))
    return combined


def _fetch_single_ticker_prices(
    ticker:     str,
    start_date: str,
    end_date:   str,
) -> Optional[pd.DataFrame]:
    """Fetch prices for one ticker, trying multiple datasets."""
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        ndl  = _get_ndl()
        time.sleep(_CALL_DELAY)

        # Strategy 1: QUOTEMEDIA/PRICES table
        try:
            raw = ndl.get_table(
                "QUOTEMEDIA/PRICES",
                ticker=ticker,
                date={"gte": start_date, "lte": end_date},
                qopts={"columns": ["date","adj_close","adj_volume"]},
            )
            if not raw.empty:
                return _standardise_prices(raw, ticker, "adj_close", "adj_volume", cache_key)
        except Exception:
            pass

        # Strategy 2: EOD time-series dataset
        try:
            raw = ndl.get(
                f"EOD/{ticker}",
                start_date=start_date,
                end_date=end_date,
                collapse="monthly",
                column_index=11,   # Adj_Close column in EOD
            )
            if raw is not None and not raw.empty:
                raw = raw.reset_index()
                raw.columns = ["date", "Adj_Close"]
                raw["Volume"] = float("nan")
                return _standardise_prices(raw, ticker, "Adj_Close", "Volume", cache_key)
        except Exception:
            pass

        log.warning("No price data found for %s on Nasdaq Data Link", ticker)
        save_data_quality_log(
            issue_type="missing_price", severity="warning",
            ticker=ticker, provider="nasdaq_data_link",
            message=f"No price data available for {ticker} — "
                    "QUOTEMEDIA and EOD datasets both unavailable. "
                    "This ticker may require a premium Nasdaq Data Link subscription.",
        )
        return None

    except Exception as exc:
        log.error("Price fetch failed for %s: %s", ticker, exc)
        save_data_quality_log(
            issue_type="api_error", severity="error",
            ticker=ticker, provider="nasdaq_data_link",
            message=f"Price fetch error: {exc}",
        )
        return None


def _standardise_prices(
    df:        pd.DataFrame,
    ticker:    str,
    price_col: str,
    vol_col:   str,
    cache_key: str,
) -> pd.DataFrame:
    """Convert raw price DataFrame to prices_clean schema."""
    df = df.copy()
    df["date"]          = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"]        = ticker
    df["adjusted_close"] = pd.to_numeric(df[price_col], errors="coerce")
    df["volume"]        = pd.to_numeric(df.get(vol_col, float("nan")), errors="coerce") if vol_col in df.columns else float("nan")
    df["monthly_return"] = df["adjusted_close"].pct_change()
    df["data_quality_flag"] = "ok"
    df.loc[df["adjusted_close"].isna(), "data_quality_flag"] = "missing"
    df.loc[df["adjusted_close"] <= 0,   "data_quality_flag"] = "suspect"

    out = df[["ticker","date","adjusted_close","monthly_return","volume","data_quality_flag"]].copy()
    out = out.sort_values("date").reset_index(drop=True)
    _save_cache(cache_key, out)
    return out


# ── Fundamental data ──────────────────────────────────────────────────────────

def fetch_fundamentals(
    tickers:    list[str],
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetch fundamental data (EPS, revenue, margins) for a list of tickers.

    Uses SHARADAR/SF1 which requires a Sharadar premium subscription.
    Returns empty DataFrame with a clear warning if not accessible.

    Returns DataFrame matching fundamentals_clean table schema.
    """
    cache_key = f"fundamentals_{'_'.join(sorted(tickers)[:5])}_{start_date}_{end_date}"
    cached    = _load_cache(cache_key)
    if cached is not None:
        log.debug("Fundamentals loaded from cache.")
        return cached

    try:
        ndl = _get_ndl()
        time.sleep(_CALL_DELAY)

        rows = []
        for ticker in tickers:
            try:
                raw = ndl.get_table(
                    "SHARADAR/SF1",
                    ticker=ticker,
                    calendardate={"gte": start_date, "lte": end_date},
                    dimension="ARQ",   # As-Reported Quarterly
                )
                if raw.empty:
                    continue

                for _, row in raw.iterrows():
                    rows.append({
                        "ticker":           ticker,
                        "fiscal_period":    _quarter_label(str(row.get("calendardate",""))),
                        "fiscal_date":      str(row.get("calendardate",""))[:10],
                        "report_date":      str(row.get("datekey",""))[:10],
                        "revenue":          _safe_float(row.get("revenue")),
                        "eps":              _safe_float(row.get("eps")),
                        "net_income":       _safe_float(row.get("netinc")),
                        "equity":           _safe_float(row.get("equity")),
                        "total_debt":       _safe_float(row.get("debt")),
                        "free_cash_flow":   _safe_float(row.get("fcf")),
                        "roe":              _safe_float(row.get("roe")),
                        "roic":             _safe_float(row.get("roic")),
                        "debt_to_equity":   _safe_float(row.get("de")),
                        "gross_margin":     _safe_float(row.get("grossmargin")),
                        "operating_margin": _safe_float(row.get("ebitdamargin")),
                    })
                time.sleep(_CALL_DELAY)
            except Exception as exc:
                log.warning("SHARADAR/SF1 unavailable for %s: %s", ticker, exc)
                save_data_quality_log(
                    issue_type="missing_fundamental", severity="warning",
                    ticker=ticker, provider="nasdaq_data_link",
                    message=f"SHARADAR/SF1 unavailable — premium subscription required: {exc}",
                )

        if not rows:
            log.warning(
                "No fundamental data fetched. SHARADAR/SF1 requires a premium "
                "Nasdaq Data Link subscription. Falling back to sample data for fundamentals."
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        _save_cache(cache_key, df)
        log.info("Fetched fundamentals: %d rows for %d tickers", len(df), len(set(r["ticker"] for r in rows)))
        return df

    except Exception as exc:
        log.error("Fundamentals fetch failed: %s", exc)
        return pd.DataFrame()


# ── FRED macro data ───────────────────────────────────────────────────────────

def fetch_fred_series(
    series_code: str,
    start_date:  str,
    end_date:    str,
    collapse:    str = "monthly",
) -> pd.DataFrame:
    """
    Fetch a FRED time series (free with any valid API key).

    Common series:
        SP500   — S&P 500 Index
        DTB3    — 3-Month T-Bill Rate
        DGS10   — 10-Year Treasury Yield
        DEXUSEU — USD/EUR Exchange Rate
        DCOILWTICO — WTI Crude Oil Price

    Returns DataFrame with columns: date, value.
    """
    cache_key = f"FRED_{series_code}_{start_date}_{end_date}_{collapse}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        ndl = _get_ndl()
        time.sleep(_CALL_DELAY)
        df  = ndl.get(
            f"FRED/{series_code}",
            start_date=start_date,
            end_date=end_date,
            collapse=collapse,
        )
        df       = df.reset_index()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        _save_cache(cache_key, df)
        log.info("Fetched FRED/%s: %d rows", series_code, len(df))
        return df
    except Exception as exc:
        log.error("FRED/%s fetch failed: %s", series_code, exc)
        return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quarter_label(date_str: str) -> str:
    try:
        month = pd.Timestamp(date_str).month
        return f"Q{(month - 1) // 3 + 1}"
    except Exception:
        return "Q?"


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        import math
        return f if not (math.isnan(f) or math.isinf(f)) else None
    except (TypeError, ValueError):
        return None
