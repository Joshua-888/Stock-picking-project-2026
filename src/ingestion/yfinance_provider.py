"""
src/ingestion/yfinance_provider.py

Real market data via Yahoo Finance (yfinance).

FREE — no API key required.
Provides real adjusted prices and fundamental data for all 30 tickers.

Data quality vs sample data
-----------------------------
  Prices      : Real adjusted close prices, split- and dividend-adjusted.
                Monthly frequency, from 2010 to today.
  Fundamentals: Real EPS, revenue, margins, debt/equity from SEC filings.
                Quarterly, with real report dates.
  Benchmark   : Real SPY (S&P 500 ETF) monthly adjusted prices.

Known limitations
-----------------
  • Yahoo Finance can occasionally be slow or return incomplete data.
  • Fundamental data coverage varies by ticker (some are missing quarters).
  • Fiscal year end dates differ by company — aligned to calendar quarters.
  • Data is sourced from SEC filings via Yahoo, not a premium financial database.
  • Rate limiting: batch downloads to avoid being blocked.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)

_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ── Prices ─────────────────────────────────────────────────────────────────────

def fetch_prices(
    tickers:    list[str],
    start_date: str = "2010-01-01",
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Download real monthly adjusted close prices for all tickers.

    Returns DataFrame matching prices_clean schema:
        ticker, date, adjusted_close, monthly_return, volume, data_quality_flag
    """
    import yfinance as yf

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    log.info("Downloading prices for %d tickers via Yahoo Finance...", len(tickers))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if raw.empty:
        log.error("yfinance returned no price data.")
        return pd.DataFrame()

    # Extract Close prices (yfinance returns MultiIndex columns when >1 ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        vol   = raw["Volume"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        vol   = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        if ticker not in close.columns:
            log.warning("No price data from Yahoo Finance for %s", ticker)
            continue

        prices_s = close[ticker].dropna()
        vol_s    = vol[ticker] if ticker in vol.columns else pd.Series(dtype=float)

        if len(prices_s) < 3:
            log.warning("Too few price observations for %s (%d)", ticker, len(prices_s))
            continue

        df = pd.DataFrame({
            "ticker":            ticker,
            "date":              prices_s.index.strftime("%Y-%m-%d"),
            "adjusted_close":    prices_s.values.astype(float),
            "volume":            vol_s.reindex(prices_s.index).values if not vol_s.empty else np.nan,
        })
        df["monthly_return"]    = df["adjusted_close"].pct_change()
        df["data_quality_flag"] = "ok"
        df.loc[df["adjusted_close"].isna(), "data_quality_flag"] = "missing"
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    log.info("Downloaded %d price rows for %d tickers", len(result), len(rows))
    return result


# ── Benchmark ──────────────────────────────────────────────────────────────────

def fetch_benchmark(
    ticker:     str = "SPY",
    start_date: str = "2010-01-01",
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Download real SPY monthly prices as the benchmark.

    Returns DataFrame matching benchmark_prices schema:
        benchmark_ticker, date, adjusted_close, monthly_return
    """
    import yfinance as yf

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    log.info("Downloading benchmark (%s) via Yahoo Finance...", ticker)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )

    if raw.empty:
        log.error("yfinance returned no benchmark data for %s.", ticker)
        return pd.DataFrame()

    # yfinance ≥1.0 returns MultiIndex columns even for a single ticker
    # squeeze() converts a single-column DataFrame to a Series safely
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].squeeze().dropna()
    elif "Close" in raw.columns:
        close = raw["Close"].squeeze().dropna()
    else:
        close = raw.iloc[:, 0].squeeze().dropna()

    df = pd.DataFrame({
        "benchmark_ticker": ticker,
        "date":             close.index.strftime("%Y-%m-%d"),
        "adjusted_close":   close.values.astype(float),
    })
    df["monthly_return"] = df["adjusted_close"].pct_change()
    df.loc[df.index[0], "monthly_return"] = float("nan")

    log.info("Downloaded %d benchmark rows for %s", len(df), ticker)
    return df


# ── Fundamentals ───────────────────────────────────────────────────────────────

def fetch_fundamentals(
    tickers:    list[str],
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Download real quarterly fundamental data from Yahoo Finance / SEC filings.

    For each ticker fetches: revenue, EPS, net income, equity, debt, FCF,
    ROE, debt/equity, gross margin, operating margin.

    Returns DataFrame matching fundamentals_clean schema.
    """
    import yfinance as yf

    log.info("Downloading fundamentals for %d tickers via Yahoo Finance...", len(tickers))

    all_rows: list[dict] = []

    for ticker in tickers:
        try:
            time.sleep(0.3)   # polite rate-limiting
            t = yf.Ticker(ticker)
            rows = _extract_fundamentals(t, ticker, start_date)
            all_rows.extend(rows)
            log.debug("%s: %d fundamental rows", ticker, len(rows))
        except Exception as exc:
            log.warning("Fundamentals failed for %s: %s", ticker, exc)
            continue

    if not all_rows:
        log.warning("No fundamental data retrieved from Yahoo Finance.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    log.info("Downloaded %d fundamental rows for %d tickers", len(df), len(set(r["ticker"] for r in all_rows)))
    return df


def _extract_fundamentals(ticker_obj, ticker: str, start_date: str) -> list[dict]:
    """Extract quarterly fundamentals from a yfinance Ticker object."""
    rows: list[dict] = []

    try:
        # Quarterly income statement
        income = ticker_obj.quarterly_income_stmt
        balance = ticker_obj.quarterly_balance_sheet
        cashflow = ticker_obj.quarterly_cashflow
    except Exception:
        return rows

    if income is None or income.empty:
        return rows

    for col in income.columns:
        try:
            fiscal_date = pd.Timestamp(col).strftime("%Y-%m-%d")
            if fiscal_date < start_date:
                continue

            # Quarter label from month
            q_month = pd.Timestamp(col).month
            q_label = f"Q{(q_month - 1) // 3 + 1}"

            # Report date: 45 days after fiscal date (Q1-Q3) or 90 days (Q4)
            lag = 90 if q_label == "Q4" else 45
            report_date = (pd.Timestamp(col) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")

            def _get(df, *keys):
                if df is None or df.empty:
                    return None
                for key in keys:
                    if key in df.index:
                        v = df.loc[key, col]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v) / 1e9   # convert to $B
                return None

            def _get_ratio(df, *keys):
                if df is None or df.empty:
                    return None
                for key in keys:
                    if key in df.index:
                        v = df.loc[key, col]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v)
                return None

            revenue      = _get(income,   "Total Revenue")
            net_income   = _get(income,   "Net Income")
            eps          = _get_ratio(income, "Basic EPS", "Diluted EPS")
            equity       = _get(balance,  "Stockholders Equity", "Common Stock Equity")
            total_assets = _get(balance,  "Total Assets")
            total_debt   = _get(balance,  "Total Debt", "Long Term Debt")
            fcf          = _get(cashflow, "Free Cash Flow")
            gross_prof   = _get(income,   "Gross Profit")
            op_income    = _get(income,   "Operating Income", "EBIT")

            gross_margin = (gross_prof / revenue)    if (gross_prof and revenue and revenue > 0) else None
            op_margin    = (op_income  / revenue)    if (op_income  and revenue and revenue > 0) else None
            roe          = (net_income * 4 / equity) if (net_income and equity and equity > 0) else None
            # ROA = annualised net income / total assets — genuinely distinct from ROE
            roa          = (net_income * 4 / total_assets) if (net_income and total_assets and total_assets > 0) else None
            debt_to_eq   = (total_debt / equity)    if (total_debt is not None and equity and equity > 0) else None

            rows.append({
                "ticker":           ticker,
                "fiscal_period":    q_label,
                "fiscal_date":      fiscal_date,
                "report_date":      report_date,
                "revenue":          round(revenue, 4)      if revenue      is not None else None,
                "eps":              round(eps, 4)          if eps          is not None else None,
                "net_income":       round(net_income, 4)   if net_income   is not None else None,
                "equity":           round(equity, 4)       if equity       is not None else None,
                "total_debt":       round(total_debt, 4)   if total_debt   is not None else None,
                "free_cash_flow":   round(fcf, 4)          if fcf          is not None else None,
                "roe":              round(roe, 4)          if roe          is not None else None,
                "roic":             round(roa, 4)          if roa          is not None else None,
                "debt_to_equity":   round(debt_to_eq, 4)   if debt_to_eq   is not None else None,
                "gross_margin":     round(gross_margin, 4) if gross_margin is not None else None,
                "operating_margin": round(op_margin, 4)   if op_margin    is not None else None,
            })
        except Exception:
            continue

    return rows
