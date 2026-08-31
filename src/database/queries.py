"""
src/database/queries.py

Named read functions for every table in the database.

All functions return pandas DataFrames.  None → no filter applied.
Date arguments are ISO-8601 strings ("YYYY-MM-DD").

Usage:
    from src.database.queries import (
        load_prices_clean,
        load_fundamentals_clean,
        load_monthly_features,
        load_latest_snapshots,
    )

    prices = load_prices_clean(tickers=["AAPL", "MSFT"], start_date="2020-01-01")
    funds  = load_fundamentals_clean(tickers=["AAPL"])
    feats  = load_monthly_features(feature_date="2024-12-31")
    snaps  = load_latest_snapshots()
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.database.db import get_engine
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Internal helper ───────────────────────────────────────────────────────────

def _read(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame. Empty = empty DataFrame."""
    engine = get_engine()
    try:
        return pd.read_sql(sql, engine, params=params)
    except Exception as exc:
        log.error("Query failed: %s | error: %s", sql[:120], exc)
        return pd.DataFrame()


def _build_where(conditions: list[str]) -> str:
    if not conditions:
        return ""
    return "WHERE " + " AND ".join(conditions)


# ── stocks ────────────────────────────────────────────────────────────────────

def load_stocks(active_only: bool = True) -> pd.DataFrame:
    """
    Return all stocks in the universe.

    Args:
        active_only: If True (default), exclude delisted stocks.
    """
    where = "WHERE is_active = 1" if active_only else ""
    return _read(f"SELECT * FROM stocks {where} ORDER BY ticker")


def load_tickers(active_only: bool = True) -> list[str]:
    """Return a plain list of ticker strings."""
    df = load_stocks(active_only=active_only)
    return df["ticker"].tolist() if not df.empty else []


# ── prices_clean ──────────────────────────────────────────────────────────────

def load_prices_clean(
    tickers:    Optional[list[str]] = None,
    start_date: Optional[str]       = None,
    end_date:   Optional[str]       = None,
) -> pd.DataFrame:
    """
    Load monthly adjusted prices.

    Args:
        tickers:    List of tickers to include (None = all).
        start_date: Earliest date to include.
        end_date:   Latest date to include.

    Returns:
        DataFrame: ticker, date, adjusted_close, monthly_return, volume, data_quality_flag
    """
    conds, params = _price_filters(tickers, start_date, end_date)
    where = _build_where(conds)
    sql = f"SELECT * FROM prices_clean {where} ORDER BY ticker, date"
    return _read(sql, tuple(params) if params else None)


def load_price_series(
    ticker:     str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.Series:
    """
    Return adjusted_close as a pd.Series indexed by date for one ticker.
    """
    df = load_prices_clean([ticker], start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    return df.set_index("date")["adjusted_close"].rename(ticker)


def load_returns_matrix(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a wide matrix: rows=date, columns=ticker, values=monthly_return.
    Useful for cross-sectional calculations (sector momentum, beta, etc.).
    """
    df = load_prices_clean(start_date=start_date, end_date=end_date)
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="monthly_return").sort_index()


# ── benchmark_prices ──────────────────────────────────────────────────────────

def load_benchmark_prices(
    benchmark_ticker: Optional[str] = None,
    start_date:       Optional[str] = None,
    end_date:         Optional[str] = None,
) -> pd.DataFrame:
    """
    Load monthly benchmark prices and returns.

    Returns:
        DataFrame: benchmark_ticker, date, adjusted_close, monthly_return
    """
    conds:  list[str] = []
    params: list      = []

    if benchmark_ticker:
        conds.append("benchmark_ticker = ?")
        params.append(benchmark_ticker)
    if start_date:
        conds.append("date >= ?")
        params.append(start_date)
    if end_date:
        conds.append("date <= ?")
        params.append(end_date)

    where = _build_where(conds)
    sql = f"SELECT * FROM benchmark_prices {where} ORDER BY benchmark_ticker, date"
    return _read(sql, tuple(params) if params else None)


def load_benchmark_returns(
    benchmark_ticker: Optional[str] = None,
    start_date:       Optional[str] = None,
    end_date:         Optional[str] = None,
) -> pd.Series:
    """
    Return benchmark monthly_return as a pd.Series indexed by date.
    """
    df = load_benchmark_prices(benchmark_ticker, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float, name="benchmark_return")
    return df.set_index("date")["monthly_return"].rename("benchmark_return")


# ── fundamentals_clean ────────────────────────────────────────────────────────

def load_fundamentals_clean(
    tickers:     Optional[list[str]] = None,
    start_date:  Optional[str]       = None,
    end_date:    Optional[str]       = None,
    lag_safe:    bool                 = True,
    as_of_date:  Optional[str]       = None,
) -> pd.DataFrame:
    """
    Load cleaned quarterly fundamental data.

    Args:
        tickers:    Filter to these tickers (None = all).
        start_date: Filter by fiscal_date >= start_date.
        end_date:   Filter by fiscal_date <= end_date.
        lag_safe:   If True and as_of_date is set, only return rows where
                    report_date <= as_of_date (prevents look-ahead bias).
        as_of_date: The prediction date. Required when lag_safe=True.

    Returns:
        DataFrame with all fundamentals_clean columns.
    """
    conds:  list[str] = []
    params: list      = []

    if tickers:
        placeholders = ", ".join(["?" for _ in tickers])
        conds.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start_date:
        conds.append("fiscal_date >= ?")
        params.append(start_date)
    if end_date:
        conds.append("fiscal_date <= ?")
        params.append(end_date)
    if lag_safe and as_of_date:
        # Only include fundamentals whose report_date is on or before
        # the prediction date (look-ahead bias prevention)
        conds.append("report_date <= ?")
        params.append(as_of_date)

    where = _build_where(conds)
    sql = f"SELECT * FROM fundamentals_clean {where} ORDER BY ticker, fiscal_date"
    return _read(sql, tuple(params) if params else None)


# ── monthly_features ──────────────────────────────────────────────────────────

def load_monthly_features(
    feature_date: Optional[str]       = None,
    tickers:      Optional[list[str]] = None,
    start_date:   Optional[str]       = None,
    end_date:     Optional[str]       = None,
) -> pd.DataFrame:
    """
    Load the feature matrix for model training or prediction.

    Args:
        feature_date: Return only this exact date (for current predictions).
        tickers:      Filter to these tickers.
        start_date:   Filter feature_date >= start_date (for training sets).
        end_date:     Filter feature_date <= end_date.

    Returns:
        DataFrame with feature_date, ticker, and all 25 feature columns.
    """
    conds:  list[str] = []
    params: list      = []

    if feature_date:
        conds.append("feature_date = ?")
        params.append(feature_date)
    if tickers:
        placeholders = ", ".join(["?" for _ in tickers])
        conds.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start_date and not feature_date:
        conds.append("feature_date >= ?")
        params.append(start_date)
    if end_date and not feature_date:
        conds.append("feature_date <= ?")
        params.append(end_date)

    where = _build_where(conds)
    sql = f"SELECT * FROM monthly_features {where} ORDER BY feature_date, ticker"
    return _read(sql, tuple(params) if params else None)


def load_feature_dates() -> list[str]:
    """Return all distinct feature_date values, sorted ascending."""
    df = _read("SELECT DISTINCT feature_date FROM monthly_features ORDER BY feature_date")
    return df["feature_date"].tolist() if not df.empty else []


def load_computed_feature_pairs() -> set:
    """Return set of (feature_date, ticker) tuples already in monthly_features."""
    df = _read("SELECT feature_date, ticker FROM monthly_features")
    if df.empty:
        return set()
    return set(zip(df["feature_date"], df["ticker"]))


def load_monthly_features_with_sector(
    feature_date: Optional[str]       = None,
    tickers:      Optional[list[str]] = None,
    start_date:   Optional[str]       = None,
    end_date:     Optional[str]       = None,
) -> pd.DataFrame:
    """
    Load monthly features with sector and industry joined from the stocks table.

    Use this instead of load_monthly_features() when sector context is needed
    for sector-relative calculations or dashboard filtering.

    Returns all monthly_features columns plus: sector, industry, company_name.
    """
    conds:  list[str] = []
    params: list      = []

    if feature_date:
        conds.append("mf.feature_date = ?")
        params.append(feature_date)
    if tickers:
        ph = ", ".join(["?" for _ in tickers])
        conds.append(f"mf.ticker IN ({ph})")
        params.extend(tickers)
    if start_date and not feature_date:
        conds.append("mf.feature_date >= ?")
        params.append(start_date)
    if end_date and not feature_date:
        conds.append("mf.feature_date <= ?")
        params.append(end_date)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""
        SELECT mf.*,
               s.sector,
               s.industry,
               s.company_name,
               s.shares_outstanding_m
        FROM   monthly_features mf
        LEFT JOIN stocks s ON mf.ticker = s.ticker
        {where}
        ORDER  BY mf.feature_date, mf.ticker
    """
    return _read(sql, tuple(params) if params else None)


# ── targets ───────────────────────────────────────────────────────────────────

def load_targets(
    feature_date: Optional[str]       = None,
    tickers:      Optional[list[str]] = None,
    start_date:   Optional[str]       = None,
    end_date:     Optional[str]       = None,
) -> pd.DataFrame:
    """
    Load target variables (future excess return + winner flag).

    Returns only rows where targets have been populated (non-NULL winner).
    Use start_date / end_date for training-set windows.
    """
    conds:  list[str] = ["winner IS NOT NULL"]
    params: list      = []

    if feature_date:
        conds.append("feature_date = ?")
        params.append(feature_date)
    if tickers:
        placeholders = ", ".join(["?" for _ in tickers])
        conds.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start_date and not feature_date:
        conds.append("feature_date >= ?")
        params.append(start_date)
    if end_date and not feature_date:
        conds.append("feature_date <= ?")
        params.append(end_date)

    where = _build_where(conds)
    sql = f"SELECT * FROM targets {where} ORDER BY feature_date, ticker"
    return _read(sql, tuple(params) if params else None)


def load_features_and_targets(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    tickers:    Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return features joined with targets for model training.

    Only rows where both features AND labelled targets exist are returned.
    This is the primary input for all model training functions.
    """
    sql = """
        SELECT f.*, t.future_12m_excess_return, t.future_12m_stock_return,
               t.future_12m_benchmark_return, t.winner
        FROM   monthly_features f
        INNER JOIN targets t
               ON f.feature_date = t.feature_date AND f.ticker = t.ticker
        WHERE  t.winner IS NOT NULL
    """
    params: list = []
    extra: list[str] = []

    if tickers:
        ph = ", ".join(["?" for _ in tickers])
        extra.append(f"f.ticker IN ({ph})")
        params.extend(tickers)
    if start_date:
        extra.append("f.feature_date >= ?")
        params.append(start_date)
    if end_date:
        extra.append("f.feature_date <= ?")
        params.append(end_date)

    if extra:
        sql += " AND " + " AND ".join(extra)
    sql += " ORDER BY f.feature_date, f.ticker"

    return _read(sql, tuple(params) if params else None)


# ── prediction_snapshots ──────────────────────────────────────────────────────

def load_prediction_snapshots(
    prediction_date: Optional[str]       = None,
    ticker:          Optional[str]        = None,
    tickers:         Optional[list[str]]  = None,
) -> pd.DataFrame:
    """
    Load prediction snapshots from the archive.

    Args:
        prediction_date: Filter to this exact date.
        ticker:          Filter to this single ticker.
        tickers:         Filter to a list of tickers.
    """
    conds:  list[str] = []
    params: list      = []

    if prediction_date:
        conds.append("prediction_date = ?")
        params.append(prediction_date)
    if ticker:
        conds.append("ticker = ?")
        params.append(ticker)
    if tickers:
        ph = ", ".join(["?" for _ in tickers])
        conds.append(f"ticker IN ({ph})")
        params.extend(tickers)

    where = _build_where(conds)
    sql = f"SELECT * FROM prediction_snapshots {where} ORDER BY prediction_date DESC, final_score DESC"
    return _read(sql, tuple(params) if params else None)


def load_latest_snapshots() -> pd.DataFrame:
    """
    Return the most recent prediction snapshot for every ticker.
    This is the primary source for the Overview and Screener pages.
    """
    sql = """
        SELECT ps.*
        FROM   prediction_snapshots ps
        INNER JOIN (
            SELECT ticker, MAX(prediction_date) AS max_date
            FROM   prediction_snapshots
            GROUP  BY ticker
        ) latest
               ON ps.ticker = latest.ticker
              AND ps.prediction_date = latest.max_date
        ORDER  BY final_score DESC
    """
    return _read(sql)


def load_snapshot_dates() -> list[str]:
    """Return all distinct prediction_date values, sorted descending."""
    df = _read(
        "SELECT DISTINCT prediction_date FROM prediction_snapshots ORDER BY prediction_date DESC"
    )
    return df["prediction_date"].tolist() if not df.empty else []


# ── realized_performance ──────────────────────────────────────────────────────

def load_realized_performance(
    model_version:   Optional[str] = None,
    prediction_date: Optional[str] = None,
    ticker:          Optional[str] = None,
) -> pd.DataFrame:
    """Load realized outcome records for backtest evaluation."""
    conds:  list[str] = []
    params: list      = []

    if model_version:
        conds.append("model_version = ?")
        params.append(model_version)
    if prediction_date:
        conds.append("prediction_date = ?")
        params.append(prediction_date)
    if ticker:
        conds.append("ticker = ?")
        params.append(ticker)

    where = _build_where(conds)
    sql = f"SELECT * FROM realized_performance {where} ORDER BY prediction_date, ticker"
    return _read(sql, tuple(params) if params else None)


def load_model_accuracy_summary() -> pd.DataFrame:
    """
    Summarise prediction accuracy by probability bucket.

    Returns a DataFrame with columns:
        probability_bucket, n_predictions, n_correct, hit_rate
    """
    sql = """
        SELECT probability_bucket,
               COUNT(*)                              AS n_predictions,
               SUM(prediction_correct)               AS n_correct,
               ROUND(AVG(CAST(prediction_correct AS FLOAT)), 4) AS hit_rate
        FROM   realized_performance
        WHERE  prediction_correct IS NOT NULL
        GROUP  BY probability_bucket
        ORDER  BY probability_bucket
    """
    return _read(sql)


# ── backtest_results ──────────────────────────────────────────────────────────

def load_backtest_results(model_version: Optional[str] = None) -> pd.DataFrame:
    """Load backtest result rows, optionally filtered by model version."""
    where = "WHERE model_version = ?" if model_version else ""
    sql = f"SELECT * FROM backtest_results {where} ORDER BY start_date DESC"
    return _read(sql, (model_version,) if model_version else None)


# ── watchlist_tickers ─────────────────────────────────────────────────────────
# Canonical implementations live in src/database/watchlist.py (new dedicated
# file, no stale-cache risk). These are thin forwards kept for backward compat.

def load_watchlist() -> pd.DataFrame:
    from src.database.watchlist import load_watchlist as _f
    return _f()

def add_watchlist_ticker(ticker: str, company_name: str = "", sector: str = "") -> None:
    from src.database.watchlist import add_watchlist_ticker as _f
    _f(ticker, company_name, sector)

def remove_watchlist_ticker(ticker: str) -> None:
    from src.database.watchlist import remove_watchlist_ticker as _f
    _f(ticker)

def update_watchlist_ticker(ticker: str, company_name: str, sector: str) -> None:
    from src.database.watchlist import update_watchlist_ticker as _f
    _f(ticker, company_name, sector)


# ── model_training_runs ───────────────────────────────────────────────────────

def load_model_training_runs(model_type: Optional[str] = None) -> pd.DataFrame:
    """Load model training run metadata."""
    where = "WHERE model_type = ?" if model_type else ""
    sql = f"SELECT * FROM model_training_runs {where} ORDER BY created_at DESC"
    return _read(sql, (model_type,) if model_type else None)


def load_latest_model_version(model_type: str) -> Optional[str]:
    """Return the model_version string for the most recently trained model of a given type."""
    sql = """
        SELECT model_version FROM model_training_runs
        WHERE  model_type = ?
        ORDER  BY created_at DESC
        LIMIT  1
    """
    df = _read(sql, (model_type,))
    if df.empty:
        return None
    return df["model_version"].iloc[0]


# ── data_quality_logs ─────────────────────────────────────────────────────────

def load_data_quality_logs(
    ticker:     Optional[str] = None,
    severity:   Optional[str] = None,
    issue_type: Optional[str] = None,
    limit:      int            = 500,
) -> pd.DataFrame:
    """
    Load recent data quality log entries.

    Args:
        ticker:     Filter to this ticker.
        severity:   'info' | 'warning' | 'error' | 'critical'
        issue_type: E.g. 'missing_price', 'stale_fundamental'
        limit:      Maximum rows to return (default 500).
    """
    conds:  list[str] = []
    params: list      = []

    if ticker:
        conds.append("ticker = ?")
        params.append(ticker)
    if severity:
        conds.append("severity = ?")
        params.append(severity)
    if issue_type:
        conds.append("issue_type = ?")
        params.append(issue_type)

    where = _build_where(conds)
    sql = f"SELECT * FROM data_quality_logs {where} ORDER BY created_at DESC LIMIT {limit}"
    return _read(sql, tuple(params) if params else None)


def load_data_quality_summary() -> pd.DataFrame:
    """Return a count of issues by severity and issue_type."""
    sql = """
        SELECT severity, issue_type, COUNT(*) AS n
        FROM   data_quality_logs
        GROUP  BY severity, issue_type
        ORDER  BY severity, n DESC
    """
    return _read(sql)


# ── provider_availability ─────────────────────────────────────────────────────

def load_provider_availability(provider: Optional[str] = None) -> pd.DataFrame:
    """Load the data availability matrix for one or all providers."""
    where = "WHERE provider = ?" if provider else ""
    sql = f"SELECT * FROM provider_availability {where} ORDER BY provider, variable_name"
    return _read(sql, (provider,) if provider else None)


# ── Shared filter builder ─────────────────────────────────────────────────────

def _price_filters(
    tickers:    Optional[list[str]],
    start_date: Optional[str],
    end_date:   Optional[str],
) -> tuple[list[str], list]:
    conds:  list[str] = []
    params: list      = []

    if tickers:
        ph = ", ".join(["?" for _ in tickers])
        conds.append(f"ticker IN ({ph})")
        params.extend(tickers)
    if start_date:
        conds.append("date >= ?")
        params.append(start_date)
    if end_date:
        conds.append("date <= ?")
        params.append(end_date)

    return conds, params
