"""
src/database/db.py

Database connection, initialisation, session management, and save functions.

This module is the single write entry point for the database.
Read/query functions live in queries.py.

Usage:
    from src.database.db import initialize_db, save_prices_clean, save_fundamentals_clean

    initialize_db()                        # create tables once
    save_prices_clean(prices_df)           # insert / update price rows
    save_fundamentals_clean(funds_df)      # insert / update fundamental rows
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from src.utils.logging import get_logger

log = get_logger(__name__)

# Path to schema.sql relative to this file
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Module-level engine cache — created once, reused everywhere
_engine: Engine | None = None


# ── Path helpers ──────────────────────────────────────────────────────────────

def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    """
    Resolve the database file path.

    Priority:
      1. Explicit argument passed by caller
      2. DATABASE_PATH environment variable (loaded via config)
      3. Default: project_root/data/stock_analysis.db
    """
    if db_path is not None:
        return Path(db_path)

    # Try to read from config without crashing if config is not yet set up
    try:
        from src.utils.config import load_config
        cfg = load_config()
        return Path(cfg.database.path)
    except Exception:
        pass

    # Fallback: two levels up from src/database/ → project root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "stock_analysis.db"


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine(db_path: str | Path | None = None) -> Engine:
    """
    Return a SQLAlchemy Engine for the SQLite database.

    The engine is cached after the first call. If you need a fresh engine
    (e.g. in tests pointing at a different file), call reset_engine() first.

    SQLite-specific settings applied:
      - WAL journal mode for better read/write concurrency
      - Foreign key enforcement
      - 30-second busy timeout to reduce "database is locked" errors
    """
    global _engine

    if _engine is not None:
        return _engine

    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection_url = f"sqlite:///{path}"
    engine = create_engine(
        connection_url,
        connect_args={"check_same_thread": False, "timeout": 30},
        echo=False,   # Set True to log every SQL statement (debug only)
    )

    # Apply pragmas on every new connection
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()

    _engine = engine
    log.debug("SQLAlchemy engine created: %s", connection_url)
    return _engine


def reset_engine() -> None:
    """
    Dispose the cached engine and clear the module-level reference.
    Call this in tests or when switching database files.
    """
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


# ── Schema initialisation ─────────────────────────────────────────────────────

def initialize_db(db_path: str | Path | None = None) -> None:
    """
    Create all tables defined in schema.sql if they do not already exist.

    Safe to call multiple times — all CREATE TABLE statements use
    IF NOT EXISTS, so existing data is never overwritten.

    Uses sqlite3.executescript() directly because it correctly handles
    multi-statement SQL files including comment blocks.

    Args:
        db_path: Optional path override for the database file.
    """
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    # executescript() handles the full SQL file in one call, including
    # comment lines and multi-line statements — no manual splitting needed.
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

    # Ensure the SQLAlchemy engine is warmed up for this path
    get_engine(db_path)
    log.info("Database initialised: %s", path)


# ── Raw connection context manager ────────────────────────────────────────────

@contextmanager
def get_connection(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a raw sqlite3 connection.

    Use this for operations that are awkward with SQLAlchemy (e.g. bulk
    inserts using executemany, or PRAGMA queries).

    The connection is committed on clean exit and rolled back on exception.

    Example:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO stocks (ticker) VALUES (?)", ("AAPL",)
            )
    """
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row   # Rows accessible by column name
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Health check ──────────────────────────────────────────────────────────────

def check_db_health(db_path: str | Path | None = None) -> dict:
    """
    Return a summary of the database state.

    Useful for the Data Quality dashboard page and for debugging.

    Returns:
        dict with keys: path, size_mb, tables, row_counts
    """
    path = _resolve_db_path(db_path)

    if not path.exists():
        return {"status": "not_found", "path": str(path)}

    size_mb = round(path.stat().st_size / 1_048_576, 2)

    with get_connection(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        row_counts = {}
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            row_counts[table] = count

    return {
        "status": "ok",
        "path": str(path),
        "size_mb": size_mb,
        "tables": tables,
        "row_counts": row_counts,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_val(v: Any) -> Any:
    """Convert NaN / NaT to None so SQLite accepts them as NULL."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _df_to_records(df: pd.DataFrame) -> list[tuple]:
    """Convert a DataFrame to a list of tuples with NaN→None."""
    return [
        tuple(_clean_val(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]


def _insert_df(
    df: pd.DataFrame,
    table: str,
    on_conflict: str = "IGNORE",
) -> int:
    """
    Bulk-insert a DataFrame into a table using INSERT OR {on_conflict}.

    on_conflict options:
      "IGNORE"  – skip rows that violate a UNIQUE constraint (safe default)
      "REPLACE" – delete the conflicting row and insert the new one (update)

    Returns the number of rows affected (approximate for IGNORE).
    """
    if df.empty:
        return 0

    cols          = list(df.columns)
    cols_sql      = ", ".join(cols)
    placeholders  = ", ".join(["?" for _ in cols])
    sql           = f"INSERT OR {on_conflict} INTO {table} ({cols_sql}) VALUES ({placeholders})"
    records       = _df_to_records(df[cols])

    with get_connection() as conn:
        conn.executemany(sql, records)

    log.debug("_insert_df: table=%s on_conflict=%s rows=%d", table, on_conflict, len(df))
    return len(df)


# ── Save functions ────────────────────────────────────────────────────────────

def save_stocks(df: pd.DataFrame) -> int:
    """
    Upsert stock metadata rows.
    Existing rows are replaced so company_name / sector updates propagate.

    Required columns: ticker
    """
    return _insert_df(df, "stocks", on_conflict="REPLACE")


def save_prices_raw(df: pd.DataFrame) -> int:
    """
    Insert raw price records exactly as received from a provider.
    Skips duplicates (same provider + ticker + date already stored).

    Required columns: provider, ticker, date
    """
    return _insert_df(df, "prices_raw", on_conflict="IGNORE")


def save_prices_clean(df: pd.DataFrame) -> int:
    """
    Upsert validated monthly price rows.
    Replaces existing rows so data_quality_flag updates are applied.

    Required columns: ticker, date, adjusted_close
    """
    return _insert_df(df, "prices_clean", on_conflict="REPLACE")


def save_benchmark_prices(df: pd.DataFrame) -> int:
    """
    Insert benchmark price rows. Skips duplicates.

    Required columns: benchmark_ticker, date, adjusted_close
    """
    return _insert_df(df, "benchmark_prices", on_conflict="IGNORE")


def save_fundamentals_raw(df: pd.DataFrame) -> int:
    """
    Insert raw fundamental records. Skips duplicates.

    Required columns: provider, ticker, fiscal_date, fiscal_period
    """
    return _insert_df(df, "fundamentals_raw", on_conflict="IGNORE")


def save_fundamentals_clean(df: pd.DataFrame) -> int:
    """
    Upsert cleaned fundamental rows.

    Required columns: ticker, fiscal_date, fiscal_period
    """
    return _insert_df(df, "fundamentals_clean", on_conflict="REPLACE")


def save_monthly_features(df: pd.DataFrame) -> int:
    """
    Upsert monthly feature rows. Called after every feature-engineering run.
    Replaces so corrected/recalculated features overwrite old values.

    Required columns: feature_date, ticker
    """
    return _insert_df(df, "monthly_features", on_conflict="REPLACE")


def save_targets(df: pd.DataFrame) -> int:
    """
    Upsert target variable rows (future_12m_excess_return, winner).
    Replaces so backfilled targets overwrite placeholder NULLs.

    Required columns: feature_date, ticker
    """
    return _insert_df(df, "targets", on_conflict="REPLACE")


def save_model_training_run(run: dict) -> None:
    """
    Record one model training run.  Skips if model_version already exists
    (training runs are immutable once committed).

    Args:
        run: dict with keys matching model_training_runs columns.
             model_version is required; all others are optional.
    """
    df = pd.DataFrame([run])
    _insert_df(df, "model_training_runs", on_conflict="IGNORE")


def save_predictions(df: pd.DataFrame) -> int:
    """
    Insert model-level prediction rows. Skips duplicates.

    Required columns: prediction_id, model_version, prediction_date, ticker
    """
    if "prediction_id" not in df.columns:
        df = df.copy()
        df["prediction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return _insert_df(df, "predictions", on_conflict="IGNORE")


def save_prediction_snapshot(snapshot: dict | pd.Series) -> None:
    """
    Write one immutable prediction snapshot.

    Snapshots must never be overwritten — they are the historical record
    of what the model predicted at a specific point in time.
    Silently skips if a snapshot for (prediction_date, ticker) already exists.

    Args:
        snapshot: dict or pd.Series with keys matching prediction_snapshots columns.
    """
    if isinstance(snapshot, pd.Series):
        snapshot = snapshot.to_dict()

    if "snapshot_id" not in snapshot or not snapshot["snapshot_id"]:
        snapshot["snapshot_id"] = str(uuid.uuid4())

    df = pd.DataFrame([snapshot])
    _insert_df(df, "prediction_snapshots", on_conflict="IGNORE")


def save_prediction_snapshots(df: pd.DataFrame) -> int:
    """
    Bulk-insert multiple prediction snapshots at once.
    Silently skips existing (prediction_date, ticker) pairs.
    """
    if "snapshot_id" not in df.columns:
        df = df.copy()
        df["snapshot_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return _insert_df(df, "prediction_snapshots", on_conflict="IGNORE")


def save_realized_performance(df: pd.DataFrame) -> int:
    """
    Upsert realized performance rows.
    Replaces so results can be updated as final prices become available.

    Required columns: prediction_date, ticker
    """
    return _insert_df(df, "realized_performance", on_conflict="REPLACE")


def save_backtest_result(result: dict) -> None:
    """
    Record one backtest run result.

    Args:
        result: dict with keys matching backtest_results columns.
                backtest_id is generated automatically if missing.
    """
    if "backtest_id" not in result or not result["backtest_id"]:
        result["backtest_id"] = str(uuid.uuid4())
    df = pd.DataFrame([result])
    _insert_df(df, "backtest_results", on_conflict="IGNORE")


def save_data_quality_log(
    issue_type: str,
    message: str,
    ticker: str | None  = None,
    date:   str | None  = None,
    provider: str | None = None,
    severity: str        = "warning",
) -> None:
    """
    Append one data quality event to the log.

    Args:
        issue_type: Short code, e.g. 'missing_price', 'stale_fundamental'.
        message:    Human-readable description of the issue.
        ticker:     Affected ticker (None for provider-level issues).
        date:       Affected date string (YYYY-MM-DD).
        provider:   Data provider name.
        severity:   'info' | 'warning' | 'error' | 'critical'.
    """
    row = {
        "log_id":     str(uuid.uuid4()),
        "date":       date,
        "ticker":     ticker,
        "provider":   provider,
        "issue_type": issue_type,
        "severity":   severity,
        "message":    message,
    }
    df = pd.DataFrame([row])
    _insert_df(df, "data_quality_logs", on_conflict="IGNORE")


def save_provider_availability(df: pd.DataFrame) -> int:
    """
    Upsert provider availability records.

    Required columns: provider, variable_name, availability_status
    """
    return _insert_df(df, "provider_availability", on_conflict="REPLACE")
