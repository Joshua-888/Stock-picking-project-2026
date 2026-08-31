"""
src/database/migrations.py

Lightweight schema migration system for SQLite.

How it works
------------
Each migration is a plain SQL string tagged with a version number.
On startup, run apply_migrations() once.  It checks which migrations
have already been applied (stored in the schema_migrations table) and
runs only the new ones in order.  Safe to call on every application start.

Adding a new migration
----------------------
1. Append a new entry to the MIGRATIONS list.
2. Give it the next integer version number.
3. Write the SQL as a string.
4. Re-run apply_migrations() — only the new migration will execute.

Usage:
    from src.database.migrations import apply_migrations
    apply_migrations()
"""

from __future__ import annotations

from dataclasses import dataclass

from src.database.db import get_connection
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Migration registry ────────────────────────────────────────────────────────

@dataclass
class Migration:
    version: int
    description: str
    sql: str


# Add new migrations here.  Never edit or delete existing ones.
MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="Initial schema (handled by schema.sql + initialize_db)",
        sql="SELECT 1",  # No-op: schema.sql already creates all tables
    ),
    Migration(
        version=2,
        description="Add predicted_rank column to prediction_snapshots",
        sql="""
            ALTER TABLE prediction_snapshots
            ADD COLUMN predicted_rank INTEGER
        """,
    ),
    Migration(
        version=3,
        description="Add correlation_score to prediction_snapshots",
        sql="""
            ALTER TABLE prediction_snapshots
            ADD COLUMN correlation_score REAL
        """,
    ),
    Migration(
        version=4,
        description="Add beta column to monthly_features",
        sql="ALTER TABLE monthly_features ADD COLUMN beta REAL",
    ),
    Migration(
        version=5,
        description="Add shares_outstanding_m column to stocks",
        sql="ALTER TABLE stocks ADD COLUMN shares_outstanding_m REAL",
    ),
    Migration(
        version=7,
        description="Add category column to stocks (Core vs Hidden Gem)",
        sql="ALTER TABLE stocks ADD COLUMN category TEXT DEFAULT 'Core'",
    ),
    Migration(
        version=8,
        description="Add category column to prediction_snapshots and backfill from stocks",
        sql="""
            ALTER TABLE prediction_snapshots ADD COLUMN category TEXT DEFAULT 'Core';
            UPDATE prediction_snapshots SET category = (
                SELECT COALESCE(s.category, 'Core') FROM stocks s WHERE s.ticker = prediction_snapshots.ticker
            ) WHERE category IS NULL OR category = 'Core';
        """,
    ),
    Migration(
        version=6,
        description="Add 10 Renaissance-inspired signal columns to monthly_features",
        sql="""
            ALTER TABLE monthly_features ADD COLUMN return_1m REAL;
            ALTER TABLE monthly_features ADD COLUMN return_3m REAL;
            ALTER TABLE monthly_features ADD COLUMN drawdown_from_52w_high REAL;
            ALTER TABLE monthly_features ADD COLUMN volatility_3m REAL;
            ALTER TABLE monthly_features ADD COLUMN downside_volatility_12m REAL;
            ALTER TABLE monthly_features ADD COLUMN abnormal_volume REAL;
            ALTER TABLE monthly_features ADD COLUMN sector_relative_ps REAL;
            ALTER TABLE monthly_features ADD COLUMN sector_relative_fcf_yield REAL;
            ALTER TABLE monthly_features ADD COLUMN peer_momentum_zscore REAL;
            ALTER TABLE monthly_features ADD COLUMN peer_valuation_zscore REAL;
        """,
    ),
    Migration(
        version=9,
        description="Add watchlist_tickers table for custom ticker analysis",
        sql="""
            CREATE TABLE IF NOT EXISTS watchlist_tickers (
                ticker       TEXT PRIMARY KEY,
                company_name TEXT,
                sector       TEXT,
                added_at     TEXT DEFAULT (datetime('now')),
                notes        TEXT
            );
        """,
    ),
]


# ── Apply migrations ──────────────────────────────────────────────────────────

def apply_migrations() -> int:
    """
    Apply any pending migrations in version order.

    Creates the schema_migrations tracking table if it does not exist.
    Skips migrations that have already been applied.

    Returns:
        Number of new migrations applied.
    """
    _ensure_migrations_table()
    applied = _get_applied_versions()
    pending = [m for m in MIGRATIONS if m.version not in applied]

    if not pending:
        log.debug("No pending migrations.")
        return 0

    count = 0
    for migration in sorted(pending, key=lambda m: m.version):
        _apply_migration(migration)
        count += 1

    log.info("Applied %d migration(s).", count)
    return count


def get_migration_status() -> list[dict]:
    """
    Return a list of all migrations with their applied status.
    Useful for the Data Quality dashboard page.
    """
    applied = _get_applied_versions()
    return [
        {
            "version":     m.version,
            "description": m.description,
            "applied":     m.version in applied,
        }
        for m in MIGRATIONS
    ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_migrations_table() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                description TEXT,
                applied_at  TEXT DEFAULT (datetime('now'))
            )
        """)


def _get_applied_versions() -> set[int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def _apply_migration(migration: Migration) -> None:
    log.info("Applying migration v%d: %s", migration.version, migration.description)
    try:
        with get_connection() as conn:
            # SQLite ALTER TABLE does not support IF NOT EXISTS, so we
            # catch the "duplicate column" error and treat it as a no-op.
            try:
                conn.executescript(migration.sql)
            except Exception as sql_err:
                if "duplicate column" in str(sql_err).lower():
                    log.debug("Migration v%d already applied (duplicate column): %s",
                              migration.version, sql_err)
                else:
                    raise

            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, description) VALUES (?, ?)",
                (migration.version, migration.description),
            )
        log.info("Migration v%d applied successfully.", migration.version)
    except Exception as exc:
        log.error("Migration v%d FAILED: %s", migration.version, exc)
        raise
