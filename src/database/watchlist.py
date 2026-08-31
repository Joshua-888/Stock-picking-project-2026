"""
src/database/watchlist.py

Watchlist CRUD operations — kept in a dedicated file so Streamlit
never loads a stale cached version (no __pycache__ conflict risk).
"""

from __future__ import annotations

import pandas as pd
from src.database.db import get_connection


def load_watchlist() -> pd.DataFrame:
    """Return all tickers in the custom watchlist, newest first."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT ticker, company_name, sector, added_at, notes "
                "FROM watchlist_tickers ORDER BY added_at DESC"
            ).fetchall()
        return pd.DataFrame(rows, columns=["ticker","company_name","sector","added_at","notes"])
    except Exception:
        return pd.DataFrame(columns=["ticker","company_name","sector","added_at","notes"])


def add_watchlist_ticker(ticker: str, company_name: str = "", sector: str = "") -> None:
    """Add a ticker (no-op if already present)."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist_tickers (ticker, company_name, sector) "
            "VALUES (?, ?, ?)",
            (ticker.strip().upper(), company_name, sector),
        )


def remove_watchlist_ticker(ticker: str) -> None:
    """Remove a ticker from the watchlist."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM watchlist_tickers WHERE ticker = ?",
            (ticker.strip().upper(),),
        )


def update_watchlist_ticker(ticker: str, company_name: str, sector: str) -> None:
    """Update company name and sector after analysis."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE watchlist_tickers SET company_name=?, sector=? WHERE ticker=?",
            (company_name, sector, ticker.strip().upper()),
        )
