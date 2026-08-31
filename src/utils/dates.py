"""
src/utils/dates.py

Date helpers used throughout the pipeline.

All dates are stored and passed as ISO-8601 strings ("YYYY-MM-DD").
Month-end snapping is critical: features and targets are always aligned
to calendar month-ends so lookups are consistent across modules.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd


def to_month_end(date_str: str) -> str:
    """
    Snap a date string to the last calendar day of its month.

    Example: "2024-01-15" → "2024-01-31"
    """
    ts = pd.Timestamp(date_str)
    return ts.to_period("M").to_timestamp("M").strftime("%Y-%m-%d")


def months_ago(date_str: str, n: int) -> str:
    """
    Return the month-end date exactly n months before date_str.

    Example: months_ago("2024-06-30", 6) → "2023-12-31"
    """
    ts = pd.Timestamp(date_str) - pd.DateOffset(months=n)
    return to_month_end(ts.strftime("%Y-%m-%d"))


def years_ago(date_str: str, n: int) -> str:
    """
    Return the month-end date exactly n years before date_str.

    Example: years_ago("2024-06-30", 5) → "2019-06-30"
    """
    return months_ago(date_str, n * 12)


def month_end_range(start: str, end: str) -> list[str]:
    """
    Return a sorted list of month-end date strings between start and end (inclusive).

    Example: month_end_range("2024-01-01", "2024-03-31")
             → ["2024-01-31", "2024-02-29", "2024-03-31"]
    """
    idx = pd.date_range(start=start, end=end, freq="ME")
    return [d.strftime("%Y-%m-%d") for d in idx]


def months_between(start: str, end: str) -> int:
    """
    Return the number of calendar months between two date strings.

    Example: months_between("2019-01-31", "2024-01-31") → 60
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return (e.year - s.year) * 12 + (e.month - s.month)


def is_valid_date(date_str: str) -> bool:
    """Return True if date_str parses as a valid date."""
    try:
        pd.Timestamp(date_str)
        return True
    except Exception:
        return False


def today_month_end() -> str:
    """Return the month-end of the current calendar month."""
    return to_month_end(pd.Timestamp.today().strftime("%Y-%m-%d"))


def add_days(date_str: str, n: int) -> str:
    """Add n calendar days to a date string."""
    return (pd.Timestamp(date_str) + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


def min_date(dates: list[str]) -> Optional[str]:
    """Return the earliest date string from a list, or None if empty."""
    valid = [d for d in dates if d]
    return min(valid) if valid else None


def max_date(dates: list[str]) -> Optional[str]:
    """Return the latest date string from a list, or None if empty."""
    valid = [d for d in dates if d]
    return max(valid) if valid else None
