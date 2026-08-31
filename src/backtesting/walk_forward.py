"""
src/backtesting/walk_forward.py

Walk-forward validation split generator.

WHY WALK-FORWARD?
-----------------
A random train/test split on financial data leaks future information:
a stock's January 2022 features might end up in the test set while its
December 2022 return ends up in the training set.

Walk-forward simulation enforces the correct temporal order:
  • Train on months 1 → T-1
  • Score / test on month T
  • Move forward one step
  • Train on months 1 → T
  • Score / test on month T+1
  • …

This mirrors exactly how the model would be used in practice.

EXPANDING vs ROLLING WINDOW
-----------------------------
Expanding window (default):
  Each period uses ALL available history up to that date.
  Pros: more training data over time.  Cons: early and late periods
  are trained on very different amounts of data.

Rolling window:
  Each period uses the most recent N months only.
  Pros: consistent training window size, more sensitive to regime changes.
  Cons: discards older data even when it is still informative.
"""

from __future__ import annotations

from typing import Generator

import pandas as pd


# ── Split generators ──────────────────────────────────────────────────────────

def expanding_splits(
    dates:          list[str],
    min_train:      int = 36,
    step:           int = 1,
) -> Generator[tuple[list[str], str], None, None]:
    """
    Yield (train_dates, test_date) tuples using an expanding window.

    Args:
        dates:      Sorted list of all available date strings.
        min_train:  Minimum number of training periods before the first test.
        step:       Number of periods to advance between splits (default 1).

    Yields:
        (train_dates, test_date) where train_dates are all dates before test_date.
    """
    for i in range(min_train, len(dates), step):
        train = dates[:i]
        test  = dates[i]
        yield train, test


def rolling_splits(
    dates:      list[str],
    train_size: int = 48,
    step:       int = 1,
) -> Generator[tuple[list[str], str], None, None]:
    """
    Yield (train_dates, test_date) tuples using a fixed-size rolling window.

    Args:
        dates:      Sorted list of all available date strings.
        train_size: Number of training periods in each window.
        step:       Number of periods to advance between splits.

    Yields:
        (train_dates, test_date) with exactly train_size training dates.
    """
    for i in range(train_size, len(dates), step):
        train = dates[i - train_size: i]
        test  = dates[i]
        yield train, test


# ── Convenience wrappers ──────────────────────────────────────────────────────

def get_backtest_dates(
    dates:          list[str],
    min_train:      int  = 36,
    window_type:    str  = "expanding",
    rolling_window: int  = 48,
) -> list[tuple[list[str], str]]:
    """
    Return all (train_dates, test_date) splits as a list.

    Args:
        dates:          Sorted list of date strings.
        min_train:      Minimum training periods (expanding mode).
        window_type:    "expanding" or "rolling".
        rolling_window: Training window size (rolling mode only).

    Returns:
        List of (train_dates, test_date) tuples.
    """
    if window_type == "rolling":
        return list(rolling_splits(dates, rolling_window))
    return list(expanding_splits(dates, min_train))


def date_to_label(date_str: str) -> str:
    """Convert YYYY-MM-DD to a short display label (MMM-YYYY)."""
    try:
        return pd.Timestamp(date_str).strftime("%b %Y")
    except Exception:
        return date_str
