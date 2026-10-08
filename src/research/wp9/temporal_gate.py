"""WP9 prospective temporal-gate primitives.

This module implements the pre-prediction controls without touching any frozen
scientific rule. It derives the last eligible trading/score date for a calendar
month from the bound certified silver price series only; no external holiday
calendar is imported. An official as-of is valid only when it is exactly that
month's last session in the certified series, is strictly after the contract
freeze timestamp, is no earlier than the latest already-written official
snapshot month, and its conservative UTC close instant has already elapsed at
run time.

Dry-run rehearsal never requires these official gates; the caller keeps dry-run
storage separated and non-evidentiary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..data import layers
from ..data.availability import price_available_at, to_utc_timestamp

ROOT = Path(__file__).resolve().parents[3]
LAYER_RECORDS_REL = Path("artifacts") / "research" / "wp2b_live" / "layer_records.json"
PRICES_NAME = "wp2b_sp500_pit_prices_silver"


class TemporalGateError(RuntimeError):
    """Raised when a prospective snapshot as-of violates the temporal gate."""


def current_utc() -> pd.Timestamp:
    """Return the current UTC instant used by the wall-clock eligibility policy."""
    return pd.Timestamp.now(tz="UTC")


def _layer_records(root: Path) -> dict:
    path = Path(root) / LAYER_RECORDS_REL
    if not path.is_file():
        raise TemporalGateError("missing certified WP2B layer records: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalGateError("invalid WP2B layer records at %s: %s" % (path, exc)) from exc


def bound_price_trade_dates(root: Path | None = None) -> list[str]:
    """Return sorted unique trade-date strings from the bound certified price series.

    The official monthly-cadence gate must be resolved against the exact bound
    silver price table (``wp2b_sp500_pit_prices_silver``), not a fuzzy calendar.
    """
    root = Path(root or ROOT)
    records = _layer_records(root)
    version = records.get("silver_prices")
    if not version:
        raise TemporalGateError("certified layer records have no silver_prices version")
    data_root = root / "data" / "research_v2"
    version_dir = layers.layer_root(data_root, "silver", PRICES_NAME) / str(version)
    table_path = version_dir / "data.parquet"
    if not table_path.is_file():
        raise TemporalGateError("bound certified price series is missing: %s" % table_path)
    try:
        frame = pd.read_parquet(table_path, columns=["trade_date"])
    except Exception as exc:  # pragma: no cover - malformed certified artifact
        raise TemporalGateError("cannot read bound certified price series: %s" % exc) from exc
    dates = sorted({str(value)[:10] for value in frame["trade_date"] if pd.notna(value)})
    if not dates:
        raise TemporalGateError("bound certified price series has no trade dates")
    return dates


def last_eligible_score_date_for_month(
    asof: str,
    trade_dates: Sequence[str],
) -> str:
    """Return the last certified trading/score date in ``asof``'s calendar month.

    ``asof`` must be a canonical date string. The returned date is the maximum
    trade date present in the certified series for that month, deterministically
    rejecting non-trading calendar month-ends.
    """
    stamp = to_utc_timestamp(asof)
    if stamp is None:
        raise TemporalGateError("as-of must be a parseable date: %r" % (asof,))
    month = stamp.strftime("%Y-%m")
    candidates = []
    for value in trade_dates:
        day = to_utc_timestamp(value)
        if day is not None and day.strftime("%Y-%m") == month:
            candidates.append(str(day.strftime("%Y-%m-%d")))
    if not candidates:
        raise TemporalGateError(
            "no eligible certified trading/score date exists for month %s" % month
        )
    return max(candidates)


def closing_utc_for_asof(asof: str) -> pd.Timestamp:
    """Conservative UTC close instant for a canonical as-of date."""
    stamp = price_available_at(asof)
    if stamp is None:
        raise TemporalGateError("cannot derive UTC close instant for as-of %r" % (asof,))
    return stamp


__all__ = [
    "TemporalGateError",
    "bound_price_trade_dates",
    "closing_utc_for_asof",
    "current_utc",
    "last_eligible_score_date_for_month",
]
