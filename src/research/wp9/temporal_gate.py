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
LIVE_LAYER_RECORDS_REL = Path("artifacts") / "research" / "wp9_live" / "layer_records.json"
LIVE_PRICES_NAME = "wp9_live_prices"


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


def live_price_trade_dates(root: Path | None = None) -> list[str]:
    """Return sorted unique trade dates from the WP9A live-forward price table.

    Reads ``artifacts/research/wp9_live/layer_records.json`` and the referenced
    ``wp9_live_prices`` parquet version. Missing or empty records/tables fail
    closed so a live-mode cadence source cannot silently fall back to a stale
    or fabricated series.
    """
    root = Path(root or ROOT)
    records_path = root / LIVE_LAYER_RECORDS_REL
    if not records_path.is_file():
        raise TemporalGateError("missing WP9 live layer records: %s" % records_path)
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalGateError("invalid WP9 live layer records at %s: %s" % (records_path, exc)) from exc
    version = records.get("silver_prices")
    if not version:
        raise TemporalGateError("WP9 live layer records have no silver_prices version")
    data_root = root / "data" / "research_v2"
    version_dir = layers.layer_root(data_root, "silver", LIVE_PRICES_NAME) / str(version)
    table_path = version_dir / "data.parquet"
    if not table_path.is_file():
        raise TemporalGateError("WP9 live price series is missing: %s" % table_path)
    try:
        frame = pd.read_parquet(table_path, columns=["trade_date"])
    except Exception as exc:  # pragma: no cover - malformed live artifact
        raise TemporalGateError("cannot read WP9 live price series: %s" % exc) from exc
    dates = sorted({str(value)[:10] for value in frame["trade_date"] if pd.notna(value)})
    if not dates:
        raise TemporalGateError("WP9 live price series has no trade dates")
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


def _month_key(value):
    stamp = to_utc_timestamp(value)
    if stamp is None:
        raise TemporalGateError("cannot parse month-bearing value: %r" % (value,))
    return stamp.strftime("%Y-%m")


def _as_utc(stamp):
    """Normalize a timestamp to tz-aware UTC for safe cross-zone comparison."""
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def calendar_last_day_of_month(month: str) -> pd.Timestamp:
    """Calendar month-end timestamp for a ``YYYY-MM`` month key."""
    stamp = to_utc_timestamp(month + "-01")
    if stamp is None:
        raise TemporalGateError("cannot parse month key: %r" % (month,))
    return pd.Timestamp(year=stamp.year, month=stamp.month, day=stamp.daysinmonth)


def last_business_day_for_month(month: str) -> str:
    """Final US weekday of a calendar month (weekend-aware only).

    No external holiday calendar is imported. A holiday month-end is therefore
    resolved from the actual trade dates by
    :func:`eligible_score_date_for_month`, never by this weekday helper alone.
    """
    day = calendar_last_day_of_month(month)
    while day.dayofweek >= 5:  # 5 == Saturday, 6 == Sunday
        day = day - pd.Timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def _max_trade_date_in_month(month: str, trade_dates: Sequence[str]) -> str | None:
    candidates = []
    for value in trade_dates:
        stamp = to_utc_timestamp(value)
        if stamp is not None and stamp.strftime("%Y-%m") == month:
            candidates.append(stamp.strftime("%Y-%m-%d"))
    return max(candidates) if candidates else None


def month_has_following_data(month: str, trade_dates: Sequence[str]) -> bool:
    """True when the series contains any trade date after ``month`` ends.

    This is the no-external-calendar coverage signal: a calendar month is proven
    complete only when the series continues into the following month. A current
    or truncated month has no following month data and is therefore incomplete.
    """
    month_end = calendar_last_day_of_month(month)
    if month_end.tzinfo is None:
        month_end = month_end.tz_localize("UTC")
    for value in trade_dates:
        stamp = to_utc_timestamp(value)
        if stamp is not None and stamp > month_end:
            return True
    return False


def eligible_score_date_for_month(
    asof: str,
    trade_dates: Sequence[str],
    *,
    require_coverage_complete: bool = True,
) -> str:
    """Return a month's last eligible trading/score date from the actual series.

    With ``require_coverage_complete=True`` the month must have at least one
    later observation before it is considered eligible; this detects a truncated
    or still-current month without importing a holiday calendar. Weekend/holiday
    month-ends therefore resolve to the last observed date in the month, while a
    data series ending before month-end (no following-month row) is rejected.
    """
    month = _month_key(asof)
    latest = _max_trade_date_in_month(month, trade_dates)
    if latest is None:
        raise TemporalGateError("no eligible trading/score date exists for month %s" % month)
    if require_coverage_complete and not month_has_following_data(month, trade_dates):
        raise TemporalGateError(
            "month %s coverage is incomplete; no following-month trade data proves month-end"
            % month
        )
    return latest


def first_eligible_official_snapshot_date(
    freeze_iso: str,
    trade_dates: Sequence[str],
    *,
    now_utc: pd.Timestamp | None = None,
) -> dict:
    """Resolve the first official monthly snapshot date without hardcoded dates.

    ``freeze_iso`` is the contract freeze timestamp. The first candidate month is
    the first full calendar month strictly after that freeze month. The exact
    eligible date is that month's last observed trade/score date if the series
    proves month coverage; otherwise the candidate is reported as not-yet
    eligible. No calendar holiday table, hardcoded October date, or future month
    is assumed.
    """
    freeze = to_utc_timestamp(freeze_iso)
    if freeze is None:
        raise TemporalGateError("freeze timestamp is not parseable: %r" % (freeze_iso,))
    # All comparisons are made in UTC. ``calendar_last_day_of_month`` returns a
    # tz-naive midnight, so normalize month-end instants to UTC before comparing
    # them with the (possibly tz-aware) freeze timestamp.
    if freeze.tzinfo is None:
        freeze = freeze.tz_localize("UTC")
    else:
        freeze = freeze.tz_convert("UTC")
    dates = sorted(
        {str(value)[:10] for value in trade_dates if to_utc_timestamp(value) is not None}
    )
    # The first candidate calendar month is the freeze's own month when its
    # calendar end is strictly after the freeze instant (e.g. an early-October
    # freeze), otherwise the following month. This never assumes a fixed date.
    freeze_month = freeze.strftime("%Y-%m")
    freeze_month_end = _as_utc(calendar_last_day_of_month(freeze_month))
    candidate_month = (
        freeze_month
        if freeze_month_end > freeze
        else (freeze + pd.DateOffset(months=1)).strftime("%Y-%m")
    )
    candidate_month_end = _as_utc(calendar_last_day_of_month(candidate_month))
    if candidate_month_end <= freeze:
        raise TemporalGateError("cannot resolve a candidate month after freeze timestamp")
    latest = _max_trade_date_in_month(candidate_month, dates)
    if latest is None:
        return {
            "eligible": False,
            "candidate_month": candidate_month,
            "first_eligible_date": None,
            "status": "NOT_READY",
            "reason": "no_trade_dates_in_first_full_month_after_freeze",
        }
    # A date inside ``candidate_month`` can still be on or before the freeze
    # instant (e.g. freeze late in the month with only an earlier session in
    # that month's series). Its conservative UTC close must be strictly after
    # the freeze before that month can ever be READY, even when following-month
    # data exists. Never silently advance here; the caller may advance one
    # whole candidate month deterministically and re-evaluate.
    latest_instant = price_available_at(latest)
    if latest_instant is None or latest_instant <= freeze:
        return {
            "eligible": False,
            "candidate_month": candidate_month,
            "first_eligible_date": latest,
            "status": "NOT_READY",
            "reason": "first_eligible_date_not_after_freeze",
        }
    if month_has_following_data(candidate_month, dates):
        return {
            "eligible": True,
            "candidate_month": candidate_month,
            "first_eligible_date": latest,
            "status": "READY",
            "reason": "coverage_complete",
        }
    return {
        "eligible": False,
        "candidate_month": candidate_month,
        "first_eligible_date": latest,
        "status": "NOT_READY",
        "reason": "coverage_incomplete_no_following_month_data",
    }


__all__ = [
    "TemporalGateError",
    "bound_price_trade_dates",
    "calendar_last_day_of_month",
    "closing_utc_for_asof",
    "current_utc",
    "eligible_score_date_for_month",
    "first_eligible_official_snapshot_date",
    "last_business_day_for_month",
    "last_eligible_score_date_for_month",
    "live_price_trade_dates",
    "month_has_following_data",
]
