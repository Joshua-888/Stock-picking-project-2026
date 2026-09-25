"""Deterministic, market-calendar-aware target date matching.

The legacy V1 behaviour (read-only reference: ``src/features/target_creation``)
matched a horizon date to the nearest available month-end within +/-15 days.
That rule is ambiguous (a date can be equidistant) and can silently pick a
pre-horizon observation. V2 replaces it with an explicit, deterministic rule:

    target_start = the observation at or immediately AFTER the intended start;
    target_end   = the observation at or immediately AFTER the intended horizon.

Never an observation strictly BEFORE the intended horizon, so a 12-month label
cannot be shortened into an 11-month label to "find" a value. When no
observation exists on/after the horizon the label is unobservable (returns
None), never approximated.

The functions are pure: they take available observation dates and return the
chosen date plus the realised horizon length. That makes weekends, holidays,
delistings, acquisition endings and missing future observations testable
without any network access.

Performance contract (why there are two layers)
-----------------------------------------------
The WP2C panel has ~121k rows over a ~4.9k-date calendar. Parsing the whole
calendar twice per row is ~1.2 billion timestamp conversions and OOM-kills the
container, so ``match_target_window_keys`` operates on pre-normalised
``YYYYMMDD`` int keys and resolves a date with ``bisect``. ``match_target_window``
stays the stable, human-facing API (accepts any parseable date-like list) and
delegates to the key path after normalising ONCE, guaranteeing byte-identical
results between the two entry points.
"""

from __future__ import annotations

import datetime
from bisect import bisect_left

import pandas as pd

from ..data.availability import to_utc_timestamp


class TargetMatchingError(ValueError):
    """Raised when target matching inputs are invalid."""


def _date_key(value):
    """``YYYYMMDD`` integer key for a date-like value, or ``None``."""
    stamp = to_utc_timestamp(value)
    if stamp is None:
        return None
    day = stamp.tz_convert("UTC").date()
    return day.year * 10000 + day.month * 100 + day.day


def _key_to_date(key):
    return datetime.date(key // 10000, (key // 100) % 100, key % 100)


def _normalise_keys(observation_dates):
    """Sorted, de-duplicated ``YYYYMMDD`` int keys for any date-like iterable."""
    keys = {_date_key(value) for value in observation_dates}
    keys.discard(None)
    return sorted(keys)


def _normalise_dates(observation_dates):
    """Return a sorted, de-duplicated list of ``datetime.date`` values."""
    return [_key_to_date(key) for key in _normalise_keys(observation_dates)]


def intended_horizon_date(start, horizon_months):
    """Calendar horizon date: ``start`` + ``horizon_months`` months, same day-of-month.

    Uses pandas DateOffset month arithmetic (which clamps e.g. Jan 31 + 1 month to
    Feb 28/29) so the intended horizon is always defined, independent of trading.
    """
    stamp = to_utc_timestamp(start)
    if stamp is None:
        raise TargetMatchingError("target start is missing or unparseable: %r" % (start,))
    shifted = stamp + pd.DateOffset(months=int(horizon_months))
    return shifted.tz_convert("UTC").date()


def intended_horizon_key(start, horizon_months, _cache=None):
    """``YYYYMMDD`` key of the intended horizon for ``start`` (optionally cached)."""
    anchor_key = _date_key(start)
    if _cache is not None and anchor_key in _cache:
        return _cache[anchor_key]
    day = intended_horizon_date(start, horizon_months)
    key = day.year * 10000 + day.month * 100 + day.day
    if _cache is not None:
        _cache[anchor_key] = key
    return key


def _first_key_on_or_after(sorted_keys, target_key):
    """Smallest sorted key >= ``target_key`` (``None`` when none exists)."""
    index = bisect_left(sorted_keys, target_key)
    if index >= len(sorted_keys):
        return None
    return sorted_keys[index]


def match_observation(observation_dates, intended_date):
    """First observation at or immediately AFTER ``intended_date``; else None.

    Returns a ``datetime.date`` or ``None`` when no observation exists on/after
    the intended date (e.g. the security delisted or the data ends first).
    """
    target = to_utc_timestamp(intended_date)
    if target is None:
        raise TargetMatchingError("intended date is missing or unparseable: %r" % (intended_date,))
    target_day = target.tz_convert("UTC").date()
    for day in _normalise_dates(observation_dates):
        if day >= target_day:
            return day
    return None


def match_target_window(observation_dates, feature_asof, horizon_months=12):
    """Resolve (target_start, target_end) and realised horizon days.

    ``observation_dates`` are the union of stock and benchmark observations.
    ``feature_asof`` anchors the window. ``target_start`` is the first
    observation on/after ``feature_asof`` (the last known price is at/ before the
    prediction instant, so the return leg begins at the anchor). ``target_end``
    is the first observation on/after the intended horizon.

    Returns a dict with ``target_start``, ``target_end``, ``intended_end``,
    ``target_horizon_days_actual`` (or None), and ``observable`` (bool).

    Kept as the human-facing API; normalises once and delegates to the key path.
    """
    return match_target_window_keys(_normalise_keys(observation_dates), feature_asof, horizon_months)


def match_target_window_keys(sorted_keys, feature_asof, horizon_months=12, horizon_cache=None):
    """Same result as :func:`match_target_window` over pre-normalised int keys.

    ``sorted_keys`` MUST be sorted ascending ``YYYYMMDD`` ints (the output of
    :func:`_normalise_keys` or :func:`_key_to_date` on such a list). Reusing it
    across rows is what keeps the build O(rows * log(dates)) instead of
    O(rows * dates).
    """
    anchor_key = _date_key(feature_asof)
    if anchor_key is None:
        raise TargetMatchingError("feature_asof is missing or unparseable: %r" % (feature_asof,))
    intended_key = intended_horizon_key(feature_asof, horizon_months, _cache=horizon_cache)
    start_key = _first_key_on_or_after(sorted_keys, anchor_key)
    end_key = _first_key_on_or_after(sorted_keys, intended_key)
    intended_iso = _key_to_date(intended_key).isoformat()
    if start_key is None or end_key is None or end_key <= start_key:
        return {
            "target_start": None if start_key is None else _key_to_date(start_key).isoformat(),
            "target_end": None if end_key is None else _key_to_date(end_key).isoformat(),
            "intended_end": intended_iso,
            "target_horizon_days_actual": None,
            "observable": False,
        }
    start_day = _key_to_date(start_key)
    end_day = _key_to_date(end_key)
    return {
        "target_start": start_day.isoformat(),
        "target_end": end_day.isoformat(),
        "intended_end": intended_iso,
        "target_horizon_days_actual": int((end_day - start_day).days),
        "observable": True,
    }
