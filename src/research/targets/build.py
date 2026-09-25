"""WP3 target construction: benchmark-relative 12-month forward labels.

Labels are computed from RAW closes and corporate actions known through the
horizon endpoint, so no provider-adjusted (look-ahead) close ever enters a
label. Only ``target_observable`` observations are labelled; censored
observations receive ``target_observable = false`` semantics and are never
converted into a 0 or a negative.

Separation of concerns:

* ``build.build_targets`` - computation over injected frames;
* ``matching`` - deterministic trading-calendar date resolution;
* ``observability`` - the training-time metadata (feature_asof / target_known_at).

No model fitting, no feature engineering, no scoring happens here.

Performance, key-correctness and memory notes
---------------------------------------------
The WP2C silver price table holds ~3.4M rows. Two naive designs both fail at
this scale and both are avoided here:

* a Python dict keyed by ISO date string for every security at once needs
  multiple GB and OOM-kills the 4 GB container - so each series is kept as a
  compact numpy ``int64`` day array (``YYYYMMDD``) plus a ``float64`` close
  array and resolved with ``np.searchsorted``;
* re-normalising a ~4.9k-date calendar per row is ~1.2 billion timestamp
  conversions - so date resolution uses pre-normalised int keys and ``bisect``
  (:func:`matching.match_target_window_keys`), with ONE merged key list per
  security and ONE horizon lookup per anchor date.

Peak Python-object footprint is one series plus one ``int`` key list, not all
of them. The per-row return is the SAME quantity
:func:`src.research.data.pit_prices.pit_return` defines; a regression test
asserts exact agreement on a fixture so the fast path cannot drift from the
audited primitive.

The price series is keyed by the permanent ``security_id`` FIRST and by the
provider ``symbol`` only as a fallback, because WP2C stores the silver price
``ticker`` as the ``security_id`` (``ADCT``) while the membership panel's
``symbol`` may hold a provider pseudo-symbol (``ADCT_OLD``). Joining on
``symbol`` alone would silently censor every reused/delisted name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import pit_prices
from .matching import match_target_window_keys

TARGET_COLUMNS = (
    "security_id", "ticker", "feature_asof", "target_start", "target_end",
    "target_horizon_days_actual", "future_12m_stock_return", "future_12m_benchmark_return",
    "future_12m_excess_return", "outperform_12m", "target_observable", "target_censored",
    "target_censor_reason",
)


class TargetBuildError(RuntimeError):
    """Raised when target inputs are structurally invalid."""


def _require_columns(frame, columns, label):
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TargetBuildError("%s is missing required column(s): %s" % (label, ", ".join(missing)))


def _clean_text(value):
    if value is None:
        return None
    text = str(value)
    return None if text in ("", "None", "nan", "NaT") else text


def _day_int(value):
    """``YYYYMMDD`` int for a date-like value, or None."""
    text = _clean_text(value)
    if text is None:
        return None
    text = text[:10].replace("-", "")
    try:
        return int(text)
    except ValueError:
        return None


def _series_arrays(frame):
    """Return (sorted ``YYYYMMDD`` int array, close float array) for one price frame."""
    days = frame["trade_date"].astype(str).str.slice(0, 10).str.replace("-", "", regex=False)
    day_values = pd.to_numeric(days, errors="coerce").to_numpy(dtype="float64")
    close_values = pd.to_numeric(frame["raw_close"], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(day_values)
    day_values = day_values[valid].astype("int64")
    close_values = close_values[valid]
    order = np.argsort(day_values, kind="mergesort")
    return day_values[order], close_values[order]


def _close_on_or_before(day_values, close_values, target_day):
    """Raw close on ``target_day`` or the nearest PRIOR session (None if none)."""
    index = int(np.searchsorted(day_values, target_day, side="right")) - 1
    if index < 0:
        return None
    value = float(close_values[index])
    return None if not np.isfinite(value) else value


def _exact_close(day_values, close_values, target_day):
    index = int(np.searchsorted(day_values, target_day))
    if index < len(day_values) and day_values[index] == target_day:
        value = float(close_values[index])
        return value if np.isfinite(value) else None
    return None


class _Series:
    """Compact PIT view of one price series plus its precomputed action factors."""

    __slots__ = ("days", "closes", "actions")

    def __init__(self, frame, actions):
        self.days, self.closes = _series_arrays(frame)
        self.actions = _precompute_actions(actions, self.days, self.closes)

    def observation_keys(self, extra_keys=None):
        """Sorted ``YYYYMMDD`` keys for this series (union with ``extra_keys``)."""
        if extra_keys is None or not len(extra_keys):
            return [int(day) for day in self.days]
        return sorted(set(int(day) for day in self.days).union(extra_keys))

    def return_between(self, start_int, end_int):
        """PIT total return over ``[start, end]`` using actions known by ``end``.

        Identical to :func:`pit_prices.pit_return`. ``None`` when either endpoint
        is missing (never approximated).
        """
        if start_int is None or end_int is None or start_int >= end_int:
            return None
        start_raw = _exact_close(self.days, self.closes, start_int)
        end_raw = _exact_close(self.days, self.closes, end_int)
        if start_raw is None or end_raw is None or start_raw == 0.0:
            return None
        factor = 1.0
        for effective, applied in self.actions:
            if start_int < effective <= end_int:
                factor *= applied
        return (end_raw / (start_raw * factor)) - 1.0


def _precompute_actions(actions, day_values, close_values):
    """Return ``[(effective_day_int, factor)]`` for the actions known at build time.

    The factor mirrors :func:`pit_prices.action_factor`; a dividend whose ex-date
    close is unknown is skipped rather than assigned an invented value.
    """
    prepared = []
    for action in actions or []:
        effective = _day_int(getattr(action, "effective_date", None))
        if effective is None:
            continue
        if action.kind == pit_prices.SPLIT:
            if not action.numerator or not action.denominator:
                continue
            prepared.append((effective, float(action.denominator) / float(action.numerator)))
        elif action.kind == pit_prices.DIVIDEND:
            close_on_ex = _close_on_or_before(day_values, close_values, effective)
            if close_on_ex in (None, 0.0):
                continue
            prepared.append((effective, (close_on_ex - float(action.amount)) / close_on_ex))
    prepared.sort(key=lambda item: item[0])
    return prepared


def _actions_by_ticker(actions):
    grouped = {}
    for action in actions or []:
        grouped.setdefault(getattr(action, "ticker", None), []).append(action)
    return grouped


def build_targets(panel, stock_prices, benchmark_prices, stock_actions=None, benchmark_actions=None,
                  horizon_months=12):
    """Build the WP3 target rows from an approved WP2C panel.

    ``panel`` must carry ``security_id``, ``snapshot_date``, ``target_observable``,
    ``target_censored`` and ``target_censor_reason``. ``stock_prices`` is the long
    raw-close frame (``security_id``/``ticker``/``trade_date``/``raw_close``);
    ``benchmark_prices`` is the SPY raw-close frame.

    Returns a DataFrame with :data:`TARGET_COLUMNS`. A row is labelled only when
    both the panel's ``target_observable`` is true AND both return legs resolve;
    otherwise the row is emitted unobservable with an explicit reason.
    """
    _require_columns(panel, ("security_id", "snapshot_date", "target_observable"), "panel")
    _require_columns(stock_prices, ("ticker", "trade_date", "raw_close"), "stock_prices")
    _require_columns(benchmark_prices, ("ticker", "trade_date", "raw_close"), "benchmark_prices")

    benchmark_view = _Series(benchmark_prices, list(benchmark_actions or []))
    benchmark_keys = benchmark_view.observation_keys()

    stock_actions = _actions_by_ticker(stock_actions)

    # Build compact per-security series; the big frame is released afterwards.
    stock_views = {}
    series_column = "security_id" if "security_id" in stock_prices.columns else "ticker"
    for key, chunk in stock_prices.groupby(series_column):
        security_key = str(key)
        stock_views[security_key] = _Series(chunk, _flatten(stock_actions, security_key))

    # Group panel rows by security so each series/calendar is expanded once.
    panel_rows = {}
    for record in panel.itertuples():
        security_id = str(record.security_id)
        panel_rows.setdefault(security_id, []).append(record)

    horizon_cache = {}
    rows = []
    for security_id, records in panel_rows.items():
        provider_symbol = None
        view = stock_views.get(security_id)
        if view is None:
            for record in records:
                candidate = _clean_text(getattr(record, "symbol", None)) or _clean_text(getattr(record, "ticker", None))
                if candidate and candidate in stock_views:
                    view = stock_views[candidate]
                    provider_symbol = candidate
                    break
        keys = view.observation_keys(benchmark_keys) if view is not None else benchmark_keys
        for record in records:
            symbol = _clean_text(getattr(record, "symbol", None)) or provider_symbol or security_id
            rows.append(_build_row(record, security_id, symbol, view, benchmark_view, keys,
                                   horizon_months, horizon_cache))

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=list(TARGET_COLUMNS))
    frame = frame[list(TARGET_COLUMNS)]
    return frame.sort_values(by=["security_id", "feature_asof"], kind="mergesort").reset_index(drop=True)


def _flatten(grouped, key):
    return list(grouped.get(key, []))


def _build_row(record, security_id, symbol, view, benchmark_view, keys, horizon_months, horizon_cache):
    feature_asof = getattr(record, "snapshot_date", None)
    panel_observable = bool(getattr(record, "target_observable", False))
    panel_reason = getattr(record, "target_censor_reason", None)
    if panel_reason is None or (isinstance(panel_reason, float) and pd.isna(panel_reason)):
        panel_reason = None

    window = match_target_window_keys(keys, feature_asof, horizon_months=horizon_months,
                                      horizon_cache=horizon_cache)

    if not panel_observable:
        return _row(security_id, symbol, feature_asof, window, None, None, None, False, True,
                    panel_reason or "panel_target_unobservable")
    if view is None or not window["observable"]:
        return _row(security_id, symbol, feature_asof, window, None, None, None, False, True,
                    "no_observation_at_or_after_horizon")

    start_int = _day_int(window["target_start"])
    end_int = _day_int(window["target_end"])
    stock_return = view.return_between(start_int, end_int)
    bench_return = benchmark_view.return_between(start_int, end_int)
    if stock_return is None or bench_return is None:
        return _row(security_id, symbol, feature_asof, window, stock_return, bench_return, None, False, True,
                    "return_leg_unavailable")

    excess = float(stock_return) - float(bench_return)
    return _row(security_id, symbol, feature_asof, window, float(stock_return), float(bench_return), excess,
                True, False, None)


def _row(security_id, symbol, feature_asof, window, stock_return, bench_return, excess,
         observable, censored, reason):
    winner = None
    if excess is not None:
        winner = 1 if excess > 0 else 0
    return {
        "security_id": security_id,
        "ticker": symbol,
        "feature_asof": str(feature_asof)[:10] if feature_asof is not None else None,
        "target_start": window["target_start"],
        "target_end": window["target_end"],
        "target_horizon_days_actual": window["target_horizon_days_actual"],
        "future_12m_stock_return": stock_return,
        "future_12m_benchmark_return": bench_return,
        "future_12m_excess_return": excess,
        "outperform_12m": winner,
        "target_observable": bool(observable),
        "target_censored": bool(censored),
        "target_censor_reason": reason,
    }
