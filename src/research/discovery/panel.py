"""WP5 point-in-time feature-panel materialisation.

This module turns the certified WP2C gold panel, the WP2B silver prices/actions,
the SPY benchmark series and the SEC EDGAR silver fundamentals into an explicit
point-in-time FEATURE PANEL: one row per (``security_id``, ``feature_asof``)
carrying a value for every candidate feature that the certified inputs actually
support.

Integrity rules (enforced, not documented only):

* every input must have been public at the prediction instant, so a feature uses
  only prices at/before the row's price date and only filings whose availability
  instant is at/before the row's prediction date;
* corporate actions are applied point-in-time through the certified WP4
  :class:`~src.research.keyes.variables.PriceHistory`;
* nothing is estimated and nothing is zero-filled: a feature without the filed
  evidence it needs stays missing, and ``abnormal_volume`` is UNAVAILABLE because
  the certified PIT price table carries no volume column;
* discovery rows are restricted to the DEVELOPMENT window (strictly before the
  locked-holdout embargo cutoff) so no locked-holdout row can ever be consumed;
* the panel is deterministic and no input frame is mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.availability import to_utc_timestamp
from ..data.roa_roic import compute_return_metrics
from ..features.return_metrics import compute_v2_roa
from ..holdout import embargo_cutoff, holdout_mask
from ..keyes.variables import (
    ComputationConfig,
    FundamentalHistory,
    PriceHistory,
    assert_no_target_leakage,
    beta_against_benchmark,
)

PANEL_COLUMNS = ("security_id", "ticker", "feature_asof", "price_date", "raw_close",
                 "adjusted_close_pit", "target_observable")

FIVE_YEAR_DAYS = 1826
ONE_YEAR_DAYS = 365


class FeaturePanelError(RuntimeError):
    """Raised when the PIT feature panel cannot be materialised honestly."""


def _day_ordinal(value):
    stamp = to_utc_timestamp(value)
    if stamp is None:
        return None
    return int(stamp.normalize().value // (24 * 3600 * 10 ** 9))


def _shift_days(day, days):
    stamp = to_utc_timestamp(day)
    if stamp is None:
        return None
    return (stamp + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _ratio(numerator, denominator, positive_denominator=True):
    """Safe ratio; a zero (or, optionally, non-positive) denominator is unavailable."""
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None:
        return None
    if bottom == 0.0 or (positive_denominator and bottom <= 0.0):
        return None
    return top / bottom


def _cagr(now, then, years=5.0):
    start = _finite(then)
    end = _finite(now)
    if start is None or end is None or start <= 0.0 or end <= 0.0:
        return None
    return (end / start) ** (1.0 / float(years)) - 1.0


def _growth(now, prior):
    start = _finite(prior)
    end = _finite(now)
    if start is None or end is None or start <= 0.0:
        return None
    return end / start - 1.0


# ── Development-row restriction (locked-holdout safety) ───────────────────────

def development_mask(frame, asof_col="feature_asof", observable_col="target_observable"):
    """Rows usable for DISCOVERY: pre-embargo AND with an observable label.

    The mask is derived from :func:`src.research.holdout.embargo_cutoff` so the
    holdout boundary can only ever change together with the frozen holdout
    definition. A holdout row can therefore never enter the discovery slice.
    """
    cutoff = embargo_cutoff()
    flags = []
    for value in frame[asof_col].tolist():
        stamp = to_utc_timestamp(value)
        flags.append(bool(stamp is not None and stamp < cutoff))
    mask = pd.Series(flags, index=frame.index)
    if observable_col in frame.columns:
        mask = mask & frame[observable_col].astype(bool)
    return mask


# ── Dividends ─────────────────────────────────────────────────────────────────

def _dividend_index(actions):
    """Per-security sorted ``(effective_day_ordinal, amount)`` dividend arrays."""
    if actions is None or len(actions) == 0 or "kind" not in actions.columns:
        return {}
    dividends = actions.loc[actions["kind"].astype(str) == "dividend"]
    index = {}
    for security_id, chunk in dividends.groupby("security_id", sort=True):
        days, amounts = [], []
        for record in chunk.to_dict("records"):
            day = _day_ordinal(record.get("effective_date"))
            amount = _finite(record.get("amount"))
            if day is None or amount is None:
                continue
            days.append(day)
            amounts.append(amount)
        if days:
            order = np.argsort(np.asarray(days, dtype="int64"), kind="mergesort")
            index[security_id] = (np.asarray(days, dtype="int64")[order],
                                  np.asarray(amounts, dtype="float64")[order])
    return index


def _dividend_trailing(index, security_id, end_day, lookback_days):
    """Declared dividends with ``end_day - lookback < effective <= end_day``.

    Returns ``0.0`` when the security has dividend records but none inside the
    window (a genuine non-payer), and ``None`` when no dividend series exists at
    all, so an absent series is never mistaken for a zero yield.
    """
    if security_id not in index:
        return None
    days, amounts = index[security_id]
    end = _day_ordinal(end_day)
    if end is None:
        return None
    low = end - int(lookback_days)
    left = int(np.searchsorted(days, low, side="right"))
    right = int(np.searchsorted(days, end, side="right"))
    if right <= left:
        return 0.0
    return float(amounts[left:right].sum())


# ── Feature computation for one observation ──────────────────────────────────

def _ttm(fundamentals, cik, field, asof):
    if cik is None:
        return None
    return fundamentals.trailing_twelve_month_value(cik, field, asof)


def _latest(fundamentals, cik, field, asof):
    if cik is None:
        return None
    return fundamentals.latest_value(cik, field, asof)


def _first(record):
    return None if record is None else record[0]


def compute_row_features(security_id, ticker, asof_day, price_day, cik, history, benchmark,
                         fundamentals, dividends, config):
    """Compute every candidate feature for one observation (missing = ``None``)."""
    values = {name: None for name in _CANDIDATE_NAMES}

    price = history.raw_on(security_id, price_day, "prior")

    # ── Price-based ────────────────────────────────────────────────────────
    values["six_month_momentum"] = _finite(
        history.trailing_return(security_id, price_day, config.momentum_short_days))
    values["twelve_month_momentum"] = _finite(
        history.trailing_return(security_id, price_day, config.momentum_long_days))
    values["five_year_price_gain"] = _finite(
        history.trailing_return(security_id, price_day, config.five_year_days))
    # abnormal_volume: the certified PIT price table carries no volume column, so
    # the feature is honestly UNAVAILABLE rather than fabricated.
    values["abnormal_volume"] = None

    beta, _observations = beta_against_benchmark(history, benchmark, security_id, price_day, config)
    values["beta"] = _finite(beta)

    declared = _dividend_trailing(dividends, security_id, price_day, ONE_YEAR_DAYS)
    values["dividend_yield"] = _ratio(declared, price)

    # ── Filed fundamentals (point-in-time) ─────────────────────────────────
    eps_now = _ttm(fundamentals, cik, "eps", asof_day)
    eps_1y = _ttm(fundamentals, cik, "eps", _shift_days(asof_day, -ONE_YEAR_DAYS))
    eps_2y = _ttm(fundamentals, cik, "eps", _shift_days(asof_day, -2 * ONE_YEAR_DAYS))
    eps_5y = _ttm(fundamentals, cik, "eps", _shift_days(asof_day, -FIVE_YEAR_DAYS))
    revenue_now = _ttm(fundamentals, cik, "revenue", asof_day)
    revenue_1y = _ttm(fundamentals, cik, "revenue", _shift_days(asof_day, -ONE_YEAR_DAYS))
    revenue_2y = _ttm(fundamentals, cik, "revenue", _shift_days(asof_day, -2 * ONE_YEAR_DAYS))
    revenue_5y = _ttm(fundamentals, cik, "revenue", _shift_days(asof_day, -FIVE_YEAR_DAYS))
    net_income = _ttm(fundamentals, cik, "net_income", asof_day)
    gross_profit = _ttm(fundamentals, cik, "gross_profit", asof_day)
    operating_income = _ttm(fundamentals, cik, "operating_income", asof_day)
    free_cash_flow = _ttm(fundamentals, cik, "free_cash_flow", asof_day)
    assets_end = _latest(fundamentals, cik, "total_assets", asof_day)
    assets_begin = _latest(fundamentals, cik, "total_assets", _shift_days(asof_day, -ONE_YEAR_DAYS))
    equity = _latest(fundamentals, cik, "equity", asof_day)
    total_debt = _latest(fundamentals, cik, "total_debt", asof_day)
    cash = _latest(fundamentals, cik, "cash", asof_day)
    shares = _latest(fundamentals, cik, "shares_outstanding", asof_day)

    market_cap = None
    if price is not None and _first(shares) is not None:
        share_count = _finite(_first(shares))
        if share_count is not None and share_count > 0.0:
            market_cap = price * share_count
    values["market_cap"] = _finite(market_cap)

    values["earnings_yield"] = _ratio(_first(net_income), market_cap)
    # current_pe mirrors Keyes X8: undefined for non-positive trailing earnings.
    eps_value = _finite(_first(eps_now))
    if price is not None and eps_value is not None and eps_value > 0.0:
        values["current_pe"] = price / eps_value
    values["price_to_book"] = _ratio(market_cap, _first(equity))
    values["price_to_sales"] = _ratio(market_cap, _first(revenue_now))
    values["free_cash_flow_yield"] = _ratio(_first(free_cash_flow), market_cap)
    values["roe"] = _ratio(_first(net_income), _first(equity))
    values["gross_margin"] = _ratio(_first(gross_profit), _first(revenue_now))
    values["operating_margin"] = _ratio(_first(operating_income), _first(revenue_now))
    values["debt_to_equity"] = _ratio(_first(total_debt), _first(equity))

    roa, _method = compute_v2_roa(_first(net_income), _first(assets_end), _first(assets_begin))
    values["roa"] = _finite(roa)

    try:
        metrics = compute_return_metrics(
            _first(net_income), _first(assets_end),
            operating_income=_first(operating_income), total_debt=_first(total_debt),
            equity=_first(equity), cash=_first(cash),
        )
        values["roic"] = _finite(metrics.roic)
    except Exception:  # pragma: no cover - engine raises on structural misuse only
        values["roic"] = None

    values["five_year_eps_growth"] = _cagr(_first(eps_now), _first(eps_5y))
    values["five_year_revenue_growth"] = _cagr(_first(revenue_now), _first(revenue_5y))

    eps_g1 = _growth(_first(eps_now), _first(eps_1y))
    eps_g0 = _growth(_first(eps_1y), _first(eps_2y))
    values["eps_growth_acceleration"] = None if eps_g1 is None or eps_g0 is None else eps_g1 - eps_g0

    revenue_g1 = _growth(_first(revenue_now), _first(revenue_1y))
    revenue_g0 = _growth(_first(revenue_1y), _first(revenue_2y))
    values["revenue_growth_acceleration"] = None if revenue_g1 is None or revenue_g0 is None else revenue_g1 - revenue_g0

    return values


# Resolved lazily to avoid a circular import at module load (catalog -> panel).
from .catalog import FEATURE_NAMES as _CANDIDATE_NAMES  # noqa: E402


def build_feature_panel(panel, prices, actions, fundamentals, cik_by_ticker,
                        benchmark_prices, benchmark_actions, config=None,
                        restrict_to_development=True, *,
                        allow_locked_holdout=False):
    """Materialise the PIT feature panel.

    Returns ``(frame, summary)`` where ``frame`` has one row per development
    (or, when unrestricted, per panel) observation and one column per candidate
    feature. ``summary`` records row counts, the embargo cutoff used and an
    explicit locked-holdout exclusion/proof count.

    ``allow_locked_holdout`` is an explicit opt-in for evaluation-only paths.
    The default is unchanged: any surviving locked-holdout row raises. When
    ``True``, those rows are permitted and are reported through
    ``locked_holdout_rows_in_panel`` as a counted proof rather than silently
    hidden.
    """
    config = config or ComputationConfig()
    for frame, what in ((panel, "panel"), (prices, "prices"), (actions, "actions"),
                        (fundamentals, "fundamentals"), (benchmark_prices, "benchmark prices")):
        assert_no_target_leakage(frame, what)

    required = ("security_id", "ticker", "feature_asof", "price_date")
    missing = [name for name in required if name not in panel.columns]
    if missing:
        raise FeaturePanelError("panel is missing column(s): %s" % ", ".join(missing))

    working = panel.copy()
    eligible = int(len(working))
    if restrict_to_development:
        mask = development_mask(working)
        working = working.loc[mask]
    working = working.drop_duplicates(subset=["security_id", "feature_asof"], keep="first")
    working = working.sort_values(["security_id", "feature_asof"], kind="mergesort").reset_index(drop=True)

    history = PriceHistory(prices, actions)
    benchmark = PriceHistory(benchmark_prices, benchmark_actions)
    fundamentals_by_cik = FundamentalHistory.from_frame(fundamentals)
    dividends = _dividend_index(actions)

    rows = []
    for record in working.itertuples():
        asof_day = str(record.feature_asof)[:10]
        price_day = str(record.price_date)[:10] if record.price_date is not None else asof_day
        ticker = str(record.ticker)
        cik = (cik_by_ticker or {}).get(ticker.upper())
        values = compute_row_features(record.security_id, ticker, asof_day, price_day, cik,
                                      history, benchmark, fundamentals_by_cik, dividends, config)
        row = {"security_id": record.security_id, "ticker": ticker, "feature_asof": asof_day}
        row.update(values)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["security_id", "ticker", "feature_asof"] + list(_CANDIDATE_NAMES))

    # Explicit exclusion proof: no locked-holdout row survived the restriction
    # unless the caller opted in for the evaluation-only unrestricted path. The
    # count is always retained in summary as a deterministic audit fact.
    if "feature_asof" in frame.columns and len(frame):
        holdout_rows = int(holdout_mask(frame).sum())
    else:
        holdout_rows = 0
    if holdout_rows and not allow_locked_holdout:
        raise FeaturePanelError("feature panel contains %d locked-holdout row(s)" % holdout_rows)

    coverage = {}
    for name in _CANDIDATE_NAMES:
        if name in frame.columns:
            series = pd.to_numeric(frame[name], errors="coerce")
            coverage[name] = {
                "rows": int(len(frame)),
                "present": int(series.notna().sum()),
                "coverage": float(series.notna().mean()) if len(frame) else 0.0,
            }
        else:
            coverage[name] = {"rows": int(len(frame)), "present": 0, "coverage": 0.0}

    summary = {
        "eligible_panel_rows": eligible,
        "rows": int(len(frame)),
        "restricted_to_development": bool(restrict_to_development),
        "embargo_cutoff": str(embargo_cutoff())[:19],
        "locked_holdout_rows_in_panel": holdout_rows,
        "coverage": coverage,
    }
    return frame, summary
