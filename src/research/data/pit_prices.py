"""Point-in-time price reconstruction from raw closes plus a corporate-action ledger.

A provider ``adjusted_close`` is NOT point-in-time safe: providers recompute the
whole adjusted history every time a new split or dividend occurs, so a stored
adjusted value silently blends information that did not exist at earlier dates.

This module rebuilds an adjusted series from the *raw* close and only the
corporate actions already known at a chosen as-of instant. For each trade date
``d`` and as-of date ``A``:

    adjusted_close_pit(d | A) = raw_close(d) * product(factor(a))
        for every action ``a`` with ``d < a.effective_date <= A``

* split factor: ``denominator / numerator`` (a 2-for-1 split halves pre-split prices);
* dividend factor: ``(close_on_ex - amount) / close_on_ex`` using the raw close
  on the ex-date, so an unavailable ex-date close simply excludes the action
  rather than inventing a factor.

Consequences that the adversarial tests rely on:

* a split/dividend dated AFTER ``A`` enters no factor, so every value used at
  ``A`` is byte-identical after that future action is added;
* a value at ``A`` itself is never adjusted by anything later than ``A``.

This module records and reconstructs provenance; it performs no feature
engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .availability import to_utc_timestamp

SPLIT = "split"
DIVIDEND = "dividend"
KINDS = (SPLIT, DIVIDEND)

PRICE_SEMANTICS = {
    "raw_close": "session close exactly as printed by the provider; never adjusted",
    "adjusted_close_provider": "provider total-return close; NOT point-in-time safe",
    "adjusted_close_pit": "raw close scaled by corporate actions known at the as-of date only",
    "available_at": "trade date plus a conservative market close time in UTC",
}


class PitPriceError(RuntimeError):
    """Raised when a price frame or action ledger is unusable."""


@dataclass(frozen=True)
class CorporateAction:
    """One dated split or dividend with its price-adjustment factor inputs."""

    ticker: str
    kind: str
    effective_date: str
    numerator: float = None
    denominator: float = None
    amount: float = None
    raw_unix: float = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise PitPriceError("unknown corporate-action kind %r" % (self.kind,))
        if to_utc_timestamp(self.effective_date) is None:
            raise PitPriceError("corporate action needs a parseable effective_date: %r" % (self.effective_date,))
        if self.kind == SPLIT and (self.numerator in (None, 0) or self.denominator in (None, 0)):
            raise PitPriceError("split action needs non-zero numerator and denominator")
        if self.kind == DIVIDEND and self.amount is None:
            raise PitPriceError("dividend action needs an amount")

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "kind": self.kind,
            "effective_date": self.effective_date,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "amount": self.amount,
            "raw_unix": self.raw_unix,
        }


def _date_str(value):
    stamp = to_utc_timestamp(value)
    return stamp.strftime("%Y-%m-%d") if stamp is not None else None


def actions_from_yahoo_events(split_events, dividend_events, ticker=None):
    """Convert Yahoo chart events into :class:`CorporateAction` records.

    Events lacking a usable date are dropped (reported via the caller), never
    assigned an invented date.
    """
    actions = []
    for event in split_events or []:
        stamp = to_utc_timestamp(pd.Timestamp(event.get("date"), unit="s", tz="UTC"))
        if stamp is None:
            continue
        actions.append(
            CorporateAction(
                ticker=ticker,
                kind=SPLIT,
                effective_date=stamp.strftime("%Y-%m-%d"),
                numerator=float(event.get("numerator")) if event.get("numerator") is not None else None,
                denominator=float(event.get("denominator")) if event.get("denominator") is not None else None,
                raw_unix=event.get("date"),
            )
        )
    for event in dividend_events or []:
        stamp = to_utc_timestamp(pd.Timestamp(event.get("date"), unit="s", tz="UTC"))
        if stamp is None:
            continue
        amount = event.get("amount")
        actions.append(
            CorporateAction(
                ticker=ticker,
                kind=DIVIDEND,
                effective_date=stamp.strftime("%Y-%m-%d"),
                amount=float(amount) if amount is not None else None,
                raw_unix=event.get("date"),
            )
        )
    actions.sort(key=lambda item: (item.effective_date, item.kind))
    return actions


def actions_known_by(actions, asof):
    """Subset of ``actions`` whose effective date is at or before ``asof``."""
    moment = to_utc_timestamp(asof)
    if moment is None:
        raise PitPriceError("asof must be a parseable timestamp")
    known = []
    for action in actions:
        stamp = to_utc_timestamp(action.effective_date)
        if stamp is not None and stamp <= moment:
            known.append(action)
    return sorted(known, key=lambda item: (item.effective_date, item.kind))


def _raw_close_on(raw_by_date, effective_date):
    """Raw close on the exact action date, else the nearest PRIOR session."""
    if effective_date in raw_by_date and pd.notna(raw_by_date[effective_date]):
        return float(raw_by_date[effective_date])
    prior = [day for day in raw_by_date if day <= effective_date and pd.notna(raw_by_date[day])]
    if not prior:
        return None
    return float(raw_by_date[max(prior)])


def action_factor(action, raw_by_date):
    """Back-adjustment factor applied to prices BEFORE the action, or None."""
    if action.kind == SPLIT:
        return float(action.denominator) / float(action.numerator)
    close_on_ex = _raw_close_on(raw_by_date, action.effective_date)
    if close_on_ex in (None, 0.0):
        return None
    return (close_on_ex - float(action.amount)) / close_on_ex


def _raw_index(frame, ticker=None):
    subset = frame
    if ticker is not None and "ticker" in subset.columns:
        subset = subset.loc[subset["ticker"] == ticker]
    index = {}
    for row in subset.itertuples():
        index[str(row.trade_date)] = row.raw_close
    return index


def validate_action_ledger(actions, asof=None):
    """Integrity checks on an action ledger; returns a problem list."""
    problems = []
    seen = set()
    for action in actions:
        if action.kind not in KINDS:
            problems.append("unknown action kind %r" % (action.kind,))
            continue
        key = (action.ticker, action.kind, action.effective_date, action.amount, action.numerator, action.denominator)
        if key in seen:
            problems.append("duplicate corporate action %s" % (key,))
        seen.add(key)
        if action.kind == SPLIT and (not action.numerator or not action.denominator):
            problems.append("split on %s has no ratio" % action.effective_date)
        if action.kind == DIVIDEND and action.amount is None:
            problems.append("dividend on %s has no amount" % action.effective_date)
        if asof is not None and action.effective_date > _date_str(asof):
            problems.append("action on %s is after as-of %s" % (action.effective_date, _date_str(asof)))
    return problems


def build_pit_adjusted_frame(raw_frame, actions, asof=None, raw_col="raw_close"):
    """Rebuild an adjusted close per trade date using only actions known at ``asof``.

    The result is keyed by ``ticker`` + ``trade_date`` and carries, per row:
    ``raw_close``, ``adjusted_close_pit``, ``actions_applied`` (count) and
    ``actions_prior`` (the effective dates of the factors used). Rows after
    ``asof`` carry ``adjusted_close_pit = None`` because their prices were not yet
    observable at the as-of instant.
    """
    for column in ("ticker", "trade_date", raw_col):
        if column not in raw_frame.columns:
            raise PitPriceError("raw price frame is missing column %r" % column)
    if asof is None:
        asof = raw_frame["trade_date"].max()
    moment = to_utc_timestamp(asof)
    if moment is None:
        raise PitPriceError("asof must be a parseable timestamp")
    asof_day = moment.strftime("%Y-%m-%d")

    rows = []
    for ticker in sorted({str(value) for value in raw_frame["ticker"]}):
        subset = raw_frame.loc[raw_frame["ticker"] == ticker]
        raw_by_date = _raw_index(subset)
        known = actions_known_by([a for a in actions if a.ticker in (None, ticker)], asof_day)
        for row in subset.itertuples():
            day = str(row.trade_date)
            if day > asof_day:
                rows.append(
                    {"ticker": ticker, "trade_date": day, raw_col: getattr(row, raw_col), "adjusted_close_pit": None,
                     "actions_applied": 0, "actions_prior": ""}
                )
                continue
            factor = 1.0
            used = []
            for action in known:
                if day < action.effective_date:
                    applied = action_factor(action, raw_by_date)
                    if applied is not None:
                        factor *= applied
                        used.append(action.effective_date)
            raw_value = getattr(row, raw_col)
            adjusted = None if pd.isna(raw_value) else float(raw_value) * factor
            rows.append(
                {"ticker": ticker, "trade_date": day, raw_col: raw_value, "adjusted_close_pit": adjusted,
                 "actions_applied": len(used), "actions_prior": "|".join(used)}
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by=["ticker", "trade_date"], kind="mergesort").reset_index(drop=True)
    frame.attrs["asof"] = asof_day
    return frame


def pit_return(raw_frame, actions, start_date, end_date, raw_col="raw_close"):
    """Point-in-time total return over ``[start, end]`` using actions known by ``end``.

    The return compares the start raw close (scaled by the corporate actions
    occurring after ``start`` and on/before ``end``) with the end raw close.
    Returns ``None`` when either endpoint is unavailable: an overlapping or
    missing label is excluded rather than approximated.
    """
    adjusted = build_pit_adjusted_frame(raw_frame, actions, asof=end_date, raw_col=raw_col)
    start_rows = adjusted.loc[adjusted["trade_date"] == _date_str(start_date)]
    end_rows = adjusted.loc[adjusted["trade_date"] == _date_str(end_date)]
    if start_rows.empty or end_rows.empty:
        return None
    start_value = start_rows.iloc[0]["adjusted_close_pit"]
    end_value = end_rows.iloc[0]["adjusted_close_pit"]
    if start_value in (None, 0.0) or pd.isna(start_value) or end_value in (None,) or pd.isna(end_value):
        return None
    return (float(end_value) / float(start_value)) - 1.0
