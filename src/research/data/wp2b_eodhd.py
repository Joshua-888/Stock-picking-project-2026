"""WP2B real EODHD survivorship-controlled dataset builder (live + offline logic).

This module turns EODHD market data plus the free Wikipedia S&P 500 change table
into a point-in-time, survivorship-AWARE research panel. It reuses the WP2/WP2A
layers (``layers``, ``membership_engine``, ``pit_prices``, ``delisting``) rather
than re-implementing them.

What it does NOT do:

* it never manufactures adjusted closes: adjusted prices are rebuilt from the raw
  close plus only the corporate actions known at the as-of instant;
* it never invents a delisting return (EODHD supplies none) and it documents the
  resulting terminal-return bias instead of papering over it;
* it never fabricates a symbol identity: an ambiguous ticker (reuse) with no name
  or date evidence is reported UNRESOLVED, not guessed.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import bisect
import difflib
import re
from dataclasses import dataclass, replace

import pandas as pd

from .availability import DEFAULT_CLOSE_UTC_TIME, price_available_at, to_utc_timestamp
from .delisting import MISSING_DELISTING_RETURN_BIAS
from .pit_prices import DIVIDEND, SPLIT, CorporateAction, action_factor

RESEARCH_WINDOW_START = "2007-01-01"
RESEARCH_WINDOW_END = "2026-09-25"
UNIVERSE_ID = "sp500_pit_wikipedia_eodhd_v1"
DATASET_NAME = "sp500_pit_research_panel_v1"

MEMBERSHIP_AVAILABILITY_RULE = (
    "Wikipedia membership change table publishes only effective dates, so universe "
    "availability uses effective_date itself (a later, conservative instant); an "
    "announced-but-not-yet-effective change is never granted membership early"
)

TERMINAL_RETURN_LIMITATION = (
    "EODHD publishes NO delisting returns of any kind; a security whose prices stop "
    "before the window end therefore has an UNKNOWN terminal return and its forward "
    "label is unobservable. " + MISSING_DELISTING_RETURN_BIAS
)

# price-completeness categories (deliverable 7)
PRICE_COMPLETE = "COMPLETE"
PRICE_PARTIAL = "PARTIAL_HISTORY"
PRICE_MISSING = "MISSING"
PRICE_IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
PRICE_TERMINAL_UNCERTAIN = "TERMINAL_RETURN_UNCERTAIN"

# terminal-behavior categories (deliverable 6)
TERMINAL_ONGOING = "ongoing_member"
TERMINAL_REMOVED_TRADING = "removed_but_still_trading"
TERMINAL_STOPPED = "stopped_before_window_end"
TERMINAL_NO_PRICE = "no_price_observations"

# terminal-RETURN observability (F5): EODHD publishes no delisting return, so a
# security whose price series stops well before its removal has an UNKNOWN
# terminal return and any forward label crossing that point is unobservable.
TERMINAL_RETURN_CONTINUES = "return_observable_series_continues"
TERMINAL_RETURN_ENDS_AFTER_REMOVAL = "return_observable_series_continues_past_removal"
TERMINAL_RETURN_ENDS_AT_REMOVAL = "return_unknown_series_ends_at_removal"
TERMINAL_RETURN_ENDS_BEFORE = "return_unknown_series_ends_before_removal"
TERMINAL_RETURN_MISSING = "return_unknown_no_price_series"
OBSERVABLE_TERMINAL_RETURN_STATES = (
    TERMINAL_RETURN_CONTINUES, TERMINAL_RETURN_ENDS_AFTER_REMOVAL)
UNKNOWN_TERMINAL_RETURN_STATES = (
    TERMINAL_RETURN_ENDS_AT_REMOVAL, TERMINAL_RETURN_ENDS_BEFORE, TERMINAL_RETURN_MISSING,
)

# Target observability / censoring (WP2C Phase B). A 12-month forward label needs a
# FUTURE price 12 months after T. When a security disappears before that horizon and
# no terminal value exists, the label cannot be measured: the observation is kept
# but CENSORED and flagged so it never enters supervised training as if observed.
CENSOR_REASON_NEAR_WINDOW_END = "horizon_exceeds_research_window_end"
CENSOR_REASON_NO_TERMINAL = "terminal_price_unobservable_series_ends"
CENSOR_REASON_NO_PRICE = "no_price_for_security"
TARGET_HORIZON_DAYS = 365

# Canonical gold panel schema (single source of truth for build + finalize).
PANEL_COLUMNS = (
    "security_id", "ticker", "symbol", "snapshot_date", "membership_start", "membership_end",
    "research_eligible", "available_at", "price_date", "raw_close", "adjusted_close_pit", "has_price",
    "target_observable", "target_censored", "target_censor_reason", "terminal_price_observable",
    "terminal_status", "membership_exit_observed",
)


def horizon_end(snapshot_date, horizon_days=TARGET_HORIZON_DAYS):
    """Calendar date ``horizon_days`` after ``snapshot_date`` (ISO), or None."""
    stamp = to_utc_timestamp(snapshot_date)
    if stamp is None:
        return None
    return (stamp + pd.Timedelta(days=int(horizon_days))).strftime("%Y-%m-%d")


def classify_target_observability(window, raw_frame, snapshot_date, window_end,
                                 horizon_days=TARGET_HORIZON_DAYS, tolerance_days=10):
    """Whether a 12-month forward outcome from ``snapshot_date`` is measurable.

    Returns ``(observable, censored, reason)``. NEVER manufactures a terminal
    return: a security whose series stops before the required horizon (and before
    the research window end) is CENSORED with an explicit reason, so it can be
    excluded from supervised labels rather than silently mislabelled.
    """
    if raw_frame is None or raw_frame.empty:
        return False, True, CENSOR_REASON_NO_PRICE
    needed = horizon_end(snapshot_date, horizon_days)
    if needed is None:
        return False, True, CENSOR_REASON_NO_TERMINAL
    if needed > window_end:
        # The horizon itself extends past the declared research window: the
        # outcome is not yet observable for ANY security (uniform, expected).
        return False, True, CENSOR_REASON_NEAR_WINDOW_END
    last = max(str(day) for day in raw_frame["trade_date"].tolist())
    if last >= _shift_day(needed, -tolerance_days):
        return True, False, None
    return False, True, CENSOR_REASON_NO_TERMINAL


def classify_terminal_status(window, raw_frame, window_end, tolerance_days=10):
    """Coarse terminal kind where determinable from PRICES only (no delisting meta).

    EODHD publishes no delisting reason, so this is deliberately structural:
    ``still_trading`` (series reaches the window end), ``series_ends_at_removal``,
    ``series_ends_before_removal`` or ``no_price``. Nothing is inferred about WHY
    a name left the index beyond what the price series shows.
    """
    if raw_frame is None or raw_frame.empty:
        return "no_price"
    last = max(str(day) for day in raw_frame["trade_date"].tolist())
    if last >= _shift_day(window_end, -tolerance_days):
        return "still_trading"
    if window.membership_end is None:
        return "series_ends_before_window_end"
    if last >= _shift_day(window.membership_end, tolerance_days):
        return "series_trades_past_removal"
    if last >= _shift_day(window.membership_end, -tolerance_days):
        return "series_ends_at_removal"
    return "series_ends_before_removal"

_NAME_STOPWORDS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos",
    "companies", "ltd", "limited", "plc", "llc", "holdings", "holding",
    "group", "the", "class", "common", "stock", "new", "old", "sa", "nv",
    "ag", "international", "intl",
})


def normalize_name(value):
    """Lower-case, strip punctuation and corporate suffixes for name comparison."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token and token not in _NAME_STOPWORDS]
    return " ".join(tokens)


def name_similarity(first, second):
    """Ratio similarity of two normalized names (0.0 when either is empty)."""
    if not first or not second:
        return 0.0
    return difflib.SequenceMatcher(None, first, second).ratio()


def build_symbol_index(symbol_rows):
    """Map EODHD symbol ``Code`` to its metadata for identity resolution."""
    index = {}
    for row in symbol_rows or []:
        code = str(row.get("Code") or "").strip().upper()
        if not code:
            continue
        index[code] = {
            "code": code,
            "name": row.get("Name"),
            "exchange": row.get("Exchange"),
            "type": row.get("Type"),
            "isin": row.get("Isin"),
        }
    return index


def symbol_code_variants(ticker):
    """Equivalent vendor spellings of one ticker.

    Wikipedia writes share classes with a dot (``BF.B``, ``BRK.B``) while EODHD
    writes them with a dash (``BF-B``, ``BRK-B``). Both spellings are candidate
    bases so a class share is never silently dropped (WP2C-A1).
    """
    text = str(ticker or "").strip().upper()
    if not text:
        return []
    variants = {text, text.replace(".", "-"), text.replace("-", ".")}
    return sorted(variant for variant in variants if variant)


def is_old_symbol(code, ticker):
    """True when ``code`` is an ``_OLD``/``_OLDn`` variant of ``ticker``."""
    for base in symbol_code_variants(ticker):
        if re.match(r"^%s_OLD\d*$" % re.escape(base), str(code), re.IGNORECASE):
            return True
    return False


def symbol_candidates(ticker, index):
    """Every symbol that could denote ``ticker``: the bare code plus ``_old`` variants."""
    codes = set()
    for base in symbol_code_variants(ticker):
        pattern = re.compile(r"^%s(?:_OLD\d*)?$" % re.escape(base), re.IGNORECASE)
        codes.update(code for code in index if pattern.match(code))
    return sorted(codes)


def resolve_eodhd_symbol(ticker, name_hint, index, probe=None):
    """Resolve a Wikipedia ticker to the correct EODHD symbol, never guessing.

    Ticker reuse is the central hazard: ``WB`` is Weibo today while ``WB_old2`` is
    Wachovia. Resolution uses company-name evidence first; when two candidates
    share the same name (e.g. ``GM`` vs ``GM_old``) an optional ``probe(code)``
    checks which symbol actually has prices, and an unresolved case returns None.
    """
    candidates = symbol_candidates(ticker, index)
    if not candidates:
        return None
    hint = normalize_name(name_hint)
    scored = []
    for code in candidates:
        normalized = normalize_name(index[code].get("name"))
        similarity = name_similarity(hint, normalized) if hint else 0.0
        scored.append({"code": code, "normalized_name": normalized, "similarity": round(similarity, 4)})
    if len(candidates) == 1:
        item = scored[0]
        return {"code": item["code"], "method": "only_candidate",
                "confidence": item["similarity"], "candidates": [item["code"]]}
    best = max(item["similarity"] for item in scored)
    top = [item for item in scored if abs(item["similarity"] - best) < 1e-9]
    if len(top) == 1 and best >= 0.55:
        return {"code": top[0]["code"], "method": "name", "confidence": best,
                "candidates": [item["code"] for item in scored]}
    if probe is not None:
        live = [item for item in top if probe(item["code"])]
        if len(live) == 1:
            return {"code": live[0]["code"], "method": "date_probe", "confidence": best,
                    "candidates": [item["code"] for item in scored]}
    return None


def _best_by_name(codes, name_hint, index):
    """Return the single best name-matching code, or None on a tie/no evidence."""
    hint = normalize_name(name_hint)
    if not hint:
        return None
    scored = [(code, name_similarity(hint, normalize_name(index[code].get("name")))) for code in codes]
    best = max(score for _, score in scored)
    if best < 0.55:
        return None
    top = [code for code, score in scored if abs(score - best) < 1e-9]
    return top[0] if len(top) == 1 else None


def choose_window_symbol(ticker, window, index, name_hint=None, active_codes=None, probe=None):
    """Resolve the EODHD provider symbol FOR ONE MEMBERSHIP WINDOW (WP2C-A2).

    Ticker reuse means the bare ticker and its ``_OLD`` variants are DIFFERENT
    companies (``DELL`` vs ``DELL_OLD``; ``WB`` vs ``WB_OLD2``). Resolution is
    therefore per-window, never per-ticker, and never guessed:

    * an ONGOING (current/active) constituent prefers its EXACT active ticker;
    * a HISTORICAL window uses price-coverage evidence: a candidate must have a
      first price at or before the window start (a series that only begins after
      the window is the REUSED company and is rejected), then name evidence;
    * an ambiguous case with no discriminating evidence returns ``None``.

    ``probe(code, start, end)`` returns the first available trade date for
    ``code`` across the window span (or ``None``); it is injected so the logic is
    live-independent in tests.
    """
    candidates = symbol_candidates(ticker, index)
    if not candidates:
        return None
    active = {str(code).upper() for code in (active_codes or ())}
    exact = [code for code in candidates if not is_old_symbol(code, ticker)]
    active_exact = [code for code in exact if code.upper() in active]
    ongoing = window is None or window.membership_end is None
    if ongoing:
        if active_exact:
            return {"code": active_exact[0], "method": "active_current", "confidence": 1.0,
                    "candidates": list(candidates)}
        if len(candidates) == 1:
            return {"code": candidates[0], "method": "only_candidate", "confidence": 1.0,
                    "candidates": list(candidates)}
    start = window.membership_start if window is not None else RESEARCH_WINDOW_START
    end = (window.membership_end if window is not None and window.membership_end else RESEARCH_WINDOW_END)
    if probe is not None:
        cover = {}
        for code in candidates:
            try:
                cover[code] = probe(code, start, end)
            except Exception:
                cover[code] = None
        window_start_plus = _shift_day(start, 10)
        viable = [code for code in candidates
                  if cover.get(code) is not None and str(cover[code])[:10] <= window_start_plus]
        if len(viable) == 1:
            return {"code": viable[0], "method": "price_coverage", "confidence": 1.0,
                    "candidates": list(candidates)}
        if len(viable) > 1:
            chosen = _best_by_name(viable, name_hint, index)
            if chosen is not None:
                return {"code": chosen, "method": "price_coverage_name", "confidence": 1.0,
                        "candidates": list(candidates)}
            return None
        # no candidate covers the window start: fall through to name evidence
    if len(candidates) == 1:
        return {"code": candidates[0], "method": "only_candidate", "confidence": 1.0,
                "candidates": list(candidates)}
    chosen = _best_by_name(candidates, name_hint, index)
    if chosen is not None:
        hint = normalize_name(name_hint)
        confidence = name_similarity(hint, normalize_name(index[chosen].get("name")))
        return {"code": chosen, "method": "name", "confidence": round(confidence, 4),
                "candidates": list(candidates)}
    return None


@dataclass
class MembershipWindow:
    """One security's reconstructed S&P 500 membership window (half-open)."""

    security_id: str
    ticker: str
    membership_start: str
    membership_end: str = None
    start_known: bool = True
    note: str = ""
    # research_eligible is False when the membership start is an ASSUMPTION
    # (pre-source-coverage) rather than a verifiable event. Such windows must be
    # excluded from research cross-sections so a present-day constituent is never
    # silently back-projected into an earlier period (WP2B-F3).
    research_eligible: bool = True
    unverifiable_before: bool = False
    # symbol is the RESOLVED EODHD provider code for this window (WP2C-A2). The
    # security_id stays the Wikipedia ticker (the universe identity); a reused
    # ticker can need a DIFFERENT provider symbol per window (DELL -> DELL /
    # DELL_OLD; WB -> WB / WB_OLD2), so resolution is per-window, never guessed.
    symbol: str = None
    # resolution_method / unresolved_reason make the resolver's decision explicit
    # (A1): every membership window ends RESOLVED or EXPLICITLY_UNRESOLVED_WITH_REASON.
    resolution_method: str = None
    unresolved_reason: str = None
    # structural terminal kind derived from PRICES only (no delisting metadata).
    terminal_status: str = None

    def overlaps(self, start, end):
        window_start = to_utc_timestamp(self.membership_start)
        window_end = to_utc_timestamp(self.membership_end) if self.membership_end else None
        lo = to_utc_timestamp(start)
        hi = to_utc_timestamp(end)
        if window_end is not None and window_end <= lo:
            return False
        if window_start is not None and window_start > hi:
            return False
        return True

    def clipped(self, start, end):
        window_start = self.membership_start if to_utc_timestamp(self.membership_start) >= to_utc_timestamp(start) else start
        window_end = self.membership_end
        if window_end is not None and to_utc_timestamp(window_end) > to_utc_timestamp(end):
            window_end = end
        return replace(self, membership_start=window_start, membership_end=window_end)


def _apply_listing_guard(window, listing_day):
    """Clamp a window start to a security's first observed price (listing) date.

    A security cannot be a universe member before it listed, so a window start
    earlier than the first available price is moved forward. A window whose start
    was an ASSUMPTION (``start_known=False``) is ALSO flagged
    ``research_eligible=False``/``unverifiable_before=True``: the source cannot
    verify membership in that span, so such rows must be excluded from research
    cross-sections instead of being silently back-projected into the past.
    """
    listing = to_utc_timestamp(listing_day)
    start = window.membership_start
    if listing is not None:
        listing_str = listing.strftime("%Y-%m-%d")
        if start is None or to_utc_timestamp(start) < listing:
            start = listing_str
    if window.start_known:
        return replace(window, membership_start=start)
    return replace(window, membership_start=start, research_eligible=False, unverifiable_before=True)


def first_price_by_security(price_frame):
    """Map each security id to its earliest observed trade date (its listing proxy)."""
    first = {}
    if price_frame is None or getattr(price_frame, "empty", True):
        return first
    for security, frame in price_frame.groupby("security_id"):
        days = [str(day) for day in frame["trade_date"].tolist()]
        if days:
            first[str(security)] = min(days)
    return first


def apply_listing_guard(windows, first_prices, first_price_by_symbol=None):
    """Apply the listing guard to every window using per-security first price dates.

    When ``first_price_by_symbol`` is supplied AND a window has a resolved
    ``symbol``, that provider symbol's first price is preferred: for a reused
    ticker the per-security proxy aggregates two different companies and could
    mask a wrong-company series (WP2C-A2), so the symbol series wins when known.
    """
    def _listing(window):
        if getattr(window, "symbol", None) and first_price_by_symbol:
            value = (first_price_by_symbol or {}).get(str(window.symbol).upper())
            if value:
                return value
        return (first_prices or {}).get(window.security_id)
    return [_apply_listing_guard(window, _listing(window)) for window in windows]


def reconstruct_memberships(change_frame, current_constituents, window_start, window_end,
                            listing_by_security=None):
    """Rebuild effective-time membership windows from change events plus today's list.

    A ``remove`` with no preceding ``add`` means the security was a member before
    the source's coverage; its start is set to ``window_start`` and flagged
    ``start_known=False`` so the assumption is visible rather than silently trusted.
    Such assumed windows are also marked ``research_eligible=False`` so a
    present-day constituent is never back-projected into an earlier period.
    ``listing_by_security`` (first observed price date) additionally forbids any
    membership before a security's listing.
    """
    by_security = {}
    for row in change_frame.to_dict(orient="records"):
        security = str(row["security_id"]).strip().upper()
        by_security.setdefault(security, []).append((str(row["effective_date"]), str(row["action"])))
    current = {str(item).strip().upper() for item in (current_constituents or [])}
    listing = {}
    for key, value in (listing_by_security or {}).items():
        stamp = to_utc_timestamp(value)
        if stamp is not None:
            listing[str(key).strip().upper()] = stamp.strftime("%Y-%m-%d")
    window_start_ts = to_utc_timestamp(window_start)
    windows = []
    assumed_starts = []
    for security, events in by_security.items():
        # On a shared effective date a REMOVE must be processed before an ADD: the
        # source reuses a ticker across an index event (FOX 2019-03-19: old
        # 21st Century Fox removed, new Fox Corp added). Sorting remove-first lets
        # the remove close the prior window and the add reopen an ongoing one, so a
        # current constituent is never left with no membership window (WP2C-A3).
        events.sort(key=lambda item: (item[0], 0 if item[1] == "remove" else 1))
        opened = None
        opened_known = True
        for effective, action in events:
            effective_ts = to_utc_timestamp(effective)
            if action == "add":
                if opened is None:
                    opened = effective
                    opened_known = True
            elif action == "remove":
                if opened is None:
                    # A removal with no known add is either (a) OUT OF WINDOW
                    # (effective at/before the research start), which must NOT
                    # remove a valid current constituent (WP2C-A3: e.g. T removed
                    # 2005-11-18, MCK 1994-09-30), or (b) an assumed pre-source
                    # membership that was live at the window start (flagged).
                    if effective_ts is not None and window_start_ts is not None and effective_ts <= window_start_ts:
                        continue
                    assumed_starts.append({"security_id": security, "removal": effective})
                    windows.append(MembershipWindow(security, security, window_start, effective, False))
                else:
                    windows.append(MembershipWindow(security, security, opened, effective, opened_known))
                    opened = None
                    opened_known = True
        if opened is not None:
            windows.append(MembershipWindow(security, security, opened, None, opened_known))
    covered = {window.security_id for window in windows}
    for security in sorted(current - covered):
        windows.append(MembershipWindow(
            security, security, window_start, None, start_known=False,
            note="current constituent with no change event in the source (assumed member across the window)",
        ))
    guarded = [_apply_listing_guard(window, listing.get(window.security_id)) for window in windows]
    selected = [window for window in guarded if window.overlaps(window_start, window_end)]
    selected = [window.clipped(window_start, window_end) for window in selected]
    selected.sort(key=lambda window: (window.security_id, window.membership_start))
    return selected, assumed_starts


def membership_diagnostics(windows, window_start, window_end):
    """Overlap, duplicate and per-month membership diagnostics (deliverable 8)."""
    by_security = {}
    for window in windows:
        by_security.setdefault(window.security_id, []).append(window)
    overlaps = []
    for security, items in by_security.items():
        items = sorted(items, key=lambda window: window.membership_start)
        for index in range(1, len(items)):
            prior = items[index - 1]
            if prior.membership_end is None or prior.membership_end > items[index].membership_start:
                overlaps.append({"security_id": security,
                                 "first": prior.membership_start,
                                 "second": items[index].membership_start})
    return {
        "windows": len(windows),
        "securities": len(by_security),
        "assumed_start_securities": sorted(
            {window.security_id for window in windows if not window.start_known}
        ),
        "overlaps": overlaps,
    }


def members_asof(windows, moment):
    """Security ids whose half-open window covers ``moment``."""
    stamp = to_utc_timestamp(moment)
    members = set()
    for window in windows:
        start = to_utc_timestamp(window.membership_start)
        end = to_utc_timestamp(window.membership_end) if window.membership_end else None
        if start <= stamp and (end is None or stamp < end):
            members.add(window.security_id)
    return members


def month_ends(window_start, window_end):
    """Month-end date strings (ISO) covering the inclusive research window.

    The final month is CLAMPED to ``window_end`` so a snapshot can never fall
    after the declared period end (e.g. a 2026-09 month-end must not exceed a
    2026-09-25 window end).
    """
    days = [str(pd.Period(str(period), freq="M").end_time.date())
            for period in pd.period_range(start=window_start[:7], end=window_end[:7], freq="M")]
    return [day if day <= window_end else window_end for day in days]


def monthly_member_counts(windows, window_start, window_end):
    """Member count at each month-end in the window (membership-completeness report)."""
    months = [str(period) for period in pd.period_range(start=window_start[:7], end=window_end[:7], freq="M")]
    rows = []
    for month in months:
        last_day = str(pd.Period(month, freq="M").end_time.date())
        rows.append({"month": month, "members": len(members_asof(windows, last_day))})
    return rows


def actions_from_eodhd_rows(rows, ticker):
    """Convert normalized EODHD action rows into PIT ``CorporateAction`` records."""
    actions = []
    for row in rows or []:
        kind = str(row.get("kind") or "").strip().lower()
        effective = row.get("effective_date")
        if not effective:
            continue
        if kind == SPLIT:
            numerator, denominator = row.get("numerator"), row.get("denominator")
            if numerator and denominator:
                actions.append(CorporateAction(ticker=ticker, kind=SPLIT, effective_date=str(effective)[:10],
                                               numerator=float(numerator), denominator=float(denominator)))
        elif kind == DIVIDEND:
            amount = row.get("amount")
            if amount is None or (isinstance(amount, float) and pd.isna(amount)):
                continue
            actions.append(CorporateAction(ticker=ticker, kind=DIVIDEND, effective_date=str(effective)[:10],
                                           amount=float(amount)))
    actions.sort(key=lambda action: (action.effective_date, action.kind))
    return actions


def _shift_day(day, days):
    stamp = to_utc_timestamp(day) + pd.Timedelta(days=int(days))
    return stamp.strftime("%Y-%m-%d")


def _cumulative_action_factors(actions, raw_by_date):
    """Cumulative back-adjustment factor at each action's effective date.

    A factor that is ``None`` OR non-positive is DROPPED (not fabricated): a
    non-positive dividend factor (amount >= close, e.g. a spinoff or a vendor
    error) would otherwise zero the cumulative factor and corrupt every earlier
    point-in-time price.
    """
    dates, cums, cumulative = [], [], 1.0
    dropped = []
    for action in actions:
        factor = action_factor(action, raw_by_date)
        if factor is not None and factor > 0.0:
            cumulative *= factor
        else:
            dropped.append({"kind": action.kind, "effective_date": action.effective_date})
        dates.append(action.effective_date)
        cums.append(cumulative)
    return dates, cums, dropped


def _cum_at(dates, cums, day):
    position = bisect.bisect_right(dates, day)
    return cums[position - 1] if position > 0 else 1.0


def panel_asof(month_ends, asof=None):
    """The single as-of instant for a panel: an explicit value or the last month."""
    if asof is not None:
        stamp = to_utc_timestamp(asof)
        return stamp.strftime("%Y-%m-%d") if stamp is not None else None
    days = [str(month)[:10] for month in (month_ends or [])]
    return max(days) if days else None


def monthly_panel_for_security(window, raw_frame, actions, month_ends, asof=None, window_end=None):
    """PIT monthly rows for one security over the month-ends it was a member.

    ``adjusted_close_pit`` is a RETURN-CONTINUOUS series: each raw close is
    back-adjusted by every split/dividend factor effective STRICTLY AFTER its
    price date and ON OR BEFORE the panel as-of instant (only actions
    known-through-T enter). The as-of defaults to the latest month-end in
    ``month_ends`` so a split or dividend can never produce a fake monthly jump;
    the unadjusted ``raw_close`` is retained for exact as-of-T reconstruction.
    ``available_at`` is the price-observation instant (the trade date's close),
    not the membership start.
    """
    asof_day = panel_asof(month_ends, asof)
    dates = []
    closes = []
    raw_by_date = {}
    if raw_frame is not None and not raw_frame.empty:
        frame = raw_frame.sort_values("trade_date")
        dates = [str(day) for day in frame["trade_date"].tolist()]
        closes = [float(value) for value in frame["raw_close"].tolist()]
        raw_by_date = dict(zip(dates, closes))
    action_dates, action_cums, _dropped = _cumulative_action_factors(actions, raw_by_date)
    rows = []
    for month in month_ends:
        start = to_utc_timestamp(window.membership_start)
        end = to_utc_timestamp(window.membership_end) if window.membership_end else None
        stamp = to_utc_timestamp(month)
        if start > stamp or (end is not None and stamp >= end):
            continue
        position = bisect.bisect_right(dates, str(month)[:10]) - 1
        if position < 0:
            observable, censored, reason = classify_target_observability(
                window, raw_frame, month, window_end or RESEARCH_WINDOW_END)
            rows.append({"security_id": window.security_id, "ticker": window.ticker,
                         "symbol": window.symbol, "snapshot_date": month, "membership_start": window.membership_start,
                         "membership_end": window.membership_end, "research_eligible": window.research_eligible,
                         "available_at": None, "price_date": None, "raw_close": None,
                         "adjusted_close_pit": None, "has_price": False,
                         "target_observable": bool(observable), "target_censored": bool(censored),
                         "target_censor_reason": reason, "terminal_price_observable": False,
                         "terminal_status": window.terminal_status if hasattr(window, "terminal_status") else None,
                         "membership_exit_observed": bool(window.membership_end is not None)})
            continue
        price_date = dates[position]
        raw_close = closes[position]
        factor = 1.0
        if asof_day is not None:
            numerator = _cum_at(action_dates, action_cums, asof_day)
            denominator = _cum_at(action_dates, action_cums, price_date)
            factor = numerator / denominator if denominator else 1.0
        available = price_available_at(price_date)
        available_at = None if available is None else available.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        observable, censored, reason = classify_target_observability(
            window, raw_frame, month, window_end or RESEARCH_WINDOW_END)
        rows.append({"security_id": window.security_id, "ticker": window.ticker,
                     "symbol": window.symbol, "snapshot_date": month, "membership_start": window.membership_start,
                     "membership_end": window.membership_end, "research_eligible": window.research_eligible,
                     "available_at": available_at, "price_date": price_date, "raw_close": raw_close,
                     "adjusted_close_pit": raw_close * factor, "has_price": True,
                     "target_observable": bool(observable), "target_censored": bool(censored),
                     "target_censor_reason": reason, "terminal_price_observable": bool(observable),
                     "terminal_status": window.terminal_status if hasattr(window, "terminal_status") else None,
                     "membership_exit_observed": bool(window.membership_end is not None)})
    return rows


def classify_terminal_return(window, raw_frame, window_end, tolerance_days=10):
    """Whether a security's TERMINAL return is observable from its price series.

    EODHD publishes no delisting return, so a series that stops at/before removal
    leaves the terminal return UNKNOWN (any forward label crossing that point is
    unobservable). A series that keeps trading to ``window_end`` is observable.
    The status is the honest observability flag, never an invented return.
    """
    if raw_frame is None or raw_frame.empty:
        return TERMINAL_RETURN_MISSING
    last = max(str(day) for day in raw_frame["trade_date"].tolist())
    if last >= _shift_day(window_end, -tolerance_days):
        return TERMINAL_RETURN_CONTINUES
    if window.membership_end is None:
        return TERMINAL_RETURN_ENDS_BEFORE
    if last >= _shift_day(window.membership_end, tolerance_days):
        # Still trading well AFTER removal (pink-sheet/OTC continuation): the
        # terminal move is observable, unlike a series cut off AT removal.
        return TERMINAL_RETURN_ENDS_AFTER_REMOVAL
    if last >= _shift_day(window.membership_end, -tolerance_days):
        return TERMINAL_RETURN_ENDS_AT_REMOVAL
    return TERMINAL_RETURN_ENDS_BEFORE


def series_ends_after_membership(window, raw_frame, tolerance_days=10):
    """True when a removed security's price series lies ENTIRELY after its removal.

    That pattern means the resolved symbol is the WRONG (reused) company, e.g. the
    bare ticker of a name that was later re-listed; the window is then quarantined
    rather than silently paired with another company's prices.
    """
    if window.membership_end is None or raw_frame is None or raw_frame.empty:
        return False
    first = min(str(day) for day in raw_frame["trade_date"].tolist())
    return first > _shift_day(window.membership_end, tolerance_days)


def classify_price_completeness(window, raw_frame, window_start, window_end, tolerance_days=10):
    """COMPLETE / PARTIAL_HISTORY / MISSING for one membership window."""
    if raw_frame is None or raw_frame.empty:
        return PRICE_MISSING
    dates = sorted(str(day) for day in raw_frame["trade_date"].tolist())
    first, last = dates[0], dates[-1]
    start = window.membership_start if to_utc_timestamp(window.membership_start) >= to_utc_timestamp(window_start) else window_start
    if window.membership_end is None:
        end = window_end
    elif to_utc_timestamp(window.membership_end) > to_utc_timestamp(window_end):
        end = window_end
    else:
        end = window.membership_end
    start_ok = first <= _shift_day(start, tolerance_days)
    end_ok = last >= _shift_day(end, -tolerance_days)
    return PRICE_COMPLETE if (start_ok and end_ok) else PRICE_PARTIAL


def classify_terminal(window, raw_frame, window_end, tolerance_days=10):
    """Structural terminal behavior; EODHD supplies no delisting reason or return."""
    if raw_frame is None or raw_frame.empty:
        return TERMINAL_NO_PRICE
    last = max(str(day) for day in raw_frame["trade_date"].tolist())
    if window.membership_end is None:
        return TERMINAL_ONGOING if last >= _shift_day(window_end, -tolerance_days) else TERMINAL_STOPPED
    return TERMINAL_REMOVED_TRADING if last >= _shift_day(window.membership_end, -tolerance_days) else TERMINAL_STOPPED
