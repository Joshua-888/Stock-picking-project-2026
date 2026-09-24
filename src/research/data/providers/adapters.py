"""Concrete provider adapters for historical universe / delisted price coverage.

Design rules enforced by this module:

* every credentialed adapter calls :func:`require_env` as its FIRST action, so a
  missing key raises :class:`ProviderCredentialError` before any socket is
  opened and can never degrade into demo or synthetic data;
* vendor payload parsing is confined to the adapter and normalized into the
  contract frames declared in ``providers/__init__``;
* adapters never invent a value: a field the vendor does not supply stays
  ``None`` and the caller must treat it as missing.

Implemented adapters
--------------------

``eodhd``
    Documented endpoints used: ``/api/eod/{ticker}.{exchange}`` for end-of-day
    prices (delisted symbols keep their history), ``/api/exchange-symbol-list/{exchange}"
    with ``delisted=1`` for delisted symbol enumeration, ``/api/splits/{ticker}``
    and ``/api/div/{ticker}`` for corporate actions. EODHD does NOT publish a
    historical index-membership endpoint, so ``fetch_membership_events`` raises
    :class:`ProviderUnsupportedError` instead of approximating.

``sharadar_ndl``
    Nasdaq Data Link / Sharadar tables: ``SEP`` (survivorship-bias-free prices
    including delisted tickers), ``SF1`` (point-in-time fundamentals),
    ``SP500`` (daily historical S&P 500 constituent membership, 1998+).

``wikipedia_sp500``
    Free, credential-free MEMBERSHIP-ONLY source parsed from the Wikipedia
    "Historical components of the S&P 500" change table. It provides added and
    removed tickers with a date, but no prices, no permanent identifiers and a
    partial early history. It is informational and is never certified alone.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import (
    CORPORATE_ACTION_FIELDS,
    DELISTED_PRICE_FIELDS,
    IDENTITY_FIELDS,
    MEMBERSHIP_EVENT_FIELDS,
    VALID_ACTIONS,
    ProviderCredentialError,
    ProviderPayloadError,
    ProviderUnsupportedError,
    register_provider,
    require_env,
)

DEFAULT_TIMEOUT = 30
USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"

# --- corporate-action kinds recognized at the ingestion boundary --------------
SPLIT = "split"
DIVIDEND = "dividend"
SPINOFF = "spinoff"


def http_get(url, timeout=DEFAULT_TIMEOUT):
    """Minimal dependency-free GET returning decoded text.

    Exposed as a module attribute so tests can monkeypatch transport without any
    network access and without importing requests.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _contract_frame(columns, rows, label):
    frame = pd.DataFrame(rows, columns=list(columns))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ProviderPayloadError("%s payload missing columns %s" % (label, missing))
    return frame


# --------------------------------------------------------------------------- #
# Normalizers (vendor payload -> V2 Silver contract)
# --------------------------------------------------------------------------- #


def normalize_delisted_prices(rows, security_id, ticker):
    """Normalize vendor price rows into :data:`DELISTED_PRICE_FIELDS`.

    ``rows`` is an iterable of mappings with ``date``/``close`` (or the already
    normalized ``trade_date``/``raw_close``). Rows without a usable close are
    dropped, never zero-filled.
    """
    normalized = []
    for row in rows or []:
        trade_date = row.get("trade_date", row.get("date"))
        raw_close = row.get("raw_close", row.get("close", row.get("adjusted_close")))
        if trade_date in (None, "") or raw_close in (None, ""):
            continue
        normalized.append(
            {
                "security_id": security_id,
                "ticker": str(row.get("ticker", ticker)).strip().upper(),
                "trade_date": str(trade_date)[:10],
                "raw_close": float(raw_close),
            }
        )
    frame = _contract_frame(DELISTED_PRICE_FIELDS, normalized, "delisted prices")
    for column in ("security_id", "ticker", "trade_date", "raw_close"):
        if frame[column].isna().any():
            raise ProviderPayloadError("delisted price payload has missing %r" % column)
    return frame


def normalize_corporate_actions(rows, ticker=None):
    """Normalize vendor actions into :data:`CORPORATE_ACTION_FIELDS`.

    Only split/dividend/spinoff rows with a parsable effective date survive;
    anything else is dropped (never guessed).
    """
    normalized = []
    for row in rows or []:
        kind = str(row.get("kind", "")).strip().lower()
        if kind not in (SPLIT, DIVIDEND, SPINOFF):
            continue
        effective = row.get("effective_date", row.get("date"))
        if effective in (None, ""):
            continue
        normalized.append(
            {
                "ticker": str(row.get("ticker", ticker) or "").strip().upper() or None,
                "kind": kind,
                "effective_date": str(effective)[:10],
                "numerator": row.get("numerator"),
                "denominator": row.get("denominator"),
                "amount": row.get("amount"),
            }
        )
    return _contract_frame(CORPORATE_ACTION_FIELDS, normalized, "corporate actions")


def normalize_identities(rows, vendor):
    """Normalize vendor identity rows into :data:`IDENTITY_FIELDS`.

    ``security_id`` is left ``None`` on purpose: deriving the permanent id is the
    security-master's job (:func:`src.research.data.security_master.security_id_for`),
    not the vendor adapter's.
    """
    normalized = []
    for row in rows or []:
        normalized.append(
            {
                "vendor": vendor,
                "vendor_security_id": row.get("vendor_security_id"),
                "security_id": row.get("security_id"),
                "cik": row.get("cik"),
                "issuer_key": row.get("issuer_key"),
                "share_class": row.get("share_class"),
                "effective_from": row.get("effective_from", row.get("start_date")),
                "effective_to": row.get("effective_to", row.get("end_date")),
                "source": row.get("source", vendor),
            }
        )
    return _contract_frame(IDENTITY_FIELDS, normalized, "identities")


def _membership_frame(rows, universe_id, source):
    normalized = []
    for row in rows or []:
        action = str(row.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            continue
        effective = row.get("effective_date")
        if effective in (None, ""):
            continue
        announcement = row.get("announcement_date")
        normalized.append(
            {
                "security_id": row.get("security_id"),
                "universe_id": row.get("universe_id", universe_id),
                "action": action,
                "announcement_date": str(announcement)[:10] if announcement not in (None, "") else None,
                "effective_date": str(effective)[:10],
                "source": row.get("source", source),
                "source_reference": row.get("source_reference"),
            }
        )
    return _contract_frame(MEMBERSHIP_EVENT_FIELDS, normalized, "membership events")


def events_from_daily_members(frame, universe_id, source, source_reference=None,
                              date_col="date", ticker_col="ticker", security_col=None):
    """Derive add/remove membership events from a DAILY constituent table.

    Vendors such as Sharadar's ``SP500`` table publish membership by day. Turning
    that into events is deterministic: the first day a security appears is an
    ``add`` effective that day, the first day it disappears is a ``remove``
    effective that day.

    LIMITATION (recorded, not hidden): a daily table carries no announcement
    date, so ``announcement_date`` is set equal to ``effective_date`` and flagged
    by :data:`INFERRED_ANNOUNCEMENT`. Membership decisions must still use
    ``effective_date`` only, so no look-ahead is introduced.
    """
    if frame is None or frame.empty:
        return _contract_frame(MEMBERSHIP_EVENT_FIELDS, [], "membership events")
    key = security_col or ticker_col
    working = frame.copy()
    working[date_col] = working[date_col].astype(str).str.slice(0, 10)
    working[key] = working[key].astype(str).str.strip().str.upper()
    days = sorted(working[date_col].unique())
    present_by_day = {
        day: set(working.loc[working[date_col] == day, key]) for day in days
    }
    rows = []
    previous = set()
    for day in days:
        current = present_by_day[day]
        for entered in sorted(current - previous):
            rows.append(
                {
                    "security_id": entered,
                    "universe_id": universe_id,
                    "action": "add",
                    "announcement_date": day,
                    "effective_date": day,
                    "source": source,
                    "source_reference": source_reference,
                }
            )
        for exited in sorted(previous - current):
            rows.append(
                {
                    "security_id": exited,
                    "universe_id": universe_id,
                    "action": "remove",
                    "announcement_date": day,
                    "effective_date": day,
                    "source": source,
                    "source_reference": source_reference,
                }
            )
        previous = current
    frame_out = _membership_frame(rows, universe_id, source)
    frame_out.attrs["announcement_inferred"] = True
    frame_out.attrs["limitation"] = INFERRED_ANNOUNCEMENT
    return frame_out


INFERRED_ANNOUNCEMENT = (
    "vendor publishes daily constituents without announcement dates; "
    "announcement_date was set equal to effective_date and membership decisions "
    "use effective_date only"
)


# --------------------------------------------------------------------------- #
# EODHD
# --------------------------------------------------------------------------- #


@register_provider
class EodhdProvider:
    """EODHD adapter (delisted prices, corporate actions, symbol identity).

    Requires ``EODHD_API_TOKEN`` in the environment. Sharadar/QuantRocket and
    other vendors are separate adapters; nothing here silently falls back.
    """

    provider_id = "eodhd"
    credential_env = ("EODHD_API_TOKEN",)
    capabilities = ("delisted_prices", "corporate_actions", "identities")
    notes = "documented EOD delisted=1 coverage; historical index membership NOT published"
    base_url = "https://eodhd.com/api"

    def __init__(self, timeout=DEFAULT_TIMEOUT, get=http_get):
        self.timeout = timeout
        self._get = get

    def _url(self, path, **params):
        _name, token = require_env(self.provider_id, self.credential_env)
        params["api_token"] = token
        params.setdefault("fmt", "json")
        return "%s/%s?%s" % (self.base_url, path.lstrip("/"), urllib.parse.urlencode(params))

    def fetch_delisted_prices(self, security_id, ticker, start, end):
        """Raw EOD history for one symbol, normalized to the Silver contract.

        EODHD keeps delisted symbols' history, so the same endpoint serves live
        and delisted names; the caller passes the permanent ``security_id``.
        """
        url = self._url("eod/%s" % ticker, **{"from": str(start)[:10], "to": str(end)[:10]})
        text = self._get(url, timeout=self.timeout)
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderPayloadError("EODHD eod payload was not JSON: %s" % exc) from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise ProviderPayloadError("EODHD eod error: %s" % payload.get("error"))
        if not isinstance(payload, list):
            raise ProviderPayloadError("EODHD eod payload was not a list of rows")
        return normalize_delisted_prices(payload, security_id, ticker)

    def fetch_exchange_symbols(self, exchange="US", delisted=True):
        """Enumerate symbols, with ``delisted=1`` per the documented API."""
        url = self._url("exchange-symbol-list/%s" % exchange, **{"delisted": 1 if delisted else 0})
        text = self._get(url, timeout=self.timeout)
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderPayloadError("EODHD symbol list was not JSON: %s" % exc) from exc
        if not isinstance(payload, list):
            raise ProviderPayloadError("EODHD symbol list payload was not a list")
        return payload

    def fetch_corporate_actions(self, ticker, start, end):
        splits_url = self._url("splits/%s" % ticker, **{"from": str(start)[:10], "to": str(end)[:10]})
        dividends_url = self._url("div/%s" % ticker, **{"from": str(start)[:10], "to": str(end)[:10]})
        rows = []
        for url, kind in ((splits_url, SPLIT), (dividends_url, DIVIDEND)):
            text = self._get(url, timeout=self.timeout)
            try:
                payload = json.loads(text)
            except ValueError as exc:
                raise ProviderPayloadError("EODHD %s payload was not JSON: %s" % (kind, exc)) from exc
            if isinstance(payload, dict) and payload.get("error"):
                raise ProviderPayloadError("EODHD %s error: %s" % (kind, payload.get("error")))
            for entry in payload if isinstance(payload, list) else []:
                rows.append(
                    {
                        "ticker": ticker,
                        "kind": kind,
                        "effective_date": entry.get("date"),
                        "numerator": entry.get("split") if kind == SPLIT else None,
                        "denominator": 1.0 if kind == SPLIT else None,
                        "amount": entry.get("unadjustedValue", entry.get("value")) if kind == DIVIDEND else None,
                    }
                )
        return normalize_corporate_actions(rows, ticker=ticker)

    def fetch_identities(self, start=None, end=None):
        """Symbol identity rows from the exchange symbol list (no permanent ids).

        EODHD exposes ticker/exchange/type; it does not publish a CIK, so
        ``cik`` stays ``None`` and the identity chain must resolve it elsewhere
        (SEC EDGAR), which is exactly what WP2B's identity chain enforces.
        """
        rows = []
        for entry in self.fetch_exchange_symbols("US", delisted=True):
            rows.append(
                {
                    "vendor_security_id": entry.get("Code"),
                    "issuer_key": entry.get("Name"),
                    "effective_from": start,
                    "effective_to": end,
                    "source": "eodhd:exchange-symbol-list",
                }
            )
        return normalize_identities(rows, self.provider_id)

    def fetch_membership_events(self, universe_id, start, end):
        raise ProviderUnsupportedError(
            "EODHD does not publish historical index membership; "
            "use a membership-capable provider (Sharadar SP500) instead"
        )


# --------------------------------------------------------------------------- #
# Sharadar via Nasdaq Data Link
# --------------------------------------------------------------------------- #


@register_provider
class SharadarNdlProvider:
    """Nasdaq Data Link / Sharadar adapter (SEP prices, SF1 facts, SP500 members).

    Requires ``NASDAQ_DATA_LINK_API_KEY`` (aliases ``QUANDL_API_KEY``,
    ``SHARADAR_API_KEY``) in the environment.
    """

    provider_id = "sharadar_ndl"
    credential_env = ("NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY", "SHARADAR_API_KEY")
    capabilities = ("delisted_prices", "corporate_actions", "identities", "historical_universe")
    notes = "survivorship-bias-free SEP/SF1; SP500 daily constituents 1998+ (no announcement dates)"
    base_url = "https://data.nasdaq.com/api/v3/datatables"

    def __init__(self, timeout=DEFAULT_TIMEOUT, get=http_get):
        self.timeout = timeout
        self._get = get

    def _table(self, table, params):
        _name, key = require_env(self.provider_id, self.credential_env)
        params = dict(params)
        params["api_key"] = key
        url = "%s/%s.json?%s" % (self.base_url, table, urllib.parse.urlencode(params))
        text = self._get(url, timeout=self.timeout)
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderPayloadError("Sharadar %s payload was not JSON: %s" % (table, exc)) from exc
        datatable = payload.get("datatable") if isinstance(payload, dict) else None
        if not isinstance(datatable, dict):
            raise ProviderPayloadError("Sharadar %s payload has no datatable envelope" % table)
        return datatable

    @staticmethod
    def _records(datatable):
        columns = [entry.get("name") for entry in datatable.get("columns", [])]
        return [dict(zip(columns, row)) for row in datatable.get("data", [])]

    def fetch_delisted_prices(self, security_id, ticker, start, end):
        datatable = self._table(
            "SEP",
            {"ticker": str(ticker).upper(), "qopts.columns": "ticker,date,close",
             "date.gte": str(start)[:10], "date.lte": str(end)[:10]},
        )
        rows = self._records(datatable)
        if not rows:
            raise ProviderPayloadError(
                "Sharadar SEP returned no rows for %r; empty coverage is reported, not filled" % ticker
            )
        return normalize_delisted_prices(rows, security_id, ticker)

    def fetch_corporate_actions(self, ticker, start, end):
        datatable = self._table(
            "SEP",
            {"ticker": str(ticker).upper(),
             "qopts.columns": "ticker,date,close,closeadj,closeunadj,lastupdated",
             "date.gte": str(start)[:10], "date.lte": str(end)[:10]},
        )
        rows = []
        for record in self._records(datatable):
            adjusted = record.get("closeadj")
            unadjusted = record.get("closeunadj")
            if adjusted in (None, 0) or unadjusted in (None, 0):
                continue
            ratio = float(adjusted) / float(unadjusted)
            if abs(ratio - 1.0) < 1e-9:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "kind": SPLIT,
                    "effective_date": record.get("date"),
                    "numerator": ratio,
                    "denominator": 1.0,
                }
            )
        frame = normalize_corporate_actions(rows, ticker=ticker)
        frame.attrs["derived"] = "SEP closeadj/closeunadj ratio; vendor does not separate splits from dividends"
        return frame

    def fetch_membership_events(self, universe_id, start, end):
        datatable = self._table(
            "SP500",
            {"qopts.columns": "date,ticker", "date.gte": str(start)[:10], "date.lte": str(end)[:10]},
        )
        rows = self._records(datatable)
        if not rows:
            raise ProviderPayloadError("Sharadar SP500 returned no membership rows")
        events = events_from_daily_members(
            pd.DataFrame(rows),
            universe_id=universe_id,
            source="sharadar_ndl:SP500",
            source_reference="nasdaq-data-link SP500 datatable",
        )
        events.attrs["limitation"] = INFERRED_ANNOUNCEMENT
        return events

    def fetch_identities(self, start=None, end=None):
        datatable = self._table("SF1", {"qopts.columns": "ticker,dimension,date"})
        rows = []
        seen = set()
        for record in self._records(datatable):
            ticker = str(record.get("ticker") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            rows.append(
                {
                    "vendor_security_id": ticker,
                    "issuer_key": ticker,
                    "effective_from": start,
                    "effective_to": end,
                    "source": "sharadar_ndl:SF1",
                }
            )
        return normalize_identities(rows, self.provider_id)


# --------------------------------------------------------------------------- #
# Wikipedia (free, membership only, informational)
# --------------------------------------------------------------------------- #

WIKIPEDIA_SP500_URL = (
    "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
)

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class WikipediaTableError(ProviderPayloadError):
    """Raised when the Wikipedia change table cannot be parsed."""


def _strip_tags(value):
    return re.sub(r"\s+", " ", _TAG_RE.sub("", value or "")).strip()


def parse_html_tables(html):
    """Dependency-free parse of ``<table>`` elements into lists of rows.

    Kept deliberately small: the Wikipedia change table is a flat grid with no
    nested tables, so regex extraction is sufficient and avoids a hard lxml
    dependency in the research runtime.
    """
    tables = []
    for table_html in _TABLE_RE.findall(html or ""):
        rows = []
        for row_html in _ROW_RE.findall(table_html):
            cells = [_strip_tags(cell) for cell in _CELL_RE.findall(row_html)]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def parse_wikipedia_sp500_changes(html, universe_id="sp500"):
    """Extract add/remove change events from the Wikipedia component table.

    Expected columns: date, added ticker, added company, removed ticker,
    removed company, reason. A blank ticker cell means no event on that side.
    """
    events = []
    for rows in parse_html_tables(html):
        for row in rows:
            if len(row) < 4:
                continue
            date = row[0]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                continue
            added = row[1].upper()
            removed = row[3].upper() if len(row) > 3 else ""
            if _TICKER_RE.match(added):
                events.append(
                    {
                        "security_id": added,
                        "universe_id": universe_id,
                        "action": "add",
                        "announcement_date": None,
                        "effective_date": date,
                        "source": "wikipedia:historical_components_sp500",
                        "source_reference": WIKIPEDIA_SP500_URL,
                    }
                )
            if _TICKER_RE.match(removed):
                events.append(
                    {
                        "security_id": removed,
                        "universe_id": universe_id,
                        "action": "remove",
                        "announcement_date": None,
                        "effective_date": date,
                        "source": "wikipedia:historical_components_sp500",
                        "source_reference": WIKIPEDIA_SP500_URL,
                    }
                )
    if not events:
        raise WikipediaTableError("no membership change rows found in Wikipedia page")
    frame = _membership_frame(events, universe_id, "wikipedia:historical_components_sp500")
    frame.attrs["limitation"] = (
        "membership-only source: no prices, no permanent identifiers, no "
        "announcement dates, partial early history"
    )
    return frame


@register_provider
class WikipediaSp500Provider:
    """Free membership-only provider for S&P 500 historical changes.

    No credential is required. It is explicitly NOT survivorship-safe on its own
    because it supplies no prices for the removed names.
    """

    provider_id = "wikipedia_sp500"
    credential_env = ()
    capabilities = ("historical_universe_membership_only",)
    notes = "free; membership events only; no prices; not certified survivorship-safe"

    def __init__(self, timeout=DEFAULT_TIMEOUT, get=http_get, universe_id="sp500"):
        self.timeout = timeout
        self._get = get
        self.universe_id = universe_id

    def fetch_membership_events(self, universe_id=None, start=None, end=None):
        html = self._get(WIKIPEDIA_SP500_URL, timeout=self.timeout)
        frame = parse_wikipedia_sp500_changes(html, universe_id=universe_id or self.universe_id)
        if start is not None:
            frame = frame.loc[frame["effective_date"] >= str(start)[:10]]
        if end is not None:
            frame = frame.loc[frame["effective_date"] <= str(end)[:10]]
        return frame.reset_index(drop=True)

    def fetch_delisted_prices(self, security_id, ticker, start, end):
        raise ProviderUnsupportedError("Wikipedia provides no price data")

    def fetch_corporate_actions(self, ticker, start, end):
        raise ProviderUnsupportedError("Wikipedia provides no corporate actions")

    def fetch_identities(self, start=None, end=None):
        raise ProviderUnsupportedError("Wikipedia provides no permanent identifiers")


__all__ = [
    "SPLIT",
    "DIVIDEND",
    "SPINOFF",
    "INFERRED_ANNOUNCEMENT",
    "WIKIPEDIA_SP500_URL",
    "http_get",
    "normalize_delisted_prices",
    "normalize_corporate_actions",
    "normalize_identities",
    "events_from_daily_members",
    "parse_html_tables",
    "parse_wikipedia_sp500_changes",
    "WikipediaTableError",
    "EodhdProvider",
    "SharadarNdlProvider",
    "WikipediaSp500Provider",
]
