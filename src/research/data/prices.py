"""Price semantics, corporate-action handling and trading-calendar alignment.

Explicit distinctions preserved here:

* ``close`` -- raw, as-printed close for the session.
* ``adjusted_close`` -- close adjusted for splits and dividends (total-return
  basis); the provider's adjustment factors change over time, so a stored
  adjusted value is only meaningful together with its ingestion timestamp.
* ``available_at`` -- a session's price information is model-visible only after
  that session closes; availability is the trade date plus a conservative close
  time expressed in UTC.

Alignment is by NEAREST PRIOR observation only. A feature at prediction date
``T`` may use the last session at or before ``T`` and must never pull a later
session backwards. All timestamps are normalised to UTC.
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

from src.research.fingerprints import fingerprint_dataframe

from .availability import DEFAULT_CLOSE_UTC_TIME, price_available_at, to_utc_timestamp

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%s"
USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"
REQUEST_TIMEOUT = 30
REQUIRED_COLUMNS = ("ticker", "trade_date", "close", "adjusted_close", "available_at", "source")


class PriceError(RuntimeError):
    """Raised when price data is missing, malformed or unusable."""


class PricesUnavailableError(PriceError):
    """Raised when a provider cannot supply real prices (never faked)."""


def _http_get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # network, HTTP, JSON
        raise PricesUnavailableError("price provider unreachable for %s: %s" % (url, exc)) from exc


def parse_yahoo_chart(payload, ticker):
    """Convert a Yahoo chart payload into normalised price rows.

    Refuses adjustments that cannot be interpreted: the raw close, the adjusted
    close and the split/dividend events are all recorded so the transformation
    is documented rather than implied.
    """
    if not isinstance(payload, dict):
        raise PriceError("chart payload must be a mapping")
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise PricesUnavailableError("provider error for %s: %s" % (ticker, chart["error"]))
    results = chart.get("result") or []
    if not results:
        raise PricesUnavailableError("no chart result for %s" % ticker)
    result = results[0]
    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]
    adjclose_blocks = indicators.get("adjclose") or [{}]
    adjusted = adjclose_blocks[0].get("adjclose") if adjclose_blocks else None
    events = (result.get("events") or {})
    splits = sorted((events.get("splits") or {}).values(), key=lambda item: item.get("date", 0))
    dividends = sorted((events.get("dividends") or {}).values(), key=lambda item: item.get("date", 0))
    timezone = meta.get("exchangeTimezoneName") or meta.get("timezone") or "UTC"
    closes = quotes.get("close") or []
    rows = []
    for position, stamp in enumerate(timestamps):
        close = closes[position] if position < len(closes) else None
        adjusted_close = adjusted[position] if adjusted is not None and position < len(adjusted) else None
        moment = pd.Timestamp(stamp, unit="s", tz="UTC")
        local = moment.tz_convert(timezone)
        trade_date = local.strftime("%Y-%m-%d")
        available_at = price_available_at(trade_date)
        rows.append(
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "close": close,
                "adjusted_close": adjusted_close,
                "raw_close": close,
                "session_utc": moment.isoformat(),
                "exchange_timezone": timezone,
                "available_at": available_at.isoformat(),
                "has_split_event": bool(splits),
                "has_dividend_event": bool(dividends),
                "split_count": len(splits),
                "dividend_count": len(dividends),
                "source": "YAHOO_CHART",
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["split_events"] = splits
    frame.attrs["dividend_events"] = dividends
    frame.attrs["exchange_timezone"] = timezone
    frame.attrs["ticker"] = ticker
    return frame


def fetch_prices(ticker, range_="5y", interval="1d"):
    """Fetch real prices for ``ticker``; raises when the provider is unusable."""
    url = "%s?range=%s&interval=%s&events=split,div" % (YAHOO_CHART_URL % ticker, range_, interval)
    payload = _http_get_json(url)
    return parse_yahoo_chart(payload, ticker)


def build_corporate_action_ledger(frame):
    """Document the splits/dividends behind the supplied price frame.

    The ledger is evidence, not advice: it records what adjustments exist so the
    adjusted-vs-raw relationship stays auditable for the prediction timeline.
    """
    return {
        "ticker": frame.attrs.get("ticker"),
        "exchange_timezone": frame.attrs.get("exchange_timezone"),
        "split_events": frame.attrs.get("split_events", []),
        "dividend_events": frame.attrs.get("dividend_events", []),
        "rows": int(len(frame)),
        "fingerprint": fingerprint_dataframe(frame),
    }


def validate_price_frame(frame):
    """Integrity checks on a normalised price frame; returns problem list."""
    problems = []
    for name in REQUIRED_COLUMNS:
        if name not in frame.columns:
            problems.append("missing required column %r" % name)
    if problems:
        return problems
    seen = {}
    for position, row in enumerate(frame.itertuples()):
        key = (row.ticker, row.trade_date)
        seen[key] = seen.get(key, 0) + 1
        if to_utc_timestamp(row.available_at) is None:
            problems.append("row %d has no parseable available_at" % position)
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    for key in duplicates:
        problems.append("duplicate (ticker, trade_date) key %s" % (key,))
    return problems


def _available_rows(frame, asof_ts):
    moment = to_utc_timestamp(asof_ts)
    if moment is None:
        raise PriceError("asof_ts must be a parseable timestamp")
    stamps = [to_utc_timestamp(value) for value in frame["available_at"]]
    return frame.loc[[stamp is not None and stamp <= moment for stamp in stamps]].copy()


def align_nearest_prior(prices, dates, ticker=None, value_col="adjusted_close", asof_col=None):
    """Align a price frame to ``dates`` using the nearest PRIOR session.

    For each requested date the most recent row whose ``trade_date`` is at or
    before that date is used. A later session is never pulled backwards. When
    ``asof_col`` is given, availability is additionally enforced per request.
    Missing alignment yields ``None`` rather than a filled or interpolated value.
    """
    frame = prices.copy()
    if ticker is not None:
        frame = frame.loc[frame["ticker"] == ticker]
    if frame.empty:
        return pd.DataFrame(columns=["request_date", "trade_date", value_col, "ticker"])
    frame["__trade"] = [to_utc_timestamp(value) for value in frame["trade_date"]]
    frame = frame.loc[frame["__trade"].notna()]
    frame = frame.sort_values(by=["__trade"], kind="mergesort")
    requests = [pd.Timestamp(value) for value in dates]
    rows = []
    for request in requests:
        request_ts = to_utc_timestamp(request)
        candidates = frame.loc[frame["__trade"] <= request_ts]
        if asof_col is not None and asof_col in candidates.columns:
            eligible = candidates.loc[
                [
                    to_utc_timestamp(value) is not None and to_utc_timestamp(value) <= request_ts
                    for value in candidates[asof_col]
                ]
            ]
        else:
            eligible = candidates
        if eligible.empty:
            rows.append(
                {
                    "request_date": request.strftime("%Y-%m-%d"),
                    "trade_date": None,
                    value_col: None,
                    "ticker": ticker,
                }
            )
            continue
        last_index = eligible["__trade"].idxmax()
        row = eligible.loc[last_index]
        rows.append(
            {
                "request_date": request.strftime("%Y-%m-%d"),
                "trade_date": row["trade_date"],
                value_col: row[value_col],
                "ticker": row["ticker"],
            }
        )
    return pd.DataFrame(rows)
