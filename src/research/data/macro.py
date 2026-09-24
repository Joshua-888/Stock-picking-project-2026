"""Macroeconomic vintage contract (FRED / ALFRED).

A macro series has three separate temporal concepts and they are never collapsed:

* ``observation_date`` -- the period the value describes.
* ``vintage_date`` -- the data snapshot in which the value appeared.
* ``release_date`` -- when that snapshot was published.

``available_at`` is the later of release and vintage, so a revised value can
never be used before the revision existed. Using today's revised series for a
historical prediction is a look-ahead error and is refused here: a record with
no vintage/release information has no ``available_at`` and is treated as
unavailable, not as the current value.

The FRED CSV endpoint accepts a ``vintage_date`` query parameter and returns the
series as it stood at that vintage. That is the only fetch path implemented, and
it is used to obtain real vintage snapshots for a series.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import urllib.error
import urllib.request

import pandas as pd

from .availability import macro_available_at, to_utc_timestamp

USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"
FRED_VINTAGE_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s&vintage_date=%s"
FRED_CURRENT_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s"
REQUEST_TIMEOUT = 30
REQUIRED_COLUMNS = ("series_id", "observation_date", "vintage_date", "release_date", "available_at")

# Macro is EXCLUDED from the initial research dataset. The vintage contract and
# leakage guard above are kept intentionally (so a future macro track inherits a
# safe foundation), but no macro observation may gate or block the initial
# dataset, and macro features are not part of the initial feature set.
EXCLUDED_FROM_INITIAL_RESEARCH = True
EXCLUSION_STATUS = "EXCLUDED_FROM_INITIAL_RESEARCH"
EXCLUSION_REASON = (
    "macroeconomic vintages are not required for the initial 12-month "
    "benchmark-relative research dataset; excluded so macro data quality cannot "
    "block certification, while the vintage/leakage guard is retained for a "
    "future, separately reviewed macro track"
)


def exclusion_note():
    """Return the explicit macro-exclusion marker for dataset certification."""
    return {
        "status": EXCLUSION_STATUS,
        "excludedFromInitialResearch": EXCLUDED_FROM_INITIAL_RESEARCH,
        "reason": EXCLUSION_REASON,
    }


class MacroError(RuntimeError):
    """Raised when macro data is malformed or temporally unsafe."""


class MacroVintageUnavailableError(MacroError):
    """Raised when a vintage snapshot cannot be obtained (never substituted)."""


class VintageNotHonoredError(MacroVintageUnavailableError):
    """Raised when a supposed vintage contains observations from the future of it.

    FRED's CSV endpoint currently ignores the ``vintage_date`` parameter and
    returns the latest revised series. Accepting that payload as a historical
    vintage would be look-ahead leakage, so it is refused outright.
    """


def _http_get_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MacroVintageUnavailableError("FRED/ALFRED unreachable for %s: %s" % (url, exc)) from exc


def fetch_vintage_csv(series_id, vintage_date):
    """Fetch one real vintage snapshot of ``series_id`` (raises on failure)."""
    text = _http_get_text(FRED_VINTAGE_CSV_URL % (series_id, vintage_date))
    if "observation_date" not in text:
        raise MacroVintageUnavailableError("unexpected FRED payload for %s @ %s" % (series_id, vintage_date))
    return text


def assert_vintage_honored(frame, vintage_date, date_col=None):
    """Raise when a payload claims a vintage but contains later observations.

    This is the leakage alarm. If a provider ignores the vintage parameter and
    returns the current series, the payload will contain observation dates after
    the requested vintage, and the correct response is to refuse it rather than
    treat revised values as historical reads.
    """
    stamp = to_utc_timestamp(vintage_date)
    if stamp is None:
        raise MacroError("vintage_date %r is unparseable" % (vintage_date,))
    column = date_col or frame.columns[0]
    future = []
    for value in frame[column]:
        observation = to_utc_timestamp(value)
        if observation is not None and observation > stamp:
            future.append(observation.strftime("%Y-%m-%d"))
    if future:
        raise VintageNotHonoredError(
            "payload for vintage %s contains %d observation(s) after the vintage "
            "(latest %s); the provider does not honour vintage_date, so this "
            "payload cannot be used as a historical vintage" % (stamp.strftime("%Y-%m-%d"), len(future), max(future))
        )
    return True


def parse_vintage_csv(text, series_id, vintage_date, release_date=None, enforce_vintage=True):
    """Parse a FRED vintage CSV into observation records with availability.

    ``available_at`` is the later of the vintage and the release date. When the
    release date is unknown, the vintage date is used as the conservative
    availability, and the record is flagged so callers can distinguish an exact
    release timestamp from a vintage-derived one.

    With ``enforce_vintage`` (the default) the payload is rejected when it holds
    observations after the vintage date, because that is evidence the provider
    ignored the vintage request and returned revised data.
    """
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        raise MacroError("FRED vintage CSV for %s has no value column" % series_id)
    date_col = frame.columns[0]
    value_col = frame.columns[1]
    vintage_stamp = to_utc_timestamp(vintage_date)
    if vintage_stamp is None:
        raise MacroError("vintage_date %r is unparseable" % (vintage_date,))
    if enforce_vintage:
        assert_vintage_honored(frame, vintage_date, date_col=date_col)
    release_stamp = to_utc_timestamp(release_date) if release_date is not None else None
    rows = []
    for record in frame.to_dict(orient="records"):
        observation = to_utc_timestamp(record.get(date_col))
        if observation is None:
            continue
        raw_value = record.get(value_col)
        value = None
        if raw_value is not None and str(raw_value).strip() not in ("", "."):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = None
        available_at = macro_available_at(release_date=release_date, vintage_date=vintage_date)
        rows.append(
            {
                "series_id": series_id,
                "observation_date": observation.strftime("%Y-%m-%d"),
                "vintage_date": vintage_stamp.strftime("%Y-%m-%d"),
                "release_date": release_stamp.strftime("%Y-%m-%d") if release_stamp is not None else None,
                "value": value,
                "is_missing": value is None,
                "available_at": available_at.isoformat() if available_at is not None else None,
                "availability_basis": "release_date" if release_stamp is not None else "vintage_date",
                "source": "FRED_ALFRED",
            }
        )
    return pd.DataFrame(rows)


def ingest_vintage(series_id, vintage_date, root, mode, release_date=None, client=None):
    """Bronze a raw vintage snapshot and silver-normalise it.

    A missing vintage is a hard failure (``MacroVintageUnavailableError``): the
    current revised series is never silently substituted for a historical
    vintage.
    """
    from src.research.modes import assert_no_synthetic_in_research

    from . import layers

    synthetic = bool((client or {}).get("synthetic")) if isinstance(client, dict) else False
    assert_no_synthetic_in_research(mode, synthetic, context="macro vintage ingestion")

    if client and "text" in client:
        text = client["text"]
    else:
        text = fetch_vintage_csv(series_id, vintage_date)
    ingestion_ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    bronze = layers.write_bronze_bytes(
        root,
        "fred_vintage_%s" % series_id,
        text,
        ext="csv",
        meta={"series_id": series_id, "vintage_date": vintage_date, "ingestion_ts": ingestion_ts},
    )
    frame = parse_vintage_csv(text, series_id, vintage_date, release_date=release_date)
    if "available_at" in frame.columns and not frame.empty:
        frame["ingestion_ts"] = ingestion_ts
    silver = None
    if not frame.empty:
        silver = layers.write_silver_table(
            root,
            "macro_vintage_%s" % series_id,
            frame,
            meta={"series_id": series_id, "vintage_date": vintage_date, "ingestion_ts": ingestion_ts},
        )
    return {
        "series_id": series_id,
        "vintage_date": vintage_date,
        "bronze": bronze,
        "silver": silver,
        "row_count": int(len(frame)),
        "ingestion_ts": ingestion_ts,
    }


def fetch_current_csv(series_id):
    """Fetch the current (revised) series -- for comparison only, never research."""
    return _http_get_text(FRED_CURRENT_CSV_URL % series_id)


def validate_macro_frame(frame):
    """Integrity checks; returns problem list (missing vintage is a problem)."""
    problems = []
    for name in REQUIRED_COLUMNS:
        if name not in frame.columns:
            problems.append("missing required column %r" % name)
    if problems:
        return problems
    for position, row in enumerate(frame.itertuples()):
        if to_utc_timestamp(row.vintage_date) is None:
            problems.append("row %d has no parseable vintage_date" % position)
        available = to_utc_timestamp(row.available_at)
        if available is None:
            problems.append("row %d has no parseable available_at" % position)
            continue
        observation = to_utc_timestamp(row.observation_date)
        vintage = to_utc_timestamp(row.vintage_date)
        if observation is not None and available < observation:
            problems.append("row %d available_at precedes its observation_date" % position)
        if vintage is not None and available < vintage:
            problems.append("row %d available_at precedes its vintage_date" % position)
    return problems


def macro_asof(frame, asof_ts, series_id=None):
    """Latest macro observation available at ``asof_ts`` (one row) or None."""
    moment = to_utc_timestamp(asof_ts)
    if moment is None:
        raise MacroError("asof_ts must be a parseable timestamp")
    subset = frame
    if series_id is not None:
        subset = subset.loc[subset["series_id"] == series_id]
    eligible = []
    for position, row in enumerate(subset.itertuples()):
        stamp = to_utc_timestamp(row.available_at)
        if stamp is not None and stamp <= moment:
            observation = to_utc_timestamp(row.observation_date)
            eligible.append((observation, position, row))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1]))
    row = eligible[-1][2]
    return {name: getattr(row, name) for name in subset.columns}
