"""WP2B deliverable A - durable, reproducible source-probe evidence generator.

Problem this fixes
------------------
The WP2B source probes were originally hand-captured into a gitignored
``artifacts/`` directory: no in-band retrieval timestamp, no content fingerprint,
no structured endpoint, no license/access note and no binding to the generator or
git commit. That made the probe evidence uncertifiable.

What this script does
---------------------
It re-emits the source-probe evidence as a *self-describing manifest*:

* raw probe snippets are preserved in the TRACKED location
  ``provenance/wp2b/source_evidence/probes/*.txt`` (bulk vendor data stays out of
  git; these snippets are tiny and are the evidence itself);
* ``provenance/wp2b/source_evidence/manifest.json`` records, per probe: source
  name, endpoint/resource URL, retrieval timestamp (ISO-8601 UTC), HTTP status,
  content sha256 (whole evidence file AND the extracted snippet), license/access
  note, parser/generator version and the generating git commit;
* the extracted snippet is DERIVED from the raw evidence file by a per-probe
  pattern -- a probe whose pattern matches nothing makes the generator FAIL
  LOUDLY, so the manifest can never assert evidence that is not in the capture;
* a small consistency gate reconciles the declared outcome with the parsed
  status code (200 -> reachable, 404 -> absent, 401/403 -> blocked);
* commercial-source license/access metadata (access tier + terms note +
  redistribution restriction) is recorded, with cost kept ``PRICE_NOT_VERIFIED``
  unless a public vendor-page figure exists.

Determinism / overwrite policy
------------------------------
Default (offline) mode is fully deterministic: retrieval timestamps come from the
frozen capture registry, so re-running reproduces a byte-identical manifest.
``--live`` re-probes the network; if a remote response now differs from the frozen
capture the generator does NOT overwrite the frozen evidence -- it writes a
timestamped sidecar under ``.../live/`` and records the divergence in the
manifest.

Run:
    /opt/venv/bin/python scripts/research_v2/wp2b_probe_sources.py            # offline, deterministic
    /opt/venv/bin/python scripts/research_v2/wp2b_probe_sources.py --live     # refresh + record divergences

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

GENERATOR_VERSION = "wp2b_probe_sources/1"
MANIFEST_SCHEMA_VERSION = "wp2b_source_evidence_manifest_v1"
PRICE_NOT_VERIFIED = "PRICE_NOT_VERIFIED"
USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVIDENCE_DIR = os.path.join(ROOT, "provenance", "wp2b", "source_evidence")
PROBES_DIR = os.path.join(EVIDENCE_DIR, "probes")
LIVE_DIR = os.path.join(EVIDENCE_DIR, "live")
MANIFEST_PATH = os.path.join(EVIDENCE_DIR, "manifest.json")

# --- capture registry --------------------------------------------------------- #
# Retrieval timestamps are the CAPTURE times of the raw evidence files (in-band,
# not the manifest run time). 2026-09-24 23:45-23:49 CEST == 21:45-21:49 UTC.
EVIDENCE_FILES = {
    "probe_delisted_dated.txt": "2026-09-24T21:45:43Z",
    "probe_free_apis.txt": "2026-09-24T21:45:14Z",
    "probe_free_delisted_2.txt": "2026-09-24T21:49:07Z",
    "probe_commercial_reach.txt": "2026-09-24T21:46:32Z",
}

FREE_LICENSE_NOTE = (
    "public unauthenticated endpoint; no credential used for this probe; "
    "vendor Terms of Service apply and automated retrieval may be rate-limited "
    "or prohibited -- probe is a single documentation request, bulk data is NOT "
    "redistributed"
)

# One entry per probed endpoint. ``pattern`` is matched against the raw evidence
# file line-by-line; the matching lines ARE the recorded snippet.
PROBES = [
    {
        "probe": "yahoo_chart_delisted_prices",
        "source": "Yahoo Finance chart API (yfinance)",
        "category": "delisted_prices",
        "endpoint": "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={epoch}&period2={epoch}",
        "method": "GET",
        "credential_env": [],
        "outcome": "absent",
        "evidence_file": "probe_delisted_dated.txt",
        "pattern": r"^[A-Z]{2,6}\s+HTTP \d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": (
            "historical daily prices exist only for currently listed tickers; "
            "every tested delisted/acquired/renamed constituent returned 404"
        ),
        "live_url": "https://query1.finance.yahoo.com/v8/finance/chart/LEH",
    },
    {
        "probe": "alphavantage_demo_quote",
        "source": "Alpha Vantage",
        "category": "delisted_prices",
        "endpoint": "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo",
        "method": "GET",
        "credential_env": ["ALPHAVANTAGE_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"Information.*demo.*API key",
        "license": FREE_LICENSE_NOTE,
        "note": "demo key is restricted to a single demo symbol; not usable for delisted coverage",
        "live_url": None,
    },
    {
        "probe": "tiingo_daily_prices",
        "source": "Tiingo",
        "category": "delisted_prices",
        "endpoint": "https://api.tiingo.com/tiingo/daily/{ticker}/prices",
        "method": "GET",
        "credential_env": ["TIINGO_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"^tiingo=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "401/403 without an API token",
        "live_url": "https://api.tiingo.com/tiingo/daily/AAPL/prices",
    },
    {
        "probe": "financialmodelingprep_delisted",
        "source": "Financial Modeling Prep",
        "category": "delisted_prices",
        "endpoint": "https://financialmodelingprep.com/api/v3/delisted-companies?apikey={key}",
        "method": "GET",
        "credential_env": ["FMP_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"Invalid API KEY",
        "license": FREE_LICENSE_NOTE,
        "note": "rejects the request key; no delisted price data returned",
        "live_url": None,
    },
    {
        "probe": "twelvedata_demo",
        "source": "Twelve Data",
        "category": "delisted_prices",
        "endpoint": "https://api.twelvedata.com/time_series?symbol={ticker}&apikey=demo",
        "method": "GET",
        "credential_env": ["TWELVE_DATA_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r'"code":401',
        "license": FREE_LICENSE_NOTE,
        "note": "demo key refused; requires a claimed key",
        "live_url": None,
    },
    {
        "probe": "marketstack_eod",
        "source": "Marketstack",
        "category": "delisted_prices",
        "endpoint": "https://api.marketstack.com/v1/eod?symbols={ticker}",
        "method": "GET",
        "credential_env": ["MARKETSTACK_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"^marketstack=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "401 without a key",
        "live_url": "https://api.marketstack.com/v1/eod?symbols=AAPL",
    },
    {
        "probe": "stooq_world",
        "source": "Stooq",
        "category": "delisted_prices",
        "endpoint": "https://stooq.com/q/d/?s={ticker}",
        "method": "GET",
        "credential_env": [],
        "outcome": "blocked_automation",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"requires JavaScript",
        "license": FREE_LICENSE_NOTE,
        "note": "JavaScript bot wall; automated retrieval blocked",
        "live_url": None,
    },
    {
        "probe": "investing_com_history",
        "source": "Investing.com",
        "category": "delisted_prices",
        "endpoint": "https://www.investing.com/equities/{slug}-historical-data",
        "method": "GET",
        "credential_env": [],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_apis.txt",
        "pattern": r"^investing=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "403 on automated request",
        "live_url": None,
    },
    {
        "probe": "stockanalysis_delisted_list",
        "source": "stockanalysis.com",
        "category": "delisted_prices",
        "endpoint": "https://stockanalysis.com/list/delisted-stocks/",
        "method": "GET",
        "credential_env": [],
        "outcome": "absent",
        "evidence_file": "probe_free_delisted_2.txt",
        "pattern": r"^sa_delisted_list=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "no delisted-stock listing endpoint; only live tickers resolve",
        "live_url": "https://stockanalysis.com/list/delisted-stocks/",
    },
    {
        "probe": "stockanalysis_delisted_ticker",
        "source": "stockanalysis.com",
        "category": "delisted_prices",
        "endpoint": "https://stockanalysis.com/stocks/{ticker}/history/",
        "method": "GET",
        "credential_env": [],
        "outcome": "mixed_absent_live",
        "evidence_file": "probe_free_delisted_2.txt",
        "pattern": r"^sa_(LEHMQ|TWTR)=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "delisted LEHMQ 404; live TWTR 200 -- live-only coverage",
        "live_url": "https://stockanalysis.com/stocks/LEHMQ/history/",
    },
    {
        "probe": "alpaca_market_data",
        "source": "Alpaca",
        "category": "delisted_prices",
        "endpoint": "https://data.alpaca.markets/v2/stocks/{ticker}/bars",
        "method": "GET",
        "credential_env": ["APCA_API_KEY_ID"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_free_delisted_2.txt",
        "pattern": r"^alpaca=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "401 without API keys",
        "live_url": None,
    },
    {
        "probe": "wayback_snapshot",
        "source": "Internet Archive Wayback Machine",
        "category": "delisted_prices",
        "endpoint": "https://web.archive.org/web/{timestamp}/{url}",
        "method": "GET",
        "credential_env": [],
        "outcome": "reachable",
        "evidence_file": "probe_free_delisted_2.txt",
        "pattern": r"^wayback_sa=\d{3}$",
        "license": FREE_LICENSE_NOTE,
        "note": "reachable, but snapshots are not a systematic survivorship-free price feed",
        "live_url": None,
    },
    {
        "probe": "nasdaq_data_link_sharadar",
        "source": "Nasdaq Data Link / Sharadar (SEP, SF1, SP500)",
        "category": "commercial_reachability",
        "endpoint": "https://data.nasdaq.com/api/v3/datatables/{table}.json",
        "method": "GET",
        "credential_env": ["NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY", "SHARADAR_API_KEY"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_commercial_reach.txt",
        "pattern": r"^ndl_sharadar=\d{3}$",
        "license": (
            "commercial subscription required; Nasdaq Data Link Terms of Service and "
            "Sharadar vendor licence apply; data is licensed for internal research "
            "and redistribution is restricted"
        ),
        "note": "403 without a subscription key",
        "live_url": "https://data.nasdaq.com/api/v3/datatables/SEP.json",
    },
    {
        "probe": "eodhd_delisted_symbol_list",
        "source": "EODHD (eodhd.com)",
        "category": "commercial_reachability",
        "endpoint": "https://eodhd.com/api/exchange-symbol-list/{exchange}?delisted=1&api_token={key}",
        "method": "GET",
        "credential_env": ["EODHD_API_TOKEN"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_commercial_reach.txt",
        "pattern": r"^eodhd_exchange=\d{3}$",
        "license": (
            "commercial subscription required; EODHD Terms of Service apply; "
            "redistribution of raw vendor data is prohibited"
        ),
        "note": "403 without an API token",
        "live_url": "https://eodhd.com/api/exchange-symbol-list/US?delisted=1",
    },
    {
        "probe": "eodhd_eod_forbidden",
        "source": "EODHD (eodhd.com)",
        "category": "commercial_reachability",
        "endpoint": "https://eodhd.com/api/eod/{ticker}.US?api_token={key}",
        "method": "GET",
        "credential_env": ["EODHD_API_TOKEN"],
        "outcome": "blocked_credential_or_terms",
        "evidence_file": "probe_commercial_reach.txt",
        "pattern": r"^Forbidden$",
        "license": (
            "commercial subscription required; EODHD Terms of Service apply; "
            "redistribution of raw vendor data is prohibited"
        ),
        "note": "403 Forbidden without an API token",
        "live_url": "https://eodhd.com/api/eod/AAPL.US",
    },
    {
        "probe": "norgate_data_page",
        "source": "Norgate Data",
        "category": "commercial_reachability",
        "endpoint": "https://norgatedata.com/",
        "method": "GET",
        "credential_env": [],
        "outcome": "reachable",
        "evidence_file": "probe_commercial_reach.txt",
        "pattern": r"^norgate=\d{3}$",
        "license": (
            "vendor page reachable; data access requires a paid licence; "
            "redistribution is prohibited by the vendor licence"
        ),
        "note": "200: vendor page reachable, licence required for data",
        "live_url": "https://norgatedata.com/",
    },
    {
        "probe": "sec_edgar_full_text_search",
        "source": "SEC EDGAR full-text search",
        "category": "free_filings",
        "endpoint": "https://efts.sec.gov/LATEST/search-index?q={query}",
        "method": "GET",
        "credential_env": [],
        "outcome": "reachable",
        "evidence_file": "probe_commercial_reach.txt",
        "pattern": r"^sec_ft=\d{3}$",
        "license": (
            "US SEC public filings; free and public; SEC fair-access policy "
            "(declared User-Agent, request rate) applies"
        ),
        "note": "200: available, but supplies filings/fundamentals, not delisted prices",
        "live_url": "https://efts.sec.gov/LATEST/search-index?q=delisting",
    },
]

# Recommended commercial sources: access tier, terms note, redistribution rule.
# ``cost`` stays PRICE_NOT_VERIFIED unless a public vendor-page figure exists.
COMMERCIAL_SOURCES = [
    {
        "source": "eodhd",
        "access_tier": "paid subscription (API token)",
        "endpoint": "https://eodhd.com/api/eod/{TICKER}.{EXCHANGE}",
        "credential_env": ["EODHD_API_TOKEN"],
        "capabilities": ["delisted_prices", "corporate_actions", "identities"],
        "cost": "~USD 19.99-99.99/mo (~199-999/yr)",
        "cost_source": "vendor page (public figure)",
        "cost_verified": PRICE_NOT_VERIFIED,
        "license_terms_note": (
            "commercial vendor subscription; EODHD Terms of Service govern use; "
            "no historical index-membership endpoint is published"
        ),
        "redistribution": "REDISTRIBUTION_PROHIBITED (raw vendor data; internal research use only)",
    },
    {
        "source": "sharadar_via_nasdaq_data_link",
        "access_tier": "paid subscription (API key, Nasdaq Data Link)",
        "endpoint": "https://data.nasdaq.com/api/v3/datatables/{SEP|SF1|SP500}.json",
        "credential_env": ["NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY", "SHARADAR_API_KEY"],
        "capabilities": ["delisted_prices", "historical_universe", "corporate_actions"],
        "cost": PRICE_NOT_VERIFIED,
        "cost_source": "not verified (no public figure captured)",
        "cost_verified": PRICE_NOT_VERIFIED,
        "license_terms_note": (
            "Nasdaq Data Link Terms of Service plus Sharadar vendor licence; SP500 "
            "daily constituents carry no announcement dates"
        ),
        "redistribution": "REDISTRIBUTION_PROHIBITED (licensed vendor data; internal research use only)",
    },
    {
        "source": "norgate_data",
        "access_tier": "paid licence (desktop data subscription)",
        "endpoint": "https://norgatedata.com/ (licensed data delivery, not a public API)",
        "credential_env": [],
        "capabilities": ["delisted_prices", "historical_universe", "corporate_actions", "identities"],
        "cost": "US ~USD 630/yr Platinum (vendor page claim)",
        "cost_source": "vendor page (public figure)",
        "cost_verified": PRICE_NOT_VERIFIED,
        "license_terms_note": (
            "vendor licence required; index constituents need a top tier; "
            "delisted coverage is a vendor claim (25,222 delisted 1950-2022)"
        ),
        "redistribution": "REDISTRIBUTION_PROHIBITED (vendor licence; internal research use only)",
    },
    {
        "source": "crsp_compustat",
        "access_tier": "institutional licence",
        "endpoint": "institutional distribution (WRDS / CRSP / Compustat)",
        "credential_env": [],
        "capabilities": ["delisted_prices", "historical_universe", "corporate_actions", "identities"],
        "cost": PRICE_NOT_VERIFIED,
        "cost_source": "not verified (institutional quote required)",
        "cost_verified": PRICE_NOT_VERIFIED,
        "license_terms_note": (
            "institutional licence; PERMNO permanent ids and delisting returns; "
            "access restricted to the licensed institution"
        ),
        "redistribution": "REDISTRIBUTION_PROHIBITED (institutional licence; no redistribution)",
    },
]

_STATUS_PATTERNS = (
    r"HTTP\s+(\d{3})",
    r"=(\d{3})\b",
    r'"status"\s*:\s*(\d{3})',
    r'"code"\s*:\s*(\d{3})',
)

_OUTCOME_BY_STATUS = {
    "200": "reachable",
    "404": "absent",
    "401": "blocked_credential_or_terms",
    "403": "blocked_credential_or_terms",
}


class ProbeEvidenceError(RuntimeError):
    """Raised when the probe evidence or registry is inconsistent."""


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def git_commit():
    """Current git commit, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def generator_sha256():
    return sha256_file(os.path.abspath(__file__))


def extract_snippet(text, pattern, label):
    """Return the lines of ``text`` matching ``pattern`` (fail loudly if none)."""
    regex = re.compile(pattern)
    lines = [line for line in text.splitlines() if regex.search(line)]
    if not lines:
        raise ProbeEvidenceError(
            "probe %s: no line in the evidence file matches %r -- refusing to "
            "assert evidence that is not in the capture" % (label, pattern)
        )
    return lines


def extract_status(lines):
    """First HTTP status code found in the snippet, else ``None``."""
    for line in lines:
        for pattern in _STATUS_PATTERNS:
            match = re.search(pattern, line)
            if match:
                return int(match.group(1))
    return None


def status_text(lines):
    """Status/outcome text when the snippet carries no numeric code."""
    return "; ".join(line.strip() for line in lines if not any(
        re.search(pattern, line) for pattern in _STATUS_PATTERNS
    )) or None


def _check_outcome_consistency(probe_name, declared_outcome, status):
    if status is None:
        return
    expected = _OUTCOME_BY_STATUS.get(str(status))
    if expected is None:
        return
    declared = declared_outcome
    if declared.startswith("mixed") and expected in ("absent", "reachable"):
        return
    if declared != expected:
        raise ProbeEvidenceError(
            "probe %s: declared outcome %r contradicts parsed HTTP status %s "
            "(expected %r)" % (probe_name, declared_outcome, status, expected)
        )


def build_manifest(live=False, timeout=20):
    """Build the evidence manifest; optionally re-probe the network."""
    evidence_files = []
    bodies = {}
    for name, captured_at in sorted(EVIDENCE_FILES.items()):
        path = os.path.join(PROBES_DIR, name)
        if not os.path.isfile(path):
            raise ProbeEvidenceError("missing tracked evidence file: %s" % path)
        text = open(path, "r", encoding="utf-8").read()
        bodies[name] = text
        evidence_files.append(
            {
                "evidence_file": name,
                "relative_path": "provenance/wp2b/source_evidence/probes/%s" % name,
                "captured_at": captured_at,
                "bytes": len(text.encode("utf-8")),
                "sha256": sha256_file(path),
            }
        )

    probes = []
    divergences = []
    for entry in PROBES:
        source_file = entry["evidence_file"]
        if source_file not in bodies:
            raise ProbeEvidenceError("probe %s references unknown evidence file" % entry["probe"])
        snippet = extract_snippet(bodies[source_file], entry["pattern"], entry["probe"])
        status = extract_status(snippet)
        _check_outcome_consistency(entry["probe"], entry["outcome"], status)
        record = {
            "probe": entry["probe"],
            "source": entry["source"],
            "category": entry["category"],
            "endpoint": entry["endpoint"],
            "method": entry["method"],
            "credential_env": list(entry["credential_env"]),
            "retrieved_at": EVIDENCE_FILES[source_file],
            "http_status": status,
            "status_text": status_text(snippet),
            "outcome": entry["outcome"],
            "evidence_file": source_file,
            "extracted_lines": snippet,
            "snippet_sha256": sha256_text("\n".join(snippet)),
            "evidence_sha256": sha256_file(os.path.join(PROBES_DIR, source_file)),
            "license_access_note": entry["license"],
            "note": entry["note"],
        }
        if live and entry.get("live_url"):
            live_result = live_probe(entry["live_url"], timeout=timeout)
            record["live_check"] = live_result
            if live_result.get("http_status") != status:
                divergence = {
                    "probe": entry["probe"],
                    "endpoint": live_result["url"],
                    "recorded_status": status,
                    "live_status": live_result.get("http_status"),
                    "live_error": live_result.get("error"),
                    "checked_at": live_result["checked_at"],
                }
                divergences.append(divergence)
                record["live_check"]["diverges_from_capture"] = True
            else:
                record["live_check"]["diverges_from_capture"] = False
        probes.append(record)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": {
            "path": "scripts/research_v2/wp2b_probe_sources.py",
            "version": GENERATOR_VERSION,
            "sha256": generator_sha256(),
        },
        "git_commit": git_commit(),
        "mode": "live" if live else "offline_deterministic",
        "evidence_files": evidence_files,
        "probes": probes,
        "commercial_sources": COMMERCIAL_SOURCES,
        "live_divergences": divergences,
        "bulk_data_committed": False,
        "notes": (
            "Raw probe snippets only (tiny, documentary). Vendor bulk data is never "
            "committed. Offline mode is deterministic: retrieval timestamps come from "
            "the frozen capture registry."
        ),
    }
    manifest["bundle_fingerprint"] = bundle_fingerprint(manifest)
    return manifest


def bundle_fingerprint(manifest):
    """Deterministic fingerprint of the evidence bundle (volatile fields excluded)."""
    payload = {
        "manifest_schema_version": manifest["manifest_schema_version"],
        "generator_version": manifest["generator"]["version"],
        "evidence_files": manifest["evidence_files"],
        "probes": [
            {key: value for key, value in probe.items() if key not in ("live_check",)}
            for probe in manifest["probes"]
        ],
        "commercial_sources": manifest["commercial_sources"],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def live_probe(url, timeout=20):
    """Single unauthenticated GET; returns status/body fingerprint, never raises."""
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "url": url,
                "http_status": response.status,
                "body_sha256": sha256_text(body),
                "body_head": body[:200],
                "checked_at": checked_at,
            }
    except urllib.error.HTTPError as exc:
        body = ""
        if exc.fp is not None:
            body = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "url": url,
            "http_status": exc.code,
            "body_sha256": sha256_text(body),
            "body_head": body[:200],
            "checked_at": checked_at,
        }
    except Exception as exc:  # noqa: BLE001 - network failure must not crash the generator
        return {
            "url": url,
            "http_status": None,
            "error": type(exc).__name__,
            "message": str(exc)[:200],
            "checked_at": checked_at,
        }


def write_manifest(manifest, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def write_live_divergences(manifest):
    """Write a timestamped sidecar per divergence; never overwrite evidence."""
    written = []
    if not manifest.get("live_divergences"):
        return written
    os.makedirs(LIVE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for divergence in manifest["live_divergences"]:
        name = "%s.%s.json" % (divergence["probe"], stamp)
        target = os.path.join(LIVE_DIR, name)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(divergence, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(target)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Emit the WP2B source-probe evidence manifest.")
    parser.add_argument("--live", action="store_true", help="re-probe the network and record divergences")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--check", action="store_true", help="verify the on-disk manifest matches the evidence")
    args = parser.parse_args(argv)

    manifest = build_manifest(live=args.live, timeout=args.timeout)

    if args.check:
        if not os.path.isfile(MANIFEST_PATH):
            print("FAIL: manifest missing at %s" % MANIFEST_PATH, file=sys.stderr)
            return 2
        on_disk = json.load(open(MANIFEST_PATH, "r", encoding="utf-8"))
        if on_disk.get("bundle_fingerprint") != manifest["bundle_fingerprint"]:
            print(
                "FAIL: manifest fingerprint %s != recomputed %s"
                % (on_disk.get("bundle_fingerprint"), manifest["bundle_fingerprint"]),
                file=sys.stderr,
            )
            return 2
        print("OK: manifest matches evidence bundle (%s)" % manifest["bundle_fingerprint"])
        return 0

    path = write_manifest(manifest)
    sidecars = write_live_divergences(manifest)
    print("wrote %s" % path)
    print("bundle_fingerprint=%s" % manifest["bundle_fingerprint"])
    print("mode=%s probes=%d evidence_files=%d" % (manifest["mode"], len(manifest["probes"]), len(manifest["evidence_files"])))
    for sidecar in sidecars:
        print("recorded divergence (no overwrite): %s" % sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
