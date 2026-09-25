"""WP4 bounded free SEC EDGAR fundamental ingestion for the certified universe.

Run (research mode, real data only):

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp4_ingest_edgar_universe.py

WP3 declared fundamental availability rules but computed no fundamental VALUE.
WP4 needs real filed fundamentals (EPS history, revenue history, equity, cash,
debt, assets) to compute the Keyes variables X5/X8/X9 and the X12 proxy. This
script performs the bounded, resumable EDGAR ingestion of the companies behind
the certified WP2C/WP3 universe, reusing the certified
``fundamentals_edgar.ingest_company`` path unchanged.

Honesty rules:

* the CIK mapping comes from the official ``company_tickers.json`` table; a
  ticker that maps to more than one distinct CIK is recorded as AMBIGUOUS and is
  NOT ingested, because a wrong CIK would attach another company's fundamentals;
* a ticker with no official mapping is recorded as UNMAPPED; its
  fundamental-dependent variables are UNAVAILABLE, never estimated;
* the mapping table is persisted as a provenance artefact so the reuse risk of
  a recycled historical ticker symbol is auditable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data.fundamentals_edgar import USER_AGENT, ingest_company
from src.research.immutability import write_json_atomic
from src.research.modes import ResearchMode

BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp4"
GOLD_TARGETS = ROOT / "data" / "research_v2" / "gold" / "sp500_pit_targets_v1" / "f244f86c22b2c2a4" / "data.parquet"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MATCH_EXACT = "EXACT"
MATCH_AMBIGUOUS = "AMBIGUOUS"
MATCH_UNMAPPED = "UNMAPPED"


class IngestionError(RuntimeError):
    """Raised when the EDGAR universe ingestion cannot proceed."""


def fetch_company_tickers():
    """Official SEC ticker->CIK table, fetched with the project User-Agent."""
    request = urllib.request.Request(
        COMPANY_TICKERS_URL,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def build_ticker_cik_map(payload, tickers):
    """Map universe tickers to a single CIK, flagging ambiguity explicitly."""
    by_ticker = {}
    for entry in (payload or {}).values():
        ticker = str(entry.get("ticker") or "").strip().upper()
        cik = entry.get("cik_str")
        if not ticker or cik is None:
            continue
        by_ticker.setdefault(ticker, set()).add(int(cik))

    mapping = {}
    for ticker in sorted({str(value).upper() for value in tickers}):
        candidates = sorted(by_ticker.get(ticker, set()))
        if len(candidates) == 1:
            mapping[ticker] = {"match": MATCH_EXACT, "cik": candidates[0], "cik_candidates": candidates}
        elif len(candidates) > 1:
            mapping[ticker] = {"match": MATCH_AMBIGUOUS, "cik": None, "cik_candidates": candidates}
        else:
            mapping[ticker] = {"match": MATCH_UNMAPPED, "cik": None, "cik_candidates": []}
    return mapping


def load_universe_tickers():
    if not GOLD_TARGETS.is_file():
        raise IngestionError("certified WP3 gold targets are missing: %s" % GOLD_TARGETS)
    frame = pd.read_parquet(GOLD_TARGETS, columns=["ticker"])
    return sorted({str(value) for value in frame["ticker"]})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="ingest at most N companies")
    parser.add_argument("--sleep", type=float, default=0.15, help="seconds between companies")
    parser.add_argument("--force", action="store_true", help="ignore the resume progress file")
    args = parser.parse_args(argv)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = Path("/tmp/wp4_edgar_progress.json")
    done = {}
    if progress_path.is_file() and not args.force:
        done = json.loads(progress_path.read_text(encoding="utf-8"))

    tickers = load_universe_tickers()
    payload = fetch_company_tickers()
    mapping = build_ticker_cik_map(payload, tickers)
    exact = {t: v for t, v in mapping.items() if v["match"] == MATCH_EXACT}
    print("universe=%d exact=%d ambiguous=%d unmapped=%d" % (
        len(tickers), len(exact),
        sum(1 for v in mapping.values() if v["match"] == MATCH_AMBIGUOUS),
        sum(1 for v in mapping.values() if v["match"] == MATCH_UNMAPPED)))

    ordered = sorted(exact.items(), key=lambda item: item[1]["cik"])
    if args.limit is not None:
        ordered = ordered[: args.limit]

    results = {}
    for position, (ticker, record) in enumerate(ordered, start=1):
        cik = record["cik"]
        key = str(cik)
        if key in done:
            results[ticker] = done[key]
            continue
        try:
            outcome = ingest_company(cik, BRONZE_ROOT, ResearchMode.RESEARCH_V2)
            results[ticker] = {
                "cik": key,
                "rows": int(outcome["row_count"]),
                "fields": list(outcome["diagnostics"]["fields_present"]),
            }
        except Exception as exc:  # noqa: BLE001 - surfaced, never hidden
            results[ticker] = {"cik": key, "error": "%s: %s" % (type(exc).__name__, exc)}
        done[key] = results[ticker]
        if position % 25 == 0 or position == len(ordered):
            write_json_atomic(progress_path, done)
            print("progress %d/%d cik=%s rows=%s" % (
                position, len(ordered), key, results[ticker].get("rows")))
        time.sleep(args.sleep)
    write_json_atomic(progress_path, done)

    mapping_payload = {
        "source": "https://www.sec.gov/files/company_tickers.json",
        "universe_tickers": len(tickers),
        "exact": len(exact),
        "ambiguous": sum(1 for v in mapping.values() if v["match"] == MATCH_AMBIGUOUS),
        "unmapped": sum(1 for v in mapping.values() if v["match"] == MATCH_UNMAPPED),
        "mapping": mapping,
        "limitations": [
            "the SEC ticker table is a CURRENT table; a historical ticker reused by another issuer could map to a different CIK",
            "ambiguous tickers are excluded rather than guessed",
            "unmapped tickers (mostly delisted suffixes such as *_OLD) yield UNAVAILABLE fundamental variables, never estimates",
        ],
    }
    write_json_atomic(ARTIFACT_DIR / "edgar_cik_mapping.json", mapping_payload)

    ingested = sorted({int(v["cik"]) for v in results.values() if "cik" in v and "error" not in v})
    summary = {
        "universe_tickers": len(tickers),
        "exact_mappings": len(exact),
        "companies_ingested": len(ingested),
        "errors": {t: v for t, v in results.items() if "error" in v},
        "total_rows": int(sum(v.get("rows", 0) for v in results.values())),
    }
    write_json_atomic(ARTIFACT_DIR / "edgar_ingestion_summary.json", summary)
    print("WP4_EDGAR_DONE companies=%d rows=%d errors=%d" % (
        len(ingested), summary["total_rows"], len(summary["errors"])))


if __name__ == "__main__":
    main()
