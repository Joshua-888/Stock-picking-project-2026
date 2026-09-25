"""SEC EDGAR point-in-time fundamentals ingestion (primary source).

Two official endpoints are used:

* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` -- reported
  values, each tagged with accession, form, fiscal year/period and filing date.
* ``https://data.sec.gov/submissions/CIK##########.json`` -- filing index used
  to recover precise acceptance timestamps and an amendment indicator.

A Silver record always carries the *filing* provenance. The fiscal period end is
recorded as what the value represents; ``available_at`` is derived from the
filings, never from the period end. Every distinct accession is retained as its
own version, so a later restatement never overwrites the original fact.

If a required concept is absent, the observation is left UNAVAILABLE: nothing is
estimated, scaled or substituted.
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
import urllib.request

import pandas as pd

from src.research.fingerprints import fingerprint_obj

from .availability import fundamental_available_at, to_utc_timestamp

USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK%s.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK%s.json"
REQUEST_TIMEOUT = 30

# Ordered candidate concepts per normalised field. The first concept present for
# a (cik, accession, period) wins; alternatives are tried only when earlier ones
# are absent for that observation. No cross-concept merging of one value occurs.
CONCEPT_CANDIDATES = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "total_debt": ("LongTermDebt", "LongTermDebtNoncurrent", "DebtLongtermAndShorttermCombinedAmount"),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
    "free_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    # WP4 additions: shares outstanding is required for a point-in-time P/E and a
    # point-in-time market capitalisation; gross profit is required for a real
    # gross margin. Both are reported facts; nothing is derived here.
    "shares_outstanding": (
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
    "gross_profit": ("GrossProfit",),
}

# Fields whose reported value already is a cash-flow figure; used for the FCF
# derivation status so downstream code can distinguish a true FCF concept from
# an operating-cash-flow proxy.
FCF_PROXY_CONCEPTS = frozenset({"NetCashProvidedByUsedInOperatingActivities"})

INSTANT_CONCEPTS = frozenset(
    {
        "Assets",
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "DebtLongtermAndShorttermCombinedAmount",
    }
)


class EdgarError(RuntimeError):
    """Raised when SEC EDGAR cannot be reached or returns unusable data."""


class MissingPublicationTimestampError(EdgarError):
    """Raised when a fact has no usable filing/publication timestamp."""


def pad_cik(cik):
    """Return the zero-padded 10-digit CIK string used by SEC URLs."""
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise EdgarError("CIK must contain digits: %r" % (cik,))
    return digits.zfill(10)


def _http_get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                import gzip

                payload = gzip.decompress(payload)
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise EdgarError("SEC EDGAR HTTP %s for %s" % (exc.code, url)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EdgarError("SEC EDGAR unreachable for %s: %s" % (url, exc)) from exc
    except json.JSONDecodeError as exc:
        raise EdgarError("SEC EDGAR returned non-JSON for %s: %s" % (url, exc)) from exc


def fetch_companyfacts(cik):
    """Raw companyfacts payload for ``cik`` (Bronze-able exactly as returned)."""
    return _http_get_json(COMPANYFACTS_URL % pad_cik(cik))


def fetch_submissions(cik):
    """Raw submissions payload for ``cik``."""
    return _http_get_json(SUBMISSIONS_URL % pad_cik(cik))


def _acceptance_index(submissions):
    """Map accession -> {acceptance_datetime, filing_date, is_amendment, form}.

    SEC serves submission history in a ``recent`` block plus optional older
    ``files``; only the ``recent`` block is parsed here, and the count of parsed
    filings is returned alongside so callers can detect truncation.
    """
    index = {}
    filings = (submissions or {}).get("filings", {})
    recent = filings.get("recent", {}) or {}
    accessions = recent.get("accessionNumber", []) or []
    forms = recent.get("form", []) or []
    filing_dates = recent.get("filingDate", []) or []
    acceptances = recent.get("acceptanceDateTime", []) or []
    for position, accession in enumerate(accessions):
        if not accession:
            continue
        form = forms[position] if position < len(forms) else ""
        index[accession] = {
            "acceptance_datetime": acceptances[position] if position < len(acceptances) else None,
            "filing_date": filing_dates[position] if position < len(filing_dates) else None,
            "form": form,
            "is_amendment": "/A" in str(form),
        }
    return index, len(accessions)


def _facts_for_concept(companyfacts, concept):
    facts = (companyfacts or {}).get("facts", {})
    for taxonomy in ("us-gaap", "ifrs-full", "dei"):
        taxonomy_facts = facts.get(taxonomy, {})
        if concept in taxonomy_facts:
            return taxonomy_facts[concept]
    return None


def _unit_values(concept_payload):
    """Yield ``(unit, fact)`` pairs for a companyfacts concept payload."""
    for unit, entries in (concept_payload or {}).get("units", {}).items():
        for entry in entries or []:
            yield unit, entry


def _period_end(fact):
    """Fiscal period end: ``end`` for every fact, ``start`` only when present."""
    end = fact.get("end")
    start = fact.get("start")
    return start, end


def build_fundamental_records(companyfacts, submissions=None, cik=None):
    """Normalise companyfacts + submissions into versioned Silver records.

    One record is produced per (cik, concept, accession, period) so restatements
    remain separate observations. Facts lacking any filing/publication timestamp
    are dropped and reported via ``unavailable``. Nothing is invented: a field
    with no reported concept is simply absent for that accession.
    """
    if companyfacts is None:
        raise EdgarError("companyfacts payload is required")
    resolved_cik = pad_cik(cik if cik is not None else companyfacts.get("cik", 0))
    acceptance_index, parsed_filings = _acceptance_index(submissions or {})
    records = []
    unavailable = []
    seen_concepts = set()
    for field, candidates in CONCEPT_CANDIDATES.items():
        concept_used = None
        for concept in candidates:
            payload = _facts_for_concept(companyfacts, concept)
            if payload is None:
                continue
            concept_used = concept
            seen_concepts.add(concept)
            for unit, fact in _unit_values(payload):
                accession = fact.get("accn")
                filing_date = fact.get("filed")
                start, end = _period_end(fact)
                meta = acceptance_index.get(accession, {})
                acceptance = meta.get("acceptance_datetime")
                form = meta.get("form") or fact.get("form")
                is_amendment = bool(meta.get("is_amendment")) or "/A" in str(fact.get("form") or "")
                available_at = fundamental_available_at(filing_date, acceptance_datetime=acceptance)
                if available_at is None:
                    unavailable.append(
                        {
                            "cik": resolved_cik,
                            "field": field,
                            "concept": concept,
                            "accession": accession,
                            "reason": "missing publication timestamp (no filing date or acceptance datetime)",
                        }
                    )
                    continue
                if end is None:
                    unavailable.append(
                        {
                            "cik": resolved_cik,
                            "field": field,
                            "concept": concept,
                            "accession": accession,
                            "reason": "missing fiscal period end",
                        }
                    )
                    continue
                value = fact.get("val")
                records.append(
                    {
                        "cik": resolved_cik,
                        "concept": concept,
                        "field": field,
                        "unit": unit,
                        "value": value,
                        "fiscal_period_start": start,
                        "fiscal_period_end": end,
                        "fiscal_year": fact.get("fy"),
                        "fiscal_period": fact.get("fp"),
                        "form": form,
                        "accession": accession,
                        "filing_date": filing_date,
                        "acceptance_datetime": acceptance,
                        "available_at": available_at.isoformat(),
                        "is_amendment": is_amendment,
                        "is_fcf_proxy": concept in FCF_PROXY_CONCEPTS,
                        "frame": fact.get("frame"),
                        "source_url": COMPANYFACTS_URL % resolved_cik,
                    }
                )
            break  # first candidate concept present wins for this field
    frame = pd.DataFrame(records)
    diagnostics = {
        "cik": resolved_cik,
        "entity_name": (companyfacts or {}).get("entityName"),
        "parsed_submission_filings": parsed_filings,
        "concepts_used": sorted(seen_concepts),
        "fields_present": sorted({record["field"] for record in records}),
        "unavailable": unavailable,
    }
    return frame, diagnostics


def ingest_company(cik, root, mode, client=None):
    """Fetch, bronze and silver one company's point-in-time fundamentals.

    ``client`` may inject pre-fetched ``companyfacts``/``submissions`` payloads
    for offline tests; it must be a mapping with those keys. In RESEARCH_V2 a
    synthetic client is refused by the research-mode gate.
    """
    from src.research.modes import assert_no_synthetic_in_research

    from . import layers

    synthetic = bool((client or {}).get("synthetic")) if isinstance(client, dict) else False
    assert_no_synthetic_in_research(mode, synthetic, context="edgar ingestion")

    if client and "companyfacts" in client:
        companyfacts = client["companyfacts"]
        submissions = client.get("submissions")
    else:
        companyfacts = fetch_companyfacts(cik)
        submissions = fetch_submissions(cik)

    resolved = pad_cik(cik)
    ingestion_ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    bronze_facts = layers.write_bronze_json(
        root, "edgar_companyfacts", companyfacts, meta={"cik": resolved, "ingestion_ts": ingestion_ts}
    )
    bronze_subs = None
    if submissions is not None:
        bronze_subs = layers.write_bronze_json(
            root, "edgar_submissions", submissions, meta={"cik": resolved, "ingestion_ts": ingestion_ts}
        )

    frame, diagnostics = build_fundamental_records(companyfacts, submissions, cik=resolved)
    if not frame.empty:
        frame = frame.copy()
        frame["ingestion_ts"] = ingestion_ts
        frame["source"] = "SEC_EDGAR"
        frame["raw_value_fingerprint"] = [
            fingerprint_obj(
                {
                    "cik": row.cik,
                    "concept": row.concept,
                    "accession": row.accession,
                    "period_end": row.fiscal_period_end,
                    "value": None if pd.isna(row.value) else row.value,
                }
            )
            for row in frame.itertuples()
        ]
    silver_record = None
    if not frame.empty:
        silver_record = layers.write_silver_table(
            root,
            "edgar_fundamentals",
            frame,
            meta={"cik": resolved, "ingestion_ts": ingestion_ts, "source": "SEC_EDGAR"},
        )
    return {
        "cik": resolved,
        "bronze_facts": bronze_facts,
        "bronze_submissions": bronze_subs,
        "silver": silver_record,
        "row_count": int(len(frame)),
        "diagnostics": diagnostics,
        "ingestion_ts": ingestion_ts,
    }


def validate_records(frame):
    """Integrity checks on a Silver fundamentals frame; returns problem list."""
    problems = []
    required = (
        "cik",
        "concept",
        "field",
        "fiscal_period_end",
        "filing_date",
        "available_at",
        "accession",
        "unit",
    )
    for name in required:
        if name not in frame.columns:
            problems.append("missing required column %r" % name)
    if problems:
        return problems
    for position, row in enumerate(frame.itertuples()):
        available = to_utc_timestamp(row.available_at)
        if available is None:
            problems.append("row %d has no parseable available_at" % position)
            continue
        period_end = to_utc_timestamp(row.fiscal_period_end)
        filing_date = to_utc_timestamp(row.filing_date)
        if period_end is not None and available <= period_end:
            problems.append(
                "row %d available_at %s is not after fiscal_period_end %s (period-end leakage)"
                % (position, row.available_at, row.fiscal_period_end)
            )
        if filing_date is not None and available < filing_date.normalize():
            problems.append("row %d available_at precedes its filing date" % position)
    return problems
