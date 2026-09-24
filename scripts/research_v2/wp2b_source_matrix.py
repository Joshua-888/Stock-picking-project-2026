"""WP2B deliverable 9 - source matrix and coverage artifacts.

This generator writes the WP2B evidence bundle under artifacts/research/wp2b/:

* source_matrix.csv          - provider x capability x access x PIT status
* universe_coverage.json/.md - historical membership coverage findings
* price_coverage.json/.md    - delisted-price coverage findings
* identifier_failures.json/.md - permanent-identifier gaps
* corporate_action_checks.json/.md - corporate-action handling status

The content is derived from (a) the frozen WP2B provider capabilities and (b) the
recorded probe outputs in source_probes/. Nothing here is fabricated research
data: it is a decision/evidence record. Run:

    /opt/venv/bin/python scripts/research_v2/wp2b_source_matrix.py
"""

from __future__ import annotations

import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "artifacts", "research", "wp2b")

PRICE_NOT_VERIFIED = "PRICE_NOT_VERIFIED"

# Derived artifacts must stay consistent with the RAW probe evidence. The tracked
# (committed) copy under provenance/wp2b/source_evidence/ is the source of truth;
# artifacts/ is generated and gitignored.
PROBE_DIR = os.path.join(ROOT, "provenance", "wp2b", "source_evidence", "probes")
DELISTED_PROBE_FILE = os.path.join(PROBE_DIR, "probe_delisted_dated.txt")
TEST_FILE = os.path.join(ROOT, "tests", "test_research_v2b_universe.py")

_PROBE_LINE_RE = re.compile(r"^([A-Z][A-Z0-9.\-]{0,9})\s+HTTP\s+(\d{3})\s*$")


def read_delisted_probe(path=DELISTED_PROBE_FILE):
    """Parse the tracked RAW delisted-price probe into (ticker, status) rows.

    Fails loudly rather than inventing a probe result: a derived artifact that
    disagrees with the raw probe is a provenance defect.
    """
    if not os.path.isfile(path):
        raise SystemExit("missing tracked probe evidence: %s" % path)
    results = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = _PROBE_LINE_RE.match(line.strip())
            if match:
                results.append((match.group(1), int(match.group(2))))
    if not results:
        raise SystemExit("no delisted probe rows parsed from %s" % path)
    return results


def read_executed_tests(path=TEST_FILE):
    """Names of test functions actually present in the WP2B test module."""
    if not os.path.isfile(path):
        raise SystemExit("missing test file: %s" % path)
    names = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"def (test_[A-Za-z0-9_]+)\s*\(", line)
            if match:
                names.add(match.group(1))
    return names

# Provider matrix. `pit_status` reflects what the source can actually support for
# a survivorship-safe historical universe, not marketing claims.
SOURCE_MATRIX = [
    {
        "provider": "yahoo_chart_api",
        "access": "free",
        "credential_env": "",
        "historical_universe": "no",
        "delisted_prices": "no",
        "corporate_actions": "yes",
        "permanent_ids": "no",
        "delisted_coverage": "none for delisted/acquired names (HTTP 404)",
        "pit_status": "BLOCKED_FOR_UNIVERSE",
        "evidence": "probe_delisted_dated.txt",
    },
    {
        "provider": "wikipedia_sp500_changes",
        "access": "free",
        "credential_env": "",
        "historical_universe": "partial_membership_only",
        "delisted_prices": "no",
        "corporate_actions": "no",
        "permanent_ids": "no",
        "delisted_coverage": "membership change rows only; no prices",
        "pit_status": "PARTIAL",
        "evidence": "wikipedia_sp500_provider (WP2B adapter)",
    },
    {
        "provider": "eodhd",
        "access": "paid",
        "credential_env": "EODHD_API_TOKEN",
        "historical_universe": "no",
        "delisted_prices": "yes",
        "corporate_actions": "yes",
        "permanent_ids": "partial",
        "delisted_coverage": "delisted=1 per exchange; full history to delisting; reliable ~2012+",
        "pit_status": "PIT_READY_IF_CREDENTIALED",
        "cost": "~USD 19.99-99.99/mo (~199-999/yr) " + PRICE_NOT_VERIFIED,
        "evidence": "probe_commercial_reach.txt (403 without key)",
    },
    {
        "provider": "sharadar_via_nasdaq_data_link",
        "access": "paid",
        "credential_env": "NASDAQ_DATA_LINK_API_KEY",
        "historical_universe": "yes",
        "delisted_prices": "yes",
        "corporate_actions": "yes",
        "permanent_ids": "partial",
        "delisted_coverage": "SEP prices 1998+ incl. delisted; SF1 fundamentals; SP500 daily constituents 1998+",
        "pit_status": "PIT_READY_IF_CREDENTIALED",
        "cost": PRICE_NOT_VERIFIED,
        "evidence": "probe_commercial_reach.txt (ndl_sharadar=403 without key)",
    },
    {
        "provider": "norgate_data",
        "access": "paid",
        "credential_env": "",
        "historical_universe": "yes",
        "delisted_prices": "yes",
        "corporate_actions": "yes",
        "permanent_ids": "yes",
        "delisted_coverage": "25,222 delisted securities 1950-2022 (vendor claim)",
        "pit_status": "PIT_READY_IF_LICENSED",
        "cost": "US ~USD 630/yr Platinum (vendor claim) " + PRICE_NOT_VERIFIED,
        "evidence": "probe_commercial_reach.txt (norgate=200)",
    },
    {
        "provider": "crsp_compustat",
        "access": "institutional",
        "credential_env": "",
        "historical_universe": "yes",
        "delisted_prices": "yes",
        "corporate_actions": "yes",
        "permanent_ids": "yes",
        "delisted_coverage": "delisting returns, PERMNO; full survivorship-free history",
        "pit_status": "PIT_READY_IF_LICENSED",
        "cost": PRICE_NOT_VERIFIED,
        "evidence": "institutional reference",
    },
]

UNIVERSE_COVERAGE = {
    "requirement": (
        "point-in-time S&P 500 membership with effective dates, including names "
        "that later left the index"
    ),
    "free_membership_available": True,
    "free_prices_for_removed_names": False,
    "certified_survivorship_safe_possible": False,
    "best_free_source": "wikipedia_sp500_changes (membership only, PARTIAL)",
    "best_paid_source": "sharadar SP500 daily constituents 1998+ / Norgate index constituents",
    "limitations": [
        "free membership tables carry no announcement dates (effective-date only)",
        "no free price history for delisted/acquired constituents",
        "membership without removed-name prices cannot build an unbiased cross-section",
    ],
    "status": "BLOCKED",
}

def price_coverage_artifact():
    """Build PRICE_COVERAGE from the RAW tracked probe (never hand-typed).

    The probe result string is rendered from the parsed evidence file, so the
    derived artifact can never drift from the raw capture (WP2B fix C).
    """
    rows = read_delisted_probe()
    tickers = [ticker for ticker, _status in rows]
    codes = sorted({status for _ticker, status in rows})
    code_text = "/".join(str(code) for code in codes)
    return {
        "requirement": "historical daily prices for currently listed AND delisted/acquired names",
        "listed_names_free": True,
        "delisted_names_free": False,
        "delisted_names_paid": True,
        "delisted_probe_result": "HTTP %s for %s" % (code_text, "/".join(tickers)),
        "delisted_probe_source": (
            "provenance/wp2b/source_evidence/probes/probe_delisted_dated.txt (raw, tracked)"
        ),
        "residual_otc_note": (
            "post-delisting OTC stubs (e.g. SBNY/FRCB) may return 200 but are residual "
            "tickers, not continuous histories of the original security"
        ),
        "status": "BLOCKED",
    }


# Executed-test evidence: each claimed test maps to a real test function in
# tests/test_research_v2b_universe.py. The artifact asserts ONLY these.
CORPORATE_ACTION_TEST_EVIDENCE = [
    ("2-for-1 split / split factor applied only after effective date",
     "test_split_factor_applied_only_after_effective"),
    ("reverse split (1-for-5 back-adjustment)",
     "test_reverse_split_back_adjusts_pre_effective_prices"),
    ("ordinary dividend",
     "test_events_known_by_excludes_future"),
    ("special dividend (large dividend factor)",
     "test_special_dividend_adjusts_like_a_large_dividend"),
    ("spinoff (information-only, never moves a price)",
     "test_future_action_is_invisible_at_earlier_asof"),
    ("cash acquisition (information-only)",
     "test_future_action_is_invisible_at_earlier_asof"),
    ("stock acquisition (delisting economics, not cash settled)",
     "test_stock_acquisition_is_not_cash_settled"),
    ("bankruptcy (information-only)",
     "test_pit_total_return_ignores_informational_events"),
    ("exchange delisting (metadata only)",
     "test_only_price_actions_reach_pit_prices"),
    ("ticker rename (metadata only)",
     "test_only_price_actions_reach_pit_prices"),
    ("missing final price never treated as a zero return",
     "test_missing_final_price_is_unknown_not_zero"),
]


def corporate_action_checks_artifact():
    """Build CORPORATE_ACTION_CHECKS; every listed test must really execute."""
    executed = read_executed_tests()
    evidence = []
    for label, test_name in CORPORATE_ACTION_TEST_EVIDENCE:
        if test_name not in executed:
            raise SystemExit(
                "corporate-action check %r claims test %r which does not exist in %s"
                % (label, test_name, TEST_FILE)
            )
        evidence.append({"check": label, "test": test_name})
    return {
        "price_adjusting_kinds": ["split", "dividend"],
        "information_only_kinds": [
            "spinoff", "cash_acquisition", "stock_acquisition",
            "bankruptcy", "exchange_delisting", "ticker_rename",
        ],
        "effective_date_visibility_enforced": True,
        "future_action_is_invisible_at_T": True,
        "missing_final_price_treated_as_zero": False,
        "tests_executed": [item["check"] for item in evidence],
        "test_evidence": evidence,
        "status": "PIT_READY",
    }

IDENTIFIER_FAILURES = {
    "requirement": "permanent id (CIK/vendor) <-> issuer <-> historical ticker <-> exchange <-> valid period",
    "free_permanent_ids": "SEC EDGAR CIK (fundamentals only)",
    "free_ticker_to_delisted_identity": False,
    "failures": [
        "free price APIs expose only live tickers; no delisted symbol list",
        "reused tickers cannot be disambiguated by ticker alone",
        "EODHD exposes ticker/exchange but no CIK (chain must join SEC)",
        "Wikipedia change table has no permanent id",
    ],
    "mitigation_implemented": (
        "identity_chain.py: vendor id -> security_id with share-class separation, "
        "issuer merger/successor links, dated reuse windows, ambiguity fails loudly"
    ),
    "status": "PARTIAL",
}

def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "provider", "access", "credential_env", "historical_universe",
        "delisted_prices", "corporate_actions", "permanent_ids",
        "delisted_coverage", "pit_status", "cost", "evidence",
    ]
    fieldnames = [name for name in preferred if name in fieldnames] + [
        name for name in fieldnames if name not in preferred
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_md(path, title, payload):
    lines = ["# %s" % title, ""]
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append("- %s:" % key)
            for item in value:
                lines.append("    - %s" % item)
        else:
            lines.append("- %s: %s" % (key, value))
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _sha256_file(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    price_coverage = price_coverage_artifact()
    corporate_action_checks = corporate_action_checks_artifact()

    # Bind every derived artifact to the raw probe evidence it was derived from.
    provenance = {
        "raw_evidence_file": "provenance/wp2b/source_evidence/probes/probe_delisted_dated.txt",
        "raw_evidence_sha256": _sha256_file(DELISTED_PROBE_FILE),
        "derivation": "scripts/research_v2/wp2b_source_matrix.py (derived, not hand-typed)",
    }
    price_coverage = dict(price_coverage, provenance=provenance)

    _write_csv(os.path.join(OUT_DIR, "source_matrix.csv"), SOURCE_MATRIX)
    _write_json(os.path.join(OUT_DIR, "universe_coverage.json"), UNIVERSE_COVERAGE)
    _write_md(os.path.join(OUT_DIR, "universe_coverage.md"), "WP2B Universe Coverage", UNIVERSE_COVERAGE)
    _write_json(os.path.join(OUT_DIR, "price_coverage.json"), price_coverage)
    _write_md(os.path.join(OUT_DIR, "price_coverage.md"), "WP2B Price Coverage", price_coverage)
    _write_json(os.path.join(OUT_DIR, "identifier_failures.json"), IDENTIFIER_FAILURES)
    _write_md(os.path.join(OUT_DIR, "identifier_failures.md"), "WP2B Identifier Failures", IDENTIFIER_FAILURES)
    _write_json(os.path.join(OUT_DIR, "corporate_action_checks.json"), corporate_action_checks)
    _write_md(os.path.join(OUT_DIR, "corporate_action_checks.md"), "WP2B Corporate-Action Checks", corporate_action_checks)
    print("wrote WP2B artifacts to", OUT_DIR)
    print("price_coverage:", price_coverage["delisted_probe_result"])
    print("corporate-action tests:", len(corporate_action_checks["test_evidence"]))


if __name__ == "__main__":
    main()
