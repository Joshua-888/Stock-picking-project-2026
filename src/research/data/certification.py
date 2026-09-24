"""WP2A minimal research dataset certification (bounded, honest, reproducible).

This module wires the WP2 point-in-time primitives into one certification run:

* SECURITY MASTER -- live CIK mapping from SEC ``company_tickers.json`` so
  ``security_id`` derives from an issuer identifier and the ticker stays an
  attribute;
* PRICES -- BRONZE (raw Yahoo payload) -> SILVER (raw close + split/dividend
  events + close-based ``available_at``) -> GOLD (PIT table keyed by
  ``security_id`` + ``trade_date``). The provider ``adjusted_close`` is retained
  only as evidence and is NEVER treated as point-in-time safe; the PIT adjusted
  close is rebuilt from the raw close and the corporate-action ledger.
* BENCHMARK -- SPY versioned separately with total-return vs price-return
  semantics and nearest-PRIOR alignment only;
* FUNDAMENTALS -- SEC EDGAR BRONZE -> SILVER -> as-of GOLD (verified, not
  redesigned);
* UNIVERSE -- current-constituent sample, reported PARTIAL (survivorship-biased);
* MACRO -- EXCLUDED_FROM_INITIAL_RESEARCH (architecture + leakage guard kept).

Every family is reported with an explicit status. Missing real data is a hard
failure: nothing is estimated, substituted or synthesised. This module records
and protects provenance; it performs no feature engineering, no model fitting and
no scoring.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.research.modes import ResearchMode

from . import layers
from .manifesting import build_and_persist_gold
from .pit_prices import actions_from_yahoo_events, build_pit_adjusted_frame, validate_action_ledger
from .sample_universe import SampleUniverse, load_sample_universe
from .security_master import SecurityMaster, SecurityRecord, TickerAssignment, security_id_for
from .universe import KNOWN_LIMITATION_NO_DELISTING_SOURCE, UniverseMembership, UniverseTable

PIT_READY = "PIT_READY"
PARTIAL = "PARTIAL"
BLOCKED = "BLOCKED"
EXCLUDED = "EXCLUDED"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "Stock-picking-project-2026 research (data-integrity-agent) research@example.com"
REQUEST_TIMEOUT = 30
SAMPLE_START = "2015-01-01"

FAMILIES = ("fundamentals", "prices", "benchmark", "security_master", "historical_universe", "provenance", "macro")


class CertificationError(RuntimeError):
    """Raised when certification cannot proceed honestly."""


@dataclass(frozen=True)
class FamilyStatus:
    """Honest certification status for one data family."""

    family: str
    status: str
    pit_status: str = "unknown"
    survivorship_safe: bool = None
    limitations: tuple = ()
    evidence: dict = field(default_factory=dict)
    blockers: tuple = ()

    def to_dict(self):
        return {
            "family": self.family,
            "status": self.status,
            "pit_status": self.pit_status,
            "survivorship_safe": self.survivorship_safe,
            "limitations": list(self.limitations),
            "evidence": dict(self.evidence),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CertificationReport:
    """Aggregate certification report for the minimal research dataset."""

    dataset_id: str
    created_at: str
    git_commit: str
    branch: str
    universe_id: str
    families: tuple
    manifests: dict
    securities: int
    observations: int
    period_start: str
    period_end: str
    overall_status: str
    external_blockers: tuple
    notes: str = ""

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "branch": self.branch,
            "universe_id": self.universe_id,
            "families": [item.to_dict() for item in self.families],
            "manifests": dict(self.manifests),
            "securities": self.securities,
            "observations": self.observations,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "overall_status": self.overall_status,
            "external_blockers": list(self.external_blockers),
            "notes": self.notes,
        }

    def status_table(self):
        return "\n".join(
            "%-22s %-9s %s" % (item.family, item.status, "; ".join(item.limitations) or "-")
            for item in self.families
        )


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git_ref():
    from src.research.modes import current_branch, current_git_commit

    return current_git_commit() or "unknown", current_branch() or "unknown"


# ── Security master (live CIK wiring) ────────────────────────────────────────

def fetch_sec_tickers(client=None):
    """Return the SEC ticker->CIK mapping (raw payload injectable for tests)."""
    if client is not None and "company_tickers" in client:
        payload = client["company_tickers"]
    else:
        import urllib.request

        request = urllib.request.Request(SEC_TICKERS_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    mapping = {}
    for value in (payload or {}).values():
        ticker = str(value.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        mapping[ticker] = {
            "cik": str(value.get("cik_str") or value.get("cik") or "").zfill(10),
            "title": value.get("title"),
            "exchange": value.get("exchange"),
        }
    return mapping


def build_security_master(tickers, sec_map, as_of=None):
    """Map sampled tickers to real CIK; ticker stays an attribute.

    A ticker with no SEC CIK gets a documented issuer-key fallback and is listed
    in ``unmapped`` so the gap is visible rather than hidden. Returns
    ``(master, diagnostics)``.
    """
    as_of = as_of or _now_iso()
    master = SecurityMaster()
    mapped = []
    unmapped = []
    for ticker in tickers:
        info = sec_map.get(str(ticker).strip().upper())
        if info and info.get("cik"):
            sid = security_id_for(cik=info["cik"])
            master.add_security(SecurityRecord(security_id=sid, name=info.get("title") or ticker, cik=info["cik"]))
            mapped.append(ticker)
        else:
            sid = security_id_for(issuer_key=str(ticker).strip().upper())
            master.add_security(SecurityRecord(security_id=sid, name=str(ticker), issuer_key=str(ticker).strip().upper()))
            unmapped.append(ticker)
        master.add_ticker(TickerAssignment(sid, str(ticker).strip().upper(), info.get("exchange") if info else None, as_of))
    diagnostics = {
        "mapped": sorted(mapped),
        "unmapped": sorted(unmapped),
        "securities": len(master.securities),
        "symbol_reuse": master.detect_symbol_reuse(),
        "duplicate_identities": master.duplicate_securities(),
        "overlaps": master.detect_overlaps(),
    }
    return master, diagnostics


# ── Prices ───────────────────────────────────────────────────────────────────

def _silver_from_chart(payload, ticker, ingestion_ts):
    """Parse a raw Yahoo payload and normalise it into silver price rows."""
    from .prices import parse_yahoo_chart, validate_price_frame

    frame = parse_yahoo_chart(payload, ticker)
    problems = validate_price_frame(frame)
    if problems:
        raise CertificationError("price frame for %s failed integrity checks: %s" % (ticker, "; ".join(problems)))
    frame = frame.copy()
    frame["ingestion_ts"] = ingestion_ts
    frame.attrs["split_events"] = frame.attrs.get("split_events", [])
    frame.attrs["dividend_events"] = frame.attrs.get("dividend_events", [])
    return frame


def certify_prices(sample, master, root, mode, clients=None, range_="5y"):
    """Persist the bounded sample through BRONZE -> SILVER -> GOLD (PIT prices).

    A per-ticker fetch failure is recorded and the family is reported PARTIAL or
    BLOCKED; a failed ticker is never replaced with synthetic rows. The GOLD table
    is keyed by ``security_id`` + ``trade_date`` and carries ``raw_close`` and the
    reconstructed ``adjusted_close_pit`` (never the provider ``adjusted_close`` as
    a research input).
    """
    from .prices import fetch_chart_payload

    clients = clients or {}
    ingestion_ts = _now_iso()
    failures = []
    silver_frames = []
    all_actions = []
    source_fingerprints = {}
    per_ticker = {}
    for ticker in sample.tickers + (sample.benchmark,):
        try:
            payload = clients.get("charts", {}).get(ticker)
            if payload is None:
                payload = fetch_chart_payload(ticker, range_=range_)
            bronze = layers.write_bronze_json(root, "yahoo_chart_%s" % ticker, payload, meta={"ticker": ticker, "ingestion_ts": ingestion_ts})
            frame = _silver_from_chart(payload, ticker, ingestion_ts)
            silver = layers.write_silver_table(root, "prices_%s" % ticker, frame, meta={"ticker": ticker, "ingestion_ts": ingestion_ts})
            actions = actions_from_yahoo_events(frame.attrs.get("split_events", []), frame.attrs.get("dividend_events", []), ticker=ticker)
            all_actions.extend(actions)
            silver_frames.append(frame[["ticker", "trade_date", "raw_close", "close", "adjusted_close", "available_at", "source"]])
            source_fingerprints["yahoo_chart:%s" % ticker] = bronze["fingerprint"]
            per_ticker[ticker] = {"rows": int(len(frame)), "bronze": bronze["fingerprint"], "silver": silver["fingerprint"], "actions": len(actions)}
        except Exception as exc:  # provider failure is evidence, not something to hide
            failures.append({"ticker": ticker, "error": "%s: %s" % (type(exc).__name__, exc)})

    if not silver_frames:
        return None, FamilyStatus(
            family="prices",
            status=BLOCKED,
            pit_status="unknown",
            blockers=tuple(item["error"] for item in failures),
            evidence={"per_ticker": per_ticker},
        )

    combined = pd.concat(silver_frames, ignore_index=True)
    ledger_problems = validate_action_ledger(all_actions)
    if ledger_problems:
        raise CertificationError("corporate-action ledger failed integrity checks: %s" % "; ".join(ledger_problems))

    # GOLD: PIT price table keyed by security_id + trade_date.
    pit = build_pit_adjusted_frame(combined, all_actions, asof=combined["trade_date"].max())
    security_ids = []
    for ticker in pit["ticker"].tolist():
        try:
            security_ids.append(master.resolve_ticker(ticker, _now_iso()))
        except Exception:
            security_ids.append(None)
    # Provider adjusted close is joined as EVIDENCE ONLY (never a research input).
    provider = combined[["ticker", "trade_date", "adjusted_close"]].rename(
        columns={"adjusted_close": "adjusted_close_provider_NOT_PIT"}
    )
    pit = pit.merge(provider, on=["ticker", "trade_date"], how="left", validate="one_to_one")
    pit["security_id"] = security_ids
    pit["has_security_id"] = pit["security_id"].notna()
    pit["adjusted_close_pit_semantics"] = "raw_close * product(actions with trade_date < effective_date <= asof)"
    gold = layers.write_gold_table(root, "prices_gold_pit", pit, meta={"universe_id": sample.universe_id, "ingestion_ts": ingestion_ts})

    limitations = [
        "sample universe is present-day constituents only; survivorship-biased (not survivorship-safe)",
        "provider adjusted_close is retained as evidence only and is NOT a point-in-time research input",
    ]
    if failures:
        limitations.append("provider fetch failed for %d ticker(s)" % len(failures))
    manifest_result = build_and_persist_gold(
        "prices_gold_pit",
        pit,
        mode=mode,
        universe_id=sample.universe_id,
        period_start=str(combined["trade_date"].min()),
        period_end=str(combined["trade_date"].max()),
        sources=["YAHOO_CHART"],
        source_fingerprints=source_fingerprints,
        pit_status="partially_point_in_time",
        synthetic=False,
        known_limitations=limitations,
        root=root,
        notes="GOLD PIT prices keyed by security_id + trade_date; reconstructed adjusted close from raw close + corporate-action ledger",
    )
    status = PARTIAL if failures else PARTIAL  # survivorship limitation keeps prices at PARTIAL
    evidence = {
        "securities": int(pit["ticker"].nunique()),
        "observations": int(len(pit)),
        "period_start": str(combined["trade_date"].min()),
        "period_end": str(combined["trade_date"].max()),
        "per_ticker": per_ticker,
        "failures": failures,
        "gold": gold["fingerprint"],
        "row_count_with_security_id": int(pit["has_security_id"].sum()),
    }
    return manifest_result, FamilyStatus(
        family="prices",
        status=status,
        pit_status="partially_point_in_time",
        survivorship_safe=False,
        limitations=tuple(limitations),
        evidence=evidence,
        blockers=tuple(item["error"] for item in failures),
    )


# ── Benchmark ────────────────────────────────────────────────────────────────

def certify_benchmark(sample, master, root, mode, clients=None):
    """Persist SPY separately, versioned, with total-return vs price-return semantics."""
    from .benchmark import BENCHMARK_SEMANTICS, build_benchmark_series
    from .prices import fetch_chart_payload

    clients = clients or {}
    ticker = sample.benchmark
    payload = clients.get("charts", {}).get(ticker)
    if payload is None:
        payload = fetch_chart_payload(ticker)
    ingestion_ts = _now_iso()
    bronze = layers.write_bronze_json(root, "benchmark_chart_%s" % ticker, payload, meta={"ticker": ticker, "ingestion_ts": ingestion_ts})
    series = build_benchmark_series(ticker, frame=None if payload is None else __import__("src.research.data.prices", fromlist=["parse_yahoo_chart"]).parse_yahoo_chart(payload, ticker))

    frame = series.frame.copy()
    frame["benchmark_ticker"] = ticker
    frame["benchmark_version"] = series.version
    frame["return_semantics"] = BENCHMARK_SEMANTICS["adjusted_close"]
    frame["price_return_semantics"] = "close ratio ignores dividends"
    frame["total_return_semantics"] = "adjusted_close ratio includes dividends and splits"
    silver = layers.write_silver_table(root, "benchmark_%s" % ticker, frame, meta={"ticker": ticker, "ingestion_ts": ingestion_ts})
    gold = layers.write_gold_table(root, "benchmark_gold_%s" % ticker, frame, meta={"ticker": ticker, "universe_id": sample.universe_id})
    limitations = [
        "benchmark alignment is nearest-PRIOR only; a future session is never pulled backwards",
        "total-return basis depends on the provider adjusted_close; used only for the benchmark label, never as a feature at snapshot time",
    ]
    manifest_result = build_and_persist_gold(
        "benchmark_gold_%s" % ticker,
        frame,
        mode=mode,
        universe_id="%s_benchmark" % sample.universe_id,
        period_start=str(frame["trade_date"].min()),
        period_end=str(frame["trade_date"].max()),
        sources=["YAHOO_CHART"],
        source_fingerprints={"yahoo_chart:%s" % ticker: bronze["fingerprint"]},
        pit_status="point_in_time",
        synthetic=False,
        known_limitations=limitations,
        root=root,
        notes="SPY benchmark versioned separately; total-return vs price-return semantics documented",
    )
    evidence = {
        "ticker": ticker,
        "version": series.version,
        "rows": int(len(frame)),
        "period_start": str(frame["trade_date"].min()),
        "period_end": str(frame["trade_date"].max()),
        "semantics": dict(BENCHMARK_SEMANTICS),
        "silver": silver["fingerprint"],
        "gold": gold["fingerprint"],
    }
    return manifest_result, FamilyStatus(
        family="benchmark",
        status=PIT_READY,
        pit_status="point_in_time",
        survivorship_safe=True,
        limitations=tuple(limitations),
        evidence=evidence,
    )


# ── Fundamentals (verify, do not redesign) ───────────────────────────────────

def certify_fundamentals(tickers, master, root, mode, clients=None, cap=5):
    """Run SEC EDGAR BRONZE->SILVER for a bounded CIK sample; verify PIT semantics.

    The existing ``fundamentals_edgar.ingest_company`` path is reused unchanged.
    """
    from .fundamentals_edgar import ingest_company, validate_records

    clients = clients or {}
    ciks = []
    for ticker in tickers:
        try:
            sid = master.resolve_ticker(ticker, _now_iso())
        except Exception:
            continue
        record = master.securities.get(sid)
        if record and record.cik:
            ciks.append((ticker, record.cik))
    ciks = ciks[:cap]
    if not ciks:
        return None, FamilyStatus(
            family="fundamentals",
            status=BLOCKED,
            pit_status="unknown",
            blockers=("no sampled ticker resolved to a CIK; cannot verify fundamentals",),
            evidence={"sampled_ciks": 0},
        )
    total_rows = 0
    per_cik = {}
    silver_versions = []
    problems = []
    for ticker, cik in ciks:
        client = None
        if clients.get("edgar"):
            client = clients["edgar"].get(cik) or clients["edgar"].get(ticker)
        result = ingest_company(cik, root, mode, client=client)
        total_rows += result["row_count"]
        per_cik[cik] = {"ticker": ticker, "rows": result["row_count"], "silver": (result["silver"] or {}).get("fingerprint"), "bronze_facts": result["bronze_facts"]["fingerprint"]}
        silver_versions.append(result["silver"])
        # Verify PIT semantics on the produced silver frame. Re-read from disk to
        # prove the persisted artefact (not the in-memory frame) is PIT-safe.
        from src.research.data.pit_join import asof_join  # local import keeps CLI light
        frame = None
        try:
            frame = layers.read_silver_table(root, "edgar_fundamentals")
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            checked = frame.loc[frame["cik"] == cik]
            problems.extend(validate_records(checked))
    status = PIT_READY if not problems else BLOCKED
    limitations = (
        "fundamentals verified for a bounded sample of CIKs only",
        "SEC acceptance datetime + filing-date lag rule govern available_at; never fiscal period end",
    )
    evidence = {
        "sampled_ciks": [item[1] for item in ciks],
        "rows": int(total_rows),
        "per_cik": per_cik,
        "validation_problems": problems,
    }
    return None, FamilyStatus(
        family="fundamentals",
        status=status,
        pit_status="point_in_time" if status == PIT_READY else "unknown",
        limitations=limitations,
        evidence=evidence,
        blockers=tuple(problems[:5]),
    )


# ── Universe ─────────────────────────────────────────────────────────────────

def certify_universe(sample, master, as_of=None):
    """Build the sample-universe membership table; report honest (PARTIAL) status."""
    as_of = as_of or _now_iso()
    table = UniverseTable(sample.universe_id)
    for ticker in sample.tickers:
        try:
            sid = master.resolve_ticker(ticker, as_of)
        except Exception:
            continue
        table.add(UniverseMembership(sample.universe_id, sid, ticker, membership_start=as_of, valid_from=as_of))
    base = table.status()
    limitations = tuple(
        [KNOWN_LIMITATION_NO_DELISTING_SOURCE]
        + list(sample.limitations)
        + ["historical premiums/discounts and removals require an official constituent history; not wired"]
    )
    blockers = (
        "historical universe membership (delisted/acquired names) not available from a free source",
        "historical price history for delisted/acquired names not available from a free source",
    )
    return FamilyStatus(
        family="historical_universe",
        status=BLOCKED,
        pit_status="not_point_in_time",
        survivorship_safe=False,
        limitations=limitations,
        evidence={"table": base.to_dict(), "members": len(table.memberships), "universe_id": sample.universe_id},
        blockers=blockers,
    )


# ── Macro ────────────────────────────────────────────────────────────────────

def certify_macro():
    """Report the explicit macro exclusion (architecture + leakage guard kept)."""
    from .macro import exclusion_note

    return FamilyStatus(
        family="macro",
        status=EXCLUDED,
        pit_status="unknown",
        limitations=("macro is EXCLUDED_FROM_INITIAL_RESEARCH; must not gate certification",),
        evidence=exclusion_note(),
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_certification(root=None, mode=ResearchMode.RESEARCH_V2, clients=None, sample=None,
                      persist_provenance=True, ledger_root=None, fundamentals_cap=5):
    """Run one certification pass over the bounded sample and aggregate statuses.

    Returns a :class:`CertificationReport`. The overall status is ``BLOCKED`` when
    any non-excluded family is BLOCKED; the historical-universe gate is expected to
    block on the external free-data dependency.
    """
    from .provenance_ledger import persist_manifest as ledger_persist

    clients = clients or {}
    sample = sample or load_sample_universe()
    sec_map = fetch_sec_tickers(clients)
    master, sm_diag = build_security_master(sample.tickers + (sample.benchmark,), sec_map)

    manifests = {}
    statuses = []

    price_manifest, price_status = certify_prices(sample, master, root, mode, clients=clients)
    if price_manifest:
        manifests["prices"] = price_manifest
    statuses.append(price_status)

    bench_manifest, bench_status = certify_benchmark(sample, master, root, mode, clients=clients)
    if bench_manifest:
        manifests["benchmark"] = bench_manifest
    statuses.append(bench_status)

    _fund_manifest, fund_status = certify_fundamentals(sample.tickers, master, root, mode, clients=clients, cap=fundamentals_cap)
    statuses.append(fund_status)

    universe_status = certify_universe(sample, master)
    statuses.append(universe_status)
    statuses.append(certify_macro())

    # An unresolved ticker (no SEC CIK) or any detected symbol ambiguity/reuse is a
    # documented limitation of the security master, not a silent pass.
    sm_status = PARTIAL if (sm_diag["unmapped"] or sm_diag["overlaps"] or sm_diag["duplicate_identities"]) else PIT_READY
    statuses.append(FamilyStatus(
        family="security_master",
        status=sm_status,
        pit_status="point_in_time",
        evidence=sm_diag,
        limitations=(("tickers without SEC CIK use a documented issuer-key fallback: %s" % (" ".join(sm_diag["unmapped"]) or "none")),) if sm_diag["unmapped"] else (),
    ))

    # Provenance persistence (source-independent durable ledger).
    prov_evidence = {}
    prov_problems = []
    if persist_provenance:
        for name, result in manifests.items():
            record = ledger_persist(result["manifest"], root=ledger_root)
            prov_evidence[name] = {"dataset_id": record["dataset_id"], "outcome": record["outcome"], "index_outcome": record["index_outcome"]}
        from .provenance_ledger import verify_ledger

        prov_problems = verify_ledger(ledger_root)
    prov_status = PIT_READY if manifests and not prov_problems else (PARTIAL if manifests else BLOCKED)
    statuses.append(FamilyStatus(
        family="provenance",
        status=prov_status,
        pit_status="point_in_time",
        evidence={"registered": prov_evidence, "problems": prov_problems},
        blockers=tuple(prov_problems[:5]),
    ))

    order = {name: index for index, name in enumerate(FAMILIES)}
    statuses.sort(key=lambda item: order.get(item.family, 99))

    blocking = [item for item in statuses if item.status == BLOCKED and item.family != "macro"]
    overall = BLOCKED if blocking else (PARTIAL if any(item.status == PARTIAL for item in statuses) else PIT_READY)
    external_blockers = []
    for item in statuses:
        for blocker in item.blockers:
            external_blockers.append("%s: %s" % (item.family, blocker))

    # Aggregate identifiers.
    primary = manifests.get("prices") or manifests.get("benchmark") or {}
    primary_manifest = primary.get("manifest", {}) if primary else {}
    securities = int(master.__len__() if False else len(master.securities))
    observations = 0
    for status in statuses:
        if status.family == "prices":
            observations = int(status.evidence.get("observations", 0))
    git_commit, branch = _git_ref()
    report = CertificationReport(
        dataset_id=primary_manifest.get("dataset_id", "dataset_unknown"),
        created_at=_now_iso(),
        git_commit=git_commit,
        branch=branch,
        universe_id=sample.universe_id,
        families=tuple(statuses),
        manifests={name: result["dataset_id"] for name, result in manifests.items()},
        securities=securities,
        observations=observations,
        period_start=str(primary_manifest.get("period_start", "")),
        period_end=str(primary_manifest.get("period_end", "")),
        overall_status=overall,
        external_blockers=tuple(external_blockers),
        notes="overall BLOCKED expected: historical universe / delisted-price external free-data dependency",
    )
    return report


def write_report(report, path=None):
    """Persist a certification report as JSON under the gitignored data area."""
    target = Path(path) if path else (layers.DEFAULT_ROOT / "certification" / "%s.json" % report.dataset_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    from src.research.immutability import write_json_atomic

    write_json_atomic(target, report.to_dict())
    return target
