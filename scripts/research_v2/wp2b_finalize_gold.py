"""WP2C finalize: SILVER -> GOLD panel, censoring diagnostics, DatasetManifest, ledger.

Reads the EXACT silver versions the build declared (never a lexicographic guess),
rebuilds the PIT monthly panel with per-window symbols and target-observability
flags, writes a SELF-CONTAINED gold table, emits censoring/selection diagnostics,
and persists the WP1 ``DatasetManifest`` plus a provenance-ledger entry bound to
the producing git commit.

Research mode only; refuses synthetic data; no feature engineering, no scoring.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers, wp2b_eodhd as wp2b
from src.research.data import provenance_ledger
from src.research.data.manifesting import build_and_persist_gold
from src.research.data.survivorship_guard import run_guardrails
from src.research.data.universe import UniverseMembership, UniverseTable
from src.research.fingerprints import fingerprint_obj
from src.research.modes import ResearchMode, current_git_commit

BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp2b_live"
GOLD_PANEL = wp2b.DATASET_NAME
PROVENANCE_ROOT = ROOT / "provenance"

# Famous failed / removed firms the source must still show as members.
KNOWN_CONSTITUENTS = [
    ("LEH", ["2007-06-30", "2008-01-31"]),
    ("WB", ["2007-06-30", "2008-06-30"]),
    ("AIG", ["2007-06-30", "2008-06-30"]),
]


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return None if text in ("", "None", "nan", "NaT") else text


def load_silver():
    """Read the EXACT silver versions the build declared (version-pinned)."""
    declared = json.loads((ARTIFACT_DIR / "layer_records.json").read_text())
    prices = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_prices_silver", version=declared["silver_prices"])
    actions = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_actions_silver", version=declared["silver_actions"])
    membership = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_membership_silver", version=declared["silver_membership"])
    print("LOAD_SILVER prices=%s actions=%s membership=%s" % (
        declared["silver_prices"], declared["silver_actions"], declared["silver_membership"]))
    return prices, actions, membership, declared


def windows_from_frame(membership):
    windows = []
    for row in membership.to_dict(orient="records"):
        windows.append(wp2b.MembershipWindow(
            security_id=row["security_id"], ticker=row.get("ticker"),
            membership_start=str(row["membership_start"])[:10],
            membership_end=_clean(row.get("membership_end")),
            start_known=bool(row.get("start_known", True)),
            research_eligible=bool(row.get("research_eligible", True)),
            unverifiable_before=bool(row.get("unverifiable_before", False)),
            symbol=_clean(row.get("symbol")),
            resolution_method=_clean(row.get("resolution_method")),
            unresolved_reason=_clean(row.get("unresolved_reason")),
        ))
    return windows


def build_pit_panel(windows, prices, actions, price_by_symbol):
    months = wp2b.month_ends(wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    price_by_security = {security: frame for security, frame in prices.groupby("security_id")}
    action_by_security = {security: frame for security, frame in actions.groupby("security_id")}
    by_security = {}
    for window in windows:
        by_security.setdefault(window.security_id, []).append(window)
    rows = []
    for security, items in by_security.items():
        raw_frame = price_by_security.get(security)
        action_frame = action_by_security.get(security)
        parsed = wp2b.actions_from_eodhd_rows(
            action_frame.to_dict("records") if action_frame is not None else [], security)
        for window in sorted(items, key=lambda item: item.membership_start):
            rows.extend(wp2b.monthly_panel_for_security(
                window, raw_frame, parsed, months, window_end=wp2b.RESEARCH_WINDOW_END))
    return pd.DataFrame(rows, columns=list(wp2b.PANEL_COLUMNS))


def _is_possible_window(window):
    """A window is impossible when its removal precedes (or equals) its start."""
    if not window.membership_end:
        return True
    start = str(window.membership_start)[:10]
    end = str(window.membership_end)[:10]
    return bool(start) and end > start


def build_universe_table(windows):
    table = UniverseTable(universe_id=wp2b.UNIVERSE_ID)
    for window in windows:
        if not _is_possible_window(window):
            continue
        table.add(UniverseMembership(
            universe_id=wp2b.UNIVERSE_ID, security_id=window.security_id,
            ticker=window.security_id,
            membership_start=window.membership_start, membership_end=window.membership_end,
        ))
    return table


def survivorship_proof(windows, prices):
    table = build_universe_table(windows)
    findings = run_guardrails(table)
    proof = []
    for ticker, dates in KNOWN_CONSTITUENTS:
        present = [day for day in dates if table.is_member(ticker, day, available_only=False)]
        proof.append({"ticker": ticker, "queried": dates, "member_on": present,
                      "proven_member": bool(present)})
    removed = sorted({window.security_id for window in windows if window.membership_end})
    price_ids = set(prices["security_id"].unique().tolist()) if len(prices) else set()
    return {
        "guardrail_findings": [finding.to_dict() for finding in findings],
        "failures_have_exits": bool(removed),
        "removed_constituent_count": len(removed),
        "famous_failed_firms": proof,
        "survivorship_guard_clean": not findings,
        "removed_ids_with_prices": sorted(set(removed) & price_ids)[:20],
    }


def censoring_diagnostics(panel, windows):
    """Phase C: understand WHERE the label space is censored (selection bias).

    Computed from the panel ONLY. Sector and market-cap buckets are NOT available
    from the EODHD symbol list, so they are reported as explicitly unavailable
    rather than fabricated.
    """
    total = int(len(panel))
    censored = int(panel["target_censored"].sum()) if total else 0
    observable = total - censored
    by_year = defaultdict(lambda: {"total": 0, "censored": 0})
    for row in panel.to_dict("records"):
        year = str(row["snapshot_date"])[:4]
        by_year[year]["total"] += 1
        if bool(row["target_censored"]):
            by_year[year]["censored"] += 1
    by_year = {year: {"total": value["total"], "censored": value["censored"],
                      "censoring_percent": round(100.0 * value["censored"] / max(1, value["total"]), 2)}
               for year, value in sorted(by_year.items())}
    removed = {window.security_id for window in windows if window.membership_end}
    removed_rows = panel[panel["security_id"].isin(removed)] if total else panel
    removed_censored = int(removed_rows["target_censored"].sum()) if len(removed_rows) else 0
    reasons = Counter(panel["target_censor_reason"].dropna().tolist()) if total else Counter()
    # Pre-T observable vs censored comparison (price level only; no fundamentals yet).
    comparison = {}
    if total and panel["has_price"].any():
        priced = panel[panel["has_price"]]
        for flag, label in ((False, "observable"), (True, "censored")):
            subset = priced[priced["target_censored"] == flag]["adjusted_close_pit"].dropna()
            comparison[label] = {
                "count": int(len(subset)),
                "pre_t_price_median": round(float(subset.median()), 4) if len(subset) else None,
            }
    return {
        "total_candidate_labels": total,
        "observable_labels": observable,
        "censored_labels": censored,
        "censoring_percent": round(100.0 * censored / total, 2) if total else 0.0,
        "by_calendar_year": by_year,
        "censor_reason_counts": dict(sorted(reasons.items())),
        "censored_among_removed_constituents": removed_censored,
        "removed_constituent_rows": int(len(removed_rows)),
        "pre_t_observable_vs_censored": comparison,
        "by_sector": None,
        "by_sector_unavailable_reason": "EODHD exchange-symbol-list exposes no sector field",
        "by_market_cap": None,
        "by_market_cap_unavailable_reason": "EODHD symbol list exposes no market-cap field",
        "note": "selection-bias documentation, not a remedy; censored rows stay in the panel flagged",
    }


def security_master_artifact(windows, membership):
    """Explicit SECURITY-MASTER VERSION (E2): the resolved identity mapping."""
    records = []
    for row in membership.to_dict("records"):
        records.append({
            "security_id": row["security_id"],
            "symbol": _clean(row.get("symbol")),
            "resolution_method": _clean(row.get("resolution_method")),
            "unresolved_reason": _clean(row.get("unresolved_reason")),
            "membership_start": str(row["membership_start"])[:10],
            "membership_end": _clean(row.get("membership_end")),
        })
    payload = {
        "security_master_version": fingerprint_obj(records),
        "securities": len({row["security_id"] for row in records}),
        "windows": len(records),
        "unresolved_windows": sum(1 for row in records if not row["symbol"]),
        "entries": records,
    }
    write_canonical(ARTIFACT_DIR / "security_master.json", payload)
    return payload


def universe_artifact(windows, diagnostics):
    """Explicit UNIVERSE VERSION (E3): the membership definition.

    ``_test`` data (impossible or non-overlapping windows) is excluded BEFORE
    fingerprinting so the version identifies exactly the research universe.
    """
    usable = [window for window in windows if _is_possible_window(window)]
    definition = [
        {"security_id": window.security_id, "symbol": window.symbol,
         "membership_start": str(window.membership_start)[:10],
         "membership_end": _clean(window.membership_end),
         "research_eligible": bool(window.research_eligible)}
        for window in sorted(usable, key=lambda item: (item.security_id, item.membership_start))
    ]
    payload = {
        "universe_id": wp2b.UNIVERSE_ID,
        "universe_version": fingerprint_obj(definition),
        "period_start": wp2b.RESEARCH_WINDOW_START,
        "period_end": wp2b.RESEARCH_WINDOW_END,
        "membership_availability_rule": wp2b.MEMBERSHIP_AVAILABILITY_RULE,
        "membership_source_limitations": MEMBERSHIP_SOURCE_LIMITATIONS,
        "windows": len(definition),
        "securities": len({row["security_id"] for row in definition}),
        "diagnostics": diagnostics,
    }
    write_canonical(ARTIFACT_DIR / "universe.json", payload)
    return payload


MEMBERSHIP_SOURCE_LIMITATIONS = [
    "Wikipedia change table omits several well-known removals (e.g. Bear Stearns BSC, "
    "Enron, Circuit City, WaMu, old GM); these are NOT representable as members from "
    "this source and are recorded as a membership-source gap, never fabricated",
    "Wikipedia publishes only effective dates; membership availability uses the "
    "effective date itself (conservative)",
]


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    prices, actions, membership, declared = load_silver()
    windows_all = windows_from_frame(membership)
    windows = [window for window in windows_all if _is_possible_window(window)]
    dropped_impossible = len(windows_all) - len(windows)
    print("WINDOWS total=%d usable=%d dropped_impossible=%d" % (len(windows_all), len(windows), dropped_impossible))

    price_by_symbol = {symbol: frame for symbol, frame in prices.groupby("ticker")}
    panel = build_pit_panel(windows, prices, actions, price_by_symbol)
    removed_ids = {window.security_id for window in windows if window.membership_end}

    diagnostics = wp2b.membership_diagnostics(windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    security_master = security_master_artifact(windows, membership)
    universe = universe_artifact(windows, diagnostics)
    censoring = censoring_diagnostics(panel, windows)

    source_versions = {
        "silver_prices": declared["silver_prices"],
        "silver_actions": declared["silver_actions"],
        "silver_membership": declared["silver_membership"],
    }
    source_fingerprints = {
        "eodhd_symbol_list_active": layers.source_fingerprint(BRONZE_ROOT, "wp2b_eodhd_symbol_list_active"),
        "eodhd_symbol_list_delisted": layers.source_fingerprint(BRONZE_ROOT, "wp2b_eodhd_symbol_list_delisted"),
        "wikipedia_sp500_changes": layers.source_fingerprint(BRONZE_ROOT, "wp2b_wikipedia_sp500_changes"),
        "wikipedia_sp500_current": layers.source_fingerprint(BRONZE_ROOT, "wp2b_wikipedia_sp500_current"),
    }

    proven = [item["ticker"] for item in survivorship_proof(windows, prices)["famous_failed_firms"] if item["proven_member"]]
    limitations = [
        wp2b.MEMBERSHIP_AVAILABILITY_RULE,
        wp2b.TERMINAL_RETURN_LIMITATION,
        "EODHD exposes no permanent security id and no CIK; identity is resolved from "
        "name/exchange/date evidence per membership window with an explicit "
        "unresolved-identities artifact (never guessed)",
    ] + MEMBERSHIP_SOURCE_LIMITATIONS + [
        "proven failed/removed firms in this build: %s" % (", ".join(proven) or "none"),
        "target observability is flagged per row (target_observable/target_censored/"
        "target_censor_reason); censored rows MUST be excluded from supervised labels",
    ]

    persisted = build_and_persist_gold(
        GOLD_PANEL, panel,
        mode=ResearchMode.RESEARCH_V2,
        universe_id=wp2b.UNIVERSE_ID,
        period_start=wp2b.RESEARCH_WINDOW_START,
        period_end=wp2b.RESEARCH_WINDOW_END,
        sources=["EODHD", "WIKIPEDIA_SP500"],
        source_fingerprints=source_fingerprints,
        pit_status="partially_point_in_time",
        synthetic=False,
        known_limitations=limitations,
        root=BRONZE_ROOT,
        notes="WP2C survivorship-controlled S&P 500 PIT monthly panel (censoring-aware)",
        source_versions=source_versions,
        security_master_version=security_master["security_master_version"],
        universe_version=universe["universe_version"],
        censoring_statistics=censoring,
    )
    manifest = persisted["manifest"]
    ledger = provenance_ledger.persist_manifest(manifest, root=PROVENANCE_ROOT, extra={
        "build_script": "scripts/research_v2/wp2b_build_live_dataset.py",
        "finalize_script": "scripts/research_v2/wp2b_finalize_gold.py",
        "producing_commit": current_git_commit() or "unknown",
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
        "universe_version": universe["universe_version"],
        "security_master_version": security_master["security_master_version"],
    })

    proof = survivorship_proof(windows, prices)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_canonical(ARTIFACT_DIR / "survivorship_proof.json", proof)
    write_canonical(ARTIFACT_DIR / "manifest.json", manifest)
    write_canonical(ARTIFACT_DIR / "censoring_diagnostics.json", censoring)
    write_canonical(ARTIFACT_DIR / "gold_summary.json", {
        "dataset_id": manifest["dataset_id"], "dataset_fingerprint": manifest["dataset_fingerprint"],
        "rows": int(len(panel)), "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
        "observable_labels": censoring["observable_labels"],
        "censored_labels": censoring["censored_labels"],
        "censoring_percent": censoring["censoring_percent"],
        "security_master_version": security_master["security_master_version"],
        "universe_version": universe["universe_version"],
        "ledger_outcome": ledger["outcome"], "guard_clean": proof["survivorship_guard_clean"],
    })
    print("GOLD_DONE rows=%d securities=%d dataset_id=%s" % (len(panel), panel["security_id"].nunique(), manifest["dataset_id"]))
    print("CENSORING total=%d observable=%d censored=%d (%.2f%%)" % (
        censoring["total_candidate_labels"], censoring["observable_labels"],
        censoring["censored_labels"], censoring["censoring_percent"]))
    print("GUARD_CLEAN=%s removed=%d proven_failed=%s" % (proof["survivorship_guard_clean"], len(removed_ids), proven))


if __name__ == "__main__":
    main()
