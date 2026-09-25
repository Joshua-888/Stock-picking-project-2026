"""WP2B finalize: SILVER -> GOLD panel, DatasetManifest, provenance, guardrail.

Reads the silver tables produced by ``wp2b_build_live_dataset.py`` (no refetch),
rebuilds the PIT monthly panel, writes the GOLD table, builds and persists the
real WP1 ``DatasetManifest`` plus a provenance-ledger entry, and runs the
survivorship guardrail proof.

Research mode only; refuses synthetic data; performs no feature engineering and
no scoring.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
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
from src.research.modes import ResearchMode

BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp2b_live"
GOLD_PANEL = wp2b.DATASET_NAME

# Famous failed / removed firms that the source must still show as members.
KNOWN_CONSTITUENTS = [
    ("LEH", ["2007-06-30", "2008-01-31"]),
    ("WB", ["2007-06-30", "2008-06-30"]),
    ("AIG", ["2007-06-30", "2008-06-30"]),
    ("GM", ["2007-06-30", "2008-06-30"]),
]


def _price_bytes(frame):
    return len(frame) if frame is not None else 0


def load_silver():
    """Read the EXACT silver versions the build declared.

    ``read_silver_table`` defaults to the lexicographically-largest version name,
    which is not a valid 'newest' rule for content-addressed version ids and once
    selected a stale empty actions table. Pinning to the declared build versions
    removes that ambiguity
    """
    declared = json.loads((ARTIFACT_DIR / "layer_records.json").read_text())
    prices = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_prices_silver", version=declared["silver_prices"])
    actions = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_actions_silver", version=declared["silver_actions"])
    membership = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_membership_silver", version=declared["silver_membership"])
    print("LOAD_SILVER prices=%s actions=%s membership=%s" % (
        declared["silver_prices"], declared["silver_actions"], declared["silver_membership"]))
    return prices, actions, membership


def windows_from_frame(membership):
    windows = []
    for row in membership.to_dict(orient="records"):
        end = row.get("membership_end")
        if isinstance(end, float) and pd.isna(end):
            end = None
        if end in ("", "None", "nan", "NaT"):
            end = None
        windows.append(wp2b.MembershipWindow(
            security_id=row["security_id"], ticker=row.get("ticker"),
            membership_start=str(row["membership_start"])[:10],
            membership_end=str(end)[:10] if end else None,
            start_known=bool(row.get("start_known", True)),
            research_eligible=bool(row.get("research_eligible", True)),
            unverifiable_before=bool(row.get("unverifiable_before", False)),
        ))
    return windows


def build_pit_panel(windows, prices, actions):
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
            rows.extend(wp2b.monthly_panel_for_security(window, raw_frame, parsed, months))
    return pd.DataFrame(rows, columns=["security_id", "ticker", "snapshot_date", "membership_start",
                                       "membership_end", "research_eligible", "available_at", "price_date", "raw_close",
                                       "adjusted_close_pit", "has_price"])


def _is_possible_window(window):
    """A window is impossible when its removal precedes (or equals) its start.

    The listing guard can clamp an assumed start forward past an early removal;
    such a window represents no tradable membership span and must be dropped
    deterministically rather than silently repaired.
    """
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


def main():
    prices, actions, membership = load_silver()
    windows_all = windows_from_frame(membership)
    windows = [window for window in windows_all if _is_possible_window(window)]
    dropped_impossible = len(windows_all) - len(windows)
    print("WINDOWS total=%d usable=%d dropped_impossible=%d" % (len(windows_all), len(windows), dropped_impossible))
    panel = build_pit_panel(windows, prices, actions)
    removed_ids = {window.security_id for window in windows if window.membership_end}

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
        "name/exchange/date evidence with an explicit unresolved-identities artifact",
        "Wikipedia change-table coverage is uneven before the mid-2000s; several "
        "well-known failures (Bear Stearns, Enron, WaMu, Circuit City) are absent and "
        "are therefore not provable as members from this source",
        "proven failed/removed firms in this build: %s" % (", ".join(proven) or "none"),
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
        notes="WP2B survivorship-controlled S&P 500 PIT monthly panel",
    )
    manifest = persisted["manifest"]
    ledger = provenance_ledger.persist_manifest(manifest, extra={
        "build_script": "scripts/research_v2/wp2b_build_live_dataset.py",
        "finalize_script": "scripts/research_v2/wp2b_finalize_gold.py",
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
    })

    proof = survivorship_proof(windows, prices)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "survivorship_proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "gold_summary.json").write_text(json.dumps({
        "dataset_id": manifest["dataset_id"], "dataset_fingerprint": manifest["dataset_fingerprint"],
        "rows": int(len(panel)), "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
        "ledger_outcome": ledger["outcome"], "guard_clean": proof["survivorship_guard_clean"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GOLD_DONE rows=%d securities=%d dataset_id=%s" % (len(panel), panel["security_id"].nunique(), manifest["dataset_id"]))
    print("GUARD_CLEAN=%s removed=%d proven_failed=%s" % (proof["survivorship_guard_clean"], len(removed_ids), proven))


if __name__ == "__main__":
    main()
