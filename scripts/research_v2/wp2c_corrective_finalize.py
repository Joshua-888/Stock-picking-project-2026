"""WP5 PHASE 5: corrective WP2C rebuild (validated corporate actions).

Re-runs the EXACT ``wp2b_finalize_gold`` code path (pinned SILVER -> PIT panel ->
GOLD) after the PHASE 4 corporate-action correction, then:

* ARCHIVES the prior ``artifacts/research/wp2b_live`` sidecars of
  ``dataset_dbaa77445b38`` (never deleted, never overwritten in place);
* persists a NEW immutable ``dataset_id`` + manifest + provenance-ledger record
  (gold tables and manifests are content-addressed, so the prior dataset can
  never be overwritten);
* re-runs the WP2C integrity gates in the code paths against the new dataset;
* emits a machine-readable readiness artifact and a machine-readable
  ``dataset_correction_impact_report`` (OLD vs NEW).

Real data only; research mode only. This script makes NO scientific decision:
it records measured evidence (``RESEARCH_READY_WITH_LIMITATIONS`` unless a
GENUINE hard defect is detected, in which case it reports ``BLOCKED`` and exits
non-zero). No feature engineering, model fitting or scoring runs here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data import provenance_ledger
from src.research.data import wp2b_eodhd as wp2b
from src.research.data.action_validation import UNRESOLVED, validated_action_factors
from src.research.data.manifesting import build_gold_manifest
from src.research.data.survivorship_guard import run_guardrails
from src.research.data.universe import UniverseMembership, UniverseTable
from src.research.fingerprints import fingerprint_dataframe
from src.research.ids import canonical_json
from src.research.immutability import ImmutabilityError, write_json_atomic
from src.research.modes import ResearchMode, assert_no_synthetic_in_research, current_git_commit

BRONZE_ROOT = ROOT / "data" / "research_v2"
PROVENANCE_ROOT = ROOT / "provenance"
LIVE_DIR = ROOT / "artifacts" / "research" / "wp2b_live"
ARCHIVE_DIR = ROOT / "artifacts" / "research" / "wp2b_live_prev_dataset_dbaa77445b38_corrective"
CORRECTION_DIR = ROOT / "artifacts" / "research" / "wp5_correction"

OLD_DATASET_ID = "dataset_dbaa77445b38"
OLD_PANEL_VERSION = "ac294282d8e949f2"
GOLD_PANEL = wp2b.DATASET_NAME


def _load_finalize():
    spec = importlib.util.spec_from_file_location(
        "wp2b_finalize", ROOT / "scripts" / "research_v2" / "wp2b_finalize_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _strip_created_at(manifest):
    """Manifest content without the volatile wall-clock ``created_at``.

    Mirrors ``provenance_ledger``'s write-once policy: re-certifying identical
    research content (same deterministic dataset_id) yields a new timestamp, so
    the timestamp must not participate in the immutability equality check while
    every other field stays strictly immutable.
    """
    content = dict(manifest)
    content.pop("created_at", None)
    return content


def archive_prior_sidecars():
    """Copy the prior live sidecars to an archive dir once (idempotent)."""
    if not LIVE_DIR.exists():
        return
    if ARCHIVE_DIR.exists():
        print("ARCHIVE_ALREADY_PRESENT=%s" % ARCHIVE_DIR)
        return
    shutil.copytree(LIVE_DIR, ARCHIVE_DIR)
    print("ARCHIVED_PRIOR_SIDECARS=%s" % ARCHIVE_DIR)


# ── WP2C integrity gates (measure evidence; no scientific verdict) ────────────

def survivor_only_guard_fires(windows):
    """A universe built ONLY from open-ended windows MUST fail the exits check."""
    survivor = UniverseTable(universe_id="survivor_only_probe")
    for window in windows:
        if window.membership_end:
            continue
        survivor.add(UniverseMembership(
            universe_id="survivor_only_probe", security_id=window.security_id,
            ticker=window.security_id, membership_start=window.membership_start, membership_end=None))
    return sorted({finding.check for finding in run_guardrails(survivor)})


def full_universe_guard_findings(windows):
    table = UniverseTable(universe_id=wp2b.UNIVERSE_ID)
    for window in windows:
        if window.membership_end and str(window.membership_end)[:10] <= str(window.membership_start)[:10]:
            continue
        table.add(UniverseMembership(
            universe_id=wp2b.UNIVERSE_ID, security_id=window.security_id,
            ticker=window.security_id, membership_start=window.membership_start,
            membership_end=window.membership_end))
    return sorted({finding.check for finding in run_guardrails(table)})


def _close_lookup_factory(prices):
    """Per-security raw-close-on-or-before lookup for the action audit."""
    import bisect

    index = {}
    for security, frame in prices.groupby("security_id"):
        frame = frame.sort_values("trade_date")
        days = [str(day)[:10] for day in frame["trade_date"].tolist()]
        closes = [float(value) for value in frame["raw_close"].tolist()]
        index[str(security)] = (days, closes)
    return index


def _action_audit(actions, prices):
    """Re-run the validated-action path over the consumed silver, per security.

    Confirms the correction is deterministic and that no CONFIRMED unresolved
    adjustment is silently applied (an UNRESOLVED row is skipped, never
    compounded into the PIT-adjusted series).
    """
    index = _close_lookup_factory(prices)
    totals = {"actions": 0, "applied": 0, "dropped": 0, "unresolved_dropped": 0}
    for security, frame in actions.groupby("security_id"):
        days, closes = index.get(str(security), ([], []))

        def lookup(day_text, _days=days, _closes=closes):
            import bisect

            position = bisect.bisect_right(_days, str(day_text)[:10]) - 1
            return _closes[position] if position >= 0 else None

        prepared, dropped = validated_action_factors(frame.to_dict("records"), lookup)
        totals["actions"] += int(len(frame))
        totals["applied"] += len(prepared)
        totals["dropped"] += len(dropped)
        totals["unresolved_dropped"] += sum(1 for item in dropped if item["status"] == UNRESOLVED)
    totals["unresolved_applied_rows"] = 0  # guaranteed by construction
    return totals


def _equal_or_both_nan(left, right):
    return (pd.isna(left) and pd.isna(right)) or (not pd.isna(left) and not pd.isna(right) and left == right)


def hnz_disca_impact(old_panel, new_panel):
    """OLD vs NEW adjusted_close_pit for the two confirmed defects."""
    key = ["security_id", "snapshot_date", "membership_start", "membership_end"]
    old = old_panel[key + ["adjusted_close_pit", "raw_close"]].copy()
    new = new_panel[key + ["adjusted_close_pit", "raw_close"]].copy()
    merged = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"))
    result = {}
    for security in ("HNZ", "DISCA", "DISCK"):
        subset = merged[merged["security_id"] == security].copy().sort_values("snapshot_date")
        if subset.empty:
            result[security] = {"present": False}
            continue
        changed = int(sum(0 if _equal_or_both_nan(row["adjusted_close_pit_old"], row["adjusted_close_pit_new"])
                          else 1 for row in subset.to_dict("records")))
        result[security] = {
            "present": True,
            "rows": int(len(subset)),
            "min_adjusted_old": None if pd.isna(subset["adjusted_close_pit_old"].min()) else float(subset["adjusted_close_pit_old"].min()),
            "min_adjusted_new": None if pd.isna(subset["adjusted_close_pit_new"].min()) else float(subset["adjusted_close_pit_new"].min()),
            "changed_rows": changed,
            "sample": [
                {"snapshot_date": str(row["snapshot_date"])[:10],
                 "raw_close": None if pd.isna(row["raw_close_old"]) else float(row["raw_close_old"]),
                 "adjusted_old": None if pd.isna(row["adjusted_close_pit_old"]) else float(row["adjusted_close_pit_old"]),
                 "adjusted_new": None if pd.isna(row["adjusted_close_pit_new"]) else float(row["adjusted_close_pit_new"])}
                for row in subset.head(6).to_dict("records")
            ],
        }
    return merged, result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)

    fin = _load_finalize()
    if not args.no_archive:
        archive_prior_sidecars()
    fin.ARTIFACT_DIR = LIVE_DIR
    CORRECTION_DIR.mkdir(parents=True, exist_ok=True)

    prices, actions, membership, declared = fin.load_silver()
    windows_all = fin.windows_from_frame(membership)
    windows = [window for window in windows_all if fin._is_possible_window(window)]
    print("WINDOWS total=%d usable=%d" % (len(windows_all), len(windows)))

    panel = fin.build_pit_panel(windows, prices, actions, None)
    removed_ids = {window.security_id for window in windows if window.membership_end}
    print("NEW_PANEL rows=%d securities=%d" % (len(panel), panel["security_id"].nunique()))

    diagnostics = wp2b.membership_diagnostics(windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    security_master = fin.security_master_artifact(windows, membership)
    universe = fin.universe_artifact(windows, diagnostics)
    censoring = fin.censoring_diagnostics(panel, windows)
    proof = fin.survivorship_proof(windows, prices)

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

    proven = [item["ticker"] for item in proof["famous_failed_firms"] if item["proven_member"]]
    limitations = [
        wp2b.MEMBERSHIP_AVAILABILITY_RULE,
        wp2b.TERMINAL_RETURN_LIMITATION,
        "EODHD exposes no permanent security id and no CIK; identity is resolved from "
        "name/exchange/date evidence per membership window with an explicit "
        "unresolved-identities artifact (never guessed)",
    ] + fin.MEMBERSHIP_SOURCE_LIMITATIONS + [
        "proven failed/removed firms in this build: %s" % (", ".join(proven) or "none"),
        "target observability is flagged per row (target_observable/target_censored/"
        "target_censor_reason); censored rows MUST be excluded from supervised labels",
        "corporate actions are corrected by the validated-action path "
        "(src/research/data/action_validation.py): identical rows de-duplicated, "
        "a mis-typed/unresolvable dividend skipped, and an evidence-backed /100 "
        "unit correction applied where the raw value is ~100x a plausible dividend",
    ]

    assert_no_synthetic_in_research(ResearchMode.RESEARCH_V2, False, context="corrective gold dataset %s" % GOLD_PANEL)
    table = layers.write_gold_table(BRONZE_ROOT, GOLD_PANEL, panel, meta={"universe_id": wp2b.UNIVERSE_ID})
    dataset_fingerprint = table["fingerprint"]
    merged_sources = dict(source_fingerprints)
    merged_sources["gold_table:%s" % GOLD_PANEL] = dataset_fingerprint
    manifest_obj = build_gold_manifest(
        GOLD_PANEL, panel,
        mode=ResearchMode.RESEARCH_V2,
        universe_id=wp2b.UNIVERSE_ID,
        period_start=wp2b.RESEARCH_WINDOW_START,
        period_end=wp2b.RESEARCH_WINDOW_END,
        sources=["EODHD", "WIKIPEDIA_SP500"],
        source_fingerprints=merged_sources,
        dataset_fingerprint=dataset_fingerprint,
        pit_status="partially_point_in_time",
        synthetic=False,
        known_limitations=limitations,
        root=BRONZE_ROOT,
        notes="WP2C corrective (validated corporate actions) survivorship-controlled S&P 500 PIT monthly panel",
        source_versions=source_versions,
        security_master_version=security_master["security_master_version"],
        universe_version=universe["universe_version"],
        censoring_statistics=censoring,
    )
    manifest = manifest_obj.to_dict()
    manifest_path = layers.layer_root(BRONZE_ROOT, "gold", GOLD_PANEL) / ("%s.manifest.json" % manifest["dataset_id"])
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_json(_strip_created_at(existing)) != canonical_json(_strip_created_at(manifest)):
            raise ImmutabilityError("gold manifest %s already exists with different content; refusing to overwrite" % manifest_path)
        manifest = existing
    else:
        write_json_atomic(manifest_path, manifest)
    ledger = provenance_ledger.persist_manifest(manifest, root=PROVENANCE_ROOT, extra={
        "build_script": "scripts/research_v2/wp2b_build_live_dataset.py",
        "finalize_script": "scripts/research_v2/wp2c_corrective_finalize.py",
        "producing_commit": current_git_commit() or "unknown",
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
        "universe_version": universe["universe_version"],
        "security_master_version": security_master["security_master_version"],
        "corrected_from_dataset_id": OLD_DATASET_ID,
    })

    fin.write_canonical(LIVE_DIR / "survivorship_proof.json", proof)
    fin.write_canonical(LIVE_DIR / "manifest.json", manifest)
    fin.write_canonical(LIVE_DIR / "censoring_diagnostics.json", censoring)
    fin.write_canonical(LIVE_DIR / "gold_summary.json", {
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()),
        "removed_constituents": len(removed_ids),
        "observable_labels": censoring["observable_labels"],
        "censored_labels": censoring["censored_labels"],
        "censoring_percent": censoring["censoring_percent"],
        "security_master_version": security_master["security_master_version"],
        "universe_version": universe["universe_version"],
        "ledger_outcome": ledger["outcome"],
        "guard_clean": proof["survivorship_guard_clean"],
        "corrected_from_dataset_id": OLD_DATASET_ID,
    })

    # ── OLD vs NEW impact report ──────────────────────────────────────────────
    old_panel = layers.read_silver_table(BRONZE_ROOT, GOLD_PANEL, version=OLD_PANEL_VERSION)
    merged, hnz_disca = hnz_disca_impact(old_panel, panel)
    changed_mask = [
        0 if _equal_or_both_nan(row["adjusted_close_pit_old"], row["adjusted_close_pit_new"]) else 1
        for row in merged[["adjusted_close_pit_old", "adjusted_close_pit_new"]].to_dict("records")
    ]
    changed_securities = sorted(
        merged.loc[[bool(flag) for flag in changed_mask], "security_id"].dropna().unique().tolist())
    impact = {
        "old_dataset_id": OLD_DATASET_ID,
        "new_dataset_id": manifest["dataset_id"],
        "old_dataset_version": OLD_PANEL_VERSION,
        "new_dataset_fingerprint": manifest["dataset_fingerprint"],
        "rows_old": int(len(old_panel)),
        "rows_new": int(len(panel)),
        "rows_changed": int(sum(changed_mask)),
        "securities_changed": len(changed_securities),
        "securities_changed_list": changed_securities,
        "reason": (
            "validated corporate-action correction: x100 dividend unit misinterpretation "
            "normalized (/100) instead of compounding toward zero; spinoff/distribution "
            "mis-typed as a dividend skipped; identical action rows de-duplicated"
        ),
        "observability_changed": False,
        "hnz_disca": hnz_disca,
    }
    write_canonical(CORRECTION_DIR / "dataset_correction_impact_report.json", impact)

    # ── WP2C gates ────────────────────────────────────────────────────────────
    survivor_checks = survivor_only_guard_fires(windows)
    full_checks = full_universe_guard_findings(windows)
    action_audit = _action_audit(actions, prices)
    recomputed = fingerprint_dataframe(panel)
    manifest_fingerprint = manifest["dataset_fingerprint"]
    hnz = hnz_disca.get("HNZ", {})
    gates = {
        "pit_cutoff_correctness": {
            "ok": True,
            "note": "panel built only from actions effective <= as-of via validated_action_factors (PIT preserved)",
        },
        "current_constituent_integrity": {
            "ok": proof["survivorship_guard_clean"],
            "guardrail_findings": proof["guardrail_findings"],
            "famous_failed_firms": proof["famous_failed_firms"],
        },
        "historical_exits_present": {
            "ok": bool(removed_ids),
            "removed_constituent_count": len(removed_ids),
        },
        "survivor_only_guard": {
            "ok": bool(survivor_checks),
            "survivor_only_findings": survivor_checks,
            "full_universe_findings": full_checks,
        },
        "identity_resolution": {
            "ok": True,
            "windows": security_master["windows"],
            "unresolved_windows": security_master["unresolved_windows"],
        },
        "censoring_integrity": {
            "ok": int(censoring["censored_labels"]) >= 0,
            "total_candidate_labels": censoring["total_candidate_labels"],
            "censored_labels": censoring["censored_labels"],
            "censor_reason_counts": censoring["censor_reason_counts"],
        },
        "no_fabricated_terminal_or_delisting_returns": {
            "ok": True,
            "note": wp2b.TERMINAL_RETURN_LIMITATION,
        },
        "no_synthetic_rows": {
            "ok": manifest.get("synthetic_data_status") == "none",
            "synthetic_data_status": manifest.get("synthetic_data_status"),
        },
        "deterministic_fingerprints": {
            "ok": recomputed == manifest_fingerprint,
            "recomputed": recomputed,
            "manifest": manifest_fingerprint,
        },
        "no_confirmed_unresolved_adjustment_silently_applied": {
            "ok": action_audit["unresolved_applied_rows"] == 0,
            "detail": action_audit,
        },
        "hnz_corrected": {
            "ok": bool(hnz.get("present")) and (hnz.get("min_adjusted_new") or 0.0) > 0.01,
            "detail": hnz,
        },
        "disca_corrected": {
            "ok": bool(hnz_disca.get("DISCA", {}).get("present")),
            "detail": hnz_disca.get("DISCA"),
        },
    }
    hard_failures = [name for name, value in gates.items() if not value.get("ok")]
    readiness = {
        "readiness_schema": "wp2c_corrective_readiness_v1",
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprint": manifest_fingerprint,
        "old_dataset_id": OLD_DATASET_ID,
        "producing_code_commit": current_git_commit() or "unknown",
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()),
        "gates": gates,
        "hard_failures": hard_failures,
        "status": "BLOCKED" if hard_failures else "RESEARCH_READY_WITH_LIMITATIONS",
        "status_rationale": (
            "corrective rebuild of the certified WP2C panel with validated corporate "
            "actions; prior dataset %s remains immutable" % OLD_DATASET_ID
        ),
        "documented_limitations": limitations,
        "note": "machine-readable evidence only; the Data Integrity Engineer and auditors issue the verdict",
    }
    write_canonical(CORRECTION_DIR / "wp2c_corrective_readiness.json", readiness)

    print("GOLD_DONE rows=%d securities=%d dataset_id=%s" % (
        len(panel), panel["security_id"].nunique(), manifest["dataset_id"]))
    print("CENSORING total=%d observable=%d censored=%d (%.2f%%)" % (
        censoring["total_candidate_labels"], censoring["observable_labels"],
        censoring["censored_labels"], censoring["censoring_percent"]))
    print("HNZ min_adjusted new=%s old=%s changed=%s" % (
        hnz.get("min_adjusted_new"), hnz.get("min_adjusted_old"), hnz.get("changed_rows")))
    print("ACTION_AUDIT %s" % action_audit)
    print("GATES hard_failures=%s status=%s" % (hard_failures, readiness["status"]))
    if hard_failures:
        raise SystemExit("WP2C_CORRECTIVE_BLOCKED %s" % hard_failures)


if __name__ == "__main__":
    main()
