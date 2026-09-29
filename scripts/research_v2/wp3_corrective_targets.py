"""WP5 PHASE 6: corrective WP3 target rebuild against the corrected WP2C dataset.

Re-runs the EXACT WP3 target code path (``build_targets`` +
``annotate_observability`` + ``TargetContract`` + ``FeatureSetManifest``) against
the corrected WP2C gold panel ``dataset_35a278e17c13`` (``553ac17bf5d4d63f``)
instead of the pre-correction ``dataset_dbaa77445b38``.

Point-in-time semantics are preserved UNCHANGED: horizon (12m), benchmark (SPY),
continuous target ``future_12m_excess_return``, classification target
``outperform_12m``, observability rules and censoring policy are identical. Only
the upstream corrected corporate actions/prices feed the SAME formula, so a NEW
immutable ``target_id`` + ``feature_set_id`` are emitted and the superseded
``target_set_888f68d1cfd0`` is never touched.

Emits ``target_correction_impact_report.json`` (OLD vs NEW). Real data only.
This script records measured evidence; it issues no scientific verdict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data import provenance_ledger
from src.research.data import wp2b_eodhd as wp2b
from src.research.data.manifesting import build_gold_manifest
from src.research.features import (
    LEGACY_ROA_ROIC_NOTE,
    V2_ROA_DEFINITION,
    V2_ROIC_DEFINITION,
    build_feature_set,
    build_initial_registry,
    v1_disposition,
)
from src.research.features.keyes import KEYES_VARIABLES, TRACK_HISTORICAL, TRACK_MODERN
from src.research.immutability import save_immutable
from src.research.modes import ResearchMode, current_git_commit
from src.research.targets import (
    CLASSIFICATION_TARGET,
    CONTINUOUS_TARGET,
    TARGET_VERSION,
    TargetContract,
    annotate_observability,
    build_targets,
)

BRONZE_ROOT = ROOT / "data" / "research_v2"
PROVENANCE_ROOT = ROOT / "provenance"
LIVE_DIR = ROOT / "artifacts" / "research" / "wp2b_live"
CORR_DIR = ROOT / "artifacts" / "research" / "wp5_correction"
CORR_WP3_DIR = CORR_DIR / "wp3_corrective"
PROVENANCE_WP3 = PROVENANCE_ROOT / "wp3"
GOLD_NAME = "sp500_pit_targets_v1"

NEW_DATASET_ID = "dataset_35a278e17c13"
NEW_PANEL_VERSION = "553ac17bf5d4d63f"
OLD_TARGET_ID = "target_set_888f68d1cfd0"
OLD_TARGET_VERSION = "f244f86c22b2c2a4"
UNIVERSE_ID = "sp500_pit_wikipedia_eodhd_v1"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wp3():
    return _load_module("wp3_build_targets", ROOT / "scripts" / "research_v2" / "wp3_build_targets.py")


def _wp2c():
    return _load_module("wp2c_corrective_finalize", ROOT / "scripts" / "research_v2" / "wp2c_corrective_finalize.py")


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _equal_or_both_nan(left, right):
    return (pd.isna(left) and pd.isna(right)) or (not pd.isna(left) and not pd.isna(right) and left == right)


def load_panel():
    return layers.read_silver_table(BRONZE_ROOT, wp2b.DATASET_NAME, version=NEW_PANEL_VERSION)


def panel_versions():
    manifest = json.loads((LIVE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != NEW_DATASET_ID:
        raise SystemExit("live manifest dataset_id %r != expected %r" % (manifest.get("dataset_id"), NEW_DATASET_ID))
    return manifest


def build_impact_report(old_targets, new_targets, contract, feature_set):
    """OLD vs NEW target comparison keyed by (security_id, feature_asof, occurrence)."""
    def keyed(frame):
        frame = frame.copy()
        frame["occ"] = frame.groupby(["security_id", "feature_asof"]).cumcount()
        return frame

    key = ["security_id", "feature_asof", "occ"]
    old = keyed(old_targets)
    new = keyed(new_targets)
    merged = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"))

    def changed(column):
        return int(sum(0 if _equal_or_both_nan(row[column + "_old"], row[column + "_new"]) else 1
                       for row in merged[[column + "_old", column + "_new"]].to_dict("records")))

    excess_changed = changed("future_12m_excess_return")
    label_changed = changed("outperform_12m")
    observable_mask = [
        0 if bool(row["target_observable_old"]) == bool(row["target_observable_new"]) else 1
        for row in merged[["target_observable_old", "target_observable_new"]].to_dict("records")
    ]
    excess_changed_mask = [
        not _equal_or_both_nan(row["future_12m_excess_return_old"], row["future_12m_excess_return_new"])
        for row in merged[["future_12m_excess_return_old", "future_12m_excess_return_new"]].to_dict("records")
    ]
    changed_securities = sorted(
        merged.loc[excess_changed_mask, "security_id"].dropna().unique().tolist())

    def security_view(security):
        subset = merged[merged["security_id"] == security].sort_values("feature_asof")
        if subset.empty:
            return {"present": False}
        sample = []
        for row in subset.head(6).to_dict("records"):
            sample.append({
                "feature_asof": str(row["feature_asof"])[:10],
                "excess_old": None if pd.isna(row["future_12m_excess_return_old"]) else float(row["future_12m_excess_return_old"]),
                "excess_new": None if pd.isna(row["future_12m_excess_return_new"]) else float(row["future_12m_excess_return_new"]),
                "outperform_old": None if pd.isna(row["outperform_12m_old"]) else bool(row["outperform_12m_old"]),
                "outperform_new": None if pd.isna(row["outperform_12m_new"]) else bool(row["outperform_12m_new"]),
            })
        return {
            "present": True,
            "rows": int(len(subset)),
            "excess_changed_rows": int(sum(0 if _equal_or_both_nan(row["future_12m_excess_return_old"], row["future_12m_excess_return_new"]) else 1 for row in subset[["future_12m_excess_return_old", "future_12m_excess_return_new"]].to_dict("records"))),
            "label_flips": int(sum(0 if _equal_or_both_nan(row["outperform_12m_old"], row["outperform_12m_new"]) else 1 for row in subset[["outperform_12m_old", "outperform_12m_new"]].to_dict("records"))),
            "sample": sample,
        }

    return {
        "old_target_id": OLD_TARGET_ID,
        "new_target_id": contract.target_id,
        "old_target_version": OLD_TARGET_VERSION,
        "new_target_version": new_targets.attrs.get("gold_version"),
        "old_feature_set_id": "feature_set_56361533cc1b",
        "new_feature_set_id": feature_set.feature_set_id,
        "target_version_semantics": TARGET_VERSION,
        "horizon_months": contract.horizon_months,
        "benchmark": contract.benchmark,
        "continuous_target": CONTINUOUS_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "rows_old": int(len(old_targets)),
        "rows_new": int(len(new_targets)),
        "rows_matched": int(len(merged)),
        "excess_changed_rows": excess_changed,
        "classification_label_flips": label_changed,
        "observability_changed_rows": int(sum(observable_mask)),
        "securities_changed": len(changed_securities),
        "securities_changed_list": changed_securities,
        "observable_old": int(old_targets["target_observable"].sum()),
        "observable_new": int(new_targets["target_observable"].sum()),
        "censored_old": int(old_targets["target_censored"].sum()),
        "censored_new": int(new_targets["target_censored"].sum()),
        "hnz": security_view("HNZ"),
        "disca": security_view("DISCA"),
        "reason": (
            "corrected corporate actions (x100 dividend unit fix; mis-typed spinoff/distribution "
            "skipped; duplicate rows de-duplicated) change the same-horizon PIT total return for "
            "affected securities; horizon/benchmark/target definitions are unchanged"
        ),
        "observability_changed": int(sum(observable_mask)) > 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    CORR_WP3_DIR.mkdir(parents=True, exist_ok=True)
    w3 = _wp3()
    wp2c = _wp2c()

    panel = load_panel()
    manifest = panel_versions()
    print("PANEL rows=%d securities=%d dataset=%s" % (len(panel), panel["security_id"].nunique(), NEW_DATASET_ID))

    prices, actions, declared = w3.load_declared_layers()
    print("SILVER prices=%d actions=%d versions=%s" % (len(prices), len(actions), declared["silver_prices"]))

    # SPY is re-read from the SAME cache the original build used; the provider is
    # never contacted (build_spy_benchmark reuses the cache when present) so the
    # benchmark leg is byte-identical to the certified build.
    benchmark, benchmark_actions_frame, spy_bronze = w3.build_spy_benchmark(None)
    print("SPY rows=%d actions=%d bronze=%s" % (len(benchmark), len(benchmark_actions_frame), spy_bronze["fingerprint"][:16]))

    contract = TargetContract(dataset_id=NEW_DATASET_ID)
    registry = build_initial_registry(include_macro=False).seal()
    feature_set = build_feature_set(NEW_DATASET_ID, current_git_commit() or "unknown", registry=registry)

    target_rows = build_targets(
        panel, prices, benchmark,
        stock_actions=w3.stock_actions_as_objects(actions),
        benchmark_actions=w3.spy_actions_as_objects(benchmark_actions_frame),
        horizon_months=contract.horizon_months,
    )
    observability = annotate_observability(target_rows)
    target_rows = target_rows.reset_index(drop=True)
    target_rows["target_known_at"] = observability["target_known_at"].to_numpy()
    duplicate_keys = int(target_rows.duplicated(subset=["security_id", "feature_asof"]).sum())
    observable = int(target_rows["target_observable"].sum())
    censored = int(target_rows["target_censored"].sum())
    print("TARGETS rows=%d observable=%d censored=%d dup_keys=%d target_id=%s feature_set=%s" % (
        len(target_rows), observable, censored, duplicate_keys, contract.target_id, feature_set.feature_set_id))

    # ── persist target contract + feature-set provenance (content-addressed) ──
    contract_payload = contract.to_dict()
    save_immutable(PROVENANCE_WP3 / ("%s.json" % contract.target_id), contract_payload)
    save_immutable(PROVENANCE_WP3 / ("%s.json" % feature_set.feature_set_id), feature_set.to_dict())
    write_canonical(CORR_WP3_DIR / "target_contract.json", contract_payload)
    write_canonical(CORR_WP3_DIR / "feature_set_manifest.json", feature_set.to_dict())

    registry_payload = {
        "registry_version": registry.version,
        "feature_count": len(registry.features),
        "enabled_count": len(registry.enabled_features()),
        "by_category": registry.by_category(),
        "features": [definition.to_dict() for definition in registry.sorted_features()],
    }
    write_canonical(CORR_WP3_DIR / "feature_registry.json", registry_payload)
    write_canonical(CORR_WP3_DIR / "v1_disposition.json", v1_disposition())
    write_canonical(CORR_WP3_DIR / "keyes_tracks.json", {
        "variables": dict(KEYES_VARIABLES),
        "tracks": {
            TRACK_HISTORICAL: "exact Keyes definitions; X12 has no PIT source -> X12_PROXY only",
            TRACK_MODERN: "benchmark-relative modern analogues; separate track, never merged",
        },
        "x12_note": (
            "no historical point-in-time analyst/Value Line expected-appreciation source exists; "
            "any stand-in is named X12_PROXY and never reported as Keyes X12"
        ),
        "qualification_rule": (
            "explicit simultaneous regression thresholds via KeyesRuleSet; NO forced top-30% quota"
        ),
    })

    known_limitations = [
        "labels are benchmark-relative vs SPY; a stock whose terminal observation is unknown is censored, never labelled 0",
        "EODHD publishes NO delisting returns; a security disappearing before the horizon end has an UNKNOWN terminal return and its 12m label is censored (target_observable=false)",
        "a row is trainable only when target_known_at <= model_date; overlapping 12m labels must be purged by WP6/WP7 using this metadata",
        "Wikipedia membership source omits several historical removals (BSC, Enron, Circuit City, WaMu, old GM); inherited from the WP2C universe limitation, not repaired here",
        "macro features are declared but DISABLED; no FRED/ALFRED vintage is wired in this package",
        "fundamental feature VALUES are not computed in WP3; the registry declares availability rules only",
    ]
    source_fingerprints = {
        "wp2c_gold_panel": w3.layer_fingerprint(wp2b.DATASET_NAME, NEW_PANEL_VERSION),
        "silver_prices": w3.layer_fingerprint("wp2b_sp500_pit_prices_silver", declared["silver_prices"]),
        "silver_actions": w3.layer_fingerprint("wp2b_sp500_pit_actions_silver", declared["silver_actions"]),
        "wp3_spy_bronze": spy_bronze["fingerprint"],
    }

    table = layers.write_gold_table(BRONZE_ROOT, GOLD_NAME, target_rows, meta={"universe_id": UNIVERSE_ID})
    dataset_fingerprint = table["fingerprint"]
    gold_version = table["version"]
    target_rows.attrs["gold_version"] = gold_version
    merged_sources = dict(source_fingerprints)
    merged_sources["gold_table:%s" % GOLD_NAME] = dataset_fingerprint
    manifest_obj = build_gold_manifest(
        GOLD_NAME, target_rows,
        mode=ResearchMode.RESEARCH_V2,
        universe_id=UNIVERSE_ID,
        period_start=str(panel["snapshot_date"].min())[:10],
        period_end=str(panel["snapshot_date"].max())[:10],
        sources=["EODHD", "WIKIPEDIA_SP500", "SEC_EDGAR(declared-only)"],
        source_fingerprints=merged_sources,
        dataset_fingerprint=dataset_fingerprint,
        pit_status="partially_point_in_time",
        known_limitations=known_limitations,
        notes="WP3 benchmark-relative 12m target + observability panel (dataset %s, corrective)" % NEW_DATASET_ID,
        source_versions={
            "wp2c_dataset_id": NEW_DATASET_ID,
            "wp2c_dataset_fingerprint": NEW_PANEL_VERSION,
            "silver_prices": declared["silver_prices"],
            "silver_actions": declared["silver_actions"],
            "feature_set_id": feature_set.feature_set_id,
            "target_id": contract.target_id,
        },
        security_master_version=manifest.get("security_master_version"),
        universe_version=manifest.get("universe_version"),
    )
    gold_manifest_path = layers.layer_root(BRONZE_ROOT, "gold", GOLD_NAME) / ("%s.manifest.json" % manifest_obj.dataset_id)
    gold_manifest, manifest_outcome = wp2c.persist_gold_manifest_idempotent(manifest_obj.to_dict(), gold_manifest_path)
    ledger = provenance_ledger.persist_manifest(gold_manifest, root=PROVENANCE_ROOT, extra={
        "target_id": contract.target_id,
        "target_version": TARGET_VERSION,
        "feature_set_id": feature_set.feature_set_id,
        "observable_labels": observable,
        "censored_labels": censored,
        "corrective": True,
        "corrected_from_dataset_id": "dataset_dbaa77445b38",
        "corrected_from_target_id": OLD_TARGET_ID,
    })

    old_targets = layers.read_silver_table(BRONZE_ROOT, GOLD_NAME, version=OLD_TARGET_VERSION)
    impact = build_impact_report(old_targets, target_rows, contract, feature_set)
    impact["new_target_version"] = gold_version
    impact["new_target_gold_dataset_id"] = manifest_obj.dataset_id
    impact["new_target_gold_fingerprint"] = dataset_fingerprint
    write_canonical(CORR_DIR / "target_correction_impact_report.json", impact)

    summary = {
        "target_id": contract.target_id,
        "target_version": TARGET_VERSION,
        "gold_dataset_id": manifest_obj.dataset_id,
        "gold_version": gold_version,
        "gold_fingerprint": dataset_fingerprint,
        "rows": int(len(target_rows)),
        "observable_labels": observable,
        "censored_labels": censored,
        "duplicate_keys": duplicate_keys,
        "feature_set_id": feature_set.feature_set_id,
        "feature_registry_version": registry.version,
        "upstream_dataset_id": NEW_DATASET_ID,
        "upstream_panel_version": NEW_PANEL_VERSION,
        "continuous_target": CONTINUOUS_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "roa_definition": V2_ROA_DEFINITION,
        "roic_definition": V2_ROIC_DEFINITION,
        "legacy_note": LEGACY_ROA_ROIC_NOTE,
        "spy_rows": int(len(benchmark)),
        "spy_actions": int(len(benchmark_actions_frame)),
        "gold_manifest_outcome": manifest_outcome,
        "ledger": ledger,
        "impact_report": "artifacts/research/wp5_correction/target_correction_impact_report.json",
    }
    write_canonical(CORR_WP3_DIR / "wp3_corrective_summary.json", summary)

    print("WP3_CORRECTIVE_DONE target_id=%s feature_set=%s gold_version=%s rows=%d observable=%d censored=%d" % (
        contract.target_id, feature_set.feature_set_id, gold_version, len(target_rows), observable, censored))
    print("IMPACT excess_changed=%d label_flips=%d observability_changed=%d securities_changed=%d" % (
        impact["excess_changed_rows"], impact["classification_label_flips"],
        impact["observability_changed_rows"], impact["securities_changed"]))


if __name__ == "__main__":
    main()
