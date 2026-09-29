"""WP5 PHASES 8-11: corrective WP5 discovery re-run.

Re-runs the EXACT WP5 discovery engine (``src.research.discovery`` -> builder)
with:

* the corrected upstream identities
  (panel ``dataset_35a278e17c13`` / ``553ac17bf5d4d63f``,
   target ``target_set_d2bb16610bce`` / ``56d0f670bdf1b47c``,
   feature set ``feature_set_4f7b43726310``);
* the corrected WP4 Keyes signals (``experiment_834a7e60f13c`` under
  ``artifacts/research/wp5_correction/wp4_corrective/``);
* the fixed dependence-aware moving/circular block bootstrap (Phase 8);
* the corrected complete-linkage redundancy grouping (Phase 9).

NO methodology redesign beyond those two corrections: development-only rows, no
locked-holdout rows, 12-month embargo, monthly cross-sectional Spearman rank IC as
the PRIMARY statistic, HAC/Newey-West on the monthly IC series, Benjamini-Hochberg
FDR over the true tested-hypothesis count, fixed quintile diagnostics, missingness
diagnostics, temporal stability, corrected block bootstrap, the placebo battery,
and an unprivileged Keyes comparison. Pooled row IC is secondary only.

Outputs are ISOLATED under ``artifacts/research/wp5/<new_experiment_id>/`` and
``provenance/wp5/<new_experiment_id>/``; the historical WP5 experiment
``experiment_902a843c7ec6`` is NEVER mutated and receives a supersession record.

Real data only; no synthetic fallback. This script records measured evidence and
issues no scientific verdict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data.manifesting import config_fingerprint
from src.research.discovery import builder
from src.research.discovery.catalog import (
    DISCOVERY_VERSION,
    FEATURE_NAMES,
    FEATURE_PANEL_VERSION,
    HANDOFF_VERSION,
    INFERENCE_VERSION,
    REDUNDANCY_VERSION,
    DiscoveryConfig,
    catalog_payload,
)
from src.research.discovery.io import discovery_payload, write_experiment
from src.research.discovery.scorecard import classification_counts
from src.research.holdout import embargo_cutoff, locked_holdout
from src.research.ids import experiment_id
from src.research.immutability import save_immutable, write_json_atomic
from src.research.modes import ResearchMode, current_git_commit

CORR_DIR = ROOT / "artifacts" / "research" / "wp5_correction"
CORR_PROV_DIR = ROOT / "provenance" / "wp5" / "dataset_corrections"
OLD_WP5_ARTIFACTS = ROOT / "artifacts" / "research" / "wp5"
CORRECTED_WP4_DIR = CORR_DIR / "wp4_corrective"

OLD_EXPERIMENT_ID = "experiment_902a843c7ec6"
OLD_DATASET_ID = "dataset_dbaa77445b38"
OLD_TARGET_ID = "target_set_888f68d1cfd0"
OLD_FEATURE_SET_ID = "feature_set_56361533cc1b"

NEW_DATASET_ID = "dataset_35a278e17c13"
NEW_PANEL_VERSION = "553ac17bf5d4d63f"
NEW_TARGET_ID = "target_set_d2bb16610bce"
NEW_TARGET_VERSION = "56d0f670bdf1b47c"
NEW_FEATURE_SET_ID = "feature_set_4f7b43726310"
CORRECTED_WP4_EXPERIMENT_ID = "experiment_834a7e60f13c"

CORRECTION_SCHEMA_VERSION = "wp5_dataset_correction_supersession_v1"

COMPARISON_FIELDS = (
    "months", "mean_ic", "icir", "hac_p_value", "bootstrap_p_value",
    "fdr_q_value", "quantile_spread", "coverage", "redundancy_cluster",
    "classification", "direction_consistent", "short_period_only", "sign_flip",
)
NUMERIC_FIELDS = (
    "months", "mean_ic", "icir", "hac_p_value", "bootstrap_p_value",
    "fdr_q_value", "quantile_spread", "coverage",
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wp5():
    return _load_module("wp5_discover_features", ROOT / "scripts" / "research_v2" / "wp5_discover_features.py")


def _num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def run_corrected_wp5(seed=20260926):
    """Execute the WP5 discovery engine against corrected upstream identities."""
    w5 = _wp5()
    # Pin the CORRECTED upstream identities + corrected WP4 signals (read at call time).
    w5.DATASET_ID = NEW_DATASET_ID
    w5.PANEL_VERSION = NEW_PANEL_VERSION
    w5.TARGET_ID = NEW_TARGET_ID
    w5.TARGET_VERSION = NEW_TARGET_VERSION
    w5.FEATURE_SET_ID = NEW_FEATURE_SET_ID
    w5.WP4_MODERN_SIGNALS = CORRECTED_WP4_DIR / "modern_signals.parquet"
    w5.WP4_HISTORICAL_SIGNALS = CORRECTED_WP4_DIR / "historical_signals.parquet"

    config = DiscoveryConfig(bootstrap_seed=seed)
    holdout = locked_holdout()

    panel = w5.load_panel()
    targets = w5.load_targets()
    prices = w5.load_prices()
    actions = w5.load_actions()
    benchmark_prices = w5.load_benchmark_prices()
    benchmark_actions = w5.load_benchmark_actions()
    fundamentals = w5.load_fundamentals()
    cik_by_ticker, cik_payload = w5.load_cik_by_ticker()
    modern, historical = w5.load_keyes_benchmark()

    print("INPUTS panel=%d targets=%d prices=%d actions=%d fundamentals=%d mapped=%d keyes_modern=%s keyes_hist=%s" % (
        len(panel), len(targets), len(prices), len(actions), len(fundamentals), len(cik_by_ticker),
        None if modern is None else len(modern), None if historical is None else len(historical)))

    bundle = builder.run_discovery(
        panel, targets, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions, config=config,
        keyes_variable_frame=modern, modern_signals=modern, historical_signals=historical,
    )

    commit = current_git_commit() or "unknown"
    binding = discovery_payload(
        NEW_DATASET_ID, NEW_TARGET_ID, NEW_FEATURE_SET_ID, catalog_payload(), config.to_dict(), commit,
        holdout_id=holdout.holdout_id,
        notes={
            "corrective": True,
            "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
            "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID,
            "bootstrap_method": "circular_moving_block_independent_starts",
            "redundancy_linkage": "complete",
        },
    )
    experiment = experiment_id(binding)

    return {
        "experiment": experiment, "config": config, "holdout": holdout, "bundle": bundle,
        "binding": binding, "commit": commit, "cik_by_ticker": cik_by_ticker,
        "cik_payload": cik_payload, "modern": modern, "historical": historical,
    }


def _limitations():
    return [
        "WP5 is discovery only: it emits a candidate universe, not a final feature set and no weighted score",
        "classifications are research categories, not a ranking",
        "SEC EDGAR companyfacts begin around 2009, so early fundamental coverage is bounded and honestly missing",
        "abnormal_volume is UNAVAILABLE because the certified PIT price table carries no volume column",
        "pooled row-level rank IC is reported as a secondary figure only (12-month labels overlap ~11.8x)",
        "the locked holdout (2022-01-01..2025-08-31) was never read during discovery",
        "corrective re-run: upstream WP2C panel and WP3 targets were corrected and the block bootstrap "
        "and redundancy grouping were fixed; no other methodology change",
        "redundancy groups are DESCRIPTIVE complete-linkage cliques; no cluster winner is elected on the full sample",
    ]


def persist_experiment(result):
    """Write the immutable WP5 experiment artifacts + provenance."""
    bundle = result["bundle"]
    config = result["config"]
    experiment = result["experiment"]
    holdout = result["holdout"]
    commit = result["commit"]
    scorecards = bundle["scorecards"]
    limitations = _limitations()

    handoff = builder.assemble_handoff(
        bundle, NEW_DATASET_ID, NEW_TARGET_ID, NEW_FEATURE_SET_ID, experiment,
        keyes_reference=bundle["keyes"].get("summary") if isinstance(bundle["keyes"], dict) else {},
        notes=limitations,
    )

    scorecard_payload = {"scorecards": scorecards,
                         "classification_counts": classification_counts(scorecards)}
    fdr_payload = {"results": bundle["fdr_results"], "summary": bundle["fdr_summary"]}
    inertia = {
        "rank_correlation_matrix": bundle["rank_matrix"].where(pd.notna(bundle["rank_matrix"]), None).to_dict(),
        "ic_series_correlation": bundle["ic_matrix"].where(pd.notna(bundle["ic_matrix"]), None).to_dict(),
        "clusters": bundle["clusters"],
    }
    summary_payload = {
        "experiment_id": experiment,
        "discovery_version": DISCOVERY_VERSION,
        "feature_panel_version": FEATURE_PANEL_VERSION,
        "inference_version": INFERENCE_VERSION,
        "redundancy_version": REDUNDANCY_VERSION,
        "handoff_version": HANDOFF_VERSION,
        "dataset_id": NEW_DATASET_ID,
        "target_id": NEW_TARGET_ID,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "holdout_id": holdout.holdout_id,
        "git_commit": commit,
        "mode": ResearchMode.RESEARCH_V2.value,
        "panel_summary": bundle["panel_summary"],
        "development_rows": bundle["development_rows"],
        "candidate_count": len(scorecards),
        "classification_counts": classification_counts(scorecards),
        "fdr_summary": bundle["fdr_summary"],
        "placebo": bundle["placebo"],
        "config": config.to_dict(),
        "cik_exact_mappings": int(len(result["cik_by_ticker"])),
        "cik_unmapped": int(result["cik_payload"].get("unmapped", 0)),
        "cik_ambiguous": int(result["cik_payload"].get("ambiguous", 0)),
        "corrective": True,
        "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
        "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID,
        "known_limitations": limitations,
    }
    quality_payload = {
        "features": [bundle["per_feature"][name]["quality"] for name in FEATURE_NAMES],
        "note": "quality is factual; nothing is discarded",
    }

    comparison = build_comparison(bundle["scorecards"], experiment)
    write_json_atomic(CORR_DIR / "wp5_old_vs_new.json", comparison)

    reference = write_experiment(
        ROOT, result["binding"],
        artifacts={
            "discovery_summary.json": summary_payload,
            "candidate_feature_universe.json": handoff,
            "feature_scorecard.json": scorecard_payload,
            "feature_quality.json": quality_payload,
            "fdr.json": fdr_payload,
            "redundancy.json": inertia,
            "keyes_benchmark.json": bundle["keyes"],
            "placebo.json": bundle["placebo"],
            "wp5_old_vs_new.json": comparison,
        },
        provenance={
            "%s.json" % experiment: {
                "experiment_id": experiment,
                "dataset_id": NEW_DATASET_ID,
                "panel_version": NEW_PANEL_VERSION,
                "target_id": NEW_TARGET_ID,
                "target_version": NEW_TARGET_VERSION,
                "feature_set_id": NEW_FEATURE_SET_ID,
                "holdout_id": holdout.holdout_id,
                "git_commit": commit,
                "mode": ResearchMode.RESEARCH_V2.value,
                "discovery_version": DISCOVERY_VERSION,
                "inference_version": INFERENCE_VERSION,
                "redundancy_version": REDUNDANCY_VERSION,
                "embargo_cutoff": str(embargo_cutoff())[:19],
                "corrective": True,
                "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
                "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID,
                "sources": [
                    "WP2C_GOLD_PANEL", "WP3_GOLD_TARGETS", "WP2B_SILVER_PRICES", "WP2B_SILVER_ACTIONS",
                    "BENCHMARK_GOLD_SPY", "WP3_SPY_BRONZE_ACTIONS", "SEC_EDGAR", "WP4_KEYES_SIGNALS_CORRECTED",
                ],
                "synthetic_data_status": None,
                "known_limitations": limitations,
            },
        },
    )
    return reference, handoff, comparison, summary_payload


def build_comparison(new_scorecards, new_experiment):
    """OLD vs NEW WP5 discovery comparison for EVERY candidate feature."""
    old_dir = OLD_WP5_ARTIFACTS / OLD_EXPERIMENT_ID
    old_rows = json.loads((old_dir / "feature_scorecard.json").read_text(encoding="utf-8"))["scorecards"]
    old_summary = json.loads((old_dir / "discovery_summary.json").read_text(encoding="utf-8"))
    old_by = {row["feature"]: row for row in old_rows}
    new_by = {row["feature"]: row for row in new_scorecards}

    features = {}
    for name in sorted(set(old_by) | set(new_by)):
        old = old_by.get(name, {})
        new = new_by.get(name, {})
        entry = {
            "old": {field: old.get(field) for field in COMPARISON_FIELDS},
            "new": {field: new.get(field) for field in COMPARISON_FIELDS},
        }
        delta = {}
        for field in NUMERIC_FIELDS:
            old_value = _num(old.get(field))
            new_value = _num(new.get(field))
            delta[field] = None if old_value is None or new_value is None else (new_value - old_value)
        entry["delta"] = delta
        entry["classification_changed"] = old.get("classification") != new.get("classification")
        entry["redundancy_group_changed"] = old.get("redundancy_cluster") != new.get("redundancy_cluster")
        features[name] = entry

    old_counts = old_summary.get("classification_counts", {})
    new_counts = classification_counts(new_scorecards)
    robust_old = int(old_counts.get("ROBUST_CANDIDATE", 0))
    robust_new = int(new_counts.get("ROBUST_CANDIDATE", 0))

    return {
        "old_experiment_id": OLD_EXPERIMENT_ID,
        "new_experiment_id": new_experiment,
        "old_upstream": {"dataset_id": OLD_DATASET_ID, "target_id": OLD_TARGET_ID,
                         "feature_set_id": OLD_FEATURE_SET_ID},
        "new_upstream": {"dataset_id": NEW_DATASET_ID, "panel_version": NEW_PANEL_VERSION,
                         "target_id": NEW_TARGET_ID, "target_version": NEW_TARGET_VERSION,
                         "feature_set_id": NEW_FEATURE_SET_ID,
                         "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID},
        "old_development_rows": old_summary.get("development_rows"),
        "features": features,
        "classification_counts": {"old": old_counts, "new": new_counts},
        "robust_candidate": {"old": robust_old, "new": robust_new, "unchanged": robust_old == robust_new},
        "changed_features": sorted(name for name, entry in features.items() if entry["classification_changed"]),
        "reason": (
            "corrected WP2C panel + corrected WP3 targets change the upstream PIT inputs; the WP5 "
            "discovery engine is unchanged except the fixed dependence-aware block bootstrap and the "
            "descriptive complete-linkage redundancy grouping; no thresholds were changed"
        ),
    }


def write_correction(result, summary_payload, comparison):
    """Supersession + dataset-correction provenance for the old WP5 experiment."""
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    CORR_PROV_DIR.mkdir(parents=True, exist_ok=True)

    corrective_summary = {
        "experiment_id": result["experiment"],
        "dataset_id": NEW_DATASET_ID,
        "panel_version": NEW_PANEL_VERSION,
        "target_id": NEW_TARGET_ID,
        "target_version": NEW_TARGET_VERSION,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID,
        "git_commit": result["commit"],
        "config_fingerprint": config_fingerprint(result["config"].to_dict()),
        "development_rows": summary_payload["development_rows"],
        "candidate_count": summary_payload["candidate_count"],
        "classification_counts": summary_payload["classification_counts"],
        "corrective": True,
        "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
    }
    write_json_atomic(CORR_DIR / "wp5_corrective_summary.json", corrective_summary)
    write_json_atomic(CORR_DIR / "wp5_old_vs_new.json", comparison)

    provenance = {
        "experiment_id": result["experiment"],
        "dataset_id": NEW_DATASET_ID,
        "panel_version": NEW_PANEL_VERSION,
        "target_id": NEW_TARGET_ID,
        "target_version": NEW_TARGET_VERSION,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "git_commit": result["commit"],
        "config_fingerprint": config_fingerprint(result["config"].to_dict()),
        "mode": ResearchMode.RESEARCH_V2.value,
        "discovery_version": DISCOVERY_VERSION,
        "inference_version": INFERENCE_VERSION,
        "redundancy_version": REDUNDANCY_VERSION,
        "corrective": True,
        "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
        "sources": [
            "WP2C_GOLD_PANEL", "WP3_GOLD_TARGETS", "WP2B_SILVER_PRICES", "WP2B_SILVER_ACTIONS",
            "BENCHMARK_GOLD_SPY", "WP3_SPY_BRONZE_ACTIONS", "SEC_EDGAR", "WP4_KEYES_SIGNALS_CORRECTED",
        ],
        "synthetic_data_status": None,
        "known_limitations": _limitations(),
    }
    prov_path = CORR_PROV_DIR / ("%s.json" % result["experiment"])
    save_immutable(prov_path, provenance)

    supersession = {
        "supersession_schema_version": CORRECTION_SCHEMA_VERSION,
        "old_experiment_id": OLD_EXPERIMENT_ID,
        "old_upstream": {"dataset_id": OLD_DATASET_ID, "target_id": OLD_TARGET_ID,
                         "feature_set_id": OLD_FEATURE_SET_ID},
        "new_experiment_id": result["experiment"],
        "new_upstream": {"dataset_id": NEW_DATASET_ID, "target_id": NEW_TARGET_ID,
                         "feature_set_id": NEW_FEATURE_SET_ID},
        "reason": (
            "SUPERSEDED_BY_DATA_CORRECTION: the WP2C panel and WP3 targets were corrected, and the "
            "degenerate full-series-rotation block bootstrap plus the pathological single-linkage "
            "redundancy grouping were fixed; this corrective experiment re-runs the WP5 discovery "
            "engine on the corrected data with those two corrections"
        ),
        "superseded_records": ["provenance/wp5/%s/%s.json" % (OLD_EXPERIMENT_ID, OLD_EXPERIMENT_ID)],
        "corrected_provenance_ref": "provenance/wp5/dataset_corrections/%s.json" % result["experiment"],
    }
    super_path = CORR_PROV_DIR / ("supersession_%s.json" % OLD_EXPERIMENT_ID)
    save_immutable(super_path, supersession)

    index = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "entries": {
            OLD_EXPERIMENT_ID: {
                "old_experiment_id": OLD_EXPERIMENT_ID,
                "new_experiment_id": result["experiment"],
                "corrected_provenance_file": "%s.json" % result["experiment"],
                "supersession_file": "supersession_%s.json" % OLD_EXPERIMENT_ID,
                "canonical_provenance_ref": "provenance/wp5/dataset_corrections/%s.json" % result["experiment"],
            }
        },
    }
    write_json_atomic(CORR_PROV_DIR / "index.json", index)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    CORR_DIR.mkdir(parents=True, exist_ok=True)

    result = run_corrected_wp5(seed=args.seed)
    reference, handoff, comparison, summary_payload = persist_experiment(result)
    write_correction(result, summary_payload, comparison)

    print("WP5_CORRECTIVE_DONE experiment=%s development_rows=%d candidates=%d counts=%s placebo_stop=%s" % (
        result["experiment"], summary_payload["development_rows"], summary_payload["candidate_count"],
        summary_payload["classification_counts"], result["bundle"]["placebo"].get("stop")))
    print("OLD_VS_NEW robust_candidate old=%s new=%s unchanged=%s" % (
        comparison["robust_candidate"]["old"], comparison["robust_candidate"]["new"],
        comparison["robust_candidate"]["unchanged"]))
    print("artifact_dir=%s provenance_dir=%s" % (reference["artifact_dir"], reference["provenance_dir"]))


if __name__ == "__main__":
    main()
