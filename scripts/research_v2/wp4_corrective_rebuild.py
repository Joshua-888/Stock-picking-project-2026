"""WP5 PHASE 7: corrective WP4 re-run against corrected upstream data/targets.

Re-runs the EXACT WP4 engine (``compute_keyes_variables`` + historical/modern
signals + diagnostics + placebos) with the corrected upstream identities:

* panel  dataset_35a278e17c13 / 553ac17bf5d4d63f (corrected WP2C)
* target target_set_d2bb16610bce / 56d0f670bdf1b47c (corrected WP3)
* featset feature_set_4f7b43726310

NO methodology redesign: historical/modern separation, X12_PROXY labelling,
explicit simultaneous Keyes thresholds (no top-30%/nlargest quota), equal-weight
sign-aware modern composite and full-sample-free weighting are all inherited from
the single ``src/research/keyes`` engine, which this script only *calls*.

Outputs are ISOLATED under ``artifacts/research/wp5_correction/wp4_corrective/``
and ``provenance/wp4/dataset_corrections/``; the superseded WP4 experiment
``experiment_5ed52dcf2f44`` and its corrected provenance are never mutated.

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
from src.research.ids import experiment_id
from src.research.immutability import save_immutable, write_json_atomic
from src.research.modes import ResearchMode, current_git_commit
from src.research.keyes import (
    DEFINITION_VERSION,
    DIAGNOSTIC_VERSION,
    SIGNAL_VERSION,
    CompositeConfig,
    ComputationConfig,
)

CORR_DIR = ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_corrective"
CORR_PROV_DIR = ROOT / "provenance" / "wp4" / "dataset_corrections"
OLD_WP4_ARTIFACTS = ROOT / "artifacts" / "research" / "wp4"

OLD_EXPERIMENT_ID = "experiment_5ed52dcf2f44"
OLD_DATASET_ID = "dataset_dbaa77445b38"
OLD_PANEL_VERSION = "ac294282d8e949f2"
OLD_TARGET_ID = "target_set_888f68d1cfd0"
OLD_TARGET_VERSION = "f244f86c22b2c2a4"
OLD_FEATURE_SET_ID = "feature_set_56361533cc1b"

NEW_DATASET_ID = "dataset_35a278e17c13"
NEW_PANEL_VERSION = "553ac17bf5d4d63f"
NEW_TARGET_ID = "target_set_d2bb16610bce"
NEW_TARGET_VERSION = "56d0f670bdf1b47c"
NEW_FEATURE_SET_ID = "feature_set_4f7b43726310"

CORRECTION_SCHEMA_VERSION = "wp4_dataset_correction_supersession_v1"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wp4():
    return _load_module("wp4_build_keyes", ROOT / "scripts" / "research_v2" / "wp4_build_keyes.py")


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def run_corrected_wp4():
    """Execute the WP4 engine against corrected upstream identities."""
    w4 = _wp4()
    # Pin the CORRECTED upstream identities on the module (read at call time).
    w4.DATASET_ID = NEW_DATASET_ID
    w4.PANEL_VERSION = NEW_PANEL_VERSION
    w4.TARGET_ID = NEW_TARGET_ID
    w4.TARGET_VERSION = NEW_TARGET_VERSION
    w4.FEATURE_SET_ID = NEW_FEATURE_SET_ID

    config = ComputationConfig()

    panel = w4.load_panel()
    targets = w4.load_targets()
    prices = w4.load_prices()
    actions = w4.load_actions()
    benchmark_prices = w4.load_benchmark_prices()
    benchmark_actions = w4.load_benchmark_actions()
    fundamentals = w4.load_fundamentals()
    cik_by_ticker, cik_payload = w4.load_cik_by_ticker()
    available_min, available_max = w4._available_at_bounds(fundamentals)
    print("INPUTS panel=%d targets=%d prices=%d actions=%d fundamentals=%d (avail %s..%s) mapped=%d" % (
        len(panel), len(targets), len(prices), len(actions), len(fundamentals),
        available_min, available_max, len(cik_by_ticker)))

    variable_frame, coverage = w4.compute_keyes_variables(
        panel, prices, actions, fundamentals, cik_by_ticker, benchmark_prices, benchmark_actions, config,
    )
    coverage["edgar_fundamentals_rows"] = int(len(fundamentals))
    coverage["edgar_available_min"] = available_min
    coverage["edgar_available_max"] = available_max
    coverage["cik_exact_mappings"] = int(len(cik_by_ticker))
    coverage["cik_unmapped"] = int(cik_payload.get("unmapped", 0))
    coverage["cik_ambiguous"] = int(cik_payload.get("ambiguous", 0))

    historical = w4.historical_signals(variable_frame, expected_rows=int(coverage["observations"]))
    composite = w4.modern_composite(variable_frame, CompositeConfig())
    diagnostics = w4.run_diagnostics(variable_frame, targets, historical, composite)
    fidelity = w4.fidelity_table(coverage)
    status = w4.replication_status(fidelity)
    placebo = w4.placebo_checks(composite, targets, seed=20260926)
    placebo["replication_status"] = status

    spec_payload = w4.build_spec_payload(config, "SPY benchmark from certified benchmark_gold_SPY silver + WP3 SPY bronze actions")
    mapping_payload = w4.variable_mapping_payload()

    payload = {
        "dataset_id": NEW_DATASET_ID,
        "panel_version": NEW_PANEL_VERSION,
        "target_id": NEW_TARGET_ID,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "definition_version": DEFINITION_VERSION,
        "signal_version": SIGNAL_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "config": config.to_dict(),
    }
    experiment = experiment_id(payload)

    return {
        "experiment": experiment, "config": config, "spec_payload": spec_payload,
        "mapping_payload": mapping_payload, "coverage": coverage, "historical": historical,
        "composite": composite, "diagnostics": diagnostics, "fidelity": fidelity,
        "status": status, "placebo": placebo, "payload": payload, "variable_frame": variable_frame,
    }


def persist_corrected(result):
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    write_canonical(CORR_DIR / "keyes_spec.json", result["spec_payload"])
    write_canonical(CORR_DIR / "variable_mapping.json", result["mapping_payload"])
    write_canonical(CORR_DIR / "coverage.json", result["coverage"])
    write_canonical(CORR_DIR / "fidelity_table.json", result["fidelity"].to_dict("records"))
    write_canonical(CORR_DIR / "placebo.json", result["placebo"])
    write_canonical(CORR_DIR / "sector_stability.json", result["diagnostics"]["sector"])
    write_canonical(CORR_DIR / "diagnostics.json", {
        "components": result["diagnostics"]["components"].to_dict("records"),
        "composite": result["diagnostics"]["composite"],
        "qualification": result["diagnostics"]["qualification"],
    })
    result["diagnostics"]["yearly"].to_csv(CORR_DIR / "yearly_stability.csv", index=False)
    result["diagnostics"]["qualification_by_year"].to_csv(CORR_DIR / "qualification_by_year.csv", index=False)
    result["historical"].to_parquet(CORR_DIR / "historical_signals.parquet", index=False)
    result["composite"].to_parquet(CORR_DIR / "modern_signals.parquet", index=False)


def build_comparison(result):
    """OLD vs CORRECTED WP4 comparison (rank-based and level-based)."""
    old_diag = json.loads((OLD_WP4_ARTIFACTS / "diagnostics.json").read_text(encoding="utf-8"))
    old_summary = json.loads((OLD_WP4_ARTIFACTS / "wp4_run_summary.json").read_text(encoding="utf-8"))
    new_diag = result["diagnostics"]
    new_comp = new_diag["composite"]
    old_comp = old_diag["composite"]

    old_hist = pd.read_parquet(OLD_WP4_ARTIFACTS / "historical_signals.parquet")
    old_mod = pd.read_parquet(OLD_WP4_ARTIFACTS / "modern_signals.parquet")
    new_hist = result["historical"]
    new_mod = result["composite"]

    def components_map(diag):
        return {row["variable"]: row for row in diag["components"]}

    old_c = components_map(old_diag)
    new_c = components_map(new_diag)

    rank_based = {
        "composite_rank_ic": {"old": _num(old_comp.get("rank_ic")), "new": _num(new_comp.get("rank_ic"))},
        "composite_rank_ic_n": {"old": old_comp.get("rank_ic_n"), "new": new_comp.get("rank_ic_n")},
        "component_rank_ic": {
            name: {"old": _num(old_c.get(name, {}).get("rank_ic")), "new": _num(new_c.get(name, {}).get("rank_ic"))}
            for name in sorted(set(old_c) | set(new_c))
        },
        "composite_outperform_top": {"old": _num(old_comp.get("composite_outperform_top")), "new": _num(new_comp.get("composite_outperform_top"))},
    }
    level_based = {
        "composite_excess_spread": {"old": _num(old_comp.get("composite_excess_spread")), "new": _num(new_comp.get("composite_excess_spread"))},
        "universe_mean_excess": {"old": _num(old_comp.get("universe_mean_excess")), "new": _num(new_comp.get("universe_mean_excess"))},
        "universe_outperform_rate": {"old": _num(old_comp.get("universe_outperform_rate")), "new": _num(new_comp.get("universe_outperform_rate"))},
        "component_excess_spread": {
            name: {"old": _num(old_c.get(name, {}).get("excess_spread")), "new": _num(new_c.get(name, {}).get("excess_spread"))}
            for name in sorted(set(old_c) | set(new_c))
        },
        "component_outperform_spread": {
            name: {"old": _num(old_c.get(name, {}).get("outperform_spread")), "new": _num(new_c.get(name, {}).get("outperform_spread"))}
            for name in sorted(set(old_c) | set(new_c))
        },
    }
    qualification = {
        "historical_qualified_old": int(old_hist["qualified"].sum()),
        "historical_qualified_new": int(new_hist["qualified"].sum()),
        "historical_rows_old": int(len(old_hist)),
        "historical_rows_new": int(len(new_hist)),
        "modern_rows_old": int(len(old_mod)),
        "modern_rows_new": int(len(new_mod)),
        "modern_scored_old": int(old_mod["composite"].notna().sum()),
        "modern_scored_new": int(new_mod["composite"].notna().sum()),
    }
    return {
        "old_experiment_id": OLD_EXPERIMENT_ID,
        "new_experiment_id": result["experiment"],
        "old_upstream": {"dataset_id": OLD_DATASET_ID, "panel_version": OLD_PANEL_VERSION,
                          "target_id": OLD_TARGET_ID, "target_version": OLD_TARGET_VERSION,
                          "feature_set_id": OLD_FEATURE_SET_ID},
        "new_upstream": {"dataset_id": NEW_DATASET_ID, "panel_version": NEW_PANEL_VERSION,
                          "target_id": NEW_TARGET_ID, "target_version": NEW_TARGET_VERSION,
                          "feature_set_id": NEW_FEATURE_SET_ID},
        "rank_based": rank_based,
        "level_based": level_based,
        "qualification": qualification,
        "replication_status": {"old": old_summary.get("replication_status"), "new": result["status"]},
        "reason": (
            "corrected corporate actions + corrected WP3 targets change only the upstream PIT "
            "inputs; the WP4 engine, tracks, thresholds, weighting and diagnostics are unchanged"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    CORR_PROV_DIR.mkdir(parents=True, exist_ok=True)

    result = run_corrected_wp4()
    persist_corrected(result)

    summary = {
        "experiment_id": result["experiment"],
        "dataset_id": NEW_DATASET_ID,
        "panel_version": NEW_PANEL_VERSION,
        "target_id": NEW_TARGET_ID,
        "target_version": NEW_TARGET_VERSION,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "definition_version": DEFINITION_VERSION,
        "signal_version": SIGNAL_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "git_commit": current_git_commit() or "unknown",
        "config_fingerprint": config_fingerprint(result["config"].to_dict()),
        "replication_status": result["status"],
        "observations": int(result["coverage"]["observations"]),
        "historical_qualified": int(result["historical"]["qualified"].sum()),
        "modern_rows": int(len(result["composite"])),
        "modern_scored": int(result["composite"]["composite"].notna().sum()),
        "corrective": True,
        "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
    }
    write_canonical(CORR_DIR / "wp4_corrective_summary.json", summary)

    comparison = build_comparison(result)
    write_canonical(CORR_DIR / "wp4_old_vs_corrected.json", comparison)
    write_canonical(ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_old_vs_corrected.json", comparison)

    provenance = {
        "experiment_id": result["experiment"],
        "dataset_id": NEW_DATASET_ID,
        "panel_version": NEW_PANEL_VERSION,
        "target_id": NEW_TARGET_ID,
        "target_version": NEW_TARGET_VERSION,
        "feature_set_id": NEW_FEATURE_SET_ID,
        "git_commit": current_git_commit() or "unknown",
        "config_fingerprint": config_fingerprint(result["config"].to_dict()),
        "mode": ResearchMode.RESEARCH_V2.value,
        "signal_version": SIGNAL_VERSION,
        "definition_version": DEFINITION_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "replication_status": result["status"],
        "sources": [
            "WP2C_GOLD_PANEL", "WP3_GOLD_TARGETS", "WP2B_SILVER_PRICES", "WP2B_SILVER_ACTIONS",
            "BENCHMARK_GOLD_SPY", "WP3_SPY_BRONZE_ACTIONS", "SEC_EDGAR",
        ],
        "known_limitations": [
            "Keyes X12 has no historical point-in-time analyst/Value Line source; X12_PROXY (five-year revenue growth) is used and never reported as X12",
            "SEC EDGAR companyfacts start around 2009, so early-universe fundamental variables are UNAVAILABLE, not estimated",
            "only genuinely observable 12m outcomes (target_observable=true) contribute to any diagnostic",
            "corrective re-run: upstream corporate actions and targets were corrected; no methodology change",
        ],
        "synthetic_data_status": None,
        "corrective": True,
        "corrected_from_experiment_id": OLD_EXPERIMENT_ID,
    }
    prov_path = CORR_PROV_DIR / ("%s.json" % result["experiment"])
    outcome = save_immutable(prov_path, provenance)

    supersession = {
        "supersession_schema_version": CORRECTION_SCHEMA_VERSION,
        "old_experiment_id": OLD_EXPERIMENT_ID,
        "old_upstream": {"dataset_id": OLD_DATASET_ID, "target_id": OLD_TARGET_ID, "feature_set_id": OLD_FEATURE_SET_ID},
        "new_experiment_id": result["experiment"],
        "new_upstream": {"dataset_id": NEW_DATASET_ID, "target_id": NEW_TARGET_ID, "feature_set_id": NEW_FEATURE_SET_ID},
        "reason": (
            "upstream corporate-action unit/type/duplicate correction changed the certified WP2C panel "
            "and WP3 targets; this corrective experiment re-runs the unchanged WP4 engine on the corrected data"
        ),
        "superseded_records": ["provenance/wp4/%s.json" % OLD_EXPERIMENT_ID,
                              "provenance/wp4/corrections/experiment_609b46519a41.json"],
        "corrected_provenance_ref": "provenance/wp4/dataset_corrections/%s.json" % result["experiment"],
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
                "canonical_provenance_ref": "provenance/wp4/dataset_corrections/%s.json" % result["experiment"],
            }
        },
    }
    write_json_atomic(CORR_PROV_DIR / "index.json", index)

    print("WP4_CORRECTIVE_DONE experiment=%s status=%s obs=%d qualified=%d modern_scored=%d" % (
        result["experiment"], result["status"], summary["observations"],
        summary["historical_qualified"], summary["modern_scored"]))
    print("RANK_IC old=%s new=%s" % (comparison["rank_based"]["composite_rank_ic"]["old"],
                                      comparison["rank_based"]["composite_rank_ic"]["new"]))


if __name__ == "__main__":
    main()
