"""WP6 DEVELOPMENT-PERIOD model research.

Loads the CERTIFIED WP5 inputs (panel ``dataset_35a278e17c13`` /
``553ac17bf5d4d63f``, target ``target_set_d2bb16610bce`` / ``56d0f670bdf1b47c``,
feature set ``feature_set_4f7b43726310``), rebuilds the WP5 point-in-time feature
panel through the EXACT WP5 engine, attaches the certified targets, restricts to
development rows (``feature_asof < 2021-01-01``), builds deterministic walk-forward
folds with a 12-month purge, enumerates the complete frozen model grid over the
frozen feature strategies and preprocessing policies, and writes an immutable
ledger plus metrics, feature-selection-per-fold records and placebos.

The locked holdout (``feature_asof >= 2022-01-01``) and its labels are NEVER read.
No final production model is fitted. The script records measured evidence and
issues no scientific verdict. Real data only; no synthetic fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data.manifesting import config_fingerprint
from src.research.holdout import locked_holdout
from src.research.ids import experiment_id
from src.research.modes import ResearchMode, current_git_commit
from src.research.modeling import models as model_registry
from src.research.modeling import runner
from src.research.modeling.contract import DEFAULT_CONFIG, contract_payload
from src.research.modeling.folds import build_folds
from src.research.modeling.io import (
    modeling_payload,
    write_experiment,
    write_canonical,
)
from src.research.modeling.panel import build_modeling_panel

# ── Certified upstream identifiers (pinned; never a lexicographic guess) ──────
DATASET_ID = "dataset_35a278e17c13"
PANEL_VERSION = "553ac17bf5d4d63f"
TARGET_ID = "target_set_d2bb16610bce"
TARGET_VERSION = "56d0f670bdf1b47c"
FEATURE_SET_ID = "feature_set_4f7b43726310"
WP5_EXPERIMENT_ID = "experiment_f985287c1315"
WP4_CORRECTIVE_EXPERIMENT_ID = "experiment_834a7e60f13c"

CORRECTED_WP4_DIR = ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_corrective"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wp5():
    return _load_module("wp5_discover_features", ROOT / "scripts" / "research_v2" / "wp5_discover_features.py")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.model_seed)
    parser.add_argument("--strategies", default="", help="comma-separated subset of strategies")
    parser.add_argument("--models", default="", help="comma-separated subset of model names")
    parser.add_argument("--no-placebo", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    w5 = _wp5()
    # Reuse the WP5 loaders VERBATIM; only the pinned identities are overridden.
    w5.DATASET_ID = DATASET_ID
    w5.PANEL_VERSION = PANEL_VERSION
    w5.TARGET_ID = TARGET_ID
    w5.TARGET_VERSION = TARGET_VERSION
    w5.FEATURE_SET_ID = FEATURE_SET_ID
    w5.WP4_MODERN_SIGNALS = CORRECTED_WP4_DIR / "modern_signals.parquet"
    w5.WP4_HISTORICAL_SIGNALS = CORRECTED_WP4_DIR / "historical_signals.parquet"

    config = DEFAULT_CONFIG
    holdout = locked_holdout()
    print("HOLDOUT canonical=%s start=%s" % (holdout.holdout_id, str(holdout.holdout_start)[:10]))

    panel = w5.load_panel()
    targets = w5.load_targets()
    prices = w5.load_prices()
    actions = w5.load_actions()
    benchmark_prices = w5.load_benchmark_prices()
    benchmark_actions = w5.load_benchmark_actions()
    fundamentals = w5.load_fundamentals()
    cik_by_ticker, _cik_payload = w5.load_cik_by_ticker()
    print("INPUTS panel=%d targets=%d prices=%d actions=%d fundamentals=%d mapped=%d" % (
        len(panel), len(targets), len(prices), len(actions), len(fundamentals), len(cik_by_ticker)))

    frame, panel_diagnostics = build_modeling_panel(
        panel, targets, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions, config=config, holdout=holdout,
    )
    print("PANEL rows=%d securities=%d months=%d range=%s..%s" % (
        panel_diagnostics["rows"], panel_diagnostics["securities"], panel_diagnostics["months"],
        panel_diagnostics["date_min"], panel_diagnostics["date_max"]))
    print("DUPLICATE key check=%s" % json.dumps(panel_diagnostics["duplicate_key_check"], default=str))

    folds, fold_diagnostics, working = build_folds(frame, config=config, holdout=holdout)
    print("FOLDS n=%d" % len(folds))
    for record in fold_diagnostics["folds"]:
        print("  fold=%d model_date=%s train=%s..%s rows=%d val=%s..%s rows=%d" % (
            record["fold"], record["model_date"], record["train_start"], record["train_end"],
            record["train_rows"], record["validation_start"], record["validation_end"],
            record["validation_rows"]))

    fold_frames = {
        fold.fold: {"train": working.iloc[list(fold.train_index)].reset_index(drop=True),
                    "validation": working.iloc[list(fold.validation_index)].reset_index(drop=True)}
        for fold in folds
    }

    strategies = [name for name in (args.strategies.split(",") if args.strategies else []) if name] or None
    models_wanted = [name for name in (args.models.split(",") if args.models else []) if name] or None

    outcome = runner.run_experiment(working, fold_frames, folds, config=config, seed=args.seed,
                                    strategies=strategies, models=models_wanted)
    ledger = outcome["ledger"]

    # Reference real mean IC: best pooled regression configuration's mean metric.
    regression_configs = [record for record in ledger if record.get("record") == "configuration"
                          and record.get("task") == "regression"]
    real_mean_ic = None
    if regression_configs:
        values = [record["pooled"].get("mean_metric") for record in regression_configs
                  if record["pooled"].get("mean_metric") is not None]
        if values:
            real_mean_ic = float(max(values, key=lambda value: abs(value)))

    placebo_result = None
    if not args.no_placebo:
        placebo_result = runner.run_placebo(working, fold_frames, folds, config=config,
                                            seed=config.placebo_seed, real_mean_ic=real_mean_ic)
        print("PLACEBO stop=%s checks=%s" % (placebo_result["overall"]["stop"],
                                             json.dumps(placebo_result["checks"], default=str)[:400]))

    stop_flag = bool(placebo_result["overall"]["stop"]) if placebo_result else False
    categories = {}
    for record in ledger:
        if record.get("record") != "configuration":
            continue
        key = "%s|%s|%s|%s" % (record["model"], sorted(record["params"].items()),
                               record["strategy"], record["task"])
        baseline = record.get("baseline_reference") or {}
        fold_metrics = [{"mean_metric": item.get("mean_metric")} for item in record["fold_metrics"]]
        record["research_category"] = runner.classify_configuration(
            fold_metrics, record["pooled"],
            {"mean_metric": baseline.get("mean_metric")}, stop_flag, config=config)
        categories[record["research_category"]] = categories.get(record["research_category"], 0) + 1

    commit = current_git_commit() or "unknown"
    binding = modeling_payload(DATASET_ID, TARGET_ID, FEATURE_SET_ID, holdout.holdout_id,
                               commit, contract_payload(config), model_registry.registry_payload(),
                               {"model_seed": config.model_seed, "bootstrap_seed": config.bootstrap_seed,
                                "placebo_seed": config.placebo_seed},
                               notes={"wp5_experiment_id": WP5_EXPERIMENT_ID,
                                      "wp4_corrective_experiment_id": WP4_CORRECTIVE_EXPERIMENT_ID})
    experiment = experiment_id(binding)

    summary_payload = {
        "experiment_id": experiment,
        "wp5_experiment_id": WP5_EXPERIMENT_ID,
        "wp4_corrective_experiment_id": WP4_CORRECTIVE_EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "target_version": TARGET_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "holdout_id": holdout.holdout_id,
        "git_commit": commit,
        "mode": ResearchMode.RESEARCH_V2.value,
        "config_fingerprint": config_fingerprint(config.to_dict()),
        "panel_diagnostics": panel_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "selection_stability": outcome["selection_stability"],
        "research_categories": categories,
        "placebo": placebo_result,
        "placebo_stop": stop_flag,
        "configuration_count": len([record for record in ledger if record.get("record") == "configuration"]),
        "runtime_seconds": round(time.time() - started, 3),
        "limitations": [
            "WP6 is development-period model research only; it fits no production model",
            "the locked holdout (2022-01-01..2025-08-31) and its labels were never read",
            "research categories are predeclared buckets, not a ranking or a verdict",
            "12-month labels overlap, so all core inference is on the monthly metric series",
            "level-return portfolio diagnostics are reported separately from predictive metrics",
        ],
    }

    legacy = _legacy_wp6_registration(experiment)
    metrics_payload = {
        "configurations": [
            {"model": record["model"], "family": record["family"], "task": record["task"],
             "strategy": record["strategy"], "params": record["params"],
             "fold_metrics": record["fold_metrics"], "pooled": record["pooled"],
             "runtime_seconds": record["runtime_seconds"],
             "research_category": record.get("research_category")}
            for record in ledger if record.get("record") == "configuration"
        ],
    }
    artifacts = {
        "summary.json": summary_payload,
        "metrics.json": metrics_payload,
        "baselines.json": outcome["baselines"],
        "selection_stability.json": outcome["selection_stability"],
    }
    # ledger records go to the immutable JSONL ledger
    ledger_records = list(ledger)
    provenance = {
        "binding.json": binding,
        "contract.json": contract_payload(config),
        "registration.json": legacy,
    }

    paths = write_experiment(ROOT, binding, artifacts, provenance, ledger_records=ledger_records)
    write_canonical(Path(paths["provenance_dir"]) / "index_entry.json", {
        "experiment_id": experiment, "wp": "wp6", "dataset_id": DATASET_ID,
        "target_id": TARGET_ID, "feature_set_id": FEATURE_SET_ID, "holdout_id": holdout.holdout_id,
        "git_commit": commit, "panel_version": PANEL_VERSION, "target_version": TARGET_VERSION,
    })
    print("EXPERIMENT %s" % experiment)
    print("ARTIFACTS %s" % paths["artifact_dir"])
    print("PROVENANCE %s" % paths["provenance_dir"])
    print("RUNTIME %.1fs" % (time.time() - started))
    return experiment


def _legacy_wp6_registration(experiment):
    return {
        "schema_version": "wp6_model_research_registration_v1",
        "experiment_id": experiment,
        "supersedes": None,
        "note": "WP6 is a NEW experiment; no historical WP6 artifact exists to supersede",
    }


if __name__ == "__main__":
    main()
