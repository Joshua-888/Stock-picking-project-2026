"""WP6 DEVELOPMENT-PERIOD model research (WP6_CORRECTIVE_CONTRACT_V2).

Loads the CERTIFIED WP5 inputs (dataset ``dataset_35a278e17c13`` /
``553ac17bf5d4d63f``, target ``target_set_d2bb16610bce`` / ``56d0f670bdf1b47c``,
feature set ``feature_set_4f7b43726310``), rebuilds the WP5 point-in-time feature
panel through the EXACT WP5 engine, attaches the certified targets, restricts to
development rows (``feature_asof < 2021-01-01``), builds deterministic walk-forward
folds with a 12-month purge, enumerates the complete frozen model grid over the
frozen feature strategies and preprocessing policies, runs the matched negative
controls and robustness checks, and writes an immutable ledger plus metrics.

Corrective contract v2 changes (see ``docs/research_v2/wp6_corrective_contract_v2.md``):

* every WP6 version string is bumped, so the content-addressed experiment id is NEW
  and can never overwrite ``experiment_f7864f37998f`` / ``experiment_05ddc3721b4a``;
* negative controls name a PREDECLARED matched real configuration by deterministic id
  (no ``max(values, key=abs)`` reference anywhere);
* classification inference runs on the AUC skill series ``AUC_t - 0.5``;
* the future-availability guard is a REAL field-based check;
* model-layer Benjamini-Hochberg FDR gates PROMISING;
* robustness checks are EXECUTED and serialized.

The locked holdout (``feature_asof >= 2022-01-01``) and its labels are NEVER read.
No final production model is fitted. The script records measured evidence and issues
no scientific verdict. Real data only; no synthetic fallback.
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
from src.research.modeling.contract import (
    DEFAULT_CONFIG,
    WP6_CORRECTIVE_CONTRACT_VERSION,
    contract_payload,
)
from src.research.modeling.folds import build_folds
from src.research.modeling.io import (
    modeling_payload,
    write_canonical,
    write_experiment,
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

# Withdrawn / misbound historical experiments that must never be overwritten.
WITHDRAWN_WP6_EXPERIMENT_ID = "experiment_f7864f37998f"
MISBOUND_WP6_EXPERIMENT_ID = "experiment_05ddc3721b4a"

CORRECTED_WP4_DIR = ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_corrective"
OLD_EXPERIMENT_DIR = ROOT / "artifacts" / "research" / "wp6" / WITHDRAWN_WP6_EXPERIMENT_ID


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wp5():
    return _load_module("wp5_discover_features", ROOT / "scripts" / "research_v2" / "wp5_discover_features.py")


# ── Aggregation helpers (factual summaries; no ranking or verdict) ────────────

def _count_categories(records, key=lambda record: record.get("research_category")):
    counts = {}
    for record in records:
        value = key(record)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _regression_summary(records):
    summary = {}
    for record in records:
        pooled = record.get("pooled") or {}
        summary[record["config_id"]] = {
            "model": record["model"], "strategy": record["strategy"],
            "params": record["params"], "family_id": record.get("family_id"),
            "mean_ic": pooled.get("mean_ic"), "mean_metric": pooled.get("mean_metric"),
            "icir": pooled.get("icir"), "null": pooled.get("null"),
            "comparator": pooled.get("comparator"),
            "hac_p_value": (pooled.get("hac") or {}).get("p_value"),
            "hac_se": (pooled.get("hac") or {}).get("hac_se"),
            "raw_p": record.get("raw_p"), "q_value": record.get("q_value"),
            "fdr_rejected": record.get("fdr_rejected"),
            "research_category": record.get("research_category"),
        }
    return summary


def _classification_summary(records):
    summary = {}
    for record in records:
        pooled = record.get("pooled") or {}
        hac = pooled.get("hac") or {}
        mean_auc = pooled.get("mean_metric")
        summary[record["config_id"]] = {
            "model": record["model"], "strategy": record["strategy"],
            "params": record["params"], "family_id": record.get("family_id"),
            "mean_auc": mean_auc,
            "mean_skill_auc_minus_half": (None if mean_auc is None else float(mean_auc) - 0.5),
            "skill_mean": hac.get("skill_mean"), "null": pooled.get("null"),
            "comparator": pooled.get("comparator"),
            "hac_se": hac.get("hac_se"), "hac_p_value": hac.get("p_value"),
            "raw_p": record.get("raw_p"), "q_value": record.get("q_value"),
            "fdr_rejected": record.get("fdr_rejected"),
            "research_category": record.get("research_category"),
        }
    values = [item["mean_auc"] for item in summary.values() if item["mean_auc"] is not None]
    aggregate = {
        "configurations": len(summary),
        "mean_auc_min": min(values) if values else None,
        "mean_auc_max": max(values) if values else None,
        "mean_auc_mean": float(np.mean(values)) if values else None,
        "mean_skill_mean": (float(np.mean(values)) - 0.5) if values else None,
    }
    return {"aggregate": aggregate, "by_config": summary}


def _old_vs_new_comparison(categories, families, controls):
    """Descriptive old-vs-new comparison data (no verdict)."""
    comparison = {"withdrawn_experiment_id": WITHDRAWN_WP6_EXPERIMENT_ID,
                  "corrective_contract_version": WP6_CORRECTIVE_CONTRACT_VERSION,
                  "note": "old counts are read from the preserved withdrawn artifact; no old result is endorsed"}
    old_summary_path = OLD_EXPERIMENT_DIR / "summary.json"
    if old_summary_path.is_file():
        try:
            old = json.loads(old_summary_path.read_text(encoding="utf-8"))
            comparison["old_research_categories"] = old.get("research_categories")
            comparison["old_placebo_stop"] = old.get("placebo_stop")
            comparison["old_contract_version"] = (old.get("git_commit"), )
        except (OSError, json.JSONDecodeError):
            comparison["old_research_categories"] = None
    comparison["new_research_categories"] = categories
    comparison["new_family_sizes"] = {key: value["number_of_hypotheses"] for key, value in families.items()}
    comparison["new_controls"] = [
        {"control_id": item["control_id"], "passed": item["passed"],
         "stop_required": item["stop_required"], "matched_real_config_id": item["matched_real_config_id"]}
        for item in controls
    ]
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.model_seed)
    parser.add_argument("--strategies", default="", help="comma-separated subset of strategies (dev only)")
    parser.add_argument("--models", default="", help="comma-separated subset of model names (dev only)")
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
    configuration_records = [record for record in ledger if record.get("record") == "configuration"]
    regression_records = [record for record in configuration_records if record.get("task") == "regression"]
    classification_records = [record for record in configuration_records if record.get("task") == "classification"]
    print("CONFIGURATIONS total=%d regression=%d classification=%d" % (
        len(configuration_records), len(regression_records), len(classification_records)))
    print("CATEGORIES %s" % json.dumps(outcome["categories"], default=str))
    print("CONTROLS %s" % json.dumps(
        [(c["control_id"], c["passed"], c["stop_required"]) for c in outcome["controls"]], default=str))
    print("OVERALL_STOP %s" % outcome["overall_stop"]["stop"])

    commit = current_git_commit() or "unknown"
    binding = modeling_payload(
        DATASET_ID, TARGET_ID, FEATURE_SET_ID, holdout.holdout_id, commit,
        contract_payload(config), model_registry.registry_payload(),
        {"model_seed": config.model_seed, "bootstrap_seed": config.bootstrap_seed,
         "placebo_seed": config.placebo_seed},
        notes={
            "corrective_contract_version": WP6_CORRECTIVE_CONTRACT_VERSION,
            "wp5_experiment_id": WP5_EXPERIMENT_ID,
            "wp4_corrective_experiment_id": WP4_CORRECTIVE_EXPERIMENT_ID,
            "withdrawn_wp6_experiment_id": WITHDRAWN_WP6_EXPERIMENT_ID,
            "misbound_wp6_experiment_id": MISBOUND_WP6_EXPERIMENT_ID,
            "executed_strategies": list(strategies) if strategies else "ALL",
            "executed_models": list(models_wanted) if models_wanted else "ALL",
            "configuration_count": len(configuration_records),
        })
    experiment = experiment_id(binding)
    if experiment in (WITHDRAWN_WP6_EXPERIMENT_ID, MISBOUND_WP6_EXPERIMENT_ID):
        raise RuntimeError("corrective experiment id collides with a preserved historical experiment")

    summary_payload = {
        "experiment_id": experiment,
        "corrective_contract_version": WP6_CORRECTIVE_CONTRACT_VERSION,
        "wp5_experiment_id": WP5_EXPERIMENT_ID,
        "wp4_corrective_experiment_id": WP4_CORRECTIVE_EXPERIMENT_ID,
        "withdrawn_wp6_experiment_id": WITHDRAWN_WP6_EXPERIMENT_ID,
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
        "research_categories": outcome["categories"],
        "categories_by_family": {
            family: _count_categories([record for record in configuration_records
                                       if record.get("family_id") == family])
            for family in outcome["families"]
        },
        "categories_by_strategy": {
            strategy: _count_categories([record for record in configuration_records
                                         if record.get("strategy") == strategy])
            for strategy in sorted({record.get("strategy") for record in configuration_records})
        },
        "families": {key: {k: v for k, v in value.items() if k not in ("q_values", "fdr_rejected")}
                     for key, value in outcome["families"].items()},
        "controls": outcome["controls"],
        "overall_stop": outcome["overall_stop"],
        "leakage_passed": outcome["leakage_passed"],
        "robustness": outcome["robustness"],
        "config_ids": outcome["config_ids"],
        "configuration_count": len(configuration_records),
        "runtime_seconds": round(time.time() - started, 3),
        "limitations": [
            "WP6 is development-period model research only; it fits no production model",
            "the locked holdout (2022-01-01..2025-08-31) and its labels were never read",
            "research categories are predeclared buckets, not a ranking or a verdict",
            "12-month labels overlap, so all core inference is on the monthly metric series",
            "classification inference is on the AUC skill series AUC_t - 0.5",
            "model-layer PROMISING requires Benjamini-Hochberg fdr_rejected within its frozen family",
        ],
    }

    metrics_payload = {
        "configurations": [
            {"config_id": record["config_id"], "model": record["model"],
             "family": record["family"], "task": record["task"], "strategy": record["strategy"],
             "preprocessing": record["preprocessing"], "params": record["params"],
             "metric": record["metric"], "fold_metrics": record["fold_metrics"],
             "pooled": record["pooled"], "runtime_seconds": record["runtime_seconds"],
             "family_id": record.get("family_id"), "raw_p": record.get("raw_p"),
             "q_value": record.get("q_value"), "fdr_rejected": record.get("fdr_rejected"),
             "research_category": record.get("research_category")}
            for record in configuration_records
        ],
    }

    artifacts = {
        "summary.json": summary_payload,
        "metrics.json": metrics_payload,
        "baselines.json": outcome["baselines"],
        "selection_stability.json": outcome["selection_stability"],
        "families.json": outcome["families"],
        "controls.json": outcome["controls"],
        "robustness.json": outcome["robustness"],
        "regression_summary.json": _regression_summary(regression_records),
        "classification_summary.json": _classification_summary(classification_records),
        "categories.json": {"overall": outcome["categories"],
                           "by_family": summary_payload["categories_by_family"],
                           "by_strategy": summary_payload["categories_by_strategy"]},
        "old_vs_new_comparison.json": _old_vs_new_comparison(
            outcome["categories"], outcome["families"], outcome["controls"]),
    }
    ledger_records = list(ledger)
    provenance = {
        "binding.json": binding,
        "contract.json": contract_payload(config),
        "registration.json": _legacy_wp6_registration(experiment),
    }

    paths = write_experiment(ROOT, binding, artifacts, provenance, ledger_records=ledger_records)
    write_canonical(Path(paths["provenance_dir"]) / "index_entry.json", {
        "experiment_id": experiment, "wp": "wp6", "dataset_id": DATASET_ID,
        "target_id": TARGET_ID, "feature_set_id": FEATURE_SET_ID, "holdout_id": holdout.holdout_id,
        "git_commit": commit, "panel_version": PANEL_VERSION, "target_version": TARGET_VERSION,
        "corrective_contract_version": WP6_CORRECTIVE_CONTRACT_VERSION,
    })
    print("EXPERIMENT %s" % experiment)
    print("ARTIFACTS %s" % paths["artifact_dir"])
    print("PROVENANCE %s" % paths["provenance_dir"])
    print("RUNTIME %.1fs" % (time.time() - started))
    return experiment


def _legacy_wp6_registration(experiment):
    return {
        "schema_version": "wp6_model_research_corrective_registration_v2",
        "experiment_id": experiment,
        "corrective_contract_version": WP6_CORRECTIVE_CONTRACT_VERSION,
        "supersedes": None,
        "note": "WP6 corrective rerun; the withdrawn experiment %s and misbound %s are preserved"
                % (WITHDRAWN_WP6_EXPERIMENT_ID, MISBOUND_WP6_EXPERIMENT_ID),
    }


if __name__ == "__main__":
    main()
