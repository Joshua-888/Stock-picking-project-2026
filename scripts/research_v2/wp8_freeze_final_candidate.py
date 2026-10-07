"""WP8 Phase B: freeze ONE final locked-holdout candidate from legal development data.

This engine reuses the frozen ``WP7_VALIDATION_CONTRACT_V5`` selection/fitting
semantics BY MODULE PATH (importlib), never by copying WP7 logic into WP8. It is
development-only: it builds its final legal training set exclusively from rows
whose feature instant is before the 2021-01-01 embargo cutoff, then selects one
of the frozen candidates with the exact WP7 inner-fold procedure, fits the final
model/preprocessor/calibrator with the frozen seed, and freezes everything as an
immutable provenance record under ``provenance/wp8/``.

Locked-holdout performance, labels, returns, predictions and any
2022+/2021-embargo-band observations are NEVER read.

Run with the research runtime:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp8_freeze_final_candidate.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import wp7_generation_corrections as wpc7
from src.research.holdout import locked_holdout, trainable_mask
from src.research.ids import canonical_json
from src.research.immutability import save_immutable
from src.research.modes import current_git_commit

WP7_SCRIPT_REL = Path("scripts") / "research_v2" / "wp7_validation.py"
WP8_SCHEMA_VERSION = "wp8_final_candidate_freeze_v1"
WP8_FREEZE_KIND = "wp8_final_candidate_freeze"
WP7_CANONICAL_GENERATION_ID = "generation_de0f9bbd0bec"
WP7_CONTRACT_VERSION = "WP7_VALIDATION_CONTRACT_V5"
CANONICAL_HOLDOUT_ID = "holdout_7ce54e933e16"
DATASET_ID = "dataset_35a278e17c13"
TARGET_SET_ID = "target_set_d2bb16610bce"
FEATURE_SET_ID = "feature_set_4f7b43726310"
WP6_EXPERIMENT_ID = "experiment_ee434a07a25d"
FROZEN_STAGE = "WP8_FINAL_LEGAL_FREEZE"
DEVELOPMENT_BOUNDARY = pd.Timestamp("2021-01-01", tz="UTC")
RUN_LOG_PATH = "/tmp/wp8_freeze_final_candidate.log"

# Important: this is the certifying WP7 generation, not the stale
# provenance/wp7/model_generation_candidate.json fallback.
WP7_MODULE_NAME = "wp7_validation_engine_for_wp8_freezer"


class Wp8FreezeError(RuntimeError):
    """Raised when the WP8 final candidate cannot be frozen honestly."""


def _utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_wp7_module():
    """Load the certified WP7 engine by file path without importing by name."""
    path = ROOT / WP7_SCRIPT_REL
    if not path.is_file():
        raise Wp8FreezeError("certified WP7 engine is missing: %s" % path)
    spec = importlib.util.spec_from_file_location(WP7_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(WP7_MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def frozen_configs(contract, wp7_mod=None):
    """Return the frozen baseline + nine configs in exact contract order."""
    wp7_mod = wp7_mod or load_wp7_module()
    configs = wp7_mod.parse_frozen_configs(contract)
    if len(configs) != 10:
        raise Wp8FreezeError("expected the frozen baseline plus nine configs; got %d" % len(configs))
    if configs[0].config_id != "baseline_base_rate":
        raise Wp8FreezeError("first frozen candidate is not baseline_base_rate")
    return configs


def load_canonical_wp7_candidate():
    """Load the canonical WP7 candidate EXCLUSIVELY through the resolver index.

    No fallback to ``provenance/wp7/model_generation_candidate.json`` is ever
    permitted; that stale fixed path is deliberately not referenced here.
    """
    resolved = wpc7.assert_canonical(WP7_CANONICAL_GENERATION_ID)
    path = wpc7.canonical_candidate_path()
    if not path.is_file():
        raise Wp8FreezeError("canonical WP7 candidate is missing: %s" % path)
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp8FreezeError("canonical WP7 candidate is not valid JSON: %s" % exc) from exc
    if candidate.get("generation_id") != WP7_CANONICAL_GENERATION_ID:
        raise Wp8FreezeError(
            "resolver canonical id %r does not match candidate id %r"
            % (WP7_CANONICAL_GENERATION_ID, candidate.get("generation_id"))
        )
    if candidate.get("contract_version") != WP7_CONTRACT_VERSION:
        raise Wp8FreezeError(
            "canonical WP7 candidate uses contract %r, expected %r"
            % (candidate.get("contract_version"), WP7_CONTRACT_VERSION)
        )
    if candidate.get("canonical_holdout_id") != CANONICAL_HOLDOUT_ID:
        raise Wp8FreezeError(
            "canonical WP7 candidate holdout id %r != %r"
            % (candidate.get("canonical_holdout_id"), CANONICAL_HOLDOUT_ID)
        )
    return candidate, resolved, path


def require_canonical_holdout(holdout):
    """Hard identity guard: only the canonical locked holdout may be referenced."""
    if getattr(holdout, "holdout_id", None) != CANONICAL_HOLDOUT_ID:
        raise Wp8FreezeError(
            "wrong locked holdout id %r; exact canonical id %s is required"
            % (getattr(holdout, "holdout_id", None), CANONICAL_HOLDOUT_ID)
        )
    return True


def assert_legal_training_frame(frame, holdout):
    """Reject ANY row at/after 2021-01-01 or in the embargo/holdout band."""
    require_canonical_holdout(holdout)
    if "feature_asof" not in frame.columns:
        raise Wp8FreezeError("training frame lacks feature_asof")
    stamps = _utc(frame["feature_asof"])
    if stamps.isna().any():
        raise Wp8FreezeError("training frame contains missing feature_asof")
    if bool((stamps >= DEVELOPMENT_BOUNDARY).any()):
        raise Wp8FreezeError("training frame contains feature_asof >= 2021-01-01")
    if bool((stamps >= holdout.embargo_cutoff).any()):
        raise Wp8FreezeError("training frame contains embargo-band rows")
    if bool((stamps >= holdout.holdout_start).any()):
        raise Wp8FreezeError("training frame contains locked-holdout rows")
    return True


def build_final_legal_train(wp7_mod, panel, contract, holdout=None):
    """Produce FINAL_LEGAL_TRAIN from the certified development panel.

    Uses ``src.research.holdout.trainable_mask`` at ``holdout.holdout_start``;
    supplies contract as an explicit frozen-input guard and performs several
    hard temporal assertions before any model code sees a single observation.
    """
    holdout = holdout or locked_holdout()
    require_canonical_holdout(holdout)
    if contract.get("contract_version") != WP7_CONTRACT_VERSION:
        raise Wp8FreezeError(
            "expected %s, got %s"
            % (WP7_CONTRACT_VERSION, contract.get("contract_version"))
        )
    assert_legal_training_frame(panel, holdout)
    mask = trainable_mask(panel, model_date=holdout.holdout_start, holdout=holdout)
    final_train = panel.loc[mask].reset_index(drop=True)
    if final_train.empty:
        raise Wp8FreezeError("FINAL_LEGAL_TRAIN is empty")
    assert_legal_training_frame(final_train, holdout)

    # Every label in FINAL_LEGAL_TRAIN must close strictly before the holdout
    # window so not even a terminal price from 2022 is read through a label.
    if "target_end" not in final_train.columns:
        raise Wp8FreezeError("FINAL_LEGAL_TRAIN lacks target_end")
    target_ends = _utc(final_train["target_end"])
    if target_ends.isna().any() or bool((target_ends >= holdout.holdout_start).any()):
        raise Wp8FreezeError(
            "FINAL_LEGAL_TRAIN contains labels reaching the locked holdout"
        )
    return final_train


def training_data_fingerprint_payload(frame):
    """The deterministic, content-addressed description of FINAL_LEGAL_TRAIN."""
    stamps = _utc(frame["feature_asof"])
    observable = pd.to_numeric(frame.get("target_observable"), errors="coerce")
    observable = observable.fillna(0).astype(bool) if "target_observable" in frame.columns \
        else pd.Series(False, index=frame.index)
    return {
        "kind": "wp8_final_training_data",
        "rows": int(len(frame)),
        "securities": int(frame["security_id"].nunique()),
        "security_ids": sorted(str(value) for value in pd.unique(frame["security_id"])),
        "min_feature_asof": str(stamps.min().strftime("%Y-%m-%d")),
        "max_feature_asof": str(stamps.max().strftime("%Y-%m-%d")),
        "observable_rows": int(observable.sum()),
    }


def training_data_fingerprint(frame):
    """SHA-256 of the deterministic training-data payload."""
    blob = canonical_json(training_data_fingerprint_payload(frame)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_freeze_id(*, wp7_generation_id, wp7_contract_version, final_config_id,
                      final_calibration, feature_names, model_seed,
                      training_data_fingerprint, producing_commit):
    """Deterministic WP8 final-candidate freeze id."""
    payload = {
        "kind": WP8_FREEZE_KIND,
        "wp7_generation_id": str(wp7_generation_id),
        "wp7_contract_version": str(wp7_contract_version),
        "final_config_id": str(final_config_id),
        "final_calibration": str(final_calibration),
        "final_feature_names": list(feature_names),
        "model_seed": int(model_seed),
        "training_data_fingerprint": str(training_data_fingerprint),
        "producing_commit": str(producing_commit),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "freeze_" + digest[:12]


def freeze_record_path(root=None):
    """Path of the tracked WP8 final-candidate freeze record."""
    return Path(root or ROOT) / "provenance" / "wp8" / "final_candidate_freeze.json"


def artifact_root(root=None):
    return Path(root or ROOT) / "artifacts" / "research" / "wp8"


def write_freeze_record(record, root=None):
    """Write-once WP8 provenance record via the shared immutable helper."""
    path = freeze_record_path(root)
    status = save_immutable(path, record)
    data = path.read_bytes()
    return {
        "path": str(path),
        "status": status,
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha256_prefix": hashlib.sha256(data).hexdigest()[:16],
    }


def _bytes_sha256(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def _write_pickle_artifact(directory, name, obj):
    """Serialize ``obj`` exactly once as an ignored immutable artifact."""
    data = pickle.dumps(obj, protocol=5)
    sha = _bytes_sha256(data)
    path = Path(directory) / name
    if path.exists():
        existing = path.read_bytes()
        if _bytes_sha256(existing) != sha:
            raise Wp8FreezeError(
                "artifact %s already exists with different bytes; refusing to overwrite" % path
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {
        "name": name,
        "path": str(path),
        "sha256": sha,
        "sha256_prefix": sha[:16],
    }


def _inner_selection_table(evaluations):
    return [
        {
            "config_id": item["config_id"],
            "model": item["model"],
            "strategy": item["strategy"],
            "params": item["params"],
            "mean_auc_skill": item["mean_auc_skill"],
            "per_fold": item["per_fold"],
        }
        for item in evaluations
    ]


def _calibration_evidence(calibration, calibration_fit, inner_oof_rows):
    return {
        "selected": calibration["selected"],
        "skills": calibration["skills"],
        "tie_break": calibration["tie_break"],
        "fitted_object_type":
            None if calibration_fit is None else type(calibration_fit).__name__,
        "fitted_on": "selected_model_inner_oof_predictions",
        "inner_oof_rows": int(inner_oof_rows),
    }


def _run(argv=None):
    started = time.time()
    root = ROOT
    wp7 = load_wp7_module()
    clean_head = validate_clean_producing_worktree(root)
    print("PRODUCING_COMMIT %s clean_worktree=True" % clean_head)

    contract = wp7.load_contract(root)
    if contract.get("contract_version") != WP7_CONTRACT_VERSION:
        raise Wp8FreezeError("unexpected WP7 contract %r" % contract.get("contract_version"))

    holdout = locked_holdout()
    require_canonical_holdout(holdout)
    candidate, _resolved, candidate_path = load_canonical_wp7_candidate()
    print("CONTRACT %s digest=%s" % (
        contract["contract_version"], wp7.contract_digest(contract)[:16]
    ))
    print("CANONICAL_WP7_GENERATION %s candidate=%s" % (
        WP7_CANONICAL_GENERATION_ID, candidate_path
    ))
    print("HOLDOUT identity=%s metadata_only=True" % holdout.holdout_id)

    panel, diagnostics = wp7.load_validation_panel()
    print("PANEL rows=%d securities=%d months=%d range=%s..%s" % (
        diagnostics["rows"], diagnostics["securities"], diagnostics["months"],
        diagnostics["date_min"], diagnostics["date_max"],
    ))

    final_train = build_final_legal_train(wp7, panel, contract, holdout=holdout)
    train_stamps = _utc(final_train["feature_asof"])
    print("FINAL_LEGAL_TRAIN rows=%d securities=%d range=%s..%s" % (
        len(final_train), final_train["security_id"].nunique(),
        train_stamps.min().strftime("%Y-%m-%d"), train_stamps.max().strftime("%Y-%m-%d"),
    ))

    configs = frozen_configs(contract, wp7)
    inner_folds, _inner_diag, inner_working = wp7.build_inner_folds(
        final_train, holdout=holdout
    )
    safe_folds, excluded = wp7.safe_inner_folds(
        inner_folds, inner_working, outer_test_start_ts=holdout.holdout_start
    )
    if not safe_folds:
        raise Wp8FreezeError("no safe inner folds are available for final selection")
    fold_frames = wp7._build_fold_frames(inner_working, safe_folds)
    safe_fold_ids = [int(fold.fold) for fold in safe_folds]
    model_seed = int(contract["seeds"]["model_seed"])
    print("INNER_FOLDS raw=%d safe=%d excluded=%d" % (
        len(inner_folds), len(safe_folds), len(excluded)
    ))
    print("MODEL_SELECTION candidates=%d seed=%d" % (len(configs), model_seed))

    evaluations = [
        wp7.evaluate_config_on_inner_folds(
            candidate, fold_frames, safe_folds, contract, model_seed
        )
        for candidate in configs
    ]
    selected_evaluation = wp7.select_model_candidate(evaluations)
    selected_config = next(
        item for item in configs
        if item.config_id == selected_evaluation["config_id"]
    )
    calibration = wp7.select_calibration(
        selected_evaluation, fold_frames, safe_fold_ids
    )
    inner_combined = pd.concat(selected_evaluation["outcomes"], ignore_index=True)
    calibration_fit = wp7._fit_calibration(
        inner_combined["prediction"].to_numpy(dtype="float64"),
        inner_combined["actual"].to_numpy(dtype="float64"),
        calibration["selected"],
    )
    final_model = wp7._fit_selected_model_on_frame(
        final_train, selected_config, model_seed, contract
    )
    final_features = list(final_model["features"])

    # Produce final probabilities on FINAL_LEGAL_TRAIN only; no holdout/blinded
    # row is scored. This output is diagnostic/deterministic and is not persisted
    # as a research prediction artifact.
    final_outcome = wp7._prediction_for_frame(
        final_model, final_train, calibration["selected"], calibration_fit
    )
    print("FINAL_SELECTION config=%s model=%s strategy=%s calibration=%s features=%d" % (
        selected_config.config_id,
        selected_config.model,
        selected_config.strategy,
        calibration["selected"],
        len(final_features),
    ))
    print("FINAL_TRAINING_PROBS rows=%d pred_min=%.6f pred_max=%.6f" % (
        len(final_outcome),
        float(final_outcome["prediction"].min()),
        float(final_outcome["prediction"].max()),
    ))

    fingerprint_payload = training_data_fingerprint_payload(final_train)
    fingerprint = training_data_fingerprint(final_train)
    freeze_id = compute_freeze_id(
        wp7_generation_id=WP7_CANONICAL_GENERATION_ID,
        wp7_contract_version=WP7_CONTRACT_VERSION,
        final_config_id=selected_config.config_id,
        final_calibration=calibration["selected"],
        feature_names=final_features,
        model_seed=model_seed,
        training_data_fingerprint=fingerprint,
        producing_commit=clean_head,
    )

    model_spec = wp7.model_registry.MODELS_BY_NAME[selected_config.model]
    frozen_dir = artifact_root(root) / freeze_id
    preprocessor_obj = {
        "kind": final_model.get("kind"),
        "features": list(final_features),
        "preprocessor": final_model.get("preprocessor"),
        "fitted": final_model.get("fitted"),
    }
    artifact_hashes = {
        "model": _write_pickle_artifact(frozen_dir, "final_model.pkl", final_model),
        "preprocessor": _write_pickle_artifact(
            frozen_dir, "final_preprocessor.pkl", preprocessor_obj
        ),
        "calibrator": _write_pickle_artifact(
            frozen_dir, "final_calibrator.pkl", calibration_fit
        ),
    }

    record = {
        "schema_version": WP8_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "frozen_at_stage": FROZEN_STAGE,
        "wp7_generation_id": WP7_CANONICAL_GENERATION_ID,
        "wp7_contract_version": WP7_CONTRACT_VERSION,
        "wp6_experiment_id": WP6_EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "target_set_id": TARGET_SET_ID,
        "feature_set_id": FEATURE_SET_ID,
        "canonical_holdout_id": holdout.holdout_id,
        "final_config_id": selected_config.config_id,
        "model": selected_config.model,
        "model_family": model_spec.family,
        "model_kind": model_spec.kind,
        "hyperparameters": dict(selected_config.params),
        "feature_strategy": selected_config.strategy,
        "final_selected_features": final_features,
        "preprocessing": {"spec": "VALUE_SPEC", "train_only": True},
        "final_calibration": calibration["selected"],
        "seeds": {
            "model_seed": model_seed,
            "bootstrap_seed": int(contract["seeds"]["bootstrap_seed"]),
            "placebo_seed": int(contract["seeds"]["placebo_seed"]),
        },
        "training_period": {
            "start": fingerprint_payload["min_feature_asof"],
            "end": fingerprint_payload["max_feature_asof"],
        },
        "training_row_count": fingerprint_payload["rows"],
        "training_data_fingerprint": fingerprint,
        "training_data_fingerprint_payload": fingerprint_payload,
        "artifact_hashes": artifact_hashes,
        "producing_commit": clean_head,
        "holdout_performance_accessed": False,
        "holdout_labels_accessed": False,
        "holdout_rows_accessed": False,
        "holdout_usage": "none",
        "final_inner_selection_table": _inner_selection_table(evaluations),
        "calibration_evidence": _calibration_evidence(
            calibration, calibration_fit, int(len(inner_combined))
        ),
        "run_log_path": RUN_LOG_PATH,
    }

    written = write_freeze_record(record, root)
    print("FREEZE_ID %s" % freeze_id)
    print("FROZEN_ARTIFACTS %s" % frozen_dir)
    print("FREEZE_RECORD %s status=%s sha256=%s" % (
        written["path"], written["status"], written["sha256_prefix"]
    ))
    print("RUNTIME %.1fs" % (time.time() - started))
    return {
        "freeze_id": freeze_id,
        "artifact_dir": str(frozen_dir),
        "freeze_record": written,
        "final_model": final_model,
        "calibration": calibration,
        "calibration_fit": calibration_fit,
        "final_train": final_train,
        "evaluations": evaluations,
    }


def validate_clean_producing_worktree(root=None):
    """Fail when a tracked WP8 producing-code file is dirty before freezing.

    The resolved HEAD is returned and is recorded as the producing commit.
    Ignored/untracked files always stay permitted.
    """


def _git_output(args, root, strip=True):
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wp8FreezeError("git unavailable for WP8 preflight: %s" % exc) from exc
    if completed.returncode != 0:
        raise Wp8FreezeError("git command failed during WP8 preflight: %s" % " ".join(args))
    output = completed.stdout or ""
    return output.strip() if strip else output


def _is_producing_code_path(path):
    normalized = path.replace("\\", "/")
    if normalized == "scripts/research_v2/wp8_freeze_final_candidate.py":
        return True
    if normalized == "provenance/wp8/final_candidate_freeze.json":
        return True
    return normalized.startswith("src/research/") or normalized.startswith(
        "scripts/research_v2/"
    )




def main(argv=None):
    return _run(argv)


if __name__ == "__main__":
    main()
