"""WP8 Phase C: deterministic ONE-SHOT locked-holdout evaluator.

This engine evaluates the Phase B frozen final candidate against the locked
holdout once and only once. It is deliberately access-controlled:

* it never starts before every pre-access gate passes;
* it refuses a stale or superseded WP7 generation candidate;
* it refuses any frozen candidate other than ``freeze_e62eac30df40``;
* it refuses any holdout other than ``holdout_7ce54e933e16``;
* it refuses to run more than once through a persistent access ledger;
* it uses only the frozen fitted model/preprocessor/calibrator and performs
  NO fitting, feature selection, calibration selection or model selection in
  the holdout path.

Locked-holdout rows, labels, predictions and performance must only be read by
``run_holdout_evaluation`` and only after the access ledger has moved to
``STARTED``. This module must not be used to inspect real holdout content
directly during development.

Run with the research runtime ONLY after the later pre-access audit gates:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp8_run_holdout_evaluation.py

Synthetic/development fixtures are the only supported test input and never
enter research artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import wp7_generation_corrections as wpc7  # noqa: E402
from src.research.holdout import locked_holdout  # noqa: E402
from src.research.ids import canonical_json  # noqa: E402
from src.research.immutability import save_immutable  # noqa: E402
from src.research.modes import current_git_commit  # noqa: E402
from src.research.modeling.metrics import (  # noqa: E402
    calibration_table,
    classification_metrics,
    monthly_auc_series,
    quantile_spread,
)

CONTRACT_REL = Path("provenance") / "wp8" / "holdout_evaluation_contract_v1.json"
FREEZE_REL = Path("provenance") / "wp8" / "final_candidate_freeze.json"
AUTHORIZATION_REL = Path("provenance") / "wp8" / "holdout_evaluation_authorization.json"
LEDGER_REL = Path("provenance") / "wp8" / "holdout_access.json"
SUMMARY_REL = Path("provenance") / "wp8" / "holdout_evaluation_summary.json"
WP7_SCRIPT_REL = Path("scripts") / "research_v2" / "wp7_validation.py"

CONTRACT_VERSION = "HOLDOUT_EVALUATION_CONTRACT_V1"
CANONICAL_HOLDOUT_ID = "holdout_7ce54e933e16"
CANONICAL_WP7_GENERATION_ID = "generation_de0f9bbd0bec"
CANONICAL_WP7_CONTRACT = "WP7_VALIDATION_CONTRACT_V5"
CANONICAL_WP6_EXPERIMENT_ID = "experiment_ee434a07a25d"
CANONICAL_FREEZE_ID = "freeze_e62eac30df40"
TARGET = "outperform_12m"

HOLDOUT_START = pd.Timestamp("2022-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2025-08-31", tz="UTC")
HOLDOUT_MIN_OBSERVATIONS = 30
HAC_LAG = 12

LEDGER_SCHEMA_VERSION = "wp8_holdout_access_ledger_v1"
STATE_AUTHORIZED_NOT_ACCESSED = "AUTHORIZED_NOT_ACCESSED"
STATE_STARTED = "STARTED"
STATE_COMPLETED = "COMPLETED"
STATE_INVALIDATED = "INVALIDATED"

RESULT_SCHEMA_VERSION = "wp8_holdout_evaluation_result_v1"
RESULT_KIND = "wp8_holdout_evaluation"
SUMMARY_SCHEMA_VERSION = "wp8_holdout_evaluation_summary_v1"


class Wp8HoldoutEvaluationError(RuntimeError):
    """Raised when the locked-holdout evaluation cannot proceed honestly."""


def _git_output(args, root, strip=True):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wp8HoldoutEvaluationError("git is unavailable: %s" % exc) from exc
    if completed.returncode != 0:
        raise Wp8HoldoutEvaluationError(
            "git command failed during preflight: %s" % " ".join(args)
        )
    output = completed.stdout or ""
    return output.strip() if strip else output


def _dirty_non_ignored_paths(root):
    output = _git_output(
        ["status", "--porcelain", "--untracked-files=no"], root, strip=False
    )
    paths = set()
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if path:
            paths.add(path)
    return sorted(paths)


def _is_producing_code_path(path):
    normalized = path.replace("\\", "/")
    if normalized == "scripts/research_v2/wp8_run_holdout_evaluation.py":
        return True
    if normalized in (
        "provenance/wp8/holdout_evaluation_contract_v1.json",
        "provenance/wp8/holdout_evaluation_authorization.json",
        "provenance/wp8/holdout_access.json",
        "provenance/wp8/final_candidate_freeze.json",
    ):
        return True
    return normalized.startswith("src/research/") or normalized.startswith(
        "scripts/research_v2/"
    )


def validate_clean_producing_worktree(root=None):
    """Fail when a tracked producing-file is modified before the one-shot run."""
    root = Path(root or ROOT)
    commit = current_git_commit(str(root))
    if not commit:
        raise Wp8HoldoutEvaluationError("no git commit available for reproducible provenance")
    dirty = [
        path for path in _dirty_non_ignored_paths(root) if _is_producing_code_path(path)
    ]
    if dirty:
        raise Wp8HoldoutEvaluationError(
            "holdout evaluation refusing to run with modified tracked producing-code "
            "files; producing commit %s would not contain the actual code. Dirty paths: %s"
            % (commit, ", ".join(dirty))
        )
    return commit


# ── Small byte/json helpers ───────────────────────────────────────────────────
def sha256_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def parse_json_file(path, expected_schema=None):
    path = Path(path)
    if not path.is_file():
        raise Wp8HoldoutEvaluationError("missing JSON file: %s" % path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp8HoldoutEvaluationError("invalid JSON file %s: %s" % (path, exc)) from exc
    if expected_schema is not None and loaded.get("schema_version") not in (expected_schema, None):
        raise Wp8HoldoutEvaluationError(
            "unexpected schema_version %r in %s; expected %r"
            % (loaded.get("schema_version"), path, expected_schema)
        )
    return loaded


def write_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp-", suffix=".json"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(obj) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


# ── WP8 Phase C contract/freeze accessors ────────────────────────────────────
def contract_path(root=None):
    return Path(root or ROOT) / CONTRACT_REL


def freeze_path(root=None):
    return Path(root or ROOT) / FREEZE_REL


def authorization_path(root=None):
    return Path(root or ROOT) / AUTHORIZATION_REL


def ledger_path(root=None):
    return Path(root or ROOT) / LEDGER_REL


def summary_path(root=None):
    return Path(root or ROOT) / SUMMARY_REL


def contract_file_digest(root=None):
    """Raw-byte SHA-256 of the frozen Phase C contract file."""
    return sha256_file(contract_path(root))


def load_evaluation_contract(root=None, path=None):
    """Load and verify the HOLDOUT_EVALUATION_CONTRACT_V1 JSON contract."""
    path = Path(path) if path is not None else contract_path(root)
    contract = parse_json_file(path, expected_schema="wp8_holdout_evaluation_contract_v1")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise Wp8HoldoutEvaluationError(
            "wrong holdout evaluation contract %r; expected %r"
            % (contract.get("contract_version"), CONTRACT_VERSION)
        )
    return contract


def require_canonical_holdout(holdout):
    """Hard identity guard: only the exact canonical holdout id is permitted."""
    if getattr(holdout, "holdout_id", None) != CANONICAL_HOLDOUT_ID:
        raise Wp8HoldoutEvaluationError(
            "wrong locked holdout id %r; exact canonical id %s is required"
            % (getattr(holdout, "holdout_id", None), CANONICAL_HOLDOUT_ID)
        )
    return True


def load_locked_canonical_holdout(root=None):
    """Load the canonical holdout definition and require the phase-C identity."""
    holdout = locked_holdout()
    require_canonical_holdout(holdout)
    return holdout


def canonical_wp7_candidate(root=None):
    """Load the canonical WP7 candidate through ONLY the corrections index.

    ``generation_04b8e2810b50`` and all other MIBOUND/NON-canonical records are
    explicitly rejected. The stale fixed-path
    ``provenance/wp7/model_generation_candidate.json`` is deliberately never
    referenced here.
    """
    resolved = wpc7.assert_canonical(CANONICAL_WP7_GENERATION_ID)
    path = wpc7.canonical_candidate_path()
    candidate = parse_json_file(path)
    if candidate.get("generation_id") != CANONICAL_WP7_GENERATION_ID:
        raise Wp8HoldoutEvaluationError(
            "resolver returned non-canonical WP7 candidate %r"
            % candidate.get("generation_id")
        )
    if candidate.get("contract_version") != CANONICAL_WP7_CONTRACT:
        raise Wp8HoldoutEvaluationError(
            "canonical WP7 candidate uses contract %r; expected %r"
            % (candidate.get("contract_version"), CANONICAL_WP7_CONTRACT)
        )
    if candidate.get("canonical_holdout_id") not in (CANONICAL_HOLDOUT_ID, None):
        raise Wp8HoldoutEvaluationError(
            "canonical WP7 candidate is bound to another holdout %r"
            % candidate.get("canonical_holdout_id")
        )
    return candidate, resolved


def require_withdrawn_generation_rejected(generation_id):
    """Explicitly reject the stale/misbound non-canonical generation."""
    if generation_id is None:
        raise Wp8HoldoutEvaluationError("no WP7 generation id supplied")
    resolved = wpc7.resolve_generation(generation_id)
    if resolved.get("canonical") is True:
        return resolved
    raise Wp8HoldoutEvaluationError(
        "%s is a withdrawn/non-canonical WP7 candidate and must never be selected"
        % generation_id
    )


def load_final_candidate_freeze(root=None, path=None):
    """Load and verify the frozen final candidate record."""
    path = Path(path) if path is not None else freeze_path(root)
    freeze = parse_json_file(path, expected_schema="wp8_final_candidate_freeze_v1")
    if freeze.get("freeze_id") != CANONICAL_FREEZE_ID:
        raise Wp8HoldoutEvaluationError(
            "wrong final candidate freeze id %r; exact freeze %s is required"
            % (freeze.get("freeze_id"), CANONICAL_FREEZE_ID)
        )
    if freeze.get("wp7_generation_id") != CANONICAL_WP7_GENERATION_ID:
        raise Wp8HoldoutEvaluationError(
            "freeze is bound to the wrong WP7 generation %r"
            % freeze.get("wp7_generation_id")
        )
    if freeze.get("wp7_contract_version") != CANONICAL_WP7_CONTRACT:
        raise Wp8HoldoutEvaluationError(
            "freeze is bound to the wrong WP7 contract %r"
            % freeze.get("wp7_contract_version")
        )
    if freeze.get("canonical_holdout_id") != CANONICAL_HOLDOUT_ID:
        raise Wp8HoldoutEvaluationError(
            "freeze is bound to another holdout %r"
            % freeze.get("canonical_holdout_id")
        )
    if freeze.get("final_calibration") not in ("none", "platt_scaling", "isotonic_regression"):
        raise Wp8HoldoutEvaluationError(
            "unknown final calibration %r" % freeze.get("final_calibration")
        )
    return freeze


def artifact_hashes_present(freeze):
    required = ("model", "preprocessor", "calibrator")
    hashes = freeze.get("artifact_hashes") or {}
    missing = [name for name in required if name not in hashes]
    if missing:
        raise Wp8HoldoutEvaluationError(
            "freeze artifact_hashes missing required artifacts: %s" % ", ".join(missing)
        )
    return {name: hashes[name] for name in required}


def verify_frozen_artifact_hashes(freeze, root=None):
    """Verify every frozen final-model artifact sha256 against the freeze record."""
    root = Path(root or ROOT)
    hashes = artifact_hashes_present(freeze)
    for name, spec in hashes.items():
        path = Path(spec.get("path"))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise Wp8HoldoutEvaluationError(
                "frozen %s artifact is missing: %s" % (name, path)
            )
        actual = sha256_file(path)
        if actual != spec.get("sha256"):
            raise Wp8HoldoutEvaluationError(
                "frozen %s artifact hash mismatch: freeze=%s actual=%s"
                % (name, spec.get("sha256"), actual)
            )
    return True


def load_frozen_bundle(freeze, root=None):
    """Load ONLY the pre-fitted final model bundle after hash verification."""
    verify_frozen_artifact_hashes(freeze, root=root)
    root = Path(root or ROOT)
    hashes = artifact_hashes_present(freeze)
    loaded = {}
    for name in ("model", "preprocessor", "calibrator"):
        path = Path(hashes[name]["path"])
        if not path.is_absolute():
            path = root / path
        with path.open("rb") as handle:
            loaded[name] = pickle.load(handle)
    return {
        "model": loaded["model"],
        "preprocessor": loaded["preprocessor"],
        "calibrator": loaded["calibrator"],
    }


# ── Authorization and one-shot access ledger ─────────────────────────────────
def load_authorization(root=None, path=None):
    path = Path(path) if path is not None else authorization_path(root)
    return parse_json_file(path, expected_schema="wp8_holdout_evaluation_authorization_v1")


def assert_authorization_ready(authorization):
    """Fail closed unless the explicit pre-access authorization is exact."""
    if authorization.get("authorization_type") != "HOLDOUT_EVALUATION_AUTHORIZATION":
        raise Wp8HoldoutEvaluationError("holdout evaluation authorization type is invalid")
    if authorization.get("scope") != "exactly one evaluation":
        raise Wp8HoldoutEvaluationError("holdout evaluation authorization scope is invalid")
    if authorization.get("authorized_wp7_generation") != CANONICAL_WP7_GENERATION_ID:
        raise Wp8HoldoutEvaluationError("holdout evaluation is not authorized for canonical WP7")
    if authorization.get("authorized_final_candidate_freeze") != CANONICAL_FREEZE_ID:
        raise Wp8HoldoutEvaluationError("holdout evaluation is not authorized for canonical freeze")
    if authorization.get("authorized_holdout") != CANONICAL_HOLDOUT_ID:
        raise Wp8HoldoutEvaluationError("holdout evaluation is not authorized for canonical holdout")
    if int(authorization.get("holdout_evaluation_count", 0)) != 1:
        raise Wp8HoldoutEvaluationError("holdout evaluation count is not exactly one")
    return True


def load_access_ledger(root=None, path=None):
    path = Path(path) if path is not None else ledger_path(root)
    return parse_json_file(path, expected_schema=LEDGER_SCHEMA_VERSION)


def assert_ledger_ready(ledger, contract_digest=None):
    """Fail closed unless the ledger is the exact pre-access init state."""
    if ledger.get("holdout_id") != CANONICAL_HOLDOUT_ID:
        raise Wp8HoldoutEvaluationError(
            "access ledger is bound to another holdout %r" % ledger.get("holdout_id")
        )
    if ledger.get("state") != STATE_AUTHORIZED_NOT_ACCESSED:
        raise Wp8HoldoutEvaluationError(
            "access ledger is not AUTHORIZED_NOT_ACCESSED (state=%r); refusing without touching data"
            % ledger.get("state")
        )
    if int(ledger.get("access_count", -1)) != 0:
        raise Wp8HoldoutEvaluationError("access ledger access_count is not zero")
    if int(ledger.get("evaluation_count", -1)) != 0:
        raise Wp8HoldoutEvaluationError("access ledger evaluation_count is not zero")
    if ledger.get("final_candidate_freeze_id") != CANONICAL_FREEZE_ID:
        raise Wp8HoldoutEvaluationError("access ledger freeze id is not canonical")
    if ledger.get("wp7_generation_id") != CANONICAL_WP7_GENERATION_ID:
        raise Wp8HoldoutEvaluationError("access ledger WP7 generation id is not canonical")
    if contract_digest is not None and ledger.get("contract_digest") != contract_digest:
        raise Wp8HoldoutEvaluationError(
            "access ledger contract_digest does not match the frozen contract file"
        )
    if not ledger.get("contract_digest"):
        raise Wp8HoldoutEvaluationError("access ledger has no contract_digest")
    return True


def ledger_with(ledger, **changes):
    record = dict(ledger)
    record.update(changes)
    return record


def write_ledger_atomic(ledger, root=None, path=None):
    path = Path(path) if path is not None else ledger_path(root)
    write_atomic(path, ledger)
    return parse_json_file(path, expected_schema=LEDGER_SCHEMA_VERSION)


def transition_ledger_to_started(ledger, *, contract_digest, freeze=None,
                                 code_digest=None, producing_commit=None,
                                 starter_payload=None, root=None, path=None):
    """Atomically transition AUTHORIZED_NOT_ACCESSED -> STARTED."""
    assert_ledger_ready(ledger, contract_digest=contract_digest)
    record = ledger_with(
        ledger,
        state=STATE_STARTED,
        access_count=1,
        started_at=starter_payload or {},
        started_freeze_id=(freeze or {}).get("freeze_id"),
        started_code_digest=code_digest,
        started_producing_commit=producing_commit,
    )
    return write_ledger_atomic(record, root=root, path=path)


def transition_ledger_to_completed(started_ledger, *, evaluation_id, result_hash,
                                   root=None, path=None):
    """Atomically transition STARTED -> COMPLETED with immutable result binding."""
    if started_ledger.get("state") != STATE_STARTED:
        raise Wp8HoldoutEvaluationError(
            "cannot complete ledger in state %r; expected STARTED" % started_ledger.get("state")
        )
    record = ledger_with(
        started_ledger,
        state=STATE_COMPLETED,
        evaluation_count=1,
        completed_evaluation_id=evaluation_id,
        completed_result_sha256=result_hash,
    )
    return write_ledger_atomic(record, root=root, path=path)


def transition_ledger_invalidate(ledger, reason, root=None, path=None):
    """Mark the ledger INVALIDATED without retrying the one-shot evaluation."""
    record = ledger_with(
        ledger,
        state=STATE_INVALIDATED,
        invalidation_reason=str(reason),
    )
    return write_ledger_atomic(record, root=root, path=path)


# ── Holdout frame construction ─────────────────────────────────────────────“─
def _utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def _parse_observable(values):
    working = pd.Series(values, dtype="object")
    return working.astype(str).str.lower().isin(("true", "1", "1.0", "yes")) | \
        (pd.to_numeric(working, errors="coerce") == 1)


def assert_holdout_window_only(frame, holdout=None):
    """Refuse any row whose feature_asof lies outside the locked holdout window."""
    if "feature_asof" not in frame.columns:
        raise Wp8HoldoutEvaluationError("frame lacks feature_asof")
    stamps = _utc(frame["feature_asof"])
    if stamps.isna().any():
        raise Wp8HoldoutEvaluationError("frame contains missing feature_asof")
    start = holdout.holdout_start if holdout is not None else HOLDOUT_START
    end = holdout.holdout_end if holdout is not None else HOLDOUT_END
    if bool((stamps < start).any()) or bool((stamps > end).any()):
        raise Wp8HoldoutEvaluationError(
            "frame contains feature_asof outside the locked holdout window "
            "%s..%s" % (start.date(), end.date())
        )
    return True


def select_holdout_rows(panel, freeze, holdout=None):
    """Select ONLY locked-holdout eligible rows; censored/missing outcomes excluded.

    A row qualifies when:

    * ``feature_asof`` is in [holdout_start, holdout_end];
    * ``target_observable`` is true;
    * the binary target is present and finite (never imputed, never class 0);
    * every required frozen final feature is available.
    """
    holdout = holdout or load_locked_canonical_holdout()
    require_canonical_holdout(holdout)
    for column in ("feature_asof", "target_observable", TARGET):
        if column not in panel.columns:
            raise Wp8HoldoutEvaluationError("panel lacks %r" % column)
    stamps = _utc(panel["feature_asof"])
    start = holdout.holdout_start
    end = holdout.holdout_end
    features = list(freeze.get("final_selected_features") or [])
    missing_features = [name for name in features if name not in panel.columns]
    if missing_features:
        raise Wp8HoldoutEvaluationError(
            "panel lacks required frozen features: %s" % ", ".join(missing_features)
        )

    in_window = (stamps >= start) & (stamps <= end) & stamps.notna()
    observable = _parse_observable(panel["target_observable"])
    label = pd.to_numeric(panel[TARGET], errors="coerce")
    label_valid = label.notna() & np.isfinite(label.to_numpy(dtype="float64")) \
        & np.isin(label.to_numpy(dtype="float64"), (0.0, 1.0))
    features_valid = panel[features].notna().all(axis=1)
    mask = in_window & observable.to_numpy() & label_valid & features_valid.to_numpy()
    selected = panel.loc[mask].reset_index(drop=True)
    if selected.empty:
        raise Wp8HoldoutEvaluationError("locked holdout evaluation frame is empty")
    assert_holdout_window_only(selected, holdout)
    return selected


def assert_no_holdout_fitting(selected_frame=None):
    """A static sentinel: the holdout path may only transform/predict, never fit.

    Model selection, feature selection, preprocessing fit, hyperparameter tuning,
    calibration fitting and calibration selection are explicitly forbidden. The
    only acceptable operations are ``transform`` with a previously fitted
    preprocessor, ``predict_proba`` with a previously fitted estimator, and
    ``predict`` with a previously fitted calibration object.
    """
    allowed = {"transform", "predict_proba", "predict", "predict_log_proba"}
    return {"forbidden_operations": ["fit", "fit_transform"], "allowed_operations": sorted(allowed)}


def _apply_frozen_calibration(calibrator, probabilities, method):
    p = np.asarray(probabilities, dtype="float64")
    if method == "none" or calibrator is None:
        return p
    if method == "platt_scaling":
        logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        return calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]
    if method == "isotonic_regression":
        return calibrator.predict(p)
    raise Wp8HoldoutEvaluationError("unknown calibration method %r" % method)


def outcome_frame(frame, prediction):
    return pd.DataFrame({
        "modeling_month": frame["modeling_month"].to_numpy()
        if "modeling_month" in frame.columns else _utc(frame["feature_asof"]).to_numpy(),
        "feature_asof": _utc(frame["feature_asof"]).to_numpy(),
        "prediction": np.asarray(prediction, dtype="float64"),
        "actual": pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype="float64"),
        "security_id": frame["security_id"].to_numpy()
        if "security_id" in frame.columns
        else np.arange(len(frame), dtype="int64"),
    })


def predict_frozen_holdout(frame, freeze, model_obj, preprocessor_obj, calibrator):
    """Score the holdout frame with the frozen fitted bundle. No fitting happens here."""
    assert_no_holdout_fitting()
    if freeze.get("final_calibration") not in ("none", "platt_scaling", "isotonic_regression"):
        raise Wp8HoldoutEvaluationError(
            "unknown final calibration %r" % freeze.get("final_calibration")
        )
    features = list(freeze.get("final_selected_features") or [])
    if not features:
        raise Wp8HoldoutEvaluationError("freeze record has no final_selected_features")

    if model_obj.get("kind") == "baseline_base_rate":
        prediction = np.full(len(frame), float(model_obj["base_rate"]), dtype="float64")
    elif model_obj.get("kind") == "estimator":
        fitted = model_obj.get("fitted")
        if fitted is None and isinstance(preprocessor_obj, dict) and "fitted" in preprocessor_obj:
            fitted = preprocessor_obj["fitted"]
        if fitted is None:
            raise Wp8HoldoutEvaluationError(
                "frozen model bundle has no fitted preprocessor input; cannot score without fitting"
            )
        preprocessor = model_obj.get("preprocessor")
        if preprocessor is None and isinstance(preprocessor_obj, dict):
            preprocessor = preprocessor_obj.get("preprocessor")
        if preprocessor is None:
            raise Wp8HoldoutEvaluationError("frozen model bundle has no preprocessor")
        x = preprocessor.transform(frame, fitted)
        probabilities = model_obj["estimator"].predict_proba(x)
        prediction = np.asarray(probabilities[:, 1], dtype="float64")
    else:
        raise Wp8HoldoutEvaluationError("unknown frozen model kind %r" % model_obj.get("kind"))

    prediction = _apply_frozen_calibration(calibrator, prediction, freeze["final_calibration"])
    return outcome_frame(frame, prediction)


# ── Deterministic primary inference ────────────────────────────────────────────
def monthly_auc_skill_series(outcome, min_observations=HOLDOUT_MIN_OBSERVATIONS):
    series, diagnostics = monthly_auc_series(
        outcome,
        "prediction",
        "actual",
        asof_col="feature_asof",
        min_observations=min_observations,
    )
    if series.empty:
        return series, diagnostics, np.array([], dtype="float64")
    return series, diagnostics, series["auc"].to_numpy(dtype="float64") - 0.5


def newey_west_hac(values, lag=HAC_LAG):
    """Deterministic one-sided HAC/Newey-West Bartlett inference on a mean-zero test.

    Returns mean, HAC SE, t-stat, one-sided upper-tail p and a 95% CI around the
    mean. The null is zero and p is computed as SF(t) for t>0; for t<=0, the
    one-sided p is exactly 1.0.
    """
    x = np.asarray(values, dtype="float64")
    n = int(x.size)
    if n < 2 or not np.isfinite(x).all():
        raise Wp8HoldoutEvaluationError(
            "primary HAC inference is invalid: need at least 2 finite monthly skill values"
        )
    mean = float(np.mean(x))
    errors = x - mean
    lrv = float(np.mean(errors * errors))
    lag_used = min(int(lag), n - 1)
    if lag_used < 0:
        lag_used = 0
    for j in range(1, lag_used + 1):
        gamma = float(np.mean(errors[j:] * errors[:-j]))
        weight = 1.0 - (j / (lag_used + 1.0))
        lrv += 2.0 * weight * gamma
    if not np.isfinite(lrv) or lrv <= 0.0:
        raise Wp8HoldoutEvaluationError(
            "primary HAC long-run variance is invalid/zero; result blocked"
        )
    se = float(np.sqrt(lrv / n))
    if not np.isfinite(se) or se <= 0.0:
        raise Wp8HoldoutEvaluationError("primary HAC standard error is invalid/zero")
    t_stat = mean / se
    from scipy import stats

    p_one_sided = float(stats.norm.sf(t_stat)) if t_stat > 0.0 else 1.0
    z = 1.959963984540054
    return {
        "mean": mean,
        "hac_se": se,
        "t_stat": t_stat,
        "one_sided_hac_p": p_one_sided,
        "ci_lower": mean - z * se,
        "ci_upper": mean + z * se,
        "lag": lag_used,
        "n": n,
        "valid": True,
    }


def verdict_from_primary(primary):
    mean = float(primary["mean"])
    p = primary.get("one_sided_hac_p")
    if mean > 0.0 and p is not None and p < 0.05:
        return "HOLDOUT_SIGNAL_CONFIRMED"
    if mean > 0.0:
        return "HOLDOUT_POSITIVE_BUT_INCONCLUSIVE"
    return "HOLDOUT_NOT_CONFIRMED"


def compute_primary_metrics(outcome, min_observations=HOLDOUT_MIN_OBSERVATIONS):
    """Equal-weighted monthly AUC skill plus deterministic one-sided HAC inference."""
    series, diagnostics, skills = monthly_auc_skill_series(
        outcome, min_observations=min_observations
    )
    if skills.size == 0:
        raise Wp8HoldoutEvaluationError(
            "no eligible monthly AUC values; primary inference blocked/invalid"
        )
    hac = newey_west_hac(skills, lag=HAC_LAG)
    aucs = series["auc"].to_numpy(dtype="float64")
    primary = {
        "months": int(len(skills)),
        "mean": float(hac["mean"]),
        "mean_auc": float(np.mean(aucs)),
        "mean_auc_skill": hac["mean"],
        "hac_se": hac["hac_se"],
        "t_stat": hac["t_stat"],
        "one_sided_hac_p": hac["one_sided_hac_p"],
        "ci_lower": hac["ci_lower"],
        "ci_upper": hac["ci_upper"],
        "median_auc_skill": float(np.median(skills)),
        "positive_month_fraction": float(np.mean(skills > 0.0)),
        "min_observations": int(min_observations),
        "monthly_auc": series.to_dict(orient="records"),
        "monthly_auc_diagnostics": diagnostics,
        "hac_lag": int(hac["lag"]),
    }
    primary["verdict"] = verdict_from_primary(primary)
    return primary


def compute_secondary_metrics(outcome, freeze=None):
    """Predeclared evaluation diagnostics. No recalibration or cutoff search."""
    pooled = classification_metrics(outcome, "prediction", "actual")
    calibration = calibration_table(outcome, "prediction", "actual")
    base_rate = pooled.get("base_rate")
    y = pd.to_numeric(outcome["actual"], errors="coerce").to_numpy(dtype="float64")
    p = np.clip(outcome["prediction"].to_numpy(dtype="float64"), 1e-9, 1 - 1e-9)
    brier_skill = None
    log_loss_skill = None
    if base_rate is not None:
        base_brier = float(np.mean((base_rate - y) ** 2))
        if pooled.get("brier") is not None:
            brier_skill = float(base_brier - pooled["brier"])
        benchmark_log_loss = float(-np.mean(
            y * np.log(np.clip(base_rate, 1e-9, 1 - 1e-9))
            + (1.0 - y) * np.log(1.0 - np.clip(base_rate, 1e-9, 1 - 1e-9))
        ))
        if pooled.get("log_loss") is not None:
            log_loss_skill = float(benchmark_log_loss - pooled["log_loss"])
    quintiles = quantile_spread(outcome, "prediction", "actual", buckets=5)
    return {
        "roc_auc": pooled.get("roc_auc"),
        "pr_auc": pooled.get("pr_auc"),
        "pr_auc_benchmark": base_rate,
        "brier": pooled.get("brier"),
        "brier_skill_vs_final_base_rate": brier_skill,
        "log_loss": pooled.get("log_loss"),
        "log_loss_skill_vs_final_base_rate": log_loss_skill,
        "calibration_slope": calibration.get("slope"),
        "calibration_intercept": calibration.get("intercept"),
        "quintile_hit_rate": quintiles,
        "pooled_metrics": pooled,
        "calibration_table": calibration,
        "sampling_note": "evaluation diagnostics only; never recalibrate or search cutoffs",
    }


# ── Immutable result binding ────────────────────────────────────────────────────
def compute_evaluation_id(result_payload):
    payload = {"kind": RESULT_KIND, "payload": result_payload}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "evaluation_" + digest[:12]


def result_hash(result_file):
    return sha256_file(result_file)


def result_dir(root=None, evaluation_id=None):
    base = Path(root or ROOT) / "artifacts" / "research" / "wp8" / "evaluations"
    return base / evaluation_id if evaluation_id else base


def build_result_record(*, evaluation_id, contract_digest, freeze_record, holdout,
                        wp7_candidate, outcome, primary, secondary, diagnostics,
                        producing_commit, code_digest):
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": contract_digest,
        "canonical_wp7_generation_id": CANONICAL_WP7_GENERATION_ID,
        "canonical_wp7_contract": CANONICAL_WP7_CONTRACT,
        "canonical_wp6_experiment_id": CANONICAL_WP6_EXPERIMENT_ID,
        "canonical_final_candidate_freeze_id": freeze_record.get("freeze_id"),
        "canonical_holdout_id": getattr(holdout, "holdout_id", None),
        "frozen_final_config_id": freeze_record.get("final_config_id"),
        "frozen_final_features": list(freeze_record.get("final_selected_features") or []),
        "frozen_final_calibration": freeze_record.get("final_calibration"),
        "producing_commit": producing_commit,
        "code_digest": code_digest,
        "total_eligible_rows": int(len(outcome)),
        "securities": int(outcome["security_id"].nunique()) if "security_id" in outcome.columns else None,
        "panel_diagnostics": diagnostics,
        "primary_metrics": primary,
        "secondary_metrics": secondary,
        "verdict": primary.get("verdict"),
    }
    recomputed = compute_evaluation_id(payload)
    if recomputed != evaluation_id:
        raise Wp8HoldoutEvaluationError(
            "evaluation id is not content-addressed: %s != %s" % (recomputed, evaluation_id)
        )
    payload["evaluation_id"] = evaluation_id
    return payload


def write_immutable_result(record, root=None):
    """Write the ONE immutable content-addressed holdout result. Never overwrite."""
    target = result_dir(root, record["evaluation_id"]) / "result.json"
    if target.exists():
        raise Wp8HoldoutEvaluationError(
            "holdout evaluation result already exists and is immutable: %s" % target
        )
    write_atomic(target, record)
    return target


def write_summary_record(summary, root=None):
    """Write the compact tracked summary; immutable write-once semantics."""
    path = summary_path(root)
    status = save_immutable(path, summary)
    data = path.read_bytes()
    return {
        "path": str(path),
        "status": status,
        "sha256": sha256_bytes(data),
        "sha256_prefix": sha256_bytes(data)[:16],
    }


# ── WP7 certified loader (read-only; called only AFTER all gates) ──────────
def load_wp7_engine(root=None):
    root = Path(root or ROOT)
    path = root / WP7_SCRIPT_REL
    if not path.is_file():
        raise Wp8HoldoutEvaluationError("certified WP7 engine is missing: %s" % path)
    module_name = "wp7_validation_engine_for_wp8_holdout"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


def load_holdout_evaluation_source(root=None):
    """Return the certified panel/diagnostics from the WP5/WP6/WP7 loader.

    This call reads real source data and must be invoked ONLY after the access
    ledger has atomically moved to ``STARTED``.
    """
    wp7 = load_wp7_engine(root)
    return wp7.load_validation_panel()


def source_diagnostics(panel):
    return {
        "rows": int(len(panel)),
        "securities": int(panel["security_id"].nunique()) if "security_id" in panel.columns else None,
        "date_min": str(_utc(panel["feature_asof"]).min().strftime("%Y-%m-%d")) if "feature_asof" in panel.columns else None,
        "date_max": str(_utc(panel["feature_asof"]).max().strftime("%Y-%m-%d")) if "feature_asof" in panel.columns else None,
    }


# ── Preflight orchestration ─────────────────────────────────────────────────────
def run_preflight_gates(root=None, code_digest=None):
    root = Path(root or ROOT)
    producing_commit = validate_clean_producing_worktree(root)
    contract = load_evaluation_contract(root)
    digest = contract_file_digest(root)

    # 1: exact canonical holdout identity; no other id is permitted.
    holdout = load_locked_canonical_holdout(root)

    # 2: canonical WP7 generation must resolve exclusively through the index
    #    and explicitly reject stale model_generation_candidate.json / withdrawn ids.
    candidate, _resolved = canonical_wp7_candidate(root)

    # 3: exact freeze identity and artifact-hash integrity.
    freeze = load_final_candidate_freeze(root)
    verify_frozen_artifact_hashes(freeze, root)

    # 4: explicit one-shot authorization.
    authorization = load_authorization(root)
    assert_authorization_ready(authorization)

    # 5: persistent one-shot access ledger must still be untouched.
    ledger = load_access_ledger(root)
    assert_ledger_ready(ledger, contract_digest=digest)

    return {
        "producing_commit": producing_commit,
        "contract": contract,
        "contract_digest": digest,
        "holdout": holdout,
        "wp7_candidate": candidate,
        "freeze": freeze,
        "authorization": authorization,
        "ledger": ledger,
        "code_digest": code_digest,
    }


def _code_digest(root=None):
    scope = [
        root / "scripts/research_v2/wp8_run_holdout_evaluation.py",
        root / "provenance/wp8/holdout_evaluation_contract_v1.json",
        root / "provenance/wp8/final_candidate_freeze.json",
    ]
    root = Path(root or ROOT)
    hasher = hashlib.sha256()
    for path in scope:
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def run_holdout_evaluation(root=None):
    """Execute the ONE-SHOT locked-holdout evaluation, fail-closed end to end.

    This function is the only real-data entry point. It must never be called
    during development and must never be called twice.
    """
    root = Path(root or ROOT)
    code_digest = _code_digest(root)
    gates = run_preflight_gates(root, code_digest=code_digest)

    # BEFORE any data read: atomically transition to STARTED and never retry.
    ledger_path_obj = ledger_path(root)
    started = transition_ledger_to_started(
        gates["ledger"],
        contract_digest=gates["contract_digest"],
        freeze=gates["freeze"],
        code_digest=code_digest,
        producing_commit=gates["producing_commit"],
        starter_payload={"start_reason": "authorized-by-mission-one-shot"},
        path=ledger_path_obj,
        root=root,
    )

    try:
        panel, loader_diagnostics = load_holdout_evaluation_source(root)
        holdout_frame = select_holdout_rows(panel, gates["freeze"], holdout=gates["holdout"])
        bundle = load_frozen_bundle(gates["freeze"], root=root)
        outcome = predict_frozen_holdout(
            holdout_frame,
            gates["freeze"],
            bundle["model"],
            bundle["preprocessor"],
            bundle["calibrator"],
        )
        primary = compute_primary_metrics(outcome, min_observations=HOLDOUT_MIN_OBSERVATIONS)
        secondary = compute_secondary_metrics(outcome, freeze=gates["freeze"])
        diagnostics = source_diagnostics(holdout_frame)

        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RESULT_KIND,
            "contract_version": CONTRACT_VERSION,
            "contract_digest": gates["contract_digest"],
            "canonical_wp7_generation_id": CANONICAL_WP7_GENERATION_ID,
            "canonical_wp7_contract": CANONICAL_WP7_CONTRACT,
            "canonical_wp6_experiment_id": CANONICAL_WP6_EXPERIMENT_ID,
            "canonical_final_candidate_freeze_id": gates["freeze"]["freeze_id"],
            "canonical_holdout_id": gates["holdout"].holdout_id,
            "frozen_final_config_id": gates["freeze"].get("final_config_id"),
            "frozen_final_features": list(gates["freeze"].get("final_selected_features") or []),
            "frozen_final_calibration": gates["freeze"].get("final_calibration"),
            "producing_commit": gates["producing_commit"],
            "code_digest": gates["code_digest"],
            "total_eligible_rows": int(len(outcome)),
            "securities": int(outcome["security_id"].nunique()) if "security_id" in outcome.columns else None,
            "panel_diagnostics": {
                "loader": loader_diagnostics,
                "holdout_frame": diagnostics,
            },
            "primary_metrics": primary,
            "secondary_metrics": secondary,
            "verdict": primary.get("verdict"),
        }
        evaluation_id = compute_evaluation_id(payload)
        record = build_result_record(
            evaluation_id=evaluation_id,
            contract_digest=gates["contract_digest"],
            freeze_record=gates["freeze"],
            holdout=gates["holdout"],
            wp7_candidate=gates["wp7_candidate"],
            outcome=outcome,
            primary=primary,
            secondary=secondary,
            diagnostics={"loader": loader_diagnostics, "holdout_frame": diagnostics},
            producing_commit=gates["producing_commit"],
            code_digest=code_digest,
        )
        result_file = write_immutable_result(record, root)
        result_sha = result_hash(result_file)

        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "contract_digest": gates["contract_digest"],
            "canonical_final_candidate_freeze_id": gates["freeze"]["freeze_id"],
            "canonical_wp7_generation_id": CANONICAL_WP7_GENERATION_ID,
            "canonical_holdout_id": gates["holdout"].holdout_id,
            "producing_commit": gates["producing_commit"],
            "code_digest": code_digest,
            "result_sha256": result_sha,
            "access_count": 1,
            "primary_metrics": primary,
            "verdict": primary.get("verdict"),
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        summary_written = write_summary_record(summary, root)

        completed = transition_ledger_to_completed(
            started,
            evaluation_id=evaluation_id,
            result_hash=result_sha,
            root=root,
            path=ledger_path_obj,
        )
        return {
            "evaluation_id": evaluation_id,
            "result_file": str(result_file),
            "result_sha256": result_sha,
            "summary": summary_written,
            "ledger": completed,
            "primary_metrics": primary,
            "verdict": primary.get("verdict"),
        }
    except BaseException:
        # One-shot rule: leave STARTED and NEVER retry/touch the data again.
        raise


def main(argv=None):
    if "--dry-run-preflight" in (argv or []):
        result = run_preflight_gates()
        print("PREFLIGHT_PASS holdout=%s freeze=%s" % (
            result["holdout"].holdout_id, result["freeze"]["freeze_id"]
        ))
        return result
    result = run_holdout_evaluation()
    print("EVALUATION_ID %s verdict=%s" % (
        result["evaluation_id"], result["verdict"]
    ))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
