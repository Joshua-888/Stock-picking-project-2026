"""WP7 PRE-HOLDOUT nested walk-forward validation engine.

Deterministic implementation of ``provenance/wp7/validation_contract_v4.json``.
The engine is development-only and never accesses locked-holdout performance,
labels, or any row whose ``feature_asof >= 2021-01-01``.

Run with the research runtime:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp7_validation.py

WP6 building blocks are reused read-only. WP7-specific geometry, selection,
calibration, controls, and artifact writing are self-contained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.discovery.multiple_testing import benjamini_hochberg
from src.research.holdout import locked_holdout
from src.research.ids import canonical_json
from src.research.immutability import save_immutable
from src.research.modes import ResearchMode, current_git_commit
from src.research.modeling import models as model_registry
from src.research.modeling.contract import DEFAULT_CONFIG as WP6_DEFAULT_CONFIG
from src.research.modeling.contract import model_feature_universe
from src.research.modeling.features import select_features, selection_stability
from src.research.modeling.folds import build_folds
from src.research.modeling.metrics import (
    calibration_table,
    classification_metrics,
    monthly_auc_series,
    monthly_brier_series,
)
from src.research.modeling.placebo import control_object, overall_stop
from src.research.modeling.placebo import shuffle_training_target
from src.research.modeling.preprocessing import PreprocessingSpec, Preprocessor

CONTRACT_REL = Path("provenance") / "wp7" / "validation_contract_v4.json"
CONTRACT_MD_REL = Path("docs") / "research_v2" / "wp7_validation_contract_v4.md"
CANONICAL_HOLDOUT_ID = "holdout_7ce54e933e16"
GENERATION_KIND = "wp7_model_generation_candidate"
MODEL_CANDIDATE_DIR_REL = Path("provenance") / "wp7" / "candidates"
# Deprecated fixed-path constant retained only for test clarity/introspection.
MODEL_CANDIDATE_REL = MODEL_CANDIDATE_DIR_REL / "<generation_id>.json"

TARGET = "outperform_12m"
CLASSIFICATION_FAMILY_ID = "wp7_classification_predictive_skill"
CALIBRATION_CANDIDATES = ("none", "platt_scaling", "isotonic_regression")
INNER_MIN_OBSERVATIONS = 30
CONTROL_SKILL_FLOOR = 0.05


class Wp7ValidationError(RuntimeError):
    """Raised when WP7 cannot proceed honestly under the frozen contract."""


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
        raise Wp7ValidationError("git is unavailable for WP7 reproducibility preflight: %s" % exc) from exc
    if completed.returncode != 0:
        raise Wp7ValidationError(
            "git command failed during WP7 preflight: %s" % " ".join(args)
        )
    output = completed.stdout or ""
    return output.strip() if strip else output


def _is_producing_code_path(path):
    """True when a modified tracked path can change the WP7 producing code/contract."""
    normalized = path.replace("\\", "/")
    if normalized == "scripts/research_v2/wp7_validation.py":
        return True
    if normalized == "provenance/wp7/validation_contract_v4.json":
        return True
    return normalized.startswith("src/research/") or normalized.startswith(
        "scripts/research_v2/"
    )


def _dirty_non_ignored_paths(root):
    # ``git status --porcelain --untracked-files=no`` reports only tracked-file
    # modifications; ignored/untracked files such as ``.a0proj/`` and ignored
    # artifacts are deliberately allowed.
    output = _git_output(
        ["status", "--porcelain", "--untracked-files=no"], root, strip=False
    )
    paths = set()
    for line in output.splitlines():
        if len(line) < 4:
            continue
        # porcelain format is ``XY PATH`` (plus rename arrows after the path);
        # with untracked files disabled every line is a tracked working-tree change.
        path = line[3:].strip()
        if path:
            paths.add(path)
    return sorted(paths)


def validate_clean_producing_worktree(root=None):
    """Fail when a tracked producing-code file is modified before an honest WP7 run.

    The resolved HEAD is returned as the producing commit. Uncommitted producing
    code/contract changes mean the HEAD hash would not contain the code that
    actually produced results. Unrelated tracked files are not required to be
    clean, and ignored/untracked files are always allowed.
    """
    root = Path(root or ROOT)
    commit = current_git_commit(str(root))
    if not commit:
        raise Wp7ValidationError("no git commit available for reproducible provenance")
    dirty = [
        path for path in _dirty_non_ignored_paths(root) if _is_producing_code_path(path)
    ]
    if dirty:
        raise Wp7ValidationError(
            "WP7 refusing to run with modified tracked producing-code files; producing "
            "commit %s would not contain the actual working-tree code. Dirty paths: %s"
            % (commit, ", ".join(dirty))
        )
    return commit


@dataclass(frozen=True)
class FrozenConfig:
    config_id: str
    model: str
    strategy: str | None
    preprocessing: tuple
    params: dict


def load_contract(root=None):
    root = Path(root or ROOT)
    path = root / CONTRACT_REL
    if not path.is_file():
        raise Wp7ValidationError("frozen WP7 validation contract is missing: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp7ValidationError("frozen WP7 contract is not valid JSON: %s" % exc) from exc


def contract_digest(contract):
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def parse_frozen_configs(contract):
    eligible = contract.get("eligible_model_set") or {}
    baseline = eligible.get("baseline")
    entries = list(eligible.get("configs") or [])
    if not baseline or not entries:
        raise Wp7ValidationError("frozen contract has no eligible model set")
    parsed = [
        FrozenConfig(
            config_id=str(baseline["config_id"]),
            model=str(baseline["model"]),
            strategy=None,
            preprocessing=tuple(baseline.get("preprocessing") or ()),
            params=dict(baseline.get("params") or {}),
        )
    ]
    for entry in entries:
        parsed.append(FrozenConfig(
            config_id=str(entry["config_id"]),
            model=str(entry["model"]),
            strategy=str(entry.get("strategy")) if entry.get("strategy") else None,
            preprocessing=tuple(entry.get("preprocessing") or ()),
            params=dict(entry.get("params") or {}),
        ))
    identifiers = [item.config_id for item in parsed]
    if len(set(identifiers)) != len(identifiers):
        raise Wp7ValidationError("frozen contract contains duplicate config_ids")
    return parsed


def frozen_feature_universe(contract):
    universe = list(contract.get("feature_universe") or [])
    if universe != list(model_feature_universe()):
        raise Wp7ValidationError(
            "frozen contract feature_universe does not match the WP6 model universe"
        )
    return tuple(universe)


def wp7_inner_config():
    return replace(
        WP6_DEFAULT_CONFIG,
        validation_window_months=12,
        min_train_months=37,
        horizon_months=12,
    )


def _utc_values(frame, column):
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _timestamp_start(month_str):
    return pd.Timestamp("%s-01" % month_str).tz_localize("UTC")


def _month_position_map(frame):
    months = sorted(frame["modeling_month"].astype(str).unique())
    return months, {month: index for index, month in enumerate(months)}


def derive_outer_geometry(frame_or_months, contract):
    """Serialize the exact frozen outer windows and verify their indices."""
    frozen = list(
        (contract.get("outer_walk_forward_design") or {}).get("outer_test_windows") or []
    )
    if len(frozen) != 4:
        raise Wp7ValidationError("frozen outer design must have exactly 4 windows")

    if isinstance(frame_or_months, pd.DataFrame):
        months, observed = _month_position_map(frame_or_months)
    else:
        months = sorted(str(value) for value in frame_or_months)
        observed = {month: index for index, month in enumerate(months)}

    windows = []
    for window in frozen:
        start = str(window["start"])
        end = str(window["end"])
        start_index = int(window["start_index"])
        end_index = int(window["end_index"])
        if observed.get(start) != start_index:
            raise Wp7ValidationError(
                "outer window %s start %s does not match observed index"
                % (window.get("window_id"), start)
            )
        if observed.get(end) != end_index:
            raise Wp7ValidationError(
                "outer window %s end %s does not match observed index"
                % (window.get("window_id"), end)
            )
        windows.append({
            "window_id": str(window["window_id"]),
            "start": start,
            "end": end,
            "start_index": start_index,
            "end_index": end_index,
            "outer_test_start_timestamp": str(_timestamp_start(start)),
        })
    return {
        "observed_month_count": len(months),
        "expected_month_count": 168,
        "windows": windows,
        "raw_inner_fold_counts_before_last_window_exclusion":
            list(contract["outer_walk_forward_design"][
                "raw_inner_fold_counts_before_last_window_exclusion"]),
        "safe_inner_fold_counts_by_outer_fold":
            list(contract["outer_walk_forward_design"][
                "safe_inner_fold_counts_by_outer_fold"]),
        "frozen_windows": [
            {key: value for key, value in item.items() if key not in ("start_index", "end_index")}
            for item in frozen
        ],
    }


def assert_development_only(frame):
    stamps = _utc_values(frame, "feature_asof")
    limit = pd.Timestamp("2021-01-01").tz_localize("UTC")
    holdout = locked_holdout()
    if bool((stamps >= limit).any()):
        raise Wp7ValidationError("frame contains feature_asof >= 2021-01-01")
    if bool((stamps >= holdout.embargo_cutoff).any()):
        raise Wp7ValidationError("frame contains embargo-band rows")
    return True


def _frame_trainable_before(frame, cutoff):
    feature_dt = _utc_values(frame, "feature_asof")
    known_dt = _utc_values(frame, "target_known_at")
    end_dt = _utc_values(frame, "target_end")
    return (
        (feature_dt < cutoff)
        & known_dt.notna()
        & (known_dt <= cutoff)
        & end_dt.notna()
        & (end_dt < cutoff)
    ).to_numpy()


def build_inner_folds(outer_train_frame, holdout=None):
    holdout = holdout or locked_holdout()
    assert_development_only(outer_train_frame)
    folds, diagnostics, working = build_folds(
        outer_train_frame, config=wp7_inner_config(), holdout=holdout
    )
    return folds, diagnostics, working


def safe_inner_folds(folds, working, outer_test_start_ts):
    safe, excluded = [], []
    for fold in folds:
        validation_frame = working.iloc[list(fold.validation_index)]
        if validation_frame.empty:
            excluded.append({"fold": fold.fold, "reason": "empty"})
            continue
        max_target_end = _utc_values(validation_frame, "target_end").max()
        if max_target_end is not None and max_target_end >= outer_test_start_ts:
            excluded.append({
                "fold": fold.fold,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "max_target_end": str(max_target_end)[:10],
                "reason": "label_overlaps_outer_test_start",
            })
            continue
        safe.append(fold)
    return safe, excluded


def _build_fold_frames(working, folds):
    return {
        int(fold.fold): {
            "train": working.iloc[list(fold.train_index)].reset_index(drop=True),
            "validation": working.iloc[list(fold.validation_index)].reset_index(drop=True),
        }
        for fold in folds
    }


def _feature_names_from_strategy(strategy, train_frame, contract):
    if strategy is None:
        return [], {"basis": "baseline_no_feature_model"}
    features, evidence = select_features(strategy, train_frame, config=WP6_DEFAULT_CONFIG)
    eligible = set(frozen_feature_universe(contract))
    features = [name for name in features if name in eligible]
    if not features:
        raise Wp7ValidationError(
            "feature strategy %s returned no eligible features on this train slice" % strategy
        )
    return features, evidence


def _finite_binary(train_frame):
    values = pd.to_numeric(train_frame[TARGET], errors="coerce")
    return values[np.isfinite(values.to_numpy(dtype="float64"))
                  & np.isin(values.to_numpy(dtype="float64"), (0.0, 1.0))]


def outcome_frame(frame, prediction):
    return pd.DataFrame({
        "modeling_month": frame["modeling_month"].to_numpy(),
        "prediction": np.asarray(prediction, dtype="float64"),
        "actual": pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype="float64"),
        "security_id": frame["security_id"].to_numpy(),
        "feature_asof": frame["feature_asof"].to_numpy(),
    })


def _baseline_predict(train_frame, val_frame):
    y_train = _finite_binary(train_frame)
    if len(y_train) == 0:
        raise Wp7ValidationError("baseline_base_rate has no usable training rows")
    return outcome_frame(val_frame, np.full(len(val_frame), float(y_train.mean())))


def _preprocess_train_predict(train_frame, val_frame, spec, params, seed, features,
                              extra_features=()):
    use_features = list(features) + list(extra_features)
    if not use_features:
        raise Wp7ValidationError("no features supplied to feature-based model")
    preprocessor = Preprocessor(PreprocessingSpec(**model_registry.VALUE_SPEC))
    fitted = preprocessor.fit(train_frame, use_features)
    x_train = preprocessor.transform(train_frame, fitted)
    x_val = preprocessor.transform(val_frame, fitted)
    y_train = _finite_binary(train_frame)
    if len(y_train) == 0:
        raise Wp7ValidationError("no usable training rows for model %s" % spec.name)
    x_train = x_train.loc[y_train.index]
    estimator = model_registry.classification_estimator(
        spec, params, seed=int(seed), columns=use_features
    )
    estimator.fit(x_train, y_train.to_numpy(dtype="float64"))
    probabilities = estimator.predict_proba(x_val)
    return outcome_frame(val_frame, probabilities[:, 1]), fitted, use_features


def evaluate_config_on_inner_folds(candidate, fold_frames, folds, contract, seed):
    outcomes = []
    per_fold = {}
    for fold in folds:
        train_frame = fold_frames[fold.fold]["train"]
        val_frame = fold_frames[fold.fold]["validation"]
        features, evidence = _feature_names_from_strategy(candidate.strategy, train_frame, contract)
        if candidate.model == "baseline_base_rate":
            outcome = _baseline_predict(train_frame, val_frame)
        else:
            spec = model_registry.MODELS_BY_NAME[candidate.model]
            outcome, _fitted, _used = _preprocess_train_predict(
                train_frame, val_frame, spec, candidate.params, seed, features
            )
        outcomes.append(outcome)
        series, _diag = monthly_auc_series(
            outcome, "prediction", "actual", min_observations=INNER_MIN_OBSERVATIONS
        )
        per_fold[fold.fold] = {
            "months": int(len(series)),
            "mean_auc": float(series["auc"].mean()) if len(series) else None,
            "mean_skill": float(series["auc"].mean() - 0.5) if len(series) else None,
            "features": list(features),
            "feature_evidence": evidence,
        }
    mean_skill, series = _mean_auc_skill_from_outcomes(outcomes)
    return {
        "config_id": candidate.config_id,
        "model": candidate.model,
        "strategy": candidate.strategy,
        "params": dict(candidate.params),
        "preprocessing": list(candidate.preprocessing),
        "outcomes": outcomes,
        "mean_auc_skill": mean_skill,
        "mean_auc_skill_series": series,
        "per_fold": per_fold,
    }


def _mean_auc_skill_from_outcomes(outcomes):
    combined = pd.concat([item for item in outcomes if len(item)], ignore_index=True)
    if combined.empty:
        return None, pd.DataFrame(columns=["month", "paired_obs", "auc"])
    series, _diag = monthly_auc_series(
        combined, "prediction", "actual", min_observations=INNER_MIN_OBSERVATIONS
    )
    if series.empty:
        return None, series
    return float(series["auc"].mean()) - 0.5, series


def select_model_candidate(evaluations):
    for item in evaluations:
        item["_sort_skill"] = -np.inf if item["mean_auc_skill"] is None \
            else float(item["mean_auc_skill"])
    return min(
        evaluations,
        key=lambda item: (
            -item["_sort_skill"],
            len(item.get("params") or {}),
            str(item["config_id"]),
        ),
    )


def _fold_brier_benchmark(outcome, train_frame):
    y = pd.to_numeric(outcome["actual"], errors="coerce").to_numpy(dtype="float64")
    p = np.asarray(outcome["prediction"], dtype="float64")
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return None, None, None
    train_y = _finite_binary(train_frame)
    if len(train_y) == 0:
        return None, None, None
    base_rate = float(train_y.mean())
    model_brier = float(np.mean((p - y) ** 2))
    base_brier = float(np.mean((base_rate - y) ** 2))
    return base_rate, model_brier, base_brier


def _fit_calibration(probabilities, labels, method):
    p = np.asarray(probabilities, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    if method == "none":
        return None
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0 or len(np.unique(y)) < 2:
        return None
    if method == "platt_scaling":
        from sklearn.linear_model import LogisticRegression

        logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(logit.reshape(-1, 1), y)
        return model
    if method == "isotonic_regression":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    raise Wp7ValidationError("unknown calibration method %r" % method)


def _apply_calibration(probabilities, fitted, method):
    p = np.asarray(probabilities, dtype="float64")
    if method == "none" or fitted is None:
        return p
    if method == "platt_scaling":
        logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        return fitted.predict_proba(logit.reshape(-1, 1))[:, 1]
    if method == "isotonic_regression":
        return fitted.predict(p)
    return p


def select_calibration(evaluated, fold_frames, safe_fold_ids):
    outcomes_by_fold = {
        fold_id: outcome
        for fold_id, outcome in zip(safe_fold_ids, evaluated["outcomes"])
    }
    ranked = []
    for method_index, method in enumerate(CALIBRATION_CANDIDATES):
        skills = []
        usable = False
        for fold_id in safe_fold_ids:
            other_ids = [item for item in safe_fold_ids if item != fold_id]
            other = pd.concat([outcomes_by_fold[item] for item in other_ids], ignore_index=True)
            if other.empty:
                continue
            fitted = _fit_calibration(
                other["prediction"].to_numpy(dtype="float64"),
                other["actual"].to_numpy(dtype="float64"),
                method,
            )
            current = outcomes_by_fold[fold_id].copy()
            current["prediction"] = _apply_calibration(
                current["prediction"].to_numpy(dtype="float64"), fitted, method
            )
            _base_rate, model_brier, base_brier = _fold_brier_benchmark(
                current, fold_frames[fold_id]["train"]
            )
            if model_brier is None or base_brier is None:
                continue
            usable = True
            skills.append(float(base_brier - model_brier))
        mean_skill = float(np.mean(skills)) if skills else None
        ranked.append({
            "method": method,
            "mean_brier_skill": mean_skill,
            "fold_skills": skills,
            "usable": usable,
            "method_index": method_index,
        })
    selected = min(
        ranked,
        key=lambda item: (
            -np.inf if item["mean_brier_skill"] is None else -item["mean_brier_skill"],
            item["method_index"],
        ),
    )
    return {
        "selected": selected["method"],
        "skills": {
            item["method"]: {
                "mean_brier_skill": item["mean_brier_skill"],
                "fold_skills": item["fold_skills"],
                "usable": item["usable"],
            }
            for item in ranked
        },
        "tie_break": "deterministic_earliest_candidate_order",
    }


def _fit_selected_model_on_frame(train_frame, selected, seed, contract):
    features, evidence = _feature_names_from_strategy(
        selected.strategy, train_frame, contract
    )
    if selected.model == "baseline_base_rate":
        y_train = _finite_binary(train_frame)
        if len(y_train) == 0:
            raise Wp7ValidationError("baseline_base_rate has no usable training rows")
        return {
            "kind": "baseline_base_rate",
            "base_rate": float(y_train.mean()),
            "features": [],
            "evidence": evidence,
        }

    use_features = list(features)
    preprocessor = Preprocessor(PreprocessingSpec(**model_registry.VALUE_SPEC))
    fitted = preprocessor.fit(train_frame, use_features)
    x_train = preprocessor.transform(train_frame, fitted)
    y_train = _finite_binary(train_frame)
    x_train = x_train.loc[y_train.index]
    estimator = model_registry.classification_estimator(
        model_registry.MODELS_BY_NAME[selected.model],
        selected.params,
        seed=int(seed),
        columns=use_features,
    )
    estimator.fit(x_train, y_train.to_numpy(dtype="float64"))
    return {
        "kind": "estimator",
        "estimator": estimator,
        "preprocessor": preprocessor,
        "fitted": fitted,
        "features": features,
        "evidence": evidence,
    }


def _prediction_for_frame(model_obj, frame, calibration_method, calibration_fit):
    if model_obj["kind"] == "baseline_base_rate":
        prediction = np.full(len(frame), model_obj["base_rate"], dtype="float64")
    else:
        x = model_obj["preprocessor"].transform(frame, model_obj["fitted"])
        probabilities = model_obj["estimator"].predict_proba(x)
        prediction = np.asarray(probabilities[:, 1], dtype="float64")
    prediction = _apply_calibration(prediction, calibration_fit, calibration_method)
    return outcome_frame(frame, prediction)


def _brier_skill(outcome, train_frame):
    _base_rate, model_brier, base_brier = _fold_brier_benchmark(outcome, train_frame)
    return None if model_brier is None or base_brier is None else float(base_brier - model_brier)


def _outer_metrics(outer_outcome, outer_train):
    series_auc, _diag = monthly_auc_series(
        outer_outcome, "prediction", "actual", min_observations=1
    )
    series_brier, _diag = monthly_brier_series(
        outer_outcome, "prediction", "actual", min_observations=1
    )
    pooled = classification_metrics(outer_outcome, "prediction", "actual")
    calibration = calibration_table(outer_outcome, "prediction", "actual")
    base_rate, model_brier, base_brier = _fold_brier_benchmark(
        outer_outcome, outer_train
    )
    auc = series_auc["auc"].to_numpy(dtype="float64") if len(series_auc) else np.array([])
    skill = auc - 0.5 if len(auc) else np.array([])
    return {
        "pooled_metrics": pooled,
        "calibration": calibration,
        "months": int(len(series_auc)),
        "mean_auc": float(np.mean(auc)) if len(auc) else None,
        "mean_auc_skill": float(np.mean(skill)) if len(skill) else None,
        "positive_month_fraction": float(np.mean(skill > 0)) if len(skill) else None,
        "monthly_auc": series_auc.to_dict(orient="records"),
        "monthly_brier": series_brier.to_dict(orient="records"),
        "model_brier": model_brier,
        "train_base_rate_brier": base_brier,
        "brier_skill_vs_train_base_rate": None if model_brier is None or base_brier is None
        else float(base_brier - model_brier),
        "train_base_rate": base_rate,
    }


def _multiple_testing_report(candidate_scores):
    rows = pd.DataFrame(candidate_scores)
    if rows.empty:
        return {
            "family_id": CLASSIFICATION_FAMILY_ID,
            "number_of_hypotheses": 0,
            "raw_p_by_config": {},
            "q_values": {},
            "fdr_rejected": {},
            "summary": {},
        }
    raw = {}
    for config_id, chunk in rows.groupby("config_id", sort=False):
        values = chunk["mean_auc_skill"].dropna().to_numpy(dtype="float64")
        if len(values) < 2:
            raw[config_id] = None
            continue
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if se <= 0.0:
            raw[config_id] = 0.0 if mean != 0.0 else 1.0
            continue
        from scipy import stats

        raw[config_id] = float(2.0 * stats.norm.sf(abs(mean / se)))

    finite = {key: float(value) for key, value in raw.items() if value is not None}
    results, summary = benjamini_hochberg(finite, alpha=0.05)
    q_values = {record.feature: record.q_value for record in results}
    rejected = {record.feature: record.rejected for record in results}
    return {
        "family_id": CLASSIFICATION_FAMILY_ID,
        "scientific_question":
            "Do frozen WP7 classification configurations show inner walk-forward AUC skill beyond chance?",
        "metric": "auc",
        "null": "H0: mean inner validation (AUC_t - 0.5) = 0",
        "number_of_hypotheses": int(len(finite)),
        "raw_p_by_config": {key: (None if key not in finite else finite[key])
                            for key in sorted(raw)},
        "q_values": q_values,
        "fdr_rejected": rejected,
        "summary": summary,
        "claim_restriction": "selection-adjusted nested estimate; NOT unbiased holdout evidence",
    }


def _robustness_report(outer_records, selected_models):
    skills = [item["outer_metrics"]["mean_auc_skill"] for item in outer_records]
    finite = [value for value in skills if value is not None]
    cut = int(len(finite) // 2)
    first, second = finite[:cut], finite[cut:]
    config_selections = [item["selected_config_id"] for item in outer_records]
    calibration_selections = [item["selected_calibration"] for item in outer_records]
    parameter_selections = [
        json.dumps(item["selected_params"], sort_keys=True) for item in outer_records
    ]
    feature_selections = {
        window_id: sorted(entry["features"])
        for window_id, entry in selected_models.items()
    }
    return {
        "outer_fold_stability": {
            "fold_auc_skills": skills,
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "positive_fold_fraction": float(np.mean(np.asarray(finite) > 0))
            if finite else None,
        },
        "subperiod_stability": {
            "first_half_mean": float(np.mean(first)) if first else None,
            "second_half_mean": float(np.mean(second)) if second else None,
            "sign_flip": bool(np.sign(np.mean(first)) != np.sign(np.mean(second)))
            if first and second else None,
        },
        "model_selection_stability": {
            "selections": config_selections,
            "unique_selected_configs": sorted(set(config_selections)),
            "unique_count": len(set(config_selections)),
        },
        "feature_selection_stability": selection_stability(feature_selections),
        "hyperparameter_stability": {
            "selections": parameter_selections,
            "unique_count": len(set(parameter_selections)),
        },
        "calibration_stability": {
            "selections": calibration_selections,
            "unique_count": len(set(calibration_selections)),
        },
        "complexity_vs_simple_baseline": {
            "note": "compared to baseline_base_rate on development outer_test windows",
        },
        "negative_controls": "computed separately in controls/control_evidence",
    }


def _noise_feature(frame, seed, name="__noise_control"):
    rng = np.random.default_rng(int(seed))
    working = frame.copy()
    working[name] = rng.standard_normal(len(working))
    return working


def _run_negative_controls_for_fold(evaluated, fold_frames, safe_folds, candidate,
                                    seed, contract):
    real_skill = evaluated["mean_auc_skill"]
    shuffled_skills, noise_skills = [], []
    spec = None if candidate.model == "baseline_base_rate" else \
        model_registry.MODELS_BY_NAME[candidate.model]
    for fold in safe_folds:
        train_frame = fold_frames[fold.fold]["train"]
        val_frame = fold_frames[fold.fold]["validation"]
        features, _evidence = _feature_names_from_strategy(
            candidate.strategy, train_frame, contract
        )
        shuffled_train = shuffle_training_target(
            train_frame, target=TARGET, seed=int(seed), asof_col="modeling_month"
        )
        if candidate.model == "baseline_base_rate":
            shuffled_outcome = _baseline_predict(shuffled_train, val_frame)
        else:
            shuffled_outcome, _fit, _used = _preprocess_train_predict(
                shuffled_train, val_frame, spec, candidate.params, seed, features
            )
        skill, _series = _mean_auc_skill_from_outcomes([shuffled_outcome])
        if skill is not None:
            shuffled_skills.append(skill)

        noisy_train = _noise_feature(train_frame, int(seed) + 1)
        noisy_val = _noise_feature(val_frame, int(seed) + 1)
        if candidate.model == "baseline_base_rate":
            noisy_outcome = _baseline_predict(noisy_train, noisy_val)
        else:
            noisy_outcome, _fit, _used = _preprocess_train_predict(
                noisy_train, noisy_val, spec, candidate.params, seed,
                features, extra_features=("__noise_control",),
            )
        skill, _series = _mean_auc_skill_from_outcomes([noisy_outcome])
        if skill is not None:
            noise_skills.append(skill)
    return {
        "real_skill": None if real_skill is None else float(real_skill),
        "shuffled_skill": float(np.mean(shuffled_skills)) if shuffled_skills else None,
        "noise_skill": float(np.mean(noise_skills)) if noise_skills else None,
        "shuffled_fold_skills": shuffled_skills,
        "noise_fold_skills": noise_skills,
    }


def evaluate_negative_control(control_id, control_type, real_metric, control_metric):
    real_magnitude = abs(float(real_metric)) if real_metric is not None else 0.0
    threshold = max(real_magnitude, CONTROL_SKILL_FLOOR)
    if control_metric is None:
        passed, observed = False, "control metric unavailable"
    else:
        magnitude = abs(float(control_metric))
        passed = bool(magnitude < threshold)
        observed = "control AUC-skill magnitude = %.6f" % magnitude
    return control_object(
        control_id=control_id,
        control_type=control_type,
        expected_behavior="negative control scores near chance (AUC skill ~ 0)",
        observed_behavior=observed,
        matched_real_config_id=None,
        real_metric=None if real_metric is None else float(real_metric),
        control_metric=None if control_metric is None else float(control_metric),
        failure_threshold=float(threshold),
        failure_condition="abs(control_metric) >= max(abs(real_metric), floor)",
        passed=passed,
        stop_required=bool(not passed),
        reason="control within frozen tolerance under the null" if passed
        else "control metric reached matched real magnitude / frozen tolerance",
        mandatory=True,
        affected_tasks=("classification",),
    )


def build_negative_controls(negative_records):
    real = [item["real_skill"] for item in negative_records
            if item.get("real_skill") is not None]
    shuffled = [item["shuffled_skill"] for item in negative_records
                if item.get("shuffled_skill") is not None]
    noise = [item["noise_skill"] for item in negative_records
             if item.get("noise_skill") is not None]
    real_mean = float(np.mean(real)) if real else None
    shuffled_mean = float(np.mean(shuffled)) if shuffled else None
    noise_mean = float(np.mean(noise)) if noise else None
    controls = [
        evaluate_negative_control(
            "shuffled_target_classification", "shuffled_target",
            real_mean, shuffled_mean,
        ),
        evaluate_negative_control(
            "noise_feature_classification", "noise_feature",
            real_mean, noise_mean,
        ),
    ]
    return controls, overall_stop(controls), {
        "real_skill_mean": real_mean,
        "shuffled_skill_mean": shuffled_mean,
        "noise_skill_mean": noise_mean,
        "per_selected_fold": negative_records,
    }


def run_pre_holdout_validation(frame, contract=None, commit=None, holdout=None):
    contract = contract or load_contract()
    holdout = holdout or locked_holdout()
    if contract.get("eligible_task", {}).get("task") != "classification":
        raise Wp7ValidationError("WP7 v3 eligible task is not classification")
    if contract.get("target") != TARGET:
        raise Wp7ValidationError("WP7 v3 target mismatch")
    assert_development_only(frame)

    started = time.time()
    configs = parse_frozen_configs(contract)
    geometry = derive_outer_geometry(frame, contract)
    seed = int(contract["seeds"]["model_seed"])
    months, _positions = _month_position_map(frame)

    outer_records = []
    selected_models = {}
    negative_records = []
    candidate_scores = []
    outer_test_predictions = []

    for window_number, window in enumerate(geometry["windows"]):
        start, end = window["start"], window["end"]
        outer_start_ts = _timestamp_start(start)
        outer_test = frame.loc[
            frame["modeling_month"].astype(str).isin(
                [month for month in months if start <= month <= end]
            )
        ].reset_index(drop=True)
        outer_before = frame.loc[
            frame["modeling_month"].astype(str) < start
        ].reset_index(drop=True)
        final_train_mask = _frame_trainable_before(outer_before, outer_start_ts)
        final_train = outer_before.loc[final_train_mask].reset_index(drop=True)
        if outer_test.empty or final_train.empty:
            raise Wp7ValidationError("empty outer segment for %s" % window["window_id"])

        inner_folds, _inner_diag, inner_working = build_inner_folds(
            outer_before, holdout=holdout
        )
        safe_folds, excluded = safe_inner_folds(inner_folds, inner_working, outer_start_ts)
        expected_raw = geometry["raw_inner_fold_counts_before_last_window_exclusion"][window_number]
        expected_safe = geometry["safe_inner_fold_counts_by_outer_fold"][window_number]
        if len(inner_folds) != expected_raw:
            raise Wp7ValidationError(
                "%s raw inner folds %d != expected %d"
                % (window["window_id"], len(inner_folds), expected_raw)
            )
        if len(safe_folds) != expected_safe:
            raise Wp7ValidationError(
                "%s safe inner folds %d != expected %d"
                % (window["window_id"], len(safe_folds), expected_safe)
            )
        fold_frames = _build_fold_frames(inner_working, safe_folds)

        evaluations = [
            evaluate_config_on_inner_folds(candidate, fold_frames, safe_folds, contract, seed)
            for candidate in configs
        ]
        selected_evaluation = select_model_candidate(evaluations)
        selected_config = next(
            item for item in configs
            if item.config_id == selected_evaluation["config_id"]
        )
        calibration = select_calibration(
            selected_evaluation, fold_frames, [fold.fold for fold in safe_folds]
        )
        inner_combined = pd.concat(selected_evaluation["outcomes"], ignore_index=True)
        calibration_fit = _fit_calibration(
            inner_combined["prediction"].to_numpy(dtype="float64"),
            inner_combined["actual"].to_numpy(dtype="float64"),
            calibration["selected"],
        )
        final_model = _fit_selected_model_on_frame(
            final_train, selected_config, seed, contract
        )
        # Fit calibration on the selected model's inner OOF predictions, then
        # apply it to outer_test probabilities only.
        outer_outcome_raw = _prediction_for_frame(
            final_model, outer_test, "none", None
        )
        outer_outcome = outer_outcome_raw.copy()
        outer_outcome["prediction"] = _apply_calibration(
            outer_outcome["prediction"].to_numpy(dtype="float64"),
            calibration_fit,
            calibration["selected"],
        )

        outer_metrics = _outer_metrics(outer_outcome, final_train)
        negative = _run_negative_controls_for_fold(
            selected_evaluation, fold_frames, safe_folds, selected_config,
            int(contract["seeds"]["placebo_seed"]), contract
        )
        negative_records.append(negative)

        per_outer_prediction = outer_outcome.copy()
        per_outer_prediction["window_id"] = window["window_id"]
        outer_test_predictions.append(per_outer_prediction)
        outer_records.append({
            "window_id": window["window_id"],
            "outer_test_start": start,
            "outer_test_end": end,
            "outer_test_rows": int(len(outer_test)),
            "outer_train_rows_after_purge": int(len(final_train)),
            "raw_inner_folds": int(len(inner_folds)),
            "safe_inner_folds": int(len(safe_folds)),
            "excluded_inner_folds": excluded,
            "selected_config_id": selected_config.config_id,
            "selected_model": selected_config.model,
            "selected_strategy": selected_config.strategy,
            "selected_params": dict(selected_config.params),
            "selected_preprocessing": list(selected_config.preprocessing),
            "selected_calibration": calibration["selected"],
            "selected_features": final_model["features"],
            "feature_evidence": final_model["evidence"],
            "selection_mean_auc_skill": selected_evaluation["mean_auc_skill"],
            "selected_allocation_brier_skill": _brier_skill(
                inner_combined, final_train
            ),
            "calibration_skills": calibration["skills"],
            "inner_selection_table": [
                {
                    "config_id": item["config_id"],
                    "model": item["model"],
                    "strategy": item["strategy"],
                    "params": item["params"],
                    "mean_auc_skill": item["mean_auc_skill"],
                    "per_fold": item["per_fold"],
                }
                for item in evaluations
            ],
            "outer_metrics": outer_metrics,
        })
        selected_models[window["window_id"]] = {
            "config_id": selected_config.config_id,
            "model": selected_config.model,
            "strategy": selected_config.strategy,
            "params": dict(selected_config.params),
            "preprocessing": list(selected_config.preprocessing),
            "calibration": calibration["selected"],
            "features": final_model["features"],
            "mean_auc_skill": selected_evaluation["mean_auc_skill"],
        }
        candidate_scores.extend(
            {
                "config_id": item["config_id"],
                "window_id": window["window_id"],
                "mean_auc_skill": item["mean_auc_skill"],
            }
            for item in evaluations
        )

    controls, stop_summary, control_evidence = build_negative_controls(negative_records)
    status = "WP7_PRE_HOLDOUT_BLOCKED" if stop_summary["stop"] else "WP7_PRE_HOLDOUT_READY"
    predictions = pd.concat(outer_test_predictions, ignore_index=True)
    predictions = predictions.sort_values(
        ["window_id", "modeling_month", "security_id"], kind="mergesort"
    ).reset_index(drop=True)
    return {
        "status": status,
        "contract_version": contract["contract_version"],
        "contract_digest": contract_digest(contract),
        "generation_kind": GENERATION_KIND,
        "dataset_id": contract["certified_wp6_inputs"]["dataset_id"],
        "target_set_id": contract["certified_wp6_inputs"]["target_set_id"],
        "feature_set_id": contract["certified_wp6_inputs"]["feature_set_id"],
        "wp6_experiment_id": contract["certified_wp6_inputs"]["wp6_experiment_id"],
        "producing_commit": str(commit or current_git_commit() or "unknown"),
        "research_mode": ResearchMode.RESEARCH_V2.value,
        "eligible_task": "classification",
        "target": TARGET,
        "canonical_holdout_id": holdout.holdout_id,
        "holdout_performance_accessed": False,
        "holdout_labels_accessed": False,
        "outer_geometry": geometry,
        "outer_folds": outer_records,
        "selected_models_by_fold": selected_models,
        "multiple_testing": _multiple_testing_report(candidate_scores),
        "controls": controls,
        "control_evidence": control_evidence,
        "overall_stop": stop_summary,
        "robustness": _robustness_report(outer_records, selected_models),
        "predictions": {
            "phase": "wp7_pre_holdout",
            "prediction_scope": "development_outer_test_windows_only",
            "rows": predictions.to_dict(orient="records"),
        },
        "seed": seed,
        "runtime_seconds": round(time.time() - started, 3),
        "limitations": [
            "WP7 pre-holdout nested estimates are selection-adjusted and are NOT unbiased holdout evidence",
            "locked holdout and rows feature_asof >= 2022-01-01/embargo band were never accessed",
            "all model/calibration/feature decisions are train/inner-only; outer_test is evaluation-only",
            "negative-control failures block progress before any holdout access",
        ],
    }


def generation_binding(result):
    contract = load_contract()
    return {
        "generation_kind": result["generation_kind"],
        "contract_version": result["contract_version"],
        "contract_digest": result["contract_digest"],
        "dataset_id": result["dataset_id"],
        "target_set_id": result["target_set_id"],
        "feature_set_id": result["feature_set_id"],
        "wp6_experiment_id": result["wp6_experiment_id"],
        "producing_commit": result["producing_commit"],
        "canonical_holdout_id": result["canonical_holdout_id"],
        "seeds": {
            "model_seed": result["seed"],
            "bootstrap_seed": int(contract["seeds"]["bootstrap_seed"]),
            "placebo_seed": int(contract["seeds"]["placebo_seed"]),
        },
        "status": result["status"],
        "outer_fold_selections": [
            {
                "window_id": item["window_id"],
                "selected_config_id": item["selected_config_id"],
                "selected_calibration": item["selected_calibration"],
                "selected_features": item["selected_features"],
            }
            for item in result["outer_folds"]
        ],
    }


def generation_id(result):
    blob = canonical_json({
        "kind": "generation",
        "payload": generation_binding(result),
    }).encode("utf-8")
    return "generation_" + hashlib.sha256(blob).hexdigest()[:12]


def _ledger_records(result):
    gid = generation_id(result)
    records = [{
        "record": "generation_header",
        "generation_id": gid,
        "status": result["status"],
        "contract_version": result["contract_version"],
        "contract_digest": result["contract_digest"],
        "producing_commit": result["producing_commit"],
    }]
    for fold in result["outer_folds"]:
        records.append({
            "record": "outer_fold",
            "window_id": fold["window_id"],
            "selected_config_id": fold["selected_config_id"],
            "selected_model": fold["selected_model"],
            "selected_strategy": fold["selected_strategy"],
            "selected_params": fold["selected_params"],
            "selected_calibration": fold["selected_calibration"],
            "selected_features": fold["selected_features"],
            "mean_selection_auc_skill": fold["selection_mean_auc_skill"],
            "outer_mean_auc_skill": fold["outer_metrics"]["mean_auc_skill"],
        })
    records.append({
        "record": "negative_controls",
        "controls": result["controls"],
        "overall_stop": result["overall_stop"],
    })
    return records


def write_generation_artifacts(result, root=None):
    root = Path(root or ROOT)
    gid = generation_id(result)
    artifact_root = root / "artifacts" / "research" / "wp7" / gid
    candidate_path = root / MODEL_CANDIDATE_DIR_REL / ("%s.json" % gid)
    candidate_record = {
        "schema_version": "wp7_model_generation_candidate_v1",
        "generation_id": gid,
        "generation_kind": GENERATION_KIND,
        "contract_version": result["contract_version"],
        "contract_digest": result["contract_digest"],
        "producing_commit": result["producing_commit"],
        "frozen_at_stage": "WP7_PRE_HOLDOUT",
        "status": result["status"],
        "canonical_holdout_id": result["canonical_holdout_id"],
        "holdout_performance_accessed": False,
        "artifact_dir": str(artifact_root),
        "selected_models_by_fold": result["selected_models_by_fold"],
        "contract_file": str(CONTRACT_REL),
        "contract_markdown_file": str(CONTRACT_MD_REL),
        "limitations": result["limitations"],
    }
    payloads = {
        "summary.json": result,
        "decision.json": {
            "status": result["status"],
            "selected_models_by_fold": result["selected_models_by_fold"],
            "overall_stop": result["overall_stop"],
        },
        "predictions.json": result["predictions"],
        "metrics.json": {
            "outer_folds": [
                {
                    "window_id": item["window_id"],
                    "outer_metrics": item["outer_metrics"],
                    "inner_selection_table": item["inner_selection_table"],
                }
                for item in result["outer_folds"]
            ],
            "multiple_testing": result["multiple_testing"],
        },
        "controls.json": {
            "controls": result["controls"],
            "overall_stop": result["overall_stop"],
            "control_evidence": result["control_evidence"],
        },
        "robustness.json": result["robustness"],
    }
    written = {}
    for name, payload in payloads.items():
        written[name] = save_immutable(artifact_root / name, payload)

    ledger_path = artifact_root / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_ledger = "".join(
        json.dumps(record, sort_keys=True, default=str) + "\n"
        for record in _ledger_records(result)
    )
    if ledger_path.exists():
        if ledger_path.read_text(encoding="utf-8") != rendered_ledger:
            raise Wp7ValidationError(
                "immutable WP7 ledger already exists with different content: %s" % ledger_path
            )
        written["ledger.jsonl"] = "verify_and_reuse"
    else:
        ledger_path.write_text(rendered_ledger, encoding="utf-8")
        written["ledger.jsonl"] = "written"

    written["model_generation_candidate"] = save_immutable(
        candidate_path, candidate_record
    )
    return {
        "generation_id": gid,
        "artifact_dir": str(artifact_root),
        "candidate_path": str(candidate_path),
        "written": written,
    }


def load_validation_panel():
    import importlib.util

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    from src.research.modeling.panel import build_modeling_panel

    contract = load_contract()
    w5 = _load_module(
        "wp5_discover_features",
        ROOT / "scripts" / "research_v2" / "wp5_discover_features.py",
    )
    w5.DATASET_ID = contract["certified_wp6_inputs"]["dataset_id"]
    w5.PANEL_VERSION = "553ac17bf5d4d63f"
    w5.TARGET_ID = contract["certified_wp6_inputs"]["target_set_id"]
    w5.TARGET_VERSION = "56d0f670bdf1b47c"
    w5.FEATURE_SET_ID = contract["certified_wp6_inputs"]["feature_set_id"]
    w5.WP4_MODERN_SIGNALS = ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_corrective" / "modern_signals.parquet"
    w5.WP4_HISTORICAL_SIGNALS = ROOT / "artifacts" / "research" / "wp5_correction" / "wp4_corrective" / "historical_signals.parquet"

    panel = w5.load_panel()
    targets = w5.load_targets()
    prices = w5.load_prices()
    actions = w5.load_actions()
    benchmark_prices = w5.load_benchmark_prices()
    benchmark_actions = w5.load_benchmark_actions()
    fundamentals = w5.load_fundamentals()
    cik_by_ticker, _cik_payload = w5.load_cik_by_ticker()
    frame, diagnostics = build_modeling_panel(
        panel, targets, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions,
        config=WP6_DEFAULT_CONFIG, holdout=locked_holdout(),
    )
    return frame, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=None, help="override producing commit (testing only)")
    parser.add_argument(
        "--allow-commit-override",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    started = time.time()
    # A dirty tracked worktree makes the recorded HEAD lie about what produced
    # results. Validate before loading any research panel, and before deciding
    # what commit to bind.
    clean_head = validate_clean_producing_worktree(ROOT)
    if args.commit is not None:
        if not args.allow_commit_override:
            raise Wp7ValidationError(
                "--commit is a test-only override; it must be used with "
                "--allow-commit-override, and actual producing code must still "
                "be committed and clean"
            )
        commit = args.commit
    else:
        commit = clean_head
    print("PRODUCING_COMMIT %s clean_worktree=True" % commit)

    contract = load_contract()
    holdout = locked_holdout()
    if holdout.holdout_id != CANONICAL_HOLDOUT_ID:
        raise Wp7ValidationError(
            "unexpected canonical holdout id %s (expected %s)"
            % (holdout.holdout_id, CANONICAL_HOLDOUT_ID)
        )
    print("CONTRACT %s digest=%s" % (
        contract["contract_version"], contract_digest(contract)[:16]
    ))
    print("HOLDOUT identity=%s pre_holdout_no_performance_access=True" % holdout.holdout_id)

    frame, diagnostics = load_validation_panel()
    max_asof = pd.to_datetime(frame["feature_asof"], utc=True).max()
    if max_asof >= pd.Timestamp("2021-01-01", tz="UTC"):
        raise Wp7ValidationError("development panel violates the 2021-01-01 boundary")
    print("PANEL rows=%d securities=%d months=%d range=%s..%s" % (
        diagnostics["rows"], diagnostics["securities"], diagnostics["months"],
        diagnostics["date_min"], diagnostics["date_max"],
    ))
    print("DEVELOPMENT_BOUNDARY_OK max_feature_asof=%s" % max_asof.strftime("%Y-%m-%d"))

    result = run_pre_holdout_validation(
        frame, contract=contract, commit=commit, holdout=holdout
    )
    paths = write_generation_artifacts(result, ROOT)
    print("STATUS %s" % result["status"])
    print("GENERATION %s" % paths["generation_id"])
    print("ARTIFACTS %s" % paths["artifact_dir"])
    print("CANDIDATE %s" % paths["candidate_path"])
    for fold in result["outer_folds"]:
        print("FOLD %s safe_inner=%d selected=%s calib=%s outer_skill=%s" % (
            fold["window_id"], fold["safe_inner_folds"],
            fold["selected_config_id"], fold["selected_calibration"],
            fold["outer_metrics"]["mean_auc_skill"],
        ))
    print("CONTROLS stop=%s failing=%s" % (
        result["overall_stop"]["stop"],
        result["overall_stop"]["mandatory_failing_controls"],
    ))
    print("RUNTIME %.1fs" % (time.time() - started))
    return result


if __name__ == "__main__":
    main()
