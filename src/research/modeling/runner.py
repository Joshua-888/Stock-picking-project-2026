"""WP6 experiment orchestration.

The runner enumerates EVERY frozen configuration (fold x feature strategy x model
x predeclared grid point) and writes a complete ledger, so the experiment is an
honest enumeration rather than the survivor of an adaptive search. It fits every
preprocessing step on the TRAINING slice of a fold only and applies it forward.

No holdout row is ever loaded. No row is scored with a label it could not have
observed. The module issues no verdict: it returns measured evidence and assigns
the predeclared research category using the FROZEN rules in
:func:`classify_configuration`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import models as model_registry
from .contract import DEFAULT_CONFIG
from .features import STRATEGIES, select_features, selection_stability
from .inference import monthly_significance, paired_across_folds, paired_monthly_comparison
from .metrics import (
    calibration_table,
    classification_metrics,
    decile_spread,
    monthly_auc_series,
    monthly_ic_series,
    quantile_spread,
    regression_errors,
    summarize_ic,
)
from .placebo import (
    complexity_comparison,
    future_guard_report,
    noise_feature_placebo,
    overall_stop_flag,
    reduced_feature_comparison,
    shuffled_target_placebo,
    subperiod_stability,
)
from .preprocessing import Preprocessor, PreprocessingSpec

PLACEBO_CONFIGS = (
    {"spec": "ridge", "params": {"alpha": 1.0}, "strategy": "A_all_eligible"},
    {"spec": "logistic", "params": {"C": 1.0}, "strategy": "A_all_eligible"},
    {"spec": "random_forest", "params": {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 50},
     "strategy": "A_all_eligible"},
)


class RunnerError(RuntimeError):
    """Raised when the WP6 grid cannot be executed honestly."""


def _profile(spec):
    if spec.name == "baseline_ew_composite":
        return PreprocessingSpec(**model_registry.RANK_SPEC)
    return PreprocessingSpec(**model_registry.VALUE_SPEC)


def _target_for(spec, config):
    if spec.family == "classification":
        return config.classification_target
    return config.regression_target


def _task_for(spec, params):
    """Return ``regression``/``classification`` for one spec/params entry."""
    if spec.family == "classification":
        return "classification"
    if spec.family == "nonlinear":
        # Nonlinear families are enumerated for BOTH tasks (declared, not chosen).
        return params.get("__task__", "regression")
    return "regression"


def enumerate_configurations(config=None):
    """Yield every frozen (spec, params, task) configuration exactly once."""
    config = config or DEFAULT_CONFIG
    configurations = []
    for spec in model_registry.regression_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), _target_for(spec, config), "regression"))
    for spec in model_registry.classification_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), config.classification_target, "classification"))
    for spec in model_registry.nonlinear_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), config.regression_target, "regression"))
            configurations.append((spec, dict(params), config.classification_target, "classification"))
    return configurations


def classification_spec_name(spec_name, task):
    return spec_name if task == "regression" else spec_name + "__clf"


# ── Frozen category rules ────────────────────────────────────────────────────

def classify_configuration(fold_metrics, pooled_summary, baseline_summary, placebo_flagged, config=None):
    """Assign the predeclared research category using the FROZEN rules.

    Metric-agnostic (works for a rank-IC or an AUC series). A configuration can
    only be PROMISING when it is repeatable across MULTIPLE folds (>= 2) AND beats
    the declared baseline on the predeclared primary metric with HAC p < alpha.
    """
    config = config or DEFAULT_CONFIG
    if placebo_flagged:
        return "REJECTED"
    months = int((pooled_summary or {}).get("months") or 0)
    mean_metric = (pooled_summary or {}).get("mean_metric")
    if months < 12 or mean_metric is None:
        return "NO_EVIDENCE"
    fold_values = [item.get("mean_metric") for item in fold_metrics if item.get("mean_metric") is not None]
    if len(fold_values) < 2:
        return "NO_EVIDENCE"
    if abs(mean_metric) < config.no_evidence_ic_floor:
        return "NO_EVIDENCE"
    direction_positive = mean_metric > 0.0
    agreement = float(np.mean([1.0 if ((value > 0.0) == direction_positive) else 0.0
                               for value in fold_values]))
    if agreement < config.fold_positive_fraction:
        return "UNSTABLE"
    hac_p = ((pooled_summary or {}).get("hac") or {}).get("p_value")
    if direction_positive and baseline_summary is not None and baseline_summary.get("mean_metric") is not None:
        if mean_metric > baseline_summary["mean_metric"] and hac_p is not None and hac_p < config.alpha:
            return "PROMISING"
    return "INCONCLUSIVE"


# ── Fold-level fitting (all fitting inside the training slice) ────────────────

def fit_and_predict(train_frame, val_frame, spec, params, task, features, config, seed):
    """Fit preprocessing+estimator on TRAIN and predict VALIDATION only."""
    features = list(features)
    preprocessor = Preprocessor(_profile(spec))
    fitted = preprocessor.fit(train_frame, features)
    x_train = preprocessor.transform(train_frame, fitted)
    x_val = preprocessor.transform(val_frame, fitted)
    target = config.regression_target if task == "regression" else config.classification_target
    y_train = pd.to_numeric(train_frame[target], errors="coerce")
    finite = np.isfinite(y_train.to_numpy(dtype="float64"))
    if task == "classification":
        finite = finite & np.isin(y_train.to_numpy(dtype="float64"), (0.0, 1.0))
    x_train = x_train.loc[finite]
    y_train = y_train.loc[finite]
    if len(y_train) == 0:
        raise RunnerError("no usable training rows for %s" % spec.name)
    if task == "regression":
        estimator = model_registry.make_estimator(spec, params, seed=seed, columns=features)
    else:
        estimator = model_registry.classification_estimator(spec, params, seed=seed, columns=features)
    estimator.fit(x_train, y_train.to_numpy(dtype="float64"))
    if task == "regression":
        prediction = np.asarray(estimator.predict(x_val), dtype="float64")
    else:
        probabilities = estimator.predict_proba(x_val)
        prediction = np.asarray(probabilities[:, 1], dtype="float64")
    outcome = pd.DataFrame({
        "modeling_month": val_frame["modeling_month"].to_numpy(),
        "prediction": prediction,
        "actual": pd.to_numeric(val_frame[target], errors="coerce").to_numpy(dtype="float64"),
        "security_id": val_frame["security_id"].to_numpy(),
        "feature_asof": val_frame["feature_asof"].to_numpy(),
    })
    return outcome, features


def fold_metric(outcome, task, config):
    """Return ``(series, summary, extra)`` for one fold's validation outcome."""
    if task == "regression":
        series, diagnostics = monthly_ic_series(outcome, "prediction", "actual",
                                                min_observations=config.min_cross_section_obs)
        summary = summarize_ic(series)
        summary["mean_metric"] = summary.get("mean_ic")
        extra = regression_errors(outcome, "prediction", "actual")
        return series, summary, extra
    series, diagnostics = monthly_auc_series(outcome, "prediction", "actual",
                                             min_observations=config.min_cross_section_obs)
    summary = {
        "months": int(len(series)),
        "mean_metric": float(series["auc"].mean()) if len(series) else None,
        "metric": "auc",
    }
    extra = classification_metrics(outcome, "prediction", "actual")
    extra["calibration"] = calibration_table(outcome, "prediction", "actual")
    return series, summary, extra


def combine_series(series_list):
    """Concatenate per-fold monthly series into one chronological series."""
    frames = [item for item in series_list if item is not None and len(item)]
    if not frames:
        return pd.DataFrame(columns=["month", "paired_obs"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("month", kind="mergesort").reset_index(drop=True)


def pooled_summary(combined, task, config):
    if combined is None or len(combined) == 0:
        return {"months": 0, "mean_metric": None, "mean_ic": None, "icir": None,
                "positive_ic_fraction": None, "hac": None, "bootstrap": None}
    column = "rank_ic" if task == "regression" else "auc"
    summary = summarize_ic(combined) if task == "regression" else {
        "months": int(len(combined)),
        "mean_metric": float(combined[column].mean()),
        "ic_std": float(combined[column].std(ddof=1)) if len(combined) > 1 else 0.0,
        "icir": None,
        "positive_ic_fraction": float((combined[column] > 0).mean()),
        "mean_paired_obs": float(combined["paired_obs"].mean()),
    }
    summary["mean_ic"] = summary.get("mean_ic")
    if summary.get("mean_metric") is None:
        summary["mean_metric"] = summary.get("mean_ic")
    icir = summary.get("icir")
    if icir is None and summary.get("ic_std", 0.0):
        icir = float(summary["mean_metric"] / summary["ic_std"])
        summary["icir"] = icir
    significance = monthly_significance(combined, config=config, column=column)
    summary["hac"] = significance.get("hac")
    summary["bootstrap"] = significance.get("bootstrap")
    return summary


# ── Dataset-path-independent numeric alignment for the report ─────────────────

def _baseline_fold_summary(series_list, task, config):
    combined = combine_series(series_list)
    return pooled_summary(combined, task, config)


def _evaluate_configuration(frame_folds, features, fold_frames, spec, params, task,
                            config, seed):
    """Fit and score ONE configuration across all folds; return per-fold + pooled."""
    records, series_list = [], []
    for fold in frame_folds:
        train_frame = fold_frames[fold.fold]["train"]
        val_frame = fold_frames[fold.fold]["validation"]
        outcome, used_features = fit_and_predict(train_frame, val_frame, spec, params, task,
                                                 features, config, seed)
        series, summary, extra = fold_metric(outcome, task, config)
        series_list.append(series)
        records.append({
            "fold": fold.fold,
            "months": summary.get("months"),
            "mean_metric": summary.get("mean_metric"),
            "mean_ic": summary.get("mean_ic"),
            "icir": summary.get("icir"),
            "positive_ic_fraction": summary.get("positive_ic_fraction"),
            "extra": extra,
        })
    combined = combine_series(series_list)
    pooled = pooled_summary(combined, task, config)
    return records, combined, pooled


def run_experiment(frame, fold_frames, folds, config=None, seed=None, strategies=None,
                   models=None, placebo=True):
    """Execute the complete frozen WP6 grid and return the full ledger.

    ``fold_frames`` maps fold number -> {'train': DataFrame, 'validation': DataFrame}.
    Every configuration is recorded; nothing is selected on performance.
    """
    config = config or DEFAULT_CONFIG
    seed = int(seed if seed is not None else config.model_seed)
    strategies = list(strategies or STRATEGIES)
    ledger = []
    config_series = {}
    per_strategy_selection = {strategy: {} for strategy in strategies}

    for strategy in strategies:
        for fold in folds:
            train_frame = fold_frames[fold.fold]["train"]
            features, evidence = select_features(strategy, train_frame, config=config)
            per_strategy_selection[strategy][fold.fold] = features
            ledger.append({
                "record": "feature_selection",
                "strategy": strategy,
                "fold": fold.fold,
                "features": list(features),
                "n_features": len(features),
                "evidence": evidence,
            })

    baselines = {}
    for task, spec_name in (("regression", "baseline_ew_composite"), ("classification", "baseline_base_rate")):
        spec = model_registry.MODELS_BY_NAME[spec_name]
        for strategy in strategies:
            all_records, all_series = [], []
            for fold in folds:
                features = per_strategy_selection[strategy][fold.fold]
                train_frame = fold_frames[fold.fold]["train"]
                val_frame = fold_frames[fold.fold]["validation"]
                outcome, _used = fit_and_predict(train_frame, val_frame, spec, {}, task,
                                                 features, config, seed)
                series, summary, extra = fold_metric(outcome, task, config)
                all_series.append(series)
                all_records.append({"fold": fold.fold, "mean_metric": summary.get("mean_metric"),
                                    "mean_ic": summary.get("mean_ic")})
            combined = combine_series(all_series)
            baselines[(task, strategy)] = {
                "model": spec_name, "task": task, "strategy": strategy,
                "fold_metrics": all_records,
                "pooled": pooled_summary(combined, task, config),
                "series": combined,
                "mean_metric": (pooled_summary(combined, task, config) or {}).get("mean_metric"),
            }

    configurations = enumerate_configurations(config)
    if models:
        wanted = set(models)
        configurations = [item for item in configurations if item[0].name in wanted]
    for spec, params, target, task in configurations:
        for strategy in strategies:
            started = time.time()
            # Feature selection is PER FOLD, so each configuration is evaluated
            # inside the fold loop using that fold's train-window selection.
            records, series_list = [], []
            for fold in folds:
                features = per_strategy_selection[strategy][fold.fold]
                train_frame = fold_frames[fold.fold]["train"]
                val_frame = fold_frames[fold.fold]["validation"]
                outcome, _used = fit_and_predict(train_frame, val_frame, spec, params, task,
                                                 features, config, seed)
                series, summary, extra = fold_metric(outcome, task, config)
                series_list.append(series)
                records.append({"fold": fold.fold, "months": summary.get("months"),
                                "mean_metric": summary.get("mean_metric"),
                                "mean_ic": summary.get("mean_ic"), "icir": summary.get("icir"),
                                "positive_ic_fraction": summary.get("positive_ic_fraction"),
                                "extra": extra})
            combined = combine_series(series_list)
            pooled = pooled_summary(combined, task, config)
            runtime = time.time() - started
            key = "%s|%s|%s|%s" % (spec.name, sorted(params.items()), strategy, task)
            config_series[key] = {
                "model": spec.name, "task": task, "strategy": strategy,
                "params": params, "series": combined,
            }
            baseline = baselines.get((task, strategy))
            ledger.append({
                "record": "configuration",
                "model": spec.name,
                "family": spec.family,
                "task": task,
                "strategy": strategy,
                "params": params,
                "target": target,
                "fold_metrics": records,
                "pooled": {k: v for k, v in pooled.items() if k not in ("hac", "bootstrap")} | {
                    "hac": pooled.get("hac"), "bootstrap": pooled.get("bootstrap")},
                "runtime_seconds": round(runtime, 3),
                "baseline_reference": {
                    "model": baseline["model"], "mean_metric": baseline["mean_metric"],
                } if baseline else None,
            })

    return {
        "ledger": ledger,
        "config_series": config_series,
        "baselines": {str(key): {"model": value["model"], "task": value["task"],
                                 "strategy": value["strategy"],
                                 "mean_metric": value["mean_metric"],
                                 "pooled": {k: v for k, v in value["pooled"].items()
                                            if k not in ("hac", "bootstrap")} | {
                                     "hac": value["pooled"].get("hac"),
                                     "bootstrap": value["pooled"].get("bootstrap")}}
                    for key, value in baselines.items()},
        "selection_stability": {strategy: selection_stability(per_strategy_selection[strategy])
                                for strategy in strategies},
        "strategy_series": {
            strategy: {fold.fold: per_strategy_selection[strategy][fold.fold]
                       for fold in folds}
            for strategy in strategies
        },
    }


# ── Placebo battery ──────────────────────────────────────────────────────────

def run_placebo(frame, fold_frames, folds, config=None, seed=None, real_mean_ic=None):
    """Run the predeclared WP6 falsification battery; return checks + stop flag.

    A pipeline that manufactures signal will show a strong IC for a shuffled
    target or for a pure-noise feature; either is an explicit STOP.
    """
    config = config or DEFAULT_CONFIG
    seed = int(seed if seed is not None else config.placebo_seed)
    from ..discovery.placebo import noise_feature as _noise
    from ..discovery.placebo import shuffled_target as _shuffle
    from .contract import model_feature_universe

    checks = {}
    features = list(model_feature_universe())

    # 1) SHUFFLED-target model placebo: train on a within-date shuffled label and
    #    score against the REAL validation label. Honest pipelines score ~0.
    shuffled_series = []
    ridge = model_registry.MODELS_BY_NAME["ridge"]
    for fold in folds:
        original = fold_frames[fold.fold]["train"]
        train_frame = _shuffle(original, target=config.regression_target, seed=seed,
                               asof_col="modeling_month")
        val_frame = fold_frames[fold.fold]["validation"]
        outcome, _used = fit_and_predict(train_frame, val_frame, ridge, {"alpha": 1.0},
                                         "regression", features, config, seed)
        series, _summary, _extra = fold_metric(outcome, "regression", config)
        shuffled_series.append(series)
    pooled_shuffled = pooled_summary(combine_series(shuffled_series), "regression", config)
    checks["shuffled_target"] = shuffled_target_placebo(real_mean_ic,
                                                       pooled_shuffled.get("mean_metric"))

    # 2) NOISE-feature control: the IC of a pure-noise feature must be ~0.
    noisy, noise_name = _noise(frame.copy(), seed=seed + 1)
    noise_series, _diag = monthly_ic_series(noisy, noise_name, config.regression_target,
                                            asof_col="modeling_month",
                                            min_observations=config.min_cross_section_obs)
    noise_mean_ic = float(noise_series["rank_ic"].mean()) if len(noise_series) else None
    checks["noise_feature"] = noise_feature_placebo(real_mean_ic, noise_mean_ic)

    # 3) LABEL-SHIFT / future guard: the PIT machinery must flag the shifted rows.
    checks["future_guard"] = future_guard_report(frame, feature="twelve_month_momentum")

    return {"checks": checks, "overall": overall_stop_flag(checks),
            "shuffled_target_mean_ic": pooled_shuffled.get("mean_metric"),
            "noise_feature_mean_ic": noise_mean_ic}
