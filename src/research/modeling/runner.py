"""WP6 experiment orchestration (corrective contract v2).

The runner enumerates EVERY frozen configuration (fold x feature strategy x model
x predeclared grid point) and writes a complete ledger, so the experiment is an
honest enumeration rather than the survivor of an adaptive search. It fits every
preprocessing step on the TRAINING slice of a fold only and applies it forward.

Corrective changes over the withdrawn experiment:

* every configuration carries a deterministic ``config_id``
  (:mod:`~src.research.modeling.identity`); negative controls name a PREDECLARED
  matched real configuration by that id -- there is no ``max(values, key=abs)`` or
  best-observed reference anywhere in the evaluation path (defect #1);
* classification inference is performed on the AUC SKILL series ``AUC_t - 0.5``;
  regression inference stays on ``IC_t`` against a zero null (defect #2);
* the placebo battery is replaced by explicit control-evaluation objects with
  real PASS / FAIL / STOP semantics (:mod:`~src.research.modeling.placebo`,
  defect #3);
* a REAL field-based future-availability guard runs over every fold
  (:mod:`~src.research.modeling.pit`, defect #4);
* the model layer applies Benjamini-Hochberg FDR inside two FROZEN hypothesis
  families, and PROMISING requires ``fdr_rejected`` (defect #6);
* the robustness checks (subperiod, complexity, reduced-feature) are EXECUTED and
  serialized (defect #8).

No holdout row is ever loaded. The module issues no verdict beyond the FROZEN
research category.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import models as model_registry
from .contract import DEFAULT_CONFIG, model_feature_universe
from .features import STRATEGIES, select_features, selection_stability
from .identity import configuration_id
from .inference import monthly_significance, paired_monthly_comparison
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
from .multiple_testing import build_families
from .pit import future_availability_control
from .placebo import (
    complexity_comparison,
    evaluate_metric_control,
    overall_stop,
    reduced_feature_comparison,
    shuffle_training_target,
    subperiod_stability,
)
from .preprocessing import Preprocessor, PreprocessingSpec

# ── Predeclared reference configurations and tolerances (frozen) ──────────────
# These are chosen from the frozen grid on METHODOLOGY, never on measured
# performance. They are the matched real reference of each negative control and
# the predeclared members of the robustness comparisons.
REAL_REG_REFERENCE = {"spec": "ridge", "params": {"alpha": 1.0}, "strategy": "A_all_eligible", "task": "regression"}
REAL_CLF_REFERENCE = {"spec": "logistic", "params": {"C": 1.0}, "strategy": "A_all_eligible", "task": "classification"}
COMPLEXITY_NONLINEAR = {"spec": "hist_gradient_boosting",
                       "params": {"max_depth": 3, "min_samples_leaf": 20, "learning_rate": 0.05, "max_iter": 300},
                       "strategy": "A_all_eligible"}
FULL_STRATEGY = "A_all_eligible"
REDUCED_STRATEGIES = ("E_ic_top_k", "F_economic_family_representatives")
PLACEBO_FLOOR = 0.05


class RunnerError(RuntimeError):
    """Raised when the WP6 grid cannot be executed honestly."""


def _profile(spec):
    if spec.name == "baseline_ew_composite":
        return PreprocessingSpec(**model_registry.RANK_SPEC)
    return PreprocessingSpec(**model_registry.VALUE_SPEC)


def _profile_name(spec):
    return "RANK_SPEC" if spec.name == "baseline_ew_composite" else "VALUE_SPEC"


def _target_for(spec, config):
    if spec.family == "classification":
        return config.classification_target
    return config.regression_target


def enumerate_configurations(config=None):
    """Yield every frozen (spec, params, task) configuration exactly once."""
    config = config or DEFAULT_CONFIG
    configurations = []
    for spec in model_registry.regression_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), config.regression_target, "regression"))
    for spec in model_registry.classification_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), config.classification_target, "classification"))
    for spec in model_registry.nonlinear_specs():
        for params in spec.grid():
            configurations.append((spec, dict(params), config.regression_target, "regression"))
            configurations.append((spec, dict(params), config.classification_target, "classification"))
    return configurations


def config_id_for(spec, params, strategy, task, folds, config=None):
    """Deterministic configuration identity (the matched-reference key)."""
    config = config or DEFAULT_CONFIG
    target = config.regression_target if task == "regression" else config.classification_target
    metric = "rank_ic" if task == "regression" else "auc"
    return configuration_id(spec.name, spec.family, task, strategy, _profile_name(spec),
                            params, folds, target, metric)


def classification_spec_name(spec_name, task):
    return spec_name if task == "regression" else spec_name + "__clf"


# ── Frozen category rules (defect #9: PROMISING requires ALL gates) ───────────

def classify_configuration(fold_metrics, pooled_summary, baseline_summary, controls_summary,
                           fdr_rejected, leakage_passed, task, config=None):
    """Assign the predeclared research category using the FROZEN rules.

    PROMISING requires ALL of: a correct metric null, beating the predeclared
    baseline on the primary metric, ``fdr_rejected`` in its frozen family, >= 2
    folds with fold agreement >= 0.75, no mandatory-control STOP affecting the
    task, and a passing leakage guard. A configuration is NEVER forced into
    PROMISING; ``PROMISING = 0`` is an acceptable outcome.
    """
    config = config or DEFAULT_CONFIG
    controls_summary = controls_summary or {}
    if bool(controls_summary.get("stop")) and task in set(controls_summary.get("affected_tasks") or ()):
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
    beats_baseline = False
    if baseline_summary is not None and baseline_summary.get("mean_metric") is not None:
        beats_baseline = bool(mean_metric > baseline_summary["mean_metric"])
    hac_p = ((pooled_summary or {}).get("hac") or {}).get("p_value")
    significant = hac_p is not None and hac_p < config.alpha
    if (direction_positive and beats_baseline and significant
            and bool(fdr_rejected) and bool(leakage_passed)):
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
        series, _diagnostics = monthly_ic_series(outcome, "prediction", "actual",
                                                 min_observations=config.min_cross_section_obs)
        summary = summarize_ic(series)
        summary["mean_metric"] = summary.get("mean_ic")
        extra = regression_errors(outcome, "prediction", "actual")
        return series, summary, extra
    series, _diagnostics = monthly_auc_series(outcome, "prediction", "actual",
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
                "positive_ic_fraction": None, "metric": None, "hac": None, "bootstrap": None}
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
    summary["metric"] = column
    icir = summary.get("icir")
    if icir is None and summary.get("ic_std", 0.0):
        summary["icir"] = float(summary["mean_metric"] / summary["ic_std"])
    significance = monthly_significance(combined, config=config, column=column)
    summary["hac"] = significance.get("hac")
    summary["bootstrap"] = significance.get("bootstrap")
    summary["null"] = significance.get("null")
    summary["comparator"] = significance.get("comparator")
    return summary


def _evaluate_configuration(frame_folds, features, fold_frames, spec, params, task,
                            config, seed):
    """Fit and score ONE configuration across all folds; return per-fold + pooled."""
    records, series_list = [], []
    for fold in frame_folds:
        train_frame = fold_frames[fold.fold]["train"]
        val_frame = fold_frames[fold.fold]["validation"]
        outcome, _used = fit_and_predict(train_frame, val_frame, spec, params, task,
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


def _baseline_summary(series_list, task, config):
    return pooled_summary(combine_series(series_list), task, config)


# ── Negative controls (matched, predeclared; defect #1 & #3) ──────────────────

def _shuffled_target_metric(fold_frames, folds, spec, params, task, features, config, seed):
    """Pooled primary metric of a model trained on the within-date shuffled TRAIN
    target and scored against the REAL validation label."""
    series_list = []
    target = config.regression_target if task == "regression" else config.classification_target
    for fold in folds:
        original = fold_frames[fold.fold]["train"]
        shuffled_train = shuffle_training_target(original, target=target, seed=seed,
                                                 asof_col="modeling_month")
        val_frame = fold_frames[fold.fold]["validation"]
        outcome, _used = fit_and_predict(shuffled_train, val_frame, spec, params, task,
                                         features, config, seed)
        series, _summary, _extra = fold_metric(outcome, task, config)
        series_list.append(series)
    return pooled_summary(combine_series(series_list), task, config).get("mean_metric")


def build_controls(frame, fold_frames, folds, config, seed, records_by_id, config_ids):
    """Build the explicit, matched negative controls (PASS/FAIL/STOP)."""
    features = list(model_feature_universe())
    controls = []

    real_reg_id = config_ids["real_regression"]
    real_clf_id = config_ids["real_classification"]
    real_reg = (records_by_id.get(real_reg_id) or {}).get("pooled", {}).get("mean_metric")
    real_clf = (records_by_id.get(real_clf_id) or {}).get("pooled", {}).get("mean_metric")

    # 1) shuffled-target control, regression (matched to the predeclared ridge).
    shuffled_reg = _shuffled_target_metric(fold_frames, folds, model_registry.MODELS_BY_NAME["ridge"],
                                           {"alpha": 1.0}, "regression", features, config, seed)
    controls.append(evaluate_metric_control(
        "shuffled_target_regression", "shuffled_target", real_reg, shuffled_reg, real_reg_id,
        PLACEBO_FLOOR,
        "a model trained on a within-date shuffled TRAIN target scores ~0 against the real validation label",
        "shuffled-target mean IC magnitude = %.6f", affected_tasks=("regression",)))

    # 2) shuffled-target control, classification (matched to the predeclared logistic).
    shuffled_clf = _shuffled_target_metric(fold_frames, folds, model_registry.MODELS_BY_NAME["logistic"],
                                           {"C": 1.0}, "classification", features, config, seed)
    controls.append(evaluate_metric_control(
        "shuffled_target_classification", "shuffled_target", real_clf, shuffled_clf, real_clf_id,
        PLACEBO_FLOOR,
        "a classifier trained on a within-date shuffled TRAIN label scores near chance (AUC ~ 0.5)",
        "shuffled-target mean AUC magnitude = %.6f", affected_tasks=("classification",)))

    # 3) noise-feature control (matched to the predeclared real regression model).
    from ..discovery.placebo import noise_feature as _noise_feature
    noisy, noise_name = _noise_feature(frame.copy(), seed=seed + 1)
    noise_series, _diag = monthly_ic_series(noisy, noise_name, config.regression_target,
                                            asof_col="modeling_month",
                                            min_observations=config.min_cross_section_obs)
    noise_mean_ic = float(noise_series["rank_ic"].mean()) if len(noise_series) else None
    controls.append(evaluate_metric_control(
        "noise_feature", "noise_feature", real_reg, noise_mean_ic, real_reg_id, PLACEBO_FLOOR,
        "a pure-noise feature has ~0 cross-sectional IC against the real target",
        "noise-feature mean IC magnitude = %.6f", affected_tasks=("regression",)))

    # 4) REAL field-based future-availability guard (defect #4).
    controls.append(future_availability_control(fold_frames, folds,
                                                matched_real_config_id=None))

    overall = overall_stop(controls)
    affected = sorted({task for item in controls
                       if item.get("mandatory") and item.get("stop_required")
                       for task in (item.get("affected_tasks") or ())})
    overall["affected_tasks"] = affected
    return controls, overall


# ── Robustness checks (defect #8: EXECUTED and serialized) ───────────────────

def _paired(col_lookup, left_id, right_id, config, column):
    left = col_lookup.get(left_id)
    right = col_lookup.get(right_id)
    if left is None or right is None:
        return None
    return paired_monthly_comparison(left, right, config=config, column=column,
                                     left_name=left_id, right_name=right_id)


def run_robustness(config_ids, series_by_id, records_by_id, selection_stability_payload,
                   config, seed):
    """Execute the frozen robustness checks over the executed configurations."""
    # (a) subperiod stability: frozen half-split for every configuration.
    subperiod = {}
    for cid, series in series_by_id.items():
        task = (records_by_id.get(cid) or {}).get("task")
        column = "rank_ic" if task == "regression" else "auc"
        subperiod[cid] = subperiod_stability(series, column=column)

    # (b) complexity comparison: predeclared nonlinear vs predeclared simple linear.
    complexity = {}
    for task, linear_key in (("regression", "real_regression"), ("classification", "real_classification")):
        linear_id = config_ids.get(linear_key)
        nonlinear_id = config_ids.get("nonlinear_%s" % task)
        column = "rank_ic" if task == "regression" else "auc"
        paired = _paired(series_by_id, nonlinear_id, linear_id, config, column)
        nonlinear_rec = records_by_id.get(nonlinear_id) or {}
        linear_rec = records_by_id.get(linear_id) or {}
        complexity[task] = complexity_comparison(
            nonlinear_rec.get("pooled"), linear_rec.get("pooled"), metric="mean_metric",
            hac=(paired or {}).get("hac"), fdr_rejected=nonlinear_rec.get("fdr_rejected"),
            logistic_metrics=linear_rec.get("pooled") if task == "classification" else None)
        complexity[task]["paired"] = paired

    # (c) reduced-feature comparison: E/F vs A on the predeclared primary model.
    reduced = {}
    for task, key in (("regression", "real_regression"), ("classification", "real_classification")):
        full_id = config_ids.get(key)
        column = "rank_ic" if task == "regression" else "auc"
        full_rec = records_by_id.get(full_id) or {}
        per_strategy = {}
        for strategy in REDUCED_STRATEGIES:
            reduced_id = config_ids.get("reduced_%s_%s" % (strategy, task))
            paired = _paired(series_by_id, reduced_id, full_id, config, column)
            reduced_rec = records_by_id.get(reduced_id) or {}
            per_strategy[strategy] = reduced_feature_comparison(
                full_rec.get("pooled"), reduced_rec.get("pooled"), metric="mean_metric",
                hac=(paired or {}).get("hac"), fdr_rejected=reduced_rec.get("fdr_rejected"),
                label="%s_vs_%s" % (strategy, FULL_STRATEGY))
            per_strategy[strategy]["paired"] = paired
        reduced[task] = per_strategy

    return {"subperiod_stability": {"by_config": subperiod},
            "complexity_comparison": complexity,
            "reduced_feature_comparison": reduced,
            "feature_selection_stability": selection_stability_payload}


def _predeclared_ids(config, strategies, folds):
    """Compute the predeclared matched-reference config ids deterministically."""
    ids = {}
    for key, spec_name, params, task in (
        ("real_regression", "ridge", {"alpha": 1.0}, "regression"),
        ("real_classification", "logistic", {"C": 1.0}, "classification"),
    ):
        spec = model_registry.MODELS_BY_NAME[spec_name]
        ids[key] = config_id_for(spec, params, FULL_STRATEGY, task, folds, config)
    for task in ("regression", "classification"):
        spec = model_registry.MODELS_BY_NAME["hist_gradient_boosting"]
        ids["nonlinear_%s" % task] = config_id_for(spec, COMPLEXITY_NONLINEAR["params"],
                                                    FULL_STRATEGY, task, folds, config)
    for strategy in REDUCED_STRATEGIES:
        for task, spec_name, params in (("regression", "ridge", {"alpha": 1.0}),
                                        ("classification", "logistic", {"C": 1.0})):
            spec = model_registry.MODELS_BY_NAME[spec_name]
            ids["reduced_%s_%s" % (strategy, task)] = config_id_for(spec, params, strategy, task,
                                                                     folds, config)
    return ids


def run_experiment(frame, fold_frames, folds, config=None, seed=None, strategies=None,
                   models=None):
    """Execute the complete frozen WP6 grid and return the full ledger.

    Every configuration is recorded; nothing is selected on performance. Model-
    layer FDR, the matched negative controls and the robustness checks are all
    computed here and returned so the caller only persists measured evidence.
    """
    config = config or DEFAULT_CONFIG
    seed = int(seed if seed is not None else config.model_seed)
    strategies = list(strategies or STRATEGIES)
    ledger = []
    series_by_id = {}
    records_by_id = {}
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
                series, summary, _extra = fold_metric(outcome, task, config)
                all_series.append(series)
                all_records.append({"fold": fold.fold, "mean_metric": summary.get("mean_metric"),
                                    "mean_ic": summary.get("mean_ic")})
            pooled = _baseline_summary(all_series, task, config)
            baselines[(task, strategy)] = {
                "model": spec_name, "task": task, "strategy": strategy,
                "fold_metrics": all_records, "pooled": pooled,
                "series": combine_series(all_series), "mean_metric": pooled.get("mean_metric"),
            }

    configurations = enumerate_configurations(config)
    if models:
        wanted = set(models)
        configurations = [item for item in configurations if item[0].name in wanted]
    for spec, params, target, task in configurations:
        for strategy in strategies:
            started = time.time()
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
            cid = config_id_for(spec, params, strategy, task, folds, config)
            series_by_id[cid] = combined
            baseline = baselines.get((task, strategy))
            record = {
                "record": "configuration",
                "config_id": cid,
                "model_config_id": cid,
                "model": spec.name,
                "family": spec.family,
                "task": task,
                "strategy": strategy,
                "preprocessing": _profile_name(spec),
                "params": params,
                "target": target,
                "metric": pooled.get("metric"),
                "fold_metrics": records,
                "pooled": pooled,
                "runtime_seconds": round(runtime, 3),
                "baseline_reference": {
                    "model": baseline["model"], "mean_metric": baseline["mean_metric"],
                } if baseline else None,
            }
            ledger.append(record)
            records_by_id[cid] = record

    # Model-layer Benjamini-Hochberg FDR inside the two frozen families.
    configuration_records = [record for record in ledger if record.get("record") == "configuration"]
    families = build_families(configuration_records, alpha=config.alpha)
    for record in configuration_records:
        cid = record["config_id"]
        family = families["regression_predictive_skill"] if record["task"] == "regression" \
            else families["classification_predictive_skill"]
        record["raw_p"] = (record["pooled"].get("hac") or {}).get("p_value")
        record["q_value"] = family["q_values"].get(cid)
        record["fdr_rejected"] = bool(family["fdr_rejected"].get(cid, False))
        record["family_id"] = family["family_id"]

    # Matched negative controls + real PIT guard.
    config_ids = _predeclared_ids(config, strategies, folds)
    controls, overall = build_controls(frame, fold_frames, folds, config, config.placebo_seed,
                                       records_by_id, config_ids)
    leakage_passed = any(item["control_id"] == "future_availability_guard" and item["passed"]
                         for item in controls)

    selection_stability_payload = {strategy: selection_stability(per_strategy_selection[strategy])
                                   for strategy in strategies}

    # Robustness checks (executed).
    robustness = run_robustness(config_ids, series_by_id, records_by_id,
                                selection_stability_payload, config, seed)

    # Research categories (all gates).
    controls_summary = {"stop": overall.get("stop"), "affected_tasks": overall.get("affected_tasks"),
                        "failed_controls": overall.get("failed_controls")}
    categories = {}
    for record in configuration_records:
        baseline = record.get("baseline_reference") or {}
        record["research_category"] = classify_configuration(
            record["fold_metrics"], record["pooled"], baseline, controls_summary,
            record.get("fdr_rejected"), leakage_passed, record["task"], config=config)
        categories[record["research_category"]] = categories.get(record["research_category"], 0) + 1

    return {
        "ledger": ledger,
        "baselines": {str(key): {"model": value["model"], "task": value["task"],
                                 "strategy": value["strategy"], "mean_metric": value["mean_metric"],
                                 "fold_metrics": value["fold_metrics"], "pooled": value["pooled"]}
                      for key, value in baselines.items()},
        "selection_stability": selection_stability_payload,
        "families": families,
        "controls": controls,
        "overall_stop": overall,
        "leakage_passed": bool(leakage_passed),
        "robustness": robustness,
        "categories": categories,
        "config_ids": config_ids,
        "strategy_series": {
            strategy: {fold.fold: per_strategy_selection[strategy][fold.fold] for fold in folds}
            for strategy in strategies
        },
    }
