"""WP6 corrective-contract regression tests (each prior failure class).

These tests exist so the corrective contract cannot silently regress: they cover
the matched predeclared placebo reference, the AUC-skill null, the explicit
control PASS/FAIL/STOP semantics, the field-based PIT guard, deterministic
within-date shuffling, model-layer Benjamini-Hochberg FDR, executed robustness
checks and the result-independent evaluation path.

Synthetic fixtures only; no research data and no network are touched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.holdout import locked_holdout
from src.research.modes import current_git_commit
from src.research.modeling import models as model_registry
from src.research.modeling import runner
from src.research.modeling.contract import (
    DEFAULT_CONFIG,
    WP6_CORRECTIVE_CONTRACT_VERSION,
    contract_payload,
    model_feature_universe,
)
from src.research.modeling.folds import build_folds
from src.research.modeling.identity import configuration_id, configuration_identity
from src.research.modeling.inference import monthly_significance
from src.research.modeling.io import modeling_payload, write_experiment
from src.research.modeling.multiple_testing import build_families, family_fdr
from src.research.modeling.pit import pit_guard_report
from src.research.modeling.placebo import (
    control_object,
    evaluate_metric_control,
    overall_stop,
    shuffle_training_target,
)
from src.research.modeling.preprocessing import Preprocessor, PreprocessingSpec
from src.research.modeling.features import select_features
from src.research.immutability import ImmutabilityError


# ── Synthetic fixtures ───────────────────────────────────────────────────────

def make_frame(months=168, per_month=40, start="2005-01-31"):
    stamps = pd.date_range(start, periods=months, freq="ME")
    rows = []
    for stamp in stamps:
        for index in range(per_month):
            record = {
                "security_id": "SEC%03d" % index,
                "ticker": "SEC%03d" % index,
                "feature_asof": stamp.strftime("%Y-%m-%d"),
                "modeling_month": stamp.strftime("%Y-%m"),
                "target_observable": True,
                "target_known_at": stamp + pd.DateOffset(months=12),
                "target_end": stamp + pd.DateOffset(months=12),
            }
            for name in model_feature_universe():
                record[name] = float((index * 7 + stamp.month) % 13) / 13.0
            record["future_12m_excess_return"] = float(index - per_month / 2) / 1000.0
            record["outperform_12m"] = 1.0 if index % 2 == 0 else 0.0
            rows.append(record)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def experiment():
    frame = make_frame()
    folds, diagnostics, working = build_folds(frame, config=DEFAULT_CONFIG)
    fold_frames = {
        fold.fold: {"train": working.iloc[list(fold.train_index)].reset_index(drop=True),
                    "validation": working.iloc[list(fold.validation_index)].reset_index(drop=True)}
        for fold in folds
    }
    outcome = runner.run_experiment(working, fold_frames, folds, config=DEFAULT_CONFIG,
                                    seed=DEFAULT_CONFIG.model_seed,
                                    strategies=["A_all_eligible"],
                                    models=["ols", "logistic", "ridge"])
    return {"frame": frame, "working": working, "folds": folds, "diagnostics": diagnostics,
            "fold_frames": fold_frames, "outcome": outcome}


# ── Defect #1: matched, predeclared reference ────────────────────────────────

def test_matched_placebo_reference_is_the_exact_predeclared_config(experiment):
    folds = experiment["folds"]
    ids = runner._predeclared_ids(DEFAULT_CONFIG, ["A_all_eligible"], folds)
    expected = runner.config_id_for(model_registry.MODELS_BY_NAME["ridge"], {"alpha": 1.0},
                                    "A_all_eligible", "regression", folds, DEFAULT_CONFIG)
    assert ids["real_regression"] == expected
    controls = {item["control_id"]: item for item in experiment["outcome"]["controls"]}
    assert controls["shuffled_target_regression"]["matched_real_config_id"] == ids["real_regression"]
    assert controls["shuffled_target_classification"]["matched_real_config_id"] == ids["real_classification"]
    assert controls["noise_feature"]["matched_real_config_id"] == ids["real_regression"]


def test_reference_is_stable_when_unrelated_results_change(experiment):
    folds = experiment["folds"]
    first = runner._predeclared_ids(DEFAULT_CONFIG, ["A_all_eligible"], folds)
    second = runner._predeclared_ids(DEFAULT_CONFIG, ["A_all_eligible"], folds)
    assert first == second
    # identity excludes every other configuration's measured metric by construction
    identity = configuration_identity("ridge", "linear", "regression", "A_all_eligible",
                                      "VALUE_SPEC", {"alpha": 1.0}, folds,
                                      DEFAULT_CONFIG.regression_target, "rank_ic")
    assert "mean_metric" not in identity and "raw_p" not in identity


def test_no_best_or_max_abs_reference_in_the_evaluation_path():
    # AST-based: docstrings may *name* the forbidden pattern; no executable call may
    # use a ``key=`` selector (``max(..., key=abs)`` / ``sorted(..., key=lambda)``).
    import ast
    for name in ("runner.py", "placebo.py", "identity.py"):
        source = (REPO_ROOT / "src" / "research" / "modeling" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    assert keyword.arg != "key", "%s selects a reference by key=" % name


# ── Defect #2: correct metric nulls ──────────────────────────────────────────

def test_auc_at_chance_is_not_significant():
    # A deterministic series whose AUC mean is exactly 0.5 (skill mean exactly 0) --
    # a pure random draw can be significant ~5% of the time, so it is not a valid test.
    values = list(np.linspace(0.30, 0.70, 48))
    series = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(48)],
                           "auc": values})
    result = monthly_significance(series, config=DEFAULT_CONFIG, column="auc")
    assert abs(result["skill_mean"]) < 1e-9
    assert result["hac"]["p_value"] > 0.05
    assert result["null"] == "H0: mean(AUC_t - 0.5) = 0"


def test_synthetic_auc_above_half_is_detected():
    rng = np.random.default_rng(11)
    series = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(48)],
                           "auc": 0.6 + rng.normal(0.0, 0.02, size=48)})
    result = monthly_significance(series, config=DEFAULT_CONFIG, column="auc")
    assert result["skill_mean"] > 0.05
    assert result["hac"]["p_value"] < 0.05


def test_classification_inference_subtracts_half():
    series = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(12)],
                           "auc": [0.6] * 12})
    result = monthly_significance(series, config=DEFAULT_CONFIG, column="auc")
    assert abs(result["skill_mean"] - 0.1) < 1e-12
    regression = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(12)],
                               "rank_ic": [0.04] * 12})
    assert abs(monthly_significance(regression, column="rank_ic")["skill_mean"] - 0.04) < 1e-12


# ── Defect #3: explicit PASS / FAIL / STOP ───────────────────────────────────

def test_placebo_pass_branch_reachable():
    control = evaluate_metric_control("c", "t", 0.05, 0.001, "cid", 0.05, "e", "o %.4f")
    assert control["passed"] is True and control["stop_required"] is False


def test_placebo_fail_branch_reachable():
    control = evaluate_metric_control("c", "t", 0.05, 0.20, "cid", 0.05, "e", "o %.4f")
    assert control["passed"] is False and control["stop_required"] is True


def test_mandatory_failure_triggers_stop():
    good = evaluate_metric_control("a", "t", 0.05, 0.0, "cid", 0.05, "e", "o %.4f")
    bad = evaluate_metric_control("b", "t", 0.05, 0.30, "cid", 0.05, "e", "o %.4f")
    assert overall_stop([good])["stop"] is False
    stopped = overall_stop([good, bad])
    assert stopped["stop"] is True
    assert stopped["mandatory_failing_controls"] == ["b"]


def test_clean_battery_experiment_has_no_stop(experiment):
    outcome = experiment["outcome"]
    assert outcome["overall_stop"]["stop"] is False
    assert outcome["leakage_passed"] is True


# ── Defect #4: field-based PIT guard ─────────────────────────────────────────

def test_injected_leakage_row_is_detected():
    train = pd.DataFrame({"feature_asof": ["2010-01-31", "2016-01-31"],
                          "target_known_at": ["2011-01-31", "2012-01-31"],
                          "target_end": ["2011-01-31", "2012-01-31"]})
    validation = pd.DataFrame({"feature_asof": ["2015-01-31"]})
    report = pit_guard_report(train, validation, "2015-01-01")
    assert report["passed"] is False and report["violations"] >= 1


def test_clean_pit_fixture_passes():
    train = pd.DataFrame({"feature_asof": ["2010-01-31", "2011-01-31"],
                          "target_known_at": ["2011-01-31", "2012-01-31"],
                          "target_end": ["2011-01-31", "2012-01-31"]})
    validation = pd.DataFrame({"feature_asof": ["2015-01-31"]})
    report = pit_guard_report(train, validation, "2015-01-01")
    assert report["passed"] is True and report["violations"] == 0


def test_experiment_fold_frames_pass_the_real_guard(experiment):
    controls = {item["control_id"]: item for item in experiment["outcome"]["controls"]}
    guard = controls["future_availability_guard"]
    assert guard["passed"] is True
    assert guard["failure_condition"] == "violations > 0"
    assert guard["mandatory"] is True


# ── Defect #5: frozen shuffle ────────────────────────────────────────────────

def test_shuffle_is_deterministic_under_seed():
    frame = pd.DataFrame({"modeling_month": ["2020-01"] * 6 + ["2020-02"] * 6,
                          "future_12m_excess_return": list(np.arange(12, dtype="float64"))})
    first = shuffle_training_target(frame, seed=3)
    second = shuffle_training_target(frame, seed=3)
    other = shuffle_training_target(frame, seed=4)
    assert list(first["future_12m_excess_return"]) == list(second["future_12m_excess_return"])
    assert list(first["future_12m_excess_return"]) != list(other["future_12m_excess_return"])


def test_shuffle_preserves_panel_structure():
    frame = pd.DataFrame({"modeling_month": ["2020-01"] * 4 + ["2020-02"] * 5,
                          "future_12m_excess_return": list(np.arange(9, dtype="float64"))})
    shuffled = shuffle_training_target(frame, seed=9)
    assert list(shuffled["modeling_month"]) == list(frame["modeling_month"])
    for month in ("2020-01", "2020-02"):
        original = sorted(frame.loc[frame["modeling_month"] == month, "future_12m_excess_return"])
        permuted = sorted(shuffled.loc[shuffled["modeling_month"] == month, "future_12m_excess_return"])
        assert original == permuted


def test_matched_real_and_placebo_share_folds_and_hyperparameters():
    folds = [type("F", (), {"fold": 1, "model_date": "2015-01-01", "train_rows": 10,
                            "validation_rows": 5, "validation_start": "2015-01-31",
                            "validation_end": "2016-12-31"})()]
    identity = configuration_identity("ridge", "linear", "regression", "A_all_eligible",
                                      "VALUE_SPEC", {"alpha": 1.0}, folds,
                                      DEFAULT_CONFIG.regression_target, "rank_ic")
    real = configuration_id("ridge", "linear", "regression", "A_all_eligible", "VALUE_SPEC",
                            {"alpha": 1.0}, folds, DEFAULT_CONFIG.regression_target, "rank_ic",
                            perturbation="none")
    placebo = configuration_id("ridge", "linear", "regression", "A_all_eligible", "VALUE_SPEC",
                               {"alpha": 1.0}, folds, DEFAULT_CONFIG.regression_target, "rank_ic",
                               perturbation="shuffled_target")
    assert identity["folds"] == [{"fold": 1, "model_date": "2015-01-01", "train_rows": 10,
                                  "validation_rows": 5, "validation_start": "2015-01-31",
                                  "validation_end": "2016-12-31"}]
    assert real != placebo  # only the perturbation differs


# ── Defect #6: model-layer BH-FDR ────────────────────────────────────────────

def test_bh_family_contains_all_frozen_hypotheses():
    records = [{"task": "regression", "config_id": "r%d" % index,
                "pooled": {"hac": {"p_value": 0.01 * (index + 1)}}} for index in range(5)]
    records += [{"task": "classification", "config_id": "c%d" % index,
                 "pooled": {"hac": {"p_value": 0.02 * (index + 1)}}} for index in range(3)]
    families = build_families(records, alpha=0.05)
    assert families["regression_predictive_skill"]["number_of_hypotheses"] == 5
    assert families["classification_predictive_skill"]["number_of_hypotheses"] == 3
    assert families["regression_predictive_skill"]["adjustment_method"] == "benjamini_hochberg"
    assert set(families["regression_predictive_skill"]["included_config_ids"]) == {"r0", "r1", "r2", "r3", "r4"}


def test_q_values_recompute_deterministically():
    raw = {"a": 0.001, "b": 0.02, "c": 0.2, "d": 0.6}
    first = family_fdr("f", "q", "rank_ic", "H0", raw, alpha=0.05)
    second = family_fdr("f", "q", "rank_ic", "H0", raw, alpha=0.05)
    assert first["q_values"] == second["q_values"]
    assert first["fdr_rejected"] == second["fdr_rejected"]


def test_raw_p_below_alpha_cannot_promote_without_fdr():
    pooled = {"months": 48, "mean_metric": 0.10, "hac": {"p_value": 0.0001}}
    fold_metrics = [{"mean_metric": 0.08}, {"mean_metric": 0.12}, {"mean_metric": 0.09}]
    baseline = {"mean_metric": 0.01}
    controls = {"stop": False, "affected_tasks": []}
    category = runner.classify_configuration(fold_metrics, pooled, baseline, controls,
                                             False, True, "regression", config=DEFAULT_CONFIG)
    assert category != "PROMISING"
    promoted = runner.classify_configuration(fold_metrics, pooled, baseline, controls,
                                             True, True, "regression", config=DEFAULT_CONFIG)
    assert promoted == "PROMISING"


def test_experiment_fdr_fields_present(experiment):
    configurations = [record for record in experiment["outcome"]["ledger"]
                      if record.get("record") == "configuration"]
    assert configurations
    for record in configurations:
        assert "raw_p" in record and "q_value" in record and "fdr_rejected" in record
        assert record["family_id"] in ("regression_predictive_skill", "classification_predictive_skill")


# ── Defect #8: executed robustness checks ────────────────────────────────────

def test_robustness_checks_are_executed(experiment):
    robustness = experiment["outcome"]["robustness"]
    assert robustness["subperiod_stability"]["by_config"]
    assert "complexity_comparison" in robustness
    assert "reduced_feature_comparison" in robustness
    assert "feature_selection_stability" in robustness
    reduced = robustness["reduced_feature_comparison"]["regression"]
    assert set(reduced.keys()) == {"E_ic_top_k", "F_economic_family_representatives"}
    for strategy in reduced.values():
        assert "reduced_justified" in strategy


# ── Evaluation-path independence & holdout safety ────────────────────────────

def test_no_holdout_rows_are_loaded(experiment):
    frame = experiment["frame"]
    stamps = pd.to_datetime(frame["feature_asof"], utc=True)
    assert (stamps < pd.Timestamp("2021-01-01", tz="UTC")).all()
    holdout = locked_holdout()
    assert holdout.holdout_start >= pd.Timestamp("2022-01-01", tz="UTC")


def test_canonical_holdout_resolves_via_index():
    import json
    index = json.loads((REPO_ROOT / "provenance" / "holdout" / "index.json").read_text(encoding="utf-8"))
    canonical = index["canonical_holdout_id"]
    assert locked_holdout().holdout_id == canonical


def test_new_experiment_id_cannot_collide_with_preserved_experiments():
    binding = modeling_payload("dataset_35a278e17c13", "target_set_d2bb16610bce",
                               "feature_set_4f7b43726310", "holdout_7ce54e933e16",
                               "deadbeef", contract_payload(DEFAULT_CONFIG),
                               model_registry.registry_payload(), {"model_seed": DEFAULT_CONFIG.model_seed})
    from src.research.ids import experiment_id
    identifier = experiment_id(binding)
    assert identifier not in ("experiment_f7864f37998f", "experiment_05ddc3721b4a")
    assert contract_payload(DEFAULT_CONFIG)["corrective_contract_version"] == WP6_CORRECTIVE_CONTRACT_VERSION


def test_new_experiment_is_immutable(tmp_path):
    binding = modeling_payload("dataset_35a278e17c13", "target_set_d2bb16610bce",
                               "feature_set_4f7b43726310", "holdout_7ce54e933e16",
                               "deadbeef", contract_payload(DEFAULT_CONFIG),
                               model_registry.registry_payload(), {"model_seed": DEFAULT_CONFIG.model_seed})
    write_experiment(tmp_path, binding, {"summary.json": {"v": 1}}, {})
    write_experiment(tmp_path, binding, {"summary.json": {"v": 1}}, {})
    with pytest.raises(ImmutabilityError):
        write_experiment(tmp_path, binding, {"summary.json": {"v": 2}}, {})


def test_producing_commit_is_a_real_tracked_commit():
    commit = current_git_commit()
    assert isinstance(commit, str) and len(commit) >= 7
    modeling_dir = REPO_ROOT / "src" / "research" / "modeling"
    for module in ("runner.py", "inference.py", "placebo.py", "pit.py", "identity.py",
                   "multiple_testing.py"):
        # the producing code must exist in the tree at the recorded commit; the commit
        # is created from these exact working-tree files at the end of the work package.
        assert (modeling_dir / module).is_file()
    subprocess.run(["git", "rev-parse", commit], cwd=str(REPO_ROOT),
                   capture_output=True, text=True, check=True)


# ── Fold geometry, train/apply boundary, feature-selection window ────────────

def test_walk_forward_folds_are_correct(experiment):
    folds = experiment["folds"]
    diagnostics = experiment["diagnostics"]
    assert len(folds) >= DEFAULT_CONFIG.min_folds
    previous_train = -1
    for record in diagnostics["folds"]:
        assert record["train_rows"] > previous_train  # expanding window
        previous_train = record["train_rows"]
        assert record["train_end"] < record["validation_start"]
    for fold in folds:
        for index in fold.train_index:
            row = experiment["working"].iloc[index]
            assert pd.Timestamp(row["target_known_at"]) <= pd.Timestamp(fold.model_date)


def test_train_apply_boundary_is_intact():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    val = pd.DataFrame({"x": [100.0, 200.0]})
    preprocessor = Preprocessor(PreprocessingSpec(**model_registry.VALUE_SPEC))
    fitted = preprocessor.fit(train, ["x"])
    train_t = preprocessor.transform(train, fitted)
    val_t = preprocessor.transform(val, fitted)
    # the transform uses the TRAIN fit only: the extreme validation values must be mapped
    # into the range implied by the TRAIN-derived clip bounds. If validation statistics
    # had leaked into the fit, the clip bounds would widen and val would exceed train.
    low = float(train_t["x"].min())
    high = float(train_t["x"].max())
    assert float(val_t["x"].min()) >= low - 1e-9
    assert float(val_t["x"].max()) <= high + 1e-9
    # refitting on the identical train window is deterministic and independent of val
    again = preprocessor.fit(train, ["x"])
    again_t = preprocessor.transform(val, again)
    assert float(again_t["x"].iloc[0]) == float(val_t["x"].iloc[0])


def test_feature_selection_uses_the_train_window_only():
    train = make_frame(months=80)
    selection_a, _ = select_features("A_all_eligible", train, config=DEFAULT_CONFIG)
    assert len(selection_a) > 0
    mutated = train.copy()
    mutated["injected_future_signal"] = 1.0
    selection_b, _ = select_features("A_all_eligible", mutated, config=DEFAULT_CONFIG)
    assert set(selection_b) <= set(selection_a) | {"injected_future_signal"}


def test_no_random_kfold_or_train_test_split_in_model_layer():
    # Only RANDOM resampling primitives are banned; sklearn estimators may legitimately
    # take the frozen ``random_state`` seed. Temporal geometry must come from folds.py.
    for path in (REPO_ROOT / "src" / "research" / "modeling").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for banned in ("KFold", "ShuffleSplit", "train_test_split"):
            assert banned not in source, "%s uses %s" % (path.name, banned)
