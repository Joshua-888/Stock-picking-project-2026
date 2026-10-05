"""WP7 pre-holdout validation tests (synthetic fixtures only; no research data).

These tests prove the WP7 engine honours the frozen
``WP7_VALIDATION_CONTRACT_V4`` geometry, purge semantics, train-only fitting,
determinism, negative-control STOP reachability, and provenance binding. No
locked-holdout row, label, or metric is ever constructed or read here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts" / "research_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wp7_validation as wp7  # noqa: E402

from src.research import wp7_generation_corrections as wpc7  # noqa: E402
from src.research.holdout import locked_holdout  # noqa: E402
from src.research.modeling.contract import model_feature_universe  # noqa: E402

WP7_SCRIPT = SCRIPTS / "wp7_validation.py"
FORBIDDEN_SPLITTERS = ("KFold", "train_test_split", "ShuffleSplit")


def _panel(months=168, per_month=40, start="2007-01-31", seed=7, signal=0.0):
    """Deterministic development-only panel ending 2020-12 (no 2021+ rows)."""
    stamps = pd.date_range(start, periods=months, freq="ME")
    rng = np.random.default_rng(seed)
    features = list(model_feature_universe())
    rows = []
    for stamp in stamps:
        for index in range(per_month):
            label = 1.0 if index % 2 == 0 else 0.0
            record = {
                "security_id": "SEC%03d" % index,
                "ticker": "SEC%03d" % index,
                "feature_asof": stamp.strftime("%Y-%m-%d"),
                "modeling_month": stamp.strftime("%Y-%m"),
                "target_observable": True,
                "target_known_at": (stamp + pd.DateOffset(months=12)).strftime("%Y-%m-%d"),
                "target_end": (stamp + pd.DateOffset(months=12)).strftime("%Y-%m-%d"),
                "future_12m_excess_return": float(rng.normal()) / 100.0,
                "outperform_12m": label,
            }
            for name in features:
                record[name] = float(rng.normal() + signal * label)
            rows.append(record)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def contract():
    return wp7.load_contract(REPO_ROOT)


@pytest.fixture(scope="module")
def panel():
    return _panel()


# --------------------------------------------------------------------------
# frozen contract integrity
# --------------------------------------------------------------------------


def test_contract_is_v4_classification_only(contract):
    assert contract["contract_version"] == "WP7_VALIDATION_CONTRACT_V4"
    assert contract["eligible_task"]["task"] == "classification"
    assert contract["eligible_task"]["regression"]["eligible"] is False
    assert contract["target"] == "outperform_12m"


def test_eligible_model_set_is_the_frozen_baseline_plus_nine(contract):
    configs = wp7.parse_frozen_configs(contract)
    assert len(configs) == 10
    assert configs[0].config_id == "baseline_base_rate"
    frozen = contract["eligible_model_set"]["configs"]
    assert len(frozen) == 9
    for parsed, raw in zip(configs[1:], frozen):
        assert parsed.config_id == raw["config_id"]
        assert parsed.model == raw["model"]
        assert parsed.strategy == raw["strategy"]
        assert list(parsed.preprocessing) == list(raw["preprocessing"])
        assert parsed.params == raw["params"]
        assert parsed.strategy in ("B_coverage_qualified", "F_economic_family_representatives")


def test_feature_universe_matches_the_frozen_twenty(contract):
    universe = wp7.frozen_feature_universe(contract)
    assert len(universe) == 20
    assert list(universe) == list(contract["feature_universe"])


def test_frozen_seeds_are_honoured(contract):
    assert contract["seeds"]["model_seed"] == 20260930
    assert contract["seeds"]["bootstrap_seed"] == 20260926
    assert contract["seeds"]["placebo_seed"] == 20260926


def test_inner_design_matches_the_frozen_fields(contract):
    inner = contract["inner_walk_forward_design"]
    config = wp7.wp7_inner_config()
    assert inner["mode"] == "chronological_expanding_folds_only"
    assert config.min_train_months == inner["min_train_months"] == 37
    assert config.validation_window_months == inner["validation_window_months"] == 12
    assert config.horizon_months == inner["horizon_months"] == 12
    assert config.min_folds == inner["min_folds"] == 4
    assert list(inner["forbidden_splitters"]) == list(FORBIDDEN_SPLITTERS)


# --------------------------------------------------------------------------
# outer geometry
# --------------------------------------------------------------------------


def test_outer_geometry_matches_the_four_frozen_windows(panel, contract):
    geometry = wp7.derive_outer_geometry(panel, contract)
    assert [item["window_id"] for item in geometry["windows"]] == [
        "outer_test_1", "outer_test_2", "outer_test_3", "outer_test_4",
    ]
    assert [(item["start"], item["end"]) for item in geometry["windows"]] == [
        ("2017-01", "2017-12"),
        ("2018-01", "2018-12"),
        ("2019-01", "2019-12"),
        ("2020-01", "2020-12"),
    ]
    assert [(item["start_index"], item["end_index"]) for item in geometry["windows"]] == [
        (120, 131), (132, 143), (144, 155), (156, 167),
    ]
    assert geometry["safe_inner_fold_counts_by_outer_fold"] == [4, 5, 6, 7]
    assert geometry["observed_month_count"] == geometry["expected_month_count"] == 168


def test_outer_geometry_is_serializable_before_evaluation(panel, contract):
    geometry = wp7.derive_outer_geometry(panel, contract)
    assert json.loads(json.dumps(geometry, default=str)) is not None


def test_outer_geometry_rejects_shifted_month_index(contract):
    shifted = _panel(months=168, per_month=4, start="2006-01-31")
    with pytest.raises(wp7.Wp7ValidationError):
        wp7.derive_outer_geometry(shifted, contract)


def test_safe_inner_fold_counts_are_four_five_six_seven(panel, contract):
    observed = []
    for window in contract["outer_walk_forward_design"]["outer_test_windows"]:
        before = panel.loc[panel["modeling_month"] < window["start"]].reset_index(drop=True)
        folds, _diagnostics, working = wp7.build_inner_folds(before)
        safe, _excluded = wp7.safe_inner_folds(
            folds, working, wp7._timestamp_start(window["start"])
        )
        observed.append(len(safe))
    assert observed == [4, 5, 6, 7]


def test_raw_inner_fold_counts_match_the_frozen_declaration(panel, contract):
    observed = []
    for window in contract["outer_walk_forward_design"]["outer_test_windows"]:
        before = panel.loc[panel["modeling_month"] < window["start"]].reset_index(drop=True)
        folds, _diagnostics, _working = wp7.build_inner_folds(before)
        observed.append(len(folds))
    assert observed == contract["outer_walk_forward_design"][
        "raw_inner_fold_counts_before_last_window_exclusion"
    ] == [5, 6, 7, 8]


# --------------------------------------------------------------------------
# purge / leakage invariants
# --------------------------------------------------------------------------


def test_inner_selection_only_uses_windows_whose_labels_close_before_outer_test(panel, contract):
    for window in contract["outer_walk_forward_design"]["outer_test_windows"]:
        outer_start = wp7._timestamp_start(window["start"])
        before = panel.loc[panel["modeling_month"] < window["start"]].reset_index(drop=True)
        folds, _diagnostics, working = wp7.build_inner_folds(before)
        safe, excluded = wp7.safe_inner_folds(folds, working, outer_start)
        assert excluded, "the last inner window must be purged before outer_test"
        for fold in safe:
            validation = working.iloc[list(fold.validation_index)]
            assert pd.to_datetime(validation["target_end"], utc=True).max() < outer_start
            train = working.iloc[list(fold.train_index)]
            assert pd.to_datetime(train["target_end"], utc=True).max() < outer_start
            assert train["feature_asof"].max() < validation["feature_asof"].min()


def test_no_training_or_validation_slice_contains_outer_test_months(panel, contract):
    for window in contract["outer_walk_forward_design"]["outer_test_windows"]:
        before = panel.loc[panel["modeling_month"] < window["start"]].reset_index(drop=True)
        folds, _diagnostics, working = wp7.build_inner_folds(before)
        safe, _excluded = wp7.safe_inner_folds(
            folds, working, wp7._timestamp_start(window["start"])
        )
        for fold in safe:
            for index in (fold.train_index, fold.validation_index):
                months = working.iloc[list(index)]["modeling_month"].astype(str)
                assert months.max() < window["start"]


def test_development_boundary_refuses_holdout_embargo_and_2021_rows(panel):
    assert wp7.assert_development_only(panel) is True
    holdout = locked_holdout()
    assert holdout.holdout_id == wp7.CANONICAL_HOLDOUT_ID

    embargo = panel.copy()
    embargo.loc[embargo.index[0], "feature_asof"] = "2021-06-30"
    with pytest.raises(wp7.Wp7ValidationError):
        wp7.assert_development_only(embargo)

    future = panel.copy()
    future.loc[future.index[0], "feature_asof"] = "2022-03-31"
    with pytest.raises(wp7.Wp7ValidationError):
        wp7.assert_development_only(future)


def test_trainable_mask_excludes_rows_whose_label_is_not_closed(panel):
    cutoff = wp7._timestamp_start("2017-01")
    before = panel.loc[panel["modeling_month"] < "2017-01"].reset_index(drop=True)
    mask = wp7._frame_trainable_before(before, cutoff)
    kept = before.loc[mask]
    assert len(kept) > 0
    assert pd.to_datetime(kept["target_end"], utc=True).max() < cutoff
    assert pd.to_datetime(kept["target_known_at"], utc=True).max() <= cutoff
    dropped = before.loc[~mask]
    assert len(dropped) > 0
    assert pd.to_datetime(dropped["target_end"], utc=True).min() >= cutoff


# --------------------------------------------------------------------------
# train-only preprocessing / feature selection / calibration
# --------------------------------------------------------------------------


def test_feature_selection_is_train_only_and_restricted_to_the_frozen_universe(panel, contract):
    train = panel.loc[panel["modeling_month"] < "2012-01"]
    other = panel.loc[panel["modeling_month"] < "2016-01"]
    universe = set(wp7.frozen_feature_universe(contract))
    for strategy in ("B_coverage_qualified", "F_economic_family_representatives"):
        features, evidence = wp7._feature_names_from_strategy(strategy, train, contract)
        assert features and set(features) <= universe
        assert evidence
        repeat, _evidence = wp7._feature_names_from_strategy(strategy, train.copy(), contract)
        assert repeat == features
        wider, _evidence = wp7._feature_names_from_strategy(strategy, other, contract)
        assert set(wider) <= universe


def test_preprocessing_is_fitted_on_train_only(panel, contract):
    from src.research.modeling import models as model_registry
    from src.research.modeling.preprocessing import PreprocessingSpec, Preprocessor

    train = panel.loc[panel["modeling_month"] < "2012-01"].reset_index(drop=True)
    validation = panel.loc[
        (panel["modeling_month"] >= "2013-01") & (panel["modeling_month"] < "2014-01")
    ].reset_index(drop=True)
    features = list(wp7.frozen_feature_universe(contract))[:5]

    preprocessor = Preprocessor(PreprocessingSpec(**model_registry.VALUE_SPEC))
    fitted_train = preprocessor.fit(train, features)
    fitted_pooled = preprocessor.fit(
        pd.concat([train, validation], ignore_index=True), features
    )
    assert json.dumps(fitted_train, default=str) != json.dumps(fitted_pooled, default=str)

    transformed = preprocessor.transform(validation, fitted_train)
    assert len(transformed) == len(validation)


def test_calibration_candidates_and_selection_are_inner_only(panel, contract):
    assert wp7.CALIBRATION_CANDIDATES == tuple(contract["calibration"]["candidate_set"])
    assert contract["calibration"]["scope"] == "train/inner-only"
    assert "outer test" in contract["calibration"]["forbidden_scope"]
    assert "holdout" in contract["calibration"]["forbidden_scope"]

    before = panel.loc[panel["modeling_month"] < "2017-01"].reset_index(drop=True)
    folds, _diagnostics, working = wp7.build_inner_folds(before)
    safe, _excluded = wp7.safe_inner_folds(folds, working, wp7._timestamp_start("2017-01"))
    fold_frames = wp7._build_fold_frames(working, safe)
    configs = wp7.parse_frozen_configs(contract)
    evaluated = wp7.evaluate_config_on_inner_folds(
        configs[1], fold_frames, safe, contract, contract["seeds"]["model_seed"]
    )
    chosen = wp7.select_calibration(evaluated, fold_frames, [fold.fold for fold in safe])
    assert chosen["selected"] in wp7.CALIBRATION_CANDIDATES
    assert set(chosen["skills"]) == set(wp7.CALIBRATION_CANDIDATES)
    assert chosen["tie_break"] == "deterministic_earliest_candidate_order"

    # every scored row originates from an inner validation window before outer_test
    for outcome in evaluated["outcomes"]:
        assert outcome["modeling_month"].astype(str).max() < "2017-01"


def test_calibration_tie_breaks_to_the_earliest_frozen_candidate():
    tied = {
        "outcomes": [
            pd.DataFrame({
                "modeling_month": ["2015-01"] * 6,
                "prediction": [0.5] * 6,
                "actual": [1.0, 0.0] * 3,
                "security_id": list("ABCDEF"),
                "feature_asof": ["2015-01-31"] * 6,
            }),
            pd.DataFrame({
                "modeling_month": ["2016-01"] * 6,
                "prediction": [0.5] * 6,
                "actual": [1.0, 0.0] * 3,
                "security_id": list("ABCDEF"),
                "feature_asof": ["2016-01-31"] * 6,
            }),
        ]
    }
    frames = {
        1: {"train": pd.DataFrame({"outperform_12m": [1.0, 0.0, 1.0, 0.0]})},
        2: {"train": pd.DataFrame({"outperform_12m": [1.0, 0.0, 1.0, 0.0]})},
    }
    chosen = wp7.select_calibration(tied, frames, [1, 2])
    assert chosen["selected"] == "none"


# --------------------------------------------------------------------------
# deterministic selection
# --------------------------------------------------------------------------


def test_model_selection_picks_highest_mean_inner_auc_skill():
    evaluations = [
        {"config_id": "cfg_b", "mean_auc_skill": 0.02, "params": {"a": 1}},
        {"config_id": "cfg_a", "mean_auc_skill": 0.04, "params": {"a": 1, "b": 2}},
        {"config_id": "cfg_c", "mean_auc_skill": None, "params": {}},
    ]
    assert wp7.select_model_candidate(evaluations)["config_id"] == "cfg_a"


def test_model_selection_tie_breaks_on_complexity_then_config_id():
    evaluations = [
        {"config_id": "cfg_z", "mean_auc_skill": 0.03, "params": {"a": 1}},
        {"config_id": "cfg_y", "mean_auc_skill": 0.03, "params": {"a": 1, "b": 2}},
        {"config_id": "cfg_a", "mean_auc_skill": 0.03, "params": {"a": 1}},
    ]
    assert wp7.select_model_candidate(evaluations)["config_id"] == "cfg_a"


def test_inner_evaluation_is_deterministic_under_the_frozen_seed(panel, contract):
    before = panel.loc[panel["modeling_month"] < "2017-01"].reset_index(drop=True)
    folds, _diagnostics, working = wp7.build_inner_folds(before)
    safe, _excluded = wp7.safe_inner_folds(folds, working, wp7._timestamp_start("2017-01"))
    fold_frames = wp7._build_fold_frames(working, safe)
    candidate = wp7.parse_frozen_configs(contract)[1]
    seed = contract["seeds"]["model_seed"]
    first = wp7.evaluate_config_on_inner_folds(candidate, fold_frames, safe, contract, seed)
    second = wp7.evaluate_config_on_inner_folds(candidate, fold_frames, safe, contract, seed)
    assert first["mean_auc_skill"] == second["mean_auc_skill"]
    for left, right in zip(first["outcomes"], second["outcomes"]):
        assert np.allclose(left["prediction"], right["prediction"])


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


def test_negative_control_pass_path_is_reachable():
    control = wp7.evaluate_negative_control(
        "noise_feature_classification", "noise_feature", 0.02, 0.001
    )
    assert control["passed"] is True
    assert control["stop_required"] is False
    assert wp7.overall_stop([control])["stop"] is False


def test_negative_control_stop_path_is_reachable_and_blocks():
    control = wp7.evaluate_negative_control(
        "shuffled_target_classification", "shuffled_target", 0.02, 0.30
    )
    assert control["passed"] is False
    assert control["stop_required"] is True
    stop = wp7.overall_stop([control])
    assert stop["stop"] is True
    assert stop["mandatory_failing_controls"] == ["shuffled_target_classification"]


def test_missing_control_metric_fails_closed():
    control = wp7.evaluate_negative_control("c", "noise_feature", 0.02, None)
    assert control["passed"] is False and control["stop_required"] is True


def test_control_battery_blocks_wp7_when_a_mandatory_control_fails():
    failing = [{"real_skill": 0.01, "shuffled_skill": 0.40, "noise_skill": 0.001}]
    controls, stop, evidence = wp7.build_negative_controls(failing)
    assert stop["stop"] is True
    assert "shuffled_target_classification" in stop["mandatory_failing_controls"]
    assert evidence["shuffled_skill_mean"] == 0.40

    passing = [{"real_skill": 0.02, "shuffled_skill": 0.002, "noise_skill": 0.003}]
    controls, stop, _evidence = wp7.build_negative_controls(passing)
    assert stop["stop"] is False
    assert all(item["mandatory"] for item in controls)


def test_shuffled_target_control_permutes_within_the_cross_section(panel):
    train = panel.loc[panel["modeling_month"] < "2010-01"].reset_index(drop=True)
    shuffled = wp7.shuffle_training_target(
        train, target=wp7.TARGET, seed=20260926, asof_col="modeling_month"
    )
    for month in sorted(train["modeling_month"].unique())[:3]:
        original = sorted(train.loc[train["modeling_month"] == month, wp7.TARGET])
        permuted = sorted(shuffled.loc[shuffled["modeling_month"] == month, wp7.TARGET])
        assert original == permuted


# --------------------------------------------------------------------------
# forbidden splitters
# --------------------------------------------------------------------------


def test_no_random_splitters_in_the_wp7_evaluation_path():
    tree = ast.parse(WP7_SCRIPT.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
    for forbidden in FORBIDDEN_SPLITTERS:
        assert forbidden not in names


def test_wp7_does_not_import_sklearn_model_selection():
    tree = ast.parse(WP7_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "model_selection" not in node.module


# --------------------------------------------------------------------------
# holdout isolation
# --------------------------------------------------------------------------


def test_wp7_never_reads_locked_holdout_performance():
    source = WP7_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called.add(function.id)
            elif isinstance(function, ast.Attribute):
                called.add(function.attr)
    # the holdout module may only be used for identity/exclusion, never scoring
    assert "locked_holdout" in called
    for forbidden in ("holdout_predictions", "holdout_metrics", "score_holdout",
                      "load_holdout_targets", "evaluate_holdout"):
        assert forbidden not in called


def test_canonical_holdout_identity_is_declaration_only():
    assert wp7.CANONICAL_HOLDOUT_ID == "holdout_7ce54e933e16"
    assert locked_holdout().holdout_id == wp7.CANONICAL_HOLDOUT_ID


# --------------------------------------------------------------------------
# multiple testing / robustness reporting
# --------------------------------------------------------------------------


def test_multiple_testing_reports_raw_and_bh_q_values():
    scores = []
    for config_id, skills in (("cfg_a", [0.05, 0.051, 0.049, 0.052]),
                              ("cfg_b", [0.0, 0.001, -0.001, 0.0005])):
        for window, value in enumerate(skills, start=1):
            scores.append({"config_id": config_id,
                           "window_id": "outer_test_%d" % window,
                           "mean_auc_skill": value})
    report = wp7._multiple_testing_report(scores)
    assert report["family_id"] == wp7.CLASSIFICATION_FAMILY_ID
    assert report["number_of_hypotheses"] == 2
    assert set(report["raw_p_by_config"]) == {"cfg_a", "cfg_b"}
    assert set(report["q_values"]) == {"cfg_a", "cfg_b"}
    for config_id, q_value in report["q_values"].items():
        assert q_value >= report["raw_p_by_config"][config_id] - 1e-12
    assert "NOT unbiased holdout evidence" in report["claim_restriction"]


def test_robustness_report_covers_every_frozen_requirement(contract):
    outer_records = [
        {
            "window_id": "outer_test_%d" % index,
            "selected_config_id": "cfg_%d" % (index % 2),
            "selected_calibration": "none",
            "selected_params": {"max_depth": 4},
            "outer_metrics": {"mean_auc_skill": 0.01 * index},
        }
        for index in range(1, 5)
    ]
    selected_models = {
        item["window_id"]: {"features": ["roa", "roe"]} for item in outer_records
    }
    report = wp7._robustness_report(outer_records, selected_models)
    for key in ("outer_fold_stability", "subperiod_stability", "model_selection_stability",
                "feature_selection_stability", "hyperparameter_stability",
                "calibration_stability", "complexity_vs_simple_baseline"):
        assert key in report
    assert report["outer_fold_stability"]["fold_auc_skills"] == [0.01, 0.02, 0.03, 0.04]
    assert report["feature_selection_stability"]["folds"] == 4
    requirements = contract["robustness_requirements"]
    assert len(requirements) == 8


# --------------------------------------------------------------------------
# provenance binding / immutability
# --------------------------------------------------------------------------


def _result_stub(status="WP7_PRE_HOLDOUT_READY", commit="a" * 40):
    contract = wp7.load_contract(REPO_ROOT)
    return {
        "status": status,
        "contract_version": contract["contract_version"],
        "contract_digest": wp7.contract_digest(contract),
        "generation_kind": wp7.GENERATION_KIND,
        "dataset_id": contract["certified_wp6_inputs"]["dataset_id"],
        "target_set_id": contract["certified_wp6_inputs"]["target_set_id"],
        "feature_set_id": contract["certified_wp6_inputs"]["feature_set_id"],
        "wp6_experiment_id": contract["certified_wp6_inputs"]["wp6_experiment_id"],
        "producing_commit": commit,
        "canonical_holdout_id": wp7.CANONICAL_HOLDOUT_ID,
        "seed": contract["seeds"]["model_seed"],
        "outer_folds": [
            {
                "window_id": "outer_test_1",
                "selected_config_id": "modelcfg_00c68894e319",
                "selected_model": "hist_gradient_boosting",
                "selected_strategy": "F_economic_family_representatives",
                "selected_params": {"max_depth": 3},
                "selected_calibration": "none",
                "selected_features": ["roa", "roe"],
                "selection_mean_auc_skill": 0.01,
                "inner_selection_table": [
                    {
                        "config_id": "modelcfg_00c68894e319",
                        "model": "hist_gradient_boosting",
                        "strategy": "F_economic_family_representatives",
                        "params": {"max_depth": 3},
                        "mean_auc_skill": 0.01,
                        "per_fold": {},
                    }
                ],
                "outer_metrics": {"mean_auc_skill": 0.005},
            }
        ],
        "selected_models_by_fold": {"outer_test_1": {"config_id": "modelcfg_00c68894e319"}},
        "controls": [],
        "overall_stop": {"stop": False, "mandatory_failing_controls": []},
        "limitations": ["selection-adjusted nested estimate"],
    }


def test_generation_id_is_deterministic_and_binds_provenance():
    first = wp7.generation_id(_result_stub())
    second = wp7.generation_id(_result_stub())
    assert first == second
    assert first.startswith("generation_") and len(first) == len("generation_") + 12

    binding = wp7.generation_binding(_result_stub())
    assert binding["dataset_id"] == "dataset_35a278e17c13"
    assert binding["target_set_id"] == "target_set_d2bb16610bce"
    assert binding["feature_set_id"] == "feature_set_4f7b43726310"
    assert binding["wp6_experiment_id"] == "experiment_ee434a07a25d"
    assert binding["canonical_holdout_id"] == wp7.CANONICAL_HOLDOUT_ID
    assert binding["seeds"]["model_seed"] == 20260930


def test_generation_id_changes_with_the_producing_commit_and_status():
    base = wp7.generation_id(_result_stub())
    assert wp7.generation_id(_result_stub(commit="b" * 40)) != base
    assert wp7.generation_id(_result_stub(status="WP7_PRE_HOLDOUT_BLOCKED")) != base


def test_ledger_records_selection_evidence_per_outer_fold():
    records = wp7._ledger_records(_result_stub())
    kinds = [item["record"] for item in records]
    assert kinds[0] == "generation_header"
    assert "outer_fold" in kinds
    assert kinds[-1] == "negative_controls"
    fold = [item for item in records if item["record"] == "outer_fold"][0]
    for key in ("selected_config_id", "selected_calibration", "selected_features",
                "selected_params", "mean_selection_auc_skill"):
        assert key in fold


def test_artifacts_are_immutable_and_reject_divergent_rewrites(tmp_path):
    result = _result_stub()
    result["predictions"] = {"phase": "wp7_pre_holdout", "rows": []}
    result["metrics"] = {}
    result["multiple_testing"] = {}
    result["control_evidence"] = {}
    result["robustness"] = {}

    first = wp7.write_generation_artifacts(result, tmp_path)
    assert set(first["written"].values()) == {"written"}
    second = wp7.write_generation_artifacts(result, tmp_path)
    assert set(second["written"].values()) == {"verify_and_reuse"}
    assert first["generation_id"] == second["generation_id"]

    candidate = json.loads(Path(first["candidate_path"]).read_text(encoding="utf-8"))
    assert candidate["holdout_performance_accessed"] is False
    assert candidate["frozen_at_stage"] == "WP7_PRE_HOLDOUT"
    assert candidate["producing_commit"] == result["producing_commit"]
    assert candidate["generation_id"] == first["generation_id"]

    tampered = dict(result)
    tampered["robustness"] = {"changed": True}
    from src.research.immutability import ImmutabilityError

    with pytest.raises(ImmutabilityError):
        wp7.write_generation_artifacts(tampered, tmp_path)


def test_predictions_artifact_is_development_outer_test_only(tmp_path):
    result = _result_stub()
    result["predictions"] = {
        "phase": "wp7_pre_holdout",
        "prediction_scope": "development_outer_test_windows_only",
        "rows": [
            {"window_id": "outer_test_1", "modeling_month": "2017-05",
             "security_id": "SEC001", "prediction": 0.5, "actual": 1.0,
             "feature_asof": "2017-05-31"},
        ],
    }
    result["metrics"] = {}
    result["multiple_testing"] = {}
    result["control_evidence"] = {}
    result["robustness"] = {}
    written = wp7.write_generation_artifacts(result, tmp_path)
    payload = json.loads(
        (Path(written["artifact_dir"]) / "predictions.json").read_text(encoding="utf-8")
    )
    assert payload["prediction_scope"] == "development_outer_test_windows_only"
    for row in payload["rows"]:
        assert row["modeling_month"] < "2021-01"


# --------------------------------------------------------------------------
# clean-worktree / committed-code preflight
# --------------------------------------------------------------------------


def _git_init_repo(root):
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root), check=True)


def _write_nested_script(root, text="# committed code\n"):
    path = root / "scripts" / "research_v2" / "wp7_validation.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_clean_commit_preflight_succeeds_on_clean_tracked_tree(tmp_path):
    _git_init_repo(tmp_path)
    _write_nested_script(tmp_path)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-qm", "clean"], cwd=str(tmp_path), check=True)

    # An unrelated tracked file may legitimately be dirty; only producing-code
    # paths in scripts/research_v2 or src/research block the run.
    (tmp_path / "README.md").write_text("unrelated", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-qm", "unrelated"], cwd=str(tmp_path), check=True)
    (tmp_path / "README.md").write_text("dirty unrelated\n", encoding="utf-8")

    commit = wp7.validate_clean_producing_worktree(tmp_path)
    assert commit
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(tmp_path),
        capture_output=True, text=True, check=True
    ).stdout.strip() == commit


def test_dirty_tracked_producing_code_guard_raises(tmp_path):
    _git_init_repo(tmp_path)
    script = _write_nested_script(tmp_path)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-qm", "clean"], cwd=str(tmp_path), check=True)

    script.write_text(
        "# uncommitted change would not be in HEAD\n", encoding="utf-8"
    )
    with pytest.raises(wp7.Wp7ValidationError, match="modified tracked producing-code files"):
        wp7.validate_clean_producing_worktree(tmp_path)


def test_commit_override_requires_test_flag_and_preflight_still_runs(monkeypatch):
    monkeypatch.setattr(
        wp7, "validate_clean_producing_worktree", lambda root=None: "a" * 40
    )
    monkeypatch.setattr(
        wp7, "load_validation_panel",
        lambda: pytest.fail("preflight must reject override before loading panel"),
    )
    with pytest.raises(wp7.Wp7ValidationError, match="test-only override"):
        wp7.main(["--commit", "b" * 40])


# --------------------------------------------------------------------------
# per-generation candidate persistence / non-overwrite
# --------------------------------------------------------------------------


def test_candidate_persistence_is_per_generation_and_non_overwriting(tmp_path):
    result = _result_stub(commit="a" * 40)
    result["predictions"] = {"phase": "wp7_pre_holdout", "rows": []}
    result["metrics"] = {}
    result["multiple_testing"] = {}
    result["control_evidence"] = {}
    result["robustness"] = {}

    first = wp7.write_generation_artifacts(result, tmp_path)
    assert first["candidate_path"].endswith(
        "%s.json" % first["generation_id"]
    )
    assert not (tmp_path / "provenance" / "wp7" / "model_generation_candidate.json").exists()
    first_bytes = Path(first["candidate_path"]).read_bytes()

    second = wp7.write_generation_artifacts(result, tmp_path)
    assert second["generation_id"] == first["generation_id"]
    assert second["written"]["model_generation_candidate"] == "verify_and_reuse"
    assert Path(first["candidate_path"]).read_bytes() == first_bytes

    # A different producing commit is a different generation; the new candidate
    # must be a separate file and the previous candidate must remain untouched.
    changed = _result_stub(commit="b" * 40)
    changed["predictions"] = result["predictions"]
    changed["metrics"] = {}
    changed["multiple_testing"] = {}
    changed["control_evidence"] = {}
    changed["robustness"] = {}
    changed["controls"] = []
    changed["overall_stop"] = {"stop": False, "mandatory_failing_controls": []}
    changed["selected_models_by_fold"] = result["selected_models_by_fold"]

    third = wp7.write_generation_artifacts(changed, tmp_path)
    assert third["generation_id"] != first["generation_id"]
    assert third["candidate_path"] != first["candidate_path"]
    assert Path(first["candidate_path"]).read_bytes() == first_bytes


# --------------------------------------------------------------------------
# index-keyed generation correction resolver
# --------------------------------------------------------------------------


def _write_wp7_correction_fixture(root, canonical="generation_aaaaaaaaaaaa"):
    (root / "corrections").mkdir(parents=True, exist_ok=True)
    canonical_path = root / "candidates" / ("zzz_%s.json" % canonical)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(json.dumps({
        "generation_id": canonical,
        "canonical_generation_id": canonical,
    }), encoding="utf-8")

    misbound = "generation_04b8e2810b50"
    supersession = {
        "schema_version": wpc7.WP7_SUPERSESSION_SCHEMA_VERSION,
        "old_generation_id": misbound,
        "new_generation_id": canonical,
        "misbound_producing_commit": "f" * 40,
        "corrected_producing_commit": "a" * 40,
        "canonical_ref": str(canonical_path),
    }
    (root / "corrections" / ("aaa_supersession_%s.json" % misbound)).write_text(
        json.dumps(supersession), encoding="utf-8"
    )

    index = {
        "schema_version": wpc7.WP7_CORRECTION_INDEX_SCHEMA_VERSION,
        "canonical_generation_id": canonical,
        "canonical_candidate_path": str(canonical_path),
        "entries": {
            misbound: {
                "status": wpc7.MISBOUND_WITHDRAWN_STATUS,
                "supersession_file": "aaa_supersession_%s.json" % misbound,
            },
            canonical: {"status": wpc7.CANONICAL_STATUS},
        },
    }
    (root / "corrections" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root / "corrections", misbound, canonical, canonical_path


def test_resolver_uses_index_id_keyed_canonical_resolution(tmp_path):
    corrections, misbound, canonical, canonical_path = _write_wp7_correction_fixture(
        tmp_path
    )

    resolved = wpc7.resolve_generation(misbound, root=corrections)
    assert resolved["canonical"] is False
    assert resolved["canonical_generation_id"] == canonical
    assert resolved["status"] == wpc7.MISBOUND_WITHDRAWN_STATUS
    assert resolved["misbound_producing_commit"] == "f" * 40

    canonical_resolved = wpc7.resolve_generation(canonical, root=corrections)
    assert canonical_resolved["canonical"] is True
    assert canonical_resolved["canonical_generation_id"] == canonical
    assert wpc7.canonical_generation_id(root=corrections) == canonical
    assert wpc7.canonical_candidate_path(root=corrections) == canonical_path
    # The deliberately misleading canonical filename proves resolution is not
    # inferred from directory ordering.
    assert wpc7.canonical_candidate_path(root=corrections).name == "zzz_%s.json" % canonical

