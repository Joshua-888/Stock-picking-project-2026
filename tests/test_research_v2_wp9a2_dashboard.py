"""WP9A.2 apply-only dashboard explainability and truth layer tests.

These tests use the REAL frozen champion only through the existing
hash-verified apply-only loader. They never fit the frozen model/preprocessor/
calibrator, never read target/holdout-label columns, and use temporary roots for
any registry behavior so immutable provenance is never mutated.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.research.dashboard import attribution, explain, schemas  # noqa: E402
from src.research.dashboard import summary as summary_mod  # noqa: E402
from src.research.dashboard.contract import (  # noqa: E402
    freshness_completeness,
    latest_shadow_ranking,
    official_snapshot_history,
)
from src.research.dashboard.integrity import audit_status  # noqa: E402
from src.research.dashboard.identity import artifact_identity  # noqa: E402
from src.research.wp9.champion import ChampionError, load_champion  # noqa: E402


FROZEN_FEATURES = list(schemas.FEATURE_ORDER)
SYNTHETIC_FRAME = pd.DataFrame(
    [
        {
            "earnings_yield": 0.04,
            "current_pe": 18.2,
            "price_to_book": 1.6,
            "free_cash_flow_yield": 0.02,
            "dividend_yield": 0.015,
            "roa": 0.05,
            "roe": 0.12,
            "eps_growth_acceleration": 0.01,
            "six_month_momentum": 0.08,
            "twelve_month_momentum": 0.13,
            "five_year_price_gain": 0.35,
            "debt_to_equity": 0.75,
            "market_cap": 8.5e10,
        },
        {
            "earnings_yield": None,
            "current_pe": 22.1,
            "price_to_book": None,
            "free_cash_flow_yield": 0.04,
            "dividend_yield": None,
            "roa": 0.03,
            "roe": None,
            "eps_growth_acceleration": -0.01,
            "six_month_momentum": -0.05,
            "twelve_month_momentum": -0.02,
            "five_year_price_gain": -0.20,
            "debt_to_equity": 1.30,
            "market_cap": 2.1e9,
        },
        {
            "earnings_yield": -0.01,
            "current_pe": 12.0,
            "price_to_book": 0.95,
            "free_cash_flow_yield": -0.03,
            "dividend_yield": 0.06,
            "roa": 0.09,
            "roe": 0.10,
            "eps_growth_acceleration": 0.2,
            "six_month_momentum": 0.11,
            "twelve_month_momentum": 0.04,
            "five_year_price_gain": 0.05,
            "debt_to_equity": 0.45,
            "market_cap": 1.9e11,
        },
        {
            "earnings_yield": 0.06,
            "current_pe": 9.5,
            "price_to_book": 1.2,
            "free_cash_flow_yield": 0.09,
            "dividend_yield": 0.03,
            "roa": 0.01,
            "roe": 0.02,
            "eps_growth_acceleration": 0.0,
            "six_month_momentum": 0.0,
            "twelve_month_momentum": 0.0,
            "five_year_price_gain": 0.0,
            "debt_to_equity": 2.0,
            "market_cap": 3.3e8,
        },
    ]
)

# exact source path -> desired value for read-only lineage checks
_LINEAGE_CASES = [
    ("wp5", lambda item: item["candidate_count"], 23),
    ("wp5", lambda item: item["robust_candidate_count"], 0),
    ("wp5", lambda item: item["fdr"]["method"], "benjamini_hochberg"),
    ("wp5", lambda item: item["fdr"]["alpha"], 0.05),
    ("wp5", lambda item: item["fdr"]["hypotheses"], 23),
    ("wp5", lambda item: item["fdr"]["rejected"], 0),
    ("wp5", lambda item: item["status"], "WP5_COMPLETE"),
    ("wp6", lambda item: item["configuration_count"], 312),
    ("wp6", lambda item: item["experiment_id"], "experiment_ee434a07a25d"),
    ("wp8", lambda item: item["holdout_primary_metrics"]["mean_auc"], 0.5414600859434548),
    ("wp8", lambda item: item["holdout_primary_metrics"]["mean_auc_skill"], 0.04146008594345476),
    ("wp8", lambda item: item["holdout_primary_metrics"]["one_sided_hac_p"], 0.13096259515179293),
    ("wp8", lambda item: item["holdout_primary_metrics"]["hac_lag"], 12),
    ("wp8", lambda item: item["holdout_primary_metrics"]["months"], 44),
    ("wp8", lambda item: item["holdout_primary_metrics"]["verdict"], "HOLDOUT_POSITIVE_BUT_INCONCLUSIVE"),
    ("wp8", lambda item: item["training"]["rows"], 82971),
    ("wp8", lambda item: item["training"]["securities"], 741),
]

_FORBIDDEN_COLUMN_TOKENS = (
    "target_observable",
    "target_censored",
    "target_censor_reason",
    "terminal_price_observable",
    "future_12m_excess_return",
    "future_12m_stock_return",
    "future_12m_benchmark_return",
    "future_return",
    "outperform_12m",
    "target_known_at",
)


@pytest.fixture(scope="module")
def champion():
    return load_champion(REPO_ROOT)


def _temp_registry_root(tmp_path: Path, registry=None, dry_entries=None, prediction_entries=None):
    wp9 = tmp_path / "provenance" / "wp9"
    (wp9 / "dry_runs").mkdir(parents=True, exist_ok=True)
    (wp9 / "predictions").mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else {
        "schema_version": "wp9_run_registry_v1",
        "current_prospective_stage": "A_OPERATIONAL_SHADOW",
        "official_snapshot_ids": [],
        "dry_run_ids": [],
        "matured_evaluations": [],
        "invalidated_snapshots": [],
    }
    (wp9 / "index.json").write_text(json.dumps(registry), encoding="utf-8")
    (wp9 / "dry_runs" / "index.json").write_text(
        json.dumps({"schema_version": "wp9_prediction_index_v1",
                     "entries": dry_entries or []}), encoding="utf-8")
    (wp9 / "predictions" / "index.json").write_text(
        json.dumps({"schema_version": "wp9_prediction_index_v1",
                     "entries": prediction_entries or []}), encoding="utf-8")
    return wp9


def _hashes(paths):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_frozen_champion_loads_hash_verified_and_is_apply_only(champion, monkeypatch):
    record = json.loads((REPO_ROOT / "provenance" / "wp8" / "final_candidate_freeze.json").read_text())
    hashes = record["artifact_hashes"]
    assert champion.model_sha256 == hashes["model"]["sha256"]
    assert champion.preprocessor_sha256 == hashes["preprocessor"]["sha256"]
    assert champion.calibrator_sha256 == hashes["calibrator"]["sha256"]
    assert champion.freeze_id == "freeze_e62eac30df40"
    assert tuple(champion.features) == schemas.FEATURE_ORDER

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen object was refit")

    preprocessor_cls = type(champion._preprocessor_obj["preprocessor"])
    estimator_cls = type(champion._model_obj["estimator"])
    calibrator_cls = type(champion._calibrator)
    for method in ("fit", "fit_transform", "fit_predict"):
        if hasattr(preprocessor_cls, method):
            monkeypatch.setattr(preprocessor_cls, method, forbidden)
        if hasattr(estimator_cls, method):
            monkeypatch.setattr(estimator_cls, method, forbidden)
        if hasattr(calibrator_cls, method):
            monkeypatch.setattr(calibrator_cls, method, forbidden)

    raw, calibrated = champion.predict(SYNTHETIC_FRAME.copy())
    assert raw.shape == (len(SYNTHETIC_FRAME),)
    assert calibrated.shape == raw.shape
    assert np.isfinite(raw).all()
    assert np.isfinite(calibrated).all()


def test_artifact_hash_verification_rejects_tampering(tmp_path):
    tampered = tmp_path / "tampered.pkl"
    tampered.write_bytes(b"not the real artifact")
    features = list(schemas.FEATURE_ORDER)
    artifacts = {
        name: {"path": str(tampered), "sha256": "0" * 64}
        for name in ("model", "preprocessor", "calibrator")
    }
    record = {
        "freeze_id": "freeze_e62eac30df40",
        "final_selected_features": features,
        "final_calibration": "platt_scaling",
        "artifact_hashes": artifacts,
    }
    with pytest.raises(ChampionError) as exc:
        load_champion(REPO_ROOT, freeze_record=record)
    assert "hash mismatch" in str(exc.value)


def test_feature_importance_collapses_26_to_13_and_sums_to_share(champion):
    importance = attribution.feature_importance(champion)
    assert len(importance) == 13
    assert list(importance) == list(schemas.FEATURE_ORDER)
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-9)
    transformed = attribution._transform_frame(champion, SYNTHETIC_FRAME)[1]
    assert list(transformed.columns) == list(schemas.TRANSFORMED_COLUMN_ORDER)
    assert len(transformed.columns) == 26
    # explicit missing-column collapse behavior
    row = pd.Series({
        "earnings_yield": 0.25,
        "earnings_yield__missing": -0.05,
        "market_cap": 0.7,
        "market_cap__missing": 0.1,
    })
    collapsed = attribution._collapse_to_base(row)
    assert collapsed["earnings_yield"] == pytest.approx(0.20)
    assert collapsed["market_cap"] == pytest.approx(0.80)


def test_tree_path_attribution_is_deterministic_and_reconciles(champion):
    frame = SYNTHETIC_FRAME.iloc[[0]].reset_index(drop=True)
    first = attribution.tree_path_local(champion, frame)
    second = attribution.tree_path_local(champion, frame.copy())
    assert first.contributions == second.contributions
    assert first.raw_model_score == pytest.approx(second.raw_model_score, abs=0)
    total = first.expected_value + sum(first.contributions.values())
    assert total == pytest.approx(first.raw_model_score, abs=attribution.ATTRIBUTION_TOLERANCE)
    assert abs(first.residual) <= attribution.ATTRIBUTION_TOLERANCE


def test_global_attribution_and_family_aggregation(champion):
    by_feature = attribution.global_attribution_by_feature(champion, SYNTHETIC_FRAME)
    assert len(by_feature) == 13
    assert list(by_feature) == list(schemas.FEATURE_ORDER)
    assert all(np.isfinite(value) for value in by_feature.values())
    by_family = attribution.family_attribution(by_feature)
    assert list(by_family) == list(schemas.FAMILY_ORDER)
    # family sums are exactly the sum of member mean-|attribution| values
    expected_family = {
        family: sum(
            value for feature, value in by_feature.items()
            if schemas.FEATURE_FAMILY_MAP[feature] == family
        )
        for family in schemas.FAMILY_ORDER
    }
    assert by_family == pytest.approx(expected_family, abs=1e-12)


def test_schema_validation_for_all_9_contracts(champion, tmp_path):
    root_contracts = {
        "research_summary": summary_mod.research_summary(REPO_ROOT),
        "model_definition": summary_mod.model_definition(REPO_ROOT),
        "latest_shadow_ranking": latest_shadow_ranking(REPO_ROOT),
        "freshness_completeness": freshness_completeness(REPO_ROOT),
        "official_snapshot_history": official_snapshot_history(REPO_ROOT),
        "integrity_audit_status": audit_status(REPO_ROOT),
    }
    importance = attribution.feature_importance(champion)
    root_contracts["global_feature_importance"] = {
        **schemas.schema_basis("global_feature_importance"),
        "freeze_id": champion.freeze_id,
        "importance_kind": attribution.IMPORTANCE_KIND,
        "basis": "frozen ExtraTrees feature_importances_ collapsed from 26 to 13 base features",
        "features": list(schemas.FEATURE_ORDER),
        "shares": importance,
        "limitations": list(attribution.ATTRIBUTION_LIMITATIONS),
        "artifact_identity": artifact_identity(REPO_ROOT),
    }
    by_feature = attribution.global_attribution_by_feature(champion, SYNTHETIC_FRAME)
    by_family = attribution.family_attribution(by_feature)
    root_contracts["feature_family_attribution"] = {
        **schemas.schema_basis("feature_family_attribution"),
        "freeze_id": champion.freeze_id,
        "attribution_kind": attribution.ATTRIBUTION_KIND,
        "attribution_basis": "mean absolute deterministic tree-path attribution",
        "sample": {
            "rows": int(len(SYNTHETIC_FRAME)),
            "securities": 1,
            "months": 1,
            "date_min": "2020-01-31",
            "date_max": "2020-01-31",
            "seed": 20261009,
            "selection_rule": "synthetic deterministic fixture",
            "sample_size": int(len(SYNTHETIC_FRAME)),
        },
        "by_feature": by_feature,
        "by_family": by_family,
        "limitations": list(attribution.ATTRIBUTION_LIMITATIONS),
        "artifact_identity": artifact_identity(REPO_ROOT),
    }
    root_contracts["stock_explanations"] = explain.explain_security(
        champion,
        SYNTHETIC_FRAME.iloc[[0]].reset_index(drop=True),
        ticker="AAA",
        security_id="AAA",
        as_of_date="2020-01-31",
    )
    for kind in schemas.CONTRACT_KINDS:
        assert schemas.validate_dashboard_artifact(kind, root_contracts[kind]) == kind


def test_every_contract_carries_the_same_validated_frozen_identity(champion):
    """All nine contracts must expose one identical identity block binding them
    to the frozen champion, feature order, WP9 contract digest, producing commit,
    and dataset/feature/target IDs."""
    expected = artifact_identity(REPO_ROOT)
    contracts = {
        "research_summary": summary_mod.research_summary(REPO_ROOT),
        "model_definition": summary_mod.model_definition(REPO_ROOT),
        "latest_shadow_ranking": latest_shadow_ranking(REPO_ROOT),
        "freshness_completeness": freshness_completeness(REPO_ROOT),
        "official_snapshot_history": official_snapshot_history(REPO_ROOT),
        "integrity_audit_status": audit_status(REPO_ROOT),
        "stock_explanations": explain.explain_security(
            champion,
            SYNTHETIC_FRAME.iloc[[0]].reset_index(drop=True),
            ticker="AAA",
            security_id="AAA",
            as_of_date="2020-01-31",
        ),
    }
    assert expected["freeze_id"] == "freeze_e62eac30df40"
    assert expected["feature_order"] == list(schemas.FEATURE_ORDER)
    assert set(expected["model_hash"]) <= set("0123456789abcdef")
    assert len(expected["model_hash"]) == 64
    assert expected["dataset_id"]
    assert expected["feature_set_id"]
    assert expected["target_set_id"]
    assert expected["producing_commit"]
    for name, contract in contracts.items():
        assert contract["artifact_identity"] == expected, name


def test_metric_lineage_exact_source_path_and_values():
    research = summary_mod.research_summary(REPO_ROOT)
    for section, accessor, expected in _LINEAGE_CASES:
        item = research[section]
        lineage = accessor(item)
        if isinstance(expected, float):
            assert lineage["value"] == pytest.approx(expected, rel=1e-12, abs=0)
        else:
            assert lineage["value"] == expected
        assert lineage["source"]["path"]
        assert lineage["source"]["field"]
    # WP7 exact fold series and pooled mean
    fold_values = [item["value"] for item in research["wp7"]["outer_fold_mean_auc_skill"]]
    expected_folds = [
        0.039048149480333354,
        0.01501194929250523,
        0.012334477480521447,
        0.027729397538139116,
    ]
    assert fold_values == pytest.approx(expected_folds, rel=1e-12, abs=0)
    assert research["wp7"]["mean_mean_auc_skill"]["value"] == pytest.approx(
        0.023530993447874786, rel=1e-12, abs=0)


def test_dashboard_package_has_no_target_or_holdout_column_access_source_level():
    package_dir = REPO_ROOT / "src" / "research" / "dashboard"
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        names = {getattr(node, "id", None) for node in ast.walk(tree) if isinstance(node, ast.Name)}
        # constants loaded from files are allowed, but column identifiers are not
        for token in _FORBIDDEN_COLUMN_TOKENS:
            assert token not in names, "%s contains forbidden identifier %s" % (path.name, token)


def test_audit_status_reports_no_refit_no_holdout_and_no_unsupported_claims():
    status = audit_status(REPO_ROOT)
    assert status["frozen_artifacts_verified"] is True
    assert status["no_refit_guard"] is True
    assert status["apply_only_check"] is True
    assert status["no_wp8_holdout_reevaluation"] is True
    assert status["read_only_registry_access"] is True
    assert status["no_unsupported_probability_claims"] is True


def test_contract_functions_are_read_only_and_do_not_mutate_registry(tmp_path):
    registry = {
        "schema_version": "wp9_run_registry_v1",
        "current_prospective_stage": "A_OPERATIONAL_SHADOW",
        "official_snapshot_ids": [],
        "dry_run_ids": [],
        "matured_evaluations": [],
        "invalidated_snapshots": [],
    }
    wp9 = _temp_registry_root(tmp_path, registry=registry)
    paths = [wp9 / "index.json", wp9 / "dry_runs" / "index.json", wp9 / "predictions" / "index.json"]
    before = _hashes(paths)
    latest_shadow_ranking(tmp_path)
    official_snapshot_history(tmp_path)
    freshness_completeness(tmp_path)
    after = _hashes(paths)
    assert before == after


def test_empty_official_history_and_dry_run_labeling(tmp_path):
    dry_entries = [{
        "snapshot_id": "dry_abc",
        "snapshot_asof": "2026-10-08",
        "model_freeze_id": "freeze_e62eac30df40",
        "model_hash": "m" * 64,
        "universe_size": 503,
    }]
    _temp_registry_root(tmp_path, dry_entries=dry_entries, prediction_entries=[])
    history = official_snapshot_history(tmp_path)
    assert history["official_snapshot_ids"] == []
    assert history["history"] == []
    ranking = latest_shadow_ranking(tmp_path)
    assert len(ranking["dry_run_rankings"]) == 1
    item = ranking["dry_run_rankings"][0]
    assert item["snapshot_kind"] == "NON_EVIDENTIARY_DRY_RUN"
    assert item["snapshot_asof"] == "2026-10-08"
    assert ranking["official_snapshot_ids"] == []


def test_stale_latest_dry_run_exposes_exact_asof(tmp_path):
    old = "2026-10-07"
    new = "2026-10-08"
    entries = [
        {"snapshot_id": "old", "snapshot_asof": old},
        {"snapshot_id": "new", "snapshot_asof": new},
    ]
    _temp_registry_root(tmp_path, dry_entries=entries)
    freshness = freshness_completeness(tmp_path)
    assert freshness["dry_run_count"] == 2
    assert freshness["official_snapshot_count"] == 0
    assert freshness["latest_dry_run"]["snapshot_asof"] == new


def test_emitted_contract_copy_has_no_unsupported_probability_claims():
    outputs = [
        summary_mod.research_summary(REPO_ROOT),
        summary_mod.model_definition(REPO_ROOT),
        latest_shadow_ranking(REPO_ROOT),
        official_snapshot_history(REPO_ROOT),
        freshness_completeness(REPO_ROOT),
    ]
    for obj in outputs:
        text = json.dumps(obj, sort_keys=True, default=str).lower()
        for phrase in schemas.FORBIDDEN_PHRASES:
            assert phrase.lower() not in text
