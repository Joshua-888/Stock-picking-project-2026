"""
Tests for the V2 research infrastructure package (``src/research``).

These tests are pure infrastructure checks: deterministic identifiers, stable
fingerprints, manifest validation, append-only registry semantics, immutability,
lineage resolution and the research-mode synthetic-data gate. They use tmp_path
only and never touch the production database.
"""

import json

import pandas as pd
import pytest

from src.research import (
    DatasetManifest,
    ExperimentRegistry,
    FeatureSetManifest,
    ImmutabilityError,
    LineageError,
    LineageGraph,
    ManifestValidationError,
    ResearchMode,
    SyntheticDataError,
    TargetManifest,
    assert_no_synthetic_in_research,
    canonical_json,
    current_branch,
    current_git_commit,
    dataset_id,
    experiment_id,
    feature_set_id,
    fingerprint_dataframe,
    fingerprint_file,
    fingerprint_obj,
    is_valid_id,
    load_immutable,
    model_id,
    parse_id,
    save_immutable,
    target_set_id,
    validate_lineage,
)
from src.research.registry import RegistryError, UnknownExperimentError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_frame():
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": ["2019-01-31", "2019-02-28", "2019-03-31"],
            "value": [1.5, 2.5, 3.5],
            "count": [10, 20, 30],
        }
    )


def _experiment_payload(**overrides):
    payload = {
        "hypothesis": "momentum predicts 12m excess return",
        "created_at": "2026-09-24T20:00:00+00:00",
        "git_commit": "f1ca964" + "0" * 33,
        "branch": "feat/v2-wp1-research-contracts",
        "dataset_id": dataset_id({"source": "db", "version": 1}),
        "feature_set_id": feature_set_id({"features": ["momentum"]}),
        "target_set_id": target_set_id({"target": "future_12m_excess_return"}),
        "model_id": None,
        "config": {"seed": 7, "model": "logistic"},
        "random_seed": 7,
        "training_period": ["2000-01-01", "2015-12-31"],
        "validation_period": ["2016-01-01", "2018-12-31"],
        "holdout_usage": "never touched",
        "artifacts": [],
        "status": "CREATED",
    }
    payload.update(overrides)
    payload["experiment_id"] = experiment_id(_experiment_id_payload(payload))
    return payload


def _experiment_id_payload(payload):
    """Identity payload used to derive the experiment ID (excludes experiment_id)."""
    return {key: value for key, value in payload.items() if key != "experiment_id"}


def _complete_dataset_manifest():
    payload = {"source": "stock_analysis.db", "version": 1}
    return DatasetManifest(
        dataset_id=dataset_id(payload),
        created_at="2026-09-24T20:00:00+00:00",
        git_commit="f1ca964" + "0" * 33,
        branch="feat/v2-wp1-research-contracts",
        sources=["yfinance", "sec_edgar"],
        universe_definition="S&P 500 point-in-time members",
        period_start="2000-01-01",
        period_end="2024-12-31",
        row_count=1234,
        schema_version="v2.0",
        config_fingerprint=fingerprint_obj({"config": "default"}),
        source_fingerprints={"yfinance": "a" * 64},
        dataset_fingerprint=fingerprint_obj(payload),
        pit_status="point_in_time",
        synthetic_data_status="none",
        known_limitations=["EDGAR coverage gaps for delisted issuers"],
    )


# ── 1. Deterministic identifiers ──────────────────────────────────────────────

def test_identifiers_are_deterministic_and_content_addressed():
    payload = {"provider": "yfinance", "version": 1, "rows": 100}
    assert dataset_id(payload) == dataset_id(dict(payload))
    assert dataset_id(payload) != dataset_id({**payload, "rows": 101})

    experiment = {"hypothesis": "h1", "seed": 7}
    assert experiment_id(experiment) == experiment_id({"seed": 7, "hypothesis": "h1"})
    assert experiment_id(experiment) != experiment_id({"hypothesis": "h2", "seed": 7})

    model = {"estimator": "logistic", "features": ["a", "b"]}
    assert model_id(model) == model_id({"estimator": "logistic", "features": ["a", "b"]})
    assert model_id(model) != model_id({"estimator": "ridge", "features": ["a", "b"]})

    for produced, kind in (
        (dataset_id(payload), "dataset"),
        (experiment_id(experiment), "experiment"),
        (model_id(model), "model"),
    ):
        assert produced.startswith(kind + "_")
        assert is_valid_id(produced, kind)
        assert parse_id(produced)[0] == kind
        assert len(produced.split("_", 1)[1]) == 12

    assert not is_valid_id("dataset_zzzz", "dataset")
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


# ── 2. Deterministic fingerprints ─────────────────────────────────────────────

def test_fingerprints_are_stable_and_data_sensitive(tmp_path):
    frame = _sample_frame()
    assert fingerprint_dataframe(frame) == fingerprint_dataframe(frame)
    assert fingerprint_dataframe(frame) == fingerprint_dataframe(_sample_frame())
    assert fingerprint_dataframe(frame) == fingerprint_dataframe(frame.copy())

    changed = _sample_frame()
    changed.loc[0, "value"] = 99.0
    assert fingerprint_dataframe(changed) != fingerprint_dataframe(frame)

    extra = _sample_frame()
    extra["new_column"] = [1, 2, 3]
    assert fingerprint_dataframe(extra) != fingerprint_dataframe(frame)

    assert fingerprint_obj({"a": 1, "b": [1, 2]}) == fingerprint_obj({"b": [1, 2], "a": 1})
    assert fingerprint_obj({"a": 1}) != fingerprint_obj({"a": 2})

    target = tmp_path / "payload.bin"
    target.write_bytes(b"research-bytes")
    assert fingerprint_file(target) == fingerprint_file(target)
    assert fingerprint_file(target) == fingerprint_file(tmp_path / "payload.bin")
    identical = tmp_path / "copy.bin"
    identical.write_bytes(b"research-bytes")
    assert fingerprint_file(identical) == fingerprint_file(target)
    different = tmp_path / "other.bin"
    different.write_bytes(b"research-bytes!")
    assert fingerprint_file(different) != fingerprint_file(target)


# ── 3. Manifest validation ────────────────────────────────────────────────────

def test_manifests_validate_complete_input():
    dataset_manifest = _complete_dataset_manifest()
    assert dataset_manifest.validate() is True
    assert DatasetManifest.from_dict(dataset_manifest.to_dict()).validate() is True

    feature_manifest = FeatureSetManifest(
        feature_set_id=feature_set_id({"features": ["momentum_12m"]}),
        dataset_id=dataset_manifest.dataset_id,
        features=["momentum_12m"],
        feature_definition_versions={"momentum_12m": "v1"},
        transformations=["winsorise_1_99"],
        availability_policy_ref="availability_policy:v1",
        missingness_policy="drop_row_if_label_missing",
        fingerprint=fingerprint_obj({"momentum_12m": "v1"}),
        git_commit=dataset_manifest.git_commit,
    )
    assert feature_manifest.validate() is True

    target_manifest = TargetManifest(
        target_set_id=target_set_id({"target": "future_12m_excess_return"}),
        dataset_id=dataset_manifest.dataset_id,
        target_definitions={"future_12m_excess_return": "stock_12m_return - spy_12m_return"},
        horizon="12m",
        benchmark="SPY",
        observability_semantics={"observable_at": "T + 12 months"},
        construction_version="v1",
        fingerprint=fingerprint_obj({"target": "future_12m_excess_return"}),
    )
    assert target_manifest.validate() is True


def test_manifests_raise_on_missing_or_malformed_fields():
    complete = _complete_dataset_manifest()

    incomplete = DatasetManifest(**{**complete.to_dict(), "git_commit": ""})
    with pytest.raises(ManifestValidationError):
        incomplete.validate()

    malformed = DatasetManifest(**{**complete.to_dict(), "config_fingerprint": "not-hex"})
    with pytest.raises(ManifestValidationError):
        malformed.validate()

    bad_id = DatasetManifest(**{**complete.to_dict(), "dataset_id": "dataset_short"})
    with pytest.raises(ManifestValidationError):
        bad_id.validate()

    wrong_kind = DatasetManifest(**{**complete.to_dict(), "dataset_id": experiment_id({"a": 1})})
    with pytest.raises(ManifestValidationError):
        wrong_kind.validate()

    with pytest.raises(ManifestValidationError):
        DatasetManifest.from_dict({**complete.to_dict(), "unexpected": 1})

    with pytest.raises(ManifestValidationError):
        FeatureSetManifest(
            feature_set_id=feature_set_id({"features": ["momentum_12m"]}),
            dataset_id=complete.dataset_id,
            features=["momentum_12m"],
            feature_definition_versions={},
            transformations=[],
            availability_policy_ref="availability_policy:v1",
            missingness_policy="drop_row_if_label_missing",
            fingerprint=fingerprint_obj({"x": 1}),
            git_commit=complete.git_commit,
        ).validate()


# ── 4. Append-only registry ───────────────────────────────────────────────────

def test_registry_add_reuse_conflict_and_persistence(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry")
    payload = _experiment_payload()

    exp_id, outcome = registry.register(payload)
    assert outcome == "written"
    assert exp_id == payload["experiment_id"]

    same_id, outcome = registry.register(payload)
    assert same_id == exp_id
    assert outcome == "verify_and_reuse"
    assert len(list((tmp_path / "registry").glob("experiment_*.json"))) == 1

    conflicting = {**payload, "hypothesis": "different claim", "status": "COMPLETED"}
    assert conflicting["experiment_id"] == exp_id
    with pytest.raises(RegistryError):
        registry.register(conflicting)

    stored = registry.get(exp_id)
    assert stored["hypothesis"] == payload["hypothesis"]
    assert stored["status"] == "CREATED"

    reloaded = ExperimentRegistry(tmp_path / "registry")
    assert reloaded.get(exp_id)["hypothesis"] == payload["hypothesis"]
    assert reloaded.list_experiments() == registry.list_experiments()
    assert len(reloaded.list_experiments()) == 1

    assert reloaded.fingerprint(exp_id) == registry.fingerprint(exp_id)
    raw = (tmp_path / "registry" / (exp_id + ".json")).read_text(encoding="utf-8")
    assert json.loads(raw)["experiment_id"] == exp_id

    with pytest.raises(UnknownExperimentError):
        registry.get(experiment_id({"never": "registered"}))


def test_registry_rejects_invalid_records(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry")
    payload = _experiment_payload()

    with pytest.raises(RegistryError):
        registry.register({**payload, "status": "NOT_A_STATUS"})
    with pytest.raises(RegistryError):
        registry.register({**payload, "random_seed": None})
    with pytest.raises(RegistryError):
        registry.register({**payload, "unknown_field": 1})
    with pytest.raises(RegistryError):
        registry.register({**payload, "dataset_id": "dataset_bogus"})

    assert registry.list_experiments() == []


# ── 5. Immutability ───────────────────────────────────────────────────────────

def test_immutability_write_once_semantics(tmp_path):
    path = tmp_path / "artifacts" / "record.json"
    original = {"b": 2, "a": 1}

    assert save_immutable(path, original) == "written"
    assert save_immutable(path, {"a": 1, "b": 2}) == "verify_and_reuse"
    assert load_immutable(path) == original

    with pytest.raises(ImmutabilityError):
        save_immutable(path, {"a": 1, "b": 3})
    assert load_immutable(path) == original

    missing = tmp_path / "artifacts" / "absent.json"
    with pytest.raises(FileNotFoundError):
        load_immutable(missing)


# ── 6. Lineage ────────────────────────────────────────────────────────────────

def test_lineage_resolves_valid_chain(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry")
    dataset_manifest = _complete_dataset_manifest()
    payload = _experiment_payload(
        dataset_id=dataset_manifest.dataset_id,
        model_id=model_id({"estimator": "logistic"}),
    )
    registry.register(payload)

    from src.research import build_lineage

    graph = build_lineage(manifests=[dataset_manifest], registry=registry)
    assert validate_lineage(graph) is True

    experiment = payload["experiment_id"]
    chain = graph.resolve_lineage(payload["model_id"])
    assert chain[-1] == payload["model_id"]
    assert experiment in chain
    assert dataset_manifest.dataset_id in chain
    assert chain.index(dataset_manifest.dataset_id) < chain.index(experiment)


def test_lineage_manual_chain_and_dangling_reference():
    graph = LineageGraph()
    ids = {
        "dataset": dataset_id({"d": 1}),
        "target_set": target_set_id({"t": 1}),
        "feature_set": feature_set_id({"f": 1}),
        "experiment": experiment_id({"e": 1}),
        "model": model_id({"m": 1}),
    }
    from src.research import audit_id, prediction_id, promotion_id, validation_id

    ids["prediction"] = prediction_id({"p": 1})
    ids["validation"] = validation_id({"v": 1})
    ids["audit"] = audit_id({"a": 1})
    ids["promotion"] = promotion_id({"pr": 1})

    graph.add_node(ids["dataset"])
    graph.add_node(ids["target_set"], parents=(ids["dataset"],))
    graph.add_node(ids["feature_set"], parents=(ids["dataset"],))
    graph.add_node(ids["experiment"], parents=(ids["target_set"], ids["feature_set"], ids["dataset"]))
    graph.add_node(ids["model"], parents=(ids["experiment"],))
    graph.add_node(ids["prediction"], parents=(ids["model"],))
    graph.add_node(ids["validation"], parents=(ids["prediction"],))
    graph.add_node(ids["audit"], parents=(ids["validation"],))
    graph.add_node(ids["promotion"], parents=(ids["audit"],))

    assert validate_lineage(graph) is True
    chain = graph.resolve_lineage(ids["promotion"])
    assert chain == [
        ids["dataset"],
        ids["target_set"],
        ids["feature_set"],
        ids["experiment"],
        ids["model"],
        ids["prediction"],
        ids["validation"],
        ids["audit"],
        ids["promotion"],
    ]

    graph.add_node(ids["model"], parents=(ids["experiment"],))
    assert len(graph) == 9

    dangling = LineageGraph()
    orphan_model = model_id({"orphan": True})
    dangling.add_node(orphan_model, parents=(experiment_id({"missing": True}),))
    with pytest.raises(LineageError):
        validate_lineage(dangling)
    with pytest.raises(LineageError):
        dangling.resolve_lineage(orphan_model)
    with pytest.raises(LineageError):
        dangling.resolve_lineage(model_id({"unknown": True}))

    wrong_direction = LineageGraph()
    wrong_direction.add_node(ids["dataset"], kind="dataset")
    wrong_direction.add_node(ids["experiment"], kind="experiment", parents=(ids["dataset"],))
    wrong_direction.add_node(ids["target_set"], kind="target_set", parents=(ids["experiment"],))
    with pytest.raises(LineageError):
        validate_lineage(wrong_direction)

    conflicting = LineageGraph()
    conflicting.add_node(ids["dataset"], kind="dataset")
    with pytest.raises(LineageError):
        conflicting.add_node(ids["dataset"], kind="experiment")


# ── 7. Research-mode synthetic gate ───────────────────────────────────────────

def test_research_mode_blocks_synthetic_data():
    with pytest.raises(SyntheticDataError):
        assert_no_synthetic_in_research(ResearchMode.RESEARCH_V2, True)
    with pytest.raises(SyntheticDataError):
        assert_no_synthetic_in_research("RESEARCH_V2", True, context="unit-test")

    assert assert_no_synthetic_in_research(ResearchMode.RESEARCH_V2, False) is True
    assert assert_no_synthetic_in_research(ResearchMode.LEGACY_V1, True) is True

    with pytest.warns(UserWarning):
        assert_no_synthetic_in_research(ResearchMode.LEGACY_V1, True)

    commit = current_git_commit()
    branch = current_branch()
    assert commit is None or isinstance(commit, str)
    assert branch is None or isinstance(branch, str)

    isolated = current_git_commit(cwd="/nonexistent-research-dir")
    assert isolated is None
