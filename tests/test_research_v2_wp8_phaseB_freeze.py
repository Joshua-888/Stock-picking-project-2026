"""WP8 Phase B final-candidate freeze tests (synthetic/development fixtures only).

These tests verify the canonical WP7 resolver, absence of stale fixed-path
fallback, exact canonical holdout identity, hard development-time guards,
frozen candidate-set order, deterministic freeze-id recomputation, and
write-once immutability. No locked-holdout row, label, return, prediction or
2022+ outcome is constructed or read here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts" / "research_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_wp8():
    if "wp8_freeze_final_candidate_test_module" in sys.modules:
        return sys.modules["wp8_freeze_final_candidate_test_module"]
    path = SCRIPTS / "wp8_freeze_final_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "wp8_freeze_final_candidate_test_module", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp8_freeze_final_candidate_test_module"] = module
    spec.loader.exec_module(module)
    return module


from src.research import wp7_generation_corrections as wpc7  # noqa: E402
from src.research.holdout import locked_holdout  # noqa: E402
from src.research.immutability import ImmutabilityError  # noqa: E402

WITHDRAWN_GENERATIONS = (
    "generation_04b8e2810b50",
    "generation_3e4e461b2e44",
    "generation_481f16d3a44b",
)
CANONICAL_GENERATION = "generation_de0f9bbd0bec"


@pytest.fixture(scope="module")
def wp8():
    return _load_wp8()


@pytest.fixture(scope="module")
def real_holdout():
    return locked_holdout()


@pytest.fixture(scope="module")
def contract(wp8):
    wp7 = wp8.load_wp7_module()
    return wp7.load_contract(REPO_ROOT)


def _feature_frame(*dates):
    """Minimal row frame carrying only the temporal access column."""
    return pd.DataFrame({"feature_asof": list(dates)})


def test_canonical_wp7_resolver_rejects_withdrawn_generations():
    for generation in WITHDRAWN_GENERATIONS:
        resolved = wpc7.resolve_generation(generation)
        assert resolved["canonical"] is False
        assert resolved["canonical_generation_id"] == CANONICAL_GENERATION
        assert resolved["status"] != wpc7.CANONICAL_STATUS
        with pytest.raises(wpc7.Wp7GenerationCorrectionError):
            wpc7.assert_canonical(generation)


def test_canonical_wp7_resolver_accepts_only_the_canonical_generation():
    resolved = wpc7.assert_canonical(CANONICAL_GENERATION)
    assert resolved["canonical"] is True
    assert resolved["canonical_generation_id"] == CANONICAL_GENERATION
    assert wpc7.canonical_generation_id() == CANONICAL_GENERATION


def test_canonical_candidate_never_falls_back_to_stale_fixed_path(wp8, monkeypatch, tmp_path):
    # Break the resolver-provided canonical candidate path; any fallback to
    # provenance/wp7/model_generation_candidate.json would violate the frozen
    # resolver contract and must not happen.
    missing = tmp_path / "missing_generation_de0f9bbd0bec.json"
    monkeypatch.setattr(wpc7, "canonical_candidate_path", lambda root=None: missing)
    with pytest.raises(wp8.Wp8FreezeError, match="canonical WP7 candidate is missing"):
        wp8.load_canonical_wp7_candidate()


def test_exact_canonical_holdout_id_is_required(wp8, real_holdout):
    wrong = SimpleNamespace(holdout_id="holdout_wrong")
    with pytest.raises(wp8.Wp8FreezeError, match="wrong locked holdout id"):
        wp8.require_canonical_holdout(wrong)
    assert wp8.require_canonical_holdout(real_holdout) is True


def test_final_freeze_rejects_2021_and_embargo_holdout_feature_asof(wp8, real_holdout):
    legal = _feature_frame("2020-12-31")
    assert wp8.assert_legal_training_frame(legal, real_holdout) is True

    for boundary_row in (
        "2021-01-01",
        "2021-06-15",
        "2021-12-31",
        "2022-01-01",
        "2023-07-01",
    ):
        frame = pd.concat([legal, _feature_frame(boundary_row)], ignore_index=True)
        with pytest.raises(wp8.Wp8FreezeError):
            wp8.assert_legal_training_frame(frame, real_holdout)


def test_frozen_candidate_set_is_baseline_plus_nine_in_exact_order(wp8, contract):
    configs = wp8.frozen_configs(contract)
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


def test_freeze_id_is_deterministic(wp8):
    args = {
        "wp7_generation_id": CANONICAL_GENERATION,
        "wp7_contract_version": "WP7_VALIDATION_CONTRACT_V5",
        "final_config_id": "modelcfg_b8151deebdc7",
        "final_calibration": "none",
        "feature_names": ["earnings_yield", "current_pe"],
        "model_seed": 20260930,
        "training_data_fingerprint": "a" * 64,
        "producing_commit": "b" * 40,
    }
    first = wp8.compute_freeze_id(**args)
    second = wp8.compute_freeze_id(**args)
    assert first == second
    assert first.startswith("freeze_")
    assert len(first) == len("freeze_") + 12

    changed = dict(args)
    changed["final_feature_names"] = ["earnings_yield"]
    with pytest.raises(TypeError):
        wp8.compute_freeze_id(**changed)


def test_freeze_record_is_immutable_and_verify_reuse(wp8, tmp_path):
    record = {
        "schema_version": wp8.WP8_SCHEMA_VERSION,
        "freeze_id": "freeze_aaaaaaaaaaaa",
        "holdout_usage": "none",
    }
    first = wp8.write_freeze_record(record, root=tmp_path)
    assert first["status"] == "written"
    assert first["path"] == str(tmp_path / "provenance" / "wp8" / "final_candidate_freeze.json")

    second = wp8.write_freeze_record(dict(record), root=tmp_path)
    assert second["status"] == "verify_and_reuse"
    assert second["sha256"] == first["sha256"]

    divergent = dict(record)
    divergent["holdout_usage"] = "accessed"
    with pytest.raises(ImmutabilityError):
        wp8.write_freeze_record(divergent, root=tmp_path)


def test_pickle_artifact_is_immutable(wp8, tmp_path):
    obj = {"kind": "estimator", "features": ["earnings_yield"]}
    first = wp8._write_pickle_artifact(tmp_path, "final_model.pkl", obj)
    second = wp8._write_pickle_artifact(tmp_path, "final_model.pkl", dict(obj))
    assert first["sha256"] == second["sha256"]

    with pytest.raises(wp8.Wp8FreezeError, match="already exists with different bytes"):
        wp8._write_pickle_artifact(tmp_path, "final_model.pkl", {"kind": "other"})
