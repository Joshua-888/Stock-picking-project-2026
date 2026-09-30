"""WP6 provenance tests: immutable ids, no overwrite, holdout independence."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.ids import experiment_id, is_valid_id
from src.research.immutability import ImmutabilityError
from src.research.modeling import models as model_registry
from src.research.modeling.contract import DEFAULT_CONFIG, contract_payload
from src.research.modeling.io import modeling_payload, write_experiment


def _binding(git_commit="deadbeef"):
    return modeling_payload("dataset_35a278e17c13", "target_set_d2bb16610bce",
                            "feature_set_4f7b43726310", "holdout_7ce54e933e16",
                            git_commit, contract_payload(DEFAULT_CONFIG),
                            model_registry.registry_payload(),
                            {"model_seed": DEFAULT_CONFIG.model_seed})


def test_experiment_id_is_deterministic_and_valid():
    first = experiment_id(_binding())
    second = experiment_id(_binding())
    assert first == second
    assert is_valid_id(first, "experiment")


def test_different_git_commit_changes_id():
    assert experiment_id(_binding("a")) != experiment_id(_binding("b"))


def test_write_experiment_is_immutable(tmp_path):
    binding = _binding()
    result = write_experiment(tmp_path, binding, {"summary.json": {"x": 1}},
                              {"binding.json": binding})
    assert is_valid_id(result["experiment_id"], "experiment")
    # identical content verifies and reuses
    again = write_experiment(tmp_path, binding, {"summary.json": {"x": 1}},
                             {"binding.json": binding})
    assert again["experiment_id"] == result["experiment_id"]
    # different content for the same path is refused
    with pytest.raises(ImmutabilityError):
        write_experiment(tmp_path, binding, {"summary.json": {"x": 2}},
                         {"binding.json": binding})


def test_ledger_jsonl_is_write_once(tmp_path):
    binding = _binding()
    first = write_experiment(tmp_path, binding, {}, {}, ledger_records=[{"a": 1}])
    second = write_experiment(tmp_path, binding, {}, {}, ledger_records=[{"a": 1}])
    assert first["experiment_id"] == second["experiment_id"]
    with pytest.raises(Exception):
        write_experiment(tmp_path, binding, {}, {}, ledger_records=[{"a": 2}])


def test_contract_excludes_never_revive():
    payload = contract_payload(DEFAULT_CONFIG)
    assert set(payload["excluded_features"]) == {"abnormal_volume", "beta", "x12_appreciation_proxy"}
    assert not (set(payload["feature_universe"]) & set(payload["excluded_features"]))
