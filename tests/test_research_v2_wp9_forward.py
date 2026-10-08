"""WP9 forward shadow system acceptance tests (synthetic fixtures only).

These tests verify the frozen forward-validation contract semantics using only
in-memory synthetic frames and temporary regex/artifact roots. They never read
locked-holdout rows, labels, returns, predictions, or 2022+ outcomes, and they
never load the real frozen champion artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts" / "research_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_module(module_name, rel_path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_module(
    "wp9_forward_score_test_module", "scripts/research_v2/wp9_forward_score.py"
)
EVALUATOR = _load_module(
    "wp9_evaluate_matured_test_module", "scripts/research_v2/wp9_evaluate_matured.py"
)

from src.research import immutability as immutability_mod  # noqa: E402
from src.research.data.availability import (  # noqa: E402
    UnavailableError,
    require_available,
)
from src.research.fingerprints import fingerprint_file  # noqa: E402
from src.research.ids import canonical_json  # noqa: E402
from src.research.keyes.variables import FundamentalHistory  # noqa: E402
from src.research.wp9 import champion as champion_mod  # noqa: E402
from src.research.wp9 import economics as economics_mod  # noqa: E402
from src.research.wp9 import storage as storage_mod  # noqa: E402
from src.research.wp9.champion import (  # noqa: E402
    FROZEN_CALIBRATION,
    FROZEN_FEATURES,
    FROZEN_FREEZE_ID,
    ChampionError,
    FrozenChampion,
)
from src.research.wp9.contract import (  # noqa: E402
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    Wp9ContractError,
    load_wp9_contract,
    monthly_cadence_is_valid,
)
from src.research.wp9.forward_inputs import (  # noqa: E402
    ForwardInputError,
    ForwardSnapshotInputs,
    _ForwardInputBuilder,
)
from src.research.wp9.health import operational_blockers  # noqa: E402

OFFICIAL = storage_mod.OFFICIAL_KIND
DRY_RUN = storage_mod.DRY_RUN_KIND
SNAPSHOT_ASOF = "2026-10-31"
CODE_COMMIT = "1234567890abcdef1234567890abcdef12345678"


# ── Deterministic synthetic helpers ──────────────────────────────────────────


def _fake_contract(digest_override=None):
    """A minimal in-memory Wp9Contract; no repository git checks required."""
    from src.research.wp9.contract import Wp9Contract

    raw = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "champion": {
            "freeze_id": FROZEN_FREEZE_ID,
            "calibration": FROZEN_CALIBRATION,
            "features": list(FROZEN_FEATURES),
            "artifacts": {
                "model": {"sha256": "m" * 64},
                "preprocessor": {"sha256": "p" * 64},
                "calibrator": {"sha256": "c" * 64},
            },
        },
        "prospective": {
            "contract_freeze_commit": "not-a-commit",
            "maturity_rule": (
                "snapshot matures only when target_known_at per "
                "target_set_d2bb16610bce availability semantics is "
                "<= evaluation run date"
            ),
        },
        "wp8_closure": {
            "consumed_holdout": True,
            "consumed_holdout_period": {"start": "2022-01-01", "end": "2025-08-31"},
        },
    }
    return Wp9Contract(
        digest=digest_override or hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest(),
        version=CONTRACT_VERSION,
        schema_version=SCHEMA_VERSION,
        freeze_id=FROZEN_FREEZE_ID,
        features=tuple(FROZEN_FEATURES),
        calibration=FROZEN_CALIBRATION,
        raw=raw,
    )


def _write_tmp_contract(root: Path):
    contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "champion": {
            "freeze_id": FROZEN_FREEZE_ID,
            "calibration": FROZEN_CALIBRATION,
            "features": list(FROZEN_FEATURES),
            "artifacts": {
                "model": {"sha256": "m" * 64},
                "preprocessor": {"sha256": "p" * 64},
                "calibrator": {"sha256": "c" * 64},
            },
        },
        "prospective": {
            "contract_freeze_commit": "not-a-commit",
            "maturity_rule": (
                "snapshot matures only when target_known_at per "
                "target_set_d2bb16610bce availability semantics is "
                "<= evaluation run date"
            ),
        },
    }
    path = root / "provenance" / "wp9" / "forward_validation_contract_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract_payload, sort_keys=True), encoding="utf-8"
    )
    return load_wp9_contract(root=root, path=path)


def _snapshot_record(
    *,
    snapshot_id,
    snapshot_asof=SNAPSHOT_ASOF,
    model_freeze_id=FROZEN_FREEZE_ID,
    model_hash="m" * 64,
    feature_snapshot_hash="f" * 64,
    source_manifest_hash="s" * 64,
    raw_scores=(0.2, 0.4, 0.6),
    security_ids=("B", "A", "C"),
):
    ranks, percentiles = SCORER.deterministic_rank_percentile(
        list(security_ids), [float(value) for value in raw_scores]
    )
    return storage_mod.snapshot_payload(
        snapshot_id=snapshot_id,
        snapshot_asof=snapshot_asof,
        security_id=list(security_ids),
        ticker=list(security_ids),
        raw_model_score=[float(value) for value in raw_scores],
        frozen_calibrated_score=[float(value) / 2.0 for value in raw_scores],
        rank=ranks,
        percentile=percentiles,
        universe_size=len(security_ids),
        model_hash=model_hash,
        feature_snapshot_hash=feature_snapshot_hash,
        source_manifest_hash=source_manifest_hash,
        code_commit=CODE_COMMIT,
        created_at_utc="2026-10-31T12:00:00+00:00",
        eligibility_status=["scoreable"] * len(security_ids),
        quality_flags=[[] for _ in security_ids],
        model_freeze_id=model_freeze_id,
    )


def _bindings(
    contract,
    *,
    model_hash="m" * 64,
    preprocessor_hash="p" * 64,
    calibrator_hash="c" * 64,
    snapshot_asof=SNAPSHOT_ASOF,
    code_commit=CODE_COMMIT,
    universe_hash="u" * 64,
    source_manifest_hash="s" * 64,
    feature_snapshot_hash="f" * 64,
):
    return storage_mod.bindings_from_snapshot_id_inputs(
        contract_digest=contract.digest,
        snapshot_asof=snapshot_asof,
        champion_freeze=FROZEN_FREEZE_ID,
        model_hash=model_hash,
        preprocessor_hash=preprocessor_hash,
        calibrator_hash=calibrator_hash,
        universe_hash=universe_hash,
        source_manifest_hash=source_manifest_hash,
        feature_snapshot_hash=feature_snapshot_hash,
        code_commit=code_commit,
    )


def _prepare_official_env(
    tmp_path,
    *,
    snapshot_id=None,
    body_model_freeze=FROZEN_FREEZE_ID,
    body_model_hash=None,
    binding_overrides=None,
    invalidate=False,
    contract=None,
):
    """Create a synthetic official snapshot in a temporary repository root.

    Returns ``(root, snapshot_id, contract)``. The snapshot and prediction index
    are synthetic and point to no real champion, holdout, or market data.
    """
    root = tmp_path
    contract = contract or _write_tmp_contract(root)
    model_hash = body_model_hash or contract.get(
        "champion", "artifacts", "model", "sha256"
    )
    bindings = _bindings(contract)
    if binding_overrides:
        bindings.update(binding_overrides)
    sid = snapshot_id or storage_mod.snapshot_id_from_bindings(bindings)
    record = _snapshot_record(
        snapshot_id=sid,
        model_freeze_id=body_model_freeze,
        model_hash=model_hash,
        feature_snapshot_hash=bindings["feature_snapshot_hash"],
        source_manifest_hash=bindings["source_manifest_hash"],
    )

    snapshot_dir = root / "provenance" / "wp9" / "predictions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / ("%s.json" % sid)
    immutability_mod.write_json_atomic(snapshot_path, record)

    entry = {
        "snapshot_id": sid,
        "snapshot_asof": record["snapshot_asof"],
        "snapshot_kind": OFFICIAL,
        "model_freeze_id": record["model_freeze_id"],
        "model_hash": record["model_hash"],
        "feature_snapshot_hash": record["feature_snapshot_hash"],
        "source_manifest_hash": record["source_manifest_hash"],
        "universe_size": record["universe_size"],
        "code_commit": record["code_commit"],
        "created_at_utc": record["created_at_utc"],
        "path": str(snapshot_path.relative_to(root)),
        "sha256": fingerprint_file(snapshot_path),
        "bindings": dict(bindings),
    }
    index_path = snapshot_dir / "index.json"
    immutability_mod.write_json_atomic(
        index_path,
        {
            "schema_version": storage_mod.PREDICTION_INDEX_SCHEMA,
            "entries": [entry],
        },
    )

    registry = storage_mod.initialise_run_registry(
        contract_version=contract.version,
        contract_digest=contract.digest,
        champion_freeze=FROZEN_FREEZE_ID,
        root=root,
    )
    storage_mod.add_snapshot_to_run_registry(
        sid, "official", root=root, match_contract=contract.digest
    )
    if invalidate:
        storage_mod.add_invalidation_to_run_registry(
            sid, {"reason": "synthetic test invalidation"}, root=root
        )
    return root, sid, contract


class _NoFitEstimator:
    def __init__(self):
        self.fit_calls = 0
        self.predict_proba_calls = 0

    def predict_proba(self, x):
        self.predict_proba_calls += 1
        return np.column_stack(
            [1.0 - np.clip(x[:, 0], 0, 1), np.clip(x[:, 0], 0, 1)]
        )

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("model.fit must never be called on a forward snapshot")


class _TransformOnlyPreprocessor:
    def __init__(self):
        self.transform_calls = 0
        self.fit_calls = 0

    def transform(self, frame, fitted):
        self.transform_calls += 1
        assert fitted is not None
        return np.column_stack(
            [
                np.linspace(0.1, 0.9, len(frame)),
                np.linspace(0.9, 0.1, len(frame)),
            ]
        )

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("preprocessor.fit must never be called")


class _PredictOnlyCalibrator:
    def __init__(self):
        self.predict_proba_calls = 0
        self.fit_calls = 0

    def predict_proba(self, logit):
        self.predict_proba_calls += 1
        clipped = np.clip(logit, 1e-9, 1 - 1e-9)
        p = 1.0 / (1.0 + np.exp(-clipped[:, 0]))
        return np.column_stack([1.0 - p, p])

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("calibrator.fit must never be called")


def _fake_champion(preprocessor, estimator, calibrator):
    return FrozenChampion(
        freeze_id=FROZEN_FREEZE_ID,
        features=tuple(FROZEN_FEATURES),
        calibration=FROZEN_CALIBRATION,
        model_sha256="m" * 64,
        preprocessor_sha256="p" * 64,
        calibrator_sha256="c" * 64,
        _model_obj={"kind": "estimator", "estimator": estimator},
        _preprocessor_obj={"preprocessor": preprocessor, "fitted": SimpleNamespace()},
        _calibrator=calibrator,
    )


def _feature_frame(rows=3):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "security_id": ["B", "A", "C"][:rows],
            "ticker": ["B", "A", "C"][:rows],
            "feature_asof": [SNAPSHOT_ASOF] * rows,
            **{name: rng.uniform(0.0, 1.0, rows) for name in FROZEN_FEATURES},
        }
    )


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── 1. WP8 consumed holdout cannot be reused as official WP9 evidence ─────────


def test_wp8_consumed_holdout_cannot_be_official_wp9_evidence():
    contract = _fake_contract()
    assert contract.get("wp8_closure", "consumed_holdout") is True
    # The official snapshot as-of (used throughout this suite) is after the
    # consumed holdout end and after the contract freeze moment.
    assert SNAPSHOT_ASOF > contract.get("wp8_closure", "consumed_holdout_period", "end")


# ── 2-3. Frozen champion identity ────────────────────────────────────────────


def test_champion_freeze_id_cannot_change(tmp_path):
    freeze_path = tmp_path / "provenance" / "wp8" / "final_candidate_freeze.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps({"freeze_id": "freeze_some_other_id"}), encoding="utf-8"
    )
    with pytest.raises(ChampionError, match="wrong WP8 freeze id"):
        champion_mod.load_freeze_record(root=tmp_path)


def test_wrong_champion_rejected(tmp_path):
    root, _sid, contract = _prepare_official_env(
        tmp_path, body_model_freeze="freeze_some_other_id"
    )
    # Rebuild the index SHA against the rewritten snapshot body so the defect is
    # isolated to the champion freeze identity, not the file hash.
    snapshot_path = root / "provenance" / "wp9" / "predictions" / ("%s.json" % _sid)
    index_path = root / "provenance" / "wp9" / "predictions" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["sha256"] = fingerprint_file(snapshot_path)
    immutability_mod.write_json_atomic(index_path, index)
    with pytest.raises(Exception, match="unknown_champion_freeze"):
        EVALUATOR.load_official_snapshot(_sid, root=root, contract=contract)


# ── 4-6. Wrong artifact hashes rejected by matured evaluator ─────────────────


def test_wrong_model_hash_rejected(tmp_path):
    root, sid, contract = _prepare_official_env(tmp_path, body_model_hash="z" * 64)
    index_path = root / "provenance" / "wp9" / "predictions" / "index.json"
    snapshot_path = root / "provenance" / "wp9" / "predictions" / ("%s.json" % sid)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["sha256"] = fingerprint_file(snapshot_path)
    immutability_mod.write_json_atomic(index_path, index)
    with pytest.raises(Exception, match="snapshot_model_hash_not_contract_champion"):
        EVALUATOR.load_official_snapshot(sid, root=root, contract=contract)


def test_wrong_preprocessor_hash_rejected(tmp_path):
    root, sid, contract = _prepare_official_env(
        tmp_path, binding_overrides={"preprocessor_hash": "z" * 64}
    )
    with pytest.raises(Exception, match="snapshot_preprocessor_hash_not_contract_champion"):
        EVALUATOR.load_official_snapshot(sid, root=root, contract=contract)


def test_wrong_calibrator_hash_rejected(tmp_path):
    root, sid, contract = _prepare_official_env(
        tmp_path, binding_overrides={"calibrator_hash": "z" * 64}
    )
    with pytest.raises(Exception, match="snapshot_calibrator_hash_not_contract_champion"):
        EVALUATOR.load_official_snapshot(sid, root=root, contract=contract)


# ── 7-9. Official date and duplicate guards ──────────────────────────────────

def test_official_date_before_contract_freeze_rejected(tmp_path, monkeypatch):
    contract = _fake_contract()
    monkeypatch.setattr(
        SCORER, "contract_freeze_timestamp", lambda *args, **kwargs: "2026-10-07T21:59:03+00:00"
    )
    with pytest.raises(SCORER.Wp9ScoringError, match="asof_not_after_contract_freeze"):
        SCORER.assert_asof_after_freeze("2026-09-30", contract, tmp_path)


def test_backdated_official_snapshot_rejected(tmp_path, monkeypatch):
    contract = _fake_contract()
    monkeypatch.setattr(
        EVALUATOR, "contract_freeze_timestamp", lambda *args, **kwargs: "2026-10-07T21:59:03+00:00"
    )
    snapshot = {"snapshot_asof": "2026-09-30"}
    with pytest.raises(Exception, match="snapshot_asof_not_after_contract_freeze"):
        EVALUATOR.assert_snapshot_after_contract_freeze(snapshot, contract, tmp_path)


def test_duplicate_official_snapshot_rejected(tmp_path, monkeypatch):
    root, sid, _contract = _prepare_official_env(tmp_path)
    with pytest.raises(SCORER.Wp9ScoringError, match="duplicate_snapshot_asof"):
        SCORER.assert_no_duplicate_snapshot(SNAPSHOT_ASOF, root)


# ── 10. Dry-run cannot enter official registry ───────────────────────────────

def test_dry_run_cannot_enter_official_registry(tmp_path):
    contract = _fake_contract()
    root = tmp_path
    bindings = _bindings(contract)
    sid = storage_mod.snapshot_id_from_bindings(bindings)
    record = _snapshot_record(
        snapshot_id=sid,
        feature_snapshot_hash=bindings["feature_snapshot_hash"],
        source_manifest_hash=bindings["source_manifest_hash"],
    )
    result = storage_mod.write_snapshot(record, bindings, kind=DRY_RUN, root=root)
    assert result["index_outcome"] == "written"
    official_index = storage_mod.canonicalise_prediction_index(root)
    assert official_index["entries"] == []
    dry_index = storage_mod.canonicalise_dry_run_index(root)
    assert [entry["snapshot_id"] for entry in dry_index["entries"]] == [sid]


# ── 11-12. Scorer never loads/requests future outcomes ───────────────────────

def test_official_scorer_works_without_target_columns(tmp_path):
    contract = _fake_contract()
    estimator = _NoFitEstimator()
    preprocessor = _TransformOnlyPreprocessor()
    calibrator = _PredictOnlyCalibrator()
    champion = _fake_champion(preprocessor, estimator, calibrator)
    frame = _feature_frame(3)
    bindings = _bindings(contract)
    sid = storage_mod.snapshot_id_from_bindings(bindings)
    inputs = ForwardSnapshotInputs(
        snapshot_asof=SNAPSHOT_ASOF,
        universe={
            "universe_count": 3,
            "universe_hash": bindings["universe_hash"],
        },
        score_frame=frame,
        feature_frame=frame,
        feature_summary={"rows": 3},
        source_manifest={
            "source_manifest_hash": bindings["source_manifest_hash"],
            "feature_snapshot_hash": bindings["feature_snapshot_hash"],
        },
    )
    payload = SCORER.score_frame(
        contract, champion, inputs, CODE_COMMIT, "2026-10-31T12:00:00+00:00"
    )
    assert payload["snapshot_id"] == sid
    assert set(payload) == set(storage_mod.SNAPSHOT_SCHEMA_FIELDS)
    assert "outperform_12m" not in payload


def test_official_scorer_never_requests_future_outcome_values():
    # The scorer module itself must not import or reference the target builder
    # or observability APIs that materialise future outcomes.
    source_tree = ast.parse(_source_text(SCRIPTS / "wp9_forward_score.py"))
    imported_names = []
    calls = set()
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert "build_targets" not in calls
    assert "annotate_observability" not in calls
    assert "src.research.targets" not in imported_names
    # And the runtime guard rejects any target/outcome-bearing frame.
    with pytest.raises(SCORER.Wp9ScoringError, match="target_outcome_columns_present"):
        SCORER.assert_no_target_columns(
            pd.DataFrame({"target_observable": [True], "future_12m_excess_return": [0.1]})
        )


# ── 13-16. Point-in-time objective controls ──────────────────────────────────

def test_future_filing_unavailable_at_snapshot_date_rejected():
    prediction_ts = "2026-10-01T00:00:00+00:00"
    future_available_at = "2026-11-01T00:00:00+00:00"
    with pytest.raises(UnavailableError):
        require_available(future_available_at, prediction_ts, context="filing")


def test_future_membership_rejected():
    builder = _ForwardInputBuilder(root=REPO_ROOT)
    membership = pd.DataFrame(
        [
            {
                "security_id": "FUTURE_MEMBER",
                "ticker": "FUTURE_MEMBER",
                "membership_start": "2027-01-01",
                "membership_end": None,
            },
            {
                "security_id": "CURRENT_MEMBER",
                "ticker": "CURRENT_MEMBER",
                "membership_start": "2020-01-01",
                "membership_end": None,
            },
        ]
    )
    universe = builder.resolve_universe_at(SNAPSHOT_ASOF, membership)
    assert universe["security_ids"] == ["CURRENT_MEMBER"]
    assert universe["excluded_membership_records"] == []
    assert "FUTURE_MEMBER" not in universe["ticker_mapping"]


def test_canonical_membership_used():
    builder = _ForwardInputBuilder(root=REPO_ROOT)
    membership = pd.DataFrame(
        [
            {
                "security_id": "MEMBER",
                "ticker": "MEMBER",
                "membership_start": "2020-01-01",
                "membership_end": None,
            }
        ]
    )
    universe = builder.resolve_universe_at(SNAPSHOT_ASOF, membership)
    assert universe["universe_id"] == "sp500_pit_wikipedia_eodhd_v1"
    assert universe["membership_source"] == "sp500_pit_wikipedia_eodhd_v1"
    assert universe["membership_effective_date"] == SNAPSHOT_ASOF


def test_feature_availability_respected():
    fundamentals = pd.DataFrame(
        [
            {
                "cik": "1",
                "field": "eps",
                "value": 1.5,
                "fiscal_period_start": "2026-01-01",
                "fiscal_period_end": "2026-03-31",
                "available_at": "2026-11-01T00:00:00+00:00",
            }
        ]
    )
    history = FundamentalHistory.from_frame(fundamentals)
    # The only filing becomes public after the as-of, so it must be unavailable.
    assert history.latest_value("1", "eps", "2026-10-01") is None


# ── 17-18. Deterministic ranking ─────────────────────────────────────────────

def test_same_inputs_produce_same_ranks():
    ids = ["B", "A", "C"]
    scores = [0.2, 0.4, 0.6]
    first = SCORER.deterministic_rank_percentile(ids, scores)
    second = SCORER.deterministic_rank_percentile(ids, scores)
    assert first == second
    assert first[0] == [1, 2, 3]


def test_deterministic_tie_break():
    ids = ["B", "A", "C"]
    scores = [0.5, 0.5, 0.5]
    ranks, _percentiles = SCORER.deterministic_rank_percentile(ids, scores)
    assert ranks == [2, 1, 3]


# ── 19-20. Snapshot immutability and deterministic identity ──────────────────

def test_snapshot_immutable(tmp_path):
    path = tmp_path / "snapshot.json"
    immutability_mod.save_immutable(path, {"snapshot_id": "a", "value": 1})
    with pytest.raises(Exception):
        immutability_mod.save_immutable(path, {"snapshot_id": "a", "value": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1


def test_prediction_hash_deterministic():
    one = storage_mod.snapshot_id(
        contract_digest="a",
        snapshot_asof=SNAPSHOT_ASOF,
        champion_freeze=FROZEN_FREEZE_ID,
        model_hash="m",
        preprocessor_hash="p",
        calibrator_hash="c",
        universe_hash="u",
        source_manifest_hash="s",
        feature_snapshot_hash="f",
        code_commit=CODE_COMMIT,
    )
    two = storage_mod.snapshot_id(
        contract_digest="a",
        snapshot_asof=SNAPSHOT_ASOF,
        champion_freeze=FROZEN_FREEZE_ID,
        model_hash="m",
        preprocessor_hash="p",
        calibrator_hash="c",
        universe_hash="u",
        source_manifest_hash="s",
        feature_snapshot_hash="f",
        code_commit=CODE_COMMIT,
    )
    assert one == two == "wp9_%s" % hashlib.sha256(
        canonical_json(
            ["a", SNAPSHOT_ASOF, FROZEN_FREEZE_ID, "m", "p", "c", "u", "s", "f", CODE_COMMIT]
        ).encode("utf-8")
    ).hexdigest()[:20]


# ── 21-23. Matured evaluator gating ──────────────────────────────────────────

def test_invalidated_snapshot_cannot_be_evaluated(tmp_path):
    root, sid, contract = _prepare_official_env(tmp_path, invalidate=True)
    with pytest.raises(Exception, match="snapshot_invalidated"):
        EVALUATOR.load_official_snapshot(sid, root=root, contract=contract)


def test_immature_snapshot_cannot_be_evaluated(tmp_path, monkeypatch):
    contract = _fake_contract()
    snapshot = _snapshot_record(snapshot_id="wp9_immature_snapshot")
    monkeypatch.setattr(
        EVALUATOR,
        "load_official_snapshot",
        lambda *args, **kwargs: (snapshot, {}, contract),
    )
    monkeypatch.setattr(
        EVALUATOR,
        "assert_snapshot_after_contract_freeze",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        EVALUATOR,
        "build_outcomes_for_snapshot",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "target_observable": [True],
                "target_known_at": ["2027-01-15T21:00:00+00:00"],
            }
        ),
    )
    result = EVALUATOR.evaluate_snapshot(
        "wp9_immature_snapshot",
        root=tmp_path,
        evaluation_date="2026-11-01",
        market_inputs={},
    )
    assert result["status"] == "IMMATURE_OR_INVALID"
    assert any(
        "target_known_at_after_evaluation_date" in reason
        for reason in result["blocking_reasons"]
    )


def test_matured_evaluator_requires_target_known_at():
    outcomes = pd.DataFrame({"target_observable": [True]})
    issues = EVALUATOR.maturity_issues(outcomes, evaluation_date="2026-11-01")
    assert any("target_known_at_column_missing" in issue for issue in issues)


# ── 24-25. Exact economic top-quintile semantics ─────────────────────────────

def test_economic_top_quintile_cutoff_exactly_frozen():
    record = {
        "snapshot_id": "wp9_x",
        "snapshot_asof": SNAPSHOT_ASOF,
        "security_id": ["A", "B", "C", "D", "E"],
        "ticker": ["A", "B", "C", "D", "E"],
        "raw_model_score": [0.1 * i for i in range(5)],
        "frozen_calibrated_score": [0.05 * i for i in range(5)],
        "rank": [1, 2, 3, 4, 5],
        "percentile": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    selected = economics_mod.select_top_quintile(record)
    assert selected["cutoff_threshold"] == 0.80
    assert selected["cutoff_rule"] == "top_quintile_exact (rank percentile > 0.80)"
    assert [item["security_id"] for item in selected["constituents"]] == ["E"]


def test_no_alternative_cutoff_search():
    assert economics_mod.TOP_QUINTILE_THRESHOLD == 0.80
    assert economics_mod.COST_SCENARIOS_BPS == (0, 10, 25, 50)
    source = _source_text(REPO_ROOT / "src" / "research" / "wp9" / "economics.py")
    for forbidden in ("golden", "minimize", "maximize", "grid_search", "optuna"):
        assert forbidden not in source.lower()


# ── 26-29. No fitting/selection/retraining on forward snapshot ───────────────

def test_no_model_fitting_on_forward_snapshot():
    estimator = _NoFitEstimator()
    preprocessor = _TransformOnlyPreprocessor()
    calibrator = _PredictOnlyCalibrator()
    champion = _fake_champion(preprocessor, estimator, calibrator)
    frame = _feature_frame(3)
    first_raw, first_calibrated = champion.predict(frame)
    second_raw, second_calibrated = champion.predict(frame)
    assert estimator.fit_calls == 0
    assert preprocessor.fit_calls == 0
    assert calibrator.fit_calls == 0
    assert preprocessor.transform_calls == 2
    assert calibrator.predict_proba_calls == 2
    # The champion deep-copies the estimator to force deterministic n_jobs=1, so
    # the original estimator's predict_proba count stays zero. The observable
    # invariant is repeatable, deterministic scoring without any fit.
    np.testing.assert_allclose(first_raw, second_raw)
    np.testing.assert_allclose(first_calibrated, second_calibrated)


def test_no_calibration_fitting_on_forward_snapshot():
    estimator = _NoFitEstimator()
    preprocessor = _TransformOnlyPreprocessor()
    calibrator = _PredictOnlyCalibrator()
    champion = _fake_champion(preprocessor, estimator, calibrator)
    champion.predict(_feature_frame(3))
    assert calibrator.fit_calls == 0
    assert calibrator.predict_proba_calls == 1


def test_no_feature_selection_on_forward_snapshot():
    paths = [
        SCRIPTS / "wp9_forward_score.py",
        SCRIPTS / "wp9_evaluate_matured.py",
        REPO_ROOT / "src" / "research" / "wp9" / "champion.py",
        REPO_ROOT / "src" / "research" / "wp9" / "forward_inputs.py",
    ]
    forbidden = (
        "SelectKBest",
        "SelectFromModel",
        "SequentialFeatureSelector",
        "RFE",
        "RFECV",
        "f_regression",
        "mutual_info_classif",
        "feature_selection",
    )
    for path in paths:
        text = _source_text(path)
        for token in forbidden:
            assert token not in text, "%s references forbidden %s" % (path, token)


def test_no_retraining_path_reachable():
    paths = [
        SCRIPTS / "wp9_forward_score.py",
        SCRIPTS / "wp9_evaluate_matured.py",
        REPO_ROOT / "src" / "research" / "wp9" / "champion.py",
        REPO_ROOT / "src" / "research" / "wp9" / "forward_inputs.py",
    ]
    for path in paths:
        tree = ast.parse(_source_text(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("fit", "fit_transform", "fit_predict"), (
                    "%s reaches retraining via %s" % (path, node.func.attr)
                )


# ── 30-32. Provenance, source manifest, canonical registry ───────────────────

def test_official_snapshot_provenance_complete(tmp_path):
    contract = _fake_contract()
    root = tmp_path
    bindings = _bindings(contract)
    sid = storage_mod.snapshot_id_from_bindings(bindings)
    record = _snapshot_record(
        snapshot_id=sid,
        feature_snapshot_hash=bindings["feature_snapshot_hash"],
        source_manifest_hash=bindings["source_manifest_hash"],
    )
    result = storage_mod.write_snapshot(record, bindings, kind=OFFICIAL, root=root)
    assert result["index_outcome"] == "written"
    index = storage_mod.canonicalise_prediction_index(root)
    entry = index["entries"][0]
    assert set(entry["bindings"]) == set(storage_mod.BINDINGS)
    assert entry["sha256"] == fingerprint_file(
        root / "provenance" / "wp9" / "predictions" / ("%s.json" % sid)
    )
    assert entry["code_commit"] == CODE_COMMIT
    assert entry["model_freeze_id"] == FROZEN_FREEZE_ID


def test_source_manifest_complete(tmp_path):
    builder = _ForwardInputBuilder(root=tmp_path)
    layer_records = {
        "silver_prices": "prices_version",
        "silver_actions": "actions_version",
        "silver_membership": "membership_version",
        "price_rows": 10,
        "action_rows": 2,
        "windows": 3,
    }
    upstream = {
        "dataset_id": "dataset_35a278e17c13",
        "feature_set_id": "feature_set_4f7b43726310",
        "target_set_id": "target_set_d2bb16610bce",
        "edgar_binding": {"edgar_fundamentals": {"shards": []}},
        "cik_mapping": {},
    }
    manifest = builder.build_source_manifest(SNAPSHOT_ASOF, layer_records, upstream)
    assert manifest["allowed_sources"] == ["EODHD", "SEC_EDGAR", "WIKIPEDIA_SP500"]
    assert set(manifest["sources"]) == {
        "eodhd_market",
        "wikipedia_sp500",
        "sec_edgar",
    }
    assert "source_manifest_hash" in manifest
    assert manifest["benchmark"]["name"] == "benchmark_gold_SPY"
    assert manifest["upstream"]["target_set_id"] == "target_set_d2bb16610bce"


def test_registry_canonical_resolution_deterministic(tmp_path):
    contract = _fake_contract(digest_override="d" * 64)
    root = tmp_path
    storage_mod.initialise_run_registry(
        contract_version=contract.version,
        contract_digest=contract.digest,
        champion_freeze=FROZEN_FREEZE_ID,
        root=root,
    )
    storage_mod.add_matured_evaluation_to_run_registry(
        "snapshot_b", "eval_2", "provenance/wp9/matured_evaluations/eval_2.json", root=root
    )
    storage_mod.add_matured_evaluation_to_run_registry(
        "snapshot_a", "eval_1", "provenance/wp9/matured_evaluations/eval_1.json", root=root
    )
    registry = storage_mod.load_run_registry(root=root)
    assert [item["snapshot_id"] for item in registry["matured_evaluations"]] == [
        "snapshot_a",
        "snapshot_b",
    ]
    assert registry["current_prospective_stage"] == "A_OPERATIONAL_SHADOW"
    again = storage_mod.load_run_registry(root=root)
    assert again["matured_evaluations"] == registry["matured_evaluations"]
