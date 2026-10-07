"""WP8 Phase C locked-holdout evaluator tests (synthetic fixtures only).

These tests verify the frozen ``HOLDOUT_EVALUATION_CONTRACT_V1`` contract, the
canonical WP7/WP6/holdout/freeze resolver guards, stale-candidate rejection,
missing-authorization/ledger fail-closed behavior, one-shot ledger transitions,
censoring rules, the no-fit holdout scoring path, deterministic equal-month
HAC inference, verdict mapping, immutable result binding, and ledger-result
binding. They never construct or read real holdout rows, labels, predictions or
2022+ outcomes, and they never invoke the real-data evaluation entry point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
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


def _load_wp8_holdout():
    module_name = "wp8_holdout_evaluation_test_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = SCRIPTS / "wp8_run_holdout_evaluation.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


from src.research import wp7_generation_corrections as wpc7  # noqa: E402
from src.research.discovery.panel import FeaturePanelError, build_feature_panel  # noqa: E402
from src.research.immutability import ImmutabilityError  # noqa: E402

ENGINE = _load_wp8_holdout()

CANONICAL_GENERATION = "generation_de0f9bbd0bec"
WITHDRAWN_GENERATION = "generation_04b8e2810b50"
CANONICAL_HOLDOUT_ID = "holdout_7ce54e933e16"
CANONICAL_FREEZE_ID = "freeze_e62eac30df40"
TARGET = "outperform_12m"


@pytest.fixture(scope="module")
def wp8():
    return _load_wp8_holdout()


@pytest.fixture()
def fake_holdout():
    return SimpleNamespace(
        holdout_id=CANONICAL_HOLDOUT_ID,
        holdout_start=pd.Timestamp("2022-01-01", tz="UTC"),
        holdout_end=pd.Timestamp("2025-08-31", tz="UTC"),
    )


def _canonical_freeze_payload(**overrides):
    payload = {
        "schema_version": "wp8_final_candidate_freeze_v1",
        "freeze_id": CANONICAL_FREEZE_ID,
        "wp7_generation_id": CANONICAL_GENERATION,
        "wp7_contract_version": "WP7_VALIDATION_CONTRACT_V5",
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
        "final_calibration": "none",
        "final_selected_features": ["x1"],
        "final_config_id": "modelcfg_158499379e80",
    }
    payload.update(overrides)
    return payload


def _write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _panel_with_holdout_rows():
    return pd.DataFrame({
        "feature_asof": [
            "2022-01-31", "2022-01-31", "2022-01-31",
            "2022-02-28", "2022-02-28", "2022-02-28",
            "2021-12-31", "2022-03-31", "2025-09-30",
        ],
        "target_observable": [
            True, True, True, True, True, True, True, True, True,
        ],
        TARGET: [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "security_id": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
    })


def _outcome_frame():
    return pd.DataFrame({
        "feature_asof": pd.to_datetime(
            ["2022-01-31"] * 2 + ["2022-02-28"] * 2, utc=True
        ),
        "prediction": [0.7, 0.4, 0.8, 0.3],
        "actual": [1.0, 0.0, 1.0, 0.0],
        "security_id": ["A", "B", "A", "B"],
    })


class _FakeCalibrator:
    def predict(self, p):
        return p

    def predict_proba(self, logit):
        return np.column_stack([np.zeros_like(logit[:, 0]), 1 / (1 + np.exp(-logit[:, 0]))])


class _FakePreprocessor:
    def __init__(self):
        self.calls = []

    def transform(self, frame, fitted):
        self.calls.append(("transform", fitted))
        return np.column_stack([np.linspace(0.0, 1.0, len(frame)), np.linspace(1.0, 0.0, len(frame))])


class _FakeEstimator:
    def __init__(self, probabilities=None):
        self.fit_called = False
        self.predict_calls = 0
        if probabilities is None:
            probabilities = np.column_stack([
                np.linspace(0.1, 0.4, 2),
                np.linspace(0.9, 0.6, 2),
            ])
        self.probabilities = probabilities

    def predict_proba(self, x):
        self.predict_calls += 1
        return np.column_stack([1 - np.clip(x[:, 0], 0, 1), np.clip(x[:, 0], 0, 1)])

    def fit(self, *args, **kwargs):  # pragma: no cover - must not be called
        self.fit_called = True
        return self


# ── Canonical resolver guards ────────────────────────────────────────────────
def test_canonical_generation_resolver_rejects_withdrawn_and_accepts_canonical(wp8):
    resolved = wpc7.assert_canonical(CANONICAL_GENERATION)
    assert resolved["canonical"] is True

    with pytest.raises(wpc7.Wp7GenerationCorrectionError):
        wpc7.assert_canonical(WITHDRAWN_GENERATION)

    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="withdrawn/non-canonical"):
        wp8.require_withdrawn_generation_rejected(WITHDRAWN_GENERATION)


def test_stale_model_generation_candidate_cannot_silently_select(wp8, monkeypatch, tmp_path):
    stale = {
        "generation_id": WITHDRAWN_GENERATION,
        "contract_version": "WP7_VALIDATION_CONTRACT_V5",
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
    }
    path = _write_json(tmp_path, "stale_candidate.json", stale)
    monkeypatch.setattr(wpc7, "canonical_candidate_path", lambda root=None: path)
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="resolver returned non-canonical"):
        wp8.canonical_wp7_candidate()


def test_exact_holdout_id_required(wp8, fake_holdout):
    assert wp8.require_canonical_holdout(fake_holdout) is True
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="wrong locked holdout id"):
        wp8.require_canonical_holdout(SimpleNamespace(holdout_id="holdout_wrong"))


def test_exact_freeze_id_required(wp8, tmp_path):
    wrong = _canonical_freeze_payload(freeze_id="freeze_wrong")
    path = _write_json(tmp_path, "freeze_wrong.json", wrong)
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="wrong final candidate freeze id"):
        wp8.load_final_candidate_freeze(path=path)


# ── Authorization/ledger fail-closed behavior ────────────────────────────────
def test_missing_authorization_refused(wp8):
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="invalid"):
        wp8.assert_authorization_ready({})


def test_authorization_must_be_exact(wp8):
    valid = {
        "authorization_type": "HOLDOUT_EVALUATION_AUTHORIZATION",
        "scope": "exactly one evaluation",
        "authorized_wp7_generation": CANONICAL_GENERATION,
        "authorized_final_candidate_freeze": CANONICAL_FREEZE_ID,
        "authorized_holdout": CANONICAL_HOLDOUT_ID,
        "holdout_evaluation_count": 1,
    }
    assert wp8.assert_authorization_ready(valid) is True
    for key, value in ("authorized_holdout", "holdout_wrong"), ("holdout_evaluation_count", 2):
        changed = dict(valid)
        changed[key] = value
        with pytest.raises(wp8.Wp8HoldoutEvaluationError):
            wp8.assert_authorization_ready(changed)


def _ready_ledger(**overrides):
    ledger = {
        "schema_version": ENGINE.LEDGER_SCHEMA_VERSION,
        "holdout_id": CANONICAL_HOLDOUT_ID,
        "state": "AUTHORIZED_NOT_ACCESSED",
        "access_count": 0,
        "evaluation_count": 0,
        "final_candidate_freeze_id": CANONICAL_FREEZE_ID,
        "wp7_generation_id": CANONICAL_GENERATION,
        "contract_digest": "b" * 64,
    }
    ledger.update(overrides)
    return ledger


def test_access_ledger_must_be_authorized_not_accessed(wp8):
    wp8.assert_ledger_ready(_ready_ledger(), contract_digest="b" * 64)
    for state in ("STARTED", "COMPLETED", "INVALIDATED"):
        with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="not AUTHORIZED_NOT_ACCESSED"):
            wp8.assert_ledger_ready(_ready_ledger(state=state), contract_digest="b" * 64)


def test_access_ledger_wrong_ids_and_counts_fail_closed(wp8):
    for ledger in (
        _ready_ledger(holdout_id="holdout_wrong"),
        _ready_ledger(access_count=1),
        _ready_ledger(evaluation_count=1),
        _ready_ledger(final_candidate_freeze_id="freeze_wrong"),
        _ready_ledger(wp7_generation_id=WITHDRAWN_GENERATION),
        _ready_ledger(contract_digest="c" * 64),
    ):
        with pytest.raises(wp8.Wp8HoldoutEvaluationError):
            wp8.assert_ledger_ready(ledger, contract_digest="b" * 64)


def test_started_cannot_start_again_and_second_execution_refused(wp8, tmp_path):
    path = tmp_path / "ledger.json"
    wp8.write_ledger_atomic(_ready_ledger(), path=path)
    started = wp8.transition_ledger_to_started(
        _ready_ledger(),
        contract_digest="b" * 64,
        freeze={"freeze_id": CANONICAL_FREEZE_ID},
        code_digest="c" * 64,
        producing_commit="d" * 40,
        path=path,
    )
    assert started["state"] == "STARTED"
    assert started["access_count"] == 1
    with pytest.raises(wp8.Wp8HoldoutEvaluationError):
        wp8.assert_ledger_ready(started, contract_digest="b" * 64)
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="not AUTHORIZED_NOT_ACCESSED"):
        wp8.transition_ledger_to_started(
            started, contract_digest="b" * 64, path=path
        )


def test_started_completed_invalidated_fail_closed_at_entry(wp8, tmp_path):
    path = tmp_path / "ledger.json"
    for state in ("STARTED", "COMPLETED", "INVALIDATED"):
        ledger = _ready_ledger(state=state)
        with pytest.raises(wp8.Wp8HoldoutEvaluationError):
            wp8.assert_ledger_ready(ledger, contract_digest="b" * 64)


def test_ledger_completed_transition_binds_evaluation_and_hash(wp8, tmp_path):
    path = tmp_path / "ledger.json"
    wp8.write_ledger_atomic(_ready_ledger(), path=path)
    started = wp8.transition_ledger_to_started(
        _ready_ledger(), contract_digest="b" * 64, path=path
    )
    completed = wp8.transition_ledger_to_completed(
        started, evaluation_id="evaluation_aaaaaaaaaaaa", result_hash="f" * 64, path=path
    )
    assert completed["state"] == "COMPLETED"
    assert completed["evaluation_count"] == 1
    assert completed["completed_evaluation_id"] == "evaluation_aaaaaaaaaaaa"
    assert completed["completed_result_sha256"] == "f" * 64


# ── Row selection / censoring / dates ───────────────────────────────────────
def test_dates_outside_holdout_bounds_refused(wp8, fake_holdout):
    frame = _panel_with_holdout_rows()
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="outside the locked holdout window"):
        wp8.assert_holdout_window_only(frame, fake_holdout)


def test_censored_missing_outcomes_excluded_not_labeled_zero(wp8, fake_holdout):
    panel = pd.DataFrame({
        "feature_asof": ["2022-01-31", "2022-01-31", "2022-01-31", "2022-02-28"],
        "target_observable": [True, True, False, True],
        TARGET: [1.0, np.nan, 0.0, 1.0],
        "x1": [0.1, 0.2, 0.3, 0.4],
        "security_id": ["A", "B", "C", "A"],
    })
    selected = wp8.select_holdout_rows(panel, _canonical_freeze_payload(), holdout=fake_holdout)
    assert list(selected["security_id"]) == ["A", "A"]
    assert selected[TARGET].tolist() == [1.0, 1.0]


def test_required_frozen_features_must_be_available(wp8, fake_holdout):
    panel = _panel_with_holdout_rows().drop(columns=["x1"])
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="panel lacks required frozen features"):
        wp8.select_holdout_rows(panel, _canonical_freeze_payload(), holdout=fake_holdout)


# ── Frozen no-fit scoring path ───────────────────────────────────────────────
def _fake_bundle():
    fit_slot = {"fitted": True}
    preprocessor = _FakePreprocessor()
    return {
        "model": {
            "kind": "estimator",
            "estimator": _FakeEstimator(),
            "preprocessor": preprocessor,
            "fitted": fit_slot,
        },
        "preprocessor": {"preprocessor": preprocessor, "fitted": fit_slot},
        "calibrator": _FakeCalibrator(),
    }


def test_holdout_path_uses_no_fitting(wp8, fake_holdout):
    bundle = _fake_bundle()
    estimator = bundle["model"]["estimator"]
    frame = pd.DataFrame({
        "feature_asof": ["2022-01-31", "2022-02-28"],
        "modeling_month": [1, 2],
        "x1": [0.1, 0.2],
        "target_observable": [True, True],
        TARGET: [1.0, 0.0],
        "security_id": ["A", "B"],
    })
    outcome = wp8.predict_frozen_holdout(
        frame,
        _canonical_freeze_payload(final_calibration="none"),
        bundle["model"],
        bundle["preprocessor"],
        None,
    )
    assert estimator.fit_called is False
    assert estimator.predict_calls == 1
    assert "prediction" in outcome.columns


def test_frozen_calibration_uses_existing_fitted_object(wp8):
    calibrator = _FakeCalibrator()
    called = wp8._apply_frozen_calibration(
        calibrator, np.array([0.6, 0.7], dtype="float64"), "platt_scaling"
    )
    assert np.isfinite(called).all()
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="unknown calibration method"):
        wp8._apply_frozen_calibration(calibrator, np.array([0.6]), "unknown")


# ── Primary metric/inference/verdict ────────────────────────────────────────
def test_primary_equal_month_weighting(wp8, monkeypatch):
    outcome = _outcome_frame()

    def fake_monthly_auc_series(frame, score, target, asof_col, min_observations):
        return pd.DataFrame({
            "month": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "paired_obs": [31, 40, 32, 41, 33, 42],
            "auc": [0.6, 0.7, 0.6, 0.7, 0.6, 0.7],
        }), {"months": 6, "min_observations": int(min_observations)}

    monkeypatch.setattr(wp8, "monthly_auc_series", fake_monthly_auc_series)
    primary = wp8.compute_primary_metrics(outcome)
    assert primary["mean_auc"] == pytest.approx(0.65)
    assert primary["mean_auc_skill"] == pytest.approx(0.15)
    assert primary["mean"] == pytest.approx(0.15)
    assert primary["months"] == 6


def test_hac_one_sided_inference_deterministic_and_positive(wp8):
    values = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15], dtype="float64")
    first = wp8.newey_west_hac(values)
    second = wp8.newey_west_hac(values.copy())
    assert first == second
    assert first["lag"] == 5
    assert first["t_stat"] > 0
    assert 0.0 <= first["one_sided_hac_p"] <= 1.0
    assert first["one_sided_hac_p"] == pytest.approx(0.0, abs=1e-9)
    assert first["ci_lower"] < first["mean"] < first["ci_upper"]


def test_hac_requires_at_least_two_values(wp8):
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="at least 2"):
        wp8.newey_west_hac(np.array([0.1]))


def test_hac_blocks_zero_long_run_variance(wp8):
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="long-run variance is invalid/zero"):
        wp8.newey_west_hac(np.array([0.1, 0.2]))


def test_verdict_mapping(wp8):
    confirmed = wp8.verdict_from_primary({"mean": 0.1, "one_sided_hac_p": 0.01})
    assert confirmed == "HOLDOUT_SIGNAL_CONFIRMED"
    inconclusive = wp8.verdict_from_primary({"mean": 0.1, "one_sided_hac_p": 0.2})
    assert inconclusive == "HOLDOUT_POSITIVE_BUT_INCONCLUSIVE"
    not_confirmed = wp8.verdict_from_primary({"mean": -0.1, "one_sided_hac_p": 0.01})
    assert not_confirmed == "HOLDOUT_NOT_CONFIRMED"
    zero_or_negative = wp8.verdict_from_primary({"mean": 0.0, "one_sided_hac_p": 1.0})
    assert zero_or_negative == "HOLDOUT_NOT_CONFIRMED"


# ── Immutable result/summary binding ────────────────────────────────────────
def test_result_binding_to_contract_candidate_holdout_and_code_hash(wp8):
    payload = {
        "schema_version": wp8.RESULT_SCHEMA_VERSION,
        "kind": wp8.RESULT_KIND,
        "contract_version": wp8.CONTRACT_VERSION,
        "contract_digest": "a" * 64,
        "canonical_wp7_generation_id": CANONICAL_GENERATION,
        "canonical_wp7_contract": "WP7_VALIDATION_CONTRACT_V5",
        "canonical_wp6_experiment_id": "experiment_ee434a07a25d",
        "canonical_final_candidate_freeze_id": CANONICAL_FREEZE_ID,
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
        "producing_commit": "c" * 40,
        "code_digest": "d" * 64,
        "frozen_final_config_id": "modelcfg_158499379e80",
        "frozen_final_features": ["x1"],
        "frozen_final_calibration": "none",
        "total_eligible_rows": 4,
        "securities": 2,
        "panel_diagnostics": {},
        "primary_metrics": {"mean_auc_skill": 0.1, "verdict": "HOLDOUT_SIGNAL_CONFIRMED"},
        "secondary_metrics": {},
        "verdict": "HOLDOUT_SIGNAL_CONFIRMED",
    }
    evaluation_id = wp8.compute_evaluation_id(payload)
    assert evaluation_id.startswith("evaluation_")

    contract_changed = dict(payload)
    contract_changed["contract_digest"] = "e" * 64
    assert wp8.compute_evaluation_id(contract_changed) != evaluation_id

    holdout_changed = dict(payload)
    holdout_changed["canonical_holdout_id"] = "holdout_wrong"
    assert wp8.compute_evaluation_id(holdout_changed) != evaluation_id

    code_changed = dict(payload)
    code_changed["code_digest"] = "f" * 64
    assert wp8.compute_evaluation_id(code_changed) != evaluation_id


def test_result_write_is_immutable(wp8, tmp_path):
    record = {
        "schema_version": wp8.RESULT_SCHEMA_VERSION,
        "kind": wp8.RESULT_KIND,
        "contract_version": wp8.CONTRACT_VERSION,
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
        "evaluation_id": "evaluation_aaaaaaaaaaaa",
    }
    target = wp8.write_immutable_result(record, root=tmp_path)
    assert target.is_file()
    with pytest.raises(wp8.Wp8HoldoutEvaluationError, match="already exists"):
        wp8.write_immutable_result(dict(record), root=tmp_path)


def test_summary_record_is_immutable_and_verify_reuse(wp8, tmp_path):
    summary = {
        "schema_version": wp8.SUMMARY_SCHEMA_VERSION,
        "evaluation_id": "evaluation_aaaaaaaaaaaa",
    }
    first = wp8.write_summary_record(summary, root=tmp_path)
    assert first["status"] == "written"
    second = wp8.write_summary_record(dict(summary), root=tmp_path)
    assert second["status"] == "verify_and_reuse"
    assert second["sha256"] == first["sha256"]

    divergent = dict(summary)
    divergent["evaluation_id"] = "evaluation_bbbbbbbbbbbb"
    with pytest.raises(ImmutabilityError):
        wp8.write_summary_record(divergent, root=tmp_path)


def test_contract_file_has_frozen_required_fields(wp8):
    contract = wp8.load_evaluation_contract(root=REPO_ROOT)
    assert contract["contract_version"] == "HOLDOUT_EVALUATION_CONTRACT_V1"
    assert contract["created_before_access"] is True
    assert contract["holdout_start"] == "2022-01-01"
    assert contract["holdout_end"] == "2025-08-31"
    assert contract["target"] == "outperform_12m"
    assert contract["evaluation_count"] == 1
    assert wp8.contract_file_digest(root=REPO_ROOT) is not None


def _source_only_feature_inputs():
    """Synthetic feature-panel inputs with one development and one holdout row."""
    panel = pd.DataFrame({
        "security_id": ["SYN"],
        "ticker": ["SYN"],
        "feature_asof": ["2020-01-31"],
        "price_date": ["2020-01-31"],
    })
    prices = pd.DataFrame({
        "security_id": ["SYN"],
        "trade_date": ["2018-01-02"],
        "raw_close": [100.0],
    })
    benchmark_prices = pd.DataFrame({
        "security_id": ["SPY"],
        "ticker": ["SPY"],
        "trade_date": ["2018-01-02"],
        "raw_close": [300.0],
    })
    fundamentals = pd.DataFrame(
        columns=["cik", "field", "value", "fiscal_period_start", "fiscal_period_end",
                 "accession", "available_at", "form"]
    )
    return panel, prices, benchmark_prices, fundamentals


def test_build_feature_panel_rejects_holdout_rows_by_default():
    panel, prices, benchmark_prices, fundamentals = _source_only_feature_inputs()
    panel = pd.concat([panel, pd.DataFrame({
        "security_id": ["SYN"],
        "ticker": ["SYN"],
        "feature_asof": ["2022-01-31"],
        "price_date": ["2022-01-31"],
    })], ignore_index=True)
    with pytest.raises(FeaturePanelError) as recorded:
        build_feature_panel(
            panel, prices, None, fundamentals, {"SYN": "0000000000"},
            benchmark_prices, None, restrict_to_development=False,
        )
    assert "locked-holdout row" in str(recorded.value)


def test_build_feature_panel_allow_locked_holdout_reports_counted_proof():
    panel, prices, benchmark_prices, fundamentals = _source_only_feature_inputs()
    panel = pd.concat([panel, pd.DataFrame({
        "security_id": ["SYN"],
        "ticker": ["SYN"],
        "feature_asof": ["2022-01-31"],
        "price_date": ["2022-01-31"],
    })], ignore_index=True)
    frame, summary = build_feature_panel(
        panel, prices, None, fundamentals, {"SYN": "0000000000"},
        benchmark_prices, None,
        restrict_to_development=False,
        allow_locked_holdout=True,
    )
    assert "feature_asof" in frame.columns
    assert summary["locked_holdout_rows_in_panel"] > 0
    assert summary["eligible_panel_rows"] == 2


def test_dedicated_holdout_loader_uses_unrestricted_flag_and_never_uses_wp7(wp8, monkeypatch):
    root = REPO_ROOT
    calls = {}

    binding = {
        "schema_version": "edgar_input_binding_v1",
        "edgar_fundamentals": {
            "shards": [],
            "shard_manifest_sha256": "a" * 64,
        },
        "edgar_cik_mapping": {
            "path": "artifacts/research/wp4/edgar_cik_mapping.json",
            "sha256": "b" * 64,
        },
    }

    monkeypatch.setattr(wp8, "assert_holdout_wp7_inputs_contract", lambda root=None: {})
    monkeypatch.setattr(
        wp8, "load_input_binding",
        lambda root=None: binding,
    )
    monkeypatch.setattr(
        wp8, "verify_edgar_input_binding",
        lambda root=None, binding=None: True,
    )
    monkeypatch.setattr(
        wp8, "load_and_verify_edgar_fundamentals",
        lambda root=None, binding=None: pd.DataFrame(),
    )
    monkeypatch.setattr(
        wp8, "load_and_verify_cik_by_ticker",
        lambda root=None, binding=None: ({}, {}),
    )
    monkeypatch.setattr(
        wp8, "load_wp5_engine",
        lambda root=None: SimpleNamespace(
            load_panel=lambda: pd.DataFrame(columns=["security_id", "ticker", "feature_asof", "price_date"]),
            load_targets=lambda: pd.DataFrame(),
            load_prices=lambda: pd.DataFrame(),
            load_actions=lambda: pd.DataFrame(),
            load_benchmark_prices=lambda: pd.DataFrame(),
            load_benchmark_actions=lambda: pd.DataFrame(),
            load_fundamentals=lambda: (_ for _ in ()).throw(
                AssertionError("holdout loader must not call unbound wp5.load_fundamentals")
            ),
            load_cik_by_ticker=lambda: (_ for _ in ()).throw(
                AssertionError("holdout loader must not call unbound wp5.load_cik_by_ticker")
            ),
        ),
    )

    def fake_build_feature_panel(panel, prices, actions, fundamentals, cik_by_ticker,
                                 benchmark_prices, benchmark_actions, **kwargs):
        calls["args"] = kwargs
        calls["cik_by_ticker"] = cik_by_ticker
        return pd.DataFrame({
            "security_id": ["SYN"],
            "ticker": ["SYN"],
            "feature_asof": pd.to_datetime(["2022-01-31"], utc=True),
            "x1": [0.1],
        }), {
            "rows": 1,
            "eligible_panel_rows": 1,
            "embargo_cutoff": "2021-01-01",
            "locked_holdout_rows_in_panel": 1,
        }

    monkeypatch.setattr(wp8, "build_feature_panel", fake_build_feature_panel)
    monkeypatch.setattr(
        wp8, "attach_targets",
        lambda frame, targets: frame.assign(outperform_12m=1.0, target_observable=True),
    )

    monkeypatch.setattr(wp8, "load_wp7_engine", lambda root=None: (_ for _ in ()).throw(
        AssertionError("dedicated holdout loader must not call WP7")
    ))

    frame, diagnostics = wp8.load_holdout_evaluation_panel(root)
    assert calls["args"]["restrict_to_development"] is False
    assert calls["args"]["allow_locked_holdout"] is True
    assert calls["cik_by_ticker"] == {}
    assert diagnostics["loader"] == "wp8_holdout_evaluation_panel"
    assert diagnostics["rows"] == 1
    assert diagnostics["feature_panel_summary"]["locked_holdout_rows_in_panel"] == 1
    assert diagnostics["edgar_input_binding"]["edgar_cik_mapping"]["sha256"] == "b" * 64


def test_wp8_code_digest_includes_input_binding(wp8, monkeypatch, tmp_path):
    read_paths = []
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(path):
        read_paths.append(path)
        return b"synthetic"

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    try:
        digest = wp8._code_digest(tmp_path)
    finally:
        monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

    assert digest == hashlib.sha256(b"synthetic" * len(read_paths)).hexdigest()
    assert any(path.name == "input_binding.json" for path in read_paths)
