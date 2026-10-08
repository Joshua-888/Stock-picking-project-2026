"""WP9A.1 operational hardening tests (synthetic fixtures only).

These tests verify the WP9A live-forward overlay, the first-eligible month-end
resolver, fail-closed readiness inputs, and atomic/idempotent snapshot storage
using only in-memory synthetic frames and temporary directories. No network
access, no target/outcome data, no official snapshot, and no frozen champion
mutation occur here.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

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
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_module("wp9_forward_score_operational_test_module", "scripts/research_v2/wp9_forward_score.py")
REFRESH = _load_module("wp9_refresh_operational_test_module", "scripts/research_v2/wp9_refresh_forward_data.py")
READINESS = _load_module("wp9_readiness_operational_test_module", "scripts/research_v2/wp9_snapshot_readiness.py")

from src.research import immutability as immutability_mod  # noqa: E402
from src.research.wp9 import champion as champion_mod  # noqa: E402
from src.research.wp9 import forward_inputs as forward_inputs_mod  # noqa: E402
from src.research.wp9 import storage as storage_mod  # noqa: E402
from src.research.wp9 import temporal_gate as temporal_gate_mod  # noqa: E402
from src.research.wp9.champion import FROZEN_FEATURES, FROZEN_FREEZE_ID, ChampionError  # noqa: E402
from src.research.wp9.forward_inputs import _ForwardInputBuilder, ForwardInputError  # noqa: E402
from src.research.wp9.temporal_gate import TemporalGateError  # noqa: E402

OFFICIAL = storage_mod.OFFICIAL_KIND
DRY_RUN = storage_mod.DRY_RUN_KIND


# ── helpers ─────────────────────────────────────────────────────────────────


def _bindings(asof="2026-11-30", contract_digest="b" * 64, code_commit="c" * 40):
    data = {
        "contract_digest": contract_digest,
        "snapshot_asof": asof,
        "champion_freeze": FROZEN_FREEZE_ID,
        "model_hash": "m" * 64,
        "preprocessor_hash": "p" * 64,
        "calibrator_hash": "k" * 64,
        "universe_hash": "u" * 64,
        "source_manifest_hash": "s" * 64,
        "feature_snapshot_hash": "f" * 64,
        "code_commit": code_commit,
    }
    return data


def _record(bindings):
    sid = storage_mod.snapshot_id_from_bindings(bindings)
    arr = ["A", "TICK"]
    return storage_mod.snapshot_payload(
        snapshot_id=sid,
        snapshot_asof=bindings["snapshot_asof"],
        security_id=arr,
        ticker=arr,
        raw_model_score=[0.5] * 2,
        frozen_calibrated_score=[0.5] * 2,
        rank=[1, 2],
        percentile=[0.0, 1.0],
        universe_size=2,
        model_hash=bindings["model_hash"],
        feature_snapshot_hash=bindings["feature_snapshot_hash"],
        source_manifest_hash=bindings["source_manifest_hash"],
        code_commit=bindings["code_commit"],
        created_at_utc="2026-11-30T21:00:00+00:00",
        eligibility_status=["scoreable", "scoreable"],
        quality_flags=[[], []],
        model_freeze_id=FROZEN_FREEZE_ID,
    )


class _RecordingBuilder(_ForwardInputBuilder):
    """Builder whose external layer/fundamental loaders are synthetic."""

    def __init__(self, root):
        super().__init__(root=root)
        self.calls = []

    def load_layer_records(self):
        self.calls.append(("certified_records",))
        return {
            "silver_prices": "cert_p",
            "silver_actions": "cert_a",
            "silver_membership": "cert_m",
            "price_rows": 10,
            "action_rows": 1,
            "windows": 2,
        }

    def _load_live_layer_records(self):
        self.calls.append(("live_records",))
        return {
            "silver_prices": "live_p",
            "silver_actions": "live_a",
            "silver_membership": "live_m",
            "silver_benchmark_prices": "live_bp",
            "silver_benchmark_actions": "live_ba",
            "price_rows": 20,
            "action_rows": 2,
            "windows": 2,
        }

    def load_fundamentals_and_cik(self):
        self.calls.append(("fundamentals",))
        return pd.DataFrame(), {}, {"mapping": {}}, {"edgar_fundamentals": {"shards": []}}

    def load_benchmark_actions(self, live_layer_records=None):
        self.calls.append(("benchmark_actions", live_layer_records))
        return pd.DataFrame(
            columns=["security_id", "kind", "effective_date", "numerator", "denominator", "amount"]
        )


def _fake_reader(calls, snapshot_date="2026-10-08"):
    """Return a monkeypatchable read_silver_table producing synthetic tables."""

    members = pd.DataFrame(
        [
            {"security_id": "AAA", "ticker": "AAA", "membership_start": "2000-01-01", "membership_end": None},
            {"security_id": "BBB", "ticker": "BBB", "membership_start": "2000-01-01", "membership_end": None},
        ]
    )
    prices = pd.DataFrame(
        [
            {"security_id": "AAA", "ticker": "AAA", "trade_date": "2026-10-07", "raw_close": 100.0},
            {"security_id": "BBB", "ticker": "BBB", "trade_date": "2026-10-07", "raw_close": 200.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "security_id": "AAA",
                "ticker": "AAA",
                "kind": "split",
                "effective_date": "2026-10-01",
                "numerator": 2.0,
                "denominator": 1.0,
                "amount": None,
            }
        ]
    )
    benchmark = pd.DataFrame(
        [{"security_id": "SPY", "ticker": "SPY", "trade_date": "2026-10-07", "raw_close": 500.0}]
    )
    benchmark_actions = pd.DataFrame(
        columns=["security_id", "kind", "effective_date", "numerator", "denominator", "amount"]
    )

    def reader(root, name, version):
        calls.append((name, str(version)))
        if name in ("wp9_live_prices", "wp2b_sp500_pit_prices_silver"):
            return prices.copy()
        if name in ("wp9_live_actions", "wp2b_sp500_pit_actions_silver"):
            return actions.copy()
        if name in ("wp9_live_membership", "wp2b_sp500_pit_membership_silver"):
            return members.copy()
        if name in ("wp9_live_benchmark_prices", "benchmark_gold_SPY"):
            return benchmark.copy()
        if name == "wp9_live_benchmark_actions":
            return benchmark_actions.copy()
        raise AssertionError("unexpected table %s" % name)

    return reader


def _feature_panel_builder(snapshot_date="2026-10-08"):
    def builder(panel_rows, *args, **kwargs):
        rows = []
        for _, row in panel_rows.iterrows():
            payload = {
                "security_id": row["security_id"],
                "ticker": row["ticker"],
                "feature_asof": row["feature_asof"],
            }
            for feature in FROZEN_FEATURES:
                payload[feature] = 1.0
            rows.append(payload)
        frame = pd.DataFrame(rows)
        summary = {"rows": int(len(frame))}
        return frame, summary

    return builder


# ── D: first-eligible month-end resolver ────────────────────────────────────


def test_resolver_rejects_incomplete_current_october():
    freeze = "2026-10-08T00:01:09+02:00"
    trade_dates = [
        "2026-10-01",
        "2026-10-02",
        "2026-10-07",
        "2026-10-08",
    ]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is False
    assert result["status"] == "NOT_READY"
    assert result["candidate_month"] == "2026-10"
    assert result["reason"] == "coverage_incomplete_no_following_month_data"


def test_resolver_weekend_month_end():
    # October 2026 ends on a Saturday; last observed eligible date is Friday.
    trade_dates = ["2026-10-29", "2026-10-30", "2026-11-02"]
    eligible = temporal_gate_mod.eligible_score_date_for_month("2026-10-15", trade_dates)
    assert eligible == "2026-10-30"


def test_resolver_holiday_month_end_uses_actual_session():
    # December 2026 ends on Thursday; simulate a holiday close on the 31st.
    trade_dates = ["2026-12-29", "2026-12-30", "2027-01-04"]
    eligible = temporal_gate_mod.eligible_score_date_for_month("2026-12-25", trade_dates)
    assert eligible == "2026-12-30"
    assert eligible != "2026-12-31"


def test_resolver_incomplete_month_no_following_data_raises():
    trade_dates = ["2026-10-01", "2026-10-08", "2026-10-20"]
    with pytest.raises(TemporalGateError, match="coverage is incomplete"):
        temporal_gate_mod.eligible_score_date_for_month("2026-10-10", trade_dates)


def test_resolver_data_ending_before_month_end_raises():
    trade_dates = ["2026-10-01", "2026-10-10"]
    with pytest.raises(TemporalGateError, match="coverage is incomplete"):
        temporal_gate_mod.eligible_score_date_for_month("2026-10-01", trade_dates)


def test_resolver_exact_final_eligible_trading_day_after_september_freeze():
    freeze = "2026-09-30T21:00:00+00:00"
    trade_dates = [
        "2026-10-29",
        "2026-10-30",
        "2026-11-02",
    ]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is True
    assert result["status"] == "READY"
    assert result["first_eligible_date"] == "2026-10-30"
    assert result["candidate_month"] == "2026-10"


def test_resolver_date_before_freeze_month_not_available():
    freeze = "2026-10-31T21:00:00+00:00"
    trade_dates = ["2026-10-30", "2026-11-02"]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is False
    assert result["candidate_month"] == "2026-11"


def test_resolver_latest_at_or_before_freeze_not_eligible():
    freeze = "2026-10-28T21:59:03Z"
    trade_dates = ["2026-10-27", "2026-11-03", "2026-11-04"]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is False
    assert result["status"] == "NOT_READY"
    assert result["candidate_month"] == "2026-10"
    assert result["first_eligible_date"] == "2026-10-27"
    assert result["reason"] == "first_eligible_date_not_after_freeze"


def test_resolver_eligible_when_latest_close_is_after_freeze():
    freeze = "2026-10-01T00:00:00Z"
    trade_dates = ["2026-10-30", "2026-11-02"]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is True
    assert result["status"] == "READY"
    assert result["first_eligible_date"] == "2026-10-30"


def test_resolver_candidate_advances_to_next_month_after_freeze():
    freeze = "2026-10-31T21:00:00Z"
    trade_dates = ["2026-10-30", "2026-11-03", "2026-11-04"]
    result = temporal_gate_mod.first_eligible_official_snapshot_date(freeze, trade_dates)
    assert result["eligible"] is False
    assert result["candidate_month"] == "2026-11"


def test_live_membership_uses_retrieval_date_not_invented_start():
    resolutions = [
        {
            "security_id": "AAPL",
            "ticker": "AAPL",
            "method": "index_symbol",
            "resolved": True,
            "reason": None,
        },
        {
            "security_id": "ZZZ",
            "ticker": "ZZZ",
            "method": None,
            "resolved": False,
            "reason": "missing_symbol_index",
        },
    ]
    frame = REFRESH.build_live_membership(resolutions, as_of="2026-10-08")
    assert "1900-01-01" not in set(frame["membership_start"])
    assert set(frame["membership_start"]) == {"2026-10-08"}
    assert frame["start_known"].eq(False).all()
    assert set(frame["membership_start_source"]) == {"current_sp500_list_retrieval_date"}


# ── B: live overlay + certified reproducibility ─────────────────────────────


def test_manifest_default_benchmark_is_certified():
    builder = _ForwardInputBuilder(root=REPO_ROOT)
    manifest = builder.build_source_manifest(
        "2026-10-08",
        {"silver_prices": "p", "silver_actions": "a", "silver_membership": "m"},
        {"edgar_binding": {"edgar_fundamentals": {"shards": []}}},
    )
    assert manifest["benchmark"]["name"] == "benchmark_gold_SPY"
    assert manifest["benchmark"]["version"] == "a0ab5bd2382518d4"


def test_manifest_live_benchmark_overlay():
    builder = _ForwardInputBuilder(root=REPO_ROOT)
    manifest = builder.build_source_manifest(
        "2026-10-08",
        {
            "silver_prices": "p",
            "silver_actions": "a",
            "silver_membership": "m",
            "silver_benchmark_prices": "bp_version",
        },
        {"edgar_binding": {"edgar_fundamentals": {"shards": []}}},
    )
    assert manifest["benchmark"]["name"] == "wp9_live_benchmark_prices"
    assert manifest["benchmark"]["version"] == "bp_version"


def test_default_build_uses_certified_layer_versions(monkeypatch, tmp_path):
    calls = []
    builder = _RecordingBuilder(root=tmp_path)
    monkeypatch.setattr(
        forward_inputs_mod.layers,
        "read_silver_table",
        _fake_reader(calls),
    )
    builder.build(
        "2026-10-08",
        feature_panel_builder=_feature_panel_builder(),
        source_data_kind=None,
    )
    used = dict(calls)
    assert used["wp2b_sp500_pit_prices_silver"] == "cert_p"
    assert used["wp2b_sp500_pit_actions_silver"] == "cert_a"
    assert used["wp2b_sp500_pit_membership_silver"] == "cert_m"
    assert used["benchmark_gold_SPY"] == "a0ab5bd2382518d4"


def test_live_build_reads_live_overlay_tables(monkeypatch, tmp_path):
    calls = []
    builder = _RecordingBuilder(root=tmp_path)
    monkeypatch.setattr(
        forward_inputs_mod.layers,
        "read_silver_table",
        _fake_reader(calls),
    )
    inputs = builder.build(
        "2026-10-08",
        feature_panel_builder=_feature_panel_builder(),
        source_data_kind="live",
    )
    used = dict(calls)
    assert used["wp9_live_prices"] == "live_p"
    assert used["wp9_live_actions"] == "live_a"
    assert used["wp9_live_membership"] == "live_m"
    assert used["wp9_live_benchmark_prices"] == "live_bp"
    assert inputs.source_manifest["benchmark"]["name"] == "wp9_live_benchmark_prices"
    assert inputs.universe["universe_count"] == 2
    assert len(inputs.score_frame) == 2


# ── E: fail-closed injection (non-network) ──────────────────────────────────


def test_missing_model_artifact_refuses_to_load(tmp_path):
    record = {
        "freeze_id": FROZEN_FREEZE_ID,
        "final_selected_features": list(FROZEN_FEATURES),
        "final_calibration": "platt_scaling",
        "artifact_hashes": {
            "model": {"sha256": "m" * 64, "path": str(tmp_path / "model.pkl")},
            "preprocessor": {"sha256": "p" * 64, "path": str(tmp_path / "pre.pkl")},
            "calibrator": {"sha256": "k" * 64, "path": str(tmp_path / "cal.pkl")},
        },
    }
    with pytest.raises(ChampionError, match="artifact is missing"):
        champion_mod.load_champion(root=tmp_path, freeze_record=record)


def test_altered_model_sha_refuses_to_load(tmp_path):
    model_path = tmp_path / "model.pkl"
    pre_path = tmp_path / "pre.pkl"
    cal_path = tmp_path / "cal.pkl"
    model_path.write_bytes(b"model")
    pre_path.write_bytes(b"pre")
    cal_path.write_bytes(b"cal")
    record = {
        "freeze_id": FROZEN_FREEZE_ID,
        "final_selected_features": list(FROZEN_FEATURES),
        "final_calibration": "platt_scaling",
        "artifact_hashes": {
            "model": {"sha256": "0" * 64, "path": str(model_path)},
            "preprocessor": {"sha256": "p" * 64, "path": str(pre_path)},
            "calibrator": {"sha256": "k" * 64, "path": str(cal_path)},
        },
    }
    with pytest.raises(ChampionError, match="hash mismatch"):
        champion_mod.load_champion(root=tmp_path, freeze_record=record)


def test_target_column_present_blocks_scoring():
    frame = pd.DataFrame({"security_id": ["A"], "outperform_12m": [1]})
    with pytest.raises(SCORER.Wp9ScoringError, match="target_outcome_columns_present"):
        SCORER.assert_no_target_columns(frame)


def test_official_mode_before_eligible_date_blocks(tmp_path):
    contract = type(
        "C",
        (),
        {},
    )()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        SCORER,
        "contract_freeze_timestamp",
        lambda contract, root: "2026-10-31T00:00:00+00:00",
    )
    with pytest.raises(SCORER.Wp9ScoringError, match="asof_not_after_contract_freeze"):
        SCORER.assert_asof_after_freeze("2026-10-08", contract, tmp_path)
    monkeypatch.undo()


def test_future_membership_or_action_row_rejected_by_bound():
    bound = pd.Timestamp("2026-10-08T21:00:00+00:00")
    frame = pd.DataFrame({"effective_date": ["2026-10-08", "2026-10-09"]})
    with pytest.raises(REFRESH.LiveRefreshError, match="after bound"):
        REFRESH._reject_payload_after_bound(
            frame, date_column="effective_date", bound=bound, label="test"
        )


def test_future_filing_is_filtered_by_availability():
    builder = _ForwardInputBuilder(root=REPO_ROOT)
    fundamentals = pd.DataFrame(
        [
            {
                "cik": "1",
                "field": "eps",
                "value": 1.0,
                "fiscal_period_start": "2026-04-01",
                "fiscal_period_end": "2026-06-30",
                "filing_date": "2026-11-30",
                "available_at": "2026-11-30T00:00:00+00:00",
            }
        ]
    )
    filtered = builder._admissible_fundamentals_for_close(
        fundamentals, pd.Timestamp("2026-10-08T21:00:00+00:00")
    )
    assert len(filtered) == 0


def test_missing_certified_binding_raises(tmp_path):
    builder = _ForwardInputBuilder(root=tmp_path)
    with pytest.raises(ForwardInputError, match="missing certified input artifact"):
        builder.load_fundamentals_and_cik()


def test_provider_network_failure_is_redacted_and_hard():
    class BrokenProvider:
        def fetch_delisted_prices(self, *args, **kwargs):
            raise RuntimeError("token=sk-12345678901234567890 boom")

        def fetch_corporate_actions(self, *args, **kwargs):
            return pd.DataFrame(columns=["kind", "effective_date", "numerator", "denominator", "amount"])

    bound = pd.Timestamp("2026-10-08T21:00:00+00:00")
    with pytest.raises(REFRESH.LiveRefreshError, match="price fetch failed"):
        REFRESH.fetch_market_for_symbol(BrokenProvider(), "AAA", bound)


def test_empty_price_payload_is_hard_failure():
    class EmptyProvider:
        def fetch_delisted_prices(self, *args, **kwargs):
            return pd.DataFrame()

        def fetch_corporate_actions(self, *args, **kwargs):
            return pd.DataFrame(columns=["kind", "effective_date", "numerator", "denominator", "amount"])

    bound = pd.Timestamp("2026-10-08T21:00:00+00:00")
    with pytest.raises(REFRESH.LiveRefreshError, match="price payload was empty"):
        REFRESH.fetch_market_for_symbol(EmptyProvider(), "AAA", bound)


def test_partial_missing_price_columns_are_hard_failure():
    class MalformedProvider:
        def fetch_delisted_prices(self, *args, **kwargs):
            return pd.DataFrame({"trade_date": ["2026-10-07"]})

        def fetch_corporate_actions(self, *args, **kwargs):
            return pd.DataFrame(columns=["kind", "effective_date", "numerator", "denominator", "amount"])

    bound = pd.Timestamp("2026-10-08T21:00:00+00:00")
    with pytest.raises(REFRESH.LiveRefreshError, match="missing columns"):
        REFRESH.fetch_market_for_symbol(MalformedProvider(), "AAA", bound)


def test_dirty_producing_path_detected():
    assert SCORER._is_wp9_producing_path("src/research/wp9/forward_inputs.py")
    assert SCORER._is_wp9_producing_path("scripts/research_v2/wp9_forward_score.py")
    assert not SCORER._is_wp9_producing_path("docs/readme.md")


# ── F: atomicity and idempotency ────────────────────────────────────────────


def test_interrupted_json_write_leaves_no_canonical_file(tmp_path, monkeypatch):
    target = tmp_path / "snapshot.json"
    original_replace = os.replace

    def exploding_replace(src, dst):
        raise OSError("simulated interrupt")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(OSError):
        immutability_mod.write_json_atomic(target, {"a": 1})
    assert not target.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []
    monkeypatch.setattr(os, "replace", original_replace)


def test_duplicate_official_asof_cannot_create_second_evidence(tmp_path):
    bindings = _bindings(asof="2026-11-30")
    record = _record(bindings)
    storage_mod.write_snapshot(record, bindings, kind=OFFICIAL, root=tmp_path)

    other_bindings = dict(bindings)
    other_bindings["model_hash"] = "z" * 64
    other_record = _record(other_bindings)
    with pytest.raises(storage_mod.Wp9StorageError, match="duplicate"):
        storage_mod.write_snapshot(other_record, other_bindings, kind=OFFICIAL, root=tmp_path)

    snapshot_files = [
        p
        for p in (tmp_path / "provenance" / "wp9" / "predictions").glob("*.json")
        if p.name != "index.json"
    ]
    assert len(snapshot_files) == 1


def test_dry_run_rerun_is_verify_and_reuse_and_non_evidentiary(tmp_path):
    bindings = _bindings(asof="2026-11-30")
    record = _record(bindings)
    first = storage_mod.write_snapshot(record, bindings, kind=DRY_RUN, root=tmp_path)
    again = storage_mod.write_snapshot(record, bindings, kind=DRY_RUN, root=tmp_path)
    assert first["index_outcome"] == "written"
    assert again["index_outcome"] == "verify_and_reuse"
    official_index = storage_mod.canonicalise_prediction_index(root=tmp_path)
    assert official_index["entries"] == []
    assert (tmp_path / "provenance" / "wp9" / "predictions").exists() is False


def test_snapshot_write_never_touches_champion_dir(tmp_path):
    bindings = _bindings(asof="2026-11-30")
    record = _record(bindings)
    storage_mod.write_snapshot(record, bindings, kind=DRY_RUN, root=tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_index_duplicate_append_fail_closed(tmp_path):
    schema = storage_mod.PREDICTION_INDEX_SCHEMA
    index_path = tmp_path / "index.json"
    entry_a = {"snapshot_asof": "2026-11-30", "x": 1}
    entry_b = {"snapshot_asof": "2026-11-30", "x": 2}
    storage_mod._append_unique_index(
        index_path, schema, entry_a, duplicate_asof_field="snapshot_asof"
    )
    with pytest.raises(storage_mod.Wp9StorageError, match="duplicate"):
        storage_mod._append_unique_index(
            index_path, schema, entry_b, duplicate_asof_field="snapshot_asof"
        )
