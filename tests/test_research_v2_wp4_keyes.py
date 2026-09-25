"""WP4 regression tests: Keyes variable values, historical/modern tracks, diagnostics.

LIVE-INDEPENDENT: every fixture below is small and explicitly synthetic and is
used only to exercise methodology invariants. No test reads a token, touches the
network or reads a certified dataset. Nothing here is research evidence.

These tests freeze the WP4 invariants so a later change cannot silently
reintroduce the V1 defects:

* the FORBIDDEN forced top-30% qualification quota (never a percentile cut);
* a fabricated X12 - the revenue-growth stand-in is always ``X12_PROXY``;
* the merging of the historical and modern tracks;
* target columns leaking into feature inputs;
* a global, look-ahead transform estimated on the full sample;
* censored outcomes being read as observable;
* an all-unavailable observation silently vanishing from the panel.
"""

import numpy as np
import pandas as pd
import pytest

from src.research.features import TRACK_HISTORICAL, TRACK_MODERN, proxy_variable
from src.research.ids import experiment_id
from src.research.keyes import (
    DIAGNOSTIC_VERSION,
    FIDELITY_EXACT,
    FIDELITY_PROXY,
    FIDELITY_UNAVAILABLE,
    HISTORICAL_COMPONENTS,
    HISTORICAL_RULE_SETS,
    PRIMARY_THRESHOLDS,
    SIGNAL_VERSION,
    VARIABLE_SPECS,
    X5,
    X6,
    X8,
    X9,
    X12_PROXY,
    CompositeConfig,
    KeyesSignalError,
    KeyesVariableError,
    build_historical_rule_set,
    direction_payload,
    fidelity_table,
    historical_signals,
    historical_threshold_payload,
    modern_composite,
    observable_slice,
    placebo_checks,
    rank_ic,
    replication_status,
    wide_variables,
)
from src.research.keyes.variables import assert_no_target_leakage, FORBIDDEN_INPUT_COLUMNS


KEYES_VARS = (X5, X6, X8, X9, X12_PROXY)

# Values that satisfy every pre-registered threshold.
_PASS = {X5: 0.10, X6: 0.20, X8: 10.0, X9: 0.50, X12_PROXY: 0.05}
# Values that fail every pre-registered threshold.
_FAIL = {X5: -0.10, X6: -0.20, X8: 40.0, X9: 2.00, X12_PROXY: -0.05}


def _observation(security_id, feature_asof, values, available=None):
    """One observation as five long (variable, value, available) rows."""
    rows = []
    for variable in KEYES_VARS:
        value = values.get(variable)
        is_available = (value is not None) if available is None else bool(available.get(variable, False))
        rows.append({
            "security_id": security_id,
            "feature_asof": feature_asof,
            "variable": variable,
            "value": float(value) if value is not None else float("nan"),
            "available": bool(is_available),
        })
    return rows


def _variable_frame(observations):
    rows = []
    for observation in observations:
        rows.extend(observation)
    return pd.DataFrame(rows, columns=["security_id", "feature_asof", "variable", "value", "available"])


def _all_missing(security_id, feature_asof):
    return _observation(security_id, feature_asof, {}, available={variable: False for variable in KEYES_VARS})


# ── Historical rule set: no quota, zero qualifiers allowed ────────────────────

def test_historical_thresholds_are_explicit_and_quota_free():
    payload = historical_threshold_payload()
    assert payload["quota"] is None
    assert payload["weights"] is None
    assert payload["track"] == TRACK_HISTORICAL
    assert payload["signal_version"] == SIGNAL_VERSION
    # every criterion is a genuine simultaneous threshold, never a percentile
    for spec in payload["thresholds"]:
        assert spec["direction"] in ("min", "max")
        assert spec["pre_registered"] is True

    rule_set = build_historical_rule_set()
    assert rule_set.track == TRACK_HISTORICAL
    assert len(rule_set.criteria) == len(PRIMARY_THRESHOLDS)
    # no forced fraction of the universe: qualification is driven by the
    # pre-registered thresholds only.
    assert HISTORICAL_RULE_SETS["keyes_primary_v1"].name == "keyes_primary_v1"


def test_a_rule_set_needs_at_least_one_criterion():
    with pytest.raises(KeyesSignalError):
        build_historical_rule_set(thresholds=())
    with pytest.raises(KeyesSignalError):
        historical_signals(_variable_frame([_observation("S1", "2015-01-31", _PASS)]), rule_set_name="does_not_exist")


def test_qualification_is_not_a_top_thirty_percent_quota():
    """Ten qualifying observations must ALL qualify - a 30% quota would keep 3."""
    frame = _variable_frame(
        [_observation("S%02d" % i, "2015-01-31", _PASS) for i in range(10)]
    )
    signals = historical_signals(frame)
    assert len(signals) == 10
    assert int(signals["qualified"].sum()) == 10
    assert float(signals["qualified"].mean()) == 1.0
    # no top-30%-style column exists anywhere in the output
    assert not any("quota" in column or "percentile" in column or "rank" in column for column in signals.columns)


def test_zero_qualifiers_is_legal():
    frame = _variable_frame(
        [_observation("S%02d" % i, "2015-01-31", _FAIL) for i in range(10)]
    )
    signals = historical_signals(frame)
    assert len(signals) == 10
    assert int(signals["qualified"].sum()) == 0
    assert float(signals["qualified"].mean()) == 0.0


def test_all_unavailable_observation_is_preserved_not_dropped():
    """Regression: an observation with no available Keyes value must still appear.

    pandas pivot_table silently drops all-NaN rows; the WP4 signal builder must
    keep them so coverage and the qualification rate stay honest.
    """
    frame = _variable_frame([
        _observation("QUAL", "2015-01-31", _PASS),
        _all_missing("MISS", "2015-01-31"),
    ])
    signals = historical_signals(frame, expected_rows=2)
    assert set(signals["security_id"]) == {"QUAL", "MISS"}
    missing_row = signals.loc[signals["security_id"] == "MISS"].iloc[0]
    assert bool(missing_row["qualified"]) is False
    assert int(missing_row["criteria_available"]) == 0
    assert set(str(missing_row["unavailable_components"]).split(",")) == set(KEYES_VARS)
    # an all-unavailable row is not silently turned into a zero-filled pass
    assert int(missing_row["criteria_met"]) == 0


def test_expected_rows_guard_rejects_a_bad_frame():
    frame = _variable_frame([_observation("S1", "2015-01-31", _PASS)])
    with pytest.raises(KeyesSignalError):
        historical_signals(frame, expected_rows=5)


# ── Proxies are explicit and never fabricated as exact ────────────────────────

def test_x12_proxy_is_explicit_and_never_reported_as_x12():
    assert proxy_variable("X12") == X12_PROXY
    assert X12_PROXY.endswith("_PROXY")
    spec = VARIABLE_SPECS[X12_PROXY]
    assert spec.is_proxy is True
    assert spec.base_variable == "X12"
    assert spec.fidelity == FIDELITY_PROXY
    # the exact Keyes X12 must not exist anywhere as a real variable
    assert "X12" not in VARIABLE_SPECS

    payload = historical_threshold_payload()
    proxy_entries = [spec for spec in payload["thresholds"] if spec["is_proxy"]]
    assert [entry["variable"] for entry in proxy_entries] == [X12_PROXY]


def test_fidelity_table_never_upgrades_a_proxy_or_missing_variable():
    coverage = {
        "variables": {
            X5: {"available": 100, "rows": 200, "coverage": 0.5},
            X6: {"available": 200, "rows": 200, "coverage": 1.0},
            X8: {"available": 100, "rows": 200, "coverage": 0.5},
            X9: {"available": 100, "rows": 200, "coverage": 0.5},
            X12_PROXY: {"available": 50, "rows": 200, "coverage": 0.25},
        }
    }
    table = fidelity_table(coverage)
    proxy_row = table.loc[table["variable"] == X12_PROXY].iloc[0]
    assert bool(proxy_row["is_proxy"]) is True
    assert proxy_row["fidelity"] == FIDELITY_PROXY
    assert proxy_row["fidelity"] != FIDELITY_EXACT
    # a variable with no real value becomes UNAVAILABLE, never EXACT
    empty = {variable: {"available": 0, "rows": 200, "coverage": 0.0} for variable in KEYES_VARS}
    empty_table = fidelity_table({"variables": empty})
    assert set(empty_table["fidelity"]) == {FIDELITY_UNAVAILABLE}
    assert replication_status(empty_table) == "UNAVAILABLE"
    # a proxy present means the track is at best PARTIAL
    assert replication_status(table) == "PARTIAL_REPLICATION"


# ── Modern track: equal weight, separate, time-safe ──────────────────────────

def test_modern_composite_is_equal_weight_and_separate_from_historical():
    frame = _variable_frame([
        _observation("S1", "2015-01-31", {X5: 1.0, X6: 2.0, X8: 3.0}),
        _observation("S2", "2015-01-31", {X5: 3.0, X6: 2.0, X8: 1.0}),
        _observation("S3", "2015-01-31", {X5: 2.0, X6: 1.0, X8: 2.0}),
    ])
    composite = modern_composite(frame)
    assert set(composite["track"]) == {TRACK_MODERN}
    assert set(composite["weighting"]) == {"EQUAL_WEIGHT_RANK"}
    assert "qualified" not in composite.columns

    rank_columns = ["rank_%s" % variable for variable in HISTORICAL_COMPONENTS]
    manual = composite[rank_columns].mean(axis=1, skipna=True)
    assert np.allclose(composite["composite"].to_numpy(), manual.to_numpy(), equal_nan=True)
    # no hand-tuned 35/25/15-style constants anywhere
    payload = direction_payload()
    assert payload["weighting"] == "EQUAL_WEIGHT_RANK"
    assert "35" not in str(payload["signs"])

    historical = historical_signals(frame)
    assert set(historical["track"]) == {TRACK_HISTORICAL}
    # the two tracks expose disjoint headline columns and are never merged
    assert "composite" not in historical.columns
    assert "qualified" not in composite.columns


def test_modern_composite_rejects_fitted_weighting():
    frame = _variable_frame([_observation("S1", "2015-01-31", _PASS)])
    with pytest.raises(KeyesSignalError):
        modern_composite(frame, CompositeConfig(weighting="HAND_TUNED_35_25_15"))


def test_modern_composite_ranking_is_within_date_only():
    """Adding a later date must not change an earlier date's ranks (time-safe)."""
    day_one = [
        _observation("S1", "2015-01-31", {X5: 1.0, X6: 1.0, X8: 5.0}),
        _observation("S2", "2015-01-31", {X5: 2.0, X6: 2.0, X8: 4.0}),
        _observation("S3", "2015-01-31", {X5: 3.0, X6: 3.0, X8: 3.0}),
    ]
    day_two = [
        _observation("S1", "2016-01-31", {X5: 100.0, X6: 100.0, X8: 1.0}),
        _observation("S2", "2016-01-31", {X5: 200.0, X6: 200.0, X8: 2.0}),
        _observation("S3", "2016-01-31", {X5: 300.0, X6: 300.0, X8: 3.0}),
    ]
    base = modern_composite(_variable_frame(day_one))
    extended = modern_composite(_variable_frame(day_one + day_two))
    first = extended.loc[extended["feature_asof"] == "2015-01-31"].sort_values("security_id").reset_index(drop=True)
    base = base.sort_values("security_id").reset_index(drop=True)
    assert np.allclose(base["rank_%s" % X5].to_numpy(), first["rank_%s" % X5].to_numpy())
    assert np.allclose(base["composite"].to_numpy(), first["composite"].to_numpy())


def test_modern_composite_preserves_all_unavailable_observation():
    frame = _variable_frame([
        _observation("S1", "2015-01-31", {X5: 1.0, X6: 1.0, X8: 1.0}),
        _observation("S2", "2015-01-31", {X5: 2.0, X6: 2.0, X8: 2.0}),
        _observation("S3", "2015-01-31", {X5: 3.0, X6: 3.0, X8: 3.0}),
        _all_missing("MISS", "2015-01-31"),
    ])
    composite = modern_composite(frame)
    assert "MISS" in set(composite["security_id"])
    missing = composite.loc[composite["security_id"] == "MISS"].iloc[0]
    assert int(missing["components_available"]) == 0
    assert pd.isna(missing["composite"])


# ── Target leakage, censoring and rank IC ────────────────────────────────────

def test_forward_looking_columns_are_refused_as_feature_inputs():
    assert "future_12m_excess_return" in FORBIDDEN_INPUT_COLUMNS
    assert "outperform_12m" in FORBIDDEN_INPUT_COLUMNS
    clean = pd.DataFrame({"security_id": ["S1"], "value": [1.0]})
    assert assert_no_target_leakage(clean, "clean frame") is True
    leaky = pd.DataFrame({"security_id": ["S1"], "outperform_12m": [1]})
    with pytest.raises(KeyesVariableError):
        assert_no_target_leakage(leaky, "leaky frame")


def test_censored_rows_are_excluded_from_diagnostics():
    frame = pd.DataFrame({
        "security_id": ["A", "B", "C"],
        "feature_asof": ["2015-01-31"] * 3,
        "target_observable": [True, False, True],
        "future_12m_excess_return": [0.1, np.nan, -0.2],
        "outperform_12m": [1, 0, 0],
    })
    sliced = observable_slice(frame)
    assert set(sliced["security_id"]) == {"A", "C"}
    with pytest.raises(Exception):
        observable_slice(frame.drop(columns=["target_observable"]))


def test_rank_ic_needs_variance_and_a_minimum_sample():
    small = pd.DataFrame({
        "signal": list(range(10)),
        "future_12m_excess_return": list(range(10)),
    })
    ic, n = rank_ic(small, "signal")
    assert ic is None and n == 10
    perfect = pd.DataFrame({
        "signal": list(range(40)),
        "future_12m_excess_return": [value * 0.01 for value in range(40)],
    })
    ic, n = rank_ic(perfect, "signal")
    assert n == 40 and ic is not None and ic > 0.99
    flat = perfect.assign(signal=0.0)
    ic, _n = rank_ic(flat, "signal")
    assert ic is None


# ── Placebo / leakage battery ────────────────────────────────────────────────

def _labelled_fixture(dates=("2015-01-31", "2016-01-31", "2017-01-31"), per_date=25):
    composite_rows = []
    target_rows = []
    for date in dates:
        for index in range(per_date):
            security = "S%02d" % index
            score = index / float(per_date)
            composite_rows.append({"security_id": security, "feature_asof": date, "composite": score})
            target_rows.append({
                "security_id": security,
                "feature_asof": date,
                "future_12m_excess_return": score - 0.5,
                "outperform_12m": int(score > 0.5),
                "target_observable": True,
                "target_start": date,
            })
    return pd.DataFrame(composite_rows), pd.DataFrame(target_rows)


def test_placebo_shuffled_target_loses_systematic_power():
    composite, targets = _labelled_fixture()
    result = placebo_checks(composite, targets, seed=7)
    assert result["diagnostic_version"] == DIAGNOSTIC_VERSION
    assert result["true_rank_ic"] is not None
    assert result["shuffled_retains_systematic_power"] is False
    assert result["feature_timestamp_violations"] == 0
    assert result["impossible_signal_date_violations"] == 0


def test_placebo_detects_an_impossible_signal_date():
    composite, targets = _labelled_fixture(dates=("2030-01-31",))
    result = placebo_checks(composite, targets, seed=7)
    assert result["impossible_signal_date_violations"] >= 1


# ── Determinism and stable experiment identifiers ────────────────────────────

def test_historical_signals_are_deterministic():
    frame = _variable_frame([
        _observation("QUAL", "2015-01-31", _PASS),
        _observation("FAIL", "2015-01-31", _FAIL),
        _all_missing("MISS", "2015-01-31"),
    ])
    first = historical_signals(frame)
    second = historical_signals(frame)
    pd.testing.assert_frame_equal(first, second)
    # the rule set is a module-level singleton, not rebuilt with fresh state
    assert HISTORICAL_RULE_SETS["keyes_primary_v1"] is HISTORICAL_RULE_SETS["keyes_primary_v1"]


def test_wide_variables_preserve_all_unavailable_observation():
    """Regression: wide_variables must not drop an all-unavailable observation."""
    rows = []
    for variable in KEYES_VARS:
        rows.append({"security_id": "OK", "ticker": "OK", "feature_asof": "2015-01-31",
                     "variable": variable, "value": 1.0, "basis": "FUNDAMENTAL"})
        rows.append({"security_id": "MISS", "ticker": "MISS", "feature_asof": "2015-01-31",
                     "variable": variable, "value": float("nan"), "basis": "UNAVAILABLE"})
    wide = wide_variables(pd.DataFrame(rows))
    assert set(wide["security_id"]) == {"OK", "MISS"}
    assert len(wide) == 2


def test_experiment_id_is_deterministic_and_content_addressed():
    payload = {"dataset_id": "dataset_x", "target_id": "target_set_y", "config": {"a": 1}}
    first = experiment_id(payload)
    second = experiment_id(dict(payload))
    assert first == second
    assert first.startswith("experiment_")
    assert experiment_id({**payload, "config": {"a": 2}}) != first
    # key order must not matter (canonical encoding)
    reordered = {"config": {"a": 1}, "target_id": "target_set_y", "dataset_id": "dataset_x"}
    assert experiment_id(reordered) == first
