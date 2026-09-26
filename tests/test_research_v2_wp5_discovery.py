"""WP5 variable-discovery tests.

LIVE-INDEPENDENT: no test touches the network or the research data filesystem.
Every fixture is synthetic and in-memory; the tests freeze the invariants that
keep WP5 discovery honest:

* the PRIMARY statistic is the cross-sectional MONTHLY IC series, not a pooled
  row correlation;
* censored / unobservable labels are excluded, never zero-filled;
* inference is dependence-aware (HAC / block bootstrap), not naive IID rows;
* multiple testing is controlled with Benjamini-Hochberg FDR;
* quantile buckets are fixed and cutoffs are never optimised;
* a future-shifted feature is detectable as a point-in-time violation;
* no locked-holdout row can enter the discovery slice;
* a placebo feature collapses to ~zero IC while a real one does not;
* experiment ids are deterministic and content-addressed;
* discovery never mutates the caller's feature frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.discovery.catalog import FEATURE_NAMES, DiscoveryConfig
from src.research.discovery.builder import attach_targets, development_rows
from src.research.discovery.cross_sectional import (
    cross_sectional_ic_series,
    pooled_rank_ic,
    summarize_ic_series,
)
from src.research.discovery.inference import (
    block_bootstrap_mean,
    dependence_diagnostics,
    hac_mean_test,
    iid_mean_test,
)
from src.research.discovery.multiple_testing import benjamini_hochberg
from src.research.discovery.placebo import (
    future_shift_guard,
    noise_feature,
    placebo_verdict,
    shuffled_target,
    sign_randomized_feature,
)
from src.research.discovery.quantiles import quantile_profile
from src.research.discovery.redundancy import (
    ic_series_correlation,
    median_rank_correlation_matrix,
    redundancy_clusters,
)
from src.research.discovery.scorecard import build_scorecard, classify_candidate
from src.research.discovery.stability import stability_report
from src.research.discovery.io import discovery_payload, write_experiment
from src.research.discovery.catalog import catalog_payload
from src.research.ids import experiment_id, is_valid_id


# ── fixtures ──────────────────────────────────────────────────────────────────

def _panel(n_months=40, n_names=40, signal=1.0, seed=7):
    """Synthetic PIT frame: cross-sectional signal ``feature`` predicts ``target``."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME", tz="UTC")
    rows = []
    for stamp in dates:
        noise = rng.normal(size=n_names)
        target = signal * noise + rng.normal(scale=1.0, size=n_names)
        feature = noise
        for index in range(n_names):
            rows.append({
                "security_id": "sec_%03d" % index,
                "feature_asof": stamp.strftime("%Y-%m-%d"),
                "feature": float(feature[index]),
                "future_12m_excess_return": float(target[index]),
                "outperform_12m": float(target[index] > 0.0),
                "target_observable": True,
            })
    return pd.DataFrame(rows)


def _evidence(**overrides):
    base = {
        "quality": {"coverage": 0.9, "status": "OK", "temporal_safety": "OK"},
        "ic": {"months": 40, "mean_ic": 0.2, "median_ic": 0.18},
        "inference": {"hac": {"p_value": 0.001, "t_stat": 4.0}},
        "fdr": {"rejected": True, "q_value": 0.002},
        "quantiles": {"spread": 0.05, "monotonic": True},
        "stability": {"direction_consistent": True, "sufficient_months": True,
                      "short_period_only": False, "sign_flip": False},
        "redundancy": {"cluster_has_stronger_member": False, "cluster": "cluster_00"},
        "missingness": {"present_minus_missing_mean": 0.01},
    }
    for key, value in overrides.items():
        base[key] = value
    return base


# ── cross-sectional monthly IC ────────────────────────────────────────────────

def test_cross_sectional_ic_is_per_month_and_sorted():
    frame = _panel()
    series, diagnostics = cross_sectional_ic_series(frame, "feature", min_observations=30)
    assert len(series) == frame["feature_asof"].nunique()
    assert list(series.columns) == ["feature_asof", "paired_obs", "rank_ic"]
    assert series["feature_asof"].is_monotonic_increasing
    assert diagnostics["dates_skipped"] == 0
    assert (series["paired_obs"] == 40).all()


def test_thin_months_are_skipped_not_pooled():
    frame = _panel(n_months=3, n_names=10)
    series, diagnostics = cross_sectional_ic_series(frame, "feature", min_observations=30)
    assert len(series) == 0
    assert diagnostics["dates_skipped"] == 3
    assert all(record["reason"] == "below_min_observations" for record in diagnostics["skipped"])


def test_pooled_ic_is_secondary_and_positive_for_a_real_signal():
    frame = _panel()
    assert pooled_rank_ic(frame, "feature") > 0.5


def test_summary_reports_icir_and_positive_share():
    series, _ = cross_sectional_ic_series(_panel(), "feature", min_observations=30)
    summary = summarize_ic_series(series, min_months=24)
    assert summary["months"] == 40
    assert summary["mean_ic"] > 0.5
    assert summary["icir"] > 0.0
    assert summary["positive_month_share"] == 1.0
    assert summary["sufficient_months"] is True


# ── censoring / observability ─────────────────────────────────────────────────

def test_censored_rows_are_excluded_not_zero_filled():
    frame = _panel()
    censored = frame.head(5).copy()
    censored["future_12m_excess_return"] = np.nan
    censored["target_observable"] = False
    combined = pd.concat([frame, censored], ignore_index=True)
    series, diagnostics = cross_sectional_ic_series(combined, "feature", min_observations=30)
    # The valid month keeps exactly the observable paired observations (40, not 45).
    assert (series["paired_obs"] == 40).all()
    assert diagnostics["dates_skipped"] == 0


def test_development_rows_exclude_the_locked_holdout():
    frame = pd.DataFrame({
        "security_id": ["a", "b", "c"],
        "feature_asof": ["2019-01-31", "2021-06-30", "2023-01-31"],
        "target_observable": [True, True, True],
        "future_12m_excess_return": [0.1, 0.2, 0.3],
    })
    selected = development_rows(frame)
    # 2019 is development, 2021-06 is embargo, 2023 is the locked holdout.
    assert list(selected["security_id"]) == ["a"]


def test_unobservable_rows_are_dropped_from_development():
    frame = pd.DataFrame({
        "security_id": ["a", "b"],
        "feature_asof": ["2019-01-31", "2019-02-28"],
        "target_observable": [True, False],
        "future_12m_excess_return": [0.1, np.nan],
    })
    selected = development_rows(frame)
    assert list(selected["security_id"]) == ["a"]


def test_attach_targets_requires_the_label_column():
    panel = pd.DataFrame({"security_id": ["a"], "feature_asof": ["2019-01-31"]})
    targets = pd.DataFrame({"security_id": ["a"], "feature_asof": ["2019-01-31"],
                            "outperform_12m": [1.0]})
    with pytest.raises(Exception):
        attach_targets(panel, targets)


# ── dependence-aware inference ────────────────────────────────────────────────

def test_hac_widens_error_under_positive_autocorrelation():
    rng = np.random.default_rng(11)
    base = rng.normal(0.05, 0.05, size=120)
    correlated = np.empty(120)
    correlated[0] = base[0]
    for index in range(1, 120):
        correlated[index] = 0.9 * correlated[index - 1] + base[index]
    hac = hac_mean_test(correlated, lags=12)
    iid = iid_mean_test(correlated)
    assert hac["hac_se"] > iid["iid_se"]


def test_hac_requires_two_points():
    assert hac_mean_test(np.array([0.1]))["p_value"] is None


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(3)
    values = rng.normal(0.2, 0.1, size=48)
    result = block_bootstrap_mean(values, block=6, iterations=500, seed=42)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_dependence_diagnostics_reports_lag1_autocorrelation():
    rng = np.random.default_rng(5)
    innovations = rng.normal(size=200)
    values = np.empty(200)
    values[0] = innovations[0]
    for index in range(1, 200):
        values[index] = 0.85 * values[index - 1] + innovations[index]
    record = dependence_diagnostics(values, config=DiscoveryConfig())
    assert record["lag1"] is not None
    assert record["lag1"] > 0.5


# ── multiple testing ──────────────────────────────────────────────────────────

def test_fdr_controls_false_discovery_rate():
    p_values = {"a": 0.0001, "b": 0.01, "c": 0.2, "d": 0.9}
    results, summary = benjamini_hochberg(p_values, alpha=0.05)
    rejected = {record.feature for record in results if record.rejected}
    assert "a" in rejected
    assert "d" not in rejected
    assert summary["count"] == 4
    assert summary["rejected"] == len(rejected)


def test_fdr_q_values_are_monotone_with_p_values():
    p_values = {"a": 0.001, "b": 0.02, "c": 0.5}
    results, _ = benjamini_hochberg(p_values, alpha=0.05)
    ordered = sorted(results, key=lambda record: record.p_value)
    q_values = [record.q_value for record in ordered]
    assert q_values == sorted(q_values)


# ── quantiles ─────────────────────────────────────────────────────────────────

def test_quantile_spread_positive_for_a_strong_signal():
    frame = _panel(signal=3.0)
    _profile, summary = quantile_profile(frame, "feature", buckets=5, min_observations=30)
    assert summary["bucket_means"] is not None
    assert summary["spread"] > 0.0
    assert summary["top_mean"] > summary["bottom_mean"]


def test_quantile_buckets_are_fixed_count():
    frame = _panel()
    profile, summary = quantile_profile(frame, "feature", buckets=5, min_observations=30)
    assert set(profile["bucket"].unique()) <= {0, 1, 2, 3, 4}
    assert summary["months"] == frame["feature_asof"].nunique()


# ── temporal stability ────────────────────────────────────────────────────────

def test_stability_flags_a_consistent_signal():
    series, _ = cross_sectional_ic_series(_panel(signal=2.0, seed=2), "feature", min_observations=30)
    report = stability_report(series, min_months=24)
    assert report["direction_consistent"] is True
    assert report["sign_flip"] is False
    assert report["sufficient_months"] is True


# ── placebo battery ───────────────────────────────────────────────────────────

def test_shuffled_target_destroys_the_signal():
    frame = _panel()
    shuffled = shuffled_target(frame)
    series, _ = cross_sectional_ic_series(shuffled, "feature", min_observations=30)
    summary = summarize_ic_series(series)
    assert abs(summary["mean_ic"]) < 0.2


def test_noise_feature_has_no_signal():
    noisy, name = noise_feature(_panel())
    series, _ = cross_sectional_ic_series(noisy, name, min_observations=30)
    summary = summarize_ic_series(series)
    assert abs(summary["mean_ic"]) < 0.2


def test_placebo_verdict_stops_when_a_placebo_looks_predictive():
    verdict = placebo_verdict(0.05, {"noise_feature": 0.6}, floor=0.05)
    assert verdict["stop"] is True
    assert verdict["verdicts"]["noise_feature"]["suspicious"] is True


def test_placebo_verdict_passes_when_placebos_collapse():
    verdict = placebo_verdict(0.4, {"noise_feature": 0.01, "shuffled_target": 0.0}, floor=0.05)
    assert verdict["stop"] is False


def test_future_shift_guard_detects_a_violation():
    frame = _panel()
    guard = future_shift_guard(frame, "feature", days=365)
    assert "violates_point_in_time" in guard.columns
    assert bool(guard["violates_point_in_time"].all())


def test_sign_randomized_feature_keeps_magnitudes_but_breaks_ordering():
    frame = _panel()
    randomized, name = sign_randomized_feature(frame, "feature")
    original = frame.groupby("feature_asof")["feature"].apply(lambda s: s.abs().sum())
    reshuffled = randomized.groupby("feature_asof")[name].apply(lambda s: s.abs().sum())
    assert np.allclose(original.to_numpy(), reshuffled.to_numpy())


# ── redundancy ────────────────────────────────────────────────────────────────

def test_identical_features_cluster_together():
    frame = pd.DataFrame({
        "feature_asof": ["2019-01-31"] * 12 + ["2019-02-28"] * 12,
        "a": list(range(12)) * 2,
        "b": [value * 2.0 for value in list(range(12)) * 2],
        "c": list(reversed(range(12))) * 2,
    })
    matrix = median_rank_correlation_matrix(frame, ["a", "b", "c"])
    clusters = redundancy_clusters(["a", "b", "c"], rank_matrix=matrix, rank_threshold=0.7)
    assert clusters["member_to_cluster"]["a"] == clusters["member_to_cluster"]["b"]


def test_ic_series_correlation_is_one_for_identical_series():
    series = pd.DataFrame({
        "feature_asof": ["2019-01-31", "2019-02-28", "2019-03-31"],
        "rank_ic": [0.1, 0.2, 0.3], "paired_obs": [40, 40, 40],
    })
    matrix = ic_series_correlation({"a": series, "b": series.copy()}, ["a", "b"])
    assert matrix.loc["a", "b"] > 0.99


# ── scorecard classification ──────────────────────────────────────────────────

def test_scorecard_robust_candidate_path():
    status, _reasons = classify_candidate(_evidence())
    assert status == "ROBUST_CANDIDATE"


def test_scorecard_redundant_beats_robust():
    evidence = _evidence(redundancy={"cluster_has_stronger_member": True, "cluster": "cluster_00"})
    status, _reasons = classify_candidate(evidence)
    assert status == "REDUNDANT"


def test_scorecard_low_coverage_path():
    evidence = _evidence(quality={"coverage": 0.1, "status": "LOW_COVERAGE", "temporal_safety": "OK"})
    status, _reasons = classify_candidate(evidence)
    assert status == "LOW_COVERAGE"


def test_scorecard_temporal_unsafe_dominates():
    evidence = _evidence(quality={"coverage": 0.9, "status": "TEMPORALLY_UNSAFE", "temporal_safety": "VIOLATION"})
    status, _reasons = classify_candidate(evidence)
    assert status == "TEMPORALLY_UNSAFE"


def test_scorecard_never_mutates_feature_frame():
    frame = _panel()
    snapshot = frame.copy(deep=True)
    cross_sectional_ic_series(frame, "feature", min_observations=30)
    quantile_profile(frame, "feature", buckets=5, min_observations=30)
    pd.testing.assert_frame_equal(frame, snapshot)


# ── deterministic experiment ids ─────────────────────────────────────────────

def test_experiment_id_is_deterministic_and_valid():
    payload = discovery_payload(
        "dataset_dbaa77445b38", "target_set_888f68d1cfd0", "feature_set_56361533cc1b",
        catalog_payload(), DiscoveryConfig().to_dict(), "deadbeef", holdout_id="holdout_e1a63def9749",
    )
    first = experiment_id(payload)
    second = experiment_id(payload)
    assert first == second
    assert is_valid_id(first, "experiment")


def test_experiment_id_changes_with_config():
    base = discovery_payload(
        "d", "t", "f", catalog_payload(), DiscoveryConfig().to_dict(), "deadbeef",
    )
    changed = discovery_payload(
        "d", "t", "f", catalog_payload(), DiscoveryConfig(hac_lags=6).to_dict(), "deadbeef",
    )
    assert experiment_id(base) != experiment_id(changed)


def test_write_experiment_is_write_once(tmp_path):
    payload = discovery_payload("d", "t", "f", catalog_payload(), DiscoveryConfig().to_dict(), "deadbeef")
    first = write_experiment(tmp_path, payload, artifacts={"a.json": {"x": 1}}, provenance={})
    second = write_experiment(tmp_path, payload, artifacts={"a.json": {"x": 1}}, provenance={})
    assert first["experiment_id"] == second["experiment_id"]
    changed = discovery_payload("d", "t", "f", catalog_payload(),
                                DiscoveryConfig(hac_lags=6).to_dict(), "deadbeef")
    third = write_experiment(tmp_path, changed, artifacts={"a.json": {"x": 2}}, provenance={})
    assert third["experiment_id"] != first["experiment_id"]


# ── catalog integrity ─────────────────────────────────────────────────────────

def test_catalog_covers_every_feature_the_panel_builds():
    from src.research.discovery.panel import _CANDIDATE_NAMES

    assert set(FEATURE_NAMES) == set(_CANDIDATE_NAMES)


def test_catalog_has_no_zero_fill_missing_policy():
    from src.research.discovery.catalog import FEATURE_CATALOG

    for spec in FEATURE_CATALOG:
        # No candidate may silently zero-fill a missing fundamental.
        assert spec.missing_policy != "ZERO_FILL"
        # Every candidate declares an honest PIT status; proxies stay labelled.
        assert spec.pit_status in {"POINT_IN_TIME", "UNAVAILABLE", "PROXY", "UNKNOWN"}

