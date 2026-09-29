"""WP5 discovery orchestration.

This module runs the full discovery pipeline over the point-in-time feature panel
and returns a complete, self-consistent result bundle. It is the ONLY place the
individual discovery modules are combined, and it enforces the research rules in
one auditable place:

* discovery operates on DEVELOPMENT rows only (pre-embargo, observable labels);
  a locked-holdout row raises;
* the PRIMARY IC is the cross-sectional monthly series, never a pooled row value;
* inference is dependence-aware (HAC + block bootstrap), never naive row p-values;
* multiple testing is controlled with Benjamini-Hochberg FDR;
* placebo checks run against the same machinery and STOP on a suspicious placebo;
* no final feature set and no weighted score is produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..holdout import embargo_cutoff, holdout_mask
from . import keyes_benchmark as _keyes
from .catalog import CATALOG_BY_NAME, FEATURE_NAMES, DiscoveryConfig
from .cross_sectional import cross_sectional_ic_series, pooled_rank_ic, summarize_ic_series
from .handoff import build_handoff
from .inference import block_bootstrap_mean, dependence_diagnostics, hac_mean_test, iid_mean_test
from .missingness import missingness_effect
from .multiple_testing import benjamini_hochberg
from .panel import build_feature_panel, development_mask
from .placebo import (
    future_shift_guard,
    noise_feature,
    placebo_verdict,
    shuffled_target,
    sign_randomized_feature,
)
from .quality import feature_quality
from .quantiles import quantile_profile
from .redundancy import (
    ic_series_correlation,
    median_rank_correlation_matrix,
    pair_similarity,
    redundancy_clusters,
)
from .scorecard import build_scorecard, classification_counts
from .stability import stability_report


class DiscoveryBuildError(RuntimeError):
    """Raised when the discovery build cannot proceed honestly."""


# ── Target labelling ──────────────────────────────────────────────────────────

def attach_targets(frame, targets, target="future_12m_excess_return"):
    """Merge the certified target columns onto the feature panel by key."""
    keys = ["security_id", "feature_asof"]
    label_columns = [name for name in (
        "future_12m_excess_return", "outperform_12m", "target_observable",
        "target_known_at", "target_end", "target_censor_reason",
    ) if name in targets.columns]
    right = targets.loc[:, keys + label_columns].copy()
    merged = frame.merge(right, on=keys, how="left")
    if target not in merged.columns:
        raise DiscoveryBuildError("targets do not carry %r" % (target,))
    return merged


def development_rows(frame):
    """Development slice with an observable label, excluding the locked holdout."""
    mask = development_mask(frame)
    selected = frame.loc[mask].copy()
    leaked = int(holdout_mask(selected).sum()) if len(selected) else 0
    if leaked:
        raise DiscoveryBuildError("discovery slice contains %d locked-holdout row(s)" % leaked)
    return selected


# ── Per-feature evidence ──────────────────────────────────────────────────────

def feature_evidence(frame, feature, config):
    """Collect the full evidence bundle for ONE candidate feature."""
    series, ic_diagnostics = cross_sectional_ic_series(
        frame, feature, min_observations=config.min_ic_observations,
    )
    ic_summary = summarize_ic_series(series, min_months=config.min_months_for_stability)
    values = series["rank_ic"].to_numpy(dtype="float64") if len(series) else np.array([])

    if len(values) >= 2:
        hac = hac_mean_test(values, lags=config.hac_lags)
        iid = iid_mean_test(values)
        bootstrap = block_bootstrap_mean(values, block=config.bootstrap_block,
                                        iterations=config.bootstrap_iterations, seed=config.bootstrap_seed)
        dependence = dependence_diagnostics(values, config=config)
    else:
        hac = {"n": len(values), "mean": None, "hac_se": None, "t_stat": None, "p_value": None}
        iid = {"n": len(values), "mean": None, "iid_se": None, "t_stat": None, "p_value": None}
        bootstrap = {"n": len(values), "mean": None, "ci_low": None, "ci_high": None,
                     "p_value": None, "iterations": 0, "block": config.bootstrap_block}
        dependence = {"autocorrelation": {}, "overlap_note": "series too short"}

    _, quantiles = quantile_profile(
        frame, feature, asof_col="feature_asof", buckets=config.quantile_buckets,
        min_observations=config.min_quantile_observations,
    )
    quality = feature_quality(frame, feature, low_coverage_threshold=config.low_coverage_threshold)
    stability = stability_report(series, frame=frame, feature=feature,
                                 min_months=config.min_months_for_stability,
                                 stable_positive_share=config.stable_positive_share,
                                 stable_negative_share=config.stable_negative_share)
    missingness = missingness_effect(frame, feature)

    return {
        "quality": quality,
        "ic": ic_summary,
        "ic_series": series,
        "ic_diagnostics": ic_diagnostics,
        "inference": {"hac": hac, "iid": iid, "bootstrap": bootstrap, "dependence": dependence},
        "quantiles": quantiles,
        "stability": stability,
        "missingness": missingness,
        "pooled_rank_ic": pooled_rank_ic(frame, feature),
    }


# ── Redundancy wiring ────────────────────────────────────────────────────────

def build_redundancy(frame, per_feature, config):
    """Compute rank/IC matrices and DESCRIPTIVE redundancy groups for all candidates.

    The grouping uses only pre-outcome similarity structure (cross-sectional rank
    correlation and monthly IC-series correlation) with complete linkage; the
    economic family is carried as factual evidence. NO feature is ever elected as a
    group winner: the full-sample outcome performance (mean IC) is deliberately NOT
    used to rank or select members, because that would be a target-driven
    full-sample selection decision belonging to WP6 inside training windows.
    """
    names = list(FEATURE_NAMES)
    families = {name: CATALOG_BY_NAME[name].cluster_family for name in names}
    rank_matrix = median_rank_correlation_matrix(frame, names)
    ic_series = {name: per_feature[name]["ic_series"] for name in names if name in per_feature}
    ic_matrix = ic_series_correlation(ic_series, names)
    clusters = redundancy_clusters(
        names, rank_matrix=rank_matrix, ic_matrix=ic_matrix, families=families,
        rank_threshold=config.redundancy_rank_corr, ic_threshold=config.redundancy_ic_corr,
    )
    for cluster_id, members in clusters["clusters"].items():
        for member in members:
            peers = [peer for peer in members if peer != member]
            similarities = [
                value for value in (pair_similarity(rank_matrix, ic_matrix, member, peer) for peer in peers)
                if value is not None
            ]
            evidence = per_feature.setdefault(member, {}).setdefault("redundancy", {})
            evidence["cluster"] = cluster_id
            evidence["group_size"] = len(members)
            # Descriptive only: membership in a similarity group is the fact.
            evidence["redundant"] = bool(len(members) > 1)
            evidence["similarity_max"] = max(similarities) if similarities else None
            evidence["similarity_min"] = min(similarities) if similarities else None
            evidence["group_families"] = clusters["cluster_families"].get(cluster_id, [])
            evidence["linkage"] = clusters.get("linkage")
    return rank_matrix, ic_matrix, clusters


# ── Placebo battery ──────────────────────────────────────────────────────────

def run_placebos(frame, config, reference_ic):
    """Run the falsification battery on the discovery slice."""
    shuffled = shuffled_target(frame)
    shuffled_series, _ = cross_sectional_ic_series(shuffled, "twelve_month_momentum",
                                                   min_observations=config.min_ic_observations)
    shuffled_summary = summarize_ic_series(shuffled_series)

    noisy, noise_name = noise_feature(frame)
    noise_series, _ = cross_sectional_ic_series(noisy, noise_name,
                                                min_observations=config.min_ic_observations)
    noise_summary = summarize_ic_series(noise_series)

    randomized, randomized_name = sign_randomized_feature(frame, "twelve_month_momentum")
    randomized_series, _ = cross_sectional_ic_series(randomized, randomized_name,
                                                     min_observations=config.min_ic_observations)
    randomized_summary = summarize_ic_series(randomized_series)

    guard = future_shift_guard(frame, "twelve_month_momentum", days=365)
    guard_violations = int(guard["violates_point_in_time"].sum())

    placebo_ics = {
        "shuffled_target": shuffled_summary.get("mean_ic"),
        "noise_feature": noise_summary.get("mean_ic"),
        "sign_randomized": randomized_summary.get("mean_ic"),
    }
    verdict = placebo_verdict(reference_ic, placebo_ics, floor=config.placeholder_ic_floor)
    verdict["future_shift_guard"] = {
        "rows": int(len(guard)),
        "violations": guard_violations,
        "guard_expected_violation": bool(guard_violations == len(guard)),
    }
    return verdict


# ── Full build ────────────────────────────────────────────────────────────────

def run_discovery(panel, targets, prices, actions, fundamentals, cik_by_ticker,
                  benchmark_prices, benchmark_actions, config=None,
                  keyes_variable_frame=None, modern_signals=None, historical_signals=None):
    """Run the full discovery pipeline and return the result bundle."""
    config = config or DiscoveryConfig()

    feature_frame, panel_summary = build_feature_panel(
        panel, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions, restrict_to_development=True,
    )
    labelled = attach_targets(feature_frame, targets)
    development = development_rows(labelled)
    if development.empty:
        raise DiscoveryBuildError("no development observations are available for discovery")

    per_feature = {}
    for name in FEATURE_NAMES:
        per_feature[name] = feature_evidence(development, name, config)

    rank_matrix, ic_matrix, clusters = build_redundancy(development, per_feature, config)

    # FDR across every candidate that produced a dependence-aware p-value.
    p_values = {}
    for name in FEATURE_NAMES:
        p_value = per_feature[name]["inference"]["hac"].get("p_value")
        p_values[name] = 1.0 if p_value is None else float(p_value)
    fdr_results, fdr_summary = benjamini_hochberg(p_values, alpha=config.fdr_alpha)
    fdr_by_name = {record.feature: record for record in fdr_results}
    for name in FEATURE_NAMES:
        per_feature[name]["fdr"] = fdr_by_name[name].to_dict()

    # Reference IC for the placebo verdict: the strongest real |mean IC|.
    real_ics = [abs(per_feature[name]["ic"].get("mean_ic") or 0.0) for name in FEATURE_NAMES]
    reference_ic = max(real_ics) if real_ics else 0.0
    placebo = run_placebos(development, config, reference_ic)

    scorecards = [
        build_scorecard(name, CATALOG_BY_NAME[name], per_feature[name], config=config)
        for name in FEATURE_NAMES
    ]
    counts = classification_counts(scorecards)

    discovery_rank = {name: per_feature[name]["ic"].get("mean_ic") for name in FEATURE_NAMES}
    benchmark = _keyes.keyes_benchmark_payload(
        keyes_variable_frame, targets, modern_signals, historical_signals, config,
        discovery_rank=discovery_rank,
    ) if keyes_variable_frame is not None else {"note": "no Keyes variable frame supplied"}

    return {
        "config": config,
        "panel_summary": panel_summary,
        "development_rows": int(len(development)),
        "feature_frame": feature_frame,
        "development": development,
        "per_feature": per_feature,
        "rank_matrix": rank_matrix,
        "ic_matrix": ic_matrix,
        "clusters": clusters,
        "fdr_results": [record.to_dict() for record in fdr_results],
        "fdr_summary": fdr_summary,
        "placebo": placebo,
        "scorecards": scorecards,
        "classification_counts": counts,
        "keyes": benchmark,
    }


def assemble_handoff(bundle, dataset_id, target_id, feature_set_id, experiment,
                     keyes_reference=None, notes=None):
    return build_handoff(
        bundle["scorecards"], bundle["clusters"], dataset_id, target_id, feature_set_id,
        experiment, keyes_reference=keyes_reference, notes=notes,
    )
