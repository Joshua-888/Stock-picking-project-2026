"""WP5 multidimensional scorecard and research classification.

Every candidate collects its factual evidence into ONE scorecard row and receives
a research category. The categories are decision aids for WP6, NOT a ranking, and
there is NO single arbitrary weighted score: a reader can see exactly which facts
drove the label.

Category precedence (deterministic, evaluated top to bottom):

* ``TEMPORALLY_UNSAFE``  - a point-in-time or holdout violation was detected;
* ``LOW_COVERAGE``       - too few usable observations to research;
* ``NO_CLEAR_SIGNAL``    - no month cleared the minimum paired-observation bar;
* ``REDUNDANT``          - statistically inseparable from a stronger peer;
* ``ROBUST_CANDIDATE``   - coherent, stable, dependence-aware significance;
* ``PROMISING_BUT_UNSTABLE`` - real signal but unstable across time;
* ``DIRECTION_UNSTABLE`` - mean and median disagree in sign, or sign flips;
* ``DEFER``              - not enough evidence to classify either way.
"""

from __future__ import annotations

from .catalog import CLASSIFICATIONS, DiscoveryConfig


class ScorecardError(ValueError):
    """Raised when a scorecard request is ill-formed."""


def _consistent_direction(mean_ic, median_ic):
    if mean_ic is None or median_ic is None:
        return None
    if mean_ic == 0.0 or median_ic == 0.0:
        return None
    return (mean_ic > 0) == (median_ic > 0)


def classify_candidate(evidence, config=None):
    """Return the research category for one candidate's evidence bundle."""
    config = config or DiscoveryConfig()
    quality = evidence.get("quality", {})
    ic = evidence.get("ic", {})
    inference = evidence.get("inference", {})
    fdr = evidence.get("fdr", {})
    quantiles = evidence.get("quantiles", {})
    stability = evidence.get("stability", {})
    redundancy = evidence.get("redundancy", {})

    reasons = []

    # 1. Temporal safety dominates everything.
    if quality.get("temporal_safety") == "VIOLATION" or quality.get("status") == "TEMPORALLY_UNSAFE":
        return "TEMPORALLY_UNSAFE", ["availability after the prediction instant"]

    months = ic.get("months") or 0
    coverage = quality.get("coverage")

    # 2. Coverage floor.
    if not coverage or coverage < config.low_coverage_threshold:
        reasons.append("coverage %.3f" % (coverage or 0.0))
        return "LOW_COVERAGE", reasons

    # 3. No evaluable month at all.
    if months == 0 or ic.get("mean_ic") is None:
        return "NO_CLEAR_SIGNAL", ["no month cleared the minimum paired-observation bar"]

    mean_ic = ic.get("mean_ic")
    median_ic = ic.get("median_ic")
    hac = inference.get("hac", {})
    hac_significant = hac.get("p_value") is not None and hac.get("p_value") <= config.fdr_alpha
    fdr_rejected = bool(fdr.get("rejected"))
    direction_ok = _consistent_direction(mean_ic, median_ic)
    stable = bool(stability.get("direction_consistent")) and bool(stability.get("sufficient_months"))
    short_period_only = bool(stability.get("short_period_only"))
    sign_flip = bool(stability.get("sign_flip"))
    spread = quantiles.get("spread")
    monotonic = quantiles.get("monotonic")

    # 4. Redundancy: descriptively inseparable from another candidate.
    #    The flag is group MEMBERSHIP only (complete-linkage similarity); it is
    #    deliberately NOT driven by full-sample outcome performance, and no
    #    cluster winner is elected on the full sample.
    if redundancy.get("redundant"):
        return "REDUNDANT", ["statistically inseparable from another candidate in a complete-linkage similarity group"]

    # 5. Direction instability. The hypothesized direction is informational only;
    #    what matters for a usable signal is a coherent average and median sign.
    if direction_ok is False or sign_flip:
        return "DIRECTION_UNSTABLE", ["mean/median disagree or a period sign flip was observed"]

    # 6. Robust candidate: coherent, stable, significant after correction.
    if (hac_significant and fdr_rejected and stable and not short_period_only
            and (monotonic is True or (spread is not None and spread != 0.0))):
        return "ROBUST_CANDIDATE", ["HAC-significant, FDR-surviving, stable across time"]

    # 7. Real but unstable: has a signal yet fails a stability requirement.
    if hac_significant or fdr_rejected:
        return "PROMISING_BUT_UNSTABLE", ["signal present but failed a stability requirement"]

    # 8. Otherwise there is not enough evidence.
    return "DEFER", ["no dependence-aware significance; insufficient evidence"]


def build_scorecard(feature, spec, evidence, config=None):
    """Assemble the full multidimensional scorecard row for one candidate."""
    config = config or DiscoveryConfig()
    category, reasons = classify_candidate(evidence, config=config)
    quality = evidence.get("quality", {})
    ic = evidence.get("ic", {})
    inference = evidence.get("inference", {})
    fdr = evidence.get("fdr", {})
    quantiles = evidence.get("quantiles", {})
    stability = evidence.get("stability", {})
    missingness = evidence.get("missingness", {})

    return {
        "feature": feature,
        "category": spec.category,
        "cluster_family": spec.cluster_family,
        "hypothesized_direction": spec.hypothesized_direction,
        "is_keyes": bool(spec.is_keyes),
        "coverage": quality.get("coverage"),
        "pit_status": spec.pit_status,
        "quality_status": quality.get("status"),
        "months": ic.get("months"),
        "mean_ic": ic.get("mean_ic"),
        "median_ic": ic.get("median_ic"),
        "ic_std": ic.get("ic_std"),
        "icir": ic.get("icir"),
        "positive_month_share": ic.get("positive_month_share"),
        "hac_t_stat": inference.get("hac", {}).get("t_stat"),
        "hac_p_value": inference.get("hac", {}).get("p_value"),
        "bootstrap_p_value": inference.get("bootstrap", {}).get("p_value"),
        "fdr_q_value": fdr.get("q_value"),
        "fdr_rejected": bool(fdr.get("rejected")),
        "quantile_spread": quantiles.get("spread"),
        "outperform_spread": quantiles.get("outperform_spread"),
        "monotonic": quantiles.get("monotonic"),
        "direction_consistent": stability.get("direction_consistent"),
        "short_period_only": stability.get("short_period_only"),
        "sign_flip": stability.get("sign_flip"),
        "coverage_drift": stability.get("coverage_drift"),
        "missingness_effect": missingness.get("present_minus_missing_mean"),
        "redundancy_cluster": evidence.get("redundancy", {}).get("cluster"),
        "classification": category,
        "classification_reasons": reasons,
        "limitations": evidence.get("limitations", []),
    }


def classification_counts(scorecards):
    """Count candidates per research category (all categories present)."""
    counts = {name: 0 for name in CLASSIFICATIONS}
    for row in scorecards:
        category = row.get("classification")
        if category in counts:
            counts[category] += 1
    return counts
