"""WP6 model-layer multiple-testing control (Benjamini-Hochberg FDR).

Corrective defect #6. Scanning hundreds of frozen configurations guarantees some
raw p-value below 0.05 by chance, so a raw p-value alone can NEVER support a
PROMISING category. Every configuration is assigned to a FROZEN hypothesis family
defined by its scientific question (regression predictive skill / classification
predictive skill) and its raw p-value is adjusted within that family. The full
tested set is retained; no family is shrunk to reduce the penalty.
"""

from __future__ import annotations

from ..discovery.multiple_testing import benjamini_hochberg


def family_fdr(family_id, scientific_question, metric, null, raw_p_by_config, alpha=0.05):
    """Apply BH-FDR within one frozen family; return the full family record."""
    raw_p_by_config = {str(key): float(value) for key, value in (raw_p_by_config or {}).items()}
    for key, value in raw_p_by_config.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError("raw p-value out of range for %s: %r" % (key, value))
    results, summary = benjamini_hochberg(dict(raw_p_by_config), alpha=alpha)
    q_values = {record.feature: record.q_value for record in results}
    rejected = {record.feature: record.rejected for record in results}
    return {
        "family_id": family_id,
        "scientific_question": scientific_question,
        "metric": metric,
        "null": null,
        "number_of_hypotheses": int(len(raw_p_by_config)),
        "raw_p_definition": "two-sided HAC (Newey-West) p-value on the monthly skill series",
        "adjustment_method": "benjamini_hochberg",
        "alpha": float(alpha),
        "included_config_ids": sorted(raw_p_by_config),
        "q_values": q_values,
        "fdr_rejected": rejected,
        "summary": summary,
    }


FROZEN_FAMILIES = (
    {
        "family_id": "regression_predictive_skill",
        "task": "regression",
        "scientific_question": "Do regression-task configurations show cross-sectional predictive skill beyond chance?",
        "metric": "rank_ic",
        "null": "H0: mean(IC_t) = 0",
    },
    {
        "family_id": "classification_predictive_skill",
        "task": "classification",
        "scientific_question": "Do classification-task configurations beat chance on ROC-AUC skill?",
        "metric": "auc",
        "null": "H0: mean(AUC_t - 0.5) = 0",
    },
)


def build_families(configuration_records, alpha=0.05):
    """Build the two frozen WP6 families from configuration ledger records."""
    families = {}
    for definition in FROZEN_FAMILIES:
        task = definition["task"]
        raw = {}
        for record in configuration_records:
            if record.get("task") != task:
                continue
            config_id = record.get("config_id")
            pooled = record.get("pooled") or {}
            hac = pooled.get("hac") or {}
            raw_p = hac.get("p_value")
            if config_id is None or raw_p is None:
                continue
            raw[config_id] = float(min(max(float(raw_p), 0.0), 1.0))
        families[definition["family_id"]] = family_fdr(
            definition["family_id"], definition["scientific_question"],
            definition["metric"], definition["null"], raw, alpha=alpha,
        )
    return families
