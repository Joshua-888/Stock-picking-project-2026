"""Deterministic WP6 configuration identity for matched, predeclared references.

Corrective defect #1: a negative control must name a PREDECLARED matched real
configuration by DETERMINISTIC identity. Identity is a frozen tuple of the design
dimensions the corrective contract enumerates - model family, task, feature
strategy, preprocessing strategy, hyper-parameters, temporal folds, prediction
target, evaluation metric and aggregation. Only the placebo perturbation may
differ between a control and its matched real configuration, so the reference can
NEVER change when unrelated measured results change. No ``max(values, key=abs)``
or best-observed selection is possible through this module.
"""

from __future__ import annotations

import hashlib

from ..ids import canonical_json


def folds_signature(folds):
    """Canonical, deterministic signature of the temporal fold geometry."""
    return [
        {
            "fold": int(getattr(fold, "fold")),
            "model_date": str(getattr(fold, "model_date")),
            "train_rows": int(getattr(fold, "train_rows")),
            "validation_rows": int(getattr(fold, "validation_rows")),
            "validation_start": str(getattr(fold, "validation_start")),
            "validation_end": str(getattr(fold, "validation_end")),
        }
        for fold in folds
    ]


def configuration_identity(model_name, model_family, task, feature_strategy,
                           preprocessing_strategy, hyperparameters, folds, target,
                           metric, aggregation="pooled_monthly"):
    """Frozen identity payload (the perturbation is deliberately excluded)."""
    return {
        "model_name": str(model_name),
        "model_family": str(model_family),
        "task": str(task),
        "feature_strategy": str(feature_strategy),
        "preprocessing_strategy": str(preprocessing_strategy),
        "hyperparameters": {str(key): hyperparameters[key] for key in sorted(hyperparameters)},
        "folds": folds_signature(folds),
        "target": str(target),
        "metric": str(metric),
        "aggregation": str(aggregation),
    }


def configuration_id(model_name, model_family, task, feature_strategy,
                     preprocessing_strategy, hyperparameters, folds, target,
                     metric, aggregation="pooled_monthly", perturbation="none"):
    """Deterministic id of a configuration (and of a perturbed control variant).

    The matched real configuration and its placebo share the SAME identity and
    differ only in ``perturbation`` (``none`` for the real model).
    """
    identity = configuration_identity(
        model_name, model_family, task, feature_strategy, preprocessing_strategy,
        hyperparameters, folds, target, metric, aggregation=aggregation,
    )
    payload = {"identity": identity, "perturbation": str(perturbation)}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "modelcfg_" + digest[:12]
