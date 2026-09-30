"""WP6 feature strategies A-F.

Every strategy is a FUNCTION OF THE TRAINING WINDOW ONLY. No strategy reads a
validation or holdout row, and no strategy uses the full sample. This is what
makes a fold's selected feature set an honest train-window decision; the WP5
categories are carried as descriptive context and are NEVER used as a permanent
full-sample feature set.

Strategies
----------
* A ``all_eligible``          - all 20 eligible features.
* B ``coverage_qualified``    - eligible features whose TRAIN-window coverage
  clears the frozen threshold.
* C ``regularization_only``   - all eligible features (regularisation, not
  selection, controls complexity).
* D ``correlation_constrained`` - all eligible features minus those dropped by a
  TRAIN-window complete-linkage redundancy filter at |rank corr| > threshold.
* E ``ic_top_k``              - the top-k eligible features by |TRAIN mean monthly
  rank IC|.
* F ``economic_family_representatives`` - one representative per economic family,
  chosen by the HIGHEST TRAIN COVERAGE (never by target performance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.cross_sectional import cross_sectional_ic_series, spearman
from .contract import DEFAULT_CONFIG, economic_family, model_feature_universe

STRATEGIES = (
    "A_all_eligible",
    "B_coverage_qualified",
    "C_regularization_only",
    "D_correlation_constrained",
    "E_ic_top_k",
    "F_economic_family_representatives",
)


class FeatureStrategyError(RuntimeError):
    """Raised when a feature strategy cannot be applied honestly."""


def _coverage(frame, name):
    if name not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[name], errors="coerce").notna().mean()) if len(frame) else 0.0


def _train_rank_correlation(frame, feature_a, feature_b):
    if feature_a not in frame.columns or feature_b not in frame.columns:
        return None
    subset = frame.loc[:, [feature_a, feature_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 2:
        return None
    return spearman(subset[feature_a].to_numpy(), subset[feature_b].to_numpy())


def select_features(strategy, train_frame, config=None, target=None):
    """Return the ordered feature list a strategy chooses on a TRAIN frame.

    ``train_frame`` is the fold's training slice ONLY. Returns
    ``(features, evidence)`` where ``evidence`` records the factual basis of the
    choice so selection stability can be measured across folds.
    """
    config = config or DEFAULT_CONFIG
    target = target or config.regression_target
    eligible = list(model_feature_universe())

    if strategy == "A_all_eligible":
        return eligible, {"basis": "all_eligible", "n_features": len(eligible)}

    if strategy == "C_regularization_only":
        return eligible, {"basis": "regularization_only", "n_features": len(eligible)}

    if strategy == "B_coverage_qualified":
        coverage = {name: _coverage(train_frame, name) for name in eligible}
        kept = [name for name in eligible if coverage[name] >= config.coverage_threshold]
        dropped = {name: coverage[name] for name in eligible if coverage[name] < config.coverage_threshold}
        if not kept:
            raise FeatureStrategyError("coverage strategy kept no features in this train window")
        return kept, {"basis": "train_coverage>=%.2f" % config.coverage_threshold,
                      "coverage": coverage, "dropped": dropped}

    if strategy == "D_correlation_constrained":
        # Complete-linkage-style greedy: keep a feature only if it is not highly
        # correlated with an ALREADY KEPT feature. Order is the frozen declaration
        # order, so the result is deterministic; correlations use TRAIN only.
        kept, dropped = [], {}
        for name in eligible:
            conflicts = []
            for peer in kept:
                correlation = _train_rank_correlation(train_frame, name, peer)
                if correlation is not None and abs(correlation) > config.correlation_threshold:
                    conflicts.append({"peer": peer, "rank_corr": correlation})
            if conflicts or not _has_train_values(train_frame, name):
                dropped[name] = conflicts if conflicts else [{"reason": "no_train_values"}]
                continue
            kept.append(name)
        if not kept:
            raise FeatureStrategyError("correlation strategy kept no features in this train window")
        return kept, {"basis": "train_rank_corr<=%.2f" % config.correlation_threshold, "dropped": dropped}

    if strategy == "E_ic_top_k":
        scored = []
        for name in eligible:
            if not _has_train_values(train_frame, name):
                continue
            series, _diag = cross_sectional_ic_series(
                train_frame, name, target=target,
                min_observations=config.min_cross_section_obs,
            )
            mean_ic = float(series["rank_ic"].mean()) if len(series) else None
            if mean_ic is not None and np.isfinite(mean_ic):
                scored.append((name, mean_ic))
        scored.sort(key=lambda item: (-abs(item[1]), item[0]))
        top = [name for name, _value in scored[: int(config.ic_top_k)]]
        if not top:
            raise FeatureStrategyError("IC strategy found no scorable feature in this train window")
        return top, {"basis": "train_ic_top_%d" % config.ic_top_k,
                     "train_mean_ic": {name: value for name, value in scored}}

    if strategy == "F_economic_family_representatives":
        families = {}
        for name in eligible:
            families.setdefault(economic_family(name), []).append(name)
        representatives, evidence = [], {}
        for family in sorted(families):
            members = families[family]
            # Representative = highest TRAIN coverage (NOT target performance).
            ranked = sorted(members, key=lambda name: (-_coverage(train_frame, name), name))
            chosen = ranked[0]
            representatives.append(chosen)
            evidence[family] = {"chosen": chosen, "members": members,
                                "train_coverage": {name: _coverage(train_frame, name) for name in members}}
        return representatives, {"basis": "train_coverage_family_representative", "families": evidence}

    raise FeatureStrategyError("unknown feature strategy %r" % (strategy,))


def _has_train_values(frame, name):
    return name in frame.columns and bool(pd.to_numeric(frame[name], errors="coerce").notna().any())


def selection_stability(per_fold_selections):
    """Measure how often each feature is selected across folds (factual).

    ``per_fold_selections`` maps fold -> ordered feature list. Returns per-feature
    selection counts/fractions plus the union and intersection.
    """
    folds = sorted(per_fold_selections)
    if not folds:
        return {"folds": 0, "features": {}, "union": [], "intersection": []}
    counts = {}
    for fold in folds:
        for name in set(per_fold_selections[fold]):
            counts[name] = counts.get(name, 0) + 1
    records = {
        name: {"folds_selected": count, "folds": len(folds),
               "selection_fraction": count / len(folds)}
        for name, count in sorted(counts.items())
    }
    sets = [set(per_fold_selections[fold]) for fold in folds]
    union = sorted(set().union(*sets))
    intersection = sorted(set.intersection(*sets)) if sets else []
    return {"folds": len(folds), "features": records, "union": union,
            "intersection": intersection}
