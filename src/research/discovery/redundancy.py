"""WP5 redundancy analysis: cross-sectional rank correlation and IC similarity.

Candidates are grouped into redundancy GROUPS by two independent, BEFORE-THE-OUTCOME
measures:

* the median cross-sectional rank correlation between two features across dates;
* the correlation of their monthly IC series.

The grouping is a DESCRIPTIVE COMPLETE-LINKAGE agglomeration: two groups may only
merge when EVERY cross-group pair is linked at the configured threshold. Complete
linkage is deliberately chosen over single linkage because single linkage chains:
one thin high-correlation edge can fuse unrelated features into a single giant
cluster (the defect this module previously exhibited). Requiring an all-pairs link
is strictly stronger and keeps a group an actual similarity clique.

Groups are NEVER used to elect a single "winner". Choosing one representative per
group on the FULL sample would itself be a target-driven selection decision, which
WP6/WP7 must make inside training windows. This module reports groups and factual
similarity statistics and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cross_sectional import spearman


class RedundancyError(ValueError):
    """Raised when a redundancy request is ill-formed."""


def cross_sectional_rank_correlation(frame, feature_a, feature_b, asof_col="feature_asof"):
    """Median per-date Spearman correlation between two features."""
    if feature_a not in frame.columns or feature_b not in frame.columns:
        return None
    working = frame.loc[:, [asof_col, feature_a, feature_b]].copy()
    working[feature_a] = pd.to_numeric(working[feature_a], errors="coerce")
    working[feature_b] = pd.to_numeric(working[feature_b], errors="coerce")
    working = working.dropna(subset=[asof_col, feature_a, feature_b])
    correlations = []
    for _asof, chunk in working.groupby(asof_col, sort=True):
        if len(chunk) < 3:
            continue
        value = spearman(chunk[feature_a].to_numpy(), chunk[feature_b].to_numpy())
        if value is not None and np.isfinite(value):
            correlations.append(value)
    if not correlations:
        return None
    return float(np.median(correlations))


def rank_correlation_matrix(frame, features, asof_col="feature_asof"):
    """Symmetric median rank-correlation matrix for the candidate features."""
    matrix = pd.DataFrame(np.nan, index=list(features), columns=list(features), dtype="float64")
    for i, feature_a in enumerate(features):
        matrix.iloc[i, i] = 1.0
        for j in range(i + 1, len(features)):
            feature_b = features[j]
            value = cross_sectional_rank_correlation(frame, feature_a, feature_b, asof_col=asof_col)
            if value is None:
                continue
            matrix.iloc[i, j] = value
            matrix.iloc[j, i] = value
    return matrix


def median_rank_correlation_matrix(frame, features, asof_col="feature_asof", min_observations=10):
    """Vectorised median per-date rank-correlation matrix.

    Ranks are computed once per date for every feature and correlated per date;
    the reported entry is the MEDIAN across dates of the per-date Spearman
    correlation. This is far cheaper than a pairwise loop while computing the
    same quantity.
    """
    features = list(features)
    matrix = pd.DataFrame(np.nan, index=features, columns=features, dtype="float64")
    for feature in features:
        matrix.loc[feature, feature] = 1.0
    if not features or asof_col not in frame.columns:
        return matrix
    present = [feature for feature in features if feature in frame.columns]
    if not present:
        return matrix
    working = frame.loc[:, [asof_col] + present].copy()
    for feature in present:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    ranked = working[[asof_col] + present].copy()
    ranked[present] = working.groupby(asof_col)[present].rank()
    per_date = []
    for _asof, chunk in ranked.groupby(asof_col, sort=True):
        if len(chunk) < int(min_observations):
            continue
        per_date.append(chunk[present].corr(min_periods=int(min_observations)))
    if not per_date:
        return matrix
    stacked = pd.concat(per_date, keys=range(len(per_date)))
    medians = stacked.groupby(level=1).median()
    for feature in present:
        for other in present:
            value = medians.loc[feature, other] if feature in medians.index and other in medians.columns else np.nan
            matrix.loc[feature, other] = value
    return matrix


def ic_series_correlation(ic_series, features):
    """Correlation between monthly IC series; ``None`` where unavailable."""
    aligned = {}
    for feature in features:
        series = ic_series.get(feature)
        if series is not None and len(series):
            aligned[feature] = series.set_index("feature_asof")["rank_ic"]
    if not aligned:
        return pd.DataFrame(index=list(features), columns=list(features), dtype="float64")
    combined = pd.DataFrame(aligned)
    return combined.corr(method="pearson", min_periods=3)


def _matrix_value(matrix, feature_a, feature_b):
    if matrix is None:
        return None
    try:
        value = matrix.loc[feature_a, feature_b]
    except (KeyError, IndexError):
        return None
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def pair_similarity(rank_matrix, ic_matrix, feature_a, feature_b):
    """|correlation| between two features: the larger of the two available measures.

    Two features are similar when EITHER their cross-sectional rank correlation OR
    their monthly IC-series correlation is high in absolute magnitude (a perfect
    negative correlation is still redundancy: one is a sign flip of the other).
    Returns ``None`` when neither measure is available.
    """
    values = []
    for matrix in (rank_matrix, ic_matrix):
        value = _matrix_value(matrix, feature_a, feature_b)
        if value is not None:
            values.append(abs(value))
    if not values:
        return None
    return float(max(values))


def _link_flags(features, rank_matrix, ic_matrix, rank_threshold, ic_threshold):
    """Boolean all-pairs link map: a pair is linked when either measure is high."""
    linked = {}
    for i, feature_a in enumerate(features):
        for feature_b in features[i + 1:]:
            flag = False
            rank_value = _matrix_value(rank_matrix, feature_a, feature_b)
            if rank_value is not None and abs(rank_value) >= float(rank_threshold):
                flag = True
            if not flag:
                ic_value = _matrix_value(ic_matrix, feature_a, feature_b)
                if ic_value is not None and abs(ic_value) >= float(ic_threshold):
                    flag = True
            linked[(feature_a, feature_b)] = flag
            linked[(feature_b, feature_a)] = flag
    return linked


def _all_linked(cluster_a, cluster_b, linked):
    for left in cluster_a:
        for right in cluster_b:
            if not linked.get((left, right), False):
                return False
    return True


def redundancy_clusters(features, rank_matrix=None, ic_matrix=None,
                        rank_threshold=0.7, ic_threshold=0.7, families=None):
    """Group candidates with COMPLETE-LINKAGE agglomeration at a strict threshold.

    Two groups merge only when EVERY cross-group pair is linked (high rank
    correlation OR high IC-series correlation). This is deliberately stricter than
    single linkage: it cannot chain, so one spurious edge cannot fuse the whole
    universe into a single cluster. ``families`` (economic family per feature) is
    carried as descriptive evidence only and never forces or blocks a merge.
    """
    features = sorted(set(features))
    families = families or {}
    linked = _link_flags(features, rank_matrix, ic_matrix, rank_threshold, ic_threshold)

    clusters = [[feature] for feature in features]
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _all_linked(clusters[i], clusters[j], linked):
                    clusters[i] = sorted(clusters[i] + clusters[j])
                    del clusters[j]
                    changed = True
                    break
            if changed:
                break

    # Deterministic cluster ids in a stable order (largest first, then alphabetical).
    clusters.sort(key=lambda members: (-len(members), members))
    named = {}
    for index, members in enumerate(clusters):
        named["cluster_%02d" % index] = sorted(members)
    member_to_cluster = {}
    family_by_cluster = {}
    for name, members in named.items():
        for member in members:
            member_to_cluster[member] = name
        family_by_cluster[name] = sorted({families.get(member) for member in members if families.get(member)})
    return {
        "version": "v2_wp5_redundancy_v2",
        "linkage": "complete",
        "clusters": named,
        "member_to_cluster": member_to_cluster,
        "cluster_families": family_by_cluster,
        "rank_threshold": float(rank_threshold),
        "ic_threshold": float(ic_threshold),
        "note": (
            "descriptive complete-linkage groups; no cluster representative is elected "
            "on the full sample and no full-sample outcome performance is used"
        ),
    }
