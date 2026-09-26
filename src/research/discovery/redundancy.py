"""WP5 redundancy analysis: cross-sectional rank correlation and IC similarity.

Candidates are grouped into redundancy CLUSTERS by two independent measures:

* the median cross-sectional rank correlation between two features across dates;
* the correlation of their monthly IC series.

Clusters are descriptive. They are NEVER used to elect a single "winner":
choosing one representative per cluster on the full sample would itself be a
selection decision, which WP6/WP7 must make inside training windows.
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


def _union_find_groups(features, links):
    parent = {feature: feature for feature in features}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in links:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    groups = {}
    for feature in features:
        groups.setdefault(find(feature), []).append(feature)
    return [sorted(members) for members in groups.values()]


def redundancy_clusters(features, rank_matrix=None, ic_matrix=None,
                        rank_threshold=0.7, ic_threshold=0.7):
    """Group candidates linked by high rank correlation OR high IC similarity."""
    features = list(features)
    links = []
    for i, feature_a in enumerate(features):
        for j in range(i + 1, len(features)):
            feature_b = features[j]
            high = False
            if rank_matrix is not None:
                value = rank_matrix.loc[feature_a, feature_b]
                if pd.notna(value) and abs(float(value)) >= float(rank_threshold):
                    high = True
            if not high and ic_matrix is not None and feature_a in ic_matrix.index and feature_b in ic_matrix.columns:
                value = ic_matrix.loc[feature_a, feature_b]
                if pd.notna(value) and abs(float(value)) >= float(ic_threshold):
                    high = True
            if high:
                links.append((feature_a, feature_b))
    groups = _union_find_groups(features, links)
    # Deterministic cluster ids in a stable order (largest first, then alphabetical).
    groups.sort(key=lambda members: (-len(members), members))
    clusters = {}
    for index, members in enumerate(groups):
        clusters["cluster_%02d" % index] = members
    member_to_cluster = {}
    for name, members in clusters.items():
        for member in members:
            member_to_cluster[member] = name
    return {
        "version": "v2_wp5_redundancy_v1",
        "clusters": clusters,
        "member_to_cluster": member_to_cluster,
        "rank_threshold": float(rank_threshold),
        "ic_threshold": float(ic_threshold),
        "note": "clusters are descriptive; no cluster representative is elected on the full sample",
    }
