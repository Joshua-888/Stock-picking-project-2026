"""Apply-only model-native importance and deterministic tree-path attribution.

This module inspects the already-frozen WP8 ExtraTrees estimator and applies the
frozen preprocessor only. It never calls ``fit`` and never reads a target,
holdout-label, future-return, or target-observable value.

Attribution semantics (explicitly NOT SHAP and NOT causal):

* ``expected_value`` is the unweighted mean class-1 probability across every
  leaf prediction of the frozen forest. It is a stable expected-value proxy for
  the tree population, not an empirical train-set base rate and not a
  probability estimate.
* For one tree, the path from the root to the reached leaf is decomposed by the
  signed transition ``child_class1 - prior_class1`` at each split. The prior at
  the root is the forest ``expected_value``. Because the transitions telescope,
  the per-tree path sum is ``leaf_class1 - expected_value`` by construction.
* Contributions are divided by ``n_trees`` so the ensemble average plus
  ``expected_value`` reproduces ``raw_model_score``.
* The 26 transformed columns (base + ``__missing``) are collapsed to the 13
  frozen base predictors; every ``__missing`` contribution is added to its base
  feature, never dropped silently.

This is local model-reliance / score attribution only. It is not SHAP, not a
causal effect, and not an attribution of rank or calibrated probability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schemas

ATTRIBUTION_KIND = "deterministic_tree_path"
IMPORTANCE_KIND = "native_gini_decrease"
ATTRIBUTION_TOLERANCE = 1e-6

ATTRIBUTION_LIMITATIONS = [
    "local model-reliance / tree-path score attribution, NOT TreeSHAP and NOT a causal effect",
    "not an attribution of rank or calibrated probability",
    "the expected value is an unweighted leaf-prediction baseline, not an empirical base rate",
    "feature importance and attribution distributions are not significance tests",
]


class AttributionError(ValueError):
    """Raised when attribution inputs violate the frozen contract."""


@dataclass(frozen=True)
class LocalAttribution:
    contributions: Dict[str, float]
    expected_value: float
    raw_model_score: float
    residual: float
    tolerance: float


@dataclass(frozen=True)
class GlobalAttribution:
    by_feature: Dict[str, float]
    expected_value: float


def _require_champion(champion) -> None:
    schemas.validate_feature_order(champion.features)
    if champion.freeze_id != "freeze_e62eac30df40":
        raise AttributionError("unexpected frozen champion freeze_id %r" % champion.freeze_id)


def _frozen_estimator(champion):
    model_obj = champion._model_obj
    if not isinstance(model_obj, dict) or model_obj.get("kind") != "estimator":
        raise AttributionError("frozen model is not an estimator bundle")
    estimator = model_obj.get("estimator")
    if estimator is None:
        raise AttributionError("frozen model bundle has no estimator")
    if not hasattr(estimator, "estimators_"):
        raise AttributionError("frozen estimator is not a tree ensemble")
    return estimator


def _positive_class_index(estimator) -> int:
    classes = list(getattr(estimator, "classes_", []))
    if 1.0 in classes:
        return classes.index(1.0)
    if 1 in classes:
        return classes.index(1)
    return 1  # sklearn fallback: positive class occupies index 1


def _tree_leaf_probabilities(tree, positive_index: int) -> np.ndarray:
    values = tree.value.astype("float64", copy=False)
    flat = values[:, 0, :]  # (n_nodes, n_outputs, n_classes) -> (n_nodes, n_classes)
    totals = flat.sum(axis=1)
    probabilities = np.zeros(len(totals), dtype="float64")
    nonzero = totals > 0.0
    probabilities[nonzero] = flat[nonzero, positive_index] / totals[nonzero]
    return probabilities


def expected_value(champion, estimator=None) -> float:
    """Unweighted mean class-1 value across all frozen tree leaves."""
    _require_champion(champion)
    estimator = estimator if estimator is not None else _frozen_estimator(champion)
    positive_index = _positive_class_index(estimator)
    values: list[float] = []
    for tree_estimator in estimator.estimators_:
        tree = tree_estimator.tree_
        probabilities = _tree_leaf_probabilities(tree, positive_index)
        is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
        values.extend(probabilities[is_leaf].tolist())
    if not values:
        raise AttributionError("frozen forest contains no leaves")
    return float(np.mean(values))


def _transform_frame(champion, frame: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    missing = [name for name in schemas.FEATURE_ORDER if name not in frame.columns]
    if missing:
        raise AttributionError("attribution frame is missing feature(s): %s" % ", ".join(missing))
    ordered = frame.loc[:, list(schemas.FEATURE_ORDER)].copy()
    preprocessor_obj = champion._preprocessor_obj
    if not isinstance(preprocessor_obj, dict):
        raise AttributionError("frozen preprocessor bundle is not a dict")
    preprocessor = preprocessor_obj.get("preprocessor")
    fitted = preprocessor_obj.get("fitted")
    if preprocessor is None or fitted is None:
        raise AttributionError("frozen preprocessor bundle is incomplete; refusing to fit")
    transformed = preprocessor.transform(ordered, fitted)
    if list(transformed.columns) != list(schemas.TRANSFORMED_COLUMN_ORDER):
        raise AttributionError("transformed column order does not match the frozen dashboard contract")
    return transformed.to_numpy(dtype="float64"), transformed


def _raw_score(estimator, x: np.ndarray) -> np.ndarray:
    deterministic = copy.deepcopy(estimator)
    if hasattr(deterministic, "n_jobs"):
        deterministic.n_jobs = 1
    positive_index = _positive_class_index(estimator)
    return np.asarray(deterministic.predict_proba(x)[:, positive_index], dtype="float64")


def feature_importance(champion) -> Dict[str, float]:
    """Model-native Gini importances collapsed from 26 columns to 13 features.

    ``__missing`` importances are added back to their base feature. The returned
    values are shares and sum to 1 (within float tolerance).
    """
    _require_champion(champion)
    estimator = _frozen_estimator(champion)
    importances = np.asarray(estimator.feature_importances_, dtype="float64")
    if importances.shape != (26,):
        raise AttributionError("frozen feature_importances_ does not have 26 entries")
    collapsed = {}
    for offset, feature in enumerate(schemas.FEATURE_ORDER):
        base_index = 2 * offset
        missing_index = base_index + 1
        collapsed[feature] = float(importances[base_index] + importances[missing_index])
    total = float(sum(collapsed.values())) or 1.0
    return {name: float(value / total) for name, value in collapsed.items()}


def tree_path_local(champion, feature_frame: pd.DataFrame) -> LocalAttribution:
    """Attribute one ``feature_frame`` row to the 13 base features."""
    if isinstance(feature_frame, pd.DataFrame) and len(feature_frame) == 0:
        raise AttributionError("tree_path_local requires exactly one row")
    if isinstance(feature_frame, pd.DataFrame) and len(feature_frame) != 1:
        raise AttributionError("tree_path_local requires exactly one row")

    _require_champion(champion)
    estimator = _frozen_estimator(champion)
    x, _transformed = _transform_frame(champion, feature_frame)
    raw = float(_raw_score(estimator, x)[0])
    base = expected_value(champion, estimator=estimator)
    contributions = _path_contributions(estimator, x, base)
    local = _collapse_to_base(contributions.iloc[0])
    residual = float(raw - (base + sum(local.values())))
    if abs(residual) > ATTRIBUTION_TOLERANCE:
        raise AttributionError(
            "local attribution reconciliation failed: residual %.12g exceeds tolerance %.3g"
            % (residual, ATTRIBUTION_TOLERANCE)
        )
    return LocalAttribution(
        contributions=local,
        expected_value=base,
        raw_model_score=raw,
        residual=residual,
        tolerance=ATTRIBUTION_TOLERANCE,
    )


def attribute_local_rows(champion, feature_frame: pd.DataFrame, raw_scores=None) -> pd.DataFrame:
    """Deterministic tree-path attribution for every row in ``feature_frame``.

    Returns a 13-column DataFrame of local contributions where each row sums to
    ``raw_model_score - expected_value`` within documented tolerance.
    """
    _require_champion(champion)
    estimator = _frozen_estimator(champion)
    x, _transformed = _transform_frame(champion, feature_frame)
    raw = _raw_score(estimator, x)
    if raw_scores is not None:
        candidate = np.asarray(raw_scores, dtype="float64")
        if candidate.shape[0] != raw.shape[0]:
            raise AttributionError("raw_scores length does not match feature_frame")
        raw = candidate
    base = float(expected_value(champion, estimator=estimator))
    contributions = _path_contributions(estimator, x, base)
    frame = pd.DataFrame([_collapse_to_base(row) for row in contributions.itertuples(index=False)],
                         index=feature_frame.index)
    for column in schemas.FEATURE_ORDER:
        if column not in frame.columns:
            frame[column] = 0.0
    frame = frame.loc[:, list(schemas.FEATURE_ORDER)].copy()
    return frame


def global_attribution_by_feature(champion, feature_frame: pd.DataFrame,
                                  raw_scores=None) -> Dict[str, float]:
    """Mean absolute local attribution over a deterministic label-free sample."""
    _require_champion(champion)
    if feature_frame is None or len(feature_frame) == 0:
        raise AttributionError("global attribution requires a non-empty label-free sample")
    estimator = _frozen_estimator(champion)
    x, _transformed = _transform_frame(champion, feature_frame)
    raw = _raw_score(estimator, x)
    if raw_scores is not None:
        candidate = np.asarray(raw_scores, dtype="float64")
        if candidate.shape[0] != raw.shape[0]:
            raise AttributionError("raw_scores length does not match feature_frame")
        raw = candidate
    base = float(expected_value(champion, estimator=estimator))
    contributions = _path_contributions(estimator, x, base)
    collapsed = pd.DataFrame(
        [_collapse_to_base(row) for row in contributions.itertuples(index=False)],
        index=feature_frame.index,
    )
    matrix = collapsed.loc[:, list(schemas.FEATURE_ORDER)].to_numpy(dtype="float64")
    finite = np.where(np.isfinite(matrix), matrix, 0.0)
    return {feature: float(np.mean(np.abs(finite[:, offset])))
            for offset, feature in enumerate(schemas.FEATURE_ORDER)}


def family_attribution(by_feature: Dict[str, float], by_family: bool = True) -> Dict[str, float]:
    """Collapse per-feature attribution values into economic families."""
    family_sums = {family: 0.0 for family in list(schemas.FAMILY_ORDER)}
    for feature, value in by_feature.items():
        family_sums[schemas.FEATURE_FAMILY_MAP[feature]] += float(value)
    return {family: float(family_sums[family]) for family in schemas.FAMILY_ORDER}


def contributor_shape(attribution_by_feature: Dict[str, float]) -> Tuple[list, list]:
    """Split feature contributions into top positive and top negative lists."""
    positive = sorted(
        ((feature, float(value)) for feature, value in attribution_by_feature.items() if value > 0),
        key=lambda item: item[1], reverse=True,
    )
    negative = sorted(
        ((feature, float(value)) for feature, value in attribution_by_feature.items() if value < 0),
        key=lambda item: item[1],
    )
    return [{"feature": name, "attribution": value} for name, value in positive], \
        [{"feature": name, "attribution": value} for name, value in negative]


def _path_contributions(estimator, x: np.ndarray, base: float) -> pd.DataFrame:
    """Vectorised tree-path contribution decomposition for all rows.

    Every split's transition ``child_probability - parent_probability`` is
    assigned to the split's transformed column. The child used is the actual
    branch the sample traverses (from ``tree.apply``), so along each path the
    transition sum telescopes to ``leaf_probability - expected_value``.
    """
    contributions = np.zeros((x.shape[0], 26), dtype="float64")
    n_trees = float(len(estimator.estimators_))
    positive_index = _positive_class_index(estimator)
    x_float32 = x.astype(np.float32, copy=False)
    for tree_estimator in estimator.estimators_:
        tree = tree_estimator.tree_
        leaf_index = tree.apply(x_float32)
        if tree.n_features != 26:
            raise AttributionError("frozen tree does not have 26 features")
        node_prob = compute_tree_probabilities(tree, base, positive_index)
        nodes = tree_index_paths(tree, leaf_index)
        for sample_index, path in enumerate(nodes):
            for position in range(len(path) - 1):
                node_id = int(path[position])
                next_node_id = int(path[position + 1])
                feature = int(tree.feature[node_id])
                if feature < 0:
                    continue
                contributions[sample_index, feature] += (
                    node_prob[next_node_id] - node_prob[node_id]
                ) / n_trees
    return pd.DataFrame(contributions, columns=list(schemas.TRANSFORMED_COLUMN_ORDER))


def compute_tree_probabilities(tree, root_prior: float, positive_index: int = 1) -> np.ndarray:
    """Assign every node a class-1 probability using leaves and ``root_prior``.

    Leaf nodes carry their observed positive-class frequency from ``tree_.value``.
    Each internal node's probability is the mean of its children's probabilities.
    The root is pinned to ``root_prior`` so the ensemble-level expected value is
    the common path baseline. Because each split contributes
    ``child_prob - parent_prob``, every root-to-leaf transition sum telescopes to
    ``leaf_prob - root_prior`` and is additive across splits.
    """
    values = tree.value.astype("float64", copy=False)[:, 0, :]
    totals = values.sum(axis=1)
    probabilities = np.zeros(len(totals), dtype="float64")
    nonzero = totals > 0.0
    probabilities[nonzero] = values[nonzero, positive_index] / totals[nonzero]
    probabilities[~nonzero] = root_prior

    children_left = tree.children_left
    children_right = tree.children_right
    # Build a post-order by DFS and propagate child values up to parents.
    sequence = []
    stack = [0]
    while stack:
        node_id = int(stack.pop())
        sequence.append(node_id)
        left, right = int(children_left[node_id]), int(children_right[node_id])
        if right >= 0:
            stack.append(right)
        if left >= 0:
            stack.append(left)
    for node_id in reversed(sequence):
        left, right = children_left[node_id], children_right[node_id]
        if left >= 0 and right >= 0:
            probabilities[node_id] = (probabilities[left] + probabilities[right]) / 2.0
    probabilities[0] = root_prior
    return probabilities


def tree_index_paths(tree, leaf_index: np.ndarray) -> list:
    """Return the root-to-leaf node-id path for every sample's leaf index."""
    children_left = tree.children_left
    children_right = tree.children_right
    parent = np.full(tree.node_count, -1, dtype=np.int64)
    for node_id in range(tree.node_count):
        left, right = children_left[node_id], children_right[node_id]
        if left >= 0:
            parent[left] = node_id
        if right >= 0:
            parent[right] = node_id

    paths = []
    for leaf_node in np.asarray(leaf_index, dtype=np.int64):
        single_path = []
        current = int(leaf_node)
        while current >= 0:
            single_path.append(current)
            current = int(parent[current])
        single_path.reverse()
        paths.append(single_path)
    return paths


def _collapse_to_base(row) -> Dict[str, float]:
    if hasattr(row, "_asdict"):
        row = row._asdict()
    elif hasattr(row, "to_dict"):
        row = row.to_dict()
    collapsed = {feature: 0.0 for feature in schemas.FEATURE_ORDER}
    for column, value in row.items():
        column = str(column)
        if column.endswith("__missing"):
            collapsed[column[:-9]] += float(value)
        elif column in collapsed:
            collapsed[column] += float(value)
    return collapsed
