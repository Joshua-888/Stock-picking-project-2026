"""WP6 model registry with SMALL FROZEN grids (no adaptive search, no Optuna).

Every model family and every hyper-parameter value is declared here before any
result is read. The grids are deliberately tiny so the experiment ledger remains
an honest, complete enumeration of what was tested rather than the survivor of an
adaptive search. ``scikit-learn`` is imported lazily so the module can be
inspected without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ModelRegistryError(RuntimeError):
    """Raised when a model request is ill-formed."""


@dataclass(frozen=True)
class ModelSpec:
    """One catalogue entry with a frozen, predeclared grid."""

    name: str
    family: str            # regression | classification | nonlinear
    kind: str              # baseline | linear | forest | boosting
    params: tuple          # tuple of dicts (the frozen grid)
    preprocessing: tuple   # ordered preprocessing spec keys
    note: str = ""

    def grid(self):
        return tuple(dict(entry) for entry in self.params)


# Preprocessing profiles (declared per family; fitted on TRAIN only).
RANK_SPEC = {"impute": "median", "indicator": True, "clip": "percentile",
             "scale": "robust", "cross_sectional_rank": True}
VALUE_SPEC = {"impute": "median", "indicator": True, "clip": "percentile",
              "scale": "robust", "cross_sectional_rank": False}

REGRESSION_MODELS = (
    ModelSpec("baseline_mean", "regression", "baseline", ({},),
              ("VALUE_SPEC",), "train-mean predictor (loss baseline)"),
    ModelSpec("baseline_ew_composite", "regression", "baseline", ({},),
              ("RANK_SPEC",), "equal-weight within-date rank composite (rank-metric baseline)"),
    ModelSpec("ols", "regression", "linear", ({},), ("VALUE_SPEC",), "ordinary least squares"),
    ModelSpec("ridge", "regression", "linear",
              tuple({"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)), ("VALUE_SPEC",), "ridge"),
    ModelSpec("lasso", "regression", "linear",
              tuple({"alpha": value} for value in (0.0005, 0.001, 0.005, 0.01)), ("VALUE_SPEC",), "lasso"),
    ModelSpec("elasticnet", "regression", "linear",
              tuple({"alpha": alpha, "l1_ratio": ratio}
                    for alpha in (0.001, 0.01) for ratio in (0.25, 0.75)),
              ("VALUE_SPEC",), "elastic net"),
)

CLASSIFICATION_MODELS = (
    ModelSpec("baseline_base_rate", "classification", "baseline", ({},),
              ("VALUE_SPEC",), "train base-rate predictor (loss baseline)"),
    ModelSpec("logistic", "classification", "linear",
              tuple({"C": value} for value in (0.01, 0.1, 1.0, 10.0)),
              ("VALUE_SPEC",), "logistic regression"),
)

NONLINEAR_MODELS = (
    ModelSpec("random_forest", "nonlinear", "forest",
              tuple({"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}
                    for depth in (4, 8) for leaf in (20, 50)),
              ("VALUE_SPEC",), "random forest"),
    ModelSpec("extra_trees", "nonlinear", "forest",
              tuple({"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}
                    for depth in (4, 8) for leaf in (20, 50)),
              ("VALUE_SPEC",), "extra trees"),
    ModelSpec("hist_gradient_boosting", "nonlinear", "boosting",
              tuple({"max_depth": depth, "min_samples_leaf": leaf,
                     "learning_rate": rate, "max_iter": 300}
                    for depth in (3, 6) for leaf in (20, 50) for rate in (0.05, 0.1)),
              ("VALUE_SPEC",), "histogram gradient boosting"),
)

ALL_MODELS = REGRESSION_MODELS + CLASSIFICATION_MODELS + NONLINEAR_MODELS
MODELS_BY_NAME = {spec.name: spec for spec in ALL_MODELS}


class MeanRegressor:
    """Constant train-mean predictor (sklearn-compatible subset)."""

    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype="float64")


class CompositeRankRegressor:
    """Equal-weight mean of the within-date percentile ranks of the features."""

    def __init__(self, columns=None):
        self.columns = list(columns or [])

    def fit(self, X, y):
        return self

    def predict(self, X):
        frame = X[self.columns] if self.columns else X
        return frame.mean(axis=1).to_numpy(dtype="float64")


def make_estimator(spec, params=None, seed=20260930, columns=None):
    """Instantiate one estimator for a spec/params pair (deterministic seed)."""
    params = dict(params or {})
    if spec.kind == "baseline" and spec.name == "baseline_mean":
        return MeanRegressor()
    if spec.kind == "baseline" and spec.name == "baseline_ew_composite":
        return CompositeRankRegressor(columns=columns)
    if spec.kind == "baseline" and spec.name == "baseline_base_rate":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="prior")
    if spec.name == "ols":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if spec.name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params.get("alpha", 1.0), random_state=None)
    if spec.name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=params.get("alpha", 0.001), max_iter=5000, random_state=seed)
    if spec.name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=params.get("alpha", 0.001), l1_ratio=params.get("l1_ratio", 0.5),
                          max_iter=5000, random_state=seed)
    if spec.name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=params.get("C", 1.0), max_iter=2000, random_state=seed)
    if spec.name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=seed, n_jobs=-1,
        )
    if spec.name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=seed, n_jobs=-1,
        )
    if spec.name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=int(params.get("max_depth", 3)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 300)),
            random_state=seed,
        )
    raise ModelRegistryError("unknown model %r" % (spec.name,))


def classification_estimator(spec, params=None, seed=20260930, columns=None):
    """Build a CLASSIFIER for the given regression/nonlinear spec."""
    params = dict(params or {})
    if spec.name == "baseline_base_rate":
        return make_estimator(spec, params, seed, columns)
    if spec.name == "logistic":
        return make_estimator(spec, params, seed, columns)
    if spec.name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=seed, n_jobs=-1,
        )
    if spec.name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=seed, n_jobs=-1,
        )
    if spec.name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=int(params.get("max_depth", 3)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 300)),
            random_state=seed,
        )
    raise ModelRegistryError("no classifier available for %r" % (spec.name,))


def regression_specs():
    return REGRESSION_MODELS


def classification_specs():
    return CLASSIFICATION_MODELS


def nonlinear_specs():
    return NONLINEAR_MODELS


def grid_size(spec):
    return len(spec.grid())


def registry_payload():
    """Deterministic registry payload used in the experiment binding."""
    return {
        spec.name: {
            "family": spec.family, "kind": spec.kind, "note": spec.note,
            "grid": [dict(entry) for entry in spec.grid()],
            "preprocessing": list(spec.preprocessing),
        }
        for spec in ALL_MODELS
    }
