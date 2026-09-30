"""WP6 preprocessing with an explicit TRAIN-FIT / APPLY-FORWARD contract.

Every fitted quantity (imputation value, clip bounds, centring, scaling) is
computed from the TRAINING frame only and then applied unchanged to the training
and validation frames. There is NO global or full-sample transform and no
randomised step. Cross-sectional ranking is a within-date transform that uses
only the rows of the date being transformed, so it cannot move information
between dates or across the train/validation boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class PreprocessingError(RuntimeError):
    """Raised when a preprocessing request is ill-formed or leaks."""


@dataclass(frozen=True)
class PreprocessingSpec:
    """Declared preprocessing pipeline (frozen per model family)."""

    impute: str = "median"          # median | none
    indicator: bool = False         # add missingness indicators (fitted on train)
    clip: str = "percentile"        # percentile | none
    scale: str = "robust"           # robust | zscore | none
    cross_sectional_rank: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FittedPreprocessor:
    """Train-fitted parameters; ``transform`` never refits them."""

    spec: PreprocessingSpec
    features: tuple
    medians: dict
    clip_low: dict
    clip_high: dict
    center: dict
    scale: dict
    indicator_features: tuple


class Preprocessor:
    """Fit on train, apply to train/validation; refuses to peek at other rows."""

    def __init__(self, spec=None):
        self.spec = spec or PreprocessingSpec()

    # ── fitting (TRAIN ONLY) ────────────────────────────────────────────────
    def fit(self, train_frame, features):
        spec = self.spec
        features = tuple(features)
        if not features:
            raise PreprocessingError("no features supplied to fit")
        medians, clip_low, clip_high, center, scale = {}, {}, {}, {}, {}
        indicator_features = []
        for name in features:
            values = pd.to_numeric(train_frame[name], errors="coerce") if name in train_frame.columns \
                else pd.Series(np.nan, index=train_frame.index, dtype="float64")
            observed = values.dropna()
            median = float(observed.median()) if len(observed) else 0.0
            medians[name] = median
            if spec.clip == "percentile" and len(observed):
                clip_low[name] = float(observed.quantile(0.01))
                clip_high[name] = float(observed.quantile(0.99))
            else:
                clip_low[name] = None
                clip_high[name] = None
            if spec.scale == "robust":
                if len(observed):
                    low = float(observed.quantile(0.25))
                    high = float(observed.quantile(0.75))
                    spread = high - low
                    if spread <= 0.0:
                        spread = float(observed.std(ddof=0))
                    centre = float(observed.median())
                else:
                    spread, centre = 0.0, 0.0
                scale[name] = spread if spread > 0.0 else 1.0
                center[name] = centre
            elif spec.scale == "zscore":
                if len(observed):
                    spread = float(observed.std(ddof=0))
                    centre = float(observed.mean())
                else:
                    spread, centre = 0.0, 0.0
                scale[name] = spread if spread > 0.0 else 1.0
                center[name] = centre
            else:
                scale[name] = 1.0
                center[name] = 0.0
            if spec.indicator:
                missing = values.isna()
                if 0 < int(missing.sum()) < len(values):
                    indicator_features.append(name)
        return FittedPreprocessor(
            spec=spec, features=features, medians=medians,
            clip_low=clip_low, clip_high=clip_high, center=center, scale=scale,
            indicator_features=tuple(indicator_features),
        )

    # ── applying (TRAIN + VALIDATION) ───────────────────────────────────────
    def transform(self, frame, fitted):
        spec = fitted.spec
        columns = {}
        for name in fitted.features:
            raw = pd.to_numeric(frame[name], errors="coerce") if name in frame.columns \
                else pd.Series(np.nan, index=frame.index, dtype="float64")
            missing_mask = raw.isna()
            values = raw.to_numpy(dtype="float64")
            if spec.impute == "median":
                values = np.where(np.isfinite(values), values, fitted.medians[name])
            if spec.clip == "percentile":
                low, high = fitted.clip_low.get(name), fitted.clip_high.get(name)
                if low is not None and high is not None:
                    values = np.clip(values, low, high)
            if spec.scale in ("robust", "zscore"):
                values = (values - fitted.center[name]) / fitted.scale[name]
            columns[name] = values
            if spec.indicator and name in fitted.indicator_features:
                columns["%s__missing" % name] = missing_mask.to_numpy(dtype="float64")
        result = pd.DataFrame(columns, index=frame.index)
        if spec.cross_sectional_rank:
            result = self._cross_sectional_rank(result, fitted.features, frame)
        return result

    @staticmethod
    def _cross_sectional_rank(result, features, frame):
        """Within-date percentile rank; uses only the rows of each date."""
        if "modeling_month" not in frame.columns:
            raise PreprocessingError("cross_sectional_rank requires modeling_month")
        months = frame["modeling_month"].to_numpy()
        ranked = result.copy()
        for name in features:
            values = result[name].to_numpy(dtype="float64")
            output = np.full(len(values), np.nan, dtype="float64")
            for month in pd.unique(months):
                positions = np.where(months == month)[0]
                local = values[positions]
                finite = np.isfinite(local)
                if finite.sum() == 0:
                    continue
                order = np.argsort(local[finite], kind="mergesort")
                count = int(finite.sum())
                percentiles = np.empty(count, dtype="float64")
                percentiles[order] = (np.arange(count, dtype="float64") + 0.5) / count
                output[positions[finite]] = percentiles
            ranked[name] = output
        return ranked


def preprocessor_payload(spec):
    """Deterministic payload for the preprocessing spec (experiment binding)."""
    return spec.to_dict() if isinstance(spec, PreprocessingSpec) else dict(spec)
