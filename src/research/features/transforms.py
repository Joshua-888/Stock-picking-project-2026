"""WP3 train-safe transforms: fit on training, apply anywhere.

Every statistical transform (winsorization percentiles, z-score mean/std,
robust median/IQR, ranking reference set) is FITTED on a training slice and
FROZEN. Applying it to validation or holdout never re-fits, so no future row can
move an earlier transform. This is the mechanism that keeps normalization from
leaking across the walk-forward boundary; the actual walk-forward split belongs
to WP6/WP7.

``log`` is the only stateless transform and is still routed through the same
fit/apply API so a caller cannot accidentally mix fitted and unfitted use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class TransformError(ValueError):
    """Raised when a transform is fitted or applied incorrectly."""


def _numeric(series):
    # Accept a Series, list, tuple or 1-D array and ALWAYS return a numeric
    # Series, so callers can use Series ops (.dropna/.apply/.clip) regardless of
    # input type. ``pd.to_numeric`` alone returns an ndarray for a bare list.
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return pd.to_numeric(series, errors="coerce")


@dataclass
class FittedTransform:
    """A transform bound to training-only statistics."""

    kind: str
    params: dict = field(default_factory=dict)
    fit_reference: str = "training"

    def to_dict(self):
        return {"kind": self.kind, "params": dict(self.params), "fit_reference": self.fit_reference}


def fit_transform(train_values, kind="TRAIN_ZSCORE", lower=0.01, upper=0.99):
    """Fit a transform on training values only.

    ``kind`` is one of ``TRAIN_ZSCORE``, ``TRAIN_RANK``, ``TRAIN_ROBUST_SCALE``,
    ``TRAIN_PERCENTILE_CLIP``, ``LOG`` or ``NONE``.
    """
    values = _numeric(train_values).dropna()
    if kind in ("NONE", "LOG"):
        return FittedTransform(kind=kind)
    if kind == "TRAIN_ZSCORE":
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std(ddof=0)) if len(values) else 0.0
        return FittedTransform(kind=kind, params={"mean": mean, "std": std if std else 1.0})
    if kind == "TRAIN_ROBUST_SCALE":
        median = float(values.median()) if len(values) else 0.0
        q1 = float(values.quantile(0.25)) if len(values) else 0.0
        q3 = float(values.quantile(0.75)) if len(values) else 0.0
        iqr = q3 - q1
        return FittedTransform(kind=kind, params={"median": median, "iqr": iqr if iqr else 1.0})
    if kind == "TRAIN_PERCENTILE_CLIP":
        low = float(values.quantile(lower)) if len(values) else 0.0
        high = float(values.quantile(upper)) if len(values) else 0.0
        return FittedTransform(kind=kind, params={"low": low, "high": high})
    if kind == "TRAIN_RANK":
        ordered = sorted(values.tolist())
        return FittedTransform(kind=kind, params={"reference": ordered})
    raise TransformError("unknown transform kind %r" % (kind,))


def apply_transform(series, fitted):
    """Apply a FITTED transform to any series (values are never re-fit here)."""
    if not isinstance(fitted, FittedTransform):
        raise TransformError("apply_transform expects a FittedTransform from fit_transform")
    values = _numeric(series)
    kind = fitted.kind
    if kind == "NONE":
        return values
    if kind == "LOG":
        # log1p keeps zeros and small positives defined; negatives stay NaN.
        return values.where(values >= 0, other=float("nan")).apply(lambda v: pd.NA if pd.isna(v) else __import__("math").log1p(v))
    if kind == "TRAIN_ZSCORE":
        return (values - fitted.params["mean"]) / fitted.params["std"]
    if kind == "TRAIN_ROBUST_SCALE":
        return (values - fitted.params["median"]) / fitted.params["iqr"]
    if kind == "TRAIN_PERCENTILE_CLIP":
        return values.clip(lower=fitted.params["low"], upper=fitted.params["high"])
    if kind == "TRAIN_RANK":
        reference = fitted.params["reference"]
        import bisect
        def _pct(value):
            if pd.isna(value):
                return float("nan")
            position = bisect.bisect_right(reference, float(value))
            return position / len(reference) if reference else float("nan")
        return values.apply(_pct)
    raise TransformError("unknown transform kind %r" % (kind,))
