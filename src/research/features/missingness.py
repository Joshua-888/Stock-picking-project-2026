"""WP3 missing-data semantics: observable, model-safe, never zero-substituted.

Rules enforced here:

* missing fundamentals are NEVER silently replaced with 0;
* a missing indicator can accompany an imputed value, so the model can tell
  "unknown" apart from "truly zero";
* any imputation (median/mean/sector-median) is FITTED ON TRAINING DATA ONLY;
  the transform records the values it fitted so the same numbers can be applied
  to validation/holdout without re-looking at them;
* ``MODEL_NATIVE`` leaves the missing value in place for estimators that accept
  it, rather than fabricating a value.

Fitting and applying are deliberately separate: ``fit_imputer`` returns an
object bound to the training slice, and ``apply_imputer`` never recomputes a
statistic from the frame it is applied to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MISSING_INDICATOR_SUFFIX = "_is_missing"


class MissingnessError(ValueError):
    """Raised when a missing-data rule is applied incorrectly."""


@dataclass
class Imputer:
    """A fitted-to-training imputer; carries the statistics it will apply."""

    strategy: str
    fill_values: dict = field(default_factory=dict)
    global_fill: float = None
    add_indicator: bool = True
    fit_reference: str = "training"

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "fill_values": dict(self.fill_values),
            "global_fill": self.global_fill,
            "add_indicator": bool(self.add_indicator),
            "fit_reference": self.fit_reference,
        }


def _indicator_name(column):
    return "%s%s" % (column, MISSING_INDICATOR_SUFFIX)


def fit_imputer(train_frame, column, strategy="INDICATOR_PLUS_TRAIN_MEDIAN", sector_column=None):
    """Fit an imputer on ``train_frame`` ONLY for one column.

    ``strategy`` is one of ``INDICATOR_PLUS_TRAIN_MEDIAN``,
    ``INDICATOR_PLUS_TRAIN_SECTOR_MEDIAN``, ``MODEL_NATIVE`` or ``EXCLUDE_ROW``.
    No strategy ever uses zero as a fill value.
    """
    if column not in train_frame.columns:
        raise MissingnessError("column %r not present in training frame" % column)
    values = pd.to_numeric(train_frame[column], errors="coerce")
    observed = values.dropna()

    if strategy == "MODEL_NATIVE":
        return Imputer(strategy=strategy, add_indicator=False)
    if strategy == "EXCLUDE_ROW":
        return Imputer(strategy=strategy, add_indicator=False)
    if strategy == "INDICATOR_PLUS_TRAIN_SECTOR_MEDIAN":
        if sector_column is None or sector_column not in train_frame.columns:
            raise MissingnessError("sector strategy requires a sector column present in training")
        fill_values = {}
        for sector, chunk in train_frame.groupby(sector_column):
            sector_values = pd.to_numeric(chunk[column], errors="coerce").dropna()
            if len(sector_values):
                fill_values[str(sector)] = float(sector_values.median())
        global_fill = float(observed.median()) if len(observed) else None
        return Imputer(strategy=strategy, fill_values=fill_values, global_fill=global_fill)
    if strategy == "INDICATOR_PLUS_TRAIN_MEDIAN":
        global_fill = float(observed.median()) if len(observed) else None
        return Imputer(strategy=strategy, global_fill=global_fill)
    raise MissingnessError("unknown missing strategy %r" % (strategy,))


def apply_imputer(frame, column, imputer, sector_column=None):
    """Apply a FITTED imputer; returns ``(values, indicator, applied)``.

    ``applied`` records how many cells were filled. The frame being applied to is
    never used to recompute a statistic, so validation/holdout rows cannot leak
    into the value used to fill them.
    """
    if column not in frame.columns:
        raise MissingnessError("column %r not present in apply frame" % column)
    values = pd.to_numeric(frame[column], errors="coerce").copy()
    missing = values.isna()
    indicator = missing.astype(int) if imputer.add_indicator else None
    if imputer.strategy in ("MODEL_NATIVE", "EXCLUDE_ROW"):
        return values, indicator, 0
    filled = 0
    if imputer.strategy == "INDICATOR_PLUS_TRAIN_SECTOR_MEDIAN" and sector_column is not None:
        if sector_column not in frame.columns:
            raise MissingnessError("sector column %r not present in apply frame" % sector_column)
        for position in frame.index[missing]:
            sector = str(frame.at[position, sector_column])
            fill = imputer.fill_values.get(sector, imputer.global_fill)
            if fill is not None:
                values.at[position] = fill
                filled += 1
    else:
        if imputer.global_fill is not None:
            values.loc[missing] = imputer.global_fill
            filled = int(missing.sum())
    return values, indicator, filled
