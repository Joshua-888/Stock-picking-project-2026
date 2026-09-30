"""WP6 deterministic walk-forward fold generation with 12-month purge/embargo.

The fold geometry is computed by a FROZEN algorithm from the observed monthly
cross-sections, so it is reproducible and cannot be tuned to a result:

* the sorted monthly cross-sections form the timeline;
* the first validation window begins after ``min_train_months + horizon_months``
  monthly cross-sections, so the first fold always has a full purge plus a
  minimum training history;
* validation windows are consecutive, non-overlapping blocks of
  ``validation_window_months`` monthly cross-sections (the final window may be
  shorter);
* training is EXPANDING: every earlier cross-section is included;
* a training row is legal only when its label is fully observed at the model
  date (``target_known_at <= validation_start``) AND its label does not reach into
  the validation window (``target_end <= validation_start``), which is the frozen
  12-month purge rule.

Every fold asserts that it contains zero locked-holdout rows and zero embargo-band
rows, and that every training row is trainable at the model date. It reads no
label value for any decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..holdout import locked_holdout, trainable_mask
from .contract import DEFAULT_CONFIG
from .panel import assert_no_holdout_or_embargo


class FoldError(RuntimeError):
    """Raised when the walk-forward folds cannot be constructed honestly."""


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold with immutable train/validation row indices."""

    fold: int
    model_date: str
    train_start: str
    train_end: str
    purge_start: str
    purge_end: str
    validation_start: str
    validation_end: str
    train_index: tuple
    validation_index: tuple
    train_rows: int
    validation_rows: int
    train_securities: int
    validation_securities: int
    train_months: int
    validation_months: int

    def to_dict(self):
        payload = asdict(self)
        payload["train_index"] = list(self.train_index)
        payload["validation_index"] = list(self.validation_index)
        return payload


def _utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def _guard(frame, holdout):
    assert_no_holdout_or_embargo(frame, holdout)


def geometry(frame, config=None):
    """Return the frozen fold window boundaries as plain dicts (no rows)."""
    config = config or DEFAULT_CONFIG
    months = sorted(frame["modeling_month"].unique())
    horizon = int(config.horizon_months)
    first = int(config.min_train_months) + horizon
    step = int(config.validation_window_months)
    windows = []
    index = first
    while index < len(months):
        block = months[index:index + step]
        if len(block) >= horizon:  # a validation window must span a full label horizon
            windows.append({
                "validation_start": block[0],
                "validation_end": block[-1],
                "validation_months": len(block),
            })
        index += step
    return {
        "months": months,
        "first_validation_index": first,
        "validation_step": step,
        "windows": windows,
        "fold_count": len(windows),
    }


def build_folds(frame, config=None, holdout=None):
    """Build the deterministic walk-forward folds for ``frame``.

    Returns ``(folds, diagnostics, frame)`` where ``frame`` is the panel with a
    stable integer ``row_id`` and reset index, so ``Fold.*_index`` positions index
    directly into ``frame.iloc``.
    """
    config = config or DEFAULT_CONFIG
    holdout = holdout or locked_holdout()
    if "modeling_month" not in frame.columns:
        raise FoldError("panel lacks modeling_month; build it through panel.py")
    _guard(frame, holdout)

    working = frame.copy()
    working["feature_asof"] = _utc(working["feature_asof"])
    working["target_known_at_dt"] = _utc(working.get("target_known_at"))
    working["target_end_dt"] = _utc(working.get("target_end"))
    working = working.sort_values(["feature_asof", "security_id"], kind="mergesort").reset_index(drop=True)
    working["row_id"] = np.arange(len(working), dtype="int64")

    plan = geometry(working, config)
    folds = []
    diagnostics = {
        "fold_count": plan["fold_count"],
        "months": len(plan["months"]),
        "first_validation_index": plan["first_validation_index"],
        "validation_step": plan["validation_step"],
        "purge_months": int(config.horizon_months),
        "folds": [],
    }
    for number, window in enumerate(plan["windows"], start=1):
        validation_start = pd.Timestamp(window["validation_start"] + "-01").tz_localize("UTC") + pd.offsets.MonthEnd(1)
        # model_date is the availability instant at the start of validation
        model_date = validation_start.tz_localize("UTC") if validation_start.tzinfo is None else validation_start

        validation_month_set = set(
            month for month in plan["months"]
            if window["validation_start"] <= month <= window["validation_end"]
        )
        validation_mask = working["modeling_month"].isin(validation_month_set)

        before = working["feature_asof"] < model_date
        known = working["target_known_at_dt"].notna() & (working["target_known_at_dt"] <= model_date)
        purged_end = working["target_end_dt"].notna() & (working["target_end_dt"] <= model_date)
        train_mask = before & known & purged_end & (~validation_mask)

        train_index = tuple(int(value) for value in working.index[train_mask].tolist())
        validation_index = tuple(int(value) for value in working.index[validation_mask].tolist())
        train_frame = working.iloc[list(train_index)]
        validation_frame = working.iloc[list(validation_index)]
        if train_frame.empty:
            raise FoldError("fold %d has no training rows" % number)
        if validation_frame.empty:
            raise FoldError("fold %d has no validation rows" % number)
        _guard(train_frame, holdout)
        _guard(validation_frame, holdout)

        if not bool(trainable_mask(train_frame, model_date, holdout).all()):
            raise FoldError("fold %d contains non-trainable training rows" % number)

        fold = Fold(
            fold=number,
            model_date=str(model_date)[:19],
            train_start=str(train_frame["feature_asof"].min())[:10],
            train_end=str(train_frame["feature_asof"].max())[:10],
            purge_start=str(train_frame["feature_asof"].max())[:10],
            purge_end=str(model_date)[:10],
            validation_start=window["validation_start"],
            validation_end=window["validation_end"],
            train_index=train_index,
            validation_index=validation_index,
            train_rows=int(len(train_frame)),
            validation_rows=int(len(validation_frame)),
            train_securities=int(train_frame["security_id"].nunique()),
            validation_securities=int(validation_frame["security_id"].nunique()),
            train_months=int(train_frame["modeling_month"].nunique()),
            validation_months=int(validation_frame["modeling_month"].nunique()),
        )
        folds.append(fold)
        diagnostics["folds"].append({
            "fold": number,
            "model_date": fold.model_date,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "purge_start": fold.purge_start,
            "purge_end": fold.purge_end,
            "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
            "train_rows": fold.train_rows,
            "validation_rows": fold.validation_rows,
            "train_securities": fold.train_securities,
            "validation_securities": fold.validation_securities,
            "train_months": fold.train_months,
            "validation_months": fold.validation_months,
        })
    if len(folds) < int(config.min_folds):
        raise FoldError("only %d fold(s) available; contract requires >= %d" % (len(folds), config.min_folds))
    return folds, diagnostics, working
