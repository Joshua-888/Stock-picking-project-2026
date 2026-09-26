"""WP5 missingness analysis.

A candidate feature's missing indicator can itself look predictive. This module
tests whether missingness relates to the realized outcome using ONLY information
available before the outcome (the fact that a value is absent at the prediction
date), and it never auto-promotes a missingness indicator: it reports the
difference and lets WP6 decide under training-window control.

The comparison is deliberately simple and dependence-aware: it contrasts the
mean outcome between present and missing groups AND reports the outcome spread,
so a reader can see whether an apparently large difference is a thin-sample
artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MissingnessError(ValueError):
    """Raised when a missingness request is ill-formed."""


def missingness_effect(frame, feature, target="future_12m_excess_return",
                       classification="outperform_12m"):
    """Compare the outcome for missing vs present observations.

    Only the PRE-OUTCOME fact of presence/absence is used, so no future label
    leaks into the split. Returns counts, mean outcome and outperform rate per
group and the present-minus-missing difference.
    """
    if feature not in frame.columns:
        raise MissingnessError("feature %r absent from frame" % (feature,))
    if target not in frame.columns:
        raise MissingnessError("target %r absent from frame" % (target,))

    values = pd.to_numeric(frame[feature], errors="coerce")
    target_values = pd.to_numeric(frame[target], errors="coerce")
    present = values.notna()
    labelled = target_values.notna()

    def _group(mask):
        selection = frame.loc[mask & labelled]
        if selection.empty:
            return {"count": 0, "mean_target": None, "outperform_rate": None}
        record = {
            "count": int(len(selection)),
            "mean_target": float(pd.to_numeric(selection[target], errors="coerce").mean()),
        }
        if classification in selection.columns:
            record["outperform_rate"] = float(
                pd.to_numeric(selection[classification], errors="coerce").mean())
        else:
            record["outperform_rate"] = None
        return record

    present_group = _group(present)
    missing_group = _group(~present)
    difference = None
    if present_group["mean_target"] is not None and missing_group["mean_target"] is not None:
        difference = present_group["mean_target"] - missing_group["mean_target"]

    # Dependence-aware evidence: split the labelled sample into quarterly blocks
    # and average the within-block difference, so overlapping labels do not
    # overstate the separation.
    block_differences = _block_differences(frame, feature, target, present, labelled)
    mean_block_difference = float(np.mean(block_differences)) if block_differences else None
    block_std = float(np.std(block_differences, ddof=1)) if len(block_differences) > 1 else None
    block_t = None
    if mean_block_difference is not None and block_std and block_std > 0.0:
        block_t = mean_block_difference / (block_std / np.sqrt(len(block_differences)))

    return {
        "feature": feature,
        "present": present_group,
        "missing": missing_group,
        "present_minus_missing_mean": difference,
        "block_count": len(block_differences),
        "block_mean_difference": mean_block_difference,
        "block_t_stat": float(block_t) if block_t is not None else None,
        "note": "missingness is tested with pre-outcome presence only; no indicator is auto-promoted",
    }


def _block_differences(frame, feature, target, present, labelled, block="QE"):
    stamps = pd.to_datetime(frame["feature_asof"], errors="coerce", utc=True)
    working = pd.DataFrame({
        "__block": stamps.dt.to_period("Q").astype(str),
        "__present": present,
        "__target": pd.to_numeric(frame[target], errors="coerce"),
        "__labelled": labelled,
    })
    working = working.loc[working["__labelled"] & working["__target"].notna()]
    differences = []
    for _block_id, chunk in working.groupby("__block", sort=True):
        present_values = chunk.loc[chunk["__present"], "__target"]
        missing_values = chunk.loc[~chunk["__present"], "__target"]
        if len(present_values) and len(missing_values):
            differences.append(float(present_values.mean() - missing_values.mean()))
    return differences
