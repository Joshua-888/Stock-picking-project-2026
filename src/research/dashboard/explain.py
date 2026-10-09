"""Apply-only per-security explanations for the WP9 dashboard truth layer.

This module produces the ``stock_explanations`` contract. It never reads target,
holdout-label, future-return, or target-observable columns and never fits the
frozen champion, preprocessor, or calibrator.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from . import attribution, schemas
from .identity import attach_identity

DEFAULT_RANK = None
DEFAULT_PERCENTILE = None


def _as_frame(record):
    """Normalize one feature-valued record into a one-row DataFrame."""
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("explain_security expects exactly one row")
        return record.copy()
    if isinstance(record, Mapping):
        return pd.DataFrame([dict(record)])
    if isinstance(record, pd.Series):
        return record.to_frame().T
    raise TypeError("explain_security expects a mapping, Series, or one-row DataFrame")


def _missing_indicators(record: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for feature in schemas.FEATURE_ORDER:
        raw = record.get(feature)
        try:
            value = float(raw)
            out[feature] = 0 if pd.notna(raw) else 1
        except (TypeError, ValueError):
            out[feature] = 1 if raw is None else 0
    return out


def explain_security(
    champion,
    record,
    *,
    ticker: str,
    security_id: str,
    as_of_date: str,
    rank: Optional[int] = DEFAULT_RANK,
    percentile: Optional[float] = DEFAULT_PERCENTILE,
) -> Dict[str, Any]:
    """Return a local ``stock_explanations`` view for one security.

    The caller supplies the already-scoreable feature values. This function
    applies the frozen preprocessor, obtains the frozen scores without fitting,
    and decomposes the raw score into 13 base-feature contributions using the
    deterministic tree-path fallback. Calibrated scores, ranks, and percentiles
    are reported as frozen operational outputs, never as validated probabilities.
    """
    frame = _as_frame(record)
    raw_score_arr, calibrated_arr = champion.predict(frame)
    raw_score = float(raw_score_arr[0])
    calibrated_score = float(calibrated_arr[0])

    values: Dict[str, Any] = {}
    record_dict = frame.iloc[0].to_dict()
    for feature in schemas.FEATURE_ORDER:
        raw = record_dict.get(feature)
        try:
            values[feature] = None if pd.isna(raw) else float(raw)
        except (TypeError, ValueError):
            values[feature] = None

    local = attribution.tree_path_local(champion, frame)
    positive, negative = attribution.contributor_shape(local.contributions)

    result = {
        **schemas.schema_basis("stock_explanations"),
        "ticker": ticker,
        "security_id": security_id,
        "as_of_date": as_of_date,
        "raw_model_score": raw_score,
        "frozen_calibrated_score": calibrated_score,
        "rank": rank,
        "percentile": percentile,
        "feature_values": values,
        "missing_indicators": _missing_indicators(record_dict),
        "attribution": local.contributions,
        "top_positive_contributors": positive,
        "top_negative_contributors": negative,
        "attribution_residual": local.residual,
        "attribution_tolerance": attribution.ATTRIBUTION_TOLERANCE,
        "limitations": attribution.ATTRIBUTION_LIMITATIONS,
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("stock_explanations", result)
    return result


__all__ = ["explain_security", "_missing_indicators"]
