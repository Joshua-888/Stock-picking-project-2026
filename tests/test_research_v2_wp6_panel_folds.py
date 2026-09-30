"""WP6 panel + fold tests (synthetic fixtures only; no research data, no network)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.modeling.contract import DEFAULT_CONFIG, model_feature_universe
from src.research.modeling.folds import FoldError, build_folds, geometry
from src.research.modeling.panel import (
    ModelingPanelError,
    assert_no_holdout_or_embargo,
    duplicate_key_report,
)


def _frame(months=90, per_month=40, start="2005-01-31", label=True):
    stamps = pd.date_range(start, periods=months, freq="ME")
    rows = []
    for stamp in stamps:
        for index in range(per_month):
            record = {
                "security_id": "SEC%03d" % index,
                "ticker": "SEC%03d" % index,
                "feature_asof": stamp.strftime("%Y-%m-%d"),
                "modeling_month": stamp.strftime("%Y-%m"),
                "target_observable": True,
                "target_known_at": stamp + pd.DateOffset(months=12),
                "target_end": stamp + pd.DateOffset(months=12),
            }
            for name in model_feature_universe():
                record[name] = float((index * 7 + stamp.month) % 13) / 13.0
            if label:
                record["future_12m_excess_return"] = float(index - per_month / 2) / 1000.0
                record["outperform_12m"] = 1.0 if index % 2 == 0 else 0.0
            rows.append(record)
    return pd.DataFrame(rows)


def test_holdout_and_embargo_rows_are_refused():
    frame = _frame()
    frame.loc[frame.index[0], "feature_asof"] = "2022-03-31"  # inside locked holdout
    with pytest.raises(ModelingPanelError):
        assert_no_holdout_or_embargo(frame)


def test_embargo_band_is_refused():
    frame = _frame()
    frame.loc[frame.index[0], "feature_asof"] = "2021-06-30"  # embargo band
    with pytest.raises(ModelingPanelError):
        assert_no_holdout_or_embargo(frame)


def test_duplicate_observable_rows_are_refused():
    frame = _frame(months=3)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ModelingPanelError):
        duplicate_key_report(doubled)


def test_duplicate_censored_rows_are_reported_not_fatal():
    frame = _frame(months=3)
    clone = frame.iloc[[0]].copy()
    clone["target_observable"] = False
    frame.loc[frame.index[0], "target_observable"] = False
    doubled = pd.concat([frame, clone], ignore_index=True)
    report = duplicate_key_report(doubled)
    assert report["duplicate_rows"] >= 2
    assert report["duplicate_observable_rows"] == 0


def test_folds_purge_and_trainability():
    frame = _frame(months=168)
    folds, diagnostics, working = build_folds(frame, config=DEFAULT_CONFIG)
    assert len(folds) >= DEFAULT_CONFIG.min_folds
    for fold in folds:
        train = working.iloc[list(fold.train_index)]
        validation = working.iloc[list(fold.validation_index)]
        assert len(train) > 0 and len(validation) > 0
        # 12-month purge: no training label reaches into the validation window
        assert pd.to_datetime(train["target_end"]).max() <= pd.Timestamp(fold.model_date)
        assert pd.to_datetime(train["target_known_at"]).max() <= pd.Timestamp(fold.model_date)
        assert train["feature_asof"].max() < validation["feature_asof"].min()


def test_fold_geometry_is_deterministic():
    frame = _frame(months=90)
    first = geometry(frame, DEFAULT_CONFIG)
    second = geometry(frame.copy(), DEFAULT_CONFIG)
    assert first == second


def test_fold_geometry_requires_minimum_folds():
    frame = _frame(months=40)
    with pytest.raises(FoldError):
        build_folds(frame, config=DEFAULT_CONFIG)


def test_no_validation_overlap_between_folds():
    frame = _frame(months=168)
    folds, _diagnostics, _working = build_folds(frame, config=DEFAULT_CONFIG)
    seen = set()
    for fold in folds:
        months = set(pd.to_datetime(frame.iloc[list(fold.validation_index)]["feature_asof"]).dt.to_period("M"))
        assert not (months & seen)
        seen |= months
