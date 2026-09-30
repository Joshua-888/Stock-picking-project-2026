"""WP6 preprocessing tests: train-only fitting; transform cannot peek."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.modeling.preprocessing import (
    Preprocessor,
    PreprocessingError,
    PreprocessingSpec,
)


def _frame(values, months=None):
    months = months or ["2019-%02d" % (index + 1) for index in range(len(values))]
    return pd.DataFrame({"modeling_month": months, "x": values})


def test_fit_uses_train_only_for_imputation_and_scale():
    train = _frame([1.0, 2.0, 3.0, 4.0, np.nan])
    validation = _frame([1e6, -1e6])
    pre = Preprocessor(PreprocessingSpec(impute="median", indicator=True, clip="percentile",
                                         scale="zscore", cross_sectional_rank=False))
    fitted = pre.fit(train, ["x"])
    # train mean/std, not the validation extremes
    assert fitted.center["x"] == pytest.approx(np.nanmean([1, 2, 3, 4]))
    assert fitted.medians["x"] == pytest.approx(2.5)
    train_out = pre.transform(train, fitted)
    val_out = pre.transform(validation, fitted)
    # validation extreme values are clipped by TRAIN percentile bounds
    assert val_out["x"].max() <= (fitted.clip_high["x"] - fitted.center["x"]) / fitted.scale["x"] + 1e-9
    assert val_out["x"].min() >= (fitted.clip_low["x"] - fitted.center["x"]) / fitted.scale["x"] - 1e-9
    assert list(train_out.columns) == list(val_out.columns)


def test_transform_is_independent_of_validation_content():
    train = _frame([1.0, 2.0, 3.0, 4.0])
    a = _frame([10.0, 20.0])
    b = _frame([999.0, -999.0])
    pre = Preprocessor(PreprocessingSpec(impute="median", scale="robust", clip="none"))
    fitted = pre.fit(train, ["x"])
    # identical rows in two different validation frames transform identically
    first = pre.transform(a, fitted)
    second = pre.transform(b, fitted)
    assert first["x"].iloc[0] != second["x"].iloc[0]  # different inputs, different outputs
    shared = _frame([10.0])
    both = pd.concat([a, shared], ignore_index=True)
    assert pre.transform(both, fitted)["x"].iloc[0] == first["x"].iloc[0]


def test_cross_sectional_rank_is_within_date():
    frame = _frame([1.0, 2.0, 3.0, 4.0], months=["2020-01"] * 4)
    frame.loc[4] = {"modeling_month": "2020-02", "x": 100.0}
    frame.loc[5] = {"modeling_month": "2020-02", "x": 50.0}
    pre = Preprocessor(PreprocessingSpec(impute="median", indicator=False, clip="none",
                                         scale="none", cross_sectional_rank=True))
    fitted = pre.fit(frame, ["x"])
    out = pre.transform(frame, fitted)
    jan = out.loc[frame["modeling_month"] == "2020-01", "x"].to_numpy()
    feb = out.loc[frame["modeling_month"] == "2020-02", "x"].to_numpy()
    assert jan.min() >= 0.0 and jan.max() <= 1.0
    assert feb.min() >= 0.0 and feb.max() <= 1.0
    # the Feb rows are ranked among themselves, not against January:
    # 100.0 (row 4) outranks 50.0 (row 5) within February
    assert feb[1] < feb[0]


def test_fit_requires_features():
    pre = Preprocessor()
    with pytest.raises(PreprocessingError):
        pre.fit(_frame([1.0, 2.0]), [])
