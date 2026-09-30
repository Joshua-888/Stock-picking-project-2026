"""WP6 metrics tests against known fixtures."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.modeling.metrics import (
    classification_metrics,
    monthly_ic_series,
    quantile_spread,
    regression_errors,
    summarize_ic,
)


def test_perfect_rank_ic_is_one():
    months = ["2020-%02d" % (index + 1) for index in range(24)]
    rows = []
    for month in months:
        for index in range(50):
            rows.append({"modeling_month": month, "prediction": float(index),
                         "actual": float(index)})
    frame = pd.DataFrame(rows)
    series, diagnostics = monthly_ic_series(frame, "prediction", "actual")
    assert len(series) == 24
    assert np.allclose(series["rank_ic"], 1.0)
    summary = summarize_ic(series)
    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["positive_ic_fraction"] == 1.0


def test_inverted_rank_ic_is_minus_one():
    rows = []
    for month in range(12):
        for index in range(40):
            rows.append({"modeling_month": "2020-%02d" % (month + 1),
                         "prediction": float(index), "actual": float(-index)})
    frame = pd.DataFrame(rows)
    series, _diagnostics = monthly_ic_series(frame, "prediction", "actual")
    assert np.allclose(series["rank_ic"], -1.0)


def test_min_observations_skip():
    frame = pd.DataFrame({"modeling_month": ["2020-01"] * 5, "prediction": range(5),
                          "actual": range(5)})
    series, diagnostics = monthly_ic_series(frame, "prediction", "actual", min_observations=30)
    assert len(series) == 0
    assert diagnostics["skipped"][0]["reason"] == "below_min_observations"


def test_regression_errors_known_values():
    frame = pd.DataFrame({"prediction": [1.0, 2.0, 3.0], "actual": [1.0, 1.0, 1.0]})
    errors = regression_errors(frame, "prediction", "actual")
    # errors are 0, 1, 2 -> MAE = 1.0 and RMSE = sqrt((0+1+4)/3)
    assert errors["mae"] == pytest.approx(1.0)
    assert errors["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_classification_metrics_perfect_separation():
    frame = pd.DataFrame({"score": [0.1, 0.2, 0.8, 0.9], "actual": [0.0, 0.0, 1.0, 1.0]})
    metrics = classification_metrics(frame, "score", "actual")
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["log_loss"] is not None and metrics["log_loss"] > 0


def test_quantile_spread_known_direction():
    rows = []
    for index in range(100):
        rows.append({"modeling_month": "2020-01", "prediction": float(index),
                     "actual": float(index)})
    frame = pd.DataFrame(rows)
    spread = quantile_spread(frame, "prediction", "actual", buckets=5)
    assert spread["spread"] > 0  # top bucket beats bottom bucket
