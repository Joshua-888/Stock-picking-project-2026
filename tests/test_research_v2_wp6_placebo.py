"""WP6 placebo tests: the falsification battery must be non-degenerate."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.modeling.placebo import (
    complexity_comparison,
    future_guard_report,
    noise_feature_placebo,
    overall_stop_flag,
    reduced_feature_comparison,
    shuffled_target,
    shuffled_target_placebo,
    subperiod_stability,
)


def test_shuffled_target_placebo_flags_reproduction():
    assert shuffled_target_placebo(0.05, 0.20)["suspicious"] is True
    assert shuffled_target_placebo(0.20, 0.02)["suspicious"] is False


def test_noise_feature_placebo_flags_dominance():
    assert noise_feature_placebo(0.03, 0.05)["suspicious"] is True
    assert noise_feature_placebo(0.10, 0.01)["suspicious"] is False


def test_overall_stop_flag_aggregates():
    checks = {"a": {"suspicious": False}, "b": {"stop": True}}
    assert overall_stop_flag(checks)["stop"] is True
    assert overall_stop_flag({"a": {"suspicious": False}})["stop"] is False


def test_shuffled_target_preserves_cross_section():
    frame = pd.DataFrame({
        "modeling_month": ["2020-01"] * 5 + ["2020-02"] * 5,
        "future_12m_excess_return": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4],
    })
    shuffled = shuffled_target(frame, seed=1, asof_col="modeling_month")
    for month in ("2020-01", "2020-02"):
        original = sorted(frame.loc[frame["modeling_month"] == month, "future_12m_excess_return"])
        permuted = sorted(shuffled.loc[shuffled["modeling_month"] == month, "future_12m_excess_return"])
        assert original == permuted  # the values are permuted within each date


def test_future_guard_requires_pit_enforcement():
    frame = pd.DataFrame({"feature_asof": ["2020-01-31", "2020-02-29", "2020-03-31"],
                          "twelve_month_momentum": [1.0, 2.0, 3.0]})
    report = future_guard_report(frame)
    assert report["violations"] == 3
    assert report["all_violate"] is True
    assert report["stop"] is False  # violations present -> the guard is working


def test_subperiod_stability_detects_sign_flip():
    series = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(12)],
                           "rank_ic": [0.1] * 6 + [-0.1] * 6})
    result = subperiod_stability(series)
    assert result["sign_flip"] is True


def test_reduced_and_complexity_comparisons():
    reduced = reduced_feature_comparison({"mean_ic": 0.05}, {"mean_ic": 0.03})
    assert reduced["difference"] < 0
    complexity = complexity_comparison({"mean_ic": 0.02}, {"mean_ic": 0.03})
    assert complexity["complexity_justified"] is False
