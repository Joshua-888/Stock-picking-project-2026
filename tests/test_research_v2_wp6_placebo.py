"""WP6 control-battery tests (corrective contract v2).

The withdrawn WP6 battery returned tautological flags (``stop = bool(violations == 0)``)
and opaque ``suspicious`` booleans, which made the STOP rule inert. The corrective
battery returns explicit control-evaluation objects with real PASS / FAIL / STOP
semantics. These tests prove all three branches are reachable and that the date
tautology was removed from the modelling path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.modeling.placebo import (
    CONTROL_FIELDS,
    complexity_comparison,
    control_object,
    evaluate_metric_control,
    overall_stop,
    reduced_feature_comparison,
    shuffle_training_target,
    shuffled_target,
    subperiod_stability,
)
from src.research.modeling.pit import pit_guard_report


def test_control_object_exposes_all_contract_fields():
    control = evaluate_metric_control("c", "shuffled_target", 0.05, 0.001, "cid", 0.05,
                                      "expected behaviour", "observed %.4f")
    for field in CONTROL_FIELDS:
        assert field in control
    assert control["passed"] is True
    assert control["stop_required"] is False


def test_control_pass_and_fail_branches_are_reachable():
    passing = evaluate_metric_control("c", "t", 0.05, 0.001, "cid", 0.05, "e", "o %.4f")
    assert passing["passed"] is True and passing["stop_required"] is False
    failing = evaluate_metric_control("c", "t", 0.05, 0.20, "cid", 0.05, "e", "o %.4f")
    assert failing["passed"] is False and failing["stop_required"] is True


def test_control_missing_metric_fails_closed():
    control = evaluate_metric_control("c", "t", 0.05, None, "cid", 0.05, "e", "o %.4f")
    assert control["passed"] is False and control["stop_required"] is True


def test_overall_stop_requires_a_mandatory_failure():
    good = control_object("a", "t", "e", "o", None, None, 0.0, 0.0, "cond",
                          True, False, "ok", mandatory=True)
    bad = control_object("b", "t", "e", "o", None, None, 0.0, 0.0, "cond",
                         False, True, "bad", mandatory=True)
    assert overall_stop([good])["stop"] is False
    stopped = overall_stop([good, bad])
    assert stopped["stop"] is True
    assert stopped["mandatory_failing_controls"] == ["b"]
    optional = control_object("c", "t", "e", "o", None, None, 0.0, 0.0, "cond",
                              False, True, "bad", mandatory=False)
    assert overall_stop([good, optional])["stop"] is False


def test_no_tautological_flags_remain_in_modelling_placebo():
    import ast
    source = (REPO_ROOT / "src" / "research" / "modeling" / "placebo.py").read_text(
        encoding="utf-8")
    defined = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
    for removed in ("shuffled_target_placebo", "noise_feature_placebo",
                    "overall_stop_flag", "future_guard_report"):
        assert removed not in defined


def test_shuffle_is_deterministic_and_preserves_cross_section():
    frame = pd.DataFrame({
        "modeling_month": ["2020-01"] * 5 + ["2020-02"] * 5,
        "future_12m_excess_return": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4],
    })
    first = shuffle_training_target(frame, seed=1, asof_col="modeling_month")
    second = shuffle_training_target(frame, seed=1, asof_col="modeling_month")
    other = shuffle_training_target(frame, seed=2, asof_col="modeling_month")
    target = "future_12m_excess_return"
    assert list(first[target]) == list(second[target])
    assert list(first[target]) != list(other[target])
    for month in ("2020-01", "2020-02"):
        original = sorted(frame.loc[frame["modeling_month"] == month, target])
        permuted = sorted(first.loc[first["modeling_month"] == month, target])
        assert original == permuted  # the values are permuted within each date
    assert shuffled_target is shuffle_training_target


def test_injected_leakage_row_is_detected_by_field_guard():
    train = pd.DataFrame({
        "feature_asof": ["2010-01-31", "2011-01-31", "2016-01-31"],  # last is future
        "target_known_at": ["2011-01-31", "2012-01-31", "2012-01-31"],
        "target_end": ["2011-01-31", "2012-01-31", "2012-01-31"],
    })
    validation = pd.DataFrame({"feature_asof": ["2015-01-31", "2015-02-28"]})
    report = pit_guard_report(train, validation, "2015-01-01")
    assert report["passed"] is False
    assert report["violations"] >= 1
    assert report["train"]["feature_after_model_rows"] == 1


def test_clean_pit_fixture_passes():
    train = pd.DataFrame({
        "feature_asof": ["2010-01-31", "2011-01-31"],
        "target_known_at": ["2011-01-31", "2012-01-31"],
        "target_end": ["2011-01-31", "2012-01-31"],
    })
    validation = pd.DataFrame({"feature_asof": ["2015-01-31", "2015-02-28"]})
    report = pit_guard_report(train, validation, "2015-01-01")
    assert report["passed"] is True
    assert report["violations"] == 0


def test_subperiod_stability_detects_sign_flip():
    series = pd.DataFrame({"month": ["2020-%02d" % (index + 1) for index in range(12)],
                           "rank_ic": [0.1] * 6 + [-0.1] * 6})
    assert subperiod_stability(series)["sign_flip"] is True


def test_reduced_and_complexity_comparisons():
    reduced = reduced_feature_comparison({"mean_ic": 0.05}, {"mean_ic": 0.03}, metric="mean_ic")
    assert reduced["difference"] < 0
    assert reduced["reduced_justified"] is False
    complexity = complexity_comparison({"mean_ic": 0.02}, {"mean_ic": 0.03}, metric="mean_ic")
    assert complexity["complexity_justified"] is False