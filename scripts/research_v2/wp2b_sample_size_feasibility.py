"""WP2B deliverable 7 - sample-size feasibility (mission sections 19/20).

Computes, for a 2000->present monthly cross-section of ~500 names, how many
security-months exist and how many remain after:

* 12-month forward-label observability (the last 12 months cannot be labelled);
* a label-horizon embargo between train and test (12 months, so no training
  observation's forward return extends into the validation window);
* walk-forward fold construction (initial train window + rolling test windows);
* a locked final holdout that must never be touched during development.

It then compares the surviving row counts against coarse rules of thumb for
OLS / Ridge / Lasso / logistic regression. It does NOT fit any model and does
NOT look at any returns: it is a pure counting exercise for experimental design.

Run: /opt/venv/bin/python scripts/research_v2/wp2b_sample_size_feasibility.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

# --- Design parameters (documented, not tuned to any result) ------------------
START = "2000-01"
PRESENT = "2026-09"          # current month
CROSS_SECTION = 500          # ~S&P 500 names per monthly cross-section
LABEL_HORIZON = 12           # 12-month forward excess return
EMBARGO = 12                 # >= label horizon, prevents label overlap leakage
INITIAL_TRAIN = 60           # 5 years initial training window
TEST_WINDOW = 12             # 1-year out-of-sample window per fold
HOLDOUT = 36                 # 3-year locked holdout (never used in development)
EXIT_TURNOVER_PER_YEAR = 25  # approximate annual index removals (documented assumption)

# Rule-of-thumb minimum observations per predictor (statistical, not alpha tuning).
OBS_PER_PREDICTOR = {
    "OLS": 20,
    "Ridge": 15,
    "Lasso": 10,
    "Logistic": 20,
}
ASSUMED_PREDICTORS = 20

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "artifacts", "research", "wp2b")


def _months(start, end):
    return [p for p in pd.period_range(start=start, end=end, freq="M")]


def compute():
    all_months = _months(START, PRESENT)
    months_total = len(all_months)
    observable_months = months_total - LABEL_HORIZON
    development_months = observable_months - HOLDOUT

    gross = months_total * CROSS_SECTION
    labelled = observable_months * CROSS_SECTION
    after_holdout = (observable_months - HOLDOUT) * CROSS_SECTION

    # Walk-forward folds over the development window only (holdout excluded).
    dev = all_months[:development_months]
    folds = []
    train_end_idx = INITIAL_TRAIN - 1
    while True:
        test_start_idx = train_end_idx + EMBARGO + 1
        test_end_idx = test_start_idx + TEST_WINDOW - 1
        if test_end_idx >= len(dev):
            break
        train_months = train_end_idx + 1
        test_months = test_end_idx - test_start_idx + 1
        folds.append({
            "fold": len(folds) + 1,
            "train_start": str(dev[0]),
            "train_end": str(dev[train_end_idx]),
            "test_start": str(dev[test_start_idx]),
            "test_end": str(dev[test_end_idx]),
            "train_rows": train_months * CROSS_SECTION,
            "test_rows": test_months * CROSS_SECTION,
        })
        train_end_idx = test_end_idx

    total_oos_rows = sum(f["test_rows"] for f in folds)
    min_train_rows = min((f["train_rows"] for f in folds), default=0)
    first_train_rows = folds[0]["train_rows"] if folds else INITIAL_TRAIN * CROSS_SECTION

    requirement = ASSUMED_PREDICTORS * OBS_PER_PREDICTOR["OLS"]

    adequacy = {
        "model": {},
    }
    for model, per_pred in OBS_PER_PREDICTOR.items():
        need = ASSUMED_PREDICTORS * per_pred
        adequacy["model"][model] = {
            "assumed_predictors": ASSUMED_PREDICTORS,
            "min_obs_per_predictor": per_pred,
            "min_rows_required_per_fold": need,
            "min_train_rows_per_fold": min_train_rows,
            "total_oos_rows": total_oos_rows,
            "sufficient_in_sample": min_train_rows >= need,
            "sufficient_out_of_sample": total_oos_rows >= need,
        }

    # Survivorship illustration: how many of a cross-section later exit.
    years = months_total / 12.0
    cumulative_exits = int(round(EXIT_TURNOVER_PER_YEAR * years))
    # Rough share of any single historical cross-section that has since delisted.
    share_later_delisted = min(0.99, (EXIT_TURNOVER_PER_YEAR * years) / (CROSS_SECTION + EXIT_TURNOVER_PER_YEAR * years))

    return {
        "parameters": {
            "start": START,
            "present": PRESENT,
            "cross_section_names": CROSS_SECTION,
            "label_horizon_months": LABEL_HORIZON,
            "embargo_months": EMBARGO,
            "initial_train_months": INITIAL_TRAIN,
            "test_window_months": TEST_WINDOW,
            "holdout_months": HOLDOUT,
            "assumed_annual_index_exits": EXIT_TURNOVER_PER_YEAR,
        },
        "counts": {
            "months_total": months_total,
            "observable_months": observable_months,
            "development_months": development_months,
            "holdout_months": HOLDOUT,
            "gross_security_months": gross,
            "labelled_security_months": labelled,
            "development_security_months": after_holdout,
            "walk_forward_folds": len(folds),
            "total_out_of_sample_rows": total_oos_rows,
            "min_train_rows_per_fold": min_train_rows,
            "first_fold_train_rows": first_train_rows,
        },
        "adequate_for": adequacy,
        "survivorship": {
            "cumulative_exits_over_period": cumulative_exits,
            "approx_share_of_cross_section_later_delisted": round(share_later_delisted, 4),
            "note": (
                "a survivorship-biased sample built only from today's constituents "
                "drops these rows; because exiting names skew to the downside, the "
                "bias is upward. This is why the historical universe family is "
                "BLOCKED until delisted prices are available."
            ),
        },
        "folds": folds,
        "verdict": _verdict(len(folds), min_train_rows, total_oos_rows, requirement),
    }


def _verdict(n_folds, min_train_rows, total_oos_rows, requirement):
    if n_folds < 1:
        return "INFEASIBLE: development window too short for any walk-forward fold"
    if min_train_rows < requirement:
        return "MARGINAL: at least one fold has fewer training rows than the OLS rule of thumb"
    return (
        "FEASIBLE (row count): %d folds, min train %d rows, %d OOS rows; "
        "count is adequate, but VALIDITY remains BLOCKED by missing delisted prices"
        % (n_folds, min_train_rows, total_oos_rows)
    )


def _markdown(result):
    c = result["counts"]
    p = result["parameters"]
    lines = [
        "# WP2B Sample-Size Feasibility (mission sections 19/20)",
        "",
        "Counting exercise only. No model is fitted and no returns are inspected.",
        "",
        "## Parameters",
        "",
    ]
    for key, value in p.items():
        lines.append("- %s: %s" % (key, value))
    lines += ["", "## Counts", ""]
    for key, value in c.items():
        lines.append("- %s: %s" % (key, value))
    lines += [
        "",
        "## Statistical adequacy (rules of thumb, %d assumed predictors)" % ASSUMED_PREDICTORS,
        "",
        "| Model | Min obs/predictor | Required/ fold | Min train/fold | Total OOS | Sufficient |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model, info in result["adequate_for"]["model"].items():
        lines.append(
            "| %s | %d | %d | %d | %d | %s |"
            % (
                model,
                info["min_obs_per_predictor"],
                info["min_rows_required_per_fold"],
                info["min_train_rows_per_fold"],
                info["total_oos_rows"],
                "yes" if info["sufficient_in_sample"] and info["sufficient_out_of_sample"] else "no",
            )
        )
    s = result["survivorship"]
    lines += [
        "",
        "## Survivorship reality check",
        "",
        "- cumulative approximate index exits over the period: %d" % s["cumulative_exits_over_period"],
        "- approx share of a historical cross-section that later delisted: %.2f%%"
        % (100 * s["approx_share_of_cross_section_later_delisted"]),
        "- %s" % s["note"],
        "",
        "## Verdict",
        "",
        "%s" % result["verdict"],
        "",
    ]
    return "\n".join(lines)


def main():
    result = compute()
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "sample_size_feasibility.json")
    md_path = os.path.join(OUT_DIR, "sample_size_feasibility.md")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    with open(md_path, "w") as handle:
        handle.write(_markdown(result))
    print("wrote", json_path)
    print("wrote", md_path)
    print(result["verdict"])


if __name__ == "__main__":
    main()
