"""
tests/test_scoring.py

Tests for the scoring model and classification logic.
"""

import pytest
import numpy as np
import pandas as pd


# ── Score bounds ───────────────────────────────────────────────────────────────

def test_final_score_in_range():
    """final_score must be in [0, 100]."""
    from src.scoring.scoring_model import compute_scores
    features = pd.DataFrame({
        "data_quality_score":    [80.0, 70.0, 50.0, 90.0, 40.0],
        "twelve_month_momentum": [0.20, -0.10, 0.05, 0.30, -0.05],
        "current_pe_ratio":      [20.0, 35.0, 15.0, 25.0, 50.0],
        "volatility_12m":        [0.20, 0.35, 0.15, 0.18, 0.40],
        "roe":                   [0.20, 0.10, 0.25, 0.18, 0.08],
        "sector_relative_pe":    [0.1, -0.2, 0.0, 0.05, 0.3],
        "sector_relative_momentum": [0.05, -0.1, 0.02, 0.08, -0.15],
    }, index=["A","B","C","D","E"])

    scores = compute_scores(features)
    assert (scores["final_score"] >= 0).all() and (scores["final_score"] <= 100).all()


def test_component_scores_in_range():
    """All component scores must be in [0, 100]."""
    from src.scoring.scoring_model import compute_scores
    features = pd.DataFrame({
        "data_quality_score": [85.0, 60.0, 40.0],
        "twelve_month_momentum": [0.15, -0.05, 0.30],
        "volatility_12m": [0.20, 0.30, 0.15],
        "roe": [0.20, 0.10, 0.30],
        "current_pe_ratio": [18.0, 35.0, 22.0],
    }, index=["X","Y","Z"])

    scores = compute_scores(features)
    for col in ["probability_score","excess_return_score","model_agreement_score",
                "risk_score","data_quality_score","valuation_score",
                "momentum_score","fundamental_score"]:
        if col in scores.columns:
            assert (scores[col] >= 0).all() and (scores[col] <= 100).all(), \
                f"{col} out of [0,100] range"


# ── Classification exhaustiveness ─────────────────────────────────────────────

def test_every_stock_gets_exactly_one_classification():
    """Every stock must receive exactly one candidate_classification."""
    from src.scoring.scoring_model import compute_scores
    from src.scoring.classifications import classify_stocks, LABELS

    features = pd.DataFrame({
        "data_quality_score": [90.0, 70.0, 40.0, 80.0, 55.0],
        "probability_of_outperformance": [0.70, 0.55, 0.30, 0.65, 0.48],
        "predicted_12m_excess_return":   [0.08, 0.03, -0.05, 0.06, 0.01],
        "model_agreement_score":         [75.0, 60.0, 40.0, 72.0, 50.0],
    }, index=["A","B","C","D","E"])

    scores = compute_scores(features)
    scores.insert(0, "ticker", list(features.index))
    classified = classify_stocks(scores)

    assert "candidate_classification" in classified.columns
    assert classified["candidate_classification"].notna().all()
    valid_labels = set(LABELS.values())
    assert classified["candidate_classification"].isin(valid_labels).all()


def test_strong_candidate_requires_all_conditions():
    """A stock only gets 'Strong candidate' if ALL threshold conditions are met."""
    from src.scoring.classifications import classify_stocks

    scores = pd.DataFrame([{
        "ticker": "A",
        "probability_of_outperformance": 0.70,  # meets threshold
        "predicted_12m_excess_return":   0.08,  # meets threshold
        "model_agreement_score":         80.0,  # meets threshold
        "data_quality_score":            80.0,  # meets threshold
        "final_score":                   75.0,
    }, {
        "ticker": "B",
        "probability_of_outperformance": 0.70,
        "predicted_12m_excess_return":   0.08,
        "model_agreement_score":         80.0,
        "data_quality_score":            50.0,  # BELOW threshold (needs >= 75)
        "final_score":                   75.0,
    }])

    result = classify_stocks(scores)
    cls_a  = result[result["ticker"] == "A"]["candidate_classification"].iloc[0]
    cls_b  = result[result["ticker"] == "B"]["candidate_classification"].iloc[0]
    assert cls_a == "Strong candidate", f"A should be Strong but got {cls_a}"
    assert cls_b != "Strong candidate", f"B has low DQ and should not be Strong"


# ── Keyes agreement flag ───────────────────────────────────────────────────────

def test_keyes_flag_requires_all_conditions():
    """Keyes flag=1 requires P(Win)≥0.5, excess_return≥0, agreement≥60, DQ≥60."""
    from src.scoring.classifications import compute_keyes_flag

    scores = pd.DataFrame([
        # All conditions met → flag=1
        {"probability_of_outperformance": 0.65, "predicted_12m_excess_return":  0.05,
         "model_agreement_score": 70.0, "data_quality_score": 80.0},
        # P(Win) too low → flag=0
        {"probability_of_outperformance": 0.45, "predicted_12m_excess_return":  0.05,
         "model_agreement_score": 70.0, "data_quality_score": 80.0},
        # excess_return negative → flag=0
        {"probability_of_outperformance": 0.65, "predicted_12m_excess_return": -0.01,
         "model_agreement_score": 70.0, "data_quality_score": 80.0},
        # DQ too low → flag=0
        {"probability_of_outperformance": 0.65, "predicted_12m_excess_return":  0.05,
         "model_agreement_score": 70.0, "data_quality_score": 40.0},
    ])

    flags = compute_keyes_flag(scores)
    assert flags.iloc[0] == 1, "All conditions met → flag should be 1"
    assert flags.iloc[1] == 0, "Low P(Win) → flag should be 0"
    assert flags.iloc[2] == 0, "Negative excess return → flag should be 0"
    assert flags.iloc[3] == 0, "Low DQ → flag should be 0"


# ── Config weights ─────────────────────────────────────────────────────────────

def test_scoring_weights_sum_to_one():
    """The default scoring weights in config must sum to exactly 1.0."""
    from src.utils.config import load_config
    cfg     = load_config()
    weights = vars(cfg.scoring.weights)
    total   = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"Scoring weights sum to {total:.6f}, expected 1.0"
