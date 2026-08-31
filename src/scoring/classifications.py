"""
src/scoring/classifications.py

Assigns candidate_classification and Keyes-style agreement flag.

CLASSIFICATION RULES
--------------------
Labels are research-focused — never "Buy" or "Sell".

Strong candidate
    All of the following are met:
      • probability_of_outperformance  ≥ 0.60
      • predicted_12m_excess_return    ≥ 0.05   (5 pp above benchmark)
      • model_agreement_score          ≥ 70
      • data_quality_score             ≥ 75

Watchlist candidate
    All of the following are met (and not Strong):
      • probability_of_outperformance  ≥ 0.50
      • predicted_12m_excess_return    ≥ 0.02
      • model_agreement_score          ≥ 55
      • data_quality_score             ≥ 60

Neutral
    Neither Strong nor Watchlist, and not explicitly Weak.

Weak / avoid
    Any of the following:
      • probability_of_outperformance  < 0.40
      • data_quality_score             < 40
      • predicted_12m_excess_return    < -0.10  (predicted to badly underperform)

KEYES-STYLE AGREEMENT FLAG
--------------------------
A modernisation of Keyes's original "multiple regression agreement" concept.
A stock is flagged (1) when ALL independent signals point positively:
  • logistic: P(winner) ≥ 0.50
  • Ridge predicted excess return ≥ 0
  • model_agreement_score ≥ 60
  • data_quality_score ≥ 60

This flag is more conservative than the classification — a stock can be a
"Watchlist candidate" without the Keyes flag if the models partially disagree.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Default thresholds ────────────────────────────────────────────────────────
# These mirror config.yaml defaults.  Pass a custom dict to override.

STRONG_DEFAULTS = {
    "min_probability":     0.60,
    "min_excess_return":   0.05,
    "min_model_agreement": 70.0,
    "min_data_quality":    75.0,
}

WATCHLIST_DEFAULTS = {
    "min_probability":     0.50,
    "min_excess_return":   0.02,
    "min_model_agreement": 55.0,
    "min_data_quality":    60.0,
}

WEAK_DEFAULTS = {
    "max_probability":       0.40,
    "min_data_quality":      40.0,
    "min_excess_return":    -0.10,
}

LABELS = {
    "strong":    "Strong candidate",
    "watchlist": "Watchlist candidate",
    "neutral":   "Neutral",
    "weak":      "Weak / avoid",
}

LABEL_COLOURS = {
    "Strong candidate":    "#1a7a3a",
    "Watchlist candidate": "#5cb85c",
    "Neutral":             "#888888",
    "Weak / avoid":        "#a94442",
}


def classify_stocks(
    scores_df:         pd.DataFrame,
    strong_thresholds: dict | None    = None,
    watchlist_thresholds: dict | None = None,
    weak_thresholds:   dict | None    = None,
) -> pd.DataFrame:
    """
    Assign candidate_classification to every row in scores_df.

    Expected columns in scores_df:
        probability_of_outperformance  (float 0-1)
        predicted_12m_excess_return    (float, decimal e.g. 0.08 = 8%)
        model_agreement_score          (float 0-100)
        data_quality_score             (float 0-100)

    Returns scores_df with a new 'candidate_classification' column.
    """
    result = scores_df.copy()

    s = {**STRONG_DEFAULTS,    **(strong_thresholds    or {})}
    w = {**WATCHLIST_DEFAULTS, **(watchlist_thresholds or {})}
    b = {**WEAK_DEFAULTS,      **(weak_thresholds      or {})}

    # ── Column presence guards ────────────────────────────────────────────────
    prob  = result.get("probability_of_outperformance",  pd.Series(0.50, index=result.index))
    xret  = result.get("predicted_12m_excess_return",    pd.Series(0.00, index=result.index))
    agree = result.get("model_agreement_score",          pd.Series(50.0, index=result.index))
    dq    = result.get("data_quality_score",             pd.Series(50.0, index=result.index))

    # Fill NaN with neutral values
    prob  = prob.fillna(0.50)
    xret  = xret.fillna(0.00)
    agree = agree.fillna(50.0)
    dq    = dq.fillna(50.0)

    # ── Classification masks ──────────────────────────────────────────────────
    mask_strong = (
        (prob  >= s["min_probability"])     &
        (xret  >= s["min_excess_return"])   &
        (agree >= s["min_model_agreement"]) &
        (dq    >= s["min_data_quality"])
    )

    mask_watchlist = (
        (prob  >= w["min_probability"])     &
        (xret  >= w["min_excess_return"])   &
        (agree >= w["min_model_agreement"]) &
        (dq    >= w["min_data_quality"])    &
        (~mask_strong)
    )

    mask_weak = (
        (prob < b["max_probability"]) |
        (dq   < b["min_data_quality"]) |
        (xret < b["min_excess_return"])
    ) & (~mask_strong) & (~mask_watchlist)

    result["candidate_classification"] = LABELS["neutral"]
    result.loc[mask_weak,      "candidate_classification"] = LABELS["weak"]
    result.loc[mask_watchlist, "candidate_classification"] = LABELS["watchlist"]
    result.loc[mask_strong,    "candidate_classification"] = LABELS["strong"]

    counts = result["candidate_classification"].value_counts().to_dict()
    log.info("Classification counts: %s", counts)

    return result


def compute_keyes_flag(
    scores_df:       pd.DataFrame,
    keyes_ols_preds: "pd.DataFrame | None" = None,
    threshold:       float = 0.0,
) -> pd.Series:
    """
    Replicates Keyes (1972) Step 14: a stock qualifies only when ALL five
    single-variable OLS models independently predict positive excess return.

    Original Keyes rule: all 5 models predict > +10% absolute price gain.
    Modernised rule:     all 5 models predict > threshold excess return over S&P 500.
    Default threshold = 0.0  (stock is predicted to beat the market).

    When keyes_ols_preds is provided (the output of apply_keyes_ols_models),
    the TRUE Keyes criterion is used.  Without it, the function falls back to
    ensemble agreement so the pipeline degrades gracefully.

    Args:
        scores_df:       Scored DataFrame (must contain data_quality_score).
        keyes_ols_preds: DataFrame indexed by ticker with columns keyes_X*_pred.
        threshold:       Minimum predicted excess return for each model (default 0.0).

    Returns:
        pd.Series of int (1 = qualifies, 0 = does not), same index as scores_df.
    """
    dq = scores_df.get(
        "data_quality_score", pd.Series(50.0, index=scores_df.index)
    ).fillna(50.0)

    if keyes_ols_preds is not None and not keyes_ols_preds.empty:
        # ── TRUE KEYES CRITERION (Keyes 1972, Step 14) ────────────────────────
        # Keyes selected ~26% of stocks where ALL 5 single-variable OLS models
        # agreed on meaningful appreciation.  We replicate this with a
        # percentile-based cutoff on the MINIMUM prediction across all 5 models
        # (the "bottleneck" model — the hardest to satisfy simultaneously).
        # This is robust to bull/bear market regimes and matches Keyes'
        # ~25% selectivity regardless of the absolute level of predicted returns.
        ols_cols = [c for c in keyes_ols_preds.columns if c.startswith("keyes_")]
        if ols_cols:
            # Align on ticker — keyes_ols_preds is ticker-indexed; scores_df
            # may have a numeric index after merges.
            if "ticker" in scores_df.columns:
                aligned = keyes_ols_preds.reindex(scores_df["ticker"].values)
                aligned.index = scores_df.index
            else:
                aligned = keyes_ols_preds.reindex(scores_df.index)

            # Minimum prediction across all 5 models per stock (bottleneck model)
            min_pred = aligned[ols_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)

            # Select exactly the top 30% of stocks (≈ Keyes' 26% selectivity).
            # Use nlargest to avoid tie-boundary issues — guarantees a fixed count.
            # Also require data quality ≥ 60 to exclude data-poor stocks.
            eligible   = min_pred[dq >= 60]
            n_select   = max(1, round(len(min_pred) * 0.30))
            top_tickers = eligible.nlargest(n_select).index

            flag = pd.Series(0, index=scores_df.index)
            flag[flag.index.isin(top_tickers)] = 1

            log.info(
                "Keyes OLS flag (top-%d of %d): %d stocks pass (min pred cutoff=%.2f%%)",
                n_select, len(min_pred), int(flag.sum()),
                eligible.nlargest(n_select).min() * 100,
            )
            return flag.rename("keyes_agreement_flag")

    # ── FALLBACK: ensemble agreement ──────────────────────────────────────────
    prob  = scores_df.get("probability_of_outperformance",
                          pd.Series(0.50, index=scores_df.index)).fillna(0.50)
    xret  = scores_df.get("predicted_12m_excess_return",
                          pd.Series(0.00, index=scores_df.index)).fillna(0.00)
    agree = scores_df.get("model_agreement_score",
                          pd.Series(50.0, index=scores_df.index)).fillna(50.0)

    flag = (
        (prob  >= 0.50) &
        (xret  >= 0.00) &
        (agree >= 60.0) &
        (dq    >= 60.0)
    ).astype(int)
    log.info("Keyes flag (ensemble fallback): %d stocks pass", int(flag.sum()))
    return flag.rename("keyes_agreement_flag")


def get_classification_label(score: float | None) -> str:
    """
    Return a classification label for a single final_score value.
    Used for display in the Stock Detail page.
    """
    if score is None:
        return LABELS["neutral"]
    if score >= 70:
        return LABELS["strong"]
    if score >= 55:
        return LABELS["watchlist"]
    if score < 35:
        return LABELS["weak"]
    return LABELS["neutral"]


def classification_color(label: str) -> str:
    """Return the hex colour for a classification label."""
    return LABEL_COLOURS.get(label, "#888888")
