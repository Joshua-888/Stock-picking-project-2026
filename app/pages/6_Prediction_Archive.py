"""
app/pages/6_Prediction_Archive.py

Prediction Archive — the historical record of every prediction the model has made.

WHY THIS PAGE EXISTS
--------------------
Backtests look good by construction — they're computed after the fact on
historical data the researcher already knows.  The prediction archive is
different: it records what the model actually said at the time of prediction.

When the 12-month horizon elapses, we look up what actually happened and
compare.  This is the only honest measure of whether the model works.

The archive answers:
  • What did the model predict for each stock each month?
  • Which predictions were correct?
  • Do Strong candidates actually outperform Neutral stocks?
  • Did the model deteriorate over time?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction Archive", layout="wide")

st.markdown(
    "<div style='background:#5c3a1a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📈 <strong>PHASE 3 — RESEARCH RESULTS</strong> &nbsp;|&nbsp; "
    "Every prediction ever made, with actual outcomes where the 12-month window has closed. "
    "This is the auditable record of model accuracy over time."
    "</div>",
    unsafe_allow_html=True,
)

st.title("Prediction Archive")
st.markdown(
    "Every prediction the model has ever made, stored permanently and immutably. "
    "This is the gold-standard test of the model: what did it actually say at the time, "
    "and what happened afterwards?"
)

with st.expander("Why this archive matters for research validity"):
    st.markdown("""
**The difference between backtesting and a live archive**

A backtest computes what *would have happened* using historical data. It is computed after the
fact and is subject to look-ahead bias if not designed carefully.

The Prediction Archive is fundamentally different: it records what the model *actually predicted*
at each month-end, before the outcomes were known. When the 12-month window closes, the actual
outcome is recorded and linked. This is the only honest measure of model performance.

**How to read the Keyes flag in the archive**

The Keyes Agreement flag changed methodology in May 2026:
- **Before May 2026**: Ensemble-based flag (logistic + Ridge agreement)
- **From May 2026 onwards**: True Keyes OLS flag (top 30% by minimum prediction across 5 single-variable models)

Do not compare Keyes flag counts across these two periods directly.

**Statistical note on realized performance**

Only 2 prediction dates have closed 12-month horizons. With 60 predictions and a 58.3% hit rate,
the 95% confidence interval is approximately **[45%, 71%]**. More months are needed before
the realized hit rate is statistically conclusive.
    """)

st.info(
    "**Snapshots are written once and never changed.** "
    "They capture exactly what the model knew and predicted at the time of writing. "
    "Realised outcomes appear automatically once the 12-month horizon elapses.",
    icon="📋",
)


def _cache_key() -> str:
    try:
        from src.database.db import get_connection
        with get_connection() as conn:
            r = conn.execute("SELECT MAX(id), MAX(prediction_date) FROM prediction_snapshots").fetchone()
        return f"{r[0]}_{r[1]}"
    except Exception:
        return ""


# ── Load archive from DB ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading prediction archive…")
def _load_archive(cache_key: str = ""):
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    from src.database.queries import (load_prediction_snapshots, load_realized_performance,
                                      load_stocks)

    initialize_db(); apply_migrations()
    snapshots   = load_prediction_snapshots()
    realized    = load_realized_performance()
    stocks_meta = load_stocks()
    return snapshots, realized, stocks_meta


snapshots_df, realized_df, stocks_meta = _load_archive(cache_key=_cache_key())

if snapshots_df.empty:
    st.warning("No prediction snapshots found. Run the monthly update to generate predictions.")
    st.stop()


# ── Summary metrics ────────────────────────────────────────────────────────────
n_snapshots  = len(snapshots_df)
n_dates      = snapshots_df["prediction_date"].nunique()
n_tickers    = snapshots_df["ticker"].nunique()
n_strong     = (snapshots_df["candidate_classification"] == "Strong candidate").sum()
n_keyes      = snapshots_df["keyes_agreement_flag"].sum()
latest_date  = snapshots_df["prediction_date"].max()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total snapshots",    f"{n_snapshots:,}")
c1.caption("One row per (ticker, month) — permanent and immutable")
c2.metric("Prediction dates",   n_dates)
c2.caption("Distinct months in which the model ran")
c3.metric("Unique tickers",     n_tickers)
c3.caption("Different stocks ever scored")
c4.metric("Strong candidates",  int(n_strong))
c4.caption("Across all prediction dates combined")
c5.metric("Keyes flag set",     int(n_keyes))
c5.caption("Note: criterion changed in May 2026 — see methodology note")
c6.metric("Latest snapshot",    pd.to_datetime(latest_date).strftime("%b %Y"))
c6.caption("Most recent prediction month")

st.markdown("---")


# ── Filters ────────────────────────────────────────────────────────────────────
st.subheader("Filter Snapshots")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    all_dates_list = sorted(snapshots_df["prediction_date"].unique(), reverse=True)
    sel_dates = st.multiselect(
        "Prediction date(s):", all_dates_list,
        default=[all_dates_list[0]] if all_dates_list else [],
    )

with col_f2:
    all_cls = ["All"] + sorted(snapshots_df["candidate_classification"].dropna().unique().tolist())
    sel_cls = st.selectbox("Classification:", all_cls)

with col_f3:
    all_sectors = ["All"] + sorted(snapshots_df["sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector:", all_sectors)

with col_f4:
    min_score = st.slider("Min final score:", 0, 100, 0)

# Apply filters
filtered = snapshots_df.copy()
if sel_dates:
    filtered = filtered[filtered["prediction_date"].isin(sel_dates)]
if sel_cls != "All":
    filtered = filtered[filtered["candidate_classification"] == sel_cls]
if sel_sector != "All":
    filtered = filtered[filtered["sector"] == sel_sector]
filtered = filtered[filtered["final_score"].fillna(0) >= min_score]
filtered = filtered.sort_values(["prediction_date", "final_score"], ascending=[False, False])

st.caption(f"Showing **{len(filtered):,}** of {n_snapshots:,} snapshots")


# ── Snapshot table ─────────────────────────────────────────────────────────────
st.subheader("Prediction Snapshots")

from src.scoring.classifications import LABEL_COLOURS

display_cols = {
    "prediction_date":               "Date",
    "ticker":                        "Ticker",
    "company_name":                  "Company",
    "sector":                        "Sector",
    "probability_of_outperformance": "P(Win)",
    "predicted_12m_excess_return":   "Pred. XRet",
    "model_agreement_score":         "Agreement",
    "data_quality_score":            "DQ",
    "final_score":                   "Score",
    "candidate_classification":      "Classification",
    "keyes_agreement_flag":          "Keyes",
}

disp = filtered[[c for c in display_cols if c in filtered.columns]].rename(columns=display_cols)

def _cls_colour(val):
    c = LABEL_COLOURS.get(val, "#888")
    return f"color: {c}; font-weight: bold"

st.dataframe(
    disp.style
        .map(_cls_colour, subset=["Classification"])
        .format({
            "P(Win)":       "{:.1%}",
            "Pred. XRet":   "{:+.2%}",
            "Agreement":    "{:.0f}",
            "DQ":           "{:.0f}",
            "Score":        "{:.1f}",
        }, na_rep="—"),
    use_container_width=True,
    height=500,
)

st.markdown("---")


# ── Single snapshot deep-dive ──────────────────────────────────────────────────
st.subheader("Snapshot Deep-Dive")
st.markdown("Select a snapshot to see the full feature values and model inputs at the time of prediction.")

# Build picker options
if not filtered.empty:
    filtered["_label"] = filtered["prediction_date"] + " — " + filtered["ticker"] + \
                         " (" + filtered["candidate_classification"].fillna("?") + ")"
    snap_options = filtered["_label"].tolist()
    sel_snap_label = st.selectbox("Select snapshot:", snap_options)
    sel_snap_row   = filtered[filtered["_label"] == sel_snap_label].iloc[0]

    # ── Header metrics ──────────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Ticker",          sel_snap_row.get("ticker"))
    s2.metric("Date",            sel_snap_row.get("prediction_date"))
    s3.metric("P(Outperform)",   f"{sel_snap_row.get('probability_of_outperformance', 0):.1%}")
    s4.metric("Final Score",     f"{sel_snap_row.get('final_score', 0):.1f}")
    s5.metric("Classification",  sel_snap_row.get("candidate_classification", "—"))

    # ── Score breakdown ─────────────────────────────────────────────────────────
    score_components = {
        "Probability":    sel_snap_row.get("probability_of_outperformance", 0.5) * 100,
        "Model Agreement": sel_snap_row.get("model_agreement_score", 50),
        "Risk":            sel_snap_row.get("risk_score", 50),
        "Valuation":       sel_snap_row.get("valuation_score", 50),
        "Momentum":        sel_snap_row.get("momentum_score", 50),
        "Fundamentals":    sel_snap_row.get("fundamental_score", 50),
        "Data Quality":    sel_snap_row.get("data_quality_score", 50),
    }
    sc_df = pd.DataFrame(list(score_components.items()), columns=["Component", "Score"])
    fig_sc = px.bar(sc_df, x="Score", y="Component", orientation="h",
                    color="Score",
                    color_continuous_scale=["#a94442", "#f0ad4e", "#1a7a3a"],
                    color_continuous_midpoint=50,
                    range_x=[0, 100],
                    title=f"Score components — {sel_snap_row.get('ticker')} on {sel_snap_row.get('prediction_date')}")
    fig_sc.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Feature values at prediction time ───────────────────────────────────────
    features_json = sel_snap_row.get("features_json")
    if features_json:
        try:
            feat_dict = json.loads(features_json)
            feat_df = pd.DataFrame(list(feat_dict.items()), columns=["Feature", "Value at Prediction"])
            feat_df = feat_df.dropna(subset=["Value at Prediction"])
            feat_df["Value at Prediction"] = feat_df["Value at Prediction"].apply(
                lambda x: round(float(x), 4) if x is not None else None
            )

            left_col, right_col = st.columns(2)
            half = len(feat_df) // 2
            with left_col:
                st.markdown("**Feature values (as of prediction date)**")
                st.dataframe(feat_df.iloc[:half], use_container_width=True, height=400)
            with right_col:
                st.markdown("&nbsp;")
                st.dataframe(feat_df.iloc[half:], use_container_width=True, height=400)
        except Exception:
            st.info("Feature values not available for this snapshot.")

    # ── Warnings ────────────────────────────────────────────────────────────────
    warnings_json = sel_snap_row.get("warnings_json")
    if warnings_json:
        try:
            warns = json.loads(warnings_json)
            if warns:
                st.subheader("Warnings at time of prediction")
                for w in warns:
                    st.warning(w, icon="⚠️")
        except Exception:
            pass

    # ── Keyes flag explanation ──────────────────────────────────────────────────
    keyes = sel_snap_row.get("keyes_agreement_flag", 0)
    if keyes:
        st.success(
            "**Keyes-style agreement flag is set.** "
            "All major models agreed on positive excess return or outperformance "
            "probability at the time of this prediction.",
            icon="✅",
        )
    else:
        st.info(
            "Keyes flag is not set — models did not fully agree on positive outperformance.",
            icon="ℹ️",
        )

st.markdown("---")


# ── Classification over time chart ────────────────────────────────────────────
st.subheader("Classification Trends Over Time")
st.markdown("How many stocks fell into each category each month?")

cls_trend = (
    snapshots_df.groupby(["prediction_date", "candidate_classification"])
    .size().reset_index(name="count")
)
fig_trend = px.bar(
    cls_trend,
    x="prediction_date", y="count",
    color="candidate_classification",
    color_discrete_map=LABEL_COLOURS,
    title="Classification distribution across prediction dates",
    labels={"prediction_date": "Prediction Date", "count": "Number of stocks"},
    barmode="stack",
)
fig_trend.update_layout(height=350, legend=dict(orientation="h", y=-0.3))
st.plotly_chart(fig_trend, use_container_width=True)


# ── Average score over time ────────────────────────────────────────────────────
avg_score = snapshots_df.groupby("prediction_date")["final_score"].mean().reset_index()
avg_score.columns = ["Date", "Avg Final Score"]

fig_avg = px.line(avg_score, x="Date", y="Avg Final Score",
                  title="Average final score across universe over time",
                  markers=True)
fig_avg.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="Neutral = 50")
fig_avg.update_layout(height=280)
st.plotly_chart(fig_avg, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 16 — REALISED PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Realised Performance — Did the Predictions Work?")
st.markdown("""
For predictions where the 12-month horizon has elapsed, we compare what the model predicted
with what actually happened.  This is the honest, out-of-sample test of the scoring model.

**Realised performance rows available:** the 12-month forward window is computed using
actual prices — no look-ahead, no reconstruction after the fact.
""")

from src.scoring.realized_performance import (
    compute_accuracy_summary, compute_accuracy_by_classification,
    compute_calibration_by_bucket, compute_rolling_hit_rate,
)

if realized_df.empty:
    st.info(
        "No realised performance data yet. "
        "This section will populate once 12 months have elapsed after a prediction date. "
        "With sample data running to 2026, predictions from before 2025-04 have realised outcomes.",
        icon="⏳",
    )
else:
    # ── Summary metrics ────────────────────────────────────────────────────────
    summ = compute_accuracy_summary(realized_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predictions evaluated",   f"{summ.get('n_predictions', 0):,}")
    c2.metric("Correct predictions",     f"{summ.get('n_correct', 0):,}")
    c3.metric("Overall hit rate",
              f"{summ.get('overall_hit_rate', 0):.1%}" if summ.get('overall_hit_rate') else "—")
    c4.metric("Actual winner rate",
              f"{summ.get('actual_winner_rate', 0):.1%}" if summ.get('actual_winner_rate') else "—")
    c5.metric("Mean actual excess return",
              f"{summ.get('mean_actual_excess_return', 0):+.2%}" if summ.get('mean_actual_excess_return') else "—")

    st.caption(
        f"Date range: {summ.get('date_range', '—')}  |  "
        f"Mean prediction error: {summ.get('mean_return_error', 0):+.4f}"
        if summ.get('mean_return_error') is not None else ""
    )
    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Hit rate by classification ─────────────────────────────────────────────
    with col_l:
        st.subheader("Hit Rate by Classification")
        st.markdown(
            "The key test: do **Strong candidates** actually beat the benchmark "
            "more often than **Neutral** or **Weak** stocks?"
        )
        acc_cls = compute_accuracy_by_classification(realized_df, snapshots_df)
        if not acc_cls.empty:
            from src.scoring.classifications import LABEL_COLOURS
            fig_cls = px.bar(
                acc_cls,
                x="candidate_classification",
                y="hit_rate",
                color="candidate_classification",
                color_discrete_map=LABEL_COLOURS,
                text=acc_cls["hit_rate"].apply(lambda v: f"{v:.1%}"),
                title="Hit rate by classification",
                labels={"candidate_classification": "Classification",
                        "hit_rate": "Prediction hit rate"},
            )
            fig_cls.add_hline(y=0.5, line_dash="dash", line_color="red",
                              annotation_text="Random = 50%")
            fig_cls.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_cls, use_container_width=True)

            st.dataframe(
                acc_cls[["candidate_classification", "n", "hit_rate",
                          "actual_winner_rate", "mean_actual_xret",
                          "mean_predicted_prob"]].rename(columns={
                    "candidate_classification": "Classification",
                    "n": "N", "hit_rate": "Hit Rate",
                    "actual_winner_rate": "Actual Win Rate",
                    "mean_actual_xret": "Mean Actual XRet",
                    "mean_predicted_prob": "Mean Pred Prob",
                }).style.format({
                    "Hit Rate": "{:.1%}", "Actual Win Rate": "{:.1%}",
                    "Mean Actual XRet": "{:+.2%}", "Mean Pred Prob": "{:.1%}",
                }, na_rep="—"),
                use_container_width=True, height=200,
            )

    # ── Calibration by probability bucket ─────────────────────────────────────
    with col_r:
        st.subheader("Calibration — Predicted vs Actual Win Rate")
        st.markdown(
            "A well-calibrated model: when it predicts 70% probability, "
            "~70% of those stocks should actually win."
        )
        cal_bkt = compute_calibration_by_bucket(realized_df)
        if not cal_bkt.empty:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(
                x=cal_bkt["probability_bucket"],
                y=cal_bkt["actual_win_rate"],
                marker_color="#1f77b4",
                name="Actual win rate",
                text=cal_bkt["actual_win_rate"].apply(lambda v: f"{v:.1%}"),
                textposition="outside",
            ))
            fig_cal.add_hline(y=0.5, line_dash="dash", line_color="red",
                              annotation_text="50% baseline")
            fig_cal.update_layout(
                height=350, title="Actual win rate by predicted probability bucket",
                xaxis_title="Probability bucket",
                yaxis_title="Actual win rate",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # ── Prediction vs actual scatter ───────────────────────────────────────────
    st.subheader("Predicted vs Actual Excess Return")
    # Join predicted return from snapshots if not already in realized_df
    if "predicted_12m_excess_return" not in realized_df.columns:
        pred_col = snapshots_df[["ticker","prediction_date","predicted_12m_excess_return"]].drop_duplicates()
        realized_df = realized_df.merge(pred_col, on=["ticker","prediction_date"], how="left")
    valid_scatter = realized_df.dropna(
        subset=["predicted_12m_excess_return", "realized_12m_excess_return"]
    )
    if not valid_scatter.empty:
        fig_scat = px.scatter(
            valid_scatter,
            x="predicted_12m_excess_return",
            y="realized_12m_excess_return",
            opacity=0.5,
            color="prediction_correct",
            color_discrete_map={1: "#1a7a3a", 0: "#a94442"},
            labels={
                "predicted_12m_excess_return":  "Predicted 12m Excess Return",
                "realized_12m_excess_return":   "Actual 12m Excess Return",
                "prediction_correct":           "Correct?",
            },
            title="Predicted vs actual excess return  (green = correct direction, red = wrong)",
            hover_data=["ticker", "prediction_date"],
        )
        # Add quadrant lines
        fig_scat.add_hline(y=0, line_color="grey", line_width=0.8)
        fig_scat.add_vline(x=0, line_color="grey", line_width=0.8)
        fig_scat.update_layout(height=400)
        st.plotly_chart(fig_scat, use_container_width=True)
        st.caption(
            "Top-right = correctly predicted winner (both positive). "
            "Bottom-left = correctly predicted loser (both negative). "
            "Off-diagonal = wrong direction."
        )

    # ── Rolling hit rate over time ─────────────────────────────────────────────
    st.subheader("Rolling Hit Rate Over Time")
    roll_hr = compute_rolling_hit_rate(realized_df)
    if not roll_hr.empty and len(roll_hr) >= 2:
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(
            x=roll_hr["prediction_date"], y=roll_hr["hit_rate"],
            name="Monthly hit rate",
            marker_color=roll_hr["hit_rate"].apply(
                lambda v: "#1a7a3a" if v >= 0.5 else "#a94442"
            ),
            opacity=0.6,
        ))
        fig_hr.add_trace(go.Scatter(
            x=roll_hr["prediction_date"], y=roll_hr["rolling_hit_rate"],
            mode="lines", name="6m rolling avg",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_hr.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="50% baseline")
        fig_hr.update_layout(
            height=320, title="Hit rate over time (green ≥ 50%, red < 50%)",
            xaxis_title="Prediction date", yaxis_title="Hit rate",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    # ── Detailed realized table ────────────────────────────────────────────────
    st.subheader("Detailed Realised Performance Table")
    real_disp = realized_df.copy()
    real_disp = real_disp.merge(
        snapshots_df[["prediction_date", "ticker", "candidate_classification",
                      "final_score", "probability_of_outperformance"]].drop_duplicates(),
        on=["prediction_date", "ticker"], how="left",
    )
    real_disp = real_disp.sort_values(["prediction_date", "ticker"], ascending=[False, True])

    st.dataframe(
        real_disp[[
            "prediction_date", "evaluation_date", "ticker",
            "probability_of_outperformance", "candidate_classification",
            "winner_predicted", "winner_actual", "prediction_correct",
            "realized_12m_excess_return", "return_error",
        ]].rename(columns={
            "prediction_date":               "Pred Date",
            "evaluation_date":               "Eval Date",
            "probability_of_outperformance": "P(Win)",
            "candidate_classification":      "Classification",
            "winner_predicted":              "Pred Winner",
            "winner_actual":                 "Actual Winner",
            "prediction_correct":            "Correct?",
            "realized_12m_excess_return":    "Actual XRet",
            "return_error":                  "Return Error",
        }).style.format({
            "P(Win)":        "{:.1%}",
            "Actual XRet":   "{:+.2%}",
            "Return Error":  "{:+.4f}",
        }, na_rep="—"),
        use_container_width=True,
        height=500,
    )

    with st.expander("How to interpret realised performance"):
        st.markdown("""
**Hit rate**
Fraction of predictions where the direction was correct (predicted winner = actual winner).
Random guessing gives 50%.  A useful model should consistently exceed 52–55%.

**Calibration**
If stocks in the "0.60–0.70 probability bucket" actually win 65% of the time,
the model is well-calibrated.  If they only win 48%, the model is overconfident.

**Return error**
predicted_excess_return − actual_excess_return.
A mean error near zero means the model is unbiased.
Systematic positive error = the model over-predicts excess return.

**Why the scatter plot matters**
The top-right quadrant (both predicted and actual positive) = correctly identified winners.
The bottom-left quadrant (both negative) = correctly identified losers.
The off-diagonal quadrants = prediction errors in direction.
A useful model has more dots in the diagonal quadrants.

**Rolling hit rate**
If the hit rate is declining over time, the model is deteriorating — possibly
because market conditions changed and the historical patterns no longer hold.
This is an early warning signal to retrain the model.

**Key limitation**
With 30 stocks, each monthly cross-section has very few observations.
Hit rates will be volatile.  At least 6–12 months of data are needed
for stable estimates.
        """)
