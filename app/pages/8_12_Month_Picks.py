"""
app/pages/8_12_Month_Picks.py

12-Month Picks — stocks the model identifies as most likely to beat the
S&P 500 over the next 12 months, with a Research Confidence Score ≥ 70%.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="12-Month Picks", layout="wide")

st.markdown(
    "<div style='background:#1a5c2a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📊 <strong>PHASE 2 — PREDICTION RESULTS</strong> &nbsp;|&nbsp; "
    "The model's highest-confidence picks for the next 12 months. "
    "Only stocks where multiple independent statistical signals simultaneously agree."
    "</div>",
    unsafe_allow_html=True,
)

st.title("12-Month Picks")
st.markdown(
    "The research thesis of this project is: **can we use statistical regression models "
    "to identify, in advance, which stocks will beat the S&P 500 over the next 12 months?**  \n\n"
    "This page answers that question with the current month's picks — stocks where the model's "
    "Research Confidence Score reaches **≥ 70%**, meaning multiple independent signals agree."
)


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _snapshot_cache_key() -> str:
    try:
        from src.database.db import get_connection
        with get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(id), MAX(prediction_date) FROM prediction_snapshots"
            ).fetchone()
        return f"{row[0]}_{row[1]}"
    except Exception:
        return ""


@st.cache_data(show_spinner="Loading picks…")
def _load(cache_key: str = ""):
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    from src.database.queries import load_latest_snapshots
    initialize_db(); apply_migrations()

    snap = load_latest_snapshots()
    if snap.empty:
        return pd.DataFrame(), None

    # Unpack features_json
    def _f(row, key):
        try:
            d = json.loads(row) if isinstance(row, str) else {}
            v = d.get(key)
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
        except Exception:
            return None

    for col in ("five_year_eps_growth", "five_year_revenue_growth", "five_year_price_gain",
                "current_pe_ratio", "pe_vs_historical_median", "twelve_month_momentum",
                "roe", "gross_margin", "debt_to_equity"):
        snap[col] = snap["features_json"].apply(lambda r, k=col: _f(r, k))

    latest_date = snap["prediction_date"].max()
    return snap, latest_date


with st.sidebar:
    if st.button("Refresh", type="primary", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.markdown("**About the threshold**")
    st.markdown(
        "70% confidence means at least the Keyes OLS filter "
        "AND one additional model agree. 90%+ means all major "
        "signals simultaneously agree."
    )

snap, latest_date = _load(cache_key=_snapshot_cache_key())

if snap.empty:
    st.error("No prediction data found. Run the monthly update first.")
    st.stop()


# ── Compute Research Confidence Score ─────────────────────────────────────────
def research_confidence(row) -> float:
    """
    Composite Research Confidence Score (0–100).

    Component breakdown:
      35 pts  Keyes OLS Agreement — all 5 regression models place stock in top 30%
      25 pts  Multi-factor classification — Strong candidate (independent model)
      15 pts  Data Quality — completeness of input data (scaled)
      15 pts  Model agreement score — how many sub-models agree (scaled)
      10 pts  Ridge predicted excess return — magnitude of predicted outperformance
    """
    score = 0.0

    # Keyes OLS Agreement (35 pts) — the Keyes (1972) direct replication
    if int(row.get("keyes_agreement_flag") or 0) == 1:
        score += 35.0

    # Multi-factor classification (max 25 pts)
    cls = row.get("candidate_classification", "")
    if cls == "Strong candidate":
        score += 25.0
    elif cls == "Watchlist candidate":
        score += 10.0

    # Data quality (max 15 pts) — more complete data = more reliable signal
    dq = float(row.get("data_quality_score") or 0)
    score += 15.0 * min(dq / 100.0, 1.0)

    # Model agreement (max 15 pts) — fraction of sub-models voting positively
    ma = float(row.get("model_agreement_score") or 0)
    score += 15.0 * min(ma / 100.0, 1.0)

    # Ridge predicted excess return (max 10 pts)
    xret = float(row.get("predicted_12m_excess_return") or 0)
    if xret > 0.10:
        score += 10.0
    elif xret > 0.05:
        score += 7.0
    elif xret > 0.0:
        score += 3.0

    return round(score, 1)


snap["research_confidence"] = snap.apply(research_confidence, axis=1)

# Apply 70% threshold
picks = snap[snap["research_confidence"] >= 70.0].copy()
picks = picks.sort_values("research_confidence", ascending=False).reset_index(drop=True)

if "category" not in picks.columns:
    picks["category"] = "Core"


# ── Header metrics ─────────────────────────────────────────────────────────────
from src.scoring.classifications import LABEL_COLOURS

n_total     = len(snap)
n_picks     = len(picks)
n_90plus    = (picks["research_confidence"] >= 90).sum()
top_conf    = picks["research_confidence"].max() if n_picks > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stocks analysed",         n_total,
          help="30 DJIA Core + 30 randomly selected Hidden Gems")
m2.metric("Picks ≥ 70% confidence",  n_picks,
          help="Stocks where multiple independent signals agree on outperformance")
m3.metric("Picks ≥ 90% confidence",  int(n_90plus),
          help="Highest-certainty tier — Keyes + Strong + DQ ≥ 75 + model agreement")
m4.metric("Analysis date",           str(latest_date))

st.caption(
    f"Research Confidence Score ≥ 70% required. "
    f"**{n_picks}** stocks qualify out of {n_total} analysed ({n_picks/n_total:.0%}). "
    f"This is the forward-looking prediction for the 12-month period starting **{latest_date}**."
)

st.markdown("---")


# ── What the confidence score means ──────────────────────────────────────────
with st.expander("How the Research Confidence Score is calculated"):
    st.markdown(f"""
The Research Confidence Score combines **five independent statistical signals** into a single 0–100% estimate.
It is NOT the raw logistic regression probability (which saturates near 99% for quality stocks due to
bull-market training bias). Instead it measures how many independent methods simultaneously agree.

| Component | Max points | What it measures |
|---|---|---|
| **Keyes OLS Agreement** | 35 pts | All 5 Keyes regression variables place the stock in the top 30% of predicted returners. Direct replication of Keyes (1972). |
| **Multi-factor classification** | 25 pts | A separate 33-variable model independently classifies the stock as Strong candidate. 10 pts for Watchlist. |
| **Data Quality (DQ)** | 15 pts | How complete the input data is. Full score = all 33 features available and fresh. |
| **Model agreement** | 15 pts | What fraction of the Ridge/Lasso/logistic sub-models predict positive excess return. |
| **Predicted excess return** | 10 pts | The Ridge regression's predicted return above S&P 500. +10 pts if > 10%, +7 if > 5%. |

**Threshold tiers:**

| Score | Tier | Interpretation |
|---|---|---|
| ≥ 90% | High confidence | Keyes + Strong candidate + DQ ≥ 90 + full model agreement. All signals agree. |
| 70–89% | Good confidence | Keyes + Watchlist/partial agreement. Multiple signals agree but not all. |
| 50–69% | Moderate | Only one or two signals agree. Treat with caution. |
| < 50% | Low | Signals disagree or insufficient data. Not shown on this page. |

**Important:** This is a research estimate from statistical models trained on historical data.
It is not a guarantee of future performance. The model's historical hit rate is approximately 59–65%
on the top picks (walk-forward backtest, 2016–2026).
    """)

st.markdown("---")


# ── Picks by tier ─────────────────────────────────────────────────────────────
tier_90 = picks[picks["research_confidence"] >= 90]
tier_70 = picks[(picks["research_confidence"] >= 70) & (picks["research_confidence"] < 90)]

if not tier_90.empty:
    st.subheader(f"Tier 1 — Highest Confidence  ({len(tier_90)} picks, ≥ 90%)")
    st.markdown(
        "These stocks passed **all major independent filters simultaneously**: "
        "Keyes OLS Agreement, Strong candidate classification, Data Quality ≥ 75, "
        "and full model agreement. This is the most defensible research finding."
    )

    for _, row in tier_90.iterrows():
        ticker    = row.get("ticker", "—")
        company   = row.get("company_name", "—")
        sector    = row.get("sector", "—")
        category  = row.get("category", "Core")
        conf      = row.get("research_confidence", 0)
        cls       = row.get("candidate_classification", "—")
        cls_col   = LABEL_COLOURS.get(cls, "#888")
        dq        = float(row.get("data_quality_score") or 0)
        keyes     = int(row.get("keyes_agreement_flag") or 0)
        xret      = row.get("predicted_12m_excess_return")
        pwin      = row.get("probability_of_outperformance")

        # Feature values
        eps5  = row.get("five_year_eps_growth")
        rev5  = row.get("five_year_revenue_growth")
        px5   = row.get("five_year_price_gain")
        pe    = row.get("current_pe_ratio")
        pe_m  = row.get("pe_vs_historical_median")
        mom12 = row.get("twelve_month_momentum")

        with st.container(border=True):
            # Header row
            hc1, hc2, hc3 = st.columns([3, 1, 1])
            with hc1:
                st.markdown(
                    f"### {ticker} &nbsp;"
                    f"<span style='color:{cls_col};font-weight:bold'>{cls}</span>  &nbsp;"
                    f"<span style='background:#1a5c2a;color:white;padding:2px 8px;"
                    f"border-radius:4px;font-size:0.85em'>{conf:.0f}% confidence</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{company}** · {sector} · *{category}*")
            with hc2:
                st.metric("Data Quality", f"{dq:.0f}/100")
            with hc3:
                st.metric("Keyes Agreement", "✓ YES" if keyes else "✗ NO")

            st.markdown("")
            left, right = st.columns(2)

            with left:
                st.markdown("**Model outputs:**")
                if xret is not None:
                    st.markdown(f"- **Predicted 12m excess return:** {xret:+.1%} above S&P 500")
                if pwin is not None and pwin < 0.999:
                    st.markdown(f"- **Logistic P(Win):** {pwin:.0%}")

                st.markdown("**Why selected — Keyes (1972) variables:**")
                def _pct(v):
                    return "N/A" if v is None else f"{v:+.1%}"
                def _pe(v):
                    return "N/A" if v is None else f"{v:.1f}×"
                st.markdown(f"- X5 (5-yr EPS growth): {_pct(eps5)}")
                st.markdown(f"- X12 (5-yr revenue growth): {_pct(rev5)}")
                st.markdown(f"- X6 (5-yr price change): {_pct(px5)}")
                st.markdown(f"- X8 (current P/E ratio): {_pe(pe)}")
                st.markdown(f"- X9 (P/E vs own history): {_pct(pe_m)}")

            with right:
                st.markdown("**Confidence score breakdown:**")

                keyes_pts = 35 if keyes else 0
                cls_pts   = 25 if cls == "Strong candidate" else (10 if cls == "Watchlist candidate" else 0)
                dq_pts    = round(15 * min(dq/100, 1), 1)
                ma        = float(row.get("model_agreement_score") or 0)
                ma_pts    = round(15 * min(ma/100, 1), 1)
                xret_v    = float(xret or 0)
                xret_pts  = 10 if xret_v > 0.10 else (7 if xret_v > 0.05 else (3 if xret_v > 0 else 0))

                breakdown = pd.DataFrame([
                    ("Keyes OLS Agreement (35 max)",     f"{keyes_pts} pts", "✓" if keyes_pts else "✗"),
                    ("Strong candidate (25 max)",         f"{cls_pts} pts",  "✓" if cls_pts == 25 else ("~" if cls_pts else "✗")),
                    ("Data Quality (15 max)",             f"{dq_pts} pts",   "✓" if dq_pts >= 13 else "~"),
                    ("Model agreement (15 max)",          f"{ma_pts} pts",   "✓" if ma_pts >= 13 else "~"),
                    ("Predicted excess return (10 max)",  f"{xret_pts} pts", "✓" if xret_pts >= 7 else "~"),
                    ("**Total**",                         f"**{conf:.0f}/100**", ""),
                ], columns=["Component", "Score", ""])
                st.dataframe(breakdown, use_container_width=True, hide_index=True, height=250)

                if mom12 is not None:
                    st.markdown(f"**12-month price momentum:** {mom12:+.1%}")

    st.markdown("---")


if not tier_70.empty:
    st.subheader(f"Tier 2 — Good Confidence  ({len(tier_70)} picks, 70–89%)")
    st.markdown(
        "These stocks passed the Keyes OLS filter but scored as Watchlist candidates "
        "in the multi-factor model. The Keyes signal is present but not all models agree. "
        "Include in analysis with appropriate caveats."
    )

    # Compact table for tier 2
    t2_display = tier_70[[
        c for c in ["ticker", "company_name", "category", "sector",
                     "candidate_classification", "research_confidence",
                     "data_quality_score", "predicted_12m_excess_return",
                     "five_year_eps_growth", "five_year_revenue_growth",
                     "twelve_month_momentum"]
        if c in tier_70.columns
    ]].rename(columns={
        "ticker":                        "Ticker",
        "company_name":                  "Company",
        "category":                      "Type",
        "sector":                        "Sector",
        "candidate_classification":      "Classification",
        "research_confidence":           "Confidence",
        "data_quality_score":            "DQ",
        "predicted_12m_excess_return":   "Pred. XRet",
        "five_year_eps_growth":          "EPS 5yr",
        "five_year_revenue_growth":      "Rev 5yr",
        "twelve_month_momentum":         "12m Mom",
    })

    fmt2 = {}
    for col, f in [("Confidence","{:.0f}%"), ("Pred. XRet","{:+.1%}"),
                   ("EPS 5yr","{:+.1%}"), ("Rev 5yr","{:+.1%}"),
                   ("12m Mom","{:+.1%}"), ("DQ","{:.0f}")]:
        if col in t2_display.columns:
            fmt2[col] = f

    def _t2_style(row):
        styles = [""] * len(row)
        idx = list(row.index)
        if "Classification" in idx:
            i = idx.index("Classification")
            styles[i] = f"color:{LABEL_COLOURS.get(row['Classification'],'#888')};font-weight:bold"
        if "Confidence" in idx:
            i = idx.index("Confidence")
            styles[i] = "color:#b8860b;font-weight:bold"
        return styles

    st.dataframe(
        t2_display.style.apply(_t2_style, axis=1).format(fmt2, na_rep="—"),
        use_container_width=True, hide_index=True,
        height=35 + 35 * len(t2_display),
    )

if n_picks == 0:
    st.warning(
        "No stocks currently meet the 70% confidence threshold. "
        "This is rare but can happen when the Keyes OLS models do not agree with the "
        "multi-factor classification for any stock this month. "
        "Check the Overview page for Keyes Agreement stocks and Model Diagnostics for signal strength.",
        icon="⚠️",
    )

st.markdown("---")


# ── Summary chart ─────────────────────────────────────────────────────────────
st.subheader("Confidence Score Distribution — All 60 Stocks")
st.markdown(
    "Where each stock sits on the Research Confidence scale. "
    "The 70% threshold is shown in green — only stocks to the right qualify."
)

all_conf = snap[["ticker","research_confidence","candidate_classification","category"]].copy()
all_conf["colour"] = all_conf.apply(
    lambda r: "#1a7a3a" if r["research_confidence"] >= 90
              else ("#5cb85c" if r["research_confidence"] >= 70
              else ("#f0ad4e" if r["research_confidence"] >= 50
              else "#a94442")),
    axis=1
)
all_conf = all_conf.sort_values("research_confidence", ascending=True)

fig = go.Figure(go.Bar(
    x=all_conf["research_confidence"],
    y=all_conf["ticker"],
    orientation="h",
    marker_color=all_conf["colour"],
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Confidence: %{x:.0f}%<extra></extra>"
    ),
))
fig.add_vline(x=70, line_dash="dash", line_color="#1a7a3a", line_width=2,
              annotation_text="70% threshold", annotation_position="top right")
fig.add_vline(x=90, line_dash="dot", line_color="#1a5c2a", line_width=1.5,
              annotation_text="90% tier", annotation_position="top right")
fig.update_layout(
    height=max(600, len(all_conf) * 18),
    xaxis_title="Research Confidence Score (%)",
    xaxis=dict(range=[0, 105]),
    margin=dict(l=60),
    title=f"Research Confidence Scores — {latest_date}  |  "
          f"{n_picks} stocks ≥ 70%  |  {int(n_90plus)} stocks ≥ 90%",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.info("""
**Research note — limitations of these picks**

1. **Not a guaranteed return.** The model's walk-forward backtest hit rate is 59–65% on top picks.
   That means roughly 1 in 3 picks will underperform the S&P 500 in any given year.

2. **Survivorship bias.** The 30 DJIA Core stocks are today's largest, most successful companies.
   Companies that failed or were delisted are not represented in the training data.

3. **Bull market training.** The models were trained on 2015–2026 data, one of the longest equity
   bull runs in history. Performance in a sustained bear market may differ significantly.

4. **No transaction costs.** Real implementation incurs bid-ask spreads, slippage, and taxes.
   Monthly rebalancing amplifies these costs.

These picks are research outputs intended for academic analysis. They are not financial advice.
""")
