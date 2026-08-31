"""
app/dashboard.py  —  Streamlit entry point.
Run with:  streamlit run app/dashboard.py
"""

import streamlit as st

st.set_page_config(
    page_title="Statistical Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Statistical Stock Analysis Dashboard")
st.caption("A modern replication of Keyes (1972) — identifying stocks likely to outperform the S&P 500")

st.markdown("---")

# ── One-sentence summary ──────────────────────────────────────────────────────
st.markdown("""
Every month, this system analyses **60 stocks** using five statistical methods trained on
10+ years of real market data, and identifies which stocks are most likely to beat the
S&P 500 over the next 12 months — replicating the methodology from Keyes (1972).
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# THE THREE PHASES
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("How to navigate this dashboard")
st.markdown(
    "The dashboard is organised into **three research phases**. "
    "Start with Phase 1 to understand what the models are built on, "
    "then go to Phase 2 to see this month's results."
)

st.markdown("")

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='background:#1a3a5c;color:white;padding:8px 14px;border-radius:4px;"
        "font-weight:bold;font-size:1.0em'>🔬 PHASE 1 — SIGNAL VALIDATION</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "**Question answered:** *Do the financial variables actually predict stock performance, "
        "or are we fitting random noise?*"
    )
    st.markdown(
        "Before any stock is ranked or scored, the system runs rigorous statistical tests "
        "on all 33 candidate variables. Only variables that survive these tests are used in Phase 2. "
        "This is the scientific foundation of the entire study."
    )
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Spearman Rank Correlation (IC)**")
        st.caption(
            "Measures how well each variable ranks stocks in the correct order month by month. "
            "IC > 0.05 = useful signal. IC > 0.10 = strong signal."
        )
    with col2:
        st.markdown("**Benjamini-Hochberg Correction**")
        st.caption(
            "When testing 33 variables simultaneously, ~2 will appear significant by pure chance. "
            "BH correction controls the false discovery rate — only genuinely significant "
            "variables pass."
        )
    with col3:
        st.markdown("**Simple OLS Regression**")
        st.caption(
            "One regression per variable. Shows the direction (positive/negative) and strength "
            "of each variable's relationship to future 12-month excess returns. "
            "Replicates Keyes (1972) directly."
        )
    with col4:
        st.markdown("**Multiple Regression + VIF**")
        st.caption(
            "All variables enter a single model simultaneously. VIF filtering removes "
            "multicollinear variables. Shows which variables remain predictive "
            "after controlling for all others."
        )

    st.markdown("")
    st.info("**→ Page: Model Diagnostics** (first item in the sidebar)", icon="🔬")

st.markdown("")

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='background:#1a5c2a;color:white;padding:8px 14px;border-radius:4px;"
        "font-weight:bold;font-size:1.0em'>📊 PHASE 2 — PREDICTION RESULTS</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "**Question answered:** *Which stocks does the model predict will outperform the S&P 500 "
        "over the next 12 months?*"
    )
    st.markdown(
        "Using the validated signals from Phase 1, four prediction models are trained and applied "
        "to this month's financial data. Stocks are ranked, scored, and classified. "
        "The primary output is the **Keyes Agreement Flag** — stocks where all five "
        "Keyes regression models simultaneously agree on outperformance."
    )
    st.markdown("")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**🔑 Keyes OLS (5 models)**")
        st.caption(
            "Five independent OLS regressions — one per Keyes variable. "
            "A stock earns the Keyes Agreement Flag if its minimum predicted return "
            "across all five models places it in the top 30% of the universe. "
            "This is the direct Keyes (1972) replication."
        )
    with col_b:
        st.markdown("**Ridge Regression**")
        st.caption(
            "Predicts 12-month excess return using all 33 variables with L2 regularisation "
            "to prevent overfitting. Contributes 25% of the final score. "
            "Selected by time-series cross-validation."
        )
    with col_c:
        st.markdown("**Logistic Regression**")
        st.caption(
            "Predicts P(stock beats S&P 500) as a probability. Contributes 35% of the final score. "
            "Note: probabilities saturate for quality stocks — used for ranking within groups, "
            "not as a literal absolute probability."
        )

    st.markdown("")

    pc1, pc2, pc3 = st.columns(3)
    pc1.success("**Overview** — this month's top picks, Keyes Agreement stocks, and high-confidence intersection picks")
    pc2.success("**Stock Screener** — filter all 60 stocks by any metric, sector, or classification")
    pc3.success("**Stock Detail** — deep-dive on one stock: all 33 feature values, signal verdicts, and prediction history")

st.markdown("")

# ── PHASE 3 ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='background:#5c3a1a;color:white;padding:8px 14px;border-radius:4px;"
        "font-weight:bold;font-size:1.0em'>📈 PHASE 3 — RESEARCH RESULTS & VALIDATION</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "**Question answered:** *How has the model actually performed? Are the predictions "
        "turning out to be correct when the 12-month window closes?*"
    )
    st.markdown(
        "These pages provide the auditable research record — historical performance, "
        "every prediction ever made with its actual outcome, and data quality logs."
    )
    st.markdown("")

    pp1, pp2, pp3 = st.columns(3)
    pp1.warning("**Backtesting** — walk-forward portfolio simulation vs. S&P 500. No hindsight — models retrained only on data available before each prediction date.")
    pp2.warning("**Prediction Archive** — every prediction ever made, with actual 12-month outcomes where the window has closed. The auditable accuracy record.")
    pp3.warning("**Data Quality** — completeness and reliability of input data. Stocks with DQ < 75 are flagged — their predictions were made with incomplete information.")

st.markdown("---")

# ── The study design ──────────────────────────────────────────────────────────
st.subheader("Study design — replicating Keyes (1972)")

d1, d2 = st.columns(2)
with d1:
    st.markdown("""
**Two stock groups — matching Keyes' original design**

| Group | This study | Keyes (1972) |
|---|---|---|
| Fixed group | DJIA 30 (actual components, Nov 2024) | Dow Jones 30 Industrials |
| Random group | 30 randomly drawn from mid-cap pool each month | 30 randomly selected from Forbes listings |

Keyes' central thesis: the same five regression variables predict outperformance equally well
for both the famous fixed group and randomly selected unknowns.
This study tests that thesis with 2015–2026 data.
    """)

with d2:
    st.markdown("""
**The five Keyes variables**

| Variable | Code | What it measures |
|---|---|---|
| 5-year EPS growth | X5 | Earnings quality and consistency |
| 5-year price gain | X6 | Market-validated performance history |
| Current P/E ratio | X8 | Valuation relative to earnings |
| P/E vs own history | X9 | Cheap or expensive vs own past? |
| 5-year revenue growth | X12 | Top-line business momentum |

A stock earns the **Keyes Agreement Flag** when all five models simultaneously
place it in the top 30% of predicted returners.
Current result: **18/60 = 30%** — consistent with Keyes' original 26–30%.
    """)

st.markdown("---")

st.caption(
    "Research tool only — not financial advice. "
    "All outputs are probabilistic estimates based on historical patterns. "
    "Past statistical relationships do not guarantee future performance."
)
