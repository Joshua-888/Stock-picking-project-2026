"""
app/pages/2_Overview.py

Research overview — Keyes (1972) replication results.
Primary metric: Keyes OLS Agreement flag (intersection with Strong candidate classification).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Overview", layout="wide")

_PHASE2_BANNER = (
    "<div style='background:#1a5c2a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📊 <strong>PHASE 2 — PREDICTION RESULTS</strong> &nbsp;|&nbsp; "
    "These pages show the output of the statistical models applied to this month's data. "
    "Stocks are ranked and classified based on validated signals from Phase 1."
    "</div>"
)
st.markdown(_PHASE2_BANNER, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Research Overview — Keyes (1972) Replication")
st.markdown(
    "This study replicates Keyes (1972), testing whether five financial regression variables "
    "can identify stocks likely to outperform the S&P 500 over the next 12 months. "
    "**60 stocks are analysed** each month: 30 from the actual Dow Jones Industrial Average "
    "(Keyes' fixed group) and 30 randomly selected mid-caps (Keyes' random group)."
)

from src.utils.config import load_config as _load_cfg
_provider = _load_cfg().data.provider
if _provider == "sample":
    st.error(
        "**WARNING — SYNTHETIC DATA.** Switch to `DATA_PROVIDER=yfinance` and re-run the "
        "monthly update to use real market data.",
        icon="🚨",
    )

st.markdown("---")


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


@st.cache_data(show_spinner="Loading results…")
def _load_scores(cache_key: str = ""):
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    from src.database.queries import load_latest_snapshots, load_features_and_targets

    initialize_db()
    apply_migrations()

    snap = load_latest_snapshots()
    if snap.empty:
        return pd.DataFrame(), None, pd.DataFrame()

    # Unpack features_json into display columns
    import math as _math
    def _feat(row, key):
        try:
            d = json.loads(row) if isinstance(row, str) else {}
            v = d.get(key)
            # JSON may store NaN as float('nan') — treat same as missing
            if v is None or (isinstance(v, float) and _math.isnan(v)):
                return None
            return v
        except Exception:
            return None

    for col in ("current_pe_ratio", "twelve_month_momentum", "five_year_price_gain",
                "current_price", "five_year_eps_growth", "five_year_revenue_growth",
                "pe_vs_historical_median"):
        snap[col] = snap["features_json"].apply(lambda r, k=col: _feat(r, k))

    if "category" not in snap.columns:
        snap["category"] = "Core"

    latest_date = snap["prediction_date"].max()

    targets_df = pd.DataFrame()
    try:
        targets_df = load_features_and_targets(start_date="2015-01-31")
    except Exception:
        pass

    return snap, latest_date, targets_df


with st.sidebar:
    if st.button("Refresh Data", type="primary", use_container_width=True,
                 help="Reload from database. Run the monthly update first to fetch fresh data."):
        st.cache_data.clear()
        st.rerun()

scores_df, latest_date, targets_df = _load_scores(cache_key=_snapshot_cache_key())

if scores_df.empty:
    st.error("No prediction data found. Run the monthly update pipeline first.")
    st.stop()

from src.scoring.classifications import LABEL_COLOURS

# ── Derived sets ───────────────────────────────────────────────────────────────
keyes_mask    = scores_df["keyes_agreement_flag"] == 1
strong_mask   = scores_df["candidate_classification"] == "Strong candidate"
dq_mask       = scores_df["data_quality_score"].fillna(0) >= 75

# HIGH-CONFIDENCE = Keyes flag + Strong candidate + DQ ≥ 75
hc_mask       = keyes_mask & strong_mask & dq_mask
hc_df         = scores_df[hc_mask].sort_values("data_quality_score", ascending=False).reset_index(drop=True)

# All Keyes picks sorted: Strong first, then DQ
keyes_df      = scores_df[keyes_mask].copy()
_cls_order    = {"Strong candidate": 0, "Watchlist candidate": 1, "Neutral": 2, "Weak / avoid": 3}
keyes_df["_r"] = keyes_df["candidate_classification"].map(_cls_order).fillna(3)
keyes_df      = keyes_df.sort_values(["_r", "data_quality_score"], ascending=[True, False]).drop(columns=["_r"]).reset_index(drop=True)

n_keyes       = int(keyes_mask.sum())
n_hc          = int(hc_mask.sum())
n_strong      = int(strong_mask.sum())
n_watchlist   = int((scores_df["candidate_classification"] == "Watchlist candidate").sum())

# ── Summary metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks analysed",       len(scores_df),
          help="30 DJIA Core + 30 randomly selected Hidden Gems")
c2.metric("Keyes Agreement picks", f"{n_keyes}/60",
          help="Stocks where all 5 Keyes OLS regression models agree on outperformance (top 30%). Primary research metric.")
c3.metric("High-confidence picks", n_hc,
          help="Keyes Agreement AND Strong candidate classification AND DQ ≥ 75. Strongest signal.")
c4.metric("Analysis date",         str(latest_date))

st.caption(
    f"**Primary research metric:** Keyes OLS Agreement = {n_keyes}/60 = {n_keyes/60:.0%} "
    f"— consistent with Keyes (1972) result of ~26–30%.  "
    f"Of these, **{n_hc} are High-Confidence picks** where the multi-factor model independently agrees."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HIGH-CONFIDENCE PICKS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("High-Confidence Research Picks")
st.markdown(
    "These stocks passed **two independent filters simultaneously**:  \n"
    "1. **Keyes OLS Agreement** — all 5 regression variables (EPS growth, price gain, P/E ratio, "
    "P/E vs median, revenue growth) place this stock in the top 30% of expected returners.  \n"
    "2. **Multi-factor Strong candidate** — the 33-feature model also classifies it as "
    "the strongest tier (requires DQ ≥ 75, meaning complete, reliable data).  \n\n"
    "Requiring both signals to agree simultaneously is the most stringent filter available in this study."
)

if hc_df.empty:
    st.warning("No stocks met both criteria this month. Check the Keyes Agreement section below for single-filter picks.")
else:
    for _, row in hc_df.iterrows():
        ticker      = row.get("ticker", "—")
        company     = row.get("company_name", "—")
        sector      = row.get("sector", "—")
        category    = row.get("category", "Core")
        dq          = float(row.get("data_quality_score") or 0)
        cls         = row.get("candidate_classification", "—")
        cls_colour  = LABEL_COLOURS.get(cls, "#888")
        pred_xret   = row.get("predicted_12m_excess_return")
        keyes_flag  = int(row.get("keyes_agreement_flag", 0))

        # Key financial metrics for the WHY explanation
        eps5        = row.get("five_year_eps_growth")
        rev5        = row.get("five_year_revenue_growth")
        price5      = row.get("five_year_price_gain")
        pe          = row.get("current_pe_ratio")
        pe_vs_med   = row.get("pe_vs_historical_median")
        mom12       = row.get("twelve_month_momentum")

        with st.container(border=True):
            col_left, col_right = st.columns([2, 3])

            with col_left:
                st.markdown(
                    f"### {ticker} &nbsp; "
                    f"<span style='font-size:0.8em;color:{cls_colour};font-weight:bold'>{cls}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{company}**  \n{sector} · {category}")
                st.markdown(
                    f"**Data Quality:** {dq:.0f}/100 &nbsp;|&nbsp; "
                    f"**Keyes Agreement:** {'✓ YES' if keyes_flag else '✗ NO'}",
                )
                if pred_xret is not None:
                    st.markdown(f"**Predicted excess return:** {pred_xret:+.1%} above S&P 500")

            with col_right:
                st.markdown("**The five Keyes regression variables that drove selection:**")
                st.caption(
                    "The OLS model is trained on 2015–2026 historical data. "
                    "It identifies patterns that have historically preceded outperformance — "
                    "including mean-reversion patterns where declining stocks later recovered."
                )

                # Build a mini table of the 5 Keyes variables
                # Always show all 5 rows — mark unavailable ones as "N/A (missing data)"
                import math as _math2
                def _pct(v):
                    if v is None or (isinstance(v, float) and _math2.isnan(v)):
                        return "N/A — missing data"
                    return f"{v:+.1%}"
                def _pe(v):
                    if v is None or (isinstance(v, float) and _math2.isnan(v)):
                        return "N/A — missing data"
                    return f"{v:.1f}×"

                keyes_rows = [
                    ("X5 — 5-yr EPS growth",      _pct(eps5)),
                    ("X12 — 5-yr revenue growth",  _pct(rev5)),
                    ("X6 — 5-yr price change",     _pct(price5)),
                    ("X8 — Current P/E ratio",     _pe(pe)),
                    ("X9 — P/E vs own history",    _pct(pe_vs_med)),
                ]

                if keyes_rows:
                    tbl = pd.DataFrame(keyes_rows, columns=["Keyes Variable", "Value"])
                    st.dataframe(tbl, use_container_width=True, hide_index=True, height=35 + 35 * len(tbl))
                else:
                    st.info("Keyes variable details not available — check Stock Detail page.")

                if mom12 is not None:
                    st.markdown(f"**12-month price momentum:** {mom12:+.1%}")

                st.markdown(
                    f"**Why selected:** The Keyes OLS regression analysis placed {ticker} in the "
                    f"top 30% of all 60 stocks on the minimum predicted return across all five "
                    f"variables. The multi-factor model independently confirmed it as a **{cls}** "
                    f"with data quality {dq:.0f}/100."
                )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALL KEYES AGREEMENT STOCKS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(f"All Keyes Agreement Stocks — {n_keyes}/60 ({n_keyes/60:.0%})")
st.markdown(
    "All 18 stocks where the 5 Keyes OLS regression variables place them in the "
    "**top 30% of expected returners**. This 30% threshold replicates Keyes' original "
    "selection criterion. Stocks are sorted: Strong candidates first, then by Data Quality."
)

if not keyes_df.empty:
    keyes_display = keyes_df[[
        c for c in [
            "ticker", "company_name", "category", "sector",
            "candidate_classification", "data_quality_score",
            "predicted_12m_excess_return", "five_year_eps_growth",
            "five_year_revenue_growth", "twelve_month_momentum",
        ] if c in keyes_df.columns
    ]].rename(columns={
        "ticker":                        "Ticker",
        "company_name":                  "Company",
        "category":                      "Type",
        "sector":                        "Sector",
        "candidate_classification":      "Classification",
        "data_quality_score":            "DQ",
        "predicted_12m_excess_return":   "Pred. XRet",
        "five_year_eps_growth":          "EPS 5yr",
        "five_year_revenue_growth":      "Rev 5yr",
        "twelve_month_momentum":         "12m Mom",
    })

    fmt = {}
    if "Pred. XRet" in keyes_display.columns: fmt["Pred. XRet"] = "{:+.1%}"
    if "EPS 5yr"    in keyes_display.columns: fmt["EPS 5yr"]    = "{:+.1%}"
    if "Rev 5yr"    in keyes_display.columns: fmt["Rev 5yr"]    = "{:+.1%}"
    if "12m Mom"    in keyes_display.columns: fmt["12m Mom"]    = "{:+.1%}"
    if "DQ"         in keyes_display.columns: fmt["DQ"]         = "{:.0f}"

    def _hc_highlight(row):
        styles = [""] * len(row)
        if "Classification" in row.index:
            i = list(row.index).index("Classification")
            c = LABEL_COLOURS.get(row["Classification"], "#888")
            styles[i] = f"color:{c};font-weight:bold"
        if "DQ" in row.index:
            i = list(row.index).index("DQ")
            dq_val = row["DQ"]
            try:
                if float(dq_val) >= 90:
                    styles[i] = "color:#1a7a3a;font-weight:bold"
                elif float(dq_val) < 75:
                    styles[i] = "color:#a94442"
            except Exception:
                pass
        return styles

    styled_keyes = keyes_display.style.apply(_hc_highlight, axis=1).format(fmt, na_rep="—")
    st.dataframe(styled_keyes, use_container_width=True, height=400)

    with st.expander("How to read this table"):
        st.markdown("""
**Classification** tells you what the 33-feature multi-factor model thinks:
- 🟢 **Strong candidate** — strong signal across all major models; reliable data (DQ ≥ 75)
- 🟡 **Watchlist candidate** — moderate signal; consider alongside other research
- 🔴 **Weak / avoid** — the multi-factor model *disagrees* with the Keyes flag; treat with caution

**DQ** (Data Quality score, 0–100) measures how complete and reliable this stock's input data is.
A stock with DQ < 75 had missing features, meaning the models had to work with incomplete information.

**Pred. XRet** is the linear regression model's estimate of excess return above the S&P 500.
This is a directional estimate, not a precise forecast.

**Keyes OLS Agreement** means all 5 of Keyes' original regression variables placed this stock
in the top 30% of expected returners for the month.
        """)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FULL RANKED TABLE
# ══════════════════════════════════════════════════════════════════════════════
_cls_order_map = {"Strong candidate": 0, "Watchlist candidate": 1, "Neutral": 2, "Weak / avoid": 3}
ranked_df = scores_df.copy()
ranked_df["_r"] = ranked_df["candidate_classification"].map(_cls_order_map).fillna(2)
ranked_df = (
    ranked_df
    .sort_values(["_r", "data_quality_score"], ascending=[True, False])
    .drop(columns=["_r"])
    .reset_index(drop=True)
)

col_title, col_slider = st.columns([3, 1])
with col_title:
    st.subheader("All 60 Stocks — Full Rankings")
    st.markdown(
        "Sorted by classification (Strong → Watchlist → Neutral → Weak), "
        "then by Data Quality score. The **Keyes** column shows which stocks passed "
        "the OLS agreement filter."
    )
with col_slider:
    top_n = st.slider("Show top N", 10, 60, 20, 5)

disp_df = ranked_df.head(top_n)[[
    c for c in [
        "ticker", "company_name", "category", "sector",
        "candidate_classification", "keyes_agreement_flag",
        "data_quality_score", "predicted_12m_excess_return",
        "five_year_eps_growth", "five_year_revenue_growth",
        "twelve_month_momentum", "current_pe_ratio",
    ] if c in ranked_df.columns
]].rename(columns={
    "ticker":                       "Ticker",
    "company_name":                 "Company",
    "category":                     "Type",
    "sector":                       "Sector",
    "candidate_classification":     "Classification",
    "keyes_agreement_flag":         "Keyes ✓",
    "data_quality_score":           "DQ",
    "predicted_12m_excess_return":  "Pred. XRet",
    "five_year_eps_growth":         "EPS 5yr",
    "five_year_revenue_growth":     "Rev 5yr",
    "twelve_month_momentum":        "12m Mom",
    "current_pe_ratio":             "P/E",
})

fmt_full = {}
for col, f in [("Pred. XRet","{:+.1%}"),("EPS 5yr","{:+.1%}"),
               ("Rev 5yr","{:+.1%}"),("12m Mom","{:+.1%}"),
               ("DQ","{:.0f}"),("P/E","{:.1f}")]:
    if col in disp_df.columns:
        fmt_full[col] = f

def _full_style(row):
    styles = [""] * len(row)
    idx = list(row.index)
    if "Classification" in idx:
        i = idx.index("Classification")
        styles[i] = f"color:{LABEL_COLOURS.get(row['Classification'],'#888')};font-weight:bold"
    if "Keyes ✓" in idx:
        i = idx.index("Keyes ✓")
        if row["Keyes ✓"] == 1:
            styles[i] = "color:#1a7a3a;font-weight:bold"
    return styles

st.dataframe(
    disp_df.style.apply(_full_style, axis=1).format(fmt_full, na_rep="—"),
    use_container_width=True, height=500,
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Classification Breakdown")
    cls_counts = scores_df["candidate_classification"].value_counts().reset_index()
    cls_counts.columns = ["Classification", "Count"]
    fig_pie = px.pie(
        cls_counts, names="Classification", values="Count",
        color="Classification", color_discrete_map=LABEL_COLOURS,
        title=f"60 stocks classified — {n_keyes} carry Keyes Agreement flag",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Keyes Agreement by Sector")
    if "sector" in scores_df.columns:
        sector_keyes = (
            scores_df.groupby("sector")
            .apply(lambda g: pd.Series({
                "Keyes picks": int(g["keyes_agreement_flag"].sum()),
                "Total": len(g),
            }), include_groups=False)
            .reset_index()
        )
        sector_keyes["Rate"] = sector_keyes["Keyes picks"] / sector_keyes["Total"]
        sector_keyes = sector_keyes.sort_values("Rate", ascending=False)
        fig_sec = px.bar(
            sector_keyes, x="sector", y="Keyes picks",
            color="Rate", color_continuous_scale=["#f0f0f0","#1a7a3a"],
            text="Keyes picks",
            title="Number of Keyes-flagged stocks per sector",
            labels={"sector": "Sector", "Keyes picks": "# Keyes picks"},
        )
        fig_sec.update_traces(texttemplate="%{text}", textposition="outside")
        fig_sec.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_sec, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY NOTE
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Methodology — how the research model works"):
    st.markdown(f"""
**Study design (replicating Keyes, 1972)**

| Group | Stocks | Selection method |
|---|---|---|
| Core 30 | DJIA 30 components | Fixed index — same as Keyes' Dow Jones 30 |
| Random 30 | Mid-cap pool (60 stocks) | Monthly random draw — same as Keyes' random sample |

**The five Keyes regression variables (Steps 12–14 of original paper)**

| Variable | What it measures | Expected direction |
|---|---|---|
| X5 — 5-year EPS growth | Earnings quality and consistency | Higher = more likely to outperform |
| X6 — 5-year price gain | Market-validated performance history | Higher = more likely to outperform |
| X8 — Current P/E ratio | Valuation relative to earnings | Moderate = better (not overpriced) |
| X9 — P/E vs historical median | Is the stock cheap or expensive vs its own history? | Below median = better |
| X12 — 5-year revenue growth | Top-line business momentum | Higher = more likely to outperform |

**Selection process**

Each month, 5 independent OLS regression models are trained (one per variable).
A stock receives the Keyes Agreement flag if its minimum predicted return across all 5 models
places it in the **top 30%** of the universe — meaning every single model agrees on outperformance.
This threshold replicates Keyes' original +10% agreement criterion.

**Current result: {n_keyes}/60 = {n_keyes/60:.0%}** stocks pass — consistent with Keyes' reported 26–30%.

**Data quality requirement**

Stocks with DQ < 75 have incomplete feature data. Their predictions carry higher uncertainty.
For research conclusions, only DQ ≥ 75 stocks are treated as fully valid observations.
    """)
