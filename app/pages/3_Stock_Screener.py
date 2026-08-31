"""
app/pages/3_Stock_Screener.py  —  Filter and rank all 60 stocks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Screener", layout="wide")

st.markdown(
    "<div style='background:#1a5c2a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📊 <strong>PHASE 2 — PREDICTION RESULTS</strong> &nbsp;|&nbsp; "
    "Filter and compare all 60 stocks ranked by the statistical models. "
    "Signals were validated in Phase 1 (Model Diagnostics)."
    "</div>",
    unsafe_allow_html=True,
)

st.title("Stock Screener")
st.markdown(
    "Filter, sort, and compare all 60 stocks in the universe. "
    "Every column is explained below the table. Use the sidebar to narrow down "
    "to the stocks most relevant to your research question."
)
st.warning(
    "Scores are statistical research estimates from historical patterns. Not financial advice.",
    icon="⚠️",
)


# ── Cache-busting ──────────────────────────────────────────────────────────────
def _cache_key() -> str:
    try:
        from src.database.db import get_connection
        with get_connection() as conn:
            r = conn.execute(
                "SELECT MAX(id), MAX(prediction_date) FROM prediction_snapshots"
            ).fetchone()
        return f"{r[0]}_{r[1]}"
    except Exception:
        return ""


@st.cache_data(show_spinner="Loading screener data…")
def _load_screener(cache_key: str = ""):
    import json
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    from src.database.queries import load_prediction_snapshots, load_prices_clean, load_stocks

    initialize_db(); apply_migrations()

    snaps = load_prediction_snapshots()
    if snaps.empty:
        return pd.DataFrame()

    latest = snaps.sort_values("prediction_date").groupby("ticker").last().reset_index()

    def _parse(row):
        try:
            d = json.loads(row.get("features_json") or "{}")
            return {
                "current_pe_ratio":      d.get("current_pe_ratio"),
                "price_to_book":         d.get("price_to_book"),
                "five_year_price_gain":  d.get("five_year_price_gain"),
                "five_year_eps_growth":  d.get("five_year_eps_growth"),
                "five_year_revenue_growth": d.get("five_year_revenue_growth"),
                "twelve_month_momentum": d.get("twelve_month_momentum"),
                "dividend_yield":        d.get("dividend_yield"),
                "roe":                   d.get("roe"),
                "debt_to_equity":        d.get("debt_to_equity"),
                "market_cap":            d.get("market_cap"),
            }
        except Exception:
            return {}

    feats = latest.apply(_parse, axis=1, result_type="expand")
    latest = pd.concat([
        latest.drop(columns=["features_json","features_std_json","warnings_json","notes"], errors="ignore"),
        feats,
    ], axis=1)

    # Current price
    prices = load_prices_clean()
    cur = (prices.sort_values("date").groupby("ticker").last()[["adjusted_close"]]
           .reset_index().rename(columns={"adjusted_close":"current_price"}))
    latest = latest.merge(cur, on="ticker", how="left")

    # Category from stocks table
    stocks = load_stocks()
    if "category" in stocks.columns:
        latest = latest.merge(stocks[["ticker","category"]], on="ticker", how="left")

    return latest


df = _load_screener(cache_key=_cache_key())

with st.sidebar:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear(); st.rerun()

if df.empty:
    st.info("No prediction data yet. Run the monthly update first.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TICKER WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
import math as _math

try:
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    # Import from dedicated watchlist module — never has a stale cache issue
    from src.database.watchlist import (load_watchlist, add_watchlist_ticker,
                                        remove_watchlist_ticker, update_watchlist_ticker)
    from src.scoring.classifications import LABEL_COLOURS
    initialize_db(); apply_migrations()
    _wl_ok = True
except Exception as _e:
    _wl_ok = False
    _wl_err = str(_e)

st.markdown("---")
st.subheader("Custom Ticker Analyser")
st.markdown(
    "Add any US stock ticker and run the same Keyes (1972) analysis as the main 60-stock universe. "
    "The result shows the full classification, all 5 Keyes variables, and where the stock "
    "would rank against the existing picks.  \n"
    "**Auto-cleanup:** after each analysis run, tickers classified as *Weak / avoid* are "
    "automatically removed from the watchlist — keeping only candidates worth tracking."
)

if not _wl_ok:
    st.error(f"Watchlist module failed to load: {_wl_err}. Restart the Streamlit server.")
    st.stop()

MAX_WATCHLIST = 20

# ── Add ticker ────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([3, 1])
with col_input:
    new_ticker = st.text_input(
        "Ticker symbol",
        placeholder="e.g. GOOG, TSLA, NOVO-B.CO",
        label_visibility="collapsed",
    ).strip().upper()
with col_btn:
    add_clicked = st.button("Add to watchlist", type="primary", use_container_width=True)

if add_clicked and new_ticker:
    watchlist_now = load_watchlist()
    if new_ticker in watchlist_now["ticker"].values:
        st.info(f"**{new_ticker}** is already in your watchlist.")
    elif len(watchlist_now) >= MAX_WATCHLIST:
        st.error(
            f"Watchlist is full ({MAX_WATCHLIST} tickers max). "
            "Run an analysis first — weak picks will be removed automatically to free space."
        )
    else:
        add_watchlist_ticker(new_ticker)
        st.success(f"Added **{new_ticker}** to watchlist.")
        st.rerun()

# ── Current watchlist ─────────────────────────────────────────────────────────
watchlist_df = load_watchlist()

if watchlist_df.empty:
    st.info("Your watchlist is empty. Type a ticker above and click **Add to watchlist**.")
else:
    st.markdown(f"**Watchlist — {len(watchlist_df)}/{MAX_WATCHLIST} tickers** (click to remove)")
    tag_cols = st.columns(min(len(watchlist_df), 8))
    for i, (_, wrow) in enumerate(watchlist_df.iterrows()):
        with tag_cols[i % len(tag_cols)]:
            label = wrow["ticker"]
            if wrow.get("company_name") and wrow["company_name"] != wrow["ticker"]:
                label = f"{wrow['ticker']} ({wrow['company_name'][:12]})"
            if st.button(f"✕  {label}", key=f"rm_{wrow['ticker']}", help=f"Remove {wrow['ticker']}"):
                remove_watchlist_ticker(wrow["ticker"])
                if "watchlist_results" in st.session_state:
                    st.session_state["watchlist_results"].pop(wrow["ticker"], None)
                st.rerun()

    st.markdown("")

    # ── Run analysis ──────────────────────────────────────────────────────────
    run_all = st.button(
        f"▶  Analyse all {len(watchlist_df)} ticker(s)",
        type="primary",
    )

    if run_all:
        from src.ingestion.custom_ticker import analyse_custom_ticker

        tickers_to_run = watchlist_df["ticker"].tolist()
        results = {}
        auto_removed = []

        progress = st.progress(0, text="Starting analysis…")
        for i, t in enumerate(tickers_to_run):
            progress.progress(i / len(tickers_to_run), text=f"Analysing {t}…")
            r = analyse_custom_ticker(t)
            results[t] = r

            if not r.get("error"):
                # Update company/sector metadata in watchlist
                update_watchlist_ticker(t, r.get("company_name", t), r.get("sector", ""))

                # Auto-remove if Weak / avoid
                cls = r.get("scores", {}).get("classification", "")
                if cls == "Weak / avoid":
                    remove_watchlist_ticker(t)
                    auto_removed.append(t)

        progress.progress(1.0, text="Done!")

        if auto_removed:
            st.warning(
                f"**Auto-removed {len(auto_removed)} weak ticker(s):** {', '.join(auto_removed)}  \n"
                "These were classified as *Weak / avoid* by the model — not worth tracking.",
                icon="🗑️",
            )

        st.session_state["watchlist_results"] = results
        st.rerun()

    # ── Display results ───────────────────────────────────────────────────────
    if "watchlist_results" in st.session_state:
        results = st.session_state["watchlist_results"]
        if not results:
            st.info("No results yet. Click **Analyse** above.")
        else:
            from src.database.queries import load_prediction_snapshots
            snaps = load_prediction_snapshots()
            latest_date = snaps["prediction_date"].max() if not snaps.empty else None
            universe_df = (snaps[snaps["prediction_date"] == latest_date].copy()
                           if latest_date else pd.DataFrame())

            st.markdown("---")
            st.subheader("Analysis Results")
            st.caption(
                "Results use the trained models from the latest monthly run. "
                "Weak / avoid tickers have already been removed from the watchlist."
            )

            for ticker, r in results.items():
                if r.get("error"):
                    with st.container(border=True):
                        st.error(f"**{ticker}** — {r['error']}")
                    continue

                scores   = r.get("scores", {})
                keyes    = r.get("keyes_vars", {})
                comp     = r.get("comparison", {})
                cls      = scores.get("classification", "Unknown")
                cls_col  = LABEL_COLOURS.get(cls, "#888")
                dq       = scores.get("data_quality_score") or 0
                pwin     = scores.get("probability_of_win")
                xret     = scores.get("predicted_xret")
                min_pred = comp.get("min_ols_pred")
                passes   = comp.get("would_pass_keyes", False)
                final_sc = scores.get("final_score")

                # Skip display of auto-removed weak tickers
                if cls == "Weak / avoid":
                    continue

                with st.container(border=True):
                    left, right = st.columns([2, 3])

                    with left:
                        st.markdown(
                            f"### {ticker} &nbsp;"
                            f"<span style='color:{cls_col};font-weight:bold;font-size:0.85em'>{cls}</span>",
                            unsafe_allow_html=True,
                        )
                        if r.get("sector") and r["sector"] != "Unknown":
                            st.caption(r["sector"])

                        m1, m2 = st.columns(2)
                        m1.metric("Data Quality", f"{dq:.0f}/100")
                        m2.metric("Keyes", "YES ✓" if passes else "NO ✗")

                        if xret is not None:
                            st.metric("Predicted excess return", f"{xret:+.1%} vs S&P 500")
                        if pwin is not None:
                            st.metric("P(Win)", f"{pwin:.0%}" if pwin < 0.999 else ">99%")
                        if min_pred is not None:
                            verdict = "passes threshold" if passes else "below threshold"
                            st.caption(f"Min OLS pred: {min_pred:+.1%} — {verdict}")

                    with right:
                        st.markdown("**Five Keyes regression variables:**")
                        def _pct(v):
                            if v is None or (isinstance(v, float) and _math.isnan(v)): return "N/A"
                            return f"{v:+.1%}"
                        def _pe(v):
                            if v is None or (isinstance(v, float) and _math.isnan(v)): return "N/A"
                            return f"{v:.1f}×"

                        kt = pd.DataFrame([
                            ("X5 — 5-yr EPS growth",     _pct(keyes.get("X5_eps_growth"))),
                            ("X12 — 5-yr revenue growth", _pct(keyes.get("X12_rev_growth"))),
                            ("X6 — 5-yr price change",    _pct(keyes.get("X6_price_gain"))),
                            ("X8 — Current P/E ratio",    _pe(keyes.get("X8_pe_ratio"))),
                            ("X9 — P/E vs own history",   _pct(keyes.get("X9_pe_vs_median"))),
                        ], columns=["Keyes Variable", "Value"])
                        st.dataframe(kt, use_container_width=True, hide_index=True, height=215)

                    # ── Rank vs universe ───────────────────────────────────
                    if not universe_df.empty and final_sc is not None:
                        st.markdown("**Where it ranks against the current 60 stocks:**")
                        u = universe_df["final_score"].dropna().sort_values(ascending=False).reset_index(drop=True)
                        rank = int((u >= final_sc).sum()) + 1
                        pct  = rank / (len(u) + 1) * 100

                        rc1, rc2, rc3 = st.columns(3)
                        rc1.metric("Final score", f"{final_sc:.1f}/100")
                        rc2.metric("Rank", f"#{rank} of {len(u)+1}")
                        rc3.metric("Percentile", f"Top {pct:.0f}%")

                        top5 = universe_df.nlargest(5, "final_score")[
                            ["ticker","candidate_classification","final_score","data_quality_score"]
                        ].copy()
                        custom_row = pd.DataFrame([{
                            "ticker": f"★ {ticker}",
                            "candidate_classification": cls,
                            "final_score": final_sc,
                            "data_quality_score": dq,
                        }])
                        combined = (pd.concat([top5, custom_row], ignore_index=True)
                                    .sort_values("final_score", ascending=False)
                                    .reset_index(drop=True))

                        def _hl_row(row):
                            return (["background-color:#fff3cd"] * len(row)
                                    if str(row.get("ticker","")).startswith("★") else [""] * len(row))

                        st.dataframe(
                            combined.rename(columns={
                                "ticker":"Ticker", "candidate_classification":"Classification",
                                "final_score":"Score", "data_quality_score":"DQ",
                            }).style.apply(_hl_row, axis=1)
                              .format({"Score":"{:.1f}", "DQ":"{:.0f}"}, na_rep="—"),
                            use_container_width=True, hide_index=True, height=245,
                        )
                        st.caption("★ = your custom ticker (highlighted)")

st.markdown("---")


# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")

    all_sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sel_sector  = st.selectbox("Sector", all_sectors)

    all_cls = ["All"] + sorted(df["candidate_classification"].dropna().unique().tolist())
    sel_cls = st.selectbox("Classification", all_cls)

    all_cat = ["All", "Core", "Hidden Gem"]
    sel_cat = st.selectbox("Type", all_cat)

    min_score = st.slider("Min Final Score (0–100)", 0, 100, 0)
    min_prob  = st.slider("Min P(Win)", 0.0, 1.0, 0.0, step=0.05)

    pe_max = st.number_input("Max P/E ratio (blank = no limit)", value=None, step=10,
                              placeholder="No limit")

    sort_by = st.selectbox("Sort by", [
        "final_score",
        "probability_of_outperformance",
        "predicted_12m_excess_return",
        "twelve_month_momentum",
        "five_year_price_gain",
        "current_pe_ratio",
        "data_quality_score",
    ], format_func=lambda x: {
        "final_score": "Final Score",
        "probability_of_outperformance": "P(Win)",
        "predicted_12m_excess_return": "Pred. Excess Return",
        "twelve_month_momentum": "12m Momentum",
        "five_year_price_gain": "5yr Price Gain",
        "current_pe_ratio": "P/E Ratio",
        "data_quality_score": "Data Quality",
    }.get(x, x))
    sort_asc = st.checkbox("Sort ascending", value=False)


# Apply filters
filtered = df.copy()
if sel_sector != "All":
    filtered = filtered[filtered["sector"] == sel_sector]
if sel_cls != "All":
    filtered = filtered[filtered["candidate_classification"] == sel_cls]
if sel_cat != "All" and "category" in filtered.columns:
    filtered = filtered[filtered["category"] == sel_cat]
filtered = filtered[filtered["final_score"].fillna(0) >= min_score]
filtered = filtered[filtered["probability_of_outperformance"].fillna(0) >= min_prob]
# P/E filter: only apply when value is present AND limit is set
if pe_max is not None and "current_pe_ratio" in filtered.columns:
    filtered = filtered[
        filtered["current_pe_ratio"].isna() | (filtered["current_pe_ratio"] <= pe_max)
    ]

# Classification-first sort to match Overview
_cls_order = {"Strong candidate": 0, "Watchlist candidate": 1, "Neutral": 2, "Weak / avoid": 3}
filtered["_cls_rank"] = filtered["candidate_classification"].map(_cls_order).fillna(2)
filtered = filtered.sort_values(
    ["_cls_rank", sort_by], ascending=[True, sort_asc], na_position="last"
).drop(columns=["_cls_rank"]).reset_index(drop=True)

st.caption(
    f"Showing **{len(filtered)}** of **{len(df)}** stocks  |  "
    f"Latest prediction: **{df['prediction_date'].max()}**"
)

st.markdown("---")


# ── Screener table ─────────────────────────────────────────────────────────────
from src.scoring.classifications import LABEL_COLOURS

display_map = {
    "ticker":                         "Ticker",
    "company_name":                   "Company",
    "category":                       "Type",
    "sector":                         "Sector",
    "current_price":                  "Price ($)",
    "current_pe_ratio":               "P/E",
    "price_to_book":                  "P/B",
    "five_year_price_gain":           "5yr Price",
    "five_year_eps_growth":           "5yr EPS",
    "twelve_month_momentum":          "12m Mom",
    "dividend_yield":                 "Div Yield",
    "roe":                            "ROE",
    "probability_of_outperformance":  "P(Win)",
    "predicted_12m_excess_return":    "Pred XRet",
    "data_quality_score":             "DQ",
    "final_score":                    "Score",
    "candidate_classification":       "Classification",
    "keyes_agreement_flag":           "Keyes",
}

cols = [c for c in display_map if c in filtered.columns]
disp = filtered[cols].rename(columns=display_map)

fmt = {
    "Price ($)":  "${:.2f}",
    "P/E":        "{:.1f}",
    "P/B":        "{:.2f}",
    "5yr Price":  "{:.1%}",
    "5yr EPS":    "{:.1%}",
    "12m Mom":    "{:.1%}",
    "Div Yield":  "{:.2%}",
    "ROE":        "{:.1%}",
    "P(Win)":     "{:.1%}",
    "Pred XRet":  "{:+.1%}",
    "DQ":         "{:.0f}",
    "Score":      "{:.1f}",
}

st.dataframe(
    disp.style
        .map(lambda v: f"color:{LABEL_COLOURS.get(v,'#888')};font-weight:bold",
             subset=["Classification"] if "Classification" in disp.columns else [])
        .format(fmt, na_rep="—"),
    use_container_width=True,
    height=550,
    hide_index=True,
)


# ── Column guide ───────────────────────────────────────────────────────────────
with st.expander("Column guide — what each column means"):
    st.markdown("""
| Column | What it measures |
|---|---|
| **Type** | Core = one of the 30 large-cap anchors. Hidden Gem = randomly selected from the mid/small-cap pool this month. |
| **P/E** | Price-to-earnings ratio. Lower generally means cheaper relative to current profits. |
| **P/B** | Price-to-book. Compares market price to accounting book value. |
| **5yr Price** | Total stock price appreciation over 5 years. Strong predictor of future performance (IC = 0.10, Stability = 1.00). |
| **5yr EPS** | 5-year compound earnings-per-share growth. Measures earnings quality over time. |
| **12m Mom** | 12-month price return. Stocks with positive recent momentum tend to continue outperforming. |
| **Div Yield** | Annual dividend ÷ price. Negative IC with future returns — high-yield stocks in this universe underperformed. |
| **ROE** | Return on equity. How efficiently management uses shareholder capital. IC = 0.10, Stability = 1.00. |
| **P(Win)** | Probability the stock beats the S&P 500 over the next 12 months. From logistic regression trained on 10+ years of data. |
| **Pred XRet** | Ridge regression estimate of how many percentage points above (or below) the S&P 500 return this stock will achieve. |
| **DQ** | Data quality score (0–100). Below 60 = key data is missing; treat predictions with caution. |
| **Score** | Composite final score combining all four models, weighted by historical reliability. |
| **Classification** | Strong candidate / Watchlist / Neutral / Weak/avoid — based on Score, P(Win), and data quality thresholds. |
| **Keyes** | 1 = passed the Keyes (1972) agreement filter: top 30% by minimum prediction across all 5 single-variable OLS models. |
""")


st.markdown("---")

# ── Distribution charts ────────────────────────────────────────────────────────
st.subheader("Score distributions")
st.markdown("Visual overview of how scores are distributed across the filtered stocks.")

col1, col2 = st.columns(2)
with col1:
    fig_s = px.histogram(
        filtered, x="final_score", nbins=20,
        color="candidate_classification",
        color_discrete_map=LABEL_COLOURS,
        title="Final score distribution by classification",
        labels={"final_score": "Final Score (0–100)"},
        barmode="stack",
    )
    fig_s.update_layout(height=320, legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig_s, use_container_width=True)

with col2:
    fig_p = px.scatter(
        filtered,
        x="probability_of_outperformance",
        y="predicted_12m_excess_return",
        color="candidate_classification",
        color_discrete_map=LABEL_COLOURS,
        hover_data=["ticker","company_name"],
        title="P(Win) vs predicted excess return",
        labels={
            "probability_of_outperformance": "P(Win) — probability of beating S&P 500",
            "predicted_12m_excess_return":   "Predicted excess return over S&P 500",
        },
    )
    fig_p.add_hline(y=0, line_color="grey", line_width=0.8,
                    annotation_text="S&P 500 baseline", annotation_position="right")
    fig_p.add_vline(x=0.5, line_color="grey", line_width=0.8, line_dash="dot")
    fig_p.update_layout(height=320, legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig_p, use_container_width=True)
