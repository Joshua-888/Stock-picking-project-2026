"""
app/pages/4_Stock_Detail.py

Deep-dive view for a single stock — written for a non-technical reader.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Detail", layout="wide")

st.markdown(
    "<div style='background:#1a5c2a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📊 <strong>PHASE 2 — PREDICTION RESULTS</strong> &nbsp;|&nbsp; "
    "Deep-dive on a single stock — all 33 feature values, score breakdown, and prediction history."
    "</div>",
    unsafe_allow_html=True,
)

# ── Friendly feature name map ──────────────────────────────────────────────────
FEATURE_LABELS = {
    "price_to_book":               ("Price-to-Book (P/B)",        "Share price ÷ book value per share. Lower = cheaper relative to assets."),
    "price_to_earnings":           ("Price-to-Earnings (P/E)",    "Share price ÷ annual earnings per share. Lower = cheaper earnings."),
    "price_to_sales":              ("Price-to-Sales (P/S)",       "Market cap ÷ annual revenue. Lower = cheaper relative to sales."),
    "current_pe_ratio":            ("Current P/E Ratio",          "Most recent price ÷ earnings. Below sector average is a positive signal."),
    "dividend_yield":              ("Dividend Yield",              "Annual dividends ÷ share price. High yield can signal value — or distress."),
    "roe":                         ("Return on Equity (ROE)",      "Net income ÷ shareholder equity. Measures how efficiently the company uses capital. Higher = better."),
    "roic":                        ("Return on Assets (ROA)",       "Net income ÷ total assets. Measures how efficiently the company uses ALL its assets (equity + debt). Distinct from ROE, which measures equity efficiency only."),
    "free_cash_flow_yield":        ("Free Cash Flow Yield",        "Free cash flow ÷ market cap. Higher = more cash generated per £ invested."),
    "five_year_revenue_growth":    ("5-Year Revenue Growth",      "Compound annual revenue growth over 5 years. Sustained growth is a strong quality signal."),
    "five_year_eps_growth":        ("5-Year EPS Growth",           "Compound annual earnings-per-share growth over 5 years."),
    "five_year_price_gain":        ("5-Year Price Gain",           "Total stock price appreciation over 5 years. Momentum persists."),
    "twelve_month_momentum":       ("12-Month Momentum",           "Price return over the past 12 months. Stocks with recent strength tend to continue outperforming."),
    "return_1m":                   ("1-Month Return",              "Price change last month."),
    "return_3m":                   ("3-Month Return",              "Price change over 3 months."),
    "market_cap":                  ("Market Capitalisation",       "Total market value of all shares. Larger companies tend to be more stable; smaller ones show higher growth potential."),
    "beta":                        ("Beta (Market Sensitivity)",   "Sensitivity to S&P 500 moves. Beta > 1 = amplified swings. Beta < 1 = more stable than the market."),
    "volatility_3m":               ("3-Month Volatility",          "Standard deviation of daily returns. Higher volatility = higher risk."),
    "downside_volatility_12m":     ("Downside Volatility",         "Volatility measured only on down days — captures the risk that matters most to investors."),
    "drawdown_from_52w_high":      ("Drawdown from 52-Week High",  "How far below its 52-week peak the stock currently trades. Large drawdown = potential opportunity or distress."),
    "abnormal_volume":             ("Abnormal Trading Volume",     "Recent volume relative to its historical average. Spikes can signal institutional buying or news."),
    "debt_to_equity":              ("Debt-to-Equity Ratio",        "Total debt ÷ equity. Higher leverage amplifies both gains and losses."),
    "current_ratio":               ("Current Ratio",               "Current assets ÷ current liabilities. Above 1.5 indicates healthy short-term liquidity."),
    "shares_outstanding_m":        ("Shares Outstanding (M)",      "Total shares in issue, in millions."),
    "sector_relative_ps":          ("Sector-Relative P/S",         "Price-to-Sales relative to the sector average. Negative = cheaper than peers."),
    "sector_relative_fcf_yield":   ("Sector-Relative FCF Yield",   "Free cash flow yield relative to sector peers. Positive = better cash generation than average."),
    "peer_momentum_zscore":        ("Peer Momentum Z-Score",       "12-month momentum relative to sector peers. Positive = outpacing its industry."),
    "peer_valuation_zscore":       ("Peer Valuation Z-Score",      "Valuation relative to sector peers. Negative = cheaper than peers on a composite basis."),
    "gross_margin":                ("Gross Margin",                 "Gross profit ÷ revenue. Higher margins signal pricing power and competitive advantage."),
    "operating_margin":            ("Operating Margin",             "Operating income ÷ revenue. Measures core business profitability."),
    # Missing entries — added to ensure all 33 features display cleanly
    "pe_vs_historical_median":     ("P/E vs Own History (X9)",      "Current P/E ÷ the stock's own 5-year median P/E, minus 1. Negative = trading cheaper than its own history. Keyes variable X9."),
    "six_month_momentum":          ("6-Month Momentum",             "Price return over the past 6 months. Medium-term trend signal."),
    "volatility_12m":              ("12-Month Volatility",          "Annualised standard deviation of monthly returns over 12 months. Higher = more uncertain future returns."),
    "sector_relative_pe":          ("Sector-Relative P/E",          "This stock's P/E minus the sector median. Negative = cheaper than peers on earnings."),
    "sector_relative_momentum":    ("Sector-Relative Momentum",     "12-month return minus the sector median return. Positive = outperforming its industry."),
    "eps_growth_acceleration":     ("EPS Growth Acceleration",      "Whether earnings growth is speeding up or slowing down. Positive = accelerating growth."),
    "revenue_growth_acceleration": ("Revenue Growth Acceleration",  "Whether revenue growth is speeding up or slowing down. Positive = accelerating top-line growth."),
}


# ── Data loader ────────────────────────────────────────────────────────────────
def _cache_key() -> str:
    try:
        from src.database.db import get_connection
        with get_connection() as conn:
            r = conn.execute("SELECT MAX(id), MAX(prediction_date) FROM prediction_snapshots").fetchone()
        return f"{r[0]}_{r[1]}"
    except Exception:
        return ""


@st.cache_data(show_spinner="Loading stock data…")
def _load_all(cache_key: str = ""):
    from src.database.db import initialize_db
    from src.database.migrations import apply_migrations
    from src.database.queries import (load_prediction_snapshots, load_realized_performance,
                                       load_stocks, load_prices_clean, load_benchmark_prices)
    initialize_db(); apply_migrations()
    return (load_prediction_snapshots(), load_realized_performance(),
            load_stocks(), load_prices_clean(), load_benchmark_prices())


snaps_df, realized_df, stocks_df, prices_df, bench_df = _load_all(cache_key=_cache_key())

if stocks_df.empty:
    st.error("No stock data available. Run the monthly update first.")
    st.stop()

# Sort tickers by classification then P(Win) so the strongest stock is first
_cls_order = {"Strong candidate": 0, "Watchlist candidate": 1, "Neutral": 2, "Weak / avoid": 3}
_latest = (
    snaps_df.sort_values("prediction_date")
    .groupby("ticker")
    .last()
    .reset_index()
)
_latest["_cls_rank"] = _latest["candidate_classification"].map(_cls_order).fillna(2)
_latest = _latest.sort_values(
    ["_cls_rank", "probability_of_outperformance"],
    ascending=[True, False],
)
all_tickers = _latest["ticker"].tolist()
if not all_tickers:
    all_tickers = sorted(stocks_df["ticker"].unique())

# ── Page intro ─────────────────────────────────────────────────────────────────
st.title("📋 Stock Detail")
st.markdown(
    "Select any stock to see its full statistical profile: what the models think, "
    "why they think it, how confident they are, and how past predictions compared "
    "to actual outcomes."
)

# ── Ticker picker ──────────────────────────────────────────────────────────────
sel_ticker = st.selectbox(
    "Choose a stock:",
    all_tickers,
    format_func=lambda t: (
        f"{t} — {stocks_df[stocks_df['ticker']==t]['company_name'].values[0]}"
        if not stocks_df[stocks_df['ticker']==t].empty else t
    ),
)

ticker_snaps = snaps_df[snaps_df["ticker"] == sel_ticker].sort_values("prediction_date")
latest_snap  = ticker_snaps.iloc[-1] if not ticker_snaps.empty else None
meta         = stocks_df[stocks_df["ticker"] == sel_ticker]
meta         = meta.iloc[0] if not meta.empty else {}

st.markdown("---")

# ── Header + verdict ───────────────────────────────────────────────────────────
from src.scoring.classifications import LABEL_COLOURS

st.subheader(f"{sel_ticker} — {meta.get('company_name', '')}")
st.caption(
    f"Sector: {meta.get('sector', '—')}  ·  {meta.get('industry', '—')}  |  "
    f"Category: **{meta.get('category', 'Core')}**  |  "
    f"Latest prediction: {latest_snap['prediction_date'] if latest_snap is not None else '—'}"
)

if latest_snap is not None:
    cls    = latest_snap.get("candidate_classification", "—")
    keyes  = int(latest_snap.get("keyes_agreement_flag", 0))
    colour = LABEL_COLOURS.get(cls, "#888")

    verdict_col, explain_col = st.columns([1, 2])
    with verdict_col:
        st.markdown(
            f"<div style='font-size:1.3rem;font-weight:bold;color:{colour}'>{cls}</div>",
            unsafe_allow_html=True,
        )
        if keyes:
            st.success("✅ Keyes Agreement — all four models agree")
        else:
            st.info("Models do not fully agree this month")

    with explain_col:
        cls_explain = {
            "Strong candidate":    "All key statistical signals align. P(Win) is high, the ensemble score is strong, and data quality is sufficient. This is the top tier.",
            "Watchlist candidate": "Good signals but not every model agrees, or one quality threshold is borderline. Worth monitoring — may strengthen next month.",
            "Neutral":             "Mixed signals. No statistically meaningful edge identified in either direction. The model has no strong view.",
            "Weak / avoid":        "Multiple models point toward below-market performance. The statistical signals are negative.",
        }
        st.markdown(
            f"*{cls_explain.get(cls, '')}*"
        )

st.markdown("---")


# ── Score metrics with explanations ───────────────────────────────────────────
st.subheader("This month's scores")
st.markdown(
    "Six numbers summarise the model's view of this stock. "
    "Hover the **ℹ️** labels below each metric for a plain-English explanation."
)

if latest_snap is not None:
    def _fmt_xret(v):
        return f"{v:+.0%}" if abs(v) >= 1.0 else f"{v:+.1%}"

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    final_score = float(latest_snap.get("final_score") or 0)
    p_win       = float(latest_snap.get("probability_of_outperformance") or 0)
    xret        = float(latest_snap.get("predicted_12m_excess_return") or 0)
    agreement   = float(latest_snap.get("model_agreement_score") or 50)
    risk        = float(latest_snap.get("risk_score") or 50)
    dq          = float(latest_snap.get("data_quality_score") or 50)
    keyes_flag  = int(latest_snap.get("keyes_agreement_flag") or 0)

    # ── Research Confidence Score (replaces raw P(Win) as headline metric) ──
    def _research_confidence(snap_row) -> float:
        s = 0.0
        if int(snap_row.get("keyes_agreement_flag") or 0): s += 35.0
        cls = snap_row.get("candidate_classification", "")
        if cls == "Strong candidate":    s += 25.0
        elif cls == "Watchlist candidate": s += 10.0
        s += 15.0 * min(float(snap_row.get("data_quality_score") or 0) / 100, 1.0)
        s += 15.0 * min(float(snap_row.get("model_agreement_score") or 0) / 100, 1.0)
        xr = float(snap_row.get("predicted_12m_excess_return") or 0)
        s += 10.0 if xr > 0.10 else (7.0 if xr > 0.05 else (3.0 if xr > 0 else 0))
        return round(s, 1)

    conf_score = _research_confidence(latest_snap)

    # Cap excess return display: Ridge extrapolates wildly for extreme stocks
    xret_display = max(min(xret, 1.0), -0.5)  # cap at +100% / -50% for display
    xret_capped  = (xret > 1.0 or xret < -0.5)

    c1.metric("Research Confidence", f"{conf_score:.0f} / 100")
    c1.caption(
        "Composite score combining Keyes OLS agreement (35 pts), "
        "multi-factor classification (25 pts), data quality (15 pts), "
        "model agreement (15 pts), and predicted return direction (10 pts). "
        "≥ 70 = qualifies as a high-confidence pick."
    )

    # P(Win) shown with cap and clear saturation warning
    if p_win >= 0.99:
        c2.metric("Logistic P(Win)", ">95% ⚠️")
        c2.caption(
            "⚠️ **Saturated.** The logistic model outputs near-100% for quality large-caps "
            "because they dominated the 2015–2026 bull market training data. "
            "This is a model artefact, not a reliable probability. "
            "Use the Research Confidence Score above instead."
        )
    else:
        c2.metric("Logistic P(Win)", f"{p_win:.0%}")
        c2.caption(
            "Probability of beating the S&P 500 over 12 months from the logistic regression model. "
            "Reliable in the 50–85% range. Values near 100% indicate model saturation."
        )

    xret_label = f"{xret_display:+.0%}" + (" (capped)" if xret_capped else "")
    c3.metric("Predicted Excess Return", xret_label)
    if xret_capped:
        c3.caption(
            f"⚠️ Raw model output was {xret:+.0%} — capped at +100% for display. "
            "The Ridge regression extrapolates outside its training range for stocks with "
            "extreme features. Treat large values as directional only, not literal forecasts."
        )
    else:
        c3.caption(
            "The Ridge regression model's estimate of return above the S&P 500. "
            "Treat as directional (positive = likely to outperform) rather than a precise number."
        )

    c4.metric("Model Agreement", f"{agreement:.0f} / 100")
    c4.caption("How consistently all sub-models (Ridge, Lasso, logistic) point the same direction. 100 = unanimous. Low scores mean models disagree — treat with caution.")

    c5.metric("Risk Score", f"{risk:.0f} / 100")
    c5.caption("Composite risk measure combining volatility, beta, and drawdown. Lower = less risky.")

    c6.metric("Data Quality", f"{dq:.0f} / 100")
    c6.caption("Completeness of input data. Below 60 = key features missing. Predictions based on incomplete data carry higher uncertainty.")

    warnings_json = latest_snap.get("warnings_json")
    if warnings_json:
        try:
            for w in json.loads(warnings_json):
                st.warning(w, icon="⚠️")
        except Exception:
            pass

st.markdown("---")


# ── Price chart ────────────────────────────────────────────────────────────────
st.subheader("Price history vs S&P 500")
st.markdown(
    "Both lines start at **100** on the same date, so you can directly compare "
    "percentage growth. A line above 100 means the stock is up from that starting point. "
    "**Orange dotted lines** mark the months when the model made a prediction — "
    "check how the stock moved after each one."
)

ticker_prices = prices_df[prices_df["ticker"] == sel_ticker].sort_values("date")

if not ticker_prices.empty and not bench_df.empty:
    t_prices = ticker_prices.set_index("date")["adjusted_close"]
    b_prices = bench_df.set_index("date")["adjusted_close"]

    common_start = max(t_prices.index.min(), b_prices.index.min())
    t_norm = t_prices / t_prices.loc[common_start:].iloc[0] * 100
    b_norm = b_prices / b_prices.loc[common_start:].iloc[0] * 100

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=t_norm.index, y=t_norm.values,
        mode="lines", name=sel_ticker,
        line=dict(color="#1f77b4", width=2.5),
    ))
    fig_p.add_trace(go.Scatter(
        x=b_norm.index, y=b_norm.values,
        mode="lines", name="SPY (S&P 500)",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
    ))
    for pred_date in ticker_snaps["prediction_date"].tolist()[-6:]:
        fig_p.add_vline(x=pred_date, line_color="orange",
                        line_width=1, line_dash="dot", opacity=0.6)
    fig_p.update_layout(
        height=380,
        xaxis_title="Date",
        yaxis_title="Growth of $100 invested",
        title=f"{sel_ticker} vs S&P 500 — indexed to 100 at first common date",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Price data not yet available for this ticker.")

st.markdown("---")


# ── Score trend ────────────────────────────────────────────────────────────────
st.subheader("Score trend over time")
st.markdown(
    "Has the model's conviction been rising or falling? "
    "A consistently high **Research Confidence Score** across multiple months is more reliable "
    "than a single strong reading. The logistic P(Win) is shown as a secondary line — "
    "values near 100% indicate model saturation, not a literal probability."
)

if not ticker_snaps.empty and len(ticker_snaps) > 1:
    sh = ticker_snaps[["prediction_date","final_score","probability_of_outperformance",
                        "model_agreement_score"]].sort_values("prediction_date")

    fig_sh = go.Figure()
    fig_sh.add_trace(go.Scatter(
        x=sh["prediction_date"], y=sh["final_score"],
        mode="lines+markers", name="Final Score (0–100)",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_sh.add_trace(go.Scatter(
        x=sh["prediction_date"],
        y=(sh["probability_of_outperformance"].astype(float).clip(0, 0.99) * 100),
        mode="lines+markers", name="Logistic P(Win) — capped at 99% (may saturate)",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
    ))
    fig_sh.add_hline(y=50, line_color="grey", line_dash="dot", line_width=0.8,
                     annotation_text="50% baseline (coin-flip)", annotation_position="right")
    fig_sh.update_layout(
        height=300,
        xaxis_title="Prediction month",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        title="Model conviction over time",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_sh, use_container_width=True)
elif not ticker_snaps.empty:
    st.info("Only one prediction month available so far — trend will build over time.")

st.markdown("---")


# ── Feature breakdown ──────────────────────────────────────────────────────────
st.subheader("What the model sees — feature values")
st.markdown(
    "These are the 25+ financial variables the models used to score this stock. "
    "The **Signal** column gives a plain-English verdict on each value: "
    "✅ positive, ⚠️ mixed or elevated, 🔴 a genuine concern."
)


def _signal(key: str, val: float) -> str:
    """Return a plain-English signal label for a given feature and its value."""
    import math as _m
    if val is None or not isinstance(val, (int, float)):
        return "—"
    v = float(val)
    if _m.isnan(v) or _m.isinf(v):
        return "—"

    rules = {
        # Momentum & returns — positive = good
        "return_1m":              [("✅ Strong momentum", v > 0.08), ("✅ Positive", v > 0), ("🔴 Negative",  True)],
        "return_3m":              [("✅ Strong",          v > 0.12), ("✅ Positive", v > 0), ("🔴 Negative",  True)],
        "twelve_month_momentum":  [("✅ Strong",          v > 0.15), ("✅ Positive", v > 0), ("🔴 Negative",  True)],
        "five_year_price_gain":   [("✅ Strong",          v > 0.5),  ("✅ Positive", v > 0), ("🔴 Declined over 5yr", True)],
        # Growth — positive = good
        "five_year_revenue_growth": [("✅ Strong",  v > 0.10), ("✅ Positive", v > 0.02), ("⚠️ Slow growth", v > -0.02), ("🔴 Revenue declining", True)],
        "five_year_eps_growth":     [("✅ Strong",  v > 0.10), ("✅ Positive", v > 0.02), ("⚠️ Weak growth",  v > -0.02), ("🔴 Earnings declining", True)],
        # Profitability — higher = better
        "roe":                    [("✅ Excellent", v > 0.20), ("✅ Good",    v > 0.10), ("⚠️ Low",    v > 0.0), ("🔴 Negative ROE", True)],
        "roic":                   [("✅ Excellent", v > 0.15), ("✅ Good",    v > 0.08), ("⚠️ Low",    v > 0.0), ("🔴 Negative ROA", True)],
        "gross_margin":           [("✅ High",      v > 0.40), ("✅ Solid",   v > 0.20), ("⚠️ Thin margins", True)],
        "operating_margin":       [("✅ High",      v > 0.15), ("✅ Positive",v > 0.05), ("⚠️ Low",   v > 0.0), ("🔴 Operating loss", True)],
        "free_cash_flow_yield":   [("✅ High yield",v > 0.05), ("✅ Positive",v > 0.01), ("⚠️ Thin",  v > 0.0), ("🔴 Negative FCF",  True)],
        # Valuation — lower is usually better
        "current_pe_ratio":       [("✅ Low P/E",   v < 15),   ("⚠️ Moderate P/E", v < 30), ("⚠️ Elevated P/E", v < 50), ("🔴 Very high P/E", True)],
        "price_to_book":          [("✅ Low P/B",   v < 2.0),  ("⚠️ Moderate", v < 5.0), ("🔴 Expensive vs assets", True)],
        "price_to_sales":         [("✅ Low P/S",   v < 2.0),  ("⚠️ Moderate", v < 8.0), ("🔴 Very expensive vs sales", True)],
        "pe_vs_historical_median":[("✅ Below own median", v < -0.05), ("⚠️ Near median", v < 0.10), ("🔴 Premium to own history", True)],
        "sector_relative_pe":     [("✅ Cheaper than sector", v < -0.05), ("⚠️ Near sector average", v < 0.10), ("🔴 Premium to sector", True)],
        "peer_valuation_zscore":  [("✅ Cheap vs peers", v < -0.5), ("⚠️ Fairly valued", v < 0.5), ("🔴 Expensive vs peers", True)],
        # Risk — lower is usually better
        "beta":                   [("✅ Defensive (β<0.8)", v < 0.8), ("⚠️ Market-like", v < 1.2), ("🔴 High β — amplified risk", True)],
        "volatility_3m":          [("✅ Low risk",   v < 0.18), ("⚠️ Moderate risk", v < 0.30), ("🔴 High volatility", True)],
        "downside_volatility_12m":[("✅ Low downside risk", v < 0.025), ("⚠️ Moderate", v < 0.05), ("🔴 High downside risk", True)],
        "drawdown_from_52w_high": [("✅ Near 52w high", v > -0.05), ("⚠️ Moderate pullback", v > -0.20), ("🔴 Deep drawdown", True)],
        # Leverage — lower is better
        "debt_to_equity":         [("✅ Low debt",  v < 0.3),  ("⚠️ Moderate debt", v < 1.0), ("🔴 High leverage", True)],
        # Dividend — neutral unless very high (could signal distress)
        "dividend_yield":         [("✅ Solid yield", v > 0.03), ("⚠️ Low yield", v > 0.005), ("— Growth stock (no div.)", True)],
        # Volume — near 1.0 is neutral; spikes can be positive or negative
        "abnormal_volume":        [("⚠️ Unusually high volume", v > 1.8), ("✅ Normal volume", v > 0.5), ("⚠️ Unusually low volume", True)],
        # Peer comparisons
        "peer_momentum_zscore":      [("✅ Outpacing peers", v > 0.5), ("⚠️ In line with peers", v > -0.5), ("🔴 Lagging peers", True)],
        # Additional features
        "six_month_momentum":        [("✅ Strong",          v > 0.10), ("✅ Positive", v > 0), ("🔴 Negative",  True)],
        "volatility_12m":            [("✅ Low risk",        v < 0.18), ("⚠️ Moderate risk", v < 0.30), ("🔴 High volatility", True)],
        "sector_relative_pe":        [("✅ Cheaper than sector", v < -0.05), ("⚠️ Near sector average", v < 0.10), ("🔴 Premium to sector", True)],
        "sector_relative_momentum":  [("✅ Outperforming sector", v > 0.05), ("⚠️ In line with sector", v > -0.05), ("🔴 Underperforming sector", True)],
        "eps_growth_acceleration":   [("✅ Accelerating EPS growth", v > 0.02), ("⚠️ Stable", v > -0.02), ("🔴 Decelerating earnings", True)],
        "revenue_growth_acceleration":[("✅ Accelerating revenue", v > 0.02), ("⚠️ Stable", v > -0.02), ("🔴 Decelerating revenue", True)],
        "pe_vs_historical_median":   [("✅ Below own median", v < -0.05), ("⚠️ Near median", v < 0.10), ("🔴 Premium to own history", True)],
    }

    if key in rules:
        for label, condition in rules[key]:
            if condition:
                return label
    return "—"


if latest_snap is not None:
    feat_json = latest_snap.get("features_json")
    if feat_json:
        try:
            feat_dict = json.loads(feat_json)
            rows = []
            import math as _mf
            for k, v in feat_dict.items():
                if v is None:
                    continue
                if isinstance(v, float) and (_mf.isnan(v) or _mf.isinf(v)):
                    continue
                # Hide internal Keyes OLS prediction columns — shown separately above
                if k.startswith("keyes_"):
                    continue
                label, description = FEATURE_LABELS.get(k, (k.replace("_", " ").title(), ""))
                rows.append({
                    "Variable":      label,
                    "_raw_val":      v,
                    "_key":          k,
                    "What it means": description,
                })
            feat_df = pd.DataFrame(rows).sort_values("Variable")
            feat_df["Signal"] = feat_df.apply(lambda r: _signal(r["_key"], r["_raw_val"]), axis=1)

            def _fmt_val(v):
                if isinstance(v, float):
                    if abs(v) >= 1000:
                        return f"{v:,.0f}"
                    if abs(v) >= 10:
                        return f"{v:,.1f}"
                    if abs(v) < 0.01:
                        return f"{v:.4f}"
                    return f"{v:.3f}"
                return str(v)

            feat_df["Value"] = feat_df["_raw_val"].apply(_fmt_val)
            feat_df = feat_df.drop(columns=["_raw_val", "_key"])

            def _colour_signal(val):
                if val.startswith("✅"): return "color:#1a7a3a;font-weight:bold"
                if val.startswith("⚠️"): return "color:#b8860b;font-weight:bold"
                if val.startswith("🔴"): return "color:#a94442;font-weight:bold"
                return "color:#888"

            # Height: 36px per row + 38px header, capped at 1200px
            tbl_height = min(38 + len(feat_df) * 36, 1200)
            st.dataframe(
                feat_df[["Variable","Value","Signal","What it means"]]
                    .style.map(_colour_signal, subset=["Signal"]),
                use_container_width=True,
                height=tbl_height,
                column_config={
                    "Variable":      st.column_config.TextColumn("Variable",      width="medium"),
                    "Value":         st.column_config.TextColumn("Value",         width="small"),
                    "Signal":        st.column_config.TextColumn("Signal",        width="medium"),
                    "What it means": st.column_config.TextColumn("What it means", width="large"),
                },
                hide_index=True,
            )

            # Summary counts
            n_pos  = feat_df["Signal"].str.startswith("✅").sum()
            n_warn = feat_df["Signal"].str.startswith("⚠️").sum()
            n_neg  = feat_df["Signal"].str.startswith("🔴").sum()
            st.caption(
                f"Signal summary: **{n_pos} positive** · **{n_warn} caution** · **{n_neg} concern**"
            )

        except Exception:
            st.info("Feature data not available for this snapshot.")

st.markdown("---")


# ── Prediction history ─────────────────────────────────────────────────────────
st.subheader("Prediction history")
st.markdown(
    "Every month a new prediction is saved permanently. "
    "This table shows every prediction made for this stock and, where the 12-month "
    "window has closed, whether the prediction turned out to be correct."
)

if not ticker_snaps.empty:
    hist_disp = ticker_snaps[[
        "prediction_date", "probability_of_outperformance",
        "predicted_12m_excess_return", "final_score",
        "candidate_classification", "keyes_agreement_flag",
        "data_quality_score",
    ]].sort_values("prediction_date", ascending=False).rename(columns={
        "prediction_date":               "Month",
        "probability_of_outperformance": "P(Win)",
        "predicted_12m_excess_return":   "Pred. Excess Return",
        "final_score":                   "Score",
        "candidate_classification":      "Classification",
        "keyes_agreement_flag":          "All Models Agreed?",
        "data_quality_score":            "Data Quality",
    })
    st.dataframe(
        hist_disp.style.format({
            "P(Win)":            "{:.1%}",
            "Pred. Excess Return": "{:+.2%}",
            "Score":             "{:.1f}",
            "Data Quality":      "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
        height=280,
        hide_index=True,
    )

st.markdown("---")


# ── Realised performance ───────────────────────────────────────────────────────
st.subheader("Did the predictions come true?")
st.markdown(
    "Once 12 months have passed since a prediction date, the actual outcome is recorded. "
    "**'Beat S&P?'** is the key column — did the stock outperform the S&P 500 "
    "in the 12 months after the prediction? "
    "The **hit rate** tells you what percentage of this stock's predictions were correct."
)

ticker_real = (
    realized_df[realized_df["ticker"] == sel_ticker]
    if not realized_df.empty else pd.DataFrame()
)

if not ticker_real.empty:
    correct = ticker_real["prediction_correct"].dropna()
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric(
        "Hit rate",
        f"{correct.mean():.1%}" if len(correct) > 0 else "—",
        help="% of predictions that correctly identified whether the stock beat the S&P 500",
    )
    rc2.metric(
        "Predictions evaluated",
        len(correct),
        help="Predictions older than 12 months where the actual outcome is now known",
    )
    rc3.metric(
        "Mean actual excess return",
        (f"{ticker_real['realized_12m_excess_return'].dropna().mean():+.2%}"
         if ticker_real["realized_12m_excess_return"].notna().any() else "—"),
        help="Average return above/below the S&P 500 in the 12 months following a prediction",
    )

    real_disp = ticker_real.merge(
        snaps_df[["prediction_date","ticker","candidate_classification"]],
        on=["prediction_date","ticker"], how="left",
    ).sort_values("prediction_date", ascending=False)

    st.dataframe(
        real_disp[[
            "prediction_date", "evaluation_date", "candidate_classification",
            "winner_predicted", "winner_actual", "prediction_correct",
            "realized_12m_excess_return",
        ]].rename(columns={
            "prediction_date":            "Predicted In",
            "evaluation_date":            "Evaluated On",
            "candidate_classification":   "Classification at time",
            "winner_predicted":           "Model said: beat S&P?",
            "winner_actual":              "Actually beat S&P?",
            "prediction_correct":         "Correct?",
            "realized_12m_excess_return": "Actual Excess Return",
        }).style.format({
            "Actual Excess Return": "{:+.2%}",
        }, na_rep="—"),
        use_container_width=True,
        height=250,
        hide_index=True,
    )
else:
    st.info(
        "No completed predictions yet for this stock. "
        "Results appear once 12 months have passed after a prediction date. "
        "Check back once the horizon closes.",
        icon="⏳",
    )
