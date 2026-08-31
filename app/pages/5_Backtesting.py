"""
app/pages/5_Backtesting.py

Backtesting dashboard — simulates an equal-weight top-N portfolio
rebalanced monthly and compares it against the S&P 500 benchmark.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Backtesting", layout="wide")

st.markdown(
    "<div style='background:#5c3a1a;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "📈 <strong>PHASE 3 — RESEARCH RESULTS</strong> &nbsp;|&nbsp; "
    "These pages show how the model performed historically and document every prediction made. "
    "Used to validate whether Phase 2 predictions translate into real-world outperformance."
    "</div>",
    unsafe_allow_html=True,
)

st.title("Backtesting — Portfolio Simulation")
st.markdown(
    "Simulates what would have happened if you had followed the model's top-N picks each month "
    "since 2016. This is the primary out-of-sample test of whether the Keyes (1972) methodology "
    "produces a statistically meaningful edge over passive S&P 500 investing."
)

with st.expander("How this backtest works — methodology and limitations"):
    st.markdown("""
**Walk-forward design (no look-ahead bias)**

Each month, the model is trained *only on data available before that month*. It then selects the
top-N stocks based purely on their historical features. No future price data is used in any
scoring decision. This mirrors real-world implementation.

**How stocks are scored in the backtest**

The backtest uses *feature scores* (the raw financial variables, rank-normalised), not the
logistic regression probabilities. This avoids a subtle look-ahead bias that would occur if
we used the same model trained on the full dataset to score historical periods.

**How performance is measured**

Each month, the model selects the top-N stocks by feature score, holds them for one month
(equal-weight), then rebalances. Portfolio return is compared to SPY (S&P 500 ETF).
**Hit rate** = percentage of months where the portfolio beat the benchmark.

**Known limitations — must be disclosed in any research paper**

| Limitation | Impact |
|---|---|
| **Survivorship bias** | All stocks existed through the full period. Companies that failed are excluded. Estimated +2–5% upward bias annually. |
| **Bull market training** | Period covers 2009–2026, one of the longest bull runs in history. Results may not generalise to bear markets. |
| **No transaction costs** | Real implementation would incur bid-ask spread, slippage, and taxes. These reduce returns meaningfully for monthly rebalancing. |
| **Universe pre-selection** | Core 30 stocks were chosen for being today's large-caps — they are quality survivors, not a random sample. |
| **Short live-test period** | Only 2 months of actual out-of-sample predictions have real outcomes. The 61.5% hit rate is simulated, not confirmed. |
    """)

st.warning(
    "**Backtesting does not guarantee future performance.** "
    "Survivorship bias and bull-market conditions inflate these results. "
    "Transaction costs and market impact are not modelled. "
    "See the methodology notes above before drawing conclusions.",
    icon="⚠️",
)


# ── Controls ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Backtest Settings")
    top_n       = st.slider("Top N stocks",        5,  20, 10)
    min_train   = st.slider("Min training months", 12, 48, 24)
    min_dq      = st.slider("Min DQ score",        0,  80, 40)
    run_btn     = st.button("Run Backtest", type="primary", use_container_width=True)
    st.caption("Changing settings and clicking Run Backtest will re-run the simulation.")


# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def _bootstrap():
    from src.database.db import initialize_db
    from src.database.queries import (load_prices_clean, load_benchmark_prices,
        load_monthly_features, load_stocks)
    from src.database.migrations import apply_migrations

    initialize_db(); apply_migrations()
    prices   = load_prices_clean()
    bench    = load_benchmark_prices()
    features = load_monthly_features(start_date="2015-01-31")
    stocks   = load_stocks()
    return prices, bench, features, stocks


prices_df, benchmark_df, features_df, stocks_df = _bootstrap()


# ── Run backtest ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running portfolio simulation…")
def _run(top_n, min_train, min_dq):
    from src.backtesting.backtest import run_backtest
    return run_backtest(
        prices_df=prices_df, benchmark_df=benchmark_df,
        stocks_df=stocks_df, features_df=features_df,
        top_n=top_n, min_train=min_train, min_dq_score=min_dq,
        save_to_db=True,
    )


# Run on first load or when button clicked
if "bt_result" not in st.session_state or run_btn:
    with st.spinner("Running simulation…"):
        st.session_state["bt_result"] = _run(top_n, min_train, min_dq)
        st.session_state["bt_params"] = (top_n, min_train, min_dq)

result = st.session_state.get("bt_result", {})

if not result:
    st.error("Backtest returned no results. Check that features and prices are loaded.")
    st.stop()

metrics   = result["metrics"]
monthly   = result["monthly_returns"]
cum       = result["cumulative_returns"]
rolling   = result["rolling_metrics"]
sector_p  = result["sector_performance"]


# ── Plain-English summary box ──────────────────────────────────────────────────
_port_ret  = metrics['total_return_portfolio']
_bench_ret = metrics['total_return_benchmark']
_excess    = metrics['total_excess_return']
_cagr_p    = metrics['ann_return_portfolio']
_cagr_b    = metrics['ann_return_benchmark']
_start     = metrics.get('start_date', '—')
_end       = metrics.get('end_date', '—')
_n_months  = metrics['n_periods']
_n_years   = round(_n_months / 12, 1)
_top_n     = st.session_state.get('bt_params', (top_n,))[0]

_val_1000_port  = 1000 * (1 + _port_ret)
_val_1000_bench = 1000 * (1 + _bench_ret)

st.info(f"""
**What does this backtest simulate?**

Starting from **{_start}**, the model selected its top **{_top_n} stocks** from the 60-stock universe each month,
held them for one month with equal weighting, then rebalanced. This was repeated for **{_n_months} months ({_n_years} years)** until **{_end}**.

**$1,000 invested example:**

| Strategy | $1,000 invested in {_start[:7]} | Value in {_end[:7]} | Annual growth (CAGR) |
|---|---|---|---|
| **This model (top {_top_n} picks)** | $1,000 | **${_val_1000_port:,.0f}** | **{_cagr_p:.1%} / year** |
| S&P 500 (buy & hold) | $1,000 | ${_val_1000_bench:,.0f} | {_cagr_b:.1%} / year |
| **Model advantage** | — | **+${_val_1000_port - _val_1000_bench:,.0f}** | **+{_cagr_p - _cagr_b:.1%} / year** |

**Which stocks?** The model rotates through its top {_top_n} picks each month — different stocks each month,
drawn from the 30 DJIA Core stocks + 30 randomly selected Hidden Gems.
The selection is based purely on the model's feature scores at that month's rebalancing date.
""", icon="💡")

st.subheader("Performance Summary")

# ── Key headline metrics ───────────────────────────────────────────────────────
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown("### Portfolio CAGR")
    st.markdown(f"<h1 style='color:#1a7a3a;margin:0'>{_cagr_p:.1%}</h1>", unsafe_allow_html=True)
    st.caption(f"Compound annual growth rate of the model portfolio — {_n_years} years")
with h2:
    st.markdown("### Benchmark CAGR (S&P 500)")
    st.markdown(f"<h1 style='color:#888;margin:0'>{_cagr_b:.1%}</h1>", unsafe_allow_html=True)
    st.caption(f"S&P 500 buy-and-hold CAGR over the same {_n_years}-year period")
with h3:
    st.markdown("### Annual outperformance")
    diff = _cagr_p - _cagr_b
    col = "#1a7a3a" if diff > 0 else "#a94442"
    st.markdown(f"<h1 style='color:{col};margin:0'>{diff:+.1%}</h1>", unsafe_allow_html=True)
    st.caption("Extra return per year vs passive S&P 500 investing")

st.markdown("")

# ── Full metrics grid ──────────────────────────────────────────────────────────
st.markdown("**All performance metrics:**")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total return (portfolio)", f"{_port_ret:.1%}")
c1.caption(f"${1000:,} → ${_val_1000_port:,.0f}  over {_n_years} years")
c2.metric("Total return (S&P 500)",   f"{_bench_ret:.1%}")
c2.caption(f"${1000:,} → ${_val_1000_bench:,.0f}  buy-and-hold")
c3.metric("Total excess return",      f"{_excess:+.1%}", delta=f"{_excess:+.1%}")
c3.caption("Model total return minus S&P 500 total return")
c4.metric("Sharpe ratio",             f"{metrics.get('sharpe_ratio', 0):.2f}" if metrics.get('sharpe_ratio') else "—")
c4.caption("Return per unit of risk. >1.0 = good. >2.0 = excellent.")
c5.metric("Hit rate",                 f"{metrics['hit_rate']:.1%}")
c5.caption("% of months the portfolio beat the S&P 500. Keyes target: >70%")
c6.metric("Max drawdown",             f"{metrics['max_drawdown_portfolio']:.1%}")
c6.caption("Worst peak-to-trough loss during the simulation")

c7, c8, c9, c10 = st.columns(4)
c7.metric("CAGR (portfolio)",  f"{_cagr_p:.1%}")
c7.caption("Compound annual growth rate — equivalent yearly return")
c8.metric("CAGR (S&P 500)",    f"{_cagr_b:.1%}")
c8.caption("S&P 500 CAGR — the passive baseline")
c9.metric("Sortino ratio",     f"{metrics.get('sortino_ratio', 0):.2f}" if metrics.get('sortino_ratio') else "—")
c9.caption("Like Sharpe but only penalises downside volatility (not upside)")
c10.metric("Months simulated", f"{_n_months}")
c10.caption(f"Walk-forward periods tested ({_n_years} years, {_start[:7]} → {_end[:7]})")

st.caption(
    f"Strategy: **{metrics.get('strategy_name','—')}**  |  "
    f"Period: **{_start}** to **{_end}**  |  "
    f"Top **{_top_n}** stocks selected each month, equal-weight, rebalanced monthly"
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# YEAR-BY-YEAR PERFORMANCE TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Year-by-Year Returns")
st.markdown(
    "How much would you have made or lost **each calendar year**, compared to simply holding the S&P 500?"
)

_mr = monthly.copy()
_mr["date"] = pd.to_datetime(_mr["date"])
_mr["year"] = _mr["date"].dt.year

# Compound monthly returns within each year
def _compound(series):
    return (1 + series).prod() - 1

yoy = (
    _mr.groupby("year")
    .agg(
        portfolio_return  = ("portfolio_return",  _compound),
        benchmark_return  = ("benchmark_return",  _compound),
        months            = ("portfolio_return",  "count"),
        beat_months       = ("hit",               "sum"),
    )
    .reset_index()
)
yoy["excess_return"] = yoy["portfolio_return"] - yoy["benchmark_return"]
yoy["hit_rate"]      = yoy["beat_months"] / yoy["months"]

# Running $1,000 across years
running = 1000.0
running_bench = 1000.0
rows_display = []
for _, r in yoy.iterrows():
    running       *= (1 + r["portfolio_return"])
    running_bench *= (1 + r["benchmark_return"])
    full = r["months"] >= 10  # flag partial years
    rows_display.append({
        "Year":            int(r["year"]),
        "Portfolio":       r["portfolio_return"],
        "S&P 500":         r["benchmark_return"],
        "Outperformed?":   "✅ Yes" if r["portfolio_return"] > r["benchmark_return"] else "❌ No",
        "Excess":          r["excess_return"],
        "Months beat / total": f"{int(r['beat_months'])}/{int(r['months'])}",
        "$1,000 → (portfolio)": f"${running:,.0f}",
        "Note":            "" if full else "⚠️ Partial year",
    })

yoy_display = pd.DataFrame(rows_display)

def _colour_yoy(row):
    styles = [""] * len(row)
    cols = list(row.index)
    p_idx = cols.index("Portfolio")
    b_idx = cols.index("S&P 500")
    e_idx = cols.index("Excess")
    p_val = row["Portfolio"]
    e_val = row["Excess"]
    if isinstance(p_val, float):
        styles[p_idx] = "color:#1a7a3a;font-weight:bold" if p_val > 0 else "color:#a94442;font-weight:bold"
    if isinstance(e_val, float):
        styles[e_idx] = "color:#1a7a3a;font-weight:bold" if e_val > 0 else "color:#a94442;font-weight:bold"
    return styles

fmt_yoy = {
    "Portfolio": "{:+.1%}",
    "S&P 500":   "{:+.1%}",
    "Excess":    "{:+.1%}",
}
st.dataframe(
    yoy_display.style.apply(_colour_yoy, axis=1).format(fmt_yoy, na_rep="—"),
    use_container_width=True,
    hide_index=True,
    height=35 + 35 * len(yoy_display),
)
st.caption(
    "Portfolio = equal-weight top-N picks rebalanced monthly.  "
    "Excess = portfolio minus S&P 500 for that year.  "
    "Partial years (< 10 months) flagged with ⚠️."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TOP PICKS HISTORY — WHICH TICKERS WERE SELECTED EACH MONTH
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Top Picks — Month-by-Month History")
st.markdown(
    f"The exact **{_top_n} tickers** selected each month, the portfolio's monthly return, "
    "and whether it beat the S&P 500 that month. "
    "Stocks are listed in the order the model ranked them."
)

_picks = monthly.copy()
_picks["date"] = pd.to_datetime(_picks["date"])
_picks = _picks.sort_values("date", ascending=False).reset_index(drop=True)

# Build display table
picks_rows = []
for _, row in _picks.iterrows():
    tickers = row.get("selected_tickers", [])
    if isinstance(tickers, str):
        import ast as _ast
        try: tickers = _ast.literal_eval(tickers)
        except: tickers = tickers.split(",")
    picks_rows.append({
        "Month":           row["date"].strftime("%Y-%m"),
        "Top picks":       "  ·  ".join(tickers) if tickers else "—",
        "# stocks":        len(tickers),
        "Portfolio return": row["portfolio_return"],
        "S&P 500 return":  row["benchmark_return"],
        "Excess":          row["excess_return"],
        "Beat S&P 500?":   "✅" if row.get("hit") else "❌",
    })

picks_df = pd.DataFrame(picks_rows)

def _colour_picks(row):
    styles = [""] * len(row)
    cols = list(row.index)
    for col_name, colour_pos in [("Portfolio return", "Portfolio return"), ("Excess", "Excess")]:
        if col_name in cols:
            i = cols.index(col_name)
            v = row[col_name]
            if isinstance(v, float):
                styles[i] = "color:#1a7a3a;font-weight:bold" if v > 0 else "color:#a94442;font-weight:bold"
    return styles

fmt_picks = {
    "Portfolio return": "{:+.1%}",
    "S&P 500 return":  "{:+.1%}",
    "Excess":          "{:+.1%}",
}
st.dataframe(
    picks_df.style.apply(_colour_picks, axis=1).format(fmt_picks, na_rep="—"),
    use_container_width=True,
    hide_index=True,
    height=min(35 + 35 * len(picks_df), 800),
    column_config={
        "Top picks": st.column_config.TextColumn("Top picks", width="large"),
        "Month":     st.column_config.TextColumn("Month",     width="small"),
    },
)
st.caption(
    "Most recent months shown first.  "
    "Each row = one rebalancing period.  "
    "Portfolio return = equal-weight average of all picks for that month."
)

st.markdown("---")


# ── Cumulative return chart ────────────────────────────────────────────────────
st.subheader("Cumulative Return: Portfolio vs Benchmark")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum["date"], y=cum["portfolio_cumret"],
    mode="lines", name="Portfolio (top-N)",
    line=dict(color="#1f77b4", width=2.5),
))
fig_cum.add_trace(go.Scatter(
    x=cum["date"], y=cum["benchmark_cumret"],
    mode="lines", name="Benchmark (SPY)",
    line=dict(color="#aaaaaa", width=1.5, dash="dash"),
))
fig_cum.add_hline(y=1, line_color="black", line_width=0.5)
fig_cum.update_layout(
    height=400, xaxis_title="Date", yaxis_title="Growth of $1",
    legend=dict(orientation="h", y=-0.2),
    title="Cumulative growth — portfolio vs benchmark",
)
st.plotly_chart(fig_cum, use_container_width=True)

# Relative performance (portfolio / benchmark)
st.subheader("Relative Performance (Portfolio / Benchmark)")
fig_rel = go.Figure(go.Scatter(
    x=cum["date"], y=cum["relative_cumret"],
    mode="lines", fill="tozeroy",
    line=dict(color="#1a7a3a", width=2),
    fillcolor="rgba(26,122,58,0.15)",
))
fig_rel.add_hline(y=1, line_color="red", line_dash="dash", line_width=1,
                  annotation_text="No excess return")
fig_rel.update_layout(
    height=280, xaxis_title="Date",
    yaxis_title="Portfolio / Benchmark",
    title="Relative cumulative return  (>1 = outperforming benchmark)",
)
st.plotly_chart(fig_rel, use_container_width=True)

st.markdown("---")


# ── Monthly excess return bars ─────────────────────────────────────────────────
st.subheader("Monthly Excess Return")
monthly_plot = monthly.copy()
monthly_plot["colour"] = monthly_plot["excess_return"].apply(
    lambda v: "#1a7a3a" if v >= 0 else "#a94442"
)
fig_bar = go.Figure(go.Bar(
    x=monthly_plot["date"],
    y=monthly_plot["excess_return"],
    marker_color=monthly_plot["colour"],
    hovertemplate="Date: %{x}<br>Excess: %{y:.2%}<extra></extra>",
))
fig_bar.add_hline(y=0, line_color="black", line_width=0.8)
fig_bar.update_layout(
    height=280,
    xaxis_title="Date", yaxis_title="Excess return (portfolio − benchmark)",
    title="Monthly excess return  (green = portfolio beat benchmark)",
    yaxis_tickformat=".1%",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")


# ── Rolling metrics ────────────────────────────────────────────────────────────
st.subheader("Rolling 12-Month Metrics")
col_r1, col_r2 = st.columns(2)

with col_r1:
    roll_valid = rolling.dropna(subset=["rolling_sharpe"])
    if not roll_valid.empty:
        fig_rs = go.Figure(go.Scatter(
            x=roll_valid["date"], y=roll_valid["rolling_sharpe"],
            mode="lines", line=dict(color="#1f77b4", width=2), name="Rolling Sharpe",
        ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
        fig_rs.add_hline(y=1, line_dash="dot", line_color="green", line_width=1,
                         annotation_text="Sharpe = 1")
        fig_rs.update_layout(height=280, title="12-month rolling Sharpe ratio",
                             xaxis_title="Date", yaxis_title="Sharpe")
        st.plotly_chart(fig_rs, use_container_width=True)

with col_r2:
    roll_hit_valid = rolling.dropna(subset=["rolling_hit"])
    if not roll_hit_valid.empty:
        fig_rh = go.Figure(go.Scatter(
            x=roll_hit_valid["date"], y=roll_hit_valid["rolling_hit"],
            mode="lines", fill="tozeroy",
            line=dict(color="#ff7f0e", width=2), name="Rolling hit rate",
            fillcolor="rgba(255,127,14,0.15)",
        ))
        fig_rh.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=1,
                         annotation_text="50% baseline")
        fig_rh.update_layout(height=280, title="12-month rolling hit rate",
                             xaxis_title="Date", yaxis_title="Hit rate",
                             yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig_rh, use_container_width=True)

st.markdown("---")


# ── Sector performance ─────────────────────────────────────────────────────────
if not sector_p.empty:
    st.subheader("Performance by Sector")
    fig_sec = px.bar(
        sector_p.sort_values("mean_excess"),
        x="mean_excess", y="sector", orientation="h",
        color="mean_excess",
        color_continuous_scale=["#a94442", "#ffffff", "#1a7a3a"],
        color_continuous_midpoint=0,
        text=sector_p.sort_values("mean_excess")["mean_excess"].apply(lambda v: f"{v:+.2%}"),
        title="Mean monthly excess return by sector (when stocks from that sector were selected)",
        labels={"mean_excess": "Mean excess return", "sector": "Sector"},
    )
    fig_sec.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_sec, use_container_width=True)

    st.dataframe(
        sector_p.rename(columns={
            "sector": "Sector",
            "n_stock_months": "Stock-months selected",
            "mean_excess": "Mean excess return",
            "hit_rate": "Hit rate",
        }).style.format({
            "Mean excess return": "{:+.2%}",
            "Hit rate": "{:.1%}",
        }),
        use_container_width=True, height=280,
    )
    st.markdown("---")


# ── Full metrics table ────────────────────────────────────────────────────────
with st.expander("Full metrics table"):
    rows = [
        ("Total return — portfolio",   f"{metrics['total_return_portfolio']:.2%}"),
        ("Total return — benchmark",   f"{metrics['total_return_benchmark']:.2%}"),
        ("Total excess return",        f"{metrics['total_excess_return']:+.2%}"),
        ("Ann. return — portfolio",    f"{metrics['ann_return_portfolio']:.2%}"),
        ("Ann. return — benchmark",    f"{metrics['ann_return_benchmark']:.2%}"),
        ("Ann. excess return",         f"{metrics['ann_excess_return']:+.2%}"),
        ("Ann. volatility — portfolio",f"{metrics['ann_volatility_portfolio']:.2%}"),
        ("Ann. volatility — benchmark",f"{metrics['ann_volatility_benchmark']:.2%}"),
        ("Sharpe ratio",               f"{metrics.get('sharpe_ratio') or '—'}"),
        ("Sortino ratio",              f"{metrics.get('sortino_ratio') or '—'}"),
        ("Information ratio",          f"{metrics.get('information_ratio') or '—'}"),
        ("Calmar ratio",               f"{metrics.get('calmar_ratio') or '—'}"),
        ("Max drawdown — portfolio",   f"{metrics['max_drawdown_portfolio']:.2%}"),
        ("Max drawdown — benchmark",   f"{metrics['max_drawdown_benchmark']:.2%}"),
        ("Hit rate",                   f"{metrics['hit_rate']:.1%}"),
        ("Months simulated",           str(metrics["n_periods"])),
        ("Start date",                 metrics.get("start_date", "—")),
        ("End date",                   metrics.get("end_date", "—")),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        use_container_width=True, hide_index=True, height=520,
    )


with st.expander("How to read the backtest"):
    st.markdown("""
**What is being simulated**
Each month, stocks are scored using feature-based signals (momentum, fundamentals,
valuation, risk, data quality).  The top-N ranked stocks are held in an equal-weight
portfolio for one month, then rebalanced.  No transaction costs are modelled.

**Why feature-based scoring (not model probabilities)**
Re-training the logistic regression and Ridge models at every month would take
hours.  The feature-based component scores are fast to compute and already
look-ahead-safe.  They represent the fundamental signal layer without needing
model training at each step.

**Hit rate**
Fraction of months where the portfolio beat the benchmark.
A random portfolio has hit rate ≈ 50%.  Above 55% is meaningful.

**Sharpe ratio**
Annualised return divided by annualised volatility.
Above 0.5 is reasonable; above 1.0 is strong.

**Sortino ratio**
Same as Sharpe but only penalises downside volatility.
A higher Sortino than Sharpe means the portfolio's volatility is mostly upside.

**Max drawdown**
The largest peak-to-trough decline in cumulative portfolio value.
A drawdown of -20% means at some point the portfolio had lost 20% from its peak.

**Information ratio**
Annualised excess return divided by tracking error.
Measures consistency: a high IR means the portfolio reliably beats the benchmark.

**Critical limitations**
• Synthetic sample data — results do not reflect real market dynamics
• No transaction costs, slippage, or market impact
• Survivorship bias: universe only includes currently-trading stocks
• The VIF filtering and feature selection used full-sample data
• With 30 stocks, results are highly sensitive to the specific random paths in the sample data
    """)
