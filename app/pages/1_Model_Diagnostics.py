"""
app/pages/1_Model_Diagnostics.py

Model Diagnostics — Steps 9 & 10

Tabs:
  1. Correlation Analysis  — IC, Spearman rank, BH correction, quintiles
  2. Simple Regression     — per-feature OLS coefficients, R², train/test split,
                             scatter plot, residuals, coefficient stability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Diagnostics", layout="wide")

st.markdown(
    "<div style='background:#1a3a5c;color:white;padding:10px 16px;border-radius:6px;"
    "font-size:0.9em;margin-bottom:8px'>"
    "🔬 <strong>PHASE 1 — SIGNAL VALIDATION</strong> &nbsp;|&nbsp; "
    "These pages test whether the statistical variables actually predict stock performance. "
    "Nothing here is a buy/sell recommendation — this is pure research validation."
    "</div>",
    unsafe_allow_html=True,
)

st.title("Model Diagnostics")
st.markdown(
    "Statistical evidence for the predictive signals — what variables are linked to future "
    "stock performance, how strong the links are, and whether they survive rigorous testing. "
    "This is the core validation of the Keyes (1972) replication methodology."
)

st.warning(
    "Results are computed on real market data from Yahoo Finance. "
    "Statistical significance does not guarantee out-of-sample performance.",
    icon="⚠️",
)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Run Analysis", type="primary", use_container_width=True,
                 help="Re-run all statistical tests. Results are cached until you click this."):
        st.cache_data.clear()
        st.rerun()

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running analysis…")
def _load_all():
    from src.database.db import initialize_db
    from src.database.queries import (load_monthly_features_with_sector,
                                      load_features_and_targets)
    from src.database.migrations import apply_migrations
    from src.models.correlation import run_correlation_analysis
    from src.models.simple_regression import run_all_simple_regressions
    from src.models.multiple_regression import run_multiple_regression
    from src.models.logistic_regression import run_logistic_regression
    from src.models.regularized_models import run_regularized_models, compute_lasso_path

    initialize_db()
    apply_migrations()

    feats_ws = load_monthly_features_with_sector(start_date="2015-01-31")
    ft       = load_features_and_targets()

    # ── Check whether model results are already stored in the DB ──────────────
    # Model training is the most expensive part (~35 s).  If the DB already
    # has results from a prior run, skip retraining and use those instead.
    # Results are invalidated when new features are computed (new feature dates).
    from src.database.queries import load_model_training_runs, load_latest_model_version
    from src.database.db import save_model_training_run
    import json, hashlib, pickle

    # Use a hash of the latest feature date as a cheap cache key
    latest_feat_date = ft["feature_date"].max() if not ft.empty else "none"
    cache_key = hashlib.md5(latest_feat_date.encode()).hexdigest()[:8]

    existing = load_model_training_runs()
    _REQUIRED_KEYS = {"corr_pickle", "reg_summary_json", "multi_pickle",
                      "logit_pickle", "reg_models_pickle"}

    def _cache_is_complete(df: pd.DataFrame, version: str) -> bool:
        if df.empty or version not in df["model_version"].values:
            return False
        row = df[df["model_version"] == version].iloc[0]
        try:
            cached = json.loads(row["metrics_json"])
            return _REQUIRED_KEYS.issubset(cached.keys())
        except Exception:
            return False

    already_trained = _cache_is_complete(existing, f"diag_{cache_key}")

    if already_trained:
        # Load cached results from DB — no model retraining needed
        from io import StringIO
        row = existing[existing["model_version"] == f"diag_{cache_key}"].iloc[0]
        cached = json.loads(row["metrics_json"])
        corr_results  = pickle.loads(bytes.fromhex(cached["corr_pickle"]))
        reg_summary   = pd.read_json(StringIO(cached["reg_summary_json"]))
        multi_result  = pickle.loads(bytes.fromhex(cached["multi_pickle"]))
        logit_result  = pickle.loads(bytes.fromhex(cached["logit_pickle"]))
        reg_models    = pickle.loads(bytes.fromhex(cached["reg_models_pickle"]))
        lasso_path_df = pd.read_json(StringIO(cached["lasso_path_json"])) if cached.get("lasso_path_json") else pd.DataFrame()
    else:
        # Train all models (first time or after new data)
        corr_results   = run_correlation_analysis(feats_ws, ft)
        reg_summary    = run_all_simple_regressions(ft)
        multi_result   = run_multiple_regression(ft)
        logit_result   = run_logistic_regression(ft)
        reg_models     = run_regularized_models(ft)
        lasso_path_df  = compute_lasso_path(ft)

        # Persist to DB so subsequent loads are instant
        try:
            # sklearn objects and complex dicts need pickle; DataFrames use JSON
            # Strip sklearn model objects (not serialisable to JSON cleanly)
            def _strip(d):
                if not isinstance(d, dict): return d
                return {k: v for k, v in d.items()
                        if k not in ("sklearn_model", "scaler", "model", "sm_result")}

            metrics_payload = {
                "corr_pickle":       pickle.dumps(corr_results).hex(),
                "reg_summary_json":  reg_summary.to_json(),
                "multi_pickle":      pickle.dumps(_strip(multi_result)).hex(),
                "logit_pickle":      pickle.dumps(_strip(logit_result)).hex(),
                "reg_models_pickle": pickle.dumps({
                    "ridge":            _strip(reg_models.get("ridge", {})),
                    "lasso":            _strip(reg_models.get("lasso", {})),
                    "comparison_table": reg_models.get("comparison_table", pd.DataFrame()),
                }).hex(),
                "lasso_path_json":   lasso_path_df.to_json() if not lasso_path_df.empty else None,
            }
            save_model_training_run({
                "model_version":       f"diag_{cache_key}",
                "training_start_date": ft["feature_date"].min() if not ft.empty else "2015-01-31",
                "training_end_date":   latest_feat_date,
                "model_type":          "diagnostics_bundle",
                "features_used":       json.dumps(list(ft.columns)),
                "metrics_json":        json.dumps(metrics_payload),
            })
        except Exception:
            pass  # Caching failure is non-fatal — results still returned

    return corr_results, reg_summary, multi_result, logit_result, reg_models, lasso_path_df, ft


corr_results, reg_summary, multi_result, logit_result, reg_models, lasso_path_df, ft_df = _load_all()

if not corr_results or reg_summary.empty:
    st.error("No labelled data available. Run the monthly update pipeline first.")
    st.stop()

summary_df  = corr_results["summary"]
ic_series   = corr_results["ic_series"]
n_obs       = corr_results["n_obs"]
n_tested    = corr_results["n_features_tested"]
top_signals = corr_results["top_features"]
merged_df   = corr_results["merged_df"]


# ── Human-readable feature name lookup ────────────────────────────────────────
FEAT_NAMES = {
    "price_to_book":             "Price-to-Book (P/B)",
    "price_to_earnings":         "Price-to-Earnings (P/E)",
    "price_to_sales":            "Price-to-Sales (P/S)",
    "current_pe_ratio":          "Current P/E Ratio",
    "dividend_yield":            "Dividend Yield",
    "roe":                       "Return on Equity (ROE)",
    "roic":                      "Return on Assets (ROA)",
    "free_cash_flow_yield":      "Free Cash Flow Yield",
    "five_year_revenue_growth":  "5yr Revenue Growth (X12 proxy)",
    "five_year_eps_growth":      "5yr EPS Growth (X5)",
    "five_year_price_gain":      "5yr Price Gain (X6)",
    "twelve_month_momentum":     "12-Month Momentum",
    "six_month_momentum":        "6-Month Momentum",
    "return_1m":                 "1-Month Return",
    "return_3m":                 "3-Month Return",
    "market_cap":                "Market Capitalisation",
    "beta":                      "Beta (Market Sensitivity)",
    "volatility_3m":             "3-Month Volatility",
    "volatility_12m":            "12-Month Volatility",
    "downside_volatility_12m":   "Downside Volatility",
    "drawdown_from_52w_high":    "Drawdown from 52w High",
    "abnormal_volume":           "Abnormal Trading Volume",
    "debt_to_equity":            "Debt-to-Equity",
    "gross_margin":              "Gross Margin",
    "operating_margin":          "Operating Margin",
    "pe_vs_historical_median":   "P/E vs Historical Median (X9)",
    "sector_relative_pe":        "Sector-Relative P/E",
    "sector_relative_ps":        "Sector-Relative P/S",
    "sector_relative_fcf_yield": "Sector-Relative FCF Yield",
    "sector_relative_momentum":  "Sector-Relative Momentum",
    "peer_momentum_zscore":      "Peer Momentum Z-Score",
    "peer_valuation_zscore":     "Peer Valuation Z-Score",
    "revenue_growth_acceleration":"Revenue Growth Acceleration",
    "eps_growth_acceleration":   "EPS Growth Acceleration",
}

# ── Top-level metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stock-months",      f"{n_obs:,}")
c1.caption("Total observations (note: 60 tickers × months — not fully independent)")
c2.metric("Features tested",   n_tested)
c2.caption("Variables tested against future 12m excess return")
c3.metric("BH-significant",    len(top_signals))
c3.caption("Survive Benjamini-Hochberg multiple-testing correction")
c4.metric("Best IC",           f"{summary_df['spearman_ic'].abs().max():.3f}")
c4.caption("Spearman rank correlation (IC > 0.10 = strong signal)")
c5.metric("Best R²",           f"{reg_summary['r_squared'].max():.3f}")
c5.caption("Single-variable OLS explanatory power (low R² is normal in finance)")

st.markdown("---")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Correlation Analysis", "Simple Regression",
    "Multiple Regression",  "Logistic Regression",
    "Ridge & Lasso",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Feature Signal Strength (IC Table)")
    st.markdown(f"""
This table answers: **which company variables have a statistically reliable relationship
with whether a stock beats the S&P 500 over the next 12 months?**

| Metric | What it means | Threshold |
|---|---|---|
| **IC** | Mean monthly Spearman rank correlation between the variable and future 12-month excess return. Positive = higher value predicts better performance. | > 0.05 useful · > 0.10 strong |
| **IC t-stat** | How many standard deviations the IC is from zero. Measures statistical reliability. | > 2.0 significant |
| **ICIR** | Information Coefficient Information Ratio = IC ÷ std(IC). Measures consistency month-to-month. | > 0.50 consistent |
| **Pearson r** | Linear correlation (less robust than Spearman for non-normal return data). | — |
| **BH p** | P-value after Benjamini-Hochberg correction for {n_tested} simultaneous tests. Controls false discovery rate. | < 0.05 significant |
| **Stability** | Fraction of calendar years in which the IC had the correct sign. 1.00 = correct direction every year. | 1.00 = perfectly stable |

**Key limitation:** 12-month return windows overlap (Jan–Feb share 11 months), which inflates t-statistics by approximately 3–5×. Treat BH-significant results as indicative, not definitive. See the Annual IC section below for overlap-free validation.
    """)

    disp = summary_df[[
        "feature", "spearman_ic", "ic_tstat", "ic_ir",
        "pearson_r", "p_bh_fdr", "bh_significant",
        "stability_score", "n_obs",
    ]].copy()
    disp["feature"] = disp["feature"].map(lambda x: FEAT_NAMES.get(x, x))
    disp = disp.rename(columns={
        "feature":        "Variable",
        "spearman_ic":    "IC",
        "ic_tstat":       "IC t-stat",
        "ic_ir":          "ICIR",
        "pearson_r":      "Pearson r",
        "p_bh_fdr":       "BH p",
        "bh_significant": "Sig.",
        "stability_score":"Stability",
        "n_obs":          "N",
    })

    def _colour_ic(val):
        if pd.isna(val): return ""
        if val >  0.10: return "background-color:#1a7a3a;color:white"
        if val >  0.05: return "background-color:#5cb85c;color:white"
        if val >  0.00: return "background-color:#d4edda"
        if val > -0.05: return "background-color:#f8d7da"
        return "background-color:#a94442;color:white"

    st.dataframe(
        disp.style
            .map(_colour_ic, subset=["IC"])
            .format({"IC":"{:.4f}","IC t-stat":"{:.2f}","ICIR":"{:.2f}",
                     "Pearson r":"{:.4f}","BH p":"{:.4f}","Stability":"{:.2f}"},
                    na_rep="—"),
        use_container_width=True, height=550,
    )

    # IC bar chart
    st.subheader("IC Ranking — All Features")
    plot_df = summary_df[["feature","spearman_ic","bh_significant"]].dropna(subset=["spearman_ic"])
    plot_df = plot_df.sort_values("spearman_ic")
    colours = plot_df.apply(
        lambda r: "#1a7a3a" if (r.bh_significant and r.spearman_ic > 0)
                  else ("#a94442" if (r.bh_significant and r.spearman_ic < 0)
                  else "#aaaaaa"), axis=1,
    )
    fig_bar = go.Figure(go.Bar(
        x=plot_df["spearman_ic"], y=plot_df["feature"],
        orientation="h", marker_color=colours,
        hovertemplate="%{y}: IC=%{x:.4f}<extra></extra>",
    ))
    fig_bar.add_vline(x=0,     line_width=1, line_color="black")
    fig_bar.add_vline(x= 0.05, line_dash="dash", line_color="green", annotation_text="0.05")
    fig_bar.add_vline(x=-0.05, line_dash="dash", line_color="red",   annotation_text="-0.05")
    fig_bar.update_layout(height=600, xaxis_title="Mean monthly IC",
                          margin=dict(l=220),
                          title="Green=BH-sig positive  |  Red=BH-sig negative  |  Grey=not significant")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Rolling IC for selected feature
    st.subheader("Rolling IC Stability")
    feat_opts = [f for f in summary_df["feature"].tolist()
                 if f in ic_series and not ic_series[f].empty]
    if feat_opts:
        sel_ic = st.selectbox("Feature:", feat_opts, key="ic_feat")
        ic_s   = ic_series[sel_ic]
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(x=ic_s.index, y=ic_s.values,
            mode="lines", name="Monthly IC", line=dict(color="#aaa", width=1), opacity=0.6))
        fig_ic.add_trace(go.Scatter(x=ic_s.rolling(36).mean().index,
            y=ic_s.rolling(36).mean().values,
            mode="lines", name="36m rolling mean", line=dict(color="#1a7a3a", width=2)))
        fig_ic.add_hline(y=0, line_color="black", line_width=1)
        fig_ic.add_hline(y= 0.05, line_dash="dash", line_color="green", opacity=0.5)
        fig_ic.add_hline(y=-0.05, line_dash="dash", line_color="red",   opacity=0.5)
        fig_ic.update_layout(height=300, title=f"IC over time: {sel_ic}",
                             xaxis_title="Date", yaxis_title="IC",
                             legend=dict(orientation="h"))
        st.plotly_chart(fig_ic, use_container_width=True)

    # ── Annual (non-overlapping) IC ───────────────────────────────────────────
    st.subheader("Annual IC — Overlap-Free Validation")
    st.markdown("""
Monthly IC uses overlapping 12-month windows (Jan–Feb share 11 months), which
artificially inflates t-statistics. **Annual IC** uses only December month-end observations
so each year's data is fully independent. If signals remain significant here,
the monthly results are not an artefact of window overlap.
    """)
    try:
        from src.database.queries import load_features_and_targets
        ft_annual = ft_df.copy() if not ft_df.empty else load_features_and_targets()
        ft_annual["feature_date"] = pd.to_datetime(ft_annual["feature_date"])
        # December only = non-overlapping annual windows
        ft_dec = ft_annual[ft_annual["feature_date"].dt.month == 12].copy()
        if len(ft_dec) >= 20:
            from src.models.correlation import run_correlation_analysis
            _target_cols = ["future_12m_excess_return", "future_12m_return", "winner"]
            feats_dec = ft_dec.drop(columns=[c for c in _target_cols if c in ft_dec.columns])
            ann_corr = run_correlation_analysis(feats_dec, ft_dec)
            if ann_corr and "summary" in ann_corr:
                ann_df = ann_corr["summary"][["feature","spearman_ic","p_bh_fdr","bh_significant","stability_score","n_obs"]].copy()
                ann_df["feature"] = ann_df["feature"].map(lambda x: FEAT_NAMES.get(x, x))
                ann_df = ann_df.rename(columns={
                    "feature":"Variable","spearman_ic":"Annual IC",
                    "p_bh_fdr":"BH p","bh_significant":"Sig.",
                    "stability_score":"Stability","n_obs":"N (years)",
                })
                st.caption(f"Based on {ann_df['N (years)'].max():.0f} non-overlapping December observations per feature")
                st.dataframe(
                    ann_df.style
                        .map(lambda v: ("background-color:#1a7a3a;color:white" if (isinstance(v,float) and v > 0.08)
                                   else ("background-color:#a94442;color:white" if (isinstance(v,float) and v < -0.08)
                                   else "")), subset=["Annual IC"])
                        .format({"Annual IC":"{:.4f}","BH p":"{:.4f}","Stability":"{:.2f}"}, na_rep="—"),
                    use_container_width=True, height=420, hide_index=True,
                )
        else:
            st.info(f"Only {len(ft_dec)} December observations available — need 20+ for reliable annual IC.")
    except Exception as e:
        st.warning(f"Could not compute annual IC: {e}")

    # Quintile analysis
    st.subheader("Quintile Analysis")
    from src.models.correlation import get_quintile_for_feature
    sel_q = st.selectbox("Feature:", feat_opts, key="q_feat")
    q_df  = get_quintile_for_feature(corr_results, sel_q)
    if not q_df.empty:
        fig_q = px.bar(q_df, x="quantile", y="mean_excess_return",
            color="mean_excess_return",
            color_continuous_scale=["#a94442","#ffffff","#1a7a3a"],
            color_continuous_midpoint=0,
            labels={"quantile":"Quintile (1=low, 5=high)",
                    "mean_excess_return":"Mean 12m Excess Return"},
            title=f"Avg future excess return by quintile: {sel_q}",
            text_auto=".3f")
        fig_q.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True)
        if "spread_vs_q1" in q_df.columns:
            st.metric("Q5 minus Q1 spread", f"{q_df['spread_vs_q1'].iloc[-1]:+.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMPLE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Simple Linear Regression — One Feature at a Time")

    train_pct   = int(reg_summary["train_cutoff"].notna().sum() /
                      max(len(reg_summary), 1) * 100)
    n_sig_reg   = int((reg_summary["p_value"] < 0.05).sum())
    best_r2_oos = reg_summary["r_squared_test"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features run",           len(reg_summary))
    c2.metric("Significant (p<0.05)",   n_sig_reg)
    c3.metric("Best in-sample R²",      f"{reg_summary['r_squared'].max():.4f}")
    c4.metric("Best out-of-sample R²",  f"{best_r2_oos:.4f}" if pd.notna(best_r2_oos) else "—")

    st.markdown("""
Each row is one OLS regression: `excess_return = α + β × feature`.

**Coefficient** = predicted change in excess return per 1-unit increase in the feature.
**Coef (std)** = predicted change per **1 SD** increase — comparable across features.
**R²** = fraction of return variance explained by this feature alone.
**R² (test)** = out-of-sample R² on the held-out 30% of dates — the honest measure.
A negative out-of-sample R² means the model predicts worse than simply using the mean.
    """)

    # ── Keyes (1972) variable replication note ────────────────────────────────
    with st.expander("Research note: Keyes (1972) variable replication"):
        st.markdown("""
The Keyes (1972) study found positive correlations for all five predictor variables
(X5–X12) with future stock price appreciation. Our replication using 2015–2026 data
reveals one important difference:

| Variable | Keyes (1972) sign | Our finding | Explanation |
|---|---|---|---|
| X5 (5yr EPS growth) | Positive | **Positive ✓** | Replicated |
| X6 (5yr price gain) | Positive | **Negative / NS** | **Does not replicate** — see note below |
| X8 (P/E ratio) | Positive | **Positive ✓** | Replicated |
| X9 (P/E vs historical) | Positive | **Negative** | Sign reversed — higher P/E vs median predicts lower excess return |
| X12 (revenue growth proxy) | Positive | **Positive ✓** | Replicated with proxy variable |

**X6 sign reversal (5-year price gain):** Keyes found stocks with higher 5-year price
appreciation tended to continue outperforming (momentum). In our 2015–2026 dataset,
this relationship is near-zero (β = -0.001, p = 0.48, R² ≈ 0). This may reflect
mean-reversion dynamics in our pre-selected quality universe, or the difference between
absolute returns (Keyes) and excess returns over the S&P 500 benchmark (this study).

**X9 sign reversal:** Stocks trading at a high premium to their own historical P/E
show lower future excess returns. This is economically sensible — expensive stocks tend
to underperform — and represents a modernisation of the Keyes finding rather than
a contradiction.

Both are important findings to disclose and discuss in the research paper.
        """)

    # Colour helper
    def _colour_r2(val):
        if pd.isna(val): return ""
        if val > 0.05:  return "background-color:#1a7a3a;color:white"
        if val > 0.02:  return "background-color:#5cb85c;color:white"
        if val > 0.00:  return "background-color:#d4edda"
        if val >= -0.02: return "background-color:#f8d7da"
        return "background-color:#a94442;color:white"

    def _colour_p(val):
        if pd.isna(val): return ""
        return "background-color:#d4edda" if val < 0.05 else ""

    reg_disp = reg_summary[[
        "feature","n_obs","coefficient","coef_std","std_error",
        "t_stat","p_value","r_squared","r_squared_test",
        "coef_ci_lower","coef_ci_upper","significant",
    ]].rename(columns={
        "coefficient":    "Coef (β)",
        "coef_std":       "Coef (std)",
        "std_error":      "Std Err",
        "t_stat":         "t-stat",
        "p_value":        "p-value",
        "r_squared":      "R²",
        "r_squared_test": "R² (test)",
        "coef_ci_lower":  "CI lower",
        "coef_ci_upper":  "CI upper",
        "significant":    "Sig.",
        "n_obs":          "N",
    })

    st.dataframe(
        reg_disp.style
            .map(_colour_r2, subset=["R²","R² (test)"])
            .map(_colour_p,  subset=["p-value"])
            .format({
                "Coef (β)":   "{:.6f}",
                "Coef (std)": "{:.4f}",
                "Std Err":    "{:.6f}",
                "t-stat":     "{:.2f}",
                "p-value":    "{:.4f}",
                "R²":         "{:.4f}",
                "R² (test)":  "{:.4f}",
                "CI lower":   "{:.4f}",
                "CI upper":   "{:.4f}",
            }, na_rep="—"),
        use_container_width=True, height=600,
    )

    st.markdown("---")

    # ── Feature deep-dive ──────────────────────────────────────────────────────
    st.subheader("Feature Deep-Dive")
    reg_feat_opts = reg_summary["feature"].tolist()
    sel_reg = st.selectbox("Select feature:", reg_feat_opts, key="reg_feat")

    row = reg_summary[reg_summary["feature"] == sel_reg].iloc[0]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Coefficient (β)",     f"{row['coefficient']:+.6f}")
    col_b.metric("Coef (std) per 1 SD", f"{row['coef_std']:+.4f}" if pd.notna(row['coef_std']) else "—")
    col_c.metric("R² in-sample",        f"{row['r_squared']:.4f}")
    col_d.metric("R² out-of-sample",    f"{row['r_squared_test']:.4f}" if pd.notna(row['r_squared_test']) else "—")

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Intercept (α)",    f"{row['intercept']:+.6f}")
    col_f.metric("t-statistic",      f"{row['t_stat']:.3f}")
    col_g.metric("p-value",          f"{row['p_value']:.4f}")
    col_h.metric("Observations",     f"{int(row['n_obs']):,}")

    if pd.notna(row['coef_ci_lower']):
        st.caption(
            f"95% confidence interval for β: "
            f"[{row['coef_ci_lower']:.6f}, {row['coef_ci_upper']:.6f}]"
        )

    st.markdown(f"**Interpretation:** {row['interpretation']}")

    # Regression scatter plot
    from src.models.simple_regression import get_regression_plot_data, get_residual_stats, compute_rolling_coefficients
    plot_df = get_regression_plot_data(ft_df, sel_reg)

    if not plot_df.empty:
        left, right = st.columns(2)

        with left:
            st.markdown("**Regression scatter: feature vs excess return**")
            fig_s = px.scatter(
                plot_df, x=sel_reg, y="future_12m_excess_return",
                opacity=0.35,
                labels={sel_reg: sel_reg, "future_12m_excess_return": "12m Excess Return"},
                trendline="ols",
                trendline_color_override="#d62728",
            )
            fig_s.update_layout(height=350)
            st.plotly_chart(fig_s, use_container_width=True)

        with right:
            st.markdown("**Residuals distribution**")
            r_stats = get_residual_stats(plot_df)
            fig_r = px.histogram(
                plot_df, x="residual", nbins=40,
                labels={"residual": "Residual"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_r.add_vline(x=0, line_color="red", line_width=2)
            fig_r.update_layout(height=350)
            st.plotly_chart(fig_r, use_container_width=True)

            st.caption(
                f"Mean={r_stats['mean']:.4f}  "
                f"Std={r_stats['std']:.4f}  "
                f"Skew={r_stats['skew']:.2f}  "
                f"Kurt={r_stats['kurtosis']:.2f}"
            )

    # Rolling coefficient stability
    st.markdown("**Coefficient stability over time (36-month rolling windows)**")
    roll_df = compute_rolling_coefficients(ft_df, sel_reg)
    if not roll_df.empty:
        pct_pos = roll_df.attrs.get("pct_positive_windows", None)
        sign_ch = roll_df.attrs.get("sign_changes", None)

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=roll_df["end_date"], y=roll_df["coefficient"],
            mode="lines+markers", name="Rolling β",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        ))
        fig_roll.add_hline(y=0, line_color="red", line_width=1, line_dash="dash")
        fig_roll.update_layout(
            height=280,
            xaxis_title="Window end date",
            yaxis_title="β coefficient",
            title=(
                f"Rolling β — {sel_reg}  |  "
                f"{pct_pos*100:.0f}% of windows positive  |  "
                f"{sign_ch} sign changes"
            ) if pct_pos is not None else f"Rolling β — {sel_reg}",
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        if pct_pos is not None:
            if pct_pos > 0.75 or pct_pos < 0.25:
                st.success(f"Stable signal: β is {'positive' if pct_pos>0.5 else 'negative'} in {pct_pos*100:.0f}% of windows.")
            else:
                st.warning(f"Unstable signal: coefficient changes sign frequently ({sign_ch} times). Treat with caution.")
    else:
        st.info("Insufficient history for rolling analysis.")

    # ── Interpretation guide ────────────────────────────────────────────────────
    with st.expander("How to read the regression table"):
        st.markdown("""
**Coefficient (β)**
The slope of the regression line. Tells you: "for every 1-unit increase
in the feature, predicted 12-month excess return changes by β percentage points."
The sign and magnitude matter, but the raw coefficient is scale-dependent.

**Coef (std)**
The standardised coefficient. Tells you: "for every 1 standard-deviation
increase in the feature, predicted excess return changes by this many percentage points."
This allows fair comparison across features with different units and ranges.

**R²**
The fraction of return variance this single feature explains in-sample.
In financial data, R² = 0.01–0.05 is meaningful. R² = 0.10 is strong.
Do not expect R² > 0.20 from a single variable.

**R² (test)**
The R² computed on the 30% of dates held back from training.
This is the **honest** measure. A negative test R² means the model
performs worse than predicting the mean — the feature overfits.

**95% Confidence Interval**
The range within which the true β likely falls (95% of the time, in theory).
If this interval crosses zero, the coefficient is not reliably non-zero.

**Coefficient stability chart**
Shows how β changes when the regression is re-run on rolling 36-month
windows. A stable signal maintains its sign consistently. A signal that
flips positive/negative is regime-dependent and unreliable.

**Key limitation**
These regressions ignore the panel structure (same 30 stocks across many months).
Standard errors are likely understated. P-values should be treated as
directional indicators, not precise probability statements.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MULTIPLE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    from src.models.multiple_regression import (
        get_actual_vs_predicted, compute_rolling_r_squared
    )

    if not multi_result:
        st.error("Multiple regression results unavailable.")
        st.stop()

    full_m    = multi_result["full_model"]
    filt_m    = multi_result["filtered_model"]
    coef_df   = multi_result["coefficient_table"]
    vif_full  = multi_result["vif_table_full"]
    vif_filt  = multi_result["vif_table_filtered"]
    feats_sel = multi_result["features_filtered"]
    removed   = multi_result["features_removed"]

    st.subheader("Multiple Linear Regression — All Features Together")
    st.markdown("""
All features enter a single OLS regression simultaneously.
The key question: is each feature still predictive **after controlling for all others**?
Two models are shown: the full model (all features) and a VIF-filtered model
(multicollinear features removed iteratively until all VIF ≤ 10).
    """)

    # ── Model comparison metrics ───────────────────────────────────────────────
    st.subheader("Model Comparison: Full vs VIF-Filtered")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Full: features",     len(multi_result["features_full"]))
    col2.metric("Full: R²",           f"{full_m['r_squared']:.4f}")
    col3.metric("Full: Adj-R²",       f"{full_m['adj_r_squared']:.4f}")
    col4.metric("Filtered: features", len(feats_sel))
    col5.metric("Filtered: R²",       f"{filt_m.get('r_squared', 0):.4f}")
    col6.metric("Filtered: Adj-R²",   f"{filt_m.get('adj_r_squared', 0):.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full: R² (test)",      f"{full_m.get('r_squared_test', 0):.4f}" if full_m.get('r_squared_test') is not None else "—")
    c2.metric("Filtered: R² (test)",  f"{filt_m.get('r_squared_test', 0):.4f}" if filt_m.get('r_squared_test') is not None else "—")
    c3.metric("Train observations",   f"{multi_result['n_obs_train']:,}")
    c4.metric("Test observations",    f"{multi_result['n_obs_test']:,}")

    if removed:
        st.info(
            f"**{len(removed)} features removed by VIF filter:** "
            + ", ".join(f"{n} (VIF={v:.1f})" for n, v in removed)
        )

    st.markdown("---")

    # ── VIF table ──────────────────────────────────────────────────────────────
    st.subheader("Variance Inflation Factor — All Features")
    st.markdown("""
VIF measures how much each feature's coefficient is inflated by correlation with others.
**VIF < 5** = acceptable · **VIF 5–10** = moderate concern · **VIF > 10** = high concern.
    """)

    def _colour_vif(val):
        if pd.isna(val): return ""
        if val > 10: return "background-color:#a94442;color:white"
        if val >  5: return "background-color:#f0ad4e"
        return "background-color:#d4edda"

    st.dataframe(
        vif_full.style.map(_colour_vif, subset=["vif"])
                      .format({"vif": "{:.2f}"}, na_rep="—"),
        use_container_width=True, height=400,
    )

    st.markdown("---")

    # ── Coefficient table (filtered model) ────────────────────────────────────
    st.subheader(f"Coefficient Table — VIF-Filtered Model ({len(feats_sel)} features)")
    st.markdown("""
These coefficients hold **after controlling for all other features in the filtered set**.
A feature that was significant in simple regression but not here was being
proxied by another correlated feature.
    """)

    if not coef_df.empty:
        coef_display = coef_df.rename(columns={
            "coefficient": "Coef (β)",
            "std_error":   "Std Err",
            "t_stat":      "t-stat",
            "p_value":     "p-value",
            "ci_lower":    "CI lower",
            "ci_upper":    "CI upper",
            "significant": "Sig.",
        }).sort_values("t-stat", key=abs, ascending=False)

        def _colour_coef(val):
            if pd.isna(val) or val == False: return ""
            return "background-color:#d4edda" if val else ""

        def _colour_p(val):
            if pd.isna(val): return ""
            return "background-color:#d4edda" if val < 0.05 else ""

        st.dataframe(
            coef_display.style
                .map(_colour_p,    subset=["p-value"])
                .map(_colour_coef, subset=["Sig."])
                .format({
                    "Coef (β)": "{:+.6f}", "Std Err":  "{:.6f}",
                    "t-stat":   "{:.3f}",  "p-value":  "{:.4f}",
                    "CI lower": "{:.4f}",  "CI upper": "{:.4f}",
                }, na_rep="—"),
            use_container_width=True, height=450,
        )

        # Coefficient bar chart
        fig_coef = go.Figure(go.Bar(
            x=coef_display["Coef (β)"],
            y=coef_display["feature"],
            orientation="h",
            marker_color=coef_display["Coef (β)"].apply(
                lambda v: "#1a7a3a" if v > 0 else "#a94442"
            ),
            hovertemplate="%{y}: β=%{x:.6f}<extra></extra>",
        ))
        fig_coef.add_vline(x=0, line_width=1, line_color="black")
        fig_coef.update_layout(
            height=400, title="Coefficients (VIF-filtered model, sorted by |t-stat|)",
            xaxis_title="Coefficient β", margin=dict(l=220),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")

    # ── Actual vs predicted ────────────────────────────────────────────────────
    st.subheader("Actual vs Predicted Excess Return")
    avp = get_actual_vs_predicted(ft_df, feats_sel)
    if not avp.empty:
        left_col, right_col = st.columns(2)
        with left_col:
            fig_avp = px.scatter(
                avp, x="predicted", y="future_12m_excess_return",
                opacity=0.3,
                labels={"predicted": "Predicted excess return",
                        "future_12m_excess_return": "Actual excess return"},
                title="Actual vs Predicted (in-sample)",
            )
            # Perfect prediction line
            min_v = min(avp["predicted"].min(), avp["future_12m_excess_return"].min())
            max_v = max(avp["predicted"].max(), avp["future_12m_excess_return"].max())
            fig_avp.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                mode="lines", name="Perfect prediction",
                line=dict(color="red", dash="dash", width=1)))
            fig_avp.update_layout(height=360)
            st.plotly_chart(fig_avp, use_container_width=True)

        with right_col:
            fig_res = px.histogram(
                avp, x="residual", nbins=40,
                labels={"residual": "Residual"},
                color_discrete_sequence=["#1f77b4"],
                title="Residual distribution",
            )
            fig_res.add_vline(x=0, line_color="red", line_width=2)
            fig_res.update_layout(height=360)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Not enough complete observations for actual vs predicted plot.")

    st.markdown("---")

    # ── Rolling R² ────────────────────────────────────────────────────────────
    st.subheader("Rolling Model R² (36-month windows)")
    st.markdown(
        "Does the model maintain its explanatory power consistently, "
        "or is it concentrated in specific market regimes?"
    )
    roll_r2 = compute_rolling_r_squared(ft_df, feats_sel)
    if not roll_r2.empty:
        fig_rr2 = go.Figure()
        fig_rr2.add_trace(go.Scatter(
            x=roll_r2["end_date"], y=roll_r2["r_squared"],
            mode="lines", name="Rolling R²", line=dict(color="#1f77b4", width=2),
        ))
        fig_rr2.add_trace(go.Scatter(
            x=roll_r2["end_date"], y=roll_r2["adj_r_squared"],
            mode="lines", name="Rolling Adj-R²",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig_rr2.add_hline(y=0, line_color="red", line_width=1, line_dash="dot")
        fig_rr2.update_layout(
            height=320, xaxis_title="Window end date", yaxis_title="R²",
            title="Rolling R² — multiple regression (VIF-filtered features)",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_rr2, use_container_width=True)

        mean_r2 = roll_r2["r_squared"].mean()
        min_r2  = roll_r2["r_squared"].min()
        pct_pos = (roll_r2["r_squared"] > 0).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean rolling R²",         f"{mean_r2:.4f}")
        c2.metric("Worst rolling R²",         f"{min_r2:.4f}")
        c3.metric("Windows with R² > 0",     f"{pct_pos*100:.0f}%")

    # ── Guide ──────────────────────────────────────────────────────────────────
    with st.expander("How to read the multiple regression results"):
        st.markdown("""
**Why adjusted R² matters more than R²**
Adding more features always increases R², even if those features are random noise.
Adjusted R² penalises for the number of features added. If adjusted R² falls
when you add a feature, that feature is not helping.

**How to interpret VIF**
If two features have VIF > 10, their coefficients are unreliable — small changes
in the data can flip their signs. The VIF filter removes the worst offender
iteratively until the model is stable.

**Comparing simple vs multiple regression coefficients**
A feature that had a large positive coefficient in simple regression may have
a small or negative coefficient in multiple regression. This means it was
"taking credit" for another feature it was correlated with.
This is not a contradiction — it is the regression doing its job correctly.

**Sign changes relative to simple regression**
If a feature flips sign from simple to multiple regression, it means its apparent
relationship with returns was actually driven by its correlation with another feature.
The multiple regression result is the more honest estimate.

**Out-of-sample R² (test set)**
The 30% of dates held back from training. If test R² is much lower than
train R², the model is overfitting. If test R² is negative, the model
predicts worse than simply using the historical mean return.

**Rolling R² interpretation**
A stable model should maintain positive R² across different time periods.
If R² drops to near zero or negative in recent windows, the feature relationships
may have changed — a sign that the model needs to be retrained or that
the signals have weakened.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not logit_result:
        st.error("Logistic regression results unavailable.")
        st.stop()

    m_tr  = logit_result["metrics_train"]
    m_te  = logit_result["metrics_test"]
    cm    = logit_result["confusion_matrix"]
    roc   = logit_result["roc_curve_data"]
    cal   = logit_result["calibration_data"]
    coef  = logit_result["coefficient_table"]
    pred  = logit_result["predictions"]
    rauc  = logit_result["rolling_auc"]

    st.subheader("Logistic Regression — Probability of Outperforming the Benchmark")
    st.markdown("""
This model estimates **P(stock beats benchmark over next 12 months)** for each stock.
The probability output is the primary input for the stock-ranking score in Step 14.

Two implementations are combined:
- **statsmodels Logit** → p-values, odds ratios, AIC/BIC (research transparency)
- **scikit-learn** → calibrated probabilities, ROC AUC, confusion matrix
    """)

    # ── Top metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Train AUC",     f"{m_tr.get('roc_auc', 0):.4f}")
    c2.metric("Test AUC",      f"{m_te.get('roc_auc', 0):.4f}" if m_te.get('roc_auc') else "—")
    c3.metric("Test Accuracy", f"{m_te['accuracy']:.4f}")
    c4.metric("Test F1",       f"{m_te['f1']:.4f}")
    c5.metric("Pseudo-R²",     f"{logit_result.get('pseudo_r2', 0):.4f}" if logit_result.get('pseudo_r2') else "—")
    c6.metric("Features used", len(logit_result["features_used"]))

    st.markdown("---")

    # ── Train vs test metrics table ────────────────────────────────────────────
    st.subheader("Train vs Test Performance")
    metric_rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score"]:
        metric_rows.append({
            "Metric":  metric.replace("_", " ").title(),
            "Train":   m_tr.get(metric),
            "Test":    m_te.get(metric),
            "Gap":     round((m_tr.get(metric) or 0) - (m_te.get(metric) or 0), 4),
        })
    metrics_df = pd.DataFrame(metric_rows)

    def _colour_gap(val):
        if pd.isna(val): return ""
        if abs(val) > 0.05: return "background-color:#f8d7da"
        return "background-color:#d4edda"

    st.dataframe(
        metrics_df.style
            .map(_colour_gap, subset=["Gap"])
            .format({"Train": "{:.4f}", "Test": "{:.4f}", "Gap": "{:+.4f}"}, na_rep="—"),
        use_container_width=True, height=310,
    )
    st.caption(
        "A large Train–Test gap indicates overfitting. "
        "Gap > 0.05 on AUC is highlighted in red."
    )
    st.markdown("---")

    # ── ROC curve + confusion matrix ──────────────────────────────────────────
    st.subheader("ROC Curve & Confusion Matrix (Test Set)")
    left, right = st.columns(2)

    with left:
        if not roc.empty:
            auc_val = m_te.get("roc_auc", 0)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                mode="lines", name=f"ROC (AUC={auc_val:.4f})",
                line=dict(color="#1f77b4", width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random (AUC=0.50)",
                line=dict(color="red", dash="dash", width=1),
            ))
            fig_roc.update_layout(
                height=380, title="ROC Curve — Test Set",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.1),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    with right:
        if cm is not None and cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            cm_df = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=["Actual Winner", "Actual Loser"],
                columns=["Predicted Winner", "Predicted Loser"],
            )
            fig_cm = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Blues",
                title="Confusion Matrix — Test Set",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── Calibration curve ─────────────────────────────────────────────────────
    st.subheader("Probability Calibration")
    st.markdown(
        "A well-calibrated model should fall on the diagonal: "
        "when it predicts 60% probability, ~60% of stocks should actually win."
    )
    if not cal.empty:
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=cal["mean_predicted_prob"], y=cal["actual_win_rate"],
            mode="lines+markers", name="Model calibration",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Perfect calibration",
            line=dict(color="red", dash="dash", width=1),
        ))
        fig_cal.update_layout(
            height=350, title="Reliability Diagram (Calibration Curve)",
            xaxis_title="Mean predicted probability",
            yaxis_title="Actual win rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        if "mean_calibration_error" in cal.attrs:
            st.metric(
                "Mean Calibration Error",
                f"{cal.attrs['mean_calibration_error']:.4f}",
                help="Average absolute deviation between predicted probability and actual win rate. Lower is better.",
            )

    st.markdown("---")

    # ── Odds ratio chart ───────────────────────────────────────────────────────
    st.subheader("Coefficient Table — Log-Odds & Odds Ratios")
    st.markdown("""
**Odds Ratio (OR)** = exp(coefficient).
OR > 1 → feature increases the odds of beating the benchmark.
OR < 1 → feature decreases the odds.
OR = 1.5 → a 1-unit increase multiplies the odds of winning by 1.5× (50% higher).
    """)

    if coef is not None and not coef.empty:
        coef_disp = coef.rename(columns={
            "coefficient": "Log-odds (β)",
            "std_error":   "Std Err",
            "z_stat":      "z-stat",
            "p_value":     "p-value",
            "odds_ratio":  "Odds Ratio",
            "or_ci_lower": "OR CI lower",
            "or_ci_upper": "OR CI upper",
            "significant": "Sig.",
        }).sort_values("z-stat", key=abs, ascending=False)

        def _colour_or(val):
            if pd.isna(val): return ""
            if val > 1.2: return "background-color:#d4edda"
            if val < 0.8: return "background-color:#f8d7da"
            return ""

        st.dataframe(
            coef_disp.style
                .map(_colour_or, subset=["Odds Ratio"])
                .format({
                    "Log-odds (β)": "{:+.4f}",
                    "Std Err":      "{:.4f}",
                    "z-stat":       "{:.3f}",
                    "p-value":      "{:.4f}",
                    "Odds Ratio":   "{:.4f}",
                    "OR CI lower":  "{:.4f}",
                    "OR CI upper":  "{:.4f}",
                }, na_rep="—"),
            use_container_width=True, height=450,
        )

        # Odds ratio forest plot
        plot_coef = coef_disp.dropna(subset=["Odds Ratio"]).copy()
        plot_coef = plot_coef[plot_coef["feature"] != "intercept"].sort_values("Odds Ratio")

        fig_or = go.Figure()
        fig_or.add_trace(go.Scatter(
            x=plot_coef["Odds Ratio"],
            y=plot_coef["feature"],
            mode="markers",
            marker=dict(
                size=10,
                color=plot_coef["Odds Ratio"].apply(
                    lambda v: "#1a7a3a" if v > 1 else "#a94442"
                ),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(plot_coef["OR CI upper"] - plot_coef["Odds Ratio"]).clip(0).tolist(),
                arrayminus=(plot_coef["Odds Ratio"] - plot_coef["OR CI lower"]).clip(0).tolist(),
                thickness=2,
            ),
            hovertemplate="%{y}: OR=%{x:.4f}<extra></extra>",
        ))
        fig_or.add_vline(x=1, line_color="red", line_width=1, line_dash="dash")
        fig_or.update_layout(
            height=450, title="Odds Ratio Forest Plot (with 95% CI)",
            xaxis_title="Odds Ratio (OR = 1 = no effect)",
            margin=dict(l=220),
        )
        st.plotly_chart(fig_or, use_container_width=True)

    st.markdown("---")

    # ── Rolling AUC ────────────────────────────────────────────────────────────
    st.subheader("Rolling AUC Stability (36-month windows, honest out-of-sample)")
    st.markdown(
        "Each window uses a 70/30 internal train/test split — AUC shown is "
        "genuinely out-of-sample within each window."
    )
    if not rauc.empty:
        fig_ra = go.Figure()
        fig_ra.add_trace(go.Scatter(
            x=rauc["end_date"], y=rauc["auc"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            name="Rolling AUC",
        ))
        fig_ra.add_hline(y=0.5, line_color="red", line_dash="dash",
                         line_width=1, annotation_text="Random = 0.50")
        fig_ra.add_hline(y=0.55, line_color="green", line_dash="dot",
                         line_width=1, annotation_text="Useful = 0.55")
        fig_ra.update_layout(
            height=320, title="Rolling out-of-sample AUC",
            xaxis_title="Window end date", yaxis_title="ROC AUC",
            yaxis=dict(range=[0.35, 0.85]),
        )
        st.plotly_chart(fig_ra, use_container_width=True)

        ca, cb, cc = st.columns(3)
        ca.metric("Mean rolling AUC",     f"{rauc['auc'].mean():.4f}")
        cb.metric("Min rolling AUC",      f"{rauc['auc'].min():.4f}")
        cc.metric("Windows AUC > 0.55",   f"{(rauc['auc']>0.55).mean()*100:.0f}%")

    # ── Probability distribution ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Predicted Probability Distribution")
    if not pred.empty:
        left2, right2 = st.columns(2)
        with left2:
            fig_pd = px.histogram(
                pred, x="probability", color="actual_winner",
                color_discrete_map={0: "#a94442", 1: "#1a7a3a"},
                nbins=25, barmode="overlay", opacity=0.6,
                labels={"probability": "Predicted probability of winning",
                        "actual_winner": "Actual winner"},
                title="Probability distribution: winners vs losers",
            )
            fig_pd.update_layout(height=320)
            st.plotly_chart(fig_pd, use_container_width=True)

        with right2:
            # Win rate by probability quintile
            pred2 = pred.copy()
            pred2["prob_q"] = pd.qcut(pred2["probability"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
            q_rates = pred2.groupby("prob_q")["actual_winner"].mean().reset_index()
            q_rates.columns = ["Quintile", "Actual win rate"]
            fig_qr = px.bar(
                q_rates, x="Quintile", y="Actual win rate",
                color="Actual win rate",
                color_continuous_scale=["#a94442","#ffffff","#1a7a3a"],
                color_continuous_midpoint=0.5,
                title="Actual win rate by predicted probability quintile",
                text_auto=".2%",
            )
            fig_qr.add_hline(y=0.5, line_dash="dash", line_color="red", line_width=1)
            fig_qr.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_qr, use_container_width=True)

    with st.expander("How to read the logistic regression results"):
        st.markdown("""
**ROC AUC**
The area under the ROC curve. AUC of 0.50 = random guessing. In financial return
prediction, AUC of 0.55–0.62 is realistic and meaningful.

**Calibration curve**
Checks whether predicted probabilities are trustworthy.
A model predicting 70% should produce winners 70% of the time.
Deviation from the diagonal = miscalibration.

**Odds Ratio**
For a feature with OR = 1.5: a 1-unit increase in that feature multiplies
the *odds* of beating the benchmark by 1.5× (50% higher odds).
If the 95% CI for OR includes 1.0, the effect is not statistically significant.

**Confusion matrix**
At the default 0.5 classification threshold:
- **TP** (top-left): predicted winner, actually won ✓
- **FN** (top-right): predicted loser, actually won ✗
- **FP** (bottom-left): predicted winner, actually lost ✗
- **TN** (bottom-right): predicted loser, actually lost ✓

**Rolling AUC**
Each window uses an internal out-of-sample test. If AUC consistently stays
above 0.55, the signal is real. Dips below 0.50 in specific windows suggest
the model failed in those market regimes.

**Probability quintile chart**
If the model is useful, the top quintile (Q5) should have the highest actual
win rate and the bottom quintile (Q1) the lowest. Monotone pattern = good signal.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RIDGE & LASSO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if not reg_models:
        st.error("Regularised model results unavailable.")
        st.stop()

    ridge = reg_models["ridge"]
    lasso = reg_models["lasso"]
    comp  = reg_models["comparison_table"]

    st.subheader("Ridge & Lasso — Regularised Regression")
    st.markdown("""
Both models address the **overfitting problem** identified in Steps 10–11, where
OLS multiple regression achieved R² = 0.15 in-sample but **−0.45 out-of-sample**.

**Ridge** shrinks all coefficients toward zero — keeps all features, reduces their magnitude.
**Lasso** sets weak features to exactly zero — automatic feature selection.

The regularisation strength α is chosen by **time-series cross-validation** (no data leakage).
    """)

    # ── Model comparison metrics ───────────────────────────────────────────────
    st.subheader("Performance Comparison: OLS vs Ridge vs Lasso")

    ols_test_r2  = multi_result.get("filtered_model", {}).get("r_squared_test", None)
    ridge_tr = ridge.get("metrics_train", {}); ridge_te = ridge.get("metrics_test", {})
    lasso_tr = lasso.get("metrics_train", {}); lasso_te = lasso.get("metrics_test", {})

    comp_rows = [
        {"Model": "OLS (VIF-filtered)",
         "Features": len(multi_result.get("features_filtered", [])),
         "Alpha": "—",
         "Train R²": multi_result.get("filtered_model", {}).get("r_squared"),
         "Test R²":  ols_test_r2,
         "Train Hit Rate": "—", "Test Hit Rate": "—"},
        {"Model": "Ridge",
         "Features": len(ridge.get("features_used", [])),
         "Alpha": f"{ridge.get('best_alpha', 0):.4f}",
         "Train R²": ridge_tr.get("r_squared"),
         "Test R²":  ridge_te.get("r_squared"),
         "Train Hit Rate": ridge_tr.get("hit_rate"),
         "Test Hit Rate":  ridge_te.get("hit_rate")},
        {"Model": "Lasso",
         "Features": f"{lasso.get('n_selected', 0)} / {len(lasso.get('features_used', []))}",
         "Alpha": f"{lasso.get('best_alpha', 0):.4f}",
         "Train R²": lasso_tr.get("r_squared"),
         "Test R²":  lasso_te.get("r_squared"),
         "Train Hit Rate": lasso_tr.get("hit_rate"),
         "Test Hit Rate":  lasso_te.get("hit_rate")},
    ]
    comp_df = pd.DataFrame(comp_rows)

    def _colour_r2_cell(val):
        if pd.isna(val) or val == "—": return ""
        try:
            v = float(val)
            if v > 0.05:  return "background-color:#1a7a3a;color:white"
            if v > 0.00:  return "background-color:#d4edda"
            return "background-color:#f8d7da"
        except (ValueError, TypeError):
            return ""

    st.dataframe(
        comp_df.style.map(_colour_r2_cell, subset=["Train R²","Test R²"]),
        use_container_width=True, height=160,
    )

    st.markdown("---")

    # ── Ridge & Lasso coefficient charts side by side ──────────────────────────
    st.subheader("Coefficient Comparison: Ridge vs Lasso")
    st.markdown(
        "Both use **standardised features** (mean=0, std=1), so coefficients are directly "
        "comparable across features. A Lasso coefficient of 0 means the feature was eliminated."
    )

    if not comp.empty:
        left, right = st.columns(2)

        with left:
            ridge_plot = comp.sort_values("ridge_coef")
            fig_r = go.Figure(go.Bar(
                x=ridge_plot["ridge_coef"],
                y=ridge_plot["feature"],
                orientation="h",
                marker_color=ridge_plot["ridge_coef"].apply(
                    lambda v: "#1a7a3a" if v > 0 else "#a94442"
                ),
                hovertemplate="%{y}: β=%{x:.4f}<extra></extra>",
            ))
            fig_r.add_vline(x=0, line_color="black", line_width=1)
            fig_r.update_layout(
                height=500, title=f"Ridge  (α={ridge.get('best_alpha', 0):.4f})",
                xaxis_title="Coefficient (per 1 SD)", margin=dict(l=220),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with right:
            lasso_plot = comp.sort_values("lasso_coef")
            colours = lasso_plot.apply(
                lambda r: "#aaaaaa" if r.lasso_coef == 0
                          else ("#1a7a3a" if r.lasso_coef > 0 else "#a94442"),
                axis=1,
            )
            fig_l = go.Figure(go.Bar(
                x=lasso_plot["lasso_coef"],
                y=lasso_plot["feature"],
                orientation="h",
                marker_color=colours,
                hovertemplate="%{y}: β=%{x:.4f}<extra></extra>",
            ))
            fig_l.add_vline(x=0, line_color="black", line_width=1)
            fig_l.update_layout(
                height=500,
                title=f"Lasso  (α={lasso.get('best_alpha', 0):.4f}, {lasso.get('n_selected', 0)} selected)",
                xaxis_title="Coefficient (grey = zeroed out)", margin=dict(l=220),
            )
            st.plotly_chart(fig_l, use_container_width=True)

    st.markdown("---")

    # ── Lasso selected features table ─────────────────────────────────────────
    n_sel = lasso.get("n_selected", 0)
    st.subheader(f"Lasso Selected Features — {n_sel} survive regularisation")

    if n_sel == 0:
        st.warning(
            f"**Lasso eliminated all features** at the cross-validated α = {lasso.get('best_alpha', 0):.3f}. "
            "This is an honest finding: with only 30 stocks, the noise-to-signal ratio is too high "
            "for Lasso to confidently select any individual feature. "
            "The regularisation path below shows which features enter first as α decreases — "
            "these are the strongest candidates for the scoring model.",
            icon="⚠️",
        )
    else:
        st.markdown(
            "These are the features the Lasso considers genuinely predictive "
            "after eliminating redundant or noise signals. "
            "They should form the core of the scoring model (Step 14)."
        )

    lasso_coef = lasso.get("coefficient_table", pd.DataFrame())
    if not lasso_coef.empty and n_sel > 0:
        selected_df = lasso_coef[lasso_coef["selected"]].sort_values(
            "coefficient", key=abs, ascending=False
        )
        st.dataframe(
            selected_df[["feature", "coefficient", "selected"]].rename(
                columns={"coefficient": "Lasso Coef (std)", "selected": "Selected"}
            ).style.map(
                lambda v: "background-color:#d4edda" if v else "background-color:#f8d7da",
                subset=["Selected"],
            ).format({"Lasso Coef (std)": "{:+.6f}"}),
            use_container_width=True, height=min(60 + len(selected_df) * 35, 500),
        )

    st.markdown("---")

    # ── Lasso regularisation path ──────────────────────────────────────────────
    st.subheader("Lasso Regularisation Path")
    st.markdown(
        "Shows how each feature's coefficient changes as we increase regularisation "
        "strength (α). Features that survive to large α are the most robust signals."
    )
    if not lasso_path_df.empty:
        feat_cols_path = [c for c in lasso_path_df.columns if c != "alpha"]
        fig_path = go.Figure()
        for feat in feat_cols_path:
            line_width = 2.5 if feat in lasso.get("selected_features", []) else 1
            opacity    = 1.0 if feat in lasso.get("selected_features", []) else 0.3
            fig_path.add_trace(go.Scatter(
                x=np.log10(lasso_path_df["alpha"]),
                y=lasso_path_df[feat],
                mode="lines",
                name=feat,
                line=dict(width=line_width),
                opacity=opacity,
                hovertemplate=f"{feat}: %{{y:.4f}}<extra></extra>",
            ))
        fig_path.add_vline(
            x=math.log10(lasso.get("best_alpha", 0.01)),
            line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"Selected α={lasso.get('best_alpha', 0):.4f}",
        )
        fig_path.add_hline(y=0, line_color="black", line_width=0.5)
        fig_path.update_layout(
            height=480,
            title="Lasso path: log₁₀(α) vs coefficient  (bold = selected features)",
            xaxis_title="log₁₀(α) — increasing regularisation →",
            yaxis_title="Coefficient (standardised)",
            showlegend=True,
            legend=dict(orientation="v", x=1.02, xanchor="left"),
        )
        st.plotly_chart(fig_path, use_container_width=True)

    with st.expander("How to read Ridge & Lasso results"):
        st.markdown("""
**Why regularisation improves out-of-sample performance**
OLS finds the single best fit for the training data, which can include fitting
noise. Regularisation constrains the coefficients — forcing the model to use
only the most robust patterns and ignore weak noise signals.

**Choosing between Ridge and Lasso**
Use Ridge when all features are expected to contribute something.
Use Lasso when you want the model to identify the strongest subset of features.
In practice, Lasso-selected features are good candidates for the scoring model.

**The regularisation path**
As α increases from left to right, the model becomes simpler (more features
hit zero). Features that remain non-zero at large α are the most robustly
predictive. The red dashed line shows the α chosen by cross-validation.

**Comparing OLS vs Ridge vs Lasso test R²**
If Ridge/Lasso have better test R² than OLS, regularisation successfully
reduced overfitting. If test R² is still negative, even with regularisation
the model cannot generalise — likely a fundamental data limitation
(30 stocks with correlated returns is a very small sample).

**Hit rate**
Fraction of test observations where the model correctly predicted the
direction (positive or negative excess return). A hit rate > 55% is
meaningful; below 50% means the model is worse than a coin flip.
        """)
