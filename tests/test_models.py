"""
tests/test_models.py

Tests for statistical models: mathematical correctness of metrics,
multiple-testing corrections, VIF positivity, and probability bounds.
"""

import math
import pytest
import numpy as np
import pandas as pd


# ── Pearson correlation ────────────────────────────────────────────────────────

def test_pearson_r_range(tiny_ft):
    """Pearson correlation values must be in [-1, 1]."""
    from src.models.correlation import run_correlation_analysis
    import warnings

    feat_cols = ["twelve_month_momentum", "current_pe_ratio", "roe", "data_quality_score"]
    # Separate features from targets to avoid column name conflicts in the merge
    features_df = tiny_ft[["feature_date","ticker"] + feat_cols].copy()
    targets_df  = tiny_ft[["feature_date","ticker",
                            "future_12m_excess_return","winner"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_correlation_analysis(features_df, targets_df, feature_cols=feat_cols)

    if results and "summary" in results:
        rs = results["summary"]["pearson_r"].dropna()
        assert (rs.abs() <= 1.0).all(), "Pearson r outside [-1, 1]"


def test_bh_correction_monotone():
    """BH-adjusted p-values must be non-decreasing when input p-values are sorted."""
    from src.models.correlation import _bh_fdr
    p_vals = np.array([0.001, 0.01, 0.03, 0.05, 0.15, 0.40, 0.80])
    adj    = _bh_fdr(p_vals)
    assert (np.diff(adj) >= -1e-12).all(), "BH adjusted p-values not monotone"


def test_bh_correction_all_le_one():
    """All BH-adjusted p-values must be ≤ 1.0."""
    from src.models.correlation import _bh_fdr
    p_vals = np.array([0.001, 0.01, 0.05, 0.10, 0.50])
    adj    = _bh_fdr(p_vals)
    assert (adj <= 1.0).all()


def test_bonferroni_large_p_capped_at_one():
    """Bonferroni correction must cap at 1.0 for large p-values."""
    from src.models.correlation import _bonferroni
    p_vals = np.array([0.001, 0.01, 0.50])
    adj    = _bonferroni(p_vals)
    assert (adj <= 1.0).all()
    assert adj[0] == pytest.approx(0.001 * 3)


def test_bonferroni_first_stays_significant():
    """A very small p-value should remain significant after Bonferroni correction."""
    from src.models.correlation import _bonferroni
    p_vals = np.array([0.0001, 0.05, 0.10])
    adj    = _bonferroni(p_vals)
    assert adj[0] < 0.05


# ── VIF ───────────────────────────────────────────────────────────────────────

def test_vif_all_positive(tiny_ft):
    """VIF values must be positive (never zero or NaN for valid data)."""
    from src.models.multiple_regression import compute_vif
    feat_cols = ["twelve_month_momentum", "current_pe_ratio", "roe", "volatility_12m"]
    available = [c for c in feat_cols if c in tiny_ft.columns]
    vif_df    = compute_vif(tiny_ft, available)
    vifs      = vif_df["vif"].dropna()
    assert (vifs > 0).all(), "VIF has non-positive values"


def test_perfectly_collinear_features_high_vif():
    """Two perfectly correlated features must produce very high VIF."""
    from src.models.multiple_regression import compute_vif
    n  = 100
    x1 = np.random.randn(n)
    df = pd.DataFrame({"A": x1, "B": x1 * 2 + 0.001 * np.random.randn(n)})
    vif_df = compute_vif(df, ["A", "B"])
    assert vif_df["vif"].max() > 50, "Highly correlated features should produce VIF >> 1"


# ── Simple OLS regression ─────────────────────────────────────────────────────

def test_ols_r_squared_in_range(tiny_ft):
    """OLS R² must be in [0, 1] for valid data."""
    from src.models.simple_regression import run_simple_regression
    result = run_simple_regression(tiny_ft, "twelve_month_momentum")
    assert result["r_squared"] is not None
    assert 0.0 <= result["r_squared"] <= 1.0


def test_ols_coefficient_finite(tiny_ft):
    """OLS coefficient must be a finite float."""
    from src.models.simple_regression import run_simple_regression
    result = run_simple_regression(tiny_ft, "twelve_month_momentum")
    assert result["coefficient"] is not None
    assert math.isfinite(result["coefficient"])


def test_ols_residuals_mean_zero(tiny_ft):
    """OLS residuals must sum to (approximately) zero."""
    from src.models.simple_regression import get_regression_plot_data
    plot_df = get_regression_plot_data(tiny_ft, "twelve_month_momentum")
    if not plot_df.empty:
        residual_mean = plot_df["residual"].mean()
        assert abs(residual_mean) < 1e-9, f"Residual mean {residual_mean:.2e} is not zero"


# ── Time-based train/test split ───────────────────────────────────────────────

def test_time_split_no_overlap(tiny_ft):
    """Train and test sets must have no overlapping dates."""
    from src.models.simple_regression import time_split
    train, test = time_split(tiny_ft, train_frac=0.70)
    train_dates = set(train["feature_date"].unique())
    test_dates  = set(test["feature_date"].unique())
    assert train_dates.isdisjoint(test_dates), "Train and test dates overlap"


def test_time_split_order(tiny_ft):
    """All training dates must be earlier than all test dates."""
    from src.models.simple_regression import time_split
    train, test = time_split(tiny_ft, train_frac=0.70)
    assert train["feature_date"].max() < test["feature_date"].min()


# ── Logistic regression probability bounds ────────────────────────────────────

def test_logistic_probabilities_in_range():
    """Logistic regression must output probabilities strictly in [0, 1]."""
    from src.models.logistic_regression import run_logistic_regression
    import warnings
    rng = np.random.default_rng(42)
    n   = 200
    df  = pd.DataFrame({
        "feature_date":             pd.date_range("2019-01-31", periods=n, freq="ME").strftime("%Y-%m-%d"),
        "ticker":                   np.tile(["A","B","C","D","E"], n // 5),
        "twelve_month_momentum":    rng.normal(0, 0.2, n),
        "current_pe_ratio":         rng.uniform(10, 50, n),
        "roe":                      rng.uniform(0.05, 0.35, n),
        "volatility_12m":           rng.uniform(0.1, 0.5, n),
        "data_quality_score":       rng.uniform(60, 100, n),
        "winner":                   rng.integers(0, 2, n).astype(float),
        "future_12m_excess_return": rng.normal(0, 0.15, n),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_logistic_regression(df, feature_cols=["twelve_month_momentum",
                                                           "current_pe_ratio", "roe"])
    if result and "predictions" in result and not result["predictions"].empty:
        probs = result["predictions"]["probability"]
        assert (probs >= 0).all() and (probs <= 1).all(), "Logistic probabilities out of [0,1]"
