"""
tests/test_target_creation.py

Tests for target variable creation: date arithmetic, mathematical identities,
and look-ahead bias prevention.
"""

import pytest
import numpy as np
import pandas as pd


def test_get_target_date_plus_12_months():
    """Adding 12 months to 2022-01-31 should give 2023-01-31."""
    from src.features.target_creation import get_target_date
    assert get_target_date("2022-01-31", 12) == "2023-01-31"


def test_get_target_date_plus_6_months():
    """Adding 6 months to 2022-06-30 should give 2022-12-31."""
    from src.features.target_creation import get_target_date
    assert get_target_date("2022-06-30", 6) == "2022-12-31"


def test_is_target_available_old_date_true():
    """A date from 13 months ago should have its 12-month target available."""
    import pandas as pd
    from src.features.target_creation import is_target_available
    thirteen_months_ago = (pd.Timestamp.today() - pd.DateOffset(months=13)).strftime("%Y-%m-%d")
    assert is_target_available(thirteen_months_ago, 12)


def test_is_target_available_future_date_false():
    """A date 6 months from now cannot have a 12-month target yet."""
    import pandas as pd
    from src.features.target_creation import is_target_available
    six_months_ahead = (pd.Timestamp.today() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    assert not is_target_available(six_months_ahead, 12)


def test_winner_matches_excess_return_sign(tiny_prices, tiny_benchmark):
    """winner=1 if and only if future_12m_excess_return > 0."""
    from src.features.target_creation import compute_all_targets
    targets = compute_all_targets(tiny_prices, tiny_benchmark, only_available=True)
    labelled = targets[targets["winner"].notna()].copy()
    if labelled.empty:
        pytest.skip("No labelled targets available with this dataset")
    assert ((labelled["winner"] == 1) == (labelled["future_12m_excess_return"] > 0)).all(), (
        "winner flag does not match sign of excess return"
    )


def test_excess_return_identity(tiny_prices, tiny_benchmark):
    """excess_return must equal stock_return - benchmark_return exactly."""
    from src.features.target_creation import compute_all_targets
    targets = compute_all_targets(tiny_prices, tiny_benchmark, only_available=True)
    labelled = targets.dropna(subset=["future_12m_excess_return",
                                      "future_12m_stock_return",
                                      "future_12m_benchmark_return"])
    if labelled.empty:
        pytest.skip("No labelled targets available")
    diff = (labelled["future_12m_excess_return"] -
            (labelled["future_12m_stock_return"] - labelled["future_12m_benchmark_return"])).abs()
    assert diff.max() < 1e-9, f"Excess return identity violated (max error={diff.max()})"


def test_winner_values_are_0_or_1(tiny_prices, tiny_benchmark):
    """winner column must contain only 0, 1, or NaN."""
    from src.features.target_creation import compute_all_targets
    targets = compute_all_targets(tiny_prices, tiny_benchmark, only_available=True)
    valid   = targets["winner"].dropna()
    assert set(valid.unique()).issubset({0, 1, 0.0, 1.0}), "winner has values other than 0/1"


def test_lookup_price_exact_match():
    """_lookup_price should return the exact price when the date is present."""
    from src.features.target_creation import _lookup_price
    idx   = pd.Series([100.0, 105.0, 110.0], index=["2022-01-31", "2022-02-28", "2022-03-31"])
    price = _lookup_price(idx, "2022-02-28")
    assert price == 105.0


def test_lookup_price_tolerance():
    """_lookup_price should find nearest price within tolerance days."""
    from src.features.target_creation import _lookup_price
    idx   = pd.Series([100.0, 110.0], index=["2022-01-31", "2022-03-31"])
    price = _lookup_price(idx, "2022-03-28", tolerance_days=5)
    assert price == 110.0


def test_lookup_price_outside_tolerance_returns_none():
    """_lookup_price should return None when no date is within tolerance."""
    from src.features.target_creation import _lookup_price
    idx   = pd.Series([100.0], index=["2022-01-31"])
    price = _lookup_price(idx, "2022-06-30", tolerance_days=15)
    assert price is None
