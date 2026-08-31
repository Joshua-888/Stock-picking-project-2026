"""
tests/test_database.py

Tests for the database layer: save/load round-trips, uniqueness constraints,
migration application, and lag-safe filtering.
"""

import pytest
import pandas as pd


# ── Schema and migrations ──────────────────────────────────────────────────────

def test_all_tables_created(test_db):
    """All 15 user tables should exist after initialize_db + apply_migrations."""
    from src.database.db import check_db_health
    health = check_db_health(test_db)
    user_tables = [t for t in health["tables"]
                   if t not in ("sqlite_sequence", "schema_migrations")]
    assert len(user_tables) == 15, f"Expected 15 user tables, got {len(user_tables)}: {user_tables}"


def test_migrations_idempotent(test_db):
    """Calling apply_migrations twice should not raise or duplicate rows."""
    from src.database.migrations import apply_migrations
    n1 = apply_migrations()
    n2 = apply_migrations()
    assert n2 == 0, "Second call should apply 0 new migrations (all already done)"


def test_all_migrations_applied(test_db):
    """Every migration in the registry should be marked applied."""
    from src.database.migrations import get_migration_status
    status = get_migration_status()
    for m in status:
        assert m["applied"], f"Migration v{m['version']} not applied: {m['description']}"


# ── Prices round-trip ──────────────────────────────────────────────────────────

def test_prices_clean_round_trip(test_db, tiny_prices):
    """Saving and loading prices_clean returns the same rows."""
    from src.database.db import save_prices_clean
    from src.database.queries import load_prices_clean
    save_prices_clean(tiny_prices)
    loaded = load_prices_clean()
    assert len(loaded) == len(tiny_prices)
    assert set(loaded["ticker"].unique()) == set(tiny_prices["ticker"].unique())


def test_prices_insert_or_ignore(test_db, tiny_prices):
    """Inserting the same prices twice should not duplicate rows."""
    from src.database.db import save_prices_clean
    from src.database.queries import load_prices_clean
    save_prices_clean(tiny_prices)
    save_prices_clean(tiny_prices)          # second insert
    loaded = load_prices_clean()
    assert len(loaded) == len(tiny_prices), "Duplicate rows inserted"


def test_benchmark_round_trip(test_db, tiny_benchmark):
    """Benchmark prices save and load correctly."""
    from src.database.db import save_benchmark_prices
    from src.database.queries import load_benchmark_prices
    save_benchmark_prices(tiny_benchmark)
    loaded = load_benchmark_prices()
    assert len(loaded) == len(tiny_benchmark)
    assert (loaded["adjusted_close"] > 0).all()


# ── Fundamentals lag-safe filter ───────────────────────────────────────────────

def test_fundamentals_lag_safe_filter(test_db, tiny_fundamentals, tiny_stocks):
    """
    Loading fundamentals with lag_safe=True and an early as_of_date
    must return fewer rows than loading with a late as_of_date.
    """
    from src.database.db import save_stocks, save_fundamentals_clean
    from src.database.queries import load_fundamentals_clean

    save_stocks(tiny_stocks)
    save_fundamentals_clean(tiny_fundamentals)

    early = load_fundamentals_clean(as_of_date="2017-01-01", lag_safe=True)
    late  = load_fundamentals_clean(as_of_date="2023-12-31", lag_safe=True)

    assert len(early) < len(late), (
        "Lag-safe filter with early date should return fewer rows than late date"
    )


def test_fundamentals_lag_safe_no_future(test_db, tiny_fundamentals, tiny_stocks):
    """No row with report_date > as_of_date should appear in lag-safe results."""
    from src.database.db import save_stocks, save_fundamentals_clean
    from src.database.queries import load_fundamentals_clean

    save_stocks(tiny_stocks)
    save_fundamentals_clean(tiny_fundamentals)

    as_of = "2019-06-30"
    loaded = load_fundamentals_clean(as_of_date=as_of, lag_safe=True)
    if not loaded.empty:
        assert (loaded["report_date"] <= as_of).all(), (
            "Lag-safe filter allowed future report_date"
        )


# ── Prediction snapshots ───────────────────────────────────────────────────────

def test_snapshot_insert_or_ignore(test_db, tiny_stocks):
    """
    Saving a snapshot twice for the same (prediction_date, ticker)
    must not raise and must not create duplicate rows.
    """
    import uuid
    from src.database.db import save_stocks, save_prediction_snapshots
    from src.database.queries import load_prediction_snapshots

    save_stocks(tiny_stocks)

    snap = pd.DataFrame([{
        "snapshot_id":                   str(uuid.uuid4()),
        "prediction_date":               "2022-01-31",
        "ticker":                        "A",
        "company_name":                  "Alpha Corp",
        "sector":                        "Technology",
        "industry":                      "Software",
        "model_version":                 "test_v1",
        "data_provider":                 "test",
        "data_update_timestamp":         "2022-01-31 00:00:00",
        "features_json":                 "{}",
        "features_std_json":             "{}",
        "predicted_12m_return":          0.10,
        "predicted_12m_excess_return":   0.05,
        "probability_of_outperformance": 0.65,
        "model_agreement_score":         70.0,
        "risk_score":                    60.0,
        "valuation_score":               55.0,
        "momentum_score":                70.0,
        "fundamental_score":             65.0,
        "data_quality_score":            90.0,
        "final_score":                   68.5,
        "candidate_classification":      "Strong candidate",
        "keyes_agreement_flag":          1,
        "warnings_json":                 "[]",
        "notes":                         None,
    }])

    save_prediction_snapshots(snap)
    save_prediction_snapshots(snap)    # duplicate

    loaded = load_prediction_snapshots(prediction_date="2022-01-31", ticker="A")
    assert len(loaded) == 1, "Duplicate snapshot was inserted"


# ── Health check ───────────────────────────────────────────────────────────────

def test_health_check_structure(test_db):
    """check_db_health returns the expected keys."""
    from src.database.db import check_db_health
    health = check_db_health(test_db)
    for key in ("status", "path", "size_mb", "tables", "row_counts"):
        assert key in health, f"Missing key '{key}' in health check result"
    assert health["status"] == "ok"
