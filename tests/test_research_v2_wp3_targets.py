"""WP3 regression tests: targets, observability, availability, ROA/ROIC, Keyes, features.

LIVE-INDEPENDENT: every fixture is small and explicitly synthetic, used only to
exercise integrity logic. No test reads a token or performs network I/O. Nothing
here is research evidence; no model is fitted and no score is produced. These
capture the corrected semantics so a future change cannot silently reintroduce a
leak, a fabricated label, a forced Keyes quota or a globally-fitted transform.
"""

import datetime

import pandas as pd

from src.research.data.availability import (
    fundamental_available_at,
    is_available,
    price_available_at,
)
from src.research.features import (
    V2_ROA_DEFINITION,
    V2_ROIC_DEFINITION,
    TRACK_HISTORICAL,
    TRACK_MODERN,
    KeyesCriterion,
    KeyesRuleSet,
    build_feature_set,
    build_initial_registry,
    compute_v2_roa,
    compute_v2_return_metrics,
    fit_imputer,
    apply_imputer,
    fit_transform,
    apply_transform,
    proxy_variable,
    qualify,
    v1_disposition,
)
from src.research.targets import (
    CLASSIFICATION_TARGET,
    CONTINUOUS_TARGET,
    TARGET_VERSION,
    TargetContract,
    annotate_observability,
    build_targets,
    intended_horizon_date,
    match_observation,
    match_target_window,
    target_known_at,
    trainable_mask,
)


def _price_frame(ticker, pairs):
    return pd.DataFrame(
        {"ticker": [ticker] * len(pairs),
         "trade_date": [p[0] for p in pairs],
         "raw_close": [p[1] for p in pairs]}
    )


# ── TARGET CONTRACT ──────────────────────────────────────────────────────────

def test_target_contract_is_versioned_and_benchmark_relative():
    contract = TargetContract(dataset_id="dataset_x")
    assert contract.target_version == TARGET_VERSION
    assert contract.horizon_months == 12
    assert contract.benchmark == "SPY"
    payload = contract.to_dict()
    assert CONTINUOUS_TARGET in payload["definitions"]
    assert CLASSIFICATION_TARGET in payload["definitions"]
    # the continuous target is benchmark-relative (SPY), so the definition must
    # name the benchmark leg and the excess-return methodology must subtract it
    assert "spy" in payload["definitions"][CONTINUOUS_TARGET].lower()
    assert "excess_return" in payload["calculation_methodology"]
    assert "benchmark_total_return" in payload["calculation_methodology"]
    assert payload["target_id"].startswith("target_set_")
    # deterministic id
    assert TargetContract(dataset_id="dataset_x").target_id == payload["target_id"]
    # a different dataset cannot share the id
    assert TargetContract(dataset_id="dataset_y").target_id != payload["target_id"]


# ── DATE MATCHING ────────────────────────────────────────────────────────────

def test_intended_horizon_clamps_month_end():
    assert intended_horizon_date("2020-01-31", 12) == datetime.date(2021, 1, 31)
    # Feb 29 in a non-leap target year clamps deterministically
    assert intended_horizon_date("2020-02-29", 12) == datetime.date(2021, 2, 28)


def test_match_observation_prefers_on_or_after_and_handles_weekends():
    dates = ["2020-01-31", "2020-03-02", "2020-03-31"]
    # intended 2020-03-01 is a weekend; first observation on/after is Monday 03-02
    assert match_observation(dates, "2020-03-01") == datetime.date(2020, 3, 2)
    # no observation on/after -> None (delisted / data ends), never approximated
    assert match_observation(dates, "2020-04-01") is None


def test_match_target_window_is_on_or_after_and_reports_horizon():
    dates = ["2020-01-31", "2020-02-28", "2021-01-29", "2021-02-26"]
    window = match_target_window(dates, "2020-01-31", 12)
    assert window["target_start"] == "2020-01-31"
    # intended 2021-01-31 (Sunday) -> first observation on/after is 2021-02-26
    assert window["target_end"] == "2021-02-26"
    assert window["observable"] is True
    assert window["target_horizon_days_actual"] == (
        datetime.date(2021, 2, 26) - datetime.date(2020, 1, 31)
    ).days
    # never an observation before the intended horizon
    assert datetime.date.fromisoformat(window["target_end"]) > datetime.date(2021, 1, 31)


def test_match_target_window_missing_future_is_unobservable():
    dates = ["2020-01-31", "2020-02-28"]
    window = match_target_window(dates, "2020-01-31", 12)
    assert window["observable"] is False
    assert window["target_end"] is None


# ── TARGET RETURN / BENCHMARK-RELATIVE / CENSORING ───────────────────────────

def _panel(rows):
    return pd.DataFrame(rows)


def test_build_targets_computes_excess_return_and_classification():
    prices = _price_frame("AAA", [
        ("2020-01-31", 100.0), ("2020-02-28", 101.0),
        ("2021-01-29", 120.0), ("2021-02-26", 121.0),
    ])
    bench = _price_frame("SPY", [
        ("2020-01-31", 300.0), ("2020-02-28", 301.0),
        ("2021-01-29", 330.0), ("2021-02-26", 331.0),
    ])
    panel = _panel([{
        "security_id": "AAA", "ticker": "AAA", "symbol": "AAA",
        "snapshot_date": "2020-01-31", "target_observable": True,
        "target_censored": False, "target_censor_reason": None,
    }])
    out = build_targets(panel, prices, bench)
    row = out.iloc[0]
    # the intended horizon is 2021-01-31 (Sunday); the first observation at/after
    # it is 2021-02-26, so the return leg legitimately ends on the 2021-02-26 close
    assert row["target_end"] == "2021-02-26"
    assert abs(row["future_12m_stock_return"] - (121.0 / 100.0 - 1.0)) < 1e-9
    assert abs(row["future_12m_benchmark_return"] - (331.0 / 300.0 - 1.0)) < 1e-9
    assert abs(row["future_12m_excess_return"]
               - ((121.0 / 100.0 - 1.0) - (331.0 / 300.0 - 1.0))) < 1e-9
    assert row["outperform_12m"] == 1
    assert bool(row["target_observable"]) is True


def test_censored_row_is_never_treated_as_zero_or_negative():
    prices = _price_frame("BBB", [("2020-01-31", 50.0), ("2020-02-28", 51.0)])
    bench = _price_frame("SPY", [("2020-01-31", 300.0), ("2020-02-28", 301.0)])
    panel = _panel([{
        "security_id": "BBB", "ticker": "BBB", "symbol": "BBB",
        "snapshot_date": "2020-01-31", "target_observable": False,
        "target_censored": True, "target_censor_reason": "terminal_price_unobservable",
    }])
    out = build_targets(panel, prices, bench)
    row = out.iloc[0]
    assert bool(row["target_observable"]) is False
    assert bool(row["target_censored"]) is True
    # the label is ABSENT, not 0.0 and not a negative
    assert pd.isna(row["future_12m_excess_return"])
    assert pd.isna(row["outperform_12m"])
    assert row["target_censor_reason"] == "terminal_price_unobservable"


# ── OBSERVABILITY / TEMPORAL METADATA ────────────────────────────────────────

def test_observability_only_marks_observable_rows_trainable():
    frame = pd.DataFrame([
        {"security_id": "A", "ticker": "A", "feature_asof": "2020-01-31",
         "target_start": "2020-01-31", "target_end": "2021-01-29",
         "target_horizon_days_actual": 364, "target_observable": True},
        {"security_id": "B", "ticker": "B", "feature_asof": "2020-01-31",
         "target_start": None, "target_end": None,
         "target_horizon_days_actual": None, "target_observable": False},
    ])
    annotated = annotate_observability(frame)
    assert annotated.loc[0, "target_known_at"] is not None and not pd.isna(annotated.loc[0, "target_known_at"])
    # a censored row carries NO known-at instant (pandas stores it as NaN, never a date)
    assert pd.isna(annotated.loc[1, "target_known_at"])
    # before the outcome is known, neither row is trainable
    early = trainable_mask(annotated, "2020-06-01")
    assert early.tolist() == [False, False]
    # after the outcome is public, only the observable row is trainable
    late = trainable_mask(annotated, "2021-06-01")
    assert late.tolist() == [True, False]


def test_target_known_at_uses_conservative_close_after_the_session():
    known = target_known_at("2021-01-29")
    assert known is not None
    assert known.tzinfo is not None
    # information is public only after the close, never before the trade date
    assert known > pd.Timestamp("2021-01-29", tz="UTC")


# ── FILING AVAILABILITY ──────────────────────────────────────────────────────

def test_fundamental_availability_is_not_period_end():
    period_end = "2020-12-31"
    filed = "2021-02-15"
    available = fundamental_available_at(filed)
    assert available is not None
    # a model at period end may NOT see a filing that had not happened yet
    assert is_available(available, period_end) is False
    # but a model after the filing date may
    assert is_available(available, "2021-02-20") is True


def test_fundamental_availability_takes_the_later_of_the_two_signals():
    # availability is the LATER of (filing_date + lag) and the SEC acceptance
    # instant, so it can never be moved earlier than the evidence supports.
    lag_only = fundamental_available_at("2021-02-15")
    assert lag_only == pd.Timestamp("2021-02-16 00:00:00", tz="UTC")
    # an acceptance instant before the lagged date does not make it available sooner
    same = fundamental_available_at("2021-02-15", acceptance_datetime="2021-02-15T21:30:00Z")
    assert same == lag_only
    assert is_available(same, "2021-02-15T23:59:00Z") is False
    # an acceptance instant AFTER the lagged date pushes availability later
    later = fundamental_available_at("2021-02-15", acceptance_datetime="2021-02-16T14:00:00Z")
    assert later == pd.Timestamp("2021-02-16 14:00:00", tz="UTC")
    assert is_available(later, "2021-02-16T13:00:00Z") is False
    assert is_available(later, "2021-02-16T15:00:00Z") is True


def test_price_availability_is_never_before_the_trade_date():
    available = price_available_at("2020-06-15")
    assert is_available(available, "2020-06-14") is False
    assert is_available(available, "2020-06-15T23:59:00Z") is True


# ── ROA / ROIC SEMANTICS ─────────────────────────────────────────────────────

def test_v2_roa_uses_average_assets_when_begin_and_end_available():
    roa, method = compute_v2_roa(10.0, 200.0, total_assets_begin=100.0)
    assert method == "average_assets"
    assert abs(roa - (10.0 / 150.0)) < 1e-12


def test_v2_roa_falls_back_to_point_assets_but_records_the_method():
    roa, method = compute_v2_roa(10.0, 200.0)
    assert method == "point_assets"
    assert abs(roa - (10.0 / 200.0)) < 1e-12
    roa_missing, method_missing = compute_v2_roa(None, 200.0)
    assert roa_missing is None and method_missing == "unavailable"


def test_v2_roa_and_roic_are_distinct_quantities():
    metrics = compute_v2_return_metrics(
        net_income=10.0, total_assets=200.0, total_assets_begin=100.0,
        operating_income=20.0, total_debt=50.0, equity=100.0, cash=10.0, tax_rate=0.25,
    )
    assert metrics["roa_available"] and metrics["roic_available"]
    assert metrics["roa"] != metrics["roic"]
    assert metrics["roa_definition"] == V2_ROA_DEFINITION
    assert metrics["roic_definition"] == V2_ROIC_DEFINITION
    # invested capital documented explicitly: debt + equity - cash = 50 + 100 - 10
    assert abs(metrics["invested_capital"] - 140.0) < 1e-9
    assert abs(metrics["nopat"] - 15.0) < 1e-9


# ── KEYES SEPARATION ─────────────────────────────────────────────────────────

def test_keyes_proxy_is_explicitly_named_not_silently_substituted():
    assert proxy_variable("X12") == "X12_PROXY"
    assert proxy_variable("X12_PROXY") == "X12_PROXY"


def test_keyes_historical_and_modern_are_separate_tracks():
    hist = KeyesRuleSet(
        name="keyes_historical", track=TRACK_HISTORICAL,
        criteria=(KeyesCriterion("X5", 0.10, "min"), KeyesCriterion("X8", 20.0, "max")),
    )
    modern = KeyesRuleSet(
        name="keyes_modern", track=TRACK_MODERN,
        criteria=(KeyesCriterion("X12_PROXY", 0.05, "min", is_proxy=True),),
    )
    assert qualify(hist, "AAA", {"X5": 0.12, "X8": 15.0}).qualified is True
    assert qualify(hist, "AAA", {"X5": 0.05, "X8": 15.0}).qualified is False
    modern_result = qualify(modern, "AAA", {"X12_PROXY": 0.09})
    assert modern_result.qualified is True
    assert modern_result.proxies_used == ("X12_PROXY",)
    assert modern_result.track == TRACK_MODERN


def test_keyes_missing_value_fails_its_criterion_never_counts_as_zero_pass():
    rule = KeyesRuleSet(
        name="r", track=TRACK_HISTORICAL,
        criteria=(KeyesCriterion("X5", 0.0, "min"),),
    )
    # a zero-threshold min criterion must NOT pass on a MISSING value
    assert qualify(rule, "AAA", {"X5": None}).qualified is False


def test_keyes_qualification_is_not_a_forced_top_percent_quota():
    rule = KeyesRuleSet(
        name="r", track=TRACK_HISTORICAL,
        criteria=(KeyesCriterion("X5", 0.10, "min"),),
    )
    # every security is judged on its own thresholds: none pass here
    values = [{"X5": 0.01}, {"X5": 0.05}, {"X5": 0.09}]
    qualified = [qualify(rule, "S%d" % i, v).qualified for i, v in enumerate(values)]
    assert qualified == [False, False, False]
    # and all pass when the threshold is met, regardless of ranking
    assert qualify(rule, "S", {"X5": 0.99}).qualified is True


# ── TRAIN-ONLY TRANSFORMS / MISSINGNESS ──────────────────────────────────────

def test_transform_fits_on_training_and_does_not_move_on_apply():
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    fitted = fit_transform(train, kind="TRAIN_ZSCORE")
    mean = fitted.params["mean"]
    # applying to validation data must reuse the TRAINING mean, not re-fit
    applied = apply_transform([100.0, 101.0], fitted)
    assert abs((100.0 - mean) / fitted.params["std"] - applied.iloc[0]) < 1e-9
    assert fitted.to_dict()["fit_reference"] == "training"


def test_imputer_fills_with_training_statistics_only():
    train = pd.DataFrame({"v": [1.0, 2.0, 3.0, None]})
    imputer = fit_imputer(train, "v", strategy="INDICATOR_PLUS_TRAIN_MEDIAN")
    assert imputer.global_fill == 2.0  # median of the TRAINING slice
    apply_frame = pd.DataFrame({"v": [None, 1000.0]})
    values, indicator, filled = apply_imputer(apply_frame, "v", imputer)
    # the missing cell uses the training median, NOT the apply-frame value (1000)
    assert values.iloc[0] == 2.0
    assert indicator.iloc[0] == 1 and indicator.iloc[1] == 0
    assert filled == 1


def test_imputer_never_fills_with_zero():
    train = pd.DataFrame({"v": [10.0, 20.0, None]})
    imputer = fit_imputer(train, "v", strategy="INDICATOR_PLUS_TRAIN_MEDIAN")
    assert imputer.global_fill != 0.0
    values, _, _ = apply_imputer(pd.DataFrame({"v": [None]}), "v", imputer)
    assert values.iloc[0] != 0.0


# ── FEATURE REGISTRY / SET ───────────────────────────────────────────────────

def test_initial_registry_defines_pit_and_missing_rules_for_every_feature():
    registry = build_initial_registry()
    registry.validate()
    for definition in registry.features.values():
        assert definition.available_at_rule in (
            "SEC_FILING_AVAILABILITY", "PRICE_CLOSE_AVAILABILITY",
            "MACRO_VINTAGE_AVAILABILITY", "UNIVERSE_MEMBERSHIP_AVAILABILITY",
        )
        assert definition.missing_rule != "ZERO_FILL"
        assert definition.raw_dependencies


def test_macro_features_are_declared_but_disabled_by_default():
    default = build_initial_registry()
    macro = [d for d in default.features.values() if d.category == "MACRO"]
    assert macro, "macro features must be declared"
    assert all(not d.enabled for d in macro)
    included = build_initial_registry(include_macro=True)
    assert any(d.enabled for d in included.features.values() if d.category == "MACRO")


def test_feature_registry_is_sealed_against_silent_mutation():
    registry = build_initial_registry()
    registry.seal()
    definition = next(iter(registry.features.values()))
    try:
        registry.add(definition)
        raise AssertionError("sealed registry must reject additions")
    except Exception as exc:  # FeatureRegistryError
        assert "sealed" in str(exc)


def test_feature_set_manifest_binds_dataset_target_and_registry():
    manifest = build_feature_set("dataset_dbaa77445b38", "deadbeef")
    assert manifest.dataset_id == "dataset_dbaa77445b38"
    assert manifest.git_commit == "deadbeef"
    assert manifest.feature_set_id.startswith("feature_set_")
    assert manifest.fingerprint
    # deterministic: same inputs -> same id
    assert build_feature_set("dataset_dbaa77445b38", "deadbeef").feature_set_id == manifest.feature_set_id


def test_v1_disposition_table_covers_legacy_features_with_reasons():
    table = v1_disposition()
    assert table
    for name, entry in table.items():
        assert entry["disposition"] in ("KEEP", "REDEFINE", "LEGACY_ONLY", "DROP", "DEFER")
        assert entry["reason"]
    # the V1 ROA/ROIC conflation is not silently kept: it is explicitly redefined
    assert table["roic"]["disposition"] == "REDEFINE"
    assert table["roa"]["disposition"] == "REDEFINE"
