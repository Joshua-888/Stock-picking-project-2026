"""Tests for the WP2 point-in-time data foundation (``src.research.data``).

All fixtures here are small, explicitly labelled synthetic data used only to
exercise the integrity logic. They are never research evidence: the research
mode gate refuses synthetic data in RESEARCH_V2 (test 7). The production
database is never touched; every filesystem write goes to ``tmp_path``.
"""

import json

import pandas as pd
import pytest

from src.research import DatasetManifest, ImmutabilityError, ManifestValidationError, ResearchMode, load_immutable, save_immutable
from src.research.data import (
    AmbiguousTickerError,
    SecurityMaster,
    SecurityRecord,
    TickerAssignment,
    UniverseMembership,
    UniverseTable,
    availability_mask,
    asof_join,
    compute_return_metrics,
    compute_roa,
    compute_roic,
    is_available,
    latest_version_asof,
    layer_root,
    read_silver_table,
    security_id_for,
    write_bronze_bytes,
    write_bronze_json,
    write_gold_table,
)
from src.research.data import benchmark as bm
from src.research.data import manifesting as mf
from src.research.data.manifesting import ManifestingError
from src.research.data import prices as px
from src.research.data.macro import VintageNotHonoredError, assert_vintage_honored, parse_vintage_csv
from src.research.modes import SyntheticDataError

SYNTHETIC = "synthetic"  # label carried by every fixture in this file


# ── Fixtures (synthetic, labelled, fixtures only) ─────────────────────────────

def _facts_frame():
    """Synthetic fundamentals: one concept with an original and a restatement."""
    return pd.DataFrame(
        [
            {"cik": "1", "concept": "Rev", "fiscal_period_end": "2020-03-31", "filing_date": "2020-05-01", "accession": "a-orig", "value": 100.0},
            {"cik": "1", "concept": "Rev", "fiscal_period_end": "2020-03-31", "filing_date": "2020-08-01", "accession": "a-amend", "value": 110.0},
            {"cik": "2", "concept": "Rev", "fiscal_period_end": "2020-03-31", "filing_date": "2020-05-02", "accession": "b-orig", "value": 200.0},
            {"cik": "3", "concept": "Rev", "fiscal_period_end": "2020-03-31", "filing_date": None, "accession": "c-missing", "value": 300.0},
        ]
    )


def _fundamental_frame():
    """Synthetic fundamentals with available_at already resolved."""
    frame = _facts_frame()
    frame["available_at"] = frame["filing_date"]
    return frame


def _price_frame():
    """Synthetic price frame spanning a weekend gap and one missing day."""
    rows = []
    for day, close, adj in [
        ("2024-01-04", 100.0, 100.0),
        ("2024-01-05", 101.0, 101.0),
        ("2024-01-08", 103.0, 103.0),
        ("2024-01-09", 104.0, 104.0),
    ]:
        rows.append(
            {
                "ticker": "SPY",
                "trade_date": day,
                "close": close,
                "adjusted_close": adj,
                "raw_close": close,
                "session_utc": "%sT20:00:00+00:00" % day,
                "exchange_timezone": "America/New_York",
                "available_at": "%sT21:00:00+00:00" % day,
                "has_split_event": False,
                "has_dividend_event": False,
                "split_count": 0,
                "dividend_count": 0,
                "source": "FIXTURE_SYNTHETIC",
            }
        )
    return pd.DataFrame(rows)


# ── 1. available_at <= prediction_ts enforced; future fixture rejected ────────

def test_availability_enforced_and_future_rejected():
    assert is_available("2020-05-01T00:00:00+00:00", "2020-05-01T00:00:00+00:00") is True
    assert is_available("2020-05-02T00:00:00+00:00", "2020-05-01T00:00:00+00:00") is False
    frame = _fundamental_frame()
    mask = availability_mask(frame, "available_at", "2020-06-01")
    assert mask.tolist() == [True, False, True, False]
    joined = asof_join(frame, "2020-06-01", ["cik", "concept"], "available_at", ["value"], version_col="accession")
    assert set(joined["cik"]) == {"1", "2"}


def test_future_dated_adversarial_fixture_rejected():
    frame = pd.DataFrame(
        [{"cik": "9", "concept": "Rev", "value": 1.0, "available_at": "2999-01-01T00:00:00+00:00"}]
    )
    joined = asof_join(frame, "2020-06-01", ["cik", "concept"], "available_at", ["value"])
    assert joined.empty


# ── 2. filing_date != fiscal_period_end ───────────────────────────────────────

def test_filing_date_distinct_from_fiscal_period_end():
    from src.research.data.availability import fundamental_available_at

    available = fundamental_available_at("2020-05-01", acceptance_datetime=None)
    assert available > pd.Timestamp("2020-03-31", tz="UTC"), "period end must not be treated as availability"
    assert available >= pd.Timestamp("2020-05-01", tz="UTC")
    frame = _fundamental_frame()
    # At the fiscal period end the value is not visible yet even though the period exists.
    mask = availability_mask(frame, "available_at", "2020-03-31")
    assert mask.tolist() == [False, False, False, False]


# ── 3. restatement/amendment does not leak backwards ─────────────────────────

def test_restatement_does_not_leak_backwards():
    frame = _fundamental_frame()
    before = asof_join(frame, "2020-06-01", ["cik", "concept"], "available_at", ["value"], version_col="accession")
    original = before.loc[before["cik"] == "1"].iloc[0]
    assert original["value"] == 100.0
    assert original["accession"] == "a-orig"
    after = asof_join(frame, "2020-09-01", ["cik", "concept"], "available_at", ["value"], version_col="accession")
    amended = after.loc[after["cik"] == "1"].iloc[0]
    assert amended["value"] == 110.0
    assert amended["accession"] == "a-amend"


def test_latest_version_isoldest_eligible_not_global_latest():
    frame = _fundamental_frame()
    only_original = latest_version_asof(frame, "2020-06-01", ["cik", "concept"], "available_at", version_col="accession")
    assert "a-amend" not in set(only_original["accession"])
    assert set(only_original["cik"]) == {"1", "2"}


# ── 4. as-of join correctness incl. duplicate keys and non-explosion ─────────

def test_asof_join_duplicate_keys_do_not_explode():
    frame = pd.DataFrame(
        [
            {"cik": "1", "field": "rev", "value": 1.0, "available_at": "2020-01-01"},
            {"cik": "1", "field": "rev", "value": 1.0, "available_at": "2020-01-01"},
            {"cik": "1", "field": "rev", "value": 2.0, "available_at": "2020-02-01"},
            {"cik": "1", "field": "rev", "value": 3.0, "available_at": "2020-03-01"},
        ]
    )
    joined = asof_join(frame, "2020-03-15", ["cik", "field"], "available_at", ["value"])
    assert len(joined) == 1
    assert joined.iloc[0]["value"] == 3.0
    # one row per key even when many versions exist
    multi = pd.DataFrame(
        [
            {"cik": str(i), "field": "rev", "value": float(i), "available_at": "2020-0%d-01" % (i + 1)}
            for i in range(1, 4)
        ]
    )
    joined_multi = asof_join(multi, "2020-12-01", ["cik", "field"], "available_at", ["value"])
    assert len(joined_multi) == 3


def test_asof_join_missing_availability_is_unavailable():
    frame = _fundamental_frame()
    joined = asof_join(frame, "2021-01-01", ["cik", "concept"], "available_at", ["value"])
    assert "3" not in set(joined["cik"]), "a fact with no filing timestamp must not be invented"


# ── 5. ticker change, symbol reuse, duplicate detection, id resolution ───────

def test_security_master_ticker_change_and_reuse():
    first = security_id_for(cik="0000320193")
    second = security_id_for(cik="0000789019")
    assert first != second
    master = SecurityMaster(
        securities=[
            SecurityRecord(security_id=first, name="First Co", cik="0000320193"),
            SecurityRecord(security_id=second, name="Second Co", cik="0000789019"),
        ]
    )
    master.add_ticker(TickerAssignment(first, "XYZ", "NYSE", "2000-01-01", "2010-12-31"))
    master.add_ticker(TickerAssignment(second, "XYZ", "NASDAQ", "2011-01-01"))
    assert master.resolve_ticker("XYZ", "2005-06-01") == first
    assert master.resolve_ticker("XYZ", "2020-06-01") == second
    assert master.detect_symbol_reuse() == {"XYZ": sorted([first, second])}
    history = master.ticker_changes(first)
    assert history[0]["ticker"] == "XYZ"


def test_security_master_ambiguous_and_unknown():
    first = security_id_for(cik="0000320193")
    second = security_id_for(cik="0000789019")
    master = SecurityMaster(
        securities=[
            SecurityRecord(security_id=first, name="First Co", cik="0000320193"),
            SecurityRecord(security_id=second, name="Second Co", cik="0000789019"),
        ]
    )
    master.add_ticker(TickerAssignment(first, "ABC", "NYSE", "2000-01-01"))
    master.add_ticker(TickerAssignment(second, "ABC", "NASDAQ", "2000-01-01"))
    with pytest.raises(AmbiguousTickerError):
        master.resolve_ticker("ABC", "2015-01-01")
    assert master.resolve_series("ABC", ["2015-01-01"]) == [None]
    with pytest.raises(Exception):
        master.resolve_ticker("NONE", "2015-01-01")


def test_security_master_duplicate_detection():
    canonical = security_id_for(cik="0000320193")
    # A deliberate duplicate: two distinct ids claiming the same CIK. The master
    # must expose this instead of silently treating the two as unrelated.
    bogus = "sec_" + "0" * 16
    master = SecurityMaster(
        securities=[
            SecurityRecord(security_id=canonical, name="First Co", cik="0000320193"),
            SecurityRecord(security_id=bogus, name="First Co (dup)", cik="0000320193"),
        ]
    )
    duplicates = master.duplicate_securities()
    assert duplicates == {"0000320193": sorted([bogus, canonical])}
    # Re-registering the same id with different content is refused outright.
    with pytest.raises(Exception):
        master.add_security(SecurityRecord(security_id=canonical, name="Different", cik="0000320193"))


# ── 6. missing publication timestamp -> rejected/unavailable ─────────────────

def test_missing_publication_timestamp_rejected():
    from src.research.data.availability import fundamental_available_at

    assert fundamental_available_at(None, acceptance_datetime=None) is None
    assert is_available(None, "2020-01-01") is False
    frame = _fundamental_frame()
    row = frame.loc[frame["cik"] == "3"].iloc[0]
    # pandas stores the missing object value as NaN; either way it is not a timestamp
    assert row["filing_date"] is None or pd.isna(row["filing_date"])


# ── 7. RESEARCH_V2 + synthetic -> raises; LEGACY_V1 -> warns ──────────────────

def test_research_v2_rejects_synthetic_legacy_warns():
    from src.research.modes import assert_no_synthetic_in_research

    with pytest.raises(SyntheticDataError):
        assert_no_synthetic_in_research(ResearchMode.RESEARCH_V2, True)
    with pytest.warns(UserWarning):
        assert_no_synthetic_in_research(ResearchMode.LEGACY_V1, True)
    frame = _price_frame()
    with pytest.raises(SyntheticDataError):
        mf.build_and_persist_gold(
            "synthetic_gold",
            frame,
            mode=ResearchMode.RESEARCH_V2,
            universe_id="u",
            period_start="2024-01-01",
            period_end="2024-12-31",
            sources=["FIXTURE_SYNTHETIC"],
            source_fingerprints={"fixture": "a" * 64},
            pit_status="not_point_in_time",
            synthetic=True,
        )


# ── 8. ROA vs ROIC semantics ─────────────────────────────────────────────────

def test_roa_and_roic_are_distinct():
    roa = compute_roa(10.0, 100.0)
    roic, nopat, invested = compute_roic(20.0, 10.0, 90.0, 0.0, tax_rate=0.21)
    assert roa == pytest.approx(0.10)
    assert roic == pytest.approx(20.0 * 0.79 / 100.0)
    assert roa != roic
    assert invested == pytest.approx(100.0)
    assert nopat == pytest.approx(15.8)


def test_roa_roic_unavailable_when_fields_missing():
    assert compute_roa(None, 100.0) is None
    assert compute_roa(10.0, None) is None
    assert compute_roa(10.0, 0.0) is None
    assert compute_roic(None, 1.0, 1.0, 0.0) == (None, None, None)
    assert compute_roic(20.0, 10.0, 90.0, None) == (None, None, None)
    metrics = compute_return_metrics(10.0, 100.0, operating_income=None, total_debt=1.0, equity=1.0, cash=0.0)
    assert metrics.roa_available is True
    assert metrics.roic_available is False
    assert metrics.roic_unavailable_reason == "operating_income unavailable"
    assert metrics.roa != metrics.roic or metrics.roic is None


def test_roa_never_relabelled_as_roic():
    metrics = compute_return_metrics(10.0, 100.0)
    assert metrics.roa == pytest.approx(0.10)
    assert metrics.roic is None
    assert metrics.roic_unavailable_reason


# ── 9. price corporate-action/split handling + nearest-prior alignment ───────

def test_price_alignment_uses_nearest_prior_only():
    frame = _price_frame()
    aligned = px.align_nearest_prior(frame, ["2024-01-06", "2024-01-09", "2024-01-01"], ticker="SPY")
    # 2024-01-06 is a Saturday; nearest prior session is 2024-01-05
    assert aligned.iloc[0]["trade_date"] == "2024-01-05"
    # 2024-01-09 aligns to itself
    assert aligned.iloc[1]["trade_date"] == "2024-01-09"
    # 2024-01-01 precedes every observation -> no future pull-back, value missing
    assert pd.isna(aligned.iloc[2]["trade_date"])
    assert pd.isna(aligned.iloc[2]["adjusted_close"])


def test_price_corporate_action_ledger_documents_adjustments():
    frame = _price_frame()
    frame.attrs["ticker"] = "SPY"
    frame.attrs["exchange_timezone"] = "America/New_York"
    frame.attrs["split_events"] = [{"date": 1704067200, "numerator": 2, "denominator": 1}]
    frame.attrs["dividend_events"] = []
    ledger = px.build_corporate_action_ledger(frame)
    assert ledger["split_events"][0]["numerator"] == 2
    assert ledger["rows"] == len(frame)
    assert len(ledger["fingerprint"]) == 64
    assert px.validate_price_frame(frame) == []


def test_benchmark_nearest_prior_alignment_no_future_obs():
    frame = _price_frame()
    frame.attrs["ticker"] = "SPY"
    series = bm.build_benchmark_series("SPY", frame=frame)
    aligned = bm.align_benchmark(series, ["2024-01-06"])
    assert aligned.iloc[0]["trade_date"] == "2024-01-05"
    # A request at 2024-01-09T00:00 cannot have seen the 2024-01-09 close, so it
    # must align to the prior session (2024-01-08) -- this is the availability
    # rule doing its job, not a future observation.
    ret = bm.benchmark_return(series, "2024-01-06", "2024-01-09")
    assert ret == pytest.approx(103.0 / 101.0 - 1.0)
    # With an explicit post-close instant the same-day close is legitimately visible.
    later = bm.benchmark_return(series, "2024-01-06T23:00:00+00:00", "2024-01-09T23:00:00+00:00")
    assert later == pytest.approx(104.0 / 101.0 - 1.0)


# ── 10. deterministic fingerprints ───────────────────────────────────────────

def test_layers_deterministic_source_and_dataset_fingerprints(tmp_path):
    root = str(tmp_path / "research_v2")
    first = write_bronze_bytes(root, "prov", b"{\"a\":1}")
    second = write_bronze_bytes(root, "prov", b"{\"a\":1}")
    assert first["fingerprint"] == second["fingerprint"]
    assert second["outcome"] == "verify_and_reuse"
    changed = write_bronze_bytes(root, "prov", b"{\"a\":2}")
    assert changed["fingerprint"] != first["fingerprint"]
    assert changed["outcome"] == "written"
    frame = _price_frame()
    table_a = write_gold_table(root, "gold_a", frame)
    table_b = write_gold_table(root, "gold_a", frame.copy())
    assert table_a["fingerprint"] == table_b["fingerprint"]


# ── 11. DatasetManifest creation validates with WP1 ──────────────────────────

def test_gold_manifest_validates_and_persists(tmp_path):
    root = str(tmp_path / "research_v2")
    result = mf.build_and_persist_gold(
        "gold_c",
        _fundamental_frame(),
        mode=ResearchMode.RESEARCH_V2,
        universe_id="sample_ciks_v1",
        period_start="2020-01-01",
        period_end="2020-12-31",
        sources=["SEC_EDGAR"],
        source_fingerprints={"edgar_companyfacts": "a" * 64},
        pit_status="partially_point_in_time",
        known_limitations=["sample only"],
        root=root,
    )
    manifest = DatasetManifest.from_dict(result["manifest"])
    manifest.validate()
    assert manifest.synthetic_data_status == "none"
    assert manifest.pit_status == "partially_point_in_time"
    stored = load_immutable(result["path"])
    assert stored["dataset_fingerprint"] == manifest.dataset_fingerprint


def test_manifest_rejects_synthetic_point_in_time():
    with pytest.raises((SyntheticDataError, ManifestValidationError, ManifestingError)):
        mf.build_gold_manifest(
            "bad",
            _price_frame(),
            mode=ResearchMode.LEGACY_V1,
            universe_id="u",
            period_start="2024-01-01",
            period_end="2024-12-31",
            sources=["FIXTURE"],
            source_fingerprints={"x": "a" * 64},
            dataset_fingerprint="b" * 64,
            pit_status="point_in_time",
            synthetic=True,
        )


# ── 12. Bronze immutability semantics ────────────────────────────────────────

def test_bronze_immutability_identical_reuse_and_new_version(tmp_path):
    root = str(tmp_path / "research_v2")
    first = write_bronze_json(root, "facts", {"cik": "1", "value": 100})
    second = write_bronze_json(root, "facts", {"value": 100, "cik": "1"})
    assert second["outcome"] == "verify_and_reuse"
    assert first["fingerprint"] == second["fingerprint"]
    third = write_bronze_json(root, "facts", {"cik": "1", "value": 999})
    assert third["outcome"] == "written"
    assert third["fingerprint"] != first["fingerprint"]
    versions = sorted(p.name for p in layer_root(root, "bronze", "facts").iterdir())
    assert len(versions) == 2


def test_bronze_and_immutability_guard_together(tmp_path):
    from src.research.data import layers

    root = str(tmp_path / "research_v2")
    record = write_bronze_json(root, "k", {"a": 1})
    raw_path = record["raw_path"]
    assert load_immutable and raw_path
    # save_immutable refuses to overwrite a different payload
    target = tmp_path / "artifact.json"
    assert save_immutable(target, {"a": 1}) == "written"
    with pytest.raises(ImmutabilityError):
        save_immutable(target, {"a": 2})


# ── Universe contract (PARTIAL by design) ────────────────────────────────────

def test_universe_membership_asof_and_partial_status():
    table = UniverseTable(
        "sp500_sample",
        [
            UniverseMembership("sp500_sample", "sec_a", "AAA", "2010-01-01", "2018-12-31"),
            UniverseMembership("sp500_sample", "sec_b", "BBB", "2015-01-01"),
        ],
    )
    assert table.members_asof("2012-06-01") == ["sec_a"]
    assert sorted(table.members_asof("2016-06-01")) == ["sec_a", "sec_b"]
    assert table.is_member("sec_b", "2010-06-01") is False
    status = table.status()
    assert status.status == "PARTIAL"
    assert status.survivorship_safe is False
    assert status.limitations
    check = table.survivorship_check("2016-06-01")
    assert check["exposes_possible_survivorship_bias"] is False


def test_universe_invalid_window_rejected():
    table = UniverseTable("u")
    with pytest.raises(Exception):
        table.add(UniverseMembership("u", "sec", "T", "2020-01-01", "2019-01-01"))


# ── Macro vintage honesty ────────────────────────────────────────────────────

def test_macro_vintage_guard_rejects_future_observations():
    dishonest = "observation_date,DGS10\n2020-05-01,0.64\n2026-01-01,4.00\n"
    with pytest.raises(VintageNotHonoredError):
        parse_vintage_csv(dishonest, "DGS10", "2020-06-15")
    honest = "observation_date,DGS10\n2020-05-01,0.64\n2020-05-04,0.66\n"
    frame = parse_vintage_csv(honest, "DGS10", "2020-06-15")
    assert len(frame) == 2
    assert frame["available_at"].iloc[0].startswith("2020-06-15")
    assert assert_vintage_honored(pd.read_csv(__import__("io").StringIO(honest)), "2020-06-15") is True


def test_macro_missing_vintage_not_substituted():
    from src.research.data.macro import validate_macro_frame

    frame = pd.DataFrame(
        [
            {
                "series_id": "DGS10",
                "observation_date": "2020-05-01",
                "vintage_date": "2020-06-15",
                "release_date": None,
                "available_at": "2020-06-15T00:00:00+00:00",
            }
        ]
    )
    assert validate_macro_frame(frame) == []
    broken = frame.copy()
    broken.loc[0, "vintage_date"] = None
    assert validate_macro_frame(broken)
