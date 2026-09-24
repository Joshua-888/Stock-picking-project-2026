"""WP2A adversarial certification tests (data-integrity hard gates).

Fixtures here are small, explicitly labelled synthetic data used only to exercise
integrity logic; RESEARCH_V2 refuses synthetic data (see tests/test_research_v2_data.py).
The real bounded price sample is produced by the certification run itself and is
never fabricated. Every filesystem write goes to ``tmp_path``; the production
database is never touched.
"""

import pandas as pd
import pytest

from src.research.data.pit_prices import (
    DIVIDEND,
    SPLIT,
    CorporateAction,
    build_pit_adjusted_frame,
    pit_return,
    validate_action_ledger,
)
from src.research.data.provenance_ledger import (
    list_entries,
    load_manifest,
    persist_manifest,
    verify_ledger,
)
from src.research.data.sample_universe import load_sample_universe
from src.research.data.universe import UniverseMembership, UniverseTable
from src.research.data.security_master import (
    AmbiguousTickerError,
    SecurityMaster,
    SecurityRecord,
    TickerAssignment,
    security_id_for,
)
from src.research.immutability import ImmutabilityError


# ── fixtures ─────────────────────────────────────────────────────────────────

def _raw_frame():
    """Three sessions around a split (2020-05-01) and a dividend (2020-07-01)."""
    return pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": "2020-01-02", "raw_close": 100.0},
            {"ticker": "AAA", "trade_date": "2020-04-30", "raw_close": 102.0},
            {"ticker": "AAA", "trade_date": "2020-06-01", "raw_close": 50.0},
            {"ticker": "AAA", "trade_date": "2020-07-01", "raw_close": 49.0},
        ]
    )


def _split():
    return CorporateAction(ticker="AAA", kind=SPLIT, effective_date="2020-05-01", numerator=2.0, denominator=1.0)


def _dividend():
    return CorporateAction(ticker="AAA", kind=DIVIDEND, effective_date="2020-07-01", amount=1.0)


def _manifest(created_at="2026-01-01T00:00:00+00:00"):
    """A realistic manifest mapping for ledger tests, including created_at."""
    return {
        "dataset_id": "dataset_abcdef012345",
        "created_at": created_at,
        "dataset_fingerprint": "f" * 64,
        "config_fingerprint": "c" * 64,
        "pit_status": "partially_point_in_time",
        "synthetic_data_status": "none",
        "row_count": 4,
        "period_start": "2020-01-01",
        "period_end": "2020-12-31",
        "universe_definition": "sample_v1",
        "git_commit": "deadbeef",
        "branch": "feat/v2-wp2a-dataset-certification",
        "schema_version": "research_v2_gold_v1",
        "sources": ["YAHOO_CHART"],
        "source_fingerprints": {"yahoo_chart:AAA": "a" * 64},
        "known_limitations": [],
        "notes": "",
    }


# ── 1. future filing unavailable pre-publication ─────────────────────────────

def test_future_filing_unavailable_before_publication():
    from src.research.data.availability import fundamental_available_at, is_available

    available = fundamental_available_at("2020-05-01", acceptance_datetime=None)
    assert is_available(available.isoformat(), "2020-04-30") is False, "must not be usable before filing"
    assert is_available(available.isoformat(), "2020-05-02") is True


# ── 2. restatement does not leak backwards ───────────────────────────────────

def test_restatement_no_backward_leak():
    from src.research.data.pit_join import asof_join

    frame = pd.DataFrame(
        [
            {"cik": "1", "concept": "Rev", "value": 100.0, "available_at": "2020-05-01"},
            {"cik": "1", "concept": "Rev", "value": 110.0, "available_at": "2020-08-01"},
        ]
    )
    before = asof_join(frame, "2020-06-01", ["cik", "concept"], "available_at", ["value"])
    assert before.iloc[0]["value"] == 100.0
    after = asof_join(frame, "2020-09-01", ["cik", "concept"], "available_at", ["value"])
    assert after.iloc[0]["value"] == 110.0


# ── 3. future split cannot alter earlier-T value ─────────────────────────────

def test_future_split_cannot_alter_earlier_value():
    raw = _raw_frame()
    asof = "2020-06-30"
    with_split = build_pit_adjusted_frame(raw, [_split()], asof=asof)
    # Adding a FUTURE split (after asof) must not change any value used at asof.
    future = CorporateAction(ticker="AAA", kind=SPLIT, effective_date="2020-12-01", numerator=3.0, denominator=1.0)
    with_extra = build_pit_adjusted_frame(raw, [_split(), future], asof=asof)
    left = with_split.set_index("trade_date")["adjusted_close_pit"].fillna(-1)
    right = with_extra.set_index("trade_date")["adjusted_close_pit"].fillna(-1)
    assert left.equals(right) is False or left.tolist() == right.tolist()
    assert left.tolist() == right.tolist()


def test_split_adjusts_only_prior_sessions():
    raw = _raw_frame()
    frame = build_pit_adjusted_frame(raw, [_split()], asof="2020-06-30")
    by_date = frame.set_index("trade_date")["adjusted_close_pit"]
    assert by_date["2020-01-02"] == pytest.approx(50.0), "pre-split price halved by 2:1 split"
    assert by_date["2020-06-01"] == pytest.approx(50.0), "post-split price untouched"


# ── 4. future dividend cannot alter earlier-T value ──────────────────────────

def test_future_dividend_cannot_alter_earlier_value():
    raw = _raw_frame()
    asof = "2020-06-30"
    base = build_pit_adjusted_frame(raw, [_split()], asof=asof)
    with_future_div = build_pit_adjusted_frame(raw, [_split(), _dividend()], asof=asof)
    assert base["adjusted_close_pit"].fillna(-1).tolist() == with_future_div["adjusted_close_pit"].fillna(-1).tolist()


def test_dividend_factor_applies_only_to_prior_sessions():
    raw = _raw_frame()
    frame = build_pit_adjusted_frame(raw, [_split(), _dividend()], asof="2020-12-31")
    by_date = frame.set_index("trade_date")["adjusted_close_pit"]
    # 2020-07-01 ex-date close = 49.0, amount 1.0 -> factor (49-1)/49 = 0.9795918...
    expected_pre = 100.0 * 0.5 * ((49.0 - 1.0) / 49.0)
    assert by_date["2020-01-02"] == pytest.approx(expected_pre)
    # the ex-date session itself is NOT adjusted by its own dividend
    assert by_date["2020-07-01"] == pytest.approx(49.0)


def test_pit_return_uses_only_actions_within_window():
    raw = _raw_frame()
    # From 2020-01-02 to 2020-06-01 the only known action is the split -> -50% -> +0%
    ret = pit_return(raw, [_split(), _dividend()], "2020-01-02", "2020-06-01")
    assert ret == pytest.approx(0.0)
    # Missing endpoints must return None, never be approximated.
    assert pit_return(raw, [_split()], "2019-01-01", "2020-06-01") is None


def test_action_ledger_integrity_flags_duplicates_and_bad_ratios():
    problems = validate_action_ledger([_split(), _split()])
    assert any("duplicate" in item for item in problems)
    bad = CorporateAction.__new__(CorporateAction)  # bypass __post_init__ to craft a bad ratio
    object.__setattr__(bad, "ticker", "AAA")
    object.__setattr__(bad, "kind", SPLIT)
    object.__setattr__(bad, "effective_date", "2020-05-01")
    object.__setattr__(bad, "numerator", None)
    object.__setattr__(bad, "denominator", None)
    object.__setattr__(bad, "amount", None)
    object.__setattr__(bad, "raw_unix", None)
    assert validate_action_ledger([bad])


def test_corporate_action_rejects_unparseable_date():
    with pytest.raises(Exception):
        CorporateAction(ticker="AAA", kind=SPLIT, effective_date="not-a-date", numerator=2.0, denominator=1.0)


# ── 5. future universe entrant not present before membership_start ───────────

def test_future_universe_entrant_absent_before_membership_start():
    table = UniverseTable(
        "sp500_sample",
        [
            UniverseMembership("sp500_sample", "sec_old", "OLD", "2010-01-01", "2018-12-31"),
            UniverseMembership("sp500_sample", "sec_new", "NEW", "2020-01-01"),
        ],
    )
    assert table.is_member("sec_new", "2019-06-01") is False
    assert table.is_member("sec_new", "2020-06-01") is True
    assert table.members_asof("2019-06-01") == []


# ── 6. delisted constituent remains present before membership_end ────────────

def test_delisted_constituent_present_before_membership_end():
    table = UniverseTable(
        "sp500_sample",
        [UniverseMembership("sp500_sample", "sec_delisted", "DEL", "2005-01-01", "2015-12-31")],
    )
    assert table.is_member("sec_delisted", "2015-12-30") is True
    assert table.is_member("sec_delisted", "2016-01-01") is False
    check = table.survivorship_check("2015-12-30")
    assert check["exposes_possible_survivorship_bias"] is False, "a recorded exit must be visible"


def test_universe_only_open_ended_flagged_as_survivorship_risk():
    table = UniverseTable("u", [UniverseMembership("u", "sec_a", "A", "2010-01-01")])
    check = table.survivorship_check("2020-01-01")
    assert check["exposes_possible_survivorship_bias"] is True


# ── 7. ticker rename -> same security; reused ticker -> correct by date ──────

def test_ticker_rename_maps_to_same_security():
    sid = security_id_for(cik="0000320193")
    master = SecurityMaster(securities=[SecurityRecord(security_id=sid, name="Co", cik="0000320193")])
    master.add_ticker(TickerAssignment(sid, "OLD", "NYSE", "2000-01-01", "2010-12-31"))
    master.add_ticker(TickerAssignment(sid, "NEW", "NYSE", "2011-01-01"))
    assert master.resolve_ticker("OLD", "2005-06-01") == master.resolve_ticker("NEW", "2015-06-01")


def test_reused_ticker_resolves_by_date():
    first = security_id_for(cik="0000320193")
    second = security_id_for(cik="0000789019")
    master = SecurityMaster(
        securities=[
            SecurityRecord(security_id=first, name="First", cik="0000320193"),
            SecurityRecord(security_id=second, name="Second", cik="0000789019"),
        ]
    )
    master.add_ticker(TickerAssignment(first, "XYZ", "NYSE", "2000-01-01", "2010-12-31"))
    master.add_ticker(TickerAssignment(second, "XYZ", "NASDAQ", "2011-01-01"))
    assert master.resolve_ticker("XYZ", "2005-01-01") == first
    assert master.resolve_ticker("XYZ", "2020-01-01") == second


def test_ambiguous_ticker_fails_loudly():
    first = security_id_for(cik="0000320193")
    second = security_id_for(cik="0000789019")
    master = SecurityMaster(
        securities=[
            SecurityRecord(security_id=first, name="First", cik="0000320193"),
            SecurityRecord(security_id=second, name="Second", cik="0000789019"),
        ]
    )
    master.add_ticker(TickerAssignment(first, "ABC", "NYSE", "2000-01-01"))
    master.add_ticker(TickerAssignment(second, "ABC", "NASDAQ", "2000-01-01"))
    with pytest.raises(AmbiguousTickerError):
        master.resolve_ticker("ABC", "2015-01-01")


def test_security_master_live_cik_wiring():
    from src.research.data.certification import build_security_master, fetch_sec_tickers

    injected = {
        "1": {"cik_str": 320193, "ticker": "AAA", "title": "A Co"},
        "2": {"cik_str": 789019, "ticker": "BBB", "title": "B Co"},
    }
    sec_map = fetch_sec_tickers({"company_tickers": injected})
    master, diag = build_security_master(["AAA", "BBB", "ZZZUNKNOWN"], sec_map, as_of="2020-01-01")
    assert diag["mapped"] == ["AAA", "BBB"]
    assert diag["unmapped"] == ["ZZZUNKNOWN"]
    aaa = master.resolve_ticker("AAA", "2020-01-01")
    assert aaa == security_id_for(cik="0000320193")


# ── 8. benchmark alignment never selects a future day ────────────────────────

def test_benchmark_alignment_never_future():
    from src.research.data.benchmark import align_benchmark, build_benchmark_series

    frame = pd.DataFrame(
        [
            {"ticker": "SPY", "trade_date": d, "close": c, "adjusted_close": c, "raw_close": c,
             "session_utc": "%sT20:00:00+00:00" % d, "exchange_timezone": "America/New_York",
             "available_at": "%sT21:00:00+00:00" % d, "has_split_event": False,
             "has_dividend_event": False, "split_count": 0, "dividend_count": 0, "source": "FIXTURE_SYNTHETIC"}
            for d, c in [("2024-01-05", 100.0), ("2024-01-08", 103.0), ("2024-01-09", 104.0)]
        ]
    )
    frame.attrs["ticker"] = "SPY"
    series = build_benchmark_series("SPY", frame=frame)
    # Saturday 2024-01-06 must align to Friday 2024-01-05, not to Monday.
    aligned = align_benchmark(series, ["2024-01-06"])
    assert aligned.iloc[0]["trade_date"] == "2024-01-05"
    # A date before all observations must not pull from the future.
    early = align_benchmark(series, ["2023-12-01"])
    assert pd.isna(early.iloc[0]["trade_date"])


# ── 9. missing real data fails (no synthetic fallback) ───────────────────────

def test_missing_real_data_fails_no_synthetic_fallback():
    from src.research.data.prices import PricesUnavailableError, fetch_chart_payload

    # A delisted ticker returns HTTP 404 -> provider unavailable -> hard failure.
    with pytest.raises(Exception):
        fetch_chart_payload("ZZZZ_DOES_NOT_EXIST_9999")
    assert issubclass(PricesUnavailableError, Exception)


def test_certification_gate_rejects_synthetic():
    from src.research.data.manifesting import build_gold_manifest
    from src.research.modes import ResearchMode, SyntheticDataError

    with pytest.raises(SyntheticDataError):
        build_gold_manifest(
            "x", _raw_frame(), mode=ResearchMode.RESEARCH_V2, universe_id="u",
            period_start="2020-01-01", period_end="2020-12-31", sources=["FIXTURE"],
            source_fingerprints={"x": "a" * 64}, dataset_fingerprint="b" * 64,
            pit_status="not_point_in_time", synthetic=True,
        )


# ── 10. duplicate joins cannot multiply observations ─────────────────────────

def test_split_join_does_not_multiply_observations():
    raw = _raw_frame()
    frame = build_pit_adjusted_frame(raw, [_split(), _dividend()], asof="2020-12-31")
    assert len(frame) == len(raw), "PIT rebuild must be one row per (ticker, trade_date)"
    assert frame[["ticker", "trade_date"]].duplicated().sum() == 0


def test_action_ledger_for_other_ticker_does_not_leak():
    raw = _raw_frame()
    other = CorporateAction(ticker="BBB", kind=SPLIT, effective_date="2020-05-01", numerator=2.0, denominator=1.0)
    frame = build_pit_adjusted_frame(raw, [other], asof="2020-06-30")
    assert frame["actions_applied"].sum() == 0


# ── 11. durable provenance ledger ────────────────────────────────────────────

def test_provenance_ledger_persist_idempotent(tmp_path):
    """Real-path idempotency: a new created_at must NOT break write-once reuse."""
    root = str(tmp_path / "provenance")
    first = persist_manifest(_manifest(created_at="2026-01-01T00:00:00+00:00"), root=root)
    assert first["outcome"] == "written"
    # Genuine re-certification of identical content -> same dataset id, new clock.
    second = persist_manifest(_manifest(created_at="2026-02-02T00:00:00+00:00"), root=root)
    assert second["outcome"] == "verify_and_reuse", "created_at must not defeat reuse"
    assert second["index_outcome"] == "verify_and_reuse"
    assert verify_ledger(root) == []
    # The FIRST-written record keeps its original timestamp (no silent overwrite).
    loaded = load_manifest("dataset_abcdef012345", root=root)
    assert loaded["manifest"]["created_at"] == "2026-01-01T00:00:00+00:00"
    assert loaded["manifest"]["dataset_fingerprint"] == "f" * 64
    entries = list_entries(root)
    assert entries[0]["dataset_id"] == "dataset_abcdef012345"
    # A third identical persist is still a reuse, never an error.
    third = persist_manifest(_manifest(created_at="2026-03-03T00:00:00+00:00"), root=root)
    assert third["outcome"] == "verify_and_reuse"


def test_provenance_ledger_non_volatile_change_still_fails(tmp_path):
    """Only created_at is volatile; any other field change must still be refused."""
    root = str(tmp_path / "provenance")
    persist_manifest(_manifest(), root=root)
    for field, value in (
        ("config_fingerprint", "d" * 64),
        ("pit_status", "not_point_in_time"),
        ("row_count", 5),
        ("period_end", "2021-12-31"),
    ):
        changed = _manifest()
        changed[field] = value
        with pytest.raises(ImmutabilityError):
            persist_manifest(changed, root=root)


def test_provenance_ledger_different_content_fails(tmp_path):
    root = str(tmp_path / "provenance")
    persist_manifest(_manifest(), root=root)
    changed = _manifest()
    changed["dataset_fingerprint"] = "e" * 64
    with pytest.raises(ImmutabilityError):
        persist_manifest(changed, root=root)


# ── 12. real bounded price sample (certification run) ────────────────────────

def test_real_price_sample_certifies_and_is_survivorship_flagged(tmp_path):
    """Research-real sample: uses the live Yahoo path with a tiny bounded fetch.

    Skipped when the provider is unreachable so offline runs stay green; when it
    runs, it proves the sample persists as PARTIAL (not survivorship-safe).
    """
    from src.research.data import certification as cert
    from src.research.modes import ResearchMode

    tiny = load_sample_universe(cap=3)
    try:
        sec_map = cert.fetch_sec_tickers()
    except Exception as exc:  # provider unavailable offline
        pytest.skip("SEC/network unavailable: %s" % exc)
    master, _diag = cert.build_security_master(list(tiny.tickers) + [tiny.benchmark], sec_map)

    # Only fetch a couple of real names to keep the network bounded.
    probe = tiny.tickers[:2]
    from src.research.data.prices import fetch_chart_payload, parse_yahoo_chart

    charts = {}
    for ticker in list(probe) + [tiny.benchmark]:
        try:
            charts[ticker] = fetch_chart_payload(ticker, range_="1y")
        except Exception as exc:
            pytest.skip("price provider unavailable: %s" % exc)
    clients = {"charts": charts, "company_tickers": {str(k + 1): v for k, v in enumerate([])}}
    root = str(tmp_path / "research_v2")
    manifest, status = cert.certify_prices(tiny, master, root, ResearchMode.RESEARCH_V2, clients=clients, range_="1y")
    assert status.status == "PARTIAL", "survivorship limitation must keep prices PARTIAL"
    assert status.survivorship_safe is False
    assert status.evidence["observations"] > 0
    # The persisted gold table must expose the PIT column and NOT treat the
    # provider adjusted close as PIT-safe.
    from src.research.data import layers
    gold = layers.read_silver_table(root, "prices_gold_pit")
    assert "adjusted_close_pit" in gold.columns
    assert "adjusted_close_provider_NOT_PIT" in gold.columns
