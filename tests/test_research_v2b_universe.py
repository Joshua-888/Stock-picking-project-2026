"""Tests for the WP2B historical-universe / delisting unblock work package.

Every fixture here is small, explicitly synthetic data used only to exercise the
integrity logic. None of it is research evidence and nothing is persisted as a
certified dataset. No test performs network I/O: credentialed adapters are
exercised only through injected transport, and the credential gate is tested by
asserting the error is raised *before* any transport call is made.

This module performs no feature engineering, no model fitting and no scoring.
"""

import pandas as pd
import pytest

from src.research.data import providers as prov
from src.research.data.providers import (
    ProviderCredentialError,
    ProviderNotConfiguredError,
    ProviderUnsupportedError,
    env_present,
    get_provider,
    provider_info,
    require_env,
)
from src.research.data.providers import adapters as ad
from src.research.data.membership_engine import (
    MembershipError,
    MembershipEvent,
    build_pit_universe,
    events_from_frame,
    membership_at,
    rebuild_membership_windows,
    validate_events,
)
from src.research.data.delisting import (
    BANKRUPTCY,
    CASH_ACQUISITION,
    CASH_SETTLED_KINDS,
    DELISTING_KINDS,
    MISSING_DELISTING_RETURN_BIAS,
    MISSING_FINAL_PRICE,
    STOCK_ACQUISITION,
    DelistingError,
    DelistingEvent,
    classify_delisting,
    terminal_return,
    validate_delistings,
)
from src.research.data.corporate_actions import (
    ALL_KINDS,
    BANKRUPTCY as CA_BANKRUPTCY,
    CASH_ACQUISITION as CA_CASH,
    SPINOFF,
    TICKER_RENAME,
    CorporateActionError,
    CorporateActionEvent,
    event_set_unchanged_by_future,
    events_known_by,
    pit_total_return,
    price_actions_from_events,
)
from src.research.data.identity_chain import (
    IdentityChain,
    IdentityChainError,
    SecurityIdentity,
    SecurityRelationship,
    SHARE_CLASS_PRIMARY,
    VendorReference,
    security_id_for_share_class,
)
from src.research.data.security_master import AmbiguousTickerError, UnknownTickerError, security_id_for
from src.research.data.survivorship_guard import (
    SurvivorshipViolation,
    assert_survivorship_clean,
    check_constituent_retained_across_history,
    check_historical_member_has_prices,
    check_no_pre_effective_membership,
    check_universe_has_exits,
    run_guardrails,
)
from src.research.data.universe import UniverseMembership, UniverseTable

SYNTHETIC = "synthetic"  # every fixture below carries this label


# ── Helpers ───────────────────────────────────────────────────────────────────

def _raw_prices(ticker="AAA", days=None, start=100.0, step=1.0):
    days = days or ["2018-01-02", "2018-01-03", "2018-01-04", "2018-06-01", "2018-06-04"]
    rows = []
    price = start
    for day in days:
        rows.append({"ticker": ticker, "trade_date": day, "raw_close": price})
        price += step
    return pd.DataFrame(rows)


# ── Provider abstraction: credential gate before network ──────────────────────

def test_env_present_and_require_env(monkeypatch):
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    assert env_present(("EODHD_API_TOKEN",)) is None
    with pytest.raises(ProviderCredentialError):
        require_env("eodhd", ("EODHD_API_TOKEN",))
    monkeypatch.setenv("EODHD_API_TOKEN", "  secret  ")
    assert env_present(("EODHD_API_TOKEN",)) == "EODHD_API_TOKEN"
    name, value = require_env("eodhd", ("EODHD_API_TOKEN",))
    assert name == "EODHD_API_TOKEN" and value == "secret"


def test_credential_error_precedes_network(monkeypatch):
    """A missing key must raise BEFORE any transport call is attempted."""
    calls = []

    def tripwire(url, timeout=30):
        calls.append(url)
        raise AssertionError("transport must not be called without a credential")

    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    provider = ad.EodhdProvider(get=tripwire)
    with pytest.raises(ProviderCredentialError):
        provider.fetch_delisted_prices("sec_x", "AAA", "2018-01-01", "2018-02-01")
    assert calls == []

    monkeypatch.delenv("NASDAQ_DATA_LINK_API_KEY", raising=False)
    monkeypatch.delenv("QUANDL_API_KEY", raising=False)
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    sharadar = ad.SharadarNdlProvider(get=tripwire)
    with pytest.raises(ProviderCredentialError):
        sharadar.fetch_membership_events("sp500", "2018-01-01", "2018-02-01")
    assert calls == []


def test_provider_registry_lists_capabilities():
    ids = sorted(item.provider_id for item in provider_info())
    assert ids == ["eodhd", "sharadar_ndl", "wikipedia_sp500"]
    by_id = {item.provider_id: item for item in provider_info()}
    assert "historical_universe" in by_id["sharadar_ndl"].capabilities
    assert isinstance(by_id["eodhd"].to_dict()["configured"], bool)
    assert by_id["eodhd"].to_dict()["credential_env"] == ["EODHD_API_TOKEN"]
    with pytest.raises(ProviderNotConfiguredError):
        get_provider("does_not_exist")


def test_eodhd_normalizes_payload(monkeypatch):
    monkeypatch.setenv("EODHD_API_TOKEN", "secret")
    payload = '[{"date":"2018-01-02","close":10.0},{"date":"2018-01-03","close":11.0}]'
    provider = ad.EodhdProvider(get=lambda url, timeout=30: payload)
    frame = provider.fetch_delisted_prices("sec_aaa", "AAA", "2018-01-01", "2018-02-01")
    assert list(frame.columns) == list(prov.DELISTED_PRICE_FIELDS)
    assert frame["raw_close"].tolist() == [10.0, 11.0]
    assert set(frame["security_id"]) == {"sec_aaa"}
    url = provider._url("eod/AAA")
    assert "api_token=secret" in url


def test_eodhd_membership_unsupported():
    provider = ad.EodhdProvider(get=lambda url, timeout=30: "[]")
    with pytest.raises(ProviderUnsupportedError):
        provider.fetch_membership_events("sp500", "2018-01-01", "2018-12-31")


def test_sharadar_daily_members_to_events():
    rows = [
        {"date": "2018-01-02", "ticker": "AAA"},
        {"date": "2018-01-02", "ticker": "BBB"},
        {"date": "2018-01-03", "ticker": "AAA"},
        {"date": "2018-01-03", "ticker": "CCC"},
    ]
    events = ad.events_from_daily_members(pd.DataFrame(rows), "sp500", "sharadar_ndl:SP500")
    adds = sorted(events.loc[events["action"] == "add", "security_id"])
    removes = sorted(events.loc[events["action"] == "remove", "security_id"])
    assert adds == ["AAA", "BBB", "CCC"]
    assert removes == ["BBB"]
    assert events.attrs["announcement_inferred"] is True


def test_wikipedia_parse_membership_only():
    html = (
        "<table><tr><th>Date</th><th>Added</th><th>Company</th><th>Removed</th></tr>"
        "<tr><td>2018-06-01</td><td>NEW</td><td>New Co</td><td>OLD</td></tr></table>"
    )
    frame = ad.parse_wikipedia_sp500_changes(html, universe_id="sp500")
    actions = sorted(frame["action"])
    assert actions == ["add", "remove"]
    assert set(frame["effective_date"]) == {"2018-06-01"}
    assert frame["announcement_date"].isna().all()


def test_wikipedia_requires_no_credential():
    provider = ad.WikipediaSp500Provider(
        get=lambda url, timeout=30: "<table><tr><td>2018-06-01</td><td>NEW</td><td>New Co</td><td></td></tr></table>"
    )
    frame = provider.fetch_membership_events(start="2018-01-01", end="2018-12-31")
    assert frame["security_id"].tolist() == ["NEW"]
    with pytest.raises(ProviderUnsupportedError):
        provider.fetch_delisted_prices("sec", "NEW", "2018-01-01", "2018-12-31")


# ── Membership engine: PIT reconstruction + look-ahead test ───────────────────

def _events():
    return [
        MembershipEvent(security_id="AAA", universe_id="sp500", action="add",
                        announcement_date="2018-05-15", effective_date="2018-06-01",
                        source="synthetic", source_reference="fixture"),
        MembershipEvent(security_id="BBB", universe_id="sp500", action="remove",
                        announcement_date="2018-05-15", effective_date="2018-06-01",
                        source="synthetic", source_reference="fixture"),
        MembershipEvent(security_id="BBB", universe_id="sp500", action="add",
                        announcement_date="2017-01-01", effective_date="2017-01-15",
                        source="synthetic", source_reference="fixture"),
    ]


def test_membership_add_is_not_effective_before_effective_date():
    events = _events()
    assert membership_at(events, "AAA", "sp500", "2018-05-31") is False
    assert membership_at(events, "AAA", "sp500", "2018-06-01") is True
    assert membership_at(events, "AAA", "sp500", "2018-06-15") is True


def test_membership_remove_stops_on_effective_date():
    events = _events()
    assert membership_at(events, "BBB", "sp500", "2017-02-01") is True
    assert membership_at(events, "BBB", "sp500", "2018-05-31") is True
    assert membership_at(events, "BBB", "sp500", "2018-06-01") is False


def test_rebuild_windows_and_table():
    events = _events()
    memberships, problems = rebuild_membership_windows(events, "sp500")
    assert problems == []
    by_id = {item.security_id: item for item in memberships}
    assert by_id["AAA"].membership_start == "2018-06-01"
    assert by_id["BBB"].membership_start == "2017-01-15"
    assert by_id["BBB"].membership_end == "2018-06-01"
    table = build_pit_universe(events, "sp500")
    assert table.is_member("AAA", "2018-06-01", available_only=False)
    assert not table.is_member("AAA", "2018-05-31", available_only=False)


def test_available_only_blocks_late_known_membership():
    """Announcement date gates availability: a late revision is not usable early."""
    events = [
        MembershipEvent(security_id="AAA", universe_id="sp500", action="add",
                        announcement_date="2018-07-01", effective_date="2018-06-01"),
    ]
    table = build_pit_universe(events, "sp500")
    assert table.is_member("AAA", "2018-06-15", available_only=False)
    assert not table.is_member("AAA", "2018-06-15", available_only=True)
    assert table.is_member("AAA", "2018-07-02", available_only=True)


def test_remove_without_add_is_flagged():
    events = [
        MembershipEvent(security_id="ZZZ", universe_id="sp500", action="remove",
                        announcement_date="2018-05-01", effective_date="2018-06-01"),
    ]
    _memberships, problems = rebuild_membership_windows(events, "sp500")
    assert any("no preceding add" in problem for problem in problems)
    with pytest.raises(MembershipError):
        build_pit_universe(events, "sp500", strict=True)
    table = build_pit_universe(events, "sp500", strict=False)
    assert table.reconstruction_problems


def test_validate_events_flags_missing_and_merged_dates():
    events = [
        MembershipEvent(security_id="AAA", universe_id="sp500", action="add", effective_date="2018-06-01"),
        MembershipEvent(security_id="BBB", universe_id="sp500", action="add",
                        announcement_date="2018-06-01", effective_date="2018-06-01"),
    ]
    problems = validate_events(events)
    assert any("no announcement date" in problem for problem in problems)
    assert any("merged/inferred" in problem for problem in problems)


def test_events_from_frame_roundtrip():
    frame = pd.DataFrame([
        {"security_id": "AAA", "universe_id": "sp500", "action": "add",
         "announcement_date": "2018-05-15", "effective_date": "2018-06-01",
         "source": "x", "source_reference": "y"},
    ])
    events = events_from_frame(frame)
    assert events[0].security_id == "AAA" and events[0].action == "add"


def test_membership_event_rejects_bad_action():
    with pytest.raises(MembershipError):
        MembershipEvent(security_id="AAA", universe_id="sp500", action="promote", effective_date="2018-06-01")


def test_membership_event_rejects_unparseable_date():
    with pytest.raises(MembershipError):
        MembershipEvent(security_id="AAA", universe_id="sp500", action="add", effective_date="not-a-date")


# ── Delisting methodology ─────────────────────────────────────────────────────

def test_missing_final_price_is_unknown_not_zero():
    event = DelistingEvent(security_id="sec_bbb", ticker="BBB", kind=MISSING_FINAL_PRICE,
                           effective_date="2018-06-01", source=SYNTHETIC)
    value, source = terminal_return(10.0, event)
    assert value is None and source == "unknown"
    assert event.return_is_known is False
    assert event.return_unknown_reason
    assert value != 0.0


def test_vendor_delisting_return_is_preserved():
    event = DelistingEvent(security_id="sec_bbb", ticker="BBB", kind=BANKRUPTCY,
                           effective_date="2018-06-01", delisting_return=-0.85, source=SYNTHETIC)
    value, source = terminal_return(10.0, event)
    assert value == pytest.approx(-0.85) and source == "vendor"


def test_cash_acquisition_uses_observable_price_only():
    event = DelistingEvent(security_id="sec_ccc", ticker="CCC", kind=CASH_ACQUISITION,
                           effective_date="2019-03-01", final_price=45.0, source=SYNTHETIC)
    value, source = terminal_return(40.0, event)
    assert value == pytest.approx(0.125) and source == "cash_price"
    bare = DelistingEvent(security_id="sec_ccc", ticker="CCC", kind=CASH_ACQUISITION,
                          effective_date="2019-03-01", source=SYNTHETIC)
    value2, source2 = terminal_return(40.0, bare)
    assert value2 is None and source2 == "unknown"


def test_stock_acquisition_is_not_cash_settled():
    event = DelistingEvent(security_id="sec_ddd", ticker="DDD", kind=STOCK_ACQUISITION,
                           effective_date="2019-03-01", final_price=50.0, source=SYNTHETIC)
    value, source = terminal_return(40.0, event)
    assert value is None and source == "unknown"
    assert STOCK_ACQUISITION not in CASH_SETTLED_KINDS


def test_classify_delisting_conservative():
    assert classify_delisting("Chapter 11 filing") == BANKRUPTCY
    assert classify_delisting("acquired by XYZ") == CASH_ACQUISITION
    assert classify_delisting("acquired via stock swap") == STOCK_ACQUISITION
    assert classify_delisting("unrecognized reason") == MISSING_FINAL_PRICE
    with pytest.raises(DelistingError):
        classify_delisting(kind="teleport")


def test_delisting_event_rejects_bad_kind():
    with pytest.raises(DelistingError):
        DelistingEvent(security_id="sec", ticker="X", kind="wormhole", effective_date="2018-01-01")


def test_validate_delistings_detects_inconsistency():
    event = DelistingEvent(security_id="sec", ticker="X", kind=BANKRUPTCY,
                           effective_date="2018-01-01", last_trade_date="2018-06-01")
    problems = validate_delistings([event, event])
    assert any("duplicate delisting" in problem for problem in problems)
    assert any("precedes last trade" in problem for problem in problems)


def test_missing_return_bias_documented():
    assert "UPWARD" in MISSING_DELISTING_RETURN_BIAS
    assert len(DELISTING_KINDS) == 8


# ── Corporate-action PIT extension ────────────────────────────────────────────

def test_only_price_actions_reach_pit_prices():
    events = [
        CorporateActionEvent(ticker="AAA", kind="split", effective_date="2018-06-01", numerator=2, denominator=1),
        CorporateActionEvent(ticker="AAA", kind=CA_BANKRUPTCY, effective_date="2019-01-01"),
        CorporateActionEvent(ticker="AAA", kind=TICKER_RENAME, effective_date="2019-02-01"),
    ]
    price_actions = price_actions_from_events(events)
    assert len(price_actions) == 1 and price_actions[0].kind == "split"


def test_future_action_is_invisible_at_earlier_asof():
    events = [CorporateActionEvent(ticker="AAA", kind=SPINOFF, effective_date="2018-03-01")]
    future = CorporateActionEvent(ticker="AAA", kind=CA_CASH, effective_date="2019-01-01")
    assert event_set_unchanged_by_future(events, "2018-06-01", future) is True
    with pytest.raises(CorporateActionError):
        event_set_unchanged_by_future(events, "2019-06-01", future)


def test_events_known_by_excludes_future():
    events = [
        CorporateActionEvent(ticker="AAA", kind="dividend", effective_date="2018-02-01", amount=1.0),
        CorporateActionEvent(ticker="AAA", kind="dividend", effective_date="2018-08-01", amount=1.0),
    ]
    known = events_known_by(events, "2018-06-01")
    assert [event.effective_date for event in known] == ["2018-02-01"]


def test_pit_total_return_ignores_informational_events():
    raw = _raw_prices(days=["2018-01-02", "2018-01-03"])
    events = [CorporateActionEvent(ticker="AAA", kind=CA_BANKRUPTCY, effective_date="2018-01-03")]
    ret = pit_total_return(raw, events, "2018-01-02", "2018-01-03")
    assert ret == pytest.approx(0.01)


def test_split_factor_applied_only_after_effective():
    raw = _raw_prices(days=["2018-01-02", "2018-01-03", "2018-01-04"])
    events = [CorporateActionEvent(ticker="AAA", kind="split", effective_date="2018-01-04", numerator=2, denominator=1)]
    ret = pit_total_return(raw, events, "2018-01-02", "2018-01-03")
    assert ret == pytest.approx(0.01)
    ret2 = pit_total_return(raw, events, "2018-01-02", "2018-01-04")
    assert ret2 is not None and ret2 != pytest.approx(0.02)


def test_corporate_action_validation():
    with pytest.raises(CorporateActionError):
        CorporateActionEvent(ticker="AAA", kind="split", effective_date="2018-01-01", numerator=2, denominator=0)
    with pytest.raises(CorporateActionError):
        CorporateActionEvent(ticker="AAA", kind="dividend", effective_date="2018-01-01")
    assert set(ALL_KINDS) >= {SPINOFF, TICKER_RENAME, CA_BANKRUPTCY, CA_CASH}


# ── Identity chain ────────────────────────────────────────────────────────────

def _chain():
    chain = IdentityChain()
    primary = SecurityIdentity(security_id=security_id_for_share_class(cik="0000320193"),
                               name="Apple Inc", cik="0000320193")
    chain.add_identity(primary)
    return chain, primary


def test_vendor_reference_resolution_and_ambiguity():
    chain, primary = _chain()
    chain.add_vendor_reference(VendorReference(vendor="eodhd", vendor_security_id="AAPL.US",
                                               security_id=primary.security_id,
                                               effective_from="2010-01-01", effective_to="2019-12-31"))
    assert chain.resolve_vendor("eodhd", "AAPL.US", as_of="2015-01-01") == primary.security_id

    other = SecurityIdentity(security_id=security_id_for(issuer_key="OTHER"), name="Other Co", issuer_key="OTHER")
    chain.add_identity(other)
    chain.add_vendor_reference(VendorReference(vendor="eodhd", vendor_security_id="AAPL.US",
                                               security_id=other.security_id,
                                               effective_from="2020-01-01"))
    assert chain.resolve_vendor("eodhd", "AAPL.US", as_of="2021-01-01") == other.security_id
    # Ambiguous when no as-of disambiguates the reused vendor id.
    with pytest.raises(AmbiguousTickerError):
        chain.resolve_vendor("eodhd", "AAPL.US")
    # Unknown vendor id fails loudly, never guessed.
    with pytest.raises(UnknownTickerError):
        chain.resolve_vendor("eodhd", "NOSUCH", as_of="2015-01-01")
    # A date before either mapping window resolves to nothing.
    with pytest.raises(UnknownTickerError):
        chain.resolve_vendor("eodhd", "AAPL.US", as_of="2005-01-01")


def test_share_classes_get_distinct_ids():
    primary = security_id_for_share_class(cik="0001652044", share_class=SHARE_CLASS_PRIMARY)
    class_c = security_id_for_share_class(cik="0001652044", share_class="class_c")
    assert primary != class_c
    assert primary == security_id_for(cik="0001652044")


def test_share_class_collision_is_rejected_by_identity_chain():
    chain = IdentityChain()
    sid = security_id_for(cik="0001652044")
    chain.add_identity(SecurityIdentity(security_id=sid, name="A", cik="0001652044", share_class="primary"))
    # Two economically distinct share classes must NOT share an id.
    with pytest.raises(IdentityChainError):
        chain.add_identity(SecurityIdentity(security_id=sid, name="A", cik="0001652044", share_class="class_c"))
    assert chain.share_class_collisions() == {}


def test_issuer_merger_keeps_securities_distinct():
    chain = IdentityChain()
    target = SecurityIdentity(security_id=security_id_for(cik="0000000001"), name="Target", cik="0000000001")
    acquirer = SecurityIdentity(security_id=security_id_for(cik="0000000002"), name="Acquirer", cik="0000000002")
    chain.add_identity(target)
    chain.add_identity(acquirer)
    assert target.security_id != acquirer.security_id
    chain.add_relationship(SecurityRelationship(kind="merged_into", predecessor_id=target.security_id,
                                                successor_id=acquirer.security_id, effective_date="2019-01-01"))
    successors = chain.successors(target.security_id)
    assert successors and successors[0]["successor_id"] == acquirer.security_id
    with pytest.raises(IdentityChainError):
        chain.add_relationship(SecurityRelationship(kind="merged_into", predecessor_id=target.security_id,
                                                     successor_id=target.security_id, effective_date="2019-01-01"))


def test_relationship_unknown_security_fails():
    chain = IdentityChain()
    with pytest.raises(IdentityChainError):
        chain.add_relationship(SecurityRelationship(kind="merged_into", predecessor_id="sec_x",
                                                     successor_id="sec_y", effective_date="2019-01-01"))


def test_identity_chain_frame_and_ticker_history():
    chain, primary = _chain()
    chain.add_ticker({"security_id": primary.security_id, "ticker": "AAPL", "exchange": "NASDAQ",
                      "valid_from": "1980-12-12"})
    frame = chain.to_frame()
    assert set(["security_id", "ticker", "exchange"]).issubset(frame.columns)
    assert frame.iloc[0]["ticker"] == "AAPL"


def test_vendor_reference_unknown_security_fails():
    chain = IdentityChain()
    with pytest.raises(IdentityChainError):
        chain.add_vendor_reference(VendorReference(vendor="v", vendor_security_id="x", security_id="sec_missing"))


# ── Survivorship adversarial guardrails ───────────────────────────────────────

def _table_with_exit():
    table = UniverseTable("sp500")
    table.add(UniverseMembership(universe_id="sp500", security_id="sec_aaa", ticker="AAA",
                                 membership_start="2017-01-01"))
    table.add(UniverseMembership(universe_id="sp500", security_id="sec_bbb", ticker="BBB",
                                 membership_start="2017-01-01", membership_end="2018-06-01"))
    return table


def test_guardrail_flags_present_day_only_universe():
    table = UniverseTable("sp500")
    table.add(UniverseMembership(universe_id="sp500", security_id="sec_aaa", ticker="AAA",
                                 membership_start="2017-01-01"))
    finding = check_universe_has_exits(table)
    assert finding is not None and finding.check == "universe_has_exits"
    with pytest.raises(SurvivorshipViolation):
        assert_survivorship_clean(table)


def test_guardrail_clean_table_passes():
    table = _table_with_exit()
    assert check_universe_has_exits(table) is None
    assert run_guardrails(table) == []
    assert assert_survivorship_clean(table) is True


def test_guardrail_flags_entrant_leaking_backwards():
    # A documented join effective 2019-06-01 must NOT be a member earlier.
    good = UniverseTable("sp500")
    good.add(UniverseMembership(universe_id="sp500", security_id="sec_new", ticker="NEW",
                                membership_start="2019-06-01", membership_end="2021-01-01"))
    assert check_no_pre_effective_membership(good, "sec_new", "2019-06-01") is None

    # Broken table: the security appears as a member before its documented join.
    bad = UniverseTable("sp500")
    bad.add(UniverseMembership(universe_id="sp500", security_id="sec_new", ticker="NEW",
                               membership_start="2018-06-01", membership_end="2021-01-01"))
    finding = check_no_pre_effective_membership(bad, "sec_new", "2019-06-01")
    assert finding is not None and finding.check == "no_pre_effective_membership"
    findings = run_guardrails(bad, known_entrants=[("sec_new", "2019-06-01")])
    assert any(item.check == "no_pre_effective_membership" for item in findings)


def test_guardrail_flags_erased_constituent():
    table = _table_with_exit()
    assert check_constituent_retained_across_history(table, "sec_bbb", ["2017-06-01", "2018-01-01"]) is None
    finding = check_constituent_retained_across_history(table, "sec_bbb", ["2019-01-01", "2020-01-01"])
    assert finding is not None and finding.check == "constituent_retained_across_history"


def test_guardrail_flags_delisted_member_without_prices():
    table = _table_with_exit()
    empty = pd.DataFrame(columns=["ticker", "trade_date", "raw_close"])
    finding = check_historical_member_has_prices(table, "sec_bbb", empty, "2018-01-01")
    assert finding is not None and finding.check == "historical_member_has_prices"
    good = pd.DataFrame([{"ticker": "BBB", "trade_date": "2018-01-02", "raw_close": 12.0}])
    assert check_historical_member_has_prices(table, "sec_bbb", good, "2018-01-01") is None
    # A non-member is irrelevant to price coverage.
    assert check_historical_member_has_prices(table, "sec_zzz", empty, "2018-01-01") is None


def test_run_guardrails_reports_price_gap():
    table = _table_with_exit()
    empty = pd.DataFrame(columns=["ticker", "trade_date", "raw_close"])
    findings = run_guardrails(table, price_frame=empty, as_of="2018-01-01")
    assert any(item.check == "historical_member_has_prices" for item in findings)


# ── Reverse split / special dividend (executed, not claimed) ──────────────────

def test_reverse_split_back_adjusts_pre_effective_prices():
    """A 1-for-5 reverse split re-bases pre-split prices by 5x, PIT-correctly."""
    raw = pd.DataFrame([
        {"ticker": "AAA", "trade_date": "2018-05-31", "raw_close": 10.0},
        {"ticker": "AAA", "trade_date": "2018-06-01", "raw_close": 52.0},
    ])
    reverse = CorporateActionEvent(ticker="AAA", kind="split", effective_date="2018-06-01",
                                   numerator=1, denominator=5)
    assert reverse.adjusts_price
    naive = 52.0 / 10.0 - 1.0
    ret = pit_total_return(raw, [reverse], "2018-05-31", "2018-06-01")
    assert ret == pytest.approx(0.04)
    assert ret != pytest.approx(naive)

    later = CorporateActionEvent(ticker="AAA", kind="split", effective_date="2018-06-02",
                                 numerator=1, denominator=5)
    # A reverse split effective AFTER the window end enters no factor.
    unadjusted = pit_total_return(raw, [later], "2018-05-31", "2018-06-01")
    assert unadjusted == pytest.approx(naive)


def test_special_dividend_adjusts_like_a_large_dividend():
    """A special dividend is a price-adjusting dividend processed PIT-correctly.

    Dividend factor is (close_on_ex - amount) / close_on_ex applied to prices
    strictly BEFORE the effective date. With ex-date close 80 and amount 20 the
    factor is 0.75, so the prior close 100 becomes 75 -> 80/75-1, which differs
    from the naive unadjusted -20%. A dividend effective AFTER the window end is
    invisible and leaves the return unadjusted.
    """
    raw = pd.DataFrame([
        {"ticker": "AAA", "trade_date": "2018-05-31", "raw_close": 100.0},
        {"ticker": "AAA", "trade_date": "2018-06-01", "raw_close": 80.0},
    ])
    special = CorporateActionEvent(ticker="AAA", kind="dividend", effective_date="2018-06-01",
                                   amount=20.0)
    assert special.adjusts_price
    actions = price_actions_from_events([special])
    assert len(actions) == 1 and actions[0].kind == "dividend"

    naive = 80.0 / 100.0 - 1.0
    adjusted = pit_total_return(raw, [special], "2018-05-31", "2018-06-01")
    assert adjusted == pytest.approx((80.0 / 75.0) - 1.0)
    assert adjusted != pytest.approx(naive)

    later = CorporateActionEvent(ticker="AAA", kind="dividend", effective_date="2018-06-02",
                                 amount=20.0)
    unadjusted = pit_total_return(raw, [later], "2018-05-31", "2018-06-01")
    assert unadjusted == pytest.approx(naive)


# ── Membership boundary convention: both callers must agree ───────────────────

def test_membership_engine_and_universe_boundary_agree():
    """add: member on/after effective; remove: NOT member on/after effective."""
    events = [
        MembershipEvent(security_id="sec_add", universe_id="sp500", action="add",
                        announcement_date="2018-05-15", effective_date="2018-06-01"),
        MembershipEvent(security_id="sec_rem", universe_id="sp500", action="add",
                        announcement_date="2017-01-01", effective_date="2017-01-15"),
        MembershipEvent(security_id="sec_rem", universe_id="sp500", action="remove",
                        announcement_date="2018-05-15", effective_date="2018-06-01"),
    ]
    table = build_pit_universe(events, "sp500")
    for security_id in ("sec_add", "sec_rem"):
        for day in ("2018-05-31", "2018-06-01", "2018-06-02"):
            assert membership_at(events, security_id, "sp500", day) == table.is_member(
                security_id, day, available_only=False
            ), "engine and table disagree on %s at %s" % (security_id, day)
    assert membership_at(events, "sec_add", "sp500", "2018-06-01") is True
    assert table.is_member("sec_add", "2018-06-01", available_only=False) is True
    assert membership_at(events, "sec_rem", "sp500", "2018-06-01") is False
    assert table.is_member("sec_rem", "2018-06-01", available_only=False) is False
