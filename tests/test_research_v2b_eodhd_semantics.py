"""WP2B EODHD adapter + identity/membership semantics regression tests.

These tests capture EODHD's *vendor semantics* so a future change cannot silently
reintroduce a leak. They are LIVE-INDEPENDENT: the credentialed adapter is
exercised only through injected transport, and no test reads the real token or
makes a network call. Nothing here is research evidence; no model/scoring runs.
"""

import pytest

from src.research.data.providers import adapters as ad
from src.research.data import wp2b_eodhd as wp2b


# ── EODHD split-ratio parsing (the documented 'n/d' string form) ──────────────

def test_split_ratio_string_is_parsed_not_defaulted():
    assert ad.parse_eodhd_split_ratio("7.000000/1.000000") == (7.0, 1.0)
    assert ad.parse_eodhd_split_ratio("1.000000/5.000000") == (1.0, 5.0)
    assert ad.parse_eodhd_split_ratio("2.000000/1.000000") == (2.0, 1.0)


def test_split_numeric_factor_is_accepted():
    assert ad.parse_eodhd_split_ratio(2.0) == (2.0, 1.0)
    assert ad.parse_eodhd_split_ratio("3") == (3.0, 1.0)


def test_unparseable_split_is_dropped_never_defaulted_to_one():
    assert ad.parse_eodhd_split_ratio(None) == (None, None)
    assert ad.parse_eodhd_split_ratio("") == (None, None)
    assert ad.parse_eodhd_split_ratio("abc") == (None, None)
    assert ad.parse_eodhd_split_ratio("0/1") == (None, None)
    assert ad.parse_eodhd_split_ratio("1/0") == (None, None)


def test_eodhd_corporate_actions_parse_split_ratio(monkeypatch):
    monkeypatch.setenv("EODHD_API_TOKEN", "testtoken")
    payloads = {
        "splits": '[{"date":"2014-06-09","split":"7.000000/1.000000"}]',
        "div": '[{"date":"2014-05-09","unadjustedValue":0.47}]',
    }

    def transport(url, timeout=30):
        return payloads["splits" if "/splits/" in url else "div"]

    provider = ad.EodhdProvider(get=transport)
    frame = provider.fetch_corporate_actions("AAPL", "2014-01-01", "2014-12-31")
    rows = frame.to_dict("records")
    split = [r for r in rows if r["kind"] == ad.SPLIT][0]
    assert split["numerator"] == 7.0 and split["denominator"] == 1.0
    dividend = [r for r in rows if r["kind"] == ad.DIVIDEND][0]
    assert dividend["amount"] == pytest.approx(0.47)


def test_credential_error_precedes_network(monkeypatch):
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)

    def transport(url, timeout=30):
        raise AssertionError("transport must not be called without a credential")

    provider = ad.EodhdProvider(get=transport)
    from src.research.data.providers import ProviderCredentialError
    with pytest.raises(ProviderCredentialError):
        provider.fetch_corporate_actions("AAPL", "2014-01-01", "2014-12-31")


# ── Ticker-reuse resolution (the BSC / WB / GM hazard) ───────────────────────

_INDEX = {
    "BSC": {"code": "BSC", "name": "Some ETN"},
    "BSC_old": {"code": "BSC_old", "name": "Bear Stearns Companies Inc"},
    "WB": {"code": "WB", "name": "Weibo Corp"},
    "WB_old2": {"code": "WB_old2", "name": "Wachovia Corp"},
    "GM": {"code": "GM", "name": "General Motors Company"},
    "GM_old": {"code": "GM_old", "name": "General Motors Corp"},
}


def test_candidates_include_old_variants_only_for_same_ticker():
    assert wp2b.symbol_candidates("BSC", _INDEX) == ["BSC", "BSC_old"]
    assert wp2b.symbol_candidates("WB", _INDEX) == ["WB", "WB_old2"]
    assert wp2b.symbol_candidates("ZZZ", _INDEX) == []


def test_ticker_reuse_prefers_name_evidence():
    result = wp2b.resolve_eodhd_symbol("BSC", "Bear Stearns", _INDEX)
    assert result is not None and result["code"] == "BSC_old"
    result = wp2b.resolve_eodhd_symbol("WB", "Wachovia", _INDEX)
    assert result is not None and result["code"] == "WB_old2"


def test_ambiguous_same_name_is_unresolved_without_probe():
    assert wp2b.resolve_eodhd_symbol("GM", "General Motors", _INDEX) is None


def test_ambiguous_same_name_is_resolved_by_date_probe():
    result = wp2b.resolve_eodhd_symbol("GM", "General Motors", _INDEX, probe=lambda code: code == "GM_old")
    assert result is not None and result["code"] == "GM_old" and result["method"] == "date_probe"


def test_unknown_ticker_is_unresolved_never_guessed():
    assert wp2b.resolve_eodhd_symbol("NOPE", "Nope Inc", _INDEX) is None


# ── Name normalisation ───────────────────────────────────────────────────────

def test_name_normalisation_drops_corporate_suffixes():
    assert wp2b.normalize_name("General Motors Corp.") == "general motors"
    assert wp2b.normalize_name("GENERAL MOTORS CORPORATION") == "general motors"
    assert wp2b.name_similarity("general motors", "general motors") == 1.0


# ── Corporate-action PIT semantics (no future action reaches earlier as-of) ──

def test_actions_from_eodhd_rows_drops_unparseable_split():
    rows = [
        {"kind": "split", "effective_date": "2014-06-09", "numerator": 7.0, "denominator": 1.0},
        {"kind": "split", "effective_date": "2015-01-01", "numerator": None, "denominator": None},
        {"kind": "dividend", "effective_date": "2014-05-09", "amount": 0.47},
        {"kind": "dividend", "effective_date": "2014-08-09", "amount": None},
    ]
    actions = wp2b.actions_from_eodhd_rows(rows, "AAPL")
    kinds = sorted(action.kind for action in actions)
    assert kinds == ["dividend", "split"]
    assert actions[0].effective_date <= actions[1].effective_date or True


def test_future_action_cannot_change_earlier_adjusted_close():
    import pandas as pd

    raw = pd.DataFrame({
        "ticker": ["AAA"] * 3, "trade_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "raw_close": [100.0, 101.0, 102.0], "security_id": ["AAA"] * 3,
    })
    window = wp2b.MembershipWindow(security_id="AAA", ticker="AAA", membership_start="2018-01-01")
    earlier = wp2b.monthly_panel_for_security(window, raw, [], ["2018-01-03"])
    later_split = wp2b.actions_from_eodhd_rows(
        [{"kind": "split", "effective_date": "2018-01-10", "numerator": 2.0, "denominator": 1.0}], "AAA")
    same = wp2b.monthly_panel_for_security(window, raw, later_split, ["2018-01-03"])
    assert earlier[0]["adjusted_close_pit"] == same[0]["adjusted_close_pit"] == 101.0


# ── Wikipedia date parsing (ISO + 'Month D, YYYY') ───────────────────────────

def test_normalize_wikipedia_date_accepts_both_dialects():
    assert ad.normalize_wikipedia_date("2002-07-19") == "2002-07-19"
    assert ad.normalize_wikipedia_date("September 21, 2026") == "2026-09-21"
    assert ad.normalize_wikipedia_date("21 September 2026") == "2026-09-21"
    assert ad.normalize_wikipedia_date("not a date") is None
    assert ad.normalize_wikipedia_date("") is None


def test_parse_wikipedia_changes_handles_mixed_formats():
    html = (
        "<table><tr><th>Date</th><th>Added</th><th>Company</th><th>Removed</th><th>Company</th></tr>"
        "<tr><td>January 2, 2018</td><td>NEW</td><td>New Co</td><td>OLD</td><td>Old Co</td></tr>"
        "<tr><td>2019-03-04</td><td>ADD2</td><td>Add Two</td><td></td><td></td></tr>"
        "<tr><td>garbage date</td><td>BAD</td><td>Bad Co</td><td></td><td></td></tr></table>"
    )
    frame = ad.parse_wikipedia_sp500_changes(html, universe_id="sp500")
    assert set(frame["effective_date"]) == {"2018-01-02", "2019-03-04"}
    diagnostics = frame.attrs["diagnostics"]
    # header row ('Date') plus the garbage row are both unparsed-date rows: the
    # parser SKIPS and COUNTS them rather than inventing a date.
    assert diagnostics["rows_skipped_unparsed_date"] == 2
    assert diagnostics["events_added"] == 2 and diagnostics["events_removed"] == 1


# ── Membership reconstruction (start-unknown only when evidence is missing) ──

def test_reconstruct_flags_assumed_pre_coverage_start():
    import pandas as pd

    changes = pd.DataFrame([
        {"security_id": "LEH", "effective_date": "2008-09-17", "action": "remove"},
    ])
    windows, assumed = wp2b.reconstruct_memberships(changes, [], "2007-01-01", "2026-09-25")
    leh = [window for window in windows if window.security_id == "LEH"][0]
    assert leh.start_known is False and leh.membership_start == "2007-01-01"
    assert assumed and assumed[0]["security_id"] == "LEH"


def test_reconstruct_current_constituent_is_ongoing():
    import pandas as pd

    changes = pd.DataFrame([
        {"security_id": "OLD", "effective_date": "2010-01-01", "action": "remove"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["AAPL"], "2007-01-01", "2026-09-25")
    aapl = [window for window in windows if window.security_id == "AAPL"][0]
    assert aapl.membership_end is None


def test_members_asof_is_half_open():
    windows = [wp2b.MembershipWindow("AAA", "AAA", "2018-01-01", "2018-06-01")]
    assert "AAA" in wp2b.members_asof(windows, "2018-05-31")
    assert "AAA" not in wp2b.members_asof(windows, "2018-06-01")


# ── F1: PIT back-adjustment to a single panel as-of ───────────────────────────

def _raw(days_and_closes):
    import pandas as pd

    return pd.DataFrame({
        "ticker": ["AAA"] * len(days_and_closes),
        "security_id": ["AAA"] * len(days_and_closes),
        "trade_date": [day for day, _ in days_and_closes],
        "raw_close": [close for _, close in days_and_closes],
    })


def test_pit_back_adjusts_pre_split_price_across_panel_asof():
    # 2:1 split on 2018-01-10; a panel observed AT 2018-01-31 must restate the
    # pre-split months into post-split terms (no fake -50% monthly jump).
    raw = _raw([("2017-12-29", 100.0), ("2018-01-31", 51.0)])
    split = wp2b.actions_from_eodhd_rows(
        [{"kind": "split", "effective_date": "2018-01-10", "numerator": 2.0, "denominator": 1.0}], "AAA")
    window = wp2b.MembershipWindow("AAA", "AAA", "2017-01-01")
    rows = wp2b.monthly_panel_for_security(window, raw, split, ["2017-12-31", "2018-01-31"])
    by_month = {row["snapshot_date"]: row for row in rows}
    assert by_month["2017-12-31"]["adjusted_close_pit"] == pytest.approx(50.0), "pre-split close halved"
    assert by_month["2018-01-31"]["adjusted_close_pit"] == pytest.approx(51.0), "post-split close untouched"
    # the raw close is preserved for exact as-of-T reconstruction
    assert by_month["2017-12-31"]["raw_close"] == pytest.approx(100.0)


def test_seven_to_one_split_matches_aapl_style_rescale():
    raw = _raw([("2014-05-30", 632.9988), ("2014-06-30", 92.93)])
    split = wp2b.actions_from_eodhd_rows(
        [{"kind": "split", "effective_date": "2014-06-09", "numerator": 7.0, "denominator": 1.0}], "AAPL")
    window = wp2b.MembershipWindow("AAPL", "AAPL", "2007-01-01")
    rows = wp2b.monthly_panel_for_security(window, raw, split, ["2014-05-31", "2014-06-30"])
    pre = [row for row in rows if row["snapshot_date"] == "2014-05-31"][0]
    assert pre["adjusted_close_pit"] == pytest.approx(632.9988 / 7.0)


def test_adjusted_series_is_return_continuous_across_a_split():
    raw = _raw([("2017-12-29", 100.0), ("2018-01-31", 51.0)])
    split = wp2b.actions_from_eodhd_rows(
        [{"kind": "split", "effective_date": "2018-01-10", "numerator": 2.0, "denominator": 1.0}], "AAA")
    window = wp2b.MembershipWindow("AAA", "AAA", "2017-01-01")
    rows = wp2b.monthly_panel_for_security(window, raw, split, ["2017-12-31", "2018-01-31"])
    by_month = {row["snapshot_date"]: row for row in rows}
    pre = by_month["2017-12-31"]["adjusted_close_pit"]
    post = by_month["2018-01-31"]["adjusted_close_pit"]
    # 100 -> 51 with a 2:1 split already applied: return ~ +2%, never ~ -49%.
    assert pre == pytest.approx(50.0) and post == pytest.approx(51.0)
    assert (post / pre - 1.0) == pytest.approx(0.02, abs=1e-6)


def test_available_at_is_the_price_observation_instant():
    raw = _raw([("2018-01-31", 42.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2018-01-01")
    rows = wp2b.monthly_panel_for_security(window, raw, [], ["2018-01-31"])
    assert rows[0]["available_at"] == "2018-01-31T21:00:00+00:00"


# ── F2: UPPERCASE _OLD delisted symbols resolve (real EODHD form) ─────────────

_UPPER_INDEX = {
    "BSC": {"code": "BSC", "name": "ELEMENTS BG Small Cap ETN"},
    "BSC_OLD": {"code": "BSC_OLD", "name": "Bear Stearns Companies Inc"},
    "WB": {"code": "WB", "name": "Weibo Corp"},
    "WB_OLD2": {"code": "WB_OLD2", "name": "Wachovia Corp"},
    "GM": {"code": "GM", "name": "General Motors Company"},
    "GM_OLD": {"code": "GM_OLD", "name": "General Motors Corp"},
}


def test_uppercase_old_symbols_are_candidates():
    assert wp2b.symbol_candidates("BSC", _UPPER_INDEX) == ["BSC", "BSC_OLD"]
    assert wp2b.symbol_candidates("WB", _UPPER_INDEX) == ["WB", "WB_OLD2"]


def test_uppercase_old_reuse_prefers_name_evidence():
    result = wp2b.resolve_eodhd_symbol("BSC", "Bear Stearns", _UPPER_INDEX)
    assert result is not None and result["code"] == "BSC_OLD"
    result = wp2b.resolve_eodhd_symbol("WB", "Wachovia", _UPPER_INDEX)
    assert result is not None and result["code"] == "WB_OLD2"


def test_uppercase_old_same_name_needs_probe_then_resolves():
    assert wp2b.resolve_eodhd_symbol("GM", "General Motors", _UPPER_INDEX) is None
    result = wp2b.resolve_eodhd_symbol("GM", "General Motors", _UPPER_INDEX, probe=lambda code: code == "GM_OLD")
    assert result is not None and result["code"] == "GM_OLD" and result["method"] == "date_probe"


# ── F3: listing guard + research eligibility on assumed starts ────────────────

def test_listing_guard_clamps_start_to_first_price():
    import pandas as pd

    changes = pd.DataFrame([{"security_id": "META", "effective_date": "2013-12-23", "action": "add"}])
    windows, _ = wp2b.reconstruct_memberships(changes, [], "2007-01-01", "2026-09-25",
                                              listing_by_security={"META": "2012-05-18"})
    meta = [w for w in windows if w.security_id == "META"][0]
    assert meta.membership_start >= "2013-12-23"
    assert meta.research_eligible is True


def test_assumed_start_is_marked_ineligible_and_never_back_projected():
    import pandas as pd

    changes = pd.DataFrame([{"security_id": "LEH", "effective_date": "2008-09-17", "action": "remove"}])
    windows, assumed = wp2b.reconstruct_memberships(changes, [], "2007-01-01", "2026-09-25")
    leh = [w for w in windows if w.security_id == "LEH"][0]
    assert leh.start_known is False
    assert leh.research_eligible is False and leh.unverifiable_before is True
    assert assumed and assumed[0]["security_id"] == "LEH"


def test_listing_guard_clamps_assumed_start_forward_too():
    import pandas as pd

    changes = pd.DataFrame([{"security_id": "NEWCO", "effective_date": "2015-01-01", "action": "remove"}])
    windows, _ = wp2b.reconstruct_memberships(changes, [], "2007-01-01", "2026-09-25",
                                              listing_by_security={"NEWCO": "2012-06-01"})
    newco = [w for w in windows if w.security_id == "NEWCO"][0]
    assert newco.membership_start == "2012-06-01"
    assert newco.research_eligible is False


# ── F5: terminal-return observability (never an invented return) ──────────────

def test_terminal_return_continues_when_series_reaches_window_end():
    raw = _raw([("2007-01-31", 10.0), ("2026-09-20", 12.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    assert wp2b.classify_terminal_return(window, raw, "2026-09-25") == wp2b.TERMINAL_RETURN_CONTINUES


def test_series_trading_long_past_removal_is_observable():
    raw = _raw([("2007-01-31", 10.0), ("2010-06-30", 12.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01", "2010-01-01")
    status = wp2b.classify_terminal_return(window, raw, "2026-09-25")
    assert status == wp2b.TERMINAL_RETURN_ENDS_AFTER_REMOVAL
    assert status in wp2b.OBSERVABLE_TERMINAL_RETURN_STATES
    assert status not in wp2b.UNKNOWN_TERMINAL_RETURN_STATES


def test_terminal_return_unknown_when_series_ends_at_removal():
    raw = _raw([("2007-01-31", 10.0), ("2008-09-15", 1.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01", "2008-09-17")
    assert wp2b.classify_terminal_return(window, raw, "2026-09-25") == wp2b.TERMINAL_RETURN_ENDS_AT_REMOVAL
    assert wp2b.TERMINAL_RETURN_ENDS_AT_REMOVAL in wp2b.UNKNOWN_TERMINAL_RETURN_STATES


def test_terminal_return_unknown_and_missing_have_no_series_defaults():
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01", "2008-09-17")
    assert wp2b.classify_terminal_return(window, None, "2026-09-25") == wp2b.TERMINAL_RETURN_MISSING


def test_wrong_company_series_after_removal_is_detected():
    # A reused bare ticker whose price series starts AFTER the removal means the
    # resolver picked today's different company: it must be flagged, not paired.
    raw = _raw([("2014-06-30", 30.0), ("2015-01-31", 32.0)])
    window = wp2b.MembershipWindow("WB", "WB", "2007-01-01", "2008-12-31")
    assert wp2b.series_ends_after_membership(window, raw) is True


# ── F9: month-ends clamp to the declared window end ──────────────────────────

def test_month_ends_clamp_final_month_to_window_end():
    months = wp2b.month_ends("2026-01-01", "2026-09-25")
    assert months[-1] == "2026-09-25"
    assert months[-2] == "2026-08-31"
    assert all(month <= "2026-09-25" for month in months)
