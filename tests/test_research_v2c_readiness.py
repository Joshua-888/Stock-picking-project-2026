"""WP2C research-readiness regression tests (A1/A2/A3 + censoring model/diagnostics).

LIVE-INDEPENDENT: every fixture is small and explicitly synthetic, used only to
exercise integrity logic. No test reads the real token or performs network I/O.
Nothing here is research evidence; no feature engineering, model fitting or
scoring runs. These capture the corrected semantics so a future change cannot
silently reintroduce a leak, a dropped constituent, or a fabricated label.
"""

import importlib.util
from pathlib import Path

import pandas as pd

from src.research.data import wp2b_eodhd as wp2b

_ROOT = Path(__file__).resolve().parents[1]


def _load_finalize():
    spec = importlib.util.spec_from_file_location("fin", _ROOT / "scripts" / "research_v2" / "wp2b_finalize_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── A1: CURRENT-CONSTITUENT INTEGRITY (the hard gate) ─────────────────────────

def _constituent_state(current, windows, explicitly_unresolved):
    """Replicate the build's A1 gate: current minus (resolved ∪ explicit)."""
    resolved = {str(window.security_id).upper() for window in windows if window.research_eligible or window.membership_end is None}
    resolved |= {str(window.security_id).upper() for window in windows}
    explicit = {str(item).upper() for item in explicitly_unresolved}
    return sorted({str(item).upper() for item in current} - resolved - explicit)


def test_current_constituents_are_never_silently_absent():
    changes = pd.DataFrame([
        {"security_id": "OLD", "effective_date": "2010-01-01", "action": "remove"},
        {"security_id": "T", "effective_date": "2005-11-18", "action": "remove"},
    ])
    current = ["AAPL", "MSFT", "T"]
    windows, _ = wp2b.reconstruct_memberships(changes, current, "2007-01-01", "2026-09-25")
    # T's only removal is OUT OF WINDOW; AAPL/MSFT have no events at all.
    silent = _constituent_state(current, windows, explicitly_unresolved=[])
    assert silent == [], "every current constituent must be RESOLVED or explicitly unresolved"


def test_hard_gate_fails_when_a_current_name_is_dropped():
    changes = pd.DataFrame([{"security_id": "OLD", "effective_date": "2010-01-01", "action": "remove"}])
    windows, _ = wp2b.reconstruct_memberships(changes, ["AAPL"], "2007-01-01", "2026-09-25")
    # Pretend AVGO silently vanished (resolver failure) while still a constituent.
    silent = _constituent_state(["AAPL", "AVGO"], windows, explicitly_unresolved=[])
    assert silent == ["AVGO"], "the gate MUST detect a silently missing current constituent"


def test_explicitly_unresolved_current_constituent_is_not_silent_absence():
    changes = pd.DataFrame([{"security_id": "OLD", "effective_date": "2010-01-01", "action": "remove"}])
    windows, _ = wp2b.reconstruct_memberships(changes, ["AAPL"], "2007-01-01", "2026-09-25")
    silent = _constituent_state(["AAPL", "Q"], windows, explicitly_unresolved=["Q"])
    assert silent == [], "an explicitly unresolved name is accounted for, not silently dropped"


# ── A2: RESOLVER EXACT-ACTIVE PREFERENCE + _OLD HISTORICAL HANDLING ───────────

_C_INDEX = {
    "BAC": {"code": "BAC", "name": "Bank of America Corp"},
    "BAC_OLD": {"code": "BAC_OLD", "name": "NationsBank Corp"},
    "C": {"code": "C", "name": "Citigroup Inc"},
    "C_OLD": {"code": "C_OLD", "name": "Citicorp"},
    "WB": {"code": "WB", "name": "Weibo Corp"},
    "WB_OLD2": {"code": "WB_OLD2", "name": "Wachovia Corp"},
    "GM": {"code": "GM", "name": "General Motors Company"},
    "GM_OLD": {"code": "GM_OLD", "name": "General Motors Corp"},
}


def test_active_constituent_prefers_exact_active_ticker():
    ongoing = wp2b.MembershipWindow("BAC", "BAC", "2007-01-01", None)
    result = wp2b.choose_window_symbol("BAC", ongoing, _C_INDEX, name_hint="Bank of America", active_codes={"BAC", "C"})
    assert result is not None and result["code"] == "BAC" and result["method"] == "active_current"


def test_active_constituent_never_falls_to_old_variant():
    ongoing = wp2b.MembershipWindow("C", "C", "2007-01-01", None)
    result = wp2b.choose_window_symbol("C", ongoing, _C_INDEX, name_hint="Citigroup", active_codes={"C"})
    assert result is not None and result["code"] == "C"


def test_historical_window_uses_name_evidence_for_old_variant():
    historical = wp2b.MembershipWindow("WB", "WB", "2007-01-01", "2008-12-31")
    result = wp2b.choose_window_symbol("WB", historical, _C_INDEX, name_hint="Wachovia")
    assert result is not None and result["code"] == "WB_OLD2"


def test_historical_window_uses_price_coverage_when_bare_ticker_is_reused():
    historical = wp2b.MembershipWindow("GM", "GM", "1997-01-01", "2011-03-31")
    first = {"GM_OLD": "1997-01-02", "GM": "2010-11-18"}
    result = wp2b.choose_window_symbol(
        "GM", historical, _C_INDEX, name_hint="General Motors", probe=lambda code, start, end: first.get(code))
    assert result is not None and result["code"] == "GM_OLD" and result["method"] == "price_coverage"


def test_ambiguous_historical_window_is_unresolved_never_guessed():
    historical = wp2b.MembershipWindow("GM", "GM", "1997-01-01", "2011-03-31")
    assert wp2b.choose_window_symbol("GM", historical, _C_INDEX, name_hint=None) is None


# ── A3: OUT-OF-WINDOW REMOVALS MUST NOT DROP A CURRENT CONSTITUENT ────────────

def test_out_of_window_removal_does_not_remove_current_constituent():
    # T was removed 2005-11-18 (before the 2007 window); it is a current member.
    changes = pd.DataFrame([{"security_id": "T", "effective_date": "2005-11-18", "action": "remove"}])
    windows, _ = wp2b.reconstruct_memberships(changes, ["T"], "2007-01-01", "2026-09-25")
    t_windows = [window for window in windows if window.security_id == "T"]
    assert any(window.membership_end is None for window in t_windows)
    assert all(start_ok(window) for window in t_windows)


def start_ok(window):
    return str(window.membership_start)[:10] >= "2007-01-01"


def test_ancient_removal_does_not_create_pre_window_assumption():
    changes = pd.DataFrame([{"security_id": "MCK", "effective_date": "1994-09-30", "action": "remove"}])
    windows, assumed = wp2b.reconstruct_memberships(changes, ["MCK"], "2007-01-01", "2026-09-25")
    assert assumed == [], "a removal before the window start is not an in-window assumption"
    assert any(window.membership_end is None for window in windows if window.security_id == "MCK")


def test_fox_remove_and_add_same_day_keeps_ongoing_membership():
    changes = pd.DataFrame([
        {"security_id": "FOX", "effective_date": "2019-03-19", "action": "remove"},
        {"security_id": "FOX", "effective_date": "2019-03-19", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["FOX"], "2007-01-01", "2026-09-25")
    assert any(window.membership_end is None for window in windows if window.security_id == "FOX")


# ── PHASE B: TARGET CENSORING MODEL (never a fabricated label) ────────────────

def _raw(days_and_closes):
    return pd.DataFrame({
        "ticker": ["AAA"] * len(days_and_closes),
        "security_id": ["AAA"] * len(days_and_closes),
        "trade_date": [day for day, _ in days_and_closes],
        "raw_close": [close for _, close in days_and_closes],
    })


def test_horizon_past_window_end_is_censored_with_reason():
    raw = _raw([("2026-09-01", 50.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    observable, censored, reason = wp2b.classify_target_observability(window, raw, "2026-09-01", "2026-09-25")
    assert observable is False and censored is True
    assert reason == wp2b.CENSOR_REASON_NEAR_WINDOW_END


def test_series_ending_before_horizon_is_censored():
    raw = _raw([("2007-01-31", 10.0), ("2007-02-06", 9.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    observable, censored, reason = wp2b.classify_target_observability(window, raw, "2007-01-31", "2026-09-25")
    assert observable is False and censored is True
    assert reason == wp2b.CENSOR_REASON_NO_TERMINAL


def test_series_reaching_horizon_is_observable():
    raw = _raw([("2007-01-31", 10.0), ("2008-01-31", 12.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    observable, censored, reason = wp2b.classify_target_observability(window, raw, "2007-01-31", "2026-09-25")
    assert observable is True and censored is False and reason is None


def test_no_price_series_is_censored_with_reason():
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    observable, censored, reason = wp2b.classify_target_observability(window, None, "2007-01-31", "2026-09-25")
    assert observable is False and censored is True and reason == wp2b.CENSOR_REASON_NO_PRICE


def test_panel_rows_carry_censoring_fields():
    raw = _raw([("2007-01-31", 10.0), ("2008-01-31", 12.0)])
    window = wp2b.MembershipWindow("AAA", "AAA", "2007-01-01")
    rows = wp2b.monthly_panel_for_security(window, raw, [], ["2007-01-31", "2008-01-31"], window_end="2026-09-25")
    for row in rows:
        for field in ("target_observable", "target_censored", "target_censor_reason",
                      "terminal_price_observable", "terminal_status", "membership_exit_observed"):
            assert field in row
    assert rows[0]["target_observable"] is True and rows[0]["target_censored"] is False


# ── PHASE C: CENSORING DIAGNOSTICS SHAPE ──────────────────────────────────────

def test_censoring_diagnostics_shape_and_percentages():
    finalize = _load_finalize()
    panel = pd.DataFrame([
        {"security_id": "AAA", "snapshot_date": "2018-01-31", "target_censored": False,
         "target_censor_reason": None, "has_price": True, "adjusted_close_pit": 10.0},
        {"security_id": "AAA", "snapshot_date": "2018-02-28", "target_censored": True,
         "target_censor_reason": wp2b.CENSOR_REASON_NO_TERMINAL, "has_price": True, "adjusted_close_pit": 9.0},
        {"security_id": "BBB", "snapshot_date": "2019-01-31", "target_censored": True,
         "target_censor_reason": wp2b.CENSOR_REASON_NO_TERMINAL, "has_price": False, "adjusted_close_pit": None},
    ])
    windows = [wp2b.MembershipWindow("AAA", "AAA", "2018-01-01", "2018-12-31")]
    result = finalize.censoring_diagnostics(panel, windows)
    assert result["total_candidate_labels"] == 3
    assert result["observable_labels"] == 1 and result["censored_labels"] == 2
    assert result["censoring_percent"] == pytest_approx(66.67)
    assert "2018" in result["by_calendar_year"] and "2019" in result["by_calendar_year"]
    assert result["censor_reason_counts"][wp2b.CENSOR_REASON_NO_TERMINAL] == 2
    # sector / market-cap buckets are explicitly unavailable, never fabricated
    assert result["by_sector"] is None and result["by_market_cap"] is None


def pytest_approx(value):
    import pytest
    return pytest.approx(value, abs=0.01)
