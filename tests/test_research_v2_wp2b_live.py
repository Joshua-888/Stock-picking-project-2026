"""Focused fail-closed regression tests for WP2B/WP2C current-constituent accounting.

A current S&P 500 constituent whose source-known add/effective date falls after
``RESEARCH_WINDOW_END`` must still produce an identity record that reaches the
per-window symbol resolver (A2) and the current-constituent hard gate (A1). It
must never be backdated into the research window, must never become
``research_eligible``, and must never be silently absent from the gate.

LIVE-INDEPENDENT: every fixture is small and synthetic; no live network, token,
or long build is used. Nothing here is research evidence and no model/scoring
runs.
"""

import importlib.util
from pathlib import Path

import pandas as pd

from src.research.data import wp2b_eodhd as wp2b

_ROOT = Path(__file__).resolve().parents[1]

WINDOW_START = "2007-01-01"
WINDOW_END = "2026-09-25"


def _load_live_build():
    spec = importlib.util.spec_from_file_location(
        "_wp2b_live_build_under_test",
        _ROOT / "scripts" / "research_v2" / "wp2b_build_live_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LIVE = _load_live_build()


def _gate_silently_absent(current, resolutions):
    """Mirror the build's WP2C-A1 gate exactly."""
    resolved = {item["security_id"] for item in resolutions if item["resolved"]}
    explicit = {item["security_id"] for item in resolutions if not item["resolved"]}
    return sorted({str(name).strip().upper() for name in current} - resolved - explicit)


def test_post_window_current_constituent_is_resolved_and_not_silently_absent():
    """TWLO added 2026-10-06 must resolve to its exact active symbol."""
    changes = pd.DataFrame([
        {"security_id": "TWLO", "effective_date": "2026-10-06", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["TWLO"], WINDOW_START, WINDOW_END)
    twlo = [window for window in windows if window.security_id == "TWLO"]
    assert len(twlo) == 1
    assert twlo[0].membership_start == "2026-10-06"
    assert twlo[0].membership_end is None
    assert twlo[0].start_known is True
    assert twlo[0].research_eligible is False

    index = {"TWLO": {"code": "TWLO", "name": "Twilio Inc"}}
    resolutions, _ = _LIVE.resolve_windows(
        windows, index, provider=None, name_hints={"TWLO": "Twilio"}, active_codes={"TWLO"},
    )
    twlo_res = [item for item in resolutions if item["security_id"] == "TWLO"]
    assert any(item["resolved"] and item["symbol"] == "TWLO" for item in twlo_res)
    assert _gate_silently_absent(["TWLO"], resolutions) == []


def test_post_window_current_constituent_is_explicitly_unresolved_with_factual_reason():
    """Ambiguous candidates must still yield an explicit factual reason, never silence."""
    changes = pd.DataFrame([
        {"security_id": "VYLR", "effective_date": "2026-10-01", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["VYLR"], WINDOW_START, WINDOW_END)
    vylr = [window for window in windows if window.security_id == "VYLR"]
    assert len(vylr) == 1
    assert vylr[0].membership_start == "2026-10-01"
    assert vylr[0].research_eligible is False

    index = {
        "VYLR": {"code": "VYLR", "name": "Valaris Ltd"},
        "VYLR_OLD": {"code": "VYLR_OLD", "name": "Valaris Ltd"},
    }
    resolutions, _ = _LIVE.resolve_windows(
        windows, index, provider=None, name_hints={"VYLR": "Valaris"}, active_codes=set(),
    )
    vylr_res = [item for item in resolutions if item["security_id"] == "VYLR"]
    assert any(not item["resolved"] for item in vylr_res)
    assert any(item["reason"] == "membership_starts_after_research_window_end" for item in vylr_res)
    assert _gate_silently_absent(["VYLR"], resolutions) == []


def test_post_window_current_constituent_with_no_vendor_candidate_is_explicit():
    changes = pd.DataFrame([
        {"security_id": "NEWCO", "effective_date": "2026-10-01", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["NEWCO"], WINDOW_START, WINDOW_END)
    resolutions, _ = _LIVE.resolve_windows(
        windows, {}, provider=None, name_hints={}, active_codes=set(),
    )
    newco_res = [item for item in resolutions if item["security_id"] == "NEWCO"]
    assert any(not item["resolved"] for item in newco_res)
    assert any(item["reason"] == "no_symbol_candidates_in_vendor" for item in newco_res)
    assert _gate_silently_absent(["NEWCO"], resolutions) == []


def test_post_window_current_constituent_contributes_zero_panel_rows():
    """A post-window identity must never materialize historical research rows."""
    changes = pd.DataFrame([
        {"security_id": "TWLO", "effective_date": "2026-10-06", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["TWLO"], WINDOW_START, WINDOW_END)
    twlo = [window for window in windows if window.security_id == "TWLO"][0]
    raw = pd.DataFrame(columns=["ticker", "security_id", "trade_date", "raw_close"])
    rows = wp2b.monthly_panel_for_security(
        twlo, raw, [], wp2b.month_ends(WINDOW_START, WINDOW_END), window_end=WINDOW_END,
    )
    assert rows == []


def test_hard_gate_still_raises_for_genuinely_unhandled_current_symbol():
    """Fail-closed is preserved: a name with neither record must still be detected."""
    changes = pd.DataFrame([
        {"security_id": "AAPL", "effective_date": "2000-01-01", "action": "add"},
    ])
    windows, _ = wp2b.reconstruct_memberships(changes, ["AAPL"], WINDOW_START, WINDOW_END)
    index = {"AAPL": {"code": "AAPL", "name": "Apple Inc"}}
    resolutions, _ = _LIVE.resolve_windows(
        windows, index, provider=None, name_hints={"AAPL": "Apple"}, active_codes={"AAPL"},
    )
    silently_absent = _gate_silently_absent(["AAPL", "ZZZZ"], resolutions)
    assert silently_absent == ["ZZZZ"]
