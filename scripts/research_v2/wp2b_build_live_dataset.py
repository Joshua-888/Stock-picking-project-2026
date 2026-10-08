"""WP2B/WP2C live build: EODHD -> BRONZE -> SILVER -> per-window identity -> universe.

Run (research mode, real data only):

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp2b_build_live_dataset.py [--limit N]

No synthetic fallback: if real data cannot be obtained the run fails loudly.
This script records provenance; it performs no feature engineering and no scoring.

WP2C changes:
* identity is resolved PER MEMBERSHIP WINDOW (ticker reuse: DELL -> DELL_OLD then
  DELL; WB -> WB_OLD2), never per-ticker and never guessed (A2);
* out-of-window removals cannot delete a valid current constituent (A3);
* EVERY current constituent must end RESOLVED or EXPLICITLY_UNRESOLVED_WITH_REASON
  (A1 hard gate), never silently absent;
* prices/actions are fetched and cached BY PROVIDER SYMBOL, so a reused ticker
  can never read another company's cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data import wp2b_eodhd as wp2b
from src.research.data.providers import adapters as ad

SECRETS_PATH = ROOT / ".a0proj" / "secrets.env"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp2b_live"
BRONZE_ROOT = ROOT / "data" / "research_v2"
CORPORATE_ACTION_FIELDS = ("ticker", "kind", "effective_date", "numerator", "denominator", "amount")
SILVER_PRICES = "wp2b_sp500_pit_prices_silver"
SILVER_ACTIONS = "wp2b_sp500_pit_actions_silver"
SILVER_MEMBERSHIP = "wp2b_sp500_pit_membership_silver"
CACHE_DIR = Path("/tmp/wp2b_cache")


def load_eodhd_token():
    """Read EODHD_API_TOKEN from .a0proj/secrets.env into the environment.

    The token is never printed, logged or fingerprinted by this script.
    """
    text = SECRETS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^EODHD_API_TOKEN=(.*)$", text, re.M)
    if not match:
        raise SystemExit("EODHD_API_TOKEN not found in secrets file")
    token = match.group(1).strip().strip('"').strip("'")
    if not token:
        raise SystemExit("EODHD_API_TOKEN is empty")
    os.environ["EODHD_API_TOKEN"] = token
    return token


def redact(message):
    token = os.environ.get("EODHD_API_TOKEN") or ""
    text = str(message)
    if token:
        text = text.replace(token, "***REDACTED***")
    return text


def _fetch_text(url, timeout=60):
    return ad.http_get(url, timeout=timeout)


def fetch_current_constituents():
    url = ad.WIKIPEDIA_SP500_URL.replace("Historical_components_of_the_S%26P_500", "List_of_S%26P_500_companies")
    html = _fetch_text(url)
    constituents = set()
    for table in ad.parse_html_tables(html):
        if not table or not any("Symbol" in cell for cell in table[0]):
            continue
        index = table[0].index("Symbol")
        for row in table[1:]:
            if index < len(row):
                symbol = row[index].strip().upper()
                if re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", symbol):
                    constituents.add(symbol)
    return sorted(constituents), html


def build_bronze(provider):
    """Fetch + bronze-cache both membership sources and both EODHD symbol lists."""
    hist_html = _fetch_text(ad.WIKIPEDIA_SP500_URL)
    bronze_hist = layers.write_bronze_bytes(BRONZE_ROOT, "wp2b_wikipedia_sp500_changes", hist_html,
                                            ext="html", meta={"source": ad.WIKIPEDIA_SP500_URL})
    constituents, current_html = fetch_current_constituents()
    layers.write_bronze_bytes(BRONZE_ROOT, "wp2b_wikipedia_sp500_current", current_html, ext="html",
                              meta={"source": "List_of_S%26P_500_companies"})
    changes = ad.parse_wikipedia_sp500_changes(hist_html)

    delisted = provider.fetch_exchange_symbols("US", delisted=True)
    active = provider.fetch_exchange_symbols("US", delisted=False)
    bronze_del = layers.write_bronze_json(BRONZE_ROOT, "wp2b_eodhd_symbol_list_delisted", delisted,
                                          meta={"endpoint": "exchange-symbol-list/US?delisted=1"})
    bronze_act = layers.write_bronze_json(BRONZE_ROOT, "wp2b_eodhd_symbol_list_active", active,
                                          meta={"endpoint": "exchange-symbol-list/US?delisted=0"})
    return changes, constituents, delisted, active, bronze_hist, bronze_del, bronze_act


def build_name_hints(bronze_hist_record):
    html = Path(bronze_hist_record["raw_path"]).read_text(encoding="utf-8")
    hints = {}
    for rows in ad.parse_html_tables(html):
        for row in rows:
            if len(row) < 5 or ad.normalize_wikipedia_date(row[0]) is None:
                continue
            if ad._TICKER_RE.match(row[1].strip().upper()):
                hints.setdefault(row[1].strip().upper(), row[2].strip())
            if ad._TICKER_RE.match(row[3].strip().upper()):
                hints.setdefault(row[3].strip().upper(), row[4].strip())
    return hints


def _cache_key(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _cache_frame(key, role, frame):
    """Persist a fetched frame keyed by the RESOLVED provider symbol."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / ("%s.%s.parquet" % (_cache_key(key), role))
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        pass


def _load_cached(key, role, columns=None):
    path = CACHE_DIR / ("%s.%s.parquet" % (_cache_key(key), role))
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _probe_first_date(provider, code, first_cache, price_cache):
    """First available trade date for one provider symbol over the full window."""
    code = str(code).upper()
    if code in first_cache:
        return first_cache[code]
    frame = price_cache.get(code)
    if frame is None:
        frame = _load_cached(code, "prices", ad.DELISTED_PRICE_FIELDS)
    if frame is None:
        try:
            frame = provider.fetch_delisted_prices("probe", code, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
            _cache_frame(code, "prices", frame)
        except Exception as exc:
            print("PROBE_FAIL %s %s" % (code, redact(exc)))
            frame = pd.DataFrame(columns=list(ad.DELISTED_PRICE_FIELDS))
    price_cache[code] = frame
    days = [str(day) for day in frame["trade_date"].tolist()] if len(frame) else []
    first = min(days) if days else None
    first_cache[code] = first
    return first


def _unresolved_reason(window, symbol_index):
    candidates = wp2b.symbol_candidates(window.security_id, symbol_index)
    if not candidates:
        return "no_symbol_candidates_in_vendor"
    if (window.membership_end is None
            and str(window.membership_start)[:10] > wp2b.RESEARCH_WINDOW_END):
        # A post-window-only current constituent is a valid identity record but can
        # never appear in a research cross-section; the factual limitation is its
        # membership start, not symbol ambiguity.
        return "membership_starts_after_research_window_end"
    return "ambiguous_no_discriminating_evidence"


def resolve_windows(windows, symbol_index, provider, name_hints, active_codes, price_cache=None):
    """Resolve ONE EODHD provider symbol PER MEMBERSHIP WINDOW (WP2C-A2).

    Returns ``(resolutions, price_cache)`` where each resolution records the
    chosen symbol, the method and -- when unresolved -- an explicit reason, so no
    window is ever silently absent.
    """
    first_cache = {}
    price_cache = price_cache if price_cache is not None else {}
    resolutions = []
    for window in windows:
        candidates = wp2b.symbol_candidates(window.security_id, symbol_index)

        def probe(code, start, end, _provider=provider):
            return _probe_first_date(_provider, code, first_cache, price_cache)

        try:
            result = wp2b.choose_window_symbol(
                window.security_id, window, symbol_index,
                name_hint=name_hints.get(window.security_id), active_codes=active_codes,
                probe=probe if len(candidates) > 1 else None)
        except Exception as exc:
            print("RESOLVE_FAIL %s %s" % (window.security_id, redact(exc)))
            result = None
        if result is None:
            resolutions.append({
                "security_id": window.security_id, "membership_start": window.membership_start,
                "membership_end": window.membership_end, "symbol": None, "resolved": False,
                "method": None, "reason": _unresolved_reason(window, symbol_index),
                "candidates": candidates, "name_hint": name_hints.get(window.security_id),
            })
        else:
            resolutions.append({
                "security_id": window.security_id, "membership_start": window.membership_start,
                "membership_end": window.membership_end, "symbol": result["code"], "resolved": True,
                "method": result["method"], "reason": None, "candidates": candidates,
                "name_hint": name_hints.get(window.security_id),
            })
    return resolutions, price_cache


def _fetch_one(provider, symbol, no_actions):
    """Fetch prices + corporate actions for ONE provider symbol (cache-keyed)."""
    failed = []
    prices = _load_cached(symbol, "prices", ad.DELISTED_PRICE_FIELDS)
    if prices is None:
        try:
            prices = provider.fetch_delisted_prices(symbol, symbol, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
            _cache_frame(symbol, "prices", prices)
        except Exception as exc:
            prices = pd.DataFrame(columns=list(ad.DELISTED_PRICE_FIELDS))
            failed.append({"symbol": symbol, "kind": "price", "error": redact(exc)})
    actions = pd.DataFrame(columns=list(CORPORATE_ACTION_FIELDS))
    if not no_actions:
        actions = _load_cached(symbol, "actions", CORPORATE_ACTION_FIELDS)
        if actions is None:
            try:
                actions = provider.fetch_corporate_actions(symbol, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
                _cache_frame(symbol, "actions", actions)
            except Exception as exc:
                actions = pd.DataFrame(columns=list(CORPORATE_ACTION_FIELDS))
                failed.append({"symbol": symbol, "kind": "action", "error": redact(exc)})
    return symbol, prices, actions, failed


def fetch_all(provider, symbols, no_actions, workers=8, price_cache=None):
    price_cache = price_cache if price_cache is not None else {}
    action_cache, failures = {}, []
    started = time.time()
    todo = [symbol for symbol in sorted(symbols) if symbol not in price_cache]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, provider, symbol, no_actions): symbol for symbol in todo}
        for future in as_completed(futures):
            symbol, prices, actions, failed = future.result()
            price_cache[symbol] = prices
            action_cache[symbol] = actions
            failures.extend(failed)
            done += 1
            if done % 50 == 0:
                print("FETCHED %d/%d elapsed=%.0fs" % (done, len(todo), time.time() - started))
    for symbol in price_cache:
        action_cache.setdefault(symbol, pd.DataFrame(columns=list(CORPORATE_ACTION_FIELDS)))
    return price_cache, action_cache, failures


def _price_frame_by_security(windows, price_cache):
    """Map security_id -> raw price frame using each window's RESOLVED symbol."""
    out = {}
    for window in windows:
        symbol = window.symbol
        if not symbol or symbol not in price_cache:
            continue
        out.setdefault(window.security_id, price_cache[symbol])
    return out


def _first_price_by_symbol(price_cache):
    first = {}
    for symbol, frame in (price_cache or {}).items():
        if frame is None or len(frame) == 0:
            continue
        days = [str(day) for day in frame["trade_date"].tolist()]
        if days:
            first[str(symbol).upper()] = min(days)
    return first


def coverage_summary(coverage_df, unresolved_windows, removed_ids, securities):
    total = len(coverage_df)
    counts = Counter(coverage_df["price_completeness"].tolist()) if total else Counter()
    removed_counts = Counter(coverage_df.loc[coverage_df["security_id"].isin(removed_ids), "price_completeness"].tolist()) if total else Counter()
    by_year, by_delisting_type = {}, {}
    if total:
        for _, row in coverage_df.iterrows():
            year = str(row.get("removal_year") or "ongoing")
            dtype = str(row.get("terminal_behavior") or "unknown")
            by_year.setdefault(year, Counter())[row["price_completeness"]] += 1
            by_delisting_type.setdefault(dtype, Counter())[row["price_completeness"]] += 1
    by_year = {key: dict(value) for key, value in sorted(by_year.items())}
    by_delisting_type = {key: dict(value) for key, value in sorted(by_delisting_type.items())}
    resolved_complete = counts.get(wp2b.PRICE_COMPLETE, 0)
    percent = round(100.0 * resolved_complete / total, 2) if total else 0.0
    removed_percent = round(100.0 * removed_counts.get(wp2b.PRICE_COMPLETE, 0) / max(1, sum(removed_counts.values())), 2)
    terminal_unknown = quarantined = removed_unknown = 0
    removed_uncertain_ids = []
    if total:
        for _, row in coverage_df.iterrows():
            status = str(row.get("terminal_return_status") or "")
            is_removed = row["security_id"] in removed_ids
            if bool(row.get("quarantined_wrong_company")):
                quarantined += 1
            if status in wp2b.UNKNOWN_TERMINAL_RETURN_STATES:
                terminal_unknown += 1
                if is_removed:
                    removed_unknown += 1
                    removed_uncertain_ids.append(str(row["security_id"]))
    removed_total = max(1, len(removed_ids))
    return {
        "securities_total": total,
        "resolved": total,
        "unresolved_windows": len(unresolved_windows),
        "unresolved_securities": len({item["security_id"] for item in unresolved_windows}),
        "complete": counts.get(wp2b.PRICE_COMPLETE, 0),
        "partial_history": counts.get(wp2b.PRICE_PARTIAL, 0),
        "missing": counts.get(wp2b.PRICE_MISSING, 0),
        "terminal_return_uncertain": counts.get(wp2b.PRICE_TERMINAL_UNCERTAIN, 0),
        "identity_unresolved": counts.get(wp2b.PRICE_IDENTITY_UNRESOLVED, 0),
        "overall_complete_percent": percent,
        "removed_constituent_complete_percent": removed_percent,
        "removed_security_count": len(removed_ids),
        "terminal_return_unknown_count": terminal_unknown,
        "removed_without_usable_terminal_price": removed_unknown,
        "removed_without_usable_terminal_price_ids": sorted(set(removed_uncertain_ids)),
        "wrong_company_quarantined": quarantined,
        "label_loss_percent": round(100.0 * removed_unknown / removed_total, 2),
        "window_count": int(len(coverage_df)),
        "by_year": by_year,
        "by_delisting_type": by_delisting_type,
    }


def build_coverage(windows, price_by_security, window_start, window_end):
    """Per-window price completeness / terminal behaviour / wrong-company quarantine."""
    rows = []
    for window in windows:
        raw_frame = price_by_security.get(window.security_id)
        quarantined = wp2b.series_ends_after_membership(window, raw_frame)
        completeness = wp2b.classify_price_completeness(window, raw_frame, window_start, window_end)
        terminal = wp2b.classify_terminal(window, raw_frame, window_end)
        terminal_return = wp2b.classify_terminal_return(window, raw_frame, window_end)
        if quarantined:
            completeness = wp2b.PRICE_IDENTITY_UNRESOLVED
        rows.append({
            "security_id": window.security_id, "symbol": window.symbol,
            "price_completeness": completeness, "terminal_behavior": terminal,
            "terminal_return_status": terminal_return,
            "quarantined_wrong_company": bool(quarantined),
            "removal_year": str(window.membership_end)[:4] if window.membership_end else "ongoing",
            "price_rows": int(len(raw_frame) if raw_frame is not None else 0),
            "resolution_method": window.resolution_method,
            "resolution_reason": window.unresolved_reason,
        })
    return pd.DataFrame(rows)


def _serialize_membership(windows):
    return pd.DataFrame([{
        "security_id": window.security_id, "ticker": window.ticker, "symbol": window.symbol,
        "membership_start": window.membership_start, "membership_end": window.membership_end,
        "start_known": window.start_known, "research_eligible": window.research_eligible,
        "unverifiable_before": window.unverifiable_before,
        "resolution_method": window.resolution_method, "unresolved_reason": window.unresolved_reason,
        "source": "wikipedia:historical_components_sp500",
    } for window in windows])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-actions", action="store_true")
    args = parser.parse_args(argv)

    load_eodhd_token()
    provider = ad.EodhdProvider(timeout=60)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    changes, constituents, delisted, active, bronze_hist, bronze_del, bronze_act = build_bronze(provider)
    symbol_index = wp2b.build_symbol_index(list(active) + list(delisted))
    active_codes = {str(row.get("Code")).upper() for row in active}
    name_hints = build_name_hints(bronze_hist)

    windows, assumed = wp2b.reconstruct_memberships(changes, constituents,
                                                     wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    print("SEED_MEMBERSHIP windows=%d securities=%d assumed_starts=%d" % (
        len(windows), len({w.security_id for w in windows}), len(assumed)))

    resolutions, price_cache = resolve_windows(windows, symbol_index, provider, name_hints, active_codes)
    index_by_key = {(item["security_id"], item["membership_start"], item["membership_end"]): item
                    for item in resolutions}
    resolved_symbols = set()
    for window in windows:
        item = index_by_key.get((window.security_id, window.membership_start, window.membership_end))
        if item and item["resolved"]:
            window.symbol = item["symbol"]
            window.resolution_method = item["method"]
            resolved_symbols.add(item["symbol"])
        elif item:
            window.unresolved_reason = item["reason"]
    print("RESOLVED windows=%d symbols=%d unresolved_windows=%d" % (
        sum(1 for item in resolutions if item["resolved"]), len(resolved_symbols),
        sum(1 for item in resolutions if not item["resolved"])))

    price_cache, action_cache, failures = fetch_all(provider, resolved_symbols, args.no_actions,
                                                    price_cache=price_cache)
    print("PRICE symbols=%d rows=%d ACTION rows=%d failures=%d" % (
        len(price_cache), sum(len(f) for f in price_cache.values()),
        sum(len(a) for a in action_cache.values()), len(failures)))

    first_by_symbol = _first_price_by_symbol(price_cache)
    first_by_security = {}
    for window in windows:
        if window.symbol and window.symbol.upper() in first_by_symbol:
            first_by_security.setdefault(window.security_id, first_by_symbol[window.symbol.upper()])
    windows = wp2b.apply_listing_guard(windows, first_by_security, first_price_by_symbol=first_by_symbol)
    ineligible = sum(1 for window in windows if not window.research_eligible)
    print("LISTING_GUARD windows=%d ineligible=%d" % (len(windows), ineligible))

    price_by_security = _price_frame_by_security(windows, price_cache)
    # Memory-safe silver assembly: ~3.5M price rows must NOT be built as a Python
    # list of dicts (that OOM-killed an earlier run in a 4GB container). Each
    # security's slim frame is projected and concatenated once.
    slims = []
    for security, frame in price_by_security.items():
        if frame is None or len(frame) == 0:
            continue
        slim = frame[["trade_date", "raw_close"]].copy()
        slim["security_id"] = security
        slim["ticker"] = security
        slims.append(slim[["security_id", "ticker", "trade_date", "raw_close"]])
    if slims:
        prices_df = pd.concat(slims, ignore_index=True)
        prices_df["trade_date"] = prices_df["trade_date"].astype(str).str.slice(0, 10)
    else:
        prices_df = pd.DataFrame(columns=["security_id", "ticker", "trade_date", "raw_close"])
    del slims
    action_rows = []
    for window in windows:
        if not window.symbol:
            continue
        frame = action_cache.get(window.symbol)
        actions = wp2b.actions_from_eodhd_rows(frame.to_dict("records") if frame is not None else [], window.security_id)
        for action in actions:
            action_rows.append({"security_id": window.security_id, "ticker": window.security_id,
                                "kind": action.kind, "effective_date": action.effective_date,
                                "numerator": action.numerator, "denominator": action.denominator,
                                "amount": action.amount})
    actions_df = pd.DataFrame(action_rows, columns=["security_id", "ticker", "kind", "effective_date", "numerator", "denominator", "amount"])
    del action_rows
    gc.collect()

    silver_prices = layers.write_silver_table(BRONZE_ROOT, SILVER_PRICES, prices_df, meta={"source": "eodhd"})
    silver_actions = layers.write_silver_table(BRONZE_ROOT, SILVER_ACTIONS, actions_df, meta={"source": "eodhd"})
    membership_df = _serialize_membership(windows)
    silver_membership = layers.write_silver_table(BRONZE_ROOT, SILVER_MEMBERSHIP, membership_df,
                                                  meta={"source": "wikipedia:historical_components_sp500"})

    coverage_df = build_coverage(windows, price_by_security, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    removed_ids = {window.security_id for window in windows if window.membership_end is not None}
    unresolved_windows = [item for item in resolutions if not item["resolved"]]
    summary = coverage_summary(coverage_df, unresolved_windows, removed_ids,
                               {window.security_id for window in windows})
    summary["membership_ineligible_assumed_starts"] = int(ineligible)

    # WP2C-A1: EVERY current constituent must be RESOLVED or explicitly unresolved.
    resolved_securities = {item["security_id"] for item in resolutions if item["resolved"]}
    explicit_unresolved = {item["security_id"] for item in resolutions if not item["resolved"]}
    current_set = {str(name).strip().upper() for name in constituents}
    in_universe = {window.security_id for window in windows}
    silently_absent = sorted(current_set - resolved_securities - explicit_unresolved)
    summary["current_constituents"] = len(current_set)
    summary["current_constituents_resolved"] = len(current_set & resolved_securities)
    summary["current_constituents_explicitly_unresolved"] = sorted(current_set & explicit_unresolved)
    summary["current_constituents_silently_absent"] = silently_absent
    if silently_absent:
        raise SystemExit("CURRENT_CONSTITUENT_INTEGRITY_FAIL silently_absent=%s" % silently_absent)

    diagnostics = wp2b.membership_diagnostics(windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    monthly = wp2b.monthly_member_counts(windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)

    unresolved_artifact = {
        "unresolved_windows": unresolved_windows,
        "resolved_windows": [item for item in resolutions if item["resolved"]],
        "explicitly_unresolved_securities": sorted(explicit_unresolved),
        "silently_absent_securities": silently_absent,
    }
    (ARTIFACT_DIR / "unresolved_identities.json").write_text(
        json.dumps(unresolved_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "price_completeness.json").write_text(
        json.dumps({"summary": summary, "per_window": coverage_df.to_dict("records")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "membership_diagnostics.json").write_text(
        json.dumps({"diagnostics": diagnostics, "monthly": monthly, "assumed_starts": assumed}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "fetch_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "layer_records.json").write_text(json.dumps({
        "silver_prices": silver_prices["version"], "silver_actions": silver_actions["version"],
        "silver_membership": silver_membership["version"],
        "price_rows": len(prices_df), "action_rows": len(actions_df),
        "windows": len(windows), "securities": len({w.security_id for w in windows}),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    print("BUILD_DONE windows=%d" % len(windows))


if __name__ == "__main__":
    main()
