"""WP2B live build: EODHD -> BRONZE -> SILVER -> identity -> universe -> GOLD.

Run (research mode, real data only):

    /opt/venv/bin/python scripts/research_v2/wp2b_build_live_dataset.py [--limit N]

No synthetic fallback: if real data cannot be obtained the run fails loudly.
This script records provenance; it performs no feature engineering and no scoring.
"""

from __future__ import annotations

import argparse
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
GOLD_PANEL = "wp2b_sp500_pit_panel_gold"


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


def resolve_identities(tickers, symbol_index, provider, name_hints):
    resolved, unresolved = {}, []
    for ticker in sorted(tickers):
        candidates = wp2b.symbol_candidates(ticker, symbol_index)

        def probe(code):
            try:
                frame = provider.fetch_delisted_prices("probe", code, wp2b.RESEARCH_WINDOW_START[:4] + "-01-01", wp2b.RESEARCH_WINDOW_START[:4] + "-12-31")
                return not frame.empty
            except Exception:
                return False

        try:
            result = wp2b.resolve_eodhd_symbol(ticker, name_hints.get(ticker, ticker), symbol_index,
                                               probe=probe if len(candidates) > 1 else None)
        except Exception as exc:
            print("RESOLVE_FAIL %s %s" % (ticker, redact(exc)))
            result = None
        if result is None:
            unresolved.append({"ticker": ticker, "candidates": candidates, "name_hint": name_hints.get(ticker)})
        else:
            result["name_hint"] = name_hints.get(ticker)
            resolved[ticker] = result
    return resolved, unresolved


CACHE_DIR = Path("/tmp/wp2b_cache")


def _cache_frame(key, role, frame):
    """Persist a fetched frame so an interrupted run does not refetch it.

    The cache is keyed by the RESOLVED provider symbol, never the Wikipedia
    ticker: a reused ticker (WB -> WB_OLD2) must never read another company's
    cached prices.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / ("%s.%s.parquet" % (_cache_key(key), role))
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        pass


def _cache_key(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _load_cached(key, role, columns):
    path = CACHE_DIR / ("%s.%s.parquet" % (_cache_key(key), role))
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _fetch_one(provider, ticker, symbol, no_actions):
    failed = []
    prices = _load_cached(symbol, "prices", ad.DELISTED_PRICE_FIELDS)
    if prices is None:
        try:
            prices = provider.fetch_delisted_prices(ticker, symbol, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
            _cache_frame(symbol, "prices", prices)
        except Exception as exc:
            prices = pd.DataFrame(columns=list(ad.DELISTED_PRICE_FIELDS))
            failed.append({"ticker": ticker, "kind": "price", "error": redact(exc)})
    actions = pd.DataFrame(columns=list(CORPORATE_ACTION_FIELDS))
    if not no_actions:
        actions = _load_cached(symbol, "actions", CORPORATE_ACTION_FIELDS)
        if actions is None:
            try:
                actions = provider.fetch_corporate_actions(symbol, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
                _cache_frame(symbol, "actions", actions)
            except Exception as exc:
                actions = pd.DataFrame(columns=list(CORPORATE_ACTION_FIELDS))
                failed.append({"ticker": ticker, "kind": "action", "error": redact(exc)})
    return ticker, prices, actions, failed


def fetch_all(provider, resolved, no_actions, workers=8):
    price_cache, action_cache, failures = {}, {}, []
    started = time.time()
    done = 0
    tickers = sorted(resolved)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, provider, ticker, resolved[ticker]["code"], no_actions): ticker
                   for ticker in tickers}
        for future in as_completed(futures):
            ticker, prices, actions, failed = future.result()
            price_cache[ticker] = prices
            action_cache[ticker] = actions
            failures.extend(failed)
            done += 1
            if done % 50 == 0:
                print("FETCHED %d/%d elapsed=%.0fs" % (done, len(tickers), time.time() - started))
    return price_cache, action_cache, failures


month_ends = wp2b.month_ends  # canonical helper lives in the wp2b module


def build_silver_and_gold(windows, price_cache, action_cache, window_start, window_end):
    by_security = {}
    for window in windows:
        by_security.setdefault(window.security_id, []).append(window)

    price_rows, action_rows, panel_rows, coverage = [], [], [], []
    months = month_ends(window_start, window_end)
    for security in sorted(by_security):
        raw_frame = price_cache.get(security)
        action_frame = action_cache.get(security)
        actions = wp2b.actions_from_eodhd_rows(
            action_frame.to_dict("records") if action_frame is not None else [], security)
        for row in (raw_frame.to_dict("records") if raw_frame is not None else []):
            price_rows.append({"security_id": security, "ticker": row.get("ticker"),
                               "trade_date": str(row.get("trade_date"))[:10], "raw_close": row.get("raw_close")})
        for action in actions:
            action_rows.append({"security_id": security, "ticker": security, "kind": action.kind,
                                "effective_date": action.effective_date, "numerator": action.numerator,
                                "denominator": action.denominator, "amount": action.amount})
        terminal = wp2b.TERMINAL_NO_PRICE
        terminal_return = wp2b.TERMINAL_RETURN_MISSING
        completeness = wp2b.PRICE_MISSING
        # F2: a series that begins ENTIRELY after a removal is the WRONG (reused)
        # company; quarantine the id instead of pairing it with another firm.
        quarantined = any(wp2b.series_ends_after_membership(window, raw_frame) for window in by_security[security])
        for window in sorted(by_security[security], key=lambda item: item.membership_start):
            panel_rows.extend(wp2b.monthly_panel_for_security(window, raw_frame, actions, months))
            completeness = wp2b.classify_price_completeness(window, raw_frame, window_start, window_end)
            terminal = wp2b.classify_terminal(window, raw_frame, window_end)
            terminal_return = wp2b.classify_terminal_return(window, raw_frame, window_end)
        if quarantined:
            completeness = wp2b.PRICE_IDENTITY_UNRESOLVED
        coverage.append({"security_id": security, "price_completeness": completeness,
                         "terminal_behavior": terminal, "terminal_return_status": terminal_return,
                         "quarantined_wrong_company": bool(quarantined),
                         "price_rows": int(len(raw_frame) if raw_frame is not None else 0),
                         "action_rows": len(actions)})
    prices_df = pd.DataFrame(price_rows, columns=["security_id", "ticker", "trade_date", "raw_close"])
    actions_df = pd.DataFrame(action_rows, columns=["security_id", "ticker", "kind", "effective_date", "numerator", "denominator", "amount"])
    panel_df = pd.DataFrame(panel_rows, columns=["security_id", "ticker", "snapshot_date", "membership_start",
                                                 "membership_end", "research_eligible", "available_at", "price_date", "raw_close",
                                                 "adjusted_close_pit", "has_price"])
    coverage_df = pd.DataFrame(coverage)
    return prices_df, actions_df, panel_df, coverage_df


def _price_frame_for_guard(price_cache):
    """Flatten the per-symbol price cache into one security_id-keyed frame."""
    rows = []
    for security, frame in (price_cache or {}).items():
        if frame is None or len(frame) == 0:
            continue
        for record in frame.to_dict("records"):
            rows.append({"security_id": security, "trade_date": str(record.get("trade_date"))[:10],
                         "raw_close": record.get("raw_close")})
    return pd.DataFrame(rows, columns=["security_id", "trade_date", "raw_close"])


def coverage_summary(coverage_df, unresolved, removed_ids):
    total = len(coverage_df)
    counts = Counter(coverage_df["price_completeness"].tolist()) if total else Counter()
    removed_counts = Counter(coverage_df.loc[coverage_df["security_id"].isin(removed_ids), "price_completeness"].tolist()) if total else Counter()
    by_year = {}
    if total:
        for _, row in coverage_df.iterrows():
            key = str(row["security_id"])
            by_year.setdefault(key, [])
    resolved_complete = counts.get(wp2b.PRICE_COMPLETE, 0)
    percent = round(100.0 * resolved_complete / total, 2) if total else 0.0
    removed_percent = round(100.0 * removed_counts.get(wp2b.PRICE_COMPLETE, 0) / max(1, sum(removed_counts.values())), 2)
    # F5: terminal-return observability + F2 wrong-company quarantine.
    terminal_unknown = 0
    quarantined = 0
    removed_unknown = 0
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
        "unresolved": len(unresolved),
        "complete": counts.get(wp2b.PRICE_COMPLETE, 0),
        "partial_history": counts.get(wp2b.PRICE_PARTIAL, 0),
        "missing": counts.get(wp2b.PRICE_MISSING, 0),
        "terminal_return_uncertain": counts.get(wp2b.PRICE_TERMINAL_UNCERTAIN, 0),
        "identity_unresolved": len(unresolved),
        "overall_complete_percent": percent,
        "removed_constituent_complete_percent": removed_percent,
        "removed_security_count": len(removed_ids),
        "terminal_return_unknown_count": terminal_unknown,
        "removed_without_usable_terminal_price": removed_unknown,
        "removed_without_usable_terminal_price_ids": sorted(removed_uncertain_ids),
        "wrong_company_quarantined": quarantined,
        "label_loss_percent": round(100.0 * removed_unknown / removed_total, 2),
    }


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
    name_hints = build_name_hints(bronze_hist)

    windows, assumed = wp2b.reconstruct_memberships(changes, constituents,
                                                     wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    all_tickers = sorted({window.security_id for window in windows})
    run_tickers = all_tickers[: args.limit] if args.limit else all_tickers
    print("SEED_MEMBERSHIP windows=%d securities=%d assumed_starts=%d run=%d" % (
        len(windows), len(all_tickers), len(assumed), len(run_tickers)))

    resolved, unresolved = resolve_identities(run_tickers, symbol_index, provider, name_hints)
    print("RESOLVED %d  UNRESOLVED %d" % (len(resolved), len(unresolved)))

    price_cache, action_cache, failures = fetch_all(provider, resolved, args.no_actions)
    price_rows = sum(len(frame) for frame in price_cache.values())
    action_rows = sum(len(frame) for frame in action_cache.values())
    print("PRICE rows=%d  ACTION rows=%d  failures=%d" % (price_rows, action_rows, len(failures)))

    # F3: apply the listing guard NOW that first price dates are known. A window
    # that opened before a security listed is clamped forward; an assumed
    # (unverifiable) start is flagged research_eligible=False so it can be
    # excluded from research cross-sections instead of back-projecting a
    # present-day constituent into the past.
    price_for_guard = _price_frame_for_guard(price_cache)
    first_prices = wp2b.first_price_by_security(price_for_guard)
    windows = wp2b.apply_listing_guard(windows, first_prices)
    ineligible = sum(1 for window in windows if not window.research_eligible)
    print("LISTING_GUARD windows=%d ineligible=%d" % (len(windows), ineligible))

    run_windows = [window for window in windows if window.security_id in set(resolved)]
    prices_df, actions_df, panel_df, coverage_df = build_silver_and_gold(
        run_windows, price_cache, action_cache, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)

    silver_prices = layers.write_silver_table(BRONZE_ROOT, SILVER_PRICES, prices_df, meta={"source": "eodhd"})
    silver_actions = layers.write_silver_table(BRONZE_ROOT, SILVER_ACTIONS, actions_df, meta={"source": "eodhd"})
    membership_df = pd.DataFrame([{
        "security_id": window.security_id, "ticker": window.ticker,
        "membership_start": window.membership_start, "membership_end": window.membership_end,
        "start_known": window.start_known, "research_eligible": window.research_eligible,
        "unverifiable_before": window.unverifiable_before,
        "source": "wikipedia:historical_components_sp500"}
        for window in run_windows])
    silver_membership = layers.write_silver_table(BRONZE_ROOT, SILVER_MEMBERSHIP, membership_df,
                                                  meta={"source": "wikipedia:historical_components_sp500"})

    diagnostics = wp2b.membership_diagnostics(run_windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
    removed_ids = {window.security_id for window in run_windows if window.membership_end is not None}
    summary = coverage_summary(coverage_df, unresolved, removed_ids)
    summary["membership_ineligible_assumed_starts"] = int(ineligible)
    monthly = wp2b.monthly_member_counts(run_windows, wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)

    (ARTIFACT_DIR / "unresolved_identities.json").write_text(
        json.dumps({"unresolved": unresolved, "resolved": resolved}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "price_completeness.json").write_text(
        json.dumps({"summary": summary, "per_security": coverage_df.to_dict("records")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "membership_diagnostics.json").write_text(
        json.dumps({"diagnostics": diagnostics, "monthly": monthly, "assumed_starts": assumed}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "fetch_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "layer_records.json").write_text(json.dumps({
        "silver_prices": silver_prices["version"], "silver_actions": silver_actions["version"],
        "silver_membership": silver_membership["version"],
        "price_rows": price_rows, "action_rows": action_rows, "panel_rows": len(panel_df),
        "windows": len(run_windows), "securities": len(resolved),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    print("DIAGNOSTICS windows=%d securities=%d overlaps=%d assumed=%d" % (
        diagnostics["windows"], diagnostics["securities"], len(diagnostics["overlaps"]), len(diagnostics["assumed_start_securities"])))
    print("BUILD_DONE panel_rows=%d" % len(panel_df))


if __name__ == "__main__":
    main()
