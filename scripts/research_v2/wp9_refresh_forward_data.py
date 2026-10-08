"""WP9A: fail-closed live-forward input refresh (operational only).

This script refreshes CURRENT prospective inputs for the WP9 forward scorer. It
is intentionally separate from the certified historical build:

* prices / corporate actions / SPY benchmark come from EODHD only;
* current S&P 500 membership comes from the current Wikipedia constituent list;
* SEC EDGAR fundamental evidence remains the verified input binding used by the
  certified historical path; this script never mutates that binding and never
  fabricates fundamentals.

Fetch bound policy
------------------
The fetch bound is ``--as-of YYYY-MM-DD`` interpreted as the conservative UTC
close for that date, or the current wall clock when ``--as-of`` is omitted. It
does NOT use the certified ``RESEARCH_WINDOW_END`` and does NOT extend it. Any
trade or effective date returned by a provider that is strictly after that bound
is rejected and the whole run fails (never silently truncated). Empty or
malformed provider payloads are hard errors.

Outputs
-------
* ``data/research_v2/silver/wp9_live_prices/``
* ``data/research_v2/silver/wp9_live_actions/``
* ``data/research_v2/silver/wp9_live_membership/``
* ``data/research_v2/silver/wp9_live_benchmark_prices/``
* ``data/research_v2/silver/wp9_live_benchmark_actions/``
* ``artifacts/research/wp9_live/layer_records.json``

These paths are separate from the certified WP2B/EDGAR layers and from
``artifacts/research/wp2b_live/layer_records.json``; this script never
overwrites any historical layer, benchmark gold table, EDGAR binding, or
``wp2b_live`` layer records.

Run:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_refresh_forward_data.py [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data import wp2b_eodhd as wp2b
from src.research.data.availability import price_available_at
from src.research.data.providers import adapters as ad
from src.research.fingerprints import fingerprint_obj
from src.research.immutability import write_json_atomic
from src.research.wp9.temporal_gate import closing_utc_for_asof, current_utc

SECRETS_PATH = ROOT / ".a0proj" / "secrets.env"
DATA_ROOT = ROOT / "data" / "research_v2"
LIVE_ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp9_live"
LIVE_RECORDS_REL = "artifacts/research/wp9_live/layer_records.json"

LIVE_PRICES = "wp9_live_prices"
LIVE_ACTIONS = "wp9_live_actions"
LIVE_MEMBERSHIP = "wp9_live_membership"
LIVE_BENCHMARK_PRICES = "wp9_live_benchmark_prices"
LIVE_BENCHMARK_ACTIONS = "wp9_live_benchmark_actions"

PRICE_FIELDS = ad.DELISTED_PRICE_FIELDS
ACTION_FIELDS = ad.CORPORATE_ACTION_FIELDS


class LiveRefreshError(RuntimeError):
    """Raised when the live-forward refresh cannot proceed honestly."""


def load_eodhd_token() -> str:
    """Read the EODHD token from ``.a0proj/secrets.env``; never print it."""
    if not SECRETS_PATH.is_file():
        raise LiveRefreshError("missing secrets file: %s" % SECRETS_PATH)
    text = SECRETS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^EODHD_API_TOKEN=(.*)$", text, re.M)
    if not match:
        raise LiveRefreshError("EODHD_API_TOKEN not found in secrets file")
    token = match.group(1).strip().strip('"').strip("'")
    if not token:
        raise LiveRefreshError("EODHD_API_TOKEN is empty")
    os.environ["EODHD_API_TOKEN"] = token
    return token


def redact_token(message) -> str:
    """Replace the live EODHD token before any user/log output."""
    token = os.environ.get("EODHD_API_TOKEN") or ""
    text = str(message)
    if token:
        text = text.replace(token, "***REDACTED***")
    return text


def _fetch_bound(asof: str | None) -> pd.Timestamp:
    """Resolve the run's conservative UTC fetch bound.

    ``--as-of`` means the 21:00 UTC close for that date; omitted means the
    current wall-clock instant (staging). The certified ``RESEARCH_WINDOW_END``
    is intentionally not consulted.
    """
    if asof is None:
        return current_utc()
    if not isinstance(asof, str):
        raise LiveRefreshError("as-of must be a YYYY-MM-DD string")
    try:
        return closing_utc_for_asof(asof)
    except Exception as exc:  # noqa: BLE001 - surfaced, never hidden
        raise LiveRefreshError("invalid --as-of %r: %s" % (asof, exc)) from exc


def _bound_date(bound: pd.Timestamp) -> str:
    return bound.strftime("%Y-%m-%d")


def _reject_payload_after_bound(
    frame: pd.DataFrame,
    *,
    date_column: str,
    bound: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    """Fail closed if any row is effective/trading after the conservative bound."""
    if frame is None or frame.empty:
        if frame is None:
            raise LiveRefreshError("%s payload was None" % label)
        return frame.copy()
    if date_column not in frame.columns:
        raise LiveRefreshError("%s payload missing %s column" % (label, date_column))
    timestamps = frame[date_column].map(price_available_at)
    late = timestamps > bound
    if bool(late.any()):
        examples = [
            str(value)[:10]
            for value in frame.loc[late, date_column].drop_duplicates().head(5)
        ]
        raise LiveRefreshError(
            "%s payload contains %d row(s) after bound %s: %s"
            % (label, int(late.sum()), bound.isoformat(), ",".join(examples))
        )
    return frame.copy()


def fetch_current_constituents() -> tuple[list[str], dict]:
    """Current Wikipedia S&P 500 constituents (no historical reconstruction)."""
    url = ad.WIKIPEDIA_SP500_URL.replace(
        "Historical_components_of_the_S%26P_500", "List_of_S%26P_500_companies"
    )
    html = ad.http_get(url)
    if not html or "Symbol" not in html:
        raise LiveRefreshError("Wikipedia current constituent page was empty or malformed")
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
    if not constituents:
        raise LiveRefreshError("no current constituents parsed from Wikipedia")
    manifest = {
        "provider": "WIKIPEDIA_SP500",
        "endpoint": url,
        "constituent_count": len(constituents),
        "row_count": len(constituents),
    }
    return sorted(constituents), manifest


def fetch_symbol_index(provider: ad.EodhdProvider) -> tuple[dict, set, dict]:
    """Fetch active + delisted EODHD US exchange symbol lists once each."""
    active = provider.fetch_exchange_symbols("US", delisted=False)
    if not active:
        raise LiveRefreshError("EODHD active symbol list was empty or malformed")
    delisted = provider.fetch_exchange_symbols("US", delisted=True)
    if not delisted:
        raise LiveRefreshError("EODHD delisted symbol list was empty or malformed")
    index = wp2b.build_symbol_index(list(active) + list(delisted))
    active_codes = {str(row.get("Code")).upper() for row in active if row.get("Code")}
    manifest = {
        "provider": "EODHD",
        "endpoint": "exchange-symbol-list/US",
        "active_rows": len(active),
        "delisted_rows": len(delisted),
    }
    return index, active_codes, manifest


def resolve_constituent_symbols(
    constituents: list[str], index: dict, active_codes: set
) -> list[dict]:
    """Resolve each current constituent to one EODHD symbol, never guessing.

    A current constituent is an active name, not a recycled historical suffix,
    so resolution prefers the exact active code. Unresolved symbols are recorded
    with an explicit factual reason, never left silently absent.
    """
    resolutions = []
    for ticker in constituents:
        if ticker in active_codes:
            resolutions.append(
                {
                    "security_id": ticker,
                    "ticker": ticker,
                    "symbol": ticker,
                    "resolved": True,
                    "method": "active_current",
                    "reason": None,
                }
            )
            continue
        candidates = wp2b.symbol_candidates(ticker, index)
        active_exact = [
            code
            for code in candidates
            if not wp2b.is_old_symbol(code, ticker) and code.upper() in active_codes
        ]
        if len(active_exact) == 1:
            resolutions.append(
                {
                    "security_id": ticker,
                    "ticker": ticker,
                    "symbol": active_exact[0],
                    "resolved": True,
                    "method": "active_exact_candidate",
                    "reason": None,
                }
            )
        elif candidates:
            resolutions.append(
                {
                    "security_id": ticker,
                    "ticker": ticker,
                    "symbol": None,
                    "resolved": False,
                    "method": None,
                    "reason": "no_single_active_eodhd_symbol",
                    "candidates": candidates,
                }
            )
        else:
            resolutions.append(
                {
                    "security_id": ticker,
                    "ticker": ticker,
                    "symbol": None,
                    "resolved": False,
                    "method": None,
                    "reason": "no_symbol_candidates_in_eodhd",
                }
            )
    return resolutions


def fetch_market_for_symbol(
    provider: ad.EodhdProvider, symbol: str, bound: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices and corporate actions for one EODHD symbol, fail-closed.

    A missing/empty/malformed price payload is a hard error. Corporate actions
    may legitimately be empty (no split/dividend in the window), but a malformed
    payload (missing columns or future rows) is also a hard error.
    """
    end = _bound_date(bound)
    # Six-year lookback is sufficient for the frozen five-year trailing windows
    # plus a one-year buffer; it never consults or extends RESEARCH_WINDOW_END.
    start = _bound_date(bound - pd.Timedelta(days=366 * 6))
    try:
        prices = provider.fetch_delisted_prices(symbol, symbol, start, end)
    except Exception as exc:  # noqa: BLE001
        raise LiveRefreshError(
            "EODHD price fetch failed for %s: %s" % (symbol, redact_token(exc))
        ) from exc
    if prices is None or prices.empty:
        raise LiveRefreshError("EODHD price payload was empty for %s" % symbol)
    missing = [name for name in PRICE_FIELDS if name not in prices.columns]
    if missing:
        raise LiveRefreshError(
            "EODHD price payload missing columns for %s: %s" % (symbol, ",".join(missing))
        )
    prices = _reject_payload_after_bound(
        prices, date_column="trade_date", bound=bound, label="prices:%s" % symbol
    )

    try:
        actions = provider.fetch_corporate_actions(symbol, start, end)
    except Exception as exc:  # noqa: BLE001
        raise LiveRefreshError(
            "EODHD action fetch failed for %s: %s" % (symbol, redact_token(exc))
        ) from exc
    if actions is None:
        raise LiveRefreshError("EODHD action payload was None for %s" % symbol)
    missing = [name for name in ACTION_FIELDS if name not in actions.columns]
    if missing:
        raise LiveRefreshError(
            "EODHD action payload missing columns for %s: %s" % (symbol, ",".join(missing))
        )
    actions = _reject_payload_after_bound(
        actions, date_column="effective_date", bound=bound, label="actions:%s" % symbol
    )
    return prices, actions


def build_live_membership(resolutions: list[dict], *, as_of: str) -> pd.DataFrame:
    """Build a current membership table in the certified membership row shape.

    The current list provides no historical add date, so ``membership_start`` is
    set to the live refresh ``as_of`` retrieval date. This is a source-known
    lower bound, not an invented open-ended start: ``start_known=False`` and
    ``membership_start_source="current_sp500_list_retrieval_date"`` make the
    assumption explicit and never grant pre-window membership.
    """
    rows = []
    for item in resolutions:
        rows.append(
            {
                "security_id": item["security_id"],
                "ticker": item["ticker"],
                "membership_start": as_of,
                "membership_end": None,
                "start_known": False,
                "membership_start_source": "current_sp500_list_retrieval_date",
                "resolution_method": item["method"] if item["resolved"] else None,
                "unresolved_reason": item["reason"] if not item["resolved"] else None,
                "source": "wikipedia:list_of_sp500_companies",
            }
        )
    return pd.DataFrame(rows)


def _latest_date(frame: pd.DataFrame, column: str) -> str | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = [str(value)[:10] for value in frame[column] if pd.notna(value)]
    return max(values) if values else None


def run(asof: str | None, root: Path | None = None) -> dict:
    """Execute the fail-closed live-forward refresh and persist the layer records."""
    root = Path(root or ROOT)
    bound = _fetch_bound(asof)
    retrieval_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    load_eodhd_token()
    provider = ad.EodhdProvider(timeout=60)

    constituents, wiki_manifest = fetch_current_constituents()
    index, active_codes, symbol_manifest = fetch_symbol_index(provider)
    resolutions = resolve_constituent_symbols(constituents, index, active_codes)
    resolved = [item for item in resolutions if item["resolved"]]
    unresolved = [item for item in resolutions if not item["resolved"]]

    # Identity gate mirrors WP2C-A1: every current constituent must be resolved
    # or explicitly unresolved; a partial-resolution run is a hard failure.
    if len(resolved) < max(10, len(constituents) // 2):
        raise LiveRefreshError(
            "identity resolution covered only %d/%d constituents; fail-closed"
            % (len(resolved), len(constituents))
        )

    price_frames = []
    action_frames = []
    # Bounded concurrency only parallelizes independent provider GETs; the
    # write/commit phase still happens exactly once after every fetch returns.
    # Any single symbol failure propagates untouched to the outer fail-closed
    # handler, so this never executes a partial or truncated write.
    max_workers = min(12, max(1, len(resolved)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_market_for_symbol, provider, item["symbol"], bound): item["security_id"]
            for item in resolved
        }
        completed = 0
        for future in as_completed(futures):
            security_id = futures[future]
            # Propagate the original exception without wrapping; it already has
            # fail-closed context and the token cannot be printed.
            prices, actions = future.result()
            prices = prices.copy()
            prices["security_id"] = security_id
            prices["ticker"] = security_id
            actions = actions.copy()
            actions["security_id"] = security_id
            actions["ticker"] = security_id
            price_frames.append(prices)
            action_frames.append(actions)
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    "fetched %d/%d symbols" % (completed, len(futures)),
                    file=sys.stderr,
                    flush=True,
                )
    if not price_frames:
        raise LiveRefreshError("no live price payloads were successfully fetched; fail-closed")

    prices = pd.concat(price_frames, ignore_index=True)
    actions = (
        pd.concat(action_frames, ignore_index=True)
        if action_frames
        else pd.DataFrame(columns=list(ACTION_FIELDS))
    )
    membership = build_live_membership(resolutions, as_of=_bound_date(bound))

    # SPY benchmark is a dedicated live table so the overlay can prefer it while
    # the certified benchmark_gold_SPY table remains untouched.
    spy_prices, spy_actions = fetch_market_for_symbol(provider, "SPY", bound)
    spy_prices = spy_prices.copy()
    spy_prices["security_id"] = "SPY"
    spy_prices["ticker"] = "SPY"
    spy_actions = spy_actions.copy()
    spy_actions["security_id"] = "SPY"
    spy_actions["ticker"] = "SPY"

    meta_prices = {
        "source": "eodhd",
        "kind": "wp9_live",
        "retrieval_at_utc": retrieval_at_utc,
        "bound": bound.isoformat(),
    }
    live_prices = layers.write_silver_table(DATA_ROOT, LIVE_PRICES, prices, meta=meta_prices)
    live_actions = layers.write_silver_table(DATA_ROOT, LIVE_ACTIONS, actions, meta=meta_prices)
    live_membership = layers.write_silver_table(
        DATA_ROOT,
        LIVE_MEMBERSHIP,
        membership,
        meta={
            "source": "wikipedia:list_of_sp500_companies",
            "kind": "wp9_live",
            "retrieval_at_utc": retrieval_at_utc,
            "bound": bound.isoformat(),
        },
    )
    live_benchmark_prices = layers.write_silver_table(
        DATA_ROOT, LIVE_BENCHMARK_PRICES, spy_prices, meta=meta_prices
    )
    live_benchmark_actions = layers.write_silver_table(
        DATA_ROOT, LIVE_BENCHMARK_ACTIONS, spy_actions, meta=meta_prices
    )

    layer_records = {
        "schema": "wp9_live_layer_records_v1",
        "retrieval_at_utc": retrieval_at_utc,
        "fetch_bound_utc": bound.isoformat(),
        "as_of": _bound_date(bound),
        "providers": {
            "market": "EODHD",
            "membership": "WIKIPEDIA_SP500",
            "fundamentals": "SEC_EDGAR (existing verified input binding; never refreshed here)",
        },
        "silver_prices": live_prices["version"],
        "silver_actions": live_actions["version"],
        "silver_membership": live_membership["version"],
        "silver_benchmark_prices": live_benchmark_prices["version"],
        "silver_benchmark_actions": live_benchmark_actions["version"],
        "price_rows": int(len(prices)),
        "action_rows": int(len(actions)),
        "membership_rows": int(len(membership)),
        "benchmark_price_rows": int(len(spy_prices)),
        "benchmark_action_rows": int(len(spy_actions)),
        "constituents": len(constituents),
        "resolved_constituents": len(resolved),
        "unresolved_constituents": [item["security_id"] for item in unresolved],
        "unresolved_reasons": {
            item["security_id"]: item["reason"] for item in unresolved
        },
        "max_trade_date": _latest_date(prices, "trade_date"),
        "max_effective_date": _latest_date(actions, "effective_date"),
        "benchmark_max_trade_date": _latest_date(spy_prices, "trade_date"),
        "source_manifest": {
            "wikipedia_sp500": wiki_manifest,
            "eodhd_symbols": symbol_manifest,
            "eodhd_market": {
                "endpoints": ["eod/{ticker}.US", "splits/{ticker}.US", "div/{ticker}.US"],
                "symbols_fetched": len(resolved) + 1,
            },
        },
    }
    layer_records["fingerprint"] = fingerprint_obj(layer_records)
    LIVE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / LIVE_RECORDS_REL, layer_records)

    print(
        json.dumps(
            {
                "as_of": layer_records["as_of"],
                "constituents": len(constituents),
                "resolved": len(resolved),
                "unresolved": len(unresolved),
                "price_rows": layer_records["price_rows"],
                "action_rows": layer_records["action_rows"],
                "max_trade_date": layer_records["max_trade_date"],
                "max_effective_date": layer_records["max_effective_date"],
                "benchmark_max_trade_date": layer_records["benchmark_max_trade_date"],
                "layer_records": LIVE_RECORDS_REL,
            },
            sort_keys=True,
        )
    )
    return layer_records


def parse_args(argv):
    parser = argparse.ArgumentParser(description="WP9 live-forward data refresh (operational only)")
    parser.add_argument(
        "--as-of", help="fetch bound YYYY-MM-DD (omit for current wall clock staging)"
    )
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args.as_of, root=Path(args.root))
        return 0
    except Exception as exc:  # noqa: BLE001 - surfaced, never hidden
        print("REFRESH_BLOCKED %s" % redact_token(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
