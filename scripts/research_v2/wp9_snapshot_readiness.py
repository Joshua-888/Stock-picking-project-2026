"""WP9A: read-only, non-evidentiary prospective-snapshot readiness preflight.

This command answers READY or NOT_READY for one candidate as-of date without
writing an official snapshot, fitting anything, or reading any target/outcome
value. It is a readiness check, never evidence.

Exit codes: 0 READY, 1 NOT_READY, 2 usage/error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data.availability import price_available_at, to_utc_timestamp
from src.research.fingerprints import fingerprint_file
from src.research.wp9.champion import load_champion
from src.research.wp9.contract import (
    CONTRACT_VERSION,
    Wp9Contract,
    Wp9ContractError,
    contract_freeze_commit,
    contract_freeze_timestamp,
    contract_path,
    load_wp9_contract,
)
from src.research.wp9.forward_inputs import (
    LIVE_LAYER_RECORDS_REL,
    LAYER_RECORDS_REL,
)
from src.research.wp9.storage import has_prediction_for_asof
from src.research.wp9.temporal_gate import (
    bound_price_trade_dates as certified_price_trade_dates,
    closing_utc_for_asof,
    current_utc,
    eligible_score_date_for_month,
)

CONTRACT_EXPECTED_DIGEST = "b18baa261248245a39649bba77be981d0b025a5730adbce0863f81b89a788eeb"
CONTRACT_FREEZE_COMMIT = "c96559cf2eb7c65e2c15c4f8014a77faf2a8f4d3"
EXPECTED_BRANCH = "feat/wp9-prospective-shadow"

TARGET_OUTCOME_COLUMNS = {
    "outperform_12m",
    "future_12m_excess_return",
    "future_12m_stock_return",
    "future_12m_benchmark_return",
    "future_return",
    "target_end",
    "target_start",
    "target_known_at",
    "target_observable",
    "target_censored",
    "target_censor_reason",
}

LIVE_PRICES = "wp9_live_prices"
LIVE_ACTIONS = "wp9_live_actions"
LIVE_MEMBERSHIP = "wp9_live_membership"
LIVE_BENCHMARK_PRICES = "wp9_live_benchmark_prices"
# Canonical source provenance written by the refresh builder for current-list
# membership rows whose historical add date is unknown. The readiness check
# uses the exact same value so the two paths cannot drift.
LIVE_MEMBERSHIP_START_SOURCE = "current_sp500_list_retrieval_date"


class ReadinessUsageError(RuntimeError):
    """Invalid usage; maps to exit code 2."""


def _git(args, root):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _git_is_ancestor(commit, root):
    return _git(["merge-base", "--is-ancestor", commit, "HEAD"], root) == ""


def _json(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _result(name, ok, detail=None):
    return {"check": name, "passed": bool(ok), "detail": detail}


def _load_live_records(root):
    return _json(root / LIVE_LAYER_RECORDS_REL)


def _live_table(root, name, records, key):
    version = records.get(key)
    if not version:
        raise RuntimeError("live layer records missing %s" % key)
    return layers.read_silver_table(
        root / "data" / "research_v2", name, version=str(version)
    )


def _latest_date(frame, column):
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = [str(value)[:10] for value in frame[column] if pd.notna(value)]
    return max(values) if values else None


def check_contract(root):
    try:
        contract = load_wp9_contract(root=root)
        actual_digest = fingerprint_file(contract_path(root))
        expected = CONTRACT_EXPECTED_DIGEST
        freeze_commit = contract_freeze_commit(contract, root=root)
        commit_ok = freeze_commit == CONTRACT_FREEZE_COMMIT
        committed = _git(
            [
                "diff",
                "--quiet",
                "--exit-code",
                "HEAD",
                "--",
                str(contract_path(root).relative_to(root)),
            ],
            root,
        ) == ""
        ok = (
            contract.version == CONTRACT_VERSION
            and actual_digest == expected
            and actual_digest == contract.digest
            and commit_ok
            and committed
        )
        detail = {
            "contract_version": contract.version,
            "contract_digest": contract.digest,
            "expected_digest": expected,
            "freeze_commit": freeze_commit,
            "expected_freeze_commit": CONTRACT_FREEZE_COMMIT,
            "committed_bytes_at_head": committed,
        }
        return contract, _result("contract_identity_and_bytes", ok, detail)
    except Wp9ContractError as exc:
        return None, _result("contract_identity_and_bytes", False, str(exc))


def check_freeze_reachable(root, contract):
    if contract is None:
        return _result("freeze_commit_reachable", False, "contract unavailable")
    try:
        commit = contract_freeze_commit(contract, root=root)
    except Wp9ContractError as exc:
        return _result("freeze_commit_reachable", False, str(exc))
    ok = commit == CONTRACT_FREEZE_COMMIT and _git_is_ancestor(commit, root)
    return _result(
        "freeze_commit_reachable",
        ok,
        {"commit": commit, "expected": CONTRACT_FREEZE_COMMIT},
    )


def check_champion(root, contract):
    if contract is None:
        return _result("champion_artifacts_match_contract_and_disk", False, "contract unavailable")
    try:
        champion = load_champion(root=root)
    except Exception as exc:  # noqa: BLE001
        return _result("champion_artifacts_match_contract_and_disk", False, str(exc))
    artifacts = (contract.get("champion", "artifacts") or {})
    expected = {
        name: (artifacts.get(name) or {}).get("sha256")
        for name in ("model", "preprocessor", "calibrator")
    }
    actual = {
        "model": champion.model_hash,
        "preprocessor": champion.preprocessor_hash,
        "calibrator": champion.calibrator_hash,
    }
    ok = (
        champion.freeze_id == "freeze_e62eac30df40"
        and champion.calibration == "platt_scaling"
        and actual == expected
    )
    return _result(
        "champion_artifacts_match_contract_and_disk",
        ok,
        {"expected": expected, "actual": actual},
    )


def check_branch_worktree(root):
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    dirty = []
    producing = []
    output = _git(["status", "--porcelain", "--untracked-files=no"], root) or ""
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().split(" -> ")[-1].split(" ")[0].replace("\\", "/")
        if path:
            dirty.append(path)
    producing_paths = {
        str(contract_path(root).relative_to(root)),
        "provenance/wp8/final_candidate_freeze.json",
    }
    for path in dirty:
        if path in producing_paths or path.startswith("src/research/") or path.startswith("scripts/research_v2/"):
            producing.append(path)
    ok = branch == EXPECTED_BRANCH and not producing
    return _result(
        "branch_and_producing_worktree_clean",
        ok,
        {"branch": branch, "expected_branch": EXPECTED_BRANCH, "producing_dirty": sorted(set(producing))},
    )


def check_no_duplicate(asof, root):
    exists = has_prediction_for_asof(asof, root=root)
    return _result("no_duplicate_official_snapshot", not exists, {"exists": exists})


def check_after_freeze(asof, root, contract):
    if contract is None:
        return _result("asof_after_contract_freeze", False, "contract unavailable")
    try:
        freeze = contract_freeze_timestamp(contract, root=root)
        asof_stamp = pd.Timestamp(asof).tz_localize("UTC")
        freeze_stamp = pd.Timestamp(freeze).tz_convert("UTC")
        ok = asof_stamp > freeze_stamp
        return _result("asof_after_contract_freeze", bool(ok), {"freeze": freeze})
    except Wp9ContractError as exc:
        return _result("asof_after_contract_freeze", False, str(exc))
    except (TypeError, ValueError) as exc:
        return _result("asof_after_contract_freeze", False, str(exc))


def _trade_dates(root):
    dates = set()
    try:
        dates.update(certified_price_trade_dates(root=root))
    except Exception:  # noqa: BLE001
        pass
    try:
        records = _load_live_records(root)
        prices = _live_table(root, LIVE_PRICES, records, "silver_prices")
        dates.update(str(value)[:10] for value in prices["trade_date"] if pd.notna(value))
    except Exception:  # noqa: BLE001
        pass
    return sorted(dates)


def check_monthly_cadence(asof, root):
    dates = _trade_dates(root)
    if not dates:
        return _result("monthly_cadence_last_eligible_score_date", False, "no trade dates")
    try:
        eligible = eligible_score_date_for_month(asof, dates, require_coverage_complete=True)
    except Exception as exc:  # noqa: BLE001
        return _result("monthly_cadence_last_eligible_score_date", False, str(exc))
    return _result(
        "monthly_cadence_last_eligible_score_date",
        asof == eligible,
        {"asof": asof, "expected_eligible": eligible},
    )


def check_not_future(asof):
    try:
        close = closing_utc_for_asof(asof)
    except Exception as exc:  # noqa: BLE001
        return _result("asof_close_elapsed", False, str(exc))
    now = current_utc()
    return _result("asof_close_elapsed", close <= now, {"close": close.isoformat(), "now": now.isoformat()})


def check_live_coverage(asof, root):
    try:
        records = _load_live_records(root)
        prices = _live_table(root, LIVE_PRICES, records, "silver_prices")
        benchmark = _live_table(root, LIVE_BENCHMARK_PRICES, records, "silver_benchmark_prices")
        membership = _live_table(root, LIVE_MEMBERSHIP, records, "silver_membership")
    except Exception as exc:  # noqa: BLE001
        return _result("live_coverage_through_asof", False, str(exc))
    latest_price = _latest_date(prices, "trade_date")
    latest_benchmark = _latest_date(benchmark, "trade_date")
    membership_ok = True
    membership_provenance_error = None
    if membership is None or membership.empty:
        membership_ok = False
        membership_provenance_error = "membership_empty"
    else:
        if "membership_start" not in membership.columns:
            membership_ok = False
            membership_provenance_error = "membership_start_column_missing"
        else:
            starts = membership["membership_start"].astype(str)
            if not (starts == str(asof)).all():
                membership_ok = False
                membership_provenance_error = "membership_start_not_equal_asof"
            if membership_ok:
                source_col = "membership_start_source"
                if source_col not in membership.columns:
                    membership_ok = False
                    membership_provenance_error = "membership_start_source_missing"
                else:
                    sources = membership[source_col].astype(str)
                    if not (sources == LIVE_MEMBERSHIP_START_SOURCE).all():
                        membership_ok = False
                        membership_provenance_error = "membership_start_source_invalid"
    ok = bool(
        latest_price
        and latest_benchmark
        and latest_price >= asof
        and latest_benchmark >= asof
        and membership_ok
    )
    return _result(
        "live_coverage_through_asof",
        ok,
        {
            "latest_price": latest_price,
            "latest_benchmark": latest_benchmark,
            "membership_rows": int(len(membership)) if membership is not None else 0,
            "membership_start_source_required": LIVE_MEMBERSHIP_START_SOURCE,
            "membership_provenance_error": membership_provenance_error,
            "required_min": asof,
        },
    )


def check_fundamental_availability(asof, root):
    try:
        from src.research.data.edgar_binding import load_and_verify_edgar_fundamentals

        binding = _json(root / "provenance" / "wp8" / "input_binding.json")
        fundamentals = load_and_verify_edgar_fundamentals(root, binding)
    except Exception as exc:  # noqa: BLE001
        return _result("fundamental_availability_through_asof", False, str(exc))
    if fundamentals is None or fundamentals.empty or "available_at" not in fundamentals.columns:
        return _result("fundamental_availability_through_asof", False, "no fundamentals")
    close = closing_utc_for_asof(asof)
    timestamps = fundamentals["available_at"].map(to_utc_timestamp)
    future = timestamps > close
    if bool(future.any()):
        return _result("fundamental_availability_through_asof", False, {"future_rows": int(future.sum())})
    available = timestamps <= close
    return _result(
        "fundamental_availability_through_asof",
        bool(available.any()),
        {"latest_available_at": _latest_date(fundamentals, "available_at")},
    )


def check_layer_records(root):
    problems = []
    try:
        certified = _json(root / LAYER_RECORDS_REL)
        for key in ("silver_prices", "silver_actions", "silver_membership"):
            if not certified.get(key):
                problems.append("certified_%s_missing" % key)
    except Exception as exc:  # noqa: BLE001
        problems.append(str(exc))
    try:
        live = _load_live_records(root)
        for key in (
            "silver_prices",
            "silver_actions",
            "silver_membership",
            "silver_benchmark_prices",
            "silver_benchmark_actions",
        ):
            if not live.get(key):
                problems.append("live_%s_missing" % key)
    except Exception as exc:  # noqa: BLE001
        problems.append(str(exc))
    return _result("source_layer_records_well_formed", not problems, problems)


def check_no_target_future(asof, root):
    close = closing_utc_for_asof(asof)
    try:
        records = _load_live_records(root)
        tables = {
            "prices": _live_table(root, LIVE_PRICES, records, "silver_prices"),
            "actions": _live_table(root, LIVE_ACTIONS, records, "silver_actions"),
            "membership": _live_table(root, LIVE_MEMBERSHIP, records, "silver_membership"),
            "benchmark": _live_table(root, LIVE_BENCHMARK_PRICES, records, "silver_benchmark_prices"),
        }
    except Exception as exc:  # noqa: BLE001
        return _result("no_target_or_future_data", False, str(exc))
    target_found = []
    future_found = []
    date_cols = ("trade_date", "effective_date", "available_at")
    for name, frame in tables.items():
        present = {str(c).lower() for c in frame.columns}
        matched = sorted(present.intersection(TARGET_OUTCOME_COLUMNS))
        if matched:
            target_found.append({"table": name, "columns": matched})
        for col in date_cols:
            if col in frame.columns:
                late = frame[col].map(price_available_at) > close
                if bool(late.any()):
                    future_found.append({"table": name, "column": col, "rows": int(late.sum())})
    ok = not target_found and not future_found
    return _result("no_target_or_future_data", ok, {"target_found": target_found, "future_found": future_found})


def check_no_empty_payloads(root):
    problems = []
    try:
        records = _load_live_records(root)
        for key, name in (
            ("silver_prices", LIVE_PRICES),
            ("silver_actions", LIVE_ACTIONS),
            ("silver_benchmark_prices", LIVE_BENCHMARK_PRICES),
        ):
            frame = _live_table(root, name, records, key)
            if frame.empty:
                problems.append(key)
    except Exception as exc:  # noqa: BLE001
        problems.append(str(exc))
    return _result("no_empty_provider_payloads", not problems, problems)





def _secret_values() -> List[str]:
    """Return secret literals to search for, without ever printing them."""
    values = []
    token = os.environ.get("EODHD_API_TOKEN")
    if token:
        values.append(token)
    secrets_path = ROOT / ".a0proj" / "secrets.env"
    if secrets_path.is_file():
        text = secrets_path.read_text(encoding="utf-8")
        match = re.search(r"^EODHD_API_TOKEN=(.*)$", text, re.M)
        if match:
            candidate = match.group(1).strip().strip('"').strip("'")
            if candidate and candidate not in values:
                values.append(candidate)
    return values


def check_secret_scan(root):
    """Deterministically detect secret material in tracked text.

    Reports file paths only, never the matched bytes. Loader-code literals that
    parse the EODHD token variable name are not treated as secrets.
    """
    matched = []
    pattern_hits = _git(
        [
            "grep",
            "--cached",
            "-I",
            "-n",
            "-E",
            r"sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}",
            ":!.a0proj/**",
            ":!data/research_v2/bronze/**",
        ],
        root,
    )
    if pattern_hits:
        matched.extend(line for line in pattern_hits.splitlines() if line)
    for token in _secret_values():
        if not token:
            continue
        token_hits = _git(
            [
                "grep",
                "--cached",
                "-I",
                "-n",
                "-F",
                "--",
                token,
                ":!.a0proj/**",
                ":!data/research_v2/bronze/**",
            ],
            root,
        )
        if token_hits:
            matched.extend(line for line in token_hits.splitlines() if line)
    ok = not matched
    return _result("secret_scan_clean", ok, {"matched": sorted(set(matched))})


def run_readiness(asof, root):
    if not isinstance(asof, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", asof):
        raise ReadinessUsageError("as-of must be a canonical YYYY-MM-DD string")
    try:
        pd.Timestamp(asof)
    except (TypeError, ValueError) as exc:
        raise ReadinessUsageError("as-of is not parseable") from exc

    checks = []
    contract, result = check_contract(root)
    checks.append(result)
    checks.append(check_freeze_reachable(root, contract))
    checks.append(check_champion(root, contract))
    checks.append(check_branch_worktree(root))
    checks.append(check_no_duplicate(asof, root))
    checks.append(check_after_freeze(asof, root, contract))
    checks.append(check_monthly_cadence(asof, root))
    checks.append(check_not_future(asof))
    checks.append(check_live_coverage(asof, root))
    checks.append(check_fundamental_availability(asof, root))
    checks.append(check_layer_records(root))
    checks.append(check_no_target_future(asof, root))
    checks.append(check_no_empty_payloads(root))
    checks.append(check_secret_scan(root))
    ready = all(bool(c["passed"]) for c in checks)
    return {
        "readiness": "READY" if ready else "NOT_READY",
        "as_of": asof,
        "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_outcome_access": False,
        "official_side_effects": "none",
        "checks": checks,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description="WP9 snapshot readiness preflight")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_readiness(args.as_of, root=Path(args.root))
    except ReadinessUsageError as exc:
        print("USAGE_ERROR %s" % exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        print("ERROR %s" % exc)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["readiness"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
