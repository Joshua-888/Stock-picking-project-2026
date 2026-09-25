"""WP4 live build: KEYES VARIABLE VALUES + HISTORICAL/MODERN SIGNALS + DIAGNOSTICS.

Run (research mode, real data only):

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp4_build_keyes.py

What this script is
-------------------
WP3 declared the Keyes CONTRACT and the fundamental availability rules but never
computed a fundamental VALUE. WP4 turns the certified WP3 gold panel and targets
into real point-in-time Keyes variable values, evaluates the two research tracks
and produces the diagnostics, fidelity scorecard and placebo battery.

Inputs (all certified, never rebuilt here)
------------------------------------------
* ``dataset_dbaa77445b38`` / ``ac294282d8e949f2``  - WP2C gold research panel
* ``target_set_888f68d1cfd0`` / ``f244f86c22b2c2a4`` - WP3 gold 12m targets
* ``feature_set_56361533cc1b``                     - WP3 feature set manifest
* ``wp2b_sp500_pit_prices_silver`` / ``a0699674d7650583``
* ``wp2b_sp500_pit_actions_silver`` / ``68bdbdf4ea75a862``
* ``benchmark_gold_SPY`` / ``a0ab5bd2382518d4``      - SPY benchmark prices
* ``bronze/wp3_spy_prices_actions``                 - SPY corporate actions
* ``silver/edgar_fundamentals/*``                    - SEC EDGAR filed fundamentals

Honesty rules
-------------
* real data only; there is no synthetic fallback in research mode;
* a variable that has no real value for an observation stays UNAVAILABLE -
  it is never zero-filled and never estimated;
* the X12 leg is a genuinely named ``X12_PROXY`` (revenue growth) and is never
  reported as Keyes X12;
* the historical rule set uses explicit simultaneous thresholds with no quota;
* the modern composite uses equal-weight signed ranks - no hand-tuned weights;
* the two tracks are persisted separately and never merged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data.manifesting import config_fingerprint
from src.research.ids import experiment_id
from src.research.immutability import save_immutable
from src.research.modes import ResearchMode, current_git_commit
from src.research.keyes import (
    DIAGNOSTIC_VERSION,
    DEFINITION_VERSION,
    SIGNAL_VERSION,
    CompositeConfig,
    ComputationConfig,
    VARIABLE_SPECS,
    X5,
    X6,
    X8,
    X9,
    X12_PROXY,
    component_diagnostics,
    composite_diagnostics,
    compute_keyes_variables,
    concentration_and_turnover,
    direction_payload,
    fidelity_table,
    historical_signals,
    historical_threshold_payload,
    modern_composite,
    placebo_checks,
    qualification_summary,
    replication_status,
    sector_component_stability,
    yearly_component_stability,
)

BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp4"
PROVENANCE_DIR = ROOT / "provenance" / "wp4"

# ── Certified upstream identifiers (pinned; never a lexicographic guess) ──────
DATASET_ID = "dataset_dbaa77445b38"
PANEL_VERSION = "ac294282d8e949f2"
TARGET_ID = "target_set_888f68d1cfd0"
TARGET_VERSION = "f244f86c22b2c2a4"
FEATURE_SET_ID = "feature_set_56361533cc1b"
SILVER_PRICES = "a0699674d7650583"
SILVER_ACTIONS = "68bdbdf4ea75a862"
SPY_BENCHMARK_VERSION = "a0ab5bd2382518d4"
CIK_MAPPING_PATH = ARTIFACT_DIR / "edgar_cik_mapping.json"
SPY_BRONZE_GLOB = "data/research_v2/bronze/wp3_spy_prices_actions/*/raw.json"

UPSTREAM_VARIABLES = (X5, X6, X8, X9, X12_PROXY)
MODERN_ONLY_VARIABLES = ("mom_6m", "mom_12m", "beta", "market_cap")


class BuildError(RuntimeError):
    """Raised when the WP4 build cannot proceed honestly."""


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _available_at_bounds(fundamentals):
    stamps = pd.to_datetime(fundamentals["available_at"], errors="coerce", utc=True)
    return str(stamps.min())[:10], str(stamps.max())[:10]


# ── Input loading ─────────────────────────────────────────────────────────────

def load_panel():
    """Certified WP2C gold panel, projected to the columns the engine needs."""
    frame = layers.read_silver_table(BRONZE_ROOT, "sp500_pit_research_panel_v1", version=PANEL_VERSION)
    frame = frame.rename(columns={"snapshot_date": "feature_asof"})
    required = ("security_id", "ticker", "feature_asof", "price_date")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise BuildError("certified panel is missing column(s): %s" % ", ".join(missing))
    return frame.loc[:, ["security_id", "ticker", "feature_asof", "price_date"]].copy()


def load_targets():
    path = BRONZE_ROOT / "gold" / "sp500_pit_targets_v1" / TARGET_VERSION / "data.parquet"
    if not path.is_file():
        raise BuildError("certified WP3 targets are missing: %s" % path)
    return pd.read_parquet(path)


def load_prices():
    return layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_prices_silver", version=SILVER_PRICES)


def load_actions():
    return layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_actions_silver", version=SILVER_ACTIONS)


def load_benchmark_prices():
    frame = layers.read_silver_table(BRONZE_ROOT, "benchmark_gold_SPY", version=SPY_BENCHMARK_VERSION)
    frame = frame.copy()
    frame["security_id"] = "SPY"
    frame["ticker"] = "SPY"
    return frame.loc[:, ["security_id", "ticker", "trade_date", "raw_close"]]


def load_benchmark_actions():
    import glob

    matches = sorted(glob.glob(str(ROOT / SPY_BRONZE_GLOB)))
    if not matches:
        raise BuildError("SPY corporate-action bronze is missing (%s)" % SPY_BRONZE_GLOB)
    payload = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
    actions = payload.get("actions") or []
    frame = pd.DataFrame(actions)
    for column in ("kind", "effective_date", "numerator", "denominator", "amount"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["security_id"] = "SPY"
    return frame.loc[:, ["security_id", "kind", "effective_date", "numerator", "denominator", "amount"]]


def load_fundamentals():
    """Concatenate every certified EDGAR silver company; no value is invented."""
    base = BRONZE_ROOT / "silver" / "edgar_fundamentals"
    files = sorted(path for path in base.glob("*/data.parquet"))
    if not files:
        raise BuildError("no EDGAR silver fundamentals are present: %s" % base)
    columns = ["cik", "field", "value", "fiscal_period_start", "fiscal_period_end",
               "accession", "available_at", "form"]
    frames = [pd.read_parquet(path, columns=columns) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["cik", "field", "fiscal_period_end", "value", "accession"])
    return frame


def load_cik_by_ticker():
    if not CIK_MAPPING_PATH.is_file():
        raise BuildError("EDGAR CIK mapping is missing: %s" % CIK_MAPPING_PATH)
    payload = json.loads(CIK_MAPPING_PATH.read_text(encoding="utf-8"))
    mapping = payload.get("mapping") or {}
    by_ticker = {}
    for ticker, record in mapping.items():
        cik = record.get("cik")
        if record.get("match") == "EXACT" and cik is not None:
            by_ticker[str(ticker).upper()] = str(int(cik)).zfill(10)
    return by_ticker, payload


# ── Build ─────────────────────────────────────────────────────────────────────

def build_spec_payload(config, benchmark_note):
    return {
        "signal_version": SIGNAL_VERSION,
        "definition_version": DEFINITION_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "historical": historical_threshold_payload(),
        "modern": direction_payload(),
        "config": config.to_dict(),
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "target_version": TARGET_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "benchmark_note": benchmark_note,
    }


def variable_mapping_payload():
    return {
        "definition_version": DEFINITION_VERSION,
        "variables": {name: VARIABLE_SPECS[name].to_dict() for name in UPSTREAM_VARIABLES},
        "modern_only": list(MODERN_ONLY_VARIABLES),
        "x12_note": VARIABLE_SPECS[X12_PROXY].fidelity_reason,
    }


def run_diagnostics(variable_frame, targets, historical, composite):
    components = component_diagnostics(variable_frame, targets)
    composite_entry, merged = composite_diagnostics(composite, targets, variable_frame=variable_frame)
    yearly = yearly_component_stability(variable_frame, targets)
    sector = sector_component_stability(variable_frame, targets, {})
    qualification, qualification_by_year = qualification_summary(historical, targets)
    concentration, turnover = concentration_and_turnover(historical, composite)
    composite_entry = dict(composite_entry)
    composite_entry["qualification_by_year"] = qualification_by_year.to_dict("records")
    composite_entry["concentration"] = concentration
    composite_entry["turnover"] = turnover
    return {
        "components": components,
        "composite": composite_entry,
        "yearly": yearly,
        "sector": sector,
        "qualification": qualification,
        "qualification_by_year": qualification_by_year,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    config = ComputationConfig()
    panel = load_panel()
    targets = load_targets()
    prices = load_prices()
    actions = load_actions()
    benchmark_prices = load_benchmark_prices()
    benchmark_actions = load_benchmark_actions()
    fundamentals = load_fundamentals()
    cik_by_ticker, cik_payload = load_cik_by_ticker()
    available_min, available_max = _available_at_bounds(fundamentals)

    print("INPUTS panel=%d targets=%d prices=%d actions=%d fundamentals=%d (avail %s..%s) mapped_tickers=%d" % (
        len(panel), len(targets), len(prices), len(actions), len(fundamentals),
        available_min, available_max, len(cik_by_ticker)))

    variable_frame, coverage = compute_keyes_variables(
        panel, prices, actions, fundamentals, cik_by_ticker, benchmark_prices, benchmark_actions, config,
    )
    coverage["edgar_fundamentals_rows"] = int(len(fundamentals))
    coverage["edgar_available_min"] = available_min
    coverage["edgar_available_max"] = available_max
    coverage["cik_exact_mappings"] = int(len(cik_by_ticker))
    coverage["cik_unmapped"] = int(cik_payload.get("unmapped", 0))
    coverage["cik_ambiguous"] = int(cik_payload.get("ambiguous", 0))

    historical = historical_signals(variable_frame, expected_rows=int(coverage["observations"]))
    composite = modern_composite(variable_frame, CompositeConfig())

    diagnostics = run_diagnostics(variable_frame, targets, historical, composite)
    fidelity = fidelity_table(coverage)
    status = replication_status(fidelity)
    placebo = placebo_checks(composite, targets, seed=args.seed)
    placebo["replication_status"] = status

    spec_payload = build_spec_payload(config, "SPY benchmark from certified benchmark_gold_SPY silver + WP3 SPY bronze actions")
    mapping_payload = variable_mapping_payload()

    # ── Deterministic experiment binding ──────────────────────────────────────
    commit = current_git_commit() or "unknown"
    config_fp = config_fingerprint(config.to_dict())
    experiment_payload = {
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "feature_set_id": FEATURE_SET_ID,
        "definition_version": DEFINITION_VERSION,
        "signal_version": SIGNAL_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "config": config.to_dict(),
    }
    experiment = experiment_id(experiment_payload)

    # ── Persist canonical artifacts (deterministic, regenerated in place) ─────
    write_canonical(ARTIFACT_DIR / "keyes_spec.json", spec_payload)
    write_canonical(ARTIFACT_DIR / "variable_mapping.json", mapping_payload)
    write_canonical(ARTIFACT_DIR / "coverage.json", coverage)
    write_canonical(ARTIFACT_DIR / "fidelity_table.json", fidelity.to_dict("records"))
    write_canonical(ARTIFACT_DIR / "placebo.json", placebo)
    write_canonical(ARTIFACT_DIR / "sector_stability.json", diagnostics["sector"])
    write_canonical(ARTIFACT_DIR / "diagnostics.json", {
        "components": diagnostics["components"].to_dict("records"),
        "composite": diagnostics["composite"],
        "qualification": diagnostics["qualification"],
    })
    diagnostics["yearly"].to_csv(ARTIFACT_DIR / "yearly_stability.csv", index=False)
    diagnostics["qualification_by_year"].to_csv(ARTIFACT_DIR / "qualification_by_year.csv", index=False)
    historical.to_parquet(ARTIFACT_DIR / "historical_signals.parquet", index=False)
    composite.to_parquet(ARTIFACT_DIR / "modern_signals.parquet", index=False)

    summary = {
        "experiment_id": experiment,
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "target_version": TARGET_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "definition_version": DEFINITION_VERSION,
        "signal_version": SIGNAL_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "git_commit": commit,
        "config_fingerprint": config_fp,
        "replication_status": status,
        "observations": int(coverage["observations"]),
        "historical_qualified": int(historical["qualified"].sum()),
        "historical_qualification_rate": float(historical["qualified"].mean()) if len(historical) else None,
        "modern_rows": int(len(composite)),
        "modern_scored": int(composite["composite"].notna().sum()),
        "variable_coverage": coverage["variables"],
        "fidelity": fidelity.to_dict("records"),
        "placebo_flags": {
            "shuffled_retains_systematic_power": placebo["shuffled_retains_systematic_power"],
            "feature_timestamp_violations": placebo["feature_timestamp_violations"],
            "impossible_signal_date_violations": placebo["impossible_signal_date_violations"],
        },
    }
    write_canonical(ARTIFACT_DIR / "wp4_run_summary.json", summary)

    # ── Immutable provenance record ───────────────────────────────────────────
    provenance = {
        "experiment_id": experiment,
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "target_version": TARGET_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "git_commit": commit,
        "config_fingerprint": config_fp,
        "mode": ResearchMode.RESEARCH_V2.value,
        "signal_version": SIGNAL_VERSION,
        "definition_version": DEFINITION_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "replication_status": status,
        "sources": [
            "WP2C_GOLD_PANEL", "WP3_GOLD_TARGETS", "WP2B_SILVER_PRICES", "WP2B_SILVER_ACTIONS",
            "BENCHMARK_GOLD_SPY", "WP3_SPY_BRONZE_ACTIONS", "SEC_EDGAR",
        ],
        "known_limitations": [
            "Keyes X12 has no historical point-in-time analyst/Value Line source; X12_PROXY (five-year revenue growth) is used and never reported as X12",
            "SEC EDGAR companyfacts start around 2009, so early-universe fundamental variables are UNAVAILABLE, not estimated",
            "CONSOLIDATED: fundamental variables are only as complete as the bounded EDGAR ingestion; missingness is observable",
            "only genuinely observable 12m outcomes (target_observable=true) contribute to any diagnostic",
        ],
        "synthetic_data_status": None,
    }
    save_immutable(PROVENANCE_DIR / ("%s.json" % experiment), provenance)
    write_canonical(PROVENANCE_DIR / "latest_experiment.json", provenance)

    print("WP4_BUILD_DONE experiment=%s status=%s obs=%d qualified=%d modern_scored=%d" % (
        experiment, status, summary["observations"], summary["historical_qualified"], summary["modern_scored"]))


if __name__ == "__main__":
    main()
