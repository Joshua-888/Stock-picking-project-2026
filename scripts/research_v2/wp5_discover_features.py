"""WP5 live build: POINT-IN-TIME FEATURE PANEL + CANDIDATE VARIABLE DISCOVERY.

Run (research mode, real data only):

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp5_discover_features.py

What this script is
-------------------
WP5 materialises a point-in-time feature panel from the certified WP2C gold panel,
WP2B silver prices/actions, the SPY benchmark and the SEC EDGAR silver
fundamentals, then characterises every candidate feature with cross-sectional
monthly IC, dependence-aware inference, fixed quantiles, temporal stability,
missingness, Benjamini-Hochberg FDR, redundancy clustering and a placebo battery.
It outputs a CANDIDATE_FEATURE_UNIVERSE for WP6 - never a final feature set.

Inputs (all certified, never rebuilt here)
------------------------------------------
* ``dataset_dbaa77445b38`` / ``ac294282d8e949f2``  - WP2C gold research panel
* ``target_set_888f68d1cfd0`` / ``f244f86c22b2c2a4`` - WP3 gold 12m targets
* ``wp2b_sp500_pit_prices_silver`` / ``a0699674d7650583``
* ``wp2b_sp500_pit_actions_silver`` / ``68bdbdf4ea75a862``
* ``benchmark_gold_SPY`` / ``a0ab5bd2382518d4``      - SPY benchmark prices
* ``bronze/wp3_spy_prices_actions``                 - SPY corporate actions
* ``silver/edgar_fundamentals/*``                    - SEC EDGAR filed fundamentals
* ``artifacts/research/wp4/modern_signals.parquet``  - Keyes benchmark (reference only)
* ``artifacts/research/wp4/historical_signals.parquet``

Honesty rules
-------------
* discovery consumes DEVELOPMENT rows only (pre-embargo, observable labels); the
  locked holdout is never read and any holdout row in the slice fails the build;
* the PRIMARY IC is the cross-sectional monthly series; pooled row IC is secondary;
* inference is dependence-aware (HAC + block bootstrap), not naive row p-values;
* multiple testing is controlled with Benjamini-Hochberg FDR over the full set;
* nothing is zero-filled, estimated or fabricated; a missing value stays missing;
* the Keyes track is measured with the same machinery and is never privileged.
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
from src.research.discovery import builder
from src.research.discovery.catalog import (
    DISCOVERY_VERSION,
    FEATURE_NAMES,
    FEATURE_PANEL_VERSION,
    HANDOFF_VERSION,
    DiscoveryConfig,
    catalog_payload,
)
from src.research.discovery.io import discovery_payload, write_experiment
from src.research.discovery.scorecard import classification_counts
from src.research.holdout import embargo_cutoff, locked_holdout
from src.research.ids import experiment_id
from src.research.modes import ResearchMode, current_git_commit

BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp5"
PROVENANCE_DIR = ROOT / "provenance" / "wp5"

# ── Certified upstream identifiers (pinned; never a lexicographic guess) ──────
DATASET_ID = "dataset_dbaa77445b38"
PANEL_VERSION = "ac294282d8e949f2"
TARGET_ID = "target_set_888f68d1cfd0"
TARGET_VERSION = "f244f86c22b2c2a4"
FEATURE_SET_ID = "feature_set_56361533cc1b"
SILVER_PRICES = "a0699674d7650583"
SILVER_ACTIONS = "68bdbdf4ea75a862"
SPY_BENCHMARK_VERSION = "a0ab5bd2382518d4"
CIK_MAPPING_PATH = ROOT / "artifacts" / "research" / "wp4" / "edgar_cik_mapping.json"
SPY_BRONZE_GLOB = "data/research_v2/bronze/wp3_spy_prices_actions/*/raw.json"
WP4_MODERN_SIGNALS = ROOT / "artifacts" / "research" / "wp4" / "modern_signals.parquet"
WP4_HISTORICAL_SIGNALS = ROOT / "artifacts" / "research" / "wp4" / "historical_signals.parquet"


class BuildError(RuntimeError):
    """Raised when the WP5 build cannot proceed honestly."""


# ── Input loading (mirrors the WP4 build; nothing is rebuilt) ─────────────────

def load_panel():
    frame = layers.read_silver_table(BRONZE_ROOT, "sp500_pit_research_panel_v1", version=PANEL_VERSION)
    frame = frame.rename(columns={"snapshot_date": "feature_asof"})
    required = ("security_id", "ticker", "feature_asof", "price_date")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise BuildError("certified panel is missing column(s): %s" % ", ".join(missing))
    keep = [name for name in ("security_id", "ticker", "feature_asof", "price_date",
                              "target_observable") if name in frame.columns]
    return frame.loc[:, keep].copy()


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


def load_keyes_benchmark():
    """WP4 signal artifacts, reused verbatim as the Keyes reference (never rebuilt)."""
    modern = pd.read_parquet(WP4_MODERN_SIGNALS) if WP4_MODERN_SIGNALS.is_file() else None
    historical = pd.read_parquet(WP4_HISTORICAL_SIGNALS) if WP4_HISTORICAL_SIGNALS.is_file() else None
    return modern, historical


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    config = DiscoveryConfig(bootstrap_seed=args.seed)
    holdout = locked_holdout()

    panel = load_panel()
    targets = load_targets()
    prices = load_prices()
    actions = load_actions()
    benchmark_prices = load_benchmark_prices()
    benchmark_actions = load_benchmark_actions()
    fundamentals = load_fundamentals()
    cik_by_ticker, cik_payload = load_cik_by_ticker()
    modern, historical = load_keyes_benchmark()

    print("INPUTS panel=%d targets=%d prices=%d actions=%d fundamentals=%d mapped_tickers=%d keyes_modern=%s keyes_hist=%s" % (
        len(panel), len(targets), len(prices), len(actions), len(fundamentals), len(cik_by_ticker),
        None if modern is None else len(modern), None if historical is None else len(historical)))

    bundle = builder.run_discovery(
        panel, targets, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions, config=config,
        keyes_variable_frame=modern, modern_signals=modern, historical_signals=historical,
    )

    # ── Deterministic experiment binding ──────────────────────────────────────
    commit = current_git_commit() or "unknown"
    binding = discovery_payload(
        DATASET_ID, TARGET_ID, FEATURE_SET_ID, catalog_payload(), config.to_dict(), commit,
        holdout_id=holdout.holdout_id,
    )
    experiment = experiment_id(binding)

    limitations = [
        "WP5 is discovery only: it emits a candidate universe, not a final feature set and no weighted score",
        "classifications are research categories, not a ranking",
        "SEC EDGAR companyfacts begin around 2009, so early fundamental coverage is bounded and honestly missing",
        "abnormal_volume is UNAVAILABLE because the certified PIT price table carries no volume column",
        "pooled row-level rank IC is reported as a secondary figure only (12-month labels overlap ~11.8x)",
        "the locked holdout (2022-01-01..2025-08-31) was never read during discovery",
    ]

    scorecards = bundle["scorecards"]
    handoff = builder.assemble_handoff(
        bundle, DATASET_ID, TARGET_ID, FEATURE_SET_ID, experiment,
        keyes_reference=bundle["keyes"].get("summary") if isinstance(bundle["keyes"], dict) else {},
        notes=limitations,
    )

    # ── Artifacts ─────────────────────────────────────────────────────────────
    scorecard_payload = {"scorecards": scorecards,
                         "classification_counts": classification_counts(scorecards)}
    fdr_payload = {"results": bundle["fdr_results"], "summary": bundle["fdr_summary"]}
    inertia = {
        "rank_correlation_matrix": bundle["rank_matrix"].where(pd.notna(bundle["rank_matrix"]), None).to_dict(),
        "ic_series_correlation": bundle["ic_matrix"].where(pd.notna(bundle["ic_matrix"]), None).to_dict(),
        "clusters": bundle["clusters"],
    }
    summary_payload = {
        "experiment_id": experiment,
        "discovery_version": DISCOVERY_VERSION,
        "feature_panel_version": FEATURE_PANEL_VERSION,
        "handoff_version": HANDOFF_VERSION,
        "dataset_id": DATASET_ID,
        "target_id": TARGET_ID,
        "feature_set_id": FEATURE_SET_ID,
        "holdout_id": holdout.holdout_id,
        "git_commit": commit,
        "mode": ResearchMode.RESEARCH_V2.value,
        "panel_summary": bundle["panel_summary"],
        "development_rows": bundle["development_rows"],
        "candidate_count": len(scorecards),
        "classification_counts": classification_counts(scorecards),
        "fdr_summary": bundle["fdr_summary"],
        "placebo": bundle["placebo"],
        "config": config.to_dict(),
        "cik_exact_mappings": int(len(cik_by_ticker)),
        "cik_unmapped": int(cik_payload.get("unmapped", 0)),
        "cik_ambiguous": int(cik_payload.get("ambiguous", 0)),
        "known_limitations": limitations,
    }
    quality_payload = {
        "features": [bundle["per_feature"][name]["quality"] for name in FEATURE_NAMES],
        "note": "quality is factual; nothing is discarded",
    }

    reference = write_experiment(
        ROOT, binding,
        artifacts={
            "discovery_summary.json": summary_payload,
            "candidate_feature_universe.json": handoff,
            "feature_scorecard.json": scorecard_payload,
            "feature_quality.json": quality_payload,
            "fdr.json": fdr_payload,
            "redundancy.json": inertia,
            "keyes_benchmark.json": bundle["keyes"],
            "placebo.json": bundle["placebo"],
        },
        provenance={
            "%s.json" % experiment: {
                "experiment_id": experiment,
                "dataset_id": DATASET_ID,
                "target_id": TARGET_ID,
                "feature_set_id": FEATURE_SET_ID,
                "holdout_id": holdout.holdout_id,
                "git_commit": commit,
                "mode": ResearchMode.RESEARCH_V2.value,
                "discovery_version": DISCOVERY_VERSION,
                "embargo_cutoff": str(embargo_cutoff())[:19],
                "sources": [
                    "WP2C_GOLD_PANEL", "WP3_GOLD_TARGETS", "WP2B_SILVER_PRICES", "WP2B_SILVER_ACTIONS",
                    "BENCHMARK_GOLD_SPY", "WP3_SPY_BRONZE_ACTIONS", "SEC_EDGAR", "WP4_KEYES_SIGNALS",
                ],
                "synthetic_data_status": None,
                "known_limitations": limitations,
            },
        },
    )

    print("WP5_DISCOVERY_DONE experiment=%s development_rows=%d candidates=%d counts=%s placebo_stop=%s" % (
        reference["experiment_id"], bundle["development_rows"], len(scorecards),
        classification_counts(scorecards), bundle["placebo"].get("stop")))
    if bundle["placebo"].get("stop"):
        print("WP5 WARNING: a placebo looked predictive - investigate before trusting any candidate")


if __name__ == "__main__":
    main()
