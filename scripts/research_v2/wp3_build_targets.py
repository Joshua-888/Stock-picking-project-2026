"""WP3 live build: TARGETS + FEATURE REGISTRY + TEMPORAL OBSERVABILITY.

Run (research mode, real data only):

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp3_build_targets.py

Consumes the certified WP2C gold panel ``dataset_dbaa77445b38``
(``ac294282d8e949f2``) under the readiness policy, forms the benchmark-relative
12-month forward labels, attaches the temporal observability metadata, and
persists the target contract, the feature registry, the V1 disposition table and
the immutable ``FeatureSetManifest`` / target artifact.

Real data only; a synthetic fallback is impossible. This script declares WHAT
the labels are and WHEN they become usable; it fits no model and computes no
feature VALUE for a model. The benchmark leg (SPY) is fetched from EODHD and
bronzed; it is never read from a provider-adjusted (look-ahead) column.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data import layers
from src.research.data import wp2b_eodhd as wp2b
from src.research.data import provenance_ledger
from src.research.data.manifesting import build_and_persist_gold, config_fingerprint
from src.research.data.providers import adapters as ad
from src.research.features import (
    LEGACY_ROA_ROIC_NOTE,
    V2_ROA_DEFINITION,
    V2_ROIC_DEFINITION,
    build_feature_set,
    build_initial_registry,
    v1_disposition,
)
from src.research.features.keyes import KEYES_VARIABLES, TRACK_HISTORICAL, TRACK_MODERN
from src.research.fingerprints import fingerprint_obj
from src.research.immutability import save_immutable
from src.research.modes import ResearchMode
from src.research.targets import (
    CLASSIFICATION_TARGET,
    CONTINUOUS_TARGET,
    TARGET_VERSION,
    TargetContract,
    annotate_observability,
    build_targets,
)

SECRETS_PATH = ROOT / ".a0proj" / "secrets.env"
BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp3"
PROVENANCE_DIR = ROOT / "provenance" / "wp3"
GOLD_NAME = "sp500_pit_targets_v1"
BENCHMARK_GOLD_NAME = "wp3_spy_benchmark_v1"
CACHE_DIR = Path("/tmp/wp3_cache")

# The certified WP2C dataset this build consumes (never a lexicographic guess).
DATASET_ID = "dataset_dbaa77445b38"
GOLD_PANEL_VERSION = "ac294282d8e949f2"
CORPORATE_ACTION_FIELDS = ("ticker", "kind", "effective_date", "numerator", "denominator", "amount")


def load_eodhd_token():
    """Read EODHD_API_TOKEN from .a0proj/secrets.env (never printed/logged)."""
    text = SECRETS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^EODHD_API_TOKEN=(.*)$", text, re.M)
    if not match:
        raise SystemExit("EODHD_API_TOKEN not found in secrets file")
    token = match.group(1).strip().strip('"').strip("'")
    if not token:
        raise SystemExit("EODHD_API_TOKEN is empty")
    os.environ["EODHD_API_TOKEN"] = token


def redact(message):
    token = os.environ.get("EODHD_API_TOKEN") or ""
    text = str(message)
    return text.replace(token, "***REDACTED***") if token else text


def write_canonical(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_panel():
    """Read the certified WP2C gold panel at its pinned content address."""
    return layers.read_silver_table(BRONZE_ROOT, wp2b.DATASET_NAME, version=GOLD_PANEL_VERSION)


def layer_fingerprint(name, version):
    """Full lower-case SHA-256 fingerprint recorded in a layer's provenance sidecar."""
    sidecar = BRONZE_ROOT / "silver" / name / version / "provenance.json"
    if not sidecar.is_file():
        sidecar = BRONZE_ROOT / "gold" / name / version / "provenance.json"
    return json.loads(sidecar.read_text(encoding="utf-8"))["fingerprint"]


def load_declared_layers():
    """Read the EXACT silver versions WP2C declared (never a lexicographic guess)."""
    declared = json.loads((ROOT / "artifacts" / "research" / "wp2b_live" / "layer_records.json").read_text())
    prices = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_prices_silver", version=declared["silver_prices"])
    actions = layers.read_silver_table(BRONZE_ROOT, "wp2b_sp500_pit_actions_silver", version=declared["silver_actions"])
    return prices, actions, declared


def build_spy_benchmark(provider):
    """Fetch/bronze SPY prices + actions (full research window) into a gold frame.

    Returns ``(benchmark_frame, benchmark_actions, bronze_fingerprint)`` where the
    frame has canonical ``ticker/trade_date/raw_close`` columns. SPY is fetched
    with the same delisted-capable endpoint used for equities so the benchmark
    leg carries the same raw-close semantics (no provider-adjusted close).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices_path = CACHE_DIR / "SPY.prices.parquet"
    actions_path = CACHE_DIR / "SPY.actions.parquet"
    if prices_path.is_file():
        frame = pd.read_parquet(prices_path)
    else:
        frame = provider.fetch_delisted_prices("SPY", "SPY", wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
        frame.to_parquet(prices_path, index=False)
    if actions_path.is_file():
        action_frame = pd.read_parquet(actions_path)
    else:
        action_frame = provider.fetch_corporate_actions("SPY", wp2b.RESEARCH_WINDOW_START, wp2b.RESEARCH_WINDOW_END)
        action_frame.to_parquet(actions_path, index=False)

    bronze = layers.write_bronze_bytes(
        BRONZE_ROOT, "wp3_spy_prices_actions",
        json.dumps({
            "prices": frame.astype(str).to_dict("records"),
            "actions": action_frame.astype(str).to_dict("records"),
        }).encode("utf-8"),
        ext="json", meta={"endpoint": "eod/SPY.US + dividends/splits", "source": "eodhd"},
    )
    benchmark = frame[["ticker", "trade_date", "raw_close"]].copy()
    benchmark["ticker"] = "SPY"
    benchmark["trade_date"] = benchmark["trade_date"].astype(str).str.slice(0, 10)
    return benchmark, action_frame, bronze


def spy_actions_as_objects(action_frame):
    return wp2b.actions_from_eodhd_rows(action_frame.to_dict("records"), "SPY")


def stock_actions_as_objects(actions):
    """Convert every silver action row into a PIT ``CorporateAction`` keyed by security_id."""
    objects = []
    for security_id, chunk in actions.groupby("security_id"):
        objects.extend(wp2b.actions_from_eodhd_rows(chunk.to_dict("records"), security_id))
    return objects


def persist_target_contract():
    contract = TargetContract(dataset_id=DATASET_ID)
    payload = contract.to_dict()
    save_immutable(PROVENANCE_DIR / "target_contract.json", payload)
    write_canonical(ARTIFACT_DIR / "target_contract.json", payload)
    return contract, payload


def persist_feature_artifacts():
    registry = build_initial_registry(include_macro=False).seal()
    registry_payload = {
        "registry_version": registry.version,
        "feature_count": len(registry.features),
        "enabled_count": len(registry.enabled_features()),
        "by_category": registry.by_category(),
        "features": [definition.to_dict() for definition in registry.sorted_features()],
    }
    write_canonical(ARTIFACT_DIR / "feature_registry.json", registry_payload)

    disposition = v1_disposition()
    write_canonical(ARTIFACT_DIR / "v1_disposition.json", disposition)

    keyes_payload = {
        "variables": dict(KEYES_VARIABLES),
        "tracks": {
            TRACK_HISTORICAL: "exact Keyes definitions; X12 has no PIT source -> X12_PROXY only",
            TRACK_MODERN: "benchmark-relative modern analogues; separate track, never merged",
        },
        "x12_note": (
            "no historical point-in-time analyst/Value Line expected-appreciation source exists; "
            "any stand-in is named X12_PROXY and never reported as Keyes X12"
        ),
        "qualification_rule": (
            "explicit simultaneous regression thresholds via KeyesRuleSet; NO forced top-30% quota"
        ),
    }
    write_canonical(ARTIFACT_DIR / "keyes_tracks.json", keyes_payload)
    return registry, registry_payload, disposition, keyes_payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spy-fetch", action="store_true", help="reuse the cached SPY frame only")
    args = parser.parse_args(argv)

    load_eodhd_token()
    provider = ad.EodhdProvider(timeout=60)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    print("PANEL rows=%d securities=%d" % (len(panel), panel["security_id"].nunique()))
    prices, actions, declared = load_declared_layers()
    print("SILVER prices=%d actions=%d versions=%s" % (
        len(prices), len(actions), declared["silver_prices"]))

    benchmark, benchmark_actions_frame, spy_bronze = build_spy_benchmark(provider)
    print("SPY rows=%d actions=%d" % (len(benchmark), len(benchmark_actions_frame)))

    contract, contract_payload = persist_target_contract()
    registry, registry_payload, disposition, keyes_payload = persist_feature_artifacts()

    target_rows = build_targets(
        panel, prices, benchmark,
        stock_actions=stock_actions_as_objects(actions),
        benchmark_actions=spy_actions_as_objects(benchmark_actions_frame),
        horizon_months=contract.horizon_months,
    )
    # ``annotate_observability`` preserves row order and count; attach positionally
    # rather than a key merge, because the certified WP2C panel legitimately holds
    # overlapping membership windows for a few securities (e.g. AGN) and is
    # therefore NOT unique on (security_id, snapshot_date). No silent de-duplication.
    observability = annotate_observability(target_rows)
    target_rows = target_rows.reset_index(drop=True)
    target_rows["target_known_at"] = observability["target_known_at"].to_numpy()
    duplicate_keys = int(target_rows.duplicated(subset=["security_id", "feature_asof"]).sum())
    print("TARGETS duplicate_(security_id,feature_asof)_rows=%d (overlapping memberships, preserved)" % duplicate_keys)
    observable = int(target_rows["target_observable"].sum())
    censored = int(target_rows["target_censored"].sum())
    print("TARGETS rows=%d observable=%d censored=%d" % (len(target_rows), observable, censored))

    # FeatureSetManifest binds the registry version to this dataset immutably.
    from src.research.modes import current_git_commit
    feature_set = build_feature_set(DATASET_ID, current_git_commit() or "unknown", registry=registry)
    save_immutable(PROVENANCE_DIR / ("%s.json" % feature_set.feature_set_id), feature_set.to_dict())
    write_canonical(ARTIFACT_DIR / "feature_set_manifest.json", feature_set.to_dict())

    known_limitations = [
        "labels are benchmark-relative vs SPY; a stock whose terminal observation is unknown is censored, never labelled 0",
        "EODHD publishes NO delisting returns; a security disappearing before the horizon end has an UNKNOWN terminal return and its 12m label is censored (target_observable=false)",
        "a row is trainable only when target_known_at <= model_date; overlapping 12m labels must be purged by WP6/WP7 using this metadata",
        "Wikipedia membership source omits several historical removals (BSC, Enron, Circuit City, WaMu, old GM); inherited from the WP2C universe limitation, not repaired here",
        "macro features are declared but DISABLED; no FRED/ALFRED vintage is wired in this package",
        "fundamental feature VALUES are not computed in WP3; the registry declares availability rules only",
    ]
    source_fingerprints = {
        "wp2c_gold_panel": layer_fingerprint(wp2b.DATASET_NAME, GOLD_PANEL_VERSION),
        "silver_prices": layer_fingerprint("wp2b_sp500_pit_prices_silver", declared["silver_prices"]),
        "silver_actions": layer_fingerprint("wp2b_sp500_pit_actions_silver", declared["silver_actions"]),
        "wp3_spy_bronze": spy_bronze["fingerprint"],
    }
    persisted = build_and_persist_gold(
        GOLD_NAME,
        target_rows,
        mode=ResearchMode.RESEARCH_V2,
        universe_id="sp500_pit_wikipedia_eodhd_v1",
        period_start=str(panel["snapshot_date"].min())[:10],
        period_end=str(panel["snapshot_date"].max())[:10],
        sources=["EODHD", "WIKIPEDIA_SP500", "SEC_EDGAR(declared-only)"],
        source_fingerprints=source_fingerprints,
        pit_status="partially_point_in_time",
        known_limitations=known_limitations,
        config=None,
        notes="WP3 benchmark-relative 12m target + observability panel (dataset %s)" % DATASET_ID,
        source_versions={
            "wp2c_dataset_id": DATASET_ID,
            "wp2c_dataset_fingerprint": GOLD_PANEL_VERSION,
            "silver_prices": declared["silver_prices"],
            "silver_actions": declared["silver_actions"],
            "feature_set_id": feature_set.feature_set_id,
            "target_id": contract.target_id,
        },
        security_master_version=json.loads((ROOT / "provenance" / "wp2c" / "security_master_version.json").read_text())["security_master_version"],
        universe_version=json.loads((ROOT / "provenance" / "wp2c" / "universe_version.json").read_text())["universe_version"],
    )
    ledger = provenance_ledger.persist_manifest(persisted["manifest"] if not isinstance(persisted["manifest"], dict) else persisted["manifest"],
                                                root=ROOT / "provenance",
                                                extra={
                                                    "target_id": contract.target_id,
                                                    "target_version": TARGET_VERSION,
                                                    "feature_set_id": feature_set.feature_set_id,
                                                    "observable_labels": observable,
                                                    "censored_labels": censored,
                                                })

    summary = {
        "dataset_id": persisted["dataset_id"],
        "dataset_fingerprint": persisted["table"]["fingerprint"],
        "gold_table_path": persisted["table"]["gold_table_path"],
        "rows": int(len(target_rows)),
        "observable_labels": observable,
        "censored_labels": censored,
        "target_id": contract.target_id,
        "target_version": TARGET_VERSION,
        "continuous_target": CONTINUOUS_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "feature_set_id": feature_set.feature_set_id,
        "feature_registry_version": registry.version,
        "feature_count": len(registry.features),
        "features_enabled": len(registry.enabled_features()),
        "features_by_category": registry.by_category(),
        "roa_definition": V2_ROA_DEFINITION,
        "roic_definition": V2_ROIC_DEFINITION,
        "legacy_note": LEGACY_ROA_ROIC_NOTE,
        "spy_rows": int(len(benchmark)),
        "spy_actions": int(len(benchmark_actions_frame)),
        "ledger": ledger,
    }
    write_canonical(ARTIFACT_DIR / "wp3_summary.json", summary)
    print("WP3_DONE dataset_id=%s rows=%d observable=%d censored=%d target_id=%s feature_set_id=%s" % (
        persisted["dataset_id"], len(target_rows), observable, censored,
        contract.target_id, feature_set.feature_set_id))


if __name__ == "__main__":
    main()
