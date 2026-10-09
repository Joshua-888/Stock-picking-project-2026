"""WP9A.2 dashboard truth-layer export + deterministic attribution sample.

Run with the research runtime:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9a2_dashboard_export.py

What this script does
---------------------
1. Builds a deterministic, label-free, development-period feature sample for
   ``2007-01-31..2020-12-31`` using the EXACT certified WP5 feature engine.
   It reads only identifier/time columns from the certified panel; it never
   reads a target, target-observable, future-return, or holdout-label column.
2. Loads the frozen WP8 champion with the existing hash-verified apply-only
   loader and computes model-native 13-feature importance plus deterministic
   tree-path family/feature attribution. The champion is never refit.
3. Writes the nine versioned dashboard contracts under
   ``artifacts/research/wp9/dashboard/`` with exact metric lineage.

The emit path is development output, not a WP9 official snapshot and not a
mutation of any immutable research provenance file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dashboard import schemas  # noqa: E402
from src.research.dashboard import summary as dashboard_summary  # noqa: E402
from src.research.dashboard.contract import (  # noqa: E402
    freshness_completeness,
    latest_shadow_ranking,
    official_snapshot_history,
)
from src.research.dashboard.integrity import audit_status  # noqa: E402
from src.research.dashboard.identity import artifact_identity  # noqa: E402
from src.research.dashboard import attribution, explain  # noqa: E402
from src.research.wp9.champion import load_champion  # noqa: E402

BRONZE_ROOT = ROOT / "data" / "research_v2"
OUTPUT_DIR = ROOT / "artifacts" / "research" / "wp9" / "dashboard"

# Certified identities used to reconstruct the feature-only historical sample.
# These match the WP6 final-model inputs used to train freeze_e62eac30df40.
PANEL_NAME = "sp500_pit_research_panel_v1"
PANEL_VERSION = "553ac17bf5d4d63f"
SAMPLE_DATE_MIN = "2007-01-31"
SAMPLE_DATE_MAX = "2020-12-31"
DASHBOARD_ATTRIBUTION_SEED = 20261009
DEFAULT_SAMPLE_SIZE = 4000


class ExportError(RuntimeError):
    """Raised when the dashboard export cannot be built honestly."""


def _load_wp5():
    path = ROOT / "scripts" / "research_v2" / "wp5_discover_features.py"
    spec = importlib.util.spec_from_file_location("wp9a2_wp5_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp9a2_wp5_engine"] = module
    spec.loader.exec_module(module)
    # Pin to the exact WP6 final dataset panel version. Everything else is the
    # certified WP5 loader body, reused verbatim (prices/actions/SPY/fundamentals).
    module.PANEL_VERSION = PANEL_VERSION
    return module


def build_feature_only_panel(sample_size, seed):
    """Deterministic, month-stratified, label-free feature sample.

    Only ``security_id``, ``ticker``, ``feature_asof``, and ``price_date`` are
    read from the certified panel. Feature values are computed by the frozen WP5
    engine in one pass for the sampled rows only.
    """
    from src.research.data.layers import layer_root
    from src.research.discovery.panel import build_feature_panel

    # Column-pruned read: only identifier/time columns ever enter this
    # feature-only sample. ``target_observable`` and all target/censoring
    # columns are never materialised, even transiently.
    panel_path = layer_root(BRONZE_ROOT, "silver", PANEL_NAME) / PANEL_VERSION / "data.parquet"
    if not panel_path.is_file():
        raise ExportError("certified panel parquet is missing: %s" % panel_path)
    keep_columns = ["security_id", "ticker", "snapshot_date", "price_date"]
    panel = pd.read_parquet(panel_path, columns=keep_columns)
    panel = panel.rename(columns={"snapshot_date": "feature_asof"})

    forbidden_columns = [name for name in panel.columns if name not in
                         ("security_id", "ticker", "feature_asof", "price_date")]
    if forbidden_columns:
        raise ExportError("unexpected non-identifier column(s) in feature-only read: %s"
                          % ", ".join(sorted(forbidden_columns)))
    panel["feature_asof"] = panel["feature_asof"].astype(str).str.slice(0, 10)
    mask = (panel["feature_asof"] >= SAMPLE_DATE_MIN) & (panel["feature_asof"] <= SAMPLE_DATE_MAX)
    panel = panel.loc[mask].copy()
    panel = panel.drop_duplicates(
        subset=["security_id", "feature_asof"], keep="first"
    ).sort_values(["feature_asof", "security_id"], kind="mergesort").reset_index(drop=True)

    months = sorted(panel["feature_asof"].unique())
    if not months:
        raise ExportError("no historical panel rows in %s..%s" % (SAMPLE_DATE_MIN, SAMPLE_DATE_MAX))
    per_month, extra = divmod(sample_size, len(months))
    if per_month == 0:
        raise ExportError("sample_size %d is smaller than month count %d" % (sample_size, len(months)))
    allocations = {month: per_month for month in months}
    for month in months[:extra]:
        allocations[month] += 1

    frames = []
    for month, count in allocations.items():
        chunk = panel.loc[panel["feature_asof"] == month]
        if count >= len(chunk):
            frames.append(chunk)
        else:
            frames.append(chunk.sample(n=count, random_state=seed))
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sort_values(
        ["feature_asof", "security_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(sample) != sample_size:
        raise ExportError(
            "deterministic sample size is %d; expected %d" % (len(sample), sample_size)
        )

    engine = _load_wp5()
    prices = engine.load_prices()
    actions = engine.load_actions()
    benchmark_prices = engine.load_benchmark_prices()
    benchmark_actions = engine.load_benchmark_actions()
    fundamentals = engine.load_fundamentals()
    cik_by_ticker, cik_payload = engine.load_cik_by_ticker()

    feature_frame, feature_summary = build_feature_panel(
        sample,
        prices,
        actions,
        fundamentals,
        cik_by_ticker,
        benchmark_prices,
        benchmark_actions,
        config=None,
        restrict_to_development=False,
        allow_locked_holdout=False,
    )
    required = ["security_id", "ticker", "feature_asof"] + list(schemas.FEATURE_ORDER)
    missing = [name for name in required if name not in feature_frame.columns]
    if missing:
        raise ExportError("feature sample is missing column(s): %s" % ", ".join(missing))
    score_frame = feature_frame.loc[:, required].copy()
    score_frame = score_frame.sort_values(
        ["security_id", "feature_asof"], kind="mergesort"
    ).reset_index(drop=True)
    return score_frame, feature_summary, cik_payload


def _write_json_atomic(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n"
    handle_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp-", suffix=".json"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _explain_sample_row(champion, feature_frame):
    row = feature_frame.iloc[[0]].copy()
    record = row.iloc[0]
    return explain.explain_security(
        champion,
        row,
        ticker=str(record["ticker"]),
        security_id=str(record["security_id"]),
        as_of_date=str(record["feature_asof"]),
        rank=None,
        percentile=None,
    )


def build_dashboard(root: Path, sample_size: int, seed: int) -> dict:
    champion = load_champion(root)
    score_frame, _feature_summary, cik_payload = build_feature_only_panel(sample_size, seed)
    feature_frame = score_frame.loc[:, ["security_id", "ticker", "feature_asof"] + list(schemas.FEATURE_ORDER)].copy()
    # Only the 13 base features ever enter scoring/attribution.
    feature_only = feature_frame.loc[
        :,
        ["security_id", "ticker", "feature_asof"] + list(schemas.FEATURE_ORDER),
    ].copy().reset_index(drop=True)

    n_rows = int(len(feature_only))
    n_securities = int(feature_only["security_id"].nunique())
    n_months = int(feature_only["feature_asof"].nunique())

    importance = attribution.feature_importance(champion)
    global_by_feature = attribution.global_attribution_by_feature(champion, feature_only)
    family = attribution.family_attribution(global_by_feature)

    contracts = {}
    contracts["research_summary"] = dashboard_summary.research_summary(root)
    contracts["model_definition"] = dashboard_summary.model_definition(root)
    contracts["global_feature_importance"] = {
        **schemas.schema_basis("global_feature_importance"),
        "freeze_id": champion.freeze_id,
        "importance_kind": attribution.IMPORTANCE_KIND,
        "basis": "frozen ExtraTrees feature_importances_ collapsed from 26 transformed columns to 13 base features",
        "features": list(schemas.FEATURE_ORDER),
        "shares": {feature: float(importance[feature]) for feature in schemas.FEATURE_ORDER},
        "limitations": list(attribution.ATTRIBUTION_LIMITATIONS),
    }
    contracts["feature_family_attribution"] = {
        **schemas.schema_basis("feature_family_attribution"),
        "freeze_id": champion.freeze_id,
        "attribution_kind": attribution.ATTRIBUTION_KIND,
        "attribution_basis": "mean absolute deterministic tree-path attribution over a label-free historical sample",
        "sample": {
            "rows": n_rows,
            "securities": n_securities,
            "months": n_months,
            "date_min": str(feature_only["feature_asof"].min()),
            "date_max": str(feature_only["feature_asof"].max()),
            "seed": seed,
            "selection_rule": "month-stratified equal allocation with deterministic fixed seed; feature-valued rows only, no label columns read",
            "sample_size": n_rows,
        },
        "by_feature": {feature: float(global_by_feature[feature]) for feature in schemas.FEATURE_ORDER},
        "by_family": {name: float(family[name]) for name in schemas.FAMILY_ORDER},
        "limitations": list(attribution.ATTRIBUTION_LIMITATIONS),
    }
    contracts["stock_explanations"] = _explain_sample_row(champion, feature_only)
    contracts["latest_shadow_ranking"] = latest_shadow_ranking(root)
    contracts["freshness_completeness"] = freshness_completeness(root)
    contracts["official_snapshot_history"] = official_snapshot_history(root)
    contracts["integrity_audit_status"] = audit_status(root)

    # Every emitted contract carries the SAME validated frozen-identity block so
    # the UI can bind numbers to the exact champion, feature order, contract
    # digest, and producing commit without research-internals access.
    identity = artifact_identity(root=root)
    for name in list(contracts):
        contracts[name]["artifact_identity"] = identity

    for kind, obj in contracts.items():
        schemas.validate_dashboard_artifact(kind, obj)

    return contracts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DASHBOARD_ATTRIBUTION_SEED)
    args = parser.parse_args(argv)

    root = Path(args.root)
    contracts = build_dashboard(root, args.sample_size, args.seed)

    identity = artifact_identity(root=root)
    manifest = {
        "schema_version": "wp9_dashboard_export_manifest_v1",
        "freeze_id": "freeze_e62eac30df40",
        "producing_script": "scripts/research_v2/wp9a2_dashboard_export.py",
        "producing_commit": identity["producing_commit"],
        "wp9_contract_digest": identity["contract_digest"],
        "attribution_seed": args.seed,
        "sample_size": args.sample_size,
        "artifact_identity": identity,
        "files": {},
    }
    for name, obj in contracts.items():
        rel = "%s.json" % name
        digest = _write_json_atomic(root / OUTPUT_DIR.relative_to(root) / rel, obj)
        manifest["files"][rel] = {"sha256": digest, "bytes": (root / OUTPUT_DIR.relative_to(root) / rel).stat().st_size}
    _write_json_atomic(root / OUTPUT_DIR.relative_to(root) / "export_manifest.json", manifest)

    print("DASHBOARD_EXPORT files=%d dir=%s" % (len(contracts), root / OUTPUT_DIR.relative_to(root)))
    for rel, meta in sorted(manifest["files"].items()):
        print("  %s %s %d" % (rel, meta["sha256"], meta["bytes"]))


if __name__ == "__main__":
    main()
