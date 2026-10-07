"""WP9 forward-snapshot input materialisation.

Build the per-snapshot universe and 13-feature score frame for the frozen WP8
champion. This module reuses the certified PIT machinery and never implements a
new provider, availability rule, membership rule, or feature definition.

Membership is resolved **effective as-of the snapshot date only**. Future or
current membership is never used to reconstruct an older snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ..data import layers
from ..data.availability import to_utc_timestamp
from ..data.universe import UniverseMembership, UniverseTable
from ..discovery.panel import build_feature_panel
from ..fingerprints import fingerprint_dataframe, fingerprint_obj
from .champion import FROZEN_FEATURES

ROOT = Path(__file__).resolve().parents[3]
BRONZE_ROOT = ROOT / "data" / "research_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "research"

PANEL_NAME = "sp500_pit_research_panel_v1"
PANEL_VERSION = "ac294282d8e949f2"
PRICES_NAME = "wp2b_sp500_pit_prices_silver"
ACTIONS_NAME = "wp2b_sp500_pit_actions_silver"
MEMBERSHIP_NAME = "wp2b_sp500_pit_membership_silver"
BENCHMARK_NAME = "benchmark_gold_SPY"
BENCHMARK_VERSION = "a0ab5bd2382518d4"
SPY_BRONZE_REL_GLOB = "data/research_v2/bronze/wp3_spy_prices_actions/*/raw.json"
INPUT_BINDING_REL = Path("provenance") / "wp8" / "input_binding.json"
CIK_MAPPING_REL = Path("artifacts") / "research" / "wp4" / "edgar_cik_mapping.json"
LAYER_RECORDS_REL = Path("artifacts") / "research" / "wp2b_live" / "layer_records.json"

UNIVERSE_ID = "sp500_pit_wikipedia_eodhd_v1"


class ForwardInputError(RuntimeError):
    """Raised when a forward snapshot cannot be built honestly."""


@dataclass(frozen=True)
class ForwardSnapshotInputs:
    """Deterministic forward inputs for one official snapshot date."""

    snapshot_asof: str
    universe: Dict[str, object]
    score_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    feature_summary: Dict[str, object]
    source_manifest: Dict[str, object]

    @property
    def universe_hash(self) -> str:
        return str(self.universe["universe_hash"])

    @property
    def source_manifest_hash(self) -> str:
        return str(self.source_manifest["source_manifest_hash"])

    @property
    def feature_snapshot_hash(self) -> str:
        return str(self.source_manifest["feature_snapshot_hash"])


class _ForwardInputBuilder:
    """Loader/builder wrapper with overridable load functions for tests."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or ROOT)
        self.bronze_root = self.root / "data" / "research_v2"

    def _json(self, rel: Path) -> dict:
        path = self.root / rel
        if not path.is_file():
            raise ForwardInputError("missing certified input artifact: %s" % path)
        return json.loads(path.read_text(encoding="utf-8"))

    def load_layer_records(self) -> dict:
        return self._json(LAYER_RECORDS_REL)

    def load_panel(self):
        frame = layers.read_silver_table(
            self.bronze_root, PANEL_NAME, version=PANEL_VERSION
        )
        required = ("security_id", "ticker", "snapshot_date", "price_date")
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ForwardInputError("certified panel missing columns: %s" % ", ".join(missing))
        keep = [name for name in ("security_id", "ticker", "snapshot_date", "price_date")
                if name in frame.columns]
        return frame.loc[:, keep].copy()

    def load_prices(self, version: str):
        return layers.read_silver_table(self.bronze_root, PRICES_NAME, version=version)

    def load_actions(self, version: str):
        return layers.read_silver_table(self.bronze_root, ACTIONS_NAME, version=version)

    def load_membership(self, version: str):
        return layers.read_silver_table(self.bronze_root, MEMBERSHIP_NAME, version=version)

    def load_benchmark_prices(self):
        frame = layers.read_silver_table(
            self.bronze_root, BENCHMARK_NAME, version=BENCHMARK_VERSION
        )
        frame = frame.copy()
        frame["security_id"] = "SPY"
        frame["ticker"] = "SPY"
        return frame.loc[:, ["security_id", "ticker", "trade_date", "raw_close"]]

    def load_benchmark_actions(self):
        import glob

        pattern = str(self.root / SPY_BRONZE_REL_GLOB)
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ForwardInputError("SPY corporate-action bronze is missing: %s" % SPY_BRONZE_REL_GLOB)
        payload = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
        actions = payload.get("actions") or []
        frame = pd.DataFrame(actions)
        for column in ("kind", "effective_date", "numerator", "denominator", "amount"):
            if column not in frame.columns:
                frame[column] = pd.NA
        frame["security_id"] = "SPY"
        return frame.loc[:, ["security_id", "kind", "effective_date", "numerator", "denominator", "amount"]]

    def load_fundamentals_and_cik(self):
        """Load the certified EDGAR input binding path.

        The real, verified loaders live in
        :mod:`src.research.data.edgar_binding`. Import lazily so synthetic tests
        can replace this method without touching the file system.
        """
        from ..data.edgar_binding import (
            load_and_verify_cik_by_ticker,
            load_and_verify_edgar_fundamentals,
        )

        binding = self._json(INPUT_BINDING_REL)
        fundamentals = load_and_verify_edgar_fundamentals(self.root, binding)
        cik_by_ticker, cik_payload = load_and_verify_cik_by_ticker(self.root, binding)
        return fundamentals, cik_by_ticker, cik_payload, binding

    def resolve_universe_at(self, snapshot_date: str, membership: pd.DataFrame) -> Dict[str, object]:
        """Resolve S&P 500 membership effective as-of ``snapshot_date`` only."""
        _require_date(snapshot_date)
        required = ("security_id", "membership_start", "membership_end")
        missing = [name for name in required if name not in membership.columns]
        if missing:
            raise ForwardInputError("membership frame missing columns: %s" % ", ".join(missing))

        records = list(membership.to_dict("records"))
        records.sort(
            key=lambda row: (
                str(row["security_id"]),
                str(row["membership_start"]),
            )
        )
        memberships = []
        for record in records:
            ticker = record.get("ticker") or record.get("symbol") or record["security_id"]
            memberships.append(
                UniverseMembership(
                    universe_id=UNIVERSE_ID,
                    security_id=str(record["security_id"]),
                    ticker=str(ticker),
                    membership_start=record["membership_start"],
                    membership_end=_nullable_text(record.get("membership_end")),
                    # The certified WP2C membership silver table omits
                    # announcement/validity columns. Availability is therefore
                    # exactly the effective-date instant (the WP2C documented
                    # conservative rule), never an announcement date.
                    valid_from=record["membership_start"],
                    valid_to=_nullable_text(record.get("membership_end")),
                )
            )
        table = UniverseTable(UNIVERSE_ID, memberships=memberships)
        security_ids = table.members_asof(snapshot_date, available_only=True)
        ticker_mapping = {
            membership.security_id: membership.ticker
            for membership in table.memberships
            if membership.security_id in set(security_ids)
        }
        record = {
            "universe_id": UNIVERSE_ID,
            "snapshot_asof": snapshot_date,
            "membership_source": "sp500_pit_wikipedia_eodhd_v1",
            "membership_effective_date": snapshot_date,
            "security_ids": security_ids,
            "ticker_mapping": ticker_mapping,
            "universe_count": len(security_ids),
            "source_retrieval_metadata": {
                "membership_layer": MEMBERSHIP_NAME,
                "availability_rule": "effective date instant only",
                "rows_loaded": int(len(membership)),
            },
        }
        record["universe_hash"] = fingerprint_obj(
            {
                "universe_id": record["universe_id"],
                "snapshot_asof": record["snapshot_asof"],
                "security_ids": record["security_ids"],
                "ticker_mapping": record["ticker_mapping"],
                "universe_count": record["universe_count"],
            }
        )
        return record

    def forward_panel_rows(self, snapshot_date, security_ids, ticker_mapping, prices):
        """One feature-panel row per as-of member with its latest eligible price date."""
        _require_date(snapshot_date)
        price_frame = prices.copy()
        price_frame["trade_date"] = price_frame["trade_date"].astype(str).str.slice(0, 10)
        eligible = price_frame.loc[price_frame["trade_date"] <= snapshot_date]
        latest = (
            eligible.sort_values("trade_date", kind="mergesort")
            .groupby("security_id", sort=True)
            .tail(1)
            .set_index("security_id")["trade_date"]
            .to_dict()
        )
        rows = []
        for security_id in security_ids:
            rows.append(
                {
                    "security_id": security_id,
                    "ticker": ticker_mapping.get(security_id, security_id),
                    "feature_asof": snapshot_date,
                    "price_date": latest.get(security_id),
                }
            )
        return pd.DataFrame(rows, columns=["security_id", "ticker", "feature_asof", "price_date"])

    def build_source_manifest(self, snapshot_date, layer_records, upstream: Dict[str, object]) -> Dict[str, object]:
        """Record source identity, versions, retrieval metadata, and row counts."""
        payload = {
            "schema_version": "wp9_source_manifest_v1",
            "snapshot_asof": snapshot_date,
            "allowed_sources": ["EODHD", "SEC_EDGAR", "WIKIPEDIA_SP500"],
            "sources": {
                "eodhd_market": {
                    "source": "EODHD",
                    "silver_prices": layer_records.get("silver_prices"),
                    "silver_actions": layer_records.get("silver_actions"),
                    "price_rows": layer_records.get("price_rows"),
                    "action_rows": layer_records.get("action_rows"),
                },
                "wikipedia_sp500": {
                    "source": "WIKIPEDIA_SP500",
                    "silver_membership": layer_records.get("silver_membership"),
                    "window_rows": layer_records.get("windows"),
                },
                "sec_edgar": upstream.get("edgar_binding"),
            },
            "benchmark": {
                "name": BENCHMARK_NAME,
                "version": BENCHMARK_VERSION,
            },
            "upstream": upstream,
        }
        digest = fingerprint_obj(payload)
        return {"source_manifest_hash": digest, **payload}

    def build(self, snapshot_date: str, *, feature_panel_builder=None) -> "ForwardSnapshotInputs":
        """Build all forward inputs for ``snapshot_date``.

        ``feature_panel_builder`` is an injection point for tests; the default is
        the certified :func:`src.research.discovery.panel.build_feature_panel`.
        """
        _require_date(snapshot_date)
        if feature_panel_builder is None:
            feature_panel_builder = build_feature_panel

        layer_records = self.load_layer_records()
        prices = self.load_prices(str(layer_records["silver_prices"]))
        actions = self.load_actions(str(layer_records["silver_actions"]))
        membership = self.load_membership(str(layer_records["silver_membership"]))
        benchmark_prices = self.load_benchmark_prices()
        benchmark_actions = self.load_benchmark_actions()
        fundamentals, cik_by_ticker, cik_payload, edgar_binding = self.load_fundamentals_and_cik()

        universe = self.resolve_universe_at(snapshot_date, membership)
        panel_rows = self.forward_panel_rows(
            snapshot_date,
            universe["security_ids"],
            universe["ticker_mapping"],
            prices,
        )
        feature_frame, feature_summary = feature_panel_builder(
            panel_rows,
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
        selected = ["security_id", "ticker", "feature_asof"] + list(FROZEN_FEATURES)
        missing = [name for name in selected if name not in feature_frame.columns]
        if missing:
            raise ForwardInputError("feature panel missing required columns: %s" % ", ".join(missing))
        score_frame = feature_frame.loc[:, selected].copy()
        score_frame = score_frame.sort_values(
            ["security_id", "feature_asof"], kind="mergesort"
        ).reset_index(drop=True)

        feature_snapshot_hash = fingerprint_dataframe(
            score_frame.loc[:, ["security_id", "feature_asof"] + list(FROZEN_FEATURES)]
        )
        upstream = {
            "dataset_id": "dataset_35a278e17c13",
            "feature_set_id": "feature_set_4f7b43726310",
            "security_master_version": (
                "d4ba066eb6cf81a115f6626cb95d57d30e842c1f4c75ec84149b35b8dfd7d349"
            ),
            "universe_version": (
                "0ce5054a2d66ce76ee68b51ac3f41cda90b8444cf134569ddd7a16044086a796"
            ),
            "target_set_id": "target_set_d2bb16610bce",
            "panel_name": PANEL_NAME,
            "panel_version": PANEL_VERSION,
            "edgar_binding": edgar_binding,
            "cik_mapping": cik_payload.get("mapping"),
        }
        manifest_payload = self.build_source_manifest(
            snapshot_date, layer_records, upstream
        )
        manifest_payload.pop("source_manifest_hash", None)
        manifest_payload["feature_snapshot_hash"] = feature_snapshot_hash
        manifest_payload["feature_row_count"] = int(len(score_frame))
        manifest_payload["feature_summary"] = feature_summary
        manifest_payload["universe_hash"] = universe["universe_hash"]
        manifest_payload["source_manifest_hash"] = fingerprint_obj(
            {
                key: value
                for key, value in manifest_payload.items()
                if key not in ("feature_summary",)
            }
        )

        return ForwardSnapshotInputs(
            snapshot_asof=snapshot_date,
            universe=universe,
            score_frame=score_frame,
            feature_frame=feature_frame,
            feature_summary=feature_summary,
            source_manifest=manifest_payload,
        )


def _nullable_text(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return None if text in ("", "None", "nan", "NaT") else text


def _require_date(value: str) -> str:
    if not isinstance(value, str):
        raise ForwardInputError("snapshot date must be a YYYY-MM-DD string")
    stamp = to_utc_timestamp(value)
    if stamp is None or value != stamp.strftime("%Y-%m-%d"):
        raise ForwardInputError("snapshot date must be a canonical YYYY-MM-DD string")
    return value


def build_forward_inputs(snapshot_date: str, root: Path | None = None) -> ForwardSnapshotInputs:
    """Materialise certified forward inputs for one snapshot date."""
    return _ForwardInputBuilder(root=root).build(snapshot_date)


__all__ = [
    "ForwardInputError",
    "ForwardSnapshotInputs",
    "build_forward_inputs",
]
