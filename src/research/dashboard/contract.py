"""Dashboard-facing read-only exports over the WP9 registry and indexes.

All functions here read immutable provenance files and never call any write
function. If the official snapshot history is empty, it is returned empty and
never invented. Dry-run rankings are always labelled
``NON_EVIDENTIARY_DRY_RUN`` and always expose their exact as-of date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..wp9 import storage
from . import schemas
from .identity import attach_identity

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

RUN_INDEX_REL = Path("provenance") / "wp9" / "index.json"
DRY_RUN_INDEX_REL = Path("provenance") / "wp9" / "dry_runs" / "index.json"
PREDICTION_INDEX_REL = Path("provenance") / "wp9" / "predictions" / "index.json"


def _run_registry(root: Path) -> dict:
    return json.loads((root / RUN_INDEX_REL).read_text(encoding="utf-8"))


def _dry_run_index(root: Path) -> dict:
    path = root / DRY_RUN_INDEX_REL
    if not path.is_file():
        return {"schema_version": storage.PREDICTION_INDEX_SCHEMA, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_index(root: Path) -> dict:
    path = root / PREDICTION_INDEX_REL
    if not path.is_file():
        return {"schema_version": storage.PREDICTION_INDEX_SCHEMA, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_shadow_ranking(root: Path | None = None) -> dict:
    """Expose the current official history honestly and label dry-runs."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    registry = _run_registry(root)
    dry_index = _dry_run_index(root)
    dry_rankings = []
    for entry in dry_index.get("entries", []):
        dry_rankings.append({
            "snapshot_id": str(entry.get("snapshot_id")),
            "snapshot_asof": str(entry.get("snapshot_asof")),
            "snapshot_kind": "NON_EVIDENTIARY_DRY_RUN",
            "model_freeze_id": str(entry.get("model_freeze_id")),
            "model_hash": str(entry.get("model_hash")),
            "universe_size": int(entry.get("universe_size")) if entry.get("universe_size") is not None else None,
        })
    result = {
        **schemas.schema_basis("latest_shadow_ranking"),
        "registry_path": str(RUN_INDEX_REL),
        "official_snapshot_ids": [str(value) for value in registry.get("official_snapshot_ids", [])],
        "dry_run_rankings": dry_rankings,
        "limitations": [
            "dry-run rankings are NON_EVIDENTIARY_DRY_RUN and never enter official history",
            "official_snapshot_ids are copied verbatim from the read-only run registry",
        ],
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("latest_shadow_ranking", result)
    return result


def official_snapshot_history(root: Path | None = None) -> dict:
    """Return the exact official snapshot history; an empty list is honest."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    registry = _run_registry(root)
    prediction_index = _prediction_index(root)
    official = [str(value) for value in registry.get("official_snapshot_ids", [])]
    history = []
    for entry in prediction_index.get("entries", []):
        snapshot_id = str(entry.get("snapshot_id"))
        if snapshot_id in official:
            history.append({
                "snapshot_id": snapshot_id,
                "snapshot_asof": str(entry.get("snapshot_asof")),
                "snapshot_kind": str(entry.get("snapshot_kind", storage.OFFICIAL_KIND)),
            })
    result = {
        **schemas.schema_basis("official_snapshot_history"),
        "registry_path": str(RUN_INDEX_REL),
        "official_snapshot_ids": official,
        "history": history,
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("official_snapshot_history", result)
    return result


def freshness_completeness(root: Path | None = None) -> dict:
    """Report current WP9 stage and counts without writing anything."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    registry = _run_registry(root)
    dry_index = _dry_run_index(root)
    dry_entries = list(dry_index.get("entries", []))
    dry_entries_sorted = sorted(dry_entries, key=lambda item: (str(item.get("snapshot_asof")), str(item.get("snapshot_id"))),
                                reverse=True)
    result = {
        **schemas.schema_basis("freshness_completeness"),
        "current_prospective_stage": str(registry.get("current_prospective_stage")),
        "official_snapshot_count": len(registry.get("official_snapshot_ids", [])),
        "dry_run_count": len(dry_entries),
        "matured_evaluation_count": len(registry.get("matured_evaluations", [])),
        "invalidated_snapshot_count": len(registry.get("invalidated_snapshots", [])),
        "latest_dry_run": dry_entries_sorted[0] if dry_entries_sorted else None,
        "limitations": [
            "the run-registry dry_run_ids list may omit separated dry-run files; this report exposes files that exist rather than hiding them",
            "no official snapshots exist yet; the prospective stage is honest and non-evidentiary",
        ],
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("freshness_completeness", result)
    return result


__all__ = ["freshness_completeness", "latest_shadow_ranking", "official_snapshot_history"]
