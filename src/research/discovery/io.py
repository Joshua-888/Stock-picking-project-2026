"""WP5 artifact and provenance IO.

WP5 persists its results as IMMUTABLE, content-addressed artifacts under
``artifacts/research/wp5/`` and ``provenance/wp5/``. Every artifact is bound to
the certified upstream identifiers through a deterministic experiment id computed
from a canonical JSON payload, so a rerun with identical inputs reproduces the
same id and a rerun with different inputs can never overwrite the previous
result - :func:`save_immutable` raises instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ids import experiment_id
from ..immutability import save_immutable

ARTIFACT_DIRNAME = "wp5"
PROVENANCE_DIRNAME = "wp5"


class DiscoveryIOError(RuntimeError):
    """Raised when a WP5 artifact cannot be written honestly."""


def artifact_dir(root):
    return Path(root) / "artifacts" / "research" / ARTIFACT_DIRNAME


def provenance_dir(root):
    return Path(root) / "provenance" / PROVENANCE_DIRNAME


def discovery_payload(dataset_id, target_id, feature_set_id, feature_catalog, config,
                      git_commit, holdout_id=None, notes=None):
    """Canonical binding payload for the WP5 discovery experiment id."""
    return {
        "discovery_version": "v2_wp5_discovery_v1",
        "dataset_id": dataset_id,
        "target_id": target_id,
        "feature_set_id": feature_set_id,
        "holdout_id": holdout_id,
        "git_commit": git_commit,
        "config": config,
        "feature_catalog": feature_catalog,
        "notes": notes or {},
    }


def write_canonical(path, payload):
    """Write ``payload`` to ``path`` as sorted, indented canonical JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def write_immutable(path, payload):
    """Write-once ``payload`` at ``path``; identical content verifies and reuses."""
    return save_immutable(Path(path), payload)


def write_experiment(root, payload, artifacts, provenance):
    """Persist a discovery experiment immutably and return its identifiers.

    ``artifacts`` and ``provenance`` are mappings of relative filename -> JSON
    payload. The experiment id is derived from ``payload`` so identical inputs
    always yield the same directory and different inputs never overwrite.
    """
    experiment = experiment_id(payload)
    artifact_root = artifact_dir(root) / experiment
    provenance_root = provenance_dir(root) / experiment
    for name, body in (artifacts or {}).items():
        write_immutable(artifact_root / name, body)
    for name, body in (provenance or {}).items():
        write_immutable(provenance_root / name, body)
    return {"experiment_id": experiment, "artifact_dir": str(artifact_root),
            "provenance_dir": str(provenance_root)}
