"""WP6 artifact and provenance IO (immutable, content-addressed).

WP6 persists its results as IMMUTABLE artifacts under ``artifacts/research/wp6/``
and ``provenance/wp6/``. The experiment id is derived from a canonical binding of
the certified upstream identities, the frozen contract, the frozen registries,
the committing Git commit and the seeds, so identical inputs reproduce the same
id and different inputs can never overwrite a historical result
(:func:`save_immutable` raises instead).

A ``ledger.jsonl`` records EVERY tested configuration - no hidden trials - so the
ledger is a complete enumeration rather than the survivor of a search.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ids import experiment_id
from ..immutability import save_immutable

ARTIFACT_DIRNAME = "wp6"
PROVENANCE_DIRNAME = "wp6"
LEDGER_NAME = "ledger.jsonl"


class ModelingIOError(RuntimeError):
    """Raised when a WP6 artifact cannot be written honestly."""


def artifact_dir(root):
    return Path(root) / "artifacts" / "research" / ARTIFACT_DIRNAME


def provenance_dir(root):
    return Path(root) / "provenance" / PROVENANCE_DIRNAME


def modeling_payload(dataset_id, target_id, feature_set_id, holdout_id, git_commit,
                     contract, registry, seeds, notes=None):
    """Canonical binding payload for the WP6 experiment id."""
    return {
        "modeling_version": contract.get("runner_version"),
        "contract_version": contract.get("contract_version"),
        "dataset_id": dataset_id,
        "target_id": target_id,
        "feature_set_id": feature_set_id,
        "holdout_id": holdout_id,
        "git_commit": git_commit,
        "contract": contract,
        "model_registry": registry,
        "seeds": seeds,
        "notes": notes or {},
    }


def write_canonical(path, payload):
    """Write ``payload`` to ``path`` as sorted, indented canonical JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def write_immutable(path, payload):
    """Write-once ``payload`` at ``path``; identical content verifies and reuses."""
    return save_immutable(Path(path), payload)


def write_jsonl_immutable(path, records):
    """Write an immutable JSONL file from an ordered list of records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        rendered = "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records)
        if existing == rendered:
            return {"path": str(path), "outcome": "verify_and_reuse"}
        raise ModelingIOError("immutable ledger already exists with different content: %s" % path)
    rendered = "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records)
    path.write_text(rendered, encoding="utf-8")
    return {"path": str(path), "outcome": "written"}


def write_experiment(root, payload, artifacts, provenance, ledger_records=None):
    """Persist a WP6 experiment immutably and return its identifiers.

    ``artifacts``/``provenance`` map relative filename -> JSON payload. The
    experiment id is derived from ``payload``.
    """
    experiment = experiment_id(payload)
    artifact_root = artifact_dir(root) / experiment
    provenance_root = provenance_dir(root) / experiment
    for name, body in (artifacts or {}).items():
        write_immutable(artifact_root / name, body)
    for name, body in (provenance or {}).items():
        write_immutable(provenance_root / name, body)
    if ledger_records is not None:
        write_jsonl_immutable(artifact_root / LEDGER_NAME, ledger_records)
    return {"experiment_id": experiment, "artifact_dir": str(artifact_root),
            "provenance_dir": str(provenance_root)}
