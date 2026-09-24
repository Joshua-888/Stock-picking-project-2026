"""Durable, source-independent provenance ledger for V2 dataset certification.

Certification metadata must survive loss of the local (gitignored) data
filesystem WITHOUT committing bulk data. This module persists tiny canonical
JSON only:

* one file per registered ``DatasetManifest`` under ``provenance/datasets/``;
* a single deterministic ``provenance/index.json`` summary.

The tracked ``provenance/`` area is intentionally NOT gitignored, while bulk
research data (``data/research_v2/``) stays out of git. Writes are write-once:
persisting identical content verifies and reuses, while different content for the
same dataset id raises ``ImmutabilityError`` and leaves the original untouched.

This module records and protects provenance; it performs no data selection, no
feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.research.ids import canonical_json
from src.research.immutability import ImmutabilityError, load_immutable, save_immutable, write_json_atomic

# Volatile manifest fields that must NOT participate in the write-once equality
# check. ``created_at`` is wall-clock wall time: a genuine re-certification of
# identical research content produces the same deterministic dataset_id but a new
# timestamp, so it must re-verify and reuse (keeping the ORIGINAL timestamp) rather
# than crash. Every other field remains strictly immutable.
VOLATILE_MANIFEST_FIELDS = ("created_at",)

DEFAULT_LEDGER_ROOT = Path("provenance")
DATASETS_SUBDIR = "datasets"
INDEX_NAME = "index.json"
LEDGER_SCHEMA_VERSION = "provenance_ledger_v1"

INDEX_FIELDS = (
    "dataset_id",
    "dataset_fingerprint",
    "pit_status",
    "synthetic_data_status",
    "row_count",
    "period_start",
    "period_end",
    "universe_definition",
    "git_commit",
    "schema_version",
    "sources",
    "manifest_file",
)


class ProvenanceLedgerError(RuntimeError):
    """Raised when a manifest cannot be registered in the ledger."""


def ledger_root(root=None):
    """Resolve the ledger root directory (default ``provenance/``)."""
    return Path(root) if root is not None else DEFAULT_LEDGER_ROOT


def datasets_dir(root=None):
    """Directory holding one canonical JSON file per registered dataset."""
    return ledger_root(root) / DATASETS_SUBDIR


def index_path(root=None):
    """Path of the ledger index file."""
    return ledger_root(root) / INDEX_NAME


def dataset_path(dataset_id, root=None):
    """Path of the stored manifest file for ``dataset_id``."""
    if not dataset_id:
        raise ProvenanceLedgerError("dataset_id is required to locate a ledger record")
    return datasets_dir(root) / ("%s.json" % dataset_id)


def _manifest_dict(manifest):
    """Return a manifest's canonical mapping from a manifest or a plain dict."""
    if hasattr(manifest, "to_dict"):
        return dict(manifest.to_dict())
    if isinstance(manifest, dict):
        return dict(manifest)
    raise ProvenanceLedgerError("manifest must be a DatasetManifest or a mapping, got %s" % type(manifest).__name__)


def _index_entry(payload):
    """Deterministic index entry derived from one ledger record."""
    data = payload.get("manifest") or {}
    entry = {name: data.get(name) for name in INDEX_FIELDS}
    entry["dataset_id"] = payload.get("dataset_id") or data.get("dataset_id")
    entry["manifest_file"] = "%s/%s.json" % (DATASETS_SUBDIR, entry["dataset_id"])
    if not entry["dataset_id"]:
        raise ProvenanceLedgerError("ledger record has no dataset_id")
    return entry


def _read_index(root):
    path = index_path(root)
    if not path.is_file():
        return {"ledger_schema_version": LEDGER_SCHEMA_VERSION, "entries": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceLedgerError("ledger index %s is not valid JSON: %s" % (path, exc)) from exc
    loaded.setdefault("ledger_schema_version", LEDGER_SCHEMA_VERSION)
    loaded.setdefault("entries", {})
    return loaded


def _update_index(root, entry):
    """Merge one entry into the index; identical content reuses, else rewrite."""
    index = _read_index(root)
    entries = index.get("entries") or {}
    key = entry["dataset_id"]
    if entries.get(key) == entry:
        return "verify_and_reuse"
    entries[key] = entry
    index["entries"] = {name: entries[name] for name in sorted(entries)}
    index["ledger_schema_version"] = LEDGER_SCHEMA_VERSION
    write_json_atomic(index_path(root), index)
    return "written"


def _strip_volatile(payload):
    """Return ``payload`` with volatile manifest fields removed.

    Used only for the write-once equality check so that a re-certification of
    identical research content (new ``created_at``) reuses the original record,
    while every other field stays immutable.
    """
    content = dict(payload)
    manifest = dict(content.get("manifest") or {})
    for field in VOLATILE_MANIFEST_FIELDS:
        manifest.pop(field, None)
    content["manifest"] = manifest
    return content


def _save_manifest_immutable(path, payload):
    """Write-once a ledger record, ignoring only the volatile timestamp.

    Returns ``"written"`` for a new record and ``"verify_and_reuse"`` when the
    stored record already holds equivalent research content. A differing field
    that is NOT volatile raises :class:`ImmutabilityError` and leaves the
    existing record untouched.
    """
    path = Path(path)
    if not path.exists():
        write_json_atomic(path, payload)
        return "written"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImmutabilityError("existing ledger record %s is not valid JSON: %s" % (path, exc)) from exc
    if canonical_json(_strip_volatile(existing)) != canonical_json(_strip_volatile(payload)):
        raise ImmutabilityError(
            "ledger record %s already exists with different research content; refusing to overwrite" % path
        )
    return "verify_and_reuse"


def persist_manifest(manifest, root=None, extra=None):
    """Persist a manifest's canonical JSON plus its index entry, idempotently.

    ``extra`` must be deterministic content (no wall-clock timestamps) so a
    repeated call for the same dataset verifies and reuses rather than diverging.
    Returns a record describing the ledger paths and both write outcomes.
    """
    data = _manifest_dict(manifest)
    dataset_id = data.get("dataset_id")
    if not dataset_id:
        raise ProvenanceLedgerError("manifest is missing dataset_id; refusing to register")
    payload = {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "manifest": data,
        "extra": dict(extra or {}),
    }
    path = dataset_path(dataset_id, root)
    outcome = _save_manifest_immutable(path, payload)
    entry = _index_entry(payload)
    index_outcome = _update_index(root, entry)
    return {
        "dataset_id": dataset_id,
        "path": str(path),
        "outcome": outcome,
        "index_path": str(index_path(root)),
        "index_outcome": index_outcome,
        "entry": entry,
    }


def load_manifest(dataset_id, root=None):
    """Load one stored ledger record (raises FileNotFoundError when absent)."""
    return load_immutable(dataset_path(dataset_id, root))


def list_entries(root=None):
    """Return the index entries ordered by dataset id."""
    index = _read_index(root)
    entries = index.get("entries") or {}
    return [entries[name] for name in sorted(entries)]


def load_all_manifests(root=None):
    """Return every stored manifest mapping, ordered by dataset id."""
    records = []
    for entry in list_entries(root):
        records.append(load_manifest(entry["dataset_id"], root)["manifest"])
    return records


def verify_ledger(root=None):
    """Cross-check the index against the stored manifests; return problem list.

    Every index entry must resolve to a stored manifest whose dataset id and
    dataset fingerprint agree with the index. A mismatch is a ledger integrity
    failure, not a warning.
    """
    problems = []
    index = _read_index(root)
    entries = index.get("entries") or {}
    stored_ids = sorted(path.stem for path in datasets_dir(root).glob("dataset_*.json")) if datasets_dir(root).is_dir() else []
    for name in sorted(set(entries) | set(stored_ids)):
        if name not in entries:
            problems.append("dataset %s is stored but absent from the index" % name)
            continue
        if name not in stored_ids:
            problems.append("dataset %s is indexed but its manifest file is missing" % name)
            continue
        manifest = load_manifest(name, root)["manifest"]
        if manifest.get("dataset_id") != name:
            problems.append("dataset %s manifest carries dataset_id %r" % (name, manifest.get("dataset_id")))
        if entries[name].get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
            problems.append("dataset %s fingerprint disagrees between index and manifest" % name)
    return problems
