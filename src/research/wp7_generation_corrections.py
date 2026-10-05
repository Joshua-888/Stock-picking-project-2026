"""Deterministic resolver for WP7 model-generation candidate corrections.

A WP7 pre-holdout generation candidate is produced per generation under
``provenance/wp7/candidates/<generation_id>.json``. The first recorded
generation, ``generation_04b8e2810b50``, was misbound to a producing commit that
does not contain the WP7 v4 contract/engine code and is superseded by a later
commit-clean generation.

This module reads a machine-readable index under
``provenance/wp7/corrections/index.json`` and resolves generations strictly by
``generation_id``. Canonical status, canonical candidate path, and supersession
records are never inferred from filenames or directory ordering.

It records and protects provenance; it performs no data selection, no feature
engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

WP7_SUPERSESSION_SCHEMA_VERSION = "wp7_model_generation_candidate_supersession_v1"
WP7_CORRECTION_INDEX_SCHEMA_VERSION = "wp7_generation_correction_index_v1"
INDEX_NAME = "index.json"

CANONICAL_STATUS = "canonical"
MISBOUND_WITHDRAWN_STATUS = "MISBOUND_WITHDRAWN_NON_CANONICAL"
NON_CANONICAL_STATUS = "NON_CANONICAL"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECTIONS_DIR = _REPO_ROOT / "provenance" / "wp7" / "corrections"


class Wp7GenerationCorrectionError(RuntimeError):
    """Raised when a WP7 generation correction cannot be resolved."""


def corrections_dir(root=None):
    """Directory holding the correction index and its supersession records."""
    return Path(root) if root is not None else DEFAULT_CORRECTIONS_DIR


def index_path(root=None):
    """Path of the machine-readable corrections index."""
    return corrections_dir(root) / INDEX_NAME


def candidate_path_provider():
    """Root used to relativize candidate paths in the corrections index."""
    return _REPO_ROOT


def _resolve_candidate_ref(raw):
    """Return an absolute candidate path from an index-provided reference."""
    path = Path(raw)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def read_index(root=None):
    """Load and validate the WP7 correction index; raise when missing/invalid."""
    path = index_path(root)
    if not path.is_file():
        raise Wp7GenerationCorrectionError("corrections index is missing: %s" % path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp7GenerationCorrectionError(
            "corrections index %s is not valid JSON: %s" % (path, exc)
        ) from exc
    if loaded.get("schema_version") not in (
        WP7_CORRECTION_INDEX_SCHEMA_VERSION,
        None,
    ):
        raise Wp7GenerationCorrectionError(
            "unexpected corrections index schema %r; expected %r"
            % (loaded.get("schema_version"), WP7_CORRECTION_INDEX_SCHEMA_VERSION)
        )
    canonical_id = loaded.get("canonical_generation_id")
    if not canonical_id:
        raise Wp7GenerationCorrectionError(
            "corrections index has no canonical_generation_id"
        )
    if not isinstance(loaded.get("entries"), dict):
        raise Wp7GenerationCorrectionError("corrections index entries must be a mapping")
    return loaded


def canonical_generation_id(root=None):
    """The canonical WP7 generation id, read from the index (never inferred)."""
    return read_index(root)["canonical_generation_id"]


def canonical_candidate_ref(root=None):
    """The raw canonical candidate reference stored in the index."""
    index = read_index(root)
    return index.get("canonical_candidate_path") \
        or index.get("canonical_candidate_ref")


def canonical_candidate_path(root=None):
    """Absolute path of the canonical WP7 candidate record.

    The file location is read from the index and is never derived from
    directory ordering.
    """
    ref = canonical_candidate_ref(root)
    if not ref:
        raise Wp7GenerationCorrectionError(
            "corrections index has no canonical_candidate_path/ref"
        )
    return _resolve_candidate_ref(ref)


def list_entries(root=None):
    """Return every corrections entry ordered by generation id."""
    entries = read_index(root).get("entries") or {}
    return [entries[name] for name in sorted(entries)]


def _entry(generation_id, root=None):
    index = read_index(root)
    entry = (index.get("entries") or {}).get(generation_id)
    if entry is None:
        raise Wp7GenerationCorrectionError(
            "no WP7 corrected candidate registration for %s" % generation_id
        )
    return entry


def _load_record(basename, root=None):
    path = corrections_dir(root) / basename
    if not path.is_file():
        raise Wp7GenerationCorrectionError("corrected record is missing: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp7GenerationCorrectionError(
            "corrected record %s is not valid JSON: %s" % (path, exc)
        ) from exc


def supersession_record(generation_id, root=None):
    """Load the withdrawal/supersession record for ``generation_id``."""
    entry = _entry(generation_id, root)
    basename = entry.get("supersession_file")
    if not basename:
        raise Wp7GenerationCorrectionError(
            "correction entry for %s has no supersession_file" % generation_id
        )
    return _load_record(basename, root)


def is_non_canonical(generation_id, root=None):
    """True when ``generation_id`` is a withdrawn, non-canonical record."""
    entries = read_index(root).get("entries") or {}
    entry = entries.get(generation_id)
    if entry is None:
        return False
    return entry.get("status") != CANONICAL_STATUS


def resolve_generation(generation_id, root=None):
    """Resolve ``generation_id`` to its canonical/non-canonical WP7 status.

    Resolution is keyed strictly by generation id, never by filenames or
    directory ordering. The canonical id returns ``canonical=True``; a misbound
    withdrawn id returns ``canonical=False`` and its supersession record.
    """
    index = read_index(root)
    canonical_id = index["canonical_generation_id"]
    entry = _entry(generation_id, root)
    canonical_ref = canonical_candidate_ref(root)
    canonical_path = _resolve_candidate_ref(canonical_ref) if canonical_ref else None

    base = {
        "generation_id": generation_id,
        "canonical_generation_id": canonical_id,
        "canonical_candidate_path": canonical_path,
        "status": entry.get("status"),
    }

    if generation_id == canonical_id:
        resolved = dict(base)
        resolved["canonical"] = entry.get("status") == CANONICAL_STATUS
        resolved["misbound_producing_commit"] = None
        resolved["supersession"] = None
        if not resolved["canonical"]:
            raise Wp7GenerationCorrectionError(
                "canonical generation %s is not marked canonical in the index"
                % generation_id
            )
        return resolved

    if is_non_canonical(generation_id, root):
        withdrawal = supersession_record(generation_id, root)
        if withdrawal.get("new_generation_id") not in (None, canonical_id):
            raise Wp7GenerationCorrectionError(
                "supersession record for %s disagrees with the index canonical id %r"
                % (generation_id, canonical_id)
            )
        resolved = dict(base)
        resolved["canonical"] = False
        resolved["misbound_producing_commit"] = withdrawal.get(
            "misbound_producing_commit"
        )
        resolved["supersession"] = withdrawal
        return resolved

    raise Wp7GenerationCorrectionError(
        "WP7 candidate %s is neither canonical nor marked non-canonical"
        % generation_id
    )


def assert_canonical(generation_id, root=None):
    """Raise when ``generation_id`` is not the canonical WP7 generation."""
    resolved = resolve_generation(generation_id, root)
    if not resolved.get("canonical"):
        raise Wp7GenerationCorrectionError(
            "%s is a non-canonical/withdrawn WP7 candidate record and must never be "
            "selected as canonical" % generation_id
        )
    return resolved
