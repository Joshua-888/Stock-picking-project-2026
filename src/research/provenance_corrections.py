"""Deterministic resolver for corrected research provenance bindings.

WP4's first provenance record was frozen with an incorrect producing-code commit
(the WP3 commit ``437f68ce``) because the WP4 build ran while its own code was
still uncommitted. Write-once immutability correctly refused to overwrite that
record, so the correction lives in NEW content-addressed artefacts plus a
machine-readable index under ``provenance/wp4/corrections/``.

This module resolves an experiment to its corrected provenance
*d-deterministically* (by experiment id, never by filename ordering) and exposes
the superseded references so the obsolete misbound record can never be selected
as canonical by downstream tooling.

It records and protects provenance; it performs no data selection, no feature
engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ids import canonical_json, make_id

CORRECTIONS_SCHEMA_VERSION = "wp4_provenance_corrections_index_v1"
INDEX_NAME = "index.json"

# Repo root derived from this module path so the default resolver is stable
# regardless of the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECTIONS_DIR = _REPO_ROOT / "provenance" / "wp4" / "corrections"


class ProvenanceResolutionError(RuntimeError):
    """Raised when a corrected provenance binding cannot be resolved."""


def corrections_dir(root=None):
    """Directory holding the correction index and its immutable records."""
    return Path(root) if root is not None else DEFAULT_CORRECTIONS_DIR


def index_path(root=None):
    """Path of the machine-readable corrections index."""
    return corrections_dir(root) / INDEX_NAME


def read_index(root=None):
    """Load and validate the corrections index; raise when missing/invalid."""
    path = index_path(root)
    if not path.is_file():
        raise ProvenanceResolutionError("corrections index is missing: %s" % path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceResolutionError("corrections index %s is not valid JSON: %s" % (path, exc)) from exc
    if loaded.get("corrections_schema_version") != CORRECTIONS_SCHEMA_VERSION:
        raise ProvenanceResolutionError(
            "unexpected corrections schema %r; expected %r"
            % (loaded.get("corrections_schema_version"), CORRECTIONS_SCHEMA_VERSION)
        )
    if not isinstance(loaded.get("entries"), dict):
        raise ProvenanceResolutionError("corrections index entries must be a mapping")
    return loaded


def list_corrections(root=None):
    """Return every correction entry ordered by experiment id."""
    entries = read_index(root).get("entries") or {}
    return [entries[name] for name in sorted(entries)]


def _entry(experiment_id, root=None):
    entries = read_index(root).get("entries") or {}
    entry = entries.get(experiment_id)
    if entry is None:
        raise ProvenanceResolutionError("no corrected provenance registered for %s" % experiment_id)
    return entry


def _load_record(basename, root=None):
    path = corrections_dir(root) / basename
    if not path.is_file():
        raise ProvenanceResolutionError("corrected record is missing: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceResolutionError("corrected record %s is not valid JSON: %s" % (path, exc)) from exc


def canonical_provenance_path(experiment_id, root=None):
    """Absolute path of the canonical (corrected) provenance for ``experiment_id``."""
    entry = _entry(experiment_id, root)
    basename = entry.get("corrected_provenance_file")
    if not basename:
        raise ProvenanceResolutionError("correction entry for %s has no corrected_provenance_file" % experiment_id)
    return corrections_dir(root) / basename


def superseded_references(experiment_id, root=None):
    """Repo-relative references that must NEVER be treated as canonical."""
    return list(_entry(experiment_id, root).get("superseded_records") or [])


def resolve_experiment(experiment_id, root=None):
    """Resolve ``experiment_id`` to its corrected provenance + supersession record.

    Returns a mapping with the canonical reference, the loaded corrected
    provenance, the supersession record and the list of obsolete references.
    The corrected record is cross-checked against the index so a mismatch is an
    integrity failure rather than a silent pass.
    """
    entry = _entry(experiment_id, root)
    corrected = load_corrected_provenance(experiment_id, root)
    supersession = _load_record(entry["supersession_record_file"], root)
    producing = entry.get("producing_code_commit")
    if corrected.get("producing_code_commit") != producing:
        raise ProvenanceResolutionError(
            "corrected provenance producing_code_commit %r disagrees with index %r"
            % (corrected.get("producing_code_commit"), producing)
        )
    if corrected.get("corrected_provenance_id") != entry.get("corrected_provenance_id"):
        raise ProvenanceResolutionError(
            "corrected provenance id %r disagrees with index %r"
            % (corrected.get("corrected_provenance_id"), entry.get("corrected_provenance_id"))
        )
    return {
        "experiment_id": experiment_id,
        "producing_code_commit": producing,
        "canonical_provenance_ref": entry.get("canonical_provenance_ref"),
        "corrected_provenance": corrected,
        "supersession": supersession,
        "superseded_records": list(entry.get("superseded_records") or []),
    }


def load_corrected_provenance(experiment_id, root=None):
    """Load the corrected provenance record for ``experiment_id``."""
    entry = _entry(experiment_id, root)
    return _load_record(entry["corrected_provenance_file"], root)


def is_superseded_reference(reference, root=None):
    """True when ``reference`` is a superseded (never-canonical) provenance ref."""
    wanted = str(reference).replace("\\", "/")
    wanted_name = Path(wanted).name
    for entry in list_corrections(root):
        for ref in entry.get("superseded_records") or []:
            normalised = str(ref).replace("\\", "/")
            if wanted == normalised or wanted_name == Path(normalised).name:
                return True
    return False


def assert_not_superseded(reference, root=None):
    """Raise when ``reference`` names an obsolete misbound provenance record."""
    if is_superseded_reference(reference, root):
        raise ProvenanceResolutionError(
            "%s is a superseded provenance record and must never be selected as canonical" % reference
        )
    return True


def corrected_provenance_id(binding):
    """Content-addressed id of a corrected binding, using the WP1 id scheme."""
    return make_id("experiment", binding)


def corrected_provenance_relative_path(corrected_id):
    """Repo-relative path convention for a corrected provenance record."""
    return "provenance/wp4/corrections/%s.json" % corrected_id


def canonical_json_text(record):
    """Canonical JSON text of a record (WP1 convention); helper for hashing/tests."""
    return canonical_json(record)
