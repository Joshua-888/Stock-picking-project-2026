"""Deterministic resolver for the WP6 provenance misbinding correction.

WP6's first provenance record was frozen with an incorrect producing-code
commit: the record for ``experiment_05ddc3721b4a`` binds
``git_commit = 8751c164b8c504628a2b9122e41397cf41f71276``, a commit that does
NOT contain the WP6 producing code (``src/research/modeling/`` and
``scripts/research_v2/wp6_model_research.py``). A corrective rerun produced
the content-addressed experiment id ``experiment_d6eda4a491ca`` whose
deterministically verified producing commit is
``1bbfed0c9a58bf1958926a67e33f5fa5f944215d``. The earlier
experiment ``experiment_f7864f37998f`` is withdrawn and superseded, and
experiment ``experiment_05ddc3721b4a`` is withdrawn/misbound; both historical
records remain byte-unchanged.

Write-once immutability correctly refuses to overwrite the frozen misbound
record, so the correction lives in NEW artefacts plus a machine-readable index
under ``provenance/wp6/corrections/`` (mirroring the WP4/WP5 correction
pattern).

This module resolves a WP6 experiment to its canonical provenance
*d-deterministically* (by experiment id, never by filename ordering) and exposes
the withdrawn, non-canonical status of the misbound and superseded records so
they can never be selected as canonical by downstream tooling.

It records and protects provenance; it performs no data selection, no feature
engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

WP6_CORRECTIONS_SCHEMA_VERSION = "wp6_provenance_correction_supersession_v1"
INDEX_NAME = "index.json"

NON_CANONICAL_STATUS = "NON_CANONICAL"
WITHDRAWN_STATUS = "MISBOUND_WITHDRAWN_NON_CANONICAL"
CANONICAL_STATUS = "canonical"

# Repo root derived from this module path so the default resolver is stable
# regardless of the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECTIONS_DIR = _REPO_ROOT / "provenance" / "wp6" / "corrections"


class Wp6ProvenanceResolutionError(RuntimeError):
    """Raised when a WP6 canonical provenance binding cannot be resolved."""


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
        raise Wp6ProvenanceResolutionError("corrections index is missing: %s" % path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp6ProvenanceResolutionError(
            "corrections index %s is not valid JSON: %s" % (path, exc)
        ) from exc
    if loaded.get("schema_version") != WP6_CORRECTIONS_SCHEMA_VERSION:
        raise Wp6ProvenanceResolutionError(
            "unexpected corrections schema %r; expected %r"
            % (loaded.get("schema_version"), WP6_CORRECTIONS_SCHEMA_VERSION)
        )
    canonical_id = loaded.get("canonical_experiment_id")
    if not canonical_id:
        raise Wp6ProvenanceResolutionError("corrections index has no canonical_experiment_id")
    if not isinstance(loaded.get("entries"), dict):
        raise Wp6ProvenanceResolutionError("corrections index entries must be a mapping")
    return loaded


def canonical_experiment_id(root=None):
    """The canonical WP6 experiment id, read from the index (never inferred)."""
    return read_index(root)["canonical_experiment_id"]


def list_corrections(root=None):
    """Return every correction entry ordered by experiment id."""
    entries = read_index(root).get("entries") or {}
    return [entries[name] for name in sorted(entries)]


def is_non_canonical(experiment_id, root=None):
    """True when ``experiment_id`` is a withdrawn, non-canonical provenance record."""
    entries = read_index(root).get("entries") or {}
    entry = entries.get(experiment_id)
    if entry is None:
        return False
    return entry.get("status") != CANONICAL_STATUS or entry.get("canonical_experiment_id") != experiment_id


def _entry(experiment_id, root=None):
    index = read_index(root)
    entries = index.get("entries") or {}
    entry = entries.get(experiment_id)
    if entry is None:
        raise Wp6ProvenanceResolutionError(
            "no WP6 corrected provenance registered for %s" % experiment_id
        )
    return entry


def _load_record(basename, root=None):
    path = corrections_dir(root) / basename
    if not path.is_file():
        raise Wp6ProvenanceResolutionError("corrected record is missing: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Wp6ProvenanceResolutionError(
            "corrected record %s is not valid JSON: %s" % (path, exc)
        ) from exc


def supersession_record(experiment_id, root=None):
    """Load the withdrawal/supersession record for ``experiment_id``."""
    entry = _entry(experiment_id, root)
    basename = entry.get("supersession_file")
    if not basename:
        raise Wp6ProvenanceResolutionError(
            "correction entry for %s has no supersession_file" % experiment_id
        )
    return _load_record(basename, root)


def canonical_provenance_path(root=None):
    """Absolute path of the canonical WP6 provenance record.

    The canonical record's file name is read from the index (keyed by the
    canonical experiment id), never derived from directory ordering.
    """
    index = read_index(root)
    basename = index.get("canonical_provenance_file")
    if not basename:
        raise Wp6ProvenanceResolutionError("corrections index has no canonical_provenance_file")
    return corrections_dir(root) / basename


def load_canonical_experiment(root=None):
    """Load the canonical WP6 provenance record.

    Cross-checks the record against the index so a mismatch is an integrity
    failure rather than a silent pass.
    """
    index = read_index(root)
    canonical_id = index["canonical_experiment_id"]
    record = _load_record(index["canonical_provenance_file"], root)
    record_id = record.get("experiment_id") or record.get("canonical_experiment_id")
    if record_id != canonical_id:
        raise Wp6ProvenanceResolutionError(
            "canonical record id %r disagrees with index %r" % (record_id, canonical_id)
        )
    if record.get("canonical_experiment_id") not in (None, canonical_id):
        raise Wp6ProvenanceResolutionError(
            "canonical record canonical_experiment_id %r disagrees with index %r"
            % (record.get("canonical_experiment_id"), canonical_id)
        )
    return record


def canonical_wp6_experiment(root=None):
    """Resolve the canonical WP6 experiment.

    Always returns the canonical experiment; the withdrawn misbound record can
    never be returned here because the canonical id comes from the index and the
    loaded record is cross-checked against it.
    """
    index = read_index(root)
    canonical_id = index["canonical_experiment_id"]
    record = load_canonical_experiment(root)
    entry = (index.get("entries") or {}).get(canonical_id)
    return {
        "experiment_id": canonical_id,
        "canonical_experiment_id": canonical_id,
        "canonical": True,
        "status": CANONICAL_STATUS,
        "canonical_provenance_ref": (entry or {}).get("canonical_provenance_ref")
        or record.get("canonical_provenance_ref"),
        "canonical_record": record,
        "withdrawn_experiment_ids": sorted(
            name
            for name, item in (index.get("entries") or {}).items()
            if item.get("status") != CANONICAL_STATUS
        ),
    }


def resolve_wp6_experiment(experiment_id, root=None):
    """Resolve ``experiment_id`` to its canonical/non-canonical WP6 status.

    Resolution is keyed strictly by experiment id (never by filename ordering).
    A withdrawn misbound id resolves with ``canonical=False`` and status
    ``MISBOUND_WITHDRAWN_NON_CANONICAL``; the canonical id resolves with
    ``canonical=True`` and the canonical provenance reference.
    """
    index = read_index(root)
    canonical_id = index["canonical_experiment_id"]

    if experiment_id == canonical_id:
        resolved = canonical_wp6_experiment(root)
        resolved["misbound_git_commit"] = None
        resolved["withdrawal"] = None
        return resolved

    if is_non_canonical(experiment_id, root):
        withdrawal = supersession_record(experiment_id, root)
        if withdrawal.get("canonical_experiment_id") not in (None, canonical_id) and \
                withdrawal.get("new_experiment_id") not in (None, canonical_id):
            raise Wp6ProvenanceResolutionError(
                "supersession record for %s disagrees with the index canonical id %r"
                % (experiment_id, canonical_id)
            )
        entry = _entry(experiment_id, root)
        return {
            "experiment_id": experiment_id,
            "canonical_experiment_id": canonical_id,
            "canonical": False,
            "status": withdrawal.get("status", NON_CANONICAL_STATUS),
            "canonical_provenance_ref": entry.get("canonical_provenance_ref"),
            "canonical_record": None,
            "misbound_git_commit": withdrawal.get("misbound_git_commit"),
            "withdrawal": withdrawal,
        }

    raise Wp6ProvenanceResolutionError(
        "no WP6 provenance registered for %s" % experiment_id
    )


def assert_canonical(experiment_id, root=None):
    """Raise when ``experiment_id`` is not the canonical WP6 experiment."""
    resolved = resolve_wp6_experiment(experiment_id, root)
    if not resolved.get("canonical"):
        raise Wp6ProvenanceResolutionError(
            "%s is a non-canonical/withdrawn WP6 provenance record and must never be "
            "selected as canonical" % experiment_id
        )
    return resolved
