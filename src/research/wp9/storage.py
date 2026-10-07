"""Immutable WP9 snapshot storage and registries.

Every official prediction snapshot is written once under
``provenance/wp9/predictions/<snapshot_id>.json``. Dry-run outputs are written
under ``provenance/wp9/dry_runs/<snapshot_id>.json`` and can never enter the
official prediction index.

The snapshot file body contains exactly the frozen
``snapshot_schema_fields``. Snapshot identity bindings are stored in the index
entry, never appended as extra columns in the snapshot body.

Snapshot identity is deterministic and binds the contract digest, as-of date,
frozen champion, all artefact hashes, universe/manifest/feature fingerprints, and
the producing git commit. Nothing is removed or overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..ids import canonical_json
from ..immutability import save_immutable, write_json_atomic

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DIR_REL = Path("provenance") / "wp9" / "predictions"
DRY_RUNS_DIR_REL = Path("provenance") / "wp9" / "dry_runs"
PREDICTION_INDEX_REL = PREDICTIONS_DIR_REL / "index.json"
DRY_RUN_INDEX_REL = DRY_RUNS_DIR_REL / "index.json"
RUN_INDEX_REL = Path("provenance") / "wp9" / "index.json"

PREDICTION_INDEX_SCHEMA = "wp9_prediction_index_v1"
RUN_INDEX_SCHEMA = "wp9_run_registry_v1"
SNAPSHOT_SCHEMA_VERSION = "wp9_forward_validation_predictions_v1"

OFFICIAL_KIND = "WP9_OFFICIAL_SNAPSHOT"
DRY_RUN_KIND = "NON_EVIDENTIARY_DRY_RUN"

BINDINGS = (
    "contract_digest",
    "snapshot_asof",
    "champion_freeze",
    "model_hash",
    "preprocessor_hash",
    "calibrator_hash",
    "universe_hash",
    "source_manifest_hash",
    "feature_snapshot_hash",
    "code_commit",
)

SNAPSHOT_SCHEMA_FIELDS = (
    "snapshot_id",
    "snapshot_asof",
    "security_id",
    "ticker",
    "raw_model_score",
    "frozen_calibrated_score",
    "rank",
    "percentile",
    "universe_size",
    "model_freeze_id",
    "model_hash",
    "feature_snapshot_hash",
    "source_manifest_hash",
    "code_commit",
    "created_at_utc",
    "eligibility_status",
    "quality_flags",
)


class Wp9StorageError(RuntimeError):
    """Raised when a WP9 snapshot or registry write would break immutability."""


class NoCanonicalRegistryError(Wp9StorageError):
    """Raised when no canonical WP9 run registry exists yet."""


def _require(root: Path | None) -> Path:
    return Path(root or ROOT)


def snapshot_id(
    *,
    contract_digest: str,
    snapshot_asof: str,
    champion_freeze: str,
    model_hash: str,
    preprocessor_hash: str,
    calibrator_hash: str,
    universe_hash: str,
    source_manifest_hash: str,
    feature_snapshot_hash: str,
    code_commit: str,
) -> str:
    """Deterministic official snapshot id per the frozen contract."""
    parts = [
        contract_digest,
        snapshot_asof,
        champion_freeze,
        model_hash,
        preprocessor_hash,
        calibrator_hash,
        universe_hash,
        source_manifest_hash,
        feature_snapshot_hash,
        code_commit,
    ]
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return "wp9_%s" % digest[:20]


def bindings_from_snapshot_id_inputs(
    *,
    contract_digest: str,
    snapshot_asof: str,
    champion_freeze: str,
    model_hash: str,
    preprocessor_hash: str,
    calibrator_hash: str,
    universe_hash: str,
    source_manifest_hash: str,
    feature_snapshot_hash: str,
    code_commit: str,
) -> Dict[str, str]:
    """Return the canonical bindings mapping used for identity and indexing."""
    return {
        "contract_digest": str(contract_digest),
        "snapshot_asof": str(snapshot_asof),
        "champion_freeze": str(champion_freeze),
        "model_hash": str(model_hash),
        "preprocessor_hash": str(preprocessor_hash),
        "calibrator_hash": str(calibrator_hash),
        "universe_hash": str(universe_hash),
        "source_manifest_hash": str(source_manifest_hash),
        "feature_snapshot_hash": str(feature_snapshot_hash),
        "code_commit": str(code_commit),
    }


def snapshot_id_from_bindings(bindings: Mapping[str, Any]) -> str:
    """Compute the snapshot id from a canonical bindings mapping."""
    return snapshot_id(**{name: bindings[name] for name in BINDINGS})


def validate_bindings(bindings: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(bindings, Mapping):
        raise Wp9StorageError("bindings must be a mapping")
    missing = [name for name in BINDINGS if name not in bindings]
    if missing:
        raise Wp9StorageError("bindings missing field(s): %s" % ", ".join(missing))
    return {name: str(bindings[name]) for name in BINDINGS}


def snapshot_payload(
    *,
    snapshot_id: str,
    snapshot_asof: str,
    security_id: Sequence[str],
    ticker: Sequence[str],
    raw_model_score: Sequence[float],
    frozen_calibrated_score: Sequence[float],
    rank: Sequence[int],
    percentile: Sequence[float],
    universe_size: int,
    model_hash: str,
    feature_snapshot_hash: str,
    source_manifest_hash: str,
    code_commit: str,
    created_at_utc: str,
    eligibility_status: Sequence[str],
    quality_flags: Sequence[object],
    model_freeze_id: str = "freeze_e62eac30df40",
) -> Dict[str, Any]:
    """Normalise one row-wise snapshot frame into the EXACT contract schema."""
    count = len(security_id)
    if not (
        count
        == len(ticker)
        == len(raw_model_score)
        == len(frozen_calibrated_score)
        == len(rank)
        == len(percentile)
        == len(eligibility_status)
        == len(quality_flags)
    ):
        raise Wp9StorageError("snapshot arrays must have equal length")
    return {
        "snapshot_id": snapshot_id,
        "snapshot_asof": snapshot_asof,
        "security_id": [str(value) for value in security_id],
        "ticker": [str(value) for value in ticker],
        "raw_model_score": [float(value) for value in raw_model_score],
        "frozen_calibrated_score": [float(value) for value in frozen_calibrated_score],
        "rank": [int(value) for value in rank],
        "percentile": [float(value) for value in percentile],
        "universe_size": int(universe_size),
        "model_freeze_id": model_freeze_id,
        "model_hash": model_hash,
        "feature_snapshot_hash": feature_snapshot_hash,
        "source_manifest_hash": source_manifest_hash,
        "code_commit": code_commit,
        "created_at_utc": created_at_utc,
        "eligibility_status": [str(value) for value in eligibility_status],
        "quality_flags": [list(value) if isinstance(value, (list, tuple)) else value for value in quality_flags],
    }


def _canonical(entry: Dict[str, Any]) -> str:
    return canonical_json(entry)


def _load_prediction_index(path: Path, schema: str) -> Dict[str, Any]:
    if not path.is_file():
        return {"schema_version": schema, "entries": []}
    import json

    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != schema:
        raise Wp9StorageError("unexpected registry schema at %s" % path)
    if not isinstance(obj.get("entries"), list):
        raise Wp9StorageError("registry entries missing at %s" % path)
    return obj


def _append_unique_index(
    path: Path,
    schema: str,
    entry: Dict[str, Any],
    *,
    duplicate_asof_field: str | None = None,
) -> str:
    """Append one canonical entry to an index without deleting anything.

    Identical rediscovery returns ``verify_and_reuse``. A different entry with
    the same duplicate key raises and leaves the file untouched.
    """
    obj = _load_prediction_index(path, schema)
    existing = obj["entries"]
    probe = _canonical(entry)
    for prior in existing:
        if _canonical(prior) == probe:
            return "verify_and_reuse"
    if duplicate_asof_field:
        for prior in existing:
            if prior.get(duplicate_asof_field) == entry.get(duplicate_asof_field):
                raise Wp9StorageError(
                    "duplicate %s %r cannot be appended to %s"
                    % (duplicate_asof_field, entry.get(duplicate_asof_field), path)
                )
    existing.append(entry)
    write_json_atomic(path, obj)
    return "written"


def _index_entry(
    snapshot_record: Dict[str, Any],
    bindings: Dict[str, str],
    kind: str,
    path: Path,
    root: Path,
) -> Dict[str, Any]:
    return {
        "snapshot_id": snapshot_record["snapshot_id"],
        "snapshot_asof": snapshot_record["snapshot_asof"],
        "snapshot_kind": kind,
        "model_freeze_id": snapshot_record["model_freeze_id"],
        "model_hash": snapshot_record["model_hash"],
        "feature_snapshot_hash": snapshot_record["feature_snapshot_hash"],
        "source_manifest_hash": snapshot_record["source_manifest_hash"],
        "universe_size": snapshot_record["universe_size"],
        "code_commit": snapshot_record["code_commit"],
        "created_at_utc": snapshot_record["created_at_utc"],
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bindings": dict(bindings),
    }


def write_snapshot(
    record: Dict[str, Any],
    bindings: Mapping[str, Any],
    *,
    kind: str,
    root: Path | None = None,
) -> Dict[str, Any]:
    """Write a snapshot record immutably and append its canonical index entry.

    ``kind`` must be :data:`OFFICIAL_KIND` or :data:`DRY_RUN_KIND`. Official
    snapshots live in the prediction index; dry runs live only in the separated
    dry-run index and are forbidden from the official prediction index.
    """
    if kind not in (OFFICIAL_KIND, DRY_RUN_KIND):
        raise Wp9StorageError("unknown WP9 snapshot kind %r" % kind)
    bindings = validate_bindings(bindings)
    root = _require(root)
    expected_id = snapshot_id_from_bindings(bindings)
    if record.get("snapshot_id") != expected_id:
        raise Wp9StorageError(
            "snapshot id mismatch: record=%s computed=%s" % (record.get("snapshot_id"), expected_id)
        )
    unexpected = sorted(set(record) - set(SNAPSHOT_SCHEMA_FIELDS))
    missing = sorted(set(SNAPSHOT_SCHEMA_FIELDS) - set(record))
    if unexpected:
        raise Wp9StorageError("snapshot body has unexpected field(s): %s" % ", ".join(unexpected))
    if missing:
        raise Wp9StorageError("snapshot body is missing field(s): %s" % ", ".join(missing))

    if kind == OFFICIAL_KIND:
        directory = root / PREDICTIONS_DIR_REL
        index_rel = PREDICTION_INDEX_REL
    else:
        directory = root / DRY_RUNS_DIR_REL
        index_rel = DRY_RUN_INDEX_REL

    path = directory / ("%s.json" % expected_id)
    outcome = save_immutable(path, record)
    entry = _index_entry(record, bindings, kind, path, root)
    index_outcome = _append_unique_index(
        root / index_rel,
        PREDICTION_INDEX_SCHEMA,
        entry,
        duplicate_asof_field="snapshot_asof",
    )
    return {"path": str(path), "outcome": outcome, "index_outcome": index_outcome, "snapshot_id": expected_id}


def canonicalise_prediction_index(
    root: Path | None = None,
) -> Dict[str, Any]:
    """Return the official prediction index (an empty one when absent)."""
    return _load_prediction_index(_require(root) / PREDICTION_INDEX_REL, PREDICTION_INDEX_SCHEMA)


def canonicalise_dry_run_index(root: Path | None = None) -> Dict[str, Any]:
    """Return the separated dry-run index (an empty one when absent)."""
    return _load_prediction_index(_require(root) / DRY_RUN_INDEX_REL, PREDICTION_INDEX_SCHEMA)


def has_prediction_for_asof(asof: str, root: Path | None = None) -> bool:
    """True when the official prediction index already has ``asof``."""
    index = canonicalise_prediction_index(root)
    return any(entry.get("snapshot_asof") == asof for entry in index["entries"])


def _current_stage(matured_count: int) -> str:
    if matured_count >= 24:
        return "C_MAJOR_STATISTICAL_GATE"
    if matured_count >= 12:
        return "B_FIRST_INTERIM_REVIEW"
    return "A_OPERATIONAL_SHADOW"


def load_run_registry(root: Path | None = None) -> Dict[str, Any]:
    """Load the canonical WP9 run registry, refusing an uninitialised one."""
    import json

    path = _require(root) / RUN_INDEX_REL
    if not path.is_file():
        raise NoCanonicalRegistryError("WP9 run registry is uninitialised")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != RUN_INDEX_SCHEMA:
        raise Wp9StorageError("unexpected run registry schema at %s" % path)
    if not obj.get("contract_digest"):
        raise NoCanonicalRegistryError("WP9 run registry is uninitialised")
    return obj


def initialise_run_registry(
    *,
    contract_version: str,
    contract_digest: str,
    champion_freeze: str,
    root: Path | None = None,
) -> Dict[str, Any]:
    """Create the canonical WP9 run registry exactly once."""
    path = _require(root) / RUN_INDEX_REL
    if path.is_file():
        existing = load_run_registry(root)
        if existing.get("contract_digest") != contract_digest:
            raise Wp9StorageError("WP9 run registry exists with a different contract")
        return existing
    payload = {
        "schema_version": RUN_INDEX_SCHEMA,
        "contract_version": contract_version,
        "contract_digest": contract_digest,
        "champion_freeze": champion_freeze,
        "official_snapshot_ids": [],
        "dry_run_ids": [],
        "invalidated_snapshots": [],
        "supersession_records": [],
        "matured_evaluations": [],
        "current_prospective_stage": "A_OPERATIONAL_SHADOW",
        "operational": {
            "status": "OPERATIONAL_SHADOW_STARTED",
            "last_updated_at_utc": None,
        },
        "updated_at_utc": None,
    }
    write_json_atomic(path, payload)
    return payload


def add_snapshot_to_run_registry(
    snapshot_id: str,
    part: str,
    *,
    root: Path | None = None,
    match_contract: str | None = None,
) -> Dict[str, Any]:
    """Append a snapshot id to the run registry under the given list part.

    Parts:
    * ``official`` -> ``official_snapshot_ids``
    * ``dry_run`` -> ``dry_run_ids``
    * ``invalidated`` -> ``invalidated_snapshots``
    """
    root = _require(root)
    path = root / RUN_INDEX_REL
    registry = load_run_registry(root)
    if match_contract is not None and registry.get("contract_digest") != match_contract:
        raise Wp9StorageError("WP9 run registry contract digest mismatch")
    key_map = {
        "official": "official_snapshot_ids",
        "dry_run": "dry_run_ids",
        "invalidated": "invalidated_snapshots",
    }
    key = key_map.get(part)
    if key is None:
        raise Wp9StorageError("unknown run-registry part %r" % part)
    values = list(registry[key])
    if snapshot_id not in values:
        values.append(snapshot_id)
    registry[key] = values
    registry["updated_at_utc"] = _now_iso()
    write_json_atomic(path, registry)
    return registry


def add_invalidation_to_run_registry(
    snapshot_id: str,
    reason: Dict[str, Any],
    *,
    root: Path | None = None,
) -> Dict[str, Any]:
    """Record an invalidated snapshot without deleting its original bytes."""
    root = _require(root)
    registry = load_run_registry(root)
    records = list(registry["invalidated_snapshots"])
    entry = {
        "snapshot_id": snapshot_id,
        "reason": reason,
        "recorded_at_utc": _now_iso(),
    }
    records.append(entry)
    registry["invalidated_snapshots"] = records
    registry["updated_at_utc"] = _now_iso()
    write_json_atomic(root / RUN_INDEX_REL, registry)
    return registry


def add_matured_evaluation_to_run_registry(
    snapshot_id: str,
    evaluation_id: str,
    evaluation_path: str,
    *,
    root: Path | None = None,
) -> Dict[str, Any]:
    """Attach one matured evaluation record reference to the run registry."""
    root = _require(root)
    registry = load_run_registry(root)
    records = list(registry["matured_evaluations"])
    records.append(
        {
            "snapshot_id": snapshot_id,
            "evaluation_id": evaluation_id,
            "evaluation_path": evaluation_path,
        }
    )
    records.sort(key=lambda item: (item["snapshot_id"], item["evaluation_id"]))
    registry["matured_evaluations"] = records
    registry["current_prospective_stage"] = _current_stage(len(records))
    registry["updated_at_utc"] = _now_iso()
    write_json_atomic(root / RUN_INDEX_REL, registry)
    return registry


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


__all__ = [
    "BINDINGS",
    "DRY_RUN_KIND",
    "OFFICIAL_KIND",
    "PREDICTION_INDEX_REL",
    "PREDICTION_INDEX_SCHEMA",
    "RUN_INDEX_REL",
    "RUN_INDEX_SCHEMA",
    "SNAPSHOT_SCHEMA_FIELDS",
    "Wp9StorageError",
    "NoCanonicalRegistryError",
    "add_invalidation_to_run_registry",
    "add_matured_evaluation_to_run_registry",
    "add_snapshot_to_run_registry",
    "bindings_from_snapshot_id_inputs",
    "canonicalise_dry_run_index",
    "canonicalise_prediction_index",
    "has_prediction_for_asof",
    "initialise_run_registry",
    "load_run_registry",
    "snapshot_id",
    "snapshot_id_from_bindings",
    "snapshot_payload",
    "validate_bindings",
    "write_snapshot",
]
