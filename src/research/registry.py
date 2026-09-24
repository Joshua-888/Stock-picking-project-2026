"""Append-only experiment registry persisted as JSON files.

Records are write-once: registering an identical payload verifies and reuses the
stored record, while a different payload under the same experiment ID raises and
leaves the original evidence untouched. Nothing is ever deleted or overwritten.
"""

from __future__ import annotations

from pathlib import Path

from .ids import canonical_json, is_valid_id
from .immutability import save_immutable

DEFAULT_REGISTRY_DIR = Path("artifacts/research/registry")

ALLOWED_STATUSES = (
    "CREATED",
    "RUNNING",
    "FAILED",
    "COMPLETED",
    "VALIDATED",
    "AUDITED",
    "REJECTED",
    "PROMOTED",
)

REQUIRED_FIELDS = (
    "experiment_id",
    "hypothesis",
    "created_at",
    "git_commit",
    "branch",
    "dataset_id",
    "feature_set_id",
    "target_set_id",
    "model_id",
    "config",
    "random_seed",
    "training_period",
    "validation_period",
    "holdout_usage",
    "artifacts",
    "status",
)

_ID_FIELDS = {
    "experiment_id": "experiment",
    "dataset_id": "dataset",
    "feature_set_id": "feature_set",
    "target_set_id": "target_set",
    "model_id": "model",
}

_OPTIONAL_ID_FIELDS = ("feature_set_id", "target_set_id", "model_id")


class RegistryError(RuntimeError):
    """Raised when a registry record is invalid or would overwrite evidence."""


class UnknownExperimentError(RegistryError, KeyError):
    """Raised when an experiment ID is not present in the registry."""


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _normalise(payload):
    """Validate a raw experiment payload and return a canonical record mapping."""
    if not isinstance(payload, dict):
        raise RegistryError("experiment payload must be a mapping, got %s" % type(payload).__name__)
    problems = []
    unknown = sorted(set(payload) - set(REQUIRED_FIELDS))
    if unknown:
        problems.append("unknown field(s): %s" % ", ".join(unknown))
    for name in REQUIRED_FIELDS:
        if name in _OPTIONAL_ID_FIELDS:
            continue
        if name not in payload or _missing(payload.get(name)):
            problems.append("missing required field %r" % name)
    for name, kind in _ID_FIELDS.items():
        if name in _OPTIONAL_ID_FIELDS and payload.get(name) is None:
            continue
        value = payload.get(name)
        if value is not None and not is_valid_id(value, kind):
            problems.append("field %r is not a valid %s identifier: %r" % (name, kind, value))
    status = payload.get("status")
    if status is not None and status not in ALLOWED_STATUSES:
        problems.append("status %r not in %s" % (status, ", ".join(ALLOWED_STATUSES)))
    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, list):
        problems.append("artifacts must be a list of paths")
    if problems:
        raise RegistryError("invalid experiment record: %s" % "; ".join(problems))
    return {name: payload.get(name) for name in REQUIRED_FIELDS}


class ExperimentRegistry:
    """File-backed, append-only registry of experiment records."""

    def __init__(self, registry_dir=DEFAULT_REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, experiment_id):
        return self.registry_dir / ("%s.json" % experiment_id)

    def register(self, payload):
        """Register an experiment; return ``(experiment_id, outcome)``.

        ``outcome`` is ``"written"`` for a new record and ``"verify_and_reuse"``
        when the identical record already existed. Raises :class:`RegistryError`
        if different content is supplied for an existing experiment ID.
        """
        record = _normalise(payload)
        target = self.path_for(record["experiment_id"])
        try:
            outcome = save_immutable(target, record)
        except Exception as exc:  # ImmutabilityError and friends
            raise RegistryError(
                "refusing to overwrite registered experiment %s: %s" % (record["experiment_id"], exc)
            ) from exc
        return record["experiment_id"], outcome

    def get(self, experiment_id):
        """Return the stored record for ``experiment_id``."""
        path = self.path_for(experiment_id)
        if not path.is_file():
            raise UnknownExperimentError("experiment not registered: %s" % experiment_id)
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def list_experiments(self):
        """Return every stored record, ordered by experiment ID."""
        import json

        records = []
        for path in sorted(self.registry_dir.glob("experiment_*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records

    def fingerprint(self, experiment_id):
        """Canonical-content fingerprint of one stored record."""
        import hashlib

        return hashlib.sha256(canonical_json(self.get(experiment_id)).encode("utf-8")).hexdigest()
