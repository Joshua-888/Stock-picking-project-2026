"""Common frozen artifact identity for the WP9 dashboard truth layer.

Every dashboard contract carries one identical identity block so a future UI
and any auditor can bind emitted numbers back to the exact frozen champion,
feature order, WP9 contract digest, and producing Git commit without access to
target data, holdout labels, or training internals.

This module is strictly read-only: it reads the WP8 freeze record and the WP9
contract, and resolves the producing commit with ``git rev-parse HEAD``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping

from . import schemas

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

FREEZE_REL = "provenance/wp8/final_candidate_freeze.json"
CONTRACT_REL = "provenance/wp9/forward_validation_contract_v1.json"
EXPECTED_FREEZE_ID = "freeze_e62eac30df40"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head(root: Path) -> str | None:
    """Resolve ``git rev-parse HEAD`` without assuming a fixed local path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def _load_json(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError("frozen identity provenance missing: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_identity(
    root: Path | None = None,
    *,
    producing_commit: str | None = None,
) -> dict:
    """Return the common frozen-identity block (read-only).

    ``producing_commit`` exists only for deterministic test injection; the
    default is resolved from the repository HEAD with ``git rev-parse HEAD``.
    The contract digest is the SHA-256 of the raw WP9 contract file bytes.
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT
    freeze = _load_json(root, FREEZE_REL)
    contract_path = root / CONTRACT_REL
    if not contract_path.is_file():
        raise FileNotFoundError("missing WP9 contract for identity: %s" % contract_path)
    contract_bytes = contract_path.read_bytes()

    freeze_id = str(freeze.get("freeze_id") or "")
    if freeze_id != EXPECTED_FREEZE_ID:
        raise ValueError("unexpected frozen freeze_id %r" % freeze_id)

    artifact_hashes = freeze.get("artifact_hashes") or {}

    def _hash(name: str) -> str:
        spec = artifact_hashes.get(name) or {}
        value = str(spec.get("sha256") or "")
        if len(value) != 64:
            raise ValueError("freeze record is missing sha256 for %s" % name)
        return value

    commit = producing_commit if producing_commit is not None else _git_head(root)
    if not commit:
        commit = str(freeze.get("producing_commit") or "")

    identity = {
        "freeze_id": freeze_id,
        "model_hash": _hash("model"),
        "preprocessor_hash": _hash("preprocessor"),
        "calibrator_hash": _hash("calibrator"),
        "feature_order": list(schemas.FEATURE_ORDER),
        "contract_digest": _sha256_bytes(contract_bytes),
        "producing_commit": commit,
        "dataset_id": str(freeze.get("dataset_id") or ""),
        "feature_set_id": str(freeze.get("feature_set_id") or ""),
        "target_set_id": str(freeze.get("target_set_id") or ""),
    }
    schemas.validate_artifact_identity(identity)
    return identity


def attach_identity(
    contract: Mapping,
    root: Path | None = None,
    *,
    producing_commit: str | None = None,
) -> dict:
    """Return ``contract`` extended with the common frozen-identity block.

    The identity is computed against the real repository (freeze record + WP9
    contract at the package's default root unless ``root`` is supplied), never
    against an arbitrary registry root.
    """
    identity_root = root if root is not None else _DEFAULT_ROOT
    out = dict(contract)
    out["artifact_identity"] = artifact_identity(
        identity_root, producing_commit=producing_commit
    )
    return out


__all__ = ["artifact_identity", "attach_identity"]
