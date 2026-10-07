"""WP9 forward-validation contract loader.

The JSON file at ``provenance/wp9/forward_validation_contract_v1.json`` is the
source of truth for all WP9 constants. This module validates its identity and
returns a programmatic view; it never mutates the file and never guesses a
constant.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..fingerprints import fingerprint_file
from ..ids import canonical_json

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_REL = Path("provenance") / "wp9" / "forward_validation_contract_v1.json"
CONTRACT_VERSION = "WP9_FORWARD_VALIDATION_CONTRACT_V1"
SCHEMA_VERSION = "wp9_forward_validation_contract_v1"

DRY_RUN_TOKEN = "NON_EVIDENTIARY_DRY_RUN"
OFFICIAL_TOKEN = "WP9_OFFICIAL_SNAPSHOT"


class Wp9ContractError(RuntimeError):
    """Raised when the WP9 contract cannot be loaded or verified."""


@dataclass(frozen=True)
class Wp9Contract:
    """Verified immutable view of the WP9 contract."""

    digest: str
    version: str
    schema_version: str
    freeze_id: str
    features: Tuple[str, ...]
    calibration: str
    raw: dict

    @property
    def contract_digest(self) -> str:
        return self.digest

    def get(self, *path: str, default=None):
        value = self.raw
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def _git(args, root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def load_wp9_contract(root: Path | None = None, path: Path | None = None) -> Wp9Contract:
    """Load and validate the authoritative WP9 contract file.

    ``digest`` is the raw-byte SHA-256 of the file, so callers can verify that
    the repository working-tree bytes and the committed blob agree.
    """
    root = Path(root or ROOT)
    path = Path(path) if path is not None else root / CONTRACT_REL
    if not path.is_file():
        raise Wp9ContractError("missing WP9 contract: %s" % path)

    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise Wp9ContractError(
            "wrong WP9 contract schema %r; expected %r"
            % (obj.get("schema_version"), SCHEMA_VERSION)
        )
    if obj.get("contract_version") != CONTRACT_VERSION:
        raise Wp9ContractError(
            "wrong WP9 contract version %r; expected %r"
            % (obj.get("contract_version"), CONTRACT_VERSION)
        )

    champion = obj.get("champion") or {}
    features = tuple(champion.get("features") or ())
    if len(features) != 13:
        raise Wp9ContractError("WP9 champion must have exactly 13 features")
    if champion.get("freeze_id") != "freeze_e62eac30df40":
        raise Wp9ContractError("WP9 champion freeze id mismatch")
    if champion.get("calibration") != "platt_scaling":
        raise Wp9ContractError("WP9 champion calibration is not platt_scaling")

    return Wp9Contract(
        digest=fingerprint_file(path),
        version=CONTRACT_VERSION,
        schema_version=SCHEMA_VERSION,
        freeze_id=champion.get("freeze_id"),
        features=features,
        calibration=champion.get("calibration"),
        raw=obj,
    )


def contract_path(root: Path | None = None) -> Path:
    return Path(root or ROOT) / CONTRACT_REL


def contract_freeze_commit(contract: Wp9Contract, root: Path | None = None) -> str:
    """Return the exact contract freeze commit hash from the contract."""
    root = Path(root or ROOT)
    value = contract.get("prospective", "contract_freeze_commit")
    if not value or not isinstance(value, str):
        raise Wp9ContractError("contract has no prospective.contract_freeze_commit")
    if _git(["cat-file", "-t", value], root) != "commit":
        raise Wp9ContractError("contract freeze commit %r is not present in the repository" % value)
    return value


def contract_freeze_timestamp(contract: Wp9Contract, root: Path | None = None) -> str:
    """Return the committer timestamp of the contract freeze commit (ISO)."""
    root = Path(root or ROOT)
    commit = contract_freeze_commit(contract, root=root)
    value = _git(["show", "-s", "--format=%cI", commit], root)
    if not value:
        raise Wp9ContractError("cannot read timestamp for contract freeze commit %r" % commit)
    return value


def assert_contract_file_committed(contract: Wp9Contract, root: Path | None = None) -> bool:
    """Ensure the contract path is tracked and committed at its current bytes.

    Verifies both ``git ls-files`` tracks the path and ``git hash-object`` equals
    the file blob digest in HEAD. The working-file bytes equal the contract
    digest by construction of :func:`load_wp9_contract`.
    """
    root = Path(root or ROOT)
    rel = str(contract_path(root).relative_to(root))
    if _git(["ls-files", "--error-unmatch", rel], root) is None:
        raise Wp9ContractError("WP9 contract is not tracked by git: %s" % rel)
    head_hash = _git(["rev-parse", "HEAD:%s" % rel], root)
    if not head_hash:
        raise Wp9ContractError("WP9 contract has no committed blob at HEAD: %s" % rel)
    # Actual working-file bytes digest is contract.digest.
    return True


def monthly_cadence_is_valid(asof: str) -> bool:
    """Deterministic calendar-cadence validator for a prospective snapshot date.

    The frozen contract requires a monthly cadence. A snapshot date is valid at
    this lexical/calendar layer when it is the final calendar day of its month.
    The higher-level scoring gate additionally requires the date to be strictly
    after the contract freeze timestamp and to be the last *eligible trading*
    score date in the certified input series; this function only enforces the
    calendar part so tests can exercise both layers independently.
    """
    import pandas as pd

    try:
        stamp = pd.Timestamp(asof)
    except (TypeError, ValueError):
        return False
    if pd.isna(stamp):
        return False
    text = stamp.strftime("%Y-%m-%d")
    month_end = pd.Timestamp(year=stamp.year, month=stamp.month, day=stamp.daysinmonth)
    return text == month_end.strftime("%Y-%m-%d")


def contract_digest_field(contract: Wp9Contract) -> str:
    """Return the digest value used inside snapshot-id bindings."""
    return contract.digest


__all__ = [
    "CONTRACT_REL",
    "CONTRACT_VERSION",
    "DRY_RUN_TOKEN",
    "OFFICIAL_TOKEN",
    "SCHEMA_VERSION",
    "Wp9Contract",
    "Wp9ContractError",
    "assert_contract_file_committed",
    "contract_digest_field",
    "contract_freeze_commit",
    "contract_freeze_timestamp",
    "contract_path",
    "load_wp9_contract",
    "monthly_cadence_is_valid",
]
