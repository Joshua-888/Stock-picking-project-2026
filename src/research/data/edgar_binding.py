"""Deterministic content binding for the SEC EDGAR input files.

The one-shot holdout evaluator must bind its feature inputs to the exact bytes
that produced the frozen model and training fingerprint.  This module builds,
verifies and loads that aggregate binding without inspecting locked-holdout
target labels, returns, predictions or outcome rows.

Binding schema
--------------
The binding is intentionally free of wall-clock values.  It contains only
byte digests and sidecar metadata, so identical input files always produce an
identical binding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..fingerprints import fingerprint_file
from ..ids import canonical_json

BINDING_SCHEMA_VERSION = "edgar_input_binding_v1"
FUNDAMENTALS_REL = Path("data") / "research_v2" / "silver" / "edgar_fundamentals"
EDGAR_CIK_RELATIVE = "artifacts/research/wp4/edgar_cik_mapping.json"

# Matches the existing wp5.load_fundamentals column selection exactly.
FUNDAMENTAL_COLUMNS = [
    "cik",
    "field",
    "value",
    "fiscal_period_start",
    "fiscal_period_end",
    "accession",
    "available_at",
    "form",
]


class EdgarBindingError(RuntimeError):
    """Raised when the EDGAR input binding cannot be built or verified honestly."""


def _fundamentals_base(root):
    return Path(root) / FUNDAMENTALS_REL


def _cik_path(root):
    return Path(root) / EDGAR_CIK_RELATIVE


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise EdgarBindingError("EDGAR input sidecar is missing: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise EdgarBindingError("invalid JSON EDGAR input sidecar %s: %s" % (path, exc)) from exc


def _shard_sidecar(provenance_path):
    """Return row_count/columns from a required provenance.json sidecar."""
    payload = _read_json(provenance_path)
    try:
        row_count = payload["row_count"]
        columns = payload["columns"]
    except KeyError as exc:
        raise EdgarBindingError(
            "EDGAR shard provenance sidecar is missing %s: %s" % (exc.args[0], provenance_path)
        ) from exc
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise EdgarBindingError(
            "EDGAR shard provenance sidecar row_count must be a non-negative integer: %s"
            % provenance_path
        )
    if not isinstance(columns, list) or not all(isinstance(name, str) for name in columns):
        raise EdgarBindingError(
            "EDGAR shard provenance sidecar columns must be a list of strings: %s"
            % provenance_path
        )
    return int(row_count), list(columns)


def _scan_shards(root):
    """Return deterministic sorted (version, data_path, provenance_path, metadata)."""
    base = _fundamentals_base(root)
    if not base.is_dir():
        raise EdgarBindingError("EDGAR fundamentals silver directory is missing: %s" % base)

    found = []
    for child in sorted(base.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        data_path = child / "data.parquet"
        if not data_path.is_file():
            continue
        version = child.name
        if len(version) != 16:
            raise EdgarBindingError(
                "EDGAR shard directory name is not 16 characters: %s" % child
            )
        provenance_path = child / "provenance.json"
        if not provenance_path.is_file():
            raise EdgarBindingError(
                "EDGAR shard has data.parquet but no provenance.json sidecar: %s" % child
            )
        row_count, columns = _shard_sidecar(provenance_path)
        found.append((version, data_path, provenance_path, row_count, columns))

    if not found:
        raise EdgarBindingError("no EDGAR silver fundamentals shards are present: %s" % base)
    # The iterdir order is already sorted by name, but re-sort for explicitness.
    return sorted(found, key=lambda entry: entry[0])


def _shard_payload(entry):
    version, data_path, provenance_path, row_count, columns = entry
    return {
        "version": version,
        "data_sha256": fingerprint_file(data_path),
        "provenance_sha256": fingerprint_file(provenance_path),
        "row_count": row_count,
        "columns": columns,
    }


def _shard_manifest_sha256(shards):
    return hashlib.sha256(canonical_json(shards).encode("utf-8")).hexdigest()


def build_edgar_input_binding(root):
    """Build the aggregate EDGAR input binding from the current local files."""
    root = Path(root)
    entries = _scan_shards(root)
    shards = [_shard_payload(entry) for entry in entries]

    cik_path = _cik_path(root)
    if not cik_path.is_file():
        raise EdgarBindingError("EDGAR CIK mapping is missing: %s" % cik_path)

    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "edgar_fundamentals": {
            "shards": shards,
            "shard_manifest_sha256": _shard_manifest_sha256(shards),
        },
        "edgar_cik_mapping": {
            "path": EDGAR_CIK_RELATIVE,
            "sha256": fingerprint_file(cik_path),
        },
    }


def _required_binding_parts(binding):
    if not isinstance(binding, dict):
        raise EdgarBindingError("EDGAR input binding must be a JSON object")
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise EdgarBindingError(
            "EDGAR input binding has unexpected schema_version %r; expected %r"
            % (binding.get("schema_version"), BINDING_SCHEMA_VERSION)
        )
    try:
        fundamentals = binding["edgar_fundamentals"]
        shards = fundamentals["shards"]
        manifest = fundamentals["shard_manifest_sha256"]
        cik_binding = binding["edgar_cik_mapping"]
        cik_path = cik_binding["path"]
        cik_sha = cik_binding["sha256"]
    except (KeyError, TypeError) as exc:
        raise EdgarBindingError(
            "EDGAR input binding is missing required field %r" % exc.args[0]
        ) from exc
    if not isinstance(shards, list):
        raise EdgarBindingError(
            "EDGAR input binding edgar_fundamentals.shards must be a list"
        )
    if not isinstance(manifest, str) or len(manifest) != 64:
        raise EdgarBindingError(
            "EDGAR input binding shard_manifest_sha256 must be a 64-character SHA-256 hex digest"
        )
    if cik_path != EDGAR_CIK_RELATIVE:
        raise EdgarBindingError(
            "EDGAR input binding CIK mapping path is %r; expected %r"
            % (cik_path, EDGAR_CIK_RELATIVE)
        )
    if not isinstance(cik_sha, str) or len(cik_sha) != 64:
        raise EdgarBindingError(
            "EDGAR input binding edgar_cik_mapping.sha256 must be a 64-character SHA-256 hex digest"
        )
    return fundamentals, shards, manifest, cik_binding, cik_sha


def _verify_edgar_fundamentals(root, shards, expected_manifest):
    entries = _scan_shards(root)
    actual_by_version = {entry[0]: entry for entry in entries}
    expected_by_version = {}
    for shard in shards:
        if not isinstance(shard, dict):
            raise EdgarBindingError("EDGAR input binding contains a non-object shard entry")
        try:
            version = shard["version"]
        except KeyError as exc:
            raise EdgarBindingError("EDGAR shard binding is missing field %r" % exc.args[0]) from exc
        if version in expected_by_version:
            raise EdgarBindingError("EDGAR shard binding contains duplicate version %r" % version)
        expected_by_version[version] = shard

    actual_versions = set(actual_by_version)
    expected_versions = set(expected_by_version)
    missing = sorted(expected_versions - actual_versions)
    extra = sorted(actual_versions - expected_versions)
    if missing:
        raise EdgarBindingError(
            "EDGAR input binding verification failed: missing shard(s) %s; "
            "action=restore the bound data or re-materialize the binding" % ", ".join(missing)
        )
    if extra:
        raise EdgarBindingError(
            "EDGAR input binding verification failed: extra shard(s) not present in the binding %s; "
            "action=remove the extra shard or re-materialize the binding" % ", ".join(extra)
        )

    for version in sorted(expected_by_version):
        expected = expected_by_version[version]
        actual = actual_by_version[version]
        actual_payload = _shard_payload(actual)

        try:
            expected_data_sha = expected["data_sha256"]
            expected_prov_sha = expected["provenance_sha256"]
            expected_row_count = expected["row_count"]
            expected_columns = expected["columns"]
        except KeyError as exc:
            raise EdgarBindingError(
                "EDGAR shard binding for version %s is missing field %r; "
                "action=re-materialize the binding" % (version, exc.args[0])
            ) from exc

        if expected_data_sha != actual_payload["data_sha256"]:
            raise EdgarBindingError(
                "EDGAR input binding verification failed for %s/data.parquet: "
                "sha256 mismatch (bound=%s actual=%s); "
                "action=restore the bound bytes or re-materialize the binding"
                % (version, expected_data_sha, actual_payload["data_sha256"])
            )
        if expected_prov_sha != actual_payload["provenance_sha256"]:
            raise EdgarBindingError(
                "EDGAR input binding verification failed for %s/provenance.json: "
                "sha256 mismatch (bound=%s actual=%s); "
                "action=restore the bound bytes or re-materialize the binding"
                % (version, expected_prov_sha, actual_payload["provenance_sha256"])
            )
        if expected_row_count != actual_payload["row_count"]:
            raise EdgarBindingError(
                "EDGAR input binding row_count changed for shard %s: "
                "bound=%r actual=%r; action=re-materialize the binding"
                % (version, expected_row_count, actual_payload["row_count"])
            )
        if list(expected_columns) != list(actual_payload["columns"]):
            raise EdgarBindingError(
                "EDGAR input binding columns changed for shard %s; "
                "action=restore the bound sidecar or re-materialize the binding" % version
            )

    actual_manifest = _shard_manifest_sha256(
        [_shard_payload(actual_by_version[version]) for version in sorted(actual_by_version)]
    )
    if actual_manifest != expected_manifest:
        raise EdgarBindingError(
            "EDGAR input binding shard_manifest_sha256 mismatch: "
            "bound=%s actual=%s; action=re-materialize the binding"
            % (expected_manifest, actual_manifest)
        )


def _verify_edgar_cik(root, expected_sha):
    cik_path = _cik_path(root)
    if not cik_path.is_file():
        raise EdgarBindingError(
            "EDGAR CIK mapping is missing at bound path: %s; action=restore the file" % cik_path
        )
    actual_sha = fingerprint_file(cik_path)
    if actual_sha != expected_sha:
        raise EdgarBindingError(
            "EDGAR CIK mapping sha256 mismatch at %s: bound=%s actual=%s; "
            "action=restore the bound bytes or re-materialize the binding"
            % (cik_path, expected_sha, actual_sha)
        )


def verify_edgar_input_binding(root, binding):
    """Recompute the current EDGAR input digests and compare them, fail-closed."""
    fundamentals, shards, manifest, _cik_binding, cik_sha = _required_binding_parts(binding)
    _verify_edgar_fundamentals(root, shards, manifest)
    _verify_edgar_cik(root, cik_sha)
    return True


def load_and_verify_edgar_fundamentals(root, binding):
    """Verify the binding and read exactly the bound fundamentals shards."""
    root = Path(root)
    fundamentals, shards, _manifest, _cik_binding, _cik_sha = _required_binding_parts(binding)
    _verify_edgar_fundamentals(root, shards, binding["edgar_fundamentals"]["shard_manifest_sha256"])

    base = _fundamentals_base(root)
    frames = []
    for shard in shards:
        data_path = base / shard["version"] / "data.parquet"
        if not data_path.is_file():
            raise EdgarBindingError(
                "bound EDGAR fundamentals shard is missing: %s; action=restore the file" % data_path
            )
        frames.append(pd.read_parquet(data_path, columns=FUNDAMENTAL_COLUMNS))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(
        subset=["cik", "field", "fiscal_period_end", "value", "accession"]
    )
    return frame


def load_and_verify_cik_by_ticker(root, binding):
    """Verify the CIK mapping digest, parse it, and build by-ticker mappings."""
    root = Path(root)
    _required_binding_parts(binding)
    _verify_edgar_cik(root, binding["edgar_cik_mapping"]["sha256"])

    cik_path = _cik_path(root)
    payload = _read_json(cik_path)
    mapping = payload.get("mapping") or {}
    by_ticker = {}
    for ticker, record in mapping.items():
        cik = record.get("cik")
        if record.get("match") == "EXACT" and cik is not None:
            by_ticker[str(ticker).upper()] = str(int(cik)).zfill(10)
    return by_ticker, payload