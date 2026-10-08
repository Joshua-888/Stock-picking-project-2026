"""Bronze / Silver / Gold storage layers for V2 research data.

BRONZE holds raw provider payloads, append-only and immutable: the same bytes
verify and reuse the stored copy, and changed bytes create a NEW version rather
than mutating history. SILVER holds normalised observations with provenance.
GOLD holds point-in-time model-ready tables.

All layouts live under ``data/research_v2/<layer>/<name>/``. Writes are atomic
(temp file + ``os.replace``) so a crash cannot leave a half-written artefact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("data/research_v2")
LAYERS = ("bronze", "silver", "gold")


class LayerError(RuntimeError):
    """Raised when a layer write would corrupt existing evidence."""


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def layer_root(root=None, layer=None, name=None):
    """Resolve ``<root>/<layer>/<name>`` (or a parent path) as a Path."""
    base = Path(root) if root is not None else DEFAULT_ROOT
    if layer is None:
        return base
    base = base / layer
    if name is None:
        return base
    return base / name


def write_bronze_bytes(root, name, payload, ext="json", meta=None):
    """Append-only, content-addressed bronze write.

    The raw payload is stored under ``bronze/<name>/<fingerprint>/raw.<ext>``.
    Re-writing identical bytes returns ``outcome='verify_and_reuse'``; different
    bytes produce a different fingerprint directory, i.e. a NEW version, and
    never overwrite the previous version.
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, (bytes, bytearray)):
        raise LayerError("bronze payload must be bytes or str, got %s" % type(payload).__name__)
    payload = bytes(payload)
    fingerprint = _sha256_bytes(payload)
    version_dir = layer_root(root, "bronze", name) / fingerprint
    raw_path = version_dir / ("raw.%s" % ext)
    outcome = "verify_and_reuse"
    if raw_path.exists():
        existing = raw_path.read_bytes()
        if _sha256_bytes(existing) != fingerprint:
            raise LayerError(
                "bronze version directory %s exists but its bytes differ from the recorded fingerprint"
                % version_dir
            )
    else:
        _atomic_write_bytes(raw_path, payload)
        outcome = "written"
    record = {
        "layer": "bronze",
        "name": name,
        "version": fingerprint,
        "fingerprint": fingerprint,
        "raw_path": str(raw_path),
        "ext": ext,
        "size_bytes": len(payload),
        "outcome": outcome,
        "meta": dict(meta or {}),
    }
    _atomic_write_bytes(version_dir / "meta.json", (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return record


def write_bronze_json(root, name, obj, meta=None):
    """Bronze write of a JSON-serialisable object using canonical encoding."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return write_bronze_bytes(root, name, payload, ext="json", meta=meta)


def write_silver_table(root, name, df, meta=None, ext="parquet"):
    """Persist a normalised silver table plus a provenance sidecar.

    Returns a record carrying the dataset fingerprint (stable content hash of
    the frame) and the artefact path. The sidecar is written with the same
    verify-and-reuse discipline as bronze metadata.
    """
    import pandas as pd

    from src.research.fingerprints import fingerprint_dataframe

    if not isinstance(df, pd.DataFrame):
        raise LayerError("silver table must be a pandas DataFrame")
    version = fingerprint_dataframe(df)
    version_dir = layer_root(root, "silver", name) / version[:16]
    version_dir.mkdir(parents=True, exist_ok=True)
    table_path = version_dir / ("data.%s" % ext)
    if ext == "parquet":
        handle_fd, tmp_path = tempfile.mkstemp(dir=str(version_dir), prefix=".tmp-", suffix=".parquet")
        os.close(handle_fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, table_path)
    elif ext == "json":
        _atomic_write_bytes(table_path, (df.to_json(orient="records", date_format="iso") + "\n").encode("utf-8"))
    else:
        raise LayerError("unsupported silver extension %r" % ext)
    record = {
        "layer": "silver",
        "name": name,
        "version": version[:16],
        "fingerprint": version,
        "table_path": str(table_path),
        "row_count": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "meta": dict(meta or {}),
    }
    _atomic_write_bytes(version_dir / "provenance.json", (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return record


def write_gold_table(root, name, df, meta=None, ext="parquet"):
    """Persist a point-in-time model-ready gold table with provenance.

    The gold version directory PHYSICALLY HOLDS the data (``data.<ext>``) as well
    as its provenance, so an approved gold dataset is self-contained and cannot be
    silently mutated by a later silver write. The bytes are copied from the same
    content-addressed table, so ``fingerprint`` and legacy reads (which go through
    :func:`read_silver_table`) stay byte-identical.
    """
    import shutil

    record = write_silver_table(root, name, df, meta=meta, ext=ext)
    record["layer"] = "gold"
    version_dir = layer_root(root, "gold", name) / record["version"]
    version_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(record["table_path"])
    gold_path = version_dir / ("data.%s" % ext)
    if source_path.is_file() and not gold_path.exists():
        shutil.copyfile(str(source_path), str(gold_path))
    record["gold_table_path"] = str(gold_path)
    _atomic_write_bytes(version_dir / "provenance.json", (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return record


def read_silver_table(root, name, version=None):
    """Read a silver table back; newest version when ``version`` is omitted."""
    import pandas as pd

    base = layer_root(root, "silver", name)
    if version is None:
        versions = sorted(path.name for path in base.iterdir() if path.is_dir())
        if not versions:
            raise LayerError("no silver versions stored for %r" % name)
        version = versions[-1]
    return pd.read_parquet(base / version / "data.parquet")


def source_fingerprint(root, name):
    """Aggregate fingerprint of every stored bronze version of ``name``."""
    base = layer_root(root, "bronze", name)
    if not base.is_dir():
        raise LayerError("no bronze data stored for %r" % name)
    versions = sorted(path.name for path in base.iterdir() if path.is_dir())
    digest = hashlib.sha256()
    for version in versions:
        digest.update(version.encode("utf-8"))
    return digest.hexdigest()


def find_bronze_by_fingerprint(root, name, fingerprint):
    """Return the bronze version record matching ``fingerprint``, or None."""
    version_dir = layer_root(root, "bronze", name) / fingerprint
    meta_path = version_dir / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
