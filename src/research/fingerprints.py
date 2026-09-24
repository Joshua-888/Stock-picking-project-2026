"""Deterministic fingerprinting helpers for V2 research artefacts.

``fingerprint_dataframe`` builds its own byte stream (domain tag, sorted column
names, row count, per-column dtype and value digests) instead of relying on
pandas hashing internals, so fingerprints stay stable across runs and pandas
versions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .ids import canonical_json

_CHUNK = 65536
_DOMAIN = b"research-df-fingerprint-v1\n"


def fingerprint_obj(obj):
    """SHA-256 over the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def fingerprint_file(path):
    """SHA-256 of the raw bytes stored at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("cannot fingerprint missing file: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_token(value):
    return canonical_json(value)


def _row_order(df):
    """Deterministic row order derived from canonicalised index values."""
    keys = []
    for value in df.index:
        keys.append(canonical_json(list(value)) if isinstance(value, tuple) else _cell_token(value))
    return sorted(range(len(df)), key=lambda position: (keys[position], position))


def fingerprint_dataframe(df):
    """Stable content fingerprint of a pandas DataFrame."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("fingerprint_dataframe expects a pandas DataFrame")
    names = [str(column) for column in df.columns]
    if len(names) != len(set(names)):
        raise ValueError("duplicate column names cannot be fingerprinted deterministically")
    label_by_name = {str(column): column for column in df.columns}
    order = _row_order(df)
    outer = hashlib.sha256()
    outer.update(_DOMAIN)
    outer.update(("rows=%d\n" % len(df)).encode("utf-8"))
    for name in sorted(names):
        series = df[label_by_name[name]]
        column_digest = hashlib.sha256()
        column_digest.update(("dtype=%s\n" % series.dtype).encode("utf-8"))
        values = series.to_numpy()
        for position in order:
            column_digest.update(_cell_token(values[position]).encode("utf-8"))
            column_digest.update(b"\x1e")
        outer.update(("column=%s\tdtype=%s\tdigest=%s\n" % (name, series.dtype, column_digest.hexdigest())).encode("utf-8"))
    return outer.hexdigest()
