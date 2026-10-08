"""Write-once artefact helpers; every V2 artefact is persisted through these.

Identical rewrites verify-and-reuse. A rewrite with different content raises
``ImmutabilityError`` and never touches the existing evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .ids import canonical_json


class ImmutabilityError(RuntimeError):
    """Raised when a write would change an already persisted artefact."""


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory so a rename is durable on POSIX.

    Some platforms/filesystems do not allow opening a directory read-only;
    failures here are intentionally ignored because the rename itself is the
    atomicity guarantee and the file payload has already been fsynced.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path, obj):
    """Atomically write ``obj`` as canonical JSON to ``path`` (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(obj) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def save_immutable(path, obj):
    """Persist ``obj`` exactly once and report what happened.

    Returns ``"written"`` for a new artefact and ``"verify_and_reuse"`` when the
    stored artefact already holds equivalent content (after canonicalisation).
    """
    path = Path(path)
    if not path.exists():
        write_json_atomic(path, obj)
        return "written"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImmutabilityError("existing artefact %s is not valid JSON: %s" % (path, exc)) from exc
    if canonical_json(existing) != canonical_json(obj):
        raise ImmutabilityError("artefact %s already exists with different content; refusing to overwrite" % path)
    return "verify_and_reuse"


def load_immutable(path):
    """Load a write-once artefact, raising FileNotFoundError when absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no artefact stored at %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))
