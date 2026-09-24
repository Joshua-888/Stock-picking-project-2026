"""Deterministic content-addressed identifiers for V2 research artefacts.

Scheme: ``<kind>_<12 hex chars>``; the digest is the first 12 hex characters of
SHA-256 over the canonical JSON encoding of {"kind": kind, "payload": payload}.
Identical payloads yield identical identifiers; no randomness and no clock input
are involved.
"""

from __future__ import annotations

import hashlib
import json
import re

HEX_LENGTH = 12

KINDS = (
    "dataset",
    "target_set",
    "feature_set",
    "experiment",
    "model",
    "prediction",
    "validation",
    "audit",
    "promotion",
)

_ID_RE = re.compile(r"^(?P<kind>[a-z_]+)_(?P<digest>[0-9a-f]{12})$")


def canonical_json(obj):
    """Canonical JSON text used for every digest in the V2 research package."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _digest(kind, payload):
    blob = canonical_json({"kind": kind, "payload": payload})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_id(kind, payload):
    """Return the deterministic identifier of ``payload`` for an artefact kind."""
    if kind not in KINDS:
        raise ValueError("unknown research artefact kind %r; expected one of %s" % (kind, ", ".join(KINDS)))
    return "%s_%s" % (kind, _digest(kind, payload)[:HEX_LENGTH])


def parse_id(identifier):
    """Split ``<kind>_<hex>`` into ``(kind, digest)``; raise ValueError if malformed."""
    if not isinstance(identifier, str):
        raise ValueError("malformed research identifier: %r" % (identifier,))
    match = _ID_RE.match(identifier)
    if not match:
        raise ValueError("malformed research identifier: %r" % (identifier,))
    kind = match.group("kind")
    if kind not in KINDS:
        raise ValueError("unknown research artefact kind in identifier: %r" % (identifier,))
    return kind, match.group("digest")


def is_valid_id(identifier, kind=None):
    """True when ``identifier`` is well formed and, when given, of ``kind``."""
    try:
        parsed_kind, _ignored = parse_id(identifier)
    except ValueError:
        return False
    return kind is None or parsed_kind == kind


def dataset_id(payload):
    return make_id("dataset", payload)


def target_set_id(payload):
    return make_id("target_set", payload)


def feature_set_id(payload):
    return make_id("feature_set", payload)


def experiment_id(payload):
    return make_id("experiment", payload)


def model_id(payload):
    return make_id("model", payload)


def prediction_id(payload):
    return make_id("prediction", payload)


def validation_id(payload):
    return make_id("validation", payload)


def audit_id(payload):
    return make_id("audit", payload)


def promotion_id(payload):
    return make_id("promotion", payload)
