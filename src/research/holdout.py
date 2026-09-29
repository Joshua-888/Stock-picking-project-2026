"""WP5/WP6/WP7 locked-holdout definition and temporal access control.

This module FREEZES the final evaluation holdout for Free Research V2 *before*
any target-based modelling or feature selection begins. It is deliberately
separated from model code so that no discovery step can silently consume the
holdout rows.

Why it exists
-------------
The 12-month forward target overlaps heavily (~11.8x): an observation at
prediction instant ``T`` carries a label that only becomes public at
``target_known_at`` (``T`` + 12 months). Two consequences follow and both are
enforced here:

* a training row may only be used once its full future outcome is observable
  (``target_known_at <= T_model``); this is already provided by
  ``src.research.targets.observability``;
* a training row's label must not overlap the holdout window at all, so the
  last ``embargo_months`` (>= the label horizon) of training observations
  before the holdout start are PURGED (never trained on, never evaluated).

The holdout boundary itself is chosen on DATA-AVAILABILITY grounds only
(observable-label coverage and censoring), never on any performance signal.
Nothing here fits a model, scores a feature, or inspects a label value.

Guarantees
----------
* the holdout definition is content-addressed and deterministic;
* it is persisted write-once under ``provenance/holdout/`` and is
  live-independent (the accessor reads only tracked provenance);
* :func:`trainable_mask` excludes holdout rows AND embargo-band rows for any
  model date;
* :func:`assert_trainable` fails loudly if holdout or embargo-band rows are
  ever handed to a training call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data.availability import to_utc_timestamp
from .ids import holdout_id
from .targets.observability import trainable_mask as _observable_trainable_mask

HOLDOUT_SCHEMA_VERSION = "wp5_locked_holdout_v1"

# ── Frozen design constants (documented, never tuned to any result) ───────────
# Boundary chosen purely from observable-label availability:
#   * observable forward labels run 2007-01-31 .. 2025-08-31;
#   * 2025 is only ~63% observable and 2026 is 0% observable (censoring rises
#     sharply because the 12-month horizon runs past the research-window end);
#   * the most recent contiguous fully-usable segment therefore ends 2025-08-31
#     and the last ~4 years of it (2022-01-01 onward) is reserved as holdout.
DEFAULT_HOLDOUT_START = "2022-01-01"
DEFAULT_HOLDOUT_END = "2025-08-31"
DEFAULT_EMBARGO_MONTHS = 12

# Baselines the holdout is frozen against (certified upstream artefacts).
DEFAULT_DATASET_ID = "dataset_dbaa77445b38"
DEFAULT_TARGET_ID = "target_set_888f68d1cfd0"
DEFAULT_FEATURE_SET_ID = "feature_set_56361533cc1b"

PROHIBITION_RULE = (
    "NO discovery, feature selection, preprocessing fit, threshold selection, "
    "model selection, hyperparameter tuning or probability calibration may use "
    "locked-holdout rows. The holdout is evaluation-only and is inspected at most "
    "once; once evaluated, no model change may be made to improve that same "
    "holdout result."
)

_PROVENANCE_REL = Path("provenance") / "holdout"


class HoldoutError(RuntimeError):
    """Raised when the holdout definition is absent, malformed or tampered with."""


class HoldoutLeakageError(HoldoutError):
    """Raised when holdout or embargo rows are offered as training data."""


# ── Deterministic, content-addressed definition ──────────────────────────────

def holdout_payload(dataset_id, target_id, feature_set_id, holdout_start,
                    holdout_end, embargo_months, selection_rationale, git_commit,
                    frozen_at, horizon_months=DEFAULT_EMBARGO_MONTHS):
    """Canonical identity payload for a locked holdout (excludes its own id)."""
    return {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "dataset_id": str(dataset_id),
        "target_id": str(target_id),
        "feature_set_id": str(feature_set_id),
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
        "embargo_months": int(embargo_months),
        "horizon_months": int(horizon_months),
        "selection_rationale": str(selection_rationale),
        "git_commit": str(git_commit),
        "frozen_at": str(frozen_at),
        "prohibition_rule": PROHIBITION_RULE,
    }


def build_holdout(payload):
    """Return ``payload`` augmented with its deterministic ``holdout_id``."""
    record = dict(payload)
    record["holdout_id"] = holdout_id(payload)
    return record


def embargo_cutoff(holdout_start=None, embargo_months=DEFAULT_EMBARGO_MONTHS):
    """First instant that is no longer usable for training.

    Any training observation at or after this instant has a forward label that
    overlaps the holdout window, so it is purged: ``holdout_start`` minus the
    embargo (>= label horizon).
    """
    start = to_utc_timestamp(holdout_start or DEFAULT_HOLDOUT_START)
    cutoff = start - pd.DateOffset(months=int(embargo_months))
    return cutoff


# ── Frozen accessor ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LockedHoldout:
    """Immutable view of a frozen locked holdout."""

    record: dict

    @property
    def holdout_id(self):
        return self.record["holdout_id"]

    @property
    def dataset_id(self):
        return self.record["dataset_id"]

    @property
    def target_id(self):
        return self.record["target_id"]

    @property
    def feature_set_id(self):
        return self.record["feature_set_id"]

    @property
    def holdout_start(self):
        return to_utc_timestamp(self.record["holdout_start"])

    @property
    def holdout_end(self):
        return to_utc_timestamp(self.record["holdout_end"])

    @property
    def embargo_months(self):
        return int(self.record["embargo_months"])

    @property
    def embargo_cutoff(self):
        return embargo_cutoff(self.record["holdout_start"], self.record["embargo_months"])

    def to_dict(self):
        return dict(self.record)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def default_provenance_dir():
    return _repo_root() / _PROVENANCE_REL


def _read_index(provenance_dir):
    index_path = Path(provenance_dir) / "index.json"
    if not index_path.is_file():
        raise HoldoutError("no locked-holdout index at %s" % index_path)
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HoldoutError("locked-holdout index is not valid JSON: %s" % exc) from exc


def _resolve_canonical_entry(index):
    """Deterministically resolve the CANONICAL holdout entry from the index.

    Resolution is by id, never by filename ordering: ``canonical_holdout_id``
    takes precedence, and any entry flagged ``status == "superseded"`` is
    refused. This keeps the corrected-bound holdout canonical after a rebind
    while the original record stays on disk as historical evidence.
    """
    entries = index.get("entries")
    canonical_id = index.get("canonical_holdout_id")
    if canonical_id and isinstance(entries, dict) and canonical_id in entries:
        entry = dict(entries[canonical_id])
        entry["holdout_id"] = entry.get("holdout_id", canonical_id)
        entry.setdefault("artifact_file", "%s.json" % canonical_id)
    else:
        # Legacy index shape (single "holdout" entry).
        entry = dict(index.get("holdout") or {})
    if str(entry.get("status") or "").lower() == "superseded":
        raise HoldoutError(
            "resolved locked holdout %r is marked superseded and must not be used"
            % entry.get("holdout_id")
        )
    return entry


def locked_holdout(provenance_dir=None):
    """Load and verify the frozen locked holdout (live-independent, deterministic).

    Reads only the tracked ``provenance/holdout/`` artefacts, recomputes the
    content-addressed id from the record's own fields and refuses to return a
    record whose id does not match. No data filesystem and no clock are used.
    """
    provenance_dir = Path(provenance_dir) if provenance_dir is not None else default_provenance_dir()
    index = _read_index(provenance_dir)
    entry = _resolve_canonical_entry(index)
    record_id = entry.get("holdout_id")
    file_name = entry.get("artifact_file")
    if not record_id or not file_name:
        raise HoldoutError("locked-holdout index is missing holdout_id/artifact_file")
    artifact_path = Path(provenance_dir) / file_name
    if not artifact_path.is_file():
        raise HoldoutError("locked-holdout artifact missing: %s" % artifact_path)
    try:
        record = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HoldoutError("locked-holdout artifact is not valid JSON: %s" % exc) from exc
    if record.get("holdout_id") != record_id:
        raise HoldoutError("locked-holdout artifact id does not match the index")
    recomputed = holdout_id({key: value for key, value in record.items() if key != "holdout_id"})
    if recomputed != record_id:
        raise HoldoutError(
            "locked-holdout artifact is not content-addressed: index=%s recomputed=%s"
            % (record_id, recomputed)
        )
    return LockedHoldout(record=record)


def canonical_holdout_id(provenance_dir=None):
    """Deterministic id of the canonical locked holdout (no filename ordering).

    Prefers the index's ``canonical_holdout_id``; falls back to the legacy
    single ``holdout`` entry. Never returns a superseded binding.
    """
    provenance_dir = Path(provenance_dir) if provenance_dir is not None else default_provenance_dir()
    index = _read_index(provenance_dir)
    entry = _resolve_canonical_entry(index)
    record_id = entry.get("holdout_id")
    if not record_id:
        raise HoldoutError("locked-holdout index has no canonical holdout_id")
    return record_id


def resolve_holdout_id(holdout_id, provenance_dir=None):
    """Map any holdout id to the current canonical id via the rebinding chain.

    A frozen experiment may still literally record an id that has since been
    superseded (only its upstream-id binding was stale). This resolves that
    reference deterministically to the canonical successor. Unknown ids that
    are not registered in the index are rejected rather than silently accepted.
    """
    provenance_dir = Path(provenance_dir) if provenance_dir is not None else default_provenance_dir()
    index = _read_index(provenance_dir)
    canonical = canonical_holdout_id(provenance_dir)
    entries = index.get("entries")
    if not isinstance(entries, dict):
        # Legacy index shape: only the canonical holdout is registered.
        legacy = (index.get("holdout") or {}).get("holdout_id")
        if holdout_id != legacy:
            raise HoldoutError("unknown locked-holdout id %r" % (holdout_id,))
        return canonical
    if holdout_id not in entries:
        raise HoldoutError("unknown locked-holdout id %r" % (holdout_id,))
    seen = set()
    current = holdout_id
    while True:
        if current in seen:
            raise HoldoutError("cyclic locked-holdout supersession chain at %r" % (current,))
        seen.add(current)
        entry = entries.get(current)
        if entry is None:
            raise HoldoutError("locked-holdout supersession references unregistered id %r" % (current,))
        if str(entry.get("status") or "").lower() != "superseded":
            return current
        nxt = entry.get("superseded_by")
        if not nxt:
            raise HoldoutError("superseded locked holdout %r has no superseded_by" % (current,))
        current = nxt


# ── Temporal access control ──────────────────────────────────────────────────

def _feature_asof_series(target_frame, feature_asof_col):
    if feature_asof_col not in target_frame.columns:
        raise HoldoutError("target frame lacks %r" % feature_asof_col)
    parsed = [_utc(value) for value in target_frame[feature_asof_col].tolist()]
    return parsed


def _utc(value):
    return to_utc_timestamp(value)


def holdout_mask(target_frame, holdout=None, feature_asof_col="feature_asof"):
    """Rows whose prediction instant falls in the locked holdout window."""
    holdout = holdout or locked_holdout()
    start = holdout.holdout_start
    flags = []
    for stamp in _feature_asof_series(target_frame, feature_asof_col):
        flags.append(bool(stamp is not None and stamp >= start))
    return pd.Series(flags, index=target_frame.index)


def embargo_mask(target_frame, holdout=None, feature_asof_col="feature_asof"):
    """Rows purged because their label overlaps the holdout window.

    These are observations between the embargo cutoff and the holdout start:
    they are neither trainable nor holdout -- they are discarded entirely so
    that no training label extends into the holdout.
    """
    holdout = holdout or locked_holdout()
    cutoff = holdout.embargo_cutoff
    start = holdout.holdout_start
    flags = []
    for stamp in _feature_asof_series(target_frame, feature_asof_col):
        flags.append(bool(stamp is not None and cutoff <= stamp < start))
    return pd.Series(flags, index=target_frame.index)


def trainable_mask(target_frame, model_date, holdout=None, feature_asof_col="feature_asof"):
    """Rows a model at ``model_date`` may legally train on.

    A row qualifies only when ALL hold:

    * it is observable (``target_observable == True``);
    * its label is public (``target_known_at <= model_date``);
    * its prediction instant is BEFORE the embargo cutoff, so its forward label
      cannot reach into the locked holdout.

    Holdout rows and embargo-band rows are therefore always excluded.
    """
    holdout = holdout or locked_holdout()
    observable_ok = _observable_trainable_mask(target_frame, model_date)
    cutoff = holdout.embargo_cutoff
    before_cutoff = []
    for stamp in _feature_asof_series(target_frame, feature_asof_col):
        before_cutoff.append(bool(stamp is not None and stamp < cutoff))
    before_cutoff = pd.Series(before_cutoff, index=target_frame.index)
    return observable_ok & before_cutoff


def assert_trainable(target_frame, model_date, holdout=None, feature_asof_col="feature_asof"):
    """Fail loudly if ``target_frame`` contains any non-trainable row.

    Intended as a guard at the top of a training call: passing holdout rows or
    embargo-band rows raises :class:`HoldoutLeakageError` rather than silently
    proceeding. Returns ``True`` when every row is legal.
    """
    holdout = holdout or locked_holdout()
    in_holdout = holdout_mask(target_frame, holdout, feature_asof_col)
    if bool(in_holdout.any()):
        raise HoldoutLeakageError(
            "attempted to train on %d locked-holdout row(s); the holdout is "
            "evaluation-only" % int(in_holdout.sum())
        )
    in_embargo = embargo_mask(target_frame, holdout, feature_asof_col)
    if bool(in_embargo.any()):
        raise HoldoutLeakageError(
            "attempted to train on %d embargo-band row(s) whose labels overlap "
            "the holdout window; purge them" % int(in_embargo.sum())
        )
    legal = trainable_mask(target_frame, model_date, holdout, feature_asof_col)
    if len(legal) != len(target_frame) or not bool(legal.all()):
        raise HoldoutLeakageError(
            "training frame contains rows that are not trainable at %r" % (model_date,)
        )
    return True
