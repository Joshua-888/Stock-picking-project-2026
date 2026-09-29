"""WP5 holdout rebinding: rebind the locked holdout to the corrected upstream ids.

Why this exists
---------------
The locked holdout (``holdout_e1a63def9749``) was frozen against the ORIGINAL
WP2C panel / WP3 targets / WP3 feature set
(``dataset_dbaa77445b38`` / ``target_set_888f68d1cfd0`` /
``feature_set_56361533cc1b``). The corrective WP5 discovery re-ran on the
CORRECTED upstream artefacts (``dataset_35a278e17c13`` /
``target_set_d2bb16610bce`` / ``feature_set_4f7b43726310``), recorded in
experiment ``experiment_f985287c1315``. The holdout *boundaries* were unchanged
by that correction (holdout 2022-01-01..2025-08-31, 12-month embargo, cutoff
2021-01-01); only the upstream-id binding was stale.

Write-once immutability correctly refuses to mutate the frozen holdout, so this
script does NOT touch or delete the original record (preserved as historical
evidence). Instead it writes a NEW content-addressed holdout bound to the
CORRECTED upstream ids, plus a supersession record linking old -> new, plus a
deterministic index that resolves the corrected-bound holdout as canonical.

Determinism / idempotence
-------------------------
* the new holdout id is content-addressed (WP1 ``holdout`` id over the payload);
* ``frozen_at`` and ``git_commit`` are read from the preserved original record,
  so the ONLY difference from the original binding is the upstream ids;
* re-running verifies and reuses identical artefacts (``save_immutable``).

Run: /opt/venv/bin/python scripts/research_v2/wp5_rebind_holdout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.fingerprints import fingerprint_obj
from src.research.holdout import (
    HOLDOUT_SCHEMA_VERSION,
    PROHIBITION_RULE,
    build_holdout,
    embargo_cutoff,
    holdout_payload,
)
from src.research.ids import holdout_id
from src.research.immutability import save_immutable, write_json_atomic

PROVENANCE_DIR = ROOT / "provenance" / "holdout"

# Preserved original binding (historical evidence; never mutated).
ORIGINAL_HOLDOUT_ID = "holdout_e1a63def9749"
OLD_UPSTREAM = {
    "dataset_id": "dataset_dbaa77445b38",
    "target_id": "target_set_888f68d1cfd0",
    "feature_set_id": "feature_set_56361533cc1b",
}
# Corrected WP5 upstream binding (experiment_f985287c1315).
NEW_UPSTREAM = {
    "dataset_id": "dataset_35a278e17c13",
    "target_id": "target_set_d2bb16610bce",
    "feature_set_id": "feature_set_4f7b43726310",
}
SUPERSESSION_SCHEMA_VERSION = "wp5_holdout_rebinding_supersession_v1"
REASON = (
    "SUPERSEDED_BY_DATA_CORRECTION: the WP2C panel and WP3 targets were corrected "
    "(dataset_35a278e17c13 / target_set_d2bb16610bce / feature_set_4f7b43726310, "
    "recorded in experiment_f985287c1315); the locked-holdout upstream-id binding was "
    "stale. Holdout boundaries and embargo are unchanged; only the upstream ids are "
    "rebound. No performance signal was examined."
)


def _load_original():
    path = PROVENANCE_DIR / ("%s.json" % ORIGINAL_HOLDOUT_ID)
    if not path.is_file():
        raise SystemExit("original locked-holdout record is missing: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify existing rebinding without rewriting")
    args = parser.parse_args(argv)

    original = _load_original()
    if original.get("holdout_id") != ORIGINAL_HOLDOUT_ID:
        raise SystemExit("original record id mismatch; refusing to rebind")

    payload = holdout_payload(
        dataset_id=NEW_UPSTREAM["dataset_id"],
        target_id=NEW_UPSTREAM["target_id"],
        feature_set_id=NEW_UPSTREAM["feature_set_id"],
        holdout_start=original["holdout_start"],
        holdout_end=original["holdout_end"],
        embargo_months=original["embargo_months"],
        selection_rationale=original["selection_rationale"],
        git_commit=original["git_commit"],
        frozen_at=original["frozen_at"],
        horizon_months=original["horizon_months"],
    )
    record = build_holdout(payload)
    new_id = record["holdout_id"]
    if new_id == ORIGINAL_HOLDOUT_ID:
        raise SystemExit("rebinding produced the original id; upstream ids unchanged?")

    # Boundary/embargo equality evidence (must be identical to the original).
    boundary_fields = ("holdout_start", "holdout_end", "embargo_months", "horizon_months")
    boundary_equality = {field: original[field] == record[field] for field in boundary_fields}
    cutoffs_equal = str(embargo_cutoff(original["holdout_start"], original["embargo_months"])) == \
        str(embargo_cutoff(record["holdout_start"], record["embargo_months"]))
    if not (all(boundary_equality.values()) and cutoffs_equal):
        raise SystemExit("rebinding changed holdout boundaries/embargo; refusing to proceed")

    supersession = {
        "supersession_schema_version": SUPERSESSION_SCHEMA_VERSION,
        "old_holdout_id": ORIGINAL_HOLDOUT_ID,
        "new_holdout_id": new_id,
        "old_upstream": dict(OLD_UPSTREAM),
        "new_upstream": dict(NEW_UPSTREAM),
        "boundary_equality_evidence": {
            "holdout_start": record["holdout_start"],
            "holdout_end": record["holdout_end"],
            "embargo_months": record["embargo_months"],
            "horizon_months": record["horizon_months"],
            "embargo_cutoff": str(embargo_cutoff(record["holdout_start"], record["embargo_months"])),
            "identical": True,
        },
        "reason": REASON,
        "no_performance_signal": True,
        "old_record_preserved": True,
        "superseded_records": ["provenance/holdout/%s.json" % ORIGINAL_HOLDOUT_ID],
    }

    index = {
        "holdout_schema_version": HOLDOUT_SCHEMA_VERSION,
        "definition_fingerprint": fingerprint_obj(payload),
        "canonical_holdout_id": new_id,
        # Legacy single-entry key retained for backward-compatible readers; it
        # points at the CANONICAL (corrected) holdout.
        "holdout": {"holdout_id": new_id, "artifact_file": "%s.json" % new_id},
        "entries": {
            new_id: {
                "holdout_id": new_id,
                "artifact_file": "%s.json" % new_id,
                "status": "canonical",
                "dataset_id": NEW_UPSTREAM["dataset_id"],
                "target_id": NEW_UPSTREAM["target_id"],
                "feature_set_id": NEW_UPSTREAM["feature_set_id"],
            },
            ORIGINAL_HOLDOUT_ID: {
                "holdout_id": ORIGINAL_HOLDOUT_ID,
                "artifact_file": "%s.json" % ORIGINAL_HOLDOUT_ID,
                "status": "superseded",
                "superseded_by": new_id,
                "dataset_id": OLD_UPSTREAM["dataset_id"],
                "target_id": OLD_UPSTREAM["target_id"],
                "feature_set_id": OLD_UPSTREAM["feature_set_id"],
                "reason": REASON,
            },
        },
        "supersession": {
            "old_holdout_id": ORIGINAL_HOLDOUT_ID,
            "new_holdout_id": new_id,
            "supersession_record_file": "supersession_%s.json" % ORIGINAL_HOLDOUT_ID,
        },
        "prohibition_rule": PROHIBITION_RULE,
    }

    artifact_path = PROVENANCE_DIR / ("%s.json" % new_id)
    supersession_path = PROVENANCE_DIR / ("supersession_%s.json" % ORIGINAL_HOLDOUT_ID)

    # Independent self-check: recompute the id from the record's own fields.
    recomputed = holdout_id({k: v for k, v in record.items() if k != "holdout_id"})
    assert recomputed == new_id, "rebound holdout id is not content-addressed"

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = {
        "rebound": save_immutable(artifact_path, record),
        "supersession": save_immutable(supersession_path, supersession),
    }
    if args.check:
        print("WP5_REBIND_HOLDOUT_CHECK old=%s new=%s outcomes=%s" % (ORIGINAL_HOLDOUT_ID, new_id, outcomes))
        return

    write_json_atomic(PROVENANCE_DIR / "index.json", index)
    print("WP5_REBIND_HOLDOUT_DONE old=%s new=%s" % (ORIGINAL_HOLDOUT_ID, new_id))
    print("boundaries=%s cutoff_equal=%s" % (boundary_equality, cutoffs_equal))
    print("rebound=%s supersession=%s index=%s" % (
        artifact_path.relative_to(ROOT), supersession_path.relative_to(ROOT),
        (PROVENANCE_DIR / "index.json").relative_to(ROOT)))


if __name__ == "__main__":
    main()
