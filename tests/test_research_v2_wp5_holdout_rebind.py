"""WP5 locked-holdout rebinding tests.

LIVE-INDEPENDENT: no network, no research data filesystem. Operates on tracked
provenance artefacts (``provenance/holdout/``) only.

They freeze the invariants of the corrective rebinding that binds the locked
holdout to the CORRECTED upstream artefacts:

* the canonical holdout resolves to the corrected upstream ids;
* boundaries and embargo are UNCHANGED vs the preserved original record;
* the original record is preserved and explicitly marked superseded;
* resolution of the canonical holdout is deterministic (by id, not filename).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.research.holdout import (
    canonical_holdout_id,
    embargo_cutoff,
    locked_holdout,
    resolve_holdout_id,
)
from src.research.ids import holdout_id

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = REPO_ROOT / "provenance" / "holdout"

ORIGINAL_HOLDOUT_ID = "holdout_e1a63def9749"
CORRECTED_HOLDOUT_ID = "holdout_7ce54e933e16"
OLD_UPSTREAM = {
    "dataset_id": "dataset_dbaa77445b38",
    "target_id": "target_set_888f68d1cfd0",
    "feature_set_id": "feature_set_56361533cc1b",
}
NEW_UPSTREAM = {
    "dataset_id": "dataset_35a278e17c13",
    "target_id": "target_set_d2bb16610bce",
    "feature_set_id": "feature_set_4f7b43726310",
}


def _load(name):
    return json.loads((PROVENANCE_DIR / name).read_text(encoding="utf-8"))


# ── 1. canonical holdout binds the CORRECTED upstream ids ────────────────────

def test_corrected_holdout_resolves_to_new_upstream_ids():
    holdout = locked_holdout()
    assert holdout.holdout_id == CORRECTED_HOLDOUT_ID
    assert holdout.record["dataset_id"] == NEW_UPSTREAM["dataset_id"]
    assert holdout.record["target_id"] == NEW_UPSTREAM["target_id"]
    assert holdout.record["feature_set_id"] == NEW_UPSTREAM["feature_set_id"]
    # Content-addressed: id reproduces from the record's own fields.
    recomputed = holdout_id({k: v for k, v in holdout.record.items() if k != "holdout_id"})
    assert recomputed == CORRECTED_HOLDOUT_ID


# ── 2. boundaries/embargo unchanged vs the preserved original ────────────────

def test_boundaries_and_embargo_unchanged_vs_original():
    original = _load("%s.json" % ORIGINAL_HOLDOUT_ID)
    corrected = _load("%s.json" % CORRECTED_HOLDOUT_ID)
    for field in ("holdout_start", "holdout_end", "embargo_months", "horizon_months"):
        assert corrected[field] == original[field], field
    assert corrected["selection_rationale"] == original["selection_rationale"]
    assert str(embargo_cutoff(corrected["holdout_start"], corrected["embargo_months"])) == \
        str(embargo_cutoff(original["holdout_start"], original["embargo_months"]))
    assert str(embargo_cutoff(corrected["holdout_start"]).date()) == "2021-01-01"
    # The only binding difference is the upstream ids.
    assert corrected["dataset_id"] != original["dataset_id"]
    assert corrected["git_commit"] == original["git_commit"]
    assert corrected["frozen_at"] == original["frozen_at"]


# ── 3. original record preserved + marked superseded ─────────────────────────

def test_original_record_preserved_and_marked_superseded():
    original = _load("%s.json" % ORIGINAL_HOLDOUT_ID)
    assert original["holdout_id"] == ORIGINAL_HOLDOUT_ID
    assert original["dataset_id"] == OLD_UPSTREAM["dataset_id"]
    assert original["target_id"] == OLD_UPSTREAM["target_id"]
    assert original["feature_set_id"] == OLD_UPSTREAM["feature_set_id"]

    index = _load("index.json")
    entry = index["entries"][ORIGINAL_HOLDOUT_ID]
    assert entry["status"] == "superseded"
    assert entry["superseded_by"] == CORRECTED_HOLDOUT_ID

    sup = _load("supersession_%s.json" % ORIGINAL_HOLDOUT_ID)
    assert sup["old_holdout_id"] == ORIGINAL_HOLDOUT_ID
    assert sup["new_holdout_id"] == CORRECTED_HOLDOUT_ID
    assert sup["old_upstream"] == OLD_UPSTREAM
    assert sup["new_upstream"] == NEW_UPSTREAM
    assert sup["old_record_preserved"] is True
    assert sup["boundary_equality_evidence"]["identical"] is True


# ── 4. deterministic resolution (by id, not filename ordering) ───────────────

def test_resolution_is_deterministic_and_independent_of_filename_order(tmp_path):
    import shutil

    assert canonical_holdout_id() == CORRECTED_HOLDOUT_ID
    # A stale reference (as frozen in the corrective WP5 experiment) maps to the
    # canonical successor deterministically.
    assert resolve_holdout_id(ORIGINAL_HOLDOUT_ID) == CORRECTED_HOLDOUT_ID
    assert resolve_holdout_id(CORRECTED_HOLDOUT_ID) == CORRECTED_HOLDOUT_ID

    copied = tmp_path / "holdout"
    shutil.copytree(PROVENANCE_DIR, copied)
    assert canonical_holdout_id(provenance_dir=copied) == CORRECTED_HOLDOUT_ID
    assert resolve_holdout_id(ORIGINAL_HOLDOUT_ID, provenance_dir=copied) == CORRECTED_HOLDOUT_ID
    assert locked_holdout(provenance_dir=copied).holdout_id == CORRECTED_HOLDOUT_ID
