"""WP5 certification artefact tests.

LIVE-INDEPENDENT: no network, no research data filesystem. Operates on tracked
provenance artefacts only, and asserts the FINAL WP5 certification faithfully
transcribes the Supervisor-authorized WP5_COMPLETE decision without mutating any
immutable scientific artefact.

Frozen invariants:

* the certification exists with schema ``wp5_certification_v1`` and status
  ``WP5_COMPLETE``;
* it binds the CORRECTED upstream ids (dataset/target/feature set);
* its canonical holdout id equals the holdout index's ``canonical_holdout_id``;
* the holdout-binding correction agrees on that same canonical id while still
  recording the pre-rebind id;
* the immutable WP5 experiment provenance still literally contains the recorded
  (superseded) holdout id, proving the correction was non-destructive;
* the recorded boundary/embargo equality is TRUE;
* both artefacts have deterministic byte content.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.research.holdout import canonical_holdout_id, embargo_cutoff, locked_holdout
from src.research.ids import canonical_json

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFICATIONS_DIR = REPO_ROOT / "provenance" / "certifications"
HOLDOUT_DIR = REPO_ROOT / "provenance" / "holdout"
WP5_DIR = REPO_ROOT / "provenance" / "wp5" / "experiment_f985287c1315"

EXPERIMENT_ID = "experiment_f985287c1315"
CERTIFICATION_PATH = CERTIFICATIONS_DIR / "wp5_experiment_f985287c1315.json"
CORRECTION_PATH = WP5_DIR / "holdout_binding_correction.json"
EXPERIMENT_PROVENANCE_PATH = WP5_DIR / "experiment_f985287c1315.json"
HOLDOUT_INDEX_PATH = HOLDOUT_DIR / "index.json"

CORRECTED_UPSTREAM = {
    "dataset_id": "dataset_35a278e17c13",
    "target_id": "target_set_d2bb16610bce",
    "feature_set_id": "feature_set_4f7b43726310",
}
RECORDED_HOLDOUT_ID = "holdout_e1a63def9749"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ── 1. certification exists with the expected schema/status ──────────────────

def test_certification_exists_with_schema_and_status():
    assert CERTIFICATION_PATH.is_file(), "WP5 certification artefact is missing"
    cert = _load(CERTIFICATION_PATH)
    assert cert["certification_schema"] == "wp5_certification_v1"
    assert cert["status"] == "WP5_COMPLETE"
    assert cert["experiment_id"] == EXPERIMENT_ID


# ── 2. certification binds the CORRECTED upstream ids ────────────────────────

def test_certification_binds_corrected_upstream_ids():
    cert = _load(CERTIFICATION_PATH)
    assert cert["dataset_id"] == CORRECTED_UPSTREAM["dataset_id"]
    assert cert["target_id"] == CORRECTED_UPSTREAM["target_id"]
    assert cert["feature_set_id"] == CORRECTED_UPSTREAM["feature_set_id"]
    # The certified upstream trio is exactly the canonical holdout's binding.
    holdout = locked_holdout()
    assert holdout.dataset_id == CORRECTED_UPSTREAM["dataset_id"]
    assert holdout.target_id == CORRECTED_UPSTREAM["target_id"]
    assert holdout.feature_set_id == CORRECTED_UPSTREAM["feature_set_id"]


# ── 3. canonical holdout id agrees with the holdout index ────────────────────

def test_certification_canonical_holdout_matches_index():
    index = _load(HOLDOUT_INDEX_PATH)
    cert = _load(CERTIFICATION_PATH)
    assert cert["canonical_holdout_id"] == index["canonical_holdout_id"]
    # Resolved by id through the API, never by filename ordering.
    assert canonical_holdout_id() == index["canonical_holdout_id"]


# ── 4. correction agrees on the canonical id and records the old id ──────────

def test_holdout_binding_correction_is_non_destructive():
    correction = _load(CORRECTION_PATH)
    index = _load(HOLDOUT_INDEX_PATH)
    assert correction["correction_schema"] == "wp5_holdout_binding_correction_v1"
    assert correction["experiment_id"] == EXPERIMENT_ID
    assert correction["canonical_holdout_id"] == index["canonical_holdout_id"]
    assert correction["recorded_holdout_id"] == RECORDED_HOLDOUT_ID
    # The correction is a NEW artefact: it never aliases the immutable record.
    assert correction["mutates_immutable_artifact"] is False
    assert CORRECTION_PATH.resolve() != EXPERIMENT_PROVENANCE_PATH.resolve()
    assert correction["immutable_artifact"].endswith(
        "provenance/wp5/experiment_f985287c1315/experiment_f985287c1315.json"
    )


def test_immutable_experiment_provenance_still_contains_recorded_holdout_id():
    # Non-destruction: the frozen provenance is byte-unchanged in the sense that
    # it still literally carries the pre-rebind holdout id it was frozen with.
    raw = EXPERIMENT_PROVENANCE_PATH.read_text(encoding="utf-8")
    assert RECORDED_HOLDOUT_ID in raw
    provenance = json.loads(raw)
    assert provenance["holdout_id"] == RECORDED_HOLDOUT_ID
    assert provenance["holdout_id"] != _load(CERTIFICATION_PATH)["canonical_holdout_id"]


# ── 5. recorded boundary/embargo equality is TRUE ────────────────────────────

def test_holdout_boundary_equality_is_recorded_and_true():
    correction = _load(CORRECTION_PATH)
    boundary = correction["boundary_equality"]
    assert boundary["identical"] is True
    # The claim is checkable against the preserved records themselves.
    old_record = _load(HOLDOUT_DIR / (RECORDED_HOLDOUT_ID + ".json"))
    new_record = _load(HOLDOUT_DIR / (correction["canonical_holdout_id"] + ".json"))
    for key in ("holdout_start", "holdout_end", "embargo_months"):
        assert old_record[key] == new_record[key]
        assert boundary[key] == new_record[key]
    # embargo_cutoff() returns a tz-aware UTC timestamp; compare on the date.
    assert str(embargo_cutoff(new_record["holdout_start"]).date()) == boundary["embargo_cutoff"]
    assert str(embargo_cutoff(new_record["holdout_start"]).date()) == "2021-01-01"


# ── 6. deterministic file content ────────────────────────────────────────────

def test_artefact_content_is_deterministic():
    # The correction follows the compact canonical-JSON record convention
    # (sorted keys, no whitespace, single trailing newline).
    correction_raw = CORRECTION_PATH.read_text(encoding="utf-8")
    assert correction_raw == canonical_json(_load(CORRECTION_PATH)) + "\n"

    # The certification follows the WP4 certification convention
    # (sorted keys, 2-space indent, no trailing newline).
    cert_raw = CERTIFICATION_PATH.read_text(encoding="utf-8")
    assert cert_raw == json.dumps(_load(CERTIFICATION_PATH), indent=2, sort_keys=True)
    assert not cert_raw.endswith("\n")
