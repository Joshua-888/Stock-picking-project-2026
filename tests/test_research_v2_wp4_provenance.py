"""WP4 provenance/certification tests.

LIVE-INDEPENDENT: no test touches the network, reads tokens or depends on the
local research data filesystem. Tests operate on the tracked provenance
artefacts (``provenance/``) and on small synthetic fixtures under ``tmp_path``.
They freeze the invariants that repair the WP4 provenance binding defect:

* the corrected producing-code commit must actually contain the producer files;
* the producing commit must be an ancestor of (or equal to) a valid repo state;
* the obsolete misbound record can never be selected as canonical;
* the correction is a NEW artefact, never an overwrite of the frozen record;
* the corrected provenance resolves deterministically to a content-addressed id;
* the experiment lineage is reproducible from the corrected binding.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.research.ids import canonical_json, make_id
from src.research.immutability import ImmutabilityError, save_immutable
from src.research import provenance_corrections as pc

REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = REPO_ROOT / "provenance" / "wp4" / "corrections"
EXPERIMENT_ID = "experiment_5ed52dcf2f44"
PRODUCING_COMMIT = "a1c0a92c072ba0f0232d363f53824984c7b58d9a"
MISBOUND_COMMIT = "437f68ce35e986d18b088a7f5f33288a1025dd96"
ORIGINAL_RECORD = REPO_ROOT / "provenance" / "wp4" / "experiment_5ed52dcf2f44.json"

REQUIRED_PRODUCER_FILES = (
    "scripts/research_v2/wp4_build_keyes.py",
    "src/research/keyes/__init__.py",
    "src/research/keyes/diagnostics.py",
    "src/research/keyes/signals.py",
    "src/research/keyes/variables.py",
)


def _git(*args):
    result = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# ── 1. producing commit contains the referenced producer files ───────────────

def test_producing_commit_contains_producer_files():
    listing = _git("ls-tree", "-r", "--name-only", PRODUCING_COMMIT)
    if listing is None:
        pytest.skip("git unavailable")
    present = set(listing.splitlines())
    missing = [name for name in REQUIRED_PRODUCER_FILES if name not in present]
    assert not missing, "producing commit %s is missing producer files: %s" % (PRODUCING_COMMIT, missing)
    # The misbound WP3 commit must NOT contain the WP4 engine.
    misbound = _git("ls-tree", "-r", "--name-only", MISBOUND_COMMIT)
    if misbound is not None:
        misbound_files = set(misbound.splitlines())
        assert not (set(REQUIRED_PRODUCER_FILES) & misbound_files), (
            "the misbound commit unexpectedly contains WP4 producer files"
        )


# ── 2. producing commit is an ancestor of / equal to a valid repo state ──────

def test_producing_commit_is_ancestor_of_certification_commit():
    ancestor = _git("merge-base", "--is-ancestor", PRODUCING_COMMIT, "HEAD")
    head = _git("rev-parse", "HEAD")
    if head is None:
        pytest.skip("git unavailable")
    equal = head == PRODUCING_COMMIT
    # merge-base --is-ancestor returns empty stdout and exit 0 on success; _git
    # returns None only when it is unavailable, not when the check fails.
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PRODUCING_COMMIT, "HEAD"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, "producing commit is not an ancestor of HEAD"
    assert equal or HEAD_IS_VALID_ANCESTOR(HEAD)


def HEAD_IS_VALID_ANCESTOR(head):
    """Pr-return that the current HEAD is a real, resolvable commit."""
    return _git("cat-file", "-t", head) == "commit"


# ── 3. superseded provenance cannot be selected as canonical ─────────────────

def test_superseded_provenance_cannot_be_canonical():
    for ref in ("provenance/wp4/experiment_5ed52dcf2f44.json", "provenance/wp4/latest_experiment.json"):
        assert pc.is_superseded_reference(ref) is True
        with pytest.raises(pc.ProvenanceResolutionError):
            pc.assert_not_superseded(ref)
    resolved = pc.resolve_experiment(EXPERIMENT_ID)
    canonical = resolved["canonical_provenance_ref"]
    assert pc.is_superseded_reference(canonical) is False
    for ref in resolved["superseded_records"]:
        assert ref != canonical


# ── 4. correction is a NEW artefact, never an overwrite ──────────────────────

def test_correction_uses_new_artifact_and_does_not_overwrite(tmp_path):
    # The original misbound record is preserved unchanged (still bound to 437f68c).
    original = json.loads(ORIGINAL_RECORD.read_text(encoding="utf-8"))
    assert original["git_commit"] == MISBOUND_COMMIT
    assert original["experiment_id"] == EXPERIMENT_ID

    # The corrected record lives under a different path and id, reusing the WP1
    # write-once mechanism: identical rewrite reuses; different content raises.
    resolved = pc.resolve_experiment(EXPERIMENT_ID)
    corrected = resolved["corrected_provenance"]
    assert corrected["producing_code_commit"] == PRODUCING_COMMIT
    assert corrected["corrected_provenance_id"] != EXPERIMENT_ID
    corrected_path = CORRECTIONS_DIR / ("%s.json" % corrected["corrected_provenance_id"])
    assert corrected_path.is_file()
    assert corrected_path.resolve() != ORIGINAL_RECORD.resolve()

    # The write-once guard still holds for the corrected artefact itself.
    assert save_immutable(corrected_path, corrected) == "verify_and_reuse"
    mutated = dict(corrected)
    mutated["producing_code_commit"] = MISBOUND_COMMIT
    with pytest.raises(ImmutabilityError):
        save_immutable(corrected_path, mutated)


# ── 5. corrected provenance resolves deterministically ───────────────────────

def test_corrected_provenance_resolves_deterministically():
    first = pc.resolve_experiment(EXPERIMENT_ID)
    second = pc.resolve_experiment(EXPERIMENT_ID)
    first_id = first["corrected_provenance"]["corrected_provenance_id"]
    second_id = second["corrected_provenance"]["corrected_provenance_id"]
    assert first_id != EXPERIMENT_ID
    assert first_id == second_id
    assert first["corrected_provenance"] == second["corrected_provenance"]

    # The corrected id is the content-addressed id of the corrected binding,
    # recomputed independently from the record's own fields (minus id/timestamp).
    binding = {
        key: value
        for key, value in first["corrected_provenance"].items()
        if key not in ("corrected_provenance_id", "correction_timestamp")
    }
    assert corrected_binding_id(binding) == first_id

    # Resolution is keyed by experiment id, never by filename ordering.
    assert pc.canonical_provenance_path(EXPERIMENT_ID).name == "%s.json" % first_id


def corrected_binding_id(binding):
    """Independent recomputation using the WP1 make_id('experiment', payload)."""
    return make_id("experiment", binding)


# ── 6. experiment lineage is reproducible ────────────────────────────────────

def test_experiment_lineage_reproducible():
    resolved = pc.resolve_experiment(EXPERIMENT_ID)
    corrected = resolved["corrected_provenance"]
    canonical = canonical_json(corrected)
    assert canonical_json(resolved["corrected_provenance"]) == canonical
    for field in ("dataset_id", "target_id", "feature_set_id"):
        assert corrected[field]
    assert corrected["upstream_ids"] == {
        "dataset_id": corrected["dataset_id"],
        "target_id": corrected["target_id"],
        "feature_set_id": corrected["feature_set_id"],
    }
    supersession = resolved["supersession"]
    assert supersession["experiment_id"] == EXPERIMENT_ID
    assert supersession["actual_producing_code_commit"] == PRODUCING_COMMIT
    assert supersession["original_bound_commit"] == MISBOUND_COMMIT
    assert supersession["corrected_provenance_ref"] == resolved["canonical_provenance_ref"]
    assert supersession["auditor_finding"] == "AUDIT_WP4_R1_2026-09-26"


# ── resolver robustness (synthetic fixture, cannot depend on committed repo) ──

def _write_fixture(root, experiment_id="experiment_000000000001"):
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "corrected_provenance_id": "experiment_ffffffffffff",
        "experiment_id": experiment_id,
        "producing_code_commit": "a" * 40,
    }
    (root / "experiment_ffffffffffff.json").write_text(json.dumps(record), encoding="utf-8")
    supersession = {
        "experiment_id": experiment_id,
        "actual_producing_code_commit": "a" * 40,
        "original_bound_commit": "b" * 40,
        "superseded_records": ["provenance/wp4/experiment_deadbeef.json"],
    }
    (root / ("supersession_%s.json" % experiment_id)).write_text(json.dumps(supersession), encoding="utf-8")
    index = {
        "corrections_schema_version": pc.CORRECTIONS_SCHEMA_VERSION,
        "entries": {
            experiment_id: {
                "experiment_id": experiment_id,
                "corrected_provenance_id": "experiment_ffffffffffff",
                "corrected_provenance_file": "experiment_ffffffffffff.json",
                "canonical_provenance_ref": "provenance/wp4/corrections/experiment_ffffffffffff.json",
                "supersession_record_file": "supersession_%s.json" % experiment_id,
                "producing_code_commit": "a" * 40,
                "superseded_records": ["provenance/wp4/experiment_deadbeef.json"],
            }
        },
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_resolver_rejects_mismatched_index(tmp_path):
    root = _write_fixture(tmp_path / "corrections")
    resolved = pc.resolve_experiment("experiment_000000000001", root=root)
    assert resolved["producing_code_commit"] == "a" * 40
    # Corrupt the index's recorded producing commit; resolution must fail loudly.
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    index["entries"]["experiment_000000000001"]["producing_code_commit"] = "c" * 40
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(pc.ProvenanceResolutionError):
        pc.resolve_experiment("experiment_000000000001", root=root)


def test_resolver_unknown_experiment_raises(tmp_path):
    root = _write_fixture(tmp_path / "corrections")
    with pytest.raises(pc.ProvenanceResolutionError):
        pc.resolve_experiment("experiment_000000000099", root=root)
