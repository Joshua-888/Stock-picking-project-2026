"""WP6 provenance misbinding cleanup tests.

LIVE-INDEPENDENT: no test touches the network or the local research data
filesystem. Tests operate on tracked provenance artefacts under ``provenance/``
and on small synthetic fixtures under ``tmp_path``.

They freeze the invariants that repair the WP6 provenance binding defect:

* ``experiment_05ddc3721b4a`` is withdrawn/non-canonical and can never resolve
  as canonical;
* ``experiment_f7864f37998f`` resolves as the canonical WP6 experiment;
* the historical misbound records are preserved byte-unchanged;
* the canonical producing commit actually contains the WP6 producing code, and
  the misbound commit does not;
* resolution is keyed by experiment id, never by filename ordering.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.research import wp6_provenance_corrections as wpc

REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = REPO_ROOT / "provenance" / "wp6" / "corrections"
MISBOUND_DIR = REPO_ROOT / "provenance" / "wp6" / "experiment_05ddc3721b4a"
CANONICAL_DIR = REPO_ROOT / "provenance" / "wp6" / "experiment_f7864f37998f"

OLD_EXPERIMENT_ID = "experiment_05ddc3721b4a"
NEW_EXPERIMENT_ID = "experiment_f7864f37998f"
CANONICAL_COMMIT = "89aadecf304d789b00daf13983641847b4a7d3bd"
MISBOUND_COMMIT = "8751c164b8c504628a2b9122e41397cf41f71276"

WP6_PRODUCER_FILES = (
    "src/research/modeling/runner.py",
    "scripts/research_v2/wp6_model_research.py",
)

# Frozen SHA-256 of the historical misbound records, captured at cleanup time.
# These files are immutable audit evidence and must never be edited in place.
MISBOUND_SHA256 = {
    "binding.json": "1fd3c2cddca6f45af2a3f1bd1c59884d05fb4c4ff9d40715f598bf4b8aac5254",
    "contract.json": "3c4ce8febb212df61727beb7feaa5ac1ba8d19dfcb154fec3cfb834ea2658bf1",
    "index_entry.json": "5eca2759bf68913ffa0caad00f78e01a3948609f46a28b75b19d0fde38d097e4",
    "registration.json": "761c159730020134b4deaa04c14ed9ab301b7f16a7d2b67db30a8e69f3c02ff6",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args):
    result = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_has_path(commit, path):
    result = subprocess.run(
        ["git", "cat-file", "-e", "%s:%s" % (commit, path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


# ── 1. the withdrawn misbound experiment is non-canonical ────────────────────

def test_misbound_experiment_is_non_canonical():
    resolved = wpc.resolve_wp6_experiment(OLD_EXPERIMENT_ID)
    assert resolved["canonical"] is False
    assert resolved["canonical_experiment_id"] == NEW_EXPERIMENT_ID
    assert resolved["experiment_id"] == OLD_EXPERIMENT_ID
    assert resolved["status"] == wpc.WITHDRAWN_STATUS
    assert resolved["misbound_git_commit"] == MISBOUND_COMMIT
    assert resolved["canonical_record"] is None
    with pytest.raises(wpc.Wp6ProvenanceResolutionError):
        wpc.assert_canonical(OLD_EXPERIMENT_ID)
    assert wpc.is_non_canonical(OLD_EXPERIMENT_ID) is True
    assert OLD_EXPERIMENT_ID != wpc.canonical_experiment_id()


# ── 2. the canonical experiment resolves as canonical ────────────────────────

def test_canonical_experiment_resolves():
    assert wpc.canonical_experiment_id() == NEW_EXPERIMENT_ID
    resolved = wpc.resolve_wp6_experiment(NEW_EXPERIMENT_ID)
    assert resolved["canonical"] is True
    assert resolved["experiment_id"] == NEW_EXPERIMENT_ID
    assert resolved["canonical_experiment_id"] == NEW_EXPERIMENT_ID
    assert resolved["status"] == wpc.CANONICAL_STATUS
    record = resolved["canonical_record"]
    assert record is not None
    assert record["canonical_experiment_id"] == NEW_EXPERIMENT_ID
    assert record["producing_code_commit"] == CANONICAL_COMMIT
    assert record["git_commit"] == CANONICAL_COMMIT
    assert record["dataset_id"] == "dataset_35a278e17c13"
    assert record["target_id"] == "target_set_d2bb16610bce"
    assert record["feature_set_id"] == "feature_set_4f7b43726310"
    assert resolved["canonical"].__class__ is bool
    assert wpc.canonical_wp6_experiment()["experiment_id"] == NEW_EXPERIMENT_ID
    # the withdrawal is visible from the canonical view
    assert OLD_EXPERIMENT_ID in resolved["withdrawn_experiment_ids"]
    assert wpc.assert_canonical(NEW_EXPERIMENT_ID)["canonical"] is True


# ── 3. historical misbound files remain byte-unchanged ───────────────────────

def test_misbound_records_are_byte_unchanged():
    for name, expected in MISBOUND_SHA256.items():
        path = MISBOUND_DIR / name
        assert path.is_file(), "historical misbound record is missing: %s" % path
        assert _sha256(path) == expected, "historical misbound record was modified: %s" % path

    # The record still carries the original misbound commit (never rewritten).
    binding = json.loads((MISBOUND_DIR / "binding.json").read_text(encoding="utf-8"))
    assert binding["git_commit"] == MISBOUND_COMMIT
    entry = json.loads((MISBOUND_DIR / "index_entry.json").read_text(encoding="utf-8"))
    assert entry["experiment_id"] == OLD_EXPERIMENT_ID
    assert entry["git_commit"] == MISBOUND_COMMIT

    # The withdrawal record points at this preserved file.
    withdrawal = wpc.supersession_record(OLD_EXPERIMENT_ID)
    assert withdrawal["superseded_records"] == [
        "provenance/wp6/experiment_05ddc3721b4a/binding.json"
    ]
    assert withdrawal["misbound_git_commit"] == MISBOUND_COMMIT
    assert withdrawal["canonical_git_commit"] == CANONICAL_COMMIT
    assert withdrawal["reason"] == "recorded commit does not contain producing WP6 code"

    # The canonical directory is also intact.
    assert (CANONICAL_DIR / "binding.json").is_file()
    canonical_binding = json.loads((CANONICAL_DIR / "binding.json").read_text(encoding="utf-8"))
    assert canonical_binding["git_commit"] == CANONICAL_COMMIT


# ── 4. the canonical producing commit really contains the WP6 code ───────────

def test_canonical_commit_contains_wp6_producing_code():
    if _git("cat-file", "-t", CANONICAL_COMMIT) != "commit":
        pytest.skip("git history unavailable")
    for path in WP6_PRODUCER_FILES:
        assert _git_has_path(CANONICAL_COMMIT, path), (
            "canonical commit %s is missing producer file %s" % (CANONICAL_COMMIT, path)
        )
    # The misbound commit must NOT contain the WP6 producing code.
    if _git("cat-file", "-t", MISBOUND_COMMIT) == "commit":
        for path in WP6_PRODUCER_FILES:
            assert not _git_has_path(MISBOUND_COMMIT, path), (
                "misbound commit %s unexpectedly contains producer file %s"
                % (MISBOUND_COMMIT, path)
            )


# ── 5. resolution is keyed by id, not by filename ordering ───────────────────

def _write_fixture(root, canonical_id="experiment_aaaaaaaaaaaa"):
    root.mkdir(parents=True, exist_ok=True)
    # deliberately misleading file names: the withdrawn record sorts FIRST.
    record = {
        "schema_version": wpc.WP6_CORRECTIONS_SCHEMA_VERSION,
        "experiment_id": canonical_id,
        "canonical_experiment_id": canonical_id,
        "producing_code_commit": "a" * 40,
        "canonical_provenance_ref": "provenance/wp6/experiment_%s/" % canonical_id,
    }
    (root / "zzz_canonical.json").write_text(json.dumps(record), encoding="utf-8")
    supersession = {
        "old_experiment_id": "experiment_bbbbbbbbbbbb",
        "new_experiment_id": canonical_id,
        "status": wpc.WITHDRAWN_STATUS,
        "misbound_git_commit": "b" * 40,
        "canonical_git_commit": "a" * 40,
    }
    (root / "aaa_supersession.json").write_text(json.dumps(supersession), encoding="utf-8")
    index = {
        "schema_version": wpc.WP6_CORRECTIONS_SCHEMA_VERSION,
        "canonical_experiment_id": canonical_id,
        "canonical_provenance_file": "zzz_canonical.json",
        "entries": {
            "experiment_bbbbbbbbbbbb": {
                "status": wpc.NON_CANONICAL_STATUS,
                "canonical_experiment_id": canonical_id,
                "supersession_file": "aaa_supersession.json",
                "canonical_provenance_ref": "provenance/wp6/experiment_%s/" % canonical_id,
            }
        },
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_resolution_independent_of_filename_ordering(tmp_path):
    root = _write_fixture(tmp_path / "corrections")
    # The withdrawn file sorts before the canonical file; resolution must not care.
    resolved = wpc.resolve_wp6_experiment("experiment_bbbbbbbbbbbb", root=root)
    assert resolved["canonical"] is False
    assert resolved["canonical_experiment_id"] == "experiment_aaaaaaaaaaaa"
    canonical = wpc.canonical_wp6_experiment(root=root)
    assert canonical["experiment_id"] == "experiment_aaaaaaaaaaaa"
    assert canonical["canonical"] is True
    assert wpc.canonical_provenance_path(root=root).name == "zzz_canonical.json"


def test_resolver_unknown_experiment_raises(tmp_path):
    root = _write_fixture(tmp_path / "corrections")
    with pytest.raises(wpc.Wp6ProvenanceResolutionError):
        wpc.resolve_wp6_experiment("experiment_cccccccccccc", root=root)


def test_resolver_rejects_index_record_mismatch(tmp_path):
    root = _write_fixture(tmp_path / "corrections")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    index["canonical_experiment_id"] = "experiment_dddddddddddd"
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(wpc.Wp6ProvenanceResolutionError):
        wpc.canonical_wp6_experiment(root=root)
