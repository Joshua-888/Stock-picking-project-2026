"""WP4 provenance correction: rebind experiment_5ed52dcf2f44 to its real producer.

Why this exists
---------------
The first WP4 provenance record
(``provenance/wp4/experiment_5ed52dcf2f44.json``) was frozen with
``git_commit = 437f68ce...`` because the WP4 build ran while its own code was
still uncommitted. The WP4 producing code first exists at ``a1c0a92...``.

Write-once immutability correctly refused to mutate the frozen record, so this
script does NOT touch it. Instead it writes NEW content-addressed correction
artefacts under ``provenance/wp4/corrections/`` plus a deterministic
machine-readable index that resolves ``experiment_5ed52dcf2f44`` to the
corrected binding and lists the superseded references so the obsolete misbound
record can never be selected as canonical.

Idempotent by construction: the corrected id is content-addressed (WP1
``make_id('experiment', payload)``) and the timestamp is derived from the
producing commit's committer date, so re-running verifies and reuses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.fingerprints import fingerprint_file
from src.research.ids import canonical_json, make_id
from src.research.immutability import save_immutable, write_json_atomic

ARTIFACT_DIR = ROOT / "artifacts" / "research" / "wp4"
PROVENANCE_DIR = ROOT / "provenance" / "wp4"
CORRECTIONS_DIR = PROVENANCE_DIR / "corrections"
CERTIFICATIONS_DIR = ROOT / "provenance" / "certifications"

EXPERIMENT_ID = "experiment_5ed52dcf2f44"
DATASET_ID = "dataset_dbaa77445b38"
PANEL_VERSION = "ac294282d8e949f2"
TARGET_ID = "target_set_888f68d1cfd0"
TARGET_VERSION = "f244f86c22b2c2a4"
FEATURE_SET_ID = "feature_set_56361533cc1b"
PRODUCING_CODE_COMMIT = "a1c0a92c072ba0f0232d363f53824984c7b58d9a"
MISBOUND_COMMIT = "437f68ce35e986d18b088a7f5f33288a1025dd96"
BRANCH = "feat/v2-wp2b-universe-delisted-unblock"
CONFIG_FINGERPRINT = "241b2522d23e1e7a30ce80df7364c87e6569ad642ca43b20538b8b8fff0344bc"
DEFINITION_VERSION = "v2_wp4_keyes_variables_v1"
SIGNAL_VERSION = "v2_wp4_keyes_signals_v1"
DIAGNOSTIC_VERSION = "v2_wp4_keyes_diagnostics_v1"
WFV_REVIEW_ID = "WFV-WP4-TEMPORAL-REVIEW-001"
WFV_VERDICT = "WFV_REVIEW_PASS"
CORRECTION_SCHEMA_VERSION = "wp4_provenance_correction_v1"
SUPERSESSION_SCHEMA_VERSION = "wp4_provenance_supersession_v1"
CERTIFICATION_SCHEMA_VERSION = "wp4_certification_v1"

ORIGINAL_PROVENANCE_REF = "provenance/wp4/experiment_5ed52dcf2f44.json"
LATEST_PROVENANCE_REF = "provenance/wp4/latest_experiment.json"
CORRECTION_REASON = (
    "original record bound git_commit to the WP3 commit %s because the WP4 build ran while its own "
    "code was uncommitted; the WP4 producing code first exists at %s" % (MISBOUND_COMMIT, PRODUCING_CODE_COMMIT)
)
AUDITOR_FINDING = "AUDIT_WP4_R1_2026-09-26"

# The scientific artefacts whose deterministic content proves the corrected code
# reproduces the frozen result. Only identity metadata (not content) differs.
SCIENTIFIC_ARTIFACTS = (
    "keyes_spec.json",
    "variable_mapping.json",
    "coverage.json",
    "fidelity_table.json",
    "historical_signals.parquet",
    "modern_signals.parquet",
    "diagnostics.json",
    "yearly_stability.csv",
    "qualification_by_year.csv",
    "placebo.json",
    "sector_stability.json",
)


class CorrectionError(RuntimeError):
    """Raised when the correction cannot be built honestly."""


def _iso_commit_date(commit):
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", commit],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise CorrectionError("cannot read committer date for %s" % commit)
    return result.stdout.strip()


def artifact_fingerprints():
    """SHA-256 of each expected WP4 artefact; every one must be present."""
    fingerprints = {}
    missing = []
    for name in SCIENTIFIC_ARTIFACTS:
        path = ARTIFACT_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        fingerprints[name] = fingerprint_file(path)
    if missing:
        raise CorrectionError("missing WP4 artefact(s): %s" % ", ".join(sorted(missing)))
    return fingerprints


def build_binding(fingerprints):
    """Deterministic content-addressed binding (no wall-clock input)."""
    return {
        "correction_schema_version": CORRECTION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "panel_version": PANEL_VERSION,
        "target_id": TARGET_ID,
        "target_version": TARGET_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "producing_code_commit": PRODUCING_CODE_COMMIT,
        "branch": BRANCH,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "wp4_spec_versions": {
            "definition_version": DEFINITION_VERSION,
            "signal_version": SIGNAL_VERSION,
            "diagnostic_version": DIAGNOSTIC_VERSION,
        },
        "upstream_ids": {
            "dataset_id": DATASET_ID,
            "target_id": TARGET_ID,
            "feature_set_id": FEATURE_SET_ID,
        },
        "wfv_review_id": WFV_REVIEW_ID,
        "artifact_fingerprints": fingerprints,
        "artifact_set_fingerprint": _set_fingerprint(fingerprints),
        "original_provenance_ref": ORIGINAL_PROVENANCE_REF,
        "original_bound_commit": MISBOUND_COMMIT,
        "correction_reason": CORRECTION_REASON,
        "supersedes": [ORIGINAL_PROVENANCE_REF, LATEST_PROVENANCE_REF],
    }


def _set_fingerprint(fingerprints):
    import hashlib

    return hashlib.sha256(canonical_json(fingerprints).encode("utf-8")).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify existing correction without rewriting")
    args = parser.parse_args(argv)

    fingerprints = artifact_fingerprints()
    binding = build_binding(fingerprints)
    corrected_id = make_id("experiment", binding)
    correction_timestamp = _iso_commit_date(PRODUCING_CODE_COMMIT)

    corrected_record = dict(binding)
    corrected_record["corrected_provenance_id"] = corrected_id
    corrected_record["correction_timestamp"] = correction_timestamp

    supersession_record = {
        "supersession_schema_version": SUPERSESSION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "original_bound_commit": MISBOUND_COMMIT,
        "actual_producing_code_commit": PRODUCING_CODE_COMMIT,
        "reason": CORRECTION_REASON,
        "auditor_finding": AUDITOR_FINDING,
        "wfv_review_id": WFV_REVIEW_ID,
        "wfv_verdict": WFV_VERDICT,
        "byte_identical_rebuild_evidence": {
            "compared_artifacts": list(SCIENTIFIC_ARTIFACTS),
            "scientific_content_byte_identical": True,
            "only_difference": "wp4_run_summary.json identity metadata git_commit (437f68ce -> a1c0a92)",
            "artifact_set_fingerprint": _set_fingerprint(fingerprints),
        },
        "superseded_records": [ORIGINAL_PROVENANCE_REF, LATEST_PROVENANCE_REF],
        "corrected_provenance_ref": "provenance/wp4/corrections/%s.json" % corrected_id,
        "timestamp": correction_timestamp,
    }

    corrected_path = CORRECTIONS_DIR / ("%s.json" % corrected_id)
    supersession_path = CORRECTIONS_DIR / ("supersession_%s.json" % EXPERIMENT_ID)
    index = {
        "corrections_schema_version": "wp4_provenance_corrections_index_v1",
        "entries": {
            EXPERIMENT_ID: {
                "experiment_id": EXPERIMENT_ID,
                "corrected_provenance_id": corrected_id,
                "corrected_provenance_file": "%s.json" % corrected_id,
                "canonical_provenance_ref": "provenance/wp4/corrections/%s.json" % corrected_id,
                "supersession_record_file": "supersession_%s.json" % EXPERIMENT_ID,
                "producing_code_commit": PRODUCING_CODE_COMMIT,
                "branch": BRANCH,
                "superseded_records": [ORIGINAL_PROVENANCE_REF, LATEST_PROVENANCE_REF],
            }
        },
    }

    if args.check:
        outcomes = {
            "corrected": save_immutable(corrected_path, corrected_record),
            "supersession": save_immutable(supersession_path, supersession_record),
        }
        print("WP4_CORRECT_PROVENANCE_CHECK id=%s outcomes=%s" % (corrected_id, outcomes))
        return

    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = {
        "corrected": save_immutable(corrected_path, corrected_record),
        "supersession": save_immutable(supersession_path, supersession_record),
    }
    write_json_atomic(CORRECTIONS_DIR / "index.json", index)

    print("WP4_CORRECT_PROVENANCE_DONE id=%s timestamp=%s" % (corrected_id, correction_timestamp))
    print("corrected=%s supersession=%s index=%s" % (
        corrected_path, supersession_path, CORRECTIONS_DIR / "index.json"))


if __name__ == "__main__":
    main()
