"""WP5 certification artefacts (final, non-destructive WP5 sign-off).

This script TRANSCRIBES a Supervisor-authorized WP5_COMPLETE decision into two
NEW tracked artefacts. It issues no verdict of its own and mutates no existing
scientific artefact.

Artefacts written
-----------------
1. ``provenance/wp5/experiment_f985287c1315/holdout_binding_correction.json``
   A non-destructive correction record: the WP5 corrective run executed BEFORE
   the locked-holdout rebind, so its frozen provenance records the then-canonical
   holdout id. The immutable experiment provenance is NEVER modified; this record
   documents the discrepancy, the identical boundaries/embargo and the canonical
   resolution rule.
2. ``provenance/certifications/wp5_experiment_f985287c1315.json``
   The WP5 certification, matching the WP4 certification convention
   (``json.dumps(obj, indent=2, sort_keys=True)``, no trailing newline).

Determinism
-----------
No RNG and no wall-clock read at import time. The certification timestamp is
``WP5_CERT_TIMESTAMP`` (or ``--certified-at``); the certification commit must be
created with the same ``GIT_COMMITTER_DATE``/``GIT_AUTHOR_DATE`` so that
``certified_at`` equals the certification commit date, exactly mirroring WP4.
Re-running verifies and refuses to overwrite differing content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.ids import canonical_json  # noqa: E402

CORRECTION_PATH = (
    ROOT / "provenance" / "wp5" / "experiment_f985287c1315" / "holdout_binding_correction.json"
)
CERTIFICATION_PATH = ROOT / "provenance" / "certifications" / "wp5_experiment_f985287c1315.json"

EXPERIMENT_ID = "experiment_f985287c1315"
IMMUTABLE_ARTIFACT = "provenance/wp5/experiment_f985287c1315/experiment_f985287c1315.json"
RECORDED_HOLDOUT_ID = "holdout_e1a63def9749"
CANONICAL_HOLDOUT_ID = "holdout_7ce54e933e16"
DATASET_ID = "dataset_35a278e17c13"
TARGET_ID = "target_set_d2bb16610bce"
FEATURE_SET_ID = "feature_set_4f7b43726310"
PANEL_VERSION = "553ac17bf5d4d63f"
TARGET_VERSION = "56d0f670bdf1b47c"
CORRECTED_WP4_EXPERIMENT_ID = "experiment_834a7e60f13c"
CORRECTED_FROM_EXPERIMENT_ID = "experiment_902a843c7ec6"
PRODUCING_CODE_COMMIT = "6bd31b760a5cdee25a0beaf0b708859d34cb4517"
BRANCH = "feat/v2-wp2b-universe-delisted-unblock"

CORRECTION_SCHEMA = "wp5_holdout_binding_correction_v1"
CERTIFICATION_SCHEMA = "wp5_certification_v1"


def _timestamp(explicit):
    if explicit:
        return explicit
    env = os.environ.get("WP5_CERT_TIMESTAMP")
    if env:
        return env
    raise SystemExit(
        "WP5_CERT_TIMESTAMP (or --certified-at) is required so certified_at is "
        "reproducible and equals the certification commit date"
    )


def build_correction(created_at):
    return {
        "correction_schema": CORRECTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "immutable_artifact": IMMUTABLE_ARTIFACT,
        "recorded_holdout_id": RECORDED_HOLDOUT_ID,
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
        "reason": (
            "The WP5 corrective run (producing commit 6bd31b7, certified 7148da8) executed BEFORE the "
            "holdout rebind (7ad2330); locked_holdout() then returned the then-canonical id bound to the "
            "pre-correction upstream ids (dataset_dbaa77445b38/target_set_888f68d1cfd0/"
            "feature_set_56361533cc1b)."
        ),
        "boundary_equality": {
            "holdout_start": "2022-01-01",
            "holdout_end": "2025-08-31",
            "embargo_months": 12,
            "embargo_cutoff": "2021-01-01",
            "identical": True,
        },
        "canonical_resolution": (
            "Resolve via provenance/holdout/index.json canonical_holdout_id or "
            "src/research/holdout.py canonical_holdout_id()/resolve_holdout_id(); never by filename ordering."
        ),
        "audit_reference": "AUDIT_WP5_R2 finding 10 (NON_BLOCKING)",
        "non_material_reason": (
            "Zero holdout rows were consumed during discovery; no performance signal was examined; "
            "boundaries and embargo are identical."
        ),
        "mutates_immutable_artifact": False,
        "created_at": created_at,
    }


def build_certification(certified_at):
    return {
        "certification_schema": CERTIFICATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "target_id": TARGET_ID,
        "feature_set_id": FEATURE_SET_ID,
        "panel_version": PANEL_VERSION,
        "target_version": TARGET_VERSION,
        "corrected_wp4_experiment_id": CORRECTED_WP4_EXPERIMENT_ID,
        "corrected_from_experiment_id": CORRECTED_FROM_EXPERIMENT_ID,
        "producing_code_commit": PRODUCING_CODE_COMMIT,
        "branch": BRANCH,
        "canonical_holdout_id": CANONICAL_HOLDOUT_ID,
        "superseded_holdout_id": RECORDED_HOLDOUT_ID,
        "holdout_binding_correction": (
            "provenance/wp5/experiment_f985287c1315/holdout_binding_correction.json"
        ),
        "candidate_count": 23,
        "robust_candidate_count": 0,
        "classification_counts": {
            "ROBUST_CANDIDATE": 0,
            "PROMISING_BUT_UNSTABLE": 0,
            "REDUNDANT": 10,
            "LOW_COVERAGE": 10,
            "NO_CLEAR_SIGNAL": 0,
            "DIRECTION_UNSTABLE": 2,
            "TEMPORALLY_UNSAFE": 0,
            "DEFER": 1,
        },
        "fdr": {"method": "benjamini_hochberg", "hypotheses": 23, "alpha": 0.05, "rejected": 0},
        "verdicts": {
            "data_integrity_engineer": "RESEARCH_READY_WITH_LIMITATIONS",
            "research_ops_engineer": "REPRODUCIBLE",
            "variable_discovery_scientist": "VDS_WP5_PASS",
            "walk_forward_validator": {
                "initial": "WFV_WP5_CONCERNS",
                "remediation": "WFV_HOLDOUT_REBOUND",
            },
            "quant_auditor": {
                "audit_id": "AUDIT_WP5_R2",
                "hard_defects": 0,
                "material_defects": 0,
                "verdict": "AUDIT_PASS",
            },
        },
        "known_limitations": [
            "No EODHD delisting returns: terminal outcomes are right-censored, never fabricated",
            "abnormal_volume UNAVAILABLE (certified PIT price table has no volume column)",
            "SEC EDGAR companyfacts begin ~2009 so early fundamental coverage is bounded",
            "~11.8x overlapping 12-month labels: monthly cross-sectional IC + HAC used as primary inference",
            "0 ROBUST_CANDIDATE: discovery found no feature surviving BH-FDR; this is the honest result, not a tuned outcome",
        ],
        "certified_at": certified_at,
        "status": "WP5_COMPLETE",
    }


def _pretty_text(obj):
    """WP4 certification convention: sorted keys, 2-space indent, no trailing newline."""
    return json.dumps(obj, indent=2, sort_keys=True)


def _canonical_text(obj):
    """Provenance-record convention: compact canonical JSON with newline."""
    return canonical_json(obj) + "\n"


def _write_new(path, text, check):
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return "verify_and_reuse"
        raise SystemExit("refusing to overwrite %s with different content" % path)
    if check:
        return "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "written"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--certified-at", default=None, help="ISO-8601 certification timestamp")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    # The correction record is created at the certification instant; both share
    # the single authorized timestamp so the artefacts stay mutually consistent.
    stamp = _timestamp(args.certified_at)

    outcomes = {
        "correction": _write_new(CORRECTION_PATH, _canonical_text(build_correction(stamp)), args.check),
        "certification": _write_new(CERTIFICATION_PATH, _pretty_text(build_certification(stamp)), args.check),
    }
    print("WP5_CERTIFY_DONE timestamp=%s outcomes=%s" % (stamp, outcomes))
    print("correction=%s" % CORRECTION_PATH)
    print("certification=%s" % CERTIFICATION_PATH)


if __name__ == "__main__":
    main()
