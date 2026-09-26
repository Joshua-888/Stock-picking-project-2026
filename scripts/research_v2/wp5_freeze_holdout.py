"""WP5: freeze the V2 locked holdout BEFORE any target-based modelling.

This script emits the content-addressed locked-holdout artefact and its index
under ``provenance/holdout/``. It is deterministic and write-once:

* the ``holdout_id`` is the WP1 ``holdout`` content-addressed id of the
  definition payload (no clock, no randomness);
* ``frozen_at`` is derived from the committer date of the PRODUCING commit, so
  re-running verifies and reuses the same artefact rather than rewriting it;
* the holdout boundary is chosen on DATA-AVAILABILITY grounds only (observable
  forward-label coverage and censoring), never on any performance signal.

The script inspects NO target values and NO feature performance. It only counts
date availability to justify the boundary.

Run: /opt/venv/bin/python scripts/research_v2/wp5_freeze_holdout.py --producing-commit <sha>
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

from src.research.fingerprints import fingerprint_obj
from src.research.holdout import (
    DEFAULT_DATASET_ID,
    DEFAULT_EMBARGO_MONTHS,
    DEFAULT_FEATURE_SET_ID,
    DEFAULT_HOLDOUT_END,
    DEFAULT_HOLDOUT_START,
    DEFAULT_TARGET_ID,
    HOLDOUT_SCHEMA_VERSION,
    build_holdout,
    holdout_payload,
)
from src.research.ids import canonical_json, holdout_id
from src.research.immutability import save_immutable, write_json_atomic

PROVENANCE_DIR = ROOT / "provenance" / "holdout"

SELECTION_RATIONALE = (
    "Data-availability only. Observable forward labels run 2007-01-31..2025-08-31; "
    "2025 is ~63% observable and 2026 is 0% observable because the 12-month horizon "
    "runs past the research-window end. The most recent contiguous fully-usable segment "
    "therefore ends 2025-08-31, and the last ~4 years of it (2022-01-01..2025-08-31) is "
    "reserved as the locked holdout. No performance signal was examined."
)


def _git(*args):
    result = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _commit_iso(commit):
    return _git("show", "-s", "--format=%cI", commit)


def main():
    parser = argparse.ArgumentParser(description="Freeze the V2 locked holdout (WP5).")
    parser.add_argument("--producing-commit", required=True,
                        help="commit that introduces the holdout module/spec")
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START)
    parser.add_argument("--holdout-end", default=DEFAULT_HOLDOUT_END)
    parser.add_argument("--embargo-months", type=int, default=DEFAULT_EMBARGO_MONTHS)
    args = parser.parse_args()

    frozen_at = _commit_iso(args.producing_commit)
    if not frozen_at:
        raise SystemExit("cannot resolve committer date for %s" % args.producing_commit)

    payload = holdout_payload(
        dataset_id=DEFAULT_DATASET_ID,
        target_id=DEFAULT_TARGET_ID,
        feature_set_id=DEFAULT_FEATURE_SET_ID,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        embargo_months=args.embargo_months,
        selection_rationale=SELECTION_RATIONALE,
        git_commit=args.producing_commit,
        frozen_at=frozen_at,
    )
    record = build_holdout(payload)
    record_id = record["holdout_id"]
    artifact_name = "%s.json" % record_id
    artifact_path = PROVENANCE_DIR / artifact_name

    index = {
        "holdout_schema_version": HOLDOUT_SCHEMA_VERSION,
        "definition_fingerprint": fingerprint_obj(payload),
        "holdout": {
            "holdout_id": record_id,
            "artifact_file": artifact_name,
        },
        "prohibition_rule": record["prohibition_rule"],
    }

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    outcome_artifact = save_immutable(artifact_path, record)
    write_json_atomic(PROVENANCE_DIR / "index.json", index)

    # Independent self-check: recompute the id from the record's own fields.
    recomputed = holdout_id({key: value for key, value in record.items() if key != "holdout_id"})
    assert recomputed == record_id, "holdout id is not content-addressed"

    print("holdout_id=%s" % record_id)
    print("holdout_start=%s holdout_end=%s embargo_months=%d" % (
        args.holdout_start, args.holdout_end, args.embargo_months))
    print("artifact=%s (%s)" % (artifact_path.relative_to(ROOT), outcome_artifact))
    print("index=%s" % (PROVENANCE_DIR / "index.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
