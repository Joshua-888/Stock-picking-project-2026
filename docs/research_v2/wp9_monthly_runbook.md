# WP9 Prospective Shadow Monthly Runbook

Status: ACTIVE
Binding contract: `provenance/wp9/forward_validation_contract_v1.json`
Scope: monthly non-evidentiary rehearsal and official prospective shadow scoring
using the frozen WP8 champion. This runbook contains **no model-changing
procedure**. No retraining, recalibration, feature selection, cutoff search, or
champion change is permitted.

Runtime:

```bash
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_forward_score.py
```

Repository root is the project root. All commands run from the project root.

## Preconditions

- Branch is the intended WP9 feature branch, never `main` for an experimental
  snapshot change.
- The frozen champion artifacts under
  `artifacts/research/wp8/freeze_e62eac30df40/` exist and their SHA-256 values
  match the contract.
- `provenance/wp9/forward_validation_contract_v1.json` is committed and its
  working-tree bytes are identical to HEAD.
- The producing code path is clean in `git status` for official scoring.

## The 13 monthly steps

### Step 1 — Confirm repository and contract state

```bash
git rev-parse HEAD
git status --short
git log --oneline -1
```

Confirm the branch and HEAD. Confirm no tracked contract or producing-code file
has uncommitted modifications intended to enter an official snapshot.

### Step 2 — Verify frozen champion artifact hashes

Use the contract-listed SHA-256 values for `model`, `preprocessor`, and
`calibrator` and compare with their on-disk files:

```bash
sha256sum \
  artifacts/research/wp8/freeze_e62eac30df40/final_model.pkl \
  artifacts/research/wp8/freeze_e62eac30df40/final_preprocessor.pkl \
  artifacts/research/wp8/freeze_e62eac30df40/final_calibrator.pkl
```

Any mismatch is a hard block. Do not replace the frozen champion.

### Step 3 — Choose the official snapshot as-of date

The snapshot date is the **last eligible trading/score date of the intended
month** and must be **strictly after the contract freeze commit timestamp**.
It must not backdate or backfill any earlier month.

Validate cadence and freeze ordering before proceeding. Do not use the calendar-only
`monthly_cadence_is_valid` helper for official cadence; route official cadence
through the scorer's fail-closed preflight, which resolves the exact last
eligible certified trading/score date for the month from the bound silver price
series (never a fuzzy holiday calendar or plain calendar month-end):

```bash
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from pathlib import Path
from scripts.research_v2.wp9_forward_score import preflight_official
root = Path('.')
asof = 'YYYY-MM-DD'  # replace with the intended last certified trading/score date
contract, code_commit = preflight_official(asof, root=root)
print('freeze_commit', contract.get('prospective', 'contract_freeze_commit'))
print('official_preflight_ok', code_commit is not None)
PY
```

Provenance note: corrected `contract_freeze_commit` is
`c96559cf2eb7c65e2c15c4f8014a77faf2a8f4d3` (the reachable commit containing the
authoritative contract bytes). Supersession record:
`provenance/wp9/corrections/contract_v1_freeze_commit_001.json`.

### Step 4 — Confirm no duplicate official snapshot exists

```bash
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from pathlib import Path
from src.research.wp9.storage import has_prediction_for_asof
print(has_prediction_for_asof('YYYY-MM-DD', root=Path('.')))
PY
```

`True` is a block. Never overwrite or replace the existing official snapshot.

### Step 5 — Run the non-evidentiary dry-run

```bash
PYTHONPATH=. /opt/venv/bin/python \
  scripts/research_v2/wp9_forward_score.py --dry-run --as-of YYYY-MM-DD
```

The dry-run writes only under `provenance/wp9/dry_runs/` and the separated
dry-run operational-health path. It must never enter
`provenance/wp9/predictions/index.json` and never contribute prospective
performance statistics.

### Step 6 — Review dry-run evidence

Check the printed fields and the dry-run health record:

- `universe_size`
- `scoreable_count`
- `excluded_count` and `coverage_rate`
- `feature_missingness` per frozen feature
- `identity_failures` and `provider_failures`
- `artifact_hash_verification` (all must be `true`)
- `score_distribution`
- `rank_concentration`
- `source_manifest_hash` and `feature_snapshot_hash`

Outcome labels must not be loaded or scored. The scorer refuses any frame with
`outperform_12m`, `future_12m_*`, `target_*`, or censoring columns.

### Step 7 — Resolve blocking operational issues

If any `operational_blockers` appear or the dry-run fails, diagnose the root
cause. Per the contract correction policy, changes may only address an
operational/data-ingestion defect discovered **before outcomes mature**. The
original snapshot, if written, must be preserved as
`INVALIDATED_OPERATIONAL_NON_EVIDENTIARY` with explicit supersession and Quant
Auditor approval. Never correct a snapshot merely because the ranks look wrong.

### Step 8 — Run the official monthly snapshot

Run only after the dry-run is clean and the as-of passes Steps 1–4:

```bash
PYTHONPATH=. /opt/venv/bin/python \
  scripts/research_v2/wp9_forward_score.py --official --as-of YYYY-MM-DD
```

Official mode preflights a clean producing worktree, exact contract bytes,
freeze ordering, monthly cadence, no duplicate as-of, and frozen artifact hash
verification.

### Step 9 — Verify the official immutable outputs

Confirm these files were written exactly once:

- `provenance/wp9/predictions/<snapshot_id>.json`
- `provenance/wp9/predictions/index.json`
- `provenance/wp9/operational_health/official/<snapshot_id>.json`
- `provenance/wp9/economics/<economic_shadow_id>.json`
- `provenance/wp9/index.json`

Confirm the snapshot body has only the contract `snapshot_schema_fields` and
that its SHA-256 appears in the official prediction index. Confirm the index
has exactly one entry for the as-of date.

### Step 10 — Update and verify the run registry

The scorer appends the snapshot id to `official_snapshot_ids` in
`provenance/wp9/index.json`. Verify:

```bash
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from pathlib import Path
from src.research.wp9.storage import load_run_registry
reg = load_run_registry(root=Path('.'))
print(reg['champion_freeze'])
print(reg['contract_digest'])
print(reg['current_prospective_stage'])
print(len(reg['official_snapshot_ids']))
PY
```

Stage A (`A_OPERATIONAL_SHADOW`) requires at least three consecutive valid
official monthly snapshots. Do not accelerate Stage B or C.

### Step 11 — Record drift and operational monitoring descriptively

Review `data_drift_alerts` and the monitoring fields. Drift is
descriptive/non-adaptive only. The only permitted automatic action is
`none; no retrain/recalibrate`. Report alerts; do not adapt the model.

### Step 12 — Produce the run handoff

Record in the monthly handoff:

- as-of date and snapshot id
- universe size, scoreable count, coverage, missingness summary
- artifact hash verification result
- source manifest hash and feature snapshot hash
- drift alerts and blockers
- exact running commit

State explicitly that no future outcome was loaded. Do not interpret the frozen
calibrated score as a probability and do not claim predictive validity.

### Step 13 — Preserve immutability and checkpoint the branch

Do not modify, rewrite, or delete any written official snapshot. Keep the
contract file and champion artifacts byte-identical to HEAD. Commit only
procedural or code changes on the feature branch, never large provider data.
Do not force-push and do not merge into `main` without Supervisor approval.
