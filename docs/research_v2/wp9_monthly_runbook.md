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

## Deterministic operational entry points

The first official month-end is executed strictly one command per step, in this
exact order. Official scoring must never run just because a calendar date
arrived; it requires the readiness preflight to answer `READY` first.

```bash
# 1. Refresh separate live-forward inputs (operational only; never mutates
#    certified historical layers, benchmark gold, or bound EDGAR binding)
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_refresh_forward_data.py --as-of YYYY-MM-DD

# 2. Read-only readiness preflight (exit 0=READY, 1=NOT_READY, 2=usage/error)
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_snapshot_readiness.py --as-of YYYY-MM-DD

# 3. Non-evidentiary dry-run (never enters the official prediction index)
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_forward_score.py --dry-run --as-of YYYY-MM-DD --live

# 4. Official scoring exactly once, only after preflight READY and git checkpoint
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_forward_score.py --official --as-of YYYY-MM-DD --live
```

The full one-command-per-step order is at the bottom of this runbook.

## Preconditions

- Branch is the intended WP9 feature branch, never `main` for an experimental
  snapshot change.
- The frozen champion artifacts under
  `artifacts/research/wp8/freeze_e62eac30df40/` exist and their SHA-256 values
  match the contract.
- `provenance/wp9/forward_validation_contract_v1.json` is committed and its
  working-tree bytes are identical to HEAD.
- The producing code path is clean in `git status` for official scoring.
- Official scoring is never run unattended or by calendar trigger alone; it is
  a human-verified step gated on a `READY` preflight.

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

### Step 3 — Refresh separate live-forward inputs

Run the operational live-forward refresh. The fetch bound is either the provided
`--as-of` close or the current wall clock (staging), and never
`RESEARCH_WINDOW_END`. Results are written only to the separate
`data/research_v2/silver/wp9_live_*/` namespace and
`artifacts/research/wp9_live/layer_records.json`; certified historical layers,
`benchmark_gold_SPY`, the bound EDGAR binding, and
`artifacts/research/wp2b_live/layer_records.json` are never overwritten.

```bash
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_refresh_forward_data.py --as-of YYYY-MM-DD
```

Any empty or malformed market/actions/benchmark provider payload causes a hard
failure, not a warning. The EODHD token is never printed or logged.

### Step 4 — Read-only readiness preflight

Run the deterministic readiness preflight. It never writes an official snapshot,
fits anything, or reads target/outcome columns.

```bash
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_snapshot_readiness.py --as-of YYYY-MM-DD
```

```text
READY        -> exit 0
NOT_READY    -> exit 1  (hard block for this as-of)
usage/error  -> exit 2
```

`READY` requires, among others: contract identity/digest and reachable freeze
commit, champion file hashes matching contract and disk, clean producing
worktree on `feat/wp9-prospective-shadow`, no duplicate official snapshot,
as-of strictly after the freeze, exact month-end cadence from observed trade
dates, UTC close already elapsed, live coverage through the as-of, fundamental
availability, well-formed layer records/manifest, no target/outcome columns, no
future rows, no empty staged payloads, and a clean secret scan.

### Step 5 — Choose the official snapshot as-of date

The snapshot date is the **last eligible trading/score date of the intended
month** and must be **strictly after the contract freeze commit timestamp**.
It must not backdate or backfill any earlier month.

Do not use the calendar-only `monthly_cadence_is_valid` helper for official
cadence. The readiness preflight and the scorer's fail-closed preflight resolve
the exact last eligible trading/score date for the month from the observed
price series (never a fuzzy holiday calendar or plain calendar month-end):

```bash
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from pathlib import Path
from scripts.research_v2.wp9_forward_score import preflight_official
root = Path('.')
asof = 'YYYY-MM-DD'  # replace with the intended last eligible trading/score date
contract, code_commit = preflight_official(asof, root=root)
print('freeze_commit', contract.get('prospective', 'contract_freeze_commit'))
print('official_preflight_ok', code_commit is not None)
PY
```

Provenance note: corrected `contract_freeze_commit` is
`c96559cf2eb7c65e2c15c4f8014a77faf2a8f4d3` (the reachable commit containing the
authoritative contract bytes). Supersession record:
`provenance/wp9/corrections/contract_v1_freeze_commit_001.json`.

### Step 6 — Confirm no duplicate official snapshot exists

```bash
PYTHONPATH=. /opt/venv/bin/python - <<'PY'
from pathlib import Path
from src.research.wp9.storage import has_prediction_for_asof
print(has_prediction_for_asof('YYYY-MM-DD', root=Path('.')))
PY
```

`True` is a block. Never overwrite or replace the existing official snapshot.

### Step 7 — Run the non-evidentiary dry-run

```bash
PYTHONPATH=. /opt/venv/bin/python \
  scripts/research_v2/wp9_forward_score.py --dry-run --as-of YYYY-MM-DD --live
```

The dry-run writes only under `provenance/wp9/dry_runs/` and the separated
dry-run operational-health path. It must never enter
`provenance/wp9/predictions/index.json` and never contribute prospective
performance statistics.

### Step 8 — Review dry-run evidence

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

### Step 9 — Resolve blocking operational issues

If any `operational_blockers` appear or the dry-run fails, diagnose the root
cause. Per the contract correction policy, changes may only address an
operational/data-ingestion defect discovered **before outcomes mature**. The
original snapshot, if written, must be preserved as
`INVALIDATED_OPERATIONAL_NON_EVIDENTIARY` with explicit supersession and Quant
Auditor approval. Never correct a snapshot merely because the ranks look wrong.

### Step 10 — Git checkpoint before official scoring

```bash
git status --short
git rev-parse HEAD
```

Sequential order of the official month-end:

1. source refresh (`wp9_refresh_forward_data.py`)
2. readiness preflight returns `READY` (`wp9_snapshot_readiness.py`)
3. git checkpoint / clean producing worktree
4. official scorer exactly once (`wp9_forward_score.py --official --as-of ... --live`)
5. health inspection
6. commit official evidence
7. push/verify
8. read-only integrity review
9. STOP (no outcome evaluation)

### Step 11 — Run the official monthly snapshot

Run only after the dry-run is clean and the as-of passes Steps 1–10:

```bash
PYTHONPATH=. /opt/venv/bin/python \
  scripts/research_v2/wp9_forward_score.py --official --as-of YYYY-MM-DD --live
```

Official mode preflights a clean producing worktree, exact contract bytes,
freeze ordering, monthly cadence, no duplicate as-of, up-to-date wall-clock
close, and frozen artifact hash verification.

### Step 12 — Verify the official immutable outputs and registry

Confirm these files were written exactly once:

- `provenance/wp9/predictions/<snapshot_id>.json`
- `provenance/wp9/predictions/index.json`
- `provenance/wp9/operational_health/official/<snapshot_id>.json`
- `provenance/wp9/economics/<economic_shadow_id>.json`
- `provenance/wp9/index.json`

Confirm the snapshot body has only the contract `snapshot_schema_fields` and
that its SHA-256 appears in the official prediction index. Confirm the index
has exactly one entry for the as-of date. Then verify the run registry:

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

### Step 13 — Drift monitoring, handoff, and immutability

Review `data_drift_alerts` and the monitoring fields. Drift is
descriptive/non-adaptive only; the only permitted automatic action is
`none; no retrain/recalibrate`.

Record in the monthly handoff: as-of date and snapshot id, universe size,
scoreable count, coverage/missingness summary, artifact hash verification,
source manifest and feature snapshot hashes, drift alerts and blockers, and the
exact running commit. State that no future outcome was loaded; do not interpret
the frozen calibrated score as a probability and do not claim predictive
validity.

Do not modify, rewrite, or delete any written official snapshot. Keep the
contract file and champion artifacts byte-identical to HEAD. Commit only
procedural or code changes on the feature branch, never large provider data.
Do not force-push and do not merge into `main` without Supervisor approval.
