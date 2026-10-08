# WP9 Matured Evaluation Runbook

Status: ACTIVE
Binding contract: `provenance/wp9/forward_validation_contract_v1.json`
Scope: matured forward-snapshot evaluation, registry updates, Stage B/C milestone
handling, and recovery. This runbook contains **no model-changing procedure**.
No retraining, recalibration, feature selection, cutoff search, challenge, or
champion change is permitted.

Runtime:

```bash
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_evaluate_matured.py
```

## Preconditions

- The snapshot to evaluate is an official WP9 snapshot listed in both
  `provenance/wp9/predictions/index.json` and the run registry
  `official_snapshot_ids`.
- The snapshot file and its index SHA-256 are unchanged and immutable.
- The producing code around evaluation is understood; evaluation does not
  modify the snapshot or champion.
- `evaluation_date` is deterministic and no earlier than every observable row's
  target maturity instant.

## Inputs and outputs

Inputs:

- `--snapshot-id <id>`: the official snapshot to evaluate.
- `--evaluation-date <ISO>`: optional deterministic evaluation date.
  Default is current UTC. Use an explicit date for replay.

Outputs:

- `provenance/wp9/matured_evaluations/<evaluation_id>.json`
- An appended reference in `provenance/wp9/index.json` under
  `matured_evaluations`.
- Updated `current_prospective_stage` and `updated_at_utc`.

## Maturity and evaluation steps

### 1 — Resolve the official snapshot

The evaluator loads the snapshot from the official prediction index, verifies
its index SHA-256, exact snapshot schema, body/filename id, champion freeze id,
contract digest, as-of binding, and every frozen champion artifact hash. A dry
run, invalidated, superseded, unknown-freeze, unknown-contract, or altered
snapshot is refused.

### 2 — Enforce the post-freeze official-date rule

The snapshot as-of must be strictly after the contract freeze commit timestamp.
Backdated or at-freeze instant snapshots are refused.

### 3 — Rebuild certified outcomes

The evaluator rebuilds the 12-month outcomes with the certified WP3 target
construction and the same `target_set_d2bb16610bce` semantics used for
`target_known_at`. Censored rows are retained and excluded from metrics; they
are never converted to zero, negative, or failure labels.

### 4 — Require maturity at the evaluation date

For every observable target row, `target_known_at` must be present and not
after `evaluation_date`. Missing, unparseable, or future maturity values return
`IMMATURE_OR_INVALID` with exact blocking reasons. A snapshot with no observable
target outcome at that date is refused.

### 5 — Record predeclared metrics

The evaluator records:

- cross-sectional ROC-AUC for the frozen calibrated score
- `auc_skill` (`roc_auc - 0.5`)
- observable/censored counts
- outcome-row evidence keys and values

No cutoff is searched and no threshold is selected. The top economic cohort is
exactly `percentile > 0.80`; the bottom descriptive band is `percentile < 0.20`.

### 6 — Persist the immutable evaluation

The evaluation identity is a deterministic digest of its canonical content. The
write-once file can never be overwritten with different content. Re-running the
same snapshot returns `ALREADY_MATURED` and reuses the stored evaluation.

### 7 — Update the run registry

The evaluator appends a sorted matured-evaluation reference and recomputes the
prospective stage from the matured count:

- fewer than 12 -> `A_OPERATIONAL_SHADOW`
- 12 to 23 -> `B_FIRST_INTERIM_REVIEW`
- 24 or more -> `C_MAJOR_STATISTICAL_GATE`

This is descriptive only and never authorizes a champion change.

## Stage B/C handling

### Stage B — 12 official monthly snapshots matured

Action: first prospective interim review. Report the predeclared metrics only.

- H0: mean prospective monthly `auc_skill <= 0`
- H1: mean prospective monthly `auc_skill > 0`
- inference: one-sided HAC/Newey-West t-test with lag 12
- primary metric: equal-weighted mean monthly `auc_skill`

**No champion change.** Record the evidence and notify the Supervisor.

### Stage C — 24 official monthly snapshots matured

Action: first major prospective statistical gate. Do not shorten the window.
Report the same predeclared metrics with the independent audit trail. No model,
preprocessor, calibrator, feature-set, or cutoff may be changed here.

## Recovery procedures

### Immature snapshot attempted too early

Do not evaluate it. Record the exact `IMMATURE_OR_INVALID` reasons and retry at
or after the latest observable `target_known_at`.

### Invalidated or superseded snapshot requested

Do not evaluate it. Resolve the supersession record in the run registry and use
the corrected official snapshot id. The original file and index entry remain as
preserved evidence.

### Altered snapshot file

The evaluator refuses on SHA-256 mismatch. Do not repair the file in place.
Escalate to the Quant Auditor/Supervisor and use the recorded supersession
procedure if correction is authorized.

### Missing market inputs

The evaluator reports the exact missing certified input and blocks. Do not
fabricate prices, returns, filing dates, or membership data. Restore the pinned
certified layer version before retrying.

### Registry drift

If the run registry contract digest, schema, or champion freeze differs from the
frozen contract, load fails closed. Do not hand-edit provenance. Escalate to the
Supervisor/Qualified specialist.

## Invariants

- The frozen champion is never loaded, fitted, or changed for the purpose of
  improving this result.
- `FROZEN_CALIBRATED_SCORE` is a ranking signal, not a validated probability.
- Probability claims are forbidden.
- Failed or mature results are retained; no provenance file is deleted.
