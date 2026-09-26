# WP4 Rebuild Runbook — Keyes variable engine

Scope: reproduce the WP4 Keyes variable values, historical/modern tracks and
diagnostics from certified upstream inputs. This runbook is provenance and
process only; it does not change any scientific methodology.

## Producing-code commit

The WP4 code first exists at commit **`a1c0a92c072ba0f0232d363f53824984c7b58d9a`**
(`WP4: Keyes variable values, historical/modern tracks, diagnostics, tests`).

The first WP4 provenance record was frozen with the WP3 commit `437f68ce...`
because the build ran while the WP4 code was still uncommitted. That record is
superseded, not deleted, and the correction lives in
`provenance/wp4/corrections/`. See "Corrected provenance" below.

## Inputs (all certified; never rebuilt here)

| Input | Identifier / version |
| --- | --- |
| WP2C gold research panel | `dataset_dbaa77445b38` / `ac294282d8e949f2` |
| WP3 gold 12m targets | `target_set_888f68d1cfd0` / `f244f86c22b2c2a4` |
| WP3 feature set | `feature_set_56361533cc1b` |
| WP2B silver prices | `a0699674d7650583` |
| WP2B silver actions | `68bdbdf4ea75a862` |
| SPY benchmark (silver) | `a0ab5bd2382518d4` |
| SPY bronze corporate actions | `data/research_v2/bronze/wp3_spy_prices_actions/*/raw.json` |
| EDGAR silver fundamentals | `data/research_v2/silver/edgar_fundamentals/*/data.parquet` |
| EDGAR CIK mapping | `artifacts/research/wp4/edgar_cik_mapping.json` |

## Build (research mode, real data only)

Run from the repository root, checked out at `a1c0a92` (or a descendant that does
not change the WP4 engine):

```
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp4_build_keyes.py
```

Outputs land in `artifacts/research/wp4/` (gitignored) and an immutable
provenance record in `provenance/wp4/`. The build is deterministic; re-running
on unchanged inputs reproduces byte-identical scientific artefacts.

## Rebuild equivalence

Rebuilding from `a1c0a92` into a sandbox reproduces the frozen scientific content
**byte-identically** for: `keyes_spec.json`, `variable_mapping.json`,
`coverage.json`, `fidelity_table.json`, `historical_signals.parquet`,
`modern_signals.parquet`, `diagnostics.json`, `yearly_stability.csv`,
`qualification_by_year.csv`, `placebo.json`, `sector_stability.json`.

The only difference is identity metadata: `wp4_run_summary.json` records
`git_commit` (`437f68ce` -> `a1c0a92`). No scientific value changes.

## Corrected provenance

- Corrected record: `provenance/wp4/corrections/experiment_609b46519a41.json`
  (`producing_code_commit = a1c0a92...`).
- Supersession record:
  `provenance/wp4/corrections/supersession_experiment_5ed52dcf2f44.json`.
- Machine-readable resolver index: `provenance/wp4/corrections/index.json`.

Rebuild or verify the correction (idempotent; write-once):

```
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp4_correct_provenance.py
PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp4_correct_provenance.py --check
```

Downstream consumers must resolve `experiment_5ed52dcf2f44` through
`src.research.provenance_corrections.resolve_experiment(...)` rather than by
filename ordering. The misbound records
(`provenance/wp4/experiment_5ed52dcf2f44.json`, `provenance/wp4/latest_experiment.json`)
are listed as superseded and must never be selected as canonical.