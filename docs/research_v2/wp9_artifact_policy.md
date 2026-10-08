# WP9 Prospective Shadow Artifact Policy

This policy classifies files produced by the WP9 prospective-shadow system and
defines what may enter version control.

## A. Tracked canonical provenance

Track in git:

- `provenance/wp9/index.json` — canonical run registry (official snapshot ids,
  matured evaluations, stage, correction/supersession records).
- `provenance/wp9/predictions/` — immutable future official prediction snapshots
  and their `index.json`.
- `provenance/wp9/operational_health/official/` — official operational-health
  records.
- `provenance/wp9/economics/` — official economic shadow vintages.

Do not add blanket ignores for these paths.

Canonical official snapshot existence and identity are proven by:

1. an immutable snapshot body under `provenance/wp9/predictions/`;
2. a corresponding entry in `provenance/wp9/predictions/index.json` whose
   `sha256` matches that body;
3. the snapshot id present in `provenance/wp9/index.json` under
   `official_snapshot_ids`;
4. the deterministic snapshot id derived from the contract digest, as-of,
   champion freeze, model/preprocessor/calibrator hashes, universe/source/
   feature fingerprints, and producing commit.

Dry-run files and dry-run ids never prove canonical snapshot existence.

## B. Generated reproducible runtime output

Ignored and reproducible; do not commit:

- `data/research_v2/` — Bronze/Silver/Gold runtime data layers.
- `artifacts/` — generated research artifacts.

## C. Ignored non-evidentiary dry-run output

Ignored and permanently NON_EVIDENTIARY:

- `provenance/wp9/dry_runs/`
- `provenance/wp9/operational_health/dry_runs/`

Dry runs are rehearsals only. They never enter the canonical run registry or
prediction index.

## D. Forbidden from commit

- `.a0proj/` — Agent Zero tooling state, memory, and local secrets/config.
- Any secret or `.env*` file.

## Hard rule

Dry-run scoring must never create or modify `provenance/wp9/index.json`,
`provenance/wp9/predictions/`, `provenance/wp9/operational_health/official/`, or
`provenance/wp9/economics/`. Dry-run output lives only under the ignored
dry-run directories above.