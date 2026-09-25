# WP2C Rebuild Runbook — Survivorship-Controlled S&P 500 PIT Panel

Deterministic rebuild of `dataset_dbaa77445b38` (`sp500_pit_research_panel_v1`).
Free sources only: EODHD (prices, corporate actions, symbol lists) + Wikipedia
S&P 500 membership change table. No synthetic data; research mode only.

## Prerequisites

- Runtime: `/opt/venv/bin/python` (task dependencies).
- Credential: `EODHD_API_TOKEN` in `.a0proj/secrets.env` (NOT shell env). The
  build parses it into `os.environ` itself; never export or print it.
- Branch: `feat/v2-wp2b-universe-delisted-unblock`.
- Do NOT touch V1 paths, `config/config.yaml` or `data/stock_analysis.db`.

## Order

1. Live build (fetch → BRONZE → SILVER, ~10 min depending on cache):
   ```
   /opt/venv/bin/python scripts/research_v2/wp2b_build_live_dataset.py
   ```
   Writes BRONZE immutables + SILVER versioned tables and declares the exact
   silver versions in `artifacts/research/wp2b_live/layer_records.json`.
   Must end with `BUILD_DONE`. The A1 gate prints
   `current_constituents_silently_absent: []` — a non-empty list is a hard FAIL.

2. Finalize (SILVER → GOLD panel, diagnostics, manifest, ledger):
   ```
   /opt/venv/bin/python scripts/research_v2/wp2b_finalize_gold.py
   ```
   Reads the pinned silver versions (never a lexicographic guess), writes a
   self-contained gold table (`data/research_v2/gold/sp500_pit_research_panel_v1/<fp>/data.parquet`),
   `security_master.json`, `universe.json`, `censoring_diagnostics.json`,
   `manifest.json`, `gold_summary.json`, and a `provenance/index.json` entry.

3. Fingerprint check — the finalize log prints `GOLD_DONE ... dataset_id=<id>`.
   Confirm it matches:
   - dataset_id: `dataset_dbaa77445b38`
   - dataset_fingerprint: `ac294282d8e949f270912682cf70a803f596c030885657b0aaf540536e2663ad`
   A different fingerprint means an input, code or config change: re-version,
   never overwrite the prior artifacts.

4. Integrity checks:
   ```
   /opt/venv/bin/python -m pytest -q -ra
   ```
   Also verify the ledger: `provenance_ledger.verify_ledger("provenance")` → `[]`.

## Memory note

The container is ~4 GB. The silver price assembly MUST use the per-security
`slim`-frame concat path (not a list of ~3.5M dicts) or the build is OOM-killed
after `LISTING_GUARD`.

## Provenance wrap

- `provenance/datasets/<dataset_id>.json` — immutable manifest ledger record.
- `provenance/index.json` — index entry (must reference a tracked file).
- `provenance/wp2c/security_master_version.json` and
  `provenance/wp2c/universe_version.json` — explicit tracked version artifacts
  (regenerated copies of `artifacts/research/wp2b_live/{security_master,universe}.json`).

## Determinism

Identical source fingerprints + silver versions + code commit + config
fingerprint reproduce the same dataset fingerprint. The build is offline-capable
from the BRONZE cache when the fetch cache under `/tmp/wp2b_cache` is warm; a
cold cache requires the EODHD token.