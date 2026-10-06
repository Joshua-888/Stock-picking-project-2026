# WP7 Validation Contract v5

- Schema: `wp7_validation_contract_v5`
- Contract version: `WP7_VALIDATION_CONTRACT_V5`
- Supersedes: `WP7_VALIDATION_CONTRACT_V4`
- `created_before_results: true`
- Frozen at stage: `WP7_PRESTART`

This contract is frozen **before any WP7 results/interpreting code exists**. It is not tuned
from any WP7 metric. The machine-readable canonical copy is
`provenance/wp7/validation_contract_v5.json`.

Supersession reason: v4 noise-feature control criterion was too vague/post-hoc; v5 adds
exact deterministic `negative_controls_spec`. No other scientific inputs change.

All other frozen WP7 inputs from v4 remain byte-for-byte identical in value, including
inner `min_train_months=37`, raw counts `[5,6,7,8]`, safe counts `[4,5,6,7]`, outer windows,
eligible model set, calibration, metrics, seeds, and holdout procedure.

See `provenance/wp7/validation_contract_v5.json` for the full machine-readable contract.
