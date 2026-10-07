# WP9_FORWARD_VALIDATION_CONTRACT_V1

- Schema version: `wp9_forward_validation_contract_v1`
- Contract version: `WP9_FORWARD_VALIDATION_CONTRACT_V1`
- Frozen at (UTC): 2026-10-07T21:59:03Z
- Contract digest: Contract digest will be computed from `provenance/wp9/forward_validation_contract_v1.json`

## Closure state

- `research_closed`: true
- `historical_tuning_closed`: true
- `consumed_holdout`: true

## WP8 closure

- `wp8_final_head`: `20e6395193047be7eb987664d51b09a241a0670e`
- `consumed_holdout_period`: `{"start": "2022-01-01", "end": "2025-08-31"}`
- `holdout_access_count`: 1
- `primary_verdict`: `HOLDOUT_POSITIVE_BUT_INCONCLUSIVE`
- `evaluation_id`: `evaluation_8db863349b51`
- `result_sha256`: `26689076bdb58631f1747a7beb922fefbc40de33f0ab2cfaeb950d61f9089be1`

## Champion

- `freeze_id`: `freeze_e62eac30df40`
- `wp7_generation_id`: `generation_de0f9bbd0bec`
- `wp7_contract`: `WP7_VALIDATION_CONTRACT_V5`
- `wp6_experiment_id`: `experiment_ee434a07a25d`
- `config_id`: `modelcfg_158499379e80`
- `model`: `extra_trees`
- `feature_strategy`: `B_coverage_qualified`
- `calibration`: `platt_scaling`
- Features:
  - `earnings_yield`
  - `current_pe`
  - `price_to_book`
  - `free_cash_flow_yield`
  - `dividend_yield`
  - `roa`
  - `roe`
  - `eps_growth_acceleration`
  - `six_month_momentum`
  - `twelve_month_momentum`
  - `five_year_price_gain`
  - `debt_to_equity`
  - `market_cap`
- `preprocessing_spec`: `VALUE_SPEC (median impute, indicator=True, percentile clip, robust scale; fitted training-only)`
- `training_period`: `{"start": "2007-01-31", "end": "2020-12-31"}`
- Seeds:
  - `bootstrap_seed`: 20260926
  - `model_seed`: 20260930
  - `placebo_seed`: 20260926
- Artifacts:
  - `model`: `artifacts/research/wp8/freeze_e62eac30df40/final_model.pkl`, sha256 `ebeb05c534e1a6025085b286c25906a7207ebf05a5a1415ff438c31a412e1d90`
  - `preprocessor`: `artifacts/research/wp8/freeze_e62eac30df40/final_preprocessor.pkl`, sha256 `591dff67b18de3fe4e215023d7c541c8c57420dcd841f6d5b05cb46cc42a4072`
  - `calibrator`: `artifacts/research/wp8/freeze_e62eac30df40/final_calibrator.pkl`, sha256 `ff285edb00aff9f64fba461d59c97d00e5e7b19d144ab5015315dfc79a63e524`

## Upstream lineage

- `dataset_id`: `dataset_35a278e17c13`
- `target_set_id`: `target_set_d2bb16610bce`
- `feature_set_id`: `feature_set_4f7b43726310`
- `universe_version`: `0ce5054a2d66ce76ee68b51ac3f41cda90b8444cf134569ddd7a16044086a796`
- `security_master_version`: `d4ba066eb6cf81a115f6626cb95d57d30e842c1f4c75ec84149b35b8dfd7d349`

## Prospective protocol

- `prediction_mode`: `frozen_ranking_signal_only`
- `probability_claim_forbidden`: true
- `calibrated_score_label`: `FROZEN_CALIBRATED_SCORE (NOT a validated probability)`
- `primary_human_output`: `rank`, `percentile`, `frozen_calibrated_score`
- `cadence`: `monthly`
- `official_snapshot_rule`: first full monthly scoring date (last eligible trading/score date) strictly AFTER this contract's freeze commit timestamp; no backdating; no official snapshot may be backfilled
- `contract_freeze_commit`: set to the actual git commit that contains this contract
- `target_horizon_months`: 12
- `maturity_rule`: snapshot matures only when target_known_at per `target_set_d2bb16610bce` availability semantics is <= evaluation run date

## Universe

- `source`: `sp500_pit_wikipedia_eodhd_v1`
- `membership_rule`: S&P 500 membership effective as-of T only; use effective_date (conservative instant); never future/current membership to reconstruct older snapshots
- `per_snapshot_recording`: `membership_source`, `membership_effective_date`, `security_ids`, `ticker_mapping`, `universe_hash`, `universe_count`, `source_retrieval_metadata`

## Sources

- `allowed`: `EODHD`, `SEC_EDGAR`, `WIKIPEDIA_SP500`
- `replacement_policy`: Data Integrity Engineer must review any proposed replacement source before Builder implements it

## Point-in-time rules

- `feature_fundamental`: SEC acceptance datetime, else filing_date + 1 day; never fiscal period end
- `feature_price`: trade date + conservative market close UTC
- `membership`: effective date instant only
- `corporate_actions_reconstruction`: apply actions with effective_date in (target_start, target_end]; raw close reconstructed, never provider-adjusted

## Ranking

- `primary_basis`: `raw_model_score`
- `calibrated_rank_equivalence`: frozen Platt logistic transform is strictly monotonic, so calibrated rank is equivalent
- `tie_break`: security_id lexical ascending

## Eligibility

- `preprocessing_rule`: use frozen preprocessor VALUE_SPEC exactly; no new live-data imputation
- `missing_handling`: record excluded securities and exclusion reasons; never silent drop

## Snapshot schema fields

`snapshot_id`, `snapshot_asof`, `security_id`, `ticker`, `raw_model_score`, `frozen_calibrated_score`, `rank`, `percentile`, `universe_size`, `model_freeze_id`, `model_hash`, `feature_snapshot_hash`, `source_manifest_hash`, `code_commit`, `created_at_utc`, `eligibility_status`, `quality_flags`

## Storage

- `predictions_dir`: `provenance/wp9/predictions`
- `index_file`: `provenance/wp9/predictions/index.json`
- `write_once`: true
- `snapshot_id_bindings`: `contract_digest`, `snapshot_asof`, `champion_freeze`, `model_hash`, `universe_hash`, `source_manifest_hash`, `feature_snapshot_hash`, `code_commit`

## Scientific protocol

- `primary_metric`: equal_weighted_mean_monthly_auc_skill (cross-sectional ROC-AUC per matured monthly snapshot minus 0.5, equal-weighted across months)
- Hypothesis:
  - H0: mean prospective monthly AUC_skill <= 0
  - H1: mean prospective monthly AUC_skill > 0
- `inference`: one-sided HAC/Newey-West t-test
- `hac_lag`: 12
- Stage A:
  - `milestone`: `OPERATIONAL_SHADOW_VALIDATED`
  - `trigger`: at least 3 consecutive valid official monthly snapshots generated
  - `alpha_claim`: none
- Stage B:
  - `trigger`: 12 official monthly snapshots matured
  - `action`: first prospective interim review; report predeclared metrics; NO champion change
- Stage C:
  - `trigger`: 24 official monthly snapshots matured
  - `action`: first major prospective statistical gate; do not shorten

## Economic shadow

- `cutoff`: top_quintile_exact (rank percentile > 0.80)
- `weighting`: equal_weight_at_entry
- `holding_months`: 12
- `benchmark`: SPY canonical total return (same reconstruction as target_set_d2bb16610bce)
- `cost_scenarios_bps`: 0, 10, 25, 50
- `cost_interpretation`: single entry-side deduction from cohort 12m return (cost_bps/10000); benchmark unadjusted; report all scenarios
- `real_trading`: false

## Immutability

- `snapshots_write_once`: true
- `correction_policy`: operational/data-ingestion defect only, discovered before outcomes mature; original preserved as INVALIDATED_OPERATIONAL_NON_EVIDENTIARY; new corrected snapshot; explicit supersession; Quant Auditor approval required; never because ranks look wrong
- `dry_run_promotion_forbidden`: true

## Operational monitoring

`universe_size`, `scoreable_count`, `coverage_rate`, `feature_missingness`, `identity_failures`, `provider_failures`, `artifact_hash_verification`, `score_distribution`, `rank_concentration`

## Drift

- `mode`: `descriptive_non_adaptive`
- `alert`: `SHADOW_DATA_DRIFT_ALERT`
- `auto_action`: none; no retrain/recalibrate

## No adaptation and challenger governance

- `no_adaptation_rule`: true
- `challenger_governance`: any future challenger requires a separate contract/model generation/evaluation program and may not use the consumed WP8 holdout as its promotion benchmark
