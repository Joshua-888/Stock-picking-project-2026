# WP7 Validation Contract v4

- Schema: `wp7_validation_contract_v4`
- Contract version: `WP7_VALIDATION_CONTRACT_V4`
- Supersedes: `WP7_VALIDATION_CONTRACT_V3`
- `created_before_results: true`
- Frozen at stage: `WP7_PRESTART`

This contract is frozen **before any WP7 results/interpreting code exists**. It is not tuned
from any WP7 metric. The machine-readable canonical copy is
`provenance/wp7/validation_contract_v4.json`.

Supersession reason: v3 inner `min_train_months=48` was not implementable with actual
point-in-time `target_end` values: actual safe inner fold counts were `[3,4,6,7]` but v3
froze `[4,5,6,7]`; `outer_test_1` excluded inner folds had `max_target_end` 2017-01-03 and
2018-01-02. Corrected before any WP7 results to inner `min_train_months=37`, keeping raw
counts `[5,6,7,8]` and safe counts `[4,5,6,7]` unchanged per `WFV_WP7_CONTRACT_FAIL`.

## Certified WP6 inputs

- Dataset: `dataset_35a278e17c13`
- Target set: `target_set_d2bb16610bce`
- Feature set: `feature_set_4f7b43726310`
- WP6 experiment: `experiment_ee434a07a25d`
- WP6 producing commit: `21eec0d907df8fc8ec191c10ba9b056bd6dfa006`
- WP4 corrective experiment: `experiment_834a7e60f13c`
- WP5 experiment: `experiment_f985287c1315`

## Eligible task

- Eligible task: **classification ONLY** (`outperform_12m`).
- Regression is **EXCLUDED** because the WP6 mandatory `shuffled_target_regression` control
triggered `overall_stop` for regression.

## Feature universe

The same 20 features listed in WP6 `binding.json` `feature_universe`:

```
earnings_yield
current_pe
price_to_book
price_to_sales
free_cash_flow_yield
dividend_yield
roa
roe
gross_margin
operating_margin
roic
five_year_eps_growth
five_year_revenue_growth
eps_growth_acceleration
revenue_growth_acceleration
six_month_momentum
twelve_month_momentum
five_year_price_gain
debt_to_equity
market_cap
```

## Preprocessing

- Spec: `VALUE_SPEC`.
- All fitted transforms must be **train-only** (fit on the permitted training slice only).

## Feature selection

- Frozen WP6 strategy semantics are reused **train-only**; **no new full-sample selection**.
- Eligible strategies only:
  - `B_coverage_qualified`
  - `F_economic_family_representatives`
- Implementation of record: `src/research/modeling/features.py`.

## Eligible model set

FROZEN list of the 9 WP6 classification PROMISING configurations plus loss baseline
`baseline_base_rate`. Each config uses `VALUE_SPEC` preprocessing.

1. `modelcfg_00c68894e319` hist_gradient_boosting F lr=0.05 depth=3 iter=300 leaf=50
2. `modelcfg_040a0c85df2c` extra_trees F depth=4 leaf=50 n=300
3. `modelcfg_158499379e80` extra_trees B depth=4 leaf=50 n=300
4. `modelcfg_1f0278c43c28` extra_trees F depth=8 leaf=50 n=300
5. `modelcfg_5dcf351cd952` extra_trees F depth=4 leaf=20 n=300
6. `modelcfg_99e145df5bc8` extra_trees B depth=4 leaf=20 n=300
7. `modelcfg_b8151deebdc7` extra_trees B depth=8 leaf=50 n=300
8. `modelcfg_e981f5ee119e` extra_trees B depth=8 leaf=20 n=300
9. `modelcfg_fecc1b87e28b` hist_gradient_boosting F lr=0.05 depth=6 iter=300 leaf=50

Baseline: `baseline_base_rate` (train base-rate predictor, `VALUE_SPEC`, no hyperparameters).

## Inner walk-forward design

- Chronological **expanding folds only**.
- `min_train_months = 37`
- `validation_window_months = 12`
- `horizon_months = 12`
- `purge_months = 12`
- `embargo_months = 12`
- `min_folds = 4`
- Forbidden: `KFold`, `train_test_split`, `ShuffleSplit`.

## Outer walk-forward design

- Nested chronological outer folds.
- Each outer fold uses its **prior data as outer_train**.
- On outer_train, perform **inner-forward model_selection_candidate and calibration
  selection** with purge/embargo, then freeze both.
- Evaluate **ONLY** on the 12-month chronological outer_test.
- Outer_test windows are **4 chronological 12-month windows within development data**:

| window_id | start | end | start_index | end_index |
| --- | --- | --- | --- | --- |
| `outer_test_1` | 2017-01 | 2017-12 | 120 | 131 |
| `outer_test_2` | 2018-01 | 2018-12 | 132 | 143 |
| `outer_test_3` | 2019-01 | 2019-12 | 144 | 155 |
| `outer_test_4` | 2020-01 | 2020-12 | 156 | 167 |

- `purge_before_outer_test_rule`: before each `outer_test`, exclude from
  model/calibration selection any inner validation window whose maximum `target_end` is
  `>= outer_test_start`. Selection uses only inner windows whose labels are fully
  observable before `outer_test_start`.
- Safe inner fold counts by outer fold: `[4, 5, 6, 7]` (outer_train months
  120/132/144/156 with `validation_window_months=12`).
- Raw inner fold counts before last-window exclusion: `[5, 6, 7, 8]`.

- The exact outer_test windows must be derived and serialized from the same deterministic
  fold geometry **before any evaluation**.

## Purge / embargo

- Apply 12-month purge before **every** inner and outer validation/test boundary.
- Before each `outer_test`, exclude from model/calibration selection any inner validation
  window whose maximum `target_end >= outer_test_start`; selection uses only inner windows
  whose labels are fully observable before `outer_test_start`.
- No overlapping 12-month labels across train/test.

## Model selection rule

- Deterministic **highest mean inner validation AUC-skill**
  (mean AUC − 0.5 across inner validation months, adjusted for fold direction/agreement)
  among eligible configs.
- No post-hoc max-abs selection of reference.
- No p-value-only selection.
- Tie-break: smaller hyperparameter complexity, then deterministic `config_id` order.

## Calibration

- Scope: train/inner-only.
- Frozen candidates: `none`, `platt_scaling`, `isotonic_regression`.
- Select the candidate with best mean inner Brier skill versus the train base-rate
  benchmark on inner validation folds.
- Tie-break: deterministic earliest candidate order.
- Never calibrate on outer test or holdout.

## Regression metrics (declared only)

Declared for possible diagnostics if any later regression candidate arises:
monthly cross-sectional Spearman IC, mean IC, ICIR, positive-month fraction, HAC
inference, MAE, RMSE.

## Classification metrics

- ROC-AUC (inferential benchmark 0.5, skill AUC − 0.5)
- PR-AUC vs train prevalence
- Brier vs train base-rate benchmark
- Log-loss vs train base-rate benchmark
- Calibration slope/intercept
- Positive-month/fold fraction

## Multiple-testing policy

- Candidate set is a frozen 9-model selection.
- Report raw and BH-FDR q-values across the frozen classification family.
- Final selected model performance is reported as a **selection-adjusted nested estimate**.
- It must **NOT** be claimed as unbiased holdout evidence.

## Robustness requirements

- Outer-fold stability
- Subperiod stability
- Model selection stability
- Feature selection stability
- Hyperparameter stability
- Calibration stability
- Complexity vs simple baseline
- Negative controls (noise-feature and shuffled-target controls with reachable STOP)

## Final model freeze conditions

Freeze: selected config, calibration, preprocessing, feature strategy, seeds, producing
commit. **Do not retune after looking at outer test or holdout.**

## Seeds

- `model_seed = 20260930`
- `bootstrap_seed = 20260926`
- `placebo_seed = 20260926`

## Holdout procedure (future)

- `canonical_holdout_id = holdout_7ce54e933e16`
- Exactly one locked-holdout evaluation, only after explicit
  `HOLDOUT_EVALUATION_AUTHORIZATION`.
- No holdout access in the WP7 pre-holdout phase.

## Stop conditions

If any mandatory negative control fails → `WP7_PRE_HOLDOUT_BLOCKED`; do not access holdout.
