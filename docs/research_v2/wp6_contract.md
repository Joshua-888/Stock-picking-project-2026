# WP6 Model Research Contract (FROZEN)

This policy is declared and frozen **before any model result is read**. Nothing in
this document is tuned from a metric. Code of record: `src/research/modeling/contract.py`.

## Scope guard

WP6 is **development-period model research only**. The locked holdout
(`feature_asof >= 2022-01-01`) and its labels are **never read**. WP7 (nested
walk-forward validation) is **not** started. No final production model is fitted.

## Targets

- Regression target: `future_12m_excess_return`.
- Classification target: `outperform_12m` (== `future_12m_excess_return > 0`).

## Eligible rows

- Development only: `feature_asof < 2021-01-01` (embargo cutoff).
- `target_observable == True`; censored labels excluded from supervised training.
- Unique `(security_id, feature_asof)`. Duplicate keys carrying an observable label
  raise; censored duplicates (e.g. AGN 2007) are reported and excluded.

## Feature universe

- The WP5 23-candidate universe minus the three **explicit, frozen exclusions**:
  `abnormal_volume` (no volume column; coverage 0.0), `beta` (coverage 0.0),
  `x12_appreciation_proxy` (X12_PROXY, never X12). Exclusions are never resurrected.
- 20 eligible features result.

## Temporal policy

- Monthly cross-sections; **expanding** training window; walk-forward forward-validation.
- 12-month purge: a training row is legal only when `target_end <= validation_start`
  **and** `target_known_at <= model_date`.
- Window geometry is computed deterministically from the observed months:
  first validation begins after `min_train_months (60) + horizon (12)` months;
  consecutive non-overlapping `validation_window_months (24)`-month windows;
  a window must span a full 12-month horizon. Minimum `min_folds = 4`.
- Exact per-fold boundaries and counts are recorded in the artifact.

## Preprocessing (train-fit / apply-forward only)

Operations: median imputation, missing indicator, percentile clipping (1%–99%),
robust scaling, z-scoring, within-date cross-sectional ranking. **Every fitted
parameter comes from the training slice of the fold** and is applied unchanged to
training and validation. No global/full-sample transform. No random splits.

## Feature strategies A–F (selection computed in TRAIN only)

| Key | Strategy | Basis |
| --- | --- | --- |
| A | `A_all_eligible` | all 20 eligible features |
| B | `B_coverage_qualified` | train coverage >= 0.35 (frozen) |
| C | `C_regularization_only` | all eligible; complexity controlled by regularisation |
| D | `D_correlation_constrained` | drop pairwise train rank-corr > 0.70 (frozen) |
| E | `E_ic_top_k` | top 5 by abs **train** mean monthly rank IC |
| F | `F_economic_family_representatives` | one per family by **highest train coverage** (never target) |

Selection stability is measured per strategy across folds. WP5 categories are
descriptive context only, never a permanent full-sample feature set.

## Model families and frozen grids

Regression: baseline mean, equal-weight rank composite, OLS, Ridge
`alpha ∈ {0.1,1,10,100}`, Lasso `alpha ∈ {5e-4,1e-3,5e-3,1e-2}`, ElasticNet
`alpha ∈ {1e-3,1e-2} × l1_ratio ∈ {0.25,0.75}`.
Classification: base-rate, Logistic `C ∈ {0.01,0.1,1,10}`.
Nonlinear (both tasks): RandomForest / ExtraTrees `depth ∈ {4,8} × leaf ∈ {20,50}`
(n=300), HistGradientBoosting `depth ∈ {3,6} × leaf ∈ {20,50} × lr ∈ {0.05,0.1}`
(iter=300). Small grids only; **no Optuna / adaptive search**.

## Metrics

Primary (regression): cross-sectional monthly Spearman rank IC, mean monthly IC,
ICIR, positive-IC-month fraction. Secondary: MAE, RMSE.
Classification: ROC-AUC, PR-AUC, log loss, Brier, calibration (reliability bins +
intercept/slope).
Level-return portfolio diagnostics (top/bottom quintile & decile observed excess
return, top-minus-bottom spread) are reported **separately** and are not predictive
metrics.

## Inference

HAC/Newey-West (`lags = 12`, PRIMARY) and circular moving-block bootstrap
(`block = 6`, `iterations = 1000`, seed `20260926`, SECONDARY) on the **monthly**
metric series. Paired monthly model comparisons across folds. No pooled-row IID claims.

## Experiment id semantics

Content-addressed via `src/research/ids.experiment_id` over a canonical binding:
dataset / target / feature ids, canonical holdout id, frozen contract, frozen
model registry, committing Git commit, seeds (`model_seed=20260930`,
`bootstrap_seed=20260926`, `placebo_seed=20260926`).

## Research categories (NOT a ranking)

`PROMISING` / `INCONCLUSIVE` / `UNSTABLE` / `NO_EVIDENCE` / `REJECTED`.

Predeclared rules: `REJECTED` if any placebo check stops; `NO_EVIDENCE` if fewer
than 12 months, |mean metric| < 0.01, or fewer than 2 folds; `UNSTABLE` if fewer
than 75% of folds agree on direction; `PROMISING` only if repeatable across
multiple folds **and** strictly beats the declared baseline on the primary metric
with HAC `p < 0.05`; else `INCONCLUSIVE`.

No arbitrary numeric composite score is defined. It is an acceptable outcome that
**no model family shows robust development-period signal**.
