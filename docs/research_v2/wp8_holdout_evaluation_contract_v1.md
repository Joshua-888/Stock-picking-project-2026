# WP8 Holdout Evaluation Contract V1

- Contract version: `HOLDOUT_EVALUATION_CONTRACT_V1`
- Frozen stage: `WP8_PRE_ACCESS`
- Created before any locked-holdout content access: `true`
- Canonical WP7 generation: `generation_de0f9bbd0bec`
- Canonical WP7 contract: `WP7_VALIDATION_CONTRACT_V5`
- Canonical final candidate freeze: `freeze_e62eac30df40`
- Canonical locked holdout: `holdout_7ce54e933e16`

## Scope

This contract freezes the exact one-shot evaluation procedure for the final
candidate against the locked holdout. It does not authorize content access by
itself; access requires the separate pre-access audit/authorization gates.

## Row definition

A row is eligible when all hold:

- `feature_asof >= 2022-01-01` and `feature_asof <= 2025-08-31`
- `target_observable == true`
- the target `outperform_12m` is present, finite, and binary
- the required frozen final features are available under point-in-time rules

Censored or missing outcomes are excluded. They are never labeled class 0 and
never imputed.

## Model rule

The evaluator uses only the static frozen model bundle produced by
`freeze_e62eac30df40`. No fitting, preprocessing fit, feature selection,
calibration fitting/selection, threshold selection, or retraining occurs in the
holdout evaluation path.

## Primary metric and inference

- Metric: cross-sectional monthly ROC-AUC skill (`AUC - 0.5`), equal monthly weighting.
- Minimum monthly observations: `30`.
- Hypothesis: `H0: mean monthly AUC skill <= 0` vs `H1: > 0`.
- Inference: deterministic one-sided Newey-West/Bartlett HAC t-test with lag `12`.
- `p = normal_sf(t)` for `t > 0`; for `t <= 0`, `p = 1`.
- If fewer than two valid monthly values or invalid HAC variance, the result is
  blocked/invalid; it is never silently passed.

## Verdict rules

| Verdict | Condition |
| --- | --- |
| `HOLDOUT_SIGNAL_CONFIRMED` | `mean_auc_skill > 0` and `one_sided_hac_p < 0.05` |
| `HOLDOUT_POSITIVE_BUT_INCONCLUSIVE` | `mean_auc_skill > 0` and `one_sided_hac_p >= 0.05` |
| `HOLDOUT_NOT_CONFIRMED` | `mean_auc_skill <= 0` |

## Secondary metrics

Predeclared evaluation diagnostics only:

- pooled ROC-AUC
- pooled PR-AUC
- PR-AUC benchmark (final base rate)
- Brier score
- Brier skill vs final base rate
- log loss
- log loss skill vs final base rate
- calibration slope/intercept diagnostics

Optional quintile hit-rate/economic descriptive metrics may be reported. No
multiple cutoff search is permitted.

## One-shot access rule

Exactly one real evaluation is authorized. A persistent access ledger prevents
second execution. If `STARTED` or `COMPLETED` or `INVALIDATED` is observed at the
next entry, the evaluator refuses without touching data.
