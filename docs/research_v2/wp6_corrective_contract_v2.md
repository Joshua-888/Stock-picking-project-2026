# WP6_CORRECTIVE_CONTRACT_V2

Frozen BEFORE any corrected WP6 scientific result is inspected or classified.
Authored by the Quant Research Supervisor. Implementation-only Builder must not
mutate these decisions; any methodological change requires a NEW contract + NEW
experiment (never a post-hoc mutation).

Supersedes, for scientific promotion only, the corrected-evidence contract of
`experiment_f7864f37998f` (which stood on `WP6_CONTRACT_VERSION =
v2_wp6_model_research_v1`). The old experiment and its artifacts are preserved
and its scientific promotion conclusions are WITHDRAWN (AUDIT_FAIL).

## 1. Corrective contract version

- `WP6_CORRECTIVE_CONTRACT_VERSION = "v2_wp6_model_research_v2_corrective"`
- Every WP6 module version string changes so the content-addressed experiment id
  is new and can never overwrite `experiment_f7864f37998f` / `experiment_05ddc3721b4a`.

## 2. Upstream data identity (unchanged, certified)

- dataset_id      = `dataset_35a278e17c13`
- panel_version   = `553ac17bf5d4d63f`
- target_id       = `target_set_d2bb16610bce`
- target_version  = `56d0f670bdf1b47c`
- feature_set_id  = `feature_set_4f7b43726310`
- WP5 experiment  = `experiment_f985287c1315`
- WP4 corrective  = `experiment_834a7e60f13c`
- Canonical holdout resolved ONLY via `provenance/holdout/index.json`
  (expected `holdout_7ce54e933e16`; stale ids in immutable artifacts are ignored).

## 3. Development period, embargo, purge

- Development rows: `feature_asof < 2021-01-01`.
- Locked holdout: `feature_asof >= 2022-01-01`; NEVER read, counted or scored.
- Embargo band (purged, neither train nor validation):
  `2021-01-01 <= feature_asof < 2022-01-01`.
- Train eligibility at a model date: `target_known_at <= model_date` AND
  `target_end <= model_date` (frozen 12-month purge), AND `feature_asof < embargo_cutoff`.
- Fold geometry UNCHANGED (passed WFV): expanding train, min_train_months=60,
  validation_window_months=24, horizon_months=12, min_folds=4.

## 4. Frozen model configuration universe (NOT shrunk)

Reuse the exact frozen registry grids of v1 (regression / classification /
nonlinear-for-both-tasks) x feature strategies A-F. Expected 312 configurations
on the full grid. No configuration may be removed after seeing corrected
outcomes. Defensible pre-results changes, if any, must be listed OLD vs NEW with
justification.

Feature strategies (TRAIN-window only): A_all_eligible, B_coverage_qualified,
C_regularization_only, D_correlation_constrained, E_ic_top_k,
F_economic_family_representatives.
Preprocessing profiles: RANK_SPEC (composite baseline only), VALUE_SPEC (all else).

## 5. Defect #1 — matched, predeclared placebo reference

- Every negative control names a PREDECLARED matched real configuration by
  deterministic identity: `model_config_id`, `placebo_config_id`,
  `matched_real_config_id`.
- Matching is EXACT on (model family, task, feature strategy, preprocessing
  strategy, hyperparameters, temporal folds, prediction target, evaluation
  metric, aggregation). Only the placebo perturbation may differ.
- FORBIDDEN anywhere in the promotion/control path:
  `max(values, key=abs)`, best-performing reference, largest-|IC| reference,
  best model observed after evaluation. Reference selection must not change when
  unrelated measured results change (proven by test).
- The matched real metric is that exact config's own pooled primary metric.

## 6. Defect #2 — correct metric nulls (explicit comparator per metric)

- ROC-AUC: null `H0: mean(AUC_t - 0.5) = 0`. Skill series = `AUC_t - 0.5`.
  Inference (HAC, primary) is performed on the SKILL series.
- Cross-sectional Spearman IC (regression): null `H0: mean(IC_t) = 0` (already correct).
- PR-AUC: comparator = frozen base prevalence (per-month), never zero.
- Accuracy: comparator = base rate / frozen benchmark, never zero.
- Brier: comparator = frozen base-rate probability benchmark `mean((base_rate - y)^2)`.
- Log loss: comparator = frozen base-rate log loss `-mean(y ln b + (1-y) ln(1-b))`,
  `b` = frozen train base rate.
- Every metric serializes its comparator/null definition string.

## 7. Defect #3 — explicit control-evaluation objects + real STOP semantics

Each mandatory control returns an object with at least:
`control_id, control_type, expected_behavior, observed_behavior,
matched_real_config_id, real_metric, control_metric, failure_threshold,
failure_condition, passed, stop_required, reason`.

Semantics:
- PASS  = control behaves as expected under no signal / no leakage.
- FAIL  = control behaves like genuine predictive signal beyond frozen tolerance.
- STOP  = at least one MANDATORY control FAILs a frozen materiality rule.

`overall_stop` is derived transparently from these objects.
PASS, FAIL and STOP branches must all be reachable (proven by fixtures).
No `stop = bool(violations == 0)`-style tautology.

## 8. Defect #4 — genuine future-availability (PIT/leakage) guard

Replace the tautological `required = ts + 365d; violates = required > ts` guard with
field-based checks over real modelling fields. For every eligible model row verify:
- `feature_asof <= model_asof` (feature availability at/ before the model instant);
- target eligibility: `target_known_at <= model_asof` for any TRAINING row;
- no future feature row is joined into a training/validation observation;
- no validation-period statistics enter training transforms (train-fit only);
- no future target information enters preprocessing/selection.
If a feature lacks required availability evidence it is classified by the existing
feature-availability rules, never silently passed.
MANDATORY FIXTURES: an injected leakage row with `feature_asof > model_asof` MUST be
detected; a clean PIT fixture MUST pass.

## 9. Defect #5 — frozen shuffle methodology + anomaly handling

- Shuffle method FROZEN before rerun: within-date CROSS-SECTIONAL permutation of the
  TRAINING target only, per `modeling_month`, deterministic seeded RNG
  (`placebo_seed`), preserving each date's marginal label distribution and the
  panel's date structure. Rationale: it destroys the cross-sectional relationship
  being tested while preserving panel structure so the null is meaningful. Method
  chosen on methodology, NOT on outcome.
- The matched real vs matched placebo comparison must be reported verbatim.
- If the correctly constructed placebo remains >= the matched real magnitude
  (beyond the frozen floor), the affected model(s) MUST be blocked (STOP) and NOT
  promoted. That is a valid scientific result and must NOT be suppressed.

## 10. Defect #6 — model-layer multiple-testing (BH-FDR)

Hypothesis families FROZEN before results, defined by scientific question, not by
p-values, and not fragmented to reduce penalty:
- F1 `regression_predictive_skill`: all regression-task configs
  (regression families + nonlinear models enumerated for regression).
- F2 `classification_predictive_skill`: all classification-task configs
  (logistic + nonlinear models enumerated for classification).

Per family record: `family_id, scientific_question, metric, null,
number_of_hypotheses, raw_p_definition, adjustment_method, alpha,
included_config_ids`.
Method: Benjamini-Hochberg FDR. alpha = 0.05.
Per configuration report `raw_p`, `q_value`, `fdr_rejected`.
PROMISING requires `fdr_rejected == True`; `raw_p < 0.05` alone is NEVER sufficient.
The FULL tested set is retained in the ledger (no hidden trials, no family shrinking).

## 11. Defect #7 — eliminate result-dependent selection

Audit the entire WP6 evaluation path for `max by metric`, `min by p-value`,
`best model / feature set / fold / subperiod / threshold`, or any winner chosen after
observing results. Descriptive ranking tables may be ordered by performance; but all
inferential/promotion reference choices are frozen independently of results.
Every searched configuration is recorded in the ledger.

## 12. Defect #8 — robustness checks executed or removed

Decision: EXECUTE all three (they are part of the WP6 scientific contract and are
implementable without redesign):
- `subperiod_stability`: frozen half-split of development months (first vs second
  half); report both halves and sign flip; no favorable-subperiod selection.
- `complexity_comparison`: nonlinear vs simple linear on the primary metric;
  complexity is justified ONLY with HAC-significant improvement surviving FDR.
- `reduced_feature_comparison`: predeclared reduced strategies (F family subset;
  E ic_top_k) vs A_all_eligible on the primary metric.
Plus `feature_selection_stability` across training windows (already computed).
All outputs serialized; any claim of a check being performed requires its serialized
output. No ghost validations.

## 13. Correct research categories (no weighted composite)

Allowed: PROMISING, INCONCLUSIVE, UNSTABLE, NO_EVIDENCE, REJECTED.
PROMISING requires ALL of:
1. correct metric null (IC vs 0; AUC-skill vs 0 i.e. AUC vs 0.5);
2. beats the predeclared baseline on the primary metric;
3. `fdr_rejected` within its frozen family (BH q <= 0.05);
4. temporal stability: >= 2 folds AND fold agreement >= 0.75;
5. no mandatory negative-control STOP affecting it;
6. leakage guard passes for its rows;
7. holdout exclusion intact.
A model is NEVER forced into PROMISING. `PROMISING = 0` is an acceptable, valid outcome.

## 14. Seeds (frozen)

model_seed=20260930, bootstrap_seed=20260926, placebo_seed=20260926.

## 15. Stop conditions

If any MANDATORY control FAILs its frozen materiality rule (including the matched
shuffled-target control, the noise-feature control, and the field-based
future-availability guard), WP6 STOPs for the affected models/families and those
models cannot be PROMISING. STOP is reported, never overridden by intuition.

## 16. Scope guard

WP6 remains DEVELOPMENT-PERIOD model research. No production model is fitted. The
locked holdout is never read. WP7 is NOT started. Human review is required before WP7.
