# V2 Data Readiness Policy

Status: ACTIVE (WP2C, supersedes the WP2A/WP2B "perfect or blocked" standard)
Scope: the V2 equity research dataset (prices, historical universe, security
identity, benchmark, point-in-time fundamentals) that WP3+ consume.

## 1. Purpose

The project does not require institutional-grade (CRSP-like) source data before
proceeding. It requires data that is **research-grade, point-in-time safe,
survivorship-controlled where practicable, with all material source limitations
measured and explicitly represented**.

A completed research dataset is certified as:

- `RESEARCH_READY_WITH_LIMITATIONS` — usable for the intended statistical
  research, with measured limitations recorded; or
- `BLOCKED` — only for a concrete methodological or software defect.

## 2. What is acceptable

- Real data only; `synthetic_data_status = none`. No fabricated rows.
- Point-in-time availability: features use only information knowable at the
  prediction instant; filing/publication dates drive fundamental availability,
  not fiscal-period ends.
- Historical universe reconstructed from effective-dated membership events,
  never by back-filling present-day constituents into earlier periods.
- Historical removals / delisted names included wherever the source supports
  them.
- Source coverage that is imperfect but **quantified** (e.g. ~96% price
  completeness, ~60% removed-constituent terminal observability).
- Explicit, persisted limitations in the `DatasetManifest` and companion
  artifacts.
- Reproducible provenance: pinned layer versions, source + config + dataset
  fingerprints, code commit, security-master version, universe version.
- Full test suite passing.

## 3. What is NOT acceptable (hard defects → BLOCKED)

1. **Known future leakage** — look-ahead in features, targets, membership,
   corporate actions, or normalization fit on future data.
2. **Silent identity corruption** — a security resolves to the wrong company
   (e.g. a reused ticker), or is dropped without an explicit record. Every
   current constituent must end `RESOLVED` or
   `EXPLICITLY_UNRESOLVED_WITH_REASON`; never silently absent.
3. **Systematic incorrect labels without detection** — e.g. returning a wrong
   forward return, or treating an unobservable outcome as 0 / −100% / last
   price.
4. **Irreproducibility** — a result that cannot be rebuilt from its dataset
   version, experiment/feature-set/model version and commit.
5. **Invalid statistical inference** — a defect that would make the intended
   WP5–WP7 inference unsound even after limitations are documented.

Absence of CRSP-equivalent completeness is **not** a hard defect.

## 4. Treatment of censoring

EODHD provides historical delisted prices but **no delisting/terminal returns**.
This is a documented limitation, not a blocker.

For each observation row the pipeline records:
`target_observable`, `target_censored`, `target_censor_reason`,
`terminal_price_observable`, `terminal_status`, `membership_exit_observed`.

If the required future outcome cannot be measured reliably (security disappears
before the horizon with no valid terminal value):

- `target_observable = false`, `target_censored = true`, with an explicit reason.
- Never assume zero return, −100%, carry-forward, or a silent drop.

Censored observations may remain in the historical panel but **must not be used
as supervised labels** for models that require a known 12-month outcome. WP3+
targets consume only `target_observable = true` rows.

Because censoring is likely non-random, WP2C persists a censoring/
selection-bias diagnostic (overall %, by year, by exit type, by trailing
performance, among removed constituents; observable vs censored compared on
pre-T information only). This measures the limitation; it does not remove it.

## 5. Treatment of missing membership events

Where a free membership source (Wikipedia S&P 500 change table) lacks a known
add/remove event (historical examples: Bear Stearns, Enron, Circuit City,
Washington Mutual, old General Motors), the gap is recorded as an explicit
**membership-source limitation**. Membership dates are **never fabricated**, and
present-day constituents are never back-projected to fill the gap.

Assumed/pre-source membership starts are flagged (`research_eligible = false`,
`unverifiable_before = true`) and excluded from research cross-sections.

## 6. When a stronger commercial source becomes genuinely necessary

A stronger source (Sharadar SEP/SF1, CRSP, or equivalent) becomes required only
if the intended research cannot be conducted soundly with the measured
limitations, specifically if:

- censoring removes so much of the removed-firm sample that survival/
  bankruptcy effects cannot be estimated at all (not merely estimated with a
  documented gap);
- benchmark-relative 12-month labels cannot be formed for a scientifically
  meaningful window;
- terminal-return bias is shown to reverse the sign of the studied effect; or
- identity/universe integrity cannot be made non-silent at acceptable cost.

Until then, limitations are documented and carried, and the reduced removed-firm
label sample is accepted as an explicit selection-bias limitation.

## 7. Certification

A dataset is certified only after:

1. Data Integrity Engineer approval (family-level PIT verdicts),
2. Research Ops reproducibility verdict `REPRODUCIBLE`,
3. Quant Auditor independent red-team result (must still fail on §3 hard
   defects; must not fail on §2/§4/§5 documented limitations),
4. deterministic `DatasetManifest` with source/universe/security-master
   versions, fingerprints, censoring statistics and known limitations.

Final status is exactly `RESEARCH_READY_WITH_LIMITATIONS` or `BLOCKED`.
