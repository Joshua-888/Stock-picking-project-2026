# Historical Universe / Delisted-Price Source Decision (WP2A item 4)

## Question

Can historical S&P 500 membership **and** historical price history for the
securities that left the index be reconstructed from free data sources, so that a
survivorship-safe point-in-time universe and its price features can be certified?

## Finding (recon evidence)

| Capability | Best free source found | Result |
| --- | --- | --- |
| Historical index membership (effective dates, added/removed pairs) | Wikipedia *Historical components of the S&P 500* changes table (~409 change rows, 1976-2026) | Usable as **PARTIAL / informational** membership evidence |
| Historical prices for currently listed names | Yahoo chart API (with a descriptive User-Agent) | Works; raw close + split/dividend events |
| Historical prices for delisted / acquired names | Yahoo chart API | **HTTP 404 for essentially all of them** (LEH, LEHMQ, ENRNQ, ATVI, VMW, ANTM, XLNX, CERN, ABMD, SGEN, CTXS, TWTR, SIVBQ, WAMUQ, BSC, CFC, MEE, DNDN, YHOO) |
| Alternative free price APIs | Stooq (JS bot wall, blocks automation); Nasdaq API (`Symbol not exists` for delisted) | Unusable |

Some post-delisting OTC stubs return HTTP 200 (e.g. SBNY/FRCB) but these are
illiquid residual tickers, **not** continuous histories of the original security;
they cannot stand in for the pre-delisting series.

## Exact missing capability

> A free, machine-readable source of **historical daily prices for delisted,
> acquired and otherwise removed index constituents**, covering each security from
> a date before its index entry through and including its exit.

Membership alone is insufficient: without the removed names' price history, any
historical cross-section that includes the losers is impossible to build from
free data, and any cross-section that excludes them is survivorship-biased by
construction.

## Source options

### Best technically suitable source (recommended for V3)

**S&P Dow Jones Indices official constituent history**, or an equivalent licensed
vendor with full delisting coverage and effective-date history (e.g. CRSP,
Compustat, Refinitiv, Bloomberg). These provide:

* point-in-time index composition with official effective dates;
* security-level price history that survives delisting, merger and bankruptcy;
* permanent security identifiers decoupled from tickers.

Cost: not free. Indicative range for a small research project is roughly
USD 5k-30k+/yr depending on vendor, universe and history depth; exact quotes must
be obtained from the vendor. Cost is **unknown/indicative**, not verified here.

### Acceptable lower-cost / free alternative

**Wikipedia *Historical components of the S&P 500*** for membership evidence:

* free and reasonably complete for effective-date add/remove pairs;
* crowd-maintained, so it may contain errors, gaps and inconsistent conventions
  (e.g. ticker reuse, mergers recorded as removals);
* carries **no** price history and therefore cannot by itself unblock the price
  side of the gate.

This table may be parsed and stored as **PARTIAL** membership evidence, clearly
labelled as informational. It **must not** be wired as certified universe
auto-membership.

## What remains blocked without the paid/delisting-complete source

1. **Survivorship-safe universe** -- not certifiable.
2. **Delisted-name price features** -- unavailable from free sources.
3. **Delisted-name forward returns / labels** -- unavailable, so a genuinely
   unbiased historical training set cannot be assembled.
4. Consequently, the minimal research dataset's universe and price families
   cannot receive a survivorship-safe certification.

## Decision

* Use the **bounded current-constituent sample** (`config/tickers.yaml`, capped)
  for pipeline construction and PIT-mechanics testing only.
* Mark its price manifest `pit_status = partially_point_in_time`,
  `survivorship_safe = False`, with the survivorship limitation recorded.
* Report the **historical-universe family as `BLOCKED`** on the external free-data
  dependency above.
* Do **not** fabricate a universe or silently drop the limitation.
