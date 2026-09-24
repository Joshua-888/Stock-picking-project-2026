# Historical Universe & Delisted-Price Source Decision (WP2B)

Scope: the external-data requirement behind the survivorship-safe historical
universe and the delisted-name price/forward-return families. This document
records what was investigated and what is blocked. It supersedes the WP2A note
`universe_source_decision.md` (kept for history).

Evidence bundle: `artifacts/research/wp2b/` (`source_matrix.csv`,
`universe_coverage.*`, `price_coverage.*`, `identifier_failures.*`,
`corporate_action_checks.*`, `sample_size_feasibility.*`, `source_probes/`).

No marketing claims. Costs are `PRICE_NOT_VERIFIED` unless a vendor page gave an
exact public number, which is labelled "vendor claim".

---

## 1. Scientific requirement

To answer the project's primary question without survivorship bias we need, for a
point-in-time S&P 500 cross-section at date `T`:

1. **Membership**: which securities were actually in the index at `T`,
   reconstructed from dated change events (effective dates), never from today's
   constituent list.
2. **Price history** for *every* member, **including names that later left the
   index** (delisted, acquired, bankrupt, renamed), continuously from before their
   entry through their exit.
3. **Corporate-action data** (splits, dividends, mergers, spin-offs, delisting
   returns) with effective dates, so that features at `T` use only actions that had
   already occurred.
4. **Permanent identifiers** so tickers (reused, renamed) never conflate
   economically different securities.

Requirement (2) is decisive: excluding removed names and their (typically bad)
future returns biases every downstream label upward. Membership alone cannot
satisfy the gate.

## 2. Sources investigated

Free / no-credential: Yahoo chart API (yfinance), stockanalysis.com, Stooq,
Nasdaq public API, Alpha Vantage demo, Tiingo, FMP, Twelve Data, Marketstack,
investing.com, Wikipedia *Historical components of the S&P 500*.

Commercial (need subscription/credentials): EODHD, Sharadar (via Nasdaq Data
Link `SEP`/`SF1`), Norgate Data, Polygon/Massive.

Institutional: CRSP, Compustat, LSEG/Refinitiv, Bloomberg, FactSet, S&P DJI.

## 3. Empirical probes (free sources)

Recorded verbatim in `source_probes/`.

### 3.1 Delisted prices — all free price APIs fail

Yahoo chart API with historical `period1/period2` windows returned **HTTP 404 for
essentially every delisted/acquired ticker** (`probe_delisted_dated.txt`):
LEH, ENRNQ, BSC, CFC, WAMUQ, YHOO, TWTR, ATVI, VMW, CERN, XLNX, ANTM, and more
(SIVBQ, ABMD, SGEN, CTXS). It works only for currently listed names.

Other free providers (`probe_free_apis.txt`, `probe_free_delisted_2.txt`):

| Source | Result |
| --- | --- |
| stockanalysis.com | LEHMQ page 404; `/list/delisted-stocks/` 404; only *live* tickers resolve |
| Alpha Vantage | demo key refused for anything but IBM |
| Tiingo | 403 without key |
| FMP | "Invalid API KEY" |
| Twelve Data | 401 (demo key insufficient) |
| Marketstack | 401 |
| Stooq | JS bot wall, blocks automation |
| Nasdaq API | `Symbol not exists` for delisted |
| investing.com | 403 |

Post-delisting OTC stubs (SBNY/FRCB) sometimes return 200, but they are residual
tickers, **not** continuous histories of the original security, so they cannot
substitute for the pre-delisting series.

### 3.2 Membership — free but partial

Wikipedia *Historical components of the S&P 500*: ~409 Added/Removed ticker pairs
with effective dates (1976–2026). Usable as **PARTIAL / informational** membership
evidence only: crowd-maintained, ticker-based, no permanent IDs, no announcement
dates, and **no prices**. It cannot by itself unblock the price side.

### 3.3 Commercial reachability (no credentials present)

`probe_commercial_reach.txt`: Nasdaq Data Link/Sharadar `403`; EODHD `403`;
Norgate page `200` (reachable, but access requires license). SEC EDGAR full-text
`200` (available, but gives filings/fundamentals, not survivorship-free prices).

## 4. Free findings

* Historical **membership** is partly obtainable free (Wikipedia), but is
  ticker-based, effective-date only, and informational.
* Historical **prices for removed securities** are **not** obtainable from any
  free source tested.
* Therefore a free, survivorship-safe historical universe + labels is not
  constructible. This is an **external data dependency**, not a code limitation.

## 5. Paid findings

| Provider | Delisted coverage (claimed) | Universe | Notes |
| --- | --- | --- | --- |
| EODHD | `delisted=1` per exchange; full history to delisting; reliable ~2012+ | add-on | cheapest plausible tier |
| Sharadar (NDL `SEP`/`SF1`) | prices 1998+ incl. delisted; S&P 500 constituents | yes | bundled prices+fundamentals+universe |
| Norgate | 25,222 delisted 1950–2022 | Platinum | index constituents need top tier |
| Polygon/Massive | survivorship-bias-free, delisted retained | add-on | — |

## 6. Coverage comparison (2000→present, survivorship-safe?)

| Provider | Membership | Delisted prices | Corp. actions | Verdict |
| --- | --- | --- | --- | --- |
| Yahoo (free) | no | no | yes (live) | BLOCKED_FOR_UNIVERSE |
| Wikipedia (free) | partial | no | no | PARTIAL |
| EODHD | no | yes | yes | PIT_READY_IF_CREDENTIALED |
| Sharadar/NDL | yes | yes | yes | PIT_READY_IF_CREDENTIALED |
| Norgate | yes | yes | yes | PIT_READY_IF_LICENSED |
| CRSP/Compustat | yes | yes | yes | PIT_READY_IF_LICENSED |

## 7. Identifier comparison

Free price APIs expose only **live tickers** and have no delisted symbol list;
EODHD exposes ticker/exchange but no CIK. SEC EDGAR provides CIK (fundamentals
only). Wikipedia has no permanent ID. Reused tickers cannot be disambiguated by
ticker alone. Mitigation already implemented: `identity_chain.py` maps vendor id
→ `security_id` with share-class separation, issuer merger/successor links, dated
reuse windows, and fails loudly on ambiguity. Status: **PARTIAL** until a
survivorship-free institutional ID crosses the delisted boundary.

## 8. Corporate-action comparison

Yahoo supplies split/dividend price-adjusting events for live names. Delisted
mergers/bankruptcies/delisting returns are only in commercial/institutional sets.
Vendor delisting returns must be preserved separately (`delisting.py`); a missing
final price is **never** treated as a zero return (see
`corporate_action_checks.*`).

## 9. PIT implications

* Membership at `T` must come from **effective historical state at `T`**;
  announcement dates never grant membership (`membership_engine.py`).
* No action with effective date `> T` may change information used at `T`
  (`pit_prices.py`); enforced by tests.
* Forward 12m labels for removed names require their post-exit prices — the exact
  data free sources lack.

## 10. Licensing / access

All viable sources require a paid subscription or institutional license and an
API credential supplied via environment variables. No credential is present in
this environment; adapters therefore raise typed errors without any network call
and never fabricate data.

## 11. Cost

`PRICE_NOT_VERIFIED` except where a vendor page showed a public figure:
EODHD ~USD 19.99–99.99/mo (~199–999/yr, vendor claim); Norgate US ~USD 630/yr
Platinum (vendor claim). Sharadar/NDL, Polygon, CRSP/Compustat: not verified.

## 12. Recommended architectures

* **Tier 1 — Free (current).** Wikipedia membership (PARTIAL) + Yahoo prices for
  **currently listed names only**, strictly for pipeline construction and
  PIT-mechanics testing. **Not** survivorship-safe; cannot certify a research
  dataset.
* **Tier 2 — Cheapest commercial (recommended to unblock).** A single vendor with
  delisted prices + constituent history (Sharadar/NDL `SEP`+`SF1`, or EODHD
  delisted add-on). Bundles membership, prices, actions, and gives a consistent
  identifier surface to join into the existing security master.
* **Tier 3 — Institutional.** CRSP/Compustat (delisting returns, PERMNO) for a
  definitive survivorship-free dataset and benchmark-grade audit.

## 13. Remaining uncertainty

Exact current pricing of Sharadar/NDL, Polygon and CRSP/Compustat; the precise
pre-2012 delisting completeness of EODHD; whether a given vendor's S&P 500
membership table includes announcement dates (needed only if a methodology
requires them). None of these affect the blocker: **no free source provides
delisted-name price history.**

## 14. Exact blocker

> No free, machine-readable source of continuous historical daily prices for
> delisted / acquired / removed S&P 500 constituents exists (verified: HTTP 404
> from Yahoo for all tested delisted tickers; 401/403 from all other free price
> APIs). Without removed-name prices, no survivorship-safe universe, price
> features, or 12-month forward labels can be built from free data.

**Decision:** the historical-universe, delisted-price and delisted-label families
are `BLOCKED`. The abstraction, membership engine, delisting methodology, identity
chain and adversarial survivorship tests are implemented and certified where they
do **not** require delisted prices. The bounded current-constituent sample remains
`survivorship_safe = False`, `pit_status = partially_point_in_time`, for mechanics
only. No certified survivorship-safe dataset was produced.
