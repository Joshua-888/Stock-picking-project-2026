"""WP5 PHASE 4 corporate-action integrity tests.

These tests pin the corrected behaviour reproduced from the pre-certified
Data Integrity Engineer report
(``artifacts/research/wp5_correction/corporate_action_anomaly_report.json``):

* HNZ -> UNIT_MISINTERPRETATION_X100 must NOT collapse the cumulative factor;
* DISCA/DISCK 2014-08-07 spinoff-as-dividend -> WRONG_EVENT_TYPE must be skipped;
* an unresolvable dividend must never be silently compounded;
* the raw provider amount stays visible;
* identical rows are de-duplicated before factors are built;
* the point-in-time action cutoff is unchanged (a future action never reaches an
  earlier as-of).

Fixtures are explicitly synthetic and exist only to exercise the validation
logic; they never enter a research artifact.
"""

import pandas as pd

from src.research.data import action_validation as av
from src.research.data import pit_prices, wp2b_eodhd as wp2b


def _price_frame(rows):
    return pd.DataFrame(
        [{"ticker": "AAA", "trade_date": day, "raw_close": close} for day, close in rows]
    )


def _dividend(day, amount, security_id="AAA"):
    return {"security_id": security_id, "kind": "dividend", "effective_date": day, "amount": amount}


# ── constants stay in lockstep with pit_prices ───────────────────────────────

def test_status_constants_and_kinds_match_pit_prices():
    assert av.SPLIT == pit_prices.SPLIT and av.DIVIDEND == pit_prices.DIVIDEND
    assert av.VALID in av.STATUSES and av.UNRESOLVED in av.STATUSES


# ── HNZ regression: an x100 dividend is corrected, never compounded away ─────

def test_hnz_x100_dividend_is_normalized_not_compounded():
    # HNZ-style: amount ~= 0.9 * close (ratio 0.715..1.255 in the report).
    close = 50.0
    status, normalized, method, _reason = av.classify_dividend(45.0, close)
    assert status == av.UNIT_MISINTERPRETATION_X100
    assert method == av.NORMALIZATION_DIVIDE_BY_100
    assert normalized == 45.0 / 100.0
    factor = av.dividend_factor(normalized, close, status)
    assert factor is not None and factor > 0.95, "a corrected dividend must barely scale prices"


def test_hnz_cumulative_factor_does_not_collapse():
    # HNZ-style: 25 quarterly dividends recorded ~x100 too large (amount ~= close).
    # Every one is corrected (/100) so each factor is ~0.99, never ~0.
    from src.research.data.pit_prices import CorporateAction

    acts = []
    raw_by_date = {}
    for quarter in range(25):
        year = 2010 + quarter // 4
        month = (quarter % 4) * 3 + 3
        day = "%d-%02d-15" % (year, month)
        raw_by_date[day] = 50.0
        acts.append(CorporateAction(ticker="HNZ", kind=pit_prices.DIVIDEND, effective_date=day, amount=49.0))
    dates, cums, dropped = wp2b._cumulative_action_factors(acts, raw_by_date)
    assert cums, "validated dividends must still produce factors"
    assert cums[-1] > 0.5, "HNZ-style cumulative factor must not collapse to ~0"
    assert dropped == [], "all 25 are correctable, none dropped"


# ── DISCA regression: spinoff recorded as dividend is skipped ────────────────

def test_disca_spinoff_as_dividend_is_wrong_event_type_and_skipped():
    # DISCA 2014-08-07: amount 39.00441, ex-date close ~40.97 (ratio ~0.95) on a
    # date that ALSO carries the spinoff+split -> not a cash dividend.
    status, normalized, _method, reason = av.classify_dividend(39.00441, 40.97, coincident_split=True)
    assert status == av.WRONG_EVENT_TYPE
    assert normalized is None
    assert av.dividend_factor(normalized, 40.97, status) is None
    assert "distribution" in reason


def test_split_and_normal_dividend_sharing_a_date_is_valid():
    # Many issuers split and pay a small cash dividend on the same day.
    status, _normalized, _method, _reason = av.classify_dividend(0.40, 40.0, coincident_split=True)
    assert status == av.VALID


# ── implausible AND unresolvable amounts ─────────────────────────────────────

def test_dividend_implausible_and_unresolvable_is_unresolved():
    # ratio 3.0, /100 = 0.03 which would be plausible -> treated as x100 (not dropped).
    status, _n, _m, _r = av.classify_dividend(150.0, 50.0)
    assert status == av.UNIT_MISINTERPRETATION_X100
    # A ratio whose /100 is still implausible is UNRESOLVED and must not apply.
    tiny_close = 0.5
    status2, normalized2, _m2, _r2 = av.classify_dividend(45.0, tiny_close)
    assert status2 == av.UNRESOLVED and normalized2 is None


def test_impossible_dividend_factor_is_refused():
    # amount >= 95% of close == an economically impossible cash dividend.
    assert av.dividend_factor(49.0, 50.0, av.VALID) is None


# ── raw provider value retained, normalized value explicit ───────────────────

def test_raw_amount_is_preserved_and_normalized_added():
    prices = _price_frame([("2014-05-09", 50.0)])
    # a genuine x100 case: raw 50.0 is 100x a plausible 0.50 dividend on a 50.0 close
    actions = pd.DataFrame([_dividend("2014-05-09", 50.0)])
    annotated, report = av.annotate_actions_frame(actions, prices)
    row = annotated.iloc[0]
    assert row["amount"] == 50.0, "provider amount is never overwritten"
    assert row[av.COLUMN_RAW_AMOUNT] == 50.0
    assert row[av.COLUMN_NORMALIZED_AMOUNT] == 0.5
    assert row[av.COLUMN_VALIDATION_STATUS] == av.UNIT_MISINTERPRETATION_X100
    assert report["status_counts"][av.UNIT_MISINTERPRETATION_X100] == 1


# ── de-duplication ───────────────────────────────────────────────────────────

def test_duplicate_rows_are_deduplicated():
    prices = _price_frame([("2020-07-01", 50.0)])
    actions = pd.DataFrame([_dividend("2020-07-01", 0.5), _dividend("2020-07-01", 0.5)])
    annotated, report = av.annotate_actions_frame(actions, prices)
    assert len(annotated) == 1
    assert report["duplicates_removed"] == 1


def test_validated_factors_dedupe_and_skip():
    lookup = lambda day: 40.97 if day == "2014-08-07" else None
    records = [
        # a split makes the same-date dividend a mis-typed distribution
        {"kind": "split", "effective_date": "2014-08-07", "numerator": 1.957, "denominator": 1.0},
        {"kind": "dividend", "effective_date": "2014-08-07", "amount": 39.00441},
        {"kind": "dividend", "effective_date": "2014-08-07", "amount": 39.00441},
    ]
    prepared, dropped = av.validated_action_factors(records, lookup)
    assert [day for day, _factor in prepared] == ["2014-08-07"]
    assert len(prepared) == 1, "only the split factor survives"
    # one duplicate + one skipped wrong-event-type
    statuses = sorted(item["status"] for item in dropped)
    assert av.DUPLICATE in statuses and av.WRONG_EVENT_TYPE in statuses


# ── PIT cutoff is unchanged ──────────────────────────────────────────────────

def test_future_action_never_reaches_earlier_asof():
    raw = _price_frame([("2020-01-02", 100.0), ("2020-06-01", 50.0)])
    split = pit_prices.CorporateAction(ticker="AAA", kind=pit_prices.SPLIT,
                                       effective_date="2020-05-01", numerator=2.0, denominator=1.0)
    at_jan = pit_prices.build_pit_adjusted_frame(raw, [split], asof="2020-01-31")
    later = pit_prices.build_pit_adjusted_frame(raw, [split], asof="2020-12-31")
    jan_value = at_jan.set_index("trade_date")["adjusted_close_pit"]["2020-01-02"]
    later_value = later.set_index("trade_date")["adjusted_close_pit"]["2020-01-02"]
    assert jan_value == 100.0, "pre-asof price is not adjusted by a future split"
    assert later_value == 50.0
