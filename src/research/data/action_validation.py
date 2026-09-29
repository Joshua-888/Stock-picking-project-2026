"""Corporate-action integrity validation and evidence-based normalization (WP5 PHASE 4).

The EODHD provider publishes three distinct defects that silently corrupt a
point-in-time adjusted price series if they are compounded verbatim:

* ``UNIT_MISINTERPRETATION_X100`` - a subset of dividend ``unadjustedValue``
  entries carry an extra 100x, so ``(close - amount) / close`` collapses toward
  0 and every earlier point-in-time price collapses with it;
* ``WRONG_EVENT_TYPE`` - a spin-off/distribution is recorded as a dividend on a
  date that ALSO carries a split, so it is not a cash dividend at all;
* ``DUPLICATE`` - identical action rows repeated inside one silver snapshot.

This module classifies each action against the price series it belongs to. It
never overwrites the provider value: ``raw_action_amount`` keeps the original and
``normalized_action_amount`` carries the evidence-based correction. A dividend is
only divided by 100 when its raw ``amount / close_on_ex`` ratio is implausible AND
its ``/100`` ratio IS plausible; nothing is divided by 100 blindly. Actions that
cannot be resolved, or that are mis-typed events, are flagged so no factor
consumer can compound them silently.

This module performs no feature engineering, no model fitting and no scoring. It
records and reconstructs provenance; it fabricates no price, no delisting return
and no amount.
"""

from __future__ import annotations

import bisect
import math

import pandas as pd

# Action kinds (kept as plain strings so this module has no import cycle with
# ``pit_prices``; a regression test asserts equality with the pit_prices constants).
SPLIT = "split"
DIVIDEND = "dividend"

# Per-action validation status.
VALID = "VALID"
UNIT_MISINTERPRETATION_X100 = "UNIT_MISINTERPRETATION_X100"
WRONG_EVENT_TYPE = "WRONG_EVENT_TYPE"
DUPLICATE = "DUPLICATE"
UNRESOLVED = "UNRESOLVED"
STATUSES = (VALID, UNIT_MISINTERPRETATION_X100, WRONG_EVENT_TYPE, DUPLICATE, UNRESOLVED)

# Documented, tested plausibility bands (see tests/test_research_v2_wp5_action_validation.py).
# Evidence behind the bands (Data Integrity Engineer report
# artifacts/research/wp5_correction/corporate_action_anomaly_report.json):
#   * normal cash dividends (KO/JNJ/PG/IBM/CVX controls) are <= ~1.5% of the
#     ex-date close;
#   * a legitimately large / one-off SPECIAL dividend can be larger, but no genuine
#     cash dividend pays out more than half the share price in cash -- the existing
#     project contract treats a 25% special dividend as valid
#     (tests/test_research_v2b_universe.py::test_special_dividend_adjusts_like_a_large_dividend),
#     so the x100 band must start strictly above that;
#   * every CONFIRMED x100 dividend (HNZ) has amount >= 0.715 * close and every
#     confirmed wrong-type distribution (DISCA/DISCK 2014-08-07) has amount ~= close.
# Hence: <= 5% is an ordinary dividend; (5%, 50%] is a large/special dividend; an
# amount above 50% of the close has no genuine cash-dividend explanation and is
# treated as a 100x unit error ONLY where dividing by 100 yields a plausible (<= 5%)
# dividend -- never blindly.
DIVIDEND_PLAUSIBLE_MAX_RATIO = 0.05
DIVIDEND_LARGE_VALID_MAX_RATIO = 0.50
DIVIDEND_BORDERLINE_MAX_RATIO = DIVIDEND_LARGE_VALID_MAX_RATIO
X100_CORRECTED_MAX_RATIO = 0.05

# A dividend factor at or below this floor (amount >= 95% of the ex-date close) is
# economically impossible as a cash dividend; an unclassified factor that low is
# refused rather than silently compounded into an adjusted price.
MIN_VALID_DIVIDEND_FACTOR = 0.05

NORMALIZATION_NONE = "none"
NORMALIZATION_DIVIDE_BY_100 = "divide_by_100_unit_misinterpretation"

# Statuses whose factor must never be applied to an adjusted price.
SKIPPED_STATUSES = (WRONG_EVENT_TYPE, UNRESOLVED)

ACTION_SIGNATURE_COLUMNS = (
    "security_id", "kind", "effective_date", "amount", "numerator", "denominator",
)

COLUMN_RAW_AMOUNT = "raw_action_amount"
COLUMN_NORMALIZED_AMOUNT = "normalized_action_amount"
COLUMN_NORMALIZATION_METHOD = "normalization_method"
COLUMN_VALIDATION_STATUS = "action_validation_status"
COLUMN_VALIDATION_REASON = "action_validation_reason"
VALIDATION_COLUMNS = (
    COLUMN_RAW_AMOUNT, COLUMN_NORMALIZED_AMOUNT, COLUMN_NORMALIZATION_METHOD,
    COLUMN_VALIDATION_STATUS, COLUMN_VALIDATION_REASON,
)


def _clean_amount(value):
    """Finite float or None (never a NaN / infinite amount)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_blank(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value) in ("", "None", "nan", "NaT")


def classify_dividend(amount, close_on_ex, coincident_split=False):
    """Classify one dividend against its ex-date close.

    Returns ``(status, normalized_amount, normalization_method, reason)``. The
    normalized amount is the value a factor may use: the raw amount for a valid
    dividend, ``amount / 100`` only for an evidence-backed unit error, and
    ``None`` for an action that must not be applied at all.
    """
    value = _clean_amount(amount)
    close = _clean_amount(close_on_ex)
    if value is None or close is None or close <= 0.0:
        return UNRESOLVED, None, NORMALIZATION_NONE, "dividend amount or ex-date close unavailable"
    if value <= 0.0:
        return UNRESOLVED, None, NORMALIZATION_NONE, "dividend amount is not positive"
    ratio = value / close
    # A dividend recorded on the SAME effective date as a split is only treated as
    # a mis-typed distribution (spin-off) when its amount is implausibly large
    # relative to the ex-date close; a split and a normal cash dividend legitimately
    # share a date for many issuers, so the ratio test guards against false flagging.
    if coincident_split and ratio > DIVIDEND_BORDERLINE_MAX_RATIO:
        return (WRONG_EVENT_TYPE, None, NORMALIZATION_NONE,
                "distribution recorded as a dividend: amount/close %.6f on a split effective date" % ratio)
    if ratio <= DIVIDEND_PLAUSIBLE_MAX_RATIO:
        return VALID, value, NORMALIZATION_NONE, "amount/close %.6f <= %.2f" % (
            ratio, DIVIDEND_PLAUSIBLE_MAX_RATIO)
    if ratio <= DIVIDEND_BORDERLINE_MAX_RATIO:
        return VALID, value, NORMALIZATION_NONE, "amount/close %.6f in the borderline band (<= %.2f)" % (
            ratio, DIVIDEND_BORDERLINE_MAX_RATIO)
    corrected = value / 100.0
    if corrected / close <= X100_CORRECTED_MAX_RATIO:
        return (UNIT_MISINTERPRETATION_X100, corrected, NORMALIZATION_DIVIDE_BY_100,
                "amount/close %.6f implausible; /100 ratio %.6f <= %.2f" % (
                    ratio, corrected / close, X100_CORRECTED_MAX_RATIO))
    return (UNRESOLVED, None, NORMALIZATION_NONE,
            "amount/close %.6f implausible and /100 ratio %.6f still implausible" % (
                ratio, corrected / close))


def dividend_factor(amount, close_on_ex, validation_status=None):
    """Back-adjustment factor for one dividend, or None when it must not apply.

    Returns ``None`` for a skipped status, for missing inputs, and for a factor at
    or below :data:`MIN_VALID_DIVIDEND_FACTOR` (an economically impossible cash
    dividend that would otherwise collapse the cumulative factor).
    """
    if validation_status in SKIPPED_STATUSES:
        return None
    value = _clean_amount(amount)
    close = _clean_amount(close_on_ex)
    if value is None or close is None or close <= 0.0:
        return None
    factor = (close - value) / close
    if not math.isfinite(factor) or factor <= MIN_VALID_DIVIDEND_FACTOR:
        return None
    return factor


def _day_text(value):
    """ISO ``YYYY-MM-DD`` text for a date-like value, or None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text in ("", "None", "nan", "NaT"):
        return None
    return text[:10]


def _get_field(record, name):
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def validated_action_factors(actions, close_lookup):
    """De-duplicated, validated ``[(effective_date, factor)]`` for ONE security.

    This is the single correction path every factor consumer uses. It:

    * de-duplicates identical rows on
      ``(kind, effective_date, amount, numerator, denominator)``;
    * SKIPS ``WRONG_EVENT_TYPE`` / ``UNRESOLVED`` actions entirely so a mis-typed
      distribution or an unresolvable amount is never compounded;
    * applies the evidence-backed ``/100`` unit correction only where
      :func:`classify_dividend` documents it (never blindly);
    * refuses any dividend factor at or below :data:`MIN_VALID_DIVIDEND_FACTOR`
      so an economically impossible cash dividend cannot collapse the series.

    ``close_lookup(day_text)`` returns the raw close on/before an effective date
    (or None) and is supplied by the caller because each consumer keys its price
    series differently. The returned list is sorted by effective date.
    """
    records = list(actions or [])
    split_dates = set()
    for record in records:
        if str(_get_field(record, "kind") or "").strip().lower() == SPLIT:
            day = _day_text(_get_field(record, "effective_date"))
            if day is not None:
                split_dates.add(day)
    prepared = []
    dropped = []
    seen = set()
    for record in records:
        kind = str(_get_field(record, "kind") or "").strip().lower()
        day = _day_text(_get_field(record, "effective_date"))
        if day is None:
            continue
        amount = _clean_amount(_get_field(record, "amount"))
        numerator = _clean_amount(_get_field(record, "numerator"))
        denominator = _clean_amount(_get_field(record, "denominator"))
        signature = (kind, day, amount, numerator, denominator)
        if signature in seen:
            dropped.append({"kind": kind, "effective_date": day, "amount": amount,
                            "status": DUPLICATE, "reason": "identical action row repeated"})
            continue
        seen.add(signature)
        if kind == SPLIT:
            if not numerator or not denominator:
                continue
            prepared.append((day, float(denominator) / float(numerator)))
        elif kind == DIVIDEND:
            close_on_ex = close_lookup(day)
            status, effective_amount, _method, reason = classify_dividend(
                amount, close_on_ex, day in split_dates)
            factor = dividend_factor(effective_amount, close_on_ex, status)
            if factor is not None:
                prepared.append((day, factor))
            else:
                dropped.append({"kind": kind, "effective_date": day, "amount": amount,
                                "status": status, "reason": reason})
    prepared.sort(key=lambda item: item[0])
    return prepared, dropped


def action_signature(row):
    """Content signature used to de-duplicate identical action rows."""
    return tuple(_clean_amount(row.get(column)) if column in ("amount", "numerator", "denominator")
                 else (None if _is_blank(row.get(column)) else str(row.get(column)))
                 for column in ACTION_SIGNATURE_COLUMNS)


def _price_index(prices):
    """Per-security (sorted day list, close list) from a raw price frame."""
    index = {}
    if prices is None or len(prices) == 0:
        return index
    key = "security_id" if "security_id" in prices.columns else (
        "ticker" if "ticker" in prices.columns else None)
    if key is None:
        return index
    frame = prices.loc[:, [key, "trade_date", "raw_close"]].rename(columns={key: "security_id"}).copy()
    frame["security_id"] = frame["security_id"].astype(str)
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    frame = frame.dropna(subset=["raw_close"])
    for security_id, chunk in frame.groupby("security_id", sort=True):
        chunk = chunk.sort_values("trade_date", kind="mergesort")
        days = chunk["trade_date"].tolist()
        closes = [float(value) for value in chunk["raw_close"].tolist()]
        index[str(security_id)] = (days, closes)
    return index


def close_on_or_before(index, security_id, day):
    """Raw close on ``day`` or the nearest PRIOR session (None when none exists)."""
    entry = index.get(str(security_id))
    if not entry:
        return None
    days, closes = entry
    position = bisect.bisect_right(days, day) - 1
    if position < 0:
        return None
    return closes[position]


def annotate_actions_frame(actions, prices, security_id=None):
    """Annotate an action frame with raw/normalized amounts and validation status.

    Inputs are never mutated. Returns ``(annotated_frame, report)`` where the frame
    keeps the provider ``amount`` untouched and adds the validation columns; the
    report carries the class counts, the de-duplication count and every flagged
    action (with its raw value visible). De-duplication is on
    :data:`ACTION_SIGNATURE_COLUMNS`, keeping the first occurrence.
    """
    frame = actions.copy() if actions is not None else pd.DataFrame()
    for column in ("security_id", "kind", "effective_date", "numerator", "denominator", "amount"):
        if column not in frame.columns:
            frame[column] = None
    if security_id is not None:
        frame["security_id"] = security_id
    frame["security_id"] = frame["security_id"].astype(str)
    frame["kind"] = frame["kind"].astype(str).str.strip().str.lower()
    frame["effective_date"] = frame["effective_date"].astype(str).str.slice(0, 10)
    frame = frame.sort_values(["security_id", "effective_date", "kind"], kind="mergesort").reset_index(drop=True)

    signature = list(ACTION_SIGNATURE_COLUMNS)
    duplicate_mask = frame.duplicated(subset=signature, keep="first").to_numpy()
    duplicates = frame.loc[duplicate_mask].copy()
    unique = frame.loc[~duplicate_mask].copy()

    index = _price_index(prices)
    split_rows = unique.loc[unique["kind"] == SPLIT]
    split_keys = set(zip(split_rows["security_id"], split_rows["effective_date"]))

    raw_amounts, normalized, methods, statuses, reasons = [], [], [], [], []
    for row in unique.itertuples(index=False):
        kind = row.kind
        raw_amounts.append(_clean_amount(row.amount))
        if kind == SPLIT:
            numerator = _clean_amount(row.numerator)
            denominator = _clean_amount(row.denominator)
            if not numerator or not denominator:
                statuses.append(UNRESOLVED)
                reasons.append("split is missing a non-zero numerator/denominator ratio")
            else:
                statuses.append(VALID)
                reasons.append("split ratio denominator/numerator")
            normalized.append(None)
            methods.append(NORMALIZATION_NONE)
        elif kind == DIVIDEND:
            close = close_on_or_before(index, row.security_id, row.effective_date)
            coincident = (str(row.security_id), str(row.effective_date)) in split_keys
            status, effective, method, reason = classify_dividend(_clean_amount(row.amount), close, coincident)
            statuses.append(status)
            normalized.append(effective)
            methods.append(method)
            reasons.append(reason)
        else:
            statuses.append(UNRESOLVED)
            normalized.append(None)
            methods.append(NORMALIZATION_NONE)
            reasons.append("unsupported action kind %r" % (kind,))

    unique[COLUMN_RAW_AMOUNT] = raw_amounts
    unique[COLUMN_NORMALIZED_AMOUNT] = normalized
    unique[COLUMN_NORMALIZATION_METHOD] = methods
    unique[COLUMN_VALIDATION_STATUS] = statuses
    unique[COLUMN_VALIDATION_REASON] = reasons

    flagged_mask = unique[COLUMN_VALIDATION_STATUS].isin(
        [UNIT_MISINTERPRETATION_X100, WRONG_EVENT_TYPE, UNRESOLVED])
    flagged = unique.loc[flagged_mask]
    flagged_records = [
        {
            "security_id": str(row["security_id"]),
            "kind": str(row["kind"]),
            "effective_date": str(row["effective_date"]),
            "raw_action_amount": row[COLUMN_RAW_AMOUNT],
            "normalized_action_amount": row[COLUMN_NORMALIZED_AMOUNT],
            "normalization_method": row[COLUMN_NORMALIZATION_METHOD],
            "action_validation_status": row[COLUMN_VALIDATION_STATUS],
            "action_validation_reason": row[COLUMN_VALIDATION_REASON],
        }
        for row in flagged.to_dict("records")
    ]
    status_counts = {name: int((unique[COLUMN_VALIDATION_STATUS] == name).sum()) for name in STATUSES}
    status_counts[DUPLICATE] = int(len(duplicates))
    report = {
        "actions_in": int(len(frame)),
        "actions_out": int(len(unique)),
        "duplicates_removed": int(len(duplicates)),
        "status_counts": status_counts,
        "flagged_action_count": int(len(flagged)),
        "flagged_security_count": int(flagged["security_id"].nunique()) if len(flagged) else 0,
        "thresholds": {
            "dividend_plausible_max_ratio": DIVIDEND_PLAUSIBLE_MAX_RATIO,
            "dividend_borderline_max_ratio": DIVIDEND_BORDERLINE_MAX_RATIO,
            "x100_corrected_max_ratio": X100_CORRECTED_MAX_RATIO,
            "min_valid_dividend_factor": MIN_VALID_DIVIDEND_FACTOR,
        },
        "flagged_actions": flagged_records,
        "note": (
            "raw provider amounts are preserved; only evidence-backed /100 corrections "
            "are applied, and mis-typed/unresolved actions are skipped rather than compounded"
        ),
    }
    return unique, report
