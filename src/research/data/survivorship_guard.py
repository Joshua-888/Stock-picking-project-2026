"""Survivorship adversarial guardrail (WP2B, mission section 22).

Reusable checks that FAIL when a historical dataset is conditioned on present-day
survival. They are deliberately adversarial: each one encodes a specific way
survivorship bias sneaks back in.

Detected failure modes
----------------------
1. only today's constituents are used historically (every membership open-ended);
2. a known future entrant appears before its membership effective date;
3. a known historical constituent vanishes from all dates because it later
   failed/delisted;
4. a delisted company is silently excluded because the price provider cannot see
   it today (historical member has no price coverage).

These are checks, not certifications: passing means no *detected* survivorship
conditioning. The absence of a complete delisted source still BLOCKS a
survivorship-safe certification (see ``universe`` and the source decision doc).

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .availability import to_utc_timestamp


class SurvivorshipViolation(AssertionError):
    """Raised when a survivorship-bias failure mode is detected."""


@dataclass(frozen=True)
class GuardrailFinding:
    """One finding with the check that produced it and the evidence."""

    check: str
    message: str
    evidence: dict = None

    def to_dict(self):
        return {"check": self.check, "message": self.message, "evidence": dict(self.evidence or {})}


def check_universe_has_exits(membership_table):
    """Check 1: a universe with zero recorded exits is present-day conditioned.

    A history that lists today's constituents and nothing else has no way to
    express that a company left the universe, so it cannot be survivorship-safe.
    """
    memberships = membership_table.memberships
    if not memberships:
        return GuardrailFinding(
            check="universe_has_exits",
            message="membership table is empty",
            evidence={"membership_rows": 0},
        )
    exits = [item for item in memberships if item.membership_end is not None]
    if not exits:
        return GuardrailFinding(
            check="universe_has_exits",
            message="every membership is open-ended; universe is present-day conditioned",
            evidence={"membership_rows": len(memberships), "exits": 0},
        )
    return None


def check_no_pre_effective_membership(membership_table, security_id, effective_date):
    """Check 2: a security must NOT be a member before its effective join date.

    Catches a future entrant leaking backwards into earlier cross-sections.
    """
    moment = to_utc_timestamp(effective_date)
    if moment is None:
        raise SurvivorshipViolation("effective_date must be parseable")
    day = moment.strftime("%Y-%m-%d")
    if membership_table.is_member(security_id, day, available_only=False):
        starts = [
            item.membership_start
            for item in membership_table.memberships
            if item.security_id == security_id
        ]
        if not any(to_utc_timestamp(start) == moment for start in starts):
            return GuardrailFinding(
                check="no_pre_effective_membership",
                message="%s is a member on %s before its effective join date" % (security_id, day),
                evidence={"security_id": security_id, "date": day, "starts": starts},
            )
    return None


def check_constituent_retained_across_history(membership_table, security_id, as_of_dates):
    """Check 3: a historical constituent must not vanish from every date.

    If a security was documented as a member at some date but is absent from all
    queried cross-sections (e.g. because it later failed), that is survivorship
    erasure. ``as_of_dates`` are the dates that must still contain it.
    """
    present = [
        day for day in as_of_dates
        if membership_table.is_member(security_id, day, available_only=False)
    ]
    if not present:
        return GuardrailFinding(
            check="constituent_retained_across_history",
            message="%s is absent from every queried date; likely erased by later failure" % security_id,
            evidence={"security_id": security_id, "queried_dates": list(as_of_dates)},
        )
    return None


def check_historical_member_has_prices(membership_table, security_id, price_frame, as_of, price_col="raw_close"):
    """Check 4: a historical member must have price coverage, not be dropped.

    Detects the case where a delisted name is a member but the price provider
    (which only sees live tickers) returns nothing, so the row disappears.
    """
    moment = to_utc_timestamp(as_of)
    if moment is None:
        raise SurvivorshipViolation("as_of must be parseable")
    if not membership_table.is_member(security_id, as_of, available_only=False):
        return None
    if price_frame is None or price_frame.empty:
        return GuardrailFinding(
            check="historical_member_has_prices",
            message="%s is a member at %s but has no price rows" % (security_id, as_of),
            evidence={"security_id": security_id, "as_of": str(as_of)},
        )
    if price_col in price_frame.columns and price_frame[price_col].notna().sum() == 0:
        return GuardrailFinding(
            check="historical_member_has_prices",
            message="%s is a member at %s but every price is missing" % (security_id, as_of),
            evidence={"security_id": security_id, "as_of": str(as_of), "rows": int(len(price_frame))},
        )
    return None


def run_guardrails(membership_table, price_frame=None, known_entrants=None,
                   known_constituents=None, as_of=None, price_col="raw_close"):
    """Run every guardrail; returns a list of findings (empty means clean).

    ``known_entrants`` is ``[(security_id, effective_date), ...]`` for joins that
    must not appear early. ``known_constituents`` is ``[(security_id, [dates]),
    ...]`` for names documented historically that must not vanish.
    """
    findings = []
    finding = check_universe_has_exits(membership_table)
    if finding is not None:
        findings.append(finding)
    for security_id, effective_date in known_entrants or []:
        finding = check_no_pre_effective_membership(membership_table, security_id, effective_date)
        if finding is not None:
            findings.append(finding)
    for security_id, dates in known_constituents or []:
        finding = check_constituent_retained_across_history(membership_table, security_id, dates)
        if finding is not None:
            findings.append(finding)
    if price_frame is not None and as_of is not None:
        for membership in membership_table.memberships:
            finding = check_historical_member_has_prices(
                membership_table, membership.security_id, price_frame, as_of, price_col=price_col
            )
            if finding is not None:
                findings.append(finding)
                break
    return findings


def assert_survivorship_clean(membership_table, **kwargs):
    """Raise :class:`SurvivorshipViolation` when any guardrail finding exists."""
    findings = run_guardrails(membership_table, **kwargs)
    if findings:
        raise SurvivorshipViolation(
            "; ".join(item.message for item in findings)
        )
    return True


__all__ = [
    "SurvivorshipViolation",
    "GuardrailFinding",
    "check_universe_has_exits",
    "check_no_pre_effective_membership",
    "check_constituent_retained_across_history",
    "check_historical_member_has_prices",
    "run_guardrails",
    "assert_survivorship_clean",
]
