"""Historical universe membership contract and as-of membership queries.

Universe membership is point-in-time information: a security belongs to a
universe only between its membership start and end. Two integrity rules are
enforced here:

* membership windows must be internally consistent (end >= start, valid_from
  and valid_to both parseable when present);
* membership must never be inferred from present-day survival. If the underlying
  membership/delisting history is incomplete, the universe is reported as
  ``PARTIAL`` and the limitation is recorded explicitly rather than implied as
  safe.

Because no complete historical index-membership and delisting source is wired up
in WP2, this module does NOT certify survivorship-safe universes. It provides the
contract, the queries and the honest status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .availability import to_utc_timestamp

UNIVERSE_STATUS_READY = "PIT_READY"
UNIVERSE_STATUS_PARTIAL = "PARTIAL"
UNIVERSE_STATUS_BLOCKED = "BLOCKED"

KNOWN_LIMITATION_NO_DELISTING_SOURCE = (
    "no complete historical index-membership or delisting source is wired up; "
    "survivorship cannot be eliminated and membership is only as complete as the "
    "supplied table"
)


class UniverseError(RuntimeError):
    """Raised for malformed membership records or unusable queries."""


@dataclass(frozen=True)
class UniverseMembership:
    """One security's membership in one universe over a validity window."""

    universe_id: str
    security_id: str
    ticker: str
    membership_start: str
    membership_end: str = None
    valid_from: str = None
    valid_to: str = None

    def to_dict(self):
        return {
            "universe_id": self.universe_id,
            "security_id": self.security_id,
            "ticker": self.ticker,
            "membership_start": self.membership_start,
            "membership_end": self.membership_end,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


@dataclass(frozen=True)
class UniverseStatus:
    """Honest certification status for one universe."""

    universe_id: str
    status: str
    pit_status: str
    survivorship_safe: bool
    limitations: tuple = ()
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "universe_id": self.universe_id,
            "status": self.status,
            "pit_status": self.pit_status,
            "survivorship_safe": self.survivorship_safe,
            "limitations": list(self.limitations),
            "evidence": dict(self.evidence),
        }


def _validate_window(start, end, label):
    parsed_start = to_utc_timestamp(start)
    parsed_end = to_utc_timestamp(end) if end is not None else None
    if parsed_start is None:
        raise UniverseError("%s requires a parseable start" % label)
    if end is not None and parsed_end is None:
        raise UniverseError("%s end %r is unparseable" % (label, end))
    if parsed_end is not None and parsed_end < parsed_start:
        raise UniverseError("%s end precedes its start" % label)
    return parsed_start, parsed_end


class UniverseTable:
    """In-memory point-in-time universe membership table."""

    def __init__(self, universe_id, memberships=None):
        self.universe_id = universe_id
        self._memberships = []
        for membership in memberships or []:
            self.add(membership)

    @property
    def memberships(self):
        return list(self._memberships)

    def add(self, membership):
        if not isinstance(membership, UniverseMembership):
            membership = UniverseMembership(**membership)
        if membership.universe_id != self.universe_id:
            raise UniverseError(
                "membership universe_id %r does not match table universe_id %r"
                % (membership.universe_id, self.universe_id)
            )
        _validate_window(membership.membership_start, membership.membership_end, "membership window")
        if membership.valid_from is not None or membership.valid_to is not None:
            _validate_window(membership.valid_from, membership.valid_to, "validity window")
        self._memberships.append(membership)
        return membership

    def members_asof(self, as_of, available_only=True):
        """Security IDs whose membership window contains ``as_of``.

        ``available_only`` additionally requires ``valid_from <= as_of`` so a
        membership revision that only becomes known later is not used early.
        """
        moment = to_utc_timestamp(as_of)
        if moment is None:
            raise UniverseError("as_of must be a parseable timestamp")
        members = []
        for membership in self._memberships:
            start, end = _validate_window(membership.membership_start, membership.membership_end, "membership window")
            if not (start <= moment and (end is None or moment <= end)):
                continue
            if available_only and membership.valid_from is not None:
                valid_start, _valid_end = _validate_window(membership.valid_from, membership.valid_to, "validity window")
                if valid_start > moment:
                    continue
            members.append(membership.security_id)
        return sorted(set(members))

    def is_member(self, security_id, as_of, available_only=True):
        return security_id in set(self.members_asof(as_of, available_only=available_only))

    def survivorship_check(self, as_of):
        """Diagnostics that expose likely survivorship conditioning.

        Reports members that have an open-ended membership (never marked as
        having ended) and the count of entries vs exits before ``as_of``. An
        all-open-ended history is a red flag that the table only lists today's
        constituents.
        """
        moment = to_utc_timestamp(as_of)
        if moment is None:
            raise UniverseError("as_of must be a parseable timestamp")
        open_ended = []
        ended = []
        for membership in self._memberships:
            start, end = _validate_window(membership.membership_start, membership.membership_end, "membership window")
            if start > moment:
                continue
            if end is None:
                open_ended.append(membership.security_id)
            else:
                ended.append(membership.security_id)
        return {
            "as_of": moment.isoformat(),
            "member_count": len(self.members_asof(moment)),
            "open_ended_members": sorted(set(open_ended)),
            "ended_members": sorted(set(ended)),
            "exposes_possible_survivorship_bias": len(ended) == 0,
        }

    def to_frame(self):
        return pd.DataFrame([membership.to_dict() for membership in self._memberships])

    def status(self):
        """Return the honest status for this universe.

        A table with no recorded exits cannot demonstrate that delisted/removed
        securities were retained; it is reported as ``PARTIAL`` with the reason.
        """
        exits = [membership for membership in self._memberships if membership.membership_end is not None]
        limitations = []
        evidence = {
            "membership_rows": len(self._memberships),
            "securities": len({membership.security_id for membership in self._memberships}),
            "rows_with_recorded_exit": len(exits),
        }
        survivorship_safe = False
        if not exits and self._memberships:
            limitations.append("every membership is open-ended; delisted/removed securities may be absent")
        limitations.append(KNOWN_LIMITATION_NO_DELISTING_SOURCE)
        return UniverseStatus(
            universe_id=self.universe_id,
            status=UNIVERSE_STATUS_PARTIAL,
            pit_status="partially_point_in_time",
            survivorship_safe=survivorship_safe,
            limitations=tuple(limitations),
            evidence=evidence,
        )


def build_universe_from_frame(universe_id, frame, ticker_col="ticker", security_col="security_id",
                              start_col="membership_start", end_col="membership_end",
                              valid_from_col="valid_from", valid_to_col="valid_to"):
    """Build a :class:`UniverseTable` from a DataFrame of membership rows."""
    table = UniverseTable(universe_id)
    for row in frame.to_dict(orient="records"):
        table.add(
            UniverseMembership(
                universe_id=universe_id,
                security_id=row[security_col],
                ticker=row[ticker_col],
                membership_start=row[start_col],
                membership_end=row.get(end_col),
                valid_from=row.get(valid_from_col),
                valid_to=row.get(valid_to_col),
            )
        )
    return table
