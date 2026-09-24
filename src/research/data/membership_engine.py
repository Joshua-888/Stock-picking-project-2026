"""Point-in-time historical universe membership reconstruction (WP2B).

Universe membership is reconstructed from *change events* (adds and removes),
never from a present-day constituent list. Each event carries two distinct dates:

* ``announcement_date`` -- when the change became public;
* ``effective_date``    -- when the membership actually changed.

The two must be different concepts and are compared, never merged. Membership at
instant ``T`` is derived from the *effective* historical state at ``T``:

* an ``add`` event makes a security a member on/after ``effective_date``;
* a ``remove`` event ends membership at ``effective_date``.

Announcement does NOT grant membership early: an announced-but-not-yet-effective
addition is NOT a member. This module never backfills future membership into
earlier dates; reconstruction is strictly forward in effective time.

The result is validated and normalized into the WP2 ``UniverseTable`` contract so
``members_asof``/``is_member`` keep working unchanged.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .availability import to_utc_timestamp
from .universe import UniverseMembership, UniverseTable

ADD = "add"
REMOVE = "remove"
VALID_ACTIONS = (ADD, REMOVE)

EVENT_FIELDS = (
    "security_id",
    "universe_id",
    "action",
    "announcement_date",
    "effective_date",
    "source",
    "source_reference",
)


class MembershipError(RuntimeError):
    """Raised for malformed membership events (fail loudly, never guess)."""


@dataclass(frozen=True)
class MembershipEvent:
    """One universe membership change with distinct announcement/effective dates."""

    security_id: str
    universe_id: str
    action: str
    effective_date: str
    announcement_date: str = None
    source: str = None
    source_reference: str = None

    def __post_init__(self):
        if self.action not in VALID_ACTIONS:
            raise MembershipError("unknown membership action %r" % (self.action,))
        if to_utc_timestamp(self.effective_date) is None:
            raise MembershipError("event needs a parseable effective_date: %r" % (self.effective_date,))
        if self.announcement_date is not None and to_utc_timestamp(self.announcement_date) is None:
            raise MembershipError("event needs a parseable announcement_date: %r" % (self.announcement_date,))

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "universe_id": self.universe_id,
            "action": self.action,
            "announcement_date": self.announcement_date,
            "effective_date": self.effective_date,
            "source": self.source,
            "source_reference": self.source_reference,
        }


def events_from_frame(frame, universe_id=None):
    """Build :class:`MembershipEvent` records from a normalized event frame."""
    events = []
    for row in frame.to_dict(orient="records"):
        events.append(
            MembershipEvent(
                security_id=row["security_id"],
                universe_id=row.get("universe_id", universe_id),
                action=str(row["action"]).strip().lower(),
                effective_date=row["effective_date"],
                announcement_date=row.get("announcement_date"),
                source=row.get("source"),
                source_reference=row.get("source_reference"),
            )
        )
    return events


def validate_events(events, require_distinct_dates=True):
    """Deterministic integrity checks; returns a problem list.

    ``require_distinct_dates`` enforces the WP2B rule that an event where the
    announcement *equals* the effective date is flagged as an inferred/merged
    announcement (some daily vendor tables supply only one date). Set it False
    only when the source genuinely uses one date for both and the limitation is
    already documented (see ``providers.INFERRED_ANNOUNCEMENT``).
    """
    problems = []
    seen = set()
    for event in events:
        key = (event.universe_id, event.security_id, event.action, event.effective_date)
        if key in seen:
            problems.append("duplicate event %s" % (key,))
        seen.add(key)
        if event.announcement_date is None:
            problems.append("event on %s has no announcement date" % event.effective_date)
            continue
        if require_distinct_dates and event.announcement_date == event.effective_date:
            problems.append(
                "event on %s has announcement_date == effective_date (merged/inferred)"
                % event.effective_date
            )
    return problems


def rebuild_membership_windows(events, universe_id, ticker_lookup=None):
    """Reconstruct membership windows from effective-time event order.

    Always forward in effective time: an ``add`` opens a window, the next
    ``remove`` closes it. A ``remove`` without an open window is an integrity
    error (the history is incomplete), reported rather than silently dropped.
    """
    ordered = sorted(
        [event for event in events if event.universe_id == universe_id],
        key=lambda item: (to_utc_timestamp(item.effective_date), item.action, item.security_id),
    )
    open_windows = {}
    windows = []
    problems = []
    for event in ordered:
        if event.action == ADD:
            if event.security_id in open_windows:
                problems.append(
                    "duplicate add for %s at %s while already a member since %s"
                    % (event.security_id, event.effective_date, open_windows[event.security_id].effective_date)
                )
                continue
            open_windows[event.security_id] = event
        else:  # REMOVE
            opened = open_windows.pop(event.security_id, None)
            if opened is None:
                problems.append(
                    "remove for %s at %s has no preceding add (incomplete history)"
                    % (event.security_id, event.effective_date)
                )
                continue
            windows.append((opened, event))
    for opened in open_windows.values():
        windows.append((opened, None))

    memberships = []
    for opened, closed in windows:
        ticker = None
        if ticker_lookup is not None:
            ticker = ticker_lookup(opened.security_id)
        memberships.append(
            UniverseMembership(
                universe_id=universe_id,
                security_id=opened.security_id,
                ticker=ticker,
                membership_start=opened.effective_date,
                membership_end=closed.effective_date if closed is not None else None,
                valid_from=opened.announcement_date,
                valid_to=closed.announcement_date if closed is not None else None,
            )
        )
    memberships.sort(key=lambda item: (item.membership_start, item.security_id))
    return memberships, problems


def build_pit_universe(events, universe_id, ticker_lookup=None, strict=True):
    """Build a :class:`UniverseTable` from change events (PIT reconstruction).

    ``strict`` raises :class:`MembershipError` when reconstruction finds an
    inconsistent event history; otherwise the problems are attached to the table
    as ``reconstruction_problems`` for reporting.
    """
    memberships, problems = rebuild_membership_windows(events, universe_id, ticker_lookup=ticker_lookup)
    if problems and strict:
        raise MembershipError(
            "membership reconstruction found %d problem(s): %s" % (len(problems), "; ".join(problems))
        )
    table = UniverseTable(universe_id)
    for membership in memberships:
        table.add(membership)
    table.reconstruction_problems = tuple(problems)
    table.event_count = len(events)
    return table


def membership_at(events, security_id, universe_id, as_of):
    """True when ``security_id`` is a member of ``universe_id`` at ``as_of``.

    Uses effective dates only; an announced future change does not grant
    membership early. The boundary convention is HALF-OPEN and SHARED with
    :meth:`UniverseTable.members_asof`: an addition is a member on/after its
    ``effective_date`` (``effective_date <= T``), while a removal is NOT a member
    on/after its ``effective_date`` (``T < effective_date``). Deletions therefore
    stop membership on the effective day -- the day before is still a member and
    the effective day is not -- and the two callers must stay in agreement.
    """
    moment = to_utc_timestamp(as_of)
    if moment is None:
        raise MembershipError("as_of must be a parseable timestamp")
    relevant = sorted(
        [event for event in events if event.universe_id == universe_id and event.security_id == security_id],
        key=lambda item: (to_utc_timestamp(item.effective_date), item.action),
    )
    status = False
    for event in relevant:
        stamp = to_utc_timestamp(event.effective_date)
        if stamp is None or stamp > moment:
            continue
        status = event.action == ADD
    return status


__all__ = [
    "ADD",
    "REMOVE",
    "VALID_ACTIONS",
    "EVENT_FIELDS",
    "MembershipError",
    "MembershipEvent",
    "events_from_frame",
    "validate_events",
    "rebuild_membership_windows",
    "build_pit_universe",
    "membership_at",
]
