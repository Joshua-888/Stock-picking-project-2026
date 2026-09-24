"""Security identity chain expansion (WP2B, mission section 12).

WP2A already resolves *which security did this ticker mean on this date?* by
issuer identity (CIK preferred) plus dated ticker assignments. WP2B needs the
richer chain that a delisted/survivorship-safe universe requires:

    vendor permanent id  <->  issuer/company  <->  CIK
        <->  historical ticker(s)  <->  exchange  <->  valid period

with explicit support for share classes, issuer mergers, successor securities and
reused tickers by date. The invariants enforced here are:

* a ``security_id`` represents exactly ONE economic security; economically
  different securities must never share an id (a share class, a successor entity
  and a merger partner are separate ids);
* a reused ticker is ambiguous unless the windows are non-overlapping; overlaps
  fail rather than being silently ordered;
* ambiguity is surfaced (:class:`AmbiguousTickerError`), never guessed.

The module builds on :mod:`security_master` and does not modify it.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .availability import to_utc_timestamp
from .security_master import (
    AmbiguousTickerError,
    SecurityMaster,
    SecurityMasterError,
    SecurityRecord,
    TickerAssignment,
    UnknownTickerError,
    security_id_for,
)

SHARE_CLASS_PRIMARY = "primary"

MERGED_INTO = "merged_into"
SUCCEEDED_BY = "succeeded_by"
SHARE_CLASS_OF = "share_class_of"

RELATIONSHIP_KINDS = (MERGED_INTO, SUCCEEDED_BY, SHARE_CLASS_OF)


class IdentityChainError(SecurityMasterError):
    """Raised when the identity chain is inconsistent (fail loudly)."""


@dataclass(frozen=True)
class SecurityIdentity:
    """One economic security identity with explicit vendor cross-references."""

    security_id: str
    name: str
    cik: str = None
    issuer_key: str = None
    share_class: str = SHARE_CLASS_PRIMARY
    vendor_ids: tuple = ()

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "name": self.name,
            "cik": self.cik,
            "issuer_key": self.issuer_key,
            "share_class": self.share_class,
            "vendor_ids": list(self.vendor_ids),
        }


@dataclass(frozen=True)
class VendorReference:
    """A vendor's permanent id mapped onto one of our ``security_id`` values."""

    vendor: str
    vendor_security_id: str
    security_id: str
    effective_from: str = None
    effective_to: str = None
    source: str = None

    def to_dict(self):
        return {
            "vendor": self.vendor,
            "vendor_security_id": self.vendor_security_id,
            "security_id": self.security_id,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "source": self.source,
        }


@dataclass(frozen=True)
class SecurityRelationship:
    """A directed relationship between two distinct securities."""

    kind: str
    predecessor_id: str
    successor_id: str
    effective_date: str
    source: str = None

    def __post_init__(self):
        if self.kind not in RELATIONSHIP_KINDS:
            raise IdentityChainError("unknown relationship kind %r" % (self.kind,))
        if self.predecessor_id == self.successor_id:
            raise IdentityChainError("relationship cannot link a security to itself: %s" % self.predecessor_id)
        if to_utc_timestamp(self.effective_date) is None:
            raise IdentityChainError("relationship needs a parseable effective_date: %r" % (self.effective_date,))

    def to_dict(self):
        return {
            "kind": self.kind,
            "predecessor_id": self.predecessor_id,
            "successor_id": self.successor_id,
            "effective_date": self.effective_date,
            "source": self.source,
        }


def security_id_for_share_class(cik=None, issuer_key=None, share_class=SHARE_CLASS_PRIMARY):
    """Deterministic id that keeps share classes distinct.

    The primary class keeps the WP2A :func:`security_id_for` id so existing
    references stay valid; a non-primary class is hashed from a class-suffixed
    issuer key so a class can never impersonate the primary listing.
    """
    if share_class in (None, "", SHARE_CLASS_PRIMARY):
        return security_id_for(cik=cik, issuer_key=issuer_key)
    base = security_id_for(cik=cik, issuer_key=issuer_key)
    marker = "cik" if (cik not in (None, "") and str(cik).strip()) else "issuer_key"
    suffix = "::share_class=%s::marker=%s" % (share_class, marker)
    return security_id_for(issuer_key=(issuer_key or base) + suffix)


class IdentityChain:
    """Identity chain over a :class:`SecurityMaster` plus vendor cross-refs."""

    def __init__(self, security_master=None):
        self._master = security_master if security_master is not None else SecurityMaster()
        self._identities = {}
        self._vendor_refs = []
        self._relationships = []

    @property
    def master(self):
        return self._master

    @property
    def vendor_references(self):
        return list(self._vendor_refs)

    @property
    def relationships(self):
        return list(self._relationships)

    def add_identity(self, identity):
        if not isinstance(identity, SecurityIdentity):
            identity = SecurityIdentity(**identity)
        existing = self._identities.get(identity.security_id)
        if existing is not None and existing != identity:
            raise IdentityChainError(
                "security %s already registered with a different identity" % identity.security_id
            )
        self._identities[identity.security_id] = identity
        self._master.add_security(
            SecurityRecord(
                security_id=identity.security_id,
                name=identity.name,
                cik=identity.cik,
                issuer_key=identity.issuer_key,
            )
        )
        return identity

    def add_vendor_reference(self, reference):
        if not isinstance(reference, VendorReference):
            reference = VendorReference(**reference)
        if reference.security_id not in self._identities:
            raise IdentityChainError(
                "vendor reference targets unknown security %s" % reference.security_id
            )
        self._vendor_refs.append(reference)
        return reference

    def add_ticker(self, assignment):
        return self._master.add_ticker(assignment)

    def add_relationship(self, relationship):
        if not isinstance(relationship, SecurityRelationship):
            relationship = SecurityRelationship(**relationship)
        for side in (relationship.predecessor_id, relationship.successor_id):
            if side not in self._identities:
                raise IdentityChainError("relationship references unknown security %s" % side)
        self._relationships.append(relationship)
        return relationship

    def resolve_vendor(self, vendor, vendor_security_id, as_of=None):
        """Map a vendor id onto our ``security_id`` without guessing ambiguity."""
        moment = to_utc_timestamp(as_of) if as_of is not None else None
        matches = []
        for reference in self._vendor_refs:
            if reference.vendor != vendor:
                continue
            if str(reference.vendor_security_id) != str(vendor_security_id):
                continue
            if moment is not None:
                start = to_utc_timestamp(reference.effective_from) if reference.effective_from else None
                end = to_utc_timestamp(reference.effective_to) if reference.effective_to else None
                if start is not None and start > moment:
                    continue
                if end is not None and moment > end:
                    continue
            matches.append(reference.security_id)
        unique = sorted(set(matches))
        if not unique:
            raise UnknownTickerError(
                "vendor %r id %r has no security mapping%s"
                % (vendor, vendor_security_id, " at %s" % as_of if as_of else "")
            )
        if len(unique) > 1:
            raise AmbiguousTickerError(
                "vendor %r id %r maps to multiple securities%s: %s"
                % (vendor, vendor_security_id, " at %s" % as_of if as_of else "", ", ".join(unique))
            )
        return unique[0]

    def share_class_collisions(self):
        """Distinct share classes that incorrectly share a ``security_id``."""
        collisions = {}
        for identity in self._identities.values():
            collisions.setdefault(identity.security_id, set()).add(identity.share_class)
        return {sid: sorted(classes) for sid, classes in collisions.items() if len(classes) > 1}

    def issuer_share_classes(self, cik):
        """All share classes recorded for one issuer CIK, each with its own id."""
        return [
            identity.to_dict()
            for identity in sorted(self._identities.values(), key=lambda item: item.security_id)
            if identity.cik == cik
        ]

    def successors(self, security_id, kind=None):
        """Securities that succeeded/received ``security_id`` (by effective date)."""
        rows = [
            rel.to_dict()
            for rel in self._relationships
            if rel.predecessor_id == security_id and (kind is None or rel.kind == kind)
        ]
        rows.sort(key=lambda item: item["effective_date"])
        return rows

    def to_frame(self):
        import pandas as pd

        rows = []
        for identity in self._identities.values():
            row = identity.to_dict()
            assignments = self._master.ticker_changes(identity.security_id)
            if not assignments:
                rows.append(row)
                continue
            for assignment in assignments:
                merged = dict(row)
                merged.update({
                    "ticker": assignment["ticker"],
                    "exchange": assignment["exchange"],
                    "valid_from": assignment["valid_from"],
                    "valid_to": assignment["valid_to"],
                })
                rows.append(merged)
        return pd.DataFrame(rows)


__all__ = [
    "SHARE_CLASS_PRIMARY",
    "MERGED_INTO",
    "SUCCEEDED_BY",
    "SHARE_CLASS_OF",
    "RELATIONSHIP_KINDS",
    "IdentityChainError",
    "SecurityIdentity",
    "VendorReference",
    "SecurityRelationship",
    "security_id_for_share_class",
    "IdentityChain",
]
