"""Permanent security identifiers and point-in-time ticker resolution.

Ticker symbols are not identifiers: they change, get reused and are ambiguous
across venues. The permanent key here is ``security_id``, a content-addressed
hash of the issuer's permanent identity (CIK when available, otherwise a
normalised issuer key). Ticker/exchange history is stored separately with
validity windows so ``resolve_ticker`` can answer ``which security did this
symbol mean on this date?`` without guessing.

Ambiguity is surfaced, never silently resolved: if two securities claim the same
symbol over the requested date, a :class:`AmbiguousTickerError` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.ids import canonical_json

from .availability import to_utc_timestamp

import hashlib

SECURITY_ID_PREFIX = "sec_"


class SecurityMasterError(RuntimeError):
    """Raised for malformed security-master records."""


class AmbiguousTickerError(SecurityMasterError):
    """Raised when a ticker maps to more than one security at a date."""


class UnknownTickerError(SecurityMasterError, KeyError):
    """Raised when a ticker has no security mapping at a date."""


def security_id_for(cik=None, issuer_key=None):
    """Deterministic ``security_id`` from a permanent identity.

    Preference is the zero-padded CIK; when no CIK exists a normalised issuer key
    must be supplied. The digest is documented and stable: SHA-256 of the
    canonical JSON of ``{'scheme': ..., 'key': ...}`` truncated to 16 hex chars.
    """
    if cik not in (None, "") and str(cik).strip():
        digits = "".join(ch for ch in str(cik) if ch.isdigit())
        if not digits:
            raise SecurityMasterError("CIK %r contains no digits" % (cik,))
        scheme, key = "cik", digits.zfill(10)
    elif issuer_key not in (None, "") and str(issuer_key).strip():
        scheme, key = "issuer_key", " ".join(str(issuer_key).split()).upper()
    else:
        raise SecurityMasterError("security identity requires a CIK or an issuer_key")
    digest = hashlib.sha256(canonical_json({"scheme": scheme, "key": key}).encode("utf-8")).hexdigest()
    return "%s%s" % (SECURITY_ID_PREFIX, digest[:16])


@dataclass(frozen=True)
class SecurityRecord:
    """One permanent security with its identity fields."""

    security_id: str
    name: str
    cik: str = None
    issuer_key: str = None

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "name": self.name,
            "cik": self.cik,
            "issuer_key": self.issuer_key,
        }


@dataclass(frozen=True)
class TickerAssignment:
    """A ticker/exchange held by a security over a validity window."""

    security_id: str
    ticker: str
    exchange: str
    valid_from: str
    valid_to: str = None

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "ticker": self.ticker,
            "exchange": self.exchange,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


class SecurityMaster:
    """In-memory security master with point-in-time ticker resolution."""

    def __init__(self, securities=None, assignments=None):
        self._securities = {}
        self._assignments = []
        for record in securities or []:
            self.add_security(record)
        for assignment in assignments or []:
            self.add_ticker(assignment)

    @property
    def securities(self):
        return dict(self._securities)

    @property
    def assignments(self):
        return list(self._assignments)

    def add_security(self, record):
        if not isinstance(record, SecurityRecord):
            record = SecurityRecord(**record)
        existing = self._securities.get(record.security_id)
        if existing is not None and existing != record:
            raise SecurityMasterError("security %s already registered with different identity" % record.security_id)
        self._securities[record.security_id] = record
        return record

    def add_ticker(self, assignment):
        if not isinstance(assignment, TickerAssignment):
            assignment = TickerAssignment(**assignment)
        if assignment.security_id not in self._securities:
            raise SecurityMasterError("ticker assignment references unknown security %s" % assignment.security_id)
        start = to_utc_timestamp(assignment.valid_from)
        if start is None:
            raise SecurityMasterError("ticker assignment needs a parseable valid_from")
        end = to_utc_timestamp(assignment.valid_to)
        if end is not None and end < start:
            raise SecurityMasterError("ticker assignment valid_to precedes valid_from")
        self._assignments.append(assignment)
        return assignment

    def resolve_ticker(self, ticker, as_of):
        """Return the ``security_id`` that ``ticker`` denoted at ``as_of``."""
        moment = to_utc_timestamp(as_of)
        if moment is None:
            raise SecurityMasterError("as_of must be a parseable timestamp")
        symbol = str(ticker).strip().upper()
        matches = []
        for assignment in self._assignments:
            if assignment.ticker.strip().upper() != symbol:
                continue
            start = to_utc_timestamp(assignment.valid_from)
            end = to_utc_timestamp(assignment.valid_to)
            if start is None:
                continue
            if start <= moment and (end is None or moment <= end):
                matches.append(assignment.security_id)
        unique = sorted(set(matches))
        if not unique:
            raise UnknownTickerError("ticker %r has no security mapping at %s" % (ticker, moment.isoformat()))
        if len(unique) > 1:
            raise AmbiguousTickerError(
                "ticker %r maps to multiple securities at %s: %s" % (ticker, moment.isoformat(), ", ".join(unique))
            )
        return unique[0]

    def resolve_series(self, ticker, dates):
        """Resolve one ticker across many dates, returning a list of security_ids.

        Unknown or ambiguous dates yield ``None`` in that position so callers can
        exclude the observation instead of receiving a wrong identity.
        """
        resolved = []
        for value in dates:
            try:
                resolved.append(self.resolve_ticker(ticker, value))
            except SecurityMasterError:
                resolved.append(None)
        return resolved

    def ticker_changes(self, security_id):
        """Ordered ticker history for a security (by valid_from)."""
        rows = [item for item in self._assignments if item.security_id == security_id]
        rows.sort(key=lambda item: (to_utc_timestamp(item.valid_from), item.ticker))
        return [item.to_dict() for item in rows]

    def detect_symbol_reuse(self):
        """Tickers used by more than one security (symbol reuse / ambiguity)."""
        by_symbol = {}
        for assignment in self._assignments:
            by_symbol.setdefault(assignment.ticker.strip().upper(), set()).add(assignment.security_id)
        return {symbol: sorted(owners) for symbol, owners in sorted(by_symbol.items()) if len(owners) > 1}

    def detect_overlaps(self):
        """Overlapping assignments for one ticker (invalid, ambiguous history)."""
        problems = []
        by_symbol = {}
        for assignment in self._assignments:
            by_symbol.setdefault(assignment.ticker.strip().upper(), []).append(assignment)
        for symbol, items in sorted(by_symbol.items()):
            windows = sorted(
                (
                    to_utc_timestamp(item.valid_from),
                    to_utc_timestamp(item.valid_to),
                    item.security_id,
                )
                for item in items
            )
            for index in range(1, len(windows)):
                prior_start, prior_end, prior_owner = windows[index - 1]
                start, _end, owner = windows[index]
                if prior_owner != owner and (prior_end is None or start <= prior_end):
                    problems.append(
                        {
                            "ticker": symbol,
                            "security_ids": sorted({prior_owner, owner}),
                            "overlap_start": start.isoformat() if start is not None else None,
                        }
                    )
        return problems

    def duplicate_securities(self):
        """Securities sharing a permanent identity (CIK / issuer key)."""
        by_identity = {}
        for record in self._securities.values():
            identity = ("cik", record.cik) if record.cik else ("issuer_key", str(record.issuer_key).upper())
            by_identity.setdefault(identity, set()).add(record.security_id)
        return {identity[1]: sorted(ids) for identity, ids in sorted(by_identity.items()) if len(ids) > 1}

    def to_frame(self):
        """Flat assignment table with security attributes attached."""
        rows = []
        for assignment in self._assignments:
            record = self._securities[assignment.security_id]
            row = assignment.to_dict()
            row["name"] = record.name
            row["cik"] = record.cik
            rows.append(row)
        return pd.DataFrame(rows)
