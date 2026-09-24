"""Delisting methodology framework (WP2B).

A security that stops trading does not simply stop existing: the position is
resolved by a corporate event whose economics differ by *how* it terminated.
Treating every delisting identically, or treating a missing final price as a zero
return, fabricates information. This module defines the classification, the
honest handling of a vendor-supplied delisting return, and the documented bias
when that return is absent.

It never manufactures a delisting return and never assumes zero.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .availability import to_utc_timestamp

# --- terminal events ---------------------------------------------------------- #
CASH_ACQUISITION = "cash_acquisition"
STOCK_ACQUISITION = "stock_acquisition"
BANKRUPTCY = "bankruptcy"
EXCHANGE_DELISTING = "exchange_delisting"
OTC_TRANSITION = "otc_transition"
LIQUIDATION = "liquidation"
TICKER_TERMINATION = "ticker_termination"
MISSING_FINAL_PRICE = "missing_final_price"

DELISTING_KINDS = (
    CASH_ACQUISITION,
    STOCK_ACQUISITION,
    BANKRUPTCY,
    EXCHANGE_DELISTING,
    OTC_TRANSITION,
    LIQUIDATION,
    TICKER_TERMINATION,
    MISSING_FINAL_PRICE,
)

# Which kinds can legitimately carry a *final cash price* the holder receives.
CASH_SETTLED_KINDS = (CASH_ACQUISITION, LIQUIDATION, EXCHANGE_DELISTING, OTC_TRANSITION)

DELISTING_SEMANTICS = {
    CASH_ACQUISITION: "holder receives cash at the acquisition price; terminal value known",
    STOCK_ACQUISITION: "holder receives shares of the acquirer; position continues in a different security",
    BANKRUPTCY: "common equity is typically worthless or heavily impaired; recovery is uncertain",
    EXCHANGE_DELISTING: "removed from an exchange but may continue trading on another venue",
    OTC_TRANSITION: "security moves to OTC; liquidity and price continuity change materially",
    LIQUIDATION: "assets are wound up; final distribution may be unknown and delayed",
    TICKER_TERMINATION: "symbol retired/reused; not itself an economic event",
    MISSING_FINAL_PRICE: "provider cannot supply the terminal price; return is NOT zero, it is unknown",
}

# Documented bias when a vendor does not provide a delisting return.
MISSING_DELISTING_RETURN_BIAS = (
    "omitting the delisting return biases forward returns UPWARD for bankruptcies "
    "and DOWNWARD/no-effect for cash acquisitions; the direction depends on the "
    "termination kind and cannot be assumed away"
)


class DelistingError(RuntimeError):
    """Raised for malformed or unsupported delisting records."""


@dataclass(frozen=True)
class DelistingEvent:
    """A terminal event with an optional, never-invented delisting return.

    ``delisting_return`` is the vendor-supplied return realized from the last
    traded price to the resolution of the event. When it is ``None`` the return
    is UNKNOWN: callers must exclude the observation or flag the label, never
    substitute 0.0.
    """

    security_id: str
    ticker: str
    kind: str
    effective_date: str
    last_trade_date: str = None
    final_price: float = None
    delisting_return: float = None
    source: str = None
    source_reference: str = None

    def __post_init__(self):
        if self.kind not in DELISTING_KINDS:
            raise DelistingError("unknown delisting kind %r" % (self.kind,))
        if to_utc_timestamp(self.effective_date) is None:
            raise DelistingError("delisting needs a parseable effective_date: %r" % (self.effective_date,))

    @property
    def return_is_known(self):
        return self.delisting_return is not None

    @property
    def return_unknown_reason(self):
        if self.return_is_known:
            return None
        if self.kind == MISSING_FINAL_PRICE:
            return "provider supplied no terminal price for this security"
        return MISSING_DELISTING_RETURN_BIAS

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "ticker": self.ticker,
            "kind": self.kind,
            "effective_date": self.effective_date,
            "last_trade_date": self.last_trade_date,
            "final_price": self.final_price,
            "delisting_return": self.delisting_return,
            "return_is_known": self.return_is_known,
            "return_unknown_reason": self.return_unknown_reason,
            "source": self.source,
            "source_reference": self.source_reference,
        }


def classify_delisting(description=None, kind=None, has_final_price=False):
    """Coarse classifier when only a free-text reason is available.

    Conservative by design: an unrecognized reason becomes
    :data:`MISSING_FINAL_PRICE` (unknown economics) rather than an optimistic
    acquisition. ``kind`` short-circuits classification when already known.
    """
    if kind is not None:
        if kind not in DELISTING_KINDS:
            raise DelistingError("unknown delisting kind %r" % (kind,))
        return kind
    text = (description or "").strip().lower()
    if "bankrupt" in text or "chapter 11" in text or "chapter 7" in text:
        return BANKRUPTCY
    if "liquidat" in text or "wind" in text or "dissolv" in text:
        return LIQUIDATION
    if "acquir" in text or "merger" in text or "taken private" in text:
        return STOCK_ACQUISITION if "stock" in text else CASH_ACQUISITION
    if "otc" in text or "pink sheet" in text:
        return OTC_TRANSITION
    if "delist" in text or "suspended" in text or "halted" in text:
        return EXCHANGE_DELISTING
    if "ticker" in text or "symbol" in text:
        return TICKER_TERMINATION
    return MISSING_FINAL_PRICE if not has_final_price else EXCHANGE_DELISTING


def validate_delistings(events):
    """Integrity checks over delisting events; returns a problem list.

    Detects duplicate terminal events, terminal dates before the last trade
    date, and known-return events whose return is non-finite.
    """
    problems = []
    seen = set()
    for event in events:
        key = (event.security_id, event.effective_date, event.kind)
        if key in seen:
            problems.append("duplicate delisting %s" % (key,))
        seen.add(key)
        if event.last_trade_date is not None:
            last = to_utc_timestamp(event.last_trade_date)
            effective = to_utc_timestamp(event.effective_date)
            if last is not None and effective is not None and effective < last:
                problems.append(
                    "delisting effective %s precedes last trade %s for %s"
                    % (event.effective_date, event.last_trade_date, event.security_id)
                )
        if event.delisting_return is not None:
            try:
                value = float(event.delisting_return)
            except (TypeError, ValueError):
                problems.append("non-numeric delisting return for %s" % event.security_id)
                continue
            if value != value or value in (float("inf"), float("-inf")):
                problems.append("non-finite delisting return for %s" % event.security_id)
    return problems


def terminal_return(last_price, event):
    """Total terminal return from ``last_price`` to the event resolution.

    Returns ``(return, source)`` where ``source`` is one of ``vendor``,
    ``cash_price`` or ``unknown``. It NEVER fabricates a return: when the event
    carries no vendor return and no observable cash settlement price, the return
    is ``None`` with source ``unknown`` and the observation must be excluded.
    """
    if event.delisting_return is not None:
        return float(event.delisting_return), "vendor"
    if event.kind in CASH_SETTLED_KINDS and event.final_price is not None:
        if last_price in (None, 0):
            return None, "unknown"
        return (float(event.final_price) / float(last_price)) - 1.0, "cash_price"
    return None, "unknown"


__all__ = [
    "CASH_ACQUISITION",
    "STOCK_ACQUISITION",
    "BANKRUPTCY",
    "EXCHANGE_DELISTING",
    "OTC_TRANSITION",
    "LIQUIDATION",
    "TICKER_TERMINATION",
    "MISSING_FINAL_PRICE",
    "DELISTING_KINDS",
    "CASH_SETTLED_KINDS",
    "DELISTING_SEMANTICS",
    "MISSING_DELISTING_RETURN_BIAS",
    "DelistingError",
    "DelistingEvent",
    "classify_delisting",
    "validate_delistings",
    "terminal_return",
]
