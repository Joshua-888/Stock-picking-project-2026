"""Corporate-action PIT extension (WP2B, mission sections 11/21).

WP2A :mod:`pit_prices` rebuilds an adjusted price from *raw* closes and only the
split/dividend factors effective at or before an as-of instant. WP2B adds the
non-price actions that a survivorship-safe history encounters -- spin-offs,
cash/stock acquisitions, bankruptcies, exchange delistings and ticker renames --
WITHOUT letting any of them leak information into an earlier date.

Design:

* a :class:`CorporateActionEvent` is a dated, typed fact (kind, effective date);
* only *price-adjusting* kinds (split, dividend) change the adjusted price, and
  they do so exclusively through :func:`pit_prices.build_pit_adjusted_frame`;
* non-price kinds are carried as metadata and NEVER move a price;
* an event effective after the as-of instant is invisible at that instant.

The invariant every test in this module targets:

    adding an action with effective_date > T must not change any information
    (price, factor, membership, event set) observable at T.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .availability import to_utc_timestamp
from .pit_prices import DIVIDEND, SPLIT, CorporateAction, actions_known_by, pit_return

# Price-adjusting kinds vs. information-only kinds.
PRICE_ADJUSTING_KINDS = (SPLIT, DIVIDEND)

SPINOFF = "spinoff"
CASH_ACQUISITION = "cash_acquisition"
STOCK_ACQUISITION = "stock_acquisition"
BANKRUPTCY = "bankruptcy"
EXCHANGE_DELISTING = "exchange_delisting"
TICKER_RENAME = "ticker_rename"

INFORMATION_ONLY_KINDS = (
    SPINOFF,
    CASH_ACQUISITION,
    STOCK_ACQUISITION,
    BANKRUPTCY,
    EXCHANGE_DELISTING,
    TICKER_RENAME,
)

ALL_KINDS = PRICE_ADJUSTING_KINDS + INFORMATION_ONLY_KINDS

# Documented price semantics per kind (why a price does or does not move).
ACTION_SEMANTICS = {
    SPLIT: "share count changes; pre-split prices are back-adjusted by denominator/numerator",
    DIVIDEND: "cash leaves the company; pre-ex prices are adjusted by (close-amount)/close",
    SPINOFF: "holder receives a new security; the parent price is NOT rebased here without a recorded ratio",
    CASH_ACQUISITION: "terminal cash event; no continuing price series for the target",
    STOCK_ACQUISITION: "target becomes acquirer shares; the target series terminates",
    BANKRUPTCY: "common equity is impaired; no adjustment is invented from the event alone",
    EXCHANGE_DELISTING: "listing venue changes; price continuity is not assumed",
    TICKER_RENAME: "identity/venue metadata; the price series is unchanged by the rename",
}


class CorporateActionError(RuntimeError):
    """Raised for malformed corporate-action events."""


@dataclass(frozen=True)
class CorporateActionEvent:
    """A dated corporate action of any kind, price-adjusting or informational."""

    ticker: str
    kind: str
    effective_date: str
    numerator: float = None
    denominator: float = None
    amount: float = None
    source: str = None
    source_reference: str = None

    def __post_init__(self):
        if self.kind not in ALL_KINDS:
            raise CorporateActionError("unknown corporate action kind %r" % (self.kind,))
        if to_utc_timestamp(self.effective_date) is None:
            raise CorporateActionError("corporate action needs a parseable effective_date: %r" % (self.effective_date,))
        if self.kind == SPLIT and (self.numerator in (None, 0) or self.denominator in (None, 0)):
            raise CorporateActionError("split needs non-zero numerator and denominator")
        if self.kind == DIVIDEND and self.amount is None:
            raise CorporateActionError("dividend needs an amount")

    @property
    def adjusts_price(self):
        return self.kind in PRICE_ADJUSTING_KINDS

    def to_price_action(self):
        """Return the :class:`CorporateAction` this event contributes, or None.

        A non-price event returns ``None`` and therefore can never alter any
        price value used by :mod:`pit_prices`.
        """
        if not self.adjusts_price:
            return None
        return CorporateAction(
            ticker=self.ticker,
            kind=self.kind,
            effective_date=self.effective_date,
            numerator=self.numerator,
            denominator=self.denominator,
            amount=self.amount,
        )

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "kind": self.kind,
            "effective_date": self.effective_date,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "amount": self.amount,
            "adjusts_price": self.adjusts_price,
            "source": self.source,
            "source_reference": self.source_reference,
        }


def price_actions_from_events(events):
    """Extract only the price-adjusting :class:`CorporateAction` records."""
    actions = []
    for event in events:
        action = event.to_price_action()
        if action is not None:
            actions.append(action)
    return actions


def events_known_by(events, asof):
    """Events whose effective date is at or before ``asof`` (nothing future)."""
    moment = to_utc_timestamp(asof)
    if moment is None:
        raise CorporateActionError("asof must be a parseable timestamp")
    known = []
    for event in events:
        stamp = to_utc_timestamp(event.effective_date)
        if stamp is not None and stamp <= moment:
            known.append(event)
    return sorted(known, key=lambda item: (item.effective_date, item.kind))


def event_set_unchanged_by_future(events, asof, future_event):
    """True when ``future_event`` (effective after ``asof``) leaks nothing at ``asof``.

    Compares the as-of event set with and without the future event, and the
    as-of membership of every event. A well-formed future event must be invisible.
    """
    moment = to_utc_timestamp(asof)
    future_stamp = to_utc_timestamp(future_event.effective_date)
    if moment is None or future_stamp is None:
        raise CorporateActionError("asof and effective_date must be parseable")
    if future_stamp <= moment:
        raise CorporateActionError("future_event is not after asof")
    before = [event.to_dict() for event in events_known_by(events, asof)]
    after = [event.to_dict() for event in events_known_by(list(events) + [future_event], asof)]
    return before == after


def pit_info_at(raw_frame, events, asof, raw_col="raw_close"):
    """Bundle the point-in-time price information observable at ``asof``.

    Returns ``(adjusted_frame, known_events)`` where the adjusted frame is built
    only from price actions known at ``asof`` and ``known_events`` lists the
    actions visible at that instant. Informational actions are returned separately
    so a caller can see that a future bankruptcy/spinoff does not move a price.
    """
    known = events_known_by(events, asof)
    price_actions = [action for action in price_actions_from_events(known)]
    from .pit_prices import build_pit_adjusted_frame

    adjusted = build_pit_adjusted_frame(raw_frame, price_actions, asof=asof, raw_col=raw_col)
    informational = [event.to_dict() for event in known if not event.adjusts_price]
    return adjusted, informational


def pit_total_return(raw_frame, events, start, end, raw_col="raw_close"):
    """Forward total return over ``[start, end]`` using price actions known by ``end``.

    Delegates to WP2A :func:`pit_prices.pit_return` with only the price-adjusting
    actions, so informational events (spinoff, delisting, rename) cannot distort
    the label.
    """
    return pit_return(
        raw_frame,
        price_actions_from_events(events),
        start,
        end,
        raw_col=raw_col,
    )


__all__ = [
    "PRICE_ADJUSTING_KINDS",
    "INFORMATION_ONLY_KINDS",
    "ALL_KINDS",
    "ACTION_SEMANTICS",
    "SPINOFF",
    "CASH_ACQUISITION",
    "STOCK_ACQUISITION",
    "BANKRUPTCY",
    "EXCHANGE_DELISTING",
    "TICKER_RENAME",
    "CorporateActionError",
    "CorporateActionEvent",
    "price_actions_from_events",
    "events_known_by",
    "event_set_unchanged_by_future",
    "pit_info_at",
    "pit_total_return",
]
