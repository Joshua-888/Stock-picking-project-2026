"""Point-in-time availability semantics for V2 research data.

Every record that may become a historical model input must carry an
``available_at`` timestamp: the moment the information became public. A record
is usable for a prediction at ``T`` only when ``available_at <= T``.

Nothing in this module derives availability from a fiscal period end. A fiscal
period end describes what a value *represents*; ``available_at`` describes when
that value *existed publicly*. Publication lag is represented explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

UTC = "UTC"
# Conservative convention: when only a calendar date is known for publication,
# the information becomes usable from the start of the following calendar day,
# never from the start of the publication date itself.
DATE_ONLY_LAG_DAYS = 1
# Conservative default US equity close instant expressed in UTC (16:00 ET is
# 20:00 UTC under EDT and 21:00 UTC under EST; 21:00 is the conservative choice).
DEFAULT_CLOSE_UTC_TIME = "21:00:00"


class AvailabilityError(ValueError):
    """Raised when availability metadata is missing or unusable."""


class UnavailableError(AvailabilityError):
    """Raised when a record is not available at the requested prediction time."""


def to_utc_timestamp(value):
    """Normalise a date/datetime/str/Timestamp to tz-aware UTC, or None.

    Date-only values are interpreted as midnight UTC. Missing, NaT and
    unparseable values return ``None`` so callers can treat them as unavailable
    rather than silently guessing.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts


def fundamental_available_at(filing_date, acceptance_datetime=None, lag_days=DATE_ONLY_LAG_DAYS):
    """Conservative availability instant for a fundamental filing.

    Two independent facts are combined and the LATER one wins, so availability
    can never be moved earlier than the evidence supports:

    * the filing date treated as fully elapsed (``filing_date + lag_days``);
      a bare publication date does not reveal the time of day, so the whole day
      must be treated as unavailable;
    * the SEC acceptance datetime, which is the precise acceptance instant.

    SEC's acceptance timestamps and filing dates can disagree by a calendar day
    (timezone/labelling artifact). Taking the maximum keeps a value from being
    declared public earlier than either signal allows. The filing date is a
    publication date and is never a fiscal period end.
    """
    candidates = []
    filed = to_utc_timestamp(filing_date)
    if filed is not None:
        candidates.append(filed.normalize() + pd.Timedelta(days=int(lag_days)))
    if acceptance_datetime is not None and not (isinstance(acceptance_datetime, float) and pd.isna(acceptance_datetime)):
        if str(acceptance_datetime).strip():
            stamp = to_utc_timestamp(acceptance_datetime)
            if stamp is not None and (filed is None or stamp >= filed.normalize()):
                candidates.append(stamp)
    if not candidates:
        return None
    return max(candidates)


def price_available_at(trade_date, close_utc_time=DEFAULT_CLOSE_UTC_TIME):
    """Availability instant for a market observation on ``trade_date``.

    Price information for a trade date is only model-visible after that
    session's close, so availability is the trade date combined with a
    conservative market-close time expressed in UTC.
    """
    stamp = to_utc_timestamp(trade_date)
    if stamp is None:
        return None
    day = stamp.tz_convert(UTC).strftime("%Y-%m-%d")
    return pd.Timestamp("%s %s" % (day, close_utc_time), tz=UTC)


def macro_available_at(release_date=None, vintage_date=None):
    """Availability instant for a macro observation.

    A value is public at the later of its release date and the vintage in which
    it appears; using the current revised series without a vintage is refused.
    """
    stamps = [to_utc_timestamp(release_date), to_utc_timestamp(vintage_date)]
    known = [stamp for stamp in stamps if stamp is not None]
    if not known:
        return None
    return max(known)


def universe_available_at(valid_from):
    """Availability instant for an explicit universe membership change."""
    return to_utc_timestamp(valid_from)


@dataclass(frozen=True)
class AvailabilityPolicy:
    """Declarative availability rule for one family of research data."""

    name: str
    source: str
    rule: str
    compute: Callable[..., Optional[pd.Timestamp]]
    description: str = ""

    def available_at(self, **kwargs):
        """Compute the availability instant for this policy."""
        return self.compute(**kwargs)


FUNDAMENTAL = AvailabilityPolicy(
    name="FUNDAMENTAL",
    source="SEC EDGAR",
    rule="available_at = authoritative filing timestamp (acceptance datetime, else filing date + 1 day)",
    compute=fundamental_available_at,
    description="Fundamental values are never available at their fiscal period end.",
)

PRICE = AvailabilityPolicy(
    name="PRICE",
    source="market data",
    rule="available_at = trade date combined with conservative close time in UTC",
    compute=price_available_at,
    description="A price observed on a trade date is only known after that close.",
)

MACRO = AvailabilityPolicy(
    name="MACRO",
    source="FRED/ALFRED",
    rule="available_at = max(release date, vintage date)",
    compute=macro_available_at,
    description="Revised current values are not historical values; the vintage governs.",
)

UNIVERSE = AvailabilityPolicy(
    name="UNIVERSE",
    source="universe definition",
    rule="available_at = explicit membership valid_from",
    compute=universe_available_at,
    description="Universe membership is point-in-time information when supplied explicitly.",
)

POLICIES = {policy.name: policy for policy in (FUNDAMENTAL, PRICE, MACRO, UNIVERSE)}


def is_available(available_at, prediction_ts):
    """True only when ``available_at <= prediction_ts`` (missing -> False)."""
    prediction = to_utc_timestamp(prediction_ts)
    if prediction is None:
        raise AvailabilityError("prediction timestamp is missing or unparseable: %r" % (prediction_ts,))
    stamp = to_utc_timestamp(available_at)
    if stamp is None:
        return False
    return bool(stamp <= prediction)


def require_available(available_at, prediction_ts, context=None):
    """Raise :class:`UnavailableError` unless ``available_at <= prediction_ts``."""
    if not is_available(available_at, prediction_ts):
        where = (" (%s)" % context) if context else ""
        raise UnavailableError(
            "record is not available at %r%s: available_at=%r" % (prediction_ts, where, available_at)
        )
    return True


def availability_mask(df, available_at_col, prediction_ts):
    """Boolean mask selecting rows usable at ``prediction_ts``."""
    if available_at_col not in df.columns:
        raise AvailabilityError("missing availability column %r" % available_at_col)
    prediction = to_utc_timestamp(prediction_ts)
    if prediction is None:
        raise AvailabilityError("prediction timestamp is missing or unparseable: %r" % (prediction_ts,))
    stamps = [to_utc_timestamp(value) for value in df[available_at_col]]
    return pd.Series([stamp is not None and stamp <= prediction for stamp in stamps], index=df.index)


def filter_available(df, available_at_col, prediction_ts):
    """Rows usable at ``prediction_ts`` with no missing availability metadata."""
    return df.loc[availability_mask(df, available_at_col, prediction_ts)].copy()
