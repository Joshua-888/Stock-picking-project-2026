"""WP2B provider abstraction for historical universes and delisted securities.

This package owns the *interfaces* (protocols), the *normalization contracts*
and a small *factory* for external data providers that WP2 needs but no free
source supplies:

* historical index/universe membership with effective + announcement dates;
* delisted/acquired historical prices;
* corporate actions sufficient to rebuild a point-in-time adjusted price;
* permanent security identities (vendor id <-> issuer <-> ticker history).

Hard rules encoded here:

* credentials come from environment variables ONLY (never hard-coded);
* an adapter MUST raise :class:`ProviderCredentialError` *before* any network
  call when its key is absent -- no request, no fabricated data;
* vendor-specific parsing stays inside the adapter; every ``fetch_*`` returns
data normalized to the V2 Silver contracts declared below.

This module performs no feature engineering, no model fitting and no scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(RuntimeError):
    """Base class for provider-layer failures."""


class ProviderCredentialError(ProviderError):
    """Raised when a provider needs a credential that is not configured.

    This is raised *before* any network call so a missing key can never fall
    back to demo/synthetic data.
    """

    def __init__(self, provider_id, env_vars):
        self.provider_id = provider_id
        self.env_vars = tuple(env_vars)
        super().__init__(
            "provider %r requires one of the environment variables %s; "
            "no request was made and no data was fabricated"
            % (provider_id, ", ".join("'%s'" % name for name in env_vars))
        )


class ProviderNotConfiguredError(ProviderError):
    """Raised when a provider cannot be used at all (unknown id, etc.)."""


class ProviderPayloadError(ProviderError):
    """Raised when a vendor payload does not match the expected contract."""


class ProviderUnsupportedError(ProviderError):
    """Raised when a provider does not implement the requested capability."""


# --------------------------------------------------------------------------- #
# Normalization contracts (V2 Silver shapes)
# --------------------------------------------------------------------------- #

MEMBERSHIP_EVENT_FIELDS = (
    "security_id",
    "universe_id",
    "action",
    "announcement_date",
    "effective_date",
    "source",
    "source_reference",
)

DELISTED_PRICE_FIELDS = (
    "security_id",
    "ticker",
    "trade_date",
    "raw_close",
)

CORPORATE_ACTION_FIELDS = (
    "ticker",
    "kind",
    "effective_date",
    "numerator",
    "denominator",
    "amount",
)

IDENTITY_FIELDS = (
    "vendor",
    "vendor_security_id",
    "security_id",
    "cik",
    "issuer_key",
    "share_class",
    "effective_from",
    "effective_to",
    "source",
)

VALID_ACTIONS = ("add", "remove")


# --------------------------------------------------------------------------- #
# Credential helpers (environment variables only)
# --------------------------------------------------------------------------- #


def env_present(names):
    """Return the first non-empty environment variable name from ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return name
    return None


def require_env(provider_id, names):
    """Return ``(name, value)`` for the first configured variable, else raise.

    Never logs or returns the value to callers, only the variable name plus the
    secret value for the request builder.
    """
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return name, value.strip()
    raise ProviderCredentialError(provider_id, names)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


class HistoricalUniverseProvider(Protocol):
    """Source of point-in-time universe membership *change events*.

    ``fetch_membership_events`` must return a frame with the columns declared in
    :data:`MEMBERSHIP_EVENT_FIELDS`, one row per add/remove event. Providers must
    never return a present-day constituent list masquerading as history.
    """

    provider_id: str
    credential_env: tuple

    def fetch_membership_events(self, universe_id, start, end):
        ...


class DelistedPriceProvider(Protocol):
    """Source of raw prices for securities that no longer trade."""

    provider_id: str
    credential_env: tuple

    def fetch_delisted_prices(self, security_id, ticker, start, end):
        ...


class CorporateActionProvider(Protocol):
    """Source of splits/dividends/other actions with effective dates."""

    provider_id: str
    credential_env: tuple

    def fetch_corporate_actions(self, ticker, start, end):
        ...


class SecurityIdentityProvider(Protocol):
    """Source of permanent security identities and ticker history."""

    provider_id: str
    credential_env: tuple

    def fetch_identities(self, start=None, end=None):
        ...


# --------------------------------------------------------------------------- #
# Factory / registry
# --------------------------------------------------------------------------- #

PROVIDER_REGISTRY = {}


@dataclass(frozen=True)
class ProviderInfo:
    """Public, secret-free description of a registered provider."""

    provider_id: str
    module: str
    credential_env: tuple
    capabilities: tuple
    notes: str = ""

    def to_dict(self):
        return {
            "provider_id": self.provider_id,
            "module": self.module,
            "credential_env": list(self.credential_env),
            "capabilities": list(self.capabilities),
            "configured": env_present(self.credential_env) is not None,
            "notes": self.notes,
        }


def register_provider(cls):
    """Class decorator registering an adapter under its ``provider_id``."""
    provider_id = getattr(cls, "provider_id", None)
    if not provider_id:
        raise ProviderNotConfiguredError("provider class %r has no provider_id" % (cls,))
    PROVIDER_REGISTRY[provider_id] = cls
    return cls


def _ensure_loaded():
    # Import adapters lazily so importing this package stays network-free.
    from . import adapters  # noqa: F401
    return adapters


def provider_info():
    """List every registered provider with its capabilities (no secrets)."""
    _ensure_loaded()
    infos = []
    for provider_id, cls in sorted(PROVIDER_REGISTRY.items()):
        infos.append(
            ProviderInfo(
                provider_id=provider_id,
                module="%s.%s" % (cls.__module__, cls.__name__),
                credential_env=tuple(getattr(cls, "credential_env", ())),
                capabilities=tuple(getattr(cls, "capabilities", ())),
                notes=getattr(cls, "notes", ""),
            )
        )
    return infos


def get_provider(provider_id, **kwargs):
    """Instantiate a registered adapter; construction never touches the network."""
    _ensure_loaded()
    cls = PROVIDER_REGISTRY.get(provider_id)
    if cls is None:
        raise ProviderNotConfiguredError(
            "unknown provider %r; registered: %s"
            % (provider_id, ", ".join(sorted(PROVIDER_REGISTRY)))
        )
    return cls(**kwargs)


__all__ = [
    "ProviderError",
    "ProviderCredentialError",
    "ProviderNotConfiguredError",
    "ProviderPayloadError",
    "ProviderUnsupportedError",
    "MEMBERSHIP_EVENT_FIELDS",
    "DELISTED_PRICE_FIELDS",
    "CORPORATE_ACTION_FIELDS",
    "IDENTITY_FIELDS",
    "VALID_ACTIONS",
    "env_present",
    "require_env",
    "HistoricalUniverseProvider",
    "DelistedPriceProvider",
    "CorporateActionProvider",
    "SecurityIdentityProvider",
    "ProviderInfo",
    "PROVIDER_REGISTRY",
    "register_provider",
    "provider_info",
    "get_provider",
]
