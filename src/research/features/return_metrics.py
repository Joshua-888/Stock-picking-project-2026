"""WP3 return-on-x semantics: extend (never rewrite) the WP2 ``roa_roic`` module.

WP2 established that ROA and ROIC are different quantities. WP3 keeps that and
adds exactly two things the research feature set needs:

* a V2 **ROA** that divides by AVERAGE total assets across the period when both
  beginning and ending assets are available, falling back to the WP2 point
  definition only when they are not (the fallback is recorded, not hidden);
* an explicit record that divides invested capital is DOCUMENTED
  (``total_debt + equity - cash``) so a reader can audit it.

The legacy V1 semantic bug (using one measure where the other was meant) is NOT
reproduced as a V2 feature. It is preserved only under an explicitly named
``LEGACY_*`` feature id so a benchmark can contrast V1 vs V2.

No feature VALUE leaves this module unless every required input is present;
there is no substitution, scaling or approximation.
"""

from __future__ import annotations

from ..data.roa_roic import (
    DEFAULT_TAX_RATE,
    ROA_DEFINITION as WP2_ROA_DEFINITION,
    ROIC_DEFINITION as WP2_ROIC_DEFINITION,
    compute_return_metrics as _wp2_return_metrics,
)

V2_ROA_DEFINITION = "net_income / average_total_assets ((begin+end)/2) when both available, else net_income / total_assets"
V2_ROIC_DEFINITION = "NOPAT / invested_capital, NOPAT = operating_income * (1 - tax_rate), invested_capital = total_debt + equity - cash"
LEGACY_ROA_ROIC_NOTE = (
    "V1 conflated ROA/ROIC semantics; reproduced only as LEGACY_* for benchmark contrast, never as a V2 feature"
)


class ReturnMetricsError(ValueError):
    """Raised when return-metric inputs are structurally invalid."""


def compute_v2_roa(net_income, total_assets, total_assets_begin=None):
    """Return ``(roa, method)`` where method records point vs average assets.

    ``method`` is ``"average_assets"`` when both beginning and ending assets are
    available, otherwise ``"point_assets"``. Either way the divisor is the real
    reported figure; nothing is fabricated.
    """
    if net_income is None or total_assets is None:
        return None, "unavailable"
    try:
        income = float(net_income)
        assets_end = float(total_assets)
    except (TypeError, ValueError):
        return None, "unavailable"
    if total_assets_begin is not None:
        try:
            assets_begin = float(total_assets_begin)
        except (TypeError, ValueError):
            assets_begin = None
        if assets_begin is not None:
            average = (assets_begin + assets_end) / 2.0
            if average == 0.0:
                return None, "zero_average_assets"
            return income / average, "average_assets"
    if assets_end == 0.0:
        return None, "zero_assets"
    return income / assets_end, "point_assets"


def compute_v2_return_metrics(net_income, total_assets, total_assets_begin=None, operating_income=None,
                              total_debt=None, equity=None, cash=None, tax_rate=DEFAULT_TAX_RATE,
                              inputs_used=None):
    """Compute V2 ROA/ROIC together, delegating ROIC to the WP2 implementation.

    ROIC reuses ``roa_roic.compute_return_metrics`` verbatim so the semantics are
    identical across WP2 and WP3; ROA is overridden with the V2 average-assets
    definition. The returned mapping separates the two and never reports one in
    place of the other.
    """
    wp2 = _wp2_return_metrics(
        net_income, total_assets, operating_income=operating_income, total_debt=total_debt,
        equity=equity, cash=cash, tax_rate=tax_rate, inputs_used=inputs_used,
    )
    roa, method = compute_v2_roa(net_income, total_assets, total_assets_begin)
    return {
        "roa": roa,
        "roa_method": method,
        "roa_available": roa is not None,
        "roa_definition": V2_ROA_DEFINITION,
        "roic": wp2.roic,
        "roic_available": wp2.roic_available,
        "roic_unavailable_reason": wp2.roic_unavailable_reason,
        "roic_definition": V2_ROIC_DEFINITION,
        "nopat": wp2.nopat,
        "invested_capital": wp2.invested_capital,
        "tax_rate": wp2.tax_rate,
        "inputs_used": dict(inputs_used or {}),
        "legacy_note": LEGACY_ROA_ROIC_NOTE,
    }
