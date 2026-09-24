"""Correct return-on-assets and return-on-invested-capital semantics.

ROA and ROIC are different quantities and are never interchangeable:

* ``ROA = net_income / total_assets`` -- a balance-sheet-efficiency ratio.
* ``ROIC = NOPAT / invested_capital`` where
  ``NOPAT = operating_income * (1 - tax_rate)`` and
  ``invested_capital = total_debt + equity - cash``.

Every input must be a period-matched, already-available fundamental value. When
any required input is missing, the result is ``None`` (UNAVAILABLE). No field is
substituted, scaled or approximated, and ROA is never reported as ROIC.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TAX_RATE = 0.21

ROA_DEFINITION = "net_income / total_assets"
ROIC_DEFINITION = "(operating_income * (1 - tax_rate)) / (total_debt + equity - cash)"


@dataclass(frozen=True)
class ReturnMetrics:
    """Return metrics for one period-matched fundamental observation."""

    roa: float
    roic: float
    nopat: float
    invested_capital: float
    tax_rate: float
    inputs_used: dict
    roa_available: bool = True
    roic_available: bool = True
    roa_unavailable_reason: str = ""
    roic_unavailable_reason: str = ""

    def to_dict(self):
        return {
            "roa": self.roa,
            "roic": self.roic,
            "nopat": self.nopat,
            "invested_capital": self.invested_capital,
            "tax_rate": self.tax_rate,
            "roa_available": self.roa_available,
            "roic_available": self.roic_available,
            "roa_unavailable_reason": self.roa_unavailable_reason,
            "roic_unavailable_reason": self.roic_unavailable_reason,
            "roa_definition": ROA_DEFINITION,
            "roic_definition": ROIC_DEFINITION,
            "inputs_used": dict(self.inputs_used),
        }


def _is_number(value):
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number  # reject NaN without importing numpy here


def compute_roa(net_income, total_assets):
    """Net income over total assets, or None when either input is missing/zero-divisor."""
    if not _is_number(net_income) or not _is_number(total_assets):
        return None
    assets = float(total_assets)
    if assets == 0.0:
        return None
    return float(net_income) / assets


def compute_roic(operating_income, total_debt, equity, cash, tax_rate=DEFAULT_TAX_RATE):
    """NOPAT over invested capital, or None when any required input is missing.

    Returns ``(roic, nopat, invested_capital)``; each element is None when the
    computation is not fully supported by reported values. ``cash`` must be
    reported (including an explicit zero) -- it is not defaulted.
    """
    for name, value in (
        ("operating_income", operating_income),
        ("total_debt", total_debt),
        ("equity", equity),
        ("cash", cash),
        ("tax_rate", tax_rate),
    ):
        if not _is_number(value):
            return None, None, None
    nopat = float(operating_income) * (1.0 - float(tax_rate))
    invested = float(total_debt) + float(equity) - float(cash)
    if invested == 0.0:
        return None, nopat, invested
    return nopat / invested, nopat, invested


def compute_return_metrics(net_income, total_assets, operating_income=None, total_debt=None,
                           equity=None, cash=None, tax_rate=DEFAULT_TAX_RATE, inputs_used=None):
    """Compute ROA and ROIC together, annotating each unavailability reason."""
    roa = compute_roa(net_income, total_assets)
    roa_reason = ""
    if roa is None:
        if not _is_number(net_income):
            roa_reason = "net_income unavailable"
        elif not _is_number(total_assets):
            roa_reason = "total_assets unavailable"
        else:
            roa_reason = "total_assets is zero"

    roic, nopat, invested = compute_roic(operating_income, total_debt, equity, cash, tax_rate=tax_rate)
    roic_reason = ""
    if roic is None:
        for name, value in (
            ("operating_income", operating_income),
            ("total_debt", total_debt),
            ("equity", equity),
            ("cash", cash),
        ):
            if not _is_number(value):
                roic_reason = "%s unavailable" % name
                break
        if not roic_reason and invested == 0.0:
            roic_reason = "invested_capital is zero"

    return ReturnMetrics(
        roa=roa,
        roic=roic,
        nopat=nopat,
        invested_capital=invested,
        tax_rate=float(tax_rate) if _is_number(tax_rate) else None,
        inputs_used=dict(inputs_used or {}),
        roa_available=roa is not None,
        roic_available=roic is not None,
        roa_unavailable_reason=roa_reason,
        roic_unavailable_reason=roic_reason,
    )


def latest_available_value(pit_frame, field, asof_ts, key_cols=("cik",), available_at_col="available_at",
                           version_col="accession"):
    """Latest point-in-time value for ``field`` at ``asof_ts``, or None.

    Uses the deterministic as-of engine so a later restatement cannot leak
    backwards into an earlier prediction timestamp.
    """
    from .pit_join import asof_join

    subset = pit_frame.loc[pit_frame["field"] == field]
    if subset.empty:
        return None, None
    joined = asof_join(
        subset,
        asof_ts,
        key_cols=list(key_cols) + ["field"],
        available_at_col=available_at_col,
        value_cols=["value", "unit", "fiscal_period_end"],
        version_col=version_col,
    )
    if joined.empty:
        return None, None
    row = joined.iloc[0]
    return row["value"], row["fiscal_period_end"]


def return_metrics_asof(pit_frame, asof_ts, key_cols=("cik",), available_at_col="available_at",
                        version_col="accession", tax_rate=DEFAULT_TAX_RATE):
    """Compute ROA/ROIC from the latest point-in-time fundamentals at ``asof_ts``.

    All inputs are selected at the same prediction timestamp and may therefore
    come from different filings; each is the newest value already public. This
    prevents a period-matched assumption from silently mixing in a future value.
    """
    fields = {}
    periods = {}
    for field in ("net_income", "total_assets", "operating_income", "total_debt", "equity", "cash"):
        value, period = latest_available_value(
            pit_frame,
            field,
            asof_ts,
            key_cols=key_cols,
            available_at_col=available_at_col,
            version_col=version_col,
        )
        fields[field] = value
        periods[field] = period
    metrics = compute_return_metrics(
        fields["net_income"],
        fields["total_assets"],
        operating_income=fields["operating_income"],
        total_debt=fields["total_debt"],
        equity=fields["equity"],
        cash=fields["cash"],
        tax_rate=tax_rate,
        inputs_used=periods,
    )
    return metrics
