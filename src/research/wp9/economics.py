"""WP9 economic shadow vintage writer.

At each official snapshot this module records the exact top quintile of the
frozen ranking signal. The cutoff is fixed by the frozen contract as
``rank percentile > 0.80``; no alternative cutoff is searched, estimated, or
optimized. Securities are equal-weighted at entry and held for 12 months in a
shadow book only. There is no real trading, position management, or execution.

This module computes only the deterministic vintage composition and planned
maturity dates. Realized returns are computed later by the matured evaluator
using the certified target semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from ..fingerprints import fingerprint_obj
from ..ids import canonical_json
from .champion import FROZEN_FREEZE_ID
from .contract import CONTRACT_VERSION, SCHEMA_VERSION

ECONOMIC_SCHEMA_VERSION = "wp9_economic_shadow_v1"
BENCHMARK_SYMBOL = "SPY"
HOLDING_MONTHS = 12
TOP_QUINTILE_THRESHOLD = 0.80
COST_SCENARIOS_BPS = (0, 10, 25, 50)
REAL_TRADING = False
WEIGHTING = "equal_weight_at_entry"
CUTOFF_RULE = "top_quintile_exact (rank percentile > 0.80)"


class EconomicShadowError(RuntimeError):
    """Raised when an economic shadow vintage cannot be constructed honestly."""


def _require_fields(record: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [name for name in fields if name not in record]
    if missing:
        raise EconomicShadowError(
            "snapshot record is missing economic-shadow field(s): %s" % ", ".join(missing)
        )


def _number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EconomicShadowError("non-numeric value %r in score column" % (value,)) from exc
    return result


def _planned_maturity(snapshot_asof: str, holding_months: int = HOLDING_MONTHS) -> str:
    stamp = pd.Timestamp(snapshot_asof)
    if pd.isna(stamp):
        raise EconomicShadowError("snapshot_asof is not a parseable date: %r" % (snapshot_asof,))
    shifted = stamp + pd.DateOffset(months=int(holding_months))
    return shifted.strftime("%Y-%m-%d")


def select_top_quintile(
    record: Mapping[str, Any],
    *,
    threshold: float = TOP_QUINTILE_THRESHOLD,
) -> Dict[str, Any]:
    """Return the deterministic top-quintile constituents of a snapshot record.

    ``record`` is the row-wise snapshot body produced by
    :func:`src.research.wp9.storage.snapshot_payload`. Selection is exclusively
    ``percentile > threshold``; no rank-percentile recomputation, no alternative
    threshold search, and no model mutation happens here.
    """
    fields = (
        "snapshot_id",
        "snapshot_asof",
        "security_id",
        "ticker",
        "raw_model_score",
        "frozen_calibrated_score",
        "rank",
        "percentile",
    )
    _require_fields(record, fields)
    snapshot_id = str(record["snapshot_id"])
    snapshot_asof = str(record["snapshot_asof"])
    count = len(record["security_id"])
    if not (count == len(record["ticker"]) == len(record["raw_model_score"]) ==
            len(record["frozen_calibrated_score"]) == len(record["rank"]) ==
            len(record["percentile"])):
        raise EconomicShadowError("snapshot arrays are not length-consistent")

    selected = []
    for position in range(count):
        if _number(record["percentile"][position]) > float(threshold):
            selected.append(
                {
                    "security_id": str(record["security_id"][position]),
                    "ticker": str(record["ticker"][position]),
                    "raw_model_score": _number(record["raw_model_score"][position]),
                    "frozen_calibrated_score": _number(
                        record["frozen_calibrated_score"][position]
                    ),
                    "rank": int(record["rank"][position]),
                    "percentile": _number(record["percentile"][position]),
                }
            )

    # The frozen ranking is deterministic. Sort selected rows by rank then
    # security_id lexical ascending so the vintage payload is reproducible.
    selected.sort(key=lambda row: (int(row["rank"]), row["security_id"]))
    weight = (1.0 / len(selected)) if selected else 0.0
    for row in selected:
        row["weight"] = weight

    return {
        "schema_version": ECONOMIC_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "snapshot_asof": snapshot_asof,
        "model_freeze_id": FROZEN_FREEZE_ID,
        "cutoff_rule": CUTOFF_RULE,
        "cutoff_threshold": float(threshold),
        "weighting": WEIGHTING,
        "holding_months": HOLDING_MONTHS,
        "entry_date": snapshot_asof,
        "planned_maturity_date": _planned_maturity(snapshot_asof),
        "benchmark": BENCHMARK_SYMBOL,
        "real_trading": REAL_TRADING,
        "cost_scenarios_bps": list(COST_SCENARIOS_BPS),
        "n_selected": int(len(selected)),
        "n_universe": int(len(record.get("security_id", []))),
        "constituents": selected,
    }


def economic_shadow_digest(economic_record: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 of the canonical economic-shadow vintage."""
    return fingerprint_obj(dict(economic_record))


def economic_shadow_id(snapshot_id: str, economic_record: Mapping[str, Any]) -> str:
    """Return a deterministic stable id bound to the official snapshot id.

    The id binds the economic vintage to the snapshot id and the canonical
    content digest. No wall-clock or random value is used.
    """
    payload = {
        "kind": "wp9_economic_shadow",
        "snapshot_id": str(snapshot_id),
        "content_digest": economic_shadow_digest(economic_record),
    }
    text = canonical_json(payload)
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "wp9eco_" + digest[:20]


def persist_economic_shadow(
    economic_record: Mapping[str, Any],
    snapshot_id: str,
    root: Any = None,
) -> str:
    """Persist an immutable economic shadow vintage under its canonical path.

    The path ``provenance/wp9/economics/<shadow_id>.json`` is write-once using
    the shared immutability helper. Returns the relative repository path.
    """
    from pathlib import Path

    from ..immutability import save_immutable

    shadow_id = economic_shadow_id(snapshot_id, economic_record)
    payload = {"economic_shadow_id": shadow_id, **dict(economic_record)}
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    path = base / "provenance" / "wp9" / "economics" / ("%s.json" % shadow_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The write-once helper refuses divergent rewrites.
    save_immutable(path, payload)
    return str(path.relative_to(base))


__all__ = [
    "BENCHMARK_SYMBOL",
    "COST_SCENARIOS_BPS",
    "CUTOFF_RULE",
    "ECONOMIC_SCHEMA_VERSION",
    "EconomicShadowError",
    "HOLDING_MONTHS",
    "REAL_TRADING",
    "TOP_QUINTILE_THRESHOLD",
    "WEIGHTING",
    "economic_shadow_digest",
    "economic_shadow_id",
    "persist_economic_shadow",
    "select_top_quintile",
]
