"""WP4 Keyes signal construction: HISTORICAL replication and MODERN composite.

Two tracks are built here and they are never merged:

* ``HISTORICAL_KEYES_REPLICATION`` applies *explicit simultaneous logical
  thresholds* (the WP3 ``KeyesRuleSet`` contract). A security qualifies only if
  it satisfies **every** criterion; the count of qualifiers is therefore free to
  be 0 or N. There is deliberately **no** percentile/top-30% quota, which is the
  forbidden V1 behaviour.
* ``MODERN_KEYES_INSPIRED`` builds a rank composite with **equal weights and
  pre-registered signs**. No hand-tuned 35/25/15 weights exist anywhere.

Threshold discipline
--------------------
Keyes' published regression implies directions (growth and momentum positive,
valuations negative), not machine-readable numeric cut-offs. WP4 therefore uses
**pre-registered, non-fitted** economic thresholds and records that choice in the
fidelity table. Nothing in this module is tuned by looking at the outcome, and
``a_priori`` thresholds are declared once in ``HISTORICAL_RULE_SETS``.

Missingness discipline
----------------------
A missing component never becomes zero. It fails its criterion, it is dropped
from the modern composite, and it is counted. Proxies are carried by name
(``X12_PROXY``) and are never relabelled as an exact Keyes variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.keyes import (
    TRACK_HISTORICAL,
    TRACK_MODERN,
    KeyesCriterion,
    KeyesRuleSet,
    qualify,
)
from .variables import (
    FIDELITY_EXACT,
    FIDELITY_UNAVAILABLE,
    KEYES_DIRECTIONS,
    VARIABLE_SPECS,
    X5,
    X6,
    X8,
    X9,
    X12_PROXY,
)

SIGNAL_VERSION = "v2_wp4_keyes_signals_v1"

HISTORICAL_COMPONENTS = (X5, X6, X8, X9, X12_PROXY)
# The X12 leg has no exact historical point-in-time source. The historical rule
# set may still be evaluated with the proxy, but the track is then PARTIAL and
# the proxy is reported by name.
PROXY_COMPONENTS = (X12_PROXY,)

REPLICATION_FULL = "FULL_REPLICATION"
REPLICATION_PARTIAL = "PARTIAL_REPLICATION"
REPLICATION_UNAVAILABLE = "UNAVAILABLE"


class KeyesSignalError(ValueError):
    """Raised when a Keyes signal request is ill-formed or unsafe."""


@dataclass(frozen=True)
class ThresholdSpec:
    """One pre-registered threshold with its provenance recorded honestly."""

    variable: str
    threshold: float
    direction: str
    rationale: str
    pre_registered: bool = True
    is_proxy: bool = False

    def to_dict(self):
        return dict(self.__dict__)


# Pre-registered, non-fitted thresholds. The numeric cut-offs are declared here
# BEFORE any evaluation and are never adjusted against target data.
PRIMARY_THRESHOLDS = (
    ThresholdSpec(
        variable=X5,
        threshold=0.0,
        direction="min",
        rationale="five-year EPS growth must be positive; zero is the neutral economic line",
    ),
    ThresholdSpec(
        variable=X6,
        threshold=0.0,
        direction="min",
        rationale="five-year price gain must be positive; zero is the neutral economic line",
    ),
    ThresholdSpec(
        variable=X8,
        threshold=25.0,
        direction="max",
        rationale=(
            "P/E ceiling of 25 is a pre-registered absolute valuation cap, chosen without "
            "reference to outcomes; a cross-sectional percentile would be the forbidden quota"
        ),
    ),
    ThresholdSpec(
        variable=X9,
        threshold=1.0,
        direction="max",
        rationale=(
            "current P/E should not exceed its own historical normal multiple; 1.0 is the "
            "pre-registered neutral line, not a fitted cut"
        ),
    ),
    ThresholdSpec(
        variable=X12_PROXY,
        threshold=0.0,
        direction="min",
        rationale=(
            "with X12 forward expected appreciation unavailable, the proxy expectation leg "
            "requires positive five-year revenue growth; the proxy nature is disclosed"
        ),
        is_proxy=True,
    ),
)


def build_historical_rule_set(name="keyes_primary_v1", track=TRACK_HISTORICAL, thresholds=PRIMARY_THRESHOLDS):
    """Construct an explicit simultaneous :class:`KeyesRuleSet` (no quota)."""
    if not thresholds:
        raise KeyesSignalError("a Keyes rule set needs at least one criterion")
    criteria = tuple(
        KeyesCriterion(
            variable=spec.variable,
            threshold=float(spec.threshold),
            direction=spec.direction,
            is_proxy=bool(spec.is_proxy),
            proxy_note=(
                "%s is a proxy for %s and is never reported as the exact variable"
                % (spec.variable, VARIABLE_SPECS[spec.variable].base_variable)
            )
            if spec.is_proxy
            else "",
        )
        for spec in thresholds
    )
    return KeyesRuleSet(name=name, track=track, criteria=criteria)


# Single, module-level historical rule set so every caller shares one definition.
HISTORICAL_RULE_SETS = {"keyes_primary_v1": build_historical_rule_set()}


def historical_threshold_payload(thresholds=PRIMARY_THRESHOLDS):
    """Serialisable description of the pre-registered thresholds."""
    return {
        "signal_version": SIGNAL_VERSION,
        "track": TRACK_HISTORICAL,
        "thresholds": [spec.to_dict() for spec in thresholds],
        "quota": None,
        "weights": None,
        "notes": [
            "qualification requires EVERY criterion; the number of qualifiers is unconstrained",
            "no percentile, decile or top-30% selection is applied anywhere",
            "numeric cut-offs are pre-registered and never fitted against target data",
        ],
    }


def historical_signals(variable_frame, rule_set_name="keyes_primary_v1", expected_rows=None):
    """Evaluate the historical rule set for every observation.

    Returns a frame with ``qualified`` and a per-criterion contribution, plus the
    honest list of which criteria were unavailable. A component that is missing
    for an observation fails its criterion and is reported as unavailable for
    that row; the row is not dropped, because the qualification rate itself is a
    research result.
    """
    if rule_set_name not in HISTORICAL_RULE_SETS:
        raise KeyesSignalError("unknown historical rule set %r" % (rule_set_name,))
    rule_set = HISTORICAL_RULE_SETS[rule_set_name]
    required = ("security_id", "feature_asof", "variable", "value", "available")
    missing = [name for name in required if name not in variable_frame.columns]
    if missing:
        raise KeyesSignalError("variable frame is missing column(s): %s" % ", ".join(missing))

    variables = [criterion.variable for criterion in rule_set.criteria]
    frame = variable_frame.loc[variable_frame["variable"].isin(variables)].copy()
    duplicated = int(frame.duplicated(subset=["security_id", "feature_asof", "variable"]).sum())
    if duplicated:
        raise KeyesSignalError("variable frame has %d duplicated key rows" % duplicated)

    # ``set_index(...).unstack(...)`` is used instead of ``pivot_table`` on
    # purpose. An observation where EVERY Keyes variable is unavailable (for
    # example a pre-2009 row with no filed fundamentals) must still appear,
    # because its unavailability - and the resulting non-qualification - is
    # itself a research result. ``pivot_table`` silently DROPS such all-NaN rows,
    # and on pandas 3.0.6 ``pivot_table(dropna=False)`` instead fabricates a full
    # Cartesian product of the index levels (duplicating observations).
    # ``unstack`` returns exactly the observed key combinations, no more, no less.
    wide = frame.set_index(["security_id", "feature_asof", "variable"])["value"].unstack("variable")
    present = frame.set_index(["security_id", "feature_asof", "variable"])["available"].unstack("variable")
    wide = wide.reindex(columns=variables)
    present = present.reindex(columns=variables)
    if expected_rows is not None and len(wide) != int(expected_rows):
        raise KeyesSignalError(
            "expected %d observations but the variable frame carries %d" % (int(expected_rows), len(wide))
        )

    rows = []
    for key, values in wide.iterrows():
        security_id, feature_asof = key
        row_present = present.loc[key]
        evaluated = {}
        unavailable = []
        contributions = {}
        for criterion in rule_set.criteria:
            available = bool(row_present.get(criterion.variable)) and pd.notna(values.get(criterion.variable))
            if available:
                evaluated[criterion.variable] = float(values.get(criterion.variable))
            else:
                unavailable.append(criterion.variable)
            contributions[criterion.variable] = bool(available and criterion.evaluate(evaluated.get(criterion.variable)))
        outcome = qualify(rule_set, security_id, evaluated)
        rows.append({
            "security_id": security_id,
            "feature_asof": str(feature_asof)[:10],
            "track": TRACK_HISTORICAL,
            "rule_set": rule_set.name,
            "qualified": bool(outcome.qualified),
            "criteria_total": len(rule_set.criteria),
            "criteria_met": int(sum(1 for flag in contributions.values() if flag)),
            "criteria_available": int(len(rule_set.criteria) - len(unavailable)),
            "unavailable_components": ",".join(unavailable),
            "proxies_used": ",".join(outcome.proxies_used),
            "proxy_variables_evaluated": ",".join(
                criterion.variable for criterion in rule_set.criteria
                if criterion.is_proxy and contributions.get(criterion.variable)
            ),
            **{"pass_%s" % name: bool(contributions.get(name)) for name in variables},
        })
    output = pd.DataFrame(rows)
    if len(output):
        output = output.sort_values(["security_id", "feature_asof"], kind="mergesort").reset_index(drop=True)
    return output


@dataclass(frozen=True)
class CompositeConfig:
    """Disclosed parameters of the modern equal-weight rank composite."""

    components: tuple = HISTORICAL_COMPONENTS
    min_components: int = 3
    weighting: str = "EQUAL_WEIGHT_RANK"
    signal_version: str = SIGNAL_VERSION

    def to_dict(self):
        return dict(self.__dict__)


# Pre-registered sign per component: +1 favours high values, -1 favours low.
COMPONENT_SIGNS = {
    X5: 1.0,
    X6: 1.0,
    X8: -1.0,
    X9: -1.0,
    X12_PROXY: 1.0,
}


def _assert_labelled_time_order(frame):
    if "feature_asof" not in frame.columns:
        raise KeyesSignalError("a composite needs a feature_asof column")
    return True


def modern_composite(variable_frame, config=None):
    """Equal-weight, sign-aware cross-sectional rank composite (time-safe).

    Each component is ranked *within its own observation date*, which uses only
    information available at that date; negative-direction components are
    flipped. The composite is the equal-weight mean of the available components,
    and a row needs at least ``min_components`` of them. No weight is fitted and
    no global transform is estimated.
    """
    config = config or CompositeConfig()
    _assert_labelled_time_order(variable_frame)
    components = [name for name in config.components]
    unknown = [name for name in components if name not in COMPONENT_SIGNS]
    if unknown:
        raise KeyesSignalError("composite has unknown component(s): %s" % ", ".join(unknown))
    if config.weighting != "EQUAL_WEIGHT_RANK":
        raise KeyesSignalError(
            "only the pre-registered EQUAL_WEIGHT_RANK weighting is allowed; got %r" % config.weighting
        )

    frame = variable_frame.loc[variable_frame["variable"].isin(components)].copy()
    duplicated = int(frame.duplicated(subset=["security_id", "feature_asof", "variable"]).sum())
    if duplicated:
        raise KeyesSignalError("variable frame has %d duplicated key rows" % duplicated)

    # ``unstack`` (not pivot_table) keeps observations whose components are all
    # unavailable, so the modern composite honestly records "no score" instead of
    # erasing them, without the Cartesian blow-up pivot_table(dropna=False)
    # exhibits on pandas 3.0.6.
    wide = frame.set_index(["security_id", "feature_asof", "variable"])["value"].unstack("variable")
    wide = wide.reindex(columns=components)
    wide = wide.sort_index()

    grouped = wide.groupby(level="feature_asof", sort=False)
    ranked = pd.DataFrame(index=wide.index, columns=components, dtype="float64")
    for name in components:
        column = wide[name]
        pct = grouped[name].rank(pct=True, method="average")
        signed = pct if COMPONENT_SIGNS[name] > 0 else (1.0 - pct)
        ranked[name] = signed

    available = ranked.notna()
    count = available.sum(axis=1)
    composite = ranked.mean(axis=1, skipna=True)
    composite = composite.where(count >= int(config.min_components))

    output = ranked.copy()
    for name in components:
        output["rank_%s" % name] = ranked[name]
    output["components_available"] = count.astype(int)
    output["composite"] = composite
    output["track"] = TRACK_MODERN
    output["weighting"] = config.weighting
    output = output.reset_index()
    output["feature_asof"] = output["feature_asof"].astype(str).str.slice(0, 10)
    return output


def fit_train_only_weights(variable_frame, labelled_frame, train_asof_max, components=HISTORICAL_COMPONENTS,
                           min_periods=None):
    """Optional sign-agnostic weights fitted ONLY on a training window.

    This exists so a later model cannot silently hard-code invented weights. It
    is never the promoted composite: it consumes labels, so it is callable only
    from a training context, and the returned weights are reported with the
    window they came from. Returned weights are *not* sign-constrained.
    """
    if labelled_frame is None:
        raise KeyesSignalError("train-only weight fitting requires a labelled frame")
    for column in ("future_12m_excess_return", "target_observable"):
        if column not in labelled_frame.columns:
            raise KeyesSignalError("labelled frame is missing %s" % column)
    cutoff = str(train_asof_max)[:10]
    labels = labelled_frame.loc[
        labelled_frame["target_observable"].astype(bool)
        & (labelled_frame["feature_asof"].astype(str).str.slice(0, 10) <= cutoff),
        ["security_id", "feature_asof", "future_12m_excess_return"],
    ]
    composed = modern_composite(variable_frame, CompositeConfig(components=tuple(components)))
    merged = composed.merge(labels, on=["security_id", "feature_asof"], how="inner")
    usable = merged.loc[merged["composite"].notna()]
    if min_periods is not None and len(usable) < int(min_periods):
        raise KeyesSignalError("train-only fitting needs %d rows, found %d" % (int(min_periods), len(usable)))
    weights = {}
    for name in components:
        column = "rank_%s" % name
        series = usable[[column, "future_12m_excess_return"]].dropna()
        if len(series) < 2:
            weights[name] = 0.0
            continue
        correlation = float(series[column].corr(series["future_12m_excess_return"], method="spearman"))
        weights[name] = 0.0 if not np.isfinite(correlation) else correlation
    total = float(sum(abs(value) for value in weights.values()))
    if total > 0:
        weights = {name: value / total for name, value in weights.items()}
    return {
        "weights": weights,
        "train_asof_max": cutoff,
        "rows_used": int(len(usable)),
        "components": list(components),
        "note": "train-window rank-IC weights; never the promoted equal-weight composite",
    }


def direction_payload():
    """Serialisable statement of the pre-registered modern directions."""
    return {
        "signal_version": SIGNAL_VERSION,
        "track": TRACK_MODERN,
        "weighting": "EQUAL_WEIGHT_RANK",
        "components": list(HISTORICAL_COMPONENTS),
        "signs": dict(COMPONENT_SIGNS),
        "directions": dict(KEYES_DIRECTIONS),
        "min_components": CompositeConfig().min_components,
        "notes": [
            "equal weights; no 35/25/15-style hand tuning exists in this codebase",
            "ranking is cross-sectional within a single observation date (time-safe)",
            "missing components are excluded, never zero-filled",
        ],
    }
