"""WP4 Keyes diagnostics: component power, stability, baselines and placebos.

Everything here is descriptive and chronological. Nothing is fitted, nothing is
shuffled except by the explicit placebo tests, and no result is computed over a
censored label: an observation contributes only where ``target_observable`` is
true, so a 12-month outcome that would not yet have been known at the prediction
date can never lift or lower a reported number.

The diagnostics deliberately separate:

* **component power** - does a Keyes variable move with the benchmark-relative
  outcome, measured by rank IC and by the spread between signal states;
* **stability** - is that relationship consistent across calendar years (and
  across sectors where the sector map actually covers the universe);
* **baselines** - a composite is only interesting if it beats an equal-weight
  universe, a valuation-only and a momentum-only rule, none of which is tuned;
* **placebos** - a shuffled target, a shifted target, a leaked future feature and
  an impossible signal date should not reproduce the observed power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import HISTORICAL_COMPONENTS, SIGNAL_VERSION
from .variables import (
    FIDELITY_EXACT,
    FIDELITY_PROXY,
    FIDELITY_UNAVAILABLE,
    VARIABLE_SPECS,
    X5,
    X6,
    X8,
    X9,
    X12_PROXY,
)

DIAGNOSTIC_VERSION = "v2_wp4_keyes_diagnostics_v1"

CONTINUOUS_TARGET = "future_12m_excess_return"
CLASSIFICATION_TARGET = "outperform_12m"
OBSERVABLE_FLAG = "target_observable"

MIN_YEAR_ROWS = 20
MIN_SECTOR_ROWS = 20
MIN_IC_ROWS = 30


class DiagnosticError(ValueError):
    """Raised when a WP4 diagnostic request is ill-formed."""


# ── Observability gate ────────────────────────────────────────────────────────

def observable_slice(frame, require_label=True):
    """Return only rows whose label was genuinely observable at the cutoff.

    Censored rows are excluded, never treated as zero and never as a negative
    outcome. ``require_label`` additionally requires a finite continuous target.
    """
    if OBSERVABLE_FLAG not in frame.columns:
        raise DiagnosticError("frame is missing %s; censoring cannot be enforced" % OBSERVABLE_FLAG)
    mask = frame[OBSERVABLE_FLAG].astype(bool)
    if require_label:
        if CONTINUOUS_TARGET not in frame.columns:
            raise DiagnosticError("frame is missing %s" % CONTINUOUS_TARGET)
        mask = mask & pd.to_numeric(frame[CONTINUOUS_TARGET], errors="coerce").notna()
    return frame.loc[mask].copy()


def rank_ic(frame, signal_column, target_column=CONTINUOUS_TARGET):
    """Spearman rank correlation between a signal and the outcome.

    Returns ``(ic, n)``; ``ic`` is None when the sample is too small or one side
    has no rank variance. The correlation is computed on ranks, so it is robust
    to the heavy tails of 12-month excess returns.
    """
    paired = frame[[signal_column, target_column]].dropna()
    if len(paired) < MIN_IC_ROWS:
        return None, int(len(paired))
    signal_rank = paired[signal_column].rank()
    target_rank = paired[target_column].rank()
    if signal_rank.nunique() < 2 or target_rank.nunique() < 2:
        return None, int(len(paired))
    value = float(signal_rank.corr(target_rank))
    if not np.isfinite(value):
        return None, int(len(paired))
    return value, int(len(paired))


def _year_of(values):
    return pd.to_datetime(pd.Series(list(values)), errors="coerce").dt.year


# ── Component diagnostics ─────────────────────────────────────────────────────

def component_diagnostics(variable_frame, targets, signal_year=None):
    """Per-component association with the observable benchmark-relative target.

    Signal state is a *signed quintile* for continuous variables: the sign is the
    pre-registered direction, so ``signal_state`` 0 is the least favourable and 4
    the most favourable. The top-minus-bottom excess spread and the outperform
    rate difference are the headline quantities; rank IC summarises the ordering.
    """
    frame = observable_slice(targets)
    key = ["security_id", "feature_asof"]
    if "feature_asof" not in frame.columns:
        raise DiagnosticError("targets frame needs feature_asof to align with variables")
    frame = frame.rename(columns={c: c for c in frame.columns})
    label = frame[key + [CONTINUOUS_TARGET, CLASSIFICATION_TARGET]].copy()

    records = []
    for variable, chunk in variable_frame.groupby("variable"):
        available = chunk.loc[chunk["available"].astype(bool), key + ["value"]].copy()
        available = available.drop_duplicates(subset=key)
        merged = available.merge(label, on=key, how="inner")
        total_rows = int(len(chunk.drop_duplicates(subset=key)))
        available_rows = int(len(available))
        usable = merged.loc[merged["value"].notna() & merged[CONTINUOUS_TARGET].notna()]
        entry = {
            "variable": variable,
            "is_proxy": bool(VARIABLE_SPECS[variable].is_proxy) if variable in VARIABLE_SPECS else None,
            "direction": VARIABLE_SPECS[variable].hypothesized_direction if variable in VARIABLE_SPECS else None,
            "observations": total_rows,
            "available": available_rows,
            "available_rate": (available_rows / total_rows) if total_rows else None,
            "matched": int(len(usable)),
        }
        ic, ic_n = rank_ic(usable, "value")
        entry["rank_ic"] = ic
        entry["rank_ic_n"] = ic_n
        entry.update(_state_spread(usable, "value", entry["direction"]))
        records.append(entry)
    return pd.DataFrame(records)


def _state_spread(usable, value_column, direction):
    """Quintile state spread; the sign is the pre-registered direction."""
    blank = {
        "quintiles_used": 0,
        "top_state_n": 0,
        "bottom_state_n": 0,
        "mean_excess_top": None,
        "mean_excess_bottom": None,
        "median_excess_top": None,
        "median_excess_bottom": None,
        "excess_spread": None,
        "outperform_rate_top": None,
        "outperform_rate_bottom": None,
        "outperform_spread": None,
    }
    if len(usable) < MIN_IC_ROWS:
        return blank
    signed = pd.to_numeric(usable[value_column], errors="coerce")
    if direction == "NEGATIVE":
        signed = -signed
    valid = signed.notna() & usable[CONTINUOUS_TARGET].notna()
    working = usable.loc[valid].copy()
    working["__signed"] = signed[valid]
    if working["__signed"].nunique() < 5:
        return blank
    try:
        working["__state"] = pd.qcut(working["__signed"], 5, labels=False, duplicates="drop")
    except ValueError:
        return blank
    if working["__state"].nunique() < 2:
        return blank
    top = working.loc[working["__state"] == working["__state"].max()]
    bottom = working.loc[working["__state"] == working["__state"].min()]

    def _rate(group):
        values = pd.to_numeric(group[CLASSIFICATION_TARGET], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None

    return {
        "quintiles_used": int(working["__state"].nunique()),
        "top_state_n": int(len(top)),
        "bottom_state_n": int(len(bottom)),
        "mean_excess_top": float(top[CONTINUOUS_TARGET].mean()) if len(top) else None,
        "mean_excess_bottom": float(bottom[CONTINUOUS_TARGET].mean()) if len(bottom) else None,
        "median_excess_top": float(top[CONTINUOUS_TARGET].median()) if len(top) else None,
        "median_excess_bottom": float(bottom[CONTINUOUS_TARGET].median()) if len(bottom) else None,
        "excess_spread": (float(top[CONTINUOUS_TARGET].mean() - bottom[CONTINUOUS_TARGET].mean())
                          if len(top) and len(bottom) else None),
        "outperform_rate_top": _rate(top),
        "outperform_rate_bottom": _rate(bottom),
        "outperform_spread": (
            (_rate(top) - _rate(bottom)) if _rate(top) is not None and _rate(bottom) is not None else None
        ),
    }


def yearly_component_stability(variable_frame, targets):
    """Rank IC and spread per component per calendar year (chronological)."""
    frame = observable_slice(targets)
    key = ["security_id", "feature_asof"]
    frame = frame[key + [CONTINUOUS_TARGET, CLASSIFICATION_TARGET]].copy()
    frame["__year"] = _year_of(frame["feature_asof"])
    records = []
    for variable, chunk in variable_frame.groupby("variable"):
        available = chunk.loc[chunk["available"].astype(bool), key + ["value"]].drop_duplicates(subset=key)
        merged = available.merge(frame, on=key, how="inner")
        direction = VARIABLE_SPECS[variable].hypothesized_direction if variable in VARIABLE_SPECS else None
        for year, group in merged.groupby("__year", sort=True):
            usable = group.loc[group["value"].notna() & group[CONTINUOUS_TARGET].notna()]
            ic, ic_n = rank_ic(usable, "value")
            spread = _state_spread(usable, "value", direction) if len(usable) >= MIN_YEAR_ROWS else {}
            records.append({
                "variable": variable,
                "year": int(year),
                "rows": int(len(usable)),
                "rank_ic": ic,
                "rank_ic_n": ic_n,
                "excess_spread": spread.get("excess_spread"),
                "outperform_spread": spread.get("outperform_spread"),
                "sufficient_sample": bool(len(usable) >= MIN_YEAR_ROWS),
            })
    return pd.DataFrame(records)


def sector_component_stability(variable_frame, targets, sector_by_ticker):
    """Sector splits, reported only where the sector map actually covers names.

    The committed sector map covers a small fraction of the universe, so this
    returns an explicit ``INSUFFICIENT_SECTOR_COVERAGE`` status rather than a
    misleading handful of rows presented as a sector result.
    """
    coverage = 0
    if "ticker" in targets.columns and sector_by_ticker:
        tickers = targets["ticker"].astype(str).str.upper()
        coverage = int(tickers.isin({str(k).upper() for k in sector_by_ticker}).mean() * 1000) / 1000.0
    status = "INSUFFICIENT_SECTOR_COVERAGE" if coverage < 0.5 else "OK"
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": status,
        "sector_coverage_fraction": coverage,
        "reason": (
            "the certified universe has no sector column and the local configuration covers only a "
            "minority of names, so no credible sector stability result can be published"
            if status != "OK" else "sector map covers the majority of the universe"
        ),
        "rows": [],
    }


# ── Composite diagnostics and baselines ───────────────────────────────────────

def _join_composite_and_targets(composite, targets):
    frame = observable_slice(targets)
    key = ["security_id", "feature_asof"]
    frame = frame[key + [CONTINUOUS_TARGET, CLASSIFICATION_TARGET]].copy()
    merged = composite.merge(frame, on=key, how="inner")
    return merged.loc[merged["composite"].notna()]


def composite_diagnostics(composite, targets, variable_frame=None):
    """Headline composite result plus untuned baselines for context.

    Baselines are deliberately simple and non-optimised: the equal-weight
    universe (every observable row, no signal), a valuation-only rule (X8/X9
    ranks) and a momentum-only rule (X6 rank). They exist so the composite is not
    credited for a result any trivial rule already has.
    """
    merged = _join_composite_and_targets(composite, targets)
    merged = merged.sort_values(["feature_asof", "security_id"], kind="mergesort")
    ic, ic_n = rank_ic(merged, "composite")
    entry = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "signal_version": SIGNAL_VERSION,
        "rows": int(len(merged)),
        "rank_ic": ic,
        "rank_ic_n": ic_n,
        "composite_excess_spread": _state_spread(merged, "composite", "POSITIVE").get("excess_spread"),
        "composite_outperform_top": _state_spread(merged, "composite", "POSITIVE").get("outperform_rate_top"),
        "universe_mean_excess": float(merged[CONTINUOUS_TARGET].mean()) if len(merged) else None,
        "universe_outperform_rate": float(merged[CLASSIFICATION_TARGET].mean()) if len(merged) else None,
    }
    entry["baselines"] = _baseline_table(merged, composite, variable_frame, targets)
    entry["yearly"] = _composite_yearly(merged)
    entry["qualification_by_year"] = None  # filled by the build script with the historical track
    return entry, merged


def _baseline_table(merged, composite, variable_frame, targets):
    baselines = {
        "equal_weight_universe": {
            "rows": int(len(merged)),
            "mean_excess": float(merged[CONTINUOUS_TARGET].mean()) if len(merged) else None,
            "outperform_rate": float(merged[CLASSIFICATION_TARGET].mean()) if len(merged) else None,
            "note": "every observable row; no signal, no selection",
        }
    }
    if variable_frame is None:
        return baselines
    key = ["security_id", "feature_asof"]
    for name, components, direction in (
        ("valuation_only", (X8, X9), "NEGATIVE"),
        ("momentum_only", (X6,), "POSITIVE"),
        ("earnings_growth_only", (X5,), "POSITIVE"),
    ):
        subset = variable_frame.loc[variable_frame["variable"].isin(components) &
                                    variable_frame["available"].astype(bool)]
        subset = subset[key + ["variable", "value"]].copy()
        if subset.empty:
            baselines[name] = {"rows": 0, "mean_excess": None, "outperform_rate": None,
                               "note": "no available observations; baseline not computable"}
            continue
        # ``unstack`` keeps exactly the observed keys: no all-NaN observation is
        # dropped and no Cartesian product is fabricated (pivot_table behaves
        # either way wrongly on pandas 3.0.6).
        wide = subset.set_index(list(key) + ["variable"])["value"].unstack("variable").reset_index()
        wide["score"] = wide[list(components)].mean(axis=1, skipna=True)
        if direction == "NEGATIVE":
            wide["score"] = -wide["score"]
        # Cross-sectional rank per date, equal weight, no fitting.
        wide["rank"] = wide.groupby("feature_asof")["score"].rank(pct=True)
        joined = wide.merge(targets[key + [CONTINUOUS_TARGET, CLASSIFICATION_TARGET, OBSERVABLE_FLAG]],
                            on=key, how="inner")
        joined = joined.loc[joined[OBSERVABLE_FLAG].astype(bool)]
        top = joined.loc[joined["rank"] >= 0.8]
        bottom = joined.loc[joined["rank"] <= 0.2]
        baselines[name] = {
            "rows": int(len(joined)),
            "mean_excess": float(joined[CONTINUOUS_TARGET].mean()) if len(joined) else None,
            "outperform_rate": float(joined[CLASSIFICATION_TARGET].mean()) if len(joined) else None,
            "top_quintile_mean_excess": float(top[CONTINUOUS_TARGET].mean()) if len(top) else None,
            "top_minus_bottom_excess": (
                (float(top[CONTINUOUS_TARGET].mean() - bottom[CONTINUOUS_TARGET].mean())
                 if len(top) and len(bottom) else None)
            ),
            "note": "cross-sectional rank, equal weight, pre-registered direction; no tuning",
        }
    return baselines


def _composite_yearly(merged):
    records = []
    for year, group in merged.groupby(_year_of(merged["feature_asof"]), sort=True):
        ic, ic_n = rank_ic(group, "composite")
        records.append({
            "year": int(year),
            "rows": int(len(group)),
            "rank_ic": ic,
            "rank_ic_n": ic_n,
            "mean_excess": float(group[CONTINUOUS_TARGET].mean()) if len(group) else None,
            "outperform_rate": float(group[CLASSIFICATION_TARGET].mean()) if len(group) else None,
        })
    return records


def qualification_summary(historical, targets):
    """Qualification counts and rates, by year and overall, for the historical track."""
    frame = observable_slice(targets)
    key = ["security_id", "feature_asof"]
    merged = historical.merge(frame[key + [CONTINUOUS_TARGET, CLASSIFICATION_TARGET]], on=key, how="left")
    merged["__year"] = _year_of(merged["feature_asof"])
    overall = {
        "rows": int(len(merged)),
        "qualified": int(merged["qualified"].sum()),
        "qualification_rate": float(merged["qualified"].mean()) if len(merged) else None,
        "min_qualified_in_a_year": None,
        "max_qualified_in_a_year": None,
    }
    by_year = []
    for year, group in merged.groupby("__year", sort=True):
        by_year.append({
            "year": int(year),
            "rows": int(len(group)),
            "qualified": int(group["qualified"].sum()),
            "qualification_rate": float(group["qualified"].mean()) if len(group) else None,
            "mean_excess_qualified": float(group.loc[group["qualified"], CONTINUOUS_TARGET].mean())
            if group["qualified"].any() else None,
            "mean_excess_unqualified": float(group.loc[~group["qualified"], CONTINUOUS_TARGET].mean())
            if (~group["qualified"]).any() else None,
        })
    if by_year:
        counts = [entry["qualified"] for entry in by_year]
        overall["min_qualified_in_a_year"] = int(min(counts))
        overall["max_qualified_in_a_year"] = int(max(counts))
    return overall, pd.DataFrame(by_year)


def concentration_and_turnover(historical, composite):
    """Concentration of qualifications per date and composite turnover.

    Turnover here is the mean symmetric difference of the composite's top decile
    between consecutive observation dates, divided by the top-decile size; it is
    a diagnostic of how much the ranking churns, not a trading simulation.
    """
    counts = historical.groupby("feature_asof")["qualified"].sum()
    concentration = {
        "dates": int(counts.shape[0]),
        "mean_qualifiers_per_date": float(counts.mean()) if len(counts) else None,
        "median_qualifiers_per_date": float(counts.median()) if len(counts) else None,
        "max_qualifiers_per_date": int(counts.max()) if len(counts) else None,
        "min_qualifiers_per_date": int(counts.min()) if len(counts) else None,
        "dates_with_zero_qualifiers": int((counts == 0).sum()),
    }

    ordered = composite.sort_values(["feature_asof", "composite"], kind="mergesort")
    turnover = {"dates": 0, "mean_top_decile_turnover": None}
    tops = []
    for date, group in ordered.groupby("feature_asof", sort=True):
        ranked = group.loc[group["composite"].notna(), ["security_id", "composite"]]
        if len(ranked) < 10:
            tops.append((str(date)[:10], set()))
            continue
        cutoff = ranked["composite"].quantile(0.9)
        tops.append((str(date)[:10], set(ranked.loc[ranked["composite"] >= cutoff, "security_id"])))
    changes = []
    for (previous_date, previous), (current_date, current) in zip(tops, tops[1:]):
        if not previous or not current:
            continue
        symmetric = len(previous.symmetric_difference(current)) / max(len(previous), 1)
        changes.append(symmetric)
    if changes:
        turnover = {"dates": len(changes), "mean_top_decile_turnover": float(np.mean(changes))}
    return concentration, turnover


def _first_per_date(frame, column="composite"):
    ordered = frame.sort_values(["feature_asof", column], kind="mergesort")
    return ordered


# ── Fidelity scorecard ────────────────────────────────────────────────────────

def fidelity_table(coverage):
    """Per-variable replication fidelity, honest about what is unavailable.

    A variable is UNAVAILABLE when it never produced a single real value for the
    universe; otherwise its declared fidelity stands. The ``X12_PROXY`` row is
    the only PROXY, and it is never promoted to an exact X12.
    """
    variables = coverage.get("variables", {})
    records = []
    for variable in (X5, X6, X8, X9, X12_PROXY):
        spec = VARIABLE_SPECS[variable]
        stats = variables.get(variable, {})
        available = int(stats.get("available", 0))
        declared = spec.fidelity
        fidelity = FIDELITY_UNAVAILABLE if available == 0 else declared
        records.append({
            "variable": variable,
            "base_variable": spec.base_variable,
            "is_proxy": bool(spec.is_proxy),
            "declared_fidelity": declared,
            "fidelity": fidelity,
            "available_rows": available,
            "rows": int(stats.get("rows", 0)),
            "coverage": stats.get("coverage"),
            "reason": spec.fidelity_reason,
            "source": spec.source,
            "availability_rule": spec.availability_rule,
        })
    return pd.DataFrame(records)


def replication_status(fidelity_frame):
    """Overall replication status derived from the fidelity table.

    * FULL_REPLICATION - every variable is EXACT or CLOSE_EQUIVALENT and none is a proxy
    * PARTIAL_REPLICATION - at least one variable is a proxy or unavailable, but
      the track is still computable for a meaningful share of the universe
    * UNAVAILABLE - nothing usable was computed
    """
    exact_like = {FIDELITY_EXACT, "CLOSE_EQUIVALENT"}
    fidelities = list(fidelity_frame["fidelity"])
    if all(value in exact_like for value in fidelities):
        return "FULL_REPLICATION"
    if all(value == FIDELITY_UNAVAILABLE for value in fidelities):
        return "UNAVAILABLE"
    return "PARTIAL_REPLICATION"


# ── Placebo and leakage checks ────────────────────────────────────────────────

def placebo_checks(composite, targets, seed=20260926):
    """Leakage/placebo battery; a shuffled target must lose its systematic power.

    The checks are:

    * ``shuffled_target`` - permute the outcome within the sample; rank IC must
      collapse toward zero (no look-ahead survives a permutation);
    * ``future_shifted_target`` - replace the target with a horizon shifted one
      year later; a genuine point-in-time signal should materially degrade;
    * ``feature_timestamp_violation`` - count feature inputs whose evidence date
      is later than their prediction date; must be zero;
    * ``impossible_signal_date`` - count signals dated on/after the target start
      in a way that would require knowing the future; must be zero.
    """
    merged = _join_composite_and_targets(composite, targets)
    rng = np.random.default_rng(int(seed))
    shuffled = merged.copy()
    shuffled[CONTINUOUS_TARGET] = rng.permutation(shuffled[CONTINUOUS_TARGET].to_numpy())
    shuffled_ic, shuffled_n = rank_ic(shuffled, "composite")
    true_ic, true_n = rank_ic(merged, "composite")

    shifted_ic, shifted_n = (None, 0)
    shifted_errors = "target_known_at not present on the target frame"
    if "target_known_at" in targets.columns:
        shifted_errors = None
        frame = observable_slice(targets)[["security_id", "feature_asof", CONTINUOUS_TARGET]].copy()
        frame = frame.sort_values(["security_id", "feature_asof"], kind="mergesort")
        frame["__shifted"] = frame.groupby("security_id")[CONTINUOUS_TARGET].shift(-1)
        shifted = composite.merge(frame, on=["security_id", "feature_asof"], how="inner")
        shifted = shifted.loc[shifted["composite"].notna() & shifted["__shifted"].notna()]
        shifted_ic, shifted_n = rank_ic(shifted, "composite", "__shifted")

    timestamp_violations = int((pd.to_datetime(targets["feature_asof"], errors="coerce") >
                                pd.to_datetime(targets.get("target_start", targets["feature_asof"]),
                                               errors="coerce")).sum()) \
        if "target_start" in targets.columns else 0
    impossible = int((pd.to_datetime(composite["feature_asof"], errors="coerce").dt.year >
                      int(pd.Timestamp.now().year) + 1).sum())

    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "seed": int(seed),
        "true_rank_ic": true_ic,
        "true_rank_ic_n": true_n,
        "shuffled_rank_ic": shuffled_ic,
        "shuffled_rank_ic_n": shuffled_n,
        "shuffled_retains_systematic_power": bool(
            shuffled_ic is not None and true_ic is not None and abs(shuffled_ic) > 0.5 * max(abs(true_ic), 1e-9)
        ),
        "shifted_target_rank_ic": shifted_ic,
        "shifted_target_rank_ic_n": shifted_n,
        "shifted_target_note": shifted_errors,
        "feature_timestamp_violations": timestamp_violations,
        "impossible_signal_date_violations": impossible,
        "notes": [
            "a shuffled target has no forward information, so any retained power would indicate a leak",
            "the shifted target tests sensitivity of the signal to the exact horizon",
            "violation counters must be zero for the run to be admissible",
        ],
    }
