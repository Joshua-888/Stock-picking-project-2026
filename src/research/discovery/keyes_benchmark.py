"""WP5 Keyes benchmark comparison.

The corrected WP4 Keyes variables (X5, X6, X8, X9, X12_PROXY), the historical
Keyes rule signal and the modern Keyes composite are evaluated with the SAME
cross-sectional IC machinery used for every discovery candidate. The Keyes track
is therefore a reference point, never a privileged one: its variables are placed
beside the discovery features and compared like any other candidate.

Nothing here reimplements the Keyes engine - the caller supplies the engine's
output (variable frame and signals) and this module only measures it.
"""

from __future__ import annotations

from .cross_sectional import cross_sectional_ic_series, summarize_ic_series
from .inference import dependence_diagnostics, hac_mean_test


def _merged(frame, targets, columns):
    keys = ["security_id", "feature_asof"]
    keep = keys + [name for name in columns if name in frame.columns]
    left = frame.loc[:, sorted(set(keep))].copy()
    available = [name for name in ("future_12m_excess_return", "outperform_12m", "target_observable")
                 if name in targets.columns]
    right = targets.loc[:, keys + available].copy()
    return left.merge(right, on=keys, how="inner")


def _measure(merged, feature, config):
    series, diagnostics = cross_sectional_ic_series(
        merged, feature, min_observations=config.min_ic_observations,
    )
    summary = summarize_ic_series(series, min_months=config.min_months_for_stability)
    values = series["rank_ic"].to_numpy(dtype="float64") if len(series) else series
    hac = hac_mean_test(values, lags=config.hac_lags) if len(series) else {"p_value": None}
    dependence = dependence_diagnostics(values, config=config) if len(series) else {"lag1": None}
    return {"feature": feature, "summary": summary, "hac": hac,
            "dependence": dependence, "diagnostics": diagnostics}


def keyes_variable_ic(variable_frame, targets, config, variables=None):
    """Measure each Keyes variable with the discovery cross-sectional IC.

    ``variable_frame`` may be either the long WP4 variable frame (columns
    ``security_id``, ``feature_asof``, ``variable``, ``value``) or a wide frame
    already carrying the variables as columns.
    """
    variables = list(variables or config.keyes_benchmark)
    if {"security_id", "feature_asof", "variable", "value"}.issubset(variable_frame.columns):
        wide = variable_frame.pivot_table(
            index=["security_id", "feature_asof"], columns="variable", values="value",
        ).reset_index()
    else:
        wide = variable_frame.copy()
    merged = _merged(wide, targets, variables)
    results = {}
    for name in variables:
        if name not in merged.columns:
            results[name] = {"feature": name, "summary": summarize_ic_series(None),
                             "hac": {"p_value": None}, "dependence": {"lag1": None},
                             "diagnostics": {"dates_evaluated": 0, "dates_skipped": 0, "skipped": []}}
            continue
        results[name] = _measure(merged, name, config)
    return results


def keyes_signal_ic(signal_frame, targets, config, columns=("composite", "qualified")):
    """Measure the modern composite and/or historical rule signal."""
    present = [name for name in columns if name in signal_frame.columns]
    merged = _merged(signal_frame, targets, present)
    results = {}
    for name in present:
        if name == "qualified":
            merged[name] = merged[name].astype(float)
        results[name] = _measure(merged, name, config)
    return results


def keyes_benchmark_payload(variable_frame, targets, modern_signals, historical_signals, config,
                            discovery_rank=None):
    """Assemble the Keyes benchmark comparison block.

    ``discovery_rank`` optionally maps discovery feature name -> mean monthly IC so
    the reader can see where the Keyes variables sit relative to the discovery
    candidates without either side being privileged.
    """
    variables = keyes_variable_ic(variable_frame, targets, config)
    signals = {}
    if modern_signals is not None and len(modern_signals):
        signals.update(keyes_signal_ic(modern_signals, targets, config, columns=("composite",)))
    if historical_signals is not None and len(historical_signals):
        signals.update(keyes_signal_ic(historical_signals, targets, config, columns=("qualified",)))

    ranking = {}
    for name, record in list(variables.items()) + list(signals.items()):
        ranking[name] = {
            "mean_ic": record["summary"].get("mean_ic"),
            "months": record["summary"].get("months"),
            "hac_p_value": record["hac"].get("p_value"),
        }
    return {
        "keyes_variables": variables,
        "keyes_signals": signals,
        "summary": ranking,
        "discovery_mean_ic": discovery_rank or {},
        "note": "Keyes variables and signals are measured with the same machinery as discovery features and are not privileged",
    }
