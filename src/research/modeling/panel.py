"""WP6 modeling panel: the development-period, point-in-time modelling frame.

This module builds the WP6 panel by REUSING the exact WP5 feature computation
(:func:`src.research.discovery.panel.build_feature_panel`) and the WP5 target
attachment (:func:`src.research.discovery.builder.attach_targets`), then applying
the development-row restriction. It adds NO feature of its own; it only selects,
excludes and documents.

Enforced invariants (raise, never silently repair):

* no locked-holdout row (``feature_asof >= 2022-01-01``) ever enters;
* no embargo-band row ever enters;
* duplicate ``(security_id, feature_asof)`` keys are reported explicitly; a
  duplicate that would DISCARD an observable target-bearing row raises instead of
  being silently dropped (censored duplicates such as the AGN keys are reported
  and excluded honestly);
* the frozen exclusions (abnormal_volume, beta, x12_appreciation_proxy) are kept
  and reported, never resurrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery import builder as _builder
from ..discovery.panel import build_feature_panel
from ..holdout import embargo_cutoff, locked_holdout
from .contract import EXCLUDED_FEATURES, model_feature_universe

PANEL_COLUMNS = ("security_id", "ticker", "feature_asof", "modeling_month")
TARGET_COLUMNS = ("future_12m_excess_return", "outperform_12m", "target_observable",
                  "target_known_at", "target_end", "target_censor_reason")


class ModelingPanelError(RuntimeError):
    """Raised when the WP6 modelling panel cannot be assembled honestly."""


# ── Temporal guards (vectorised; equivalent to the WP5 row-wise masks) ────────

def _utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def holdout_and_embargo_counts(frame, holdout=None):
    """Count locked-holdout and embargo-band rows without reading any label."""
    holdout = holdout or locked_holdout()
    stamps = _utc(frame["feature_asof"])
    in_holdout = stamps >= holdout.holdout_start
    in_embargo = (stamps >= holdout.embargo_cutoff) & (stamps < holdout.holdout_start)
    return int(in_holdout.sum()), int(in_embargo.sum())


def assert_no_holdout_or_embargo(frame, holdout=None):
    """Raise when any locked-holdout or embargo-band row is present."""
    holdout_rows, embargo_rows = holdout_and_embargo_counts(frame, holdout)
    if holdout_rows:
        raise ModelingPanelError("panel contains %d locked-holdout row(s)" % holdout_rows)
    if embargo_rows:
        raise ModelingPanelError("panel contains %d embargo-band row(s)" % embargo_rows)
    return True


# ── Duplicate-key diagnostics (explicit, never silent) ───────────────────────

def duplicate_key_report(frame, key=("security_id", "feature_asof")):
    """Factual duplicate-key report; raises when an observable row would be lost."""
    keys = list(key)
    duplicated = frame.duplicated(subset=keys, keep=False)
    dup_rows = int(duplicated.sum())
    observable = frame["target_observable"].astype(bool) if "target_observable" in frame.columns \
        else pd.Series(False, index=frame.index)
    dup_observable = int((duplicated & observable).sum())
    dup_censored = int((duplicated & ~observable).sum())
    duplicate_groups = 0
    if dup_rows:
        duplicate_groups = int(frame.loc[duplicated, keys].drop_duplicates().shape[0])
    report = {
        "key": keys,
        "duplicate_rows": dup_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_observable_rows": dup_observable,
        "duplicate_censored_rows": dup_censored,
        "note": "duplicate keys with an observable label are refused; censored duplicates are excluded",
    }
    if dup_observable:
        raise ModelingPanelError(
            "%d duplicate (security_id, feature_asof) row(s) carry an observable label; "
            "refusing to silently drop target-bearing rows" % dup_observable
        )
    return report


# ── Panel assembly ───────────────────────────────────────────────────────────

def assemble_from_feature_frame(feature_frame, targets, holdout=None):
    """Attach certified targets, restrict to development rows and de-duplicate.

    Returns ``(frame, diagnostics)``. ``feature_frame`` must already be the WP5
    point-in-time feature frame (development-restricted or not).
    """
    labelled = _builder.attach_targets(feature_frame, targets)
    if "feature_asof" not in labelled.columns:
        raise ModelingPanelError("labelled frame lacks feature_asof")

    raw_duplicate_report = duplicate_key_report(labelled)

    deduped = labelled.drop_duplicates(subset=["security_id", "feature_asof"], keep="first")
    development = _builder.development_rows(deduped)
    if development.empty:
        raise ModelingPanelError("no development observations are available for modelling")
    assert_no_holdout_or_embargo(development, holdout)

    frame = development.copy()
    frame["feature_asof"] = _utc(frame["feature_asof"]).dt.strftime("%Y-%m-%d")
    frame["modeling_month"] = pd.to_datetime(frame["feature_asof"]).dt.to_period("M").astype(str)
    frame = frame.sort_values(["feature_asof", "security_id"], kind="mergesort").reset_index(drop=True)

    diagnostics = panel_diagnostics(frame, duplicate_report=raw_duplicate_report)
    diagnostics["eligible_panel_rows"] = int(len(labelled))
    diagnostics["rows_after_dedupe"] = int(len(deduped))
    diagnostics["development_rows"] = int(len(development))
    return frame, diagnostics


def build_modeling_panel(panel, targets, prices, actions, fundamentals, cik_by_ticker,
                         benchmark_prices, benchmark_actions, config=None, holdout=None):
    """Materialise the WP6 modeling panel from the certified WP2C/WP3 inputs.

    The feature values are produced by the EXACT WP5 engine; WP6 adds no feature
    computation. ``panel`` must already expose ``feature_asof``.
    """
    feature_frame, panel_summary = build_feature_panel(
        panel, prices, actions, fundamentals, cik_by_ticker,
        benchmark_prices, benchmark_actions, restrict_to_development=True,
    )
    frame, diagnostics = assemble_from_feature_frame(feature_frame, targets, holdout=holdout)
    diagnostics["wp5_panel_summary"] = {
        "rows": panel_summary.get("rows"),
        "eligible_panel_rows": panel_summary.get("eligible_panel_rows"),
        "embargo_cutoff": panel_summary.get("embargo_cutoff"),
        "locked_holdout_rows_in_panel": panel_summary.get("locked_holdout_rows_in_panel"),
    }
    return frame, diagnostics


# ── Diagnostics ──────────────────────────────────────────────────────────────

def _coverage(frame, names):
    records = {}
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
            present = int(values.notna().sum())
            records[name] = {
                "rows": int(len(frame)),
                "present": present,
                "coverage": float(present / len(frame)) if len(frame) else 0.0,
            }
        else:
            records[name] = {"rows": int(len(frame)), "present": 0, "coverage": 0.0}
    return records


def panel_diagnostics(frame, duplicate_report=None):
    """Factual WP6 panel diagnostics (no labels are used for any choice)."""
    eligible = model_feature_universe()
    months = sorted(frame["modeling_month"].unique())
    month_counts = {str(month): int(count) for month, count in
                    frame.groupby("modeling_month").size().items()}
    counts = np.asarray(list(month_counts.values()), dtype="float64") if month_counts else np.array([0.0])

    regression = pd.to_numeric(frame.get("future_12m_excess_return"), errors="coerce")
    classification = pd.to_numeric(frame.get("outperform_12m"), errors="coerce")
    balance = {
        "rows": int(len(frame)),
        "regression_mean": float(regression.mean()) if regression.notna().any() else None,
        "regression_std": float(regression.std(ddof=1)) if regression.notna().sum() > 1 else None,
        "classification_rate": float(classification.mean()) if classification.notna().any() else None,
        "classification_positive": int(classification.sum()) if classification.notna().any() else 0,
    }
    return {
        "rows": int(len(frame)),
        "securities": int(frame["security_id"].nunique()),
        "date_min": str(frame["feature_asof"].min()),
        "date_max": str(frame["feature_asof"].max()),
        "months": int(len(months)),
        "month_min": months[0] if months else None,
        "month_max": months[-1] if months else None,
        "monthly_counts": month_counts,
        "monthly_count_min": int(counts.min()),
        "monthly_count_median": float(np.median(counts)),
        "monthly_count_max": int(counts.max()),
        "feature_coverage": _coverage(frame, eligible),
        "excluded_features": {
            name: {
                "present": bool(name in frame.columns),
                "coverage": _coverage(frame, [name])[name]["coverage"],
            }
            for name in EXCLUDED_FEATURES
        },
        "label_balance": balance,
        "duplicate_key_check": duplicate_report or duplicate_key_report(frame),
        "embargo_cutoff": str(embargo_cutoff())[:19],
    }
