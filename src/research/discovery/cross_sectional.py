"""WP5 cross-sectional rank IC.

The PRIMARY discovery statistic is the cross-sectional Spearman rank IC computed
at EACH feature ``asof`` date. Pooling rows across dates would treat heavily
overlapping 12-month labels (WP4 measured ~11.8x overlap) as independent and
would overstate significance, so a pooled correlation is, at most, a secondary
figure and never the basis of an inference claim.

For every date the module requires a minimum number of paired observations and a
non-degenerate rank profile; a date that fails is reported as skipped with its
reason rather than silently contributing a flattering number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IC_SERIES_COLUMNS = ("feature_asof", "paired_obs", "rank_ic")


class CrossSectionalICError(ValueError):
    """Raised when a cross-sectional IC request is ill-formed."""


def _ranks(values):
    """Average ranks with ties sharing the mean rank (deterministic)."""
    values = np.asarray(values, dtype="float64")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    sorted_values = values[order]
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def spearman(x, y):
    """Spearman correlation; ``None`` when either side has no rank variance."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if len(x) != len(y) or len(x) < 2:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    rx = _ranks(x)
    ry = _ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cross_sectional_ic_series(frame, feature, target="future_12m_excess_return",
                              asof_col="feature_asof", min_observations=30):
    """Monthly cross-sectional rank IC series for one feature.

    Returns ``(series, diagnostics)``: ``series`` is a date-sorted frame with the
    paired count and rank IC per ``feature_asof``; ``diagnostics`` records every
    skipped date and why, so a thin or degenerate month is visible, not hidden.
    """
    if feature not in frame.columns:
        raise CrossSectionalICError("feature %r absent from frame" % (feature,))
    if target not in frame.columns:
        raise CrossSectionalICError("target %r absent from frame" % (target,))
    if asof_col not in frame.columns:
        raise CrossSectionalICError("asof column %r absent from frame" % (asof_col,))

    working = frame.loc[:, [asof_col, feature, target]].copy()
    working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[asof_col, feature, target])

    records = []
    skipped = []
    for asof, chunk in working.groupby(asof_col, sort=True):
        paired = int(len(chunk))
        if paired < int(min_observations):
            skipped.append({"feature_asof": str(asof)[:10], "paired_obs": paired,
                            "reason": "below_min_observations"})
            continue
        ic = spearman(chunk[feature].to_numpy(), chunk[target].to_numpy())
        if ic is None or not np.isfinite(ic):
            skipped.append({"feature_asof": str(asof)[:10], "paired_obs": paired,
                            "reason": "no_rank_variance"})
            continue
        records.append({"feature_asof": str(asof)[:10], "paired_obs": paired, "rank_ic": ic})

    series = pd.DataFrame(records, columns=list(IC_SERIES_COLUMNS))
    if not series.empty:
        series = series.sort_values("feature_asof", kind="mergesort").reset_index(drop=True)

    diagnostics = {
        "feature": feature,
        "dates_evaluated": int(len(series)),
        "dates_skipped": int(len(skipped)),
        "skipped": skipped,
        "min_observations": int(min_observations),
    }
    return series, diagnostics


def summarize_ic_series(series, min_months=1):
    """Mean/median IC, IC std, ICIR and positive-month share from a monthly series."""
    if series is None or len(series) == 0:
        return {
            "months": 0, "mean_ic": None, "median_ic": None, "ic_std": None,
            "icir": None, "positive_month_share": None, "negative_month_share": None,
            "mean_paired_obs": None, "sufficient_months": False,
        }
    values = series["rank_ic"].to_numpy(dtype="float64")
    months = int(len(values))
    mean_ic = float(np.mean(values))
    std_ic = float(np.std(values, ddof=1)) if months > 1 else 0.0
    icir = float(mean_ic / std_ic) if std_ic > 0.0 else None
    return {
        "months": months,
        "mean_ic": mean_ic,
        "median_ic": float(np.median(values)),
        "ic_std": std_ic,
        "icir": icir,
        "positive_month_share": float(np.mean(values > 0.0)),
        "negative_month_share": float(np.mean(values < 0.0)),
        "mean_paired_obs": float(series["paired_obs"].mean()),
        "sufficient_months": months >= int(min_months),
    }


def pooled_rank_ic(frame, feature, target="future_12m_excess_return"):
    """SECONDARY pooled rank IC (non-independence acknowledged, never primary)."""
    if feature not in frame.columns or target not in frame.columns:
        return None
    subset = frame.loc[:, [feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 2:
        return None
    return spearman(subset[feature].to_numpy(), subset[target].to_numpy())
