"""WP3 temporal observability: when a labelled row becomes usable for training.

With a 12-month forward target, an observation at prediction instant T cannot be
used to fit a model until its full future outcome has actually occurred. This
module attaches, per row:

* ``feature_asof``      - the prediction instant T;
* ``target_start``      - the first observation that begins the return leg;
* ``target_end``        - the observation at/after the 12-month horizon;
* ``target_known_at``   - the instant the outcome is public, i.e. the availability
                          instant of the terminal observation (trade date + close,
                          in UTC) for BOTH the stock and the benchmark;
* ``target_horizon_days_actual`` - realised calendar length of the label.

A row is trainable for a model whose ``model_date`` is ``T_model`` only when
``target_known_at <= T_model``. This is the infrastructure WP6/WP7 use to purge
overlapping labels; it does not itself fit or validate any model.
"""

from __future__ import annotations

import pandas as pd

from ..data.availability import price_available_at, to_utc_timestamp

OBSERVABILITY_COLUMNS = (
    "security_id", "ticker", "feature_asof", "target_start", "target_end",
    "target_horizon_days_actual", "target_known_at", "target_observable",
)


class ObservabilityError(ValueError):
    """Raised when observability inputs cannot be interpreted."""


def _max_timestamp(*values):
    stamps = [to_utc_timestamp(value) for value in values]
    known = [stamp for stamp in stamps if stamp is not None]
    return max(known) if known else None


def target_known_at(target_end, benchmark_end=None):
    """Availability instant for a label ending at ``target_end`` (+ benchmark).

    The label requires both legs, so the later of the two terminal availability
    instants governs. ``price_available_at`` applies the conservative UTC close
    rule (information is only public after the session closes).
    """
    if target_end is None:
        return None
    stock = price_available_at(target_end)
    bench = price_available_at(benchmark_end if benchmark_end is not None else target_end)
    return _max_timestamp(stock, bench)


def annotate_observability(target_frame, benchmark_end_col=None):
    """Attach ``target_known_at`` to a built target frame (feature_asof already present).

    Only observable rows receive a concrete ``target_known_at``; censored rows
    keep ``None`` so they can never be mistaken for trainable labels. The input
    is not mutated; a new frame with :data:`OBSERVABILITY_COLUMNS` is returned.
    """
    if not isinstance(target_frame, pd.DataFrame):
        raise ObservabilityError("annotate_observability expects a DataFrame")
    missing = [name for name in ("security_id", "feature_asof", "target_end",
                                 "target_horizon_days_actual", "target_observable")
               if name not in target_frame.columns]
    if missing:
        raise ObservabilityError("target frame is missing column(s): %s" % ", ".join(missing))
    rows = []
    for record in target_frame.itertuples():
        observable = bool(getattr(record, "target_observable"))
        bench_end = getattr(record, benchmark_end_col) if benchmark_end_col else None
        known = target_known_at(getattr(record, "target_end"), bench_end) if observable else None
        rows.append(
            {
                "security_id": str(record.security_id),
                "ticker": str(getattr(record, "ticker", record.security_id)),
                "feature_asof": getattr(record, "feature_asof"),
                "target_start": getattr(record, "target_start", None),
                "target_end": getattr(record, "target_end", None),
                "target_horizon_days_actual": getattr(record, "target_horizon_days_actual", None),
                "target_known_at": known.isoformat() if known is not None else None,
                "target_observable": observable,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=list(OBSERVABILITY_COLUMNS))
    return frame[list(OBSERVABILITY_COLUMNS)]


def trainable_mask(target_frame, model_date, known_col="target_known_at",
                   observable_col="target_observable"):
    """Boolean mask of rows a model at ``model_date`` may legally train on.

    A row qualifies only when it is observable AND its ``target_known_at`` is at
    or before ``model_date``. Rows with missing/invalid metadata are excluded
    (never silently included).
    """
    if known_col not in target_frame.columns or observable_col not in target_frame.columns:
        raise ObservabilityError("target frame lacks %r/%r" % (known_col, observable_col))
    moment = to_utc_timestamp(model_date)
    if moment is None:
        raise ObservabilityError("model_date is missing or unparseable: %r" % (model_date,))
    flags = []
    observable_values = target_frame[observable_col].tolist()
    known_values = target_frame[known_col].tolist()
    for observable, known_raw in zip(observable_values, known_values):
        if not bool(observable):
            flags.append(False)
            continue
        known = to_utc_timestamp(known_raw)
        flags.append(bool(known is not None and known <= moment))
    return pd.Series(flags, index=target_frame.index)
