"""Deterministic point-in-time as-of join engine.

Two invariants drive every function here:

1. A record is eligible for a prediction at ``T`` only when
   ``available_at <= T``. Records with missing/unparseable availability are
   dropped, never guessed.
2. When several versions of the same fact are eligible (an original filing and
   a later restatement/amendment), the version selected is the newest one that
   was *already* available at ``T``. A restatement filed after ``T`` can never
   be selected for ``T``: that would be backward restatement leakage.

Joins are many-to-one by construction: an arbitrary number of historical
versions collapse to exactly one row per key, so a join cannot silently
multiply observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .availability import availability_mask, to_utc_timestamp


class PitJoinError(ValueError):
    """Raised when an as-of join is requested with unusable inputs."""


def _require_columns(df, columns, what):
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise PitJoinError("%s is missing column(s): %s" % (what, ", ".join(missing)))


def _key_tuples(df, key_cols):
    if not key_cols:
        return [()] * len(df)
    columns = [df[name] for name in key_cols]
    return list(zip(*[list(column) for column in columns]))


def _sort_tokens(series):
    """Deterministic, type-stable sort token per value (None sorts first)."""
    import json

    return [json.dumps(value, sort_keys=True, default=str) if value is not None else "" for value in series]


def _eligible(facts_df, asof_ts, available_at_col):
    if available_at_col not in facts_df.columns:
        raise PitJoinError("availability column %r is absent" % available_at_col)
    if facts_df.empty:
        return facts_df.copy()
    mask = availability_mask(facts_df, available_at_col, asof_ts)
    return facts_df.loc[mask].copy()


def latest_version_asof(facts_df, asof_ts, key_cols, available_at_col, version_col=None, extra_cols=()):
    """Select, per key, the newest version already available at ``asof_ts``.

    Returns one row per distinct key tuple. Ordering is fully deterministic:
    rows are ranked by ``available_at`` ascending, then by the canonical JSON
    token of ``version_col`` (when given), then by original position; the last
    row in that ranking wins. The result is therefore independent of input row
    order and of pandas sort stability assumptions.
    """
    key_cols = list(key_cols)
    extra_cols = [name for name in extra_cols if name not in key_cols]
    _require_columns(facts_df, key_cols, "facts")
    if version_col is not None:
        _require_columns(facts_df, [version_col], "facts")
    _require_columns(facts_df, extra_cols, "facts")

    output_cols = key_cols + extra_cols + [available_at_col] + ([version_col] if version_col else [])
    eligible = _eligible(facts_df, asof_ts, available_at_col)
    if eligible.empty:
        return eligible.reindex(columns=output_cols).reset_index(drop=True)

    ordered = eligible.copy()
    ordered["__row_pos"] = range(len(ordered))
    ordered["__avail"] = [to_utc_timestamp(value) for value in ordered[available_at_col]]
    ordered["__avail_token"] = [stamp.isoformat() if stamp is not None else "" for stamp in ordered["__avail"]]
    ordered["__key"] = _key_tuples(ordered, key_cols)
    if version_col is not None:
        ordered["__version_token"] = _sort_tokens(ordered[version_col])
    else:
        ordered["__version_token"] = ""

    ordered = ordered.sort_values(
        by=["__key", "__avail_token", "__version_token", "__row_pos"],
        kind="mergesort",
    )
    selected = ordered.groupby("__key", sort=True, dropna=False).tail(1)
    selected = selected.sort_values(by=["__key"], kind="mergesort")
    result = selected.reindex(columns=output_cols).reset_index(drop=True)
    if version_col is not None and version_col in [available_at_col]:
        pass
    return result


def asof_join(facts_df, asof_ts, key_cols, available_at_col, value_cols, version_col=None, extra_cols=()):
    """Many-to-one point-in-time join selecting the latest version per key.

    ``facts_df`` may hold any number of versions/restatements per key; exactly
    one row per key is returned, carrying ``value_cols`` (plus ``extra_cols``)
    from the version that was already public at ``asof_ts``.
    """
    value_cols = list(value_cols)
    _require_columns(facts_df, value_cols, "facts")
    keep = list(value_cols) + [name for name in extra_cols if name not in value_cols]
    selected = latest_version_asof(
        facts_df,
        asof_ts,
        key_cols=key_cols,
        available_at_col=available_at_col,
        version_col=version_col,
        extra_cols=keep,
    )
    ordered_cols = list(key_cols) + keep + [available_at_col] + ([version_col] if version_col else [])
    seen = []
    for name in ordered_cols:
        if name not in seen:
            seen.append(name)
    return selected.reindex(columns=seen).reset_index(drop=True)


def asof_join_at_times(facts_df, requests_df, key_cols, request_time_col, available_at_col,
                       value_cols, version_col=None, extra_cols=()):
    """Apply :func:`asof_join` independently at several prediction times.

    ``requests_df`` carries the query keys plus a per-row prediction timestamp.
    Each request row sees only facts available at its own timestamp, so one
    call can build a full historical panel without any request observing data
    from a later request's future.
    """
    key_cols = list(key_cols)
    _require_columns(requests_df, key_cols + [request_time_col], "requests")
    value_cols = list(value_cols)
    frames = []
    unique_times = sorted(
        {to_utc_timestamp(value) for value in requests_df[request_time_col]},
        key=lambda stamp: (stamp is None, stamp.isoformat() if stamp is not None else ""),
    )
    for moment in unique_times:
        if moment is None:
            continue
        joined = asof_join(
            facts_df,
            moment,
            key_cols=key_cols,
            available_at_col=available_at_col,
            value_cols=value_cols,
            version_col=version_col,
            extra_cols=extra_cols,
        )
        selected_rows = requests_df.loc[
            [to_utc_timestamp(value) == moment for value in requests_df[request_time_col]]
        ]
        if selected_rows.empty:
            continue
        request_keys = selected_rows[key_cols + [request_time_col]].copy()
        request_keys["__key"] = _key_tuples(request_keys, key_cols)
        joined = joined.copy()
        joined["__key"] = _key_tuples(joined, key_cols)
        merged = request_keys.merge(joined.drop(columns=key_cols), on="__key", how="left")
        frames.append(merged.drop(columns=["__key"]))
    ordered_cols = key_cols + [request_time_col] + value_cols + [name for name in extra_cols if name not in value_cols] + [available_at_col]
    seen = []
    for name in ordered_cols:
        if name not in seen:
            seen.append(name)
    if not frames:
        return requests_df.reindex(columns=seen).reset_index(drop=True)
    combined = pd.concat(frames, ignore_index=True)
    return combined.reindex(columns=seen).reset_index(drop=True)


@dataclass(frozen=True)
class AsofJoinSpec:
    """Declarative description of one point-in-time join."""

    key_cols: tuple
    available_at_col: str
    value_cols: tuple
    version_col: str = None

    def apply(self, facts_df, asof_ts):
        return asof_join(
            facts_df,
            asof_ts,
            key_cols=list(self.key_cols),
            available_at_col=self.available_at_col,
            value_cols=list(self.value_cols),
            version_col=self.version_col,
        )
