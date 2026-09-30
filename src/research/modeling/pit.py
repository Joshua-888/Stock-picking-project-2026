"""WP6 field-based future-availability (point-in-time / leakage) guard.

Corrective defect #4. The legacy guard in ``src/research/discovery/placebo.py``
used a DATE tautology (``required = ts + 365d; violates = required > ts``, which is
always true) and is therefore inert. That module is SHARED WITH CERTIFIED WP5
(``src/research/discovery/builder.py`` consumes it) so it is left byte-unchanged to
preserve WP5 history; WP6 instead implements a REAL field-based guard here over the
actual modelling fields ``feature_asof``, ``model_asof``/``model_date``,
``target_known_at`` and ``target_end``.

For every fold:

* every TRAINING row must satisfy ``feature_asof <= model_asof`` (the feature was
  available at the model instant);
* every TRAINING row must satisfy ``target_known_at <= model_asof`` and
  ``target_end <= model_asof`` (its label was public AND fully realised; the frozen
  12-month purge);
* every VALIDATION row must satisfy ``feature_asof >= model_asof`` (it is predicted
  at/after the model instant, never a backwards leak);
* the training window must be strictly before the validation window.
"""

from __future__ import annotations

import pandas as pd


class PitGuardError(RuntimeError):
    """Raised when the field-based guard cannot be evaluated."""


def to_utc(value):
    return pd.to_datetime(value, errors="coerce", utc=True)


def _as_scalar(stamp):
    series = pd.to_datetime(pd.Series([stamp]), errors="coerce", utc=True)
    return series.iloc[0]


def _column(frame, name, what):
    if name not in frame.columns:
        raise PitGuardError("%s frame lacks required field %r" % (what, name))
    return to_utc(frame[name])


def training_availability_report(frame, model_asof, feature_asof_col="feature_asof",
                                target_known_col="target_known_at", target_end_col="target_end"):
    """Field-based availability report for a TRAINING frame at one model instant."""
    asof = _column(frame, feature_asof_col, "training")
    known = _column(frame, target_known_col, "training")
    end = _column(frame, target_end_col, "training")
    model = _as_scalar(model_asof)
    feature_future = asof > model
    known_future = known.isna() | (known > model)
    end_future = end.isna() | (end > model)
    violations = feature_future | known_future | end_future
    return {
        "rows": int(len(frame)),
        "feature_after_model_rows": int(feature_future.sum()),
        "target_known_after_model_rows": int(known_future.sum()),
        "target_end_after_model_rows": int(end_future.sum()),
        "violations": int(violations.sum()),
        "sample": frame.loc[violations, [feature_asof_col]].head(5).astype(str).to_dict(orient="records"),
    }


def validation_availability_report(frame, model_asof, feature_asof_col="feature_asof"):
    """Field-based report for a VALIDATION frame: rows must be at/after the instant."""
    asof = _column(frame, feature_asof_col, "validation")
    model = _as_scalar(model_asof)
    backwards = asof < model
    return {
        "rows": int(len(frame)),
        "feature_before_model_rows": int(backwards.sum()),
        "violations": int(backwards.sum()),
        "sample": frame.loc[backwards, [feature_asof_col]].head(5).astype(str).to_dict(orient="records"),
    }


def pit_guard_report(train_frame, validation_frame, model_asof,
                     feature_asof_col="feature_asof", target_known_col="target_known_at",
                     target_end_col="target_end"):
    """Full field-based PIT guard for one fold's train/validation split."""
    train = training_availability_report(train_frame, model_asof, feature_asof_col,
                                         target_known_col, target_end_col)
    validation = validation_availability_report(validation_frame, model_asof, feature_asof_col)
    train_max = _column(train_frame, feature_asof_col, "training").max()
    val_min = _column(validation_frame, feature_asof_col, "validation").min()
    chronology_ok = bool(pd.isna(train_max) or pd.isna(val_min) or train_max < val_min)
    violations = int(train["violations"] + validation["violations"]) + (0 if chronology_ok else 1)
    return {
        "model_asof": str(model_asof),
        "train": train,
        "validation": validation,
        "chronology_ok": chronology_ok,
        "violations": violations,
        "passed": bool(violations == 0),
    }


def future_availability_control(fold_frames, folds, control_id="future_availability_guard",
                                matched_real_config_id=None):
    """Aggregate the field-based guard over every fold into one control object."""
    from .placebo import control_object
    reports = []
    total = 0
    for fold in folds:
        bundle = fold_frames.get(fold.fold) if isinstance(fold_frames, dict) else None
        if bundle is None:
            continue
        report = pit_guard_report(bundle["train"], bundle["validation"], fold.model_date)
        report["fold"] = fold.fold
        reports.append(report)
        total += int(report["violations"])
    failed = bool(total > 0)
    return control_object(
        control_id=control_id,
        control_type="future_availability_guard",
        expected_behavior="no feature instant after the model instant; every training label public and realised at the model date; validation rows at/after the model instant",
        observed_behavior="%d point-in-time availability violation(s) across %d fold(s)" % (total, len(reports)),
        matched_real_config_id=matched_real_config_id,
        real_metric=None,
        control_metric=float(total),
        failure_threshold=0.0,
        failure_condition="violations > 0",
        passed=not failed,
        stop_required=failed,
        reason=("clean point-in-time availability" if not failed
                else "future feature/label information detected in the modelling rows"),
        mandatory=True,
        affected_tasks=("regression", "classification"),
        extra={"fold_reports": reports},
    )
