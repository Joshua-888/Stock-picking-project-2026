"""WP5 locked-holdout tests.

LIVE-INDEPENDENT: no test touches the network or the research data filesystem.
Tests operate on the tracked provenance artefacts (``provenance/holdout/``) and
on small synthetic fixtures built in-memory or under ``tmp_path``.

They freeze the invariants that keep the V2 locked holdout evaluation-only:

* the holdout boundary is deterministic and content-addressed;
* the accessor is live-independent;
* holdout rows are excluded from every training mask;
* the embargo is honored (label-overlap purge band);
* rows whose labels overlap the holdout are purged from training;
* training on holdout or embargo rows fails loudly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.research.holdout import (
    DEFAULT_EMBARGO_MONTHS,
    DEFAULT_HOLDOUT_START,
    HOLDOUT_SCHEMA_VERSION,
    HoldoutError,
    HoldoutLeakageError,
    assert_trainable,
    build_holdout,
    embargo_cutoff,
    embargo_mask,
    holdout_mask,
    holdout_payload,
    locked_holdout,
    trainable_mask,
)
from src.research.ids import KINDS, holdout_id, is_valid_id

REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = REPO_ROOT / "provenance" / "holdout"


# ── fixtures ──────────────────────────────────────────────────────────────────

def _target_frame():
    """Synthetic observable target frame spanning development, embargo and holdout.

    ``feature_asof`` covers monthly points; ``target_known_at`` is +12 months.
    Values are irrelevant here; only date availability is exercised.
    """
    asof = pd.date_range("2019-01-31", "2025-08-31", freq="ME", tz="UTC")
    rows = []
    for stamp in asof:
        known = stamp + pd.DateOffset(months=12)
        rows.append({
            "security_id": "sec_%s" % stamp.strftime("%Y%m"),
            "ticker": "T%s" % stamp.strftime("%y%m"),
            "feature_asof": stamp.isoformat(),
            "target_known_at": known.isoformat(),
            "target_observable": True,
        })
    return pd.DataFrame(rows)


# ── 1. deterministic, content-addressed boundary ─────────────────────────────

def test_holdout_id_is_content_addressed_and_deterministic():
    assert "holdout" in KINDS
    payload = holdout_payload(
        dataset_id="dataset_x", target_id="target_set_x", feature_set_id="feature_set_x",
        holdout_start="2022-01-01", holdout_end="2025-08-31", embargo_months=12,
        selection_rationale="availability", git_commit="a" * 40,
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    first = holdout_id(payload)
    second = holdout_id(dict(payload))
    assert first == second
    assert is_valid_id(first, "holdout")
    changed = dict(payload)
    changed["holdout_start"] = "2023-01-01"
    assert holdout_id(changed) != first

    record = build_holdout(payload)
    assert record["holdout_id"] == first
    assert record["prohibition_rule"].startswith("NO discovery")


# ── 2. frozen artefact + accessor ────────────────────────────────────────────

def test_committed_holdout_artifact_is_content_addressed():
    index = json.loads((PROVENANCE_DIR / "index.json").read_text(encoding="utf-8"))
    assert index["holdout_schema_version"] == HOLDOUT_SCHEMA_VERSION
    entry = index["holdout"]
    record = json.loads((PROVENANCE_DIR / entry["artifact_file"]).read_text(encoding="utf-8"))
    assert record["holdout_id"] == entry["holdout_id"]
    recomputed = holdout_id({k: v for k, v in record.items() if k != "holdout_id"})
    assert recomputed == record["holdout_id"]
    # The upstream bindings must reference the CORRECTED (WP2C/WP3) artefacts.
    assert record["dataset_id"] == "dataset_35a278e17c13"
    assert record["target_id"] == "target_set_d2bb16610bce"
    assert record["feature_set_id"] == "feature_set_4f7b43726310"
    assert record["holdout_start"] == DEFAULT_HOLDOUT_START


def test_locked_holdout_accessor_is_live_independent(tmp_path):
    holdout = locked_holdout()
    assert holdout.holdout_id == "holdout_7ce54e933e16"
    assert str(holdout.holdout_start.date()) == "2022-01-01"
    assert str(holdout.holdout_end.date()) == "2025-08-31"
    assert holdout.embargo_months == DEFAULT_EMBARGO_MONTHS
    assert str(holdout.embargo_cutoff.date()) == "2021-01-01"

    copied = tmp_path / "holdout"
    shutil.copytree(PROVENANCE_DIR, copied)
    assert locked_holdout(provenance_dir=copied).to_dict() == holdout.to_dict()


def test_tampered_artifact_is_rejected(tmp_path):
    copied = tmp_path / "holdout"
    shutil.copytree(PROVENANCE_DIR, copied)
    index = json.loads((copied / "index.json").read_text(encoding="utf-8"))
    record_path = copied / index["holdout"]["artifact_file"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["holdout_start"] = "2010-01-01"  # tamper with the frozen boundary
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(HoldoutError):
        locked_holdout(provenance_dir=copied)


# ── 3. holdout rows excluded from every training mask ────────────────────────

def test_holdout_rows_excluded_from_train_mask():
    frame = _target_frame()
    holdout = locked_holdout()
    for model_date in ("2020-06-30", "2023-06-30", "2026-09-25"):
        mask = trainable_mask(frame, model_date, holdout)
        asof = pd.to_datetime(frame["feature_asof"], format="ISO8601", utc=True)
        in_holdout = asof >= holdout.holdout_start
        assert not bool((mask & in_holdout).any())
        assert bool(holdout_mask(frame, holdout).equals(in_holdout))


def test_embargo_band_is_honored_and_purged():
    frame = _target_frame()
    holdout = locked_holdout()
    mask = trainable_mask(frame, "2026-09-25", holdout)
    band = embargo_mask(frame, holdout)
    asof = pd.to_datetime(frame["feature_asof"], format="ISO8601", utc=True)
    expected_band = (asof >= holdout.embargo_cutoff) & (asof < holdout.holdout_start)
    assert bool(band.equals(expected_band))
    # No embargo-band row is ever trainable, even once its label is observable.
    assert not bool((mask & band).any())
    assert band.any()


def test_overlapping_label_rows_are_purged_from_training():
    frame = _target_frame()
    holdout = locked_holdout()
    mask = trainable_mask(frame, "2026-09-25", holdout)
    kept = frame[mask.values]
    cutoff = holdout.embargo_cutoff
    # Every kept training row's forward label ends before the holdout begins.
    for _, row in kept.iterrows():
        asof = pd.Timestamp(row["feature_asof"])
        known = pd.Timestamp(row["target_known_at"])
        assert asof < cutoff
        assert known <= holdout.holdout_start  # no label overlaps the holdout


# ── 4. training on holdout/embargo rows fails loudly ─────────────────────────

def test_training_on_holdout_rows_asserts():
    frame = _target_frame()
    holdout = locked_holdout()
    asof = pd.to_datetime(frame["feature_asof"], format="ISO8601", utc=True)
    holdout_only = frame[(asof >= holdout.holdout_start).values]
    assert not holdout_only.empty
    with pytest.raises(HoldoutLeakageError):
        assert_trainable(holdout_only, "2026-09-25", holdout)


def test_training_on_embargo_rows_asserts():
    frame = _target_frame()
    holdout = locked_holdout()
    band = frame[embargo_mask(frame, holdout).values]
    assert not band.empty
    with pytest.raises(HoldoutLeakageError):
        assert_trainable(band, "2026-09-25", holdout)


def test_training_on_fully_legal_rows_passes():
    frame = _target_frame()
    holdout = locked_holdout()
    mask = trainable_mask(frame, "2026-09-25", holdout)
    legal = frame[mask.values]
    assert not legal.empty
    assert assert_trainable(legal, "2026-09-25", holdout) is True


def test_embargo_cutoff_default_matches_definition():
    assert str(embargo_cutoff(DEFAULT_HOLDOUT_START, DEFAULT_EMBARGO_MONTHS).date()) == "2021-01-01"
