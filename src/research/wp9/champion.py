"""Frozen WP8 champion loader for WP9 prospective scoring.

This module applies the immutable final model. It performs no fitting, no
feature selection, no imputation refitting and no calibration refitting.

The authoritative artifact identity is
``provenance/wp8/final_candidate_freeze.json``. Every pickle is verified by
raw-byte SHA-256 before it is unpickled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

FROZEN_FREEZE_ID = "freeze_e62eac30df40"
FROZEN_CALIBRATION = "platt_scaling"
FROZEN_FEATURES: Tuple[str, ...] = (
    "earnings_yield",
    "current_pe",
    "price_to_book",
    "free_cash_flow_yield",
    "dividend_yield",
    "roa",
    "roe",
    "eps_growth_acceleration",
    "six_month_momentum",
    "twelve_month_momentum",
    "five_year_price_gain",
    "debt_to_equity",
    "market_cap",
)

FREEZE_RECORD_REL = Path("provenance") / "wp8" / "final_candidate_freeze.json"


class ChampionError(RuntimeError):
    """Raised when the frozen champion cannot be loaded or verified."""


@dataclass(frozen=True)
class FrozenChampion:
    """Verified, apply-only view of the immutable WP8 champion."""

    freeze_id: str
    features: Tuple[str, ...]
    calibration: str
    model_sha256: str
    preprocessor_sha256: str
    calibrator_sha256: str
    _model_obj: object
    _preprocessor_obj: object
    _calibrator: object

    @property
    def model_hash(self) -> str:
        return self.model_sha256

    @property
    def preprocessor_hash(self) -> str:
        return self.preprocessor_sha256

    @property
    def calibrator_hash(self) -> str:
        return self.calibrator_sha256

    def predict(self, score_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the frozen bundle to ``score_frame`` without fitting anything.

        Returns ``(raw_model_score, frozen_calibrated_score)``, both float64
        arrays in input row order. The frozen Platt logistic transform is
        applied exactly as in WP8: logit clip, then the logistic calibrator's
        positive-class column.
        """
        if not isinstance(score_frame, pd.DataFrame):
            raise ChampionError("score_frame must be a pandas DataFrame")
        missing = [name for name in self.features if name not in score_frame.columns]
        if missing:
            raise ChampionError(
                "score_frame is missing frozen feature column(s): %s" % ", ".join(missing)
            )
        # Keep only the frozen columns in frozen feature order for determinism.
        frame = score_frame.loc[:, list(self.features)].copy()

        preprocessor_obj = self._preprocessor_obj
        if not isinstance(preprocessor_obj, dict):
            raise ChampionError("frozen preprocessor bundle is not a dict")
        preprocessor = preprocessor_obj.get("preprocessor")
        fitted = preprocessor_obj.get("fitted")
        if preprocessor is None or fitted is None:
            raise ChampionError("frozen preprocessor bundle is incomplete; refusing to fit")

        x = preprocessor.transform(frame, fitted)

        model_obj = self._model_obj
        if not isinstance(model_obj, dict) or model_obj.get("kind") != "estimator":
            raise ChampionError("frozen model bundle is not an estimator bundle")
        estimator = model_obj.get("estimator")
        if estimator is None:
            raise ChampionError("frozen model bundle has no estimator")
        # The frozen ExtraTrees object is configured with n_jobs=-1, whose tiny
        # thread-scheduling nondeterminism breaks the WP9 deterministic-rank and
        # snapshot-hash guarantees. Use an ephemeral copy with n_jobs=1 for every
        # apply call; the frozen bundle is never mutated and no fit happens.
        deterministic_estimator = copy.deepcopy(estimator)
        if hasattr(deterministic_estimator, "n_jobs"):
            deterministic_estimator.n_jobs = 1
        class_columns = getattr(estimator, "classes_", None)
        if class_columns is not None and not hasattr(deterministic_estimator, "n_features_in_"):
            deterministic_estimator.classes_ = class_columns
        if hasattr(deterministic_estimator, "n_outputs_"):
            deterministic_estimator.n_outputs_ = getattr(estimator, "n_outputs_", 1)
        raw_proba = deterministic_estimator.predict_proba(x)[:, 1]
        raw_score = np.asarray(raw_proba, dtype="float64")

        calibrated = self._apply_platt(raw_score)
        return raw_score, calibrated

    def _apply_platt(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the frozen Platt logistic calibration object (no refit)."""
        if self.calibration != "platt_scaling":
            raise ChampionError("frozen calibration is not platt_scaling")
        p = np.asarray(probabilities, dtype="float64")
        clipped = np.clip(p, 1e-9, 1 - 1e-9)
        logit = np.log(clipped / (1 - clipped))
        calibrator = self._calibrator
        if calibrator is None or not hasattr(calibrator, "predict_proba"):
            raise ChampionError("frozen Platt calibrator is unavailable")
        output = calibrator.predict_proba(logit.reshape(-1, 1))
        return np.asarray(output[:, 1], dtype="float64")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_freeze_record(root: Path | None = None) -> dict:
    root = Path(root or Path(__file__).resolve().parents[3])
    path = root / FREEZE_RECORD_REL
    if not path.is_file():
        raise ChampionError("missing WP8 final candidate freeze record: %s" % path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("freeze_id") != FROZEN_FREEZE_ID:
        raise ChampionError(
            "wrong WP8 freeze id %r; expected %s" % (record.get("freeze_id"), FROZEN_FREEZE_ID)
        )
    return record


def load_champion(root: Path | None = None, freeze_record: dict | None = None) -> FrozenChampion:
    """Verify and load the frozen WP8 champion.

    ``root`` is the repository root; it defaults to the repository containing
    this source file (three parents above ``src/research/wp9/champion.py``).
    """
    root = Path(root or Path(__file__).resolve().parents[3])
    record = freeze_record or load_freeze_record(root)

    features = tuple(record.get("final_selected_features") or ())
    if features != FROZEN_FEATURES:
        raise ChampionError(
            "frozen final_selected_features mismatch; expected the 13 WP9 champion features"
        )
    calibration = record.get("final_calibration")
    if calibration != FROZEN_CALIBRATION:
        raise ChampionError(
            "wrong final calibration %r; expected %s" % (calibration, FROZEN_CALIBRATION)
        )

    artifact_hashes = record.get("artifact_hashes") or {}
    required = ("model", "preprocessor", "calibrator")
    missing = [name for name in required if name not in artifact_hashes]
    if missing:
        raise ChampionError(
            "freeze artifact_hashes missing required artifacts: %s" % ", ".join(missing)
        )

    loaded: Dict[str, object] = {}
    digests: Dict[str, str] = {}
    for name in required:
        spec = artifact_hashes[name]
        expected = spec.get("sha256")
        if not expected or not isinstance(expected, str):
            raise ChampionError("freeze record is missing a sha256 for %s" % name)
        path = Path(spec.get("path") or "")
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ChampionError("frozen %s artifact is missing: %s" % (name, path))
        actual = _sha256_file(path)
        if actual != expected:
            raise ChampionError(
                "frozen %s artifact hash mismatch: expected=%s actual=%s" % (name, expected, actual)
            )
        digests[name] = actual
        with path.open("rb") as handle:
            loaded[name] = pickle.load(handle)

    champion = FrozenChampion(
        freeze_id=FROZEN_FREEZE_ID,
        features=features,
        calibration=calibration,
        model_sha256=digests["model"],
        preprocessor_sha256=digests["preprocessor"],
        calibrator_sha256=digests["calibrator"],
        _model_obj=loaded["model"],
        _preprocessor_obj=loaded["preprocessor"],
        _calibrator=loaded["calibrator"],
    )
    return champion
