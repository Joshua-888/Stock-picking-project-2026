"""WP9 matured forward-snapshot evaluator.

Load one immutable official WP9 prediction snapshot, rebuild its 12-month
outcomes using the certified WP3 target construction, refuse immature
snapshots, compute the predeclared cross-sectional and economic shadow metrics,
and persist an immutable evaluation record while updating the WP9 run registry.

This script NEVER retrains, recalibrates, selects features, searches cutoffs, or
challenges the frozen champion. Censored rows are retained as censored and are
never converted into zero, negative, or failure labels.

Run with the research runtime:

    PYTHONPATH=. /opt/venv/bin/python \\
        scripts/research_v2/wp9_evaluate_matured.py --snapshot-id <id>

Optional deterministic evaluation date (defaults to current UTC):

    PYTHONPATH=. /opt/venv/bin/python \\
        scripts/research_v2/wp9_evaluate_matured.py --snapshot-id <id> \\
        --evaluation-date 2027-10-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.data.wp2b_eodhd import actions_from_eodhd_rows  # noqa: E402
from src.research.fingerprints import (  # noqa: E402
    fingerprint_dataframe,
    fingerprint_file,
    fingerprint_obj,
)
from src.research.immutability import save_immutable  # noqa: E402
from src.research.modes import current_git_commit  # noqa: E402
from src.research.targets import annotate_observability, build_targets  # noqa: E402
from src.research.targets.contract import TARGET_VERSION  # noqa: E402
from src.research.wp9.contract import (  # noqa: E402
    Wp9Contract,
    Wp9ContractError,
    contract_freeze_timestamp,
    load_wp9_contract,
)
from src.research.wp9.economics import (  # noqa: E402
    BENCHMARK_SYMBOL,
    COST_SCENARIOS_BPS,
    HOLDING_MONTHS,
    select_top_quintile,
)
from src.research.wp9.forward_inputs import (  # noqa: E402
    _ForwardInputBuilder,
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
)
from src.research.wp9.storage import (  # noqa: E402
    BINDINGS,
    OFFICIAL_KIND,
    PREDICTION_INDEX_REL,
    PREDICTION_INDEX_SCHEMA,
    SNAPSHOT_SCHEMA_FIELDS,
    Wp9StorageError,
    add_matured_evaluation_to_run_registry,
    load_run_registry,
    snapshot_id_from_bindings,
)

MATURED_EVALUATION_SCHEMA = "wp9_matured_evaluation_v1"
TARGET_SET_ID = "target_set_d2bb16610bce"
FROZEN_CHAMPION = "freeze_e62eac30df40"
HAC_LAG = 12
TOP_QUINTILE_THRESHOLD = 0.80
BOTTOM_QUINTILE_THRESHOLD = 0.20

PREDICTIONS_DIR = Path("provenance") / "wp9" / "predictions"
DRY_RUNS_DIR = Path("provenance") / "wp9" / "dry_runs"
EVALUATIONS_DIR = Path("provenance") / "wp9" / "matured_evaluations"

BLOCK_PREFIX = "MATURED_EVALUATION_BLOCK:"


class MaturedEvaluationError(RuntimeError):
    """Raised when a matured evaluation cannot proceed honestly."""


def _utc_stamp(value):
    """Return a timezone-aware UTC ``pd.Timestamp`` or ``pd.NaT``."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _utc_iso(value):
    """Normalise a date-like value to a UTC ISO-8601 string."""
    stamp = _utc_stamp(value)
    return None if pd.isna(stamp) else stamp.isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise MaturedEvaluationError("missing file: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaturedEvaluationError("invalid JSON at %s: %s" % (path, exc)) from exc


# ── Immutable snapshot loading/verification ───────────────────────────────────

def _prediction_index(root: Path) -> Dict[str, Any]:
    index = _read_json(root / PREDICTION_INDEX_REL)
    if index.get("schema_version") != PREDICTION_INDEX_SCHEMA:
        raise MaturedEvaluationError("unexpected WP9 prediction index schema")
    return index


def snapshot_file_path(root: Path, snapshot_id: str) -> Path:
    return root / PREDICTIONS_DIR / ("%s.json" % snapshot_id)


def _index_entry_by_id(root: Path, snapshot_id: str) -> Dict[str, Any]:
    index = _prediction_index(root)
    matching = [
        entry for entry in index.get("entries", [])
        if entry.get("snapshot_id") == snapshot_id
    ]
    if not matching:
        raise MaturedEvaluationError(
            "%ssnapshot_not_in_official_prediction_index:snapshot_id=%s"
            % (BLOCK_PREFIX, snapshot_id)
        )
    if len(matching) != 1:
        raise MaturedEvaluationError(
            "%smultiple_index_entries:snapshot_id=%s" % (BLOCK_PREFIX, snapshot_id)
        )
    return matching[0]


def _snapshot_in_dry_run_index(root: Path, snapshot_id: str) -> bool:
    index_path = root / DRY_RUNS_DIR / "index.json"
    if not index_path.is_file():
        return False
    index = _read_json(index_path)
    return any(
        entry.get("snapshot_id") == snapshot_id for entry in index.get("entries", [])
    )


def _assert_run_registry_status(
    root: Path, snapshot_id: str, contract: Wp9Contract
) -> None:
    """Reject dry-run, invalidated, superseded, unknown-freeze, unknown-contract ids."""
    if _snapshot_in_dry_run_index(root, snapshot_id):
        raise MaturedEvaluationError(
            "%sdry_run_snapshot_forbidden_for_matured_evaluation:%s"
            % (BLOCK_PREFIX, snapshot_id)
        )
    try:
        registry = load_run_registry(root=root)
    except Wp9StorageError as exc:
        raise MaturedEvaluationError(
            "%srun_registry_unavailable:%s" % (BLOCK_PREFIX, exc)
        ) from exc
    if registry.get("contract_digest") != contract.digest:
        raise MaturedEvaluationError(
            "%srun_registry_contract_digest_mismatch" % BLOCK_PREFIX
        )
    if registry.get("champion_freeze") != FROZEN_CHAMPION:
        raise MaturedEvaluationError(
            "%srun_registry_unknown_champion_freeze:%s"
            % (BLOCK_PREFIX, registry.get("champion_freeze"))
        )
    if snapshot_id not in set(registry.get("official_snapshot_ids", [])):
        raise MaturedEvaluationError(
            "%ssnapshot_not_in_run_registry_official_list:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    invalidated = {
        record.get("snapshot_id") for record in registry.get("invalidated_snapshots", [])
    }
    if snapshot_id in invalidated:
        raise MaturedEvaluationError(
            "%ssnapshot_invalidated:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    for record in registry.get("supersession_records", []):
        superseded = {record.get("invalidated_snapshot_id")}
        superseded.update(record.get("superseded_snapshot_ids") or [])
        if snapshot_id in superseded:
            raise MaturedEvaluationError(
                "%ssnapshot_superseded:%s" % (BLOCK_PREFIX, snapshot_id)
            )


def _assert_snapshot_schema(snapshot: Mapping[str, Any]) -> None:
    missing = sorted(set(SNAPSHOT_SCHEMA_FIELDS) - set(snapshot))
    unexpected = sorted(set(snapshot) - set(SNAPSHOT_SCHEMA_FIELDS))
    if missing:
        raise MaturedEvaluationError(
            "%ssnapshot_schema_missing_fields:%s" % (BLOCK_PREFIX, ",".join(missing))
        )
    if unexpected:
        raise MaturedEvaluationError(
            "%ssnapshot_schema_unexpected_fields:%s" % (BLOCK_PREFIX, ",".join(unexpected))
        )


def load_official_snapshot(
    snapshot_id: str,
    *,
    root: Path | None = None,
    contract: Wp9Contract | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Wp9Contract]:
    """Load and verify an official prediction snapshot and its index bindings.

    The file raw-byte SHA-256 must match the official prediction index; the body
    and bindings must agree with the file name and frozen WP9 contract.
    """
    root = Path(root or ROOT)
    contract = contract or load_wp9_contract(root=root)
    entry = _index_entry_by_id(root, snapshot_id)
    if entry.get("snapshot_kind") != OFFICIAL_KIND:
        raise MaturedEvaluationError(
            "%snon_official_snapshot:%s" % (BLOCK_PREFIX, entry.get("snapshot_kind"))
        )

    path = snapshot_file_path(root, snapshot_id)
    actual_sha = fingerprint_file(path)
    expected_sha = entry.get("sha256")
    if actual_sha != expected_sha:
        raise MaturedEvaluationError(
            "%ssnapshot_file_hash_mismatch:expected=%s actual=%s"
            % (BLOCK_PREFIX, expected_sha, actual_sha)
        )

    snapshot = _read_json(path)
    _assert_snapshot_schema(snapshot)
    if snapshot.get("snapshot_id") != snapshot_id:
        raise MaturedEvaluationError(
            "%ssnapshot_id_body_filename_mismatch:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    if snapshot.get("model_freeze_id") != FROZEN_CHAMPION:
        raise MaturedEvaluationError(
            "%sunknown_champion_freeze:%s" % (BLOCK_PREFIX, snapshot.get("model_freeze_id"))
        )
    champion_artifacts = contract.get("champion", "artifacts", default={}) or {}
    expected_model_hash = (champion_artifacts.get("model") or {}).get("sha256")
    if snapshot.get("model_hash") != expected_model_hash:
        raise MaturedEvaluationError(
            "%ssnapshot_model_hash_not_contract_champion" % BLOCK_PREFIX
        )

    bindings = entry.get("bindings") or {}
    missing_bindings = [name for name in BINDINGS if name not in bindings]
    if missing_bindings:
        raise MaturedEvaluationError(
            "%ssnapshot_index_bindings_missing:%s" % (BLOCK_PREFIX, ",".join(missing_bindings))
        )
    if snapshot_id_from_bindings(bindings) != snapshot_id:
        raise MaturedEvaluationError(
            "%ssnapshot_index_bindings_recompute_mismatch:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    if bindings.get("contract_digest") != contract.digest:
        raise MaturedEvaluationError(
            "%ssnapshot_contract_digest_not_current:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    if bindings.get("champion_freeze") != FROZEN_CHAMPION:
        raise MaturedEvaluationError(
            "%ssnapshot_unknown_champion_freeze:%s"
            % (BLOCK_PREFIX, bindings.get("champion_freeze"))
        )
    if bindings.get("snapshot_asof") != snapshot.get("snapshot_asof"):
        raise MaturedEvaluationError(
            "%ssnapshot_asof_binding_body_mismatch:%s" % (BLOCK_PREFIX, snapshot_id)
        )
    champion_artifact_keys = ("model", "preprocessor", "calibrator")
    for artifact_name in champion_artifact_keys:
        expected_hash = (champion_artifacts.get(artifact_name) or {}).get("sha256")
        if bindings.get("%s_hash" % artifact_name) != expected_hash:
            raise MaturedEvaluationError(
                "%ssnapshot_%s_hash_not_contract_champion:%s"
                % (BLOCK_PREFIX, artifact_name, snapshot_id)
            )

    _assert_run_registry_status(root, snapshot_id, contract)
    return snapshot, entry, contract


def assert_snapshot_after_contract_freeze(
    snapshot: Mapping[str, Any], contract: Wp9Contract, root: Path
) -> None:
    freeze_iso = contract_freeze_timestamp(contract, root=root)
    asof_stamp = _utc_stamp(snapshot.get("snapshot_asof"))
    freeze_stamp = _utc_stamp(freeze_iso)
    if pd.isna(asof_stamp) or pd.isna(freeze_stamp):
        raise MaturedEvaluationError(
            "%sunparseable_asof_or_freeze_timestamp" % BLOCK_PREFIX
        )
    if not asof_stamp > freeze_stamp:
        raise MaturedEvaluationError(
            "%ssnapshot_asof_not_after_contract_freeze:asof=%s freeze=%s"
            % (BLOCK_PREFIX, snapshot.get("snapshot_asof"), freeze_iso)
        )


def snapshot_frame(snapshot: Mapping[str, Any]) -> pd.DataFrame:
    """Normal row-wise frame of an official snapshot plus ``feature_asof``."""
    keys = ["security_id", "ticker", "raw_model_score", "frozen_calibrated_score", "rank", "percentile"]
    lengths = {key: len(snapshot[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        raise MaturedEvaluationError("snapshot arrays are not length-consistent")
    count = lengths["security_id"]
    return pd.DataFrame(
        {
            "security_id": [str(value) for value in snapshot["security_id"]],
            "ticker": [str(value) for value in snapshot["ticker"]],
            "feature_asof": [str(snapshot["snapshot_asof"])] * count,
            "raw_model_score": [float(value) for value in snapshot["raw_model_score"]],
            "frozen_calibrated_score": [
                float(value) for value in snapshot["frozen_calibrated_score"]
            ],
            "rank": [int(value) for value in snapshot["rank"]],
            "percentile": [float(value) for value in snapshot["percentile"]],
        }
    )


# ── Certified outcome reconstruction ──────────────────────────────────────────

def certified_market_inputs(root: Path) -> Dict[str, Any]:
    """Load certified silver prices/actions and SPY benchmark inputs."""
    builder = _ForwardInputBuilder(root=root)
    layer_records = builder.load_layer_records()
    return {
        "prices": builder.load_prices(str(layer_records["silver_prices"])),
        "actions": builder.load_actions(str(layer_records["silver_actions"])),
        "benchmark_prices": builder.load_benchmark_prices(),
        "benchmark_actions": builder.load_benchmark_actions(),
        "layer_records": layer_records,
    }


def stock_action_objects(actions: pd.DataFrame) -> List[Any]:
    """Convert every silver action row to a PIT action keyed by ``security_id``."""
    objects: List[Any] = []
    if actions is None or actions.empty:
        return objects
    for security_id, chunk in actions.groupby("security_id", sort=False):
        objects.extend(actions_from_eodhd_rows(chunk.to_dict("records"), str(security_id)))
    return objects


def benchmark_action_objects(actions: pd.DataFrame) -> List[Any]:
    if actions is None or actions.empty:
        return []
    return actions_from_eodhd_rows(actions.to_dict("records"), BENCHMARK_SYMBOL)


def build_outcomes_for_snapshot(
    snapshot: Mapping[str, Any],
    *,
    root: Path | None = None,
    market_inputs: Mapping[str, Any] | None = None,
    horizon_months: int = HOLDING_MONTHS,
) -> pd.DataFrame:
    """Rebuild 12-month outcomes for the snapshot via the certified WP3 target path."""
    frame = snapshot_frame(snapshot)
    inputs = market_inputs or certified_market_inputs(Path(root or ROOT))

    panel = pd.DataFrame(
        {
            "security_id": frame["security_id"].tolist(),
            "ticker": frame["ticker"].tolist(),
            "snapshot_date": frame["feature_asof"].tolist(),
            "target_observable": [True] * len(frame),
        }
    )
    targets = build_targets(
        panel,
        inputs["prices"],
        inputs["benchmark_prices"],
        stock_actions=stock_action_objects(inputs["actions"]),
        benchmark_actions=benchmark_action_objects(inputs["benchmark_actions"]),
        horizon_months=horizon_months,
    )
    observability = annotate_observability(targets)
    targets = targets.reset_index(drop=True).copy()
    targets["target_known_at"] = observability["target_known_at"].to_numpy()

    target_keys = [
        "security_id",
        "feature_asof",
        "target_start",
        "target_end",
        "target_horizon_days_actual",
        "future_12m_stock_return",
        "future_12m_benchmark_return",
        "future_12m_excess_return",
        "outperform_12m",
        "target_observable",
        "target_censored",
        "target_censor_reason",
        "target_known_at",
    ]
    joined = frame.merge(targets.loc[:, target_keys], on=["security_id", "feature_asof"], how="left")
    if len(joined) != len(frame):
        raise MaturedEvaluationError(
            "%starget_join_changed_row_count:%d_to_%d" % (BLOCK_PREFIX, len(frame), len(joined))
        )
    joined["target_observable"] = (
        pd.to_numeric(joined["target_observable"], errors="coerce").fillna(0).astype(bool)
    )
    joined["target_censored"] = (
        pd.to_numeric(joined["target_censored"], errors="coerce").fillna(0).astype(bool)
    )
    return joined


def maturity_issues(
    outcomes: pd.DataFrame, evaluation_date: str | dt.datetime | None = None
) -> List[str]:
    """Return exact blocking reasons when a snapshot has not matured.

    A snapshot is matured only when every observable target row has a concrete
    ``target_known_at <= evaluation_date``. Censored rows carry no
    ``target_known_at`` and are excluded from metrics, never treated as failures.
    If no observable outcome exists at the evaluation date the snapshot is
    refused (its horizon is still open or its sources are missing).
    """
    if "target_known_at" not in outcomes.columns:
        return ["%starget_known_at_column_missing" % BLOCK_PREFIX]
    run_at = _utc_stamp(evaluation_date or dt.datetime.now(dt.timezone.utc))
    if pd.isna(run_at):
        return ["%sunparseable_evaluation_date" % BLOCK_PREFIX]

    observable = outcomes["target_observable"].astype(bool)
    known_values = outcomes.loc[observable, "target_known_at"]
    if int(observable.sum()) == 0:
        return [
            "%sno_observable_outcomes_at_evaluation_date:%s"
            % (BLOCK_PREFIX, run_at.isoformat())
        ]

    issues = []
    for value in known_values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            issues.append("%sobservable_row_missing_target_known_at" % BLOCK_PREFIX)
            break
        stamp = _utc_stamp(value)
        if pd.isna(stamp):
            issues.append("%sobservable_row_unparseable_target_known_at" % BLOCK_PREFIX)
            break
        if stamp > run_at:
            issues.append(
                "%starget_known_at_after_evaluation_date:%s>%s"
                % (BLOCK_PREFIX, stamp.isoformat(), run_at.isoformat())
            )
            break
    return issues


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_auc(frame: pd.DataFrame) -> Dict[str, Any]:
    """Cross-sectional ROC-AUC for the frozen calibrated score."""
    metric_frame = frame.loc[
        :, ["frozen_calibrated_score", "outperform_12m"]
    ].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(metric_frame))
    if n == 0:
        return {"n": 0, "roc_auc": None, "auc_skill": None}
    labels = metric_frame["outperform_12m"].to_numpy(dtype="float64")
    scores = metric_frame["frozen_calibrated_score"].to_numpy(dtype="float64")
    roc_auc = None
    if 0 < labels.sum() < n:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(labels, scores))
    auc_skill = None if roc_auc is None else roc_auc - 0.5
    return {"n": n, "roc_auc": roc_auc, "auc_skill": auc_skill}


def _finite_return(values: Sequence[Any]) -> float | None:
    valid = []
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            valid.append(number)
    return float(np.mean(valid)) if valid else None


def economic_shadow_metrics(
    snapshot: Mapping[str, Any], outcomes: pd.DataFrame
) -> Dict[str, Any]:
    """Predeclared economic shadow results with the fixed top-quintile cutoff.

    The top cohort is exactly ``percentile > 0.80``. The bottom cohort is the
    symmetric fixed ``percentile < 0.20`` band used only for descriptive
    top-minus-bottom spread reporting; neither cutoff is searched.
    """
    economic = select_top_quintile(snapshot, threshold=TOP_QUINTILE_THRESHOLD)
    top_ids = [str(item["security_id"]) for item in economic["constituents"]]
    bottom_ids = [
        str(security_id)
        for position, security_id in enumerate(snapshot["security_id"])
        if float(snapshot["percentile"][position]) < BOTTOM_QUINTILE_THRESHOLD
    ]

    rows_by_security: Dict[str, Dict[str, Any]] = {}
    for row in outcomes.to_dict("records"):
        rows_by_security.setdefault(str(row.get("security_id")), row)

    def cohort_metrics(security_ids: Sequence[str]) -> Dict[str, Any]:
        matched = [rows_by_security[str(security_id)] for security_id in security_ids if str(security_id) in rows_by_security]
        observable = [row for row in matched if bool(row.get("target_observable", False))]
        censored = [row for row in matched if not bool(row.get("target_observable", False))]
        stock_mean = _finite_return([row.get("future_12m_stock_return") for row in observable])
        bench_mean = _finite_return([row.get("future_12m_benchmark_return") for row in observable])
        excess = None
        if stock_mean is not None and bench_mean is not None:
            excess = float(stock_mean) - float(bench_mean)
        winners = [
            bool(row["outperform_12m"])
            for row in observable
            if row.get("outperform_12m") is not None
            and (not isinstance(row["outperform_12m"], float) or pd.notna(row["outperform_12m"]))
        ]
        return {
            "n_selected": int(len(security_ids)),
            "n_observable": int(len(observable)),
            "n_censored": int(len(censored)),
            "mean_12m_return": stock_mean,
            "mean_benchmark_12m_return": bench_mean,
            "mean_12m_excess_return": excess,
            "hit_rate": float(np.mean([1.0 if value else 0.0 for value in winners])) if winners else None,
            "constituent_security_ids": [str(value) for value in security_ids],
            "censored_security_ids": sorted(str(row["security_id"]) for row in censored),
        }

    top = cohort_metrics(top_ids)
    bottom = cohort_metrics(bottom_ids)
    spread = None
    if top.get("mean_12m_return") is not None and bottom.get("mean_12m_return") is not None:
        spread = float(top["mean_12m_return"]) - float(bottom["mean_12m_return"])

    cost_scenarios = []
    for cost_bps in COST_SCENARIOS_BPS:
        net_return = None
        excess = None
        gross_return = top.get("mean_12m_return")
        benchmark_return = top.get("mean_benchmark_12m_return")
        if gross_return is not None:
            net_return = float(gross_return) - float(cost_bps) / 10000.0
        if net_return is not None and benchmark_return is not None:
            excess = float(net_return) - float(benchmark_return)
        cost_scenarios.append(
            {
                "entry_cost_bps": int(cost_bps),
                "cohort_net_12m_return": net_return,
                "cohort_net_excess_return": excess,
            }
        )

    return {
        "top_quintile": top,
        "bottom_quintile": bottom,
        "top_minus_bottom_12m_return_spread": spread,
        "cost_scenarios": cost_scenarios,
        "benchmark_symbol": BENCHMARK_SYMBOL,
        "holding_months": HOLDING_MONTHS,
        "weighting": "equal_weight_at_entry",
        "real_trading": False,
        "cost_interpretation": (
            "single entry-side deduction from cohort 12m return (cost_bps/10000); "
            "benchmark unadjusted; report all scenarios"
        ),
    }


def newey_west_one_sided(values: Sequence[float], lag: int = HAC_LAG) -> Dict[str, Any]:
    """One-sided Newey-West/Bartlett HAC test that mean skill is <= 0.

    Returns the mean, HAC SE, t-statistic, and one-sided upper-tail p-value.
    For an empty or short series the result is ``valid=False``, never a
    fabricated p-value.
    """
    x = np.asarray(values, dtype="float64")
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {
            "n": n,
            "mean": float(x.mean()) if n else None,
            "hac_se": None,
            "t_stat": None,
            "one_sided_p": None,
            "lag": int(lag),
            "valid": False,
        }
    mean = float(x.mean())
    errors = x - mean
    long_run_variance = float(np.mean(errors * errors))
    lag_used = min(int(lag), n - 1)
    for j in range(1, lag_used + 1):
        gamma = float(np.mean(errors[j:] * errors[:-j]))
        weight = 1.0 - (j / (lag_used + 1.0))
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return {
            "n": n,
            "mean": mean,
            "hac_se": None,
            "t_stat": None,
            "one_sided_p": None,
            "lag": lag_used,
            "valid": False,
        }
    se = float(np.sqrt(long_run_variance / n))
    if not np.isfinite(se) or se <= 0.0:
        return {
            "n": n,
            "mean": mean,
            "hac_se": se,
            "t_stat": None,
            "one_sided_p": None,
            "lag": lag_used,
            "valid": False,
        }
    t_stat = float(mean / se)
    from scipy import stats

    p_one_sided = float(stats.norm.sf(t_stat)) if t_stat > 0.0 else 1.0
    return {
        "n": n,
        "mean": mean,
        "hac_se": se,
        "t_stat": t_stat,
        "one_sided_p": p_one_sided,
        "lag": lag_used,
        "valid": True,
    }


def matured_auc_skill_series(root: Path) -> np.ndarray:
    """Collect existing matured ``auc_skill`` values from the evaluation directory."""
    directory = root / EVALUATIONS_DIR
    if not directory.is_dir():
        return np.array([], dtype="float64")
    values = []
    for path in sorted(directory.glob("*.json")):
        record = _read_json(path)
        metric = record.get("primary_metrics") or {}
        value = metric.get("auc_skill")
        if value is not None and float(value) == value:
            values.append(float(value))
    return np.asarray(values, dtype="float64")


def prospective_inference(
    auc_skills: Sequence[float], *, lag: int = HAC_LAG
) -> Dict[str, Any]:
    """One-sided HAC evidence over equal-weighted monthly AUC skills."""
    values = [float(value) for value in auc_skills if value is not None]
    hac = newey_west_one_sided(values, lag=lag)
    return {
        "n_matured_snapshots": len(values),
        "equal_weighted_mean_auc_skill": float(np.mean(values)) if values else None,
        "hac": hac,
    }


def _market_input_fingerprints(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic data-source identity for outcome reconstruction inputs."""
    payload: Dict[str, Any] = {}
    for name in ("prices", "actions", "benchmark_prices", "benchmark_actions"):
        value = inputs.get(name)
        if isinstance(value, pd.DataFrame):
            payload[name] = fingerprint_dataframe(value)
        else:
            payload[name] = fingerprint_obj(value)
    payload["layer_records"] = dict(inputs.get("layer_records") or {})
    payload["benchmark_name"] = BENCHMARK_NAME
    payload["benchmark_version"] = BENCHMARK_VERSION
    return payload


def build_evaluation_content(
    snapshot: Mapping[str, Any],
    outcome: pd.DataFrame,
    *,
    evaluation_date: str,
    contract: Wp9Contract,
    code_commit: str,
    snapshot_file_sha256: str,
    market_source_hashes: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct the deterministic matured-evaluation content record."""
    evaluated_at_utc = _utc_iso(evaluation_date)
    if evaluated_at_utc is None:
        raise MaturedEvaluationError(
            "%sunparseable_evaluation_date" % BLOCK_PREFIX
        )
    primary = classification_auc(outcome.loc[outcome["target_observable"].astype(bool)])
    evidence_keys = [
        "security_id",
        "ticker",
        "feature_asof",
        "target_start",
        "target_end",
        "target_known_at",
        "target_observable",
        "target_censored",
        "target_censor_reason",
        "future_12m_stock_return",
        "future_12m_benchmark_return",
        "future_12m_excess_return",
        "outperform_12m",
    ]
    evidence = (
        outcome.loc[:, evidence_keys]
        .sort_values(["security_id", "feature_asof"], kind="mergesort")
        .to_dict("records")
    )
    for row in evidence:
        for key, value in list(row.items()):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                row[key] = None
    return {
        "schema_version": MATURED_EVALUATION_SCHEMA,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_asof": snapshot["snapshot_asof"],
        "evaluation_date": evaluation_date,
        "evaluated_at_utc": evaluated_at_utc,
        "target_set_id": TARGET_SET_ID,
        "target_version": TARGET_VERSION,
        "contract_version": contract.version,
        "contract_digest": contract.digest,
        "champion_freeze": FROZEN_CHAMPION,
        "code_commit": code_commit,
        "snapshot_file_sha256": snapshot_file_sha256,
        "market_source_hashes": market_source_hashes,
        "censoring_policy": (
            "censored rows retained and excluded from metrics; never treated as failures"
        ),
        "primary_metrics": primary,
        "outcome_counts": {
            "scoreable": int(len(outcome)),
            "observable": int(outcome["target_observable"].astype(bool).sum()),
            "censored": int(outcome["target_censored"].astype(bool).sum()),
        },
        "economic_shadow": economic_shadow_metrics(snapshot, outcome),
        "row_evidence_outcome_keys": evidence_keys,
        "outcome_rows": evidence,
    }


def evaluation_identity(content: Mapping[str, Any]) -> str:
    """Deterministic matured-evaluation identity bound to canonical content."""
    digest = fingerprint_obj(dict(content))
    return "wp9mat_%s" % digest[:20]


def persist_evaluation(
    content: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> Tuple[str, str]:
    """Persist one immutable matured evaluation; return ``(id, relative path)``."""
    root = Path(root or ROOT)
    evaluation_id = evaluation_identity(content)
    payload = {**dict(content), "evaluation_id": evaluation_id}
    path = root / EVALUATIONS_DIR / ("%s.json" % evaluation_id)
    save_immutable(path, payload)
    return evaluation_id, str(path.relative_to(root))


def _existing_matued_reference(registry: Mapping[str, Any], snapshot_id: str) -> Dict[str, Any] | None:
    for record in registry.get("matured_evaluations", []):
        if record.get("snapshot_id") == snapshot_id:
            return dict(record)
    return None


def evaluate_snapshot(
    snapshot_id: str,
    *,
    root: Path | None = None,
    evaluation_date: str | None = None,
    market_inputs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full matured evaluation for one official snapshot.

    ``market_inputs`` and ``evaluation_date`` are injection points for tests and
    deterministic replay. When absent, live certified inputs are loaded.
    """
    root = Path(root or ROOT)
    snapshot, _entry, contract = load_official_snapshot(snapshot_id, root=root)
    assert_snapshot_after_contract_freeze(snapshot, contract, root)

    eval_iso = _utc_iso(evaluation_date or dt.datetime.now(dt.timezone.utc))
    if eval_iso is None:
        raise MaturedEvaluationError(
            "%sunparseable_evaluation_date" % BLOCK_PREFIX
        )

    inputs = market_inputs if market_inputs is not None else certified_market_inputs(root)
    outcomes = build_outcomes_for_snapshot(snapshot, root=root, market_inputs=inputs)
    issues = maturity_issues(outcomes, eval_iso)
    if issues:
        return {
            "snapshot_id": snapshot_id,
            "evaluation_date": eval_iso,
            "status": "IMMATURE_OR_INVALID",
            "blocking_reasons": issues,
        }

    code_commit = current_git_commit(str(root)) or "<unavailable>"
    snapshot_file_sha256 = fingerprint_file(snapshot_file_path(root, snapshot_id))
    source_hashes = _market_input_fingerprints(inputs)
    content = build_evaluation_content(
        snapshot,
        outcomes,
        evaluation_date=eval_iso,
        contract=contract,
        code_commit=code_commit,
        snapshot_file_sha256=snapshot_file_sha256,
        market_source_hashes=source_hashes,
    )

    registry = load_run_registry(root=root)
    existing = _existing_matued_reference(registry, snapshot_id)
    if existing:
        return {
            "snapshot_id": snapshot_id,
            "evaluation_id": existing["evaluation_id"],
            "evaluation_path": existing["evaluation_path"],
            "status": "ALREADY_MATURED",
            "evaluation_date": eval_iso,
            "primary_metrics": content["primary_metrics"],
            "outcome_counts": content["outcome_counts"],
        }

    evaluation_id, rel_path = persist_evaluation(content, root=root)
    registry = add_matured_evaluation_to_run_registry(
        snapshot_id, evaluation_id, rel_path, root=root
    )
    skills = matured_auc_skill_series(root)
    inference = prospective_inference(skills)
    return {
        "snapshot_id": snapshot_id,
        "evaluation_id": evaluation_id,
        "evaluation_path": rel_path,
        "status": "WRITTEN",
        "evaluation_date": eval_iso,
        "primary_metrics": content["primary_metrics"],
        "outcome_counts": content["outcome_counts"],
        "prospective_inference": inference,
        "registry_stage": registry.get("current_prospective_stage"),
        "matured_evaluations_count": len(registry.get("matured_evaluations", [])),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP9 matured forward-snapshot evaluator")
    parser.add_argument("--snapshot-id", required=True, help="official snapshot id to evaluate")
    parser.add_argument("--evaluation-date", help="deterministic evaluation date (ISO); default now")
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    try:
        result = evaluate_snapshot(
            args.snapshot_id,
            root=root,
            evaluation_date=args.evaluation_date,
        )
    except (MaturedEvaluationError, Wp9ContractError, Wp9StorageError) as exc:
        print("EVALUATION_BLOCKED %s" % exc)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
