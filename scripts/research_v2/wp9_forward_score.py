"""WP9 prospective shadow scoring engine.

Apply the immutable WP8 champion to one forward snapshot, never fitting,
selecting features, retraining, recalibrating, or adapting anything. The
snapshot rank is deterministic: ascending by ``raw_model_score`` with
``security_id`` lexical ascending as the tie-break.

Modes
-----
``--dry-run`` writes to the separated dry-run location and can never enter the
official prediction index.

``--official --as-of YYYY-MM-DD`` writes an immutable official snapshot after a
fail-closed preflight. Official snapshots require the frozen contract bytes to
match the committed repository blob, an as-of date strictly after the contract
freeze timestamp, a valid monthly-cadence date, exact frozen-artifact hashes,
no duplicate as-of, no target/outcome columns, and a clean tracked producing
code worktree.

the scorer neither reads nor requests future outcomes. This is enforced by the
column-name guard; the WP9 import surface loads only frozen inference and
point-in-time input-builders, never target-evaluation modules.

Run with the research runtime:

    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_forward_score.py --dry-run
    PYTHONPATH=. /opt/venv/bin/python scripts/research_v2/wp9_forward_score.py --official --as-of 2026-10-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.ids import canonical_json  # noqa: E402
from src.research.modes import current_git_commit  # noqa: E402
from src.research.wp9.champion import FrozenChampion, load_champion  # noqa: E402
from src.research.wp9.contract import (  # noqa: E402
    Wp9Contract,
    Wp9ContractError,
    assert_contract_file_committed,
    contract_freeze_timestamp,
    contract_path,
    load_wp9_contract,
    monthly_cadence_is_valid,
)
from src.research.wp9.economics import persist_economic_shadow, select_top_quintile  # noqa: E402
from src.research.wp9.forward_inputs import (  # noqa: E402
    ForwardInputError,
    ForwardSnapshotInputs,
    build_forward_inputs,
)
from src.research.wp9.health import compute_operational_health, operational_blockers  # noqa: E402
from src.research.wp9.storage import (  # noqa: E402
    DRY_RUN_KIND,
    OFFICIAL_KIND,
    Wp9StorageError,
    add_snapshot_to_run_registry,
    bindings_from_snapshot_id_inputs,
    has_prediction_for_asof,
    initialise_run_registry,
    load_run_registry,
    snapshot_id_from_bindings,
    snapshot_payload,
    write_snapshot,
)

SCHEMA_VERSION = "wp9_forward_validation_predictions_v1"
SCORING_ENGINE_VERSION = "wp9_forward_scoring_engine_v1"
OFFICIAL_MODE_TOKEN = "WP9_OFFICIAL_SNAPSHOT"
DRY_RUN_MODE_TOKEN = "NON_EVIDENTIARY_DRY_RUN"

BANNED_TARGET_COLUMNS = (
    "outperform_12m",
    "future_12m_excess_return",
    "future_12m_stock_return",
    "future_12m_benchmark_return",
    "future_return",
    "target_end",
    "target_start",
    "target_known_at",
    "target_observable",
    "target_censored",
    "target_censor_reason",
)

OFFICIAL_BLOCK_PREFIX = "WP9_OFFICIAL_BLOCK:"


class Wp9ScoringError(RuntimeError):
    """Raised when WP9 scoring cannot proceed honestly."""


def _git(args: Sequence[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    return output or None


def _git_required(args: Sequence[str], root: Path) -> str:
    value = _git(args, root)
    if value is None:
        raise Wp9ScoringError(
            "%sgit_required_failed:%s" % (OFFICIAL_BLOCK_PREFIX, " ".join(args))
        )
    return value


def assert_contract_bytes_committed(contract: Wp9Contract, root: Path) -> None:
    """Ensure the contract working bytes are tracked, committed, and unchanged.

    The contract digest is a raw-byte SHA-256. Git's default blob hash is
    SHA-1, so identity here is proven by asking Git whether the working
    tracked file differs from HEAD rather than comparing those two hash
    formats directly.
    """
    path = contract_path(root)
    rel = str(path.relative_to(root))
    assert_contract_file_committed(contract, root=root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != contract.digest:
        raise Wp9ScoringError(
            "%scontract_digest_mismatch:working_file_digest=%s contract_digest=%s"
            % (OFFICIAL_BLOCK_PREFIX, digest, contract.digest)
        )
    # Check the tracked file is byte-identical to HEAD. `git diff --quiet`
    # exits 0 only when they match, including a rename of the working path.
    try:
        completed = subprocess.run(
            ["git", "diff", "--quiet", "--exit-code", "HEAD", "--", rel],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wp9ScoringError(
            "%scontract_working_bytes_not_at_head:git_diff_unavailable:%s"
            % (OFFICIAL_BLOCK_PREFIX, exc)
        ) from exc
    if completed.returncode != 0:
        raise Wp9ScoringError(
            "%scontract_working_bytes_not_at_head:path=%s" % (OFFICIAL_BLOCK_PREFIX, rel)
        )


def assert_no_target_columns(*frames: pd.DataFrame) -> None:
    """Reject any frame containing a target, censoring, or future-return column."""
    for position, frame in enumerate(frames):
        if frame is None:
            continue
        present = [str(name).lower() for name in frame.columns]
        banned = sorted(BANNED_TARGET_COLUMNS)
        matched = sorted(set(present).intersection(banned))
        if matched:
            raise Wp9ScoringError(
                "%starget_outcome_columns_present:frame=%d columns=%s"
                % (OFFICIAL_BLOCK_PREFIX, position, ",".join(matched))
            )


def assert_asof_after_freeze(asof: str, contract: Wp9Contract, root: Path) -> None:
    freeze_iso = contract_freeze_timestamp(contract, root=root)
    try:
        asof_stamp = pd.Timestamp(asof).tz_localize("UTC")
        freeze_stamp = pd.Timestamp(freeze_iso).tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise Wp9ScoringError(
            "%sasof_or_freeze_timestamp_unparseable:%s" % (OFFICIAL_BLOCK_PREFIX, exc)
        ) from exc
    if pd.isna(asof_stamp) or pd.isna(freeze_stamp):
        raise Wp9ScoringError(
            "%sasof_or_freeze_timestamp_unparseable" % OFFICIAL_BLOCK_PREFIX
        )
    if not asof_stamp > freeze_stamp:
        raise Wp9ScoringError(
            "%sasof_not_after_contract_freeze:asof=%s freeze=%s"
            % (OFFICIAL_BLOCK_PREFIX, asof, freeze_iso)
        )


def assert_monthly_cadence(asof: str) -> None:
    if not monthly_cadence_is_valid(asof):
        raise Wp9ScoringError(
            "%sasof_invalid_monthly_cadence:asof=%s" % (OFFICIAL_BLOCK_PREFIX, asof)
        )


def assert_no_duplicate_snapshot(asof: str, root: Path) -> None:
    if has_prediction_for_asof(asof, root=root):
        raise Wp9ScoringError(
            "%sduplicate_snapshot_asof:asof=%s" % (OFFICIAL_BLOCK_PREFIX, asof)
        )


def _dirty_non_ignored_paths(root: Path) -> List[str]:
    output = _git_required(
        ["status", "--porcelain", "--untracked-files=no"], root
    )
    paths = set()
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().split(" -> ")[-1].split(" ")[0]
        if path:
            paths.add(path.replace("\\", "/"))
    return sorted(paths)


def _is_wp9_producing_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized == "scripts/research_v2/wp9_forward_score.py":
        return True
    if normalized in (
        "provenance/wp9/forward_validation_contract_v1.json",
        "provenance/wp8/final_candidate_freeze.json",
    ):
        return True
    return normalized.startswith("src/research/") or normalized.startswith(
        "scripts/research_v2/"
    )


def assert_clean_producing_worktree(root: Path) -> str:
    commit = current_git_commit(str(root))
    if not commit:
        raise Wp9ScoringError(
            "%sno_git_commit_for_reproducible_provenance" % OFFICIAL_BLOCK_PREFIX
        )
    dirty = [
        path for path in _dirty_non_ignored_paths(root) if _is_wp9_producing_path(path)
    ]
    if dirty:
        raise Wp9ScoringError(
            "%sdirty_producing_code:commit=%s paths=%s"
            % (OFFICIAL_BLOCK_PREFIX, commit, ",".join(dirty))
        )
    return commit


def preflight_official(
    asof: str,
    root: Path,
) -> Tuple[Wp9Contract, str]:
    """Fail-closed official-mode preflight. Returns ``(contract, code_commit)``."""
    try:
        contract = load_wp9_contract(root=root)
    except Wp9ContractError as exc:
        raise Wp9ScoringError(
            "%scontract_missing_or_invalid:%s" % (OFFICIAL_BLOCK_PREFIX, exc)
        ) from exc
    assert_contract_bytes_committed(contract, root)
    assert_asof_after_freeze(asof, contract, root)
    assert_monthly_cadence(asof)
    assert_no_duplicate_snapshot(asof, root)
    code_commit = assert_clean_producing_worktree(root)
    # Artifact hashes are verified by load_champion, which refuses any byte drift.
    load_champion(root=root)
    return contract, code_commit


def deterministic_rank_percentile(
    security_ids: Sequence[str],
    raw_scores: Sequence[float],
) -> Tuple[List[int], List[float]]:
    """Deterministic ascending raw-score rank with lexical security-id tie-break.

    Rank is 1-based sequential after sorting by (``raw_model_score`` ascending,
    ``security_id`` lexical ascending). Percentile is ``(rank-1)/(n-1)`` for
    ``n > 1`` and ``1.0`` for ``n == 1``.
    """
    count = len(security_ids)
    if count != len(raw_scores):
        raise Wp9ScoringError("security_ids and raw_scores must have equal length")
    order = sorted(range(count), key=lambda idx: (float(raw_scores[idx]), str(security_ids[idx])))
    ranks = [0] * count
    for position, idx in enumerate(order, start=1):
        ranks[idx] = position
    if count > 1:
        percentiles = [(rank - 1) / (count - 1) for rank in ranks]
    elif count == 1:
        percentiles = [1.0]
    else:
        percentiles = []
    return ranks, percentiles


def quality_flags(score_frame: pd.DataFrame, features: Sequence[str]) -> List[List[str]]:
    """Record per-row missing feature flags; imputation remains frozen-only."""
    rows: List[List[str]] = []
    for row_index in range(len(score_frame)):
        flags: List[str] = []
        for name in features:
            value = score_frame.iloc[row_index].get(name)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                flags.append("feature_missing:%s" % name)
        rows.append(flags)
    return rows


def score_frame(
    contract: Wp9Contract,
    champion: FrozenChampion,
    inputs: ForwardSnapshotInputs,
    code_commit: str,
    created_at_utc: str,
) -> Dict[str, Any]:
    """Score an input snapshot and return the exact row-wise snapshot body."""
    frame = inputs.score_frame
    assert_no_target_columns(frame, inputs.feature_frame)
    raw, calibrated = champion.predict(frame)
    ranks, percentiles = deterministic_rank_percentile(
        [str(value) for value in frame["security_id"]],
        [float(value) for value in raw],
    )
    bindings = bindings_from_snapshot_id_inputs(
        contract_digest=contract.digest,
        snapshot_asof=inputs.snapshot_asof,
        champion_freeze=champion.freeze_id,
        model_hash=champion.model_hash,
        preprocessor_hash=champion.preprocessor_hash,
        calibrator_hash=champion.calibrator_hash,
        universe_hash=inputs.universe_hash,
        source_manifest_hash=inputs.source_manifest_hash,
        feature_snapshot_hash=inputs.feature_snapshot_hash,
        code_commit=code_commit,
    )
    snapshot_id = snapshot_id_from_bindings(bindings)
    payload = snapshot_payload(
        snapshot_id=snapshot_id,
        snapshot_asof=inputs.snapshot_asof,
        security_id=[str(value) for value in frame["security_id"]],
        ticker=[str(value) for value in frame["ticker"]],
        raw_model_score=[float(value) for value in raw],
        frozen_calibrated_score=[float(value) for value in calibrated],
        rank=ranks,
        percentile=percentiles,
        universe_size=int(inputs.universe["universe_count"]),
        model_hash=champion.model_hash,
        feature_snapshot_hash=inputs.feature_snapshot_hash,
        source_manifest_hash=inputs.source_manifest_hash,
        code_commit=code_commit,
        created_at_utc=created_at_utc,
        eligibility_status=["scoreable"] * len(frame),
        quality_flags=quality_flags(frame, champion.features),
        model_freeze_id=champion.freeze_id,
    )
    return payload


def operational_health_record(
    inputs: ForwardSnapshotInputs,
    champion: FrozenChampion,
    payload: Mapping[str, Any],
    artifact_verification: Mapping[str, bool],
) -> Dict[str, Any]:
    return compute_operational_health(
        inputs.feature_frame,
        raw_scores=payload["raw_model_score"],
        calibrated_scores=payload["frozen_calibrated_score"],
        ranks=payload["rank"],
        universe={
            "snapshot_asof": inputs.snapshot_asof,
            "universe_count": inputs.universe["universe_count"],
        },
        feature_summary=inputs.feature_summary,
        artifact_verification=artifact_verification,
    )


def persist_health(
    payload: Mapping[str, Any],
    health: Mapping[str, Any],
    *,
    official: bool,
    root: Path,
) -> str:
    from src.research.immutability import save_immutable

    sub = "official" if official else "dry_runs"
    base = root / "provenance" / "wp9" / "operational_health" / sub
    path = base / ("%s.json" % payload["snapshot_id"])
    save_immutable(path, {"snapshot_id": payload["snapshot_id"], **dict(health)})
    return str(path.relative_to(root))


def ensure_run_registry(contract: Wp9Contract, root: Path) -> Dict[str, Any]:
    try:
        return load_run_registry(root=root)
    except Wp9StorageError:
        return initialise_run_registry(
            contract_version=contract.version,
            contract_digest=contract.digest,
            champion_freeze=contract.freeze_id,
            root=root,
        )


def execute(
    *,
    mode: str,
    asof: str,
    root: Path,
    inputs: ForwardSnapshotInputs | None = None,
    champion: FrozenChampion | None = None,
) -> Dict[str, Any]:
    """Run one scoring snapshot, returning a machine-readable evidence summary."""
    if mode not in (OFFICIAL_MODE_TOKEN, DRY_RUN_MODE_TOKEN):
        raise Wp9ScoringError("unknown scoring mode %r" % mode)
    root = Path(root)
    official = mode == OFFICIAL_MODE_TOKEN

    if official:
        contract, code_commit = preflight_official(asof, root)
    else:
        contract = load_wp9_contract(root=root)
        # Dry-run also records the current HEAD, but a dirty worktree does not
        # block a non-evidentiary rehearsal.
        code_commit = current_git_commit(str(root)) or "<unavailable>"
    champion = champion or load_champion(root=root)
    if inputs is None:
        inputs = build_forward_inputs(asof, root=root)

    created_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = score_frame(contract, champion, inputs, code_commit, created_at_utc)
    assert_no_target_columns(inputs.score_frame)

    artifact_verification = {
        "model": True,
        "preprocessor": True,
        "calibrator": True,
    }
    health = operational_health_record(inputs, champion, payload, artifact_verification)
    blockers = operational_blockers(health)
    if official and blockers:
        raise Wp9ScoringError(
            "%soperational_health_blocked:%s" % (OFFICIAL_BLOCK_PREFIX, ",".join(blockers))
        )

    bindings = bindings_from_snapshot_id_inputs(
        contract_digest=contract.digest,
        snapshot_asof=inputs.snapshot_asof,
        champion_freeze=champion.freeze_id,
        model_hash=champion.model_hash,
        preprocessor_hash=champion.preprocessor_hash,
        calibrator_hash=champion.calibrator_hash,
        universe_hash=inputs.universe_hash,
        source_manifest_hash=inputs.source_manifest_hash,
        feature_snapshot_hash=inputs.feature_snapshot_hash,
        code_commit=code_commit,
    )
    write_result = write_snapshot(
        payload,
        bindings,
        kind=OFFICIAL_KIND if official else DRY_RUN_KIND,
        root=root,
    )
    registry = ensure_run_registry(contract, root)
    add_snapshot_to_run_registry(
        payload["snapshot_id"],
        "official" if official else "dry_run",
        root=root,
        match_contract=contract.digest,
    )
    health_rel = persist_health(payload, health, official=official, root=root)

    economics_rel = None
    if official:
        economic = select_top_quintile(payload)
        economics_rel = persist_economic_shadow(economic, payload["snapshot_id"], root=root)

    return {
        "mode": mode,
        "snapshot_id": payload["snapshot_id"],
        "snapshot_asof": inputs.snapshot_asof,
        "universe_size": int(inputs.universe["universe_count"]),
        "scoreable_count": int(len(inputs.score_frame)),
        "excluded_count": max(0, int(inputs.universe["universe_count"]) - int(len(inputs.score_frame))),
        "artifact_hash_verification": artifact_verification,
        "operational_blockers": blockers,
        "operational_health_path": health_rel,
        "economics_shadow_path": economics_rel,
        "snapshot_path": write_result["path"] if isinstance(write_result, dict) else str(write_result),
        "write_outcome": write_result.get("outcome") if isinstance(write_result, dict) else None,
        "index_outcome": write_result.get("index_outcome") if isinstance(write_result, dict) else None,
        "code_commit": code_commit,
        "contract_digest": contract.digest,
        "registry_schema": registry.get("schema_version"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP9 forward snapshot scorer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="non-evidentiary rehearsal output")
    group.add_argument("--official", action="store_true", help="write an immutable official snapshot")
    parser.add_argument("--as-of", help="snapshot date YYYY-MM-DD")
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if args.dry_run:
        mode = DRY_RUN_MODE_TOKEN
        asof = args.as_of or dt.date.today().isoformat()
    else:
        mode = OFFICIAL_MODE_TOKEN
        if not args.as_of:
            print("OFFICIAL_REQUIRES_AS_OF")
            return 2
        asof = args.as_of
    try:
        result = execute(mode=mode, asof=asof, root=root)
    except (Wp9ScoringError, Wp9ContractError, Wp9StorageError, ForwardInputError) as exc:
        print("SCORING_BLOCKED %s" % exc)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
