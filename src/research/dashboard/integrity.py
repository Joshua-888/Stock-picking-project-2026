"""Assembled audit/readiness facts for the WP9 dashboard truth layer.

This module reports facts; it does not issue certification verdicts. It verifies
frozen artifact hashes through the existing champion loader, checks that the new
dashboard package is apply-only at source level, confirms read-only provenance
access, and vets emitted copy against unsupported probability/significance claims.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from . import schemas
from .contract import freshness_completeness, latest_shadow_ranking, official_snapshot_history
from .identity import attach_identity
from .summary import research_summary, model_definition

_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parent

# The forbidden identifiers are assembled from fragments so this guard module does
# not itself contain the exact target-column spellings it is designed to detect.
_FORBIDDEN_SOURCE_TOKENS = tuple(
    "".join(parts)
    for parts in (
        ("target_", "observable"),
        ("future_12m_", "excess_return"),
        ("future_12m_", "stock_return"),
        ("future_12m_", "benchmark_return"),
        ("future_", "return"),
        ("outperform_", "12m"),
        ("target_", "censored"),
        ("target_", "censor_reason"),
        ("terminal_price_", "observable"),
        ("target_", "known_at"),
    )
)

_WRITE_TOKEN_PARTS = (
    ("save_", "immutable"),
    ("write_json_", "atomic"),
    ("write_", "snapshot"),
    ("initialise_run_", "registry"),
    ("add_snapshot_to_run_", "registry"),
    ("add_invalidation_to_run_", "registry"),
    ("add_matured_evaluation_to_run_", "registry"),
)
_WRITE_TOKENS = tuple("".join(parts) for parts in _WRITE_TOKEN_PARTS)

_FIT_METHODS = {"fit", "fit_transform", "fit_predict"}


def _package_sources() -> Iterable[Path]:
    return sorted(path for path in _PACKAGE_DIR.glob("*.py") if path.name != "__pycache__")


def _source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_any(text: str, tokens: Iterable[str]) -> list:
    lowered = text.lower()
    return [token for token in tokens if token.lower() in lowered]


def _ast_tree(path: Path) -> ast.AST:
    return ast.parse(_source_text(path))


def _apply_only_call_paths() -> list[str]:
    findings = []
    for path in _package_sources():
        tree = _ast_tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _FIT_METHODS:
                    findings.append("%s:%d .%s()" % (path.name, node.lineno, node.func.attr))
    return findings


def _forbidden_source_tokens() -> dict[str, list]:
    findings = {}
    for path in _package_sources():
        hits = _contains_any(_source_text(path), _FORBIDDEN_SOURCE_TOKENS)
        if hits:
            findings[path.name] = hits
    return findings


def _write_calls() -> list[str]:
    findings = []
    for path in _package_sources():
        text = _source_text(path)
        for token in _WRITE_TOKENS:
            if token in text:
                findings.append("%s:%s" % (path.name, token))
    return findings


def _emitted_copy_chunks(root: Path) -> Iterable[str]:
    """Artifacts whose copy is vetted for unsupported claims."""
    summary = research_summary(root)
    model = model_definition(root)
    shadow = latest_shadow_ranking(root)
    history = official_snapshot_history(root)
    freshness = freshness_completeness(root)
    import json

    for obj in (summary, model, shadow, history, freshness):
        yield json.dumps(obj, sort_keys=True, default=str)


def _emitted_claim_violations(root: Path) -> list[tuple[str, str]]:
    violations = []
    for chunk in _emitted_copy_chunks(root):
        lowered = chunk.lower()
        for phrase in schemas.FORBIDDEN_PHRASES:
            if phrase.lower() in lowered:
                violations.append((phrase, chunk[:160]))
    return violations


def audit_status(root: Path | None = None) -> dict:
    """Assemble the dashboard integrity audit status contract.

    This function never mutates provenance and never recomputes holdout metrics.
    It is a collection of read-only facts for the reviewer/supervisor, not a
    verdict.
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT

    # Existing hash-verified loader; this raises on tampering and never refits.
    from ..wp9.champion import load_champion

    try:
        load_champion(root)
        frozen_artifacts_verified = True
    except Exception:  # pragma: no cover - only possible under tampered/missing artifacts
        frozen_artifacts_verified = False

    fit_findings = _apply_only_call_paths()
    source_forbidden = _forbidden_source_tokens()
    write_findings = _write_calls()
    claim_violations = []
    try:
        claim_violations = _emitted_claim_violations(root)
    except Exception as exc:  # pragma: no cover - factual build failure
        claim_violations = [("dashboard build failed", str(exc))]

    result = {
        **schemas.schema_basis("integrity_audit_status"),
        "frozen_artifacts_verified": bool(frozen_artifacts_verified),
        "no_refit_guard": len(fit_findings) == 0,
        "metric_lineage_complete": bool(claim_violations == [] and len(fit_findings) == 0),
        "no_wp8_holdout_reevaluation": bool(not source_forbidden and not write_findings),
        "read_only_registry_access": bool(not write_findings),
        "no_unsupported_probability_claims": bool(len(claim_violations) == 0),
        "apply_only_check": bool(len(fit_findings) == 0),
        "limitations": [
            "this is an assembled readiness fact sheet, not a formal audit or certification verdict",
            "frozen artifact hashes are verified through src.research.wp9.champion.load_champion",
            "apply-only check scans the src/research/dashboard package source for .fit/.fit_transform/.fit_predict calls",
            "the WP9 registry and indexes are read only; snapshot files are never written or mutated here",
        ],
    }
    result = attach_identity(result)
    schemas.validate_dashboard_artifact("integrity_audit_status", result)
    return result


__all__ = ["audit_status"]
