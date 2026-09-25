"""Build and persist WP1 ``DatasetManifest`` records for GOLD datasets.

A manifest is only ever labelled ``point_in_time`` when a deterministic check
supports it. Otherwise the honest status (``partially_point_in_time`` /
``not_point_in_time``) is recorded. Synthetic data can never yield a
``point_in_time`` status, and in RESEARCH_V2 synthetic data is refused outright
by the research-mode gate before a manifest is written.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

from src.research.fingerprints import fingerprint_file, fingerprint_obj
from src.research.ids import dataset_id as make_dataset_id
from src.research.immutability import save_immutable
from src.research.manifests import DatasetManifest
from src.research.modes import (
    ResearchMode,
    assert_no_synthetic_in_research,
    current_branch,
    current_git_commit,
)

from .layers import layer_root

SCHEMA_VERSION = "research_v2_gold_v1"


class ManifestingError(RuntimeError):
    """Raised when a dataset cannot be honestly manifested."""


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def config_fingerprint(config=None):
    """Deterministic SHA-256 of the research configuration.

    Independent of the dataset fingerprint. Priority:

    * an explicit mapping -> canonical-JSON digest of that mapping;
    * an explicit path/string hash is not accepted (values only);
    * otherwise the bytes of ``config/config.yaml``.
    """
    if isinstance(config, dict):
        return fingerprint_obj(dict(config))
    if config is not None:
        raise ManifestingError("config must be a mapping when supplied, got %s" % type(config).__name__)
    return fingerprint_file(CONFIG_PATH)


def dataset_payload(name, universe_id, period_start, period_end, schema, source_fingerprints, dataset_fingerprint):
    """Canonical payload that determines a dataset id (deterministic)."""
    return {
        "name": name,
        "universe_id": universe_id,
        "period_start": str(period_start),
        "period_end": str(period_end),
        "schema": list(schema),
        "source_fingerprints": dict(source_fingerprints),
        "dataset_fingerprint": dataset_fingerprint,
        "schema_version": SCHEMA_VERSION,
    }


def build_gold_manifest(
    name,
    frame,
    *,
    mode,
    universe_id,
    period_start,
    period_end,
    sources,
    source_fingerprints,
    dataset_fingerprint,
    pit_status,
    synthetic=False,
    known_limitations=(),
    root=None,
    notes="",
    config=None,
    source_versions=None,
    security_master_version=None,
    universe_version=None,
    censoring_statistics=None,
):
    """Build (and validate) a ``DatasetManifest`` for one GOLD dataset.

    ``synthetic=True`` is rejected in RESEARCH_V2 and forces a non-PIT status in
    LEGACY_V1, so a manifest can never misrepresent synthetic rows as reliable
    research evidence.
    """
    import pandas as pd  # local import keeps module import light

    resolved_mode = mode if isinstance(mode, ResearchMode) else ResearchMode(mode)
    assert_no_synthetic_in_research(resolved_mode, synthetic, context="gold manifest %s" % name)

    if not isinstance(frame, pd.DataFrame):
        raise ManifestingError("gold frame for %r must be a pandas DataFrame" % name)
    if pit_status not in ("point_in_time", "partially_point_in_time", "not_point_in_time"):
        raise ManifestingError("unsupported pit_status %r" % (pit_status,))
    if synthetic and pit_status == "point_in_time":
        raise ManifestingError("synthetic data can never be labelled point_in_time")
    if not sources:
        raise ManifestingError("at least one source is required")

    synthetic_status = "synthetic" if synthetic else "none"
    payload = dataset_payload(
        name,
        universe_id,
        period_start,
        period_end,
        [str(column) for column in frame.columns],
        source_fingerprints,
        dataset_fingerprint,
    )
    identifier = make_dataset_id(payload)
    manifest = DatasetManifest(
        dataset_id=identifier,
        created_at=_now_iso(),
        git_commit=current_git_commit() or "unknown",
        branch=current_branch() or "unknown",
        sources=list(sources),
        universe_definition=str(universe_id),
        period_start=str(period_start),
        period_end=str(period_end),
        row_count=int(len(frame)),
        schema_version=SCHEMA_VERSION,
        config_fingerprint=config_fingerprint(config),
        source_fingerprints=dict(source_fingerprints),
        dataset_fingerprint=dataset_fingerprint,
        pit_status=pit_status,
        synthetic_data_status=synthetic_status,
        known_limitations=list(known_limitations),
        notes=notes,
        source_versions=dict(source_versions) if source_versions else None,
        security_master_version=security_master_version,
        universe_version=universe_version,
        censoring_statistics=dict(censoring_statistics) if censoring_statistics else None,
    )
    manifest.validate()
    return manifest


def persist_manifest(manifest, root=None, name=None):
    """Persist a manifest immutably under ``gold/<name>/manifest.json``.

    Identical content verifies and reuses; different content for the same
    dataset id raises via :func:`save_immutable`, so an approved dataset version
    can never be silently overwritten.
    """
    directory = layer_root(root, "gold", name or manifest.universe_definition)
    path = directory / ("%s.manifest.json" % manifest.dataset_id)
    outcome = save_immutable(path, manifest.to_dict())
    return {"path": str(path), "outcome": outcome, "dataset_id": manifest.dataset_id, "manifest": manifest.to_dict()}


def build_and_persist_gold(
    name,
    frame,
    *,
    mode,
    universe_id,
    period_start,
    period_end,
    sources,
    source_fingerprints,
    pit_status,
    synthetic=False,
    known_limitations=(),
    root=None,
    notes="",
    config=None,
    source_versions=None,
    security_master_version=None,
    universe_version=None,
    censoring_statistics=None,
):
    """Convenience wrapper: fingerprint, manifest, persist one GOLD dataset.

    The research-mode / synthetic gate runs FIRST, before any byte is written,
    so synthetic data can never reach the gold layer even transiently.
    """
    from . import layers

    resolved_mode = mode if isinstance(mode, ResearchMode) else ResearchMode(mode)
    assert_no_synthetic_in_research(resolved_mode, synthetic, context="gold dataset %s" % name)

    table = layers.write_gold_table(root, name, frame, meta={"universe_id": universe_id})
    dataset_fingerprint = table["fingerprint"]
    merged_sources = dict(source_fingerprints)
    merged_sources["gold_table:%s" % name] = dataset_fingerprint
    manifest = build_gold_manifest(
        name,
        frame,
        mode=mode,
        universe_id=universe_id,
        period_start=period_start,
        period_end=period_end,
        sources=sources,
        source_fingerprints=merged_sources,
        dataset_fingerprint=dataset_fingerprint,
        pit_status=pit_status,
        synthetic=synthetic,
        known_limitations=known_limitations,
        root=root,
        notes=notes,
        config=config,
        source_versions=source_versions,
        security_master_version=security_master_version,
        universe_version=universe_version,
        censoring_statistics=censoring_statistics,
    )
    persisted = persist_manifest(manifest, root=root, name=name)
    persisted["table"] = table
    return persisted


def verify_manifest_roundtrip(manifest):
    """Re-parse a manifest from its dict form and confirm stability."""
    restored = DatasetManifest.from_dict(manifest.to_dict())
    restored.validate()
    if restored.dataset_fingerprint != manifest.dataset_fingerprint:
        raise ManifestingError("manifest fingerprint changed across a roundtrip")
    return True
