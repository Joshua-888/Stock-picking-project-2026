"""WP3 feature-set manifest: bind a registry version to a dataset immutably.

A ``FeatureSetManifest`` states exactly which features, under which dataset
version and which transformation/missingness policy, a model may consume. The
``feature_set_id`` is deterministic over the manifest payload, so two engineers
who assemble the same set get the same id, and any change to a feature
(normalisation, missing rule, lookback, availability) produces a new id rather
than silently mutating an approved one.

This module reuses :class:`src.research.manifests.FeatureSetManifest` and
:func:`src.research.ids.feature_set_id`; it does not re-declare their contract.
"""

from __future__ import annotations

from ..fingerprints import fingerprint_obj
from ..ids import feature_set_id
from ..manifests import FeatureSetManifest
from .initial_set import build_initial_registry
from .registry import REGISTRY_VERSION

AVAILABILITY_POLICY_REF = "src.research.data.availability: FUNDAMENTAL / PRICE policies"
DEFAULT_MISSINGNESS_POLICY = "per-feature missing_rule; fitted on training only; never zero-filled"


class FeatureSetError(ValueError):
    """Raised when a feature set cannot be assembled consistently."""


def build_feature_set(dataset_id, git_commit, registry=None, include_macro=False,
                      availability_policy_ref=AVAILABILITY_POLICY_REF,
                      missingness_policy=DEFAULT_MISSINGNESS_POLICY, notes=""):
    """Assemble a validated :class:`FeatureSetManifest` for ``dataset_id``.

    Only ENABLED features are included, so a declared-but-disabled macro feature
    can never enter a model by accident. The fingerprint is taken over the
    registry's canonical payload.
    """
    if not dataset_id:
        raise FeatureSetError("dataset_id is required")
    if not git_commit:
        raise FeatureSetError("git_commit is required for reproducibility")
    registered = registry if registry is not None else build_initial_registry(include_macro=include_macro)
    enabled = registered.enabled_features()
    if not enabled:
        raise FeatureSetError("feature set has no enabled features")

    feature_names = [definition.feature_name for definition in enabled]
    duplicate = sorted({name for name in feature_names if feature_names.count(name) > 1})
    if duplicate:
        raise FeatureSetError("duplicate feature_name(s) in set: %s" % ", ".join(duplicate))

    transformations = sorted({definition.transformation for definition in enabled})
    fingerprint = fingerprint_obj(registered.fingerprint_payload())
    manifest = FeatureSetManifest(
        feature_set_id="feature_set_placeholder00",  # replaced below via payload id
        dataset_id=dataset_id,
        features=feature_names,
        feature_definition_versions=registered.definition_versions(),
        transformations=transformations,
        availability_policy_ref=availability_policy_ref,
        missingness_policy=missingness_policy,
        fingerprint=fingerprint,
        git_commit=git_commit,
        notes=notes,
    )
    # Deterministic id over the manifest payload minus the id itself.
    payload = manifest.to_dict()
    payload.pop("feature_set_id", None)
    payload["registry_version"] = REGISTRY_VERSION
    identifier = feature_set_id(payload)
    manifest = FeatureSetManifest(**{**manifest.to_dict(), "feature_set_id": identifier})
    manifest.validate()
    return manifest
