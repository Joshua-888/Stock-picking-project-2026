"""Lineage graph over research artefacts.

Edge direction is fixed by the research pipeline:
DATASET -> TARGET_SET -> FEATURE_SET -> EXPERIMENT -> MODEL -> PREDICTION ->
VALIDATION -> AUDIT -> PROMOTION. A parent must always precede its child and
every referenced node must resolve; dangling references raise ``LineageError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ids import parse_id

KIND_ORDER = (
    "dataset",
    "target_set",
    "feature_set",
    "experiment",
    "model",
    "prediction",
    "validation",
    "audit",
    "promotion",
)

_KIND_RANK = {kind: rank for rank, kind in enumerate(KIND_ORDER)}


class LineageError(RuntimeError):
    """Raised for dangling edges, duplicate conflicting nodes or cycles."""


@dataclass(frozen=True)
class LineageNode:
    """One artefact node and the artefacts it directly depends on."""

    node_id: str
    kind: str
    parents: tuple = ()
    refs: dict = field(default_factory=dict)


class LineageGraph:
    """In-memory lineage DAG built from manifests and registry records."""

    def __init__(self):
        self._nodes = {}

    @property
    def nodes(self):
        return dict(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node_id):
        return node_id in self._nodes

    def add_node(self, node_id, kind=None, parents=(), refs=None):
        """Add or idempotently re-add a node; conflicting re-adds raise."""
        if kind is None:
            kind, _digest = parse_id(node_id)
        if kind not in KIND_ORDER:
            raise LineageError("unknown lineage kind %r" % (kind,))
        node = LineageNode(node_id=node_id, kind=kind, parents=tuple(parents), refs=dict(refs or {}))
        existing = self._nodes.get(node_id)
        if existing is not None:
            if existing != node:
                raise LineageError("node %s already exists with different lineage" % node_id)
            return existing
        self._nodes[node_id] = node
        return node

    def ensure_node(self, node_id, kind=None):
        """Add ``node_id`` as a parentless node only when it is not present yet."""
        if node_id in self._nodes:
            return self._nodes[node_id]
        return self.add_node(node_id, kind=kind)

    def parents_of(self, node_id):
        if node_id not in self._nodes:
            raise LineageError("unknown lineage node: %s" % node_id)
        return self._nodes[node_id].parents

    def _order_key(self, node_id):
        node = self._nodes.get(node_id)
        rank = _KIND_RANK[node.kind] if node is not None else len(KIND_ORDER)
        return (rank, node_id)

    def resolve_lineage(self, node_id):
        """Return the ancestor chain of ``node_id``, root-first."""
        if node_id not in self._nodes:
            raise LineageError("unknown lineage node: %s" % node_id)
        ordered = []
        seen = set()

        def visit(current):
            if current in seen:
                return
            if current not in self._nodes:
                raise LineageError("dangling lineage reference: %s" % current)
            seen.add(current)
            for parent in sorted(self._nodes[current].parents, key=self._order_key):
                visit(parent)
            ordered.append(current)

        visit(node_id)
        return ordered

    def validate_lineage(self):
        """Raise :class:`LineageError` unless every edge resolves and is ordered."""
        for node_id, node in sorted(self._nodes.items()):
            for parent in node.parents:
                if parent not in self._nodes:
                    raise LineageError("dangling lineage reference: %s -> %s" % (node_id, parent))
                parent_kind = self._nodes[parent].kind
                if _KIND_RANK[parent_kind] >= _KIND_RANK[node.kind]:
                    raise LineageError(
                        "invalid lineage edge %s (%s) -> %s (%s)" % (parent, parent_kind, node_id, node.kind)
                    )
        return True


def validate_lineage(graph):
    """Module-level convenience wrapper around ``LineageGraph.validate_lineage``."""
    return graph.validate_lineage()


def _manifest_rows(manifests):
    if manifests is None:
        return []
    if isinstance(manifests, dict):
        manifests = list(manifests.values())
    rows = []
    for manifest in manifests:
        rows.append(manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest))
    return rows


_REF_FIELDS = (("dataset_id", "dataset"), ("target_set_id", "target_set"), ("feature_set_id", "feature_set"))


def _manifest_edge(row):
    """Return ``(node_id, kind, parents)`` for one manifest row, or None."""
    if row.get("target_set_id"):
        return row["target_set_id"], "target_set", tuple(filter(None, (row.get("dataset_id"),)))
    if row.get("feature_set_id"):
        return row["feature_set_id"], "feature_set", tuple(filter(None, (row.get("dataset_id"),)))
    if row.get("dataset_id"):
        return row["dataset_id"], "dataset", ()
    return None


def build_lineage(manifests=None, registry=None):
    """Build a :class:`LineageGraph` from manifests and/or a registry.

    Manifest nodes are added with their recorded edges first. Registry records
    then add experiment/model nodes and ensure every referenced artefact exists,
    so a broken reference surfaces in ``validate_lineage`` instead of being
    silently absorbed.
    """
    graph = LineageGraph()
    rows = _manifest_rows(manifests)
    for row in rows:
        edge = _manifest_edge(row)
        if edge is None:
            continue
        node_id, kind, parents = edge
        if node_id not in graph:
            graph.add_node(node_id, kind=kind, parents=parents)
        for parent in parents:
            graph.ensure_node(parent)

    if registry is not None:
        records = registry.list_experiments()
        for record in records:
            for field_name, kind in _REF_FIELDS:
                referenced = record.get(field_name)
                if referenced:
                    graph.ensure_node(referenced, kind=kind)
        for record in records:
            experiment = record.get("experiment_id")
            if not experiment:
                continue
            parents = tuple(
                sorted(filter(None, (record.get("dataset_id"), record.get("target_set_id"), record.get("feature_set_id"))))
            )
            graph.add_node(experiment, kind="experiment", parents=parents)
        for record in records:
            model = record.get("model_id")
            experiment = record.get("experiment_id")
            if not model or not experiment:
                continue
            graph.ensure_node(experiment, kind="experiment")
            graph.add_node(model, kind="model", parents=(experiment,))
    return graph
