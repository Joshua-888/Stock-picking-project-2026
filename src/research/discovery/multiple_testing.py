"""WP5 multiple-hypothesis control: Benjamini-Hochberg FDR.

Scanning many candidate features guarantees some impressive-looking p-value by
chance. Every candidate is therefore reported with BOTH its raw p-value and its
Benjamini-Hochberg adjusted q-value, and negatives are kept in the output rather
than dropped. The FDR verdict is descriptive (does this raw finding survive the
multiple-comparison correction?); it is NOT a promotion decision.
"""

from __future__ import annotations

from dataclasses import dataclass


class MultipleTestingError(ValueError):
    """Raised when an FDR request is ill-formed."""


@dataclass(frozen=True)
class FDRResult:
    feature: str
    p_value: float
    q_value: float
    rank: int
    rejected: bool

    def to_dict(self):
        return {"feature": self.feature, "p_value": self.p_value, "q_value": self.q_value,
                "rank": self.rank, "rejected": self.rejected}


def benjamini_hochberg(p_values, alpha=0.05):
    """Return ``(results, summary)`` for a mapping of feature -> raw p-value.

    ``results`` preserves the input order and is always the FULL set (a feature is
    never hidden for failing to reach significance). Ties in p-values share the
    same q-value.
    """
    if not isinstance(p_values, dict):
        raise MultipleTestingError("p_values must be a mapping of feature -> p-value")
    items = [(name, float(value)) for name, value in p_values.items()]
    for name, value in items:
        if not 0.0 <= value <= 1.0:
            raise MultipleTestingError("p-value out of range for %s: %r" % (name, value))
    count = len(items)
    if count == 0:
        return [], {"count": 0, "alpha": float(alpha), "rejected": 0}

    ordered = sorted(range(count), key=lambda index: (items[index][1], items[index][0]))
    q_values = [1.0] * count
    running_min = 1.0
    for position in range(count - 1, -1, -1):
        index = ordered[position]
        rank = position + 1
        candidate = items[index][1] * count / rank
        running_min = min(running_min, candidate)
        q_values[index] = min(1.0, running_min)

    results = []
    for index, (name, value) in enumerate(items):
        results.append(FDRResult(
            feature=name, p_value=value, q_value=q_values[index],
            rank=ordered.index(index) + 1, rejected=bool(q_values[index] <= alpha),
        ))
    rejected = sum(1 for record in results if record.rejected)
    summary = {"count": count, "alpha": float(alpha), "rejected": int(rejected),
               "note": "full candidate set retained; failing features are reported, not hidden"}
    return results, summary
