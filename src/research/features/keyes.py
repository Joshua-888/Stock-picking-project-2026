"""WP3 Keyes tracks: A) HISTORICAL_KEYES_REPLICATION, B) MODERN_KEYES_INSPIRED.

James R. Keyes' 1972 study used a small set of variables and an explicit
logical rule set. The project preserves that as a SEPARATE, clearly labelled
track so the historical replication and the modern benchmark-relative model are
never conflated.

Two V1 behaviours are explicitly NOT carried into V2:

1. **Quota selection.** V1 forced roughly the top 30% of stocks to be
   "Keyes-qualified". Replication instead applies actual simultaneous logical
   thresholds (a stock qualifies only if it satisfies every criterion), never a
   percentile quota.
2. **Silent proxy substitution.** When the original variable is unavailable, the
   substitute is named ``*_PROXY`` with the mismatch recorded; a revenue-growth
   series is never silently reported as X12 expected appreciation.

Keyes variables: X5 five-year EPS growth, X6 five-year price gain, X8 P/E,
X9 current P/E vs historical P/E, X12 forward expected appreciation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRACK_HISTORICAL = "HISTORICAL_KEYES_REPLICATION"
TRACK_MODERN = "MODERN_KEYES_INSPIRED"

X12_PROXY_SUFFIX = "_PROXY"

# Canonical Keyes variables and their exact historical meaning.
KEYES_VARIABLES = {
    "X5": "five-year EPS growth",
    "X6": "five-year price gain",
    "X8": "P/E ratio",
    "X9": "current P/E relative to historical P/E",
    "X12": "forward expected appreciation",
}


class KeyesError(ValueError):
    """Raised when a Keyes qualification request is ill-formed."""


@dataclass(frozen=True)
class KeyesCriterion:
    """One explicit logical threshold in the replication rule set.

    ``direction`` is ``"min"`` (value must be >= threshold) or ``"max"`` (value
    must be <= threshold). This is a genuine simultaneous regression threshold,
    not a percentile cut.
    """

    variable: str
    threshold: float
    direction: str = "min"
    is_proxy: bool = False
    proxy_note: str = ""

    def evaluate(self, value):
        if value is None:
            return False
        if self.direction == "min":
            return float(value) >= float(self.threshold)
        if self.direction == "max":
            return float(value) <= float(self.threshold)
        raise KeyesError("direction must be 'min' or 'max', got %r" % (self.direction,))


@dataclass(frozen=True)
class KeyesRuleSet:
    """A named, explicit replication rule set (no quota, no free parameters)."""

    name: str
    track: str
    criteria: tuple = ()

    def qualifies(self, values):
        """True only when EVERY criterion is satisfied simultaneously.

        ``values`` maps variable name -> value. A missing value fails its
        criterion (never treated as zero or as passing).
        """
        if not self.criteria:
            raise KeyesError("rule set %r has no criteria" % self.name)
        for criterion in self.criteria:
            if not criterion.evaluate(values.get(criterion.variable)):
                return False
        return True

    def proxies_used(self):
        return [criterion.variable for criterion in self.criteria if criterion.is_proxy]


@dataclass
class KeyesQualification:
    """Per-security qualification outcome for one rule set."""

    security_id: str
    qualified: bool
    track: str
    rule_set: str
    detail: dict = field(default_factory=dict)
    proxies_used: tuple = ()

    def to_dict(self):
        return {
            "security_id": self.security_id,
            "qualified": bool(self.qualified),
            "track": self.track,
            "rule_set": self.rule_set,
            "proxies_used": list(self.proxies_used),
            "detail": dict(self.detail),
        }


def proxy_variable(base_variable):
    """Explicit name for a proxy of ``base_variable`` (never a silent substitute).

    For example, revenue growth standing in for X12 forward expected
    appreciation is named ``X12_PROXY``, so no downstream reader can mistake it
    for the true Keyes variable.
    """
    if not base_variable or not str(base_variable).strip():
        raise KeyesError("proxy_variable requires a base variable name")
    base = str(base_variable)
    if base.endswith(X12_PROXY_SUFFIX):
        return base
    return base + X12_PROXY_SUFFIX


def qualify(rule_set, security_id, values):
    """Evaluate one security against a rule set and record proxies honestly."""
    if rule_set.track not in (TRACK_HISTORICAL, TRACK_MODERN):
        raise KeyesError("unknown Keyes track %r" % (rule_set.track,))
    return KeyesQualification(
        security_id=str(security_id),
        qualified=rule_set.qualifies(values),
        track=rule_set.track,
        rule_set=rule_set.name,
        detail={criterion.variable: values.get(criterion.variable) for criterion in rule_set.criteria},
        proxies_used=tuple(rule_set.proxies_used()),
    )
