"""Research-mode gate and Git provenance helpers.

RESEARCH_V2 forbids synthetic data entirely; LEGACY_V1 keeps its historical
behaviour and only warns, so frozen V1 results can still be re-run.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from enum import Enum


class ResearchMode(str, Enum):
    LEGACY_V1 = "LEGACY_V1"
    RESEARCH_V2 = "RESEARCH_V2"


class SyntheticDataError(RuntimeError):
    """Raised when synthetic data would enter a research-mode pipeline."""


def assert_no_synthetic_in_research(mode, synthetic_flag, context=None):
    """Gate synthetic data out of RESEARCH_V2 pipelines.

    Returns True when the combination is admissible; raises SyntheticDataError
    when RESEARCH_V2 is combined with synthetic data.
    """
    resolved = mode if isinstance(mode, ResearchMode) else ResearchMode(mode)
    where = (" (%s)" % context) if context else ""
    if resolved is ResearchMode.RESEARCH_V2 and bool(synthetic_flag):
        raise SyntheticDataError(
            "synthetic data is not admissible in RESEARCH_V2%s; exclude the observation "
            "or fail dataset certification" % where
        )
    if resolved is ResearchMode.LEGACY_V1 and bool(synthetic_flag):
        warnings.warn("LEGACY_V1 run used synthetic data%s; not valid research evidence" % where, stacklevel=2)
    return True


def _git(args, cwd=None):
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd or os.getcwd(), capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    return output or None


def current_git_commit(cwd=None):
    """Current HEAD commit hash, or None when Git is unavailable."""
    return _git(["rev-parse", "HEAD"], cwd=cwd)


def current_branch(cwd=None):
    """Current branch name, or None when Git is unavailable."""
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
