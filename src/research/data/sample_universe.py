"""Bounded sample-universe loader for WP2A dataset certification.

The sample universe is read deterministically from ``config/tickers.yaml`` (core
universe plus the hidden-gems pool) together with the configured benchmark. This
is a CURRENT-constituent sample: it is survivorship-biased by construction and
is therefore explicitly NOT certified survivorship-safe (see the universe
source decision doc).

This module only reads configuration; it performs no fetching, no feature
engineering, no model fitting and no scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TICKERS_PATH = Path("config/tickers.yaml")
DEFAULT_BENCHMARK = "SPY"
DEFAULT_SAMPLE_CAP = 60
SAMPLE_UNIVERSE_ID = "config_tickers_current_sample_v1"

KNOWN_LIMITATION_CURRENT_CONSTITUENTS = (
    "sample is drawn from present-day config constituents only; historical "
    "entrants/exits, delisted and acquired names are absent, so the sample is "
    "survivorship-biased and is not certified survivorship-safe"
)


class SampleUniverseError(RuntimeError):
    """Raised when the sample universe cannot be read."""


def _load_yaml(path):
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SampleUniverseError("PyYAML is required to read %s" % path) from exc
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _tickers_from_block(block, key="ticker"):
    tickers = []
    for entry in block or []:
        if isinstance(entry, dict) and entry.get(key):
            tickers.append(str(entry[key]).strip().upper())
        elif isinstance(entry, str):
            tickers.append(entry.strip().upper())
    return tickers


@dataclass(frozen=True)
class SampleUniverse:
    """A deterministic, bounded sample universe read from configuration."""

    universe_id: str
    benchmark: str
    tickers: tuple
    source_path: str
    core_count: int
    pool_count: int
    cap: int = None
    limitations: tuple = field(default_factory=tuple)

    def to_dict(self):
        return {
            "universe_id": self.universe_id,
            "benchmark": self.benchmark,
            "tickers": list(self.tickers),
            "source_path": self.source_path,
            "core_count": self.core_count,
            "pool_count": self.pool_count,
            "cap": self.cap,
            "survivorship_safe": False,
            "limitations": list(self.limitations),
        }


def load_sample_universe(path=DEFAULT_TICKERS_PATH, cap=DEFAULT_SAMPLE_CAP):
    """Load benchmark + bounded ticker sample from ``config/tickers.yaml``.

    The sample is the core universe followed by the hidden-gems pool, de-duplicated
    and capped at ``cap`` (None = no cap). Order is preserved from configuration so
    the sample is reproducible across runs.
    """
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise SampleUniverseError("tickers file %s did not parse to a mapping" % path)
    benchmark_block = data.get("benchmark") or {}
    if isinstance(benchmark_block, dict):
        benchmark = str(benchmark_block.get("ticker") or DEFAULT_BENCHMARK).strip().upper()
    else:
        benchmark = str(benchmark_block or DEFAULT_BENCHMARK).strip().upper()

    core = _tickers_from_block(data.get("universe"))
    pool = _tickers_from_block(data.get("hidden_gems_pool"))
    ordered = []
    seen = set()
    for ticker in core + pool:
        if ticker and ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    if cap is not None:
        ordered = ordered[: int(cap)]
    return SampleUniverse(
        universe_id=SAMPLE_UNIVERSE_ID,
        benchmark=benchmark,
        tickers=tuple(ordered),
        source_path=str(path),
        core_count=len(core),
        pool_count=len(pool),
        cap=cap,
        limitations=(KNOWN_LIMITATION_CURRENT_CONSTITUENTS,),
    )
