"""Benchmark (SPY) series: separately versioned and temporally aligned.

The benchmark is a first-class, independently versioned artefact because the
primary research target is benchmark-relative. Alignment reuses the price
semantics (nearest PRIOR session only, UTC timestamps, availability after the
close) so a benchmark value can never be pulled from the future of a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .availability import to_utc_timestamp
from .prices import align_nearest_prior, fetch_prices, validate_price_frame

DEFAULT_BENCHMARK_TICKER = "SPY"
BENCHMARK_SEMANTICS = {
    "raw_close": "session close as printed by the provider",
    "adjusted_close": "provider-adjusted total-return close (splits and dividends)",
    "available_at": "trade date plus a conservative market close time in UTC",
    "alignment": "nearest prior trading session at or before each request date",
}


class BenchmarkError(RuntimeError):
    """Raised when benchmark data is malformed or unusable."""


@dataclass(frozen=True)
class BenchmarkSeries:
    """One versioned benchmark series (default SPY)."""

    ticker: str
    frame: pd.DataFrame
    version: str
    source: str
    semantics: dict

    def to_metadata(self):
        return {
            "ticker": self.ticker,
            "version": self.version,
            "source": self.source,
            "semantics": dict(self.semantics),
            "row_count": int(len(self.frame)),
        }


def build_benchmark_series(ticker=DEFAULT_BENCHMARK_TICKER, frame=None, range_="5y", interval="1d"):
    """Build a validated and versioned benchmark series from real prices.

    ``frame`` may be injected for offline tests; it is validated the same way as
    a freshly fetched frame. Fetching failures propagate, because a benchmark
    must never be substituted with fabricated values.
    """
    from src.research.fingerprints import fingerprint_dataframe

    if frame is None:
        frame = fetch_prices(ticker, range_=range_, interval=interval)
    problems = validate_price_frame(frame)
    if problems:
        raise BenchmarkError("benchmark frame failed integrity checks: %s" % "; ".join(problems))
    version = fingerprint_dataframe(frame)
    source = "YAHOO_CHART" if frame.attrs.get("ticker") else frame["source"].iloc[0]
    return BenchmarkSeries(
        ticker=ticker,
        frame=frame,
        version=version,
        source=source,
        semantics=dict(BENCHMARK_SEMANTICS),
    )


def align_benchmark(series, dates, value_col="adjusted_close"):
    """Nearest-prior alignment of the benchmark to each request date."""
    return align_nearest_prior(series.frame, dates, ticker=series.ticker, value_col=value_col, asof_col="available_at")


def benchmark_return(series, start_date, end_date, value_col="adjusted_close"):
    """Total benchmark return between two nearest-prior aligned sessions.

    Returns ``None`` when either endpoint cannot be aligned, so overlapping
    labels are excluded rather than silently mis-measured.
    """
    aligned = align_benchmark(series, [start_date, end_date], value_col=value_col)
    if len(aligned) != 2 or aligned[value_col].isna().any():
        return None
    start_value = float(aligned[value_col].iloc[0])
    end_value = float(aligned[value_col].iloc[1])
    if start_value == 0.0:
        return None
    return (end_value / start_value) - 1.0


def benchmark_frame_asof(series, dates, value_col="adjusted_close"):
    """Tidy frame of aligned benchmark values with provenance columns."""
    aligned = align_benchmark(series, dates, value_col=value_col)
    aligned = aligned.copy()
    aligned["benchmark_ticker"] = series.ticker
    aligned["benchmark_version"] = series.version
    aligned["available_at_rule"] = BENCHMARK_SEMANTICS["available_at"]
    return aligned


def frames_overlap(first, second, key_cols=("ticker", "trade_date")):
    """True when two benchmark frames share any observation key (duplication risk)."""
    keys_a = set(map(tuple, first[list(key_cols)].to_records(index=False)))
    keys_b = set(map(tuple, second[list(key_cols)].to_records(index=False)))
    if keys_a & keys_b:
        return True
    return False
