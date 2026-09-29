"""WP4 Keyes variable engine: point-in-time X5, X6, X8, X9 and X12_PROXY.

WP3 declared the Keyes CONTRACT (``src.research.features.keyes``) but computed no
feature VALUE. This module computes the values, and it computes them under one
rule: every input must have been public at the prediction instant
(``feature_asof`` combined with the conservative market close in UTC).

Two integrity properties are deliberate:

1. **Corporate actions are applied point-in-time.** A split or dividend is only
   priced into a return when it was already effective at the prediction instant.
   Trailing returns are computed from a raw close series scaled by the actions
   in force, never from a provider-adjusted column.
2. **Nothing is estimated.** A variable without the filed evidence it needs is
   reported UNAVAILABLE with a reason. No zero fill, no interpolation, no
   substitution, and no proxy is ever labelled as an exact Keyes variable.

X12 forward expected appreciation has no historical point-in-time source; its
stand-in is named ``X12_PROXY`` and is a revenue-growth series, never "X12".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.availability import price_available_at, to_utc_timestamp
from ..data.action_validation import validated_action_factors
from ..features.keyes import proxy_variable

DEFINITION_VERSION = "v2_wp4_keyes_variables_v1"

X5 = "X5"
X6 = "X6"
X8 = "X8"
X9 = "X9"
X12_PROXY = proxy_variable("X12")

FIDELITY_EXACT = "EXACT"
FIDELITY_CLOSE_EQUIVALENT = "CLOSE_EQUIVALENT"
FIDELITY_PROXY = "PROXY"
FIDELITY_UNAVAILABLE = "UNAVAILABLE"
FIDELITY_LEVELS = (FIDELITY_EXACT, FIDELITY_CLOSE_EQUIVALENT, FIDELITY_PROXY, FIDELITY_UNAVAILABLE)

# The Keyes hypothesis directions are pre-registered here (never fitted): they
# are the economic signs Keyes argued for, and the modern composite uses them
# unchanged with equal weights.
KEYES_DIRECTIONS = {
    X5: "POSITIVE",
    X6: "POSITIVE",
    X8: "NEGATIVE",
    X9: "NEGATIVE",
    X12_PROXY: "POSITIVE",
}


class KeyesVariableError(ValueError):
    """Raised when the Keyes variable engine is given unusable inputs."""


@dataclass(frozen=True)
class VariableSpec:
    """Declared identity, source and fidelity of one WP4 Keyes variable."""

    variable: str
    base_variable: str
    definition: str
    source: str
    is_proxy: bool
    hypothesized_direction: str
    fidelity: str
    fidelity_reason: str
    availability_rule: str
    definition_version: str = DEFINITION_VERSION

    def to_dict(self):
        return dict(self.__dict__)


VARIABLE_SPECS = {
    X5: VariableSpec(
        variable=X5,
        base_variable=X5,
        definition="compound annual growth rate of trailing-twelve-month diluted EPS over five years",
        source="SEC EDGAR XBRL companyfacts (EarningsPerShareDiluted/EarningsPerShareBasic)",
        is_proxy=False,
        hypothesized_direction="POSITIVE",
        fidelity=FIDELITY_CLOSE_EQUIVALENT,
        fidelity_reason=(
            "same reported-EPS concept as the original five-year earnings growth, but the source is "
            "XBRL companyfacts rather than the Value Line series Keyes used, and it starts around "
            "2009, so early observations are UNAVAILABLE"
        ),
        availability_rule="FUNDAMENTAL: acceptance datetime, else filing_date + 1 day",
    ),
    X6: VariableSpec(
        variable=X6,
        base_variable=X6,
        definition="point-in-time total return over the trailing five years",
        source="market raw closes plus corporate actions known at the prediction instant",
        is_proxy=False,
        hypothesized_direction="POSITIVE",
        fidelity=FIDELITY_EXACT,
        fidelity_reason="price-only quantity; the committed raw-close series carries the full history",
        availability_rule="PRICE: trade date plus conservative close in UTC",
    ),
    X8: VariableSpec(
        variable=X8,
        base_variable=X8,
        definition="trailing-twelve-month diluted EPS divided into the prediction-date price",
        source="SEC EDGAR XBRL companyfacts plus market raw closes",
        is_proxy=False,
        hypothesized_direction="NEGATIVE",
        fidelity=FIDELITY_CLOSE_EQUIVALENT,
        fidelity_reason=(
            "P/E over filed XBRL EPS is the same economic ratio Keyes reported, but the earnings "
            "series is not the Value Line earnings series and negative-EPS companies have no "
            "meaningful P/E, so they are UNAVAILABLE rather than fabricated"
        ),
        availability_rule="FUNDAMENTAL for EPS, PRICE for the price leg",
    ),
    X9: VariableSpec(
        variable=X9,
        base_variable=X9,
        definition="current P/E divided by the median P/E of the security's prior 60 available monthly observations",
        source="the engine's own point-in-time P/E series (SEC EDGAR EPS plus market raw closes)",
        is_proxy=False,
        hypothesized_direction="NEGATIVE",
        fidelity=FIDELITY_CLOSE_EQUIVALENT,
        fidelity_reason=(
            "Keyes compared the current multiple with a normal multiple; V1 left that window "
            "unspecified, so WP4 fixes it a priori at 60 monthly observations with a 24-observation "
            "minimum and reports the window it used"
        ),
        availability_rule="PRICE for the price leg, FUNDAMENTAL for each month's EPS as of that month",
    ),
    X12_PROXY: VariableSpec(
        variable=X12_PROXY,
        base_variable="X12",
        definition="X12_PROXY: five-year compound annual growth rate of filed annual revenue",
        source="SEC EDGAR XBRL companyfacts (RevenueFromContractWithCustomerExcludingAssessedTax/Revenues)",
        is_proxy=True,
        hypothesized_direction="POSITIVE",
        fidelity=FIDELITY_PROXY,
        fidelity_reason=(
            "no historical point-in-time analyst or Value Line expected-appreciation series exists; "
            "trailing revenue growth is a stand-in and is NEVER reported as Keyes X12"
        ),
        availability_rule="FUNDAMENTAL: acceptance datetime, else filing_date + 1 day",
    ),
    # Modern-only companion quantities. These are NOT Keyes variables: they are
    # price-only diagnostics companions used by the modern track and its
    # baselines. They are declared here so the engine's blank-row builder can
    # label them honestly rather than crash, and they are excluded from the
    # fidelity scorecard (which only covers the five Keyes variables).
    "mom_6m": VariableSpec(
        variable="mom_6m",
        base_variable="mom_6m",
        definition="point-in-time total return over the trailing six months",
        source="market raw closes plus corporate actions known at the prediction instant",
        is_proxy=False,
        hypothesized_direction="POSITIVE",
        fidelity=FIDELITY_EXACT,
        fidelity_reason="price-only companion quantity; not a Keyes variable",
        availability_rule="PRICE: trade date plus conservative close in UTC",
    ),
    "mom_12m": VariableSpec(
        variable="mom_12m",
        base_variable="mom_12m",
        definition="point-in-time total return over the trailing twelve months",
        source="market raw closes plus corporate actions known at the prediction instant",
        is_proxy=False,
        hypothesized_direction="POSITIVE",
        fidelity=FIDELITY_EXACT,
        fidelity_reason="price-only companion quantity; not a Keyes variable",
        availability_rule="PRICE: trade date plus conservative close in UTC",
    ),
    "beta": VariableSpec(
        variable="beta",
        base_variable="beta",
        definition="trailing-window OLS slope of stock returns on the SPY benchmark",
        source="market raw closes plus corporate actions known at the prediction instant",
        is_proxy=False,
        hypothesized_direction="NONE",
        fidelity=FIDELITY_EXACT,
        fidelity_reason="price-only risk companion quantity; not a Keyes variable",
        availability_rule="PRICE: trailing window ending at the prediction date",
    ),
    "market_cap": VariableSpec(
        variable="market_cap",
        base_variable="market_cap",
        definition="point-in-time raw close times the latest filed share count",
        source="market raw closes plus SEC EDGAR filed shares outstanding",
        is_proxy=False,
        hypothesized_direction="NONE",
        fidelity=FIDELITY_CLOSE_EQUIVALENT,
        fidelity_reason="size companion quantity; not a Keyes variable",
        availability_rule="FUNDAMENTAL for shares, PRICE for the price leg",
    ),
}

# Variables that may never be handed to the variable engine as raw inputs.
FORBIDDEN_INPUT_COLUMNS = ("future_12m_excess_return", "outperform_12m", "future_12m_stock_return",
                           "future_12m_benchmark_return", "target_known_at")


@dataclass(frozen=True)
class ComputationConfig:
    """Explicit, disclosed parameters of the WP4 variable computation."""

    momentum_short_days: int = 182
    momentum_long_days: int = 365
    five_year_days: int = 1826
    beta_window_days: int = 365
    beta_min_observations: int = 120
    pe_history_window: int = 60
    pe_history_minimum: int = 24
    eps_ttm_quarter_min_days: int = 70
    eps_ttm_quarter_max_days: int = 105
    eps_ttm_max_gap_days: int = 105
    definition_version: str = DEFINITION_VERSION

    def to_dict(self):
        return dict(self.__dict__)


# ── Input guards ──────────────────────────────────────────────────────────────

def assert_no_target_leakage(frame, context):
    """Refuse a frame that carries a forward-looking label into the engine."""
    if frame is None:
        return True
    present = [name for name in FORBIDDEN_INPUT_COLUMNS if name in getattr(frame, "columns", [])]
    if present:
        raise KeyesVariableError(
            "%s was handed forward-looking column(s) %s; labels are never feature inputs"
            % (context, ", ".join(present))
        )
    return True


def _date_series(values):
    return pd.to_datetime(pd.Series(list(values)), errors="coerce")


def _ordinals(values):
    stamps = pd.to_datetime(pd.Series(list(values)), errors="coerce")
    return stamps.values.astype("datetime64[D]").astype("int64")


def _day_ordinal(day):
    """Whole-day ordinal for a date-like value, or None when it is missing.

    A genuinely price-less panel observation carries a null ``price_date``;
    that must resolve to "unavailable", never to another session's price and
    never to an exception.
    """
    stamp = pd.to_datetime(day, errors="coerce")
    if stamp is None or pd.isna(stamp):
        return None
    return int(pd.Timestamp(stamp).normalize().value // (24 * 3600 * 10 ** 9))


def _available_ordinals(values):
    ordinals = np.full(len(values), np.iinfo("int64").min, dtype="int64")
    for position, value in enumerate(values):
        stamp = to_utc_timestamp(value)
        if stamp is not None:
            ordinals[position] = stamp.normalize().value // (24 * 3600 * 10 ** 9)
    return ordinals


# ── Point-in-time price series ────────────────────────────────────────────────

class PriceHistory:
    """Raw closes scaled by the corporate actions in force at the prediction date.

    ``adjusted(d | A) = raw(d) * product(factor(a))`` over actions with
    ``d < a.effective_date <= A``, exactly as
    :func:`src.research.data.pit_prices.build_pit_adjusted_frame` defines it. The
    factor lookup is precomputed as a suffix product, which makes one pass over
    the committed history possible instead of one pass per prediction date.

    A ratio of two adjusted closes inside a window is independent of the as-of
    horizon once that horizon is past the later endpoint, because the factors
    outside the window cancel. Trailing returns therefore use the committed
    series and stay byte-identical to the per-date reconstruction.
    """

    def __init__(self, prices, actions):
        assert_no_target_leakage(prices, "price history")
        required = ("security_id", "trade_date", "raw_close")
        missing = [name for name in required if name not in prices.columns]
        if missing:
            raise KeyesVariableError("price frame is missing column(s): %s" % ", ".join(missing))
        self._raw = {}
        self._days = {}
        self._adjusted = {}
        action_frames = self._group(actions, "actions")
        for security_id, chunk in prices.groupby("security_id", sort=True):
            chunk = chunk.sort_values("trade_date", kind="mergesort")
            days = chunk["trade_date"].astype(str).str.slice(0, 10).values
            ordinals = np.array([int(pd.Timestamp(day).value // (24 * 3600 * 10 ** 9)) for day in days],
                                dtype="int64")
            raw = pd.to_numeric(chunk["raw_close"], errors="coerce").to_numpy(dtype="float64")
            self._days[security_id] = ordinals
            self._raw[security_id] = raw
            self._adjusted[security_id] = self._apply_actions(ordinals, raw, action_frames.get(security_id, []))

    @staticmethod
    def _group(frame, what):
        if frame is None or len(frame) == 0:
            return {}
        if "security_id" not in frame.columns:
            raise KeyesVariableError("%s frame is missing security_id" % what)
        return {key: chunk for key, chunk in frame.groupby("security_id", sort=True)}

    def _apply_actions(self, ordinals, raw, action_frame):
        """Back-adjust raw closes by VALIDATED corporate-action factors.

        Every action is routed through :func:`action_validation.validated_action_factors`
        so identical rows are de-duplicated, a mis-typed/unresolvable dividend is
        skipped, an evidence-backed ``/100`` unit correction is applied, and an
        economically impossible dividend factor is refused. No invalid action is
        silently compounded into the trailing-return series.
        """
        if len(action_frame) == 0:
            return raw.astype("float64").copy()

        def close_lookup(day_text):
            effective = int(pd.Timestamp(day_text).value // (24 * 3600 * 10 ** 9))
            position = int(np.searchsorted(ordinals, effective, side="right")) - 1
            if position < 0:
                return None
            value = raw[position]
            return None if not np.isfinite(value) else float(value)

        prepared, _dropped = validated_action_factors(action_frame.to_dict("records"), close_lookup)
        if not prepared:
            return raw.astype("float64").copy()
        dated = [(int(pd.Timestamp(str(day)).value // (24 * 3600 * 10 ** 9)), float(factor))
                 for day, factor in prepared]
        dated.sort(key=lambda item: item[0])
        effective = np.array([item[0] for item in dated], dtype="int64")
        all_factors = np.array([item[1] for item in dated], dtype="float64")
        # suffix[k] = product of factors[k:] applied to dates strictly before effective[k]
        suffix = np.ones(len(effective) + 1, dtype="float64")
        for position in range(len(effective) - 1, -1, -1):
            suffix[position] = suffix[position + 1] * all_factors[position]
        counts = np.searchsorted(effective, ordinals, side="right")
        return raw * suffix[counts]

    def has(self, security_id):
        return security_id in self._adjusted

    def adjusted_on(self, security_id, day, direction="prior"):
        """Adjusted close on ``day``, else the nearest prior (or later) session.

        A missing endpoint returns None: an unavailable price never silently
        becomes another session's price. A missing or unparseable ``day`` (a
        genuinely price-less panel observation) also returns None.
        """
        if security_id not in self._adjusted:
            return None
        target = _day_ordinal(day)
        if target is None:
            return None
        ordinals = self._days[security_id]
        if direction == "prior":
            position = int(np.searchsorted(ordinals, target, side="right")) - 1
        else:
            position = int(np.searchsorted(ordinals, target, side="left"))
        if position < 0 or position >= len(ordinals):
            return None
        value = self._adjusted[security_id][position]
        return float(value) if np.isfinite(value) else None

    def raw_on(self, security_id, day, direction="prior"):
        if security_id not in self._raw:
            return None
        target = _day_ordinal(day)
        if target is None:
            return None
        ordinals = self._days[security_id]
        if direction == "prior":
            position = int(np.searchsorted(ordinals, target, side="right")) - 1
        else:
            position = int(np.searchsorted(ordinals, target, side="left"))
        if position < 0 or position >= len(ordinals):
            return None
        value = self._raw[security_id][position]
        return float(value) if np.isfinite(value) else None

    def trailing_return(self, security_id, end_day, lookback_days):
        """Point-in-time total return over the trailing window ending ``end_day``."""
        end_stamp = pd.to_datetime(end_day, errors="coerce")
        if end_stamp is None or pd.isna(end_stamp):
            return None
        start_target = pd.Timestamp(end_stamp) - pd.Timedelta(days=int(lookback_days))
        start = self.adjusted_on(security_id, start_target.strftime("%Y-%m-%d"), "prior")
        end = self.adjusted_on(security_id, end_day, "prior")
        if start in (None, 0.0) or end is None:
            return None
        return (end / start) - 1.0

    def daily_returns(self, security_id, end_day, window_days):
        """Chronological daily PIT returns inside the trailing window ending at ``end_day``."""
        if security_id not in self._adjusted:
            return None, None
        ordinals = self._days[security_id]
        adjusted = self._adjusted[security_id]
        end_target = _day_ordinal(end_day)
        if end_target is None:
            return None, None
        start_target = end_target - int(window_days)
        low = int(np.searchsorted(ordinals, start_target, side="left"))
        high = int(np.searchsorted(ordinals, end_target, side="right"))
        if high - low < 3:
            return None, None
        segment_days = ordinals[low:high]
        segment = adjusted[low:high]
        valid = np.isfinite(segment) & (segment > 0)
        segment_days = segment_days[valid]
        segment = segment[valid]
        if len(segment) < 3:
            return None, None
        returns = segment[1:] / segment[:-1] - 1.0
        return segment_days[1:], returns


def beta_against_benchmark(price_history, benchmark_history, security_id, asof_day, config, benchmark_id="SPY"):
    """Trailing-window OLS slope of stock returns on benchmark returns.

    The window ends at ``asof_day`` and contains only sessions up to that date,
    so the estimate never uses a future session. Returns ``(beta, observations)``
    or ``(None, observations)``.
    """
    stock_days, stock_returns = price_history.daily_returns(security_id, asof_day, config.beta_window_days)
    bench_days, bench_returns = benchmark_history.daily_returns(benchmark_id, asof_day, config.beta_window_days)
    if stock_days is None or bench_days is None:
        return None, 0
    lookup = {int(day): value for day, value in zip(bench_days, bench_returns)}
    paired = [(lookup[int(day)], value) for day, value in zip(stock_days, stock_returns) if int(day) in lookup]
    if len(paired) < config.beta_min_observations:
        return None, len(paired)
    market = np.array([pair[0] for pair in paired], dtype="float64")
    stock = np.array([pair[1] for pair in paired], dtype="float64")
    variance = float(np.var(market))
    if variance == 0.0:
        return None, len(paired)
    return float(np.cov(stock, market, ddof=0)[0, 1] / variance), len(paired)


# ── Point-in-time fundamentals ────────────────────────────────────────────────

@dataclass
class FundamentalHistory:
    """Filed fundamental facts indexed by CIK, resolved point-in-time per date.

    Each (fiscal_period_end, accession) pair stays a separate version, so a
    restatement filed later can never be selected for an earlier date.
    """

    records: dict = field(default_factory=dict)

    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    OTHER = "OTHER"

    @classmethod
    def from_frame(cls, frame):
        required = ("cik", "field", "value", "fiscal_period_end", "available_at")
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise KeyesVariableError("fundamental frame is missing column(s): %s" % ", ".join(missing))
        history = cls()
        working = frame.copy()
        working["__available"] = _available_ordinals(working["available_at"])
        working["__end"] = _ordinals(working["fiscal_period_end"])
        starts = _ordinals(working["fiscal_period_start"]) if "fiscal_period_start" in working.columns else \
            np.full(len(working), np.iinfo("int64").min, dtype="int64")
        working["__start"] = starts
        working["__value"] = pd.to_numeric(working["value"], errors="coerce")
        working = working.loc[working["__value"].notna() & (working["__available"] > np.iinfo("int64").min)]
        for (cik, field_name), chunk in working.groupby(["cik", "field"], sort=True):
            history.records[(str(cik), str(field_name))] = {
                "available": chunk["__available"].to_numpy(dtype="int64"),
                "end": chunk["__end"].to_numpy(dtype="int64"),
                "start": chunk["__start"].to_numpy(dtype="int64"),
                "value": chunk["__value"].to_numpy(dtype="float64"),
            }
        return history

    @staticmethod
    def classify(start, end):
        if start > 0 and end > start:
            length = end - start
            if 70 <= length <= 105:
                return FundamentalHistory.QUARTERLY
            if 340 <= length <= 380:
                return FundamentalHistory.ANNUAL
            return FundamentalHistory.OTHER
        return FundamentalHistory.OTHER

    def _eligible(self, cik, field_name, asof_ordinal):
        record = self.records.get((str(cik), str(field_name)))
        if record is None:
            return None
        mask = record["available"] <= asof_ordinal
        if not mask.any():
            return None
        return {name: values[mask] for name, values in record.items()}

    @staticmethod
    def _latest_period_version(ends, order):
        """Map each fiscal period end to the index of its latest-public version.

        ``order`` sorts by ``(fiscal_period_end, available_at)`` ascending, so the
        last occurrence of a period end inside that order is the restatement that
        was public latest at the as-of date. The reverse-unique trick resolves it
        without a Python loop over every fact, which matters because this runs
        once per observation and per field.
        """
        ordered_ends = ends[order]
        _unique_ends, first_in_reverse = np.unique(ordered_ends[::-1], return_index=True)
        positions = order[::-1][first_in_reverse]
        return {int(end): int(position) for end, position in zip(_unique_ends, positions)}

    def latest_value(self, cik, field_name, asof, period=None):
        """Latest filed value whose period already ended before ``asof``.

        Returns ``(value, period_end, available_at_ordinal, basis)`` or None.
        """
        asof_ordinal = int(pd.Timestamp(str(asof)).value // (24 * 3600 * 10 ** 9))
        eligible = self._eligible(cik, field_name, asof_ordinal)
        if eligible is None:
            return None
        order = np.lexsort((eligible["available"], eligible["end"]))
        chosen = self._latest_period_version(eligible["end"], order)
        candidates = []
        for end, position in chosen.items():
            if period is not None and self.classify(int(eligible["start"][position]), end) != period:
                continue
            if end > asof_ordinal:
                continue
            candidates.append(position)
        if not candidates:
            return None
        best = max(candidates, key=lambda position: (int(eligible["end"][position]), int(eligible["available"][position])))
        return (
            float(eligible["value"][best]),
            int(eligible["end"][best]),
            int(eligible["available"][best]),
            self.classify(int(eligible["start"][best]), int(eligible["end"][best])),
        )

    def trailing_twelve_month_value(self, cik, field_name, asof):
        """Sum of four consecutive filed quarters, else the latest filed year.

        Returns ``(value, basis, period_end, available_at_ordinal)`` where
        ``basis`` is explicitly ``"TTM_QUARTERLY"`` or ``"LATEST_ANNUAL"``;
        the two are never confused, and an incomplete quarter set never becomes
        a fabricated four-quarter total.
        """
        asof_ordinal = int(pd.Timestamp(str(asof)).value // (24 * 3600 * 10 ** 9))
        eligible = self._eligible(cik, field_name, asof_ordinal)
        if eligible is not None:
            order = np.lexsort((eligible["available"], eligible["end"]))
            chosen = self._latest_period_version(eligible["end"], order)
            quarters = []
            for end, position in chosen.items():
                if self.classify(int(eligible["start"][position]), end) != self.QUARTERLY:
                    continue
                if end > asof_ordinal:
                    continue
                quarters.append((end, position))
            quarters.sort()
            if len(quarters) >= 4:
                window = quarters[-4:]
                gaps = [window[index + 1][0] - window[index][0] for index in range(3)]
                if all(60 <= gap <= 110 for gap in gaps):
                    total = float(sum(float(eligible["value"][position]) for _end, position in window))
                    available = int(max(int(eligible["available"][position]) for _end, position in window))
                    return total, "TTM_QUARTERLY", int(window[-1][0]), available
        annual = self.latest_value(cik, field_name, asof, period=self.ANNUAL)
        if annual is not None:
            value, end, available, _basis = annual
            return value, "LATEST_ANNUAL", end, available
        return None


# ── Variable computation ──────────────────────────────────────────────────────

def _blank(variable):
    spec = VARIABLE_SPECS[variable]
    return {
        "variable": variable,
        "value": None,
        "available": False,
        "basis": "UNAVAILABLE",
        "reason": "",
        "is_proxy": bool(spec.is_proxy),
        "fidelity": spec.fidelity,
        "source_observed_at": None,
        "definition_version": DEFINITION_VERSION,
    }


def compute_keyes_variables(panel, prices, actions, fundamentals, cik_by_ticker, benchmark_prices,
                            benchmark_actions, config=None):
    """Compute the five WP4 Keyes variables for every panel observation.

    ``panel`` carries ``security_id``, ``ticker``, ``feature_asof`` and the
    prediction-date price. ``fundamentals`` is the committed SEC EDGAR silver
    frame. No label column may appear on any input.
    """
    config = config or ComputationConfig()
    for frame, what in ((panel, "panel"), (prices, "prices"), (actions, "actions"),
                        (fundamentals, "fundamentals"), (benchmark_prices, "benchmark prices")):
        assert_no_target_leakage(frame, what)
    required = ("security_id", "ticker", "feature_asof", "price_date")
    missing = [name for name in required if name not in panel.columns]
    if missing:
        raise KeyesVariableError("panel is missing column(s): %s" % ", ".join(missing))

    history = PriceHistory(prices, actions)
    benchmark = PriceHistory(benchmark_prices, benchmark_actions)
    fundamentals_by_cik = FundamentalHistory.from_frame(fundamentals)

    observations = panel.loc[:, ["security_id", "ticker", "feature_asof", "price_date"]].copy()
    duplicates = int(observations.duplicated(subset=["security_id", "feature_asof"]).sum())
    observations = observations.drop_duplicates(subset=["security_id", "feature_asof"], keep="first")
    observations = observations.sort_values(["security_id", "feature_asof"], kind="mergesort").reset_index(drop=True)

    rows = []
    pe_history = {}
    for record in observations.itertuples():
        asof_day = str(record.feature_asof)[:10]
        price_day = str(record.price_date)[:10]
        prediction = price_available_at(asof_day)
        prediction_ordinal = int(prediction.value // (24 * 3600 * 10 ** 9)) if prediction is not None else None
        ticker = str(record.ticker)
        cik = (cik_by_ticker or {}).get(ticker.upper())

        # X6 five-year price gain (price-only, exact).
        entry = _blank(X6)
        value = history.trailing_return(record.security_id, price_day, config.five_year_days)
        if value is not None:
            entry.update({"value": value, "available": True, "basis": "PIT_TOTAL_RETURN_5Y",
                          "source_observed_at": price_day})
        else:
            entry["reason"] = "no point-in-time adjusted close at one or both endpoints of the five-year window"
        rows.append(_row(record, asof_day, entry))

        # EPS legs: trailing twelve month EPS, point-in-time.
        eps_now = fundamentals_by_cik.trailing_twelve_month_value(cik, "eps", asof_day) if cik else None
        eps_then = fundamentals_by_cik.trailing_twelve_month_value(
            cik, "eps", (pd.Timestamp(asof_day) - pd.Timedelta(days=config.five_year_days)).strftime("%Y-%m-%d")
        ) if cik else None

        # X5 five-year EPS growth.
        entry = _blank(X5)
        if eps_now is None or eps_then is None:
            entry["reason"] = "no filed EPS available at one or both ends of the five-year window"
        elif eps_now[0] <= 0 or eps_then[0] <= 0:
            entry["reason"] = "a five-year compound growth rate is undefined unless both EPS values are positive"
        else:
            growth = (eps_now[0] / eps_then[0]) ** (1.0 / 5.0) - 1.0
            entry.update({"value": growth, "available": True,
                          "basis": "EPS_%s_TO_%s" % (eps_then[1], eps_now[1]),
                          "source_observed_at": _ordinal_to_day(max(eps_now[3], eps_then[3]))})
        rows.append(_row(record, asof_day, entry))

        # X8 current P/E.
        entry = _blank(X8)
        price = history.raw_on(record.security_id, price_day, "prior")
        if cik is None:
            entry["reason"] = "ticker has no unambiguous SEC CIK mapping, so filed EPS is not attributable"
        elif eps_now is None:
            entry["reason"] = "no filed EPS available at the prediction date"
        elif eps_now[0] <= 0:
            entry["reason"] = "a P/E is undefined for non-positive trailing earnings"
        elif price in (None, 0.0):
            entry["reason"] = "no point-in-time price at the prediction date"
        else:
            entry.update({"value": price / eps_now[0], "available": True,
                          "basis": "PRICE_OVER_EPS_%s" % eps_now[1],
                          "source_observed_at": _ordinal_to_day(max(eps_now[3],
                                                                     int(pd.Timestamp(price_day).value // (24 * 3600 * 10 ** 9))))})
        rows.append(_row(record, asof_day, entry))
        if entry["available"] and entry["value"] > 0:
            pe_history.setdefault(record.security_id, []).append((asof_day, float(entry["value"])))

        # X9 current P/E relative to the trailing median P/E.
        entry = _blank(X9)
        series = pe_history.get(record.security_id, [])
        prior = [value for day, value in series[:-1]][-config.pe_history_window:]
        current = series[-1][1] if series and series[-1][0] == asof_day else None
        if current is None:
            entry["reason"] = "current P/E is unavailable, so no ratio can be formed"
        elif len(prior) < config.pe_history_minimum:
            entry["reason"] = ("only %d prior monthly P/E observations are available; %d are required "
                              "before a historical normal multiple is defined" % (len(prior), config.pe_history_minimum))
        else:
            median = float(np.median(prior))
            if median <= 0:
                entry["reason"] = "historical median P/E is non-positive"
            else:
                entry.update({"value": current / median, "available": True,
                              "basis": "CURRENT_OVER_MEDIAN_%d_MONTHS" % len(prior),
                              "source_observed_at": price_day})
        rows.append(_row(record, asof_day, entry))

        # X12_PROXY five-year revenue growth.
        entry = _blank(X12_PROXY)
        revenue_now = fundamentals_by_cik.trailing_twelve_month_value(cik, "revenue", asof_day) if cik else None
        revenue_then = fundamentals_by_cik.trailing_twelve_month_value(
            cik, "revenue", (pd.Timestamp(asof_day) - pd.Timedelta(days=config.five_year_days)).strftime("%Y-%m-%d")
        ) if cik else None
        if revenue_now is None or revenue_then is None:
            entry["reason"] = "no filed revenue available at one or both ends of the five-year window"
        elif revenue_now[0] <= 0 or revenue_then[0] <= 0:
            entry["reason"] = "a five-year compound growth rate is undefined unless both revenue values are positive"
        else:
            entry.update({"value": (revenue_now[0] / revenue_then[0]) ** (1.0 / 5.0) - 1.0, "available": True,
                          "basis": "REVENUE_%s_TO_%s" % (revenue_then[1], revenue_now[1]),
                          "source_observed_at": _ordinal_to_day(max(revenue_now[3], revenue_then[3]))})
        rows.append(_row(record, asof_day, entry))

        # Price-only modern companions (diagnostics, not part of the Keyes composite).
        for variable, lookback, basis in (("mom_6m", config.momentum_short_days, "PIT_TOTAL_RETURN_6M"),
                                          ("mom_12m", config.momentum_long_days, "PIT_TOTAL_RETURN_12M")):
            entry = _blank(variable)
            value = history.trailing_return(record.security_id, price_day, lookback)
            if value is not None:
                entry.update({"value": value, "available": True, "basis": basis, "source_observed_at": price_day})
            else:
                entry["reason"] = "no point-in-time adjusted close at one or both endpoints of the window"
            entry["is_proxy"] = False
            entry["fidelity"] = FIDELITY_EXACT
            rows.append(_row(record, asof_day, entry))

        entry = _blank("beta")
        value, observations_used = beta_against_benchmark(history, benchmark, record.security_id, price_day, config)
        if value is not None:
            entry.update({"value": value, "available": True, "source_observed_at": price_day,
                          "basis": "OLS_%d_OBSERVATIONS" % observations_used})
        else:
            entry["reason"] = ("only %d paired benchmark observations in the trailing window; %d are required"
                               % (observations_used, config.beta_min_observations))
        entry["is_proxy"] = False
        entry["fidelity"] = FIDELITY_EXACT
        rows.append(_row(record, asof_day, entry))

        entry = _blank("market_cap")
        shares = fundamentals_by_cik.latest_value(cik, "shares_outstanding", asof_day) if cik else None
        if shares is None:
            entry["reason"] = "no filed share count available at the prediction date"
        elif shares[0] <= 0:
            entry["reason"] = "filed share count is non-positive"
        elif price in (None, 0.0):
            entry["reason"] = "no point-in-time price at the prediction date"
        else:
            entry.update({"value": price * shares[0], "available": True,
                          "basis": "PRICE_TIMES_SHARES", "source_observed_at": _ordinal_to_day(shares[2])})
        entry["is_proxy"] = False
        entry["fidelity"] = FIDELITY_CLOSE_EQUIVALENT
        rows.append(_row(record, asof_day, entry))

        if prediction_ordinal is None:
            continue

    frame = pd.DataFrame(rows)
    coverage = {
        "panel_rows": int(len(observations)),
        "duplicate_panel_rows_removed": duplicates,
        "observations": int(len(observations)),
        "variables": {},
    }
    for variable, chunk in frame.groupby("variable"):
        available = int(chunk["available"].sum())
        coverage["variables"][variable] = {
            "rows": int(len(chunk)),
            "available": available,
            "missing": int(len(chunk) - available),
            "coverage": (available / len(chunk)) if len(chunk) else 0.0,
        }
    coverage["config"] = config.to_dict()
    return frame, coverage


def _row(record, asof_day, entry):
    payload = {
        "security_id": record.security_id,
        "ticker": record.ticker,
        "feature_asof": asof_day,
    }
    payload.update(entry)
    return payload


def _ordinal_to_day(ordinal):
    if ordinal is None or ordinal <= 0:
        return None
    return (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(ordinal))).strftime("%Y-%m-%d")


def wide_variables(variable_frame, variables=None):
    """Pivot the long variable frame into one column per variable.

    Duplicate (security_id, feature_asof, variable) rows are refused rather than
    silently aggregated, because an aggregate would hide a duplicated panel row.
    """
    frame = variable_frame
    if variables is not None:
        frame = frame.loc[frame["variable"].isin(list(variables))]
    duplicated = int(frame.duplicated(subset=["security_id", "feature_asof", "variable"]).sum())
    if duplicated:
        raise KeyesVariableError("variable frame has %d duplicated key rows" % duplicated)
    # ``unstack`` returns exactly the observed keys: it neither drops all-NaN
    # observations (pivot_table's default) nor fabricates a Cartesian product
    # (pivot_table(dropna=False) on pandas 3.0.6).
    keys = ["security_id", "ticker", "feature_asof"]
    wide = frame.set_index(keys + ["variable"])["value"].unstack("variable").reset_index()
    wide.columns.name = None
    basis = frame.set_index(keys + ["variable"])["basis"].unstack("variable").reset_index()
    for column in basis.columns:
        if column in ("security_id", "ticker", "feature_asof"):
            continue
        wide["%s_basis" % column] = basis[column]
    return wide


class VariableEngine:
    """Thin object wrapper so a caller cannot forget the input guard."""

    def __init__(self, panel, prices, actions, fundamentals, cik_by_ticker, benchmark_prices,
                 benchmark_actions, config=None):
        self.panel = panel
        self.prices = prices
        self.actions = actions
        self.fundamentals = fundamentals
        self.cik_by_ticker = cik_by_ticker or {}
        self.benchmark_prices = benchmark_prices
        self.benchmark_actions = benchmark_actions
        self.config = config or ComputationConfig()

    def build(self):
        return compute_keyes_variables(
            self.panel, self.prices, self.actions, self.fundamentals, self.cik_by_ticker,
            self.benchmark_prices, self.benchmark_actions, self.config,
        )
