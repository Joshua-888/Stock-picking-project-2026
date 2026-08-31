"""
src/ingestion/sample_data.py

Generates realistic synthetic stock market data for pipeline testing.

Improvements over first version
---------------------------------
* Regime shifts: 2011 debt-ceiling, 2015 China slowdown, 2018 Dec selloff,
  2020 COVID crash+recovery, 2022 rate-hike bear market, 2023 bank stress.
* Sector drifts are now centred on the market drift (~10 % annually) so
  the winner rate is ~50 % rather than artificially high.
* Fundamentals use fully consistent units throughout:
    eps            – dollars per share ($/share)
    revenue        – billions of USD per quarter
    net_income     – billions of USD per quarter
    equity         – billions of USD (cumulative)
    total_debt     – billions of USD
    free_cash_flow – billions of USD per quarter
* shares_outstanding_m is generated per stock and stored in stocks metadata,
  enabling a clean market_cap = price × shares / 1000 calculation in features.

Usage
-----
    from src.ingestion.sample_data import load_sample_data
    data = load_sample_data()
    prices       = data["prices_clean"]
    benchmark    = data["benchmark_prices"]
    fundamentals = data["fundamentals_clean"]
    stocks       = data["stocks"]
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GLOBAL_SEED       = 42
SAMPLE_START_DATE = "2009-10-31"
BENCHMARK_TICKER  = "SPY"

MARKET_ANNUAL_DRIFT = 0.10   # 10 % long-run equity premium
MARKET_ANNUAL_VOL   = 0.18

# Real-world-inspired one-time market shocks (year, month): additional return
MARKET_SHOCKS: dict[tuple[int, int], float] = {
    (2011, 8):  -0.07,   # US debt-ceiling crisis
    (2015, 8):  -0.06,   # China slowdown fears
    (2018, 12): -0.09,   # Fed tightening panic
    (2020, 2):  -0.08,   # COVID onset
    (2020, 3):  -0.13,   # COVID crash trough
    (2020, 4):  +0.13,   # V-shaped recovery begins
    (2020, 5):  +0.05,   # Recovery continues
    (2022, 1):  -0.05,   # Rate-hike fears
    (2022, 4):  -0.08,   # Aggressive Fed hikes
    (2022, 6):  -0.08,   # Bear market deepens
    (2022, 9):  -0.09,   # Near trough
    (2022, 10): +0.08,   # Bear-market bounce
    (2023, 3):  -0.04,   # Regional bank stress
}


# ── Sector profiles ───────────────────────────────────────────────────────────
# annual_drift is now centred on MARKET_ANNUAL_DRIFT so ~50 % of stocks
# beat the benchmark.  Each tuple is (min, max).

SECTOR_PROFILES: dict[str, dict] = {
    "Technology": {
        "annual_drift":       (0.04, 0.20),   # wide spread: some underperform, some soar
        "annual_vol":         (0.25, 0.42),
        "market_corr":        (0.55, 0.75),
        "initial_pe":         (22.0, 45.0),
        "eps_growth":         (0.06, 0.22),
        "eps_vol":            (0.20, 0.40),
        "revenue_growth":     (0.05, 0.20),
        "gross_margin":       (0.50, 0.80),
        "operating_margin":   (0.18, 0.38),
        "net_margin":         (0.12, 0.28),
        "dividend_yield":     (0.00, 0.012),
        "debt_to_equity":     (0.10, 0.80),
        "roe":                (0.18, 0.45),
        "fcf_conversion":     (0.80, 1.10),   # FCF / net_income
        "market_cap_B_range": (80,  3000),    # initial market cap in $B
    },
    "Communication Services": {
        "annual_drift":       (0.03, 0.18),
        "annual_vol":         (0.22, 0.38),
        "market_corr":        (0.50, 0.70),
        "initial_pe":         (18.0, 38.0),
        "eps_growth":         (0.04, 0.18),
        "eps_vol":            (0.20, 0.40),
        "revenue_growth":     (0.04, 0.18),
        "gross_margin":       (0.40, 0.70),
        "operating_margin":   (0.14, 0.32),
        "net_margin":         (0.10, 0.22),
        "dividend_yield":     (0.00, 0.015),
        "debt_to_equity":     (0.20, 1.00),
        "roe":                (0.14, 0.32),
        "fcf_conversion":     (0.75, 1.05),
        "market_cap_B_range": (60,  1500),
    },
    "Consumer Discretionary": {
        "annual_drift":       (0.02, 0.18),
        "annual_vol":         (0.22, 0.40),
        "market_corr":        (0.50, 0.70),
        "initial_pe":         (18.0, 40.0),
        "eps_growth":         (0.03, 0.16),
        "eps_vol":            (0.18, 0.38),
        "revenue_growth":     (0.03, 0.16),
        "gross_margin":       (0.28, 0.55),
        "operating_margin":   (0.07, 0.22),
        "net_margin":         (0.05, 0.15),
        "dividend_yield":     (0.00, 0.020),
        "debt_to_equity":     (0.30, 1.50),
        "roe":                (0.14, 0.38),
        "fcf_conversion":     (0.70, 1.00),
        "market_cap_B_range": (40,  1500),
    },
    "Consumer Staples": {
        "annual_drift":       (0.05, 0.13),   # defensive — tighter range
        "annual_vol":         (0.12, 0.20),
        "market_corr":        (0.35, 0.55),
        "initial_pe":         (18.0, 28.0),
        "eps_growth":         (0.03, 0.09),
        "eps_vol":            (0.08, 0.18),
        "revenue_growth":     (0.02, 0.07),
        "gross_margin":       (0.28, 0.52),
        "operating_margin":   (0.08, 0.20),
        "net_margin":         (0.06, 0.14),
        "dividend_yield":     (0.020, 0.045),
        "debt_to_equity":     (0.50, 1.80),
        "roe":                (0.14, 0.28),
        "fcf_conversion":     (0.85, 1.10),
        "market_cap_B_range": (20,   500),
    },
    "Healthcare": {
        "annual_drift":       (0.04, 0.16),
        "annual_vol":         (0.18, 0.30),
        "market_corr":        (0.40, 0.60),
        "initial_pe":         (16.0, 30.0),
        "eps_growth":         (0.04, 0.14),
        "eps_vol":            (0.15, 0.32),
        "revenue_growth":     (0.04, 0.13),
        "gross_margin":       (0.55, 0.78),
        "operating_margin":   (0.14, 0.30),
        "net_margin":         (0.10, 0.22),
        "dividend_yield":     (0.010, 0.035),
        "debt_to_equity":     (0.30, 1.20),
        "roe":                (0.14, 0.32),
        "fcf_conversion":     (0.80, 1.05),
        "market_cap_B_range": (40,   500),
    },
    "Financials": {
        "annual_drift":       (0.04, 0.15),
        "annual_vol":         (0.20, 0.32),
        "market_corr":        (0.55, 0.72),
        "initial_pe":         (10.0, 20.0),
        "eps_growth":         (0.03, 0.13),
        "eps_vol":            (0.18, 0.36),
        "revenue_growth":     (0.03, 0.11),
        "gross_margin":       (0.55, 0.80),
        "operating_margin":   (0.25, 0.42),
        "net_margin":         (0.18, 0.30),
        "dividend_yield":     (0.015, 0.040),
        "debt_to_equity":     (1.50, 4.00),
        "roe":                (0.10, 0.22),
        "fcf_conversion":     (0.70, 0.95),
        "market_cap_B_range": (50,   500),
    },
    "Energy": {
        "annual_drift":       (-0.01, 0.14),  # can trail the market
        "annual_vol":         (0.25, 0.40),
        "market_corr":        (0.45, 0.65),
        "initial_pe":         (12.0, 25.0),
        "eps_growth":         (0.00, 0.10),
        "eps_vol":            (0.30, 0.55),   # commodity-driven
        "revenue_growth":     (0.00, 0.08),
        "gross_margin":       (0.18, 0.38),
        "operating_margin":   (0.07, 0.20),
        "net_margin":         (0.05, 0.15),
        "dividend_yield":     (0.025, 0.055),
        "debt_to_equity":     (0.40, 1.20),
        "roe":                (0.07, 0.18),
        "fcf_conversion":     (0.65, 0.90),
        "market_cap_B_range": (30,   300),
    },
}


# ── Stock profile ─────────────────────────────────────────────────────────────

@dataclass
class StockProfile:
    ticker:               str
    sector:               str
    annual_drift:         float
    annual_vol:           float
    market_corr:          float
    initial_pe:           float
    shares_outstanding_m: float   # millions of shares outstanding
    initial_market_cap_B: float   # initial market cap in $B
    initial_price:        float   # = market_cap_B * 1000 / shares_m
    initial_eps_annual:   float   # = initial_price / initial_pe
    initial_revenue_B:    float   # annual revenue in $B
    eps_growth:           float
    eps_vol:              float
    revenue_growth:       float
    gross_margin:         float
    operating_margin:     float
    net_margin:           float
    dividend_yield:       float
    debt_to_equity:       float
    roe:                  float
    fcf_conversion:       float


# ── Deterministic per-ticker RNG ──────────────────────────────────────────────

def _ticker_rng(ticker: str) -> np.random.Generator:
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2 ** 32)
    return np.random.default_rng(seed)


def _build_profile(ticker: str, sector: str) -> StockProfile:
    rng = _ticker_rng(ticker)
    p   = SECTOR_PROFILES.get(sector, SECTOR_PROFILES["Technology"])

    def draw(key: str) -> float:
        lo, hi = p[key]
        return float(rng.uniform(lo, hi))

    # ── Derive a self-consistent set of financials ────────────────────────────
    # 1. Draw market cap and a realistic per-share price.
    # 2. Derive shares from cap / price (not independently) — this prevents
    #    unrealistically large revenue when shares × EPS is blown up.
    mkt_cap_lo, mkt_cap_hi = p["market_cap_B_range"]
    initial_market_cap_B   = float(rng.uniform(mkt_cap_lo, mkt_cap_hi))

    # Price range anchored to sector (cheap stocks vs expensive growth stocks)
    price_lo = 15.0
    price_hi = min(500.0, initial_market_cap_B * 10)   # no $500 price for a $5B company
    initial_price_ref  = float(rng.uniform(price_lo, price_hi))
    shares_outstanding_m = initial_market_cap_B * 1_000 / initial_price_ref
    # Clamp to a plausible range: 100M – 20B shares
    shares_outstanding_m = min(max(shares_outstanding_m, 100.0), 20_000.0)
    initial_price        = round(initial_market_cap_B * 1_000 / shares_outstanding_m, 2)

    initial_pe           = draw("initial_pe")
    # Cap initial EPS at $30/share — prevents extreme compounding over 15+ years
    initial_eps_annual   = min(initial_price / initial_pe, 30.0)
    gross_margin         = draw("gross_margin")
    operating_margin     = draw("operating_margin")
    net_margin           = draw("net_margin")

    # Annual revenue in $B — cap to prevent runaway values
    net_income_annual_B  = initial_eps_annual * shares_outstanding_m / 1_000
    raw_revenue_B        = net_income_annual_B / max(net_margin, 0.03)
    # Hard cap: no company starts with >$150B quarterly revenue ($600B annual)
    initial_revenue_B    = max(min(raw_revenue_B, 150.0), 0.1)

    return StockProfile(
        ticker               = ticker,
        sector               = sector,
        annual_drift         = draw("annual_drift"),
        annual_vol           = draw("annual_vol"),
        market_corr          = draw("market_corr"),
        initial_pe           = initial_pe,
        shares_outstanding_m = shares_outstanding_m,
        initial_market_cap_B = initial_market_cap_B,
        initial_price        = initial_price,
        initial_eps_annual   = initial_eps_annual,
        initial_revenue_B    = initial_revenue_B,
        eps_growth           = draw("eps_growth"),
        eps_vol              = draw("eps_vol"),
        revenue_growth       = draw("revenue_growth"),
        gross_margin         = gross_margin,
        operating_margin     = operating_margin,
        net_margin           = net_margin,
        dividend_yield       = draw("dividend_yield"),
        debt_to_equity       = draw("debt_to_equity"),
        roe                  = draw("roe"),
        fcf_conversion       = draw("fcf_conversion"),
    )


# ── Date helpers ──────────────────────────────────────────────────────────────

def _month_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="ME")


def _quarter_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="QE")


# ── Market factor with regime shifts ─────────────────────────────────────────

def _generate_market_returns(dates: pd.DatetimeIndex) -> pd.Series:
    """
    GBM market returns with discrete regime-shift shocks layered on top.
    Shocks are additive so their magnitude is intuitive.
    """
    rng           = np.random.default_rng(GLOBAL_SEED)
    n             = len(dates)
    monthly_drift = MARKET_ANNUAL_DRIFT / 12
    monthly_vol   = MARKET_ANNUAL_VOL / math.sqrt(12)

    returns = rng.normal(monthly_drift, monthly_vol, n)

    # Apply one-time shocks
    for i, ts in enumerate(dates):
        key = (ts.year, ts.month)
        if key in MARKET_SHOCKS:
            returns[i] += MARKET_SHOCKS[key]

    return pd.Series(returns, index=dates, name="market_return")


# ── Benchmark prices ──────────────────────────────────────────────────────────

def generate_benchmark_prices(
    start_date: str = SAMPLE_START_DATE,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    dates          = _month_ends(start_date, end_date)
    market_returns = _generate_market_returns(dates)

    initial_price = 100.0
    prices = [initial_price]
    for r in market_returns.values[1:]:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({
        "benchmark_ticker": BENCHMARK_TICKER,
        "date":             dates.strftime("%Y-%m-%d"),
        "adjusted_close":   prices,
        "monthly_return":   market_returns.values,
    })
    df.loc[df.index[0], "monthly_return"] = float("nan")
    return df


# ── Stock prices ──────────────────────────────────────────────────────────────

def _generate_single_stock_prices(
    profile:        StockProfile,
    dates:          pd.DatetimeIndex,
    market_returns: pd.Series,
) -> list[float]:
    rng          = _ticker_rng(profile.ticker + "_prices")
    n            = len(dates)
    monthly_drift = profile.annual_drift / 12
    monthly_vol   = profile.annual_vol / math.sqrt(12)
    rho           = profile.market_corr
    mkt_monthly_vol = MARKET_ANNUAL_VOL / math.sqrt(12)
    idio_vol      = math.sqrt(max(monthly_vol ** 2 - (rho * mkt_monthly_vol) ** 2, 1e-8))

    idio_eps      = rng.normal(0, idio_vol, n)
    mkt_comp      = rho * market_returns.values
    stock_returns = monthly_drift + mkt_comp + idio_eps

    prices = [profile.initial_price]
    for r in stock_returns[1:]:
        prices.append(max(prices[-1] * (1 + r), 0.01))
    return prices


def generate_stock_prices(
    tickers_with_sectors: list[tuple[str, str]],
    start_date:           str = SAMPLE_START_DATE,
    end_date:             Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    dates          = _month_ends(start_date, end_date)
    market_returns = _generate_market_returns(dates)
    rows: list[pd.DataFrame] = []

    for ticker, sector in tickers_with_sectors:
        profile     = _build_profile(ticker, sector)
        prices_list = _generate_single_stock_prices(profile, dates, market_returns)
        prices_arr  = np.array(prices_list, dtype=float)
        returns_arr = np.full(len(prices_arr), float("nan"))
        returns_arr[1:] = prices_arr[1:] / prices_arr[:-1] - 1.0

        rng_vol  = _ticker_rng(ticker + "_volume")
        base_vol = rng_vol.uniform(5.0, 100.0)
        volume   = np.abs(rng_vol.normal(base_vol, base_vol * 0.3, len(dates)))

        rows.append(pd.DataFrame({
            "ticker":            ticker,
            "date":              dates.strftime("%Y-%m-%d"),
            "adjusted_close":    prices_arr,
            "monthly_return":    returns_arr,
            "volume":            np.round(volume * 1e6, 0),
            "data_quality_flag": "ok",
        }))

    return pd.concat(rows, ignore_index=True)


# ── Quarterly fundamentals (unit-consistent) ──────────────────────────────────

def _quarter_label(dt: pd.Timestamp) -> str:
    return f"Q{(dt.month - 1) // 3 + 1}"


def _report_lag_days(quarter_label: str) -> int:
    return 90 if quarter_label == "Q4" else 45


def generate_fundamentals(
    tickers_with_sectors: list[tuple[str, str]],
    prices_df:            pd.DataFrame,
    start_date:           str = SAMPLE_START_DATE,
    end_date:             Optional[str] = None,
) -> pd.DataFrame:
    """
    Unit conventions
    ----------------
    eps            $/share   (e.g. 1.50)
    revenue        $B/qtr    (e.g. 25.3)
    net_income     $B/qtr
    equity         $B        (cumulative book value)
    total_debt     $B
    free_cash_flow $B/qtr
    roe            net_income_ttm / equity  (decimal, e.g. 0.18)
    roic           ~ roe * 0.85
    gross_margin   decimal (e.g. 0.62)
    operating_margin decimal
    debt_to_equity decimal
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    q_start       = (pd.Timestamp(start_date) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    quarter_dates = _quarter_ends(q_start, end_date)
    all_rows: list[dict] = []

    for ticker, sector in tickers_with_sectors:
        profile = _build_profile(ticker, sector)
        rng     = _ticker_rng(ticker + "_funds")

        q_eps_growth  = (1 + profile.eps_growth)     ** (1 / 4) - 1
        q_rev_growth  = (1 + profile.revenue_growth) ** (1 / 4) - 1

        # Initial quarterly values (consistent with stock profile)
        eps_q    = profile.initial_eps_annual / 4          # $/share
        rev_q_B  = profile.initial_revenue_B   / 4         # $B/qtr
        equity_B = (profile.initial_market_cap_B /          # rough book value
                    max(profile.initial_pe * profile.roe, 0.1))

        for qend in quarter_dates:
            q_label  = _quarter_label(qend)
            lag_days = _report_lag_days(q_label)
            report_dt = qend + pd.Timedelta(days=lag_days)

            # Grow EPS and revenue with noise
            eps_noise = float(rng.normal(0, profile.eps_vol / math.sqrt(4)))
            rev_noise = float(rng.normal(0, 0.04))
            # Cap EPS: $20/share per quarter covers even the most profitable companies
            eps_q     = max(min(eps_q * (1 + q_eps_growth + eps_noise), 20.0), -10.0)
            rev_q_B   = max(rev_q_B * (1 + q_rev_growth + rev_noise), 0.01)
            # Cap quarterly revenue: $200B covers largest real companies ($90-130B peak)
            rev_q_B   = min(rev_q_B, 200.0)

            # Derive quarterly financials (all in $B)
            net_income_B  = rev_q_B * profile.net_margin
            gross_profit_B = rev_q_B * profile.gross_margin   # noqa: F841
            op_income_B   = rev_q_B * profile.operating_margin
            fcf_B         = net_income_B * profile.fcf_conversion * float(rng.uniform(0.9, 1.1))
            total_debt_B  = equity_B * profile.debt_to_equity
            # Update book equity: add retained earnings (net income minus dividends)
            annual_div_ps = eps_q * 4 * profile.dividend_yield / max(profile.initial_price, 1)
            retained_B    = net_income_B - (annual_div_ps * profile.shares_outstanding_m / 1_000 / 4)
            equity_B      = max(equity_B + retained_B, 0.1)

            # ROE (trailing approximation — annual net income / equity)
            roe_q  = (net_income_B * 4) / equity_B if equity_B > 0 else float("nan")

            all_rows.append({
                "ticker":           ticker,
                "fiscal_period":    q_label,
                "fiscal_date":      qend.strftime("%Y-%m-%d"),
                "report_date":      report_dt.strftime("%Y-%m-%d"),
                "revenue":          round(rev_q_B, 4),          # $B
                "eps":              round(eps_q,   4),           # $/share
                "net_income":       round(net_income_B, 4),      # $B
                "equity":           round(equity_B, 4),          # $B
                "total_debt":       round(total_debt_B, 4),      # $B
                "free_cash_flow":   round(fcf_B, 4),             # $B
                "roe":              round(roe_q, 4) if not math.isnan(roe_q) else None,
                "roic":             round(roe_q * 0.85, 4) if not math.isnan(roe_q) else None,
                "debt_to_equity":   round(profile.debt_to_equity, 4),
                "gross_margin":     round(profile.gross_margin, 4),
                "operating_margin": round(profile.operating_margin, 4),
            })

    return pd.DataFrame(all_rows)


# ── Stocks metadata ───────────────────────────────────────────────────────────

def generate_stocks_metadata(
    tickers_config: list[dict],
    start_date:     str = SAMPLE_START_DATE,
    end_date:       Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    rows = []
    for t in tickers_config:
        ticker  = t["ticker"]
        sector  = t.get("sector", "Technology")
        profile = _build_profile(ticker, sector)
        rows.append({
            "ticker":               ticker,
            "company_name":         t.get("name", ticker),
            "sector":               sector,
            "industry":             t.get("industry", "Unknown"),
            "exchange":             "NASDAQ",
            "currency":             "USD",
            "is_active":            1,
            "shares_outstanding_m": round(profile.shares_outstanding_m, 2),
            "first_available_date": start_date,
            "last_available_date":  end_date,
        })
    return pd.DataFrame(rows)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _get_sample_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sample"


def _cache_is_fresh(cache_dir: Path) -> bool:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for name in ("stocks", "prices_clean", "benchmark_prices", "fundamentals_clean"):
        path = cache_dir / f"{name}.csv"
        if not path.exists():
            return False
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d")
        if mtime < today:
            return False
    return True


def clear_sample_cache() -> None:
    """Delete cached CSV files so data is regenerated on the next call."""
    cache_dir = _get_sample_dir()
    for name in ("stocks", "prices_clean", "benchmark_prices", "fundamentals_clean"):
        path = cache_dir / f"{name}.csv"
        if path.exists():
            path.unlink()
            log.info("Deleted sample cache: %s", path)


def _save_to_cache(data: dict[str, pd.DataFrame], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(cache_dir / f"{name}.csv", index=False)


def _load_from_cache(cache_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(cache_dir / f"{name}.csv")
        for name in ("stocks", "prices_clean", "benchmark_prices", "fundamentals_clean")
    }


# ── Ticker config helper ──────────────────────────────────────────────────────

_FALLBACK_TICKERS = [
    # Actual DJIA 30 components (as of November 2024) — matches tickers.yaml universe
    {"ticker": "AAPL",  "name": "Apple Inc.",                    "sector": "Technology",              "industry": "Consumer Electronics"},
    {"ticker": "MSFT",  "name": "Microsoft Corp.",               "sector": "Technology",              "industry": "Software - Infrastructure"},
    {"ticker": "NVDA",  "name": "NVIDIA Corp.",                  "sector": "Technology",              "industry": "Semiconductors"},
    {"ticker": "CSCO",  "name": "Cisco Systems",                 "sector": "Technology",              "industry": "Communication Equipment"},
    {"ticker": "CRM",   "name": "Salesforce Inc.",               "sector": "Technology",              "industry": "Software - Application"},
    {"ticker": "IBM",   "name": "IBM",                           "sector": "Technology",              "industry": "Information Technology Services"},
    {"ticker": "DIS",   "name": "The Walt Disney Company",       "sector": "Communication Services",  "industry": "Entertainment"},
    {"ticker": "VZ",    "name": "Verizon Communications",        "sector": "Communication Services",  "industry": "Telecom Services"},
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",               "sector": "Consumer Discretionary",  "industry": "Internet Retail"},
    {"ticker": "MCD",   "name": "McDonald's Corp.",              "sector": "Consumer Discretionary",  "industry": "Restaurants"},
    {"ticker": "NKE",   "name": "Nike Inc.",                     "sector": "Consumer Discretionary",  "industry": "Footwear & Accessories"},
    {"ticker": "HD",    "name": "Home Depot Inc.",               "sector": "Consumer Discretionary",  "industry": "Home Improvement Retail"},
    {"ticker": "WMT",   "name": "Walmart Inc.",                  "sector": "Consumer Staples",        "industry": "Discount Stores"},
    {"ticker": "PG",    "name": "Procter & Gamble",              "sector": "Consumer Staples",        "industry": "Household & Personal Products"},
    {"ticker": "KO",    "name": "Coca-Cola Co.",                 "sector": "Consumer Staples",        "industry": "Beverages - Non-Alcoholic"},
    {"ticker": "UNH",   "name": "UnitedHealth Group",            "sector": "Healthcare",              "industry": "Healthcare Plans"},
    {"ticker": "JNJ",   "name": "Johnson & Johnson",             "sector": "Healthcare",              "industry": "Drug Manufacturers - General"},
    {"ticker": "MRK",   "name": "Merck & Co.",                   "sector": "Healthcare",              "industry": "Drug Manufacturers - General"},
    {"ticker": "AMGN",  "name": "Amgen Inc.",                    "sector": "Healthcare",              "industry": "Biotechnology"},
    {"ticker": "JPM",   "name": "JPMorgan Chase",                "sector": "Financials",              "industry": "Banks - Diversified"},
    {"ticker": "V",     "name": "Visa Inc.",                     "sector": "Financials",              "industry": "Credit Services"},
    {"ticker": "GS",    "name": "Goldman Sachs Group",           "sector": "Financials",              "industry": "Capital Markets"},
    {"ticker": "AXP",   "name": "American Express",              "sector": "Financials",              "industry": "Credit Services"},
    {"ticker": "TRV",   "name": "Travelers Companies",           "sector": "Financials",              "industry": "Insurance"},
    {"ticker": "BA",    "name": "Boeing Company",                "sector": "Industrials",             "industry": "Aerospace & Defense"},
    {"ticker": "CAT",   "name": "Caterpillar Inc.",              "sector": "Industrials",             "industry": "Farm & Heavy Construction Machinery"},
    {"ticker": "HON",   "name": "Honeywell International",       "sector": "Industrials",             "industry": "Conglomerates"},
    {"ticker": "MMM",   "name": "3M Company",                    "sector": "Industrials",             "industry": "Conglomerates"},
    {"ticker": "SHW",   "name": "Sherwin-Williams",              "sector": "Materials",               "industry": "Specialty Chemicals"},
    {"ticker": "CVX",   "name": "Chevron Corp.",                 "sector": "Energy",                  "industry": "Oil & Gas Integrated"},
]


def _get_tickers_config(cfg=None) -> list[dict]:
    if cfg is not None:
        try:
            return [
                {"ticker": t.ticker, "name": t.name, "sector": t.sector, "industry": t.industry}
                for t in cfg.tickers.universe
            ]
        except Exception:
            pass
    try:
        from src.utils.config import load_config
        loaded = load_config()
        return [
            {"ticker": t.ticker, "name": t.name, "sector": t.sector, "industry": t.industry}
            for t in loaded.tickers.universe
        ]
    except Exception:
        log.warning("Could not load tickers from config — using fallback list.")
        return _FALLBACK_TICKERS


# ── Main entry point ──────────────────────────────────────────────────────────

def load_sample_data(
    cfg=None,
    start_date: str = SAMPLE_START_DATE,
    end_date:   Optional[str] = None,
    force:      bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Return all sample data as a dict of DataFrames.

    Keys: "stocks", "prices_clean", "benchmark_prices", "fundamentals_clean"

    Args:
        force: Bypass the CSV cache and regenerate from scratch.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    tickers_config       = _get_tickers_config(cfg)
    tickers_with_sectors = [(t["ticker"], t.get("sector", "Technology")) for t in tickers_config]
    cache_dir            = _get_sample_dir()

    if not force and _cache_is_fresh(cache_dir):
        log.info("Loading sample data from cache: %s", cache_dir)
        return _load_from_cache(cache_dir)

    log.info("Generating sample data for %d tickers (%s to %s)",
             len(tickers_with_sectors), start_date, end_date)

    stocks_df = generate_stocks_metadata(tickers_config, start_date, end_date)
    bench_df  = generate_benchmark_prices(start_date, end_date)
    prices_df = generate_stock_prices(tickers_with_sectors, start_date, end_date)
    funds_df  = generate_fundamentals(tickers_with_sectors, prices_df, start_date, end_date)

    data = {
        "stocks":             stocks_df,
        "prices_clean":       prices_df,
        "benchmark_prices":   bench_df,
        "fundamentals_clean": funds_df,
    }
    _save_to_cache(data, cache_dir)
    log.info("Sample data generated and cached.")
    return data
