"""
src/features/feature_engineering.py

Calculates all 25 stock-level features for a given prediction month.

LOOK-AHEAD BIAS RULE
--------------------
At each feature_date, this module uses ONLY data that would have been
available on or before that date:
  • Prices:       all monthly closes up to and including feature_date
  • Fundamentals: only rows where report_date <= feature_date
                  (quarterly results are available ~45 days after quarter end)

Violating this rule inflates backtest performance and makes the model
look better than it really is.

FEATURE GROUPS
--------------
Group A – Keyes-inspired core (6 features)
  five_year_price_gain, five_year_eps_growth, five_year_revenue_growth,
  current_pe_ratio, pe_vs_historical_median, dividend_yield

Group B – Modern fundamental quality (8 features)
  roe, roic, debt_to_equity, free_cash_flow_yield,
  gross_margin, operating_margin, price_to_book, price_to_sales

Group C – Price momentum & risk (5 features)
  six_month_momentum, twelve_month_momentum,
  volatility_12m, market_cap, beta (via rolling_beta)

Group D – Growth acceleration (2 features)
  revenue_growth_acceleration, eps_growth_acceleration

Group E – Sector-relative (2 features)
  sector_relative_pe, sector_relative_momentum

Plus: data_quality_score (0–100)

Usage
-----
    from src.features.feature_engineering import compute_features_for_date

    features_df = compute_features_for_date(
        feature_date   = "2024-01-31",
        prices_df      = prices_clean_df,
        benchmark_df   = benchmark_prices_df,
        fundamentals_df= fundamentals_clean_df,
        stocks_df      = stocks_df,
    )
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.features.lag_rules import (
    filter_fundamentals_lag_safe,
    get_ttm_fundamentals,
    get_historical_fundamentals,
    staleness_days,
)
from src.features.transformations import winsorize_features, FEATURE_COLS
from src.utils.math_utils import cagr, annualize_vol, rolling_beta, clip_pe, safe_divide, trailing_sum
from src.utils.logging import get_logger

log = get_logger(__name__)

# Minimum months of price history required to calculate momentum
MIN_PRICE_MONTHS = 13

# Data quality deduction table: (feature_name, points_deducted_if_missing)
_DQ_DEDUCTIONS = [
    ("current_pe_ratio",         12),
    ("five_year_eps_growth",     10),
    ("five_year_revenue_growth",  8),
    ("five_year_price_gain",      8),
    ("twelve_month_momentum",     6),
    ("six_month_momentum",        5),
    ("volatility_12m",            5),
    ("roe",                       5),
    ("debt_to_equity",            4),
    ("free_cash_flow_yield",      4),
    ("gross_margin",              3),
    ("operating_margin",          3),
    ("pe_vs_historical_median",   3),
    ("dividend_yield",            3),
    ("roic",                      3),
    ("price_to_book",             3),
    ("price_to_sales",            3),
    ("revenue_growth_acceleration", 2),
    ("eps_growth_acceleration",   2),
    # New signals — smaller deductions (derivable from price/volume)
    ("return_1m",                  1),
    ("return_3m",                  1),
    ("drawdown_from_52w_high",     2),
    ("volatility_3m",              2),
    ("downside_volatility_12m",    2),
    ("abnormal_volume",            2),
]


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_features_for_date(
    feature_date:    str,
    prices_df:       pd.DataFrame,
    benchmark_df:    pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    stocks_df:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all features for every ticker as of feature_date.

    Args:
        feature_date:    Month-end prediction date (YYYY-MM-DD).
        prices_df:       prices_clean DataFrame (all tickers, all dates).
        benchmark_df:    benchmark_prices DataFrame.
        fundamentals_df: fundamentals_clean DataFrame (all tickers, all dates).
        stocks_df:       stocks DataFrame (provides sector mapping).

    Returns:
        DataFrame with one row per ticker and all feature columns populated
        where data is available (NaN otherwise), plus data_quality_score.
    """
    log.info("Computing features for date: %s", feature_date)

    # ── 1. Filter to look-ahead-safe data ────────────────────────────────────
    prices_avail = prices_df[prices_df["date"] <= feature_date].copy()
    bench_avail  = benchmark_df[benchmark_df["date"] <= feature_date].copy()
    funds_avail  = filter_fundamentals_lag_safe(fundamentals_df, feature_date)

    # Pre-index prices by ticker for speed
    price_groups = {
        ticker: grp.sort_values("date").reset_index(drop=True)
        for ticker, grp in prices_avail.groupby("ticker")
    }

    # Pre-index fundamentals by ticker for speed
    fund_groups = {
        ticker: grp.sort_values("fiscal_date").reset_index(drop=True)
        for ticker, grp in funds_avail.groupby("ticker")
    }

    # Benchmark return series indexed by date
    bench_returns = bench_avail.set_index("date")["monthly_return"]

    # ── 2. Build lookups ─────────────────────────────────────────────────────
    sector_map = dict(zip(stocks_df["ticker"], stocks_df["sector"]))
    shares_map = (
        dict(zip(stocks_df["ticker"], stocks_df["shares_outstanding_m"]))
        if "shares_outstanding_m" in stocks_df.columns else {}
    )

    # ── 3. Per-ticker feature calculation ────────────────────────────────────
    rows: list[dict] = []
    for _, stock_row in stocks_df.iterrows():
        ticker    = stock_row["ticker"]
        sector    = sector_map.get(ticker, "Unknown")
        shares_m  = shares_map.get(ticker)

        row = _compute_ticker_features(
            ticker        = ticker,
            sector        = sector,
            feature_date  = feature_date,
            price_grp     = price_groups.get(ticker, pd.DataFrame()),
            bench_returns = bench_returns,
            fund_grp      = fund_groups.get(ticker, pd.DataFrame()),
            funds_avail   = funds_avail,
            shares_m      = float(shares_m) if shares_m is not None else None,
        )
        rows.append(row)

    features_df = pd.DataFrame(rows)

    # ── 4. Sector-relative features ───────────────────────────────────────────
    features_df = _add_sector_relative_features(features_df)

    # ── 5. Winsorise numeric columns ─────────────────────────────────────────
    features_df = winsorize_features(features_df)

    # ── 6. Data quality score ─────────────────────────────────────────────────
    features_df["data_quality_score"] = features_df.apply(
        lambda r: _compute_dq_score(r, funds_avail, feature_date), axis=1
    )

    log.info(
        "Features computed for %s: %d tickers, mean DQ score = %.1f",
        feature_date,
        len(features_df),
        features_df["data_quality_score"].mean(),
    )
    return features_df


def compute_features_for_all_dates(
    prices_df:       pd.DataFrame,
    benchmark_df:    pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    stocks_df:       pd.DataFrame,
    start_date:      Optional[str] = None,
    end_date:        Optional[str] = None,
    skip_existing:   bool          = True,
) -> pd.DataFrame:
    """
    Compute features for every available month-end date.

    Used to build the full historical feature matrix for model training.
    Respects look-ahead bias rules at every date.

    Args:
        start_date:    First feature date to compute (default: earliest available).
        end_date:      Last feature date to compute (default: most recent prices date).
        skip_existing: If True, skip dates already stored in the database.
                       Dramatically speeds up monthly update runs.

    Returns:
        DataFrame containing only newly computed rows (empty if all exist).
    """
    all_dates = sorted(prices_df["date"].unique())
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    if not all_dates:
        log.warning("No dates in range %s to %s", start_date, end_date)
        return pd.DataFrame()

    # Skip dates where ALL tickers in the current universe already have features.
    # Uses (date, ticker) pairs so new tickers added mid-run are not skipped.
    if skip_existing:
        try:
            from src.database.queries import load_computed_feature_pairs
            existing_pairs = load_computed_feature_pairs()
            universe_tickers = set(stocks_df["ticker"].unique()) if stocks_df is not None else set()
            pending = [
                d for d in all_dates
                if any((d, t) not in existing_pairs for t in universe_tickers)
            ]
            skipped = len(all_dates) - len(pending)
            if skipped:
                log.info("Skipping %d already-computed dates; %d remaining.", skipped, len(pending))
            all_dates = pending
        except Exception as exc:
            log.debug("Could not check existing feature pairs (DB may not be ready): %s", exc)

    if not all_dates:
        log.info("All feature dates are already computed and stored.")
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    for i, date in enumerate(all_dates):
        log.debug("Computing features %d/%d: %s", i + 1, len(all_dates), date)
        df = compute_features_for_date(date, prices_df, benchmark_df, fundamentals_df, stocks_df)
        # Only keep rows for ticker/date pairs not already in the DB
        if skip_existing and 'existing_pairs' in dir():
            df = df[~df.apply(
                lambda r: (r["feature_date"], r["ticker"]) in existing_pairs, axis=1
            )]
        chunks.append(df)

    non_empty = [c for c in chunks if not c.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def backfill_new_tickers(
    new_tickers:     list[str],
    prices_df:       pd.DataFrame,
    benchmark_df:    pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    stocks_df:       pd.DataFrame,
    start_date:      Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute full historical features for tickers that have no prior feature rows.

    Called when new Hidden Gems enter the universe for the first time.
    Recomputes ALL tickers at each affected date so sector-relative features
    (sector_relative_pe, sector_relative_momentum, etc.) remain accurate.
    Only the new ticker rows are returned — existing Core rows are not re-saved.

    Args:
        new_tickers: Tickers with price data but no feature history.
        start_date:  Earliest date to backfill (default: earliest in prices).

    Returns:
        DataFrame of new feature rows for the new tickers only.
    """
    if not new_tickers:
        return pd.DataFrame()

    from src.database.queries import load_computed_feature_pairs
    existing_pairs = load_computed_feature_pairs()
    new_set = set(new_tickers)

    # Dates where at least one new ticker has price data and is missing features
    ticker_dates = prices_df[prices_df["ticker"].isin(new_set)]["date"].unique()
    if start_date:
        ticker_dates = [d for d in ticker_dates if d >= start_date]
    dates_needed = sorted([
        d for d in ticker_dates
        if any((d, t) not in existing_pairs for t in new_set
               if not prices_df[(prices_df["ticker"] == t) & (prices_df["date"] == d)].empty)
    ])

    if not dates_needed:
        log.info("No backfill needed — all new tickers already have features.")
        return pd.DataFrame()

    log.info(
        "Backfilling %d new tickers across %d historical dates: %s...",
        len(new_tickers), len(dates_needed), ", ".join(new_tickers[:5]),
    )

    chunks: list[pd.DataFrame] = []
    for i, date in enumerate(dates_needed):
        full_df = compute_features_for_date(
            date, prices_df, benchmark_df, fundamentals_df, stocks_df
        )
        # Only keep rows for new tickers that are genuinely missing
        new_rows = full_df[
            full_df["ticker"].isin(new_set) &
            ~full_df.apply(lambda r: (r["feature_date"], r["ticker"]) in existing_pairs, axis=1)
        ]
        if not new_rows.empty:
            chunks.append(new_rows)
        if (i + 1) % 20 == 0:
            log.info("Backfill progress: %d/%d dates done", i + 1, len(dates_needed))

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    log.info("Backfill complete: %d new feature rows for %d tickers.",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ── Per-ticker calculation ────────────────────────────────────────────────────

def _compute_ticker_features(
    ticker:        str,
    sector:        str,
    feature_date:  str,
    price_grp:     pd.DataFrame,
    bench_returns: pd.Series,
    fund_grp:      pd.DataFrame,
    funds_avail:   pd.DataFrame,
    shares_m:      Optional[float] = None,
) -> dict:
    """Compute all features for one ticker at one date."""

    row: dict = {
        "feature_date": feature_date,
        "ticker":       ticker,
        "sector":       sector,
    }

    # Initialise all feature columns to None
    for col in FEATURE_COLS:
        row[col] = None

    # ── Price series ──────────────────────────────────────────────────────────
    if price_grp.empty:
        return row

    prices    = price_grp["adjusted_close"]
    returns   = price_grp["monthly_return"]
    n_months  = len(price_grp)
    cur_price = float(prices.iloc[-1])

    if cur_price <= 0:
        return row

    # ── Momentum (12-1 skip-month convention) ─────────────────────────────────
    # Standard financial momentum skips the most recent month (t-1) to avoid
    # short-term reversal (Jegadeesh & Titman 1993).
    # 6-month:  price[t-1] / price[t-7]  — return from 6 months ago to 1 month ago
    # 12-month: price[t-1] / price[t-13] — return from 12 months ago to 1 month ago
    # Minimum data needed: 8 months for 6m, 14 months for 12m.
    if n_months >= 8:
        row["six_month_momentum"] = safe_divide(float(prices.iloc[-2]), float(prices.iloc[-8])) - 1

    if n_months >= 14:
        row["twelve_month_momentum"] = safe_divide(float(prices.iloc[-2]), float(prices.iloc[-14])) - 1

    # ── 1-month return (short-term reversal signal — no skip) ─────────────────
    if n_months >= 2:
        row["return_1m"] = safe_divide(float(prices.iloc[-1]), float(prices.iloc[-2])) - 1

    # ── 3-month return ────────────────────────────────────────────────────────
    if n_months >= 4:
        row["return_3m"] = safe_divide(float(prices.iloc[-1]), float(prices.iloc[-4])) - 1

    # ── 5-year price gain ─────────────────────────────────────────────────────
    if n_months >= 61:
        row["five_year_price_gain"] = safe_divide(cur_price, float(prices.iloc[-61])) - 1

    # ── Volatility (12m, 3m) ──────────────────────────────────────────────────
    if n_months >= 13:
        row["volatility_12m"] = annualize_vol(returns.iloc[-12:])

    if n_months >= 4:
        row["volatility_3m"] = annualize_vol(returns.iloc[-3:])

    # ── Downside volatility (12m) ─────────────────────────────────────────────
    # Only uses negative monthly returns — basis of Sortino ratio.
    if n_months >= 13:
        neg_returns = returns.iloc[-12:].dropna()
        neg_returns = neg_returns[neg_returns < 0]
        if len(neg_returns) >= 3:
            row["downside_volatility_12m"] = float(neg_returns.std() * math.sqrt(12))

    # ── Drawdown from 52-week high (George & Hwang 2004) ─────────────────────
    # Negative value: -0.15 means 15 % below the 52-week high.
    if n_months >= 13:
        high_52w = float(prices.iloc[-13:].max())
        if high_52w > 0:
            row["drawdown_from_52w_high"] = safe_divide(cur_price - high_52w, high_52w)

    # ── Abnormal volume ───────────────────────────────────────────────────────
    # current volume / 12-month average volume.  >1 = above-average activity.
    if n_months >= 13 and "volume" in price_grp.columns:
        cur_vol  = price_grp["volume"].iloc[-1]
        avg_vol  = price_grp["volume"].iloc[-13:-1].mean()
        if avg_vol and avg_vol > 0 and not math.isnan(avg_vol):
            row["abnormal_volume"] = safe_divide(float(cur_vol), float(avg_vol))

    # ── Beta ─────────────────────────────────────────────────────────────────
    # Align stock and benchmark returns by date
    stock_ret_series = price_grp.set_index("date")["monthly_return"]
    common_dates     = stock_ret_series.index.intersection(bench_returns.index)
    if len(common_dates) >= 12:
        row["beta"] = rolling_beta(
            stock_ret_series.loc[common_dates].tail(24),
            bench_returns.loc[common_dates].tail(24),
        )

    # ── Fundamental features ──────────────────────────────────────────────────
    if fund_grp.empty:
        return row

    ttm = fund_grp.tail(4)  # Trailing twelve months (last 4 quarters)
    latest = fund_grp.iloc[-1]

    # TTM sums
    ttm_eps     = trailing_sum(ttm["eps"],     4)
    ttm_revenue = trailing_sum(ttm["revenue"], 4)

    # ── P/E ratio ─────────────────────────────────────────────────────────────
    if ttm_eps and ttm_eps > 0:
        raw_pe = safe_divide(cur_price, ttm_eps)
        row["current_pe_ratio"] = clip_pe(raw_pe)

    # ── Historical median P/E ─────────────────────────────────────────────────
    # Compute trailing PE for all historical quarters available
    _add_pe_vs_median(row, funds_avail, ticker, cur_price)

    # ── 5-year EPS growth ─────────────────────────────────────────────────────
    hist_funds = get_historical_fundamentals(funds_avail, ticker, 5, feature_date)
    if not hist_funds.empty:
        old_ttm_eps = trailing_sum(hist_funds["eps"], min(4, len(hist_funds)))
        if old_ttm_eps and old_ttm_eps > 0 and ttm_eps and ttm_eps > 0:
            row["five_year_eps_growth"] = cagr(old_ttm_eps, ttm_eps, 5.0)

    # ── 5-year revenue growth ─────────────────────────────────────────────────
    if not hist_funds.empty:
        old_ttm_rev = trailing_sum(hist_funds["revenue"], min(4, len(hist_funds)))
        if old_ttm_rev and old_ttm_rev > 0 and ttm_revenue and ttm_revenue > 0:
            row["five_year_revenue_growth"] = cagr(old_ttm_rev, ttm_revenue, 5.0)

    # ── Fundamental quality ratios from latest quarter ────────────────────────
    for col in ("roe", "roic", "debt_to_equity", "gross_margin", "operating_margin"):
        val = latest.get(col)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            row[col] = float(val)

    # ── Dividend yield ────────────────────────────────────────────────────────
    row["dividend_yield"] = _estimate_dividend_yield(fund_grp, cur_price, sector)

    # ── Market cap (price × shares, all in $B) ───────────────────────────────
    # Prefer shares_outstanding_m from stocks table (clean, authoritative).
    # Fall back to deriving from net_income / EPS if shares not available.
    market_cap_B: Optional[float] = None
    if shares_m and shares_m > 0:
        market_cap_B = cur_price * shares_m / 1_000          # $B
    else:
        ttm_net_income = trailing_sum(ttm["net_income"], 4)   # $B
        if ttm_net_income and ttm_eps and ttm_eps > 0:
            implied_shares_m = ttm_net_income / ttm_eps * 1_000  # rough shares in M
            if implied_shares_m > 0:
                market_cap_B = cur_price * implied_shares_m / 1_000

    if market_cap_B and market_cap_B > 0:
        row["market_cap"] = market_cap_B * 1_000              # store in $M

    # ── Free cash flow yield ──────────────────────────────────────────────────
    ttm_fcf = trailing_sum(ttm["free_cash_flow"], 4)           # $B
    if ttm_fcf is not None and market_cap_B and market_cap_B > 0:
        row["free_cash_flow_yield"] = safe_divide(ttm_fcf, market_cap_B)

    # ── Price-to-sales ────────────────────────────────────────────────────────
    # revenue is in $B/quarter; TTM revenue is sum of 4 quarters → $B
    if ttm_revenue and ttm_revenue > 0 and market_cap_B and market_cap_B > 0:
        row["price_to_sales"] = safe_divide(market_cap_B, ttm_revenue)

    # ── Price-to-book ─────────────────────────────────────────────────────────
    # equity is in $B (cumulative)
    equity = latest.get("equity")
    if equity and equity > 0 and market_cap_B and market_cap_B > 0:
        row["price_to_book"] = safe_divide(market_cap_B, float(equity))

    # ── Growth acceleration ───────────────────────────────────────────────────
    _add_growth_acceleration(row, fund_grp)

    return row


# ── Feature helpers ───────────────────────────────────────────────────────────

def _add_pe_vs_median(
    row: dict,
    funds_avail: pd.DataFrame,
    ticker: str,
    cur_price: float,
) -> None:
    """
    Compute pe_vs_historical_median:
        current_pe / median(historical_pe over past 5 years) - 1

    Uses all available lag-safe fundamentals to build a PE history.
    """
    ticker_funds = (
        funds_avail[funds_avail["ticker"] == ticker]
        .sort_values("fiscal_date")
    )
    if len(ticker_funds) < 8:  # need at least 2 years of quarterly data
        return

    # Build rolling TTM EPS at each quarter and compute PE
    eps_vals = ticker_funds["eps"].values
    pe_history = []
    for i in range(3, len(eps_vals)):
        ttm_eps_hist = sum(eps_vals[i - 3: i + 1])
        if ttm_eps_hist > 0:
            pe_history.append(cur_price / ttm_eps_hist)

    if len(pe_history) >= 4:
        median_pe = float(np.median(pe_history[-20:]))  # last 5 years of quarterly PEs
        current_pe = row.get("current_pe_ratio")
        if current_pe and median_pe > 0:
            row["pe_vs_historical_median"] = safe_divide(current_pe, median_pe) - 1


def _estimate_dividend_yield(
    fund_grp: pd.DataFrame,
    cur_price: float,
    sector: str,
) -> Optional[float]:
    """
    Estimate annual dividend yield from sector-typical payout ratios.

    In real data this comes from dividend history.  For sample data we
    use operating_margin and sector as a proxy for the payout ratio.
    Returns None if estimation is not possible.
    """
    if fund_grp.empty or cur_price <= 0:
        return None

    # Sector-typical payout ratios used as a rough proxy
    _SECTOR_PAYOUT = {
        "Consumer Staples":       0.55,
        "Energy":                 0.45,
        "Financials":             0.40,
        "Healthcare":             0.30,
        "Technology":             0.15,
        "Communication Services": 0.20,
        "Consumer Discretionary": 0.20,
    }
    payout = _SECTOR_PAYOUT.get(sector, 0.25)

    ttm_eps = trailing_sum(fund_grp.tail(4)["eps"], 4)
    if ttm_eps is None or ttm_eps <= 0:
        return None

    annual_dividend = ttm_eps * payout
    return safe_divide(annual_dividend, cur_price)


def _add_growth_acceleration(row: dict, fund_grp: pd.DataFrame) -> None:
    """
    Compute revenue_growth_acceleration and eps_growth_acceleration.

    Acceleration = recent YoY growth rate - prior YoY growth rate.
    Positive = growth is speeding up; negative = slowing down.
    Requires at least 8 quarters of data.
    """
    if len(fund_grp) < 8:
        return

    q = fund_grp.sort_values("fiscal_date")

    # Recent YoY: last quarter vs same quarter 4 quarters ago
    recent_eps   = q["eps"].iloc[-1]
    prior_eps    = q["eps"].iloc[-5]
    earlier_eps  = q["eps"].iloc[-9] if len(q) >= 10 else None

    if prior_eps and prior_eps != 0:
        recent_eps_growth = safe_divide(recent_eps - prior_eps, abs(prior_eps))
    else:
        recent_eps_growth = None

    if earlier_eps and earlier_eps != 0 and prior_eps is not None:
        prior_eps_growth = safe_divide(prior_eps - earlier_eps, abs(earlier_eps))
    else:
        prior_eps_growth = None

    if recent_eps_growth is not None and prior_eps_growth is not None:
        row["eps_growth_acceleration"] = recent_eps_growth - prior_eps_growth

    # Same for revenue
    recent_rev  = q["revenue"].iloc[-1]
    prior_rev   = q["revenue"].iloc[-5]
    earlier_rev = q["revenue"].iloc[-9] if len(q) >= 10 else None

    if prior_rev and prior_rev > 0:
        recent_rev_growth = safe_divide(recent_rev - prior_rev, prior_rev)
    else:
        recent_rev_growth = None

    if earlier_rev and earlier_rev > 0 and prior_rev is not None:
        prior_rev_growth = safe_divide(prior_rev - earlier_rev, earlier_rev)
    else:
        prior_rev_growth = None

    if recent_rev_growth is not None and prior_rev_growth is not None:
        row["revenue_growth_acceleration"] = recent_rev_growth - prior_rev_growth


def _add_sector_relative_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional, sector-relative features.

    sector_relative_pe       = stock PE / sector median PE - 1
    sector_relative_momentum = stock 12m return / sector median 12m return - 1

    These require all tickers to be computed first, then we normalise
    within sector groups.
    """
    if "sector" not in features_df.columns:
        return features_df

    result = features_df.copy()

    for sector, group in result.groupby("sector"):
        idx = group.index

        # Sector-relative P/E
        sector_pe = group["current_pe_ratio"].median()
        if sector_pe and sector_pe > 0:
            result.loc[idx, "sector_relative_pe"] = (
                result.loc[idx, "current_pe_ratio"] / sector_pe - 1
            )

        # Sector-relative 12m momentum
        sector_mom = group["twelve_month_momentum"].median()
        if sector_mom is not None and not math.isnan(float(sector_mom or float("nan"))):
            result.loc[idx, "sector_relative_momentum"] = (
                result.loc[idx, "twelve_month_momentum"] - sector_mom
            )

        # ── NEW: Sector-relative P/S ──────────────────────────────────────────
        sector_ps = group["price_to_sales"].median()
        if sector_ps and sector_ps > 0:
            result.loc[idx, "sector_relative_ps"] = (
                result.loc[idx, "price_to_sales"] / sector_ps - 1
            )

        # ── NEW: Sector-relative FCF yield (additive difference) ──────────────
        fcf_vals = group["free_cash_flow_yield"].dropna()
        if len(fcf_vals) >= 3:
            sector_fcf = float(fcf_vals.median())
            result.loc[idx, "sector_relative_fcf_yield"] = (
                result.loc[idx, "free_cash_flow_yield"] - sector_fcf
            )

        # ── NEW: Peer momentum z-score (rank within sector) ───────────────────
        # Minimum 3 peers to compute a meaningful z-score
        mom_vals = group["twelve_month_momentum"].dropna()
        if len(mom_vals) >= 3:
            mu_m  = float(mom_vals.mean())
            std_m = float(mom_vals.std())
            if std_m > 0:
                result.loc[idx, "peer_momentum_zscore"] = (
                    (result.loc[idx, "twelve_month_momentum"] - mu_m) / std_m
                )

        # ── NEW: Peer valuation z-score (rank within sector) ─────────────────
        pe_vals = group["current_pe_ratio"].dropna()
        if len(pe_vals) >= 3:
            mu_v  = float(pe_vals.mean())
            std_v = float(pe_vals.std())
            if std_v > 0:
                result.loc[idx, "peer_valuation_zscore"] = (
                    (result.loc[idx, "current_pe_ratio"] - mu_v) / std_v
                )

    return result


# ── Data quality score ────────────────────────────────────────────────────────

def _compute_dq_score(
    row: pd.Series,
    funds_avail: pd.DataFrame,
    feature_date: str,
) -> float:
    """
    Calculate a 0–100 data quality score for one stock at one date.

    Starts at 100 and deducts points for:
      • Each missing feature (weighted by importance)
      • Stale fundamental data (>6 months old)
    """
    score = 100.0

    for feat, deduction in _DQ_DEDUCTIONS:
        val = row.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            score -= deduction

    # Stale fundamentals penalty
    ticker = row.get("ticker")
    if ticker:
        stale = staleness_days(funds_avail, ticker, feature_date)
        if stale is None:
            score -= 20  # No fundamentals at all
        elif stale > 180:
            score -= 10  # Very stale (>6 months)
        elif stale > 90:
            score -= 5   # Moderately stale (3-6 months)

    return max(0.0, min(100.0, score))
