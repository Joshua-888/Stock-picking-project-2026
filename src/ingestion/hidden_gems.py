"""
src/ingestion/hidden_gems.py

Monthly random selection of "Hidden Gems" from the broader stock pool.

Replicates the Keyes (1972) methodology:
  His Dow Jones 30 Industrials  =  our Core 30 large-caps (config/tickers.yaml: universe)
  His 30 randomly-selected firms =  our monthly draw from hidden_gems_pool

KEY DESIGN DECISION — Seeded randomness
  The selection is seeded by the prediction month (e.g., "2025-05") so:
  • Every run in the same month picks the SAME 30 gems → stable predictions
  • Each new month rotates to a fresh set → exposure across the full pool
  • Over 12 months all ~60 pool stocks get analysed at least once

This is exactly how Keyes ensured his randomly-selected group was independent
of his Dow Jones group, while still being reproducible.

Usage
-----
    from src.ingestion.hidden_gems import get_monthly_gems, get_all_tickers

    # 30 randomly selected gems for May 2025
    gems = get_monthly_gems(prediction_date="2025-05-31")
    # [{"ticker": "CDNS", "name": "...", "sector": "...", ...}, ...]

    # Combined universe: core 30 + 30 gems
    all_tickers = get_all_tickers(prediction_date="2025-05-31")
"""

from __future__ import annotations

import hashlib
from typing import Optional

from src.utils.logging import get_logger

log = get_logger(__name__)


def get_monthly_gems(
    prediction_date:   str,
    cfg=None,
    n:                 Optional[int] = None,
    available_tickers: Optional[set] = None,
) -> list[dict]:
    """
    Return N randomly-selected Hidden Gems for the given prediction month.

    The selection is deterministic within a month: calling this function
    multiple times with the same prediction_date returns identical results.

    Args:
        prediction_date:   Month-end date (YYYY-MM-DD).
        cfg:               AppConfig object (loaded if None).
        n:                 Override number of gems (defaults to config value).
        available_tickers: Set of tickers that successfully downloaded data.
                           Failed tickers are replaced with pool alternatives.

    Returns:
        List of ticker dicts: [{ticker, name, sector, industry, category}, ...]
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    pool = _load_pool(cfg)
    if not pool:
        log.warning("Hidden gems pool is empty — check tickers.yaml")
        return []

    if not getattr(getattr(cfg, "hidden_gems", None), "enabled", True):
        log.info("Hidden gems disabled in config.")
        return []

    n_select = n or getattr(getattr(cfg, "hidden_gems", None), "n_select", 30)
    n_select = min(n_select, len(pool))

    # Seed from the month prefix (YYYY-MM) for within-month stability
    month_key = prediction_date[:7]                     # e.g. "2025-05"
    seed      = int(hashlib.md5(month_key.encode()).hexdigest()[:8], 16)

    import random
    rng = random.Random(seed)
    selected = rng.sample(pool, n_select)

    # Substitute any tickers that failed to download with pool alternatives
    if available_tickers is not None:
        confirmed  = [s for s in selected if s["ticker"] in available_tickers]
        failed     = [s["ticker"] for s in selected if s["ticker"] not in available_tickers]
        if failed:
            selected_tickers = {s["ticker"] for s in selected}
            reserve = [
                t for t in pool
                if t["ticker"] not in selected_tickers
                and t["ticker"] in available_tickers
            ]
            rng.shuffle(reserve)
            substitutes = reserve[:len(failed)]
            if len(substitutes) < len(failed):
                log.warning(
                    "Only %d substitutes available for %d failed tickers: %s",
                    len(substitutes), len(failed), failed,
                )
            confirmed.extend(substitutes)
            log.info(
                "Substituted %d failed tickers (%s) with: %s",
                len(failed), failed,
                [s["ticker"] for s in substitutes],
            )
            selected = confirmed

    # Tag as Hidden Gem
    for s in selected:
        s["category"] = "Hidden Gem"

    log.info(
        "Selected %d Hidden Gems for %s (seed=%d): %s...",
        len(selected), month_key, seed,
        ", ".join(s["ticker"] for s in selected[:5]),
    )
    return selected


def get_core_tickers(cfg=None) -> list[dict]:
    """
    Return the permanent Core 30 large-cap tickers, tagged as 'Core'.
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    core = [
        {
            "ticker":   t.ticker,
            "name":     t.name,
            "sector":   t.sector,
            "industry": t.industry,
            "category": "Core",
        }
        for t in cfg.tickers.universe
    ]
    return core


def get_all_tickers(
    prediction_date:   str,
    cfg=None,
    available_tickers: Optional[set] = None,
) -> list[dict]:
    """
    Return the combined universe: Core 30 + N randomly-selected Hidden Gems.

    This is the full ticker list that the monthly analysis runs against —
    exactly as Keyes combined his DJ30 with his randomly-selected 30.

    Args:
        prediction_date: Month-end date (YYYY-MM-DD).
        cfg:             AppConfig object (loaded if None).

    Returns:
        List of ticker dicts with 'category' = 'Core' or 'Hidden Gem'.
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    core = get_core_tickers(cfg)
    gems = get_monthly_gems(prediction_date, cfg, available_tickers=available_tickers)

    # Deduplicate: gems that happen to match a core ticker are skipped
    core_tickers = {t["ticker"] for t in core}
    gems_deduped = [g for g in gems if g["ticker"] not in core_tickers]

    combined = core + gems_deduped
    log.info(
        "Combined universe for %s: %d Core + %d Hidden Gems = %d total",
        prediction_date[:7], len(core), len(gems_deduped), len(combined),
    )
    return combined


def pool_summary(cfg=None) -> dict:
    """
    Return statistics about the hidden gems pool.
    Useful for the Data Quality dashboard page.
    """
    if cfg is None:
        from src.utils.config import load_config
        cfg = load_config()

    pool = _load_pool(cfg)
    if not pool:
        return {"n_pool": 0, "sectors": {}, "n_select": 0}

    sectors: dict[str, int] = {}
    for t in pool:
        s = t.get("sector", "Unknown")
        sectors[s] = sectors.get(s, 0) + 1

    n_select = getattr(getattr(cfg, "hidden_gems", None), "n_select", 30)
    return {
        "n_pool":    len(pool),
        "n_select":  n_select,
        "sectors":   sectors,
        "tickers":   [t["ticker"] for t in pool],
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_pool(cfg) -> list[dict]:
    """Load hidden_gems_pool from tickers.yaml via config."""
    try:
        import yaml
        from pathlib import Path
        tickers_path = Path(__file__).resolve().parents[2] / "config" / "tickers.yaml"
        with tickers_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        pool_raw = raw.get("hidden_gems_pool", [])
        return [
            {
                "ticker":   t["ticker"],
                "name":     t.get("name", t["ticker"]),
                "sector":   t.get("sector", "Unknown"),
                "industry": t.get("industry", "Unknown"),
            }
            for t in pool_raw
            if t.get("ticker")
        ]
    except Exception as exc:
        log.error("Failed to load hidden gems pool: %s", exc)
        return []
