"""
src/utils/config.py

Loads config.yaml and tickers.yaml, merges in environment variables,
and exposes a single AppConfig object used throughout the project.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg.data.provider)           # "sample" or "nasdaq_data_link"
    print(cfg.database.path)           # "data/stock_analysis.db"
    print(cfg.tickers.universe)        # list of ticker dicts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file if it exists (does nothing if the file is absent)
load_dotenv()

# Paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"
_TICKERS_PATH = _PROJECT_ROOT / "config" / "tickers.yaml"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    provider: str         = "sample"
    benchmark_ticker: str = "SPY"
    start_date: str       = "2010-01-01"
    end_date: str | None  = None


@dataclass
class DatabaseConfig:
    type: str  = "sqlite"
    path: str  = "data/stock_analysis.db"


@dataclass
class ReportingLagsConfig:
    quarterly_earnings_days: int = 45
    annual_report_days: int      = 90


@dataclass
class FeaturesConfig:
    lookback_years: int           = 5
    momentum_short_months: int    = 6
    momentum_long_months: int     = 12
    volatility_months: int        = 12
    winsorize_lower: float        = 0.01
    winsorize_upper: float        = 0.99
    min_price_history_months: int = 24


@dataclass
class TargetsConfig:
    horizon_months: int = 12


@dataclass
class ScoringWeights:
    probability_of_outperformance: float = 0.35
    expected_excess_return: float        = 0.25
    model_agreement: float               = 0.15
    risk_score: float                    = 0.10
    data_quality: float                  = 0.10
    valuation: float                     = 0.05


@dataclass
class ClassificationThreshold:
    min_probability: float     = 0.60
    min_excess_return: float   = 0.05
    min_model_agreement: float = 0.70
    min_data_quality: float    = 75.0


@dataclass
class ScoringConfig:
    weights: ScoringWeights                           = field(default_factory=ScoringWeights)
    strong_candidate: ClassificationThreshold         = field(default_factory=ClassificationThreshold)
    watchlist_candidate: ClassificationThreshold      = field(
        default_factory=lambda: ClassificationThreshold(
            min_probability=0.50,
            min_excess_return=0.02,
            min_model_agreement=0.55,
            min_data_quality=60.0,
        )
    )


@dataclass
class ModelsConfig:
    ridge_alpha: float        = 1.0
    lasso_alpha: float        = 0.1
    vif_threshold: float      = 10.0
    min_observations: int     = 30
    significance_level: float = 0.05


@dataclass
class BacktestingConfig:
    min_training_months: int   = 36
    top_n_stocks: int          = 10
    rebalance_frequency: str   = "monthly"


@dataclass
class LoggingConfig:
    level: str        = "INFO"
    file: str         = "logs/stock_analysis.log"
    max_bytes: int    = 5_242_880
    backup_count: int = 3


@dataclass
class TickerEntry:
    ticker: str
    name: str
    sector: str
    industry: str


@dataclass
class BenchmarkEntry:
    ticker: str
    name: str
    description: str = ""


@dataclass
class TickersConfig:
    benchmark: BenchmarkEntry          = field(default_factory=lambda: BenchmarkEntry("SPY", "SPDR S&P 500 ETF Trust"))
    universe: list[TickerEntry]        = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [t.ticker for t in self.universe]


@dataclass
class AppConfig:
    data:            DataConfig          = field(default_factory=DataConfig)
    database:        DatabaseConfig      = field(default_factory=DatabaseConfig)
    reporting_lags:  ReportingLagsConfig = field(default_factory=ReportingLagsConfig)
    features:        FeaturesConfig      = field(default_factory=FeaturesConfig)
    targets:         TargetsConfig       = field(default_factory=TargetsConfig)
    scoring:         ScoringConfig       = field(default_factory=ScoringConfig)
    models:          ModelsConfig        = field(default_factory=ModelsConfig)
    backtesting:     BacktestingConfig   = field(default_factory=BacktestingConfig)
    logging:         LoggingConfig       = field(default_factory=LoggingConfig)
    tickers:         TickersConfig       = field(default_factory=TickersConfig)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _deep_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: Path = _CONFIG_PATH,
    tickers_path: Path = _TICKERS_PATH,
) -> AppConfig:
    """
    Load application config from YAML files and environment variables.

    Environment variables take precedence over YAML values:
      DATA_PROVIDER     → cfg.data.provider
      DATABASE_PATH     → cfg.database.path
    """
    raw = _load_yaml(config_path)
    raw_tickers = _load_yaml(tickers_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_raw = raw.get("data", {})
    data = DataConfig(
        provider         = os.getenv("DATA_PROVIDER", data_raw.get("provider", "sample")),
        benchmark_ticker = data_raw.get("benchmark_ticker", "SPY"),
        start_date       = data_raw.get("start_date", "2010-01-01"),
        end_date         = data_raw.get("end_date", None),
    )

    # ── Database ──────────────────────────────────────────────────────────────
    db_raw = raw.get("database", {})
    database = DatabaseConfig(
        type = db_raw.get("type", "sqlite"),
        path = os.getenv("DATABASE_PATH", db_raw.get("path", "data/stock_analysis.db")),
    )

    # ── Reporting lags ────────────────────────────────────────────────────────
    lags_raw = raw.get("reporting_lags", {})
    reporting_lags = ReportingLagsConfig(
        quarterly_earnings_days = lags_raw.get("quarterly_earnings_days", 45),
        annual_report_days      = lags_raw.get("annual_report_days", 90),
    )

    # ── Features ──────────────────────────────────────────────────────────────
    feat_raw = raw.get("features", {})
    features = FeaturesConfig(
        lookback_years           = feat_raw.get("lookback_years", 5),
        momentum_short_months    = feat_raw.get("momentum_short_months", 6),
        momentum_long_months     = feat_raw.get("momentum_long_months", 12),
        volatility_months        = feat_raw.get("volatility_months", 12),
        winsorize_lower          = feat_raw.get("winsorize_lower", 0.01),
        winsorize_upper          = feat_raw.get("winsorize_upper", 0.99),
        min_price_history_months = feat_raw.get("min_price_history_months", 24),
    )

    # ── Targets ───────────────────────────────────────────────────────────────
    tgt_raw = raw.get("targets", {})
    targets = TargetsConfig(horizon_months=tgt_raw.get("horizon_months", 12))

    # ── Scoring ───────────────────────────────────────────────────────────────
    sc_raw = raw.get("scoring", {})
    w_raw  = sc_raw.get("weights", {})
    weights = ScoringWeights(
        probability_of_outperformance = w_raw.get("probability_of_outperformance", 0.35),
        expected_excess_return        = w_raw.get("expected_excess_return", 0.25),
        model_agreement               = w_raw.get("model_agreement", 0.15),
        risk_score                    = w_raw.get("risk_score", 0.10),
        data_quality                  = w_raw.get("data_quality", 0.10),
        valuation                     = w_raw.get("valuation", 0.05),
    )
    thr = sc_raw.get("thresholds", {})
    strong_raw   = thr.get("strong_candidate", {})
    watchlist_raw = thr.get("watchlist_candidate", {})
    scoring = ScoringConfig(
        weights=weights,
        strong_candidate=ClassificationThreshold(
            min_probability     = strong_raw.get("min_probability", 0.60),
            min_excess_return   = strong_raw.get("min_excess_return", 0.05),
            min_model_agreement = strong_raw.get("min_model_agreement", 0.70),
            min_data_quality    = strong_raw.get("min_data_quality", 75.0),
        ),
        watchlist_candidate=ClassificationThreshold(
            min_probability     = watchlist_raw.get("min_probability", 0.50),
            min_excess_return   = watchlist_raw.get("min_excess_return", 0.02),
            min_model_agreement = watchlist_raw.get("min_model_agreement", 0.55),
            min_data_quality    = watchlist_raw.get("min_data_quality", 60.0),
        ),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    mod_raw = raw.get("models", {})
    models = ModelsConfig(
        ridge_alpha        = mod_raw.get("ridge_alpha", 1.0),
        lasso_alpha        = mod_raw.get("lasso_alpha", 0.1),
        vif_threshold      = mod_raw.get("vif_threshold", 10.0),
        min_observations   = mod_raw.get("min_observations", 30),
        significance_level = mod_raw.get("significance_level", 0.05),
    )

    # ── Backtesting ───────────────────────────────────────────────────────────
    bt_raw = raw.get("backtesting", {})
    backtesting = BacktestingConfig(
        min_training_months  = bt_raw.get("min_training_months", 36),
        top_n_stocks         = bt_raw.get("top_n_stocks", 10),
        rebalance_frequency  = bt_raw.get("rebalance_frequency", "monthly"),
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level        = log_raw.get("level", "INFO"),
        file         = log_raw.get("file", "logs/stock_analysis.log"),
        max_bytes    = log_raw.get("max_bytes", 5_242_880),
        backup_count = log_raw.get("backup_count", 3),
    )

    # ── Tickers ───────────────────────────────────────────────────────────────
    bm_raw = raw_tickers.get("benchmark", {})
    benchmark = BenchmarkEntry(
        ticker      = bm_raw.get("ticker", "SPY"),
        name        = bm_raw.get("name", "SPDR S&P 500 ETF Trust"),
        description = bm_raw.get("description", ""),
    )
    universe = [
        TickerEntry(
            ticker   = t["ticker"],
            name     = t.get("name", t["ticker"]),
            sector   = t.get("sector", "Unknown"),
            industry = t.get("industry", "Unknown"),
        )
        for t in raw_tickers.get("universe", [])
    ]
    tickers = TickersConfig(benchmark=benchmark, universe=universe)

    return AppConfig(
        data           = data,
        database       = database,
        reporting_lags = reporting_lags,
        features       = features,
        targets        = targets,
        scoring        = scoring,
        models         = models,
        backtesting    = backtesting,
        logging        = logging_cfg,
        tickers        = tickers,
    )
