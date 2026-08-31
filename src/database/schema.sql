-- =============================================================================
-- schema.sql
-- SQLite database schema for the Statistical Stock Analysis platform.
--
-- Design principles:
--   • All dates stored as ISO-8601 TEXT ("YYYY-MM-DD") for portability.
--   • JSON columns (raw_json, features_json, metrics_json, …) store flexible
--     payloads that would otherwise require many nullable columns.
--   • UNIQUE constraints on business keys prevent duplicate ingestion runs.
--   • All tables have a created_at timestamp for audit purposes.
--   • Foreign keys are declared but SQLite requires PRAGMA foreign_keys = ON
--     at connection time to enforce them.
-- =============================================================================

PRAGMA journal_mode = WAL;       -- Write-Ahead Logging: better concurrency
PRAGMA foreign_keys = ON;        -- Enforce FK constraints


-- =============================================================================
-- 1. stocks
--    Master list of every ticker in the analysis universe.
-- =============================================================================
CREATE TABLE IF NOT EXISTS stocks (
    ticker               TEXT NOT NULL PRIMARY KEY,
    company_name         TEXT,
    sector               TEXT,
    industry             TEXT,
    exchange             TEXT,
    currency             TEXT    DEFAULT 'USD',
    is_active            INTEGER DEFAULT 1    CHECK (is_active IN (0, 1)),
    first_available_date TEXT,
    last_available_date  TEXT,
    created_at           TEXT    DEFAULT (datetime('now')),
    updated_at           TEXT    DEFAULT (datetime('now'))
);


-- =============================================================================
-- 2. prices_raw
--    Unmodified price records exactly as received from a data provider.
--    raw_json stores the full API response for debugging and re-processing.
-- =============================================================================
CREATE TABLE IF NOT EXISTS prices_raw (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    provider       TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL,
    adjusted_close REAL,
    volume         REAL,
    raw_json       TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    UNIQUE (provider, ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_raw_ticker_date
    ON prices_raw (ticker, date);


-- =============================================================================
-- 3. prices_clean
--    Validated, split-adjusted monthly prices ready for feature engineering.
--    monthly_return = (adjusted_close / prev_adjusted_close) - 1
-- =============================================================================
CREATE TABLE IF NOT EXISTS prices_clean (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT NOT NULL,
    date              TEXT NOT NULL,
    adjusted_close    REAL NOT NULL CHECK (adjusted_close > 0),
    monthly_return    REAL,
    volume            REAL,
    -- 'ok' | 'split_adjusted' | 'estimated' | 'suspect'
    data_quality_flag TEXT DEFAULT 'ok',
    created_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_clean_ticker_date
    ON prices_clean (ticker, date);

CREATE INDEX IF NOT EXISTS idx_prices_clean_date
    ON prices_clean (date);


-- =============================================================================
-- 4. benchmark_prices
--    Monthly prices and returns for the benchmark index (e.g. SPY).
-- =============================================================================
CREATE TABLE IF NOT EXISTS benchmark_prices (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_ticker TEXT NOT NULL,
    date             TEXT NOT NULL,
    adjusted_close   REAL NOT NULL CHECK (adjusted_close > 0),
    monthly_return   REAL,
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE (benchmark_ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_ticker_date
    ON benchmark_prices (benchmark_ticker, date);


-- =============================================================================
-- 5. fundamentals_raw
--    Unmodified fundamental records from a data provider.
--    fiscal_period: 'Q1', 'Q2', 'Q3', 'Q4', 'FY'
--    fiscal_date:   last day of the reported period
--    report_date:   date the filing was actually published
-- =============================================================================
CREATE TABLE IF NOT EXISTS fundamentals_raw (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    fiscal_period TEXT,
    fiscal_date   TEXT,
    report_date   TEXT,
    raw_json      TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (provider, ticker, fiscal_date, fiscal_period)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_raw_ticker
    ON fundamentals_raw (ticker, fiscal_date);


-- =============================================================================
-- 6. fundamentals_clean
--    Processed fundamental data with derived ratios.
--    All monetary values in the stock's reported currency.
--    Margins are stored as decimals (0.25 = 25%).
-- =============================================================================
CREATE TABLE IF NOT EXISTS fundamentals_clean (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    fiscal_period    TEXT,
    fiscal_date      TEXT NOT NULL,
    -- report_date is the lag-safe date: data available to the model from here
    report_date      TEXT,
    revenue          REAL,
    eps              REAL,
    net_income       REAL,
    equity           REAL,
    total_debt       REAL,
    free_cash_flow   REAL,
    roe              REAL,   -- net_income / equity
    roic             REAL,   -- nopat / invested_capital (if derivable)
    debt_to_equity   REAL,
    gross_margin     REAL,   -- decimal, e.g. 0.42
    operating_margin REAL,   -- decimal
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE (ticker, fiscal_date, fiscal_period)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_clean_ticker_date
    ON fundamentals_clean (ticker, fiscal_date);

CREATE INDEX IF NOT EXISTS idx_fundamentals_clean_report_date
    ON fundamentals_clean (ticker, report_date);


-- =============================================================================
-- 7. monthly_features
--    One row per (feature_date, ticker): the complete feature vector used
--    as model input for that month's prediction.
--
--    IMPORTANT: All features are calculated using only data that would have
--    been available on or before feature_date (look-ahead bias prevention).
--
--    Returns and growth rates stored as decimals (0.15 = 15%).
--    PE ratios stored as raw multiples (e.g. 25.0).
--    Market cap stored in millions of USD.
-- =============================================================================
CREATE TABLE IF NOT EXISTS monthly_features (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_date                 TEXT NOT NULL,
    ticker                       TEXT NOT NULL,

    -- ── Keyes-inspired core variables ────────────────────────────────────────
    five_year_price_gain         REAL,   -- total return over 5 years
    five_year_eps_growth         REAL,   -- annualised EPS CAGR over 5 years
    five_year_revenue_growth     REAL,   -- annualised revenue CAGR over 5 years
    current_pe_ratio             REAL,   -- price / trailing-12m EPS
    pe_vs_historical_median      REAL,   -- current_pe / 5yr_median_pe - 1
    dividend_yield               REAL,   -- annual dividend / price

    -- ── Modern additions ──────────────────────────────────────────────────────
    roe                          REAL,
    roic                         REAL,
    debt_to_equity               REAL,
    free_cash_flow_yield         REAL,   -- FCF per share / price
    six_month_momentum           REAL,   -- 6-month price return
    twelve_month_momentum        REAL,   -- 12-month price return
    volatility_12m               REAL,   -- annualised std dev of monthly returns
    market_cap                   REAL,   -- millions USD
    revenue_growth_acceleration  REAL,   -- recent revenue growth minus prior period
    eps_growth_acceleration      REAL,   -- recent EPS growth minus prior period
    sector_relative_pe           REAL,   -- stock PE / sector median PE - 1
    sector_relative_momentum     REAL,   -- stock 12m return / sector 12m return - 1
    price_to_sales               REAL,
    price_to_book                REAL,
    gross_margin                 REAL,
    operating_margin             REAL,

    -- ── Data quality ──────────────────────────────────────────────────────────
    -- 0–100 score; lower means more variables are missing or stale
    data_quality_score           REAL    CHECK (data_quality_score BETWEEN 0 AND 100),

    created_at                   TEXT    DEFAULT (datetime('now')),
    UNIQUE (feature_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_monthly_features_date
    ON monthly_features (feature_date);

CREATE INDEX IF NOT EXISTS idx_monthly_features_ticker
    ON monthly_features (ticker, feature_date);


-- =============================================================================
-- 8. targets
--    The outcome variables calculated after the fact (12 months later).
--    Populated once the 12-month horizon has elapsed.
--    winner = 1 if stock outperformed the benchmark over the next 12 months.
-- =============================================================================
CREATE TABLE IF NOT EXISTS targets (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_date                 TEXT NOT NULL,
    ticker                       TEXT NOT NULL,
    future_12m_stock_return      REAL,
    future_12m_benchmark_return  REAL,
    future_12m_excess_return     REAL,  -- stock_return - benchmark_return
    winner                       INTEGER CHECK (winner IN (0, 1, NULL)),
    created_at                   TEXT DEFAULT (datetime('now')),
    UNIQUE (feature_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_targets_date
    ON targets (feature_date);

CREATE INDEX IF NOT EXISTS idx_targets_ticker
    ON targets (ticker, feature_date);


-- =============================================================================
-- 9. model_training_runs
--    One row per trained model version. features_used and metrics_json are
--    stored as JSON so the schema does not need to change when we add models.
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_training_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version       TEXT NOT NULL UNIQUE,  -- e.g. "v20240101_ridge"
    training_start_date TEXT NOT NULL,
    training_end_date   TEXT NOT NULL,
    test_start_date     TEXT,
    test_end_date       TEXT,
    -- 'correlation' | 'simple_ols' | 'multiple_ols' | 'logistic' | 'ridge' | 'lasso'
    model_type          TEXT NOT NULL,
    features_used       TEXT,   -- JSON array: ["five_year_eps_growth", "pe_ratio", ...]
    metrics_json        TEXT,   -- JSON dict: {"r2": 0.12, "auc": 0.61, ...}
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_runs_type
    ON model_training_runs (model_type, training_end_date);


-- =============================================================================
-- 10. predictions
--    Individual model-level predictions before scoring.
--    Multiple rows per (prediction_date, ticker) — one per model_version.
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id                   TEXT NOT NULL UNIQUE,  -- UUID
    model_version                   TEXT NOT NULL,
    prediction_date                 TEXT NOT NULL,
    ticker                          TEXT NOT NULL,
    predicted_12m_return            REAL,
    predicted_12m_excess_return     REAL,
    probability_of_outperformance   REAL CHECK (
        probability_of_outperformance IS NULL
        OR probability_of_outperformance BETWEEN 0.0 AND 1.0
    ),
    created_at                      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (model_version) REFERENCES model_training_runs (model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date_ticker
    ON predictions (prediction_date, ticker);

CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON predictions (model_version);


-- =============================================================================
-- 11. prediction_snapshots
--    The most important table in the system.
--    One row per (prediction_date, ticker): a complete, frozen record of
--    everything the model knew and predicted at that exact point in time.
--
--    This is the gold standard for honest out-of-sample evaluation.
--    Once written, a snapshot must never be modified.
-- =============================================================================
CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id                     TEXT NOT NULL UNIQUE,  -- UUID
    prediction_date                 TEXT NOT NULL,
    ticker                          TEXT NOT NULL,
    company_name                    TEXT,
    sector                          TEXT,
    industry                        TEXT,
    model_version                   TEXT NOT NULL,
    data_provider                   TEXT,
    data_update_timestamp           TEXT,

    -- Complete feature vector at prediction time (raw values)
    features_json                   TEXT,
    -- Standardised (z-score) feature values used by models
    features_std_json               TEXT,

    -- Model output
    predicted_12m_return            REAL,
    predicted_12m_excess_return     REAL,
    probability_of_outperformance   REAL CHECK (
        probability_of_outperformance IS NULL
        OR probability_of_outperformance BETWEEN 0.0 AND 1.0
    ),

    -- Component scores (all 0–100 scale)
    model_agreement_score           REAL CHECK (model_agreement_score IS NULL OR model_agreement_score BETWEEN 0 AND 100),
    risk_score                      REAL CHECK (risk_score IS NULL OR risk_score BETWEEN 0 AND 100),
    valuation_score                 REAL CHECK (valuation_score IS NULL OR valuation_score BETWEEN 0 AND 100),
    momentum_score                  REAL CHECK (momentum_score IS NULL OR momentum_score BETWEEN 0 AND 100),
    fundamental_score               REAL CHECK (fundamental_score IS NULL OR fundamental_score BETWEEN 0 AND 100),
    data_quality_score              REAL CHECK (data_quality_score IS NULL OR data_quality_score BETWEEN 0 AND 100),
    final_score                     REAL CHECK (final_score IS NULL OR final_score BETWEEN 0 AND 100),

    -- Classification
    -- 'Strong candidate' | 'Watchlist candidate' | 'Neutral' | 'Weak / avoid'
    candidate_classification        TEXT,
    -- 1 if ALL major models agree on positive excess return
    keyes_agreement_flag            INTEGER DEFAULT 0 CHECK (keyes_agreement_flag IN (0, 1)),

    warnings_json                   TEXT,   -- JSON array of warning strings
    notes                           TEXT,
    created_at                      TEXT DEFAULT (datetime('now')),
    UNIQUE (prediction_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date
    ON prediction_snapshots (prediction_date);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
    ON prediction_snapshots (ticker, prediction_date);

CREATE INDEX IF NOT EXISTS idx_snapshots_classification
    ON prediction_snapshots (candidate_classification, prediction_date);


-- =============================================================================
-- 12. realized_performance
--    Filled in 12 months after each prediction snapshot.
--    Answers the question: "Did the model actually work?"
--    prediction_correct = 1 if winner_actual == winner_predicted.
-- =============================================================================
CREATE TABLE IF NOT EXISTS realized_performance (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date                 TEXT NOT NULL,
    evaluation_date                 TEXT NOT NULL,  -- prediction_date + 12 months
    ticker                          TEXT NOT NULL,
    realized_12m_stock_return       REAL,
    realized_12m_benchmark_return   REAL,
    realized_12m_excess_return      REAL,
    winner_actual                   INTEGER CHECK (winner_actual IN (0, 1, NULL)),
    winner_predicted                INTEGER CHECK (winner_predicted IN (0, 1, NULL)),
    prediction_correct              INTEGER CHECK (prediction_correct IN (0, 1, NULL)),
    return_error                    REAL,   -- predicted_excess_return - realized_excess_return
    -- Probability bucket at time of prediction, e.g. "0.60-0.70"
    probability_bucket              TEXT,
    model_version                   TEXT,
    created_at                      TEXT DEFAULT (datetime('now')),
    UNIQUE (prediction_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_realized_eval_date
    ON realized_performance (evaluation_date);

CREATE INDEX IF NOT EXISTS idx_realized_ticker
    ON realized_performance (ticker, prediction_date);

CREATE INDEX IF NOT EXISTS idx_realized_model
    ON realized_performance (model_version);


-- =============================================================================
-- 13. backtest_results
--    One row per backtest run. metrics_json stores the full breakdown
--    (by sector, by classification, by probability bucket, monthly series).
-- =============================================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id      TEXT NOT NULL UNIQUE,
    model_version    TEXT,
    start_date       TEXT NOT NULL,
    end_date         TEXT NOT NULL,
    strategy_name    TEXT,   -- e.g. "top_10_by_final_score"
    benchmark_return REAL,
    strategy_return  REAL,
    excess_return    REAL,
    hit_rate         REAL,   -- fraction of periods strategy beat benchmark
    drawdown         REAL,   -- maximum drawdown (negative decimal)
    volatility       REAL,   -- annualised volatility
    sharpe_ratio     REAL,
    sortino_ratio    REAL,
    metrics_json     TEXT,   -- full JSON breakdown
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_backtest_model
    ON backtest_results (model_version, start_date);


-- =============================================================================
-- 14. data_quality_logs
--    One row per data quality event. severity levels:
--    'info'     – informational only
--    'warning'  – data present but suspect (e.g. stale fundamentals)
--    'error'    – data missing or invalid (excluded from features)
--    'critical' – API failure or whole ticker excluded
-- =============================================================================
CREATE TABLE IF NOT EXISTS data_quality_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id     TEXT NOT NULL UNIQUE,
    date       TEXT,
    ticker     TEXT,
    provider   TEXT,
    -- 'missing_price' | 'missing_fundamental' | 'stale_fundamental' |
    -- 'negative_pe' | 'extreme_pe' | 'negative_equity' | 'api_error' |
    -- 'rate_limit' | 'delisted' | 'split_detected' | 'duplicate' | 'other'
    issue_type TEXT NOT NULL,
    severity   TEXT DEFAULT 'warning'
        CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message    TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dq_logs_ticker_date
    ON data_quality_logs (ticker, date);

CREATE INDEX IF NOT EXISTS idx_dq_logs_severity
    ON data_quality_logs (severity, created_at);

CREATE INDEX IF NOT EXISTS idx_dq_logs_issue_type
    ON data_quality_logs (issue_type);


-- =============================================================================
-- 15. provider_availability
--    Records which data fields each provider can supply.
--    The availability checker writes here after probing each provider.
--    availability_status:
--      'available'                – field present and reliable
--      'partial'                  – field present for some tickers/dates only
--      'missing provider required' – field not available from this provider
--      'untested'                 – not yet checked
-- =============================================================================
CREATE TABLE IF NOT EXISTS provider_availability (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    provider             TEXT NOT NULL,
    variable_name        TEXT NOT NULL,
    availability_status  TEXT NOT NULL DEFAULT 'untested',
    notes                TEXT,
    last_checked_at      TEXT DEFAULT (datetime('now')),
    UNIQUE (provider, variable_name)
);
