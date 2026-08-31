# Statistical Stock Analysis Dashboard

A modern replication of Keyes (1972): *"Evaluation and Prediction of Common Stock Prices:
A Statistical Study Using Correlation and Regression Analysis."*

---

## Disclaimer

> **This is a statistical research tool — not financial advice.**
> All outputs are probabilistic estimates based on historical patterns in data.
> Past statistical relationships do not guarantee future performance.
> Do not make investment decisions based solely on this software.
> Scores and rankings are for research purposes only.

---

## What This Is

This platform asks the same question Keyes asked in 1972:
*"Can publicly available company data predict which stocks will outperform the market?"*

It rebuilds that experiment with 50 years of methodological improvements:

| Keyes (1972) | This System |
|---|---|
| ~30 stocks, one time period | 30 stocks MVP, monthly updates, scalable |
| Absolute price movement | **Benchmark-relative** excess return |
| Correlation + simple regression | + Multiple regression, VIF, Logistic, Ridge, Lasso |
| No out-of-sample test | **Walk-forward validation** enforced |
| No prediction record | **Immutable prediction snapshots** |
| Static study | Live dashboard with 7 interactive pages |

The core definition of a "winner": a stock that **beats the S&P 500** over the next 12 months.

---

## Quick Start

```powershell
# 1. Clone and navigate to the project
cd stock-dashboard

# 2. Create virtual environment (Python 3.13 required)
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (default is sample data mode — no API key required)

# 5. Run the monthly update (builds the database)
python run_monthly_update.py

# 6. Launch the dashboard
streamlit run app/dashboard.py
```

Dashboard opens at **http://localhost:8501**

---

## Setup Instructions

### Requirements

- Python 3.13+
- Windows / macOS / Linux
- ~500 MB disk space (sample data + database)

### Virtual Environment

**Windows (PowerShell):**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env` and set your values:

```bash
# Sample data mode — no API key required (default)
DATA_PROVIDER=sample

# Nasdaq Data Link mode — requires a free API key
# DATA_PROVIDER=nasdaq_data_link
# NASDAQ_DATA_LINK_API_KEY=your_key_here
```

Get a free Nasdaq Data Link API key at: **https://data.nasdaq.com/sign-up**

---

## Running the System

### Monthly Update (builds / updates everything)

```bash
python run_monthly_update.py
```

**Options:**
```bash
python run_monthly_update.py --date 2024-01-31          # specific date
python run_monthly_update.py --provider nasdaq_data_link # use real data
python run_monthly_update.py --force-retrain             # retrain models
python run_monthly_update.py --top-n 15                  # 15-stock portfolio
python run_monthly_update.py --quiet                     # suppress console output
```

The update runs 10 steps automatically:
1. Setup — config, database initialisation
2. Data ingestion — prices, fundamentals, benchmark
3. Determine prediction date
4. Feature engineering — compute 33 signals
5. Target backfill — calculate forward returns
6. Model training — logistic, Ridge, Lasso
7. Score stocks — classify all 30 stocks
8. Save prediction snapshot — immutable record
9. Update realised performance — compare old predictions to actuals
10. Update backtest — portfolio simulation

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

**Restart the server when code changes affect `src/` modules:**
```bash
# PowerShell
Stop-Process -Name "streamlit" -Force
streamlit run app/dashboard.py
```

### Run Tests

```bash
pytest                         # run all 61 tests
pytest tests/ -v               # verbose output
pytest tests/ --cov=src        # with coverage report
```

---

## Dashboard Pages

### 1. Overview (`/Overview`)

The main ranking page. Shows:
- Top 20 stocks ranked by **final_score** (0–100)
- Score component breakdown chart (what drives each stock's score)
- Classification distribution (Strong / Watchlist / Neutral / Weak)
- Sector average scores

Auto-saves a prediction snapshot every time it loads.

### 2. Stock Screener (`/Stock_Screener`)

Filterable table of all stocks. Sidebar controls:
- Sector, classification, min score
- Min P(outperform), max P/E, min data quality

### 3. Stock Detail (`/Stock_Detail`)

Deep-dive for one stock:
- Price chart vs S&P 500 (normalised to 100)
- Score history over time
- All 33 feature values at latest prediction
- Full prediction history
- Realised performance (actual outcomes vs predictions)

### 4. Model Diagnostics (`/Model_Diagnostics`)

Five tabs covering all statistical models:
- **Correlation Analysis** — IC table, BH correction, rolling IC, quintile analysis
- **Simple Regression** — per-feature OLS, R², residuals, rolling β stability
- **Multiple Regression** — VIF table, coefficient table, actual vs predicted, rolling R²
- **Logistic Regression** — ROC curve, confusion matrix, calibration, odds ratios, rolling AUC
- **Ridge & Lasso** — coefficient comparison, regularisation path, alpha selection

Results are cached in the database so page loads are fast after the first visit.

### 5. Backtesting (`/Backtesting`)

Portfolio simulation vs benchmark. Configure via sidebar:
- Top-N stocks to hold
- Minimum training months
- Minimum data quality threshold

Shows: cumulative return chart, monthly excess return bars, rolling Sharpe and hit rate,
sector performance breakdown, full metrics table.

### 6. Prediction Archive (`/Prediction_Archive`)

Every historical prediction with its frozen feature values.
Once the 12-month horizon elapses, actual outcomes are shown alongside predictions.

Realised performance section:
- Hit rate by classification
- Calibration by probability bucket
- Predicted vs actual excess return scatter
- Rolling hit rate over time

### 7. Data Quality (`/Data_Quality`)

- DQ score distribution per stock
- Feature availability heatmap (33 features × 30 stocks)
- Missing feature counts
- Data quality log
- Provider availability table (what data each provider can supply)

---

## How the Scoring Model Works

Each stock receives a **final_score** (0–100) each month, built from 6 components:

| Component | Weight | Source |
|---|---|---|
| Probability of outperforming benchmark | 35% | Logistic regression |
| Expected excess return (rank-normalised) | 25% | Ridge regression |
| Model agreement (% of models pointing positively) | 15% | All models |
| Risk score (inverse of volatility + leverage) | 10% | Features |
| Data quality score | 10% | Feature engineering |
| Valuation (sector-relative cheapness) | 5% | Features |

Weights are configurable in `config/config.yaml`.

### Candidate Classifications

| Classification | Criteria |
|---|---|
| **Strong candidate** | P(Win)≥60%, excess return≥5pp, agreement≥70, DQ≥75 |
| **Watchlist candidate** | P(Win)≥50%, excess return≥2pp, agreement≥55, DQ≥60 |
| **Neutral** | Neither Strong nor Watchlist |
| **Weak / avoid** | P(Win)<40%, DQ<40, or predicted excess return<−10% |

### Keyes-Style Agreement Flag

A stock receives this flag when **all** major models simultaneously agree:
- Logistic: P(Win) ≥ 50%
- Ridge: predicted excess return ≥ 0
- Model agreement score ≥ 60
- Data quality ≥ 60

This modernises Keyes's original principle: select only when multiple
independent statistical methods agree.

---

## How to Interpret Model Diagnostics

### Information Coefficient (IC)

Monthly Spearman rank correlation between feature values and future excess returns.

| IC Range | Interpretation |
|---|---|
| > 0.10 | Strong signal |
| 0.05–0.10 | Useful signal |
| 0.02–0.05 | Weak — marginal value |
| < 0.02 | Likely noise |

### Multiple Testing Correction

When testing 33 features simultaneously at α = 0.05, we expect ~1–2 false
positives by chance. The **Benjamini-Hochberg FDR correction** is applied
to all features simultaneously. Only BH-significant features should be
considered reliably predictive.

### VIF (Variance Inflation Factor)

| VIF | Interpretation |
|---|---|
| < 5 | Acceptable |
| 5–10 | Moderate concern |
| > 10 | High — coefficient unreliable |

The system automatically removes the highest-VIF feature iteratively until
all remaining features are below the threshold (default: 10).

### Backtesting Metrics

| Metric | What it measures |
|---|---|
| Hit rate | % of months portfolio beat benchmark (random = 50%) |
| Sharpe ratio | Return per unit of volatility (> 0.5 = reasonable) |
| Sortino ratio | Sharpe but only penalises downside moves (higher = better) |
| Max drawdown | Largest peak-to-trough loss |
| Information ratio | Consistency of outperformance vs tracking error |

---

## Connecting Nasdaq Data Link

1. Get a free API key: **https://data.nasdaq.com/sign-up**
2. Add to `.env`:
   ```
   DATA_PROVIDER=nasdaq_data_link
   NASDAQ_DATA_LINK_API_KEY=your_key_here
   ```
3. Run the update:
   ```bash
   python run_monthly_update.py
   ```

**What's available for free:**
- S&P 500 benchmark prices (FRED/SP500)
- FRED macro series (interest rates, oil prices, etc.)
- Limited EOD price history for some tickers

**What requires a premium subscription:**
- Full adjusted price history (QUOTEMEDIA/PRICES)
- Fundamental data — EPS, revenue, margins (SHARADAR/SF1)
- Analyst estimates, earnings surprises

When premium data is unavailable, the system falls back to synthetic sample data
for fundamentals and logs a clear warning. All fallbacks are transparent.

Run the availability checker:
```bash
python -c "
from src.ingestion.availability_checker import check_availability, print_availability_report
report = check_availability()
print_availability_report(report)
"
```

---

## The Statistical Foundation

### Why benchmark-relative returns?

In a bull market, most stocks rise. A model that says "everything will go up"
looks accurate but adds no value. We define winners as stocks that **beat the S&P 500**
— this is the only meaningful test.

### Why walk-forward validation?

Random train/test splits allow future data to leak into the training set via
cross-sectional correlations. Walk-forward validation trains on all data up to
month T and tests on month T+1 — exactly how the model would be used in practice.

### Why prediction snapshots?

Backtests are computed after the fact and can be unconsciously optimised.
Prediction snapshots record exactly what the model predicted *before* outcomes
are known. When 12 months elapse, the actual outcome is recorded and compared.
This is the only honest test of whether the model actually works.

### Why regularisation (Ridge / Lasso)?

With 30 stocks and 33 features, standard OLS overfits badly:
- OLS in-sample R² = 0.15, out-of-sample R² = −0.45
- Ridge (α=100): hit rate = 55.7% on test data (meaningful improvement)
- Lasso automatically zeros out weak signals

This confirms the model needs regularisation to generalise. The feature-based
scoring (without model predictions) provides the most stable ranking signal.

---

## Project Architecture

```
stock-dashboard/
├── app/                       Streamlit dashboard
│   ├── dashboard.py           Entry point (home page)
│   └── pages/                 7 dashboard pages
├── config/
│   ├── config.yaml            All tunable settings
│   └── tickers.yaml           30-stock MVP universe
├── data/
│   ├── raw/                   API response cache
│   ├── sample/                Synthetic data cache (CSV)
│   └── stock_analysis.db      SQLite database (15 tables)
├── src/
│   ├── ingestion/             Data fetching (sample + Nasdaq)
│   ├── features/              Feature engineering (33 signals)
│   ├── models/                Correlation, OLS, logistic, Ridge, Lasso
│   ├── scoring/               Scoring model + classification
│   ├── backtesting/           Portfolio simulation + metrics
│   ├── database/              SQLite layer (schema, queries, migrations)
│   ├── workflows/             Monthly update orchestration
│   └── utils/                 Config, logging, date math
├── tests/                     61 pytest tests
├── run_monthly_update.py      CLI entry point
└── requirements.txt
```

---

## Known Limitations

| Limitation | Impact |
|---|---|
| **30 stocks only** | Small sample — statistical power is limited |
| **Survivorship bias** | Universe = currently-trading stocks only |
| **Synthetic data** | Results on sample data are not real-world performance |
| **No transaction costs** | Real returns will be lower after trading costs |
| **OLS overfits** | Multiple regression test R² is negative (expected with 30 stocks) |
| **Lasso selects 0 features** | Signal too weak relative to noise at this sample size |
| **Non-stationarity** | Relationships may change over time |
| **Look-ahead in VIF** | Feature selection used full sample — minor bias |

The negative out-of-sample R² and Lasso zeroing everything are honest findings,
not bugs. They confirm that 30 stocks × monthly data is too small for reliable
multi-factor regression generalisation. The feature-based scoring and single-signal
correlations show more promise (hit rate ~55–65% on synthetic data).

---

## Future Improvements

After the MVP is validated on real data:

- Expand to full S&P 500 universe
- Add delisted stocks to reduce survivorship bias
- Add macroeconomic features (VIX, yield curve, credit spreads)
- Add analyst revision data (via premium provider)
- Add transaction cost modelling
- Add sector-neutral ranking
- Add earnings surprise signals
- Only add ML (XGBoost, Random Forest) *after* the statistical baseline is proven

---

## Commands Reference

| Command | What it does |
|---|---|
| `python run_monthly_update.py` | Run complete monthly pipeline |
| `streamlit run app/dashboard.py` | Launch the dashboard |
| `pytest` | Run all 61 tests |
| `pytest --cov=src` | Tests with coverage report |
| `python run_monthly_update.py --force-retrain` | Retrain models |
| `python run_monthly_update.py --provider nasdaq_data_link` | Use real data |
