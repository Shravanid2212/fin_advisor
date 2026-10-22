# Financial Market Analytics & Investment Decision Support System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-teal.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.2+-FFF000.svg)](https://duckdb.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **Educational Decision-Support Tool**  
> This software is an educational decision-support and market analytics tool for Indian equities (NSE). It is **not** a trading terminal and does **not** provide SEBI-registered investment advice. No direct buy or sell terminology is used anywhere in the application or documentation.

---

## 1. Executive Overview

This platform provides structured quantitative decision support across historical Indian equity market data (NSE). It combines statistical modeling, investor risk profiling, portfolio diagnostic analytics, and transparent rule-based idea generation into an interactive web application.

### Core Architectural Principles
1. **Explainable Quantitative Analytics**: Every metric, recommendation score, and portfolio insight decomposes into visible mathematical formulas and empirical inputs. No black-box machine learning models are used for stock ranking.
2. **Strict Tail-Risk & Outlier Preservation (AGENTS.md Hard Rule 1)**: Genuine market crash days (such as March 2020 during the global COVID-19 pandemic) are **never trimmed or dropped**. They are preserved and tagged with `is_extreme = True`. As demonstrated in our statistical audits, removing extreme days via standard statistical heuristics (e.g., 1.5× IQR) underestimates annualized market volatility by **26.3%**, creating dangerous illusions of safety.
3. **No Min-Max Scaling (Hard Rule 2)**: Tickers are never compared using global min-max scaling, which distorts cross-sectional volatility. All growth comparisons utilize per-ticker rebased series ($I_0 = 100$ at start) or cumulative simple returns.
4. **Offline Resilience & Read-Only Market Database (Hard Rule 4)**: The Streamlit application operates strictly read-only against precomputed local analytical databases (`data/market.duckdb`). It never triggers live yfinance network calls during user interaction, guaranteeing instantaneous response times and resilience to external API downtime.
5. **Secure Local Authentication & User Privacy (Hard Rule 6 & Indian DPDP Act 2023)**: User account credentials, risk questionnaire responses, and portfolio holdings reside in a dedicated local SQLite database (`data/users.db`), completely separated from the read-only market database. Passwords are salted with random 16-byte salts and hashed using PBKDF2-HMAC-SHA256 (200,000 iterations). Plain text passwords are never stored or logged. Users can save, reset their portfolio, or permanently delete all their account data at any time.

---

## 2. System Architecture & Layout

The system is organized following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

```
dsa_cp/
├── .github/
│   └── workflows/
│       └── refresh.yml            # Automated weekly data refresh pipeline
├── app/                           # Streamlit Web Application
│   ├── Home.py                    # Landing page with Authentication, KPIs & navigation
│   ├── theme.py                   # Design system tokens (teal palette, light/dark)
│   ├── components/                # Modular UI widgets (KPI card, risk gauge, banners)
│   └── pages/                     # Multipage application views
│       ├── 1_Explore_Market.py    # Universe overview, sector performance, leaderboards
│       ├── 2_Stock_Detail.py      # Historical analytics, drawdowns, peer correlations
│       ├── 3_Compare_and_Screen.py# Multi-criteria screening & side-by-side comparison
│       ├── 4_My_Profile.py        # 14-question risk engine (capacity vs tolerance)
│       ├── 5_My_Portfolio.py      # Holdings concentration, drift & portfolio risk
│       └── 6_Recommendations.py   # Rule-based stock ideas with transparent scoring
├── data/
│   ├── raw/                       # Immutable raw price and fundamental CSV extracts
│   ├── clean/                     # Cleaned, validated, calendar-aligned Parquet files
│   ├── features/                  # Rolling volatility, metrics, correlation tables
│   ├── reference/                 # Official AMFI market-cap list & NSE holiday calendar
│   ├── manifests/                 # Pipeline execution run manifests and audit logs
│   ├── market.duckdb              # High-performance analytical columnar database
│   └── users.db                   # Secure user account & profile SQLite database
├── docs/
│   └── PRD.md                     # Comprehensive Product Requirements Document
├── notebooks/                     # 9 CRISP-DM Lifecycle Notebooks (01 to 09)
├── reports/                       # Generated audit reports and figures
├── src/                           # Pure Python Analytical Engine
│   ├── accounts/                  # Secure account store, PBKDF2 hashing, lockout
│   ├── app_data.py                # Cached DuckDB queries & data loaders for Streamlit
│   ├── config.py                  # Dataclass configuration loader (config.yaml)
│   ├── ingest/                    # Data collection, benchmarks & data quality checks
│   ├── clean/                     # Validation, calendar alignment, crash preservation
│   ├── features/                  # Return metrics, CAGR, drawdown, VaR calculations
│   ├── stats/                     # Hypothesis testing (Kruskal-Wallis, Levene), GARCH
│   ├── profile/                   # Risk engine scoring & asset allocation mappings
│   ├── portfolio/                 # Concentration (HHI), volatility & portfolio risk
│   └── recommend/                 # 5-factor explainable idea recommendation engine
├── tests/                         # Automated pytest suite
├── config.yaml                    # Single source of truth for thresholds and weights
├── Makefile                       # GNU Make automation runner
├── run.py                         # Cross-platform CLI runner (data, test, app)
├── run.ps1                        # Native Windows PowerShell runner
└── requirements.txt               # Pinned Python package dependencies
```

---

## 3. Quickstart & Installation

### 3.1 Environment Prerequisites
- **Python**: Version **3.11+** (tested and verified on Python 3.13.2).
- **Virtual Environment**: Recommended to avoid dependency conflicts.

### 3.2 Setup Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd dsa_cp

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. Simple Commands

To make reproduction effortless across Windows, macOS, and Linux, the project provides three simple command targets: `data`, `test`, and `app`.

### Using the Universal Python Runner (`run.py`)
```bash
# Build data pipeline (cleaning, features, statistics, DQ checks)
python run.py data

# Run automated test suite
python run.py test

# Launch interactive Streamlit web dashboard
python run.py app
```

### Using Makefile (`make`)
```bash
make data   # Rebuild DuckDB analytical database from raw data
make test   # Run full pytest suite with coverage verification
make app    # Launch Streamlit web application on port 8501
```

### Using Windows PowerShell (`run.ps1`)
```powershell
.\run.ps1 data
.\run.ps1 test
.\run.ps1 app
```

### Exact Python Module Commands
If you prefer running modules directly without wrappers:
```bash
# 1. Clean raw market data and construct synthetic benchmarks:
python -m src.clean.clean_market_data

# 2. Precompute stock features and metrics:
python -m src.features.build_features

# 3. Compute statistical hypothesis tests and GARCH volatility models:
python -m src.stats.build_stats_tables

# 4. Verify database data quality and primary key integrity:
python -m src.ingest.dq_checks

# 5. Launch the Streamlit application:
streamlit run app/Home.py
```

---

## 5. Clean Clone Reproduction Guide

To build the entire system from a fresh clone without relying on pre-existing artifacts:

1. **Clone and Setup Dependencies**:
   ```bash
   git clone <repo-url>
   cd dsa_cp
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Initialize Analytical Database**:
   The raw market data snapshots are stored in `data/raw/`. Running `python run.py data` reads `data/raw/`, cleans price series, detects holidays, builds synthetic midcap and smallcap benchmarks, computes all features and rolling metrics, and populates `data/market.duckdb`:
   ```bash
   python run.py data
   ```

3. **Verify System Integrity**:
   Run the comprehensive test suite to confirm that all tests pass:
   ```bash
   python run.py test
   ```

4. **Launch Application**:
   ```bash
   python run.py app
   ```
   Open your browser to `http://localhost:8501`. Register an account or sign in to begin.

---

## 6. System Configuration (`config.yaml`)

In accordance with **Hard Rule 5**, there are **no magic numbers in code**. All analytical parameters, thresholds, and weights reside in `config.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `risk_free_rate` | `0.065` (6.5%) | Annualized risk-free rate based on the RBI 91-day Treasury Bill rate |
| `extreme_z_threshold` | `4.0` | Z-score threshold for tagging extreme market shock days (`|z| > 4`) |
| `spike_review_threshold` | `0.20` (20%) | Single-day return magnitude requiring manual audit logging |
| `trading_days_per_year` | `252` | Trading calendar annualization standard for Indian equities |
| `max_trading_days_stale`| `5` | Maximum business days before triggering the stale-data warning banner |
| `random_seed` | `42` | Seed for deterministic, reproducible statistical sampling |
| `recommender.weights` | See below | Transparent weights for the 5-factor stock recommendation engine |

### Recommender Factor Weights
```yaml
recommender:
  weights:
    sharpe: 0.30              # Risk-adjusted historical return
    drawdown: 0.20            # Resilience to maximum drawdown
    diversification: 0.25     # Volatility reduction & correlation benefit to user portfolio
    sector_gap: 0.15          # Alignment with target sector allocation
    yield_momentum: 0.10      # Dividend yield (or 12-month momentum for low-dividend stocks)
  max_picks: 8                # Maximum stock suggestions per run
  max_per_sector: 2           # Sector concentration ceiling
  per_stock_cap: 0.20         # 20% maximum suggested weight per stock
```

---

## 7. Data Sources, Governance & Limitations

### 7.1 Stock Universe & Market-Cap Classification
- The analytical universe consists of active stocks across 10 economic sectors selected for historical longevity and liquidity (`data/reference/universe.csv`).
- Market capitalization classifications are updated semi-annually under SEBI guidelines.
- **Look-Ahead Bias Limitation**: Using a static universe classification introduces look-ahead bias when analyzing returns across early periods. Stocks categorized as large-cap today may have been mid-cap or small-cap in earlier cycles.

### 7.2 Synthetic Cap Benchmarks & Survivorship Bias
- While `^NSEI` (NIFTY 50) provides continuous large-cap benchmark coverage, pure historical index tickers for mid-cap and small-cap on free feeds exhibit data gaps or mislabeling.
- To provide robust comparisons, the pipeline constructs daily equal-weighted synthetic benchmarks (`synthetic_midcap` and `synthetic_smallcap`) rebalanced daily and compounded from 100 ($I_0 = 100$).
- **Survivorship Bias Disclaimer**: These synthetic indices are computed from surviving universe constituents that maintained active listings through the period. They do not account for companies delisted or merged during historical periods.

### 7.3 Crash Day Preservation vs. IQR Counterfactual
Standard quantitative toolkits often automatically discard return outliers exceeding 1.5× the Interquartile Range (IQR). In our data cleaning pipeline, we explicitly preserve all genuine trading crashes:
- Out of 210,039 clean daily price rows, an automated 1.5× IQR filter would discard **16,664 trading days (7.9%)**.
- If discarded, the average annualized volatility across universe stocks would collapse from **31.3%** down to **23.1%**.
- This **26.3% volatility understatement** hides the true frequency of extreme tail events and severe drawdowns. Our platform preserves these days, tagging them with `is_extreme = True` so risk metrics accurately reflect real-world tail risks.

### 7.4 Stale Data Detection & Graceful Diagnostics
- The application evaluates data freshness dynamically using trading calendar day counts (`np.busday_count`).
- If the latest date in `prices` lags current calendar dates by more than **5 trading days**, an amber notice banner appears across all pages indicating the data date and staleness status.
- If the database file is missing or uninitialized, pages render helpful diagnostic troubleshooting cards instead of unhandled Python exceptions.

### 7.5 Mutual Fund Dataset & Analytics Scope
Selection rule: at least 3 years of history (5 preferred), no more than 3 funds from one fund house per category, a mix of well-known fund houses.

- **Data Source**: Historical daily Net Asset Value (NAV) data is ingested from the open AMFI-backed MFAPI service (`https://api.mfapi.in`) for 109 curated direct-growth schemes across 9 categories (`data/reference/funds.csv`).
- **Expenses & Holdings Scope**: All NAV series are net of Total Expense Ratio (TER), as reported daily by fund houses. Expense ratio, AUM (fund size), and portfolio holdings breakdown are not included in the dataset or ranking model.
- **Explainable Ranking**: Funds are ranked strictly within their categories using composite z-scores of Sharpe ratio, maximum drawdown, and 3-year rolling return consistency against the benchmark.


---

## 8. Ethics, Privacy & Regulatory Compliance

### 8.1 User Privacy & Secure Local Storage (Indian DPDP Act 2023)
In strict compliance with the **Digital Personal Data Protection (DPDP) Act 2023**:
- **What is Stored**: User account credentials, risk questionnaire responses and assessed profile levels, and stock holdings and fund bucket values are stored in a dedicated local SQLite database (`data/users.db`), completely segregated from the read-only analytical database (`market.duckdb`).
- **Cryptographic Password Hashing**: Passwords are never stored in plain text. Each user is assigned a cryptographically random 16-byte salt (`os.urandom(16)`), and passwords are hashed using `PBKDF2-HMAC-SHA256` with 200,000 iterations. Verification uses constant-time comparison (`hmac.compare_digest`), and accounts enforce a temporary 15-minute lockout after 5 consecutive failed attempts.
- **Data Deletion & Rights**: Users retain full sovereignty over their data. The application provides instant controls to:
  1. *Reset Portfolio*: Clears all saved stock holdings and mutual fund buckets while preserving the risk profile.
  2. *Delete My Account*: Permanently and completely purges all user credentials, profile records, and portfolio holdings from `data/users.db`.
- **Zero Third-Party Telemetry**: No user data, holdings, passwords, or profile calculations are ever sent to external services, remote telemetry, or third-party cloud trackers.

### 8.2 Transparent, Rule-Based Recommendations
- Stock idea generation uses a transparent, deterministic scoring rubric with publicly documented weights.
- It operates strictly within the investor's computed risk tier:
  - **Level 1 (Conservative)**: Suggestions restricted to Bucket 1-2 stocks (low volatility).
  - **Level 2 (Moderately Conservative)**: Bucket 1-2 stocks.
  - **Level 3 (Moderate)**: Bucket 1-3 stocks.
  - **Level 4 (Growth)**: Bucket 1-4 stocks.
  - **Level 5 (Aggressive)**: All buckets eligible.
- Weights are assigned using inverse-volatility sizing capped at 20% per stock, ensuring high-volatility suggestions do not dominate the proposed basket.

### 8.3 Regulatory Notice & Disclaimer
- This project is developed exclusively for academic, research, and educational decision-support purposes.
- It is **not** registered with SEBI under the SEBI (Investment Advisers) Regulations, 2013, or SEBI (Research Analysts) Regulations, 2014.
- Nothing in this software, documentation, or user interface constitutes financial, legal, tax, or investment advice.

---

## 9. Streamlit Community Cloud Deployment

To deploy this repository to **Streamlit Community Cloud**:

1. **Repository Setup**:
   Ensure all source code, `requirements.txt`, `config.yaml`, and `data/market.duckdb` are committed to your GitHub repository.
2. **Configure Streamlit Cloud**:
   - **Repository**: `your-username/dsa_cp`
   - **Branch**: `main`
   - **Main file path**: `app/Home.py`
3. **Precomputed Database Advantage**:
   Because `data/market.duckdb` is precomputed and checked into the repository, Streamlit Cloud instances launch with instantaneous zero-latency cold starts without making external network calls or hitting yfinance rate limits.
4. **Environment Variables**:
   No external API keys or secrets are required. The entire analytical engine runs self-contained.

---

## 10. Test Suite & Code Quality

The codebase enforces strict unit testing, pure functions, and complete type annotations:

```bash
# Run full automated test suite:
pytest -v
```

---

## 11. License & Citation

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
