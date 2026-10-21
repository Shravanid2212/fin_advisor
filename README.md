# Financial Market Analytics & Investment Decision Support System

> **Educational Decision-Support Tool**  
> This application is built strictly for educational and analytical purposes. It is **not** a trading terminal and **not** SEBI-registered investment advice. No recommendations constitute financial advice or direct buy/sell instructions.

---

## 1. Overview
A financial analytics platform analyzing 10 years of Indian equity market data (NSE). Based on an investor's risk profile, preferences, and existing holdings, it explains market risk and return behavior and suggests diversified stock ideas with transparent, explainable reasoning.

### Key Differentiators
- **Explainable Analytics**: Every recommendation and metric traces back to underlying statistics that can be inspected.
- **Strict Outlier Preservation**: Genuine market crash days (e.g. March 2020) are preserved with is_extreme flags rather than arbitrarily trimmed.
- **Offline Reliability**: The application operates read-only against precomputed local databases and does not rely on fragile live API calls during user sessions.

---

## 2. Architecture & Technology Stack
- **Language**: Python 3.11+
- **Data & Computation**: pandas, 
umpy, scipy, statsmodels, rch, scikit-learn
- **Visualization & UI**: plotly, streamlit
- **Storage**: duckdb (or SQLite) and Parquet
- **Configuration & Quality**: pyyaml, pytest, lack, 
uff
- **Market Data Ingestion**: yfinance with NSE bhavcopy fallbacks

```
data/
  ├── raw/         # Immutable ingested snapshots
  ├── clean/       # Validated clean tables
  ├── features/    # Precomputed analytics & metrics
  ├── reference/   # NSE index constituents & holiday calendars
  └── manifests/   # Data pipeline run manifests
src/
  ├── ingest/      # Data collection scripts
  ├── clean/       # Validation and cleaning pipeline
  ├── features/    # Feature engineering (Sharpe, CAGR, Drawdown, etc.)
  ├── stats/       # Statistical tests (normality, ANOVA, GARCH)
  ├── profile/     # Investor risk questionnaire scoring
  ├── portfolio/   # Concentration, volatility, diversification analysis
  └── recommend/   # Explainable rule-based stock idea generator
notebooks/         # 01_problem through 09_ethics (CRISP-DM lifecycle)
app/               # Streamlit application UI & components
tests/             # Automated test suite
config.yaml        # Centralized system configuration
```

---

## 3. Quickstart

### 3.1 Setup Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3.2 Run Test Suite
```bash
pytest
```

### 3.3 Launch Streamlit Application
```bash
streamlit run app/Home.py
```

---

## 4. Stock Universe & AMFI Market-Cap Categorization

The stock universe is defined in `data/reference/universe.csv` (80 stocks across 10 sectors).

### Market Capitalization Classification (AMFI Semi-Annual Cycle)
- Stock market capitalization tiers (`large`, `mid`, `small`) are sourced strictly from the official **AMFI Average Market Capitalization List** dated **2026-06-30** (`data/reference/amfi_market_cap_2026-06-30.csv`).
- **Semi-Annual Refresh Cycle**: In accordance with SEBI circular guidelines, AMFI releases updated average market-cap categorizations semi-annually:
  - **July Release**: Evaluates average market cap for January 1 to June 30 (effective August).
  - **January Release**: Evaluates average market cap for July 1 to December 31 (effective February).
- **Next Refresh**: The next release after 30 Jun 2026 covers 31 Dec 2026 and must be refreshed in January 2027 to update `data/reference/universe.csv` with the latest official cap classifications.
