# Sample Portfolio CSV Data (`sample_data/`)

This directory provides curated CSV test fixtures for validating portfolio uploads, concentration checks, data hygiene handling, and portfolio analytics in accordance with PRD §15.6.

All tracked tickers correspond to genuine constituents defined in `data/reference/universe.csv`.

---

## Files and Expected Test Results

### 1. `bank_heavy.csv`
- **Purpose**: Tests sector concentration warning and sector skew analytics.
- **Holdings**: 6 large-cap financial stocks (HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, BAJFINANCE).
- **Expected Outcome**:
  - Parses with zero errors.
  - Generates a **Sector Concentration Alert** (> 40% threshold) as Banking & Financials represents ~100% of equity value.
  - Herfindahl-Hirschman Index (HHI) concentration warning triggered.

### 2. `balanced_multi_sector.csv`
- **Purpose**: Tests healthy diversification across multiple sectors and cap classes.
- **Holdings**: 10 stocks across 10 distinct sectors (Banking, IT, Pharma, FMCG, Auto, Energy, Metals, Infra, Consumer, Chemicals).
- **Expected Outcome**:
  - Parses with zero validation errors.
  - High diversification score (> 75/100).
  - No single sector exceeds the 40% concentration ceiling.
  - Sector allocation donut displays balanced wedges across all 10 sectors.

### 3. `tech_heavy.csv`
- **Purpose**: Tests IT sector overweighting and correlation clustering.
- **Holdings**: 7 IT services and technology stocks (TCS, INFY, HCLTECH, WIPRO, PERSISTENT, COFORGE, MPHASIS).
- **Expected Outcome**:
  - Parses with zero validation errors.
  - Triggers sector concentration warning for Information Technology (> 40%).
  - Correlation analytics identify high pairwise co-movement within the tech basket.

### 4. `conservative_large_cap.csv`
- **Purpose**: Tests portfolio analytics for defensive, low-volatility blue-chip large caps.
- **Holdings**: 8 low-beta large-cap stocks (HDFCBANK, TCS, HINDUNILVR, ITC, NESTLEIND, POWERGRID, NTPC, TITAN).
- **Expected Outcome**:
  - Parses with zero errors.
  - Annualized volatility is low (~14% to 18%).
  - Maximum drawdown is shallow relative to the broad market index.

### 5. `small_mid_cap_growth.csv`
- **Purpose**: Tests cap-class distribution with a focus on mid and small caps.
- **Holdings**: 9 mid and small-cap stocks (FEDERALBNK, IDFCFIRSTB, PERSISTENT, LUPIN, MARICO, BHARATFORG, CYIENT, BSOFT, EXIDEIND).
- **Expected Outcome**:
  - Parses with zero errors.
  - Cap-class donut confirms 0% large cap, ~65% mid cap, and ~35% small cap.
  - Volatility reflects the higher-beta growth profile.

### 6. `single_stock.csv`
- **Purpose**: Tests edge cases with exactly one equity holding.
- **Holdings**: 50 shares of RELIANCE.NS.
- **Expected Outcome**:
  - Parses cleanly with zero errors.
  - Stock concentration alert triggered (100% in single stock vs 25% config limit).
  - Volatility and risk metrics equal the individual stock's volatility.

### 7. `messy_with_errors.csv`
- **Purpose**: Tests data hygiene, input validation, row-level error reporting, and isolation.
- **Contents**:
  - `UNKNOWNCO.NS`: Non-existent ticker.
  - `ICICIBANK.NS`: Negative quantity (`-10`).
  - `SBIN.NS`: Zero quantity (`0`).
  - `TCS.NS`: Duplicate ticker (two rows: 15 and 10 shares).
  - `INFY.NS`: Missing price (empty price field).
  - `   wipro.ns   `: Lowercase ticker with leading and trailing whitespace.
  - `ZOMATO.NS`: Real listed company not tracked in our 80-stock universe.
- **Expected Outcome**:
  - Generates validation warning banners on the Portfolio page:
    1. Warning: Negative quantity for `ICICIBANK.NS` dropped.
    2. Warning: Zero quantity for `SBIN.NS` dropped.
    3. Notice: Duplicate rows for `TCS.NS` consolidated.
    4. Notice: Missing buy price for `INFY.NS` handled gracefully (defaults to latest market price).
    5. Success: Whitespace and lowercase cleaned for `wipro.ns` -> successfully mapped to `WIPRO.NS`.
    6. Informational: `ZOMATO.NS` and `UNKNOWNCO.NS` marked as unanalysed.
  - Valid rows (`HDFCBANK.NS`, `TCS.NS`, `INFY.NS`, `WIPRO.NS`) are still analyzed without crashing the application.

### 8. `empty_header_only.csv`
- **Purpose**: Tests zero-byte or headers-only CSV edge case.
- **Contents**: Header line `ticker,quantity,avg_buy_price` with no data rows.
- **Expected Outcome**:
  - No crash or traceback.
  - Displays friendly empty state: `"No valid stock holdings found in CSV"`.

### 9. `tickers_without_ns.csv`
- **Purpose**: Tests automatic ticker normalization when users omit the exchange suffix.
- **Contents**: 5 tickers written without `.NS` (`HDFCBANK`, `TCS`, `INFY`, `ITC`, `RELIANCE`).
- **Expected Outcome**:
  - Parses with zero errors.
  - Automatically appends `.NS` and matches all 5 stocks to the tracked universe.
  - Computes complete portfolio analytics seamlessly.
