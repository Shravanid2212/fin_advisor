# PRD: Financial Market Analytics & Investment Decision Support System

*Product Requirements Document, v1.0, 20 Oct 2026. Data science and analytics project following the full CRISP-DM lifecycle.*

## 1. Overview

**Product:** A web application that analyses 10 years of Indian equity market data and, based on an investor's risk profile, preferences and existing holdings, explains the market's risk and return behaviour and suggests diversified stock ideas with reasons.

**Product type:** Educational decision-support tool. It is *not* a trading terminal and *not* SEBI-registered investment advice.

**Differentiator:** Commercial Indian platforms (robo-advisors such as Kuvera, Scripbox, ET Money, INDmoney, Groww) typically work as a black box: a risk questionnaire leads to a suggested allocation, then rebalancing. Ours is **explainable**: every number and recommendation traces back to a statistic the user can inspect.

**Deliverables:** (1) live Streamlit app, (2) data pipeline with daily refresh, (3) one notebook per lifecycle phase, (4) written report, (5) tests, (6) README and deployment.

## 2. Problem Formulation

**Problem statement.** Retail investors can see raw price charts or read opinions, but they lack a structured, statistically sound way to (a) compare companies and sectors on a risk-adjusted basis, (b) quantify risk, and (c) turn that analysis into guidance that fits *their* portfolio and temperament.

**Sub-problems**

1. **Comparison:** like-for-like, risk-adjusted comparison of stocks and sectors.
2. **Risk evaluation:** volatility, drawdown, tail risk and consistency, explained simply.
3. **Decision-support gap:** analysis that ends in "what this means for you".
4. **Personalization:** matching suggestions to risk capacity, risk tolerance, cap-size preference and current holdings.

**Goals**

- G1: Answer the analytical questions in section 3 with correct, reproducible statistics.
- G2: Produce a risk profile from a validated questionnaire and explain it.
- G3: Analyse a user portfolio for concentration, volatility and diversification.
- G4: Generate ranked, explained stock suggestions.
- G5: Ship a working, deployed app, not only a dashboard.

**Non-goals (v1)**

- Real-time or intraday prices, order placement, brokerage integration.
- Mutual-fund-level recommendations (funds are treated as asset-class buckets).
- Price prediction as the basis for recommendations.
- User accounts, stored personal data.

## 3. Analytical Questions (each has a metric and a decision)

| # | Question | Metric / method | Hypothesis or decision |
| --- | --- | --- | --- |
| Q1 | How big is the dataset and what is the typical return? | Counts, mean daily return, CAGR | Context KPIs |
| Q2 | Which sectors grew most over 10 years? | Rebased price (100 at start), CAGR by sector | Rank sectors by growth |
| Q3 | Which sectors give the best risk-adjusted return? | Sharpe, Sortino by sector | Best return per unit of risk |
| Q4 | How has one stock behaved: growth, crash, recovery? | Price, drawdown curve, recovery time | Stock profile |
| Q5 | Is a price move backed by volume? | Volume vs absolute return, volume z-score | Flag low-volume moves |
| Q6 | How risky is a stock day to day? | Return histogram, rolling 90-day volatility, VaR 95% | Risk label |
| Q7 | Which stocks move together? | Correlation matrix, clustering | Avoid redundant holdings |
| Q8 | Is the return worth the risk? | Risk-return scatter, Sharpe, beta | Efficient picks |
| Q9 | Are sectors and cap classes *significantly* different in risk? | Kruskal-Wallis / ANOVA, Levene | H1: large caps have lower volatility than small caps |
| Q10 | How do stocks behave in crashes? | Drawdown in COVID 2020, 2022; downside beta | Defensive vs cyclical |
| Q11 | What is this investor's risk profile? | Questionnaire score | Profile 1 to 5 |
| Q12 | How diversified is their portfolio? | Sector and cap weights, avg pairwise correlation, diversification ratio | Concentration warnings |
| Q13 | Which stocks fit the profile and improve diversification? | Recommender score | Ranked suggestions |

## 4. Benchmark Research: What Existing Platforms Do

| Pattern | Seen in | What we do |
| --- | --- | --- |
| Digital risk questionnaire (goals, horizon, income, tolerance) | Robo-advisors: Scripbox, Kuvera, ET Money | Two-part score: capacity and tolerance |
| Target allocation by profile, rebalancing on drift (e.g. when equity drifts about 5% from target) | Robo-advisors | Target vs current allocation view |
| Portfolio tracking and aggregation | INDmoney, Kuvera | Manual or CSV entry in v1 (automatic import is future work) |
| Screeners and stock pages with ratios and charts | Screener.in, Tickertape | Stock deep dive and compare pages |
| SEBI-style five-level riskometer | Mutual fund factsheets | Our risk gauge |
| SEBI Investment Adviser registration needed for personalised advice | Regulatory | Educational disclaimer, no-advice wording |

*Build note: re-verify these platform features against their current sites while writing the report's literature section.*

## 5. Users

- **Aarav, 24, first-time investor:** wants simple explanations, fears losses.
- **Meera, 38, salaried, holds 12 stocks and 2 funds:** wants to know if she is diversified.
- **Rohit, 55, nearing retirement:** wants low volatility and dividends.
- **Evaluator (faculty or reviewer):** wants to see every lifecycle phase and be able to try the app in two minutes.

## 6. Data Collection

### 6.1 Universe (about 75 stocks)

- **Large cap (about 30):** Nifty 50 members, chosen to cover each sector (reuse the existing 29 stocks).
- **Mid cap (about 25):** from the Nifty Midcap 150.
- **Small cap (about 20):** from the Nifty Smallcap 250.
- **Selection rule:** top names by average daily traded value within each sector and cap class, with at least 8 years of listed history.
- **Benchmarks:** ^NSEI plus a midcap and a smallcap index. Verify the exact Yahoo tickers for the mid/small indices at build time.
- **Cap category:** based on the SEBI/AMFI rank convention (rank 1 to 100 large, 101 to 250 mid, 251+ small), using the official AMFI list dated **2026-06-30** (`data/reference/amfi_market_cap_2026-06-30.csv`). In accordance with SEBI guidelines, AMFI publishes updated categorisations semi-annually (every January and July). The universe cap labels must be refreshed on each release (the next release after 30 Jun 2026 covers 31 Dec 2026). The limitation that this is today's label applied to history is documented in the ethics section.
- **Sector taxonomy (10 sectors):** Banking and Financials, IT, Pharma and Healthcare, FMCG, Auto, Energy and Power, Metals, Capital Goods and Infra, Consumer, Chemicals.

### 6.2 Sources

| Data | Source | Method |
| --- | --- | --- |
| Daily OHLCV, adjusted for splits and dividends | Yahoo Finance via `yfinance` (`.NS` suffix) | `yf.download(..., auto_adjust=True)`, 2016-01-01 to latest |
| Market cap, P/E, dividend yield, sector, industry | `yfinance` `Ticker.info` | Snapshot on each refresh, stored with its date |
| Constituent lists | NSE index factsheets (CSV) | Manual download once, stored in `data/reference/` |
| Trading calendar | NSE holiday list | Used for gap checks |
| Risk-free rate | 10-year G-Sec yield | Configurable default (about 6.5%), cited in the report |
| Fallback prices | NSE bhavcopy archives | Used if Yahoo fails or looks wrong |

**Known caveat:** `yfinance` is an unofficial scraper, quotes can be delayed, and it can break. This is why the app reads from our own database, never live from the API.

### 6.3 Collection process

1. `pull_prices.py` downloads each ticker in batches of 10 with retry and backoff, saving raw files untouched to `data/raw/YYYY-MM-DD/`.
2. `pull_fundamentals.py` stores a dated snapshot of market cap, P/E and dividend yield.
3. Every run writes a manifest (ticker, rows, first and last date, status) to `data/manifests/`.
4. Raw data is immutable. Cleaning writes to a separate layer.

### 6.4 Refresh strategy (answers "prices change every second")

- All metrics use **daily closes**, so tick data adds nothing.
- A scheduled job (GitHub Actions, about 17:00 IST on weekdays, after the 15:30 close) pulls new rows, validates them, appends them to the database and recomputes metrics.
- If the refresh fails, the app keeps serving the last good data and shows a stale-data banner.
- Every page shows **"Data as of DD MMM YYYY, close"**.
- Optional delayed "last price" on stock pages, labelled delayed. No logic depends on it.

## 7. Data Cleaning and Preprocessing

**Layers:** `raw` (as downloaded), `clean` (validated), `features` (metrics). Storage is SQLite (or DuckDB) plus parquet exports.

### 7.1 Validation rules

| Check | Rule | Action |
| --- | --- | --- |
| Schema | Required columns present, correct types, dates parsed as ISO | Fail the run |
| Duplicates | Same ticker and date | Keep last, log count |
| Invalid prices | Open, high, low, close <= 0 or null; high < low; close outside \[low, high\] | Flag, then fix or drop the row with a logged reason |
| Missing days | Compare against the NSE calendar | Forward-fill gaps of 1 to 2 days; flag longer gaps |
| Volume | Negative or null | Fix or flag |
| Return spikes | Absolute daily return > 20% | Check against known corporate actions before deciding |
| Listing history | Fewer than 3 years | Excluded from recommendations, kept for analysis |
| Date nulls | Any null dates | Reject row, log it (the old dataset had 28) |

### 7.2 The extreme-day policy (key change from the old pipeline)

The old pipeline deleted any return outside 1.5 times the IQR per ticker. That wiped out every real crash day, including most of March 2020, and understated all risk metrics. **New rule: genuine market moves are never removed.** Days are flagged (`is_extreme = |z| > 4`) and only confirmed data errors (zero or negative price, bad ticks, unadjusted splits) are corrected or dropped. The cleaning report lists rows removed per reason, and the report discusses the before/after impact on volatility as a documented lesson.

### 7.3 Preprocessing outputs

- Daily simple return and log return per ticker.
- Per-ticker **rebased price** (100 at the start date). Never a global min-max scale across tickers.
- Standardised and normalised columns only where a model needs them, never for comparing sectors.
- Consistent labels ("IT", "FMCG" capitalised correctly).

## 8. Exploratory Data Analysis

Each item becomes a notebook section and a figure in the report:

1. Dataset profile: rows, tickers, date range, missingness heat map.
2. Return distributions: histograms and QQ plots (expect fat tails), skew and kurtosis.
3. Rebased growth curves by sector and cap class.
4. Volatility by sector and cap class, including rolling 90-day volatility regimes.
5. Volume behaviour: volume vs absolute return, spikes.
6. Correlation matrix, hierarchical clustering, and correlation rising in crashes.
7. Crash studies: COVID 2020, 2022, with drawdown depth and recovery time.
8. Fundamentals: P/E and dividend yield vs return and risk.

## 9. Feature Engineering

| Feature | Definition |
| --- | --- |
| Annualised return (CAGR) | (end / start)^(1 / years) minus 1 |
| Annualised volatility | Daily standard deviation times sqrt(252) |
| Sharpe ratio | (Annual return minus risk-free) / annual volatility |
| Sortino ratio | Same, using downside deviation |
| Max drawdown | Largest peak-to-trough fall |
| Drawdown recovery days | Days from trough back to previous peak |
| Beta | Slope of stock returns on ^NSEI returns |
| VaR 95% (historical) | 5th percentile of daily returns |
| Rolling volatility | 30, 90 and 252-day windows |
| Momentum | 6 and 12-month return |
| Liquidity | 60-day average traded value (price times volume) |
| Volume z-score | Volume relative to its 60-day mean and standard deviation |
| Risk bucket | Quintiles of annual volatility, mapped to labels 1 to 5 (Very Low to Very High) |
| Date features | Year, quarter, month, day of week |

## 10. Statistical Analysis

| Test | Purpose |
| --- | --- |
| Jarque-Bera, Shapiro-Wilk | Are returns normal? (Expect no, which justifies historical VaR) |
| Kruskal-Wallis (and ANOVA as a check) | Do sectors and cap classes differ in return or volatility? |
| Levene test | Do variances differ between groups? |
| Welch t-test and Mann-Whitney U | Large vs small cap pairwise comparisons, with multiple-testing correction |
| Bootstrap confidence intervals | Uncertainty on Sharpe and mean return |
| Pearson and Spearman correlation | Co-movement, with stability across sub-periods |
| Markowitz optimisation | Efficient frontier for a chosen basket |
| GARCH(1,1) volatility forecast | 5-day-ahead volatility, backtested against a naive rolling estimate |
| ARIMA on returns | Shown to illustrate low predictability, with honest error bands |

Every test result is stored as a table and shown in plain English ("Small caps are significantly more volatile than large caps, p < 0.001").

## 11. Risk Questionnaire

Two separate scores are computed, and the final profile is the **lower** of the two. A nervous person with a high income is not pushed into aggressive picks.

**Section A: Risk capacity (score each 1 to 4)**

1. Age: under 25 / 25-35 / 36-50 / 51+ (younger scores higher)
2. Income stability: fixed salary / variable / business / retired
3. Dependents: none / 1 / 2-3 / 4+
4. Emergency fund: under 3 months / 3-6 / 6-12 / 12+
5. Income spent on EMIs: over 50% / 30-50% / 10-30% / under 10%
6. Investment horizon: under 1 yr / 1-3 / 3-7 / 7+ yrs (counted double)

**Section B: Risk tolerance (score each 1 to 4)**

7. Your portfolio falls 20% in 3 months: sell all / sell some / hold / buy more
8. Acceptable one-year outcome range: +5/-3, +10/-8, +18/-15, +30/-25 (percent)
9. Equity experience: none / under 2 yrs / 2-5 / 5+
10. Which matters more: protecting capital / steady income / balanced / maximum growth

**Section C: Preferences (filters, not scored)**

11. Goal: retirement / education / house / wealth growth / short-term
12. Cap preference: large only / large + mid / include small
13. Sectors to favour or avoid (multi-select)
14. Dividend preference: yes / no / no preference

**Scoring**

- Capacity = normalised sum of Q1 to Q6 (0 to 100). Tolerance = normalised sum of Q7 to Q10 (0 to 100).
- Each maps to levels 1 to 5 (cut-offs at 20, 40, 60, 80). **Profile level = min(capacity level, tolerance level).**
- Level labels: 1 Conservative, 2 Moderately Conservative, 3 Moderate, 4 Moderately Aggressive, 5 Aggressive.
- **Guardrails:** horizon under 1 year caps the level at 2. Emergency fund under 3 months caps the level at 3. The result page states which guardrail applied.

**Target allocation by profile**

| Level | Equity | Debt | Gold | Equity split (large / mid / small) |
| --- | --- | --- | --- | --- |
| 1 | 20% | 70% | 10% | 100 / 0 / 0 |
| 2 | 35% | 55% | 10% | 90 / 10 / 0 |
| 3 | 50% | 40% | 10% | 70 / 25 / 5 |
| 4 | 70% | 25% | 5% | 55 / 30 / 15 |
| 5 | 85% | 10% | 5% | 40 / 35 / 25 |

*These are illustrative educational assumptions, documented as such.*

## 12. Portfolio Input

- **Manual (default):** stock autocomplete from the universe, quantity, optional average buy price. Fund categories (large-cap, mid/small-cap, index, debt, gold/other) with amount invested.
- **CSV (optional):** columns `ticker, quantity, avg_buy_price`. A template is downloadable. The upload shows a validation report: unknown tickers, negative quantities, duplicates (merged), missing prices.
- Stocks are valued at the latest close. Holdings outside the universe are listed as "not analysed" and excluded from metrics with a visible warning.
- **Privacy:** portfolio lives in Streamlit session state only. Nothing is written to disk or logged.

**Computed outputs:** total value, weights by stock, sector and cap; concentration flags (single stock over 25%, sector over 40%); portfolio volatility, max drawdown, VaR; average pairwise correlation and a 0 to 100 diversification score; current vs target allocation with drift.

## 13. Recommendation Engine (rule-based, explainable)

1. **Eligibility filter:** cap class within preference; volatility bucket not above the profile's maximum (level 1 to 5 permits bucket 2 to 5); not in excluded sectors; history of at least 3 years; liquidity above a minimum traded value; not already above its weight limit in the portfolio.
2. **Score** (each component z-scored across eligible stocks):
   - 35% Sharpe ratio
   - 20% shallow max drawdown
   - 25% diversification gain (drop in portfolio volatility and average correlation if 5% is added)
   - 10% fills an under-represented sector versus target
   - 10% dividend yield if the user prefers dividends, otherwise momentum
3. **Selection:** top 8 with at most 2 per sector.
4. **Suggested weights:** inverse-volatility weights, 10% per-stock cap, scaled to the equity share of the target allocation.
5. **Explanation template:** "Why this stock: Sharpe 0.9 (top 20% of eligible), correlation with your portfolio 0.28, adds exposure to Pharma where you hold 0%."
6. Output includes a "what would change" note: if the user's risk level went up or down by one, which picks would change.

The recommender is deterministic and unit-tested, with fixture portfolios and expected outputs. Forecasts are not used for ranking.

## 14. System Architecture

```
GitHub Actions (daily) -> pull_prices / pull_fundamentals
   -> raw/ -> validate + clean -> clean DB (SQLite/DuckDB)
   -> features + stats jobs -> features tables, stats tables
   -> Streamlit app (read-only) <- user session (profile, portfolio)
```

**Stack:** Python 3.11, pandas, numpy, scipy, statsmodels, arch, scikit-learn, Plotly, Streamlit, DuckDB or SQLite, pytest, GitHub Actions, Streamlit Community Cloud. Power BI is optional, as a supplementary report.

**Repository**

```
project/
  data/{raw,clean,features,reference,manifests}
  src/{ingest,clean,features,stats,profile,portfolio,recommend}
  notebooks/01_problem ... 09_ethics
  app/{Home.py, pages/, components/, theme.py}
  tests/  reports/  .github/workflows/refresh.yml
  README.md  requirements.txt  config.yaml
```

`config.yaml` holds the universe, dates, risk-free rate, thresholds and score weights, so nothing is hard-coded.

## 15. UI Specification

**Design principles (borrowed from Zerodha Kite/Varsity, Tickertape, Screener.in, Kuvera, Scripbox, Groww):** one primary action per screen, numbers with plain-language labels, a stepper for onboarding, colour used for meaning only, and a risk gauge as the visual anchor.

**Visual system**

- Light and dark themes. Teal (#00B386-like) for positive and primary actions, red for negative, amber for warnings, neutral grey for everything else.
- Sans-serif type (Inter), tabular numerals for prices.
- Indian formatting: ₹ and lakh/crore grouping, dates as DD MMM YYYY.
- Persistent top bar: logo, global stock search, **Data as of** chip, theme toggle.
- Left sidebar navigation (Streamlit pages). Persistent disclaimer in the footer.

**Navigation:** Home | Explore Market | Stock Detail | Compare & Screen | My Profile | My Portfolio | Recommendations | Methodology & Ethics

### 15.1 Home

```
[Logo]  [ Search stocks...            ]  Data as of 20 Oct 2026   [Theme]
+------------------------------------------------------------------+
| Understand your risk. Build a smarter portfolio.                 |
| [ Start risk profile ]   [ Explore the market ]                  |
+------------------------------------------------------------------+
| KPI: Stocks tracked | Years of data | Sectors | Median volatility|
+------------------------------------------------------------------+
| Nifty rebased chart (1Y/5Y/10Y)  |  Sector heat map (1Y return)  |
+------------------------------------------------------------------+
| Most volatile today*  | Lowest drawdown | Highest Sharpe (cards)  |
+------------------------------------------------------------------+
```

*"Today" means the latest close. Not intraday.*

### 15.2 Explore Market (Q1 to Q3, Q7 to Q10)

```
Filters: [Sector v] [Cap v] [Period v]                [Export CSV]
+---------------------------------+--------------------------------+
| Rebased growth by sector (line) | Risk vs return scatter         |
|                                 | (colour=sector, size=mkt cap)  |
+---------------------------------+--------------------------------+
| Sector risk bars (vol, VaR)     | Correlation heat map + cluster |
+---------------------------------+--------------------------------+
| Stat insight cards: "Small caps are significantly riskier (p<.001)"|
+------------------------------------------------------------------+
```

Every chart has an "i" tooltip explaining how to read it.

### 15.3 Stock Detail (Q4 to Q6)

```
TICKER  Company name   [Sector][Cap]   Last close  1D%   [+ Add to portfolio]
[1M][6M][1Y][5Y][10Y][Max]
+------------------------------------------------------+-----------+
| Price chart (log toggle) with COVID / 2022 annotations| Risk gauge|
|                                                       | (1 to 5)  |
+------------------------------------------------------+-----------+
| Volume bars (aligned)                                | Key stats |
+------------------------------------------------------+ Sharpe,   |
| Return histogram + normal overlay | Drawdown curve   | Beta, VaR,|
+--------------------------------------+---------------+ P/E, Div  |
| Peers table (same sector): vol, Sharpe, correlation   +-----------+
+------------------------------------------------------------------+
```

### 15.4 Compare & Screen

```
[Screen]  Filters: Cap | Sector | Volatility bucket | Sharpe > x | Div yield > y
+------------------------------------------------------------------+
| Results table: sortable, risk badge, sparkline, [Compare] checkbox|
+------------------------------------------------------------------+
[Compare 2-4 stocks]: rebased chart | metrics table | correlation pairs
```

### 15.5 My Profile (questionnaire)

Stepper in three steps (Capacity, Tolerance, Preferences), one question card per screen, a progress bar and Back/Next, with answers stored in session.

```
Step 2 of 3  Risk tolerance         [#######-----]
+--------------------------------------------------+
| Your portfolio falls 20% in three months. You:   |
|  ( ) Sell everything     ( ) Sell some           |
|  ( ) Hold                ( ) Buy more            |
+--------------------------------------------------+
[ Back ]                                   [ Next ]
```

**Result screen:** riskometer gauge with the final level, capacity and tolerance shown as two bars with a note on which one limited the result, the target allocation donut, any guardrail messages, and "Retake" and "Continue to portfolio" buttons.

### 15.6 My Portfolio

```
[Manual entry] [Upload CSV]            [Download template]
Stocks:  [Ticker v] [Qty] [Avg price] [+ Add]     Funds: category + amount
+------------------------------------------------------------------+
| Holdings table: weight, value, gain/loss, risk badge, [remove]   |
+------------------------------------------------------------------+
| KPIs: Value | Volatility | Max drawdown | VaR | Diversification   |
+------------------------+-----------------------------------------+
| Sector donut           | Cap donut                               |
+------------------------+-----------------------------------------+
| Current vs target allocation bars (with drift arrows)            |
| Warnings: "Banking is 46% of your equity (limit 40%)"            |
+------------------------------------------------------------------+
| Portfolio correlation heat map                                    |
+------------------------------------------------------------------+
```

Empty state: a friendly prompt with a "Load a sample portfolio" button.

### 15.7 Recommendations

```
Profile: Moderate (3)   Cap: Large+Mid   Excluded: none   [Edit]
+------------------------------------------------------------------+
| Suggested portfolio additions (cards, 8)                         |
|  [Ticker] [Sector][Cap]  Risk badge   Suggested weight 6%        |
|  Why: Sharpe 0.9 (top 20%), correlation with yours 0.28, fills   |
|  Pharma gap.   [View stock] [Add to what-if]                     |
+------------------------------------------------------------------+
| What-if: before vs after (volatility, diversification, sectors)  |
| What would change if your risk level were one higher/lower       |
+------------------------------------------------------------------+
| Educational use only. Not investment advice.                      |
+------------------------------------------------------------------+
```

### 15.8 Methodology and Ethics

Tabs: Lifecycle (the phases linked to notebooks), Data and Sources, Metrics glossary, Statistical results, Limitations, Ethics and Privacy, Disclaimer.

### 15.9 UX requirements

Responsive layout down to tablet. Every chart has alt text and a table view. No colour-only signals (risk badges also show text). Loading skeletons. Friendly error states for data failures and invalid inputs.

## 16. Functional Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| F1 | Daily pipeline pulls, validates and stores prices, with manifest and logs | Must |
| F2 | Extreme days are flagged, never auto-deleted | Must |
| F3 | App shows a "data as of" date on all pages | Must |
| F4 | All Q1 to Q10 views implemented with correct metrics | Must |
| F5 | Questionnaire with capacity/tolerance scoring and guardrails | Must |
| F6 | Manual portfolio entry with validation | Must |
| F7 | CSV upload with validation report | Should |
| F8 | Portfolio analytics (concentration, volatility, diversification) | Must |
| F9 | Recommender with explanations and what-if | Must |
| F10 | Screener and compare | Should |
| F11 | Methodology and ethics pages | Must |
| F12 | Statistical test results page with plain-English readouts | Must |
| F13 | Export of tables to CSV | Could |
| F14 | GARCH volatility forecast panel | Could |

## 17. Non-Functional Requirements

- **Performance:** pages render in under 3 seconds (metrics are precomputed, the app reads only).
- **Reliability:** stale-data fallback, retry and backoff on ingestion.
- **Reproducibility:** pinned dependencies, seeded random operations, a one-command rebuild.
- **Testing:** unit tests for returns, metrics, scoring, validators and recommender. Data-quality tests on every refresh. Target at least 80% coverage on `src/`.
- **Security and privacy:** no user data persisted or logged, HTTPS via the host, no secrets in the repo.

## 18. Ethics, Bias, Fairness and Privacy

| Issue | Mitigation |
| --- | --- |
| **Survivorship bias:** only currently listed stocks are included, inflating historical returns | State it prominently, and discuss it quantitatively where possible |
| **Label look-ahead:** current cap class applied to history | Documented limitation |
| **Outlier bias:** deleting extremes understates risk | Extremes preserved, with the before/after impact reported |
| **Past performance is not future performance** | Shown with every ranking and forecast |
| **Questionnaire fairness:** wording or age scoring may disadvantage some users, and it could push novices into excess risk | Min-of-two scoring, guardrails, explanations, user override with warning |
| **Recommendation bias:** large-cap favouritism from liquidity filters | Report the cap distribution of recommendations |
| **Privacy:** portfolio data is sensitive (India's DPDP Act context) | Session-only, no persistence, no third-party analytics |
| **Regulatory:** SEBI rules for investment advisers and research analysts | Educational-use disclaimer, no "buy/sell" wording, explainable output |
| **Data licensing:** Yahoo data terms | Academic use, attribution, fallback to NSE data |
| **Overconfidence from forecasts** | Error bands, backtest results shown |

## 19. Milestones

| Phase | Output | Est. |
| --- | --- | --- |
| 0. Setup | Repo, config, universe list | 1 day |
| 1. Data | Ingestion, validation, cleaning report, database | 3 to 4 days |
| 2. EDA and features | Notebooks 02 to 04, feature tables | 3 days |
| 3. Statistics | Tests, tables, forecast backtest | 2 days |
| 4. Profile and portfolio engines | `src/profile`, `src/portfolio` with tests | 3 days |
| 5. Recommender | Scoring, explanations, tests | 2 days |
| 6. App | All pages, theming, empty and error states | 5 days |
| 7. Report and ethics | Report, methodology page | 3 days |
| 8. Deploy and polish | CI refresh, Streamlit Cloud, README, demo script | 2 days |

## 20. Success Metrics

- Pipeline runs unattended for 14 days without failure.
- All 13 questions answered with reproducible outputs.
- A new user can complete profile, portfolio and recommendations in under 5 minutes.
- Fixture portfolios give the expected recommender outputs in tests.
- All nine lifecycle phases appear as notebooks, report sections and app content.

## 21. Risks and Open Questions

| Risk or question | Plan |
| --- | --- |
| `yfinance` breaks or rate-limits | Local database as source of truth, NSE bhavcopy fallback |
| Small-cap data is sparse or illiquid | Liquidity and history filters, wider outlier review |
| Cap labels drift over time | Re-snapshot each refresh, record the date |
| Recommender seems arbitrary | Explanations, what-if, published weights in `config.yaml` |
| Scope creep | Anything beyond section 16 "Must" waits for v2 |
| Open: final universe list | To be fixed in Phase 0 |
| Open: mutual fund recommendations | v2, using AMFI NAV data |
