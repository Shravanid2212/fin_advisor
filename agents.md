# AGENTS.md

## Project
Financial Market Analytics & Investment Decision Support System for Indian equities.
Spec: docs/PRD.md (source of truth). If a request conflicts with the PRD, stop and ask.
This is an educational tool, not investment advice. Never use buy/sell wording in UI text.

## Stack
Python 3.11, pandas, numpy, scipy, statsmodels, arch, scikit-learn, plotly, streamlit,
duckdb (or sqlite3), pyyaml, pytest, yfinance. Pin versions in requirements.txt.

## Layout
data/{raw,clean,features,reference,manifests}
src/{ingest,clean,features,stats,profile,portfolio,recommend,accounts}
notebooks/01_problem ... 09_ethics
app/{Home.py,pages/,components/,theme.py}
tests/  reports/  .github/workflows/refresh.yml
config.yaml  README.md  requirements.txt

## Hard rules
1. NEVER delete genuine extreme return days (crashes like March 2020). Flag them with
   is_extreme. Only confirmed data errors (price <= 0, high < low, close outside [low, high],
   bad ticks, unadjusted splits) may be fixed or dropped, and every removal is logged with a reason.
2. NEVER compare tickers using a global min-max scale. Use per-ticker rebased price (100 at start)
   or cumulative return.
3. Raw data in data/raw is immutable. Cleaning writes to data/clean only.
4. The app is read-only against the database. It never calls yfinance live.
5. All thresholds, weights, dates, the risk-free rate and the stock universe live in config.yaml.
   No magic numbers in code.
6. User accounts, profiles and portfolios are stored exclusively in a dedicated local SQLite database (data/users.db), completely separated from the read-only market.duckdb. Passwords are never stored in plain text; they are salted and hashed using PBKDF2-HMAC-SHA256 with 200,000 iterations. Users can save, reset their portfolio, and permanently delete all their account data at any time via 'Delete my account'. User data is never sent to any external service or third party.
7. Every metric function needs a docstring with the formula and a unit test.
8. Annualisation uses 252 trading days. Returns are simple unless named log_return.
9. Dates are ISO (YYYY-MM-DD) in storage and DD MMM YYYY in the UI. Currency uses the rupee
   sign with lakh/crore grouping in the UI.
10. Every page shows 'Data as of <date>'. Recommendations keep neutral language without buy/sell advice.
11. Ranking scores stay transparent and rule-based. ML may add labels and context, and may exclude stocks flagged 'risk rising' for profile levels 1 and 2 only.

## Code style
Type hints, small pure functions, no notebooks logic duplicated in src (notebooks import from src).
Format with black and ruff. Seed all randomness (seed in config.yaml).

## Working style
- Work only on the phase requested. Do not touch other modules except to fix imports.
- Before coding, state a short plan. After coding, run pytest and report results.
- Never invent data. If a source fails, report it and use the documented fallback.
- End every phase with: files changed, tests run (pass/fail counts), assumptions made, and
  anything unverified.
- Ask before adding a new dependency.
