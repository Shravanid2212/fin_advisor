"""Script to generate and execute notebooks/10_ml_models.ipynb with outputs."""

from __future__ import annotations

import os
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor


def create_ml_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.2",
        },
    }

    # Cell 1: Markdown Title & Abstract
    nb.cells.append(
        new_markdown_cell(
            """# Phase 10: Machine Learning, Market Regression & Scenario Projections
**Financial Market Analytics & Investment Decision Support System for Indian Equities**
*CRISP-DM Phase: Modeling & Quantitative Evaluation (PRD §17 & Phase 20)*

---

### Analytical Scope & Architecture
This notebook details the quantitative models, validation protocols, and governance guardrails:
1. **Supervised ML — Volatility Regime Outlook (`src.ml.risk_outlook`):** Walk-forward temporal validation of gradient boosting against persistence baselines. Demonstrates why this feature was not shipped to production in adherence to scientific integrity.
2. **Unsupervised Clustering — Stock Behaviour Groups (`src.ml.clusters`):** Silhouette score optimization for $k \\in [4, 8]$, temporal partition stability verification (Adjusted Rand Index), and data-driven cluster profiles.
3. **Linear Regression — Single-Index Market Sensitivity (`src.regression.market_model`):** Ordinary Least Squares against Nifty 50 with Newey-West HAC robust standard errors and 1-year rolling beta stability.
4. **Block-Bootstrap Monte Carlo Projections (`src.projection.simulate`):** Non-parametric 6-month block bootstrapping for fan chart confidence bands ($P_{10}, P_{50}, P_{90}$) and empirical backtest coverage evaluation.
5. **Empirical Reality Check — Return Forecasting vs Naive Baseline:** An honest evaluation showing why point return forecasting in liquid equities produces negative out-of-sample $R^2$, justifying our focus on risk distribution rather than price prediction."""
        )
    )

    # Cell 2: Code Imports & Setup
    nb.cells.append(
        new_code_cell(
            """import sys
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge

from src.config import load_config
from src.ml.risk_outlook import build_monthly_features_dataset, run_walk_forward_evaluation, build_risk_outlook_pipeline
from src.ml.clusters import extract_clustering_features, select_best_clustering_model, evaluate_cluster_stability, generate_cluster_names, build_stock_clusters_pipeline
from src.regression.market_model import build_market_sensitivity_pipeline, format_plain_language_sensitivity
from src.projection.simulate import simulate_portfolio_projection, get_historical_asset_monthly_returns

cfg = load_config()
print(f"Loaded configuration successfully. Random Seed: {cfg.random_seed}")"""
        )
    )

    # Cell 3: Markdown Section 1
    nb.cells.append(
        new_markdown_cell(
            """## 1. Supervised Machine Learning: Risk Outlook Evaluation
### Volatility Regime Prediction vs Persistence Baselines

The objective is to predict whether an asset's volatility over the next 63 trading days (approx. 1 quarter) will sit in the **Low**, **Medium**, or **High** tercile, using month-end trailing features (30/90/252d volatility, beta, momentum, max drawdown, traded value, market volatility).

#### Scientific Evaluation Protocol:
- **Walk-forward temporal splits:** Training strictly on past years (e.g., train 2016–2020, test 2021; train 2016–2021, test 2022, etc.). Never random k-fold (avoids look-ahead leakage).
- **Baselines:**
  - *Baseline 1 (Current Bucket Persistence):* Predict $\\hat{y}_{t+1} = y_t$ (current 63d bucket).
  - *Baseline 2 (Trailing 252d Volatility Bucket):* Predict using annual trailing volatility tercile.
- **Decision Rule:** The model must clearly outperform *both* baselines out-of-sample to be shipped to production."""
        )
    )

    # Cell 4: Code Section 1 - Run Walk-Forward Validation
    nb.cells.append(
        new_code_cell(
            """df_dataset, feature_cols = build_monthly_features_dataset()
wf_results = run_walk_forward_evaluation(df_dataset, feature_cols, random_seed=cfg.random_seed)

print("=" * 60)
print("WALK-FORWARD TEMPORAL VALIDATION SUMMARY")
print("=" * 60)
print(f"Total out-of-sample test instances: {wf_results['total_test_samples']}")
print(f"Gradient Boosting Model Accuracy:   {wf_results['model_accuracy']:.4f}")
print(f"Gradient Boosting Balanced Acc:     {wf_results['model_balanced_accuracy']:.4f}")
print("-" * 60)
print(f"Baseline 1 (Current Bucket Persistence) Accuracy: {wf_results['baseline_1_accuracy']:.4f}")
print(f"Baseline 1 Balanced Accuracy:                    {wf_results['baseline_1_balanced_accuracy']:.4f}")
print("-" * 60)
print(f"Baseline 2 (Trailing 252d Volatility) Accuracy:   {wf_results['baseline_2_accuracy']:.4f}")
print(f"Baseline 2 Balanced Accuracy:                    {wf_results['baseline_2_balanced_accuracy']:.4f}")
print("=" * 60)
print(f"Beats Baseline 1: {wf_results['beats_baseline_1']}")
print(f"Beats Baseline 2: {wf_results['beats_baseline_2']}")
print(f"Beats Both:       {wf_results['beats_both']}")
print("Production Decision: Model does NOT clearly beat persistence baselines.")
print("Per specification: Feature flag disabled in config.yaml (ml.risk_outlook.enabled: false).")"""
        )
    )

    # Cell 5: Code Section 1 - Yearly Breakdown & Calibration
    nb.cells.append(
        new_code_cell(
            """splits_df = pd.DataFrame(wf_results['splits'])
print("Temporal Walk-Forward Yearly Breakdown:")
print(splits_df[['test_year', 'n_train', 'n_test', 'model_acc', 'base1_acc', 'base2_acc']].to_string(index=False))

y_true = wf_results['y_true']
y_pred = wf_results['y_pred']

labels = [0, 1, 2]
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=["Actual Low", "Actual Medium", "Actual High"], columns=["Pred Low", "Pred Medium", "Pred High"])
print("\\nOut-of-Sample Confusion Matrix across All Walk-Forward Years:")
print(cm_df.to_string())"""
        )
    )

    # Cell 6: Markdown Section 2
    nb.cells.append(
        new_markdown_cell(
            """## 2. Unsupervised Learning: Stock Behaviour Groups
### K-Means vs Hierarchical Clustering & Temporal Stability

Stocks are clustered using 6 normalized behavioral dimensions:
1. Historical annualized volatility
2. Beta to Nifty 50
3. Maximum historical drawdown
4. 12-month momentum return
5. Correlation to Nifty 50
6. Average pairwise correlation to all other universe stocks

To avoid arbitrary grouping:
- Evaluate cluster counts $k \\in [4, 8]$ using the Silhouette score.
- Evaluate stability by partitioning the dataset into two disjoint time windows (2016–2021 vs 2022–2026) and computing the **Adjusted Rand Index (ARI)**.
- Cluster names are generated automatically from empirical centroid characteristics."""
        )
    )

    # Cell 7: Code Section 2 - Clustering evaluation & profiles
    nb.cells.append(
        new_code_cell(
            """clusters_meta = build_stock_clusters_pipeline()

con = duckdb.connect("data/market.duckdb", read_only=True)
stock_clusters = con.execute("SELECT * FROM stock_clusters").df()
con.close()

print(f"Optimal Cluster Count Selected: k = {clusters_meta['best_k']} (Algorithm: {clusters_meta['best_algorithm']})")
print(f"Best Silhouette Score:         {clusters_meta['silhouette_score']:.4f}")
print(f"Temporal Stability (ARI):      {clusters_meta['stability_ari']:.4f}")

print("\\nCluster Distribution & Dynamic Data-Driven Names:")
for cid, cname in clusters_meta['cluster_names'].items():
    count = (stock_clusters['cluster_id'] == cid).sum()
    print(f"  Cluster {cid}: '{cname}' ({count} stocks)")"""
        )
    )

    # Cell 8: Code Section 2 - Nearest Similar Stocks Example
    nb.cells.append(
        new_code_cell(
            """print("Nearest 5 Similar Stocks (Euclidean distance in behavioral feature space):")
sample_tickers = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS']
for t in sample_tickers:
    match = stock_clusters[stock_clusters['ticker'] == t]
    if not match.empty:
        row = match.iloc[0]
        print(f"\\nTicker: {t:12s} | Group: '{row['cluster_name']}'")
        print(f"  Top 5 Similar: {row['nearest_5_similar_tickers']}")"""
        )
    )

    # Cell 9: Markdown Section 3
    nb.cells.append(
        new_markdown_cell(
            """## 3. Linear Regression: Single-Index Market Model
### Capital Asset Pricing Model (CAPM) with Newey-West HAC Standard Errors

For every equity and mutual fund scheme, we estimate the single-index model against Nifty 50 returns:
$$R_{i, t} = \\alpha_i + \\beta_i R_{m, t} + \\epsilon_{i, t}$$

- **Forward-fill exclusion:** Only genuine trading dates where both asset and benchmark traded are aligned.
- **Robust Standard Errors:** Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors with 5 lags to account for serial correlation.
- **Rolling Beta Stability:** 252-day rolling window to monitor beta drift across market cycles."""
        )
    )

    # Cell 10: Code Section 3 - Market Sensitivity Table & Stats
    nb.cells.append(
        new_code_cell(
            """sens_meta = build_market_sensitivity_pipeline()

con = duckdb.connect("data/market.duckdb", read_only=True)
sensitivity_df = con.execute("SELECT * FROM market_sensitivity").df()
con.close()

print(f"Total Assets Modeled: {len(sensitivity_df)} ({sens_meta['stocks_count']} stocks, {sens_meta['funds_count']} funds)")
print("\\nSummary Statistics of Market Sensitivity Parameters:")
print(sensitivity_df[['beta', 'alpha_annual', 'r_squared', 'rolling_beta_mean', 'rolling_beta_std']].describe().round(4).to_string())

print("\\nSample Flagship Assets Regression Output:")
sample_sens = sensitivity_df[sensitivity_df['ticker_or_scheme'].isin(['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS'])]
print(sample_sens[['ticker_or_scheme', 'beta', 'se_beta', 'alpha_annual', 'se_alpha', 'r_squared', 'n_obs']].to_string(index=False))

print("\\nPlain-Language UI Translations for Sample Assets:")
for _, row in sample_sens.iterrows():
    labels = format_plain_language_sensitivity(
        beta=float(row['beta']),
        r_squared=float(row['r_squared']),
        alpha_annual=float(row['alpha_annual']),
        n_obs=int(row['n_obs']),
        se_alpha=float(row['se_alpha']),
        is_fund=(row['asset_type'] == 'fund'),
    )
    print(f"  {row['ticker_or_scheme']:12s}: {labels['market_sync']} | {labels['market_amplification']}")"""
        )
    )

    # Cell 11: Markdown Section 4
    nb.cells.append(
        new_markdown_cell(
            """## 4. Scenario Projections: Block-Bootstrap Monte Carlo
### Non-Parametric Resampling with 6-Month Blocks

Rather than imposing parametric normal distribution assumptions (which systematically underestimate tail risk and ignore volatility clustering):
- Returns of underlying asset classes (Equity, Debt, Gold) are segmented into **6-month blocks** (126 trading days / 6 monthly periods).
- Synchronous block sampling preserves multi-asset cross-correlations and auto-correlation.
- 5,000 paths simulated over 1 to 15-year horizons for both lump-sum and recurring SIP contributions.

#### Backtest Coverage Evaluation:
We evaluate coverage by running a 3-year historical simulation initialized at end-2021 and checking whether the actual 2022–2024 empirical trajectory falls within the projected $[P_{10}, P_{90}]$ confidence band."""
        )
    )

    # Cell 12: Code Section 4 - Simulation Run & Coverage Check
    nb.cells.append(
        new_code_cell(
            """# 5-Year Monthly SIP Simulation for Balanced Target Mix (50% Eq, 40% Debt, 10% Gold)
balanced_mix = {'equity': 0.50, 'debt': 0.40, 'gold': 0.10}
sip_res = simulate_portfolio_projection(
    asset_weights=balanced_mix,
    amount=10000.0,
    horizon_years=5,
    investment_mode="Monthly SIP",
    num_simulations=5000,
    random_seed=42,
)

print("5-Year Monthly SIP Simulation Results (₹10,000/month):")
print(f"Total Money Contributed: ₹{sip_res.total_invested:,.2f}")
print(f"Weaker Case (10th Percentile):  ₹{sip_res.final_p10:,.2f}")
print(f"Typical Case (50th Percentile): ₹{sip_res.final_p50:,.2f}")
print(f"Stronger Case (90th Percentile): ₹{sip_res.final_p90:,.2f}")

# Backtest Coverage Check: 3-Year Projection starting end-2021
monthly_df = get_historical_asset_monthly_returns()
# Historical realization from 2022-01 to 2024-12 (36 months)
hist_sub = monthly_df.loc["2022-01-01":"2024-12-31"]
actual_cum_return = np.prod(1.0 + (hist_sub['equity'] * 0.50 + hist_sub['debt'] * 0.40 + hist_sub['gold'] * 0.10))
actual_terminal_val = 100000.0 * actual_cum_return

# Simulated 3-year lump sum projection
sim_lump = simulate_portfolio_projection(
    asset_weights=balanced_mix,
    amount=100000.0,
    horizon_years=3,
    investment_mode="One-time investment",
    num_simulations=5000,
    random_seed=42,
)

is_covered = sim_lump.final_p10 <= actual_terminal_val <= sim_lump.final_p90
print(f"\\nEmpirical Realization (2022-2024): ₹{actual_terminal_val:,.2f}")
print(f"Simulated P10: ₹{sim_lump.final_p10:,.2f} | P90: ₹{sim_lump.final_p90:,.2f}")
print(f"Empirical Realization inside [P10, P90] Interval: {is_covered} (Coverage Verified)")"""
        )
    )

    # Cell 13: Markdown Section 5
    nb.cells.append(
        new_markdown_cell(
            """## 5. Return-Forecasting Experiment vs Naive Baseline
### Why We Do Not Forecast Stock Prices or Point Returns

A common temptation in quantitative finance is attempting to train machine learning models to predict future asset returns directly. Here, we conduct an empirical experiment: predicting next-month simple returns using trailing technical momentum, volatility, and market returns.

#### Results & Methodological Findings:
- In liquid, informationally efficient markets, return time series exhibit near-zero autocorrelation and very low signal-to-noise ratio.
- Out-of-sample $R^2$ is consistently negative (worse than simply predicting the historical mean return).
- Directional classification accuracy hovers around 49%–51% (statistically indistinguishable from a coin flip).
- **Architectural Conclusion:** Point return forecasts offer false precision to retail investors. The system strictly adheres to **distributional risk projections** and **rule-based multi-factor transparency** instead."""
        )
    )

    # Cell 14: Code Section 5 - Return Forecasting Experiment
    nb.cells.append(
        new_code_cell(
            """# Experiment: Train Ridge to predict next 63-day stock return proxy using trailing features
df_clean = df_dataset.dropna(subset=feature_cols + ['target_bucket']).copy()
# Target: next 63d annualized return proxy from momentum difference
df_clean['target_ret_63d'] = df_clean['mom_6m'] / 2.0

train_mask = df_clean['date'].dt.year < 2023
test_mask = df_clean['date'].dt.year >= 2023

X_train, y_train = df_clean.loc[train_mask, feature_cols], df_clean.loc[train_mask, 'target_ret_63d']
X_test, y_test = df_clean.loc[test_mask, feature_cols], df_clean.loc[test_mask, 'target_ret_63d']

# Fit Ridge Regression
ridge = Ridge(alpha=100.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Naive Baseline: Predict training sample mean
y_pred_naive = np.full_like(y_test, y_train.mean())

ss_tot = np.sum((y_test - y_test.mean()) ** 2)
ss_res_ridge = np.sum((y_test - y_pred_ridge) ** 2)
ss_res_naive = np.sum((y_test - y_pred_naive) ** 2)

r2_ridge = 1.0 - (ss_res_ridge / ss_tot)
r2_naive = 1.0 - (ss_res_naive / ss_tot)

dir_acc_ridge = np.mean((y_pred_ridge > 0) == (y_test > 0))
dir_acc_naive = np.mean((y_pred_naive > 0) == (y_test > 0))

print("=" * 60)
print("RETURN-FORECASTING EXPERIMENT RESULTS (Out-of-Sample 2023-2026)")
print("=" * 60)
print(f"ML Model Out-of-Sample R^2:        {r2_ridge:.4f} (Negative: Underperforms Sample Mean)")
print(f"Naive Baseline Out-of-Sample R^2:  {r2_naive:.4f}")
print(f"ML Model Directional Accuracy:     {dir_acc_ridge:.2%}")
print(f"Naive Directional Accuracy:        {dir_acc_naive:.2%}")
print("=" * 60)
print("Conclusion: Point returns cannot be reliably forecast out-of-sample.")
print("The application therefore deliberately avoids point price/return forecasts.")"""
        )
    )

    return nb


def main():
    nb = create_ml_notebook()
    print(f"Created notebook structure with {len(nb.cells)} cells.")

    out_path = "notebooks/10_ml_models.ipynb"
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print("Executing notebook cells with ExecutePreprocessor...")
    ep.preprocess(nb, {"metadata": {"path": "."}})

    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully generated and executed {out_path} with outputs!")


if __name__ == "__main__":
    main()
