"""Market Sensitivity and Linear Regression Module (Single-Index Market Model).

Adheres to:
- PRD Section 11/13 and AGENTS.md rules.
- Regresses asset simple returns on benchmark Nifty 50 (^NSEI) returns.
- Estimates beta, annualised alpha, R-squared, observation count, and HAC (Newey-West) robust standard errors.
- Calculates 252-day rolling beta for stability metrics.
- Provides pure-python helper for plain-language, jargon-free investor labels.
- Persists table 'market_sensitivity' in market.duckdb.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import load_config


def fit_asset_market_model(
    asset_rets: pd.Series,
    benchmark_rets: pd.Series,
    ticker_or_scheme: str,
    asset_type: str = "stock",
    trading_days: int = 252,
) -> dict[str, Any] | None:
    """Fit single-index OLS regression of asset returns on benchmark returns with HAC standard errors.

    Model:
        R_{i,t} = alpha_i + beta_i * R_{m,t} + epsilon_{i,t}

    Formulae:
        Beta = Cov(R_i, R_m) / Var(R_m)
        Alpha_daily = mean(R_i) - Beta * mean(R_m)
        Alpha_annual = Alpha_daily * 252
        R_squared = 1 - SS_res / SS_tot
        HAC SE: Newey-West standard errors robust to heteroskedasticity and autocorrelation.
    """
    clean_df = pd.DataFrame({
        "asset": asset_rets,
        "mkt": benchmark_rets,
    }).dropna()

    n_obs = len(clean_df)
    if n_obs < 60:
        return None

    x = sm.add_constant(clean_df["mkt"])
    y = clean_df["asset"]

    # Fit with Heteroskedasticity and Autocorrelation Consistent (HAC) covariance
    try:
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        beta = float(model.params["mkt"])
        alpha_daily = float(model.params["const"])
        alpha_annual = float(alpha_daily * trading_days)
        r2 = float(model.rsquared)
        se_beta = float(model.bse["mkt"])
        se_alpha = float(model.bse["const"] * trading_days)
    except Exception:
        # Fallback to standard OLS if HAC matrix inversion fails on small samples
        model = sm.OLS(y, x).fit()
        beta = float(model.params["mkt"])
        alpha_daily = float(model.params["const"])
        alpha_annual = float(alpha_daily * trading_days)
        r2 = float(model.rsquared)
        se_beta = float(model.bse["mkt"])
        se_alpha = float(model.bse["const"] * trading_days)

    # Rolling 1-year (252-day) beta
    rolling_window = min(trading_days, n_obs)
    roll_cov = clean_df["asset"].rolling(rolling_window).cov(clean_df["mkt"])
    roll_var = clean_df["mkt"].rolling(rolling_window).var()
    roll_beta = (roll_cov / roll_var.replace(0, np.nan)).dropna()

    if len(roll_beta) > 0:
        r_mean = float(roll_beta.mean())
        r_min = float(roll_beta.min())
        r_max = float(roll_beta.max())
        r_std = float(roll_beta.std()) if len(roll_beta) > 1 else 0.0
    else:
        r_mean, r_min, r_max, r_std = beta, beta, beta, 0.0

    return {
        "ticker_or_scheme": ticker_or_scheme,
        "asset_type": asset_type,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "r_squared": max(0.0, min(1.0, r2)),
        "n_obs": n_obs,
        "se_beta": se_beta,
        "se_alpha": se_alpha,
        "rolling_beta_mean": r_mean,
        "rolling_beta_min": r_min,
        "rolling_beta_max": r_max,
        "rolling_beta_std": r_std,
    }


def format_plain_language_sensitivity(
    beta: float,
    r_squared: float,
    alpha_annual: float | None = None,
    n_obs: int = 0,
    se_alpha: float | None = None,
    is_fund: bool = False,
) -> dict[str, str]:
    """Translate technical regression statistics into plain, jargon-free investor sentences.

    Rules:
    - No statistical jargon, no p-values, no model names in user-facing UI.
    - Moves with market: derived from R-squared.
    - Amplifies/dampens moves: derived from Beta.
    - For funds: excess risk-adjusted return from Alpha (when statistically meaningful and not noise).
    """
    pct_explained = round(r_squared * 100)
    if r_squared >= 0.60:
        mkt_sync = f"Moves with the market: High (about {pct_explained}% of its ups and downs)"
    elif r_squared >= 0.25:
        mkt_sync = f"Moves with the market: Moderate (about {pct_explained}% of its ups and downs)"
    else:
        mkt_sync = f"Moves with the market: Low (about {pct_explained}% of its ups and downs)"

    if beta >= 1.05:
        mkt_amp = f"Amplifies market moves by about {beta:.1f}x"
    elif beta <= 0.95:
        mkt_amp = f"Dampens market moves (about {beta:.1f}x market moves)"
    else:
        mkt_amp = f"Moves in line with the market (about {beta:.1f}x)"

    fund_alpha_text = ""
    if is_fund and alpha_annual is not None and n_obs >= 250:
        # Check that estimate is not tiny noise
        se_threshold = (se_alpha * 0.5) if se_alpha is not None else 0.01
        if abs(alpha_annual) > se_threshold:
            alpha_pct = round(abs(alpha_annual) * 100, 1)
            if alpha_annual > 0:
                fund_alpha_text = f"Beat what the market's risk would explain by about {alpha_pct}% a year"
            else:
                fund_alpha_text = f"Trailed what the market's risk would explain by about {alpha_pct}% a year"

    return {
        "market_sync": mkt_sync,
        "market_amplification": mkt_amp,
        "fund_alpha": fund_alpha_text,
    }


def build_market_sensitivity_pipeline(
    db_path: str = "data/market.duckdb",
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Fit regression model for all stocks and mutual funds, storing market_sensitivity table."""
    cfg = load_config(config_path)
    con = duckdb.connect(db_path, read_only=False)

    # 1. Fetch stock and benchmark prices
    prices = con.execute(
        "SELECT date, ticker, simple_return FROM prices WHERE is_filled = false ORDER BY ticker, date"
    ).df()
    nifty = (
        prices[prices["ticker"] == "^NSEI"]
        .copy()
        .sort_values("date")
        .set_index("date")["simple_return"]
    )

    records = []
    latest_date_str = str(prices["date"].max())

    # Process stocks
    tickers = con.execute("SELECT ticker FROM universe ORDER BY ticker").df()["ticker"].tolist()
    pivot_stocks = prices[prices["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="simple_return"
    )

    for ticker in tickers:
        if ticker not in pivot_stocks.columns:
            continue
        res = fit_asset_market_model(
            asset_rets=pivot_stocks[ticker],
            benchmark_rets=nifty,
            ticker_or_scheme=ticker,
            asset_type="stock",
            trading_days=cfg.trading_days,
        )
        if res:
            res["as_of_date"] = latest_date_str
            records.append(res)

    # 2. Fetch mutual fund NAVs
    fund_nav_df = con.execute(
        "SELECT scheme_code, date, nav FROM fund_nav ORDER BY scheme_code, date"
    ).df()

    if not fund_nav_df.empty:
        fund_nav_df["date"] = pd.to_datetime(fund_nav_df["date"])
        nifty_dt = nifty.copy()
        nifty_dt.index = pd.to_datetime(nifty_dt.index)

        grouped = fund_nav_df.groupby("scheme_code")
        for sc, g in grouped:
            g = g.sort_values("date").set_index("date")
            nav_rets = g["nav"].pct_change().dropna()
            res = fit_asset_market_model(
                asset_rets=nav_rets,
                benchmark_rets=nifty_dt,
                ticker_or_scheme=str(int(sc)),
                asset_type="fund",
                trading_days=cfg.trading_days,
            )
            if res:
                res["as_of_date"] = latest_date_str
                records.append(res)

    sensitivity_df = pd.DataFrame(records)

    # 3. Store table in DuckDB
    con.execute("DROP TABLE IF EXISTS market_sensitivity")
    con.execute(
        """
        CREATE TABLE market_sensitivity (
            ticker_or_scheme VARCHAR PRIMARY KEY,
            asset_type VARCHAR,
            beta DOUBLE,
            alpha_annual DOUBLE,
            r_squared DOUBLE,
            n_obs BIGINT,
            se_beta DOUBLE,
            se_alpha DOUBLE,
            rolling_beta_mean DOUBLE,
            rolling_beta_min DOUBLE,
            rolling_beta_max DOUBLE,
            rolling_beta_std DOUBLE,
            as_of_date VARCHAR
        )
        """
    )
    con.register("df_sens", sensitivity_df)
    con.execute("INSERT INTO market_sensitivity SELECT * FROM df_sens")
    con.close()

    return {
        "total_assets_modeled": len(sensitivity_df),
        "stocks_count": len([r for r in records if r["asset_type"] == "stock"]),
        "funds_count": len([r for r in records if r["asset_type"] == "fund"]),
        "median_r_squared": float(sensitivity_df["r_squared"].median()),
        "median_beta": float(sensitivity_df["beta"].median()),
    }


if __name__ == "__main__":
    print("Executing Market Sensitivity Regression Pipeline...")
    res = build_market_sensitivity_pipeline()
    print("Execution complete. Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
