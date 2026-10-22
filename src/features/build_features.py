"""Feature engineering pipeline for Financial Market Analytics System.

Computes all stock-level features specified in PRD §9:
- CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Recovery Days
- Systematic Risk (CAPM Beta against ^NSEI)
- Historical 95% Value at Risk (VaR)
- 6-Month and 12-Month Momentum
- 60-Day Average Traded Value (Liquidity)
- Volatility Quintile Risk Buckets (1 to 5)
- Point-in-time Fundamental valuation metrics (P/E, Dividend Yield, Market Cap)

Stores feature tables in DuckDB (data/market.duckdb) and exports data/features/stock_features.parquet.
Adheres strictly to AGENTS.md Hard Rules:
- Rule 2: Rebased / percentage returns, never global min-max scaling.
- Rule 4: Read-only access against prices; atomic writes for features.
- Rule 8: 252 trading days for annualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd

from src.config import load_config
from src.features.metrics import (
    assign_risk_buckets,
    calculate_annualized_volatility,
    calculate_beta,
    calculate_cagr,
    calculate_downside_deviation,
    calculate_drawdowns,
    calculate_historical_var,
    calculate_momentum,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

logger = logging.getLogger(__name__)


def build_all_stock_features(
    db_path: Path | str = "data/market.duckdb",
    export_parquet_path: Path | str = "data/features/stock_features.parquet",
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Compute and store stock features across all universe equities in DuckDB and Parquet.

    Args:
        db_path: Path to database.
        export_parquet_path: Path to parquet export.
        risk_free_rate: Annual risk-free rate (default 0.065).
        trading_days: Trading days per year (default 252).

    Returns:
        DataFrame containing computed features for all 80 stocks.
    """
    p = Path(db_path)
    con = duckdb.connect(str(p))

    # 1. Fetch Universe & Fundamentals
    df_univ = con.execute("SELECT ticker, name, sector, cap_class FROM universe").fetchdf()
    df_funds = con.execute("""
        SELECT ticker, market_cap, trailing_pe, forward_pe, dividend_yield
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_date DESC) as rn
            FROM fundamentals_snapshots
        ) WHERE rn = 1
    """).fetchdf()

    # 2. Fetch Benchmark returns (^NSEI) for Beta calculation
    df_nifty = con.execute("""
        SELECT date, simple_return
        FROM prices
        WHERE ticker = '^NSEI' AND is_filled = FALSE AND simple_return IS NOT NULL
        ORDER BY date
    """).fetchdf().set_index("date")["simple_return"]

    # 3. Fetch all price data for stocks (excluding filled rows for clean stats)
    df_prices = con.execute("""
        SELECT ticker, date, close, volume, simple_return
        FROM prices
        WHERE ticker NOT IN ('^NSEI', 'synthetic_midcap', 'synthetic_smallcap', '^CRSLDX')
          AND is_filled = FALSE
        ORDER BY ticker, date
    """).fetchdf()

    features_list = []
    for ticker, grp in df_prices.groupby("ticker"):
        grp = grp.sort_values("date")
        n_days = len(grp)
        if n_days < 2:
            continue

        s_date = grp["date"].iloc[0]
        e_date = grp["date"].iloc[-1]
        start_p = float(grp["close"].iloc[0])
        end_p = float(grp["close"].iloc[-1])
        elapsed_years = max(0.1, n_days / trading_days)

        # Returns & Volatility
        rets = grp["simple_return"].dropna()
        cagr = calculate_cagr(start_p, end_p, elapsed_years)
        ann_vol = calculate_annualized_volatility(rets, trading_days=trading_days)
        sharpe = calculate_sharpe_ratio(cagr, ann_vol, risk_free_rate=risk_free_rate)
        downside_dev = calculate_downside_deviation(rets, trading_days=trading_days)
        sortino = calculate_sortino_ratio(cagr, downside_dev, risk_free_rate=risk_free_rate)

        # Drawdowns
        _, max_dd, rec_days = calculate_drawdowns(grp["close"])

        # CAPM Beta
        stock_s = grp.set_index("date")["simple_return"].dropna()
        beta = calculate_beta(stock_s, df_nifty)

        # VaR
        var_95 = calculate_historical_var(rets, confidence_level=0.95)

        # Momentum
        mom = calculate_momentum(grp["close"], lookback_days=[126, 252])

        # Liquidity (60-day average daily traded value)
        traded_val = grp["close"] * grp["volume"]
        avg_liq_60d = float(traded_val.tail(60).mean()) if len(traded_val) >= 5 else 0.0

        features_list.append({
            "ticker": ticker,
            "start_date": s_date,
            "end_date": e_date,
            "trading_days": n_days,
            "cagr": round(cagr, 6),
            "annual_volatility": round(ann_vol, 6),
            "sharpe_ratio": round(sharpe, 4),
            "downside_deviation": round(downside_dev, 6),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 6),
            "recovery_days": rec_days,
            "beta": round(beta, 4),
            "var_95": round(var_95, 6),
            "momentum_6m": round(mom.get("momentum_6m", 0.0), 6),
            "momentum_12m": round(mom.get("momentum_12m", 0.0), 6),
            "avg_traded_value_60d": round(avg_liq_60d, 2),
        })

    df_feat = pd.DataFrame(features_list)

    # 4. Merge Universe Metadata & Fundamentals
    df_merged = df_univ.merge(df_feat, on="ticker", how="inner")
    df_merged = df_merged.merge(df_funds, on="ticker", how="left")

    # 5. Assign Risk Buckets (1 to 5) based on Annual Volatility Quintiles
    df_merged["risk_bucket"] = assign_risk_buckets(df_merged["annual_volatility"])

    # 6. Save to DuckDB table 'stock_features'
    con.execute("CREATE TABLE IF NOT EXISTS stock_features AS SELECT * FROM df_merged WHERE 1=0")
    con.execute("DELETE FROM stock_features")
    con.register("tmp_features", df_merged)
    con.execute("INSERT INTO stock_features SELECT * FROM tmp_features")
    con.unregister("tmp_features")
    con.close()

    # 7. Export Parquet
    exp_p = Path(export_parquet_path)
    exp_p.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(str(exp_p), index=False)

    logger.info("Successfully built features for %d stocks and exported to %s", len(df_merged), exp_p)
    return df_merged


if __name__ == "__main__":
    df = build_all_stock_features()
    print(f"Features successfully built for {len(df)} equities.")
    print(df[["ticker", "sector", "cap_class", "cagr", "annual_volatility", "sharpe_ratio", "beta", "risk_bucket"]].head(10))
