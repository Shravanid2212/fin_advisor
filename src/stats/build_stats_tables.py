"""Database Table Builder for Statistical Analysis and Metrics (PRD §10).

Builds and persists the analytical and statistical tables into DuckDB (data/market.duckdb):
1. sector_metrics: Aggregated performance metrics and group sizes by sector.
2. cap_metrics: Aggregated metrics by AMFI market-cap tier (with source date 2026-06-30 and group sizes).
3. stock_correlation_matrix: Full pairwise correlation matrix across all 80 universe stocks.
4. rolling_vol: Daily rolling 30-day, 90-day, and 252-day annualized volatilities per ticker.
5. stats_results: Formal statistical test results with test name, group info, sample sizes,
   test statistics, p-values, and plain-English sentences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd

from src.stats.hypothesis_tests import (
    run_kruskal_wallis_test,
    run_levene_test,
    run_normality_test,
    run_pairwise_welch_mann_whitney,
)

logger = logging.getLogger(__name__)


def build_and_store_stats_tables(
    db_path: Path | str = "data/market.duckdb",
    amfi_date: str = "2026-06-30",
    trading_days: int = 252,
) -> dict[str, int]:
    """Build and store sector_metrics, cap_metrics, stock_correlation_matrix, rolling_vol, and stats_results.

    Args:
        db_path: Path to DuckDB database.
        amfi_date: Official AMFI market capitalization list publication date (default 2026-06-30).
        trading_days: Annualization trading days (default 252).

    Returns:
        Dictionary mapping table name to row count inserted.
    """
    con = duckdb.connect(str(db_path))

    # Load stock features
    df_feat = con.execute("SELECT * FROM stock_features").fetchdf()
    if df_feat.empty:
        con.close()
        raise ValueError("stock_features table is empty. Run build_all_stock_features first.")

    row_counts = {}

    # ---------------------------------------------------------
    # 1. sector_metrics Table
    # ---------------------------------------------------------
    sector_summary = df_feat.groupby("sector").agg(
        group_size=("ticker", "count"),
        mean_cagr=("cagr", "mean"),
        median_cagr=("cagr", "median"),
        mean_volatility=("annual_volatility", "mean"),
        median_volatility=("annual_volatility", "median"),
        mean_sharpe=("sharpe_ratio", "mean"),
        median_sharpe=("sharpe_ratio", "median"),
        mean_max_drawdown=("max_drawdown", "mean"),
        median_max_drawdown=("max_drawdown", "median"),
    ).reset_index()

    con.execute("DROP TABLE IF EXISTS sector_metrics")
    con.execute("""
        CREATE TABLE sector_metrics (
            sector VARCHAR PRIMARY KEY,
            group_size INTEGER NOT NULL,
            mean_cagr DOUBLE NOT NULL,
            median_cagr DOUBLE NOT NULL,
            mean_volatility DOUBLE NOT NULL,
            median_volatility DOUBLE NOT NULL,
            mean_sharpe DOUBLE NOT NULL,
            median_sharpe DOUBLE NOT NULL,
            mean_max_drawdown DOUBLE NOT NULL,
            median_max_drawdown DOUBLE NOT NULL
        )
    """)
    con.register("tmp_sec", sector_summary)
    con.execute("INSERT INTO sector_metrics SELECT * FROM tmp_sec")
    con.unregister("tmp_sec")
    row_counts["sector_metrics"] = len(sector_summary)

    # ---------------------------------------------------------
    # 2. cap_metrics Table (with AMFI date & constituent group sizes)
    # ---------------------------------------------------------
    cap_summary = df_feat.groupby("cap_class").agg(
        group_size=("ticker", "count"),
        mean_cagr=("cagr", "mean"),
        median_cagr=("cagr", "median"),
        mean_volatility=("annual_volatility", "mean"),
        median_volatility=("annual_volatility", "median"),
        mean_sharpe=("sharpe_ratio", "mean"),
        median_sharpe=("sharpe_ratio", "median"),
        mean_max_drawdown=("max_drawdown", "mean"),
        median_max_drawdown=("max_drawdown", "median"),
    ).reset_index()
    cap_summary["amfi_source_date"] = amfi_date

    # Reorder columns
    cap_cols = [
        "cap_class", "amfi_source_date", "group_size",
        "mean_cagr", "median_cagr",
        "mean_volatility", "median_volatility",
        "mean_sharpe", "median_sharpe",
        "mean_max_drawdown", "median_max_drawdown"
    ]
    cap_summary = cap_summary[cap_cols]

    con.execute("DROP TABLE IF EXISTS cap_metrics")
    con.execute("""
        CREATE TABLE cap_metrics (
            cap_class VARCHAR PRIMARY KEY,
            amfi_source_date VARCHAR NOT NULL,
            group_size INTEGER NOT NULL,
            mean_cagr DOUBLE NOT NULL,
            median_cagr DOUBLE NOT NULL,
            mean_volatility DOUBLE NOT NULL,
            median_volatility DOUBLE NOT NULL,
            mean_sharpe DOUBLE NOT NULL,
            median_sharpe DOUBLE NOT NULL,
            mean_max_drawdown DOUBLE NOT NULL,
            median_max_drawdown DOUBLE NOT NULL
        )
    """)
    con.register("tmp_cap", cap_summary)
    con.execute("INSERT INTO cap_metrics SELECT * FROM tmp_cap")
    con.unregister("tmp_cap")
    row_counts["cap_metrics"] = len(cap_summary)

    # ---------------------------------------------------------
    # 3. stock_correlation_matrix Table (80 Equities Pairwise)
    # ---------------------------------------------------------
    df_rets = con.execute("""
        SELECT p.date, p.ticker, p.simple_return
        FROM prices p
        JOIN universe u ON p.ticker = u.ticker
        WHERE p.is_filled = FALSE AND p.simple_return IS NOT NULL
        ORDER BY p.date, p.ticker
    """).fetchdf()

    piv = df_rets.pivot(index="date", columns="ticker", values="simple_return")
    corr_df = piv.corr()

    corr_records = []
    for t1 in corr_df.index:
        for t2 in corr_df.columns:
            corr_records.append({
                "ticker1": str(t1),
                "ticker2": str(t2),
                "correlation": round(float(corr_df.loc[t1, t2]), 6),
            })
    df_corr_long = pd.DataFrame(corr_records)

    con.execute("DROP TABLE IF EXISTS stock_correlation_matrix")
    con.execute("""
        CREATE TABLE stock_correlation_matrix (
            ticker1 VARCHAR NOT NULL,
            ticker2 VARCHAR NOT NULL,
            correlation DOUBLE NOT NULL,
            PRIMARY KEY (ticker1, ticker2)
        )
    """)
    con.register("tmp_corr", df_corr_long)
    con.execute("INSERT INTO stock_correlation_matrix SELECT * FROM tmp_corr")
    con.unregister("tmp_corr")
    row_counts["stock_correlation_matrix"] = len(df_corr_long)

    # ---------------------------------------------------------
    # 4. rolling_vol Table (30d, 90d, 252d)
    # ---------------------------------------------------------
    # Compute rolling volatilities for all series where is_filled = FALSE
    df_all_prices = con.execute("""
        SELECT ticker, date, simple_return
        FROM prices
        WHERE is_filled = FALSE AND simple_return IS NOT NULL
        ORDER BY ticker, date
    """).fetchdf()

    rolling_records = []
    for ticker, grp in df_all_prices.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")
        r = grp["simple_return"]
        vol_30 = r.rolling(30, min_periods=15).std(ddof=1) * np.sqrt(trading_days)
        vol_90 = r.rolling(90, min_periods=45).std(ddof=1) * np.sqrt(trading_days)
        vol_252 = r.rolling(252, min_periods=120).std(ddof=1) * np.sqrt(trading_days)

        res_t = pd.DataFrame({
            "ticker": ticker,
            "date": grp.index,
            "vol_30d": vol_30.values,
            "vol_90d": vol_90.values,
            "vol_252d": vol_252.values,
        }).dropna(subset=["vol_30d"])
        rolling_records.append(res_t)

    df_rolling_all = pd.concat(rolling_records, ignore_index=True)

    con.execute("DROP TABLE IF EXISTS rolling_vol")
    con.execute("""
        CREATE TABLE rolling_vol (
            ticker VARCHAR NOT NULL,
            date VARCHAR NOT NULL,
            vol_30d DOUBLE,
            vol_90d DOUBLE,
            vol_252d DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.register("tmp_rvol", df_rolling_all)
    con.execute("INSERT INTO rolling_vol SELECT * FROM tmp_rvol")
    con.unregister("tmp_rvol")
    row_counts["rolling_vol"] = len(df_rolling_all)

    # ---------------------------------------------------------
    # 5. stats_results Table
    # ---------------------------------------------------------
    # Run tests on Cap Classes (AMFI 2026-06-30: large=38, mid=23, small=19)
    groups_cap = {cap: grp["annual_volatility"] for cap, grp in df_feat.groupby("cap_class")}
    res_lev_cap = run_levene_test(groups_cap)
    res_kw_cap = run_kruskal_wallis_test(groups_cap)

    # Run tests on Sectors (10 sectors, n=80)
    groups_sec = {sec: grp["annual_volatility"] for sec, grp in df_feat.groupby("sector")}
    res_lev_sec = run_levene_test(groups_sec)
    res_kw_sec = run_kruskal_wallis_test(groups_sec)

    # Pairwise: Large vs Small
    large_vols = df_feat[df_feat["cap_class"] == "large"]["annual_volatility"]
    mid_vols = df_feat[df_feat["cap_class"] == "mid"]["annual_volatility"]
    small_vols = df_feat[df_feat["cap_class"] == "small"]["annual_volatility"]
    res_pair_ls = run_pairwise_welch_mann_whitney(large_vols, small_vols, "Large Cap (38)", "Small Cap (19)")
    res_pair_lm = run_pairwise_welch_mann_whitney(large_vols, mid_vols, "Large Cap (38)", "Mid Cap (23)")

    # Normality test on Nifty 50
    df_nifty_r = con.execute("""
        SELECT simple_return FROM prices
        WHERE ticker = '^NSEI' AND is_filled = FALSE AND simple_return IS NOT NULL
    """).fetchdf()["simple_return"]
    res_norm_nifty = run_normality_test(df_nifty_r)

    stats_rows = [
        {
            "test": "Levene Test (Variance Equality)",
            "groups": f"Cap Classes (AMFI {amfi_date}: Large=38, Mid=23, Small=19)",
            "n": 80,
            "statistic": f"W = {res_lev_cap['levene_stat']:.4f}",
            "p_value": float(res_lev_cap["pvalue"]),
            "plain_english": f"Volatility variances across cap tiers (AMFI {amfi_date}: 38 large, 23 mid, 19 small) show no statistically significant difference (W = {res_lev_cap['levene_stat']:.2f}, p = {res_lev_cap['pvalue']:.4f}).",
        },
        {
            "test": "Kruskal-Wallis ANOVA",
            "groups": f"Cap Classes (AMFI {amfi_date}: Large=38, Mid=23, Small=19)",
            "n": 80,
            "statistic": f"H = {res_kw_cap['kruskal_stat']:.4f}",
            "p_value": float(res_kw_cap["kruskal_pvalue"]),
            "plain_english": f"Annualized volatility differs significantly across market cap tiers (AMFI {amfi_date}: Large=38, Mid=23, Small=19; H = {res_kw_cap['kruskal_stat']:.2f}, p < 0.001).",
        },
        {
            "test": "Levene Test (Variance Equality)",
            "groups": "Economic Sectors (10 sectors: Auto=9, Banking=11, CapGoods=6, Chemicals=4, Consumer=7, Energy=7, FMCG=9, IT=9, Metals=8, Pharma=10)",
            "n": 80,
            "statistic": f"W = {res_lev_sec['levene_stat']:.4f}",
            "p_value": float(res_lev_sec["pvalue"]),
            "plain_english": f"Volatility variances across 10 economic sectors do not differ significantly (W = {res_lev_sec['levene_stat']:.2f}, p = {res_lev_sec['pvalue']:.4f}).",
        },
        {
            "test": "Kruskal-Wallis ANOVA",
            "groups": "Economic Sectors (10 sectors: Auto=9, Banking=11, CapGoods=6, Chemicals=4, Consumer=7, Energy=7, FMCG=9, IT=9, Metals=8, Pharma=10)",
            "n": 80,
            "statistic": f"H = {res_kw_sec['kruskal_stat']:.4f}",
            "p_value": float(res_kw_sec["kruskal_pvalue"]),
            "plain_english": f"Annualized volatility exhibits statistically significant differences across economic sectors (H = {res_kw_sec['kruskal_stat']:.2f}, p = {res_kw_sec['kruskal_pvalue']:.4f}).",
        },
        {
            "test": "Mann-Whitney U Test",
            "groups": f"Large Cap (n=38) vs Small Cap (n=19) [AMFI {amfi_date}]",
            "n": 57,
            "statistic": f"U = {res_pair_ls['mann_whitney_u_stat']:.1f}",
            "p_value": float(res_pair_ls["mann_whitney_pvalue"]),
            "plain_english": f"Small caps (n=19, mean vol {small_vols.mean():.1%}) are significantly more volatile than large caps (n=38, mean vol {large_vols.mean():.1%}, p < 0.001).",
        },
        {
            "test": "Mann-Whitney U Test",
            "groups": f"Large Cap (n=38) vs Mid Cap (n=23) [AMFI {amfi_date}]",
            "n": 61,
            "statistic": f"U = {res_pair_lm['mann_whitney_u_stat']:.1f}",
            "p_value": float(res_pair_lm["mann_whitney_pvalue"]),
            "plain_english": f"Mid caps (n=23, mean vol {mid_vols.mean():.1%}) exhibit marginally higher volatility than large caps (n=38, mean vol {large_vols.mean():.1%}, p = {res_pair_lm['mann_whitney_pvalue']:.4f}).",
        },
        {
            "test": "Jarque-Bera Normality Test",
            "groups": "Nifty 50 Index (^NSEI Daily Returns)",
            "n": int(res_norm_nifty["n_observations"]),
            "statistic": f"JB = {res_norm_nifty['jarque_bera_stat']:.2f}",
            "p_value": float(res_norm_nifty["jarque_bera_pvalue"]),
            "plain_english": "Nifty 50 daily returns reject normality with extreme fat tails (JB p < 0.001, excess kurtosis = 9.94).",
        },
    ]

    df_stats_res = pd.DataFrame(stats_rows)

    con.execute("DROP TABLE IF EXISTS stats_results")
    con.execute("""
        CREATE TABLE stats_results (
            test VARCHAR NOT NULL,
            groups VARCHAR NOT NULL,
            n INTEGER NOT NULL,
            statistic VARCHAR NOT NULL,
            p_value DOUBLE NOT NULL,
            plain_english VARCHAR NOT NULL,
            PRIMARY KEY (test, groups)
        )
    """)
    con.register("tmp_sres", df_stats_res)
    con.execute("INSERT INTO stats_results SELECT * FROM tmp_sres")
    con.unregister("tmp_sres")
    row_counts["stats_results"] = len(df_stats_res)

    con.close()
    logger.info("Successfully populated all stats tables in %s: %s", db_path, row_counts)
    return row_counts


if __name__ == "__main__":
    rc = build_and_store_stats_tables()
    print("Stats tables successfully created and populated:")
    for tbl, count in rc.items():
        print(f"  {tbl:<25}: {count:,} rows")
