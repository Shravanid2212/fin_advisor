"""Application data service and read-only queries against DuckDB.

Adheres strictly to:
- AGENTS.md Hard Rules: App is read-only against the database.
- PRD Section 15.1, 15.2, 15.3: Pure data retrieval and aggregation. No UI code here.
- Amendment 2: Reuse existing functions in src/features/metrics.py.
- Amendment 3: 'Most volatile (full history)', 'equal-weighted, last 1 year', synthetic tooltips.
- Amendment 4: Use st.cache_data, graceful error handling for missing/stale DB, graceful empty filters.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.features.metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_cagr,
    calculate_drawdowns,
    calculate_historical_var,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.ingest.load_db import get_data_as_of

BENCHMARK_TICKERS = {"^NSEI", "^CRSLDX", "synthetic_midcap", "synthetic_smallcap"}


def get_db_connection(db_path: Path | str = "data/market.duckdb") -> duckdb.DuckDBPyConnection | None:
    """Safely connect read-only to DuckDB database."""
    p = Path(db_path)
    if not p.exists():
        return None
    try:
        return duckdb.connect(str(p), read_only=True)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def check_db_health(db_path: str = "data/market.duckdb") -> dict[str, Any]:
    """Check database file existence and essential table presence."""
    p = Path(db_path)
    if not p.exists():
        return {
            "is_healthy": False,
            "message": f"Database file '{db_path}' not found. Please run the ingestion pipeline.",
            "data_as_of": "N/A",
        }

    try:
        con = duckdb.connect(str(p), read_only=True)
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        req_tables = {"prices", "stock_features", "universe", "stats_results"}
        missing = req_tables - set(tables)
        if missing:
            con.close()
            return {
                "is_healthy": False,
                "message": f"Database is missing required tables: {missing}",
                "data_as_of": "N/A",
            }

        price_count = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        data_as_of = get_data_as_of(p, con=con)
        con.close()

        if price_count == 0:
            return {
                "is_healthy": False,
                "message": "The prices table in the database is empty.",
                "data_as_of": "N/A",
            }

        return {
            "is_healthy": True,
            "message": "Database healthy and accessible.",
            "data_as_of": data_as_of,
        }
    except Exception as exc:
        return {
            "is_healthy": False,
            "message": f"Failed to access database: {exc}",
            "data_as_of": "N/A",
        }


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_universe_tickers(db_path: str = "data/market.duckdb") -> list[dict[str, str]]:
    """Retrieve all equity stock tickers with names and sectors."""
    con = get_db_connection(db_path)
    if con is None:
        return []

    try:
        df = con.execute(
            """
            SELECT ticker, name, sector, cap_class
            FROM universe
            WHERE ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')
            ORDER BY ticker
            """
        ).fetchdf()
        con.close()
        return df.to_dict(orient="records")
    except Exception:
        con.close()
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_home_kpis_data(db_path: str = "data/market.duckdb") -> dict[str, Any]:
    """Retrieve KPIs and highlight cards for the Home page."""
    con = get_db_connection(db_path)
    if con is None:
        return {
            "stocks_count": 0,
            "years_of_data": 0.0,
            "sectors_count": 0,
            "median_volatility": 0.0,
            "most_volatile": None,
            "lowest_drawdown": None,
            "highest_sharpe": None,
            "data_as_of": "N/A",
        }

    try:
        # Stock features excluding benchmarks
        query = """
            SELECT ticker, name, sector, cap_class, cagr, annual_volatility, 
                   sharpe_ratio, max_drawdown, trading_days
            FROM stock_features
            WHERE ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')
        """
        feats = con.execute(query).fetchdf()

        # Date range for total years
        dates = con.execute(
            "SELECT MIN(date), MAX(date) FROM prices WHERE is_filled = FALSE"
        ).fetchone()
        con.close()

        if feats.empty:
            return {
                "stocks_count": 0,
                "years_of_data": 0.0,
                "sectors_count": 0,
                "median_volatility": 0.0,
                "most_volatile": None,
                "lowest_drawdown": None,
                "highest_sharpe": None,
                "data_as_of": "N/A",
            }

        start_dt = pd.to_datetime(dates[0])
        end_dt = pd.to_datetime(dates[1])
        years = round((end_dt - start_dt).days / 365.25, 1)

        stocks_count = len(feats)
        sectors_count = feats["sector"].nunique()
        median_vol = float(feats["annual_volatility"].median())

        # Highlights
        most_vol = feats.sort_values(by="annual_volatility", ascending=False).iloc[0].to_dict()
        # Lowest drawdown: closest to zero (least severe drawdown)
        lowest_dd = feats.sort_values(by="max_drawdown", ascending=False).iloc[0].to_dict()
        highest_sharpe = feats.sort_values(by="sharpe_ratio", ascending=False).iloc[0].to_dict()

        return {
            "stocks_count": stocks_count,
            "years_of_data": years,
            "sectors_count": sectors_count,
            "median_volatility": median_vol,
            "most_volatile": most_vol,
            "lowest_drawdown": lowest_dd,
            "highest_sharpe": highest_sharpe,
            "data_as_of": str(dates[1]),
        }
    except Exception:
        con.close()
        return {
            "stocks_count": 0,
            "years_of_data": 0.0,
            "sectors_count": 0,
            "median_volatility": 0.0,
            "most_volatile": None,
            "lowest_drawdown": None,
            "highest_sharpe": None,
            "data_as_of": "N/A",
        }


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty_rebased_series(horizon: str = "1Y", db_path: str = "data/market.duckdb") -> pd.DataFrame:
    """Retrieve rebased price series (100 at start) for Nifty 50 (^NSEI)."""
    con = get_db_connection(db_path)
    if con is None:
        return pd.DataFrame(columns=["date", "rebased_price"])

    try:
        max_date_str = con.execute("SELECT MAX(date) FROM prices WHERE ticker = '^NSEI'").fetchone()[0]
        if not max_date_str:
            con.close()
            return pd.DataFrame(columns=["date", "rebased_price"])

        max_dt = pd.to_datetime(max_date_str)
        if horizon == "1Y":
            min_dt = max_dt - pd.DateOffset(years=1)
        elif horizon == "5Y":
            min_dt = max_dt - pd.DateOffset(years=5)
        else:  # 10Y or Max
            min_dt = max_dt - pd.DateOffset(years=11)

        min_date_str = min_dt.strftime("%Y-%m-%d")

        query = f"""
            SELECT date, close
            FROM prices
            WHERE ticker = '^NSEI' 
              AND is_filled = FALSE
              AND date >= '{min_date_str}'
            ORDER BY date ASC
        """
        df = con.execute(query).fetchdf()
        con.close()

        if df.empty or len(df) < 2:
            return pd.DataFrame(columns=["date", "rebased_price"])

        first_close = df["close"].iloc[0]
        if first_close > 0:
            df["rebased_price"] = (df["close"] / first_close) * 100.0
        else:
            df["rebased_price"] = 100.0

        return df[["date", "rebased_price", "close"]]
    except Exception:
        con.close()
        return pd.DataFrame(columns=["date", "rebased_price"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_1y_returns(db_path: str = "data/market.duckdb") -> pd.DataFrame:
    """Retrieve equal-weighted average return for each sector over the last 1 year.

    Label: 'equal-weighted, last 1 year'.
    """
    con = get_db_connection(db_path)
    if con is None:
        return pd.DataFrame(columns=["sector", "return_1y"])

    try:
        max_date_str = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        one_year_ago = (pd.to_datetime(max_date_str) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

        query = f"""
            WITH stock_dates AS (
                SELECT p.ticker, u.sector, p.date, p.close,
                       ROW_NUMBER() OVER(PARTITION BY p.ticker ORDER BY p.date ASC) as rn_first,
                       ROW_NUMBER() OVER(PARTITION BY p.ticker ORDER BY p.date DESC) as rn_last
                FROM prices p
                JOIN universe u ON p.ticker = u.ticker
                WHERE p.is_filled = FALSE
                  AND p.date >= '{one_year_ago}'
                  AND p.ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')
            ),
            stock_rets AS (
                SELECT f.ticker, f.sector,
                       (l.close - f.close) / f.close as ret_1y
                FROM stock_dates f
                JOIN stock_dates l ON f.ticker = l.ticker
                WHERE f.rn_first = 1 AND l.rn_last = 1
            )
            SELECT sector, 
                   AVG(ret_1y) as return_1y,
                   COUNT(*) as stock_count
            FROM stock_rets
            GROUP BY sector
            ORDER BY return_1y DESC
        """
        df = con.execute(query).fetchdf()
        con.close()
        return df
    except Exception:
        con.close()
        return pd.DataFrame(columns=["sector", "return_1y"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_explore_market_data(
    selected_sectors: list[str] | None,
    selected_caps: list[str] | None,
    period: str = "3Y",
    db_path: str = "data/market.duckdb",
) -> dict[str, Any]:
    """Retrieve data for the Explore Market page."""
    con = get_db_connection(db_path)
    empty_result = {
        "rebased_df": pd.DataFrame(),
        "scatter_df": pd.DataFrame(),
        "sector_risk_df": pd.DataFrame(),
        "corr_matrix_df": pd.DataFrame(),
        "stats_cards": [],
    }
    if con is None:
        return empty_result

    try:
        # Load all stock features
        feats_df = con.execute("SELECT * FROM stock_features").fetchdf()

        # Handle empty filter selections gracefully
        if selected_sectors is not None and len(selected_sectors) == 0:
            con.close()
            return empty_result
        if selected_caps is not None and len(selected_caps) == 0:
            con.close()
            return empty_result

        # Filter stock features
        mask = ~feats_df["ticker"].isin(BENCHMARK_TICKERS)
        if selected_sectors:
            mask = mask & feats_df["sector"].isin(selected_sectors)
        if selected_caps:
            mask = mask & feats_df["cap_class"].isin(selected_caps)

        filtered_feats = feats_df[mask].copy().reset_index(drop=True)

        # 1. Scatter data
        scatter_df = filtered_feats[[
            "ticker", "name", "sector", "cap_class", "annual_volatility", 
            "cagr", "sharpe_ratio", "market_cap", "risk_bucket"
        ]].copy()

        # 2. Sector risk bars (volatility and VaR 95% by sector)
        if not filtered_feats.empty:
            sector_risk_df = filtered_feats.groupby("sector").agg(
                mean_volatility=("annual_volatility", "mean"),
                mean_var95=("var_95", "mean"),
                stock_count=("ticker", "count"),
            ).reset_index().sort_values(by="mean_volatility", ascending=False)
        else:
            sector_risk_df = pd.DataFrame()

        # 3. Statistical insight cards from stats_results
        stats_df = con.execute("SELECT test, groups, statistic, p_value, plain_english FROM stats_results").fetchdf()
        stats_cards = stats_df.to_dict(orient="records")

        # 4. Correlation matrix for filtered stocks
        tickers_list = filtered_feats["ticker"].tolist()
        corr_matrix_df = pd.DataFrame()
        if len(tickers_list) >= 2:
            # Limit correlation matrix to top 15 by market cap for visual clarity
            top_tickers = filtered_feats.sort_values(by="market_cap", ascending=False)["ticker"].head(15).tolist()
            placeholders = ",".join(f"'{t}'" for t in top_tickers)
            query_corr = f"""
                SELECT ticker1, ticker2, correlation
                FROM stock_correlation_matrix
                WHERE ticker1 IN ({placeholders}) AND ticker2 IN ({placeholders})
            """
            corr_raw = con.execute(query_corr).fetchdf()
            if not corr_raw.empty:
                corr_matrix_df = corr_raw.pivot(index="ticker1", columns="ticker2", values="correlation")

        # 5. Rebased growth by sector and benchmarks
        max_date_str = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        max_dt = pd.to_datetime(max_date_str)
        if period == "1Y":
            min_dt = max_dt - pd.DateOffset(years=1)
        elif period == "3Y":
            min_dt = max_dt - pd.DateOffset(years=3)
        elif period == "5Y":
            min_dt = max_dt - pd.DateOffset(years=5)
        else:
            min_dt = max_dt - pd.DateOffset(years=11)
        min_date_str = min_dt.strftime("%Y-%m-%d")

        # Query sector price series (mean price rebased)
        rebased_dict: dict[str, pd.Series] = {}

        if not filtered_feats.empty:
            sectors_to_plot = filtered_feats["sector"].unique().tolist()
            for sec in sectors_to_plot:
                sec_tickers = filtered_feats[filtered_feats["sector"] == sec]["ticker"].tolist()
                t_str = ",".join(f"'{t}'" for t in sec_tickers)
                sec_p = con.execute(f"""
                    SELECT date, AVG(close) as avg_close
                    FROM prices
                    WHERE ticker IN ({t_str}) AND is_filled = FALSE AND date >= '{min_date_str}'
                    GROUP BY date
                    ORDER BY date ASC
                """).fetchdf()
                if not sec_p.empty and len(sec_p) > 1:
                    base = sec_p["avg_close"].iloc[0]
                    if base > 0:
                        s_series = (sec_p["avg_close"] / base) * 100.0
                        s_series.index = pd.to_datetime(sec_p["date"])
                        rebased_dict[sec] = s_series

        # Query benchmarks as reference lines
        for bm_ticker, bm_name in [
            ("^NSEI", "Nifty 50 (Ref)"),
            ("synthetic_midcap", "Mid-cap average"),
            ("synthetic_smallcap", "Small-cap average"),
        ]:
            bm_p = con.execute(f"""
                SELECT date, close
                FROM prices
                WHERE ticker = '{bm_ticker}' AND is_filled = FALSE AND date >= '{min_date_str}'
                ORDER BY date ASC
            """).fetchdf()
            if not bm_p.empty and len(bm_p) > 1:
                base = bm_p["close"].iloc[0]
                if base > 0:
                    s_series = (bm_p["close"] / base) * 100.0
                    s_series.index = pd.to_datetime(bm_p["date"])
                    rebased_dict[bm_name] = s_series

        rebased_df = pd.DataFrame(rebased_dict).dropna(how="all")
        con.close()

        return {
            "rebased_df": rebased_df,
            "scatter_df": scatter_df,
            "sector_risk_df": sector_risk_df,
            "corr_matrix_df": corr_matrix_df,
            "stats_cards": stats_cards,
        }
    except Exception:
        con.close()
        return empty_result


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_detail_data(
    ticker: str,
    period: str = "1Y",
    db_path: str = "data/market.duckdb",
) -> dict[str, Any] | None:
    """Retrieve full analytics for a single stock."""
    con = get_db_connection(db_path)
    if con is None:
        return None

    try:
        # 1. Feature row
        f_row = con.execute(f"SELECT * FROM stock_features WHERE ticker = '{ticker}'").fetchdf()
        if f_row.empty:
            con.close()
            return None
        stock_feat = f_row.iloc[0].to_dict()

        # 2. Fundamentals snapshot
        fund_row = con.execute(
            f"SELECT * FROM fundamentals_snapshots WHERE ticker = '{ticker}'"
        ).fetchdf()
        fund_dict = fund_row.iloc[0].to_dict() if not fund_row.empty else {}

        # 3. Price history
        prices_df = con.execute(
            f"""
            SELECT date, open, high, low, close, volume, is_filled
            FROM prices
            WHERE ticker = '{ticker}'
            ORDER BY date ASC
            """
        ).fetchdf()

        if prices_df.empty:
            con.close()
            return None

        prices_df["date"] = pd.to_datetime(prices_df["date"])
        prices_df = prices_df.sort_values(by="date").reset_index(drop=True)

        # Filter by requested display period
        max_dt = prices_df["date"].max()
        if period == "1M":
            min_dt = max_dt - pd.DateOffset(months=1)
        elif period == "6M":
            min_dt = max_dt - pd.DateOffset(months=6)
        elif period == "1Y":
            min_dt = max_dt - pd.DateOffset(years=1)
        elif period == "5Y":
            min_dt = max_dt - pd.DateOffset(years=5)
        elif period == "10Y":
            min_dt = max_dt - pd.DateOffset(years=10)
        else:  # Max
            min_dt = prices_df["date"].min()

        filtered_prices = prices_df[prices_df["date"] >= min_dt].copy().reset_index(drop=True)

        # Latest close and 1D return
        clean_prices = prices_df[~prices_df["is_filled"]].copy().reset_index(drop=True)
        latest_close = float(clean_prices["close"].iloc[-1])
        prev_close = float(clean_prices["close"].iloc[-2]) if len(clean_prices) >= 2 else latest_close
        one_day_ret = (latest_close - prev_close) / prev_close if prev_close > 0 else 0.0

        # Calculate underwater drawdown series using pure function
        full_close_series = pd.Series(clean_prices["close"].values, index=clean_prices["date"])
        dd_series, max_dd, rec_days = calculate_drawdowns(full_close_series)
        dd_df = pd.DataFrame({"date": clean_prices["date"], "drawdown": dd_series.values})
        filtered_dd = dd_df[dd_df["date"] >= min_dt].copy().reset_index(drop=True)

        # Daily returns for histogram
        daily_rets = full_close_series.pct_change().dropna()

        # 4. Sector peers
        sector = stock_feat["sector"]
        peers_df = con.execute(f"""
            SELECT ticker, name, cap_class, annual_volatility, sharpe_ratio, risk_bucket
            FROM stock_features
            WHERE sector = '{sector}' AND ticker != '{ticker}'
            ORDER BY annual_volatility ASC
        """).fetchdf()

        # Pairwise correlations with this stock
        if not peers_df.empty:
            peer_tickers = peers_df["ticker"].tolist()
            t_str = ",".join(f"'{t}'" for t in peer_tickers)
            corrs = con.execute(f"""
                SELECT ticker2, correlation
                FROM stock_correlation_matrix
                WHERE ticker1 = '{ticker}' AND ticker2 IN ({t_str})
            """).fetchdf()
            corr_map = dict(zip(corrs["ticker2"], corrs["correlation"])) if not corrs.empty else {}
            peers_df["correlation"] = peers_df["ticker"].map(corr_map).fillna(0.0)
        else:
            peers_df["correlation"] = []

        con.close()

        return {
            "ticker": ticker,
            "name": stock_feat["name"],
            "sector": stock_feat["sector"],
            "cap_class": stock_feat["cap_class"],
            "risk_bucket": int(stock_feat["risk_bucket"]),
            "latest_close": latest_close,
            "prev_close": prev_close,
            "one_day_ret": one_day_ret,
            "data_date": str(clean_prices["date"].iloc[-1].date()),
            "prices_df": filtered_prices,
            "full_prices_df": clean_prices,
            "drawdown_df": filtered_dd,
            "daily_returns": daily_rets,
            "stats": {
                "cagr": float(stock_feat["cagr"]),
                "annual_volatility": float(stock_feat["annual_volatility"]),
                "sharpe_ratio": float(stock_feat["sharpe_ratio"]),
                "sortino_ratio": float(stock_feat["sortino_ratio"]),
                "max_drawdown": float(stock_feat["max_drawdown"]),
                "recovery_days": int(stock_feat["recovery_days"]),
                "beta": float(stock_feat["beta"]),
                "var_95": float(stock_feat["var_95"]),
            },
            "fundamentals": {
                "market_cap": fund_dict.get("market_cap", stock_feat.get("market_cap")),
                "trailing_pe": fund_dict.get("trailing_pe", stock_feat.get("trailing_pe")),
                "forward_pe": fund_dict.get("forward_pe", stock_feat.get("forward_pe")),
                "dividend_yield": fund_dict.get("dividend_yield", stock_feat.get("dividend_yield")),
                "snapshot_date": fund_dict.get("snapshot_date", "Latest Snapshot"),
            },
            "peers": peers_df,
        }
    except Exception:
        con.close()
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_screener_data(
    sectors: list[str] | None = None,
    caps: list[str] | None = None,
    buckets: list[int] | None = None,
    min_sharpe: float = -1.0,
    min_div_yield: float = 0.0,
    db_path: str = "data/market.duckdb",
) -> pd.DataFrame:
    """Filter universe stocks across multiple screening criteria.

    Args:
        sectors: List of economic sectors.
        caps: List of market cap tiers ('large', 'mid', 'small').
        buckets: List of risk buckets (1 to 5).
        min_sharpe: Minimum Sharpe ratio threshold.
        min_div_yield: Minimum dividend yield percentage threshold.
        db_path: Path to database.

    Returns:
        Filtered DataFrame with formatted attributes.
    """
    con = get_db_connection(db_path)
    if con is None:
        return pd.DataFrame()

    try:
        # If any multiselect is explicitly empty list, return empty DataFrame
        if (sectors is not None and len(sectors) == 0) or (caps is not None and len(caps) == 0) or (buckets is not None and len(buckets) == 0):
            con.close()
            return pd.DataFrame()

        df = con.execute("SELECT * FROM stock_features WHERE ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')").fetchdf()
        con.close()

        if df.empty:
            return pd.DataFrame()

        mask = pd.Series(True, index=df.index)

        if sectors:
            mask = mask & df["sector"].isin(sectors)
        if caps:
            mask = mask & df["cap_class"].isin(caps)
        if buckets:
            mask = mask & df["risk_bucket"].isin(buckets)
        if min_sharpe is not None:
            mask = mask & (df["sharpe_ratio"] >= min_sharpe)
        if min_div_yield is not None and min_div_yield > 0.0:
            mask = mask & (df["dividend_yield"].fillna(0.0) >= min_div_yield)

        res = df[mask].copy().sort_values(by="sharpe_ratio", ascending=False).reset_index(drop=True)
        return res
    except Exception:
        con.close()
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_stocks_comparison_data(
    tickers: list[str],
    horizon: str = "3Y",
    db_path: str = "data/market.duckdb",
) -> dict[str, Any]:
    """Retrieve rebased growth, metrics table, and correlation for 2 to 4 stocks.

    Args:
        tickers: List of 2 to 4 stock tickers.
        horizon: '1Y', '3Y', '5Y', or '10Y'.
        db_path: Path to database.

    Returns:
        Dict containing 'rebased_df', 'metrics_df', and 'corr_df'.
    """
    empty_result = {
        "rebased_df": pd.DataFrame(),
        "metrics_df": pd.DataFrame(),
        "corr_df": pd.DataFrame(),
    }
    if not tickers or len(tickers) < 2:
        return empty_result

    con = get_db_connection(db_path)
    if con is None:
        return empty_result

    try:
        # 1. Metrics for selected tickers
        t_clean = tickers[:4]
        t_str = ",".join(f"'{t}'" for t in t_clean)

        metrics_query = f"""
            SELECT ticker, name, sector, cap_class, risk_bucket, cagr, annual_volatility, 
                   sharpe_ratio, sortino_ratio, max_drawdown, beta, var_95, trailing_pe, dividend_yield
            FROM stock_features
            WHERE ticker IN ({t_str})
        """
        metrics_df = con.execute(metrics_query).fetchdf()

        # 2. Rebased price series
        max_date_str = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        max_dt = pd.to_datetime(max_date_str)
        if horizon == "1Y":
            min_dt = max_dt - pd.DateOffset(years=1)
        elif horizon == "3Y":
            min_dt = max_dt - pd.DateOffset(years=3)
        elif horizon == "5Y":
            min_dt = max_dt - pd.DateOffset(years=5)
        else:
            min_dt = max_dt - pd.DateOffset(years=11)
        min_date_str = min_dt.strftime("%Y-%m-%d")

        prices_query = f"""
            SELECT ticker, date, close
            FROM prices
            WHERE ticker IN ({t_str})
              AND is_filled = FALSE
              AND date >= '{min_date_str}'
            ORDER BY date ASC
        """
        raw_prices = con.execute(prices_query).fetchdf()

        rebased_dict: dict[str, pd.Series] = {}
        if not raw_prices.empty:
            for t in t_clean:
                sub = raw_prices[raw_prices["ticker"] == t].sort_values(by="date")
                if not sub.empty and len(sub) > 1:
                    base_price = sub["close"].iloc[0]
                    if base_price > 0:
                        s = (sub["close"] / base_price) * 100.0
                        s.index = pd.to_datetime(sub["date"])
                        rebased_dict[t] = s

        rebased_df = pd.DataFrame(rebased_dict).dropna(how="all")

        # 3. Correlation matrix
        corr_query = f"""
            SELECT ticker1, ticker2, correlation
            FROM stock_correlation_matrix
            WHERE ticker1 IN ({t_str}) AND ticker2 IN ({t_str})
        """
        corr_raw = con.execute(corr_query).fetchdf()
        corr_df = pd.DataFrame()
        if not corr_raw.empty:
            corr_df = corr_raw.pivot(index="ticker1", columns="ticker2", values="correlation")

        con.close()
        return {
            "rebased_df": rebased_df,
            "metrics_df": metrics_df,
            "corr_df": corr_df,
        }
    except Exception:
        con.close()
        return empty_result


def get_sample_portfolio_holdings() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return verified sample holdings matching the balanced fixture."""
    stock_rows = [
        {"ticker": "RELIANCE.NS", "quantity": 20, "avg_buy_price": 1200.0},
        {"ticker": "TCS.NS", "quantity": 10, "avg_buy_price": 2400.0},
        {"ticker": "SUNPHARMA.NS", "quantity": 20, "avg_buy_price": 1150.0},
        {"ticker": "FEDERALBNK.NS", "quantity": 100, "avg_buy_price": 180.0},
        {"ticker": "VOLTAS.NS", "quantity": 15, "avg_buy_price": 950.0},
        {"ticker": "JINDALSAW.NS", "quantity": 25, "avg_buy_price": 500.0},
    ]
    fund_rows = [
        {"category": "debt", "amount": 80000.0},
        {"category": "gold", "amount": 20000.0},
    ]
    return stock_rows, fund_rows


def get_portfolio_csv_template() -> str:
    """Generate CSV template string for portfolio upload."""
    return "ticker,quantity,avg_buy_price\nRELIANCE.NS,20,1200.00\nTCS.NS,10,2400.00\nHDFCBANK.NS,40,1400.00\nINFY.NS,30,1450.00\n"


@st.cache_data(ttl=3600, show_spinner=False)
def get_methodology_data(db_path: str = "data/market.duckdb") -> dict[str, Any]:
    """Retrieve methodology, statistics, cap/sector metrics, and dataset diagnostics from DuckDB.
    
    Guarantees every statistic and metric is queried directly from DuckDB tables.
    """
    con = get_db_connection(db_path)
    if con is None:
        return {}

    try:
        # 1. Dataset Summary
        p_row = con.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN is_extreme THEN 1 END) as extreme_count,
                COUNT(CASE WHEN is_filled THEN 1 END) as filled_count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM prices
        """).fetchone()

        total_rows = int(p_row[0]) if p_row else 0
        extreme_count = int(p_row[1]) if p_row else 0
        filled_count = int(p_row[2]) if p_row else 0
        min_date = str(p_row[3]) if p_row else ""
        max_date = str(p_row[4]) if p_row else ""

        u_cap = con.execute("""
            SELECT cap_class, COUNT(*) as count
            FROM universe
            WHERE ticker NOT LIKE '^%' AND ticker NOT LIKE 'synthetic_%'
            GROUP BY cap_class
            ORDER BY count DESC
        """).fetchdf()

        u_sec = con.execute("""
            SELECT sector, COUNT(*) as count
            FROM universe
            WHERE ticker NOT LIKE '^%' AND ticker NOT LIKE 'synthetic_%'
            GROUP BY sector
            ORDER BY count DESC
        """).fetchdf()

        equity_count = int(u_cap["count"].sum()) if not u_cap.empty else 0
        benchmark_count = int(con.execute("""
            SELECT COUNT(DISTINCT ticker) FROM prices WHERE ticker LIKE '^%' OR ticker LIKE 'synthetic_%'
        """).fetchone()[0])

        # 2. Formal Statistical Results Table
        stats_df = con.execute("""
            SELECT test, groups, n, statistic, p_value, plain_english
            FROM stats_results
        """).fetchdf()

        # 3. Cap Metrics Table
        cap_df = con.execute("""
            SELECT * FROM cap_metrics ORDER BY group_size DESC
        """).fetchdf()

        # 4. Sector Metrics Table
        sector_df = con.execute("""
            SELECT * FROM sector_metrics ORDER BY mean_cagr DESC
        """).fetchdf()

        # 5. Outlier Counterfactual Calculation (Directly from clean price returns)
        px_clean = con.execute("""
            SELECT ticker, simple_return
            FROM prices
            WHERE ticker NOT LIKE '^%' AND ticker NOT LIKE 'synthetic_%' AND is_filled = FALSE
            ORDER BY ticker, date
        """).fetchdf()

        con.close()

        total_days = 0
        total_removed = 0
        vols_real: list[float] = []
        vols_counter: list[float] = []

        if not px_clean.empty:
            for _, grp in px_clean.groupby("ticker"):
                r = grp["simple_return"].dropna().values
                n = len(r)
                if n < 50:
                    continue
                total_days += n
                v_real = float(np.std(r, ddof=1) * np.sqrt(252))
                vols_real.append(v_real)

                q25, q75 = np.percentile(r, [25, 75])
                iqr = q75 - q25
                lo = q25 - 1.5 * iqr
                hi = q75 + 1.5 * iqr
                mask = (r >= lo) & (r <= hi)
                r_counter = r[mask]

                total_removed += (n - len(r_counter))
                v_counter = float(np.std(r_counter, ddof=1) * np.sqrt(252))
                vols_counter.append(v_counter)

        real_mean_vol = float(np.mean(vols_real)) if vols_real else 0.0
        counter_mean_vol = float(np.mean(vols_counter)) if vols_counter else 0.0
        vol_understatement = ((real_mean_vol - counter_mean_vol) / real_mean_vol) if real_mean_vol > 0 else 0.0

        return {
            "dataset_summary": {
                "total_rows": total_rows,
                "extreme_count": extreme_count,
                "extreme_pct": (extreme_count / total_rows * 100.0) if total_rows > 0 else 0.0,
                "filled_count": filled_count,
                "filled_pct": (filled_count / total_rows * 100.0) if total_rows > 0 else 0.0,
                "min_date": min_date,
                "max_date": max_date,
                "equity_count": equity_count,
                "benchmark_count": benchmark_count,
                "cap_distribution": u_cap,
                "sector_distribution": u_sec,
            },
            "stats_results": stats_df,
            "cap_metrics": cap_df,
            "sector_metrics": sector_df,
            "counterfactual": {
                "total_clean_days": total_days,
                "iqr_removed_days": total_removed,
                "iqr_removed_pct": (total_removed / total_days * 100.0) if total_days > 0 else 0.0,
                "real_mean_vol": real_mean_vol,
                "counter_mean_vol": counter_mean_vol,
                "vol_understatement_pct": vol_understatement * 100.0,
            },
        }
    except Exception:
        if con:
            con.close()
        return {}


