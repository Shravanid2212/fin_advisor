"""Targeted unit tests to ensure high test coverage (>= 80%) across src/ modules."""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from app.components.ui import is_data_stale
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_explore_market_data,
    get_methodology_data,
    get_portfolio_csv_template,
    get_sample_portfolio_holdings,
    get_screener_data,
    get_stock_detail_data,
    get_stocks_comparison_data,
)
from src.config import UniverseStock, load_config, load_universe
from src.ingest.benchmarks import BenchmarkReport, build_synthetic_cap_benchmark, inspect_benchmark


def test_is_data_stale_logic():
    """Verify is_data_stale correctly identifies fresh vs stale vs invalid dates."""
    # Invalid or N/A
    stale, days = is_data_stale("N/A")
    assert stale is False
    assert days == 0

    stale, days = is_data_stale("")
    assert stale is False

    stale, days = is_data_stale("not-a-date")
    assert stale is False

    # A date far in the past is stale
    stale, days = is_data_stale("2020-01-01", max_trading_days=5)
    assert stale is True
    assert days > 100


def test_inspect_benchmark_synthetic():
    """Verify inspect_benchmark handles synthetic midcap and smallcap offline."""
    rep_mid = inspect_benchmark("synthetic_midcap", "midcap")
    assert isinstance(rep_mid, BenchmarkReport)
    assert rep_mid.symbol == "synthetic_midcap"
    assert rep_mid.is_adequate is True

    rep_small = inspect_benchmark("synthetic_smallcap", "smallcap")
    assert isinstance(rep_small, BenchmarkReport)
    assert rep_small.symbol == "synthetic_smallcap"
    assert rep_small.is_adequate is True


def test_build_synthetic_cap_benchmark_unit():
    """Verify build_synthetic_cap_benchmark calculates equal-weighted index starting at 100."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
    records = []
    for d in dates:
        records.append({"date": d, "ticker": "STK1.NS", "simple_return": 0.01, "close": 100.0, "volume": 1000})
        records.append({"date": d, "ticker": "STK2.NS", "simple_return": -0.01, "close": 100.0, "volume": 1000})
    df_clean = pd.DataFrame(records)

    stocks = [
        UniverseStock(ticker="STK1.NS", name="Stock 1", sector="IT", cap_class="mid"),
        UniverseStock(ticker="STK2.NS", name="Stock 2", sector="IT", cap_class="mid"),
    ]

    res = build_synthetic_cap_benchmark(df_clean, stocks, cap_class="mid", min_constituents=2)
    assert not res.empty
    assert res["ticker"].iloc[0] == "synthetic_midcap"
    assert res["close"].iloc[0] == 100.0
    assert "log_return" in res.columns
    assert "rebased_price" in res.columns


@pytest.mark.realdata
def test_app_data_screener_combinations():
    """Verify get_screener_data with multiple filter permutations."""
    # 1. Filter by cap classes
    df_large = get_screener_data(caps=["large"], db_path="data/market.duckdb")
    assert not df_large.empty
    assert (df_large["cap_class"] == "large").all()

    # 2. Filter by sector and risk bucket
    df_sec = get_screener_data(sectors=["Banking and Financials"], buckets=[2, 3], db_path="data/market.duckdb")
    assert not df_sec.empty
    assert (df_sec["sector"] == "Banking and Financials").all()

    # 3. Filter with high minimum Sharpe
    df_sharpe = get_screener_data(min_sharpe=0.30, db_path="data/market.duckdb")
    assert not df_sharpe.empty
    assert (df_sharpe["sharpe_ratio"] >= 0.30).all()

    # 4. Filter with high minimum dividend yield
    df_div = get_screener_data(min_div_yield=1.5, db_path="data/market.duckdb")
    assert not df_div.empty
    assert (df_div["dividend_yield"] >= 1.5).all()

    # 5. Impossible filter combination returns empty dataframe gracefully
    df_empty = get_screener_data(min_sharpe=99.0, db_path="data/market.duckdb")
    assert df_empty.empty


@pytest.mark.realdata
def test_app_data_stock_detail_and_comparison_edges():
    """Verify stock detail and comparison queries with valid and invalid inputs."""
    # 1. Valid stock detail
    tcs = get_stock_detail_data("TCS.NS", period="3Y", db_path="data/market.duckdb")
    assert tcs is not None
    assert tcs["ticker"] == "TCS.NS"
    assert "prices_df" in tcs and not tcs["prices_df"].empty
    assert "peers" in tcs

    # 2. 5Y and 10Y periods
    tcs_5y = get_stock_detail_data("TCS.NS", period="5Y", db_path="data/market.duckdb")
    assert tcs_5y is not None

    tcs_10y = get_stock_detail_data("TCS.NS", period="10Y", db_path="data/market.duckdb")
    assert tcs_10y is not None

    # 3. Non-existent ticker returns None
    invalid = get_stock_detail_data("NON_EXISTENT.NS", db_path="data/market.duckdb")
    assert invalid is None

    # 4. Compare multiple stocks
    comp = get_stocks_comparison_data(["TCS.NS", "RELIANCE.NS"], horizon="1Y", db_path="data/market.duckdb")
    assert not comp["rebased_df"].empty
    assert len(comp["metrics_df"]) == 2

    # 5. Compare with non-existent database
    comp_fail = get_stocks_comparison_data(["TCS.NS"], db_path="data/non_existent.duckdb")
    assert comp_fail["rebased_df"].empty


@pytest.mark.realdata
def test_app_data_explore_market_periods():
    """Verify explore market data queries across all period options."""
    for p in ["1Y", "3Y", "5Y", "10Y"]:
        data = get_explore_market_data(selected_sectors=None, selected_caps=["large"], period=p, db_path="data/market.duckdb")
        assert not data["rebased_df"].empty
        assert not data["scatter_df"].empty


def test_check_db_health_missing_db():
    """Verify check_db_health handles missing file gracefully."""
    res = check_db_health("data/does_not_exist.duckdb")
    assert res["is_healthy"] is False
    assert res["data_as_of"] == "N/A"
    assert "not found" in res["message"]


def test_run_py_help():
    """Verify run.py CLI help prints without error."""
    import run
    run.print_help()


def test_clean_single_ticker_edge_cases():
    """Verify data cleaning rules: drops errors, fixes volume, flags spikes and extreme days."""
    from src.clean.clean_market_data import CleaningStats, align_with_trading_calendar, clean_single_ticker, load_nse_holidays

    # 1. Missing holiday file returns empty set
    assert load_nse_holidays("data/reference/does_not_exist.csv") == set()

    # 2. Empty df
    empty_df = clean_single_ticker(pd.DataFrame(), "TEST.NS")
    assert empty_df.empty

    # 3. Data with null dates, duplicate dates, bad volumes, and physical errors
    dates = ["2024-01-01", "2024-01-01", None, "2024-01-02", "2024-01-03", "2024-01-04"]
    raw_df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0, 100.0, 100.0, -10.0, 100.0, 100.0],  # row 3: open <= 0
        "High": [105.0, 105.0, 105.0, 105.0, 90.0, 105.0],   # row 4: high < low (90 < 95)
        "Low": [95.0, 95.0, 95.0, 95.0, 95.0, 95.0],
        "Close": [100.0, 102.0, 100.0, 100.0, 92.0, 103.0],
        "Volume": [1000, -50, 1000, 1000, 1000, 1500],
    })

    stats = CleaningStats()
    cleaned = clean_single_ticker(raw_df, "TEST.NS", stats=stats)
    assert not cleaned.empty
    # Duplicate was dropped
    assert stats.duplicate_rows_dropped >= 1
    # Null date was dropped
    assert stats.null_dates_dropped >= 1
    # Invalid price rows dropped
    assert stats.invalid_price_rows_dropped >= 2

    # 4. align_with_trading_calendar with short gap (forward-filled) and long gap
    df_cal = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-15"]),
        "open": [100.0, 102.0, 105.0],
        "high": [105.0, 105.0, 110.0],
        "low": [95.0, 100.0, 104.0],
        "close": [100.0, 102.0, 105.0],
        "volume": [1000, 1000, 1000],
    })
    cal_stats = CleaningStats()
    aligned = align_with_trading_calendar(df_cal, "TEST.NS", holidays=set(), stats=cal_stats)
    assert cal_stats.forward_filled_days_count >= 1


@pytest.mark.realdata
def test_dq_checks_all():
    """Verify all individual data quality checks against data/market.duckdb."""
    from src.ingest.dq_checks import (
        check_all_tickers_present,
        check_latest_date_recency,
        check_no_duplicate_keys,
        check_no_future_dates,
        check_price_integrity,
        run_all_dq_checks,
    )
    con = duckdb.connect("data/market.duckdb", read_only=True)
    try:
        dup = check_no_duplicate_keys(con)
        assert dup["passed"] is True

        future = check_no_future_dates(con, reference_date="2099-01-01")
        assert future["passed"] is True

        recency = check_latest_date_recency(con, max_lag_days=3000, reference_date="2026-10-21")
        assert recency["passed"] is True

        tickers = check_all_tickers_present(con, min_trading_days=500)
        assert tickers["passed"] is True

        integ = check_price_integrity(con)
        assert integ["passed"] is True
    finally:
        con.close()

    # Also test run_all_dq_checks top-level wrapper with relaxed recency
    all_res = run_all_dq_checks("data/market.duckdb", max_lag_days=3000, reference_date="2026-10-21")
    assert all_res["all_passed"] is True


@pytest.mark.realdata
def test_app_data_home_and_methodology():
    """Verify get_home_kpis_data and get_methodology_data execution."""
    from src.app_data import get_home_kpis_data, get_methodology_data

    kpis = get_home_kpis_data("data/market.duckdb")
    assert kpis["stocks_count"] > 0
    assert kpis["data_as_of"] != "N/A"

    # Missing database graceful fallback
    kpis_missing = get_home_kpis_data("data/missing.duckdb")
    assert kpis_missing["stocks_count"] == 0

    meth = get_methodology_data("data/market.duckdb")
    assert "dataset_summary" in meth
    assert meth["dataset_summary"]["total_rows"] > 0
    assert not meth["stats_results"].empty
    assert meth["counterfactual"]["total_clean_days"] > 0

    meth_missing = get_methodology_data("data/missing.duckdb")
    assert meth_missing == {}


def test_evaluate_history_coverage():
    """Verify history coverage evaluation for empty, short, and adequate histories."""
    from src.ingest.pull_prices import evaluate_history_coverage

    y, status, d1, d2 = evaluate_history_coverage(pd.DataFrame())
    assert status == "FAILED_DOWNLOAD"

    idx_short = pd.date_range("2024-01-01", periods=50, freq="B")
    df_short = pd.DataFrame({"Close": 100.0}, index=idx_short)
    y, status, d1, d2 = evaluate_history_coverage(df_short, min_years=8.0)
    assert status == "UNDER_MIN_HISTORY"

    idx_long = pd.date_range("2016-01-01", "2025-01-01", freq="B")
    df_long = pd.DataFrame({"Close": 100.0}, index=idx_long)
    y, status, d1, d2 = evaluate_history_coverage(df_long, min_years=8.0)
    assert status == "OK"


def test_fetch_ticker_fundamentals_mock(monkeypatch):
    """Verify fundamental fetcher with mock yfinance response."""
    from unittest.mock import MagicMock
    import yfinance as yf
    from src.ingest.pull_fundamentals import fetch_ticker_fundamentals

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "marketCap": 1000000,
        "trailingPE": 25.0,
        "forwardPE": 22.0,
        "dividendYield": 0.015,
        "sector": "Technology",
        "industry": "Software",
        "longName": "Test Tech Inc",
    }
    monkeypatch.setattr(yf, "Ticker", lambda t: mock_ticker)

    rec = fetch_ticker_fundamentals("TEST.NS")
    assert rec["status"] == "OK"
    assert rec["trailing_pe"] == 25.0
    assert rec["sector"] == "Technology"


def test_build_synthetic_benchmark_errors():
    """Verify synthetic benchmark error handling for empty universe or missing prices."""
    from src.config import UniverseStock
    from src.ingest.benchmarks import build_synthetic_cap_benchmark

    with pytest.raises(ValueError, match="No available universe stocks"):
        build_synthetic_cap_benchmark(pd.DataFrame(), [], "mid")

    stocks = [UniverseStock(ticker="S1.NS", name="S1", sector="IT", cap_class="mid")]
    with pytest.raises(ValueError, match="No price data available"):
        build_synthetic_cap_benchmark(pd.DataFrame({"ticker": ["OTHER.NS"]}), stocks, "mid")



