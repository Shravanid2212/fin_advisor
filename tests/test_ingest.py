"""Tests for ingestion helpers, history coverage evaluation, and benchmark analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import UniverseStock
from src.ingest.benchmarks import build_synthetic_cap_benchmark, inspect_benchmark
from src.ingest.pull_prices import evaluate_history_coverage


def test_evaluate_history_coverage_empty_df():
    """Verify empty DataFrame maps to FAILED_DOWNLOAD."""
    df = pd.DataFrame()
    years, status, first_d, last_d = evaluate_history_coverage(df)
    assert years == 0.0
    assert status == "FAILED_DOWNLOAD"
    assert first_d is None
    assert last_d is None


def test_evaluate_history_coverage_sufficient_and_insufficient():
    """Verify history threshold (> 8 years) calculation."""
    # 5 years
    dates_5y = pd.date_range("2018-01-01", "2023-01-01", freq="B")
    df_5y = pd.DataFrame({"Close": [100.0] * len(dates_5y)}, index=dates_5y)
    years, status, _, _ = evaluate_history_coverage(df_5y, min_years=8.0)
    assert years == pytest.approx(5.0, abs=0.1)
    assert status == "UNDER_MIN_HISTORY"

    # 10 years
    dates_10y = pd.date_range("2016-01-01", "2026-01-01", freq="B")
    df_10y = pd.DataFrame({"Close": [100.0] * len(dates_10y)}, index=dates_10y)
    years, status, _, _ = evaluate_history_coverage(df_10y, min_years=8.0)
    assert years >= 10.0
    assert status == "OK"


def test_build_synthetic_cap_benchmark():
    """Verify synthetic equal-weighted benchmark generation from clean prices."""
    dates = pd.date_range("2023-01-01", periods=5, freq="B").strftime("%Y-%m-%d").tolist()
    records = []
    # Stock A: +2% each day
    # Stock B: +4% each day
    for idx, d in enumerate(dates):
        records.append({
            "date": d,
            "ticker": "STOCK_A.NS",
            "open": 100.0 * (1.02**idx),
            "high": 101.0 * (1.02**idx),
            "low": 99.0 * (1.02**idx),
            "close": 100.0 * (1.02**idx),
            "volume": 1000,
            "simple_return": 0.02 if idx > 0 else 0.0,
        })
        records.append({
            "date": d,
            "ticker": "STOCK_B.NS",
            "open": 200.0 * (1.04**idx),
            "high": 201.0 * (1.04**idx),
            "low": 199.0 * (1.04**idx),
            "close": 200.0 * (1.04**idx),
            "volume": 2000,
            "simple_return": 0.04 if idx > 0 else 0.0,
        })

    clean_df = pd.DataFrame(records)
    universe = [
        UniverseStock(ticker="STOCK_A.NS", name="A", sector="IT", cap_class="mid"),
        UniverseStock(ticker="STOCK_B.NS", name="B", sector="Auto", cap_class="mid"),
        UniverseStock(ticker="OTHER.NS", name="Other", sector="FMCG", cap_class="large"),
    ]

    benchmark_df = build_synthetic_cap_benchmark(
        clean_df, universe, cap_class="mid", min_constituents=2
    )
    assert len(benchmark_df) == 5
    assert benchmark_df["ticker"].iloc[0] == "synthetic_midcap"
    assert benchmark_df["close"].iloc[0] == pytest.approx(100.0)
    # Average return is ~3% per day -> (1.03)^4 * 100 ~ 112.55
    assert benchmark_df["close"].iloc[-1] > 110.0
    assert "rebased_price" in benchmark_df.columns


def test_build_synthetic_cap_benchmark_no_stocks_raises():
    """Verify ValueError when no stocks match target cap class."""
    clean_df = pd.DataFrame({
        "date": ["2023-01-01"],
        "ticker": ["A.NS"],
        "close": [100.0],
        "simple_return": [0.0],
        "volume": [1000],
    })
    universe = [UniverseStock(ticker="A.NS", name="A", sector="IT", cap_class="large")]
    with pytest.raises(ValueError, match="No available universe stocks"):
        build_synthetic_cap_benchmark(clean_df, universe, cap_class="small")

