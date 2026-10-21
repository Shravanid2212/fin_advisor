"""Tests for data cleaning, validation, and extreme return preservation."""

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.clean.clean_market_data import CleaningStats, clean_single_ticker, run_cleaning_pipeline


def test_clean_single_ticker_preserves_extreme_returns():
    """Verify that genuine extreme returns (e.g. March 2020 crash) are NOT deleted,

    but flagged with is_extreme = True per AGENTS.md Hard Rule 1.
    """
    dates = pd.date_range("2020-03-01", periods=30, freq="B")
    # Base prices
    prices = np.linspace(100, 110, len(dates))
    # Inject a genuine crash day (e.g. 15% drop on index 15)
    prices[15] = prices[14] * 0.85
    # Subsequent recovery
    for i in range(16, len(dates)):
        prices[i] = prices[i - 1] * 1.01

    raw_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 1000000,
        }
    )

    stats = CleaningStats()
    clean_df = clean_single_ticker(raw_df, ticker="TEST.NS", extreme_z_threshold=3.0, stats=stats)

    # 1. Row count should be preserved (none dropped)
    assert len(clean_df) == 30
    # 2. Crash day must be retained
    assert clean_df["is_extreme"].any()
    # 3. Crash day index has is_extreme True
    assert clean_df.iloc[15]["is_extreme"]
    # 4. Returns and rebased price computed correctly
    assert clean_df["rebased_price"].iloc[0] == pytest.approx(100.0)
    assert clean_df["rebased_price"].iloc[15] < 100.0


def test_clean_single_ticker_drops_confirmed_physical_errors():
    """Verify that only confirmed data errors (price <= 0, high < low, close outside [low, high]) are dropped."""
    dates = pd.date_range("2021-01-01", periods=6, freq="B")
    raw_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, -10.0, 100.0, 100.0, 100.0, 100.0],  # row 1: negative open
            "High": [105.0, 105.0, 95.0, 105.0, 105.0, 105.0],   # row 2: high < low (95 < 98)
            "Low": [98.0, 98.0, 98.0, 98.0, 98.0, 98.0],
            "Close": [102.0, 102.0, 102.0, 120.0, 102.0, 102.0], # row 3: close (120) > high (105)
            "Volume": [1000, 1000, 1000, 1000, 1000, 1000],
        }
    )

    stats = CleaningStats()
    clean_df = clean_single_ticker(raw_df, ticker="BAD.NS", stats=stats)

    # Out of 6 rows, rows 1, 2, 3 must be dropped, leaving rows 0, 4, 5 (3 rows)
    assert len(clean_df) == 3
    assert stats.invalid_price_rows_dropped == 3


def test_clean_single_ticker_deduplication_and_null_dates():
    """Verify duplicate dates (keeping last) and null dates are removed."""
    raw_df = pd.DataFrame(
        {
            "Date": ["2022-01-03", "2022-01-04", "2022-01-04", None, "2022-01-05"],
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 105.0, 106.0, 107.0, 108.0],
            "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],  # on duplicate 2022-01-04, 103 should be kept
            "Volume": [1000, 1000, 2000, 1000, 1000],
        }
    )

    stats = CleaningStats()
    clean_df = clean_single_ticker(raw_df, ticker="DUP.NS", stats=stats)

    assert len(clean_df) == 3  # Jan 3, Jan 4 (second one), Jan 5
    assert stats.null_dates_dropped == 1
    assert stats.duplicate_rows_dropped == 1
    # Check kept value
    jan4 = clean_df[clean_df["date"] == "2022-01-04"]
    assert len(jan4) == 1
    assert jan4["close"].iloc[0] == 103.0


def test_run_cleaning_pipeline_duckdb_and_parquet(tmp_path: Path):
    """Verify full pipeline processes raw files, exports to DuckDB and Parquet, and writes report."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    clean_dir = tmp_path / "clean"
    report_file = tmp_path / "reports" / "cleaning_report.md"

    # Create dummy raw file
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    dummy_raw = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 101.0,
            "Volume": 50000,
        }
    )
    dummy_raw.to_csv(raw_dir / "HDFCBANK.csv", index=False)

    df_clean, stats = run_cleaning_pipeline(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        report_file=report_file,
    )

    assert len(df_clean) == 10
    assert (clean_dir / "market_clean.parquet").exists()
    assert report_file.exists()
    # Parquet contains the clean columns including is_filled
    df_pq = pd.read_parquet(clean_dir / "market_clean.parquet")
    assert len(df_pq) == 10
    assert "is_filled" in df_pq.columns
    assert pytest.approx(df_pq["rebased_price"].mean()) == 100.0


@pytest.mark.realdata
def test_real_clean_data_nifty_worst_return_below_negative_ten_percent():
    """Verify from real clean DuckDB data that Nifty 50's worst daily return is below -10%

    (confirming preservation of March 23, 2020 crash day per AGENTS.md Hard Rule 1).
    """
    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not found (run pipeline first)")

    con = duckdb.connect(str(db_path), read_only=True)
    res = con.execute("SELECT min(simple_return) FROM prices WHERE ticker = '^NSEI'").fetchall()
    con.close()

    worst_return = res[0][0]
    assert worst_return is not None, "No return data found for ^NSEI"
    assert worst_return < -0.10, f"Expected worst return < -10%, got {worst_return * 100:.2f}%"
    assert worst_return == pytest.approx(-0.1298, abs=0.005)

