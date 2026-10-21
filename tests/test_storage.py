"""Unit tests for DuckDB storage, incremental refresh, and Data Quality suite."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.ingest.dq_checks import (
    DataQualityError,
    check_all_tickers_present,
    check_latest_date_recency,
    check_no_duplicate_keys,
    check_no_future_dates,
    check_price_integrity,
    run_all_dq_checks,
)
from src.ingest.load_db import (
    filter_unclosed_intraday_bars,
    get_data_as_of,
    init_db,
    is_market_closed,
    load_full_db,
    load_incremental,
)

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """Fixture providing clean price rows."""
    return pd.DataFrame([
        {
            "date": "2026-10-16",
            "ticker": "HDFCBANK.NS",
            "open": 1700.0,
            "high": 1720.0,
            "low": 1690.0,
            "close": 1710.0,
            "volume": 1000000.0,
            "simple_return": 0.01,
            "log_return": 0.00995,
            "z_score": 0.5,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        },
        {
            "date": "2026-10-19",
            "ticker": "HDFCBANK.NS",
            "open": 1710.0,
            "high": 1730.0,
            "low": 1705.0,
            "close": 1725.0,
            "volume": 1200000.0,
            "simple_return": 0.00877,
            "log_return": 0.00873,
            "z_score": 0.45,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.88,
        },
    ])


@pytest.fixture
def sample_fundamentals_df() -> pd.DataFrame:
    """Fixture providing fundamental snapshot rows."""
    return pd.DataFrame([
        {
            "ticker": "HDFCBANK.NS",
            "snapshot_date": "2026-10-20",
            "market_cap": 1300000000000.0,
            "trailing_pe": 18.5,
            "forward_pe": 16.2,
            "dividend_yield": 0.012,
            "sector": "Financial Services",
            "industry": "Banks - Diversified",
            "long_name": "HDFC Bank Limited",
            "status": "OK",
            "universe_cap_class": "large",
            "universe_sector": "Banking and Financials",
        }
    ])


def test_init_db_creates_tables(tmp_path: Path):
    """Verify that init_db creates all 4 required tables with primary keys."""
    db_file = tmp_path / "test_market.duckdb"
    con = init_db(db_file)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    con.close()

    assert "prices" in tables
    assert "fundamentals_snapshots" in tables
    assert "universe" in tables
    assert "run_log" in tables


def test_get_data_as_of_empty_and_populated(tmp_path: Path, sample_clean_df: pd.DataFrame):
    """Verify get_data_as_of returns 'N/A' on empty DB and max(date) on populated DB."""
    db_file = tmp_path / "test_market.duckdb"
    assert get_data_as_of(db_file) == "N/A"

    init_db(db_file)
    assert get_data_as_of(db_file) == "N/A"

    load_full_db(db_file, clean_df=sample_clean_df)
    assert get_data_as_of(db_file) == "2026-10-19"


def test_market_close_filter_before_and_after_1600():
    """Verify filter_unclosed_intraday_bars drops today's bar before 16:00 IST and keeps it after."""
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    yesterday_str = (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")

    df = pd.DataFrame([
        {"date": yesterday_str, "close": 100.0},
        {"date": today_str, "close": 105.0},
    ])

    # 1. Simulate 14:30 IST (before 16:00 close) -> Today's bar must be dropped
    time_before_close = datetime.now(IST).replace(hour=14, minute=30)
    filtered_early = filter_unclosed_intraday_bars(df, current_time=time_before_close)
    assert len(filtered_early) == 1
    assert filtered_early.iloc[0]["date"] == yesterday_str

    # 2. Simulate 16:30 IST (after 16:00 close) -> Both bars must be kept
    time_after_close = datetime.now(IST).replace(hour=16, minute=30)
    filtered_late = filter_unclosed_intraday_bars(df, current_time=time_after_close)
    assert len(filtered_late) == 2


def test_load_incremental_idempotent_no_duplicates(tmp_path: Path):
    """Verify that running load_incremental twice produces identical row counts (idempotent)."""
    db_file = tmp_path / "market_idempotent.duckdb"
    con = init_db(db_file)

    # Populate initial row
    initial_df = pd.DataFrame([
        {
            "date": "2026-10-16",
            "ticker": "HDFCBANK.NS",
            "open": 1700.0,
            "high": 1720.0,
            "low": 1690.0,
            "close": 1710.0,
            "volume": 1000000.0,
            "simple_return": 0.01,
            "log_return": 0.00995,
            "z_score": 0.5,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        }
    ])
    con.register("tmp_p", initial_df)
    con.execute("INSERT INTO prices SELECT * FROM tmp_p")
    con.unregister("tmp_p")
    con.close()

    # Pre-run count
    con = duckdb.connect(str(db_file), read_only=True)
    count_before = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()
    assert count_before == 1

    # First incremental run with offline injected data
    injected_raw = {
        "HDFCBANK.NS": pd.DataFrame([
            {"Date": "2026-10-16", "Open": 1700.0, "High": 1720.0, "Low": 1690.0, "Close": 1710.0, "Volume": 1000000.0}
        ])
    }
    res1 = load_incremental(db_path=db_file, injected_raw_dfs=injected_raw)
    con = duckdb.connect(str(db_file), read_only=True)
    count_after1 = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()

    # Second incremental run with same data
    res2 = load_incremental(db_path=db_file, injected_raw_dfs=injected_raw)
    con = duckdb.connect(str(db_file), read_only=True)
    count_after2 = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()

    # Must be completely identical
    assert count_after1 == count_after2
    assert res2["rows_added"] == 0


def test_simulated_failure_leaves_db_unchanged(tmp_path: Path):
    """Verify that an error during incremental load rolls back and leaves DB untouched."""
    db_file = tmp_path / "market_rollback.duckdb"
    init_db(db_file)

    # Initial state
    initial_df = pd.DataFrame([
        {
            "date": "2026-10-16",
            "ticker": "HDFCBANK.NS",
            "open": 1700.0,
            "high": 1720.0,
            "low": 1690.0,
            "close": 1710.0,
            "volume": 1000000.0,
            "simple_return": 0.01,
            "log_return": 0.00995,
            "z_score": 0.5,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        }
    ])
    con = duckdb.connect(str(db_file))
    con.register("tmp_p", initial_df)
    con.execute("INSERT INTO prices SELECT * FROM tmp_p")
    con.unregister("tmp_p")
    con.close()

    # Record initial counts
    con = duckdb.connect(str(db_file), read_only=True)
    count_before = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()

    # Trigger simulated failure
    with pytest.raises(RuntimeError, match="Simulated transaction failure"):
        load_incremental(db_path=db_file, simulate_failure=True, injected_raw_dfs={})

    # Verify counts remain 100% unchanged
    con = duckdb.connect(str(db_file), read_only=True)
    count_after = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()

    assert count_after == count_before


@pytest.mark.realdata
def test_dq_checks_pass_on_clean_db():
    """Verify that all DQ checks pass on the official clean database."""
    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not found (run pipeline first)")
    report = run_all_dq_checks(db_path=str(db_path), reference_date="2026-10-21")
    assert report["all_passed"] is True
    assert report["passed_checks"] == report["total_checks"]


def test_dq_checks_detect_duplicate_key_failure(tmp_path: Path):
    """Verify check_no_duplicate_keys fails when duplicate keys exist."""
    db_file = tmp_path / "dup_key.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date VARCHAR)")
    con.execute("INSERT INTO prices VALUES ('TCS.NS', '2026-10-19'), ('TCS.NS', '2026-10-19')")

    res = check_no_duplicate_keys(con)
    con.close()

    assert res["passed"] is False
    assert "FAILED" in res["message"]


def test_dq_checks_detect_future_date_failure(tmp_path: Path):
    """Verify check_no_future_dates fails when date > reference_date."""
    db_file = tmp_path / "future_date.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date VARCHAR)")
    con.execute("INSERT INTO prices VALUES ('TCS.NS', '2026-10-25')")

    res = check_no_future_dates(con, reference_date="2026-10-20")
    con.close()

    assert res["passed"] is False
    assert "FAILED" in res["message"]


def test_check_all_tickers_present_fails_under_1000_trading_days(tmp_path: Path):
    """Verify that any ticker with fewer than 1000 trading days fails DQ checks."""
    from src.ingest.dq_checks import check_all_tickers_present

    db_file = tmp_path / "short_history.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date VARCHAR)")
    # Insert 500 rows for HDFCBANK.NS (< 1000 threshold)
    con.execute("""
        INSERT INTO prices
        SELECT 'HDFCBANK.NS', '2020-01-01'
        FROM range(500)
    """)
    res = check_all_tickers_present(con, min_trading_days=1000)
    con.close()

    assert res["passed"] is False
    assert "HDFCBANK.NS" in res["short_history_tickers"]
    assert res["short_history_tickers"]["HDFCBANK.NS"] == 500


def test_check_all_tickers_present_fails_if_cnxsc_present(tmp_path: Path):
    """Verify that presence of unsanitized ^CNXSC fails check_all_tickers_present."""
    from src.ingest.dq_checks import check_all_tickers_present

    db_file = tmp_path / "with_cnxsc.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("CREATE TABLE prices (ticker VARCHAR, date VARCHAR)")
    con.execute("INSERT INTO prices VALUES ('^CNXSC', '2026-10-19')")

    res = check_all_tickers_present(con)
    con.close()

    assert res["passed"] is False
    assert "^CNXSC" in res["message"]


def test_incremental_simulated_new_day_with_synthetic_compounding_and_quorum(tmp_path: Path):
    """Verify that incremental additions compound synthetic midcap/smallcap from previous close with quorum."""
    db_file = tmp_path / "incremental_synthetic.duckdb"
    init_db(db_file)

    # Base date: 2026-10-16 with synthetic base level 100.0
    base_rows = [
        {
            "date": "2026-10-16",
            "ticker": "synthetic_midcap",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 500.0,  # Prior level is 500.0
            "volume": 100000.0,
            "simple_return": 0.0,
            "log_return": 0.0,
            "z_score": 0.0,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 500.0,
        },
        {
            "date": "2026-10-16",
            "ticker": "synthetic_smallcap",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 400.0,  # Prior level is 400.0
            "volume": 80000.0,
            "simple_return": 0.0,
            "log_return": 0.0,
            "z_score": 0.0,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 400.0,
        },
        {
            "date": "2026-10-16",
            "ticker": "POLYCAB.NS",
            "open": 6000.0,
            "high": 6100.0,
            "low": 5950.0,
            "close": 6050.0,
            "volume": 200000.0,
            "simple_return": 0.01,
            "log_return": 0.0099,
            "z_score": 0.2,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        },
    ]
    con = duckdb.connect(str(db_file))
    df_base = pd.DataFrame(base_rows)
    con.register("tmp_b", df_base)
    con.execute("INSERT INTO prices SELECT * FROM tmp_b")
    con.unregister("tmp_b")
    con.close()

    # Load universe to get midcap and smallcap constituents
    from src.config import load_universe
    universe = load_universe()
    mid_stocks = [s.ticker for s in universe if s.cap_class == "mid"]
    small_stocks = [s.ticker for s in universe if s.cap_class == "small"]

    # Simulate new day: 2026-10-19 with +2.0% return across all midcap stocks and +1.0% across smallcap
    injected_raw: dict[str, pd.DataFrame] = {}
    for ticker in mid_stocks:
        injected_raw[ticker] = pd.DataFrame([
            {"Date": "2026-10-16", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 40000.0},
            {"Date": "2026-10-19", "Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0, "Volume": 50000.0},
        ])
    for ticker in small_stocks:
        injected_raw[ticker] = pd.DataFrame([
            {"Date": "2026-10-16", "Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0, "Volume": 20000.0},
            {"Date": "2026-10-19", "Open": 50.0, "High": 52.0, "Low": 49.0, "Close": 50.5, "Volume": 25000.0},
        ])

    # Run incremental with injected data
    res = load_incremental(db_path=db_file, injected_raw_dfs=injected_raw)
    assert res["status"] == "SUCCESS"
    assert res["rows_added"] > 0

    # Verify synthetic compounding from prior close
    con = duckdb.connect(str(db_file), read_only=True)
    mid_row = con.execute("SELECT close, rebased_price, simple_return FROM prices WHERE ticker = 'synthetic_midcap' AND date = '2026-10-19'").fetchone()
    small_row = con.execute("SELECT close, rebased_price, simple_return FROM prices WHERE ticker = 'synthetic_smallcap' AND date = '2026-10-19'").fetchone()
    con.close()

    assert mid_row is not None, "synthetic_midcap incremental row was not created"
    assert small_row is not None, "synthetic_smallcap incremental row was not created"
    # Compounding check: prior was 500.0, new close = 500.0 * (1 + 0.02) = 510.0
    assert mid_row[0] == pytest.approx(510.0, rel=1e-3)
    assert mid_row[1] == pytest.approx(510.0, rel=1e-3)
    # Prior was 400.0, new close = 400.0 * (1 + 0.01) = 404.0
    assert small_row[0] == pytest.approx(404.0, rel=1e-3)


def test_incremental_simulated_extreme_day_minus_15_percent_kept_never_deleted(tmp_path: Path):
    """Verify that a simulated -15% return on a new day is retained with is_extreme=True and not deleted."""
    db_file = tmp_path / "extreme_ret.duckdb"
    init_db(db_file)

    base_rows = [
        {
            "date": "2026-10-16",
            "ticker": "INFY.NS",
            "open": 1800.0,
            "high": 1820.0,
            "low": 1790.0,
            "close": 1800.0,
            "volume": 500000.0,
            "simple_return": 0.0,
            "log_return": 0.0,
            "z_score": 0.0,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        }
    ]
    con = duckdb.connect(str(db_file))
    df_base = pd.DataFrame(base_rows)
    con.register("tmp_b", df_base)
    con.execute("INSERT INTO prices SELECT * FROM tmp_b")
    con.unregister("tmp_b")
    con.close()

    # Simulate new day with -15% crash
    injected_raw = {
        "INFY.NS": pd.DataFrame([
            {"Date": "2026-10-16", "Open": 1800.0, "High": 1820.0, "Low": 1790.0, "Close": 1800.0, "Volume": 500000.0},
            {"Date": "2026-10-19", "Open": 1800.0, "High": 1800.0, "Low": 1500.0, "Close": 1530.0, "Volume": 2500000.0},
        ])
    }

    res = load_incremental(db_path=db_file, injected_raw_dfs=injected_raw)
    assert res["status"] == "SUCCESS"

    con = duckdb.connect(str(db_file), read_only=True)
    row = con.execute("SELECT date, simple_return, is_extreme, is_spike FROM prices WHERE ticker = 'INFY.NS' AND date = '2026-10-19'").fetchone()
    con.close()

    assert row is not None, "Crash day row was deleted"
    # Return: (1530 - 1800) / 1800 = -0.15 (-15%)
    assert pytest.approx(row[1], abs=0.01) == -0.15
    assert row[2] is True, "is_extreme must be True for -15% crash move"


def test_incremental_clock_mocking_before_and_after_1600_ist(tmp_path: Path):
    """Verify that before 16:00 IST today's bar is dropped, and at/after 16:00 IST it is retained."""
    mock_today = datetime(2026, 10, 20, 14, 30, tzinfo=IST)  # 14:30 IST (market open)
    df_raw = pd.DataFrame([
        {"Date": "2026-10-19", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000.0},
        {"Date": "2026-10-20", "Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0, "Volume": 1200.0},
    ])

    filtered_before = filter_unclosed_intraday_bars(df_raw, current_time=mock_today)
    assert len(filtered_before) == 1
    assert filtered_before["Date"].iloc[-1] == "2026-10-19"

    mock_after = datetime(2026, 10, 20, 16, 15, tzinfo=IST)  # 16:15 IST (settled)
    filtered_after = filter_unclosed_intraday_bars(df_raw, current_time=mock_after)
    assert len(filtered_after) == 2
    assert filtered_after["Date"].iloc[-1] == "2026-10-20"


def test_incremental_gap_forward_fill_1_2_days_and_longer_gap_flagged(tmp_path: Path):
    """Verify that a 1-2 day gap is forward-filled and a longer gap (>2 days) is flagged."""
    from src.clean.clean_market_data import clean_single_ticker, CleaningStats

    # 1 day missing between Monday 2026-10-12 and Wednesday 2026-10-14 (Tuesday 2026-10-13 missing)
    df_short_gap = pd.DataFrame([
        {"Date": "2026-10-12", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000.0},
        {"Date": "2026-10-14", "Open": 101.0, "High": 104.0, "Low": 100.0, "Close": 103.0, "Volume": 1200.0},
    ])
    stats_short = CleaningStats()
    cleaned_short = clean_single_ticker(df_short_gap, ticker="TEST.NS", holidays=set(), stats=stats_short)
    # The missing day 2026-10-13 must be forward-filled
    assert len(cleaned_short) == 3
    dates = cleaned_short["date"].dt.strftime("%Y-%m-%d").tolist()
    assert "2026-10-13" in dates
    ff_row = cleaned_short[cleaned_short["date"] == "2026-10-13"].iloc[0]
    assert ff_row["is_filled"] is True or ff_row["is_filled"] == 1
    assert ff_row["volume"] == 0.0

    # 4 days missing (> 2 days threshold)
    df_long_gap = pd.DataFrame([
        {"Date": "2026-10-05", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000.0},
        {"Date": "2026-10-12", "Open": 101.0, "High": 104.0, "Low": 100.0, "Close": 103.0, "Volume": 1200.0},
    ])
    stats_long = CleaningStats()
    cleaned_long = clean_single_ticker(df_long_gap, ticker="TEST.NS", holidays=set(), stats=stats_long)
    # Should NOT forward fill the 4 days
    assert len(cleaned_long) == 2
    long_gap_logs = [log for log in stats_long.audit_log if log.issue_type == "LONG_CALENDAR_GAP"]
    assert len(long_gap_logs) > 0


def test_incremental_yahoo_fetch_failure_leaves_db_byte_identical(tmp_path: Path):
    """Verify that a failure during Yahoo fetch leaves the database byte-identical."""
    import hashlib

    db_file = tmp_path / "byte_identical.duckdb"
    init_db(db_file)

    initial_df = pd.DataFrame([
        {
            "date": "2026-10-16",
            "ticker": "HDFCBANK.NS",
            "open": 1700.0,
            "high": 1720.0,
            "low": 1690.0,
            "close": 1710.0,
            "volume": 1000000.0,
            "simple_return": 0.01,
            "log_return": 0.00995,
            "z_score": 0.5,
            "is_extreme": False,
            "is_spike": False,
            "is_filled": False,
            "rebased_price": 100.0,
        }
    ])
    con = duckdb.connect(str(db_file))
    con.register("tmp_p", initial_df)
    con.execute("INSERT INTO prices SELECT * FROM tmp_p")
    con.unregister("tmp_p")
    con.close()

    # Capture initial file hash
    with open(db_file, "rb") as f:
        bytes_before = hashlib.sha256(f.read()).hexdigest()

    # Attempt incremental load with simulated fetch failure
    with pytest.raises(RuntimeError, match="Simulated network/Yahoo Finance failure"):
        load_incremental(db_path=db_file, simulate_fetch_failure=True)

    # Capture file hash after failure
    with open(db_file, "rb") as f:
        bytes_after = hashlib.sha256(f.read()).hexdigest()

    assert bytes_after == bytes_before, "Database file modified despite failure during fetch"

