"""Storage and Database Refresh Engine for Market Data.

Loads clean prices, fundamental snapshots, universe reference data, and run logs
into DuckDB (data/market.duckdb).
Adheres strictly to AGENTS.md Hard Rules:
- Rule 1: Never delete genuine extreme return days.
- Rule 3: Raw data is immutable.
- Rule 4: The application is read-only against the database.
- Idempotent: running incremental twice produces identical row counts.
- Market Close Safeguard: drops current day's bar if market is not yet closed (< 16:00 IST).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.clean.clean_market_data import clean_single_ticker, load_nse_holidays
from src.config import AppConfig, UniverseStock, load_config, load_universe
from src.ingest.benchmarks import build_synthetic_cap_benchmark
from src.ingest.pull_fundamentals import pull_all_fundamentals
from src.ingest.pull_prices import download_ticker_with_retry

logger = logging.getLogger(__name__)

# Indian Standard Time (IST) timezone: UTC+05:30
IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_CUTOFF = time(16, 0)  # 16:00 IST: trading halts at 15:30, Bhavcopy finalized by 16:00


def get_current_ist_time() -> datetime:
    """Return the current datetime in Indian Standard Time (IST)."""
    return datetime.now(IST)


def is_market_closed(current_time: datetime | None = None) -> bool:
    """Check if the current IST time is at or after official market settlement (16:00 IST)."""
    if current_time is None:
        current_time = get_current_ist_time()
    return current_time.time() >= MARKET_CLOSE_CUTOFF


def filter_unclosed_intraday_bars(
    df: pd.DataFrame,
    current_time: datetime | None = None,
) -> pd.DataFrame:
    """Filter out today's bar if called before market close settlement (16:00 IST).

    Ensures that an incomplete intraday bar never enters the historical database.
    """
    if df.empty:
        return df

    if current_time is None:
        current_time = get_current_ist_time()

    if not is_market_closed(current_time):
        today_ist = current_time.strftime("%Y-%m-%d")
        initial_len = len(df)
        if "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"
        else:
            return df

        # Convert to string format if necessary
        date_series = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df_filtered = df[date_series < today_ist].copy()
        dropped = initial_len - len(df_filtered)
        if dropped > 0:
            logger.info(
                "Market is open / pre-close (before 16:00 IST). Dropped %d unclosed bars for today (%s)",
                dropped,
                today_ist,
            )
        return df_filtered

    return df


def init_db(db_path: Path | str = "data/market.duckdb") -> duckdb.DuckDBPyConnection:
    """Initialize database tables, schemas, and primary key constraints if not present."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p))

    # 1. prices table
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            simple_return DOUBLE,
            log_return DOUBLE,
            z_score DOUBLE,
            is_extreme BOOLEAN,
            is_spike BOOLEAN,
            is_filled BOOLEAN,
            rebased_price DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)

    # Migration: ensure is_filled column exists on existing databases
    cols_info = [r[1] for r in con.execute("PRAGMA table_info('prices')").fetchall()]
    if cols_info and "is_filled" not in cols_info:
        con.execute("ALTER TABLE prices ADD COLUMN is_filled BOOLEAN DEFAULT FALSE")

    # 2. fundamentals_snapshots table
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_snapshots (
            ticker VARCHAR NOT NULL,
            snapshot_date VARCHAR NOT NULL,
            market_cap DOUBLE,
            trailing_pe DOUBLE,
            forward_pe DOUBLE,
            dividend_yield DOUBLE,
            sector VARCHAR,
            industry VARCHAR,
            long_name VARCHAR,
            status VARCHAR,
            universe_cap_class VARCHAR,
            universe_sector VARCHAR,
            PRIMARY KEY (ticker, snapshot_date)
        )
    """)

    # 3. universe table
    con.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            ticker VARCHAR NOT NULL PRIMARY KEY,
            name VARCHAR,
            sector VARCHAR,
            cap_class VARCHAR,
            cap_source_date VARCHAR
        )
    """)

    # 4. run_log table
    con.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            run_id VARCHAR NOT NULL PRIMARY KEY,
            run_timestamp VARCHAR NOT NULL,
            run_type VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            rows_added INTEGER,
            tickers_updated INTEGER,
            details VARCHAR
        )
    """)

    return con


def get_data_as_of(
    db_path: Path | str = "data/market.duckdb",
    con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """Retrieve the latest clean price date in the database as an ISO date string (YYYY-MM-DD).

    Returns 'N/A' if the database or table is empty or missing.
    """
    if con is not None:
        try:
            res = con.execute("SELECT MAX(date) FROM prices").fetchone()
            if res and res[0]:
                return str(res[0])
            return "N/A"
        except Exception as exc:
            logger.warning("Error reading data_as_of with existing connection: %s", exc)
            return "N/A"

    p = Path(db_path)
    if not p.exists():
        return "N/A"

    try:
        c = duckdb.connect(str(p), read_only=True)
        res = c.execute("SELECT MAX(date) FROM prices").fetchone()
        c.close()
        if res and res[0]:
            return str(res[0])
        return "N/A"
    except Exception as exc:
        logger.warning("Error reading data_as_of from %s: %s", db_path, exc)
        return "N/A"


def load_full_db(
    db_path: Path | str = "data/market.duckdb",
    clean_df: pd.DataFrame | None = None,
    fundamentals_df: pd.DataFrame | None = None,
    universe_path: Path | str = "data/reference/universe.csv",
    run_details: str = "Full database initialization",
) -> dict[str, Any]:
    """Perform a clean, full load of market data into DuckDB with atomic transaction handling."""
    p = Path(db_path)
    con = init_db(p)
    run_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()

    try:
        con.execute("BEGIN TRANSACTION")

        # 1. Load Universe
        stocks = load_universe(universe_path)
        univ_data = [
            {
                "ticker": s.ticker,
                "name": s.name,
                "sector": s.sector,
                "cap_class": s.cap_class,
                "cap_source_date": s.cap_source_date,
            }
            for s in stocks
        ]
        univ_df = pd.DataFrame(univ_data)
        con.execute("DELETE FROM universe")
        con.register("tmp_universe", univ_df)
        con.execute("INSERT INTO universe SELECT * FROM tmp_universe")
        con.unregister("tmp_universe")

        # 2. Load Prices
        prices_count = 0
        if clean_df is not None and not clean_df.empty:
            con.execute("DELETE FROM prices")
            if "is_filled" not in clean_df.columns:
                clean_df = clean_df.copy()
                clean_df["is_filled"] = False
            con.register("tmp_prices", clean_df)
            con.execute("""
                INSERT INTO prices (
                    date, ticker, open, high, low, close, volume,
                    simple_return, log_return, z_score, is_extreme, is_spike, is_filled, rebased_price
                )
                SELECT
                    date, ticker, open, high, low, close, volume,
                    simple_return, log_return, z_score, is_extreme, is_spike, is_filled, rebased_price
                FROM tmp_prices
            """)
            con.unregister("tmp_prices")
            prices_count = len(clean_df)

        # 3. Load Fundamentals
        funds_count = 0
        if fundamentals_df is not None and not fundamentals_df.empty:
            con.execute("DELETE FROM fundamentals_snapshots")
            con.register("tmp_funds", fundamentals_df)
            con.execute("""
                INSERT INTO fundamentals_snapshots (
                    ticker, snapshot_date, market_cap, trailing_pe, forward_pe,
                    dividend_yield, sector, industry, long_name, status,
                    universe_cap_class, universe_sector
                )
                SELECT
                    ticker, snapshot_date, market_cap, trailing_pe, forward_pe,
                    dividend_yield, sector, industry, long_name, status,
                    universe_cap_class, universe_sector
                FROM tmp_funds
            """)
            con.unregister("tmp_funds")
            funds_count = len(fundamentals_df)

        # 4. Log run
        con.execute(
            """
            INSERT INTO run_log (run_id, run_timestamp, run_type, status, rows_added, tickers_updated, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [run_id, ts, "full", "SUCCESS", prices_count, len(stocks), run_details],
        )

        con.execute("COMMIT")
        logger.info("Successfully executed full load into %s: %d prices, %d universe", db_path, prices_count, len(stocks))
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "prices_rows": prices_count,
            "universe_count": len(stocks),
            "fundamentals_rows": funds_count,
        }

    except Exception as exc:
        con.execute("ROLLBACK")
        logger.error("Full load failed, rolled back changes: %s", exc)
        raise exc
    finally:
        con.close()


def compute_incremental_synthetic_benchmarks(
    con: duckdb.DuckDBPyConnection,
    new_stocks_df: pd.DataFrame,
    universe: list[UniverseStock],
) -> pd.DataFrame:
    """Incrementally compound synthetic midcap and smallcap benchmarks from their previous level in DB."""
    synth_rows = []
    for cap_class, bench_ticker in [("mid", "synthetic_midcap"), ("small", "synthetic_smallcap")]:
        constituents = {s.ticker for s in universe if s.cap_class == cap_class}
        if not constituents:
            continue
        min_constituents = int(np.ceil(0.80 * len(constituents)))

        # Get previous close from prices table
        prev = con.execute(
            "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            [bench_ticker],
        ).fetchone()
        prev_close = float(prev[0]) if (prev and prev[0] is not None) else 100.0

        # Unique dates in new_stocks_df sorted
        new_dates = sorted(new_stocks_df["date"].unique())
        for d in new_dates:
            day_cap_stocks = new_stocks_df[
                (new_stocks_df["date"] == d) & (new_stocks_df["ticker"].isin(constituents))
            ]
            active_count = len(day_cap_stocks)
            if active_count < min_constituents:
                logger.warning(
                    "Insufficient constituents for %s on %s: %d active, required %d (quorum)",
                    bench_ticker,
                    d,
                    active_count,
                    min_constituents,
                )
                continue

            mean_return = float(day_cap_stocks["simple_return"].mean())
            total_vol = float(day_cap_stocks["volume"].sum())

            new_open = prev_close
            new_close = prev_close * (1.0 + mean_return)
            new_high = max(new_open, new_close)
            new_low = min(new_open, new_close)
            log_ret = float(np.log(new_close / new_open)) if new_open > 0 else 0.0

            synth_rows.append({
                "date": d,
                "ticker": bench_ticker,
                "open": new_open,
                "high": new_high,
                "low": new_low,
                "close": new_close,
                "volume": total_vol,
                "simple_return": mean_return,
                "log_return": log_ret,
                "z_score": 0.0,
                "is_extreme": False,
                "is_spike": abs(mean_return) > 0.20,
                "is_filled": False,
                "rebased_price": new_close,
            })
            prev_close = new_close

    return pd.DataFrame(synth_rows)


def load_incremental(
    db_path: Path | str = "data/market.duckdb",
    universe_path: Path | str = "data/reference/universe.csv",
    holidays_path: Path | str = "data/reference/nse_holidays.csv",
    config: AppConfig | None = None,
    current_time: datetime | None = None,
    simulate_failure: bool = False,
    injected_raw_dfs: dict[str, pd.DataFrame] | None = None,
    simulate_fetch_failure: bool = False,
) -> dict[str, Any]:
    """Perform an incremental data refresh into DuckDB.

    Requirements:
    - Queries max(date) currently in prices.
    - Pulls new data starting from max(date).
    - Drops current day's bar if before 16:00 IST.
    - Aligns against trading calendar and forward-fills gaps.
    - Never deletes genuine extreme days (Hard Rule 1).
    - Recomputes synthetic midcap and smallcap benchmarks for the new dates with proportional quorum.
    - Idempotent: running twice on the same day/data adds 0 duplicate rows and keeps counts identical.
    - Atomic: if an error or simulate_failure occurs, changes are rolled back completely.
    """
    if config is None:
        config = load_config()

    p = Path(db_path)
    con = init_db(p)

    latest_date_in_db = get_data_as_of(p, con=con)
    if latest_date_in_db == "N/A":
        con.close()
        raise ValueError("Database is empty. Please run full load first.")

    run_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    holidays = load_nse_holidays(holidays_path)
    universe = config.universe or load_universe(universe_path)

    # 1. Determine date range for incremental pull
    # Start 2 calendar days before latest date to catch potential adjustments
    d_last = datetime.strptime(latest_date_in_db, "%Y-%m-%d")
    fetch_start = (d_last - timedelta(days=2)).strftime("%Y-%m-%d")

    target_tickers = [s.ticker for s in universe] + ["^NSEI"]
    logger.info("Executing incremental pull for %d tickers starting from %s", len(target_tickers), fetch_start)

    try:
        con.execute("BEGIN TRANSACTION")

        if simulate_fetch_failure:
            raise RuntimeError("Simulated network/Yahoo Finance failure during fetch.")

        new_clean_dfs: list[pd.DataFrame] = []
        for ticker in target_tickers:
            if injected_raw_dfs is not None:
                if ticker not in injected_raw_dfs:
                    continue
                df_raw = injected_raw_dfs[ticker].copy()
            else:
                df_raw = download_ticker_with_retry(ticker, start_date=fetch_start)

            if df_raw.empty:
                continue

            # Market close safeguard: drop today's unclosed bar if < 16:00 IST
            df_raw = filter_unclosed_intraday_bars(df_raw, current_time=current_time)
            if df_raw.empty:
                continue

            # Clean raw bars
            clean_df = clean_single_ticker(
                df_raw,
                ticker=ticker,
                extreme_z_threshold=config.extreme_z_threshold,
                spike_threshold=config.spike_review_threshold,
                holidays=holidays,
            )
            if not clean_df.empty:
                # Retain only dates strictly newer than latest_date_in_db
                clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.strftime("%Y-%m-%d")
                clean_df = clean_df[clean_df["date"] > latest_date_in_db]
                if not clean_df.empty:
                    new_clean_dfs.append(clean_df)

        rows_added = 0
        new_combined_df = pd.DataFrame()

        if new_clean_dfs:
            new_combined_df = pd.concat(new_clean_dfs, ignore_index=True)

            # Rebuild synthetic benchmarks for the new dates (proportional quorum compounding from previous level)
            try:
                synth_df = compute_incremental_synthetic_benchmarks(con, new_combined_df, universe)
                if not synth_df.empty:
                    new_combined_df = pd.concat([new_combined_df, synth_df], ignore_index=True)
            except Exception as bench_exc:
                logger.warning("Synthetic basket incremental compounding: %s", bench_exc)

            if "is_filled" not in new_combined_df.columns:
                new_combined_df["is_filled"] = False

            # Insert new rows into prices table (idempotent: avoid duplicates)
            con.register("tmp_new_prices", new_combined_df)
            con.execute("""
                INSERT INTO prices (
                    date, ticker, open, high, low, close, volume,
                    simple_return, log_return, z_score, is_extreme, is_spike, is_filled, rebased_price
                )
                SELECT
                    np.date, np.ticker, np.open, np.high, np.low, np.close, np.volume,
                    np.simple_return, np.log_return, np.z_score, np.is_extreme, np.is_spike, np.is_filled, np.rebased_price
                FROM tmp_new_prices np
                WHERE NOT EXISTS (
                    SELECT 1 FROM prices p WHERE p.ticker = np.ticker AND p.date = np.date
                )
            """)
            con.unregister("tmp_new_prices")
            rows_added = len(new_combined_df)

        # Simulated failure test check
        if simulate_failure:
            raise RuntimeError("Simulated transaction failure triggered for testing atomicity.")

        # Log execution
        con.execute(
            """
            INSERT INTO run_log (run_id, run_timestamp, run_type, status, rows_added, tickers_updated, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                run_id,
                ts,
                "incremental",
                "SUCCESS",
                rows_added,
                len(target_tickers),
                f"Incremental refresh from {latest_date_in_db}. Added {rows_added} rows.",
            ],
        )

        con.execute("COMMIT")
        logger.info("Incremental refresh completed. Rows added: %d", rows_added)

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "rows_added": rows_added,
            "previous_latest_date": latest_date_in_db,
            "new_latest_date": get_data_as_of(p, con=con),
        }

    except Exception as exc:
        con.execute("ROLLBACK")
        logger.error("Incremental refresh failed or was simulated to fail. Transaction rolled back: %s", exc)
        raise exc
    finally:
        con.close()
