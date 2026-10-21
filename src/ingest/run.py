"""Unified CLI runner for data ingestion and refresh pipeline.

Usage:
    python -m src.ingest.run --full
    python -m src.ingest.run --incremental
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.clean.clean_market_data import run_cleaning_pipeline
from src.config import load_config
from src.ingest.dq_checks import run_all_dq_checks
from src.ingest.load_db import load_full_db, load_incremental
from src.ingest.pull_fundamentals import pull_all_fundamentals
from src.ingest.pull_prices import pull_all_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest.run")


def run_full_pipeline() -> None:
    """Execute complete end-to-end full ingest, clean, store, and DQ verification pipeline."""
    print("\n" + "=" * 60)
    print("STARTING FULL DATA INGESTION & PIPELINE RUN")
    print("=" * 60)
    config = load_config()
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. Ingest prices & fundamentals
    print("\n[STEP 1/4] Downloading full price history & fundamentals snapshot...")
    manifest = pull_all_prices(config=config, batch_size=10, min_years=8.0)
    funds_df = pull_all_fundamentals(config=config)

    # 2. Clean & synthesize
    print("\n[STEP 2/4] Validating, cleaning data, and constructing synthetic benchmarks...")
    clean_df, stats = run_cleaning_pipeline(raw_dir=f"data/raw/{today}", config=config)

    # 3. Load DuckDB
    print("\n[STEP 3/4] Populating DuckDB tables (data/market.duckdb)...")
    load_res = load_full_db(
        db_path="data/market.duckdb",
        clean_df=clean_df,
        fundamentals_df=funds_df,
        run_details="Full pipeline execution",
    )
    print(f"Loaded: {load_res['prices_rows']:,} prices rows, {load_res['universe_count']} universe stocks.")

    # 4. Data quality checks
    print("\n[STEP 4/4] Executing automated Data Quality checks...")
    dq_res = run_all_dq_checks("data/market.duckdb")
    print(f"Data Quality: {dq_res['passed_checks']}/{dq_res['total_checks']} checks passed.")
    print("\nFull pipeline completed successfully.")


def run_incremental_pipeline() -> None:
    """Execute daily incremental refresh, recompute synthetic benchmarks, and verify DQ."""
    print("\n" + "=" * 60)
    print("STARTING INCREMENTAL DATA REFRESH")
    print("=" * 60)
    config = load_config()

    # 1. Incremental load into DuckDB
    print("\n[STEP 1/2] Fetching new dates and updating tables in data/market.duckdb...")
    res = load_incremental(
        db_path="data/market.duckdb",
        config=config,
    )
    print(f"Incremental Refresh Status: {res['status']}")
    print(f"Rows Added: {res['rows_added']}")
    print(f"Previous Latest Date: {res['previous_latest_date']} -> New Latest Date: {res['new_latest_date']}")

    # 2. Run Data Quality checks
    print("\n[STEP 2/2] Running Data Quality verification...")
    dq_res = run_all_dq_checks("data/market.duckdb")
    print(f"Data Quality: {dq_res['passed_checks']}/{dq_res['total_checks']} checks passed.")
    print("\nIncremental refresh completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Data Ingestion & Refresh Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Execute full history pipeline")
    group.add_argument("--incremental", action="store_true", help="Execute daily incremental refresh")

    args = parser.parse_args()

    try:
        if args.full:
            run_full_pipeline()
        elif args.incremental:
            run_incremental_pipeline()
    except Exception as exc:
        logger.error("Pipeline failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
