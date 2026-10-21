"""CLI runner for Phase 1 data pipeline: benchmarks, ingestion, and cleaning."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean.clean_market_data import run_cleaning_pipeline
from src.config import load_config
from src.ingest.benchmarks import inspect_benchmark
from src.ingest.pull_prices import pull_all_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def main() -> None:
    config = load_config()
    today = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("STEP 1: INSPECTING CONFIGURED BENCHMARKS")
    print("=" * 60)
    bench_items = [
        (config.benchmarks.broad_market, "broad_market"),
        (config.benchmarks.midcap, "midcap"),
        (config.benchmarks.smallcap, "smallcap"),
    ]
    for sym, role in bench_items:
        rep = inspect_benchmark(sym, role)
        print(f"Ticker: {sym:10} | Role: {role:12}")
        print(f"  - Yahoo Long/Short Name : {rep.yahoo_name}")
        print(f"  - Latest Price Level    : {rep.latest_close} (as of {rep.latest_date})")
        print(f"  - Historical Depth      : {rep.total_rows} trading rows over {rep.historical_depth_years} years ({rep.first_date} to {rep.latest_date})")
        print(f"  - True Index Identity   : {rep.true_identity}")
        print(f"  - Status & Assessment   : {rep.status_note}")
        print()

    print("=" * 60)
    print("STEP 2: DOWNLOADING HISTORICAL PRICES IN BATCHES OF 10")
    print("=" * 60)
    manifest = pull_all_prices(config=config, batch_size=10, min_years=8.0)
    print(f"\nDownload Summary: {manifest['successful_downloads']} successful, {manifest['failed_downloads']} failed.")
    print(f"Flagged Issues for Review: {manifest['flagged_issues_count']}")
    for issue in manifest["flagged_issues"]:
        print(f"\n[ALERT] Ticker: {issue['ticker']} ({issue['name']})")
        print(f"  Sector & Cap Class: {issue['sector']} / {issue['cap_class']}")
        print(f"  Rows & History    : {issue['rows']} rows ({issue['years']} years)")
        print(f"  Failure Reason    : {issue['reason']}")
        print(f"  Proposed Replace  : {issue['proposed_replacement']} ({issue['proposed_name']})")

    print("\n" + "=" * 60)
    print("STEP 3: CLEANING & VALIDATING MARKET DATA")
    print("=" * 60)
    raw_dir = Path(f"data/raw/{today}")
    df_clean, stats = run_cleaning_pipeline(raw_dir=raw_dir, config=config)
    print(f"\nCleaned Tickers Processed: {stats.tickers_processed}")
    print(f"Raw Rows Processed       : {stats.total_raw_rows:,}")
    print(f"Physical Errors Dropped  : {stats.invalid_price_rows_dropped:,}")
    print(f"Extreme Days Flagged     : {stats.extreme_return_days_flagged:,} (|z| > {config.extreme_z_threshold})")
    print(f"Return Spikes Flagged    : {stats.spikes_flagged:,} (> {config.spike_review_threshold * 100:.0f}%)")
    print(f"Clean Rows Retained      : {stats.clean_rows_retained:,}")
    print("\nPhase 1 pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
