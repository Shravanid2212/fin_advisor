"""Data Quality Test Suite for Market Data.

Enforces critical data integrity rules on data/market.duckdb:
1. No duplicate primary keys (ticker, date).
2. No future dates relative to current IST date.
3. Latest date recency is within configured trading calendar threshold (<= 5 days).
4. Each universe ticker and benchmark series has active historical data.
5. Physical price integrity (prices > 0, high >= low, close within [low, high]).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from src.config import load_universe

logger = logging.getLogger(__name__)

# Indian Standard Time (IST)
IST = timezone(timedelta(hours=5, minutes=30))


class DataQualityError(Exception):
    """Raised when one or more data quality checks fail."""


def check_no_duplicate_keys(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Verify that (ticker, date) is strictly unique across all rows in prices table."""
    query = """
        SELECT ticker, date, COUNT(*) AS count
        FROM prices
        GROUP BY ticker, date
        HAVING COUNT(*) > 1
        LIMIT 10
    """
    dups = con.execute(query).fetchall()
    passed = len(dups) == 0
    return {
        "check": "no_duplicate_keys",
        "passed": passed,
        "violations_sample": dups,
        "message": "Passed: No duplicate (ticker, date) records found." if passed else f"FAILED: Found duplicate keys: {dups}",
    }


def check_no_future_dates(
    con: duckdb.DuckDBPyConnection,
    reference_date: str | None = None,
) -> dict[str, Any]:
    """Verify that no price dates exceed the current calendar date."""
    if reference_date is None:
        reference_date = datetime.now(IST).strftime("%Y-%m-%d")

    query = """
        SELECT ticker, date
        FROM prices
        WHERE date > ?
        LIMIT 10
    """
    future_rows = con.execute(query, [reference_date]).fetchall()
    passed = len(future_rows) == 0
    return {
        "check": "no_future_dates",
        "passed": passed,
        "reference_date": reference_date,
        "violations_sample": future_rows,
        "message": (
            f"Passed: No future dates beyond {reference_date} found."
            if passed
            else f"FAILED: Found future dates beyond {reference_date}: {future_rows}"
        ),
    }


def check_latest_date_recency(
    con: duckdb.DuckDBPyConnection,
    max_lag_days: int = 5,
    reference_date: str | None = None,
) -> dict[str, Any]:
    """Verify that the latest date in prices is within max_lag_days of the reference date."""
    if reference_date is None:
        reference_date = datetime.now(IST).strftime("%Y-%m-%d")

    res = con.execute("SELECT MAX(date) FROM prices").fetchone()
    latest_date_str = str(res[0]) if (res and res[0]) else None

    if not latest_date_str:
        return {
            "check": "latest_date_recency",
            "passed": False,
            "message": "FAILED: Table 'prices' has no date records.",
        }

    d_ref = datetime.strptime(reference_date, "%Y-%m-%d")
    d_latest = datetime.strptime(latest_date_str, "%Y-%m-%d")
    lag_days = (d_ref - d_latest).days

    passed = 0 <= lag_days <= max_lag_days
    return {
        "check": "latest_date_recency",
        "passed": passed,
        "latest_date": latest_date_str,
        "reference_date": reference_date,
        "lag_days": lag_days,
        "max_lag_days": max_lag_days,
        "message": (
            f"Passed: Latest date {latest_date_str} is within {max_lag_days} days of {reference_date} (lag: {lag_days}d)."
            if passed
            else f"FAILED: Latest date {latest_date_str} exceeds {max_lag_days} days lag from {reference_date} (lag: {lag_days}d)."
        ),
    }


def check_all_tickers_present(
    con: duckdb.DuckDBPyConnection,
    universe_path: Path | str = "data/reference/universe.csv",
    min_trading_days: int = 1000,
    documented_new_listings: set[str] | None = None,
) -> dict[str, Any]:
    """Verify that every ticker in universe.csv and required benchmarks has data in prices,
    and that every ticker/benchmark has at least min_trading_days (1000) of history,
    failing if fewer unless documented in documented_new_listings.
    """
    if documented_new_listings is None:
        documented_new_listings = set()

    stocks = load_universe(universe_path)
    expected_tickers = set(s.ticker for s in stocks)
    expected_tickers.add("^NSEI")
    expected_tickers.add("synthetic_midcap")
    expected_tickers.add("synthetic_smallcap")

    # Fetch tickers and trading day counts
    ticker_counts = dict(
        con.execute("SELECT ticker, COUNT(date) FROM prices GROUP BY ticker").fetchall()
    )
    present_tickers = set(ticker_counts.keys())
    missing = expected_tickers - present_tickers

    # Check for short history (< min_trading_days)
    short_history = {
        ticker: cnt
        for ticker, cnt in ticker_counts.items()
        if cnt < min_trading_days and ticker not in documented_new_listings
    }

    # Verify ^CNXSC is purged
    has_cnxsc = "^CNXSC" in present_tickers

    passed = (len(missing) == 0) and (len(short_history) == 0) and (not has_cnxsc)

    failure_reasons = []
    if missing:
        failure_reasons.append(f"Missing {len(missing)} tickers: {sorted(missing)}")
    if short_history:
        failure_reasons.append(f"Tickers with < {min_trading_days} trading days: {short_history}")
    if has_cnxsc:
        failure_reasons.append("Unsanitized benchmark '^CNXSC' is present in database.")

    message = (
        f"Passed: All {len(expected_tickers)} required tickers/benchmarks present with >= {min_trading_days} trading days."
        if passed
        else f"FAILED: {'; '.join(failure_reasons)}"
    )

    return {
        "check": "all_tickers_present",
        "passed": passed,
        "expected_count": len(expected_tickers),
        "present_count": len(present_tickers),
        "missing_tickers": sorted(missing),
        "short_history_tickers": short_history,
        "message": message,
    }


def check_price_integrity(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Verify physical price integrity: prices > 0, high >= low, close within [low, high]."""
    query = """
        SELECT ticker, date, open, high, low, close
        FROM prices
        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
           OR high < low
           OR close < (low * 0.999) OR close > (high * 1.001)
        LIMIT 10
    """
    bad_rows = con.execute(query).fetchall()
    passed = len(bad_rows) == 0
    return {
        "check": "price_integrity",
        "passed": passed,
        "violations_sample": bad_rows,
        "message": "Passed: All price values meet physical integrity bounds." if passed else f"FAILED: Found physical price errors: {bad_rows}",
    }


def run_all_dq_checks(
    db_path: Path | str = "data/market.duckdb",
    universe_path: Path | str = "data/reference/universe.csv",
    max_lag_days: int = 5,
    reference_date: str | None = None,
) -> dict[str, Any]:
    """Run all data quality checks on the DuckDB database and return a summary report."""
    p = Path(db_path)
    if not p.exists():
        raise DataQualityError(f"Database not found at {db_path}")

    con = duckdb.connect(str(p), read_only=True)
    try:
        results = [
            check_no_duplicate_keys(con),
            check_no_future_dates(con, reference_date=reference_date),
            check_latest_date_recency(con, max_lag_days=max_lag_days, reference_date=reference_date),
            check_all_tickers_present(con, universe_path=universe_path),
            check_price_integrity(con),
        ]

        all_passed = all(r["passed"] for r in results)
        summary = {
            "all_passed": all_passed,
            "checks": results,
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r["passed"]),
        }

        if not all_passed:
            failed_msgs = [r["message"] for r in results if not r["passed"]]
            raise DataQualityError("Data quality checks failed:\n" + "\n".join(failed_msgs))

        return summary
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Data Quality checks on DuckDB database.")
    parser.add_argument("--db-path", default="data/market.duckdb", help="Path to DuckDB database")
    parser.add_argument("--universe-path", default="data/reference/universe.csv", help="Path to universe reference CSV")
    parser.add_argument("--max-lag-days", type=int, default=5, help="Maximum allowable lag for latest price date")
    parser.add_argument("--reference-date", default=None, help="Reference date YYYY-MM-DD (defaults to current IST date)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RUNNING DATA QUALITY (DQ) CHECKS")
    print(f"Target Database: {args.db_path}")
    print("=" * 60)

    try:
        report = run_all_dq_checks(
            db_path=args.db_path,
            universe_path=args.universe_path,
            max_lag_days=args.max_lag_days,
            reference_date=args.reference_date,
        )
        for check in report["checks"]:
            print(f"[PASS] {check['check']:25} -> {check['message']}")
        print("\nAll DQ checks passed successfully! (100% integrity)")
        sys.exit(0)
    except DataQualityError as err:
        print(f"\n[FAIL] {err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error during DQ verification: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
