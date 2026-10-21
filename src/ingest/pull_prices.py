"""Download historical market prices for the stock universe and benchmarks.

Features:
- Batches of 10 tickers with retry and exponential backoff.
- Immutable storage in data/raw/YYYY-MM-DD/ (CSV / Parquet).
- Manifest output in data/manifests/ detailing status, rows, dates.
- Detects failures and insufficient listing history (< 8 years) without silent substitution.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yfinance as yf

from src.config import AppConfig, UniverseStock, load_config, load_universe

logger = logging.getLogger(__name__)

# Proposed replacements for known flawed tickers (must be explicitly reported to user)
KNOWN_PROPOSED_REPLACEMENTS: dict[str, dict[str, str]] = {
    "TATAMOTORS.NS": {
        "reason": "Delisted / not found on Yahoo Finance due to corporate action/demerger",
        "proposed_ticker": "HEROMOTOCO.NS",
        "proposed_name": "Hero MotoCorp Limited",
        "sector": "Auto",
        "cap_class": "large",
    },
    "EQUITASBNK.NS": {
        "reason": "Under 8 years of listing history (IPO in late 2020; ~6.0 years available)",
        "proposed_ticker": "KARURVYSYA.NS",
        "proposed_name": "The Karur Vysya Bank Limited",
        "sector": "Banking and Financials",
        "cap_class": "small",
    },
}


def download_ticker_with_retry(
    ticker: str,
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker with retry and exponential backoff.

    Args:
        ticker: Yahoo ticker symbol.
        start_date: ISO date string for start.
        end_date: ISO date string for end, or None for latest.
        max_retries: Number of retry attempts.
        backoff_factor: Multiplier for sleep duration on retry.

    Returns:
        DataFrame containing OHLCV history, or empty DataFrame on failure.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            tk = yf.Ticker(ticker)
            params: dict[str, Any] = {"start": start_date, "auto_adjust": True}
            if end_date and end_date.lower() != "latest":
                params["end"] = end_date

            df = tk.history(**params)
            if df is not None and not df.empty:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                return df
            # If empty DataFrame returned, try once more or break
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff_factor**attempt)
        except Exception as exc:
            attempt += 1
            logger.warning("Attempt %d failed for %s: %s", attempt, ticker, exc)
            if attempt < max_retries:
                time.sleep(backoff_factor**attempt)

    return pd.DataFrame()


def evaluate_history_coverage(df: pd.DataFrame, min_years: float = 8.0) -> tuple[float, str, str | None, str | None]:
    """Calculate the years of listing history and evaluate whether coverage meets requirement."""
    if df.empty:
        return 0.0, "FAILED_DOWNLOAD", None, None

    first_date = df.index[0].strftime("%Y-%m-%d")
    last_date = df.index[-1].strftime("%Y-%m-%d")
    d1 = datetime.strptime(first_date, "%Y-%m-%d")
    d2 = datetime.strptime(last_date, "%Y-%m-%d")
    years = round((d2 - d1).days / 365.25, 2)

    if years < min_years:
        status = "UNDER_MIN_HISTORY"
    else:
        status = "OK"

    return years, status, first_date, last_date


def pull_all_prices(
    config: AppConfig | None = None,
    output_dir: Path | str | None = None,
    manifest_dir: Path | str | None = None,
    batch_size: int = 10,
    min_years: float = 8.0,
) -> dict[str, Any]:
    """Execute batch price downloads for all universe stocks and benchmarks.

    Writes raw CSV files to data/raw/YYYY-MM-DD/ and writes run manifest to data/manifests/.

    Returns:
        Summary dictionary containing manifests, warnings, and replacement proposals.
    """
    if config is None:
        config = load_config()

    date_str = datetime.now().strftime("%Y-%m-%d")
    raw_path = Path(output_dir or f"data/raw/{date_str}")
    raw_path.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_dir or "data/manifests")
    manifest_path.mkdir(parents=True, exist_ok=True)

    start_date = config.date_range.start
    end_date = config.date_range.end

    # List of all targets: universe stocks + benchmarks
    universe = config.universe or load_universe()
    tickers_to_pull = [s.ticker for s in universe]
    benchmark_tickers = [
        config.benchmarks.broad_market,
        config.benchmarks.midcap,
        config.benchmarks.smallcap,
    ]

    all_symbols = tickers_to_pull + [b for b in benchmark_tickers if b not in tickers_to_pull]

    manifest_records: list[dict[str, Any]] = []
    flagged_issues: list[dict[str, Any]] = []

    logger.info("Starting ingestion of %d symbols in batches of %d", len(all_symbols), batch_size)

    # Process in batches of 10
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i : i + batch_size]
        logger.info("Processing batch %d/%d: %s", (i // batch_size) + 1, (len(all_symbols) + batch_size - 1) // batch_size, batch)

        for sym in batch:
            df = download_ticker_with_retry(sym, start_date=start_date, end_date=end_date)
            years, status, first_d, last_d = evaluate_history_coverage(df, min_years=min_years)

            # Save untouched raw file
            safe_sym = sym.replace("^", "IDX_").replace(".NS", "")
            raw_file = raw_path / f"{safe_sym}.csv"
            if not df.empty:
                df.to_csv(raw_file)

            record = {
                "symbol": sym,
                "rows": len(df),
                "first_date": first_d,
                "last_date": last_d,
                "years_history": years,
                "status": status,
                "raw_file": str(raw_file),
            }
            manifest_records.append(record)

            # Check for issues in universe stocks
            matching_stocks = [s for s in universe if s.ticker == sym]
            if matching_stocks and status != "OK":
                stock = matching_stocks[0]
                reason = "Download failed or quote not found on Yahoo Finance" if status == "FAILED_DOWNLOAD" else f"Listing history ({years} yrs) is under required {min_years} years"
                proposal = KNOWN_PROPOSED_REPLACEMENTS.get(sym, {
                    "reason": reason,
                    "proposed_ticker": "N/A",
                    "proposed_name": "N/A",
                    "sector": stock.sector,
                    "cap_class": stock.cap_class,
                })
                flagged_issues.append({
                    "ticker": sym,
                    "name": stock.name,
                    "sector": stock.sector,
                    "cap_class": stock.cap_class,
                    "status": status,
                    "rows": len(df),
                    "years": years,
                    "reason": proposal["reason"],
                    "proposed_replacement": proposal.get("proposed_ticker"),
                    "proposed_name": proposal.get("proposed_name"),
                })

        # Pause slightly between batches to be respectful to endpoints
        time.sleep(1.0)

    # Write manifest
    manifest_file = manifest_path / f"manifest_{date_str}.json"
    manifest_data = {
        "run_date": date_str,
        "start_date": start_date,
        "end_date": end_date,
        "total_symbols": len(all_symbols),
        "successful_downloads": sum(1 for r in manifest_records if r["rows"] > 0),
        "failed_downloads": sum(1 for r in manifest_records if r["rows"] == 0),
        "flagged_issues_count": len(flagged_issues),
        "records": manifest_records,
        "flagged_issues": flagged_issues,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Manifest saved to %s", manifest_file)
    return manifest_data
