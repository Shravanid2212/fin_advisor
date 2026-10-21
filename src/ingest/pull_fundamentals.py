"""Snapshot fundamental metrics for the stock universe.

Captures market cap, P/E, dividend yield, sector, and industry for each ticker.
Stores dated snapshot in data/raw/YYYY-MM-DD/fundamentals.csv.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.config import AppConfig, load_config, load_universe

logger = logging.getLogger(__name__)


def fetch_ticker_fundamentals(ticker: str, max_retries: int = 3) -> dict[str, Any]:
    """Fetch fundamental snapshot for a ticker from yfinance."""
    record = {
        "ticker": ticker,
        "market_cap": None,
        "trailing_pe": None,
        "forward_pe": None,
        "dividend_yield": None,
        "sector": None,
        "industry": None,
        "long_name": None,
        "status": "FAILED",
    }

    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            if info:
                record["market_cap"] = info.get("marketCap")
                record["trailing_pe"] = info.get("trailingPE")
                record["forward_pe"] = info.get("forwardPE")
                record["dividend_yield"] = info.get("dividendYield")
                record["sector"] = info.get("sector")
                record["industry"] = info.get("industry")
                record["long_name"] = info.get("longName") or info.get("shortName")
                record["status"] = "OK"
                return record
        except Exception as exc:
            logger.warning("Attempt %d failed fetching fundamentals for %s: %s", attempt + 1, ticker, exc)
            time.sleep(1.0)

    return record


def pull_all_fundamentals(
    config: AppConfig | None = None,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Pull fundamental snapshots for all stocks in the universe and store to raw data layer.

    Args:
        config: Application configuration.
        output_dir: Output directory (defaults to data/raw/YYYY-MM-DD).

    Returns:
        DataFrame of captured fundamentals.
    """
    if config is None:
        config = load_config()

    date_str = datetime.now().strftime("%Y-%m-%d")
    raw_path = Path(output_dir or f"data/raw/{date_str}")
    raw_path.mkdir(parents=True, exist_ok=True)

    universe = config.universe or load_universe()
    records: list[dict[str, Any]] = []

    for idx, stock in enumerate(universe, start=1):
        fund = fetch_ticker_fundamentals(stock.ticker)
        fund["snapshot_date"] = date_str
        fund["universe_cap_class"] = stock.cap_class
        fund["universe_sector"] = stock.sector
        records.append(fund)
        if idx % 10 == 0:
            time.sleep(0.5)

    df = pd.DataFrame(records)
    out_file = raw_path / "fundamentals.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved fundamentals snapshot to %s (%d rows)", out_file, len(df))
    return df
