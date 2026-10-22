"""Unit tests for sample CSV portfolio data files (Item C).

Verifies:
1. All clean sample files parse with zero errors.
2. messy_with_errors.csv produces expected validation messages and analyses valid rows.
3. empty_header_only.csv and single_stock.csv do not crash parsing, analytics, or the UI.
4. tickers_without_ns.csv normalizes and matches all stocks to universe.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config import load_config, load_universe
from src.portfolio.analytics import analyze_portfolio
from src.portfolio.holdings import parse_holdings_csv


@pytest.fixture
def universe():
    return load_universe()


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_data_dir():
    return Path(__file__).resolve().parent.parent / "sample_data"


CLEAN_FILES = [
    "bank_heavy.csv",
    "balanced_multi_sector.csv",
    "tech_heavy.csv",
    "conservative_large_cap.csv",
    "small_mid_cap_growth.csv",
    "tickers_without_ns.csv",
    "single_stock.csv",
]


@pytest.mark.realdata
@pytest.mark.parametrize("filename", CLEAN_FILES)
def test_clean_sample_files_parse_without_errors(filename, universe, sample_data_dir, config):
    """Verify each clean sample file parses with zero errors and computes valid analytics."""
    file_path = sample_data_dir / filename
    assert file_path.exists(), f"File {filename} is missing!"

    port = parse_holdings_csv(file_path, universe=universe)
    assert port.validation_report.is_valid
    assert len(port.holdings) >= 1

    # Analytics must compute without exceptions
    analytics = analyze_portfolio(port, config=config)
    assert analytics.total_value > 0.0
    assert analytics.equity_value > 0.0
    assert analytics.volatility > 0.0


@pytest.mark.realdata
def test_messy_with_errors_produces_validation_messages_and_analyses_valid_rows(
    universe, sample_data_dir, config
):
    """Verify messy_with_errors.csv produces validation issues and still analyses valid rows."""
    file_path = sample_data_dir / "messy_with_errors.csv"
    assert file_path.exists()

    port = parse_holdings_csv(file_path, universe=universe)
    report = port.validation_report

    # Issues must be reported for negative quantity, zero quantity, duplicate, or unknown tickers
    assert len(report.issues) >= 2
    issue_text = " ".join(report.issues).lower()
    assert "quantity" in issue_text or "invalid" in issue_text or "dropped" in issue_text or "consolidat" in issue_text

    # Valid stocks must be retained and analysed
    analysed_tickers = {h.ticker for h in port.holdings if h.is_analysed}
    assert "HDFCBANK.NS" in analysed_tickers
    assert "TCS.NS" in analysed_tickers
    assert "WIPRO.NS" in analysed_tickers  # Cleaned from '   wipro.ns   '

    # Analytics still executes successfully on the valid rows
    analytics = analyze_portfolio(port, config=config)
    assert analytics.equity_value > 0.0
    assert analytics.volatility > 0.0


@pytest.mark.realdata
def test_empty_header_only_does_not_crash(universe, sample_data_dir, config):
    """Verify empty_header_only.csv does not crash and returns empty portfolio."""
    file_path = sample_data_dir / "empty_header_only.csv"
    assert file_path.exists()

    port = parse_holdings_csv(file_path, universe=universe)
    assert len(port.holdings) == 0

    analytics = analyze_portfolio(port, config=config)
    assert analytics.total_value == 0.0
    assert analytics.volatility == 0.0


@pytest.mark.realdata
def test_tickers_without_ns_matches_universe(universe, sample_data_dir):
    """Verify tickers without .NS suffix are normalized and matched to the universe."""
    file_path = sample_data_dir / "tickers_without_ns.csv"
    port = parse_holdings_csv(file_path, universe=universe)

    assert port.validation_report.is_valid
    for h in port.holdings:
        assert h.ticker.endswith(".NS")
        assert h.is_analysed
