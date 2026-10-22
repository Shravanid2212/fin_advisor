"""Unit tests for the portfolio engine (src/portfolio/holdings.py and src/portfolio/analytics.py).

Verifies:
1. Bank-heavy fixture portfolio: hand-checkable values, sector concentration flag triggered (>40%).
2. Balanced fixture portfolio: multi-cap, multi-sector, no concentration breaches, drift vs profile.
3. Fund-only fixture portfolio: zero stocks, asset allocation drift, no crashes.
4. Messy CSV parser: bad rows isolated without crash, unknown tickers flagged, duplicates merged.
5. Empty portfolio and single-stock portfolio edge cases.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import TargetAllocationLevel, EquitySplitConfig, load_config, load_universe
from src.portfolio.analytics import (
    analyze_portfolio,
    check_concentration_limits,
    compute_asset_allocation,
    compute_portfolio_risk_metrics,
    compute_portfolio_valuation,
)
from src.portfolio.holdings import (
    PortfolioInput,
    StockHolding,
    parse_fund_buckets,
    parse_holdings_csv,
    parse_manual_holdings,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def universe():
    return load_universe()


@pytest.fixture
def moderate_target():
    """Level 3 Moderate target allocation from config."""
    return TargetAllocationLevel(
        level=3,
        label="Moderate",
        equity=0.50,
        debt=0.40,
        gold=0.10,
        equity_split=EquitySplitConfig(large=0.70, mid=0.25, small=0.05),
    )


# ---------------------------------------------------------------------------
# Fixture 1: Bank-Heavy Portfolio Test
# ---------------------------------------------------------------------------
def test_bank_heavy_fixture_portfolio(config, universe):
    """Test bank-heavy portfolio triggering sector concentration limit (> 40%)."""
    # 3 bank stocks
    holdings_input = [
        {"ticker": "HDFCBANK.NS", "quantity": 100, "avg_buy_price": 700.0},
        {"ticker": "ICICIBANK.NS", "quantity": 50, "avg_buy_price": 1350.0},
        {"ticker": "KOTAKBANK.NS", "quantity": 40, "avg_buy_price": 1700.0},
    ]
    prices = {
        "HDFCBANK.NS": 712.0,
        "ICICIBANK.NS": 1381.0,
        "KOTAKBANK.NS": 1776.0,
    }

    # Synthetic daily returns with high bank correlation (~0.7)
    np.random.seed(42)
    days = 500
    base_factor = np.random.normal(0.0005, 0.015, days)
    ret_hdfc = base_factor + np.random.normal(0, 0.005, days)
    ret_icici = base_factor + np.random.normal(0, 0.006, days)
    ret_kotak = base_factor + np.random.normal(0, 0.007, days)

    returns_df = pd.DataFrame(
        {"HDFCBANK.NS": ret_hdfc, "ICICIBANK.NS": ret_icici, "KOTAKBANK.NS": ret_kotak},
        index=pd.date_range("2024-01-01", periods=days, freq="B"),
    )

    port = parse_manual_holdings(holdings_input, universe=universe)
    analytics = analyze_portfolio(
        port,
        config=config,
        returns_df=returns_df,
        latest_prices=prices,
    )

    # 1. Hand-checkable valuation:
    # HDFC: 100 * 712.0 = 71,200
    # ICICI: 50 * 1381.0 = 69,050
    # KOTAK: 40 * 1776.0 = 71,040
    # Total = 211,290.0
    expected_total = 71200.0 + 69050.0 + 71040.0
    assert analytics.total_value == expected_total
    assert analytics.equity_value == expected_total

    # 2. Sector concentration:
    # All 3 stocks are 'Banking and Financials'
    assert analytics.sector_weights.get("Banking and Financials") == 1.0
    # Must trigger sector concentration warning (limit is 40%)
    assert any("Banking and Financials" in w and "40%" in w for w in analytics.concentration_warnings)

    # 3. High correlation among banks:
    assert analytics.avg_correlation > 0.65

    # 4. Diversification score should be modest due to high correlation and single-sector exposure
    assert analytics.diversification_score < 60.0


# ---------------------------------------------------------------------------
# Fixture 2: Balanced Multi-Cap Portfolio Test
# ---------------------------------------------------------------------------
def test_balanced_fixture_portfolio(config, universe, moderate_target):
    """Test balanced multi-cap portfolio with stocks, debt, and gold."""
    stock_rows = [
        {"ticker": "RELIANCE.NS", "quantity": 20, "avg_buy_price": 1200.0},   # Energy & Power (Large)
        {"ticker": "TCS.NS", "quantity": 10, "avg_buy_price": 2400.0},        # IT (Large)
        {"ticker": "SUNPHARMA.NS", "quantity": 20, "avg_buy_price": 1150.0},  # Pharma (Large)
        {"ticker": "FEDERALBNK.NS", "quantity": 100, "avg_buy_price": 180.0}, # Banking (Mid)
        {"ticker": "VOLTAS.NS", "quantity": 15, "avg_buy_price": 950.0},      # Consumer (Mid)
        {"ticker": "JINDALSAW.NS", "quantity": 25, "avg_buy_price": 500.0},   # Metals (Small)
    ]
    fund_rows = [
        {"category": "debt", "amount": 80000.0},
        {"category": "gold", "amount": 20000.0},
    ]

    prices = {
        "RELIANCE.NS": 1200.0,    # 24,000
        "TCS.NS": 2400.0,         # 24,000
        "SUNPHARMA.NS": 1200.0,   # 24,000
        "FEDERALBNK.NS": 190.0,   # 19,000
        "VOLTAS.NS": 1000.0,      # 15,000
        "JINDALSAW.NS": 540.0,    # 13,500
    }
    # Total stock equity: 24,000 + 24,000 + 24,000 + 19,000 + 15,000 + 13,500 = 119,500
    # Total portfolio value: 119,500 + 80,000 + 20,000 = 219,500

    # Independent synthetic returns
    np.random.seed(123)
    days = 500
    ret_dict = {
        t: np.random.normal(0.0004, 0.015 + 0.002 * i, days)
        for i, t in enumerate(prices.keys())
    }
    returns_df = pd.DataFrame(ret_dict, index=pd.date_range("2024-01-01", periods=days, freq="B"))

    port = parse_manual_holdings(stock_rows, fund_rows=fund_rows, universe=universe)
    analytics = analyze_portfolio(
        port,
        target_allocation=moderate_target,
        config=config,
        returns_df=returns_df,
        latest_prices=prices,
    )

    # 1. Valuation
    assert analytics.total_value == 219500.0
    assert analytics.stock_equity_value == 119500.0
    assert analytics.debt_value == 80000.0
    assert analytics.gold_value == 20000.0

    # 2. No concentration limit warnings (all stocks < 25% of equity, all sectors < 40%)
    assert len(analytics.concentration_warnings) == 0

    # 3. Multi-cap presence
    assert analytics.cap_weights["large"] > 0.50
    assert analytics.cap_weights["mid"] > 0.20
    assert analytics.cap_weights["small"] > 0.10

    # 4. Asset allocation and drift vs Level 3 Moderate (50% eq, 40% debt, 10% gold)
    # Current equity %: 119,500 / 219,500 = 54.44% (drift ~ +4.44%)
    # Current debt %: 80,000 / 219,500 = 36.45% (drift ~ -3.55%)
    # Current gold %: 20,000 / 219,500 = 9.11% (drift ~ -0.89%)
    assert math.isclose(analytics.asset_allocation_current["equity"], 0.5444, abs_tol=1e-3)
    assert math.isclose(analytics.drift["equity"], 0.0444, abs_tol=1e-3)
    assert math.isclose(analytics.drift["debt"], -0.0355, abs_tol=1e-3)

    # 5. Low correlation across independent sectors -> high diversification score
    assert analytics.avg_correlation < 0.20
    assert analytics.diversification_score > 70.0


# ---------------------------------------------------------------------------
# Fixture 3: Fund-Only Portfolio Test
# ---------------------------------------------------------------------------
def test_fund_only_fixture_portfolio(config, universe, moderate_target):
    """Test fund-only portfolio: 0 stocks, handles risk metrics gracefully, computes drift."""
    fund_rows = [
        {"category": "debt", "amount": 40000.0},
        {"category": "gold", "amount": 10000.0},
        {"category": "index", "amount": 50000.0},
    ]

    port = parse_manual_holdings([], fund_rows=fund_rows, universe=universe)
    analytics = analyze_portfolio(
        port,
        target_allocation=moderate_target,
        config=config,
    )

    # 1. Valuation
    assert analytics.total_value == 100000.0
    assert analytics.equity_value == 50000.0
    assert analytics.debt_value == 40000.0
    assert analytics.gold_value == 10000.0
    assert analytics.stock_equity_value == 0.0

    # 2. Risk metrics should be 0.0 without errors
    assert analytics.volatility == 0.0
    assert analytics.max_drawdown == 0.0
    assert analytics.var_95 == 0.0
    assert analytics.avg_correlation == 0.0
    assert analytics.diversification_score == 0.0

    # 3. Exact target match (50% eq, 40% debt, 10% gold)
    assert analytics.asset_allocation_current["equity"] == 0.50
    assert analytics.asset_allocation_current["debt"] == 0.40
    assert analytics.asset_allocation_current["gold"] == 0.10
    assert analytics.drift["equity"] == 0.0
    assert analytics.drift["debt"] == 0.0
    assert analytics.drift["gold"] == 0.0


# ---------------------------------------------------------------------------
# Messy CSV Parsing and Robustness Test
# ---------------------------------------------------------------------------
def test_messy_csv_parsing_and_isolation(universe):
    """Verify parser never crashes on bad rows, isolates errors, and merges duplicates."""
    raw_csv = """ticker,quantity,avg_buy_price
    INFY.NS, 50, 1400.0
    tcs, 25, 3200.0
    UNKNOWN_CO, 100, 50.0
    SBIN.NS, -10, 500.0
    HDFCBANK.NS, invalid_qty, 1600.0
    infy.ns, 50, 1600.0
    RELIANCE.NS, 0, 1200.0
    , 10, 100.0
    """

    port = parse_holdings_csv(raw_csv, universe=universe)
    rep = port.validation_report

    # 1. Never crashed!
    assert port is not None

    # 2. Bad quantity rows caught
    # Negative (-10), non-numeric ("invalid_qty"), zero (0)
    assert len(rep.bad_quantity_rows) == 3
    bad_reasons = [r["reason"] for r in rep.bad_quantity_rows]
    assert any("positive" in r.lower() for r in bad_reasons)

    # 3. Duplicate merging:
    # INFY.NS appears twice (50 @ 1400.0 and 50 @ 1600.0)
    # Merged: total qty = 100, avg buy price = 1500.0
    assert len(rep.duplicates_merged) == 1
    dup = rep.duplicates_merged[0]
    assert dup["ticker"] == "INFY.NS"
    assert dup["total_quantity"] == 100.0
    assert dup["merged_avg_price"] == 1500.0

    # 4. Unknown ticker flagged as not analysed
    assert "UNKNOWN_CO" in rep.unknown_tickers
    unknown_holding = next(h for h in port.holdings if h.ticker == "UNKNOWN_CO")
    assert unknown_holding.is_analysed is False
    assert unknown_holding.sector == "Unanalysed"

    # 5. Case-insensitive ticker normalization (tcs -> TCS.NS)
    tcs_holding = next(h for h in port.holdings if h.ticker == "TCS.NS")
    assert tcs_holding.is_analysed is True
    assert tcs_holding.quantity == 25.0

    # 6. Valid active holdings count: INFY.NS, TCS.NS, UNKNOWN_CO
    assert len(port.holdings) == 3


# ---------------------------------------------------------------------------
# Edge Cases: Empty and Single-Stock Portfolios
# ---------------------------------------------------------------------------
def test_empty_portfolio(config):
    """Empty portfolio returns zero values without errors."""
    port = PortfolioInput(holdings=[], funds=[])
    analytics = analyze_portfolio(port, config=config)

    assert analytics.total_value == 0.0
    assert analytics.equity_value == 0.0
    assert analytics.volatility == 0.0
    assert analytics.max_drawdown == 0.0
    assert analytics.var_95 == 0.0
    assert analytics.avg_correlation == 0.0
    assert analytics.diversification_score == 0.0
    assert len(analytics.concentration_warnings) == 0


def test_single_stock_portfolio(config, universe):
    """Single-stock portfolio has avg correlation 1.0, 0 diversification score, and flags concentration."""
    port = parse_manual_holdings(
        [{"ticker": "RELIANCE.NS", "quantity": 100, "avg_buy_price": 1000.0}],
        universe=universe,
    )

    np.random.seed(42)
    days = 300
    ret_reliance = pd.DataFrame(
        {"RELIANCE.NS": np.random.normal(0.0005, 0.015, days)},
        index=pd.date_range("2024-01-01", periods=days, freq="B"),
    )
    prices = {"RELIANCE.NS": 1202.0}

    analytics = analyze_portfolio(
        port,
        config=config,
        returns_df=ret_reliance,
        latest_prices=prices,
    )

    assert analytics.total_value == 120200.0
    assert analytics.stock_weights["RELIANCE.NS"] == 1.0
    assert analytics.avg_correlation == 1.0
    assert analytics.diversification_score == 0.0
    assert analytics.diversification_ratio == 1.0

    # Both stock limit (>25%) and sector limit (>40%) triggered
    assert len(analytics.concentration_warnings) == 2
    assert any("RELIANCE.NS is 100.0%" in w for w in analytics.concentration_warnings)
    assert any("Energy and Power is 100.0%" in w for w in analytics.concentration_warnings)


# ---------------------------------------------------------------------------
# Real Data Integration Test (against data/market.duckdb)
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_real_portfolio_analytics_on_duckdb(config, universe, moderate_target):
    """Verify end-to-end portfolio analysis querying real DuckDB database directly."""
    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not present")

    stock_rows = [
        {"ticker": "HDFCBANK.NS", "quantity": 20, "avg_buy_price": 680.0},
        {"ticker": "TCS.NS", "quantity": 10, "avg_buy_price": 2400.0},
        {"ticker": "RELIANCE.NS", "quantity": 15, "avg_buy_price": 1150.0},
    ]
    fund_rows = [
        {"category": "debt", "amount": 50000.0},
    ]

    port = parse_manual_holdings(stock_rows, fund_rows=fund_rows, universe=universe)
    analytics = analyze_portfolio(
        port,
        target_allocation=moderate_target,
        config=config,
        db_path=db_path,
    )

    # 1. Prices loaded from DB
    assert analytics.total_value > 50000.0
    assert analytics.stock_equity_value > 0.0
    assert analytics.debt_value == 50000.0

    # 2. Real risk metrics computed from prices table
    assert 0.10 < analytics.volatility < 0.40  # Annualized volatility between 10% and 40%
    assert analytics.max_drawdown < 0.0        # Max drawdown is negative
    assert analytics.var_95 < 0.0              # 5th percentile return is negative
    assert 0.0 < analytics.avg_correlation < 1.0  # Pairwise correlation between stocks
    assert 0.0 < analytics.diversification_score <= 100.0  # Diversification score

    # 3. Allocation and drift
    assert analytics.asset_allocation_current["equity"] > 0.0
    assert analytics.asset_allocation_current["debt"] > 0.0
    assert analytics.drift is not None

