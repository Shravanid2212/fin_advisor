"""Portfolio analytics engine.

Source of truth: docs/PRD.md Section 12.
Adheres strictly to AGENTS.md Hard Rules:
- Rule 2: Per-ticker rebased and relative returns, never global min-max scaling.
- Rule 5: User portfolio lives in session memory only (never written to disk or logs).
- Rule 7: Every metric function needs a docstring with the formula and a unit test.
- Rule 8: 252 trading days for annualisation.

Computes:
1. Portfolio valuation and gain/loss by stock, sector, and cap class.
2. Asset allocation (equity, debt, gold) and drift vs target profile.
3. Concentration limit flags (stock > 25%, sector > 40%).
4. Portfolio annualized volatility, historical 95% VaR, and max drawdown.
5. Average pairwise stock correlation and 0-100 diversification score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.config import AppConfig, TargetAllocationLevel, load_config
from src.features.metrics import (
    calculate_annualized_volatility,
    calculate_drawdowns,
    calculate_historical_var,
)
from src.portfolio.holdings import FundBucket, PortfolioInput, StockHolding, ValidationReport


@dataclass
class PortfolioAnalytics:
    """Consolidated portfolio analytics output."""

    total_value: float
    equity_value: float
    debt_value: float
    gold_value: float
    other_value: float
    stock_equity_value: float
    fund_equity_value: float
    total_unrealized_gain_loss: float
    total_gain_loss_pct: float
    stock_weights: dict[str, float]  # ticker -> weight in equity (0.0 to 1.0)
    stock_portfolio_weights: dict[str, float]  # ticker -> weight in total portfolio
    sector_weights: dict[str, float]  # sector -> weight in equity
    cap_weights: dict[str, float]  # 'large', 'mid', 'small', 'unanalysed' -> weight in equity
    concentration_warnings: list[str]  # Flags for breached stock or sector limits
    volatility: float  # Annualized portfolio volatility (252 days)
    max_drawdown: float  # Historical portfolio max drawdown
    var_95: float  # Daily historical 95% VaR
    avg_correlation: float  # Average pairwise correlation of stock holdings
    diversification_score: float  # 0 to 100 diversification score
    diversification_ratio: float  # Ratio of weighted vols to portfolio vol
    asset_allocation_current: dict[str, float]  # 'equity', 'debt', 'gold' percentages
    asset_allocation_target: dict[str, float] | None = None
    drift: dict[str, float] | None = None
    equity_split_current: dict[str, float] = field(default_factory=dict)
    equity_split_drift: dict[str, float] | None = None
    not_analysed_holdings: list[str] = field(default_factory=list)
    validation_report: ValidationReport = field(default_factory=ValidationReport)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_latest_prices_from_db(
    tickers: list[str],
    db_path: Path | str = "data/market.duckdb",
) -> dict[str, float]:
    """Retrieve latest close price for specified tickers from DuckDB.

    Args:
        tickers: List of ticker strings.
        db_path: Path to database.

    Returns:
        Mapping of ticker to latest close price.
    """
    if not tickers:
        return {}

    p = Path(db_path)
    if not p.exists():
        return {}

    con = duckdb.connect(str(p), read_only=True)
    try:
        # Get maximum date present in prices table
        max_date_row = con.execute("SELECT MAX(date) FROM prices").fetchone()
        if not max_date_row or not max_date_row[0]:
            return {}
        max_date = max_date_row[0]

        placeholders = ", ".join(["?"] * len(tickers))
        query = f"SELECT ticker, close FROM prices WHERE ticker IN ({placeholders}) AND date = ?"
        params = list(tickers) + [max_date]
        rows = con.execute(query, params).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    finally:
        con.close()


def load_stock_daily_returns_from_db(
    tickers: list[str],
    db_path: Path | str = "data/market.duckdb",
) -> pd.DataFrame:
    """Retrieve daily simple returns for specified tickers from DuckDB.

    Excludes filled/interpolated rows (is_filled = FALSE).

    Args:
        tickers: List of ticker strings.
        db_path: Path to database.

    Returns:
        DataFrame indexed by date with columns for each ticker.
    """
    if not tickers:
        return pd.DataFrame()

    p = Path(db_path)
    if not p.exists():
        return pd.DataFrame()

    con = duckdb.connect(str(p), read_only=True)
    try:
        placeholders = ", ".join(["?"] * len(tickers))
        query = f"""
            SELECT date, ticker, simple_return
            FROM prices
            WHERE ticker IN ({placeholders}) AND is_filled = FALSE
            ORDER BY date, ticker
        """
        df = con.execute(query, tickers).fetchdf()
        if df.empty:
            return pd.DataFrame()

        pivoted = df.pivot(index="date", columns="ticker", values="simple_return")
        return pivoted
    finally:
        con.close()


def compute_portfolio_valuation(
    portfolio: PortfolioInput,
    latest_prices: dict[str, float] | None = None,
    db_path: Path | str = "data/market.duckdb",
) -> PortfolioInput:
    """Value all holdings at latest close prices and compute unrealized gains/losses.

    Formula:
        current_value = quantity * latest_price
        gain_loss = (latest_price - avg_buy_price) * quantity
        gain_loss_pct = ((latest_price - avg_buy_price) / avg_buy_price) * 100

    Args:
        portfolio: Input portfolio with holdings and funds.
        latest_prices: Optional pre-loaded mapping of ticker to price.
        db_path: Path to DuckDB if prices need to be fetched.

    Returns:
        Same portfolio with updated valuation fields on StockHolding objects.
    """
    if latest_prices is None:
        tickers = [h.ticker for h in portfolio.holdings if h.is_analysed]
        latest_prices = load_latest_prices_from_db(tickers, db_path=db_path)

    for h in portfolio.holdings:
        if h.ticker in latest_prices:
            h.latest_price = float(latest_prices[h.ticker])
        elif h.avg_buy_price is not None and not h.is_analysed:
            # For unanalysed stocks without live DB prices, use average buy price for valuation
            h.latest_price = float(h.avg_buy_price)
        else:
            h.latest_price = 0.0
            if h.is_analysed and h.ticker not in portfolio.validation_report.missing_price_tickers:
                portfolio.validation_report.missing_price_tickers.append(h.ticker)
                portfolio.validation_report.issues.append(
                    f"No recent market price found for {h.ticker}; current value set to 0.0."
                )

        h.current_value = round(h.quantity * h.latest_price, 2)

        if h.avg_buy_price is not None and h.avg_buy_price > 0:
            h.gain_loss = round((h.latest_price - h.avg_buy_price) * h.quantity, 2)
            h.gain_loss_pct = round(((h.latest_price - h.avg_buy_price) / h.avg_buy_price) * 100.0, 2)
        else:
            h.gain_loss = 0.0
            h.gain_loss_pct = 0.0

    return portfolio


def compute_asset_allocation(
    portfolio: PortfolioInput,
    target_allocation: TargetAllocationLevel | None = None,
) -> dict[str, Any]:
    """Compute current asset allocation across Equity, Debt, and Gold, and drift vs target.

    Formulas:
        Equity Value = Analysed Stocks + Unanalysed Stocks + Equity Funds (large, mid/small, index)
        Debt Value = Debt Funds
        Gold Value = Gold Funds
        Total Value = Equity + Debt + Gold + Other
        Drift = Current Asset % - Target Asset %

    Args:
        portfolio: Valued portfolio.
        target_allocation: Optional TargetAllocationLevel from risk profile.

    Returns:
        Dictionary containing values, current percentages, equity split, and drifts.
    """
    stock_equity = sum(h.current_value for h in portfolio.holdings)

    fund_large = sum(f.amount for f in portfolio.funds if f.category == "large_cap")
    fund_mid_small = sum(f.amount for f in portfolio.funds if f.category == "mid_small_cap")
    fund_index = sum(f.amount for f in portfolio.funds if f.category == "index")
    fund_equity = fund_large + fund_mid_small + fund_index

    debt_value = sum(f.amount for f in portfolio.funds if f.category == "debt")
    gold_value = sum(f.amount for f in portfolio.funds if f.category == "gold")
    other_value = sum(f.amount for f in portfolio.funds if f.category == "other")

    equity_value = stock_equity + fund_equity
    total_value = equity_value + debt_value + gold_value + other_value

    if total_value > 0.0:
        current_pct = {
            "equity": round(equity_value / total_value, 4),
            "debt": round(debt_value / total_value, 4),
            "gold": round(gold_value / total_value, 4),
        }
    else:
        current_pct = {"equity": 0.0, "debt": 0.0, "gold": 0.0}

    # Cap breakdown within equity bucket
    large_stock_val = sum(h.current_value for h in portfolio.holdings if h.cap_class == "large")
    mid_stock_val = sum(h.current_value for h in portfolio.holdings if h.cap_class == "mid")
    small_stock_val = sum(h.current_value for h in portfolio.holdings if h.cap_class == "small")
    unanalysed_val = sum(h.current_value for h in portfolio.holdings if not h.is_analysed)

    total_large = large_stock_val + fund_large + fund_index
    total_mid = mid_stock_val + (0.5 * fund_mid_small)
    total_small = small_stock_val + (0.5 * fund_mid_small)

    if equity_value > 0.0:
        equity_split = {
            "large": round(total_large / equity_value, 4),
            "mid": round(total_mid / equity_value, 4),
            "small": round(total_small / equity_value, 4),
            "unanalysed": round(unanalysed_val / equity_value, 4),
        }
    else:
        equity_split = {"large": 0.0, "mid": 0.0, "small": 0.0, "unanalysed": 0.0}

    # Drift vs target profile
    drift: dict[str, float] | None = None
    target_pct: dict[str, float] | None = None
    split_drift: dict[str, float] | None = None

    if target_allocation is not None:
        target_pct = {
            "equity": target_allocation.equity,
            "debt": target_allocation.debt,
            "gold": target_allocation.gold,
        }
        drift = {
            "equity": round(current_pct["equity"] - target_allocation.equity, 4),
            "debt": round(current_pct["debt"] - target_allocation.debt, 4),
            "gold": round(current_pct["gold"] - target_allocation.gold, 4),
        }
        split_drift = {
            "large": round(equity_split["large"] - target_allocation.equity_split.large, 4),
            "mid": round(equity_split["mid"] - target_allocation.equity_split.mid, 4),
            "small": round(equity_split["small"] - target_allocation.equity_split.small, 4),
        }

    return {
        "total_value": round(total_value, 2),
        "equity_value": round(equity_value, 2),
        "debt_value": round(debt_value, 2),
        "gold_value": round(gold_value, 2),
        "other_value": round(other_value, 2),
        "stock_equity_value": round(stock_equity, 2),
        "fund_equity_value": round(fund_equity, 2),
        "current_pct": current_pct,
        "target_pct": target_pct,
        "drift": drift,
        "equity_split": equity_split,
        "equity_split_drift": split_drift,
    }


def check_concentration_limits(
    portfolio: PortfolioInput,
    config: AppConfig | None = None,
    equity_value: float | None = None,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Check stock and sector concentration against limits in config.yaml.

    PRD Section 12 rule:
        Single stock concentration limit: default 25% (0.25) of equity.
        Single sector concentration limit: default 40% (0.40) of equity.

    Args:
        portfolio: Valued portfolio.
        config: Optional AppConfig.
        equity_value: Optional total equity value. If None, computed from holdings and equity funds.

    Returns:
        Tuple of (warnings, stock_weights, sector_weights).
    """
    if config is None:
        config = load_config()

    stock_limit = config.concentration_limits.stock
    sector_limit = config.concentration_limits.sector

    if equity_value is None:
        stock_equity = sum(h.current_value for h in portfolio.holdings)
        fund_equity = sum(
            f.amount for f in portfolio.funds if f.category in ["large_cap", "mid_small_cap", "index"]
        )
        equity_value = stock_equity + fund_equity

    warnings: list[str] = []
    stock_weights: dict[str, float] = {}
    sector_values: dict[str, float] = {}

    if equity_value <= 0.0:
        return warnings, {}, {}

    # Stock weights and concentration
    for h in portfolio.holdings:
        w = h.current_value / equity_value
        stock_weights[h.ticker] = round(w, 4)
        if w > stock_limit:
            pct = round(w * 100.0, 1)
            lim_pct = round(stock_limit * 100.0, 0)
            warnings.append(
                f"Concentration warning: {h.ticker} is {pct}% of your equity (limit {lim_pct:.0f}%)."
            )

        if h.is_analysed and h.sector:
            sector_values[h.sector] = sector_values.get(h.sector, 0.0) + h.current_value

    # Sector weights and concentration
    sector_weights: dict[str, float] = {}
    for sector, val in sector_values.items():
        sw = val / equity_value
        sector_weights[sector] = round(sw, 4)
        if sw > sector_limit:
            pct = round(sw * 100.0, 1)
            lim_pct = round(sector_limit * 100.0, 0)
            warnings.append(
                f"Concentration warning: {sector} is {pct}% of your equity (limit {lim_pct:.0f}%)."
            )

    return warnings, stock_weights, sector_weights


def compute_portfolio_risk_metrics(
    portfolio: PortfolioInput,
    returns_df: pd.DataFrame | None = None,
    db_path: Path | str = "data/market.duckdb",
    trading_days: int = 252,
) -> dict[str, Any]:
    """Compute portfolio volatility, max drawdown, historical VaR, pairwise correlation, and diversification score.

    Formulas:
        Portfolio Daily Return:
            R_p(t) = sum(w_i * R_i(t))
            where w_i = value_i / sum(analysed_values).
        Portfolio Volatility:
            sigma_p = std(R_p) * sqrt(252).
        Portfolio Max Drawdown:
            Peak-to-trough drop of cumulative compounded returns prod(1 + R_p).
        Portfolio Historical VaR (95%):
            5th percentile of daily R_p.
        Average Pairwise Correlation:
            rho_bar = (2 / (N*(N-1))) * sum_{i < j} corr(i, j).
            rho_bar = 1.0 for single stock; 0.0 for empty portfolio.
        Diversification Ratio:
            DR = sum(w_i * sigma_i) / sigma_p.
        Diversification Score (0 to 100):
            Score = 100 * min(1.0, max(0.0, 1 - sigma_p / sum(w_i * sigma_i)) / (1 - 1/sqrt(N))).
            Score = 0 for single stock or perfect co-movement; approaches 100 for uncorrelated baskets.

    Args:
        portfolio: Valued portfolio.
        returns_df: Optional pre-loaded daily returns DataFrame.
        db_path: Database path if returns need to be loaded.
        trading_days: Annualization factor (default 252).

    Returns:
        Dictionary of risk metrics.
    """
    analysed_holdings = [h for h in portfolio.holdings if h.is_analysed and h.current_value > 0]
    unanalysed_tickers = [h.ticker for h in portfolio.holdings if not h.is_analysed]

    # Edge Case 1: Empty or Fund-Only portfolio
    if not analysed_holdings:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "avg_correlation": 0.0,
            "diversification_score": 0.0,
            "diversification_ratio": 1.0,
            "correlation_matrix": pd.DataFrame(),
            "analysed_stock_count": 0,
            "unanalysed_stock_count": len(unanalysed_tickers),
        }

    tickers = [h.ticker for h in analysed_holdings]
    total_analysed_val = sum(h.current_value for h in analysed_holdings)
    weights = np.array([h.current_value / total_analysed_val for h in analysed_holdings])

    if returns_df is None:
        returns_df = load_stock_daily_returns_from_db(tickers, db_path=db_path)

    # Filter to available tickers and drop incomplete rows
    available_tickers = [t for t in tickers if t in returns_df.columns]
    if not available_tickers:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "avg_correlation": 0.0,
            "diversification_score": 0.0,
            "diversification_ratio": 1.0,
            "correlation_matrix": pd.DataFrame(),
            "analysed_stock_count": 0,
            "unanalysed_stock_count": len(unanalysed_tickers),
        }

    # Re-normalize weights if some tickers lacked price history in returns_df
    sub_df = returns_df[available_tickers].dropna()
    if sub_df.empty:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "avg_correlation": 0.0,
            "diversification_score": 0.0,
            "diversification_ratio": 1.0,
            "analysed_stock_count": len(available_tickers),
            "unanalysed_stock_count": len(unanalysed_tickers),
        }

    w_dict = {h.ticker: h.current_value for h in analysed_holdings if h.ticker in available_tickers}
    tot_w = sum(w_dict.values())
    w_vec = np.array([w_dict[t] / tot_w for t in available_tickers])

    # Edge Case 2: Single-Stock Portfolio
    if len(available_tickers) == 1:
        series = sub_df[available_tickers[0]]
        vol = calculate_annualized_volatility(series, trading_days=trading_days)
        # Reconstruct price series from daily returns for drawdown calculation
        cum_prices = (1.0 + series).cumprod() * 100.0
        _, mdd, _ = calculate_drawdowns(cum_prices)
        var_95 = calculate_historical_var(series, confidence_level=0.95)

        return {
            "volatility": round(vol, 4),
            "max_drawdown": round(mdd, 4),
            "var_95": round(var_95, 4),
            "avg_correlation": 1.0,
            "diversification_score": 0.0,
            "diversification_ratio": 1.0,
            "correlation_matrix": pd.DataFrame([[1.0]], index=available_tickers, columns=available_tickers),
            "analysed_stock_count": 1,
            "unanalysed_stock_count": len(unanalysed_tickers),
        }

    # Multi-Stock Portfolio (N >= 2)
    ret_matrix = sub_df.values
    port_returns = ret_matrix @ w_vec
    port_series = pd.Series(port_returns, index=sub_df.index)

    # 1. Volatility
    port_vol = calculate_annualized_volatility(port_series, trading_days=trading_days)

    # 2. Max Drawdown
    cum_port = (1.0 + port_series).cumprod() * 100.0
    _, port_mdd, _ = calculate_drawdowns(cum_port)

    # 3. VaR 95%
    port_var = calculate_historical_var(port_series, confidence_level=0.95)

    # 4. Correlation Matrix and Average Pairwise Correlation
    corr_df = sub_df.corr()
    corr_matrix = corr_df.values
    n = len(available_tickers)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_corrs = corr_matrix[triu_indices]
    avg_corr = float(np.mean(pairwise_corrs)) if len(pairwise_corrs) > 0 else 1.0

    # 5. Diversification Ratio and 0-100 Score
    # Individual asset volatilities
    indiv_vols = np.array([
        calculate_annualized_volatility(sub_df[col], trading_days=trading_days)
        for col in available_tickers
    ])
    weighted_vol = float(np.sum(w_vec * indiv_vols))

    if port_vol > 0.0:
        dr = weighted_vol / port_vol
    else:
        dr = 1.0

    # Theoretical maximum diversification benefit for N equal uncorrelated assets: 1 - 1/sqrt(N)
    max_theoretical_benefit = 1.0 - (1.0 / math.sqrt(n))
    actual_benefit = max(0.0, 1.0 - (port_vol / weighted_vol)) if weighted_vol > 0 else 0.0

    if max_theoretical_benefit > 0:
        div_score = min(100.0, (actual_benefit / max_theoretical_benefit) * 100.0)
    else:
        div_score = 0.0

    return {
        "volatility": round(port_vol, 4),
        "max_drawdown": round(port_mdd, 4),
        "var_95": round(port_var, 4),
        "avg_correlation": round(avg_corr, 4),
        "diversification_score": round(div_score, 2),
        "diversification_ratio": round(dr, 4),
        "correlation_matrix": corr_df,
        "analysed_stock_count": n,
        "unanalysed_stock_count": len(unanalysed_tickers),
    }


def analyze_portfolio(
    portfolio: PortfolioInput,
    target_allocation: TargetAllocationLevel | None = None,
    config: AppConfig | None = None,
    db_path: Path | str = "data/market.duckdb",
    returns_df: pd.DataFrame | None = None,
    latest_prices: dict[str, float] | None = None,
    trading_days: int = 252,
) -> PortfolioAnalytics:
    """Run full end-to-end portfolio analytics pipeline.

    Args:
        portfolio: Validated PortfolioInput container.
        target_allocation: Optional investor TargetAllocationLevel.
        config: Optional AppConfig.
        db_path: Database path for market data.
        returns_df: Optional daily returns matrix.
        latest_prices: Optional latest prices dictionary.
        trading_days: 252 trading days per year.

    Returns:
        Consolidated PortfolioAnalytics object.
    """
    if config is None:
        config = load_config()

    # 1. Valuation
    valued_portfolio = compute_portfolio_valuation(portfolio, latest_prices=latest_prices, db_path=db_path)

    # 2. Asset Allocation & Drift
    allocation = compute_asset_allocation(valued_portfolio, target_allocation=target_allocation)

    # 3. Concentration Limits
    warnings, stock_eq_weights, sector_weights = check_concentration_limits(
        valued_portfolio, config=config, equity_value=allocation["equity_value"]
    )

    # Stock weights in total portfolio
    tot_val = allocation["total_value"]
    stock_port_weights = {
        h.ticker: round(h.current_value / tot_val, 4) if tot_val > 0 else 0.0
        for h in valued_portfolio.holdings
    }

    # 4. Risk Metrics
    risk_metrics = compute_portfolio_risk_metrics(
        valued_portfolio, returns_df=returns_df, db_path=db_path, trading_days=trading_days
    )

    # 5. Unrealized Gain/Loss across analysed stocks
    total_gain_loss = sum(h.gain_loss for h in valued_portfolio.holdings)
    total_cost_basis = sum(
        (h.avg_buy_price * h.quantity)
        for h in valued_portfolio.holdings
        if h.avg_buy_price is not None and h.avg_buy_price > 0
    )
    total_gain_loss_pct = (
        round((total_gain_loss / total_cost_basis) * 100.0, 2) if total_cost_basis > 0 else 0.0
    )

    not_analysed = [h.ticker for h in valued_portfolio.holdings if not h.is_analysed]

    return PortfolioAnalytics(
        total_value=allocation["total_value"],
        equity_value=allocation["equity_value"],
        debt_value=allocation["debt_value"],
        gold_value=allocation["gold_value"],
        other_value=allocation["other_value"],
        stock_equity_value=allocation["stock_equity_value"],
        fund_equity_value=allocation["fund_equity_value"],
        total_unrealized_gain_loss=round(total_gain_loss, 2),
        total_gain_loss_pct=total_gain_loss_pct,
        stock_weights=stock_eq_weights,
        stock_portfolio_weights=stock_port_weights,
        sector_weights=sector_weights,
        cap_weights=allocation["equity_split"],
        concentration_warnings=warnings,
        volatility=risk_metrics["volatility"],
        max_drawdown=risk_metrics["max_drawdown"],
        var_95=risk_metrics["var_95"],
        avg_correlation=risk_metrics["avg_correlation"],
        diversification_score=risk_metrics["diversification_score"],
        diversification_ratio=risk_metrics["diversification_ratio"],
        asset_allocation_current=allocation["current_pct"],
        asset_allocation_target=allocation["target_pct"],
        drift=allocation["drift"],
        equity_split_current=allocation["equity_split"],
        equity_split_drift=allocation["equity_split_drift"],
        not_analysed_holdings=not_analysed,
        validation_report=valued_portfolio.validation_report,
        correlation_matrix=risk_metrics.get("correlation_matrix", pd.DataFrame()),
    )
