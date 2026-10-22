"""Pure mathematical and financial metric functions for mutual funds (PRD Phase 19).

Formulas:
1. CAGR: ((end_nav / start_nav) ** (1 / elapsed_years)) - 1
2. Annualized Volatility: std(daily_simple_returns) * sqrt(trading_days)
3. Max Drawdown: min((nav - peak_nav) / peak_nav)
4. Sharpe Ratio: (cagr - risk_free_rate) / annualized_volatility
5. Sortino Ratio: (cagr - risk_free_rate) / annualized_downside_deviation
6. Rolling 3-Year Return Consistency: share of 3-year rolling windows outperforming Nifty 50
7. Beta: cov(fund_returns, benchmark_returns) / var(benchmark_returns)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def calculate_cagr(start_val: float, end_val: float, elapsed_years: float) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Formula:
        CAGR = ((end_val / start_val) ** (1.0 / elapsed_years)) - 1.0

    Args:
        start_val: Initial NAV (> 0).
        end_val: Final NAV (> 0).
        elapsed_years: Time elapsed in years (> 0).

    Returns:
        Float CAGR. Returns 0.0 if invalid inputs.
    """
    if start_val <= 0 or end_val <= 0 or elapsed_years <= 0:
        return 0.0
    try:
        return float((end_val / start_val) ** (1.0 / elapsed_years) - 1.0)
    except (ZeroDivisionError, OverflowError, ValueError):
        return 0.0


def calculate_annualized_volatility(returns: pd.Series | np.ndarray, trading_days: int = 252) -> float:
    """Calculate annualized standard deviation of daily returns.

    Formula:
        sigma_ann = std(returns, ddof=1) * sqrt(trading_days)
    """
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    return float(std * math.sqrt(trading_days))


def calculate_max_drawdown(nav_series: pd.Series | list[float]) -> float:
    """Calculate maximum peak-to-trough drawdown from an NAV series.

    Formula:
        drawdown_t = (nav_t - max(nav_0..t)) / max(nav_0..t)
        max_drawdown = min(drawdown_t) (<= 0)
    """
    s = pd.Series(nav_series).dropna().astype(float)
    if s.empty:
        return 0.0
    rolling_peak = s.cummax()
    drawdowns = (s - rolling_peak) / rolling_peak
    return float(drawdowns.min())


def calculate_sharpe_ratio(cagr: float, annualized_volatility: float, risk_free_rate: float = 0.065) -> float:
    """Calculate annualized Sharpe ratio.

    Formula:
        Sharpe = (CAGR - r_f) / annualized_volatility
    """
    if annualized_volatility <= 0:
        return 0.0
    return float((cagr - risk_free_rate) / annualized_volatility)


def calculate_sortino_ratio(
    daily_returns: pd.Series | np.ndarray,
    cagr: float,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> float:
    """Calculate Sortino ratio using downside standard deviation below target return.

    Formula:
        Sortino = (CAGR - r_f) / downside_deviation_annualized
        downside_dev = sqrt(mean(min(0, r_t - r_f_daily)^2)) * sqrt(trading_days)
    """
    clean = pd.Series(daily_returns).dropna().values
    if len(clean) < 2:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    negative_excess = np.minimum(0.0, clean - daily_rf)
    downside_var = np.mean(negative_excess ** 2)
    downside_dev = math.sqrt(downside_var) * math.sqrt(trading_days)

    if downside_dev <= 0:
        return 0.0
    return float((cagr - risk_free_rate) / downside_dev)


def calculate_beta(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate market Beta against the benchmark index (Nifty 50).

    Formula:
        Beta = cov(fund, bench) / var(bench)
    """
    df = pd.DataFrame({"fund": fund_returns, "bench": benchmark_returns}).dropna()
    if len(df) < 20:
        return 1.0  # Default neutral market beta
    var_bench = df["bench"].var(ddof=1)
    if var_bench <= 0:
        return 1.0
    cov = df["fund"].cov(df["bench"])
    return float(cov / var_bench)


def calculate_rolling_consistency(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    window_days: int = 756,  # ~3 calendar trading years
) -> float:
    """Calculate rolling 3-year return consistency (share of windows beating benchmark or achieving > 0).

    Formula:
        For each rolling window of length W:
            R_fund = (nav_{t} / nav_{t-W}) ** (252 / W) - 1
            R_bench = (bench_{t} / bench_{t-W}) ** (252 / W) - 1
            is_win = 1 if R_fund >= R_bench else 0
        Consistency Score = sum(is_win) / total_windows in [0.0, 1.0]

    Args:
        fund_nav: Clean daily NAV series sorted by date.
        benchmark_nav: Optional benchmark NAV / price series aligned on dates.
        window_days: Trading days in rolling window (default 756 for 3 years).

    Returns:
        Float consistency score in [0.0, 1.0].
    """
    s_fund = pd.Series(fund_nav).dropna().astype(float)
    if len(s_fund) < window_days:
        return 0.5  # Neutral default for funds with insufficient 3Y rolling history

    # Compute 3-year rolling returns
    fund_rolling_cagr = (s_fund / s_fund.shift(window_days)) ** (252.0 / window_days) - 1.0
    fund_rolling_cagr = fund_rolling_cagr.dropna()

    if fund_rolling_cagr.empty:
        return 0.5

    if benchmark_nav is not None:
        s_bench = pd.Series(benchmark_nav).dropna().astype(float)
        bench_rolling_cagr = (s_bench / s_bench.shift(window_days)) ** (252.0 / window_days) - 1.0
        # Align series
        aligned = pd.DataFrame({"fund": fund_rolling_cagr, "bench": bench_rolling_cagr}).dropna()
        if not aligned.empty:
            wins = (aligned["fund"] >= aligned["bench"]).sum()
            return float(wins / len(aligned))

    # Fallback if benchmark not provided: share of windows with positive return
    wins = (fund_rolling_cagr > 0.0).sum()
    return float(wins / len(fund_rolling_cagr))
