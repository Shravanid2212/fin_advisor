"""Financial feature engineering and risk metrics pure functions.

Adheres strictly to AGENTS.md rules:
- Rule 2: No global min-max scaling across tickers.
- Rule 7: Every metric function includes a docstring with its exact mathematical formula and a unit test.
- Rule 8: Annualization uses 252 trading days. Returns are simple unless named log_return.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def calculate_cagr(start_price: float, end_price: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Formula:
        CAGR = (P_end / P_start) ** (1 / years) - 1.0

    Args:
        start_price: Initial price (> 0).
        end_price: Final price (> 0).
        years: Elapsed time in years (> 0).

    Returns:
        Annualized compound growth rate as a float.
    """
    if start_price <= 0 or end_price <= 0:
        raise ValueError(f"Prices must be strictly positive: start={start_price}, end={end_price}")
    if years <= 0:
        raise ValueError(f"Years must be strictly positive: years={years}")

    return float((end_price / start_price) ** (1.0 / years) - 1.0)


def calculate_annualized_volatility(daily_returns: pd.Series | np.ndarray, trading_days: int = 252) -> float:
    """Calculate annualized standard deviation of simple returns.

    Formula:
        sigma_annual = std(daily_returns, ddof=1) * sqrt(trading_days)

    Args:
        daily_returns: Series or 1D array of daily simple returns.
        trading_days: Number of trading days in a year (default 252).

    Returns:
        Annualized volatility as a float.
    """
    clean_rets = pd.Series(daily_returns).dropna()
    if len(clean_rets) < 2:
        return 0.0
    return float(clean_rets.std(ddof=1) * np.sqrt(trading_days))


def calculate_sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float = 0.065,
) -> float:
    """Calculate the Sharpe Ratio.

    Formula:
        Sharpe = (R_annual - R_f) / sigma_annual

    Args:
        annual_return: Annualized return (e.g. CAGR or annualized mean).
        annual_volatility: Annualized return volatility (> 0).
        risk_free_rate: Risk-free rate (default 0.065 for 10Y Indian G-Sec).

    Returns:
        Sharpe ratio as a float. Returns 0.0 if volatility is zero or negative.
    """
    if annual_volatility <= 0:
        return 0.0
    return float((annual_return - risk_free_rate) / annual_volatility)


def calculate_downside_deviation(
    daily_returns: pd.Series | np.ndarray,
    target_return: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Calculate annualized downside deviation (semi-deviation below target return).

    Formula:
        downside_diff = min(0, r_i - r_target)
        downside_dev = sqrt( (1 / N) * sum(downside_diff ** 2) ) * sqrt(trading_days)

    Args:
        daily_returns: Series or array of daily returns.
        target_return: Minimum acceptable daily return (default 0.0).
        trading_days: Number of trading days in a year (default 252).

    Returns:
        Annualized downside deviation as a float.
    """
    clean_rets = pd.Series(daily_returns).dropna()
    if len(clean_rets) == 0:
        return 0.0
    diff = clean_rets - target_return
    downside_diff = np.minimum(diff, 0.0)
    variance_downside = np.mean(downside_diff**2)
    return float(np.sqrt(variance_downside) * np.sqrt(trading_days))


def calculate_sortino_ratio(
    annual_return: float,
    downside_deviation: float,
    risk_free_rate: float = 0.065,
) -> float:
    """Calculate the Sortino Ratio using downside risk.

    Formula:
        Sortino = (R_annual - R_f) / sigma_downside

    Args:
        annual_return: Annualized return.
        downside_deviation: Annualized downside deviation (> 0).
        risk_free_rate: Risk-free rate (default 0.065).

    Returns:
        Sortino ratio as a float. Returns 0.0 if downside deviation is zero.
    """
    if downside_deviation <= 0:
        return 0.0
    return float((annual_return - risk_free_rate) / downside_deviation)


def calculate_drawdowns(price_series: pd.Series) -> tuple[pd.Series, float, int]:
    """Calculate cumulative underwater drawdown series, maximum drawdown, and peak-to-recovery days.

    Formulas:
        Peak_t = max_{s <= t} P_s
        Drawdown_t = (P_t - Peak_t) / Peak_t
        MaxDrawdown = min(Drawdown_t)
        RecoveryDays = Number of trading days from trough date until price reaches or exceeds Peak_t

    Args:
        price_series: Time series of prices indexed chronologically.

    Returns:
        Tuple containing:
        - drawdown_series: Series of percentage drawdowns (<= 0.0).
        - max_drawdown: Minimum drawdown value (<= 0.0).
        - recovery_days: Trading days from trough to full recovery (-1 if not yet recovered).
    """
    s = pd.Series(price_series).dropna()
    if len(s) == 0:
        return pd.Series(dtype=float), 0.0, 0

    peaks = s.cummax()
    drawdowns = (s - peaks) / peaks

    max_dd = float(drawdowns.min())
    if max_dd >= 0.0:
        return drawdowns, 0.0, 0

    trough_idx = drawdowns.idxmin()
    trough_pos = s.index.get_loc(trough_idx)
    peak_val_at_trough = peaks.loc[trough_idx]

    # Look for recovery after trough
    post_trough = s.iloc[trough_pos:]
    recovered_subset = post_trough[post_trough >= peak_val_at_trough]

    if len(recovered_subset) > 0:
        recovery_idx = recovered_subset.index[0]
        recovery_pos = s.index.get_loc(recovery_idx)
        recovery_days = int(recovery_pos - trough_pos)
    else:
        recovery_days = -1  # Still underwater / not yet recovered

    return drawdowns, max_dd, recovery_days


def calculate_beta(
    stock_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
) -> float:
    """Calculate Systematic Risk (CAPM Beta) against benchmark.

    Formula:
        Beta = Cov(r_stock, r_benchmark) / Var(r_benchmark)

    Args:
        stock_returns: Array or Series of stock returns.
        benchmark_returns: Array or Series of benchmark returns.

    Returns:
        Beta coefficient as a float. Returns 1.0 if benchmark variance is zero.
    """
    df = pd.DataFrame({"stock": stock_returns, "bench": benchmark_returns}).dropna()
    if len(df) < 2:
        return 1.0

    cov_matrix = np.cov(df["stock"], df["bench"], ddof=1)
    bench_var = cov_matrix[1, 1]
    if bench_var <= 0:
        return 1.0
    return float(cov_matrix[0, 1] / bench_var)


def calculate_historical_var(
    daily_returns: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Historical Value at Risk (VaR).

    Formula:
        VaR_alpha = Percentile(daily_returns, (1 - confidence_level) * 100)

    Args:
        daily_returns: Series or array of daily returns.
        confidence_level: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        Value at Risk as a daily return quantile (typically negative, e.g. -0.025).
    """
    clean_rets = pd.Series(daily_returns).dropna()
    if len(clean_rets) == 0:
        return 0.0
    alpha_pct = (1.0 - confidence_level) * 100.0
    return float(np.percentile(clean_rets, alpha_pct))


def calculate_rolling_volatility(
    daily_returns: pd.Series,
    windows: list[int] | None = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate rolling annualized volatility across multiple lookback windows.

    Formula:
        vol_rolling_w = std(daily_returns, window=w) * sqrt(trading_days)

    Args:
        daily_returns: Daily returns series indexed by date.
        windows: List of window lengths in trading days (default [30, 90, 252]).
        trading_days: Trading days per year (default 252).

    Returns:
        DataFrame with columns 'vol_{w}d' indexed by date.
    """
    if windows is None:
        windows = [30, 90, 252]

    res = pd.DataFrame(index=daily_returns.index)
    for w in windows:
        min_p = max(5, int(w * 0.6))
        res[f"vol_{w}d"] = daily_returns.rolling(window=w, min_periods=min_p).std(ddof=1) * np.sqrt(trading_days)
    return res


def calculate_momentum(
    price_series: pd.Series,
    lookback_days: list[int] | None = None,
) -> dict[str, float]:
    """Calculate historical price momentum returns over standard horizons.

    Formulas:
        mom_6m = P_t / P_{t - 126} - 1.0
        mom_12m = P_t / P_{t - 252} - 1.0

    Args:
        price_series: Daily price series sorted chronologically.
        lookback_days: Trading day lookbacks (default [126, 252] representing 6M and 12M).

    Returns:
        Dictionary of momentum return percentages keyed by horizon.
    """
    if lookback_days is None:
        lookback_days = [126, 252]

    s = pd.Series(price_series).dropna()
    mom = {}
    n = len(s)
    for d in lookback_days:
        label = "6m" if d == 126 else ("12m" if d == 252 else f"{d}d")
        if n > d and s.iloc[-d - 1] > 0:
            mom[f"momentum_{label}"] = float(s.iloc[-1] / s.iloc[-d - 1] - 1.0)
        else:
            mom[f"momentum_{label}"] = 0.0
    return mom


def calculate_liquidity_and_volume_zscore(
    df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Compute average daily traded value (liquidity) and 60-day rolling volume z-scores.

    Formulas:
        TradedValue = Close * Volume
        Volume_Z = (Volume_t - mean_60d(Volume)) / std_60d(Volume)

    Args:
        df: DataFrame containing 'close' and 'volume' columns.
        window: Rolling lookback window in trading days (default 60).

    Returns:
        DataFrame with 'traded_value', 'avg_traded_value_60d', and 'volume_zscore'.
    """
    out = pd.DataFrame(index=df.index)
    out["traded_value"] = df["close"] * df["volume"]
    min_p = max(5, int(window * 0.5))
    out["avg_traded_value_60d"] = out["traded_value"].rolling(window=window, min_periods=min_p).mean()

    vol = df["volume"]
    vol_mean = vol.rolling(window=window, min_periods=min_p).mean()
    vol_std = vol.rolling(window=window, min_periods=min_p).std(ddof=1)

    out["volume_zscore"] = np.where(vol_std > 0, (vol - vol_mean) / vol_std, 0.0)
    return out


def assign_risk_buckets(annual_volatilities: pd.Series) -> pd.Series:
    """Assign equities into 5 risk buckets based on volatility quintiles (PRD §9).

    Mapping:
        1: Very Low (Quintile 1)
        2: Low (Quintile 2)
        3: Moderate (Quintile 3)
        4: High (Quintile 4)
        5: Very High (Quintile 5)

    Args:
        annual_volatilities: Series of annualized volatilities across the universe.

    Returns:
        Series of integer risk bucket labels (1 to 5).
    """
    s = pd.Series(annual_volatilities)
    if len(s) < 5:
        return pd.Series(3, index=s.index)

    try:
        labels = [1, 2, 3, 4, 5]
        return pd.qcut(s, q=5, labels=labels).astype(int)
    except ValueError:
        ranks = s.rank(method="first")
        return pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
