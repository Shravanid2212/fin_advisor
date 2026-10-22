"""Unit tests for feature engineering and financial risk metrics (PRD §9).

Adheres to AGENTS.md Hard Rule 7: Every metric function needs a docstring with the formula and a unit test.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.metrics import (
    assign_risk_buckets,
    calculate_annualized_volatility,
    calculate_beta,
    calculate_cagr,
    calculate_downside_deviation,
    calculate_drawdowns,
    calculate_historical_var,
    calculate_momentum,
    calculate_rolling_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_liquidity_and_volume_zscore,
)
from src.features.build_features import build_all_stock_features


def test_calculate_cagr():
    """Verify CAGR calculation against known mathematical doubling."""
    # 100 to 200 in 1 year = 100%
    assert math.isclose(calculate_cagr(100.0, 200.0, 1.0), 1.0, rel_tol=1e-5)
    # 100 to 400 in 2 years = 100% (sqrt(4) - 1)
    assert math.isclose(calculate_cagr(100.0, 400.0, 2.0), 1.0, rel_tol=1e-5)
    # Zero or negative inputs raise ValueError
    with pytest.raises(ValueError):
        calculate_cagr(0.0, 100.0, 1.0)
    with pytest.raises(ValueError):
        calculate_cagr(100.0, 100.0, 0.0)


def test_calculate_annualized_volatility():
    """Verify annualized volatility scales by sqrt(252)."""
    # Daily returns with constant standard deviation 0.01 (1%)
    # Annualized vol = 0.01 * sqrt(252) ~= 0.158745
    rets = np.array([0.01, -0.01, 0.01, -0.01] * 250)
    vol = calculate_annualized_volatility(rets, trading_days=252)
    expected = float(pd.Series(rets).std(ddof=1) * np.sqrt(252))
    assert math.isclose(vol, expected, rel_tol=1e-5)
    assert calculate_annualized_volatility([]) == 0.0


def test_calculate_sharpe_ratio():
    """Verify Sharpe Ratio: (R - Rf) / vol."""
    # R = 15%, vol = 10%, Rf = 5% -> Sharpe = (0.15 - 0.05) / 0.10 = 1.0
    assert math.isclose(calculate_sharpe_ratio(0.15, 0.10, 0.05), 1.0, rel_tol=1e-5)
    assert calculate_sharpe_ratio(0.15, 0.0, 0.05) == 0.0


def test_calculate_downside_deviation_and_sortino():
    """Verify downside deviation and Sortino ratio penalizing only negative returns."""
    rets = pd.Series([0.05, 0.02, -0.02, -0.04])
    downside_dev = calculate_downside_deviation(rets, target_return=0.0, trading_days=252)
    assert downside_dev > 0.0
    # Sortino ratio
    sortino = calculate_sortino_ratio(0.12, downside_dev, risk_free_rate=0.065)
    assert math.isclose(sortino, (0.12 - 0.065) / downside_dev, rel_tol=1e-5)
    assert calculate_sortino_ratio(0.12, 0.0) == 0.0


def test_calculate_drawdowns_and_recovery_days():
    """Verify underwater drawdown calculation, max drawdown, and recovery duration."""
    # Prices: 100 -> 120 (peak) -> 90 (trough, -25% from 120) -> 110 -> 120 (recovered)
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 120.0], index=pd.date_range("2026-01-01", periods=5))
    dd_series, max_dd, rec_days = calculate_drawdowns(prices)

    assert math.isclose(max_dd, (90.0 - 120.0) / 120.0, rel_tol=1e-5)  # -25%
    # Trough is index 2 (90.0), recovered is index 4 (120.0) -> 2 days recovery
    assert rec_days == 2
    assert dd_series.iloc[1] == 0.0
    assert dd_series.iloc[2] == -0.25

    # Monotonically increasing series -> max_dd = 0.0, recovery_days = 0
    mono_inc = pd.Series([10.0, 20.0, 30.0])
    _, m_dd, r_d = calculate_drawdowns(mono_inc)
    assert m_dd == 0.0
    assert r_d == 0


def test_calculate_beta():
    """Verify CAPM Beta slope."""
    bench = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    # Exactly 2x benchmark return
    stock = bench * 2.0
    assert math.isclose(calculate_beta(stock, bench), 2.0, rel_tol=1e-5)


def test_calculate_historical_var():
    """Verify 95% Historical VaR returns 5th percentile."""
    rets = np.linspace(-0.10, 0.10, 101)
    var95 = calculate_historical_var(rets, confidence_level=0.95)
    expected = float(np.percentile(rets, 5.0))
    assert math.isclose(var95, expected, rel_tol=1e-5)
    assert var95 < 0.0


def test_calculate_rolling_volatility():
    """Verify rolling volatility output dataframe schema."""
    dates = pd.date_range("2026-01-01", periods=100)
    rets = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    df_vol = calculate_rolling_volatility(rets, windows=[30, 60])
    assert "vol_30d" in df_vol.columns
    assert "vol_60d" in df_vol.columns
    assert len(df_vol) == 100


def test_calculate_momentum():
    """Verify 6M and 12M momentum lookbacks."""
    prices = pd.Series(np.linspace(100.0, 200.0, 300))
    mom = calculate_momentum(prices, lookback_days=[126, 252])
    assert "momentum_6m" in mom
    assert "momentum_12m" in mom
    assert mom["momentum_6m"] > 0.0
    assert mom["momentum_12m"] > 0.0


def test_calculate_liquidity_and_volume_zscore():
    """Verify traded value and volume z-scores."""
    df = pd.DataFrame({
        "close": [100.0] * 70,
        "volume": [1000.0] * 69 + [5000.0]
    })
    res = calculate_liquidity_and_volume_zscore(df, window=60)
    assert "traded_value" in res.columns
    assert "avg_traded_value_60d" in res.columns
    assert "volume_zscore" in res.columns
    # Last observation has abnormally high volume -> z-score must be high positive
    assert res["volume_zscore"].iloc[-1] > 2.0


def test_assign_risk_buckets():
    """Verify assignment of 5 volatility quintile risk buckets."""
    vols = pd.Series(np.linspace(0.10, 0.50, 50))
    buckets = assign_risk_buckets(vols)
    assert set(buckets.unique()) == {1, 2, 3, 4, 5}
    # Lowest volatility must be bucket 1, highest must be bucket 5
    assert buckets.iloc[0] == 1
    assert buckets.iloc[-1] == 5


@pytest.mark.realdata
def test_build_all_stock_features_on_real_db(tmp_path: Path):
    """Verify that build_all_stock_features executes and produces valid features for all 80 stocks."""
    db_file = Path("data/market.duckdb")
    if not db_file.exists():
        pytest.skip("data/market.duckdb does not exist")

    out_parquet = tmp_path / "stock_features_test.parquet"
    df_feat = build_all_stock_features(
        db_path=db_file,
        export_parquet_path=out_parquet,
    )
    assert len(df_feat) == 80
    assert "cagr" in df_feat.columns
    assert "sharpe_ratio" in df_feat.columns
    assert "risk_bucket" in df_feat.columns
    assert out_parquet.exists()
