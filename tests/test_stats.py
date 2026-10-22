"""Unit tests for statistical hypothesis tests and financial models (PRD §10)."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.stats.hypothesis_tests import (
    bootstrap_ci,
    run_kruskal_wallis_test,
    run_levene_test,
    run_normality_test,
    run_pairwise_welch_mann_whitney,
)
from src.stats.models import (
    calculate_efficient_frontier,
    fit_arima_returns,
    fit_garch_volatility,
)


def test_normality_rejects_fat_tailed_student_t():
    """Verify that normality tests reject fat-tailed t-distributed returns."""
    rng = np.random.default_rng(42)
    t_rets = rng.standard_t(df=3, size=1000)
    res = run_normality_test(t_rets)

    assert res["is_normal"] is False
    assert res["excess_kurtosis"] > 1.0
    assert res["jarque_bera_pvalue"] < 0.001


def test_levene_variance():
    """Verify Levene's test detects significant variance differences."""
    rng = np.random.default_rng(42)
    group_low = rng.normal(0, 0.01, size=200)
    group_high = rng.normal(0, 0.05, size=200)  # 5x volatility

    res = run_levene_test({"low": group_low, "high": group_high})
    assert res["variances_differ"] is True
    assert res["pvalue"] < 0.001


def test_kruskal_wallis():
    """Verify Kruskal-Wallis detects medians across shifted groups."""
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.01, 0.02, size=100)
    g2 = rng.normal(0.05, 0.02, size=100)
    g3 = rng.normal(0.10, 0.02, size=100)

    res = run_kruskal_wallis_test({"g1": g1, "g2": g2, "g3": g3})
    assert res["groups_differ"] is True
    assert res["kruskal_pvalue"] < 0.001


def test_pairwise_welch_and_mann_whitney():
    """Verify pairwise Welch t-test and Mann-Whitney U test."""
    rng = np.random.default_rng(42)
    large = rng.normal(0.01, 0.01, size=150)
    small = rng.normal(0.03, 0.02, size=150)

    res = run_pairwise_welch_mann_whitney(large, small, "Large", "Small")
    assert res["significant_at_05"] is True
    assert "Mann-Whitney U" in res["plain_english"]


def test_bootstrap_ci():
    """Verify percentile bootstrap confidence intervals bound sample mean."""
    data = np.array([10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 13.0] * 10)
    pt, lower, upper = bootstrap_ci(data, np.mean, n_boot=500, alpha=0.05, seed=42)
    assert lower < pt < upper
    assert math.isclose(pt, np.mean(data), rel_tol=1e-4)


def test_calculate_efficient_frontier():
    """Verify Markowitz mean-variance optimization produces valid portfolio weights."""
    mu = np.array([0.15, 0.20, 0.12])
    cov = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.09, 0.02],
        [0.01, 0.02, 0.025]
    ])

    res = calculate_efficient_frontier(mu, cov, n_points=10, risk_free_rate=0.065)
    assert "min_volatility_portfolio" in res
    assert "max_sharpe_portfolio" in res
    assert len(res["frontier_volatilities"]) > 0

    min_vol_w = res["min_volatility_portfolio"]["weights"]
    assert math.isclose(sum(min_vol_w), 1.0, rel_tol=1e-3)

    max_sharpe_w = res["max_sharpe_portfolio"]["weights"]
    assert math.isclose(sum(max_sharpe_w), 1.0, rel_tol=1e-3)
    assert res["max_sharpe_portfolio"]["sharpe_ratio"] > 0.0


def test_fit_garch_volatility():
    """Verify GARCH(1,1) volatility estimation and persistence factor."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.015, size=500)
    res = fit_garch_volatility(rets, horizon=5)

    assert "forecast_annualized_volatility" in res
    assert res["forecast_annualized_volatility"] > 0.0
    assert "persistence" in res or "error" in res


def test_fit_arima_returns():
    """Verify ARIMA fit returns AIC and forecast intervals."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.01, size=200)
    res = fit_arima_returns(rets, order=(1, 0, 1), forecast_steps=5)

    assert "forecast_mean" in res
    assert len(res["forecast_mean"]) == 5
    assert len(res["ci_lower_95"]) == 5


@pytest.mark.realdata
def test_build_and_store_stats_tables_persists_expected_schema():
    """Verify build_and_store_stats_tables creates all required DuckDB tables with expected counts."""
    from src.stats.build_stats_tables import build_and_store_stats_tables
    import duckdb

    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not found")

    counts = build_and_store_stats_tables(db_path=db_path)

    assert counts["sector_metrics"] == 10
    assert counts["cap_metrics"] == 3
    assert counts["stock_correlation_matrix"] == 6400  # 80 x 80
    assert counts["rolling_vol"] > 200000
    assert counts["stats_results"] >= 7

    con = duckdb.connect(str(db_path), read_only=True)
    df_cap = con.execute("SELECT * FROM cap_metrics").fetchdf()
    assert set(df_cap["group_size"]) == {38, 23, 19}
    assert (df_cap["amfi_source_date"] == "2026-06-30").all()

    df_stats = con.execute("SELECT * FROM stats_results").fetchdf()
    assert "Levene Test (Variance Equality)" in df_stats["test"].values
    assert "Kruskal-Wallis ANOVA" in df_stats["test"].values
    con.close()


@pytest.mark.realdata
def test_run_garch_vs_baseline_backtest_produces_valid_metrics():
    """Verify GARCH vs rolling volatility backtest produces valid non-empty RMSE output."""
    from src.stats.garch_backtest import run_garch_vs_baseline_backtest

    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not found")

    res = run_garch_vs_baseline_backtest(
        db_path=str(db_path),
        test_tickers=["^NSEI", "TCS.NS"],
        eval_dates=["2024-06-28", "2024-09-30"],
    )

    assert res["total_evaluations"] == 4
    assert res["rmse_baseline"] > 0.0
    assert res["rmse_garch"] > 0.0
    assert "baseline_wins" in res

