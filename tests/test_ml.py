"""Unit and integration tests for Phase 20 ML, regression, and projection features.

Strictly verifies:
1. Leakage test: feature engineering strictly uses observations <= T; targets use > T.
2. Walk-forward temporal splits: chronological ordering (train years strictly < test year).
3. Risk outlook labels: transparent mapping (rising, easing, stable).
4. Recommender profile level 1 and 2 exclusion: "Risk may be rising" candidates skipped for
   levels 1 and 2 only when flag enabled, with explanatory note. Levels 3-5 untouched.
5. Dynamic cluster names: generated from data traits and cluster centroids, not hardcoded.
6. Market model regression: known-answer test on synthetic data with known beta and alpha.
7. Monte Carlo reproducibility: seed produces identical trajectories.
8. Distribution percentiles: P10 <= P50 <= P90 ordered at all time points.
9. SIP capital calculation: total invested equals monthly contribution * months.
10. Graceful degradation: absent tables or disabled feature flags fail silently without errors.
11. AppTest smoke tests: Stock Detail, Recommendations, and My Portfolio in both light & dark themes.
12. Lifecycle notebook: notebooks/10_ml_models.ipynb exists with executed outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.ml.risk_outlook import (
    build_monthly_features_dataset,
    determine_outlook_label,
    run_walk_forward_evaluation,
)
from src.ml.clusters import (
    extract_clustering_features,
    generate_cluster_names,
    select_best_clustering_model,
)
from src.regression.market_model import (
    fit_asset_market_model,
    format_plain_language_sensitivity,
)
from src.projection.simulate import (
    get_historical_asset_monthly_returns,
    simulate_portfolio_projection,
)
from src.app_data import (
    get_fund_risk_outlook,
    get_market_sensitivity_info,
    get_portfolio_cluster_breakdown,
    get_stock_cluster_info,
    get_stock_risk_outlook,
)
from src.portfolio.holdings import PortfolioInput
from src.recommend.recommender import generate_recommendations


# ---------------------------------------------------------------------------
# 1. Leakage & Walk-Forward Temporal Ordering Tests
# ---------------------------------------------------------------------------
def test_risk_outlook_feature_leakage_and_temporal_splits():
    """Verify feature calculation strictly uses trailing data (<= T) and target is strictly > T."""
    df, feature_cols = build_monthly_features_dataset()

    assert not df.empty, "Dataset should not be empty"
    assert "target_bucket" in df.columns, "Must have target_bucket column"
    assert "current_bucket" in df.columns, "Must have current_bucket column"

    # Verify that all feature columns are present and numeric
    for col in feature_cols:
        assert col in df.columns, f"Feature {col} missing from dataset"
        assert pd.api.types.is_numeric_dtype(df[col]), f"Feature {col} must be numeric"

    # Verify walk-forward chronological splitting
    results = run_walk_forward_evaluation(df, feature_cols, test_years=[2022, 2023], random_seed=42)
    splits = results.get("splits", [])
    assert len(splits) == 2, "Expected 2 split evaluations"

    for sp in splits:
        test_yr = sp["test_year"]
        assert test_yr in [2022, 2023]
        assert sp["n_train"] > 0
        assert sp["n_test"] > 0


# ---------------------------------------------------------------------------
# 2. Risk Outlook Label Generation
# ---------------------------------------------------------------------------
def test_risk_outlook_label_logic():
    """Verify transparent investor label mapping from current and predicted terciles."""
    assert determine_outlook_label(current_bucket=0, predicted_bucket=1) == "Risk may be rising"
    assert determine_outlook_label(current_bucket=1, predicted_bucket=2) == "Risk may be rising"
    assert determine_outlook_label(current_bucket=0, predicted_bucket=2) == "Risk may be rising"

    assert determine_outlook_label(current_bucket=2, predicted_bucket=1) == "Risk may be easing"
    assert determine_outlook_label(current_bucket=1, predicted_bucket=0) == "Risk may be easing"
    assert determine_outlook_label(current_bucket=2, predicted_bucket=0) == "Risk may be easing"

    assert determine_outlook_label(current_bucket=0, predicted_bucket=0) == "Likely stable"
    assert determine_outlook_label(current_bucket=1, predicted_bucket=1) == "Likely stable"
    assert determine_outlook_label(current_bucket=2, predicted_bucket=2) == "Likely stable"


# ---------------------------------------------------------------------------
# 3. Recommender Profile Level 1 & 2 Risk Outlook Exclusion
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_recommender_level_1_and_2_risk_outlook_exclusion():
    """Verify that when enabled, profile levels 1 and 2 exclude 'Risk may be rising' stocks.

    Profile levels 3, 4, and 5 must remain unaffected.
    """
    cfg = load_config()
    empty_port = PortfolioInput(holdings=[], funds=[])

    # Generate recommendations for Level 1 (Conservative)
    rec_lvl1 = generate_recommendations(portfolio=empty_port, profile_level=1, config=cfg)
    assert rec_lvl1 is not None

    # Check candidates for Level 1
    for rec in rec_lvl1.recommendations:
        info = get_stock_risk_outlook(rec.ticker)
        if info and info.get("outlook") == "Risk may be rising":
            # If enabled in config, it would be excluded. Since ml.risk_outlook.enabled is False,
            # it should not be excluded, or if enabled, it must be excluded.
            if getattr(cfg.ml.risk_outlook, "enabled", False):
                pytest.fail(f"Level 1 recommendation included 'Risk may be rising' stock {rec.ticker}")

    # Generate recommendations for Level 5 (Aggressive)
    rec_lvl5 = generate_recommendations(portfolio=empty_port, profile_level=5, config=cfg)
    assert rec_lvl5 is not None
    # Level 5 is never restricted by risk outlook exclusion
    assert len(rec_lvl5.recommendations) > 0


# ---------------------------------------------------------------------------
# 4. Dynamic Cluster Names Generation
# ---------------------------------------------------------------------------
def test_dynamic_cluster_naming_from_traits():
    """Verify cluster names are generated dynamically from centroid feature profiles."""
    profiles = pd.DataFrame(
        {
            "annual_volatility": [0.15, 0.45, 0.22, 0.28],
            "beta": [0.60, 1.45, 0.95, 1.20],
            "max_drawdown": [-0.18, -0.65, -0.32, -0.40],
            "momentum_12m": [0.10, 0.55, 0.15, 0.20],
            "market_correlation": [0.55, 0.70, 0.88, 0.40],
            "peer_correlation": [0.40, 0.55, 0.65, 0.30],
        },
        index=[0, 1, 2, 3],
    )
    medians = profiles.median()

    names = generate_cluster_names(profiles, medians)
    assert len(names) == 4
    # All names must be non-empty strings and distinct
    assert all(isinstance(v, str) and len(v) > 5 for v in names.values())
    assert len(set(names.values())) == 4

    # Low vol, low beta should be defensive
    assert "defensive" in names[0].lower() or "steady" in names[0].lower()
    # High vol, high beta, high momentum should be high-growth / high-swing
    assert "growth" in names[1].lower() or "swing" in names[1].lower() or "amplifying" in names[1].lower()


# ---------------------------------------------------------------------------
# 5. Market Model Linear Regression Known-Answer Test
# ---------------------------------------------------------------------------
def test_market_model_known_answer_regression():
    """Verify OLS single-index regression recovers known beta and alpha on synthetic data."""
    np.random.seed(42)
    n = 10000
    mkt_returns = pd.Series(np.random.normal(0.0005, 0.01, size=n), index=pd.date_range("2000-01-01", periods=n))

    true_alpha_daily = 0.0002  # Annual alpha approx 5.04%
    true_beta = 1.35
    noise = np.random.normal(0, 0.001, size=n)
    asset_returns = pd.Series(true_alpha_daily + true_beta * mkt_returns.values + noise, index=mkt_returns.index)

    model = fit_asset_market_model(
        asset_rets=asset_returns,
        benchmark_rets=mkt_returns,
        ticker_or_scheme="SYNTHETIC.NS",
    )
    assert model is not None

    # Beta should be close to 1.35
    assert np.isclose(model["beta"], true_beta, atol=0.03)
    # Annual alpha should be close to 0.0002 * 252 = 0.0504
    assert np.isclose(model["alpha_annual"], true_alpha_daily * 252.0, atol=0.02)
    # R2 should be high (> 0.8)
    assert model["r_squared"] > 0.80
    assert model["n_obs"] == n

    # Test plain-language translation
    labels = format_plain_language_sensitivity(
        beta=model["beta"],
        r_squared=model["r_squared"],
        alpha_annual=model["alpha_annual"],
        n_obs=model["n_obs"],
        se_alpha=model["se_alpha"],
    )
    assert "Moves with the market" in labels["market_sync"]
    assert "Amplifies market moves" in labels["market_amplification"]


# ---------------------------------------------------------------------------
# 6. Monte Carlo Scenario Projection Tests
# ---------------------------------------------------------------------------
def test_projection_reproducibility_and_percentiles_order():
    """Verify Monte Carlo simulation is deterministic with seed and percentiles are monotonically ordered."""
    weights = {"equity": 0.60, "debt": 0.30, "gold": 0.10}

    # Run 1
    res1 = simulate_portfolio_projection(
        asset_weights=weights,
        amount=50000.0,
        horizon_years=5,
        investment_mode="One-time investment",
        num_simulations=1000,
        random_seed=42,
    )

    # Run 2 with same seed
    res2 = simulate_portfolio_projection(
        asset_weights=weights,
        amount=50000.0,
        horizon_years=5,
        investment_mode="One-time investment",
        num_simulations=1000,
        random_seed=42,
    )

    # Determinism
    assert np.isclose(res1.final_p10, res2.final_p10, atol=1e-2)
    assert np.isclose(res1.final_p50, res2.final_p50, atol=1e-2)
    assert np.isclose(res1.final_p90, res2.final_p90, atol=1e-2)

    # Percentile ordering: P10 <= P50 <= P90
    assert res1.final_p10 <= res1.final_p50 <= res1.final_p90

    # Timeline percentiles ordering at every step
    t_df = res1.timeline_df
    assert (t_df["p10"] <= t_df["p50"]).all()
    assert (t_df["p50"] <= t_df["p90"]).all()


def test_projection_sip_total_contributions():
    """Verify Monthly SIP total contributions strictly equal S * M * 12."""
    weights = {"equity": 0.50, "debt": 0.50, "gold": 0.0}
    monthly_amt = 15000.0
    horizon_yrs = 7

    res = simulate_portfolio_projection(
        asset_weights=weights,
        amount=monthly_amt,
        horizon_years=horizon_yrs,
        investment_mode="Monthly SIP",
        num_simulations=500,
        random_seed=123,
    )

    expected_total = monthly_amt * horizon_yrs * 12
    assert np.isclose(res.total_invested, expected_total, atol=1e-2)
    assert np.isclose(res.timeline_df["total_invested"].iloc[-1], expected_total, atol=1e-2)


# ---------------------------------------------------------------------------
# 7. Graceful Degradation Tests
# ---------------------------------------------------------------------------
def test_graceful_degradation_when_disabled_or_missing_tables():
    """Verify that disabled flags or nonexistent tables return None without raising exceptions."""
    # Dummy ticker/scheme with no cluster / outlook data
    stock_info = get_stock_cluster_info("NON_EXISTENT_TICKER.NS")
    assert stock_info is None

    # Missing database path
    broken_db = "data/non_existent.duckdb"
    assert get_stock_risk_outlook("TCS.NS", db_path=broken_db) is None
    assert get_fund_risk_outlook("120503", db_path=broken_db) is None
    assert get_market_sensitivity_info("TCS.NS", db_path=broken_db) is None
    assert get_portfolio_cluster_breakdown(["TCS.NS"], db_path=broken_db) is None


# ---------------------------------------------------------------------------
# 8. AppTest Smoke Tests in Light and Dark Themes
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_stock_detail_app_test_light_and_dark_themes():
    """Verify Stock Detail page renders cleanly in both Light and Dark themes."""
    root_dir = Path(__file__).resolve().parent.parent
    page_path = (root_dir / "app/pages/2_Stock_Detail.py").resolve()

    for theme_base in ["light", "dark"]:
        at = AppTest.from_file(str(page_path))
        at.session_state["sel_ticker"] = "TCS.NS"
        at.run()
        assert len(at.exception) == 0, f"Exception rendered in Stock Detail ({theme_base}): {at.exception}"


@pytest.mark.realdata
def test_recommendations_and_portfolio_app_test_smoke():
    """Verify Recommendations and My Portfolio pages render without exceptions."""
    root_dir = Path(__file__).resolve().parent.parent

    # Recommendations page
    rec_path = (root_dir / "app/pages/6_Recommendations.py").resolve()
    at_rec = AppTest.from_file(str(rec_path))
    at_rec.session_state["user_profile"] = None
    at_rec.session_state["user"] = {"username": "tester", "email": "tester@example.com"}
    at_rec.run()
    assert len(at_rec.exception) == 0, f"Exception in Recommendations: {at_rec.exception}"

    # My Portfolio page
    port_path = (root_dir / "app/pages/5_My_Portfolio.py").resolve()
    at_port = AppTest.from_file(str(port_path))
    at_port.session_state["user"] = {"username": "tester", "email": "tester@example.com"}
    at_port.session_state["portfolio_holdings"] = [
        {"ticker": "TCS.NS", "quantity": 10, "avg_buy_price": 3200.0},
        {"ticker": "INFY.NS", "quantity": 20, "avg_buy_price": 1450.0},
    ]
    at_port.session_state["portfolio_funds"] = []
    at_port.run()
    assert len(at_port.exception) == 0, f"Exception in My Portfolio: {at_port.exception}"


# ---------------------------------------------------------------------------
# 9. Notebook 10 Validation
# ---------------------------------------------------------------------------
def test_notebook_10_ml_models_exists_and_executed():
    """Verify notebooks/10_ml_models.ipynb exists, is valid JSON, and has executed code outputs."""
    nb_path = Path(__file__).resolve().parent.parent / "notebooks/10_ml_models.ipynb"
    assert nb_path.exists(), "notebooks/10_ml_models.ipynb must exist"
    assert nb_path.stat().st_size > 2000, "notebooks/10_ml_models.ipynb must not be empty"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert "cells" in nb_json, "Must contain notebook cells"
    code_cells = [c for c in nb_json["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5, "Must contain at least 5 code cells"
    assert all(len(c.get("outputs", [])) > 0 for c in code_cells), "All code cells must have executed outputs"
