"""Unit tests for the recommendation engine (src/recommend/recommender.py).

Verifies:
1. Excluded sectors never appear in recommendations.
2. Suggested weights sum to 100% of the basket, no weight exceeds 20% cap, and weights differ between stocks.
3. Level 1 strictly contains no small caps and no bucket above 2.
4. Level 2 and below contain no bucket above 2. Level 3 contains no bucket above 3.
5. Overweight sectors are penalized in sector gap and explanation text never says 'adds exposure' for them.
6. Empty / fund-only portfolios use real correlation against ^NSEI instead of 0.
7. Level sensitivity explanations are generated dynamically from pick list differences.
8. Deterministic reproducibility across repeated calls.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
import pytest

from src.config import load_config, load_universe
from src.portfolio.analytics import analyze_portfolio
from src.portfolio.holdings import PortfolioInput, parse_manual_holdings
from src.recommend.recommender import (
    generate_recommendations,
    load_stock_features_from_db,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def universe():
    return load_universe()


@pytest.fixture
def features_df():
    db_path = Path("data/market.duckdb")
    if not db_path.exists():
        pytest.skip("data/market.duckdb not present")
    return load_stock_features_from_db(db_path=db_path)


# ---------------------------------------------------------------------------
# Test 1: Excluded sectors never appear
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_excluded_sectors_never_appear(config, universe, features_df):
    """Excluded sectors must never appear in recommendations."""
    excluded = ["Banking and Financials", "IT", "Energy and Power"]
    empty_port = PortfolioInput(holdings=[], funds=[])

    res = generate_recommendations(
        portfolio=empty_port,
        profile_level=3,
        excluded_sectors=excluded,
        features_df=features_df,
        config=config,
    )

    assert len(res.recommendations) > 0
    recommended_sectors = [r.sector for r in res.recommendations]
    for sec in recommended_sectors:
        assert sec not in excluded


# ---------------------------------------------------------------------------
# Test 2: Weights sum to 100%, capped at 20%, and differ between stocks
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_weights_sum_to_100_percent_and_differ(config, universe, features_df):
    """Suggested weights must sum to 100%, never exceed 20%, and vary across picks."""
    empty_port = PortfolioInput(holdings=[], funds=[])

    res = generate_recommendations(
        portfolio=empty_port,
        profile_level=3,
        features_df=features_df,
        config=config,
    )

    assert len(res.recommendations) == 8
    # Sum to 100% of the suggested basket
    assert math.isclose(res.total_suggested_weight, 1.0, abs_tol=1e-3)
    # Each <= 20% cap from config.yaml
    cap = config.recommender.per_stock_weight_cap
    assert cap == 0.20
    for r in res.recommendations:
        assert r.suggested_weight <= 0.20
        assert r.suggested_weight > 0.0

    # Weights must differ between stocks (inverse volatility proportions)
    distinct_weights = set(r.suggested_weight for r in res.recommendations)
    assert len(distinct_weights) > 1

    # Target equity share is separate and matches profile level
    assert res.target_equity_share == 0.50


# ---------------------------------------------------------------------------
# Test 3: Risk bucket and cap constraints across profile levels
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_risk_bucket_and_cap_limits_by_level(config, universe, features_df):
    """Verify Level 1 has no small caps and no bucket > 2; Level 2 max bucket 2; Level 3 max bucket 3."""
    empty_port = PortfolioInput(holdings=[], funds=[])

    # Level 1: no small caps, no bucket > 2
    res_lvl1 = generate_recommendations(
        portfolio=empty_port, profile_level=1, features_df=features_df, config=config
    )
    for r in res_lvl1.recommendations:
        assert r.cap_class == "large"
        assert r.risk_bucket <= 2

    # Level 2: no small caps, no bucket > 2
    res_lvl2 = generate_recommendations(
        portfolio=empty_port, profile_level=2, features_df=features_df, config=config
    )
    for r in res_lvl2.recommendations:
        assert r.cap_class in ["large", "mid"]
        assert r.risk_bucket <= 2

    # Level 3: no bucket > 3
    res_lvl3 = generate_recommendations(
        portfolio=empty_port, profile_level=3, features_df=features_df, config=config
    )
    for r in res_lvl3.recommendations:
        assert r.risk_bucket <= 3


# ---------------------------------------------------------------------------
# Test 4: Sector gap penalizes overweight sectors and phrasing check
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_overweight_sector_phrasing_and_gap(config, universe, features_df):
    """Explanation text must never say 'adds exposure' for an overweight sector."""
    # Bank-heavy portfolio has 100% in Banking and Financials
    port = parse_manual_holdings(
        [
            {"ticker": "HDFCBANK.NS", "quantity": 100, "avg_buy_price": 700.0},
            {"ticker": "ICICIBANK.NS", "quantity": 50, "avg_buy_price": 1350.0},
        ],
        universe=universe,
    )

    res = generate_recommendations(
        portfolio=port, profile_level=3, features_df=features_df, config=config
    )

    for r in res.recommendations:
        # If holding in that sector was >= 10%, explanation must not say 'adds exposure'
        if r.sector == "Banking and Financials":
            assert "adds exposure" not in r.explanation.lower()
            assert "maintains exposure" in r.explanation.lower()


# ---------------------------------------------------------------------------
# Test 5: Empty portfolios use real correlation against ^NSEI
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_empty_portfolio_uses_nifty_correlation(config, universe, features_df):
    """Portfolios with no stocks must compute correlation with ^NSEI instead of 0."""
    empty_port = PortfolioInput(holdings=[], funds=[])

    res = generate_recommendations(
        portfolio=empty_port, profile_level=3, features_df=features_df, config=config
    )

    for r in res.recommendations:
        # Correlation with ^NSEI should be non-zero and reasonable (typically between 0.2 and 0.8)
        corr_nifty = r.metrics["corr_with_port"]
        assert corr_nifty != 0.0
        assert 0.10 <= corr_nifty <= 0.90


# ---------------------------------------------------------------------------
# Test 6: Deterministic reproducibility
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_deterministic_reproducibility(config, universe, features_df):
    """The same input must always produce the exact same output."""
    port = parse_manual_holdings(
        [{"ticker": "INFY.NS", "quantity": 50, "avg_buy_price": 1400.0}],
        universe=universe,
    )

    res1 = generate_recommendations(
        portfolio=port, profile_level=3, features_df=features_df, config=config
    )
    res2 = generate_recommendations(
        portfolio=port, profile_level=3, features_df=features_df, config=config
    )

    picks1 = [r.ticker for r in res1.recommendations]
    picks2 = [r.ticker for r in res2.recommendations]
    weights1 = [r.suggested_weight for r in res1.recommendations]
    weights2 = [r.suggested_weight for r in res2.recommendations]

    assert picks1 == picks2
    assert weights1 == weights2


# ---------------------------------------------------------------------------
# Test 7: Strictly no buy/sell advice phrasing
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_no_buy_sell_wording(config, universe, features_df):
    """UI explanations and summaries must strictly avoid buy/sell advice phrasing."""
    empty_port = PortfolioInput(holdings=[], funds=[])
    res = generate_recommendations(
        portfolio=empty_port, profile_level=3, features_df=features_df, config=config
    )

    forbidden_pattern = re.compile(r"\b(buy|sell|strong buy|strong sell)\b", re.IGNORECASE)

    for r in res.recommendations:
        assert not forbidden_pattern.search(r.explanation), f"Forbidden wording in: {r.explanation}"

    for note in res.notes:
        assert not forbidden_pattern.search(note), f"Forbidden wording in note: {note}"


# ---------------------------------------------------------------------------
# Test 8: Dynamic level sensitivity explanations
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_dynamic_level_sensitivity(config, universe, features_df):
    """Level sensitivity summaries must be generated dynamically from actual pick changes."""
    port = parse_manual_holdings(
        [
            {"ticker": "HDFCBANK.NS", "quantity": 100, "avg_buy_price": 700.0},
            {"ticker": "ICICIBANK.NS", "quantity": 50, "avg_buy_price": 1350.0},
        ],
        universe=universe,
    )

    res = generate_recommendations(
        portfolio=port, profile_level=3, features_df=features_df, config=config
    )

    sens = res.level_sensitivity
    assert sens.level_plus_1 == 4
    assert sens.level_minus_1 == 2
    assert "At Level 4:" in sens.level_plus_1_summary
    assert "At Level 2:" in sens.level_minus_1_summary
    if sens.level_plus_1_new_picks:
        assert any(t in sens.level_plus_1_summary for t in sens.level_plus_1_new_picks)


# ---------------------------------------------------------------------------
# Test 9: Favoured sector boosts score, adds explanation note, and obeys hard filters
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_favoured_sector_preference_end_to_end(config, universe, features_df):
    """Verify favoured sectors rank higher, get tagged in explanation, produce note when no stocks fit, and never bypass bucket limits."""
    empty_port = PortfolioInput(holdings=[], funds=[])

    # 1. Favoured sector ranks higher and gets explanation note
    res_base = generate_recommendations(
        portfolio=empty_port,
        profile_level=3,
        features_df=features_df,
        config=config,
    )
    # Pick a sector from eligible list
    fav_sec = "Pharma and Healthcare"
    res_fav = generate_recommendations(
        portfolio=empty_port,
        profile_level=3,
        favoured_sectors=[fav_sec],
        features_df=features_df,
        config=config,
    )
    # Check that picks in fav_sec have matching reason
    fav_picks = [r for r in res_fav.recommendations if r.sector == fav_sec]
    assert len(fav_picks) > 0
    for p in fav_picks:
        assert f"Matches your preference for {fav_sec}" in p.explanation

    # Check composite score of any stock in fav_sec is boosted compared to baseline
    base_scores = {r.ticker: r.composite_score for r in res_base.recommendations}
    for p in fav_picks:
        if p.ticker in base_scores:
            assert p.composite_score > base_scores[p.ticker]

    # 2. Avoided sectors never appear
    res_avoid = generate_recommendations(
        portfolio=empty_port,
        profile_level=3,
        excluded_sectors=["IT", "FMCG"],
        features_df=features_df,
        config=config,
    )
    avoided_picks = [r for r in res_avoid.recommendations if r.sector in ["IT", "FMCG"]]
    assert len(avoided_picks) == 0

    # 3. Favoured sector with no eligible stock produces note
    res_note = generate_recommendations(
        portfolio=empty_port,
        profile_level=1,  # Level 1: large cap only, risk bucket <= 2
        favoured_sectors=["Metals"],  # Metals has no large cap bucket <= 2 stock
        features_df=features_df,
        config=config,
    )
    assert any("No stocks in Metals fit your risk level" in n for n in res_note.notes)

    # 4. Conservative profile favouring high-risk sector still gets NO pick above bucket 2
    for r in res_note.recommendations:
        assert r.risk_bucket <= 2
        assert r.cap_class == "large"

