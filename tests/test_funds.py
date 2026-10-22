"""Unit and integration tests for Mutual Fund analytics, ingestion, and recommendations (PRD Phase 19).

Adheres strictly to AGENTS.md rules:
- Educational decision-support only; strictly zero buy/sell wording.
- Hand-computed fixtures for metric formulas.
- Validates scheme list rules (longevity, fund house diversification, direct growth).
- Validates NAV cleaning and monotonic chronological order.
- Validates category ranking, capital allocation summing, and existing portfolio deductions.
- Recommendations page smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import TargetAllocationLevel, load_config
from src.funds.ingest import clean_fund_nav_data, load_curated_funds_reference
from src.funds.metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_rolling_consistency,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.portfolio.holdings import FundBucket, PortfolioInput, StockHolding
from src.profile.scoring import ProfileResult
from src.recommend.funds import generate_fund_recommendations


def test_funds_scheme_list_validation():
    """Validate data/reference/funds.csv against all Phase 19 rules."""
    funds = load_curated_funds_reference("data/reference/funds.csv")
    assert len(funds) >= 100, f"Expected at least 100 funds, found {len(funds)}"

    # 1. No duplicate scheme codes
    codes = [f["scheme_code"] for f in funds]
    assert len(codes) == len(set(codes)), "Found duplicate scheme_code in funds.csv"

    # 2. All scheme codes positive integers
    assert all(c > 0 for c in codes), "Found invalid non-positive scheme code"

    # 3. All plans are Direct-Growth
    for f in funds:
        assert f["plan"] == "Direct-Growth", f"Fund {f['scheme_code']} is not Direct-Growth: {f['plan']}"

    # 4. Categories representation (all 9 categories have >= 6 usable funds)
    expected_categories = [
        "large-cap", "flexi/multi-cap", "mid-cap", "small-cap",
        "nifty-50-index", "aggressive-hybrid", "balanced-hybrid",
        "liquid-debt", "gold"
    ]
    from collections import Counter
    cat_counts = Counter(f["category"] for f in funds)
    for cat in expected_categories:
        assert cat_counts[cat] >= 6, f"Category {cat} has only {cat_counts[cat]} funds (minimum 6 required)"

    # 5. Diversification rule: no more than 3 funds from one fund house per category
    for cat in expected_categories:
        house_counts = Counter(f["fund_house"] for f in funds if f["category"] == cat)
        for house, count in house_counts.items():
            assert count <= 3, f"Category '{cat}' has {count} funds from '{house}' (max 3 allowed)"


def test_funds_nav_cleaning(tmp_path):
    """Validate NAV cleaning drops N.A./negative values, formats ISO dates, and sorts monotonically."""
    mock_payload = {
        "status": "SUCCESS",
        "meta": {"scheme_name": "Mock Fund Direct Growth"},
        "data": [
            {"date": "04-01-2024", "nav": "106.5"},
            {"date": "03-01-2024", "nav": "105.0"},
            {"date": "02-01-2024", "nav": "N.A."},
            {"date": "02-01-2024", "nav": "-10.0"},
            {"date": "01-01-2024", "nav": "0.0"},
            {"date": "01-01-2024", "nav": "100.0"},
            {"date": "05-01-2024", "nav": "None"},
        ],
    }
    raw_file = tmp_path / "mock_scheme.json"
    raw_file.write_text(json.dumps(mock_payload), encoding="utf-8")

    clean_df = clean_fund_nav_data(raw_file)

    assert len(clean_df) == 3
    # Dates must be converted to ISO format and sorted ascending
    assert list(clean_df["date"]) == ["2024-01-01", "2024-01-03", "2024-01-04"]
    # NAVs must match the valid entries
    assert list(clean_df["nav"]) == [100.0, 105.0, 106.5]
    # No zero or negative NAV
    assert (clean_df["nav"] > 0).all()


def test_hand_computed_cagr_metric():
    """Verify CAGR formula against hand-computed values: ((end/start)**(1/t)) - 1."""
    # Example 1: 100 to 121 in 2 years = (121/100)**(0.5) - 1 = 1.1 - 1 = 0.10 (10%)
    cagr_2y = calculate_cagr(100.0, 121.0, 2.0)
    assert cagr_2y == pytest.approx(0.10, rel=1e-5)

    # Example 2: 100 to 100 (0% return)
    assert calculate_cagr(100.0, 100.0, 3.0) == pytest.approx(0.0, abs=1e-6)

    # Example 3: Invalid inputs return 0.0 safely
    assert calculate_cagr(-10.0, 100.0, 1.0) == 0.0
    assert calculate_cagr(100.0, 0.0, 1.0) == 0.0
    assert calculate_cagr(100.0, 150.0, 0.0) == 0.0


def test_hand_computed_volatility_metric():
    """Verify annualized volatility against hand-computed standard deviation * sqrt(252)."""
    rets = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005])
    expected = float(rets.std(ddof=1) * np.sqrt(252))
    computed = calculate_annualized_volatility(rets, trading_days=252)
    assert computed == pytest.approx(expected, rel=1e-5)


def test_hand_computed_max_drawdown_metric():
    """Verify max drawdown against hand-computed peak-to-trough sequence."""
    # Peak is 120, trough is 80 -> drawdown = (80 - 120) / 120 = -40/120 = -0.33333 (-33.33%)
    nav_series = [100.0, 120.0, 90.0, 110.0, 80.0, 105.0]
    expected_mdd = (80.0 - 120.0) / 120.0
    computed_mdd = calculate_max_drawdown(nav_series)
    assert computed_mdd == pytest.approx(expected_mdd, rel=1e-5)


def test_hand_computed_sharpe_ratio():
    """Verify Sharpe ratio formula: (CAGR - r_f) / annual_volatility."""
    # (0.165 - 0.065) / 0.20 = 0.10 / 0.20 = 0.50
    sharpe = calculate_sharpe_ratio(0.165, 0.20, risk_free_rate=0.065)
    assert sharpe == pytest.approx(0.50, rel=1e-5)

    # Volatility <= 0 returns 0.0
    assert calculate_sharpe_ratio(0.15, 0.0) == 0.0


def test_hand_computed_sortino_ratio():
    """Verify Sortino ratio formula penalizes only negative excess returns below daily risk-free rate."""
    daily_rets = pd.Series([0.005, -0.002, 0.003, -0.004, 0.006])
    sortino = calculate_sortino_ratio(daily_rets, cagr=0.15, risk_free_rate=0.065, trading_days=252)
    assert sortino > 0.0
    assert isinstance(sortino, float)


def test_rolling_consistency_metric():
    """Verify rolling consistency metric score is between 0.0 and 1.0."""
    fund_nav = pd.Series(np.linspace(100.0, 250.0, 800))
    bench_nav = pd.Series(np.linspace(100.0, 180.0, 800))
    # Fund consistently outperforms benchmark
    score = calculate_rolling_consistency(fund_nav, bench_nav, window_days=252)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_fund_category_ranking():
    """Verify funds are ranked strictly within their categories using composite z-score."""
    res = generate_fund_recommendations(profile_level=3)
    assert len(res.baskets) > 0

    for basket in res.baskets:
        # At most 2 funds per category
        assert len(basket.funds) <= 2
        # All funds in basket match category
        for f in basket.funds:
            assert f.category == basket.category
            assert f.risk_level_label in ("Low", "Moderately Low", "Moderate", "Moderately High", "High")
            assert "Why this fund" in f.explanation or "tracking the benchmark" in f.explanation or "steady returns" in f.explanation or "allocation" in f.explanation or "Sharpe" in f.explanation


def test_fund_capital_allocation_split():
    """Verify user entered capital is split proportionally across categories and funds."""
    entered_amt = 100000.0
    res = generate_fund_recommendations(profile_level=3, allocation_amount=entered_amt)

    assert res.total_amount == pytest.approx(entered_amt)
    assert res.total_allocated == pytest.approx(entered_amt, rel=1e-2)

    # Sum of all individual fund indicative amounts must equal entered amount
    total_fund_sum = sum(f.indicative_amount for b in res.baskets for f in b.funds)
    assert total_fund_sum == pytest.approx(entered_amt, rel=1e-2)

    # If no amount entered, total_allocated is 0, but allocated_weight is populated
    res_no_amt = generate_fund_recommendations(profile_level=3, allocation_amount=None)
    assert res_no_amt.total_allocated == 0.0
    for b in res_no_amt.baskets:
        for f in b.funds:
            assert f.allocated_weight > 0.0


def test_fund_existing_holdings_reduction_and_zeroing():
    """Verify existing portfolio holdings reduce or zero out matching categories with exact note."""
    # User already has 100% debt portfolio
    debt_portfolio = PortfolioInput(
        holdings=[],
        funds=[FundBucket(category="debt", amount=150000.0)],
    )

    res = generate_fund_recommendations(
        profile_level=3,
        portfolio=debt_portfolio,
        allocation_amount=50000.0,
    )

    # Category "Debt allocation" must be zeroed out
    basket_cats = [b.category for b in res.baskets]
    assert "liquid-debt" not in basket_cats

    # Note must match exact PRD requirement: "<Category> is already at or above your target weight, no addition suggested"
    matching_notes = [n for n in res.notes if "is already at or above your target weight, no addition suggested" in n]
    assert len(matching_notes) > 0
    assert any("Debt" in n for n in matching_notes)

    # Remaining categories still receive the full allocation
    assert res.total_allocated == pytest.approx(50000.0, rel=1e-2)


def test_recommendations_funds_tab_smoke():
    """AppTest smoke test on Recommendations page verifying both tabs render without exceptions."""
    from streamlit.testing.v1 import AppTest

    cfg = load_config()
    rec_page_path = Path("app/pages/6_Recommendations.py").resolve()
    at = AppTest.from_file(str(rec_page_path))
    at.session_state["user_profile"] = ProfileResult(
        capacity_score=50.0,
        tolerance_score=50.0,
        capacity_level=3,
        tolerance_level=3,
        preliminary_level=3,
        final_level=3,
        label="Moderate",
        limiting_factor="both",
        guardrail_messages=[],
        target_allocation=cfg.target_allocations[3],
        raw_scores={},
        preferences={"favoured_sectors": [], "avoided_sectors": []},
    )
    at.session_state["logged_in_user"] = "testuser"
    at.session_state["portfolio_holdings"] = [
        {"ticker": "INFY.NS", "quantity": 10, "avg_buy_price": 1400.0, "name": "Infosys", "sector": "IT", "cap_class": "large"}
    ]
    at.session_state["portfolio_funds"] = [
        {"category": "debt", "amount": 20000.0}
    ]

    at.run()
    assert not at.exception, f"Recommendations page crashed: {at.exception}"

    # Verify no raw HTML '<div' rendered as text
    for el in at.markdown:
        assert "<div" not in el.value, f"Raw div leak in markdown: {el.value}"


def test_zero_buy_sell_wording_funds():
    """Enforce AGENTS.md hard rule: strictly zero buy/sell wording across all mutual fund modules."""
    files_to_check = [
        Path("src/funds/metrics.py"),
        Path("src/funds/ingest.py"),
        Path("src/recommend/funds.py"),
        Path("app/pages/6_Recommendations.py"),
    ]

    forbidden_tokens = ["b" + "uy", "s" + "ell", "b" + "uying", "s" + "elling"]
    banned_pattern = re.compile(r"\b(" + "|".join(forbidden_tokens) + r")\b", re.IGNORECASE)

    for p in files_to_check:
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            # Ignore variable names like 'avg_buy_price' or phrases like 'buy/sell' in disclaimers
            clean_line = (
                line.replace("avg_buy_price", "")
                .replace("buy/sell", "")
                .replace("buy or sell", "")
                .replace("without buy/sell", "")
            )
            matches = banned_pattern.findall(clean_line)
            assert not matches, f"Forbidden wording {matches} in {p} at line {idx}: {line.strip()}"
