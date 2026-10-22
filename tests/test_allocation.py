"""Unit tests for the recommendation allocation engine (PRD §13, §15.7).

Verifies:
1. Quantities are non-negative whole numbers (integers).
2. Total actual allocated amount never exceeds the user-entered amount.
3. Leftover is shown and correct (leftover == total_amount - total_actual_amount >= 0).
4. A stock above its budget share is handled (quantity 0, shows minimum amount needed for 1 share, leftover redistributed).
5. The per-stock cap (default 20%) still holds across all picks.
6. Zero or empty amount is handled gracefully with clear messages and 0 quantities.
7. End-to-end execution on the bank-heavy fixture with Rs 1,00,000 at Level 3.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import pytest
import pandas as pd

from src.config import load_config, load_universe
from src.portfolio.analytics import analyze_portfolio
from src.portfolio.holdings import PortfolioInput, parse_manual_holdings
from src.recommend.recommender import (
    AllocationPlan,
    AllocationItem,
    compute_recommendation_allocations,
    generate_recommendations,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def universe():
    return load_universe()


@pytest.fixture
def bank_heavy_portfolio(universe):
    """Bank-heavy fixture portfolio: 100 HDFCBANK, 50 ICICIBANK, 40 KOTAKBANK + debt + gold."""
    holdings = [
        {"ticker": "HDFCBANK.NS", "quantity": 100, "avg_buy_price": 700.0},
        {"ticker": "ICICIBANK.NS", "quantity": 50, "avg_buy_price": 1350.0},
        {"ticker": "KOTAKBANK.NS", "quantity": 40, "avg_buy_price": 1700.0},
    ]
    funds = [
        {"category": "debt", "amount": 50000.0},
        {"category": "gold", "amount": 10000.0},
    ]
    return parse_manual_holdings(holdings, fund_rows=funds, universe=universe)


@pytest.mark.realdata
def test_bank_heavy_fixture_allocation_rs_1_lakh(config, universe, bank_heavy_portfolio):
    """Verify bank-heavy fixture with Rs 1,00,000 at Level 3."""
    res = generate_recommendations(
        portfolio=bank_heavy_portfolio,
        profile_level=3,
        config=config,
        allocation_amount=100000.0,
    )
    plan = res.allocation_plan
    assert plan is not None
    assert isinstance(plan, AllocationPlan)

    # 1. Total amount entered
    assert plan.total_amount == 100000.0

    # 2. Total actual amount never exceeds entered amount
    assert plan.total_actual_amount <= 100000.0
    assert plan.total_actual_amount > 90000.0  # High budget utilization

    # 3. Leftover is non-negative and exactly equals difference
    assert math.isclose(plan.unallocated_leftover, 100000.0 - plan.total_actual_amount, abs_tol=1e-2)
    assert plan.unallocated_leftover >= 0.0

    # 4. Check all items
    assert len(plan.items) == len(res.recommendations)
    print("\n--- BANK-HEAVY FIXTURE ALLOCATION TABLE (Rs 1,00,000 @ Level 3) ---")
    header = f"{'Name (Ticker)':<35} | {'Last Close (Date)':<24} | {'Suggested Qty':<14} | {'Indicative Amt':<15} | {'Basket Wt':<10} | {'Post-Add Port Wt':<16}"
    print(header)
    print("-" * len(header))
    for it in plan.items:
        close_str = f"Rs {it.last_close:,.2f} ({it.price_date})"
        print(f"{it.name + ' (' + it.ticker + ')':<35} | {close_str:<24} | {it.suggested_quantity:<14} | Rs {it.indicative_amount:<12,.2f} | {it.basket_weight*100:>8.2f}% | {it.weight_after_addition*100:>14.2f}%")
    print(f"\nTotal Actual Allocated: Rs {plan.total_actual_amount:,.2f}")
    print(f"Unallocated Leftover  : Rs {plan.unallocated_leftover:,.2f}")
    print(f"New Portfolio Value   : Rs {plan.new_portfolio_value:,.2f}")
    print(f"Current Equity Share  : {plan.current_equity_share*100:.2f}% -> New Equity Share: {plan.new_equity_share*100:.2f}%")

    for item in plan.items:
        # Quantities are whole integers >= 0
        assert isinstance(item.suggested_quantity, int)
        assert item.suggested_quantity >= 0

        # Actual amount = quantity * price
        assert math.isclose(item.actual_amount, item.suggested_quantity * item.last_close, abs_tol=1e-2)

        # Per-stock cap check: each stock's actual allocated share <= 20% cap + epsilon
        assert (item.actual_amount / plan.total_amount) <= (config.recommender.per_stock_weight_cap + 1e-3)

        # Minimum amount for one share is positive
        assert item.min_amount_for_one_share > 0.0
        assert math.isclose(item.min_amount_for_one_share, item.last_close, abs_tol=1e-2)

        # Post-addition portfolio weight is positive and realistic
        assert 0.0 <= item.weight_after_addition <= 1.0


@pytest.mark.realdata
def test_stock_above_budget_share_handled(config, universe, bank_heavy_portfolio):
    """When allocation is too small for expensive stocks (e.g. Bajaj Auto > 10,000), quantity is 0 and leftover redistributed."""
    # With Rs 25,000 total, a stock priced at ~10,287 (like BAJAJ-AUTO) will have indicative budget ~3,145 < price
    res = generate_recommendations(
        portfolio=bank_heavy_portfolio,
        profile_level=3,
        config=config,
        allocation_amount=25000.0,
    )
    plan = res.allocation_plan
    assert plan is not None

    # Total never exceeds entered amount
    assert plan.total_actual_amount <= 25000.0
    assert math.isclose(plan.unallocated_leftover, 25000.0 - plan.total_actual_amount, abs_tol=1e-2)

    # Check for zero quantity handling
    zero_qty_items = [it for it in plan.items if it.suggested_quantity == 0]
    assert len(zero_qty_items) >= 1  # At least one expensive stock cannot buy 1 share

    for it in zero_qty_items:
        assert it.actual_amount == 0.0
        assert it.min_amount_for_one_share == it.last_close
        assert "Minimum amount needed for 1 share is" in it.note
        assert f"₹{it.last_close:,.2f}" in it.note

    # Non-zero items received redistributed capital without violating 20% cap
    cap = config.recommender.per_stock_weight_cap
    for it in plan.items:
        assert (it.actual_amount / 25000.0) <= (cap + 1e-3)


@pytest.mark.realdata
def test_zero_and_negative_amount_handling(config, universe, bank_heavy_portfolio):
    """Zero or negative amount returns empty/zero quantities safely."""
    res_zero = generate_recommendations(
        portfolio=bank_heavy_portfolio,
        profile_level=3,
        config=config,
        allocation_amount=0.0,
    )
    # Allocation plan with 0 amount
    plan_zero = compute_recommendation_allocations(
        recommendations=res_zero.recommendations,
        total_amount=0.0,
        portfolio=bank_heavy_portfolio,
    )
    assert plan_zero.total_amount == 0.0
    assert plan_zero.total_actual_amount == 0.0
    assert plan_zero.unallocated_leftover == 0.0
    assert len(plan_zero.items) == 0

    plan_neg = compute_recommendation_allocations(
        recommendations=res_zero.recommendations,
        total_amount=-5000.0,
        portfolio=bank_heavy_portfolio,
    )
    assert plan_neg.total_actual_amount == 0.0
    assert len(plan_neg.items) == 0


@pytest.mark.realdata
def test_what_if_updates_with_rounded_quantities(config, universe, bank_heavy_portfolio):
    """What-if panel reflects exact rounded share additions."""
    # Run with Rs 1,00,000 allocation
    res_alloc = generate_recommendations(
        portfolio=bank_heavy_portfolio,
        profile_level=3,
        config=config,
        allocation_amount=100000.0,
    )
    wi_alloc = res_alloc.what_if
    plan = res_alloc.allocation_plan
    assert plan is not None

    # Before volatility must match current portfolio
    base_analytics = analyze_portfolio(bank_heavy_portfolio, config=config)
    assert math.isclose(wi_alloc.before_volatility, base_analytics.volatility, abs_tol=1e-3)

    # After volatility must be computed and valid
    assert wi_alloc.after_volatility > 0.0
    assert 0.05 <= wi_alloc.after_volatility <= 0.40

    # Sector weights must shift toward added sectors (e.g. FMCG, Consumer, Pharma)
    assert "FMCG" in wi_alloc.after_sector_weights
    assert wi_alloc.after_sector_weights["FMCG"] > 0.0


@pytest.mark.realdata
def test_recommendations_page_interactive_allocation_apptest():
    """Verify Streamlit Recommendations page with allocation amounts via AppTest."""
    from streamlit.testing.v1 import AppTest
    from src.app_data import get_sample_portfolio_holdings
    from src.profile.scoring import calculate_profile

    root_dir = Path(__file__).resolve().parent.parent
    page_path = root_dir / "app" / "pages" / "6_Recommendations.py"

    answers = {f"q{i}": 3 for i in range(1, 11)}
    sample_profile = calculate_profile(answers)
    stock_rows, fund_rows = get_sample_portfolio_holdings()

    at = AppTest.from_file(str(page_path))
    at.session_state["logged_in"] = True
    at.session_state["username"] = "alloc_test_user"
    at.session_state["profile_completed"] = True
    at.session_state["user_profile"] = sample_profile
    at.session_state["quiz_answers"] = answers
    at.session_state["portfolio_holdings"] = stock_rows
    at.session_state["portfolio_funds"] = fund_rows

    at.run(timeout=30)
    assert len(at.exception) == 0, f"Exceptions found: {[e.value for e in at.exception]}"

    # Verify download button rendered
    download_buttons = [b for b in at.download_button if "Suggested Allocations" in b.label]
    assert len(download_buttons) == 1
    assert "buy" not in download_buttons[0].label.lower()
    assert "sell" not in download_buttons[0].label.lower()

    # Verify no buy/sell advice in any rendered markdown
    for m in at.markdown:
        text = m.value.lower()
        assert not re.search(r"\b(buy|sell)\b", text), f"Found prohibited wording in markdown: {m.value}"

