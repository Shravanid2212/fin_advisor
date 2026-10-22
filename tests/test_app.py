"""Automated tests for Phase 10 & Phase 11 Streamlit application.

Tests:
1. Formatting pure functions (Rupee with Lakh/Crore grouping, DD MMM YYYY dates).
2. Backend data service (read-only queries, health checks, empty filter handling).
3. Exact verification of TCS.NS numbers against both stored features and fresh calculation from clean prices.
4. Page smoke tests using streamlit.testing.v1.AppTest across all 8 pages.
5. Strict audit ensuring zero buy/sell terminology in UI text.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.theme import (
    format_date,
    format_dividend_yield,
    format_lakh_crore,
    format_pct,
    format_ratio,
    format_rupee,
)
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_explore_market_data,
    get_home_kpis_data,
    get_screener_data,
    get_stock_detail_data,
    get_stocks_comparison_data,
)
from src.features.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# 1. Formatting Pure Functions
# ---------------------------------------------------------------------------
def test_rupee_formatting_lakh_crore():
    """Verify Indian rupee formatting with lakh/crore grouping and thousands commas."""
    assert format_rupee(15_234_500) == "₹1.52 Cr"
    assert format_rupee(450_000) == "₹4.50 L"
    assert format_rupee(1_250.50) == "₹1,250.50"
    assert format_rupee(0) == "₹0.00"
    assert format_rupee(None) == "₹0.00"
    assert format_rupee(-500_000) == "-₹5.00 L"

    assert format_lakh_crore(25_000_000) == "2.50 Cr"
    assert format_lakh_crore(350_000) == "3.50 L"


def test_date_formatting():
    """Verify ISO dates are converted to DD MMM YYYY format."""
    assert format_date("2026-10-19") == "19 Oct 2026"
    assert format_date("2026-06-30") == "30 Jun 2026"
    assert format_date("2020-03-23") == "23 Mar 2020"
    assert format_date(None) == "N/A"
    assert format_date("") == "N/A"


def test_pct_and_ratio_formatting():
    """Verify percentage and ratio formatting."""
    assert format_pct(0.1234) == "12.34%"
    assert format_pct(0.1234, show_plus=True) == "+12.34%"
    assert format_pct(-0.052, show_plus=True) == "-5.20%"
    assert format_ratio(1.456) == "1.46"


def test_dividend_yield_formatting_and_no_displayed_yield_exceeds_20_percent():
    """Verify dividend yield formatting and guarantee that no displayed yield exceeds 20%."""
    # 1. Pure function formatting
    assert format_dividend_yield(0.47) == "0.47%"
    assert format_dividend_yield(0.34) == "0.34%"
    assert format_dividend_yield(1.95) == "1.95%"
    assert format_dividend_yield(6.14) == "6.14%"
    assert format_dividend_yield(1.46) == "1.46%"
    assert format_dividend_yield(0.0) == "0.00%"
    assert format_dividend_yield(None) == "0.00%"

    # 2. Audit all stocks in stock_features table in DuckDB
    db_path = "data/market.duckdb"
    con = duckdb.connect(db_path, read_only=True)
    df_feats = con.execute("SELECT ticker, name, dividend_yield FROM stock_features").fetchdf()
    con.close()

    assert not df_feats.empty
    for _, row in df_feats.iterrows():
        raw_val = row["dividend_yield"]
        displayed = format_dividend_yield(raw_val)
        disp_num = float(displayed.rstrip("%"))
        assert 0.0 <= disp_num <= 20.0, f"Displayed yield {displayed} for {row['ticker']} exceeds 20% limit!"

    # 3. Known key tickers exact display checks
    ticker_yields = dict(zip(df_feats["ticker"], df_feats["dividend_yield"]))
    assert format_dividend_yield(ticker_yields["BHARATFORG.NS"]) == "0.47%"
    assert format_dividend_yield(ticker_yields["TITAN.NS"]) == "0.34%"
    assert format_dividend_yield(ticker_yields["TCS.NS"]) == "1.95%"
    assert format_dividend_yield(ticker_yields["ITC.NS"]) == "6.14%"
    assert format_dividend_yield(ticker_yields["BAJAJ-AUTO.NS"]) == "1.46%"

    # 4. AppTest headless render check on Stock Detail for Bharat Forge
    page_path = Path(__file__).resolve().parent.parent / "app" / "pages" / "2_Stock_Detail.py"
    at = AppTest.from_file(str(page_path))
    at.session_state["logged_in"] = True
    at.session_state["username"] = "test_user"
    at.session_state["selected_ticker"] = "BHARATFORG.NS"
    at.run()

    assert len(at.exception) == 0
    # Collect all rendered text from all html, markdown, info elements
    rendered_strings = []
    for h in getattr(at, "html", []):
        rendered_strings.append(h.value)
    for m in at.markdown:
        rendered_strings.append(m.value)
    all_rendered = " ".join(rendered_strings)

    assert "47.00%" not in all_rendered, "Bug regression: Bharat Forge still displays 47.00% dividend yield!"
    assert "0.47%" in all_rendered, "Bharat Forge does not display corrected 0.47% dividend yield!"



# ---------------------------------------------------------------------------
# 2. Backend Service Layer Tests
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_check_db_health():
    """Verify database health checker."""
    health = check_db_health("data/market.duckdb")
    assert health["is_healthy"] is True
    assert health["data_as_of"] != "N/A"


@pytest.mark.realdata
def test_home_kpis_and_highlights():
    """Verify Home KPIs and highlights."""
    kpis = get_home_kpis_data("data/market.duckdb")
    assert kpis["stocks_count"] >= 75
    assert kpis["years_of_data"] >= 9.0
    assert kpis["sectors_count"] == 10
    assert 0.15 <= kpis["median_volatility"] <= 0.45
    assert kpis["most_volatile"] is not None
    assert kpis["lowest_drawdown"] is not None
    assert kpis["highest_sharpe"] is not None


@pytest.mark.realdata
def test_explore_market_empty_selection_graceful():
    """Verify graceful handling of empty filter selections in Explore Market."""
    res_empty_sec = get_explore_market_data(selected_sectors=[], selected_caps=["large"])
    assert res_empty_sec["scatter_df"].empty

    res_empty_cap = get_explore_market_data(selected_sectors=["IT"], selected_caps=[])
    assert res_empty_cap["scatter_df"].empty


# ---------------------------------------------------------------------------
# 3. Exact Verification of TCS.NS (Stored vs Fresh Computed from Clean Prices)
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_tcs_metrics_exact_match_against_clean_prices():
    """Verify TCS.NS CAGR, volatility, and Sharpe match stored values AND fresh calculation from clean prices."""
    db_path = "data/market.duckdb"
    con = duckdb.connect(db_path, read_only=True)

    # 1. Read stored values from stock_features
    stored = con.execute("SELECT * FROM stock_features WHERE ticker = 'TCS.NS'").fetchdf().iloc[0]
    stored_cagr = float(stored["cagr"])
    stored_vol = float(stored["annual_volatility"])
    stored_sharpe = float(stored["sharpe_ratio"])

    # 2. Fresh calculation directly from clean prices (is_filled = FALSE)
    prices_df = con.execute(
        """
        SELECT date, close, simple_return
        FROM prices
        WHERE ticker = 'TCS.NS' AND is_filled = FALSE
        ORDER BY date ASC
        """
    ).fetchdf()
    con.close()

    assert len(prices_df) > 2000

    start_price = float(prices_df["close"].iloc[0])
    end_price = float(prices_df["close"].iloc[-1])
    n_days = len(prices_df)
    elapsed_years = max(0.1, n_days / 252)

    computed_cagr = calculate_cagr(start_price, end_price, elapsed_years)

    daily_rets = prices_df["simple_return"].dropna()
    computed_vol = calculate_annualized_volatility(daily_rets, trading_days=252)
    computed_sharpe = calculate_sharpe_ratio(computed_cagr, computed_vol, risk_free_rate=0.065)

    # 3. Read via get_stock_detail_data
    detail = get_stock_detail_data("TCS.NS", period="Max", db_path=db_path)
    assert detail is not None
    service_stats = detail["stats"]

    # Verify stored vs computed match within tight tolerance
    assert math.isclose(service_stats["cagr"], stored_cagr, abs_tol=1e-4)
    assert math.isclose(service_stats["cagr"], computed_cagr, abs_tol=1e-4)

    assert math.isclose(service_stats["annual_volatility"], stored_vol, abs_tol=1e-4)
    assert math.isclose(service_stats["annual_volatility"], computed_vol, abs_tol=1e-4)

    assert math.isclose(service_stats["sharpe_ratio"], stored_sharpe, abs_tol=1e-3)
    assert math.isclose(service_stats["sharpe_ratio"], computed_sharpe, abs_tol=1e-3)

    # Check known values
    assert math.isclose(stored_cagr, 0.0968, abs_tol=1e-3)  # ~9.68%
    assert math.isclose(stored_vol, 0.2432, abs_tol=1e-3)   # ~24.32%
    assert math.isclose(stored_sharpe, 0.1308, abs_tol=1e-3) # ~0.13


# ---------------------------------------------------------------------------
# 4. Streamlit AppTest Headless Smoke Tests Across All Active Pages
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_all_pages_render_without_exception_and_no_raw_html():
    """Verify all active Streamlit pages run headless without raising exceptions,

    with a logged-in user, completed profile, and sample portfolio loaded,
    and assert no raw '<div' text in any rendered markdown or text element.
    """
    from src.app_data import get_sample_portfolio_holdings
    from src.profile.scoring import calculate_profile

    root_dir = Path(__file__).resolve().parent.parent
    pages = [
        "app/Home.py",
        "app/pages/1_Explore_Market.py",
        "app/pages/2_Stock_Detail.py",
        "app/pages/3_Compare_and_Screen.py",
        "app/pages/4_My_Profile.py",
        "app/pages/5_My_Portfolio.py",
        "app/pages/6_Recommendations.py",
    ]

    answers = {f"q{i}": 3 for i in range(1, 11)}
    sample_profile = calculate_profile(answers)
    stock_rows, fund_rows = get_sample_portfolio_holdings()

    for page_path in pages:
        p = (root_dir / page_path).resolve()
        assert p.exists(), f"Page {page_path} is missing!"
        at = AppTest.from_file(str(p))

        # Setup logged-in state, profile, and sample portfolio
        at.session_state["logged_in"] = True
        at.session_state["username"] = "smoke_test_user"
        at.session_state["profile_completed"] = True
        at.session_state["user_profile"] = sample_profile
        at.session_state["quiz_answers"] = answers
        at.session_state["portfolio_holdings"] = stock_rows
        at.session_state["portfolio_funds"] = fund_rows
        at.session_state["selected_ticker"] = "TCS.NS"

        at.run(timeout=30)
        assert len(at.exception) == 0, f"Exception in {page_path}: {[e.value for e in at.exception]}"

        # Assert no raw "<div" rendered as markdown code block or unescaped text
        for elem in at.markdown:
            assert "<div" not in elem.value, f"Raw '<div' found in markdown element on {page_path}: {elem.value}"
        for elem in at.caption:
            assert "<div" not in elem.value, f"Raw '<div' found in caption element on {page_path}: {elem.value}"
        for elem in at.info:
            assert "<div" not in elem.value, f"Raw '<div' found in info element on {page_path}: {elem.value}"
        for elem in at.warning:
            assert "<div" not in elem.value, f"Raw '<div' found in warning element on {page_path}: {elem.value}"
        for elem in at.error:
            assert "<div" not in elem.value, f"Raw '<div' found in error element on {page_path}: {elem.value}"


# ---------------------------------------------------------------------------
# 5. Strict Prohibition of Buy/Sell Advice Terminology in App Code
# ---------------------------------------------------------------------------
def test_no_buy_sell_advice_wording_in_app():
    """Ensure no buy/sell recommendation advice wording exists in app pages or UI text."""
    app_files = list(Path("app").rglob("*.py"))
    assert len(app_files) >= 7

    # Forbidden advice phrases
    forbidden_patterns = [
        re.compile(r"\b(buy|sell)\s+(this|the|now|recommendation|call)\b", re.IGNORECASE),
        re.compile(r"\b(strong\s+buy|strong\s+sell)\b", re.IGNORECASE),
        re.compile(r"\b(we\s+recommend\s+buying|we\s+recommend\s+selling)\b", re.IGNORECASE),
        re.compile(r"\b(target\s+price)\b", re.IGNORECASE),
    ]

    for f in app_files:
        text = f.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            matches = pat.findall(text)
            assert not matches, f"Found forbidden advice terminology {matches} in {f}"


# ---------------------------------------------------------------------------
# 6. Phase 12 Screener & Comparison Tests
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_screener_filtering_matches_underlying_tables():
    """Verify screener filtering results match underlying stock_features table exactly."""
    db_path = "data/market.duckdb"
    con = duckdb.connect(db_path, read_only=True)

    # Filter criteria
    sectors = ["IT", "Banking and Financials", "FMCG"]
    caps = ["large", "mid"]
    buckets = [1, 2]
    min_sharpe = 0.2
    min_div = 0.5

    # Direct DuckDB reference query
    query = f"""
        SELECT ticker, sharpe_ratio
        FROM stock_features
        WHERE ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')
          AND sector IN ('IT', 'Banking and Financials', 'FMCG')
          AND cap_class IN ('large', 'mid')
          AND risk_bucket IN (1, 2)
          AND sharpe_ratio >= {min_sharpe}
          AND COALESCE(dividend_yield, 0.0) >= {min_div}
        ORDER BY sharpe_ratio DESC
    """
    db_expected = con.execute(query).fetchdf()
    con.close()

    # Screener service function
    screener_res = get_screener_data(
        sectors=sectors,
        caps=caps,
        buckets=buckets,
        min_sharpe=min_sharpe,
        min_div_yield=min_div,
        db_path=db_path,
    )

    assert len(screener_res) == len(db_expected)
    if not db_expected.empty:
        assert list(screener_res["ticker"]) == list(db_expected["ticker"])


@pytest.mark.realdata
def test_stocks_comparison_analytics():
    """Verify head-to-head comparison generates valid rebased curves and correlation."""
    from src.app_data import get_stocks_comparison_data

    tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]
    comp = get_stocks_comparison_data(tickers, horizon="3Y")

    rebased_df = comp["rebased_df"]
    metrics_df = comp["metrics_df"]
    corr_df = comp["corr_df"]

    # Rebased prices start at 100
    assert not rebased_df.empty
    for t in tickers:
        assert t in rebased_df.columns
        assert math.isclose(rebased_df[t].dropna().iloc[0], 100.0, abs_tol=1e-2)

    # Metrics comparison
    assert len(metrics_df) == 3
    assert set(metrics_df["ticker"]) == set(tickers)

    # Correlation matrix
    assert corr_df.shape == (3, 3)
    for t in tickers:
        assert math.isclose(corr_df.loc[t, t], 1.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# 7. Phase 13 Questionnaire Refresh Stability Test
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_questionnaire_mid_quiz_refresh_no_crash():
    """Verify refreshing mid-quiz preserves answers in session_state and does not crash."""
    root_dir = Path(__file__).resolve().parent.parent
    page_path = (root_dir / "app/pages/4_My_Profile.py").resolve()

    at = AppTest.from_file(str(page_path))
    # Simulate user being on Step 2 with partial answers
    at.session_state["quiz_step"] = 2
    at.session_state["quiz_answers"] = {
        "q1": "25-35",
        "q2": "fixed salary",
        "q3": "none",
        "q4": "6-12 months",
        "q5": "10-25 lakh",
        "q6": "5-10 years",
    }
    at.run()

    # Must render without any exception
    assert len(at.exception) == 0
    # Step and answers must be preserved
    assert at.session_state["quiz_step"] == 2
    assert at.session_state["quiz_answers"]["q1"] == "25-35"


# ---------------------------------------------------------------------------
# 8. Phase 14 Portfolio & Messy CSV Handling Test
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_messy_csv_handling_with_clear_message():
    """Verify messy CSV input is handled gracefully with clear validation report messages."""
    import io
    from src.config import load_config, load_universe
    from src.portfolio.analytics import analyze_portfolio
    from src.portfolio.holdings import parse_holdings_csv

    cfg = load_config()
    univ = load_universe()

    messy_csv_text = """ticker,quantity,avg_buy_price
TCS.NS,10,2400.00
TCS.NS,15,2500.00
UNKNOWN_CORP.NS,5,150.00
BAD_ROW_QTY,-20,300.00
RELIANCE.NS,20,1200.00
"""

    port = parse_holdings_csv(io.StringIO(messy_csv_text), universe=univ)
    report = port.validation_report

    # Check clear reporting
    assert report.is_valid is False  # Negative quantity dropped
    assert len(report.bad_quantity_rows) == 1
    assert any("BAD_ROW_QTY" in iss for iss in report.issues)
    assert any("Dropped BAD_ROW_QTY due to invalid quantity" in iss for iss in report.issues)

    # Duplicates merged
    assert len(report.duplicates_merged) == 1
    assert any("Merged 2 duplicate entries for TCS.NS" in iss for iss in report.issues)

    # Unknown tickers flagged
    assert "UNKNOWN_CORP.NS" in report.unknown_tickers
    assert any("UNKNOWN_CORP.NS is outside the tracked universe" in iss for iss in report.issues)

    # Downstream analysis still succeeds without crashing
    analytics = analyze_portfolio(port, config=cfg, db_path="data/market.duckdb")
    assert analytics.total_value > 0.0
    assert analytics.volatility > 0.0


@pytest.mark.realdata
def test_sample_portfolio_load_and_analytics():
    """Verify sample portfolio generates valid KPIs, asset allocation, and concentration."""
    from src.app_data import get_sample_portfolio_holdings
    from src.config import load_config, load_universe
    from src.portfolio.analytics import analyze_portfolio
    from src.portfolio.holdings import parse_manual_holdings

    cfg = load_config()
    univ = load_universe()

    stock_rows, fund_rows = get_sample_portfolio_holdings()
    port = parse_manual_holdings(stock_rows, fund_rows=fund_rows, universe=univ)

    target_moderate = cfg.target_allocations[3]
    analytics = analyze_portfolio(port, target_allocation=target_moderate, config=cfg, db_path="data/market.duckdb")

    assert analytics.total_value > 100_000.0
    assert analytics.debt_value == 80_000.0
    assert analytics.gold_value == 20_000.0
    assert 0.10 <= analytics.volatility <= 0.40
    assert analytics.diversification_score > 0.0


# ---------------------------------------------------------------------------
# 9. Phase 16 Methodology, Statistics & Ethics Tests
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_methodology_data_service():
    """Verify get_methodology_data returns complete, uncorrupted live database numbers."""
    from src.app_data import get_methodology_data

    meth = get_methodology_data("data/market.duckdb")
    assert bool(meth) is True

    # 1. Dataset summary
    summ = meth["dataset_summary"]
    assert summ["total_rows"] >= 200_000
    assert summ["equity_count"] == 80
    assert summ["extreme_count"] >= 1_000
    assert summ["filled_count"] >= 500
    assert summ["min_date"] == "2016-01-01"
    assert summ["max_date"] >= "2026-10-15"

    # 2. Stats results table
    stats_df = meth["stats_results"]
    assert len(stats_df) == 7
    expected_tests = {"Levene Test (Variance Equality)", "Kruskal-Wallis ANOVA", "Mann-Whitney U Test", "Jarque-Bera Normality Test"}
    assert set(stats_df["test"].unique()) == expected_tests

    # 3. Cap metrics table (AMFI 2026-06-30)
    cap_df = meth["cap_metrics"]
    assert len(cap_df) == 3
    assert set(cap_df["cap_class"].tolist()) == {"large", "mid", "small"}
    assert (cap_df["amfi_source_date"] == "2026-06-30").all()
    # Group sizes: Large 38, Mid 23, Small 19
    sizes = dict(zip(cap_df["cap_class"], cap_df["group_size"]))
    assert sizes == {"large": 38, "mid": 23, "small": 19}

    # 4. Sector metrics table (10 sectors)
    sec_df = meth["sector_metrics"]
    assert len(sec_df) == 10
    assert sec_df["group_size"].sum() == 80

    # 5. Outlier counterfactual calculation
    counter = meth["counterfactual"]
    assert counter["total_clean_days"] >= 200_000
    assert counter["iqr_removed_days"] >= 9_000
    assert counter["vol_understatement_pct"] >= 20.0


@pytest.mark.realdata
def test_all_nine_crisp_dm_notebooks_exist_and_valid():
    """Verify that all 9 CRISP-DM lifecycle notebooks exist, are non-empty, and contain cells."""
    import json

    expected_notebooks = [
        "01_problem_formulation.ipynb",
        "02_data_collection.ipynb",
        "03_cleaning.ipynb",
        "04_eda.ipynb",
        "05_features_and_stats.ipynb",
        "06_profile_and_portfolio.ipynb",
        "07_recommender.ipynb",
        "08_decision_support.ipynb",
        "09_ethics.ipynb",
    ]

    nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
    for nb_name in expected_notebooks:
        nb_path = nb_dir / nb_name
        assert nb_path.exists(), f"Missing required notebook: {nb_name}"
        assert nb_path.stat().st_size > 1000, f"Notebook {nb_name} appears empty or truncated"

        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data and len(data["cells"]) >= 2, f"Notebook {nb_name} must contain at least 2 cells"


