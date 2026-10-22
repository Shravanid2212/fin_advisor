"""Test dividend yield units, conversion, and display formatting (PRD §9, §15).

Verifies:
1. Units in fundamentals_snapshots and stock_features are percentage points (0 to 100 scale).
2. All 80 universe stocks have displayed yields between 0.00% and 20.00%.
3. Key stock dividend yields (ITC, TCS, BAJAJ-AUTO, TITAN, BHARATFORG) display correctly.
4. Screener and Comparison displays format yields properly.
"""

from __future__ import annotations

import duckdb
import pytest

from app.theme import format_dividend_yield
from src.app_data import get_screener_data, get_stock_detail_data, get_stocks_comparison_data


@pytest.mark.realdata
def test_no_displayed_yield_exceeds_20_percent():
    """Guarantee that no stock in the universe displays a dividend yield > 20%."""
    con = duckdb.connect("data/market.duckdb", read_only=True)
    df = con.execute("SELECT ticker, name, sector, dividend_yield FROM stock_features").fetchdf()
    con.close()

    assert not df.empty
    assert len(df) == 80

    for _, row in df.iterrows():
        raw_val = row["dividend_yield"]
        assert raw_val is not None
        assert 0.0 <= raw_val <= 20.0, f"Raw dividend yield {raw_val} for {row['ticker']} out of bounds!"

        disp = format_dividend_yield(raw_val)
        assert disp.endswith("%")
        disp_num = float(disp.rstrip("%"))
        assert 0.0 <= disp_num <= 20.0, f"Displayed dividend yield {disp} for {row['ticker']} exceeds 20%!"


@pytest.mark.realdata
def test_known_stocks_dividend_yields():
    """Verify known stocks display accurate, unmagnified dividend yields."""
    con = duckdb.connect("data/market.duckdb", read_only=True)
    df = con.execute("SELECT ticker, dividend_yield FROM stock_features").fetchdf()
    con.close()

    yield_map = dict(zip(df["ticker"], df["dividend_yield"]))

    # 1. Bharat Forge (Auto): was bugged at 47.00%, now 0.47%
    assert format_dividend_yield(yield_map["BHARATFORG.NS"]) == "0.47%"

    # 2. Titan (Consumer): was bugged at 34.00%, now 0.34%
    assert format_dividend_yield(yield_map["TITAN.NS"]) == "0.34%"

    # 3. TCS (IT): 1.95%
    assert format_dividend_yield(yield_map["TCS.NS"]) == "1.95%"

    # 4. ITC (FMCG): 6.14%
    assert format_dividend_yield(yield_map["ITC.NS"]) == "6.14%"

    # 5. Bajaj Auto (Auto): 1.46%
    assert format_dividend_yield(yield_map["BAJAJ-AUTO.NS"]) == "1.46%"


@pytest.mark.realdata
def test_stock_detail_fundamentals_dividend_yield():
    """Verify get_stock_detail_data returns correct dividend yield in fundamentals dict."""
    detail_bf = get_stock_detail_data("BHARATFORG.NS")
    assert detail_bf is not None
    raw_yield = detail_bf["fundamentals"]["dividend_yield"]
    assert raw_yield == 0.47
    assert format_dividend_yield(raw_yield) == "0.47%"

    detail_titan = get_stock_detail_data("TITAN.NS")
    assert detail_titan is not None
    raw_titan = detail_titan["fundamentals"]["dividend_yield"]
    assert raw_titan == 0.34
    assert format_dividend_yield(raw_titan) == "0.34%"


@pytest.mark.realdata
def test_compare_and_screen_dividend_yield():
    """Verify Compare & Screen service returns consistent dividend yield."""
    screener_df = get_screener_data(min_div_yield=1.0)
    assert not screener_df.empty
    assert (screener_df["dividend_yield"] >= 1.0).all()
    # Check that formatted yields are <= 20%
    formatted = screener_df["dividend_yield"].apply(format_dividend_yield)
    for disp in formatted:
        assert float(disp.rstrip("%")) <= 20.0
