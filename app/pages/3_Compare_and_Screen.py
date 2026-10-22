"""Compare & Screen page for Indian equities.

Adheres to:
- Multi-factor screening and stock comparison.
- Rebased price chart (base = 100) and side-by-side metrics table (no correlation heat map).
- Pure investor metrics with plain-language labels.
- Read-only queries via src/app_data.py.
- Requires user authentication.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Any

from app.theme import (
    apply_theme_to_plotly,
    format_dividend_yield,
    format_pct,
    format_ratio,
)
from app.components.ui import (
    render_disclaimer,
    render_empty_state,
    render_html,
    render_page_header,
    require_login,
)
from src.app_data import (
    check_db_health,
    get_screener_data,
    get_stocks_comparison_data,
)

st.set_page_config(
    page_title="Compare & Screen | Financial Analytics",
    page_icon="⚖️",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Health check
health = check_db_health()
if not health["is_healthy"]:
    st.error(f"⚠️ {health['message']}")
    render_disclaimer()
    st.stop()

data_date = health["data_as_of"]
render_page_header("Compare & Screen", data_date)

SECTORS_LIST = [
    "Banking and Financials",
    "IT",
    "Pharma and Healthcare",
    "FMCG",
    "Auto",
    "Energy and Power",
    "Metals",
    "Capital Goods and Infra",
    "Consumer",
    "Chemicals",
]

CAPS_LIST = ["large", "mid", "small"]
BUCKETS_LIST = [1, 2, 3, 4, 5]

# 3. Screener Filters
st.subheader("Multi-Factor Universe Screener")
st.caption("Filter analyzed stocks by market cap tier, sector, risk volatility bucket, Sharpe ratio, and dividend yield.")

with st.expander("Filter Controls", expanded=True):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_caps = st.multiselect(
            "Market Cap Tier",
            options=CAPS_LIST,
            default=CAPS_LIST,
            format_func=lambda c: f"{c.capitalize()} Cap",
            help="Market capitalization tier classification.",
        )
        sel_sectors = st.multiselect(
            "Economic Sectors",
            options=SECTORS_LIST,
            default=SECTORS_LIST,
            help="Economic sectors.",
        )
    with f_col2:
        sel_buckets = st.multiselect(
            "Volatility Risk Bucket",
            options=BUCKETS_LIST,
            default=BUCKETS_LIST,
            format_func=lambda b: f"Level {b}",
            help="Assigned risk bucket (1: lowest volatility, 5: highest volatility).",
        )
        min_sharpe = st.slider(
            "Minimum Sharpe Ratio",
            min_value=-0.5,
            max_value=1.5,
            value=0.0,
            step=0.05,
            help="Excess return over risk-free rate per unit of annualized volatility.",
        )
    with f_col3:
        min_div = st.slider(
            "Minimum Dividend Yield (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Snapshot dividend yield percentage.",
        )

# Query Screener Data
screener_df = get_screener_data(
    sectors=sel_sectors,
    caps=sel_caps,
    buckets=sel_buckets,
    min_sharpe=min_sharpe,
    min_div_yield=min_div,
)

# Screener Results Table
if not screener_df.empty:
    st.markdown(f"**Screened Stocks:** `{len(screener_df)} matched`")

    display_df = screener_df[[
        "ticker", "name", "sector", "cap_class", "risk_bucket",
        "cagr", "annual_volatility", "sharpe_ratio", "max_drawdown", "dividend_yield"
    ]].copy()

    display_df["cap_class"] = display_df["cap_class"].str.capitalize()
    display_df["cagr"] = display_df["cagr"].apply(format_pct)
    display_df["annual_volatility"] = display_df["annual_volatility"].apply(format_pct)
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(format_ratio)
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(format_pct)
    display_df["dividend_yield"] = display_df["dividend_yield"].apply(format_dividend_yield)

    display_df.columns = [
        "Ticker", "Company Name", "Sector", "Cap Tier", "Risk Level",
        "CAGR", "Annual Volatility", "Sharpe", "Max Drawdown", "Div Yield"
    ]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = screener_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Table (CSV)",
        data=csv_data,
        file_name="screened_stocks.csv",
        mime="text/csv",
        help="Download filtered screener output as CSV.",
    )
else:
    render_empty_state("No Matches Found", "Try adjusting the filter criteria above to broaden results.")

render_html("<div style='margin-bottom: 32px;'></div>")
st.markdown("---")

# 4. Side-by-Side Stock Comparison
st.subheader("Compare Stocks Side-by-Side")
st.caption("Select 2 to 4 stocks to evaluate historical rebased growth and comparative risk metrics.")

all_tickers_list = screener_df["ticker"].tolist() if not screener_df.empty else ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
initial_compare = [t for t in ["TCS.NS", "INFY.NS"] if t in all_tickers_list]
if len(initial_compare) < 2 and len(all_tickers_list) >= 2:
    initial_compare = all_tickers_list[:2]

c_cmp_sel, c_cmp_hor = st.columns([3, 1])
with c_cmp_sel:
    selected_compare = st.multiselect(
        "Choose Stocks to Compare (2 - 4)",
        options=all_tickers_list,
        default=initial_compare,
        max_selections=4,
        help="Select between 2 and 4 stocks for comparative analysis.",
    )

with c_cmp_hor:
    comp_horizon = st.selectbox(
        "Comparison Horizon",
        options=["1Y", "3Y", "5Y", "10Y"],
        index=1,
    )

if len(selected_compare) < 2:
    st.info("ℹ️ Select at least 2 stocks above to display comparative charts and metrics.")
    render_disclaimer()
    st.stop()

# Load comparison data
comp_data = get_stocks_comparison_data(selected_compare, horizon=comp_horizon)
rebased_comp = comp_data["rebased_df"]
metrics_comp = comp_data["metrics_df"]

# Rebased Performance Chart
st.markdown(f"#### Rebased Price Trajectory ({comp_horizon})")
st.caption("All stocks rebased to 100 at the start of the horizon to eliminate nominal price level distortions.")

if not rebased_comp.empty:
    fig_comp = px.line(
        rebased_comp,
        labels={"index": "Date", "value": "Rebased Level (Base = 100)", "variable": "Ticker"},
    )
    fig_comp.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_theme_to_plotly(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    render_empty_state("No Price Data", "Could not generate rebased price curves for the selected tickers.")

render_html("<div style='margin-bottom: 24px;'></div>")

# Key Metrics Comparison Table (Full Width)
st.markdown("#### Key Metrics Comparison")
if not metrics_comp.empty:
    m_display = metrics_comp.set_index("ticker").copy()

    metrics_rows = [
        ("Company Name", lambda t: m_display.loc[t, "name"]),
        ("Sector", lambda t: m_display.loc[t, "sector"]),
        ("Cap Tier", lambda t: str(m_display.loc[t, "cap_class"]).capitalize()),
        ("Risk Level", lambda t: f"Level {int(m_display.loc[t, 'risk_bucket'])}"),
        ("CAGR", lambda t: format_pct(m_display.loc[t, "cagr"])),
        ("Annual Volatility", lambda t: format_pct(m_display.loc[t, "annual_volatility"])),
        ("Sharpe Ratio", lambda t: format_ratio(m_display.loc[t, "sharpe_ratio"])),
        ("Sortino Ratio", lambda t: format_ratio(m_display.loc[t, "sortino_ratio"])),
        ("Max Drawdown", lambda t: format_pct(m_display.loc[t, "max_drawdown"])),
        ("Beta (vs Nifty 50)", lambda t: format_ratio(m_display.loc[t, "beta"])),
        ("Daily VaR 95%", lambda t: format_pct(m_display.loc[t, "var_95"])),
        ("Trailing P/E", lambda t: format_ratio(m_display.loc[t, "trailing_pe"])),
        ("Dividend Yield", lambda t: format_dividend_yield(m_display.loc[t, "dividend_yield"]) if t in m_display.index else "0.00%"),
    ]

    table_dict: dict[str, list[Any]] = {"Metric": [r[0] for r in metrics_rows]}
    for t in selected_compare:
        if t in m_display.index:
            table_dict[t] = [r[1](t) for r in metrics_rows]

    cmp_table_df = pd.DataFrame(table_dict)
    st.dataframe(cmp_table_df, use_container_width=True, hide_index=True)

render_disclaimer()
