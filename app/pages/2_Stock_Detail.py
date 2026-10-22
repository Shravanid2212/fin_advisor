"""Stock Detail deep dive page.

Adheres strictly to:
- Educational decision-support tool; no buy/sell terminology.
- Read-only against database via src/app_data.py. No calculations in this page.
- Label: 'Fundamentals are a today-only snapshot'.
- Price chart with log scale toggle and crisis annotations.
- SEBI-style 5-level riskometer gauge.
- Key stats, return distribution histogram (without normal overlay), underwater drawdown curve.
- Peers table with pairwise correlation.
- Requires user authentication.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from app.theme import (
    GREY_BAR_COLOR,
    NEGATIVE_RED,
    PRIMARY_TEAL,
    RED_BG_OVERLAY,
    apply_theme_to_plotly,
    format_date,
    format_dividend_yield,
    format_lakh_crore,
    format_pct,
    format_ratio,
    format_rupee,
)
from app.components.ui import (
    create_riskometer_fig,
    render_disclaimer,
    render_empty_state,
    render_html,
    render_page_header,
    render_risk_badge,
    require_login,
)
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_stock_detail_data,
)

st.set_page_config(
    page_title="Stock Detail | Financial Analytics",
    page_icon="🔍",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Health Check
health = check_db_health()
if not health["is_healthy"]:
    st.error(f"⚠️ {health['message']}")
    render_disclaimer()
    st.stop()

data_date = health["data_as_of"]
render_page_header("Stock Detail", data_date)

# 3. Ticker Selection
tickers = get_all_universe_tickers()
if not tickers:
    render_empty_state("No Stocks Found", "No universe stocks are available in the database.")
    render_disclaimer()
    st.stop()

saved_ticker = st.session_state.get("selected_ticker", "TCS.NS")
ticker_list = [t["ticker"] for t in tickers]
default_idx = ticker_list.index(saved_ticker) if saved_ticker in ticker_list else 0

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_ticker = st.selectbox(
        "Select Stock",
        options=ticker_list,
        index=default_idx,
        format_func=lambda t: f"{t} - {next((x['name'] for x in tickers if x['ticker'] == t), '')} ({next((x['sector'] for x in tickers if x['ticker'] == t), '')})",
        help="Choose a stock to view its comprehensive risk-return analysis.",
    )
    st.session_state["selected_ticker"] = selected_ticker

with col_sel2:
    period = st.radio(
        "Lookback Period",
        options=["1M", "6M", "1Y", "5Y", "10Y", "Max"],
        index=2,
        horizontal=True,
    )

# 4. Load Stock Data
detail = get_stock_detail_data(selected_ticker, period=period)
if not detail:
    render_empty_state("Data Error", f"Could not load data for {selected_ticker}.")
    render_disclaimer()
    st.stop()

# 5. Stock Header Bar
st.markdown("---")
c_hdr1, c_hdr2, c_hdr3 = st.columns([3, 2, 2])

with c_hdr1:
    render_html(
        f"""
        <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
            <h2 style="margin: 0; font-weight: 700;">{detail['name']}</h2>
            <code style="font-size: 1.1rem; padding: 2px 8px; border-radius: 4px;">{detail['ticker']}</code>
        </div>
        <div style="margin-top: 8px;">
            <span style="font-size: 0.9rem; opacity: 0.8; margin-right: 12px;"><b>Sector:</b> {detail['sector']}</span>
            <span style="font-size: 0.9rem; opacity: 0.8; margin-right: 12px;"><b>Tier:</b> {detail['cap_class'].capitalize()} Cap</span>
            {render_risk_badge(detail['risk_bucket'])}
        </div>
        """
    )

with c_hdr2:
    ret_color = PRIMARY_TEAL if detail["one_day_ret"] >= 0 else NEGATIVE_RED
    render_html(
        f"""
        <div style="text-align: right;">
            <div style="font-size: 1.9rem; font-weight: 700;">{format_rupee(detail['latest_close'])}</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: {ret_color};">
                {format_pct(detail['one_day_ret'], show_plus=True)} (1D)
            </div>
            <div style="font-size: 0.75rem; opacity: 0.6;">Latest Clean Close: {format_date(detail['data_date'])}</div>
        </div>
        """
    )

with c_hdr3:
    render_html("<div style='height: 10px;'></div>")
    if st.button("➕ Add to portfolio", type="secondary", use_container_width=True):
        if "portfolio_holdings" not in st.session_state:
            st.session_state["portfolio_holdings"] = []
        existing = [h for h in st.session_state["portfolio_holdings"] if h.get("ticker") == detail["ticker"]]
        if not existing:
            st.session_state["portfolio_holdings"].append({
                "ticker": detail["ticker"],
                "quantity": 10,
                "avg_buy_price": detail["latest_close"],
                "name": detail["name"],
                "sector": detail["sector"],
                "cap_class": detail["cap_class"],
            })
            st.success(f"Added {detail['ticker']} (10 units @ {format_rupee(detail['latest_close'])}) to your portfolio!")
        else:
            st.info(f"{detail['ticker']} is already in your portfolio.")

render_html("<div style='margin-bottom: 24px;'></div>")

# 6. Price & Volume Charts + Riskometer Gauge
col_chart, col_gauge = st.columns([3, 1.2])

with col_chart:
    st.subheader(f"Price Chart ({period})")
    log_scale = st.toggle("Log scale", value=False, help="Toggle logarithmic vertical axis for proportional growth evaluation.")

    prices_df = detail["prices_df"]
    if not prices_df.empty:
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=prices_df["date"],
                y=prices_df["close"],
                mode="lines",
                name="Close Price",
                line=dict(color=PRIMARY_TEAL, width=2.2),
                hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>",
            )
        )

        min_date_view = prices_df["date"].min()
        max_date_view = prices_df["date"].max()

        covid_dt = pd.to_datetime("2020-03-23")
        if min_date_view <= covid_dt <= max_date_view:
            covid_row = prices_df[prices_df["date"] >= covid_dt].head(1)
            if not covid_row.empty:
                y_val = covid_row["close"].values[0]
                fig_price.add_annotation(
                    x=covid_dt,
                    y=y_val,
                    text="COVID-19 Shock (Mar 2020)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=NEGATIVE_RED,
                    arrowsize=1,
                    arrowwidth=1.5,
                    ax=0,
                    ay=-45,
                    bgcolor=RED_BG_OVERLAY,
                    bordercolor=NEGATIVE_RED,
                    font=dict(size=10),
                )

        fig_price.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=10),
            hovermode="x unified",
            yaxis=dict(type="log" if log_scale else "linear", tickprefix="₹"),
            xaxis=dict(title=""),
        )
        apply_theme_to_plotly(fig_price)
        st.plotly_chart(fig_price, use_container_width=True)

        fig_vol = go.Figure(
            go.Bar(
                x=prices_df["date"],
                y=prices_df["volume"],
                name="Volume",
                marker_color=GREY_BAR_COLOR,
                hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Volume:</b> %{y:,.0f}<extra></extra>",
            )
        )
        fig_vol.update_layout(
            height=120,
            margin=dict(l=20, r=20, t=5, b=20),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Volume", showgrid=False),
        )
        apply_theme_to_plotly(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True)

with col_gauge:
    st.subheader("Risk Level")
    fig_gauge = create_riskometer_fig(detail["risk_bucket"], title=detail["ticker"])
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### Key Statistics (Full History)")
    stats_dict = detail["stats"]
    render_html(
        f"""
        <table style="width: 100%; font-size: 0.9rem; line-height: 2;">
            <tr><td><b>CAGR:</b></td><td style="text-align: right;">{format_pct(stats_dict['cagr'])}</td></tr>
            <tr><td><b>Annual Volatility:</b></td><td style="text-align: right;">{format_pct(stats_dict['annual_volatility'])}</td></tr>
            <tr><td><b>Sharpe Ratio:</b></td><td style="text-align: right;">{format_ratio(stats_dict['sharpe_ratio'])}</td></tr>
            <tr><td><b>Sortino Ratio:</b></td><td style="text-align: right;">{format_ratio(stats_dict['sortino_ratio'])}</td></tr>
            <tr><td><b>Max Drawdown:</b></td><td style="text-align: right; color: {NEGATIVE_RED};">{format_pct(stats_dict['max_drawdown'])}</td></tr>
            <tr><td><b>Beta (vs Nifty 50):</b></td><td style="text-align: right;">{format_ratio(stats_dict['beta'])}</td></tr>
            <tr><td><b>Daily VaR (95%):</b></td><td style="text-align: right;">{format_pct(stats_dict['var_95'])}</td></tr>
        </table>
        """
    )

    st.markdown("---")
    st.markdown("#### Company Fundamentals")
    st.caption("*Fundamentals reflect the latest available snapshot*")
    f_dict = detail["fundamentals"]
    render_html(
        f"""
        <table style="width: 100%; font-size: 0.9rem; line-height: 2;">
            <tr><td><b>Market Cap:</b></td><td style="text-align: right;">{format_rupee(f_dict.get('market_cap'))}</td></tr>
            <tr><td><b>Trailing P/E:</b></td><td style="text-align: right;">{format_ratio(f_dict.get('trailing_pe'))}</td></tr>
            <tr><td><b>Forward P/E:</b></td><td style="text-align: right;">{format_ratio(f_dict.get('forward_pe'))}</td></tr>
            <tr><td><b>Dividend Yield:</b></td><td style="text-align: right;">{format_dividend_yield(f_dict.get('dividend_yield'))}</td></tr>
        </table>
        """
    )

render_html("<div style='margin-bottom: 24px;'></div>")

# 7. Return Histogram & Underwater Drawdown Curve
col_hist, col_dd = st.columns(2)

with col_hist:
    st.subheader("Daily Return Distribution")
    st.caption("Empirical distribution of historical daily percentage returns.")

    daily_rets = detail["daily_returns"]
    if len(daily_rets) > 10:
        fig_hist = px.histogram(
            x=daily_rets,
            nbins=60,
            histnorm="probability density",
            labels={"x": "Daily Return"},
            opacity=0.75,
            color_discrete_sequence=[PRIMARY_TEAL],
        )
        fig_hist.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(title="Density"),
        )
        apply_theme_to_plotly(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        render_empty_state("Insufficient Returns", "Not enough trading data for return distribution.")

with col_dd:
    st.subheader(f"Underwater Drawdown ({period})")
    st.caption("Percentage decline from previous historical peak water mark.")

    drawdown_df = detail["drawdown_df"]
    if not drawdown_df.empty:
        fig_dd = px.area(
            drawdown_df,
            x="date",
            y="drawdown",
            labels={"date": "Date", "drawdown": "Drawdown from Peak"},
            color_discrete_sequence=[NEGATIVE_RED],
        )
        fig_dd.update_traces(
            line=dict(width=1.5, color=NEGATIVE_RED),
            fillcolor=RED_BG_OVERLAY,
            hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Drawdown:</b> %{y:.2%}<extra></extra>",
        )
        fig_dd.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(tickformat=".1%", title="Drawdown"),
            xaxis=dict(title="Date"),
            hovermode="x unified",
        )
        apply_theme_to_plotly(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        render_empty_state("No Drawdown Data", "Drawdown history not available.")

render_html("<div style='margin-bottom: 24px;'></div>")

# 8. Sector Peers Table
st.subheader(f"Sector Peers ({detail['sector']})")
st.caption("Comparison with peer companies in the same economic sector, including pairwise correlation.")

peers_df = detail["peers"]
if not peers_df.empty:
    display_peers = peers_df.copy()
    display_peers["Cap Tier"] = display_peers["cap_class"].str.capitalize()
    display_peers["Annual Volatility"] = display_peers["annual_volatility"].apply(format_pct)
    display_peers["Sharpe Ratio"] = display_peers["sharpe_ratio"].apply(format_ratio)
    display_peers["Correlation with " + detail["ticker"]] = display_peers["correlation"].apply(format_ratio)

    st.dataframe(
        display_peers[[
            "ticker", "name", "Cap Tier", "Annual Volatility", "Sharpe Ratio", "Correlation with " + detail["ticker"]
        ]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No other peer companies found in this sector within the universe.")

render_disclaimer()
