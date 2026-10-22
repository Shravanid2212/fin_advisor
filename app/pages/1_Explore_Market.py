"""Explore Market analytical dashboard.

Adheres to:
- Practical market views: Top gainers/losers, sector performance, risk-adjusted leaderboard,
  lowest-drawdown list, and cap-class/sector breakdown donuts.
- Benchmarks and synthetic baskets appear as reference lines, never as a sector.
- Pure investor metrics with plain-language labels (no statistical tests or p-values).
- Read-only queries via src/app_data.py.
- Requires user authentication.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from app.theme import (
    ACCENT_BLUE,
    CARD_BORDER_OVERLAY,
    NEGATIVE_RED,
    NEUTRAL_GREY,
    PRIMARY_TEAL,
    WARNING_AMBER,
    apply_theme_to_plotly,
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
    get_explore_market_data,
    get_sector_1y_returns,
)

st.set_page_config(
    page_title="Explore Market | Financial Analytics",
    page_icon="🌐",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Database Health Check
health = check_db_health()
if not health["is_healthy"]:
    st.error(f"⚠️ {health['message']}")
    render_disclaimer()
    st.stop()

data_date = health["data_as_of"]
render_page_header("Explore Market", data_date)

# 3. Interactive Filters
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

col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1.5, 1, 1])

with col_f1:
    selected_sectors = st.multiselect(
        "Economic Sectors",
        options=SECTORS_LIST,
        default=SECTORS_LIST,
        help="Filter stocks and sector series by economic sector.",
    )

with col_f2:
    selected_caps = st.multiselect(
        "Market Cap Tier",
        options=CAPS_LIST,
        default=CAPS_LIST,
        format_func=lambda c: f"{c.capitalize()} Cap",
        help="Market capitalization tier classification.",
    )

with col_f3:
    period = st.selectbox(
        "Time Horizon",
        options=["1Y", "3Y", "5Y", "10Y"],
        index=1,
        help="Historical lookback window for rebased performance.",
    )

# Handle empty filter selections gracefully
if not selected_sectors or not selected_caps:
    st.warning("⚠️ Please select at least one sector and one market cap category to display analysis.")
    render_empty_state(
        "No Data Selected",
        "Select one or more sectors and market cap tiers above to load market views and charts.",
    )
    render_disclaimer()
    st.stop()

# 4. Load Data from Backend Service
data = get_explore_market_data(
    selected_sectors=selected_sectors,
    selected_caps=selected_caps,
    period=period,
)

scatter_df = data["scatter_df"]
rebased_df = data["rebased_df"]

with col_f4:
    if not scatter_df.empty:
        csv_data = scatter_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"market_explore_{period.lower()}.csv",
            mime="text/csv",
            help="Download filtered stock metrics as CSV.",
        )

render_html("<div style='margin-bottom: 20px;'></div>")

# Live Filter Count Banner
st.markdown(f"**Showing results for:** `{len(scatter_df)} active stocks` across `{len(selected_sectors)} sectors`")

# 5. Row 1: Rebased Growth by Sector vs Benchmarks & Risk vs. Return
c_r1_left, c_r1_right = st.columns(2)

with c_r1_left:
    st.subheader("Rebased Growth by Sector")
    st.caption("Indexed to 100 at the start of the selected window for direct like-for-like comparison.")

    if not rebased_df.empty:
        fig_reb = go.Figure()
        for col in rebased_df.columns:
            if "(Ref)" in col or "average" in col:
                dash_style = "dot" if "average" in col else "dash"
                line_color = ACCENT_BLUE if ("^NSEI" in col or "Nifty" in col) else NEUTRAL_GREY
                if "Mid-cap average" in col:
                    tooltip_txt = "Average of the mid-cap stocks in this app"
                elif "Small-cap average" in col:
                    tooltip_txt = "Average of the small-cap stocks in this app"
                else:
                    tooltip_txt = "Broad market index"
                fig_reb.add_trace(
                    go.Scatter(
                        x=rebased_df.index,
                        y=rebased_df[col],
                        mode="lines",
                        name=col,
                        line=dict(dash=dash_style, width=2, color=line_color),
                        hovertemplate=f"<b>{col}</b><br>Level: %{{y:.1f}}<br><i>({tooltip_txt})</i><extra></extra>",
                    )
                )
            else:
                fig_reb.add_trace(
                    go.Scatter(
                        x=rebased_df.index,
                        y=rebased_df[col],
                        mode="lines",
                        name=col,
                        line=dict(width=1.8),
                        hovertemplate=f"<b>{col}</b><br>Level: %{{y:.1f}}<extra></extra>",
                    )
                )

        fig_reb.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Rebased Level (Base = 100)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        )
        apply_theme_to_plotly(fig_reb)
        st.plotly_chart(fig_reb, use_container_width=True)
    else:
        render_empty_state("No Growth Data", "Unable to compute rebased growth for the selected criteria.")

with c_r1_right:
    st.subheader("Risk vs. Return")
    st.caption("Top-left = higher growth for less risk. Bubble size represents market cap.")

    if not scatter_df.empty:
        # Sensible axis padding so small differences don't look huge
        x_min, x_max = float(scatter_df["annual_volatility"].min()), float(scatter_df["annual_volatility"].max())
        y_min, y_max = float(scatter_df["cagr"].min()), float(scatter_df["cagr"].max())
        x_span = max(0.10, x_max - x_min)
        y_span = max(0.12, y_max - y_min)
        x_pad = x_span * 0.15
        y_pad = y_span * 0.15
        x_range = [max(0.0, x_min - x_pad), x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]

        fig_scat = px.scatter(
            scatter_df,
            x="annual_volatility",
            y="cagr",
            color="sector",
            size="market_cap",
            text="ticker",
            hover_name="ticker",
            opacity=0.75,
            size_max=32,
            hover_data={
                "name": True,
                "sector": True,
                "cap_class": True,
                "annual_volatility": ":.2%",
                "cagr": ":.2%",
                "sharpe_ratio": ":.2f",
                "market_cap": False,
                "ticker": False,
            },
            labels={
                "annual_volatility": "Annualized Volatility (Risk)",
                "cagr": "Compound Annual Growth Rate",
                "sector": "Sector",
            },
        )
        fig_scat.update_traces(
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(line=dict(width=1, color=CARD_BORDER_OVERLAY)),
        )
        fig_scat.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(tickformat=".1%", range=x_range),
            yaxis=dict(tickformat=".1%", range=y_range),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        )
        apply_theme_to_plotly(fig_scat)
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        render_empty_state("No Scatter Data", "No stocks matched the selected filters.")

render_html("<div style='margin-bottom: 24px;'></div>")

# 6. Practical View: Top Gainers and Top Losers
st.subheader(f"Top Gainers & Losers ({period})")
c_gain, c_loss = st.columns(2)

if not scatter_df.empty:
    sorted_stocks = scatter_df.sort_values(by="cagr", ascending=False)
    top_gainers = sorted_stocks.head(5)[["ticker", "name", "sector", "cap_class", "cagr", "annual_volatility"]]
    top_losers = sorted_stocks.tail(5).iloc[::-1][["ticker", "name", "sector", "cap_class", "cagr", "annual_volatility"]]

    with c_gain:
        st.markdown("##### 🚀 Top Performers")
        gain_df = top_gainers.copy()
        gain_df["cagr"] = gain_df["cagr"].apply(lambda v: format_pct(v, show_plus=True))
        gain_df["annual_volatility"] = gain_df["annual_volatility"].apply(format_pct)
        gain_df["cap_class"] = gain_df["cap_class"].str.capitalize()
        gain_df.columns = ["Ticker", "Company", "Sector", "Tier", "Annual Return", "Volatility"]
        st.dataframe(gain_df, use_container_width=True, hide_index=True)

    with c_loss:
        st.markdown("##### 📉 Lagging Performers")
        loss_df = top_losers.copy()
        loss_df["cagr"] = loss_df["cagr"].apply(lambda v: format_pct(v, show_plus=True))
        loss_df["annual_volatility"] = loss_df["annual_volatility"].apply(format_pct)
        loss_df["cap_class"] = loss_df["cap_class"].str.capitalize()
        loss_df.columns = ["Ticker", "Company", "Sector", "Tier", "Annual Return", "Volatility"]
        st.dataframe(loss_df, use_container_width=True, hide_index=True)

render_html("<div style='margin-bottom: 24px;'></div>")

# 7. Sector Performance (Table and Bar Chart, equal-weighted)
st.subheader("Sector Performance Breakdown")
st.caption("Equal-weighted historical returns and median volatility across economic sectors.")

c_sec_tbl, c_sec_bar = st.columns([1.2, 1.8])
sector_df = get_sector_1y_returns()

with c_sec_tbl:
    if not sector_df.empty:
        sec_tbl_display = sector_df.copy().sort_values(by="return_1y", ascending=False)
        sec_tbl_display["return_1y"] = sec_tbl_display["return_1y"].apply(lambda v: format_pct(v, show_plus=True))
        sec_tbl_display.columns = ["Sector", "1Y Return (Equal-Weighted)", "Active Constituents"]
        st.dataframe(sec_tbl_display, use_container_width=True, hide_index=True)

with c_sec_bar:
    if not sector_df.empty:
        fig_sec_bar = px.bar(
            sector_df.sort_values(by="return_1y", ascending=True),
            x="return_1y",
            y="sector",
            orientation="h",
            labels={"return_1y": "1-Year Return", "sector": "Sector"},
            color="return_1y",
            color_continuous_scale=[NEGATIVE_RED, WARNING_AMBER, PRIMARY_TEAL],
        )
        fig_sec_bar.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".1%"),
        )
        apply_theme_to_plotly(fig_sec_bar)
        st.plotly_chart(fig_sec_bar, use_container_width=True)

render_html("<div style='margin-bottom: 24px;'></div>")

# 8. Leaderboards: Best Risk-Adjusted & Lowest Drawdown
c_ldr_sharpe, c_ldr_dd = st.columns(2)

with c_ldr_sharpe:
    st.subheader("Best Risk-Adjusted Stocks")
    st.caption("Top stocks ranked by Sharpe ratio (excess return per unit of total risk).")
    if not scatter_df.empty:
        top_sharpe = scatter_df.sort_values(by="sharpe_ratio", ascending=False).head(5).copy()
        top_sharpe["sharpe_ratio"] = top_sharpe["sharpe_ratio"].apply(format_ratio)
        top_sharpe["cagr"] = top_sharpe["cagr"].apply(format_pct)
        top_sharpe["annual_volatility"] = top_sharpe["annual_volatility"].apply(format_pct)
        top_sharpe["cap_class"] = top_sharpe["cap_class"].str.capitalize()
        top_sharpe = top_sharpe[["ticker", "name", "sector", "cap_class", "sharpe_ratio", "cagr", "annual_volatility"]]
        top_sharpe.columns = ["Ticker", "Company", "Sector", "Tier", "Sharpe", "Return", "Volatility"]
        st.dataframe(top_sharpe, use_container_width=True, hide_index=True)

with c_ldr_dd:
    st.subheader("Lowest Historical Drawdown")
    st.caption("Stocks exhibiting the shallowest historical peak-to-trough declines.")
    if not scatter_df.empty:
        # Load max drawdown from stock_features
        from src.app_data import get_db_connection
        con_dd = get_db_connection()
        if con_dd:
            dd_query = """
                SELECT ticker, name, sector, cap_class, max_drawdown, recovery_days, risk_bucket
                FROM stock_features
                WHERE ticker NOT IN ('^NSEI', '^CRSLDX', 'synthetic_midcap', 'synthetic_smallcap')
                ORDER BY max_drawdown DESC
                LIMIT 5
            """
            dd_df = con_dd.execute(dd_query).fetchdf()
            con_dd.close()
            if not dd_df.empty:
                dd_df["max_drawdown"] = dd_df["max_drawdown"].apply(format_pct)
                dd_df["cap_class"] = dd_df["cap_class"].str.capitalize()
                dd_df.columns = ["Ticker", "Company", "Sector", "Tier", "Max Drawdown", "Recovery (Days)", "Risk Level"]
                st.dataframe(dd_df, use_container_width=True, hide_index=True)

render_html("<div style='margin-bottom: 24px;'></div>")

# 9. Breakdown Donut Charts: Market Cap and Sector Composition
st.subheader("Market Composition Breakdown")
c_dnt_cap, c_dnt_sec = st.columns(2)

with c_dnt_cap:
    st.markdown("##### Market Cap Tier Distribution")
    if not scatter_df.empty:
        cap_counts = scatter_df["cap_class"].value_counts().reset_index()
        cap_counts.columns = ["Tier", "Count"]
        cap_counts["Tier"] = cap_counts["Tier"].str.capitalize() + " Cap"
        fig_cap_donut = px.pie(
            cap_counts,
            values="Count",
            names="Tier",
            hole=0.55,
            color="Tier",
            color_discrete_sequence=[PRIMARY_TEAL, ACCENT_BLUE, WARNING_AMBER],
        )
        fig_cap_donut.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        apply_theme_to_plotly(fig_cap_donut)
        st.plotly_chart(fig_cap_donut, use_container_width=True)

with c_dnt_sec:
    st.markdown("##### Sector Representation")
    if not scatter_df.empty:
        sec_counts = scatter_df["sector"].value_counts().reset_index()
        sec_counts.columns = ["Sector", "Count"]
        fig_sec_donut = px.pie(
            sec_counts,
            values="Count",
            names="Sector",
            hole=0.55,
        )
        fig_sec_donut.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        apply_theme_to_plotly(fig_sec_donut)
        st.plotly_chart(fig_sec_donut, use_container_width=True)

render_disclaimer()
