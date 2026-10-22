"""My Portfolio analysis & asset allocation rebalancing page.

Adheres strictly to:
- Educational decision-support tool; no buy/sell advice wording.
- Read-only against database. All metrics computed by src/portfolio.
- Manual entry (autocomplete, quantity, avg price), CSV upload with template, fund bucket amounts.
- Displays validation report for messy CSVs (bad rows dropped, duplicates merged).
- 1-click 'Load sample portfolio' button.
- KPIs, sector/cap/asset donuts, target vs current bars with drift, concentration warnings, correlation heatmap.
- Secure persistence to user account in data/users.db.
- Requires user authentication.
"""

from __future__ import annotations

import io
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from app.theme import (
    ACCENT_BLUE,
    NEGATIVE_RED,
    NEUTRAL_GREY,
    PRIMARY_TEAL,
    WARNING_AMBER,
    apply_theme_to_plotly,
    format_pct,
    format_ratio,
    format_rupee,
)
from app.components.ui import (
    get_current_user,
    is_logged_in,
    render_disclaimer,
    render_empty_state,
    render_html,
    render_kpi_card,
    render_page_header,
    render_risk_badge,
    require_login,
)
from src.accounts.store import save_user_data
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_portfolio_csv_template,
    get_sample_portfolio_holdings,
)
from src.config import load_config, load_universe
from src.portfolio.analytics import analyze_portfolio
from src.portfolio.holdings import (
    FundBucket,
    PortfolioInput,
    parse_holdings_csv,
    parse_manual_holdings,
)

st.set_page_config(
    page_title="My Portfolio | Financial Analytics",
    page_icon="💼",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Health Check
health = check_db_health()
data_date = health.get("data_as_of", "N/A")
render_page_header("My Portfolio Analysis", data_date)

cfg = load_config()
universe = load_universe()

# Helper to persist portfolio to SQLite when modified
def _persist_portfolio():
    if is_logged_in():
        user = get_current_user()
        if user:
            save_user_data(
                user,
                portfolio_data={
                    "holdings": st.session_state.get("portfolio_holdings", []),
                    "funds": st.session_state.get("portfolio_funds", []),
                },
            )

# Session State Initialization
if "portfolio_holdings" not in st.session_state:
    st.session_state["portfolio_holdings"] = []

if "portfolio_funds" not in st.session_state:
    st.session_state["portfolio_funds"] = []

if "portfolio_validation_issues" not in st.session_state:
    st.session_state["portfolio_validation_issues"] = []

# Target allocation from assessed profile or default Level 3 Moderate
user_prof = st.session_state.get("user_profile")
if user_prof:
    target_alloc = user_prof.target_allocation
    target_label = f"Level {user_prof.final_level} — {user_prof.label}"
else:
    target_alloc = cfg.target_allocations[3]
    target_label = "Level 3 — Moderate (Default)"

# 3. Portfolio Management Controls
st.subheader("Manage Portfolio Holdings")
st.caption("Add stock positions and mutual fund / ETF allocations. Your data is stored securely in your account.")

tab_manual, tab_csv, tab_sample = st.tabs([
    "Manual Entry", "CSV Upload", "Sample Portfolio & Actions"
])

all_tickers = get_all_universe_tickers()
ticker_symbols = [t["ticker"] for t in all_tickers]

with tab_manual:
    col_stk1, col_stk2, col_stk3, col_stk4 = st.columns([2.5, 1, 1.2, 1])
    with col_stk1:
        sel_sym = st.selectbox(
            "Stock Ticker",
            options=ticker_symbols,
            format_func=lambda t: f"{t} - {next((x['name'] for x in all_tickers if x['ticker'] == t), '')}",
            key="sb_manual_ticker",
        )
    with col_stk2:
        stk_qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="num_manual_qty")
    with col_stk3:
        stk_price = st.number_input("Avg Buy Price (₹)", min_value=0.0, value=1000.0, step=10.0, key="num_manual_price")
    with col_stk4:
        render_html("<div style='height: 28px;'></div>")
        if st.button("➕ Add Stock", key="btn_add_stock", use_container_width=True):
            st.session_state["portfolio_holdings"].append({
                "ticker": sel_sym,
                "quantity": int(stk_qty),
                "avg_buy_price": float(stk_price),
            })
            _persist_portfolio()
            st.success(f"Added {int(stk_qty)} shares of {sel_sym}")
            st.rerun()

    st.markdown("---")
    col_fnd1, col_fnd2, col_fnd3 = st.columns([2, 1.5, 1])
    with col_fnd1:
        fund_cat = st.selectbox(
            "Fund Category",
            options=["debt", "gold", "liquid", "large_cap_fund", "mid_cap_fund", "small_cap_fund", "flexi_cap_fund", "index_fund"],
            format_func=lambda c: c.replace("_", " ").title(),
            key="sb_fund_cat",
        )
    with col_fnd2:
        fund_amt = st.number_input("Fund Value (₹)", min_value=100.0, value=25000.0, step=1000.0, key="num_fund_amt")
    with col_fnd3:
        render_html("<div style='height: 28px;'></div>")
        if st.button("➕ Add Fund", key="btn_add_fund", use_container_width=True):
            existing_f = [f for f in st.session_state["portfolio_funds"] if f["category"] == fund_cat]
            if existing_f:
                existing_f[0]["amount"] += fund_amt
            else:
                st.session_state["portfolio_funds"].append({
                    "category": fund_cat,
                    "amount": fund_amt,
                })
            _persist_portfolio()
            st.success(f"Added ₹{fund_amt:,.2f} to {fund_cat} bucket")
            st.rerun()

with tab_csv:
    st.markdown("Upload a CSV file containing your stock holdings. Required columns: `ticker`, `quantity`, optional `avg_buy_price`.")
    col_up, col_tmpl = st.columns([3, 1])
    with col_tmpl:
        template_str = get_portfolio_csv_template()
        st.download_button(
            "📥 Download CSV Template",
            data=template_str,
            file_name="portfolio_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_up:
        uploaded_file = st.file_uploader("Upload Holdings CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                parsed_port = parse_holdings_csv(
                    csv_source=io.StringIO(content),
                    universe=universe,
                )
                report = parsed_port.validation_report

                st.session_state["portfolio_holdings"] = [
                    {"ticker": h.ticker, "quantity": h.quantity, "avg_buy_price": h.avg_buy_price}
                    for h in parsed_port.holdings
                ]
                st.session_state["portfolio_validation_issues"] = report.issues
                _persist_portfolio()
                st.success(f"Successfully loaded {len(parsed_port.holdings)} holdings from CSV.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse CSV file: {e}")

with tab_sample:
    st.markdown("Quickly test the analytics engine by loading a pre-configured multi-cap balanced portfolio or clearing existing data.")
    c_s1, c_s2, _ = st.columns([1.5, 1.5, 3])
    with c_s1:
        if st.button("🚀 Load Sample Portfolio", type="primary", use_container_width=True):
            sample_stocks, sample_funds = get_sample_portfolio_holdings()
            st.session_state["portfolio_holdings"] = sample_stocks
            st.session_state["portfolio_funds"] = sample_funds
            st.session_state["portfolio_validation_issues"] = []
            _persist_portfolio()
            st.success("Loaded balanced sample portfolio (6 stocks, debt & gold funds)!")
            st.rerun()
    with c_s2:
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state["portfolio_holdings"] = []
            st.session_state["portfolio_funds"] = []
            st.session_state["portfolio_validation_issues"] = []
            _persist_portfolio()
            st.rerun()

# 4. Display Validation Report Issues (if any)
if st.session_state["portfolio_validation_issues"]:
    with st.expander("⚠️ Data Quality & Parsing Notice", expanded=True):
        for issue in st.session_state["portfolio_validation_issues"]:
            st.info(f"&bull; {issue}")

# 5. Check If Portfolio Is Empty
holdings_list = st.session_state["portfolio_holdings"]
funds_list = st.session_state["portfolio_funds"]

if not holdings_list and not funds_list:
    render_empty_state(
        "Your Portfolio is Empty",
        "Add stock holdings manually, upload a CSV file, or click 'Load Sample Portfolio' above to view risk analytics.",
        action_label="Load Sample Portfolio",
    )
    if st.session_state.get("btn_empty_Your Portfolio is Empty"):
        s_stocks, s_funds = get_sample_portfolio_holdings()
        st.session_state["portfolio_holdings"] = s_stocks
        st.session_state["portfolio_funds"] = s_funds
        _persist_portfolio()
        st.rerun()

    render_disclaimer()
    st.stop()

# 6. Parse and Analyze Portfolio
port_input = parse_manual_holdings(
    stock_rows=holdings_list,
    fund_rows=funds_list,
    universe=universe,
)

analytics = analyze_portfolio(
    portfolio=port_input,
    target_allocation=target_alloc,
    config=cfg,
)

# Concentration Limit Warnings
if analytics.concentration_warnings:
    for w in analytics.concentration_warnings:
        st.warning(f"⚠️ **Concentration Limit Breached:** {w}")

# 7. Portfolio Valuation & Risk KPI Ribbon
st.markdown("---")
st.subheader("Valuation & Key Risk Metrics")

k_c1, k_c2, k_c3, k_c4 = st.columns(4)
with k_c1:
    gain_loss_delta = f"{format_rupee(analytics.total_unrealized_gain_loss)} ({format_pct(analytics.total_gain_loss_pct / 100.0, show_plus=True)})"
    gain_loss_type = "positive" if analytics.total_unrealized_gain_loss >= 0 else "negative"
    render_kpi_card(
        "Total Portfolio Value",
        format_rupee(analytics.total_value),
        delta=gain_loss_delta,
        delta_type=gain_loss_type,
        help_text="Combined value of all stock holdings and fund bucket assets",
    )

with k_c2:
    render_kpi_card(
        "Annualized Volatility",
        format_pct(analytics.volatility),
        help_text="Weighted portfolio historical annual volatility (252 days)",
    )

with k_c3:
    render_kpi_card(
        "Max Historical Drawdown",
        format_pct(analytics.max_drawdown),
        delta_type="negative",
        help_text="Deepest historical peak-to-trough decline of current portfolio weights",
    )

with k_c4:
    render_kpi_card(
        "Diversification Score",
        f"{analytics.diversification_score:.1f} / 100",
        help_text="0-100 diversification benefit score based on correlation matrix and volatility reduction",
    )

render_html("<div style='margin-bottom: 24px;'></div>")

# 8. Composition Donut Charts
st.subheader("Portfolio Asset & Sector Breakdown")
c_don1, c_don2, c_don3 = st.columns(3)

with c_don1:
    st.markdown("##### Broad Asset Mix")
    asset_df = pd.DataFrame([
        {"Asset Class": "Equity", "Value": analytics.equity_value},
        {"Asset Class": "Debt", "Value": analytics.debt_value},
        {"Asset Class": "Gold", "Value": analytics.gold_value},
        {"Asset Class": "Other", "Value": analytics.other_value},
    ])
    asset_df = asset_df[asset_df["Value"] > 0]
    fig_asset = px.pie(
        asset_df,
        values="Value",
        names="Asset Class",
        hole=0.55,
        color="Asset Class",
        color_discrete_map={"Equity": PRIMARY_TEAL, "Debt": ACCENT_BLUE, "Gold": WARNING_AMBER, "Other": NEUTRAL_GREY},
    )
    fig_asset.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    apply_theme_to_plotly(fig_asset)
    st.plotly_chart(fig_asset, use_container_width=True)

with c_don2:
    st.markdown("##### Equity Sector Weights")
    sec_data = [{"Sector": k, "Weight": v} for k, v in analytics.sector_weights.items() if v > 0]
    if sec_data:
        sec_df = pd.DataFrame(sec_data)
        fig_sec = px.pie(sec_df, values="Weight", names="Sector", hole=0.55)
        fig_sec.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        apply_theme_to_plotly(fig_sec)
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        render_empty_state("No Equity Positions", "No direct stock equity positions held.")

with c_don3:
    st.markdown("##### Equity Cap Tier Allocation")
    cap_data = [{"Tier": k.capitalize() + " Cap", "Weight": v} for k, v in analytics.cap_weights.items() if v > 0]
    if cap_data:
        cap_df = pd.DataFrame(cap_data)
        fig_cap = px.pie(cap_df, values="Weight", names="Tier", hole=0.55)
        fig_cap.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        apply_theme_to_plotly(fig_cap)
        st.plotly_chart(fig_cap, use_container_width=True)
    else:
        render_empty_state("No Cap Allocation", "No equity holdings available.")

render_html("<div style='margin-bottom: 24px;'></div>")

# 9. Detailed Holdings Table
st.subheader("Current Stock Holdings Detail")

holdings_records = []
for h in port_input.holdings:
    val = h.current_value
    p_weight = analytics.stock_portfolio_weights.get(h.ticker, 0.0)
    eq_weight = analytics.stock_weights.get(h.ticker, 0.0)
    cost = (h.avg_buy_price * h.quantity) if h.avg_buy_price else 0.0
    gain_loss = h.gain_loss
    gain_pct = (gain_loss / cost) if cost > 0 else 0.0

    holdings_records.append({
        "Ticker": h.ticker,
        "Quantity": h.quantity,
        "Avg Buy Price": format_rupee(h.avg_buy_price) if h.avg_buy_price else "N/A",
        "Current Price": format_rupee(h.latest_price) if h.latest_price else "N/A",
        "Current Value": format_rupee(val),
        "Portfolio %": format_pct(p_weight),
        "Equity %": format_pct(eq_weight),
        "P&L (₹)": format_rupee(gain_loss),
        "Return %": format_pct(gain_pct, show_plus=True),
    })

if holdings_records:
    df_holdings_ui = pd.DataFrame(holdings_records)
    st.dataframe(df_holdings_ui, use_container_width=True, hide_index=True)

render_html("<div style='margin-bottom: 28px;'></div>")

# 10. Target vs. Current Asset Allocation & Drift
st.subheader("Asset Allocation Drift Analysis")
st.caption(f"Comparing current asset distribution against your target profile: **{target_label}**")

c_drift_chart, c_drift_txt = st.columns([1.5, 1])

drift_records = []
curr_alloc = analytics.asset_allocation_current
targ_alloc = analytics.asset_allocation_target or {"equity": 0.50, "debt": 0.40, "gold": 0.10}
drift_vals = analytics.drift or {}

for asset_k in ["equity", "debt", "gold"]:
    c_w = curr_alloc.get(asset_k, 0.0)
    t_w = targ_alloc.get(asset_k, 0.0)
    d_w = drift_vals.get(asset_k, c_w - t_w)
    drift_records.append({
        "Asset Class": asset_k.capitalize(),
        "Current Weight": c_w,
        "Target Weight": t_w,
        "Drift": d_w,
    })

drift_df = pd.DataFrame(drift_records)

with c_drift_chart:
    drift_plot_df = pd.melt(
        drift_df,
        id_vars=["Asset Class"],
        value_vars=["Current Weight", "Target Weight"],
        var_name="Allocation Type",
        value_name="Weight",
    )
    fig_drift = px.bar(
        drift_plot_df,
        x="Asset Class",
        y="Weight",
        color="Allocation Type",
        barmode="group",
        color_discrete_map={"Current Weight": PRIMARY_TEAL, "Target Weight": NEUTRAL_GREY},
    )
    fig_drift.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_theme_to_plotly(fig_drift)
    st.plotly_chart(fig_drift, use_container_width=True)

with c_drift_txt:
    st.markdown("#### Drift Analysis")
    for _, row in drift_df.iterrows():
        d_val = row["Drift"]
        arrow = "🔺 Overweight" if d_val > 0.02 else ("🔻 Underweight" if d_val < -0.02 else "✅ Balanced")
        color = NEGATIVE_RED if abs(d_val) > 0.05 else (WARNING_AMBER if abs(d_val) > 0.02 else PRIMARY_TEAL)
        render_html(
            f"""
            <div style="padding: 10px 14px; background: var(--secondary-background-color); border-radius: 6px; margin-bottom: 8px;">
                <b>{row['Asset Class']}:</b> {format_pct(row['Current Weight'])} (Target: {format_pct(row['Target Weight'])})
                <span style="float: right; color: {color}; font-weight: 600;">{arrow} ({format_pct(d_val, show_plus=True)})</span>
            </div>
            """
        )

# 11. Portfolio Correlation Heatmap
if hasattr(analytics, "correlation_matrix") and analytics.correlation_matrix is not None and not analytics.correlation_matrix.empty and len(analytics.correlation_matrix) >= 2:
    st.markdown("---")
    st.subheader("Holdings Correlation Heat Map")
    st.caption("Pairwise daily return correlations between held stocks.")
    fig_port_corr = px.imshow(
        analytics.correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
    )
    fig_port_corr.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=10, b=20),
    )
    apply_theme_to_plotly(fig_port_corr)
    st.plotly_chart(fig_port_corr, use_container_width=True)

render_disclaimer()
