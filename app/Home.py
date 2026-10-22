"""Home landing page with Authentication and Market Overview.

Adheres to:
- PRD Section 15.1 and AGENTS.md rules.
- Sign-up and login entry point.
- Read-only queries against DuckDB via src/app_data.py.
- User authentication and persistent profile/portfolio saving in data/users.db.
- Zero raw HTML code blocks via render_html.
- Live counts only; no dataset-size hardcoding ("10 years", "80 stocks", etc.).
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px

from app.theme import (
    BANNER_BORDER,
    BANNER_GRADIENT,
    NEGATIVE_RED,
    PRIMARY_TEAL,
    WARNING_AMBER,
    apply_theme_to_plotly,
    format_pct,
    format_ratio,
)
from app.components.ui import (
    get_current_user,
    is_logged_in,
    render_disclaimer,
    render_empty_state,
    render_html,
    render_kpi_card,
    render_page_header,
)
from src.accounts.store import (
    authenticate,
    change_password,
    delete_account,
    reset_portfolio_data,
    save_user_data,
    signup,
)
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_home_kpis_data,
    get_nifty_rebased_series,
    get_sector_1y_returns,
)
from src.profile.scoring import ProfileResult, rebuild_profile_result, TargetAllocationLevel

st.set_page_config(
    page_title="Financial Market Analytics | Home",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Database Health Check
health = check_db_health()
if not health["is_healthy"]:
    st.error(f"⚠️ {health['message']}")
    render_disclaimer()
    st.stop()

data_date = health["data_as_of"]

# Initialize session state for user session
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "profile_completed" not in st.session_state:
    st.session_state["profile_completed"] = False
if "quiz_step" not in st.session_state:
    st.session_state["quiz_step"] = 1
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}
if "portfolio_holdings" not in st.session_state:
    st.session_state["portfolio_holdings"] = []
if "portfolio_funds" not in st.session_state:
    st.session_state["portfolio_funds"] = []


# ===========================================================================
# VIEW 1: AUTHENTICATION (Sign In / Sign Up) for logged-out visitors
# ===========================================================================
if not is_logged_in():
    render_page_header("Sign In or Register", data_date)

    render_html(
        f"""
        <div style="background: {BANNER_GRADIENT}; 
                    border-radius: 12px; padding: 24px 28px; margin: 12px 0 24px 0; border: {BANNER_BORDER};">
            <h2 style="margin-top: 0; font-weight: 700;">Welcome to Financial Market Analytics</h2>
            <p style="font-size: 1.05rem; opacity: 0.85; max-width: 800px; margin-bottom: 0;">
                Evaluate risk-adjusted performance, explore economic sectors, and discover diversified portfolio ideas tailored to your risk profile.
                Please sign in or create an account to get started.
            </p>
        </div>
        """
    )

    tab_signin, tab_signup = st.tabs(["🔐 Sign In", "📝 Create Account"])

    with tab_signin:
        st.markdown("#### Access Your Account")
        with st.form("form_login"):
            login_user = st.text_input("Username or Email", placeholder="Enter your username or email")
            login_pwd = st.text_input("Password", type="password", placeholder="Enter your password")
            btn_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

        if btn_login:
            if not login_user or not login_pwd:
                st.warning("Please provide both username and password.")
            else:
                ok, msg, user_data = authenticate(login_user, login_pwd)
                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user.strip().lower()

                    # Load saved profile if present using unified rebuilder
                    if user_data and user_data.get("profile_data"):
                        p_dict = user_data["profile_data"]
                        rebuilt_prof = rebuild_profile_result(p_dict, cfg)
                        if rebuilt_prof is not None:
                            st.session_state["user_profile"] = rebuilt_prof
                            st.session_state["profile_completed"] = True
                            st.session_state["quiz_answers"] = p_dict.get("answers", {})

                    # Load saved portfolio if present
                    if user_data and user_data.get("portfolio_data"):
                        port_dict = user_data["portfolio_data"]
                        st.session_state["portfolio_holdings"] = port_dict.get("holdings", [])
                        st.session_state["portfolio_funds"] = port_dict.get("funds", [])

                    # Load saved settings / theme if present
                    if user_data and user_data.get("settings_data"):
                        sett_dict = user_data["settings_data"]
                        if isinstance(sett_dict, dict) and "theme" in sett_dict:
                            st.session_state["theme_mode"] = sett_dict["theme"]

                    st.success(f"Welcome back, {st.session_state['username']}!")
                    st.rerun()
                else:
                    st.error(f"⚠️ {msg}")

    with tab_signup:
        st.markdown("#### New Account Registration")
        with st.form("form_register"):
            new_user = st.text_input("Choose Username or Email", placeholder="At least 3 characters")
            new_pwd = st.text_input("Choose Password", type="password", placeholder="At least 8 characters")
            new_pwd2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            btn_signup = st.form_submit_button("Register Account", type="primary", use_container_width=True)

        if btn_signup:
            if not new_user or not new_pwd:
                st.warning("Please fill in all registration fields.")
            elif new_pwd != new_pwd2:
                st.error("Passwords do not match. Please retype carefully.")
            else:
                ok, msg = signup(new_user, new_pwd)
                if ok:
                    st.success("Account created successfully! You can now sign in above.")
                else:
                    st.error(f"⚠️ {msg}")

    render_disclaimer()
    st.stop()


# ===========================================================================
# VIEW 2: AUTHENTICATED DASHBOARD (Logged-In User)
# ===========================================================================
current_user = get_current_user() or "User"
render_page_header(f"Welcome, {current_user}", data_date)

# Account Management Controls
with st.expander("👤 Account Settings & Saved Data", expanded=False):
    c_acc1, c_acc2 = st.columns(2)
    with c_acc1:
        st.markdown("#### Change Password")
        with st.form("form_change_pwd"):
            old_p = st.text_input("Current Password", type="password")
            new_p = st.text_input("New Password (min 8 chars)", type="password")
            btn_cp = st.form_submit_button("Update Password")
        if btn_cp:
            ok, msg = change_password(current_user, old_p, new_p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with c_acc2:
        st.markdown("#### Account Actions")
        if st.button("🗑️ Reset Saved Portfolio", help="Clear saved portfolio holdings for your account"):
            reset_portfolio_data(current_user)
            st.session_state["portfolio_holdings"] = []
            st.session_state["portfolio_funds"] = []
            st.success("Your portfolio has been reset.")
            st.rerun()

        st.markdown("---")
        st.markdown("##### Danger Zone")
        if st.button("⚠️ Delete My Account", type="secondary", help="Permanently delete your account and all saved data"):
            delete_account(current_user)
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["user_profile"] = None
            st.session_state["portfolio_holdings"] = []
            st.session_state["portfolio_funds"] = []
            st.success("Account permanently deleted.")
            st.rerun()

# Quick stock search / jump
tickers = get_all_universe_tickers()
if tickers:
    ticker_labels = [f"{t['ticker']} - {t['name']} ({t['sector']})" for t in tickers]
    col_search, _ = st.columns([2, 1])
    with col_search:
        selected_search = st.selectbox(
            "🔍 Quick Stock Search",
            options=[""] + ticker_labels,
            index=0,
            help="Select a stock to view its deep dive profile on the Stock Detail page.",
        )
        if selected_search:
            sym = selected_search.split(" - ")[0].strip()
            st.session_state["selected_ticker"] = sym
            st.switch_page("pages/2_Stock_Detail.py")

# Hero Navigation Cards
render_html(
    f"""
    <div style="background: {BANNER_GRADIENT}; 
                border-radius: 12px; padding: 24px 28px; margin: 16px 0 20px 0; border: {BANNER_BORDER};">
        <h2 style="margin-top: 0; font-weight: 700;">Understand your risk. Build a smarter portfolio.</h2>
        <p style="font-size: 1.05rem; opacity: 0.85; max-width: 800px; margin-bottom: 12px;">
            Evaluate risk-adjusted performance across Indian equities, assess asset allocation drift, and discover ideas tailored to your risk profile.
        </p>
    </div>
    """
)

col_hero_1, col_hero_2, col_hero_3 = st.columns(3)
with col_hero_1:
    if st.button("📋 My Risk Profile", use_container_width=True, type="primary"):
        st.switch_page("pages/4_My_Profile.py")
with col_hero_2:
    if st.button("💼 My Portfolio", use_container_width=True):
        st.switch_page("pages/5_My_Portfolio.py")
with col_hero_3:
    if st.button("🌐 Explore Market", use_container_width=True):
        st.switch_page("pages/1_Explore_Market.py")

render_html("<div style='margin-bottom: 24px;'></div>")

# Context KPIs
kpis = get_home_kpis_data()

c1, c2, c3 = st.columns(3)
with c1:
    render_kpi_card("Active Stocks Tracked", f"{kpis['stocks_count']}", help_text="Analyzed stocks across economic sectors")
with c2:
    render_kpi_card("Economic Sectors", f"{kpis['sectors_count']}", help_text="Sectors represented in universe")
with c3:
    render_kpi_card("Median Market Volatility", format_pct(kpis["median_volatility"]), help_text="Annualized median volatility across stocks")

render_html("<div style='margin-bottom: 28px;'></div>")

# Charts Side-by-Side: Nifty Rebased & Sector 1Y Returns
col_chart_left, col_chart_right = st.columns(2)

with col_chart_left:
    st.subheader("Nifty 50 Rebased Growth")
    h_col1, _ = st.columns([1, 1])
    with h_col1:
        horizon = st.radio("Lookback Window", options=["1Y", "5Y", "10Y"], horizontal=True, index=0)

    nifty_df = get_nifty_rebased_series(horizon)
    if not nifty_df.empty:
        fig_nifty = px.line(
            nifty_df,
            x="date",
            y="rebased_price",
            title=f"Nifty 50 (^NSEI) Rebased to 100 ({horizon})",
            labels={"date": "Date", "rebased_price": "Rebased Level (Base = 100)"},
        )
        fig_nifty.update_traces(line=dict(color=PRIMARY_TEAL, width=2.5))
        fig_nifty.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )
        apply_theme_to_plotly(fig_nifty)
        st.plotly_chart(fig_nifty, use_container_width=True)
    else:
        render_empty_state("Data Unavailable", "Nifty 50 historical data not available.")

with col_chart_right:
    st.subheader("Sector Performance (1Y)")
    st.caption("Equal-weighted, last 1 year return across universe stocks")
    sector_df = get_sector_1y_returns()
    if not sector_df.empty:
        sector_sorted = sector_df.sort_values(by="return_1y", ascending=True)
        fig_sec = px.bar(
            sector_sorted,
            x="return_1y",
            y="sector",
            orientation="h",
            labels={"return_1y": "1-Year Return", "sector": "Sector"},
            color="return_1y",
            color_continuous_scale=[NEGATIVE_RED, WARNING_AMBER, PRIMARY_TEAL],
        )
        fig_sec.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".1%"),
        )
        apply_theme_to_plotly(fig_sec)
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        render_empty_state("Data Unavailable", "Sector return data not available.")

render_html("<div style='margin-bottom: 24px;'></div>")

# Bottom Highlight Cards
st.subheader("Market Highlights")

h1, h2, h3 = st.columns(3)
with h1:
    mv = kpis["most_volatile"]
    if mv:
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Most Volatile (Full History)</div>
                <div class="kpi-value" style="font-size: 1.4rem;">{mv['name']}</div>
                <div style="margin: 6px 0;"><code>{mv['ticker']}</code> &bull; {mv['sector']}</div>
                <div class="kpi-delta kpi-delta-negative">Annual Volatility: {format_pct(mv['annual_volatility'])}</div>
            </div>
            """
        )

with h2:
    ld = kpis["lowest_drawdown"]
    if ld:
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Lowest Max Drawdown</div>
                <div class="kpi-value" style="font-size: 1.4rem;">{ld['name']}</div>
                <div style="margin: 6px 0;"><code>{ld['ticker']}</code> &bull; {ld['sector']}</div>
                <div class="kpi-delta kpi-delta-positive">Max Drawdown: {format_pct(ld['max_drawdown'])}</div>
            </div>
            """
        )

with h3:
    hs = kpis["highest_sharpe"]
    if hs:
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Highest Sharpe Ratio</div>
                <div class="kpi-value" style="font-size: 1.4rem;">{hs['name']}</div>
                <div style="margin: 6px 0;"><code>{hs['ticker']}</code> &bull; {hs['sector']}</div>
                <div class="kpi-delta kpi-delta-positive">Sharpe Ratio: {format_ratio(hs['sharpe_ratio'])}</div>
            </div>
            """
        )

render_disclaimer()
