"""Explainable Decision Support Recommendations page.

Adheres strictly to:
- Educational decision-support tool; never use buy/sell wording.
- Phrasing: 'Ideas that fit your profile', 'Suggested portfolio additions'.
- Needs a risk profile (prompt if missing) and works seamlessly with or without an existing portfolio.
- Summary bar, up to 8 recommendation cards with factual numerical explanations.
- What-if panel (before vs after volatility and diversification), +1/-1 level comparison, and cap-class mix.
- Pure HTML rendering with render_html (no raw div text in UI).
- Requires user authentication.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from app.theme import (
    ACCENT_BLUE,
    CARD_BG_OVERLAY,
    NEGATIVE_RED,
    NEUTRAL_GREY,
    PRIMARY_TEAL,
    TEAL_BG_OVERLAY,
    WARNING_AMBER,
    apply_theme_to_plotly,
    format_date,
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
    render_page_header,
    render_projection_panel,
    render_risk_badge,
    require_login,
)
from src.accounts.store import save_user_data
from src.app_data import (
    check_db_health,
    get_all_universe_tickers,
    get_fund_risk_outlook,
)
from src.config import load_config, load_universe
from src.portfolio.analytics import analyze_portfolio
from src.portfolio.holdings import PortfolioInput, parse_manual_holdings
from src.profile.scoring import ProfileResult
from src.recommend.recommender import generate_recommendations
from src.recommend.funds import generate_fund_recommendations

st.set_page_config(
    page_title="Ideas & Recommendations | Financial Analytics",
    page_icon="💡",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Database Health Check
health = check_db_health()
data_date = health.get("data_as_of", "N/A")
render_page_header("Ideas That Fit Your Profile", data_date)

cfg = load_config()
universe = load_universe()

# 3. Risk Profile Check (Needs a profile, prompt if missing)
user_prof: ProfileResult | None = st.session_state.get("user_profile")

if user_prof is None:
    st.info("ℹ️ **Personalized recommendations require an investor risk profile.**")
    c_p1, c_p2 = st.columns([1.5, 2])
    with c_p1:
        if st.button("📋 Take the Risk Questionnaire", type="primary", use_container_width=True):
            st.switch_page("pages/4_My_Profile.py")
    with c_p2:
        st.markdown("**Or select a risk profile level to preview ideas:**")
        preview_level = st.selectbox(
            "Risk Profile Level Preview",
            options=[1, 2, 3, 4, 5],
            index=2,
            format_func=lambda lvl: f"Level {lvl} — {cfg.target_allocations[lvl].label}",
            label_visibility="collapsed",
        )
        active_level = preview_level
        profile_label = cfg.target_allocations[active_level].label
        target_equity_share = cfg.target_allocations[active_level].equity
else:
    active_level = user_prof.final_level
    profile_label = user_prof.label
    target_equity_share = user_prof.target_allocation.equity

render_html("<div style='margin-bottom: 16px;'></div>")

# 4. Preference Overrides & Filters
prof_prefs = getattr(user_prof, "preferences", {}) or {}
pref_fav = prof_prefs.get("favoured_sectors", [])
pref_avoid = prof_prefs.get("avoided_sectors", [])

SECTORS_LIST = [
    "Banking and Financials", "IT", "Pharma and Healthcare", "FMCG",
    "Auto", "Energy and Power", "Metals", "Capital Goods and Infra",
    "Consumer", "Chemicals"
]

with st.expander("⚙️ Refine Preferences & Sector Focus", expanded=False):
    c_prf1, c_prf2 = st.columns(2)
    with c_prf1:
        cap_pref = st.selectbox(
            "Market Cap Preference",
            options=["no preference", "large only", "large + mid", "include small"],
            index=0,
            help="Filter candidate universe by market capitalization tier.",
        )
        div_pref = st.selectbox(
            "Dividend Focus",
            options=["no", "yes"],
            format_func=lambda d: "Prioritize Dividend Yield" if d == "yes" else "Prioritize 12M Momentum",
            index=0,
            help="Favor high dividend yield stocks over price momentum.",
        )
    with c_prf2:
        curr_avoided = st.session_state.get("rec_excluded", pref_avoid)
        favoured = st.multiselect(
            "Favoured Sectors (Score Bonus)",
            options=[s for s in SECTORS_LIST if s not in curr_avoided],
            default=[s for s in pref_fav if s in SECTORS_LIST and s not in curr_avoided],
            key="rec_favoured",
            help="Stocks in favoured sectors receive a score boost.",
        )
        excluded = st.multiselect(
            "Excluded Sectors (Hard Filter)",
            options=[s for s in SECTORS_LIST if s not in favoured],
            default=[s for s in pref_avoid if s in SECTORS_LIST and s not in favoured],
            key="rec_excluded",
            help="Exclude specific sectors from candidate consideration.",
        )

# 5. Resolve Active Portfolio (Works with or without portfolio)
holdings_list = st.session_state.get("portfolio_holdings", [])
funds_list = st.session_state.get("portfolio_funds", [])

if holdings_list or funds_list:
    active_port = parse_manual_holdings(
        stock_rows=holdings_list,
        fund_rows=funds_list,
        universe=universe,
    )
    port_status_msg = f"Customized for your current portfolio of {len(active_port.holdings)} stock holdings."
else:
    active_port = PortfolioInput(holdings=[], funds=[])
    port_status_msg = "No existing portfolio loaded. Diversification gains evaluated against the broad market index (^NSEI)."

# 5b. Investment Capital Allocation Input & Default Suggestion
has_portfolio = bool(holdings_list or funds_list)
has_profile = (user_prof is not None)
default_suggested_amount = None
curr_equity_share = 0.0

if has_portfolio:
    port_analytics = analyze_portfolio(active_port, config=cfg)
    total_val = port_analytics.total_value
    eq_val = port_analytics.equity_value
    curr_equity_share = (eq_val / total_val) if total_val > 0 else 0.0
    if has_profile:
        eq_gap_share = max(0.0, target_equity_share - curr_equity_share)
        default_suggested_amount = round(eq_gap_share * total_val)

st.markdown("### 💰 Allocate Capital to Ideas")
c_amt1, c_amt_mode, c_amt2 = st.columns([1.5, 1.3, 2.2])
with c_amt1:
    if default_suggested_amount is not None:
        init_val = int(default_suggested_amount)
    else:
        init_val = None

    alloc_amount = st.number_input(
        "Amount to allocate to these ideas (₹)",
        min_value=0,
        value=init_val,
        step=5000,
        placeholder="e.g. 100,000",
        help="Target investment amount in rupees to distribute across recommended stock ideas.",
    )

with c_amt_mode:
    investment_mode = st.selectbox(
        "Investment Horizon Mode",
        options=["One-time investment", "Monthly SIP"],
        index=0,
        help="Choose whether this capital represents a one-time allocation or a recurring monthly SIP.",
    )
    is_monthly_sip = (investment_mode == "Monthly SIP")
    sip_label = "per month" if is_monthly_sip else "one-time"

with c_amt2:
    if default_suggested_amount is not None:
        render_html(
            f"""
            <div style="margin-top: 28px; font-size: 0.9rem; line-height: 1.5; opacity: 0.85;">
                💡 <b>Profile Target Gap Suggestion:</b> {format_rupee(default_suggested_amount)} ({sip_label})<br>
                <span style="font-size: 0.8rem; opacity: 0.75;">(Target Equity: {format_pct(target_equity_share)} | Current Equity: {format_pct(curr_equity_share)})</span>
            </div>
            """
        )
    else:
        render_html(
            f"""
            <div style="margin-top: 28px; font-size: 0.9rem; line-height: 1.5; opacity: 0.75;">
                💡 Enter an amount in rupees to calculate suggested share quantities, actual amounts, and post-addition portfolio weights ({sip_label}).
            </div>
            """
        )

# 6. Generate Recommendations
with st.spinner("Evaluating universe stocks and calculating portfolio diversification impact..."):
    rec_result = generate_recommendations(
        profile_level=active_level,
        portfolio=active_port,
        cap_preference=cap_pref,
        excluded_sectors=excluded,
        favoured_sectors=favoured,
        dividend_preference=div_pref,
        config=cfg,
        allocation_amount=float(alloc_amount) if (alloc_amount is not None and alloc_amount > 0) else None,
    )
    fund_rec_result = generate_fund_recommendations(
        profile_level=active_level,
        portfolio=active_port,
        user_profile=user_prof,
        allocation_amount=float(alloc_amount) if (alloc_amount is not None and alloc_amount > 0) else None,
        cap_preference=cap_pref,
        excluded_sectors=excluded,
        config=cfg,
    )

tab_stocks, tab_funds = st.tabs(["📈 Stock Ideas", "🏦 Mutual Fund Ideas"])

with tab_stocks:
    # 7. Summary Ribbon
    st.markdown("---")
    c_sum1, c_sum2, c_sum3 = st.columns([2, 1, 1])

    with c_sum1:
        st.markdown(
            f"### Showing {len(rec_result.recommendations)} Ideas for Level {active_level} ({profile_label})\n"
            f"*{port_status_msg} Target profile equity share: **{format_pct(target_equity_share)}**.*"
        )

    with c_sum2:
        st.metric(
            label="Top Pick Weight",
            value=format_pct(rec_result.recommendations[0].suggested_weight) if rec_result.recommendations else "N/A",
            help="Max recommended weight is capped at 20% in config",
        )

    with c_sum3:
        st.metric(
            label="Total Suggested Additions",
            value=f"{len(rec_result.recommendations)} stocks",
            help="Capped at top 8 ideas, max 2 per sector",
        )

    for n in rec_result.notes:
        if "fit your risk level" in n:
            st.info(f"ℹ️ **Preference Note:** {n}")

    render_html("<div style='margin-bottom: 24px;'></div>")

    # 8. Suggested Quantities and Amounts Table & CSV Download
    plan = rec_result.allocation_plan
    if plan and plan.items and alloc_amount and alloc_amount > 0:
        st.subheader("Suggested Allocation & Whole Share Quantities")
        st.caption("*Prices reflect last close, not a live price.*")

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.metric("Entered Capital", format_rupee(plan.total_amount))
        with c_m2:
            st.metric("Total Actual Allocated", format_rupee(plan.total_actual_amount))
        with c_m3:
            st.metric(
                "Unallocated Leftover",
                format_rupee(plan.unallocated_leftover),
                help="Cash remaining due to whole share price constraints or per-stock weight caps.",
            )

        table_rows = []
        csv_rows = []
        for item in plan.items:
            qty_str = (
                f"{item.suggested_quantity:,}"
                if item.suggested_quantity > 0
                else f"0 (Min for 1 share: {format_rupee(item.min_amount_for_one_share)})"
            )
            close_str = f"{format_rupee(item.last_close)} ({format_date(item.price_date)})"
            table_rows.append({
                "Idea / Company": f"{item.name} ({item.ticker})",
                "Last Close (Date)*": close_str,
                "Suggested Quantity": qty_str,
                "Indicative Amount": format_rupee(item.indicative_amount),
                "Actual Amount": format_rupee(item.actual_amount),
                "Basket Weight": format_pct(item.basket_weight),
                "Portfolio Weight After Addition": format_pct(item.weight_after_addition),
            })
            csv_rows.append({
                "Ticker": item.ticker,
                "Company Name": item.name,
                "Sector": item.sector,
                "Last Close": item.last_close,
                "Price Date": item.price_date,
                "Suggested Quantity": item.suggested_quantity,
                "Indicative Amount": round(item.indicative_amount, 2),
                "Actual Amount": round(item.actual_amount, 2),
                "Basket Weight": format_pct(item.basket_weight),
                "Portfolio Weight After Addition": format_pct(item.weight_after_addition),
                "Note": item.note,
            })

        alloc_df = pd.DataFrame(table_rows)
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        csv_df = pd.DataFrame(csv_rows)
        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Suggested Allocations (CSV)",
            data=csv_bytes,
            file_name="suggested_allocations.csv",
            mime="text/csv",
            use_container_width=True,
        )
        render_html("<div style='margin-bottom: 24px;'></div>")
    elif alloc_amount == 0 or alloc_amount is None:
        st.info("💡 **Enter an investment amount above** to calculate suggested share quantities, actual amounts, and post-addition portfolio weights.")
        render_html("<div style='margin-bottom: 24px;'></div>")

    # 9. Recommendation Cards (Up to 8 cards, 2 per row)
    st.subheader("Candidate Ideas & Rationale")
    st.caption("Multi-factor rule-based rankings balancing risk-adjusted return (Sharpe), shallow drawdown, diversification gain, sector gaps, and momentum/dividends.")

    col_c1, col_c2 = st.columns(2)
    plan_item_map = {it.ticker: it for it in plan.items} if plan else {}

    for idx, rec in enumerate(rec_result.recommendations):
        target_col = col_c1 if idx % 2 == 0 else col_c2
        p_item = plan_item_map.get(rec.ticker)
        qty_badge = ""
        if p_item and p_item.suggested_quantity > 0:
            qty_badge = f'<div style="font-size: 0.85rem; color: {PRIMARY_TEAL}; font-weight: 600;">Qty: {p_item.suggested_quantity:,} shares ({format_rupee(p_item.actual_amount)})</div>'
        elif p_item and p_item.suggested_quantity == 0:
            qty_badge = f'<div style="font-size: 0.8rem; color: {WARNING_AMBER};">Qty: 0 (Min for 1 share: {format_rupee(p_item.min_amount_for_one_share)})</div>'

        with target_col:
            render_html(
                f"""
                <div class="kpi-card" style="min-height: 220px; border-left: 4px solid {PRIMARY_TEAL}; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 1.2rem; font-weight: 700;">{rec.name}</span>
                            <code style="font-size: 0.95rem; margin-left: 8px;">{rec.ticker}</code>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: {PRIMARY_TEAL};">{format_pct(rec.suggested_weight)}</span>
                            <div style="font-size: 0.75rem; opacity: 0.65;">Basket Weight</div>
                            {qty_badge}
                        </div>
                    </div>
                    <div style="margin: 8px 0;">
                        <span style="font-size: 0.85rem; opacity: 0.8; margin-right: 10px;"><b>Sector:</b> {rec.sector}</span>
                        <span style="font-size: 0.85rem; opacity: 0.8; margin-right: 10px;"><b>Tier:</b> {rec.cap_class.capitalize()} Cap</span>
                        {render_risk_badge(rec.risk_bucket)}
                    </div>
                    <div style="font-size: 0.85rem; margin: 10px 0; line-height: 1.5; padding: 8px 12px; background: {CARD_BG_OVERLAY}; border-radius: 6px;">
                        {rec.explanation}
                    </div>
                    <div style="font-size: 0.75rem; opacity: 0.7; margin-bottom: 8px;">
                        <b>Annual Volatility:</b> {format_pct(rec.annual_volatility)} &bull; 
                        <b>Sharpe:</b> {format_ratio(rec.sharpe_ratio)} &bull; 
                        <b>Max Drawdown:</b> {format_pct(rec.max_drawdown)}
                    </div>
                </div>
                """
            )

            c_act1, c_act2 = target_col.columns([1, 1])
            with c_act1:
                if st.button("🔍 View Stock", key=f"view_{rec.ticker}", use_container_width=True):
                    st.session_state["selected_ticker"] = rec.ticker
                    st.switch_page("pages/2_Stock_Detail.py")
            with c_act2:
                add_qty = p_item.suggested_quantity if (p_item and p_item.suggested_quantity > 0) else 10
                add_price = p_item.last_close if p_item else rec.metrics.get("latest_price", 100.0)
                if st.button("➕ Add to Portfolio", key=f"add_{rec.ticker}", use_container_width=True):
                    if "portfolio_holdings" not in st.session_state:
                        st.session_state["portfolio_holdings"] = []
                    already_in = [h for h in st.session_state["portfolio_holdings"] if h.get("ticker") == rec.ticker]
                    if not already_in:
                        st.session_state["portfolio_holdings"].append({
                            "ticker": rec.ticker,
                            "quantity": add_qty,
                            "avg_buy_price": add_price,
                            "name": rec.name,
                            "sector": rec.sector,
                            "cap_class": rec.cap_class,
                        })
                        if is_logged_in():
                            u = get_current_user()
                            if u:
                                save_user_data(
                                    u,
                                    portfolio_data={
                                        "holdings": st.session_state.get("portfolio_holdings", []),
                                        "funds": st.session_state.get("portfolio_funds", []),
                                    },
                                )
                        st.success(f"Added {rec.ticker} ({add_qty} units @ {format_rupee(add_price)}) to your portfolio holdings!")
                    else:
                        st.info(f"{rec.ticker} is already in your portfolio.")

    render_html("<div style='margin-bottom: 24px;'></div>")

    # 10. Cap-Class Composition of the Suggestions
    st.subheader("Market Cap Composition of Suggested Ideas")
    cap_mix_dict = getattr(rec_result, "cap_class_mix", {}) or {
        "large": sum(1 for r in rec_result.recommendations if r.cap_class == "large"),
        "mid": sum(1 for r in rec_result.recommendations if r.cap_class == "mid"),
        "small": sum(1 for r in rec_result.recommendations if r.cap_class == "small"),
    }
    cap_mix_df = pd.DataFrame([
        {"Cap Tier": "Large", "Stock Count": cap_mix_dict.get("large", 0), "Percentage": cap_mix_dict.get("large", 0) / len(rec_result.recommendations) if rec_result.recommendations else 0.0},
        {"Cap Tier": "Mid", "Stock Count": cap_mix_dict.get("mid", 0), "Percentage": cap_mix_dict.get("mid", 0) / len(rec_result.recommendations) if rec_result.recommendations else 0.0},
        {"Cap Tier": "Small", "Stock Count": cap_mix_dict.get("small", 0), "Percentage": cap_mix_dict.get("small", 0) / len(rec_result.recommendations) if rec_result.recommendations else 0.0},
    ])

    c_cmix1, c_cmix2 = st.columns([1.5, 1])
    with c_cmix1:
        fig_cmix = px.bar(
            cap_mix_df,
            x="Percentage",
            y="Cap Tier",
            orientation="h",
            text="Percentage",
            color="Cap Tier",
            color_discrete_map={"Large": PRIMARY_TEAL, "Mid": ACCENT_BLUE, "Small": WARNING_AMBER},
        )
        fig_cmix.update_traces(texttemplate="%{text:.0%}", textposition="inside")
        fig_cmix.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            showlegend=False,
        )
        apply_theme_to_plotly(fig_cmix)
        st.plotly_chart(fig_cmix, use_container_width=True)

    with c_cmix2:
        render_html(
            f"""
            <div class="kpi-card" style="height: 180px;">
                <div class="kpi-title">Cap Tier Composition</div>
                <div style="margin-top: 10px; line-height: 1.8;">
                    {'<br>'.join(f"&bull; <b>{row['Cap Tier']} Cap:</b> {row['Stock Count']} stocks ({format_pct(row['Percentage'])})" for _, row in cap_mix_df.iterrows())}
                </div>
            </div>
            """
        )

    render_html("<div style='margin-bottom: 24px;'></div>")

    # 11. What-If Before vs After Analysis Panel
    st.subheader("What-If Portfolio Impact Analysis")
    if plan and plan.items and alloc_amount and alloc_amount > 0:
        st.caption(f"Evaluates simulated portfolio risk and diversification shift using the rounded whole share additions totaling {format_rupee(plan.total_actual_amount)}.")
    else:
        st.caption("Evaluates hypothetical portfolio volatility and diversification change if the suggested additions are integrated.")

    wi = rec_result.what_if

    col_wi1, col_wi2 = st.columns(2)

    with col_wi1:
        st.markdown("#### Portfolio Risk & Diversification Shift")
        render_html(
            f"""
            <div class="kpi-card">
                <table style="width: 100%; line-height: 2.2; font-size: 0.95rem;">
                    <tr>
                        <td><b>Metric</b></td>
                        <td style="text-align: right;"><b>Current</b></td>
                        <td style="text-align: right;"><b>Simulated After</b></td>
                        <td style="text-align: right;"><b>Net Change</b></td>
                    </tr>
                    <tr>
                        <td>Annualized Volatility:</td>
                        <td style="text-align: right;">{format_pct(wi.before_volatility)}</td>
                        <td style="text-align: right;">{format_pct(wi.after_volatility)}</td>
                        <td style="text-align: right; color: {PRIMARY_TEAL if wi.volatility_change <= 0 else NEGATIVE_RED};">
                            {format_pct(wi.volatility_change, show_plus=True)}
                        </td>
                    </tr>
                    <tr>
                        <td>Diversification Score (0-100):</td>
                        <td style="text-align: right;">{wi.before_diversification_score:.1f}</td>
                        <td style="text-align: right;">{wi.after_diversification_score:.1f}</td>
                        <td style="text-align: right; color: {PRIMARY_TEAL if wi.diversification_score_change >= 0 else NEGATIVE_RED};">
                            {format_ratio(wi.diversification_score_change)} pts
                        </td>
                    </tr>
                </table>
            </div>
            """
        )

    with col_wi2:
        st.markdown("#### Sector Allocation Impact")
        sec_names = list(set(list(wi.before_sector_weights.keys()) + list(wi.after_sector_weights.keys())))
        if sec_names:
            sec_shift_df = pd.DataFrame({
                "Sector": sec_names,
                "Current": [wi.before_sector_weights.get(s, 0.0) for s in sec_names],
                "Simulated After": [wi.after_sector_weights.get(s, 0.0) for s in sec_names],
            })
            fig_sec_shift = go.Figure()
            fig_sec_shift.add_trace(go.Bar(name="Current", x=sec_shift_df["Sector"], y=sec_shift_df["Current"], marker_color=NEUTRAL_GREY))
            fig_sec_shift.add_trace(go.Bar(name="Simulated After", x=sec_shift_df["Sector"], y=sec_shift_df["Simulated After"], marker_color=PRIMARY_TEAL))
            fig_sec_shift.update_layout(
                barmode="group",
                height=280,
                margin=dict(l=20, r=20, t=10, b=20),
                yaxis=dict(tickformat=".0%"),
                xaxis=dict(tickangle=-30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_theme_to_plotly(fig_sec_shift)
            st.plotly_chart(fig_sec_shift, use_container_width=True)

    render_html("<div style='margin-bottom: 24px;'></div>")

    # 12. Dynamic +/- 1 Risk Level Sensitivity Comparison
    st.subheader("Risk Level Sensitivity (What Would Change)")
    st.caption("Understand how candidate stock ideas shift if your risk profile level were one notch higher or lower.")

    sens = rec_result.level_sensitivity

    c_s_down, c_s_up = st.columns(2)

    with c_s_down:
        st.markdown(f"#### Shift Down: Level {active_level - 1 if active_level > 1 else 1}")
        if sens.level_minus_1 is not None:
            render_html(
                f"""
                <div class="kpi-card" style="border-left: 4px solid {ACCENT_BLUE};">
                    <div style="font-size: 0.95rem; line-height: 1.6;">
                        {sens.level_minus_1_summary}
                    </div>
                </div>
                """
            )
        else:
            st.info("Already at lowest risk level (Level 1: Conservative).")

    with c_s_up:
        st.markdown(f"#### Shift Up: Level {active_level + 1 if active_level < 5 else 5}")
        if sens.level_plus_1 is not None:
            render_html(
                f"""
                <div class="kpi-card" style="border-left: 4px solid {WARNING_AMBER};">
                    <div style="font-size: 0.95rem; line-height: 1.6;">
                        {sens.level_plus_1_summary}
                    </div>
                </div>
                """
            )
        else:
            st.info("Already at highest risk level (Level 5: Aggressive).")

with tab_funds:
    st.markdown("---")
    st.markdown(
        f"### Mutual Fund Ideas for Level {active_level} ({profile_label})\n"
        f"Rule-based fund ideas mapped directly to your risk profile's target asset allocation (equity, debt, and gold). "
        f"Ranks funds within each category by Sharpe ratio, shallow drawdown, and 3-year return consistency."
    )
    st.caption("🏷️ **Direct-growth plans only.** Values reflect the **last reported NAV**, not a live traded price.")
    st.info("ℹ️ **This is how your amount could be split across funds, not a projection.**")

    # Funds Summary Metrics
    c_fsum1, c_fsum2, c_fsum3 = st.columns(3)
    with c_fsum1:
        st.metric("Entered Capital", format_rupee(alloc_amount) if (alloc_amount and alloc_amount > 0) else "None entered")
    with c_fsum2:
        st.metric(
            "Total Suggested Fund Additions",
            f"{format_rupee(fund_rec_result.total_allocated)} ({sip_label})" if (alloc_amount and alloc_amount > 0) else "Target weights only",
            help="Adds up to the user entered capital across active categories",
        )
    with c_fsum3:
        st.metric(
            "Category Baskets",
            f"{len(fund_rec_result.baskets)} categories",
            help="Max 2 fund ideas per category",
        )

    # Allocation Notes (including categories already at/above target)
    for n in fund_rec_result.notes:
        st.info(f"ℹ️ **Allocation Note:** {n}")

    render_html("<div style='margin-bottom: 20px;'></div>")

    if not fund_rec_result.baskets:
        st.warning("No mutual fund additions suggested based on your current portfolio and preferences.")
    else:
        for b_idx, basket in enumerate(fund_rec_result.baskets):
            if alloc_amount and alloc_amount > 0:
                cat_sub_str = f"Target Category Weight: **{format_pct(basket.target_weight)}** &bull; Indicative Basket Amount: **{format_rupee(basket.allocated_amount)}** ({sip_label})"
            else:
                cat_sub_str = f"Target Category Weight: **{format_pct(basket.target_weight)}**"

            st.markdown(f"#### 📂 {basket.category_label}")
            st.caption(cat_sub_str)

            f_col1, f_col2 = st.columns(2)
            for f_idx, f_item in enumerate(basket.funds):
                target_fcol = f_col1 if f_idx % 2 == 0 else f_col2
                with target_fcol:
                    cagr3_str = format_pct(f_item.cagr_3y) if f_item.cagr_3y != 0 else "N/A"
                    cagr5_str = format_pct(f_item.cagr_5y) if f_item.cagr_5y != 0 else "N/A"
                    nav_date_str = format_date(f_item.last_nav_date) if f_item.last_nav_date else "N/A"

                    if alloc_amount and alloc_amount > 0:
                        fund_amt_badge = f'<div style="font-size: 1.15rem; font-weight: 700; color: {PRIMARY_TEAL};">{format_rupee(f_item.indicative_amount)} <span style="font-size: 0.8rem; font-weight: normal;">({sip_label})</span></div>'
                        sub_badge_lbl = f"Indicative Amount ({sip_label})"
                    else:
                        fund_amt_badge = f'<div style="font-size: 1.15rem; font-weight: 700; color: {PRIMARY_TEAL};">{format_pct(f_item.allocated_weight)}</div>'
                        sub_badge_lbl = "Category Weight"

                    fund_outlook = get_fund_risk_outlook(str(f_item.scheme_code))
                    outlook_badge_html = ""
                    if fund_outlook and fund_outlook.get("outlook"):
                        outlook_badge_html = f'&bull; <span style="background: {CARD_BG_OVERLAY}; color: var(--text-color); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Outlook: {fund_outlook["outlook"]}</span>'

                    render_html(
                        f"""
                        <div class="kpi-card" style="min-height: 240px; border-left: 4px solid {ACCENT_BLUE}; margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="max-width: 68%;">
                                    <span style="font-size: 1.15rem; font-weight: 700;">{f_item.name}</span>
                                    <div style="font-size: 0.85rem; opacity: 0.75; margin-top: 4px;">
                                        {f_item.fund_house} &bull; 
                                        <span style="background: {TEAL_BG_OVERLAY}; color: {PRIMARY_TEAL}; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Direct Growth</span>
                                        {outlook_badge_html}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    {fund_amt_badge}
                                    <div style="font-size: 0.75rem; opacity: 0.65;">{sub_badge_lbl}</div>
                                </div>
                            </div>
                            <div style="margin: 10px 0;">
                                <span style="font-size: 0.85rem; opacity: 0.85; margin-right: 12px;"><b>Risk Level:</b> {f_item.risk_level_label}</span>
                                <span style="font-size: 0.85rem; opacity: 0.85; margin-right: 12px;"><b>3Y CAGR:</b> {cagr3_str}</span>
                                <span style="font-size: 0.85rem; opacity: 0.85; margin-right: 12px;"><b>5Y CAGR:</b> {cagr5_str}</span>
                                <span style="font-size: 0.85rem; opacity: 0.85;"><b>Max Drawdown:</b> {format_pct(f_item.max_drawdown)}</span>
                            </div>
                            <div style="font-size: 0.85rem; margin: 10px 0; line-height: 1.5; padding: 8px 12px; background: {CARD_BG_OVERLAY}; border-radius: 6px;">
                                💡 {f_item.explanation}
                            </div>
                            <div style="font-size: 0.75rem; opacity: 0.7; margin-bottom: 10px;">
                                <b>Last NAV:</b> {format_rupee(f_item.last_nav)} as of {nav_date_str} (Last NAV, not a live price) &bull; 
                                <b>Rolling Consistency:</b> {f_item.consistency_score * 100:.0f}% &bull; 
                                <b>Sharpe:</b> {format_ratio(f_item.sharpe_ratio)}
                            </div>
                        </div>
                        """
                    )

                    f_add_amt = f_item.indicative_amount if (alloc_amount and alloc_amount > 0) else 10000.0
                    if st.button("➕ Add to Portfolio Funds", key=f"add_fund_{f_item.scheme_code}", use_container_width=True):
                        if "portfolio_funds" not in st.session_state:
                            st.session_state["portfolio_funds"] = []
                        already_in_f = [f for f in st.session_state["portfolio_funds"] if f.get("scheme_code") == f_item.scheme_code]
                        if not already_in_f:
                            st.session_state["portfolio_funds"].append({
                                "category": f_item.category,
                                "amount": f_add_amt,
                                "name": f_item.name,
                                "scheme_code": f_item.scheme_code,
                            })
                            if is_logged_in():
                                u = get_current_user()
                                if u:
                                    save_user_data(
                                        u,
                                        portfolio_data={
                                            "holdings": st.session_state.get("portfolio_holdings", []),
                                            "funds": st.session_state.get("portfolio_funds", []),
                                        },
                                    )
                            st.success(f"Added {f_item.name} ({format_rupee(f_add_amt)}) to your portfolio funds!")
                        else:
                            st.info(f"{f_item.name} is already in your portfolio funds.")

            render_html("<div style='margin-bottom: 20px;'></div>")

# 13. Scenario Forecast Projection (Feature 4)
rec_target_alloc = {
    "equity": cfg.target_allocations[active_level].equity,
    "debt": cfg.target_allocations[active_level].debt,
    "gold": cfg.target_allocations[active_level].gold,
}
render_projection_panel(
    asset_weights=rec_target_alloc,
    default_amount=float(alloc_amount) if (alloc_amount and alloc_amount > 0) else (10000.0 if is_monthly_sip else 100000.0),
    default_mode=investment_mode,
    default_horizon=5,
    key_prefix="rec_proj",
)

render_disclaimer()
