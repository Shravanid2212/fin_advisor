"""Reusable UI components for Streamlit application.

Adheres to:
- PRD Section 15.1 - 15.3 and AGENTS.md rules.
- Persistent 'Data as of' chip.
- Risk badges show both color AND text (no color-only signals).
- SEBI-style 5-level riskometer gauge.
- Dedented, safe HTML rendering preventing raw markdown code block issues.
- User authentication helpers (require_login, render_user_sidebar).
"""

from __future__ import annotations

import datetime
import textwrap
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    COLOR_WHITE,
    GAUGE_STEPS,
    NEUTRAL_GREY,
    PRIMARY_TEAL,
    RISK_COLORS,
    RISK_LABELS,
    TEAL_BG_OVERLAY,
    TRANSPARENT_BG,
    WARNING_AMBER,
    apply_theme_to_plotly,
    format_date,
    format_rupee,
    get_gauge_pointer_color,
    get_plotly_template,
    get_theme_mode,
    inject_theme_css,
    set_theme_mode,
)


def render_html(html_str: str) -> None:
    """Render HTML safely without Markdown parser treating indented lines as code blocks."""
    dedented = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(dedented)
    else:
        st.markdown(dedented, unsafe_allow_html=True)


def is_logged_in() -> bool:
    """Check if a user is currently authenticated in session state."""
    return bool(st.session_state.get("logged_in", False))


def get_current_user() -> str | None:
    """Return the currently authenticated username or None."""
    return st.session_state.get("username")


def require_login() -> None:
    """Enforce login requirement on protected pages."""
    if not is_logged_in():
        inject_theme_css()
        st.warning("🔐 **Authentication Required**  \nPlease sign in or register an account on the Home page to access this page.")
        if st.button("⬅️ Go to Sign In / Sign Up", type="primary"):
            st.switch_page("Home.py")
        st.stop()


def render_user_sidebar() -> None:
    """Render user profile status, theme selector, save, and logout controls in the sidebar."""
    from src.accounts.store import save_user_data

    # Display Theme Toggle (Light / Dark / System)
    theme_options = ["System", "Light", "Dark"]
    current_theme = get_theme_mode()
    if current_theme not in theme_options:
        current_theme = "System"
    theme_idx = theme_options.index(current_theme)

    selected_theme = st.sidebar.selectbox(
        "🎨 Display Theme",
        options=theme_options,
        index=theme_idx,
        key="app_theme_mode_selector",
        help="Select display appearance: Light, Dark, or System preference.",
    )
    if selected_theme != current_theme:
        set_theme_mode(selected_theme)
        if is_logged_in():
            user = get_current_user()
            if user:
                save_user_data(user, settings_data={"theme": selected_theme})
        st.rerun()

    if is_logged_in():
        user = get_current_user() or "User"
        st.sidebar.markdown(f"**👤 Signed in as:** `{user}`")

        c_save, c_logout = st.sidebar.columns(2)
        with c_save:
            if st.button("💾 Save", use_container_width=True, help="Save current profile and portfolio to your account"):
                prof_data = None
                if "user_profile" in st.session_state and st.session_state["user_profile"] is not None:
                    prof = st.session_state["user_profile"]
                    prof_data = {
                        "final_level": getattr(prof, "final_level", 3),
                        "label": getattr(prof, "label", "Moderate"),
                        "capacity_score": getattr(prof, "capacity_score", 50.0),
                        "capacity_level": getattr(prof, "capacity_level", 3),
                        "tolerance_score": getattr(prof, "tolerance_score", 50.0),
                        "tolerance_level": getattr(prof, "tolerance_level", 3),
                        "preliminary_level": getattr(prof, "preliminary_level", 3),
                        "limiting_factor": getattr(prof, "limiting_factor", "capacity"),
                        "guardrail_messages": getattr(prof, "guardrail_messages", []),
                        "preferences": getattr(prof, "preferences", {}),
                        "raw_scores": getattr(prof, "raw_scores", {}),
                        "answers": st.session_state.get("quiz_answers", {}),
                    }
                port_data = {
                    "holdings": st.session_state.get("portfolio_holdings", []),
                    "funds": st.session_state.get("portfolio_funds", []),
                }
                save_user_data(user, profile_data=prof_data, portfolio_data=port_data, settings_data={"theme": selected_theme})
                st.sidebar.success("Saved!")
        with c_logout:
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                st.session_state["user_profile"] = None
                st.session_state["profile_completed"] = False
                st.session_state["portfolio_holdings"] = []
                st.session_state["portfolio_funds"] = []
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_step"] = 1
                st.switch_page("Home.py")


def is_data_stale(data_date: str, max_trading_days: int = 5) -> tuple[bool, int]:
    """Check if the latest data date is more than max_trading_days business days old."""
    if not data_date or data_date == "N/A":
        return False, 0
    try:
        dt = datetime.date.fromisoformat(str(data_date).strip())
        today = datetime.date.today()
        if dt < today:
            bus_days = int(np.busday_count(dt, today))
            if bus_days > max_trading_days:
                return True, bus_days
            return False, bus_days
    except Exception:
        pass
    return False, 0


def render_friendly_error(
    title: str,
    message: str,
    resolution_steps: list[str] | None = None,
) -> None:
    """Render a user-friendly error callout with actionable resolution steps instead of tracebacks."""
    inject_theme_css()
    st.error(f"### ⚠️ {title}\n\n{message}")
    if resolution_steps:
        st.markdown("**Recommended Resolution:**")
        for step in resolution_steps:
            st.markdown(f"- {step}")


def render_page_header(page_title: str, data_date: str) -> None:
    """Render top bar with page title, CSS injection, Data as of chip, and stale data banner."""
    inject_theme_css()
    render_user_sidebar()

    col_title, col_chip = st.columns([3, 1])
    with col_title:
        st.title(page_title)
    with col_chip:
        render_html(
            f"""
            <div style="text-align: right; margin-top: 15px;">
                <span class="data-as-of-chip">📅 Data as of {format_date(data_date)}</span>
            </div>
            """
        )

    # 1. Uninitialized or missing database callout
    if not data_date or data_date == "N/A":
        st.error(
            "⚠️ **Database Unavailable or Uninitialized**  \n"
            "The market analytics database (`data/market.duckdb`) is currently uninitialized or missing required tables.\n\n"
            "**To initialize the database, execute:**\n"
            "```powershell\n"
            "python run.py data\n"
            "# or: python -m src.ingest.run_pipeline\n"
            "```"
        )
        return

    # 2. Stale data warning banner (if > 5 trading days old)
    stale, days_old = is_data_stale(data_date, max_trading_days=5)
    if stale:
        st.warning(
            f"⚠️ **Stale Data Notice**: The market dataset reflects prices as of **{format_date(data_date)}** "
            f"({days_old} trading days old). The automated data ingestion pipeline runs on market trading days. "
            f"To update prices to the latest close, run `python run.py data` or execute the CI refresh workflow."
        )


def render_kpi_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
    help_text: str | None = None,
) -> None:
    """Render a styled KPI card."""
    delta_html = ""
    if delta:
        delta_class = f"kpi-delta-{delta_type}"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    help_attr = f'title="{help_text}"' if help_text else ""
    render_html(
        f"""
        <div class="kpi-card" {help_attr}>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """
    )


def render_risk_badge(bucket: int | str, label: str | None = None) -> str:
    """Generate HTML for a risk badge with both color and text."""
    try:
        b_int = int(bucket)
    except (ValueError, TypeError):
        b_int = 3

    color = RISK_COLORS.get(b_int, "#828282")
    text_label = label or RISK_LABELS.get(b_int, f"Level {b_int}")
    return (
        f'<span class="risk-badge" style="background-color: {color}; color: {COLOR_WHITE};">'
        f'Risk Level {b_int}: {text_label}</span>'
    )


def create_riskometer_fig(level: int, title: str = "Riskometer") -> go.Figure:
    """Create a SEBI-style 5-level riskometer gauge."""
    try:
        b_int = int(level)
    except (ValueError, TypeError):
        b_int = 3

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=b_int,
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{RISK_LABELS.get(b_int, '')}</span>"},
            gauge={
                "axis": {"range": [1, 5], "tickvals": [1, 2, 3, 4, 5], "ticktext": ["Low", "Mod.Low", "Mod", "Mod.High", "High"]},
                "bar": {"color": RISK_COLORS.get(b_int, PRIMARY_TEAL), "thickness": 0.3},
                "steps": GAUGE_STEPS,
                "threshold": {
                    "line": {"color": get_gauge_pointer_color(), "width": 4},
                    "thickness": 0.75,
                    "value": b_int,
                },
            },
        )
    )
    fig.update_layout(
        template=get_plotly_template(),
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=TRANSPARENT_BG,
        plot_bgcolor=TRANSPARENT_BG,
    )
    apply_theme_to_plotly(fig)
    return fig


def render_empty_state(title: str, message: str, action_label: str | None = None) -> bool:
    """Render an empty state callout with optional action button."""
    render_html(
        f"""
        <div class="empty-state-box">
            <h3>{title}</h3>
            <p style="opacity: 0.8;">{message}</p>
        </div>
        """
    )
    if action_label:
        return st.button(action_label, key=f"btn_empty_{title}")
    return False


def render_disclaimer() -> None:
    """No-op following removal of educational disclaimer per Phase B6 requirements."""
    pass


def render_projection_panel(
    asset_weights: dict[str, float],
    default_amount: float = 10000.0,
    default_mode: str = "Monthly SIP",
    default_horizon: int = 5,
    key_prefix: str = "proj",
) -> None:
    """Render interactive scenario forecast panel with block-bootstrap Monte Carlo fan chart.

    Adheres to:
    - Feature flag check: silently returns if ml.projection.enabled is False.
    - Plain investor wording: 'Weaker case', 'Typical case', 'Stronger case'.
    - No statistical jargon, no p-values, no model names.
    - Mandatory disclosure sentence directly below fan chart.
    """
    from src.config import load_config
    cfg = load_config()
    ml_cfg = getattr(cfg, "ml", None)
    if not ml_cfg or not getattr(ml_cfg.projection, "enabled", False):
        return

    from src.projection.simulate import simulate_portfolio_projection

    render_html("<div style='margin-bottom: 24px;'></div>")
    st.subheader("What could this grow to?")
    st.caption("Scenario projections based on how this asset mix behaved historically over various market cycles.")

    c_mode, c_amt, c_horiz = st.columns([1.5, 1.5, 2])
    with c_mode:
        mode_idx = 1 if default_mode == "Monthly SIP" else 0
        inv_mode = st.selectbox(
            "Contribution Type",
            options=["One-time investment", "Monthly SIP"],
            index=mode_idx,
            key=f"{key_prefix}_mode",
            help="Choose between a single lump-sum investment or a recurring monthly systematic investment plan.",
        )
    with c_amt:
        is_sip = (inv_mode == "Monthly SIP")
        def_amt = 10000 if is_sip else 100000
        step_val = 1000 if is_sip else 10000
        min_val = 500 if is_sip else 5000
        amt = st.number_input(
            f"Amount (₹) ({'per month' if is_sip else 'starting lump sum'})",
            min_value=min_val,
            value=int(default_amount) if (default_amount and default_amount >= min_val) else def_amt,
            step=step_val,
            key=f"{key_prefix}_amt_{inv_mode}",
        )
    with c_horiz:
        horizon = st.slider(
            "Investment Horizon (Years)",
            min_value=1,
            max_value=15,
            value=int(default_horizon) if 1 <= default_horizon <= 15 else 5,
            step=1,
            key=f"{key_prefix}_horizon",
            help="Number of years you plan to keep this capital invested.",
        )

    try:
        res = simulate_portfolio_projection(
            asset_weights=asset_weights,
            amount=float(amt),
            horizon_years=int(horizon),
            investment_mode=inv_mode,
            random_seed=cfg.random_seed,
        )
    except Exception as exc:
        st.warning(f"Could not calculate projection: {exc}")
        return

    freq_label = "per month" if inv_mode == "Monthly SIP" else "one-time"

    # 4 Key Outcome Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Money Put In", format_rupee(res.total_invested), help=f"Total capital contributed ({freq_label})")
    with m2:
        st.metric("Weaker Case (10th percentile)", format_rupee(res.final_p10), help="Challenging market conditions")
    with m3:
        st.metric("Typical Case (50th percentile)", format_rupee(res.final_p50), help="Median historical trajectory")
    with m4:
        st.metric("Stronger Case (90th percentile)", format_rupee(res.final_p90), help="Favorable market conditions")

    # Plotly Fan Chart
    t_df = res.timeline_df
    fig_fan = go.Figure()

    # Upper boundary: Stronger case (90th percentile)
    fig_fan.add_trace(
        go.Scatter(
            x=t_df["year"],
            y=t_df["p90"],
            mode="lines",
            line=dict(width=1.5, color=PRIMARY_TEAL, dash="dash"),
            name="Stronger case (90th %ile)",
            hovertemplate="Year %{x:.1f}: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    # Shaded band fill down to Weaker case (10th percentile)
    fig_fan.add_trace(
        go.Scatter(
            x=t_df["year"],
            y=t_df["p10"],
            mode="lines",
            line=dict(width=1.5, color=WARNING_AMBER, dash="dash"),
            fill="tonexty",
            fillcolor=TEAL_BG_OVERLAY,
            name="Weaker case (10th %ile)",
            hovertemplate="Year %{x:.1f}: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    # Middle trajectory: Typical case (50th percentile)
    fig_fan.add_trace(
        go.Scatter(
            x=t_df["year"],
            y=t_df["p50"],
            mode="lines",
            line=dict(width=2.5, color=PRIMARY_TEAL),
            name="Typical case (50th %ile)",
            hovertemplate="Year %{x:.1f}: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    # Baseline: Total invested
    fig_fan.add_trace(
        go.Scatter(
            x=t_df["year"],
            y=t_df["total_invested"],
            mode="lines",
            line=dict(width=1.8, color=NEUTRAL_GREY, dash="dot"),
            name="Total money put in",
            hovertemplate="Year %{x:.1f}: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    fig_fan.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Years elapsed", dtick=1 if horizon <= 5 else 2),
        yaxis=dict(title="Estimated Value (₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor=TRANSPARENT_BG,
        plot_bgcolor=TRANSPARENT_BG,
    )
    apply_theme_to_plotly(fig_fan)
    st.plotly_chart(fig_fan, use_container_width=True)

    # Mandatory Disclosure Sentence directly under the chart
    st.caption("Projections are based on historical return patterns reshuffled at random in 6-month blocks. They do not predict the future and do not guarantee returns.")
    if res.warning:
        st.info(f"ℹ️ {res.warning}")

