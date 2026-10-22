"""My Profile (Investor Risk Profiling Questionnaire) page.

Adheres strictly to:
- Educational decision-support tool; no buy/sell advice wording.
- 3-step questionnaire (Capacity Q1-Q6, Tolerance Q7-Q10, Preferences Q11-Q14).
- Safe session state handling: refreshing mid-quiz retains answers.
- Result screen: riskometer, capacity/tolerance scores, limiting factor, guardrail alerts, target asset allocation donut.
- User data saved securely to user account in data/users.db.
- Pure HTML rendering with render_html (no raw div text in UI).
- Requires user authentication.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

from app.theme import (
    ACCENT_BLUE,
    CARD_BORDER_OVERLAY,
    PRIMARY_TEAL,
    WARNING_AMBER,
    apply_theme_to_plotly,
)
from app.components.ui import (
    create_riskometer_fig,
    get_current_user,
    is_logged_in,
    render_disclaimer,
    render_html,
    render_page_header,
    require_login,
)
from src.accounts.store import save_user_data
from src.app_data import check_db_health
from src.profile.scoring import (
    calculate_profile,
    load_questions,
)

st.set_page_config(
    page_title="My Risk Profile | Financial Analytics",
    page_icon="📋",
    layout="wide",
)

# 1. Require Authentication
require_login()

# 2. Health check
health = check_db_health()
data_date = health.get("data_as_of", "N/A")
render_page_header("My Risk Profile", data_date)

# 3. Session State Initialization
if "quiz_step" not in st.session_state:
    st.session_state["quiz_step"] = 1

if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}

if "profile_completed" not in st.session_state:
    st.session_state["profile_completed"] = False

# Load questions
questions = load_questions()

# ---------------------------------------------------------------------------
# View A: Result Screen (Profile Completed)
# ---------------------------------------------------------------------------
if st.session_state["profile_completed"] and "user_profile" in st.session_state:
    prof = st.session_state["user_profile"]

    st.subheader(f"Your Assessed Profile: Level {prof.final_level} — {prof.label}")
    st.caption("Computed from your verified risk capacity, risk tolerance, and safety guardrails.")

    # Guardrails callout
    if prof.guardrail_messages:
        for g_msg in prof.guardrail_messages:
            st.warning(f"🛡️ **Safety Guardrail Applied:** {g_msg}")

    c_gauge, c_scores = st.columns([1.5, 2])

    with c_gauge:
        fig_g = create_riskometer_fig(prof.final_level, title="Risk Profile Level")
        st.plotly_chart(fig_g, use_container_width=True)

    with c_scores:
        st.markdown("#### Score Breakdown")
        render_html(
            f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <b>Risk Capacity (Objective Ability):</b>
                    <span><b>{prof.capacity_score:.0f}/100</b> (Level {prof.capacity_level})</span>
                </div>
                <div style="background: {CARD_BORDER_OVERLAY}; border-radius: 6px; height: 10px; width: 100%; margin-bottom: 16px;">
                    <div style="background: {ACCENT_BLUE}; height: 10px; border-radius: 6px; width: {prof.capacity_score}%;"></div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <b>Risk Tolerance (Psychological Comfort):</b>
                    <span><b>{prof.tolerance_score:.0f}/100</b> (Level {prof.tolerance_level})</span>
                </div>
                <div style="background: {CARD_BORDER_OVERLAY}; border-radius: 6px; height: 10px; width: 100%; margin-bottom: 12px;">
                    <div style="background: {WARNING_AMBER}; height: 10px; border-radius: 6px; width: {prof.tolerance_score}%;"></div>
                </div>

                <div style="font-size: 0.9rem; margin-top: 12px; padding-top: 8px; border-top: 1px solid {CARD_BORDER_OVERLAY};">
                    <b>Limiting Factor:</b> Your risk level is governed by your <b>{prof.limiting_factor}</b> (the lower-of-two conservative rule).
                </div>
            </div>
            """
        )

    st.markdown("---")
    st.subheader("Target Asset Allocation")
    st.caption("Strategic asset mix aligned with your risk tolerance and investment capacity.")

    c_donut, c_split = st.columns(2)
    alloc = prof.target_allocation

    with c_donut:
        alloc_df = pd.DataFrame({
            "Asset Class": ["Equity", "Debt", "Gold"],
            "Target Weight": [alloc.equity, alloc.debt, alloc.gold],
        })
        fig_donut = px.pie(
            alloc_df,
            values="Target Weight",
            names="Asset Class",
            hole=0.55,
            color="Asset Class",
            color_discrete_map={"Equity": PRIMARY_TEAL, "Debt": ACCENT_BLUE, "Gold": WARNING_AMBER},
        )
        fig_donut.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        apply_theme_to_plotly(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_split:
        st.markdown("#### Suggested Equity Sub-Split")
        st.caption("Distribution of equity allocation across market capitalization tiers:")
        split = alloc.equity_split
        split_df = pd.DataFrame({
            "Cap Tier": ["Large Cap (Top 100)", "Mid Cap (101-250)", "Small Cap (251+)"],
            "Equity Share": [split.large, split.mid, split.small],
        })
        fig_bar = px.bar(
            split_df,
            x="Equity Share",
            y="Cap Tier",
            orientation="h",
            text="Equity Share",
            color="Cap Tier",
            color_discrete_sequence=[PRIMARY_TEAL, ACCENT_BLUE, WARNING_AMBER],
        )
        fig_bar.update_traces(texttemplate="%{text:.0%}", textposition="inside")
        fig_bar.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            showlegend=False,
        )
        apply_theme_to_plotly(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Actions
    c_btn1, c_btn2, _ = st.columns([1.5, 2, 3])
    with c_btn1:
        if st.button("🔄 Retake Questionnaire", use_container_width=True):
            st.session_state["profile_completed"] = False
            st.session_state["quiz_step"] = 1
            st.rerun()

    with c_btn2:
        if st.button("💼 Continue to My Portfolio", type="primary", use_container_width=True):
            st.switch_page("pages/5_My_Portfolio.py")

    render_disclaimer()
    st.stop()


# ---------------------------------------------------------------------------
# View B: Questionnaire Stepper (Steps 1, 2, 3)
# ---------------------------------------------------------------------------
step = st.session_state["quiz_step"]

step_names = {
    1: "Section A: Risk Capacity (Questions 1 to 6)",
    2: "Section B: Risk Tolerance (Questions 7 to 10)",
    3: "Section C: Investment Preferences (Questions 11 to 15)",
}

progress_vals = {1: 0.33, 2: 0.66, 3: 1.0}

st.progress(progress_vals.get(step, 0.33), text=f"Step {step} of 3 — {step_names.get(step, '')}")
render_html("<div style='margin-bottom: 20px;'></div>")

# Filter questions for active step
cat_map = {1: "capacity", 2: "tolerance", 3: "preference"}
target_cat = cat_map.get(step, "capacity")
step_questions = [q for q in questions if q.category == target_cat]

# Render question cards
for q in step_questions:
    render_html(
        f"""
        <div class="kpi-card" style="margin-bottom: 16px;">
            <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                Q{q.number}. {q.prompt}
            </div>
        </div>
        """
    )

    opt_texts = [opt.text for opt in q.options]
    saved_val = st.session_state["quiz_answers"].get(q.id)

    if q.multi_select:
        # Mutual exclusivity: a sector chosen in one must disappear from the other
        filtered_options = list(opt_texts)
        if q.id == "q13":
            avoided = st.session_state["quiz_answers"].get("q14", [])
            filtered_options = [opt for opt in opt_texts if opt not in avoided]
        elif q.id == "q14":
            favoured = st.session_state["quiz_answers"].get("q13", [])
            filtered_options = [opt for opt in opt_texts if opt not in favoured]

        default_ms = [v for v in (saved_val if isinstance(saved_val, list) else []) if v in filtered_options]
        chosen_ms = st.multiselect(
            label=f"Options for Q{q.number}",
            options=filtered_options,
            default=default_ms,
            key=f"input_{q.id}",
            label_visibility="collapsed",
        )
        st.session_state["quiz_answers"][q.id] = chosen_ms
    else:
        default_idx = opt_texts.index(saved_val) if saved_val in opt_texts else 0
        chosen_radio = st.radio(
            label=f"Options for Q{q.number}",
            options=opt_texts,
            index=default_idx,
            key=f"input_{q.id}",
            label_visibility="collapsed",
        )
        st.session_state["quiz_answers"][q.id] = chosen_radio

    render_html("<div style='margin-bottom: 12px;'></div>")

# Navigation Buttons
c_prev, c_next, _ = st.columns([1, 1.5, 3])

with c_prev:
    if step > 1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state["quiz_step"] -= 1
            st.rerun()

with c_next:
    if step < 3:
        if st.button("Next ➡️", type="primary", use_container_width=True):
            st.session_state["quiz_step"] += 1
            st.rerun()
    else:
        if st.button("📊 Calculate My Risk Profile", type="primary", use_container_width=True):
            try:
                prof_result = calculate_profile(
                    answers=st.session_state["quiz_answers"],
                    questions=questions,
                )
                st.session_state["user_profile"] = prof_result
                st.session_state["profile_completed"] = True

                # Persist to database if logged in
                if is_logged_in():
                    user = get_current_user()
                    if user:
                        prof_data = {
                            "final_level": prof_result.final_level,
                            "label": prof_result.label,
                            "capacity_score": prof_result.capacity_score,
                            "capacity_level": prof_result.capacity_level,
                            "tolerance_score": prof_result.tolerance_score,
                            "tolerance_level": prof_result.tolerance_level,
                            "preliminary_level": prof_result.preliminary_level,
                            "limiting_factor": prof_result.limiting_factor,
                            "guardrail_messages": prof_result.guardrail_messages,
                            "preferences": prof_result.preferences,
                            "raw_scores": prof_result.raw_scores,
                            "answers": st.session_state["quiz_answers"],
                        }
                        save_user_data(user, profile_data=prof_data)

                st.rerun()
            except Exception as e:
                st.error(f"Error calculating profile: {e}")

render_disclaimer()
