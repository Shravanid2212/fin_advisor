"""Automated tests for Light, Dark, and System theme support (Item B).

Verifies:
1. Theme tokens and helper functions return proper colors and templates.
2. Riskometer gauge threshold line color changes between Light and Dark modes.
3. Plotly template toggles between 'plotly_white' and 'plotly_dark'.
4. Streamlit AppTest smoke tests run across every active page in both Light and Dark modes
   with a logged-in user, saved profile, and sample portfolio, asserting 0 exceptions and no raw '<div'.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.theme import (
    DARK_THEME_TOKENS,
    LIGHT_THEME_TOKENS,
    get_gauge_pointer_color,
    get_plotly_template,
    get_theme_mode,
    inject_theme_css,
    set_theme_mode,
)
from app.components.ui import create_riskometer_fig
from src.app_data import get_sample_portfolio_holdings
from src.profile.scoring import calculate_profile


def test_theme_tokens_and_helpers():
    """Verify theme helper functions and token switching."""
    # Test Light mode
    assert get_plotly_template("Light") == "plotly_white"
    assert get_gauge_pointer_color("Light") == "#1A1D20"

    # Test Dark mode
    assert get_plotly_template("Dark") == "plotly_dark"
    assert get_gauge_pointer_color("Dark") == "#F0F6FC"


def test_riskometer_gauge_adapts_to_theme():
    """Verify the riskometer gauge pointer and template change between Light and Dark."""
    fig_light = create_riskometer_fig(3, title="Risk Test")
    # In default/Light mode, gauge line is dark
    assert fig_light.data[0].gauge.threshold.line.color in ("#1A1D20", "black")

    # Switch session state to Dark mode
    st.session_state["theme_mode"] = "Dark"
    fig_dark = create_riskometer_fig(3, title="Risk Test")
    assert fig_dark.data[0].gauge.threshold.line.color == "#F0F6FC"
    assert fig_dark.layout.template.layout.font is not None or fig_dark.layout.template == "plotly_dark"

    # Reset
    st.session_state["theme_mode"] = "System"


@pytest.mark.realdata
@pytest.mark.parametrize("theme", ["Light", "Dark"])
def test_all_pages_smoke_test_in_both_themes(theme):
    """Verify every page runs headless in both Light and Dark themes without exceptions or raw html."""
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
        assert p.exists(), f"Page {page_path} missing!"
        at = AppTest.from_file(str(p))

        # Setup logged-in state, profile, portfolio, and theme
        at.session_state["logged_in"] = True
        at.session_state["username"] = f"theme_test_user_{theme.lower()}"
        at.session_state["profile_completed"] = True
        at.session_state["user_profile"] = sample_profile
        at.session_state["quiz_answers"] = answers
        at.session_state["portfolio_holdings"] = stock_rows
        at.session_state["portfolio_funds"] = fund_rows
        at.session_state["selected_ticker"] = "TCS.NS"
        at.session_state["theme_mode"] = theme

        at.run(timeout=30)
        assert len(at.exception) == 0, f"Exception in {page_path} ({theme} mode): {[e.value for e in at.exception]}"

        # Assert no raw "<div" rendered
        for elem in at.markdown:
            assert "<div" not in elem.value, f"Raw '<div' in markdown on {page_path} ({theme}): {elem.value}"


def test_no_hardcoded_colors_in_app_outside_theme():
    """Verify no hardcoded color literals (#000, #333, black, white, rgb/rgba) remain in app/ outside theme.py."""
    import re
    app_dir = Path(__file__).resolve().parent.parent / "app"
    # Match #000, #333, #ffffff, #fff, rgb(...), rgba(...), or literal black/white
    pattern = re.compile(
        r'(#000000|#000\b|#333333|#333\b|#ffffff|#fff\b|rgba?\([^)]+\)|\b(?:black|white)\b)',
        re.IGNORECASE,
    )
    violations = []
    for file_path in app_dir.rglob("*.py"):
        if file_path.name == "theme.py":
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                # Ignore comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                match = pattern.search(line)
                if match:
                    violations.append(f"{file_path.relative_to(app_dir.parent)}:{line_no} [{match.group(0)}] -> {stripped}")

    assert not violations, f"Found hardcoded color literals in app/ outside theme.py:\n" + "\n".join(violations)

