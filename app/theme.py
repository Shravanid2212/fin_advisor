"""Theme configuration, visual tokens, and formatting pure functions for Streamlit.

Adheres to:
- AGENTS.md Rule 9: Dates are DD MMM YYYY in UI, Rupee sign with lakh/crore grouping.
- PRD Section 15: Primary teal (#00B386), negative red (#EB5757), warning amber (#F2994A).
- Supports both Light and Dark modes.
"""

from __future__ import annotations

import datetime
import math
from typing import Any
import streamlit as st

# Color Tokens
PRIMARY_TEAL = "#00B386"
NEGATIVE_RED = "#EB5757"
WARNING_AMBER = "#F2994A"
NEUTRAL_GREY = "#828282"
MUTED_TEXT = "#6C757D"
ACCENT_BLUE = "#2F80ED"

# Semantic Color Constants (used across app to eliminate raw color literals)
COLOR_WHITE = "#FFFFFF"
COLOR_BLACK = "#000000"
COLOR_MUTED_DARK = "#333333"
COLOR_MUTED_LIGHT = "#F8F9FA"
TRANSPARENT_BG = "rgba(0, 0, 0, 0)"
CARD_BG_OVERLAY = "rgba(128, 128, 128, 0.1)"
CARD_BORDER_OVERLAY = "rgba(128, 128, 128, 0.2)"
TEAL_BG_OVERLAY = "rgba(0, 179, 134, 0.15)"
BLUE_BG_OVERLAY = "rgba(45, 156, 219, 0.15)"
RED_BG_OVERLAY = "rgba(235, 87, 87, 0.15)"
AMBER_BG_OVERLAY = "rgba(242, 153, 74, 0.15)"
GREY_BAR_COLOR = "rgba(128, 128, 128, 0.45)"
BANNER_GRADIENT = "linear-gradient(135deg, rgba(0, 179, 134, 0.08) 0%, rgba(47, 128, 237, 0.08) 100%)"
BANNER_BORDER = "1px solid rgba(0, 179, 134, 0.25)"

# Gauge step color ranges
GAUGE_STEPS = [
    {"range": [1, 1.8], "color": "rgba(33, 150, 83, 0.25)"},
    {"range": [1.8, 2.6], "color": "rgba(45, 156, 219, 0.25)"},
    {"range": [2.6, 3.4], "color": "rgba(242, 153, 74, 0.25)"},
    {"range": [3.4, 4.2], "color": "rgba(242, 201, 76, 0.25)"},
    {"range": [4.2, 5], "color": "rgba(235, 87, 87, 0.25)"},
]

# Semantic Theme Tokens (chosen for contrast readability in both modes)
LIGHT_THEME_TOKENS = {
    "bg": "#FFFFFF",
    "sidebar_bg": "#F8F9FA",
    "surface": "#F8F9FA",
    "card_bg": "#FFFFFF",
    "text": "#1A1D20",
    "text_muted": "#6C757D",
    "border": "rgba(0, 0, 0, 0.12)",
    "plotly_template": "plotly_white",
    "gauge_pointer": "#1A1D20",
}

DARK_THEME_TOKENS = {
    "bg": "#0E1117",
    "sidebar_bg": "#161B22",
    "surface": "#161B22",
    "card_bg": "#1E232A",
    "text": "#F0F6FC",
    "text_muted": "#8B949E",
    "border": "rgba(255, 255, 255, 0.15)",
    "plotly_template": "plotly_dark",
    "gauge_pointer": "#F0F6FC",
}

# Risk Bucket Colors (1 to 5)
RISK_COLORS = {
    1: "#27AE60",  # Low - Green
    2: "#2D9CDB",  # Moderately Low - Blue
    3: "#F2994A",  # Moderate - Amber
    4: "#E26D5C",  # Moderately High - Dark Amber/Coral
    5: "#EB5757",  # High - Red
}

RISK_LABELS = {
    1: "Low",
    2: "Moderately Low",
    3: "Moderate",
    4: "Moderately High",
    5: "High",
}


def format_rupee(amount: float | int | None) -> str:
    """Format an amount in Indian Rupees using Lakh / Crore grouping.

    Examples:
        format_rupee(15234500) -> "₹1.52 Cr"
        format_rupee(450000)   -> "₹4.50 L"
        format_rupee(1250.50)  -> "₹1,250.50"
        format_rupee(None)     -> "₹0.00"
    """
    if amount is None or (isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount))):
        return "₹0.00"

    sign = "-" if amount < 0 else ""
    abs_amt = abs(amount)

    if abs_amt >= 10_000_000:
        cr = abs_amt / 10_000_000
        return f"{sign}₹{cr:.2f} Cr"
    elif abs_amt >= 100_000:
        lakh = abs_amt / 100_000
        return f"{sign}₹{lakh:.2f} L"
    else:
        # Standard formatting with 2 decimals
        return f"{sign}₹{abs_amt:,.2f}"


def format_lakh_crore(amount: float | int | None) -> str:
    """Format numerical value without rupee symbol in Lakh / Crore grouping."""
    if amount is None or (isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount))):
        return "0.00"

    sign = "-" if amount < 0 else ""
    abs_amt = abs(amount)

    if abs_amt >= 10_000_000:
        return f"{sign}{abs_amt / 10_000_000:.2f} Cr"
    elif abs_amt >= 100_000:
        return f"{sign}{abs_amt / 100_000:.2f} L"
    else:
        return f"{sign}{abs_amt:,.2f}"


def format_date(date_val: str | datetime.date | datetime.datetime | None) -> str:
    """Format an ISO date string (YYYY-MM-DD) or date object to 'DD MMM YYYY'.

    Examples:
        format_date("2026-10-19") -> "19 Oct 2026"
        format_date(None)         -> "N/A"
    """
    if not date_val or str(date_val).strip() in ("", "None", "nan", "N/A"):
        return "N/A"

    if isinstance(date_val, (datetime.date, datetime.datetime)):
        return date_val.strftime("%d %b %Y")

    s = str(date_val).strip()[:10]
    try:
        dt = datetime.datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except ValueError:
        return s


def format_pct(val: float | None, decimals: int = 2, show_plus: bool = False) -> str:
    """Format a decimal float as percentage (e.g. 0.125 -> '12.50%')."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "0.00%"
    sign = "+" if show_plus and val > 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def format_ratio(val: float | None, decimals: int = 2) -> str:
    """Format a ratio float (e.g. 1.234 -> '1.23')."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "0.00"
    return f"{val:.{decimals}f}"


def format_dividend_yield(val: float | None, decimals: int = 2) -> str:
    """Format dividend yield percentage stored in percentage points (e.g. 1.95 -> '1.95%', 0.47 -> '0.47%').

    Handles nulls and guarantees values reflect realistic Indian equity dividend yields (0% to 20%).
    """
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "0.00%"
    return f"{float(val):.{decimals}f}%"


def get_theme_mode() -> str:
    """Retrieve the active theme mode ('Light', 'Dark', or 'System') from session state."""
    return st.session_state.get("theme_mode", "System")


def set_theme_mode(mode: str) -> None:
    """Set the active theme mode."""
    if mode in ("Light", "Dark", "System"):
        st.session_state["theme_mode"] = mode


def get_plotly_template(theme: str | None = None) -> str:
    """Get the appropriate Plotly chart template for the active theme mode."""
    t = theme or get_theme_mode()
    if t == "Dark":
        return DARK_THEME_TOKENS["plotly_template"]
    return LIGHT_THEME_TOKENS["plotly_template"]


def get_gauge_pointer_color(theme: str | None = None) -> str:
    """Get high-contrast gauge pointer color for the riskometer."""
    t = theme or get_theme_mode()
    if t == "Dark":
        return DARK_THEME_TOKENS["gauge_pointer"]
    return LIGHT_THEME_TOKENS["gauge_pointer"]


def apply_theme_to_plotly(fig: Any, theme: str | None = None) -> Any:
    """Apply consistent dark/light theme styling to any Plotly figure.

    Sets template, font color, legend text/title colors, axis line/grid/tick colors,
    and hoverlabel font and background colors based on active theme mode.
    """
    t = theme or get_theme_mode()
    is_dark = (t == "Dark")
    tokens = DARK_THEME_TOKENS if is_dark else LIGHT_THEME_TOKENS

    font_color = tokens["text"]
    grid_color = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(0, 0, 0, 0.08)"
    axis_color = tokens["text_muted"]
    card_bg = tokens["card_bg"]

    fig.update_layout(
        template=tokens["plotly_template"],
        paper_bgcolor=TRANSPARENT_BG,
        plot_bgcolor=TRANSPARENT_BG,
        font=dict(color=font_color),
        legend=dict(
            font=dict(color=font_color),
            title=dict(font=dict(color=font_color)),
        ),
        hoverlabel=dict(
            bgcolor=card_bg,
            font=dict(color=font_color),
            bordercolor=tokens["border"],
        ),
    )
    fig.update_xaxes(
        tickfont=dict(color=axis_color),
        title_font=dict(color=font_color),
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )
    fig.update_yaxes(
        tickfont=dict(color=axis_color),
        title_font=dict(color=font_color),
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )
    return fig


def inject_theme_css(theme: str | None = None) -> None:
    """Inject theme styling CSS compatible with Light, Dark, and System modes."""
    active_theme = theme or get_theme_mode()

    if active_theme == "Dark":
        theme_specific_css = f"""
        :root {{
            --bg-color: {DARK_THEME_TOKENS['bg']};
            --sidebar-bg: {DARK_THEME_TOKENS['sidebar_bg']};
            --card-bg: {DARK_THEME_TOKENS['card_bg']};
            --text-color: {DARK_THEME_TOKENS['text']};
            --text-muted: {DARK_THEME_TOKENS['text_muted']};
            --border-color: {DARK_THEME_TOKENS['border']};
        }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {DARK_THEME_TOKENS['bg']} !important;
            color: {DARK_THEME_TOKENS['text']} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {DARK_THEME_TOKENS['sidebar_bg']} !important;
            color: {DARK_THEME_TOKENS['text']} !important;
        }}
        """
    elif active_theme == "Light":
        theme_specific_css = f"""
        :root {{
            --bg-color: {LIGHT_THEME_TOKENS['bg']};
            --sidebar-bg: {LIGHT_THEME_TOKENS['sidebar_bg']};
            --card-bg: {LIGHT_THEME_TOKENS['card_bg']};
            --text-color: {LIGHT_THEME_TOKENS['text']};
            --text-muted: {LIGHT_THEME_TOKENS['text_muted']};
            --border-color: {LIGHT_THEME_TOKENS['border']};
        }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {LIGHT_THEME_TOKENS['bg']} !important;
            color: {LIGHT_THEME_TOKENS['text']} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {LIGHT_THEME_TOKENS['sidebar_bg']} !important;
            color: {LIGHT_THEME_TOKENS['text']} !important;
        }}
        """
    else:  # System mode
        theme_specific_css = f"""
        :root {{
            --bg-color: {LIGHT_THEME_TOKENS['bg']};
            --sidebar-bg: {LIGHT_THEME_TOKENS['sidebar_bg']};
            --card-bg: {LIGHT_THEME_TOKENS['card_bg']};
            --text-color: {LIGHT_THEME_TOKENS['text']};
            --text-muted: {LIGHT_THEME_TOKENS['text_muted']};
            --border-color: {LIGHT_THEME_TOKENS['border']};
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: {DARK_THEME_TOKENS['bg']};
                --sidebar-bg: {DARK_THEME_TOKENS['sidebar_bg']};
                --card-bg: {DARK_THEME_TOKENS['card_bg']};
                --text-color: {DARK_THEME_TOKENS['text']};
                --text-muted: {DARK_THEME_TOKENS['text_muted']};
                --border-color: {DARK_THEME_TOKENS['border']};
            }}
            .stApp, [data-testid="stAppViewContainer"] {{
                background-color: {DARK_THEME_TOKENS['bg']} !important;
                color: {DARK_THEME_TOKENS['text']} !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: {DARK_THEME_TOKENS['sidebar_bg']} !important;
                color: {DARK_THEME_TOKENS['text']} !important;
            }}
        }}
        """

    st.markdown(
        f"""
        <style>
        {theme_specific_css}

        /* 1. Global Typography: text, paragraph, spans, list items */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] p, p, span, li {{
            color: var(--text-color);
        }}

        /* 2. Headings: h1, h2, h3, h4, h5, h6 */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {{
            color: var(--text-color) !important;
        }}

        /* 3. Labels & Captions */
        label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
        .stSelectbox label, .stNumberInput label, .stTextInput label, .stMultiSelect label {{
            color: var(--text-color) !important;
        }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, small {{
            color: var(--text-muted) !important;
        }}

        /* 4. Inputs & Select boxes */
        input, textarea, [data-baseweb="input"], [data-baseweb="base-input"] {{
            color: var(--text-color) !important;
            background-color: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }}

        /* 5. Tables */
        table, th, td, [data-testid="stTable"] {{
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}

        /* 6. Tabs */
        [data-baseweb="tab-list"] {{
            border-bottom-color: var(--border-color) !important;
        }}
        [data-baseweb="tab"] {{
            color: var(--text-muted) !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: var(--text-color) !important;
            border-bottom-color: #00B386 !important;
        }}

        /* 7. Expanders */
        [data-testid="stExpander"] {{
            background-color: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stExpander"] summary {{
            color: var(--text-color) !important;
        }}

        /* 8. Metrics */
        [data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--text-color) !important;
        }}

        /* 9. Custom Cards & HTML Containers */
        .kpi-card {{
            background-color: var(--card-bg, var(--secondary-background-color)) !important;
            color: var(--text-color) !important;
            border-radius: 8px;
            padding: 16px 20px;
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2)) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 12px;
        }}
        .kpi-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted, var(--text-color)) !important;
            opacity: 0.85;
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-color) !important;
            line-height: 1.2;
        }}
        .kpi-delta {{
            font-size: 0.85rem;
            margin-top: 4px;
            font-weight: 500;
        }}
        .kpi-delta-positive {{
            color: #00B386;
        }}
        .kpi-delta-negative {{
            color: #EB5757;
        }}
        .kpi-delta-neutral {{
            color: var(--text-muted, var(--text-color));
            opacity: 0.7;
        }}

        /* Risk badge */
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
        }}

        /* Data as of chip */
        .data-as-of-chip {{
            display: inline-flex;
            align-items: center;
            background-color: rgba(0, 179, 134, 0.12);
            color: #00B386;
            border: 1px solid rgba(0, 179, 134, 0.3);
            border-radius: 16px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        /* Disclaimer footer */
        .disclaimer-footer {{
            margin-top: 48px;
            padding: 16px 20px;
            border-top: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
            font-size: 0.8rem;
            color: var(--text-muted, var(--text-color));
            opacity: 0.75;
            text-align: center;
            line-height: 1.5;
        }}

        /* Empty state */
        .empty-state-box {{
            text-align: center;
            padding: 40px 20px;
            background-color: var(--card-bg, var(--secondary-background-color)) !important;
            color: var(--text-color) !important;
            border-radius: 8px;
            border: 1px dashed var(--border-color, rgba(128, 128, 128, 0.3)) !important;
            margin: 20px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

