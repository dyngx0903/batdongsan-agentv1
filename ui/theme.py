from __future__ import annotations

from typing import Dict

import streamlit as st


THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "colors": {
        "primary_600": "#7C3AED",
        "primary_700": "#6D28D9",
        "primary_300": "#A78BFA",
        "secondary_500": "#06B6D4",
        "secondary_300": "#67E8F9",
        "background": "#F8FAFC",
        "surface": "#FFFFFF",
        "border": "#E5E7EB",
        "text_main": "#111827",
        "text_sub": "#6B7280",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "info": "#3B82F6",
    },
    "gradients": {
        "hero": "linear-gradient(135deg, #6D28D9, #9333EA, #06B6D4)",
        "primary_button": "linear-gradient(135deg, #7C3AED 0%, #6D28D9 45%, #06B6D4 100%)",
    },
    "radius": {
        "sm": "10px",
        "md": "14px",
        "lg": "18px",
    },
    "shadow": {
        "soft": "0 10px 24px rgba(15, 23, 42, 0.08)",
        "hero": "0 16px 36px rgba(109, 40, 217, 0.28)",
    },
    "typography": {
        "font_family": "'Manrope', sans-serif",
        "text_xs": "0.78rem",
        "text_sm": "0.9rem",
        "text_md": "0.96rem",
        "text_lg": "1.08rem",
        "text_xl": "1.4rem",
        "weight_regular": "400",
        "weight_semibold": "600",
        "weight_bold": "700",
        "weight_extrabold": "800",
        "line_height_body": "1.55",
        "line_height_heading": "1.25",
    },
    "spacing": {
        "xs": "4px",
        "sm": "8px",
        "md": "12px",
        "lg": "16px",
        "xl": "20px",
        "2xl": "24px",
    },
}


def get_global_css() -> str:
    colors = THEME_TOKENS["colors"]
    gradients = THEME_TOKENS["gradients"]
    radius = THEME_TOKENS["radius"]
    shadow = THEME_TOKENS["shadow"]
    typography = THEME_TOKENS["typography"]
    spacing = THEME_TOKENS["spacing"]

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    /* Core design tokens */
    :root {{
        --color-primary-600: {colors['primary_600']};
        --color-primary-700: {colors['primary_700']};
        --color-primary-300: {colors['primary_300']};
        --color-secondary-500: {colors['secondary_500']};
        --color-secondary-300: {colors['secondary_300']};
        --color-background: {colors['background']};
        --color-surface: {colors['surface']};
        --color-border: {colors['border']};
        --color-text-main: {colors['text_main']};
        --color-text-sub: {colors['text_sub']};
        --color-success: {colors['success']};
        --color-warning: {colors['warning']};
        --color-danger: {colors['danger']};
        --color-info: {colors['info']};

        --gradient-hero: {gradients['hero']};
        --gradient-primary-button: {gradients['primary_button']};

        --radius-sm: {radius['sm']};
        --radius-md: {radius['md']};
        --radius-lg: {radius['lg']};

        --shadow-soft: {shadow['soft']};
        --shadow-hero: {shadow['hero']};

        --font-family-base: {typography['font_family']};
        --text-xs: {typography['text_xs']};
        --text-sm: {typography['text_sm']};
        --text-md: {typography['text_md']};
        --text-lg: {typography['text_lg']};
        --text-xl: {typography['text_xl']};
        --font-weight-regular: {typography['weight_regular']};
        --font-weight-semibold: {typography['weight_semibold']};
        --font-weight-bold: {typography['weight_bold']};
        --font-weight-extrabold: {typography['weight_extrabold']};
        --line-height-body: {typography['line_height_body']};
        --line-height-heading: {typography['line_height_heading']};

        --space-xs: {spacing['xs']};
        --space-sm: {spacing['sm']};
        --space-md: {spacing['md']};
        --space-lg: {spacing['lg']};
        --space-xl: {spacing['xl']};
        --space-2xl: {spacing['2xl']};
    }}

    /* Global canvas + typography */
    html, body, [class*="css"] {{
        font-family: var(--font-family-base);
        color: var(--color-text-main);
        line-height: var(--line-height-body);
    }}

    .stApp {{
        background:
            radial-gradient(circle at 8% 6%, rgba(167, 139, 250, 0.22) 0%, rgba(167, 139, 250, 0) 36%),
            radial-gradient(circle at 94% 14%, rgba(103, 232, 249, 0.18) 0%, rgba(103, 232, 249, 0) 34%),
            var(--color-background);
    }}

    /* Sidebar panel */
    [data-testid="stSidebar"] > div:first-child {{
        background: rgba(255, 255, 255, 0.86);
        border-right: 1px solid var(--color-border);
    }}

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] label {{
        color: var(--color-text-sub);
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{
        color: var(--color-text-main);
    }}

    /* Hero + shared card surfaces */
    .hero {{
        background: var(--gradient-hero);
        border-radius: var(--radius-lg);
        padding: calc(var(--space-lg) + 2px) var(--space-xl);
        color: #ffffff;
        box-shadow: var(--shadow-hero);
        margin-bottom: var(--space-md);
    }}

    .hero h1 {{
        margin: 0;
        font-size: var(--text-xl);
        font-weight: var(--font-weight-extrabold);
        line-height: var(--line-height-heading);
        letter-spacing: 0.01em;
    }}

    .hero p {{
        margin: calc(var(--space-sm) - 2px) 0 0;
        opacity: 0.93;
        font-size: var(--text-md);
    }}

    .panel {{
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-soft);
        padding: var(--space-md) calc(var(--space-md) + 2px);
        margin-bottom: calc(var(--space-sm) + 2px);
    }}

    .panel-title {{
        color: var(--color-text-main);
        font-weight: var(--font-weight-bold);
        font-size: var(--text-lg);
        margin: 0 0 calc(var(--space-sm) - 2px);
        line-height: var(--line-height-heading);
    }}

    .panel-sub {{
        color: var(--color-text-sub);
        margin: 0;
        font-size: var(--text-md);
    }}

    .quick-tip {{
        background: rgba(167, 139, 250, 0.10);
        border: 1px dashed var(--color-primary-300);
        border-radius: var(--radius-md);
        padding: calc(var(--space-sm) + 2px) var(--space-md);
        color: var(--color-primary-700);
        margin-bottom: var(--space-sm);
        font-size: var(--text-sm);
    }}

    /* Buttons: primary and quick actions */
    .stButton > button {{
        background: var(--gradient-primary-button);
        color: #ffffff;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        font-weight: var(--font-weight-bold);
        font-size: var(--text-sm);
        padding: calc(var(--space-sm) - 1px) var(--space-md);
        box-shadow: 0 8px 18px rgba(124, 58, 237, 0.20);
    }}

    .stButton > button:hover {{
        filter: brightness(1.03);
        transform: translateY(-1px);
    }}

    [data-testid="stSidebar"] .stButton > button {{
        border: 1px solid var(--color-primary-300);
    }}

    /* Inputs, chat container, and text hierarchy */
    .stChatInput [data-testid="stChatInput"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div,
    [data-testid="stSlider"] {{
        border-radius: var(--radius-sm);
    }}

    .stChatInput [data-testid="stChatInput"] {{
        border: 1px solid var(--color-border);
        background: var(--color-surface);
    }}

    .stChatMessage {{
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        background: var(--color-surface);
    }}

    .stCaption {{
        color: var(--color-text-sub);
        font-size: var(--text-xs);
    }}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(get_global_css(), unsafe_allow_html=True)
