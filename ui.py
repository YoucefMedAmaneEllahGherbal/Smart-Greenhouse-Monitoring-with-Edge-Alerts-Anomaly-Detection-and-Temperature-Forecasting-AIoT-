"""UI Presentation Layer for Smart Greenhouse AIoT Streamlit Dashboard.

Provides modern dark greenhouse components:
- Inline SVG icon library (Lucide / Material style)
- Custom HTML KPI cards and responsive grid
- Semicircular gauge for anomaly scoring
- Interactive dark Plotly charts with smooth spline and gradient fills
- Live connection header pill with CSS pulse animation
- Clean status banner and architecture flow diagram
- HTML sanitizer helper: render_html() to prevent Markdown parser interference
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants
COLOR_BG = "#0B1210"
COLOR_SURFACE = "#121C18"
COLOR_BORDER = "rgba(255, 255, 255, 0.06)"
COLOR_TEXT = "#E6F0EA"
COLOR_MUTED = "#8FA39A"

COLOR_TEMP = "#FB923C"       # Orange
COLOR_HUMIDITY = "#38BDF8"   # Sky Blue
COLOR_FORECAST = "#A78BFA"   # Violet
COLOR_NORMAL = "#34D399"     # Emerald
COLOR_WARNING = "#FBBF24"    # Amber
COLOR_CRITICAL = "#F87171"   # Red

# -----------------------------------------------------------------------------
# SVG Icons Library (Lucide / Material Minimalist Style)
# -----------------------------------------------------------------------------
SVG_LEAF = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>"""

SVG_THERMOMETER = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>"""

SVG_DROP = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>"""

SVG_VPD = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>"""

SVG_ALERT = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""

SVG_SHIELD = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""

SVG_FORECAST = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""

SVG_CHECK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""

SVG_ARROW_UP = """<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>"""

SVG_ARROW_DOWN = """<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>"""

SVG_CHIP = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/></svg>"""

SVG_CLOUD = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>"""

SVG_LAYOUT = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>"""

SVG_BRAIN = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-4.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-4.04z"/></svg>"""

SVG_SEARCH = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>"""

SVG_INFO = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>"""


# -----------------------------------------------------------------------------
# HTML Sanitizer & Markdown Helper
# -----------------------------------------------------------------------------
def clean_html(html: str) -> str:
    """Sanitize HTML string to eliminate Markdown parser interference.

    1. Removes all common leading indentation via textwrap.dedent.
    2. Collapses whitespace and newlines between tags (re.sub(r'>\\s+<', '><', ...)).
    3. Collapses any remaining newlines and surrounding spaces into a single space.
    4. Strips leading and trailing whitespace.
    """
    if not html:
        return ""
    dedented = textwrap.dedent(html).strip()
    # Collapse whitespace between tags
    collapsed = re.sub(r">\s+<", "><", dedented)
    # Collapse any remaining newlines into spaces so no markdown blank lines / code blocks exist
    compact = re.sub(r"\s*\n\s*", " ", collapsed)
    return compact.strip()


def render_html(html: str) -> None:
    """Render compact, dedented HTML without Markdown parsing glitches."""
    cleaned = clean_html(html)
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CSS Loader
# -----------------------------------------------------------------------------
def load_custom_css(css_path: Optional[Path] = None) -> None:
    """Inject styles.css into the Streamlit session."""
    if css_path is None:
        css_path = Path(__file__).resolve().parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Component: Header with Live Connection Pill
# -----------------------------------------------------------------------------
def render_header(
    title: str = "Smart Greenhouse AIoT",
    subtitle: str = "Edge ESP32 Sensing • ThingSpeak Cloud Telemetry • 1-Hour Forecast ML • Isolation Forest Anomaly Detection",
    connection_state: str = "live",
    age_seconds: Optional[float] = None,
) -> None:
    """Render the modern dark greenhouse header with live connection pill."""
    if connection_state == "live":
        pill_class = "pulse-green"
        pill_status = "Live"
        pill_color = "#34D399"
    elif connection_state == "delayed":
        pill_class = "pulse-amber"
        pill_status = "Delayed"
        pill_color = "#FBBF24"
    else:
        pill_class = "pulse-red"
        pill_status = "Offline"
        pill_color = "#F87171"

    if age_seconds is not None and age_seconds != float("inf"):
        age_str = f"Updated {int(age_seconds)}s ago"
    else:
        age_str = "No feed data"

    html = f"""
    <div class="app-header-container">
        <div class="app-title-group">
            <div class="app-logo-badge">
                {SVG_LEAF.strip()}
            </div>
            <div>
                <div class="app-title-text">{title.strip()}</div>
                <div class="app-subtitle-text">{subtitle.strip()}</div>
            </div>
        </div>
        <div class="live-pill">
            <span class="pulse-dot {pill_class}"></span>
            <span style="font-weight:600; color:{pill_color};">{pill_status}</span>
            <span style="color:#8FA39A; font-size:0.75rem;">•</span>
            <span style="color:#8FA39A;">{age_str}</span>
        </div>
    </div>
    """
    render_html(html)


# -----------------------------------------------------------------------------
# Component: Status Banner
# -----------------------------------------------------------------------------
def render_status_banner(
    state: str,
    title: str,
    message: str,
    time_summary: str = "",
    stale_notice: str = "",
) -> None:
    """Render a single wide status banner with soft tinted gradient background."""
    if state == "critical":
        card_class = "status-banner-critical"
        icon_bg = "rgba(248, 113, 113, 0.2)"
        icon_color = "#F87171"
        icon_svg = SVG_ALERT
    elif state == "warning":
        card_class = "status-banner-warning"
        icon_bg = "rgba(251, 191, 36, 0.2)"
        icon_color = "#FBBF24"
        icon_svg = SVG_ALERT
    else:
        card_class = "status-banner-normal"
        icon_bg = "rgba(52, 211, 153, 0.2)"
        icon_color = "#34D399"
        icon_svg = SVG_CHECK

    stale_html = ""
    if stale_notice:
        stale_html = f'<div style="margin-top:6px; color:#FCA5A5; font-size:0.8rem; font-weight:500;">{stale_notice.strip()}</div>'

    time_html = ""
    if time_summary:
        time_html = f'<div class="status-banner-time">{time_summary.strip()}</div>'

    html = f"""
    <div class="status-banner-card {card_class}">
        <div class="status-banner-icon" style="background:{icon_bg}; color:{icon_color};">
            {icon_svg.strip()}
        </div>
        <div style="flex-grow:1;">
            <div class="status-banner-title" style="color:{icon_color};">
                <span>{title.strip()}</span>
            </div>
            <div class="status-banner-desc">{message.strip()}</div>
            {time_html}
            {stale_html}
        </div>
    </div>
    """
    render_html(html)


# -----------------------------------------------------------------------------
# Component: Custom KPI Cards & Grid
# -----------------------------------------------------------------------------
def render_kpi_grid(cards: List[Dict[str, Any]], grid_cols: int = 7) -> None:
    """Render a responsive grid of custom KPI cards in a single compact HTML block."""
    grid_class = f"kpi-grid-{grid_cols}" if grid_cols in (5, 7) else ""
    card_htmls = []

    for c in cards:
        icon_svg = str(c.get("icon_svg", SVG_THERMOMETER)).strip()
        icon_bg = str(c.get("icon_bg", "rgba(52, 211, 153, 0.15)")).strip()
        icon_color = str(c.get("icon_color", "#34D399")).strip()
        label = str(c.get("label", "METRIC")).strip()
        value = str(c.get("value", "--")).strip()
        unit = str(c.get("unit", "")).strip()
        subtext = str(c.get("subtext", "")).strip()
        subtext_type = str(c.get("subtext_type", "neutral")).strip()
        arrow = c.get("arrow", None)

        arrow_html = ""
        if arrow == "up":
            arrow_html = SVG_ARROW_UP.strip()
        elif arrow == "down":
            arrow_html = SVG_ARROW_DOWN.strip()

        sub_class = f"kpi-subtext-{subtext_type}"
        sub_html = f'<div class="kpi-subtext {sub_class}">{arrow_html}<span>{subtext}</span></div>' if subtext else ""
        unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""

        card_html = (
            f'<div class="kpi-card">'
            f'<div class="kpi-top">'
            f'<div class="kpi-icon-box" style="background:{icon_bg}; color:{icon_color};">{icon_svg}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
            f'<div>'
            f'<div class="kpi-value-row"><span class="kpi-value">{value}</span>{unit_html}</div>'
            f'{sub_html}'
            f'</div>'
            f'</div>'
        )
        card_htmls.append(card_html)

    full_grid_html = f'<div class="kpi-grid {grid_class}">{"".join(card_htmls)}</div>'
    render_html(full_grid_html)


# -----------------------------------------------------------------------------
# Component: Chart Header
# -----------------------------------------------------------------------------
def render_chart_header(
    title: str,
    icon_svg: str,
    icon_color: str,
    badge_text: str,
    badge_bg: str = "rgba(255, 255, 255, 0.08)",
) -> None:
    """Render a standalone card-like header for Plotly charts."""
    html = f"""
    <div class="chart-card-header">
        <div class="chart-card-title">
            <span style="color:{icon_color};">{icon_svg.strip()}</span>
            <span>{title.strip()}</span>
        </div>
        <div class="chart-card-badge" style="background:{badge_bg}; color:{icon_color};">
            {badge_text.strip()}
        </div>
    </div>
    """
    render_html(html)


# -----------------------------------------------------------------------------
# Component: Anomaly Details & Contributing Chips
# -----------------------------------------------------------------------------
def render_anomaly_chips(
    score: float,
    threshold: float,
    status: str,
    severity: str,
    unusual_features: List[str],
) -> None:
    """Render anomaly status, threshold, severity chip, and contributing features."""
    chips_html = "".join([
        f'<span class="feature-chip feature-chip-accent">{str(f).strip()}</span>'
        for f in unusual_features
    ])

    status_chip_class = "feature-chip-normal" if status == "Normal" else "feature-chip-danger"
    if severity == "MEDIUM":
        severity_chip_class = "feature-chip-warning"
    elif severity == "HIGH":
        severity_chip_class = "feature-chip-danger"
    else:
        severity_chip_class = "feature-chip-normal"

    html = f"""
    <div style="background:#121C18; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px 16px; margin-top:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:0.8rem; color:#8FA39A;">
                Threshold: <span style="color:#E6F0EA; font-weight:600;">{threshold:.4f}</span>
            </div>
            <div class="chip-container" style="margin:0;">
                <span class="feature-chip {status_chip_class}">{status.strip()}</span>
                <span class="feature-chip {severity_chip_class}">Severity: {severity.strip()}</span>
            </div>
        </div>
        <div style="margin-top:10px;">
            <div style="font-size:0.75rem; text-transform:uppercase; color:#8FA39A; font-weight:600; letter-spacing:0.05em; margin-bottom:4px;">
                Unusual Contributing Features
            </div>
            <div class="chip-container">
                {chips_html}
            </div>
        </div>
        <div style="font-size:0.75rem; color:#8FA39A; font-style:italic; margin-top:8px;">
            Indication, not a diagnosis of the cause.
        </div>
    </div>
    """
    render_html(html)


# -----------------------------------------------------------------------------
# Component: Forecast Diagnostics Card
# -----------------------------------------------------------------------------
def render_forecast_diagnostics_card(
    forecast_res: Optional[Dict[str, Any]],
    forecast_err: Optional[str] = None,
) -> None:
    """Render full forecast diagnostics container as a single clean HTML block."""
    header_html = f"""
    <div class="detail-card-title">
        <span style="color:{COLOR_FORECAST};">{SVG_FORECAST.strip()}</span>
        <span>Temperature Forecast Diagnostics</span>
    </div>
    """

    if forecast_res:
        model_used = str(forecast_res.get("model_used", "N/A")).strip()
        hist_mins = forecast_res.get("history_minutes", 0)
        is_lag = "lag" in model_used.lower()
        badge_color = COLOR_NORMAL if is_lag else COLOR_FORECAST

        if is_lag:
            desc_html = """
            <div style="background:rgba(52,211,153,0.08); border-left:3px solid #34D399; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#E6F0EA; line-height:1.45;">
                <b>Lag model active:</b> Leveraging over 70 minutes of continuous resampled telemetry to capture 10m, 30m, and 60m momentum dynamics for enhanced forecast accuracy.
            </div>
            """
        else:
            desc_html = """
            <div style="background:rgba(167,139,250,0.08); border-left:3px solid #A78BFA; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#E6F0EA; line-height:1.45;">
                <b>Instant model active:</b> Operating on instantaneous T, RH, and solar diurnal cycle harmonics. The system will automatically upgrade to the lag model once 70+ minutes of continuous telemetry accumulate.
            </div>
            """

        content_html = f"""
        <div class="detail-card">
            {header_html}
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">
                <div style="background:#0E1613; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px;">
                    <div style="font-size:0.7rem; color:#8FA39A; text-transform:uppercase; font-weight:600;">Active Model</div>
                    <div style="font-size:0.92rem; font-weight:600; color:{badge_color}; margin-top:2px;">{model_used}</div>
                </div>
                <div style="background:#0E1613; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px;">
                    <div style="font-size:0.7rem; color:#8FA39A; text-transform:uppercase; font-weight:600;">Continuous History</div>
                    <div style="font-size:0.92rem; font-weight:600; color:#E6F0EA; margin-top:2px;">{hist_mins} minutes</div>
                </div>
            </div>
            {desc_html}
        </div>
        """
    elif forecast_err:
        content_html = f"""
        <div class="detail-card">
            {header_html}
            <div style="background:rgba(251,191,36,0.08); border-left:3px solid #FBBF24; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#E6F0EA;">
                Forecast computation notice: {forecast_err.strip()}
            </div>
        </div>
        """
    else:
        content_html = f"""
        <div class="detail-card">
            {header_html}
            <div style="font-size:0.85rem; color:#8FA39A;">
                Insufficient continuous telemetry points to compute +1 hour forecast projection.
            </div>
        </div>
        """

    render_html(content_html)


# -----------------------------------------------------------------------------
# Component: Plotly Temperature Chart
# -----------------------------------------------------------------------------
def build_temperature_chart(
    df: pd.DataFrame,
    forecast_point: Optional[Dict[str, Any]],
    tz_name: str,
    time_window: str = "All",
) -> go.Figure:
    """Create interactive Plotly chart with transparent dark styling and spline interpolation."""
    fig = go.Figure()

    if df.empty or "ts" not in df.columns:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[dict(text="No temperature data", showarrow=False, font=dict(color=COLOR_MUTED))],
            height=320,
        )
        return fig

    # Localize timestamps
    plot_df = df.copy()
    if plot_df["ts"].dt.tz is not None:
        plot_df["local_ts"] = plot_df["ts"].dt.tz_convert(tz_name)
    else:
        plot_df["local_ts"] = plot_df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_name)

    # Filter by time window for display
    if not plot_df.empty:
        latest_time = plot_df["local_ts"].max()
        if time_window == "30 min":
            cutoff = latest_time - pd.Timedelta(minutes=30)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]
        elif time_window == "1 hour":
            cutoff = latest_time - pd.Timedelta(hours=1)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]
        elif time_window == "3 hours":
            cutoff = latest_time - pd.Timedelta(hours=3)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]

    # Historical temperature spline line with gradient fill
    fig.add_trace(
        go.Scatter(
            x=plot_df["local_ts"],
            y=plot_df["T"],
            mode="lines",
            name="Temperature",
            line=dict(color=COLOR_TEMP, width=2.5, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillcolor="rgba(251, 146, 60, 0.08)",
            hovertemplate="%{x|%H:%M:%S}<br><b>%{y:.1f} °C</b><extra></extra>",
        )
    )

    # Highlight latest measured point with a halo marker
    if not plot_df.empty:
        last_t = plot_df["T"].iloc[-1]
        last_time = plot_df["local_ts"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_time],
                y=[last_t],
                mode="markers",
                name="Latest",
                marker=dict(
                    size=10,
                    color=COLOR_TEMP,
                    line=dict(color="rgba(251, 146, 60, 0.4)", width=6),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Shaded band for safe agronomic range (5 °C to 30 °C)
    fig.add_hrect(
        y0=5.0,
        y1=30.0,
        fillcolor="rgba(52, 211, 153, 0.03)",
        line_width=0,
        layer="below",
    )

    # Limit annotations
    fig.add_hline(
        y=35.0,
        line_dash="dash",
        line_color="rgba(248, 113, 113, 0.6)",
        line_width=1.2,
        annotation_text="Critical 35 °C",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(248, 113, 113, 0.8)"),
    )
    fig.add_hline(
        y=30.0,
        line_dash="dash",
        line_color="rgba(251, 191, 36, 0.6)",
        line_width=1.2,
        annotation_text="Warning 30 °C",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(251, 191, 36, 0.8)"),
    )
    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color="rgba(251, 191, 36, 0.4)",
        line_width=1.0,
        annotation_text="Warning 5 °C",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="rgba(251, 191, 36, 0.7)"),
    )

    # 1-Hour Forecast Point and Violet Dashed Connector
    if forecast_point and not plot_df.empty:
        fc_time = forecast_point["forecast_time"]
        if hasattr(fc_time, "tzinfo") and fc_time.tzinfo is None:
            try:
                import pytz
                fc_time = pytz.timezone(tz_name).localize(fc_time)
            except Exception:
                pass

        fc_val = forecast_point["prediction"]

        # Dashed connector
        fig.add_trace(
            go.Scatter(
                x=[last_time, fc_time],
                y=[last_t, fc_val],
                mode="lines",
                name="Projection",
                line=dict(color=COLOR_FORECAST, width=1.8, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Violet Diamond Marker
        fig.add_trace(
            go.Scatter(
                x=[fc_time],
                y=[fc_val],
                mode="markers+text",
                name="Forecast +1h",
                marker=dict(symbol="diamond", size=11, color=COLOR_FORECAST, line=dict(color="#ffffff", width=1.5)),
                text=[f" {fc_val:.1f} °C"],
                textposition="top right",
                textfont=dict(size=11, color=COLOR_FORECAST),
                hovertemplate="Forecast %{x|%H:%M}:<br><b>%{y:.1f} °C</b><extra></extra>",
                showlegend=False,
            )
        )

    # Clean Dark Layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLOR_MUTED, size=11),
        margin=dict(l=30, r=20, t=10, b=30),
        height=320,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLOR_SURFACE, font_size=12, font_family="Inter"),
        uirevision="greenhouse_temp_chart",
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickformat="%H:%M",
            color=COLOR_MUTED,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_BORDER,
            showline=False,
            zeroline=False,
            color=COLOR_MUTED,
            tickfont=dict(size=10),
            ticksuffix=" °C",
        ),
        showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# Component: Plotly Humidity Chart
# -----------------------------------------------------------------------------
def build_humidity_chart(
    df: pd.DataFrame,
    tz_name: str,
    time_window: str = "All",
) -> go.Figure:
    """Create interactive Plotly humidity chart with transparent dark styling."""
    fig = go.Figure()

    if df.empty or "ts" not in df.columns:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[dict(text="No humidity data", showarrow=False, font=dict(color=COLOR_MUTED))],
            height=320,
        )
        return fig

    # Localize timestamps
    plot_df = df.copy()
    if plot_df["ts"].dt.tz is not None:
        plot_df["local_ts"] = plot_df["ts"].dt.tz_convert(tz_name)
    else:
        plot_df["local_ts"] = plot_df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_name)

    # Filter by time window
    if not plot_df.empty:
        latest_time = plot_df["local_ts"].max()
        if time_window == "30 min":
            cutoff = latest_time - pd.Timedelta(minutes=30)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]
        elif time_window == "1 hour":
            cutoff = latest_time - pd.Timedelta(hours=1)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]
        elif time_window == "3 hours":
            cutoff = latest_time - pd.Timedelta(hours=3)
            plot_df = plot_df[plot_df["local_ts"] >= cutoff]

    # Humidity spline curve with sky-blue gradient fill
    fig.add_trace(
        go.Scatter(
            x=plot_df["local_ts"],
            y=plot_df["RH"],
            mode="lines",
            name="Humidity",
            line=dict(color=COLOR_HUMIDITY, width=2.5, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
            hovertemplate="%{x|%H:%M:%S}<br><b>%{y:.1f} %</b><extra></extra>",
        )
    )

    # Highlight latest point
    if not plot_df.empty:
        last_rh = plot_df["RH"].iloc[-1]
        last_time = plot_df["local_ts"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_time],
                y=[last_rh],
                mode="markers",
                name="Latest",
                marker=dict(
                    size=10,
                    color=COLOR_HUMIDITY,
                    line=dict(color="rgba(56, 189, 248, 0.4)", width=6),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Safe humidity range shaded band (40 % - 90 %)
    fig.add_hrect(
        y0=40.0,
        y1=90.0,
        fillcolor="rgba(52, 211, 153, 0.03)",
        line_width=0,
        layer="below",
    )

    # Limits
    fig.add_hline(
        y=90.0,
        line_dash="dash",
        line_color="rgba(251, 191, 36, 0.6)",
        line_width=1.2,
        annotation_text="Humid 90 %",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(251, 191, 36, 0.8)"),
    )
    fig.add_hline(
        y=40.0,
        line_dash="dash",
        line_color="rgba(251, 191, 36, 0.6)",
        line_width=1.2,
        annotation_text="Dry 40 %",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="rgba(251, 191, 36, 0.8)"),
    )

    y_min = max(0, plot_df["RH"].min() - 8) if not plot_df.empty else 0
    y_max = min(100, plot_df["RH"].max() + 8) if not plot_df.empty else 100

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLOR_MUTED, size=11),
        margin=dict(l=30, r=20, t=10, b=30),
        height=320,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLOR_SURFACE, font_size=12, font_family="Inter"),
        uirevision="greenhouse_rh_chart",
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickformat="%H:%M",
            color=COLOR_MUTED,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_BORDER,
            showline=False,
            zeroline=False,
            color=COLOR_MUTED,
            tickfont=dict(size=10),
            ticksuffix=" %",
            range=[y_min, y_max],
        ),
        showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# Component: Semicircular Anomaly Gauge (Plotly Indicator)
# -----------------------------------------------------------------------------
def build_anomaly_gauge(
    score: float,
    threshold: float = 0.6315,
    high_threshold: float = 0.6315,
) -> go.Figure:
    """Create a semicircular gauge for Isolation Forest anomaly score vs threshold."""
    gauge_max = max(0.75, round(score + 0.05, 2))
    gauge_min = 0.45
    bar_color = COLOR_CRITICAL if score > threshold else COLOR_NORMAL

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f", "font": {"family": "Inter", "size": 24, "color": COLOR_TEXT}},
            gauge={
                "axis": {
                    "range": [gauge_min, gauge_max],
                    "tickwidth": 1,
                    "tickcolor": COLOR_MUTED,
                    "tickfont": {"size": 9, "color": COLOR_MUTED},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [gauge_min, threshold], "color": "rgba(52, 211, 153, 0.12)"},
                    {"range": [threshold, gauge_max], "color": "rgba(248, 113, 113, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": COLOR_WARNING, "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=25, r=25, t=15, b=5),
        height=175,
        uirevision="greenhouse_gauge",
    )
    return fig


# -----------------------------------------------------------------------------
# Component: Architecture Flow Diagram
# -----------------------------------------------------------------------------
def render_architecture_flow() -> None:
    """Render clean horizontal HTML/CSS flow diagram with flowing line animation."""
    html = f"""
    <div class="arch-flow-container">
        <div class="arch-flow">
            <!-- Node 1: Edge -->
            <div class="arch-node">
                <div class="arch-node-header">
                    <div class="arch-node-icon">{SVG_CHIP.strip()}</div>
                    <div class="arch-node-title">ESP32 + DHT22</div>
                </div>
                <div class="arch-node-desc">
                    Wokwi edge simulation. Reads every 5s, evaluates limits, uploads every 20s.
                </div>
            </div>

            <!-- Connector 1 -->
            <div class="arch-connector">
                <div class="arch-connector-label">HTTP POST (20s)</div>
            </div>

            <!-- Node 2: IoT Cloud -->
            <div class="arch-node">
                <div class="arch-node-header">
                    <div class="arch-node-icon" style="color:#38BDF8; background:rgba(56,189,248,0.12);">{SVG_CLOUD.strip()}</div>
                    <div class="arch-node-title">ThingSpeak</div>
                </div>
                <div class="arch-node-desc">
                    Feeds REST API. Channels telemetry: Field 1 (T), Field 2 (RH), Field 3 (Alert).
                </div>
            </div>

            <!-- Connector 2 -->
            <div class="arch-connector">
                <div class="arch-connector-label">REST feeds.json</div>
            </div>

            <!-- Node 3: Streamlit UI -->
            <div class="arch-node">
                <div class="arch-node-header">
                    <div class="arch-node-icon" style="color:#A78BFA; background:rgba(167,139,250,0.12);">{SVG_LAYOUT.strip()}</div>
                    <div class="arch-node-title">Streamlit AIoT</div>
                </div>
                <div class="arch-node-desc">
                    Dark greenhouse dashboard. Fragments auto-refresh, dynamic Plotly curves.
                </div>
            </div>

            <!-- Connector 3 -->
            <div class="arch-connector">
                <div class="arch-connector-label">Inference Pipelines</div>
            </div>

            <!-- Node 4: ML Engines -->
            <div class="arch-node">
                <div class="arch-node-header">
                    <div class="arch-node-icon" style="color:#FB923C; background:rgba(251,146,60,0.12);">{SVG_BRAIN.strip()}</div>
                    <div class="arch-node-title">ML Engines</div>
                </div>
                <div class="arch-node-desc">
                    1h Forecast (Ridge/Lag) & Climate Anomaly (Isolation Forest on [T, RH, VPD]).
                </div>
            </div>
        </div>
    </div>
    """
    render_html(html)


# -----------------------------------------------------------------------------
# Component: Stat Cards Grid (for Model Metrics)
# -----------------------------------------------------------------------------
def render_stat_cards(stats: List[Dict[str, str]]) -> None:
    """Render small clean stat cards in a responsive grid."""
    cards_html = []
    for s in stats:
        label = str(s.get("label", "")).strip()
        value = str(s.get("value", "")).strip()
        color = str(s.get("color", COLOR_TEXT)).strip()
        card = (
            f'<div class="stat-card">'
            f'<div class="stat-card-label">{label}</div>'
            f'<div class="stat-card-value" style="color:{color};">{value}</div>'
            f'</div>'
        )
        cards_html.append(card)

    full_html = f'<div class="stat-card-grid">{"".join(cards_html)}</div>'
    render_html(full_html)


# -----------------------------------------------------------------------------
# Component: Empty / Error State Card
# -----------------------------------------------------------------------------
def render_empty_state(
    title: str,
    message: str,
    icon_svg: str = SVG_SEARCH,
) -> None:
    """Render friendly centered state card without raw errors or emojis."""
    html = f"""
    <div class="empty-state-card">
        <div class="empty-state-icon">
            {icon_svg.strip()}
        </div>
        <div class="empty-state-title">{title.strip()}</div>
        <div class="empty-state-desc">{message.strip()}</div>
    </div>
    """
    render_html(html)
