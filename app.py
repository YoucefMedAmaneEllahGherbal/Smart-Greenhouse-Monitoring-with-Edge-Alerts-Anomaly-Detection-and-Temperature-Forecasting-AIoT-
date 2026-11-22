"""Smart Greenhouse AIoT Streamlit Application.

Redesigned with a modern dark greenhouse aesthetic:
- Clean typography and tight spacing
- Responsive HTML KPI cards
- Interactive transparent Plotly charts with spline smoothing and gradient fills
- Semicircular anomaly gauge
- Live connection indicator with pulsing status dot
- Horizontal animated architecture diagram
- Robust HTML sanitizer (render_html) preventing markdown parser code blocks
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Configure models directory in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# Local ML imports
try:
    from predict import load_bundle, predict_next_hour
except ImportError:
    from greenhouse_predict import load_bundle, predict_next_hour

from anomaly import calculate_vpd, score_reading
from thingspeak import (
    calculate_esp32_alert,
    fetch_thingspeak_data,
    get_default_credentials,
    get_latest_reading_age,
    get_reading_age_seconds,
    is_data_stale,
)
from ui import (
    COLOR_CRITICAL,
    COLOR_FORECAST,
    COLOR_HUMIDITY,
    COLOR_MUTED,
    COLOR_NORMAL,
    COLOR_SURFACE,
    COLOR_TEMP,
    COLOR_TEXT,
    COLOR_WARNING,
    SVG_ALERT,
    SVG_BRAIN,
    SVG_CHECK,
    SVG_CHIP,
    SVG_DROP,
    SVG_FORECAST,
    SVG_INFO,
    SVG_LEAF,
    SVG_SHIELD,
    SVG_THERMOMETER,
    SVG_VPD,
    build_anomaly_gauge,
    build_humidity_chart,
    build_temperature_chart,
    clean_html,
    load_custom_css,
    render_anomaly_chips,
    render_architecture_flow,
    render_chart_header,
    render_empty_state,
    render_forecast_diagnostics_card,
    render_header,
    render_html,
    render_kpi_grid,
    render_stat_cards,
    render_status_banner,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Greenhouse AIoT",
    page_icon=":material/potted_plant:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom Modern Dark Stylesheet
load_custom_css()

# Alert and Model Constants
ALERT_LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}


# -----------------------------------------------------------------------------
# Model Resource Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_ml_resources() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Load both ML inference bundles and model metadata from disk."""
    forecast_path = MODELS_DIR / "greenhouse_forecast_bundle.joblib"
    anomaly_path = MODELS_DIR / "climate_inference_bundle.joblib"
    metadata_path = MODELS_DIR / "model_metadata_and_metrics.json"

    forecast_bundle = None
    anomaly_bundle = None
    metadata = None
    errors = []

    if not forecast_path.exists():
        errors.append(f"Forecast model bundle missing at: {forecast_path}")
    else:
        try:
            forecast_bundle = load_bundle(str(forecast_path))
        except Exception as exc:
            errors.append(f"Failed to load forecast bundle: {exc}")

    if not anomaly_path.exists():
        errors.append(f"Anomaly model bundle missing at: {anomaly_path}")
    else:
        try:
            import joblib
            anomaly_bundle = joblib.load(str(anomaly_path))
        except Exception as exc:
            errors.append(f"Failed to load anomaly bundle: {exc}")

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            errors.append(f"Failed to read model metadata JSON: {exc}")

    return forecast_bundle, anomaly_bundle, metadata, "\n".join(errors)


# -----------------------------------------------------------------------------
# Application Flow
# -----------------------------------------------------------------------------
def main() -> None:
    forecast_bundle, anomaly_bundle, metadata, model_errors = load_ml_resources()

    if model_errors:
        st.error(f"Model Loading Error:\n{model_errors}")
        st.stop()

    # Pre-fetch default credentials
    default_chan_id, default_read_key = get_default_credentials()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        render_html(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="color:#34D399; display:flex;">{SVG_LEAF.strip()}</span>
                <span style="font-weight:700; font-size:1.05rem; color:#E6F0EA;">Greenhouse AIoT</span>
            </div>
        """)

        render_html("""
            <div class="sidebar-section-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>
                <span>Telemetry Stream</span>
            </div>
        """)

        channel_id = st.text_input(
            "Channel ID",
            value=default_chan_id,
            help="Numeric Channel ID from ThingSpeak (e.g. 3580105)",
        ).strip()

        read_key = st.text_input(
            "Read API Key (Optional)",
            value=default_read_key,
            type="password",
            help="Read API Key for private ThingSpeak channels",
        ).strip()

        history_size = st.slider(
            "History Window (readings)",
            min_value=50,
            max_value=1200,
            value=600,
            step=50,
            help="Default 600 points (~3.3 hours at 20s intervals)",
        )

        render_html("""
            <div class="sidebar-section-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21h5v-5"/></svg>
                <span>Refresh Settings</span>
            </div>
        """)

        refresh_options = {
            "20 seconds (Default)": 20,
            "10 seconds": 10,
            "30 seconds": 30,
            "60 seconds": 60,
            "Off (Manual only)": 0,
        }
        selected_refresh = st.selectbox(
            "Auto-refresh interval",
            options=list(refresh_options.keys()),
            index=0,
            help="Fragment background update frequency",
        )
        auto_refresh_sec = refresh_options[selected_refresh]

        stale_threshold_min = st.slider(
            "Stale threshold (minutes)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Mark telemetry as stale if latest reading exceeds this age.",
        )

        render_html("""
            <div class="sidebar-section-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <span>Display & Timezone</span>
            </div>
        """)

        timezone_options = [
            "Africa/Algiers",
            "UTC",
            "Europe/Paris",
            "America/New_York",
            "Asia/Tokyo",
            "Asia/Dubai",
        ]
        selected_tz = st.selectbox(
            "Greenhouse Timezone",
            options=timezone_options,
            index=0,
            help="Used for local timestamp formatting and solar feature calculations",
        )

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        if st.button("Refresh now", icon=":material/refresh:", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # --- TOP TABS ---
    tab_live, tab_manual, tab_models = st.tabs([
        ":material/sensors: Live Dashboard",
        ":material/tune: Manual Simulation",
        ":material/psychology: Models & Architecture",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Live Dashboard
    # -------------------------------------------------------------------------
    with tab_live:
        render_live_dashboard(
            channel_id=channel_id,
            read_key=read_key,
            history_size=history_size,
            auto_refresh_sec=auto_refresh_sec,
            tz_name=selected_tz,
            stale_threshold_min=stale_threshold_min,
            forecast_bundle=forecast_bundle,
            anomaly_bundle=anomaly_bundle,
        )

    # -------------------------------------------------------------------------
    # TAB 2: Manual Simulation
    # -------------------------------------------------------------------------
    with tab_manual:
        render_manual_simulation(
            forecast_bundle=forecast_bundle,
            anomaly_bundle=anomaly_bundle,
            tz_name=selected_tz,
        )

    # -------------------------------------------------------------------------
    # TAB 3: Models & Architecture
    # -------------------------------------------------------------------------
    with tab_models:
        render_models_and_about(
            forecast_bundle=forecast_bundle,
            anomaly_bundle=anomaly_bundle,
            metadata=metadata,
        )


# -----------------------------------------------------------------------------
# Fragment: Live Dashboard
# -----------------------------------------------------------------------------
def render_live_dashboard(
    channel_id: str,
    read_key: str,
    history_size: int,
    auto_refresh_sec: int,
    tz_name: str,
    stale_threshold_min: int,
    forecast_bundle: Dict[str, Any],
    anomaly_bundle: Dict[str, Any],
) -> None:
    """Encapsulates Tab 1 in an auto-refreshing Streamlit fragment."""
    interval = auto_refresh_sec if auto_refresh_sec > 0 else None

    @st.fragment(run_every=interval)
    def _fragment_live_content():
        if not channel_id:
            render_header(
                connection_state="offline",
                age_seconds=None,
            )
            render_empty_state(
                title="ThingSpeak Channel ID Required",
                message="Please configure a valid numeric ThingSpeak Channel ID in the sidebar to stream real-time greenhouse microclimate telemetry.",
            )
            return

        # Fetch telemetry
        df = fetch_thingspeak_data(channel_id, read_key, results=history_size)

        if df.empty:
            render_header(
                connection_state="offline",
                age_seconds=None,
            )
            render_empty_state(
                title="No Telemetry Received",
                message="ThingSpeak returned no records. Verify that your ESP32 Wokwi simulation is running and broadcasting to this channel.",
            )
            return

        # Check stale data using strictly tz-aware UTC comparison
        device_stale = is_data_stale(df, max_minutes=float(stale_threshold_min))
        age_sec = get_latest_reading_age(df)
        age_sec_int = int(age_sec) if age_sec != float("inf") else 0

        # Connection state for header pill
        if device_stale:
            conn_state = "offline"
        elif age_sec > 60:
            conn_state = "delayed"
        else:
            conn_state = "live"

        # Render Header
        render_header(
            title="Smart Greenhouse AIoT",
            subtitle="Edge ESP32 Sensing • ThingSpeak Cloud Telemetry • 1-Hour Forecast ML • Isolation Forest Anomaly Detection",
            connection_state=conn_state,
            age_seconds=age_sec,
        )

        latest_reading = df.iloc[-1]
        curr_t = float(latest_reading["T"])
        curr_rh = float(latest_reading["RH"])
        curr_alert = int(latest_reading.get("alert", 0))
        latest_ts_raw = latest_reading["ts"]

        # Parse UTC timestamp for display conversions
        if isinstance(latest_ts_raw, pd.Timestamp):
            ts_utc = latest_ts_raw.tz_localize("UTC") if latest_ts_raw.tzinfo is None else latest_ts_raw.tz_convert("UTC")
        elif isinstance(latest_ts_raw, datetime):
            ts_utc = latest_ts_raw.replace(tzinfo=timezone.utc) if latest_ts_raw.tzinfo is None else latest_ts_raw.astimezone(timezone.utc)
        else:
            ts_utc = pd.to_datetime(latest_ts_raw, utc=True)

        local_time_str = ts_utc.tz_convert(tz_name).strftime("%H:%M:%S")
        utc_time_str = ts_utc.strftime("%H:%M:%S UTC")
        time_summary = f"Last Reading: {local_time_str} ({tz_name}) • {utc_time_str}"

        # Run Anomaly Detection
        anomaly_res = score_reading(curr_t, curr_rh, anomaly_bundle)

        # Run 1-Hour Forecast
        forecast_res = None
        forecast_err = None
        try:
            forecast_res = predict_next_hour(df, forecast_bundle, tz=tz_name)
        except Exception as exc:
            forecast_err = str(exc)

        # Status Banner Evaluation
        forecast_alert = forecast_res["alert_level"] if forecast_res else 0
        is_critical = (curr_alert == 2) or (forecast_alert == 2) or (anomaly_res.status == "Anomaly")
        is_warning = (curr_alert == 1) or (forecast_alert == 1)

        stale_notice = ""
        if device_stale:
            stale_notice = f"Telemetry is delayed ({age_sec_int}s since last upload, exceeding the {stale_threshold_min}m threshold). Displaying last verified sensor measurements."

        if is_critical:
            banner_state = "critical"
            banner_title = "CRITICAL RISK / CLIMATE ANOMALY DETECTED"
            messages = []
            if anomaly_res.status == "Anomaly":
                messages.append(f"Statistical climate anomaly flagged by Isolation Forest (Severity: {anomaly_res.severity}).")
            if curr_alert == 2:
                messages.append(f"Current temperature ({curr_t:.1f} °C) is outside safe survival thresholds.")
            if forecast_alert == 2 and forecast_res:
                messages.append(f"Forecast (+1h) predicts critical breach ({forecast_res['prediction']:.1f} °C).")
            if not messages:
                messages.append("Microclimate parameters have exceeded critical tolerance limits.")
            banner_msg = " ".join(messages)
        elif is_warning:
            banner_state = "warning"
            banner_title = "WARNING: ENVIRONMENTAL CAUTION"
            messages = []
            if curr_alert == 1:
                messages.append(f"ESP32 warning threshold reached (T: {curr_t:.1f} °C, RH: {curr_rh:.1f} %).")
            if forecast_alert == 1 and forecast_res:
                messages.append(f"Forecast (+1h) projects caution boundary ({forecast_res['prediction']:.1f} °C).")
            banner_msg = " ".join(messages) if messages else "Greenhouse microclimate is approaching environmental limit bands."
        else:
            banner_state = "normal"
            banner_title = "ALL NORMAL: OPTIMAL GREENHOUSE MICROCLIMATE"
            banner_msg = "Current temperature, humidity, VPD, and 1-hour projected forecast are within safe agronomic parameters."

        # Render Status Banner
        render_status_banner(
            state=banner_state,
            title=banner_title,
            message=banner_msg,
            time_summary=time_summary,
            stale_notice=stale_notice,
        )

        # Build KPI Cards
        delta_forecast = (forecast_res["prediction"] - curr_t) if forecast_res else 0.0
        fc_arrow = "up" if delta_forecast > 0 else ("down" if delta_forecast < 0 else None)

        kpi_cards = [
            {
                "icon_svg": SVG_THERMOMETER,
                "icon_bg": "rgba(251, 146, 60, 0.15)",
                "icon_color": COLOR_TEMP,
                "label": "TEMPERATURE",
                "value": f"{curr_t:.1f}",
                "unit": "°C",
                "subtext": "Stale" if device_stale else "Current Measure",
                "subtext_type": "danger" if device_stale else "neutral",
            },
            {
                "icon_svg": SVG_DROP,
                "icon_bg": "rgba(56, 189, 248, 0.15)",
                "icon_color": COLOR_HUMIDITY,
                "label": "HUMIDITY",
                "value": f"{curr_rh:.1f}",
                "unit": "%",
                "subtext": "Stale" if device_stale else "Current Measure",
                "subtext_type": "danger" if device_stale else "neutral",
            },
            {
                "icon_svg": SVG_VPD,
                "icon_bg": "rgba(52, 211, 153, 0.15)",
                "icon_color": COLOR_NORMAL,
                "label": "VPD",
                "value": f"{anomaly_res.vpd:.2f}",
                "unit": "kPa",
                "subtext": "Optimal: 0.8 - 1.2",
                "subtext_type": "success" if (0.8 <= anomaly_res.vpd <= 1.2) else "neutral",
            },
            {
                "icon_svg": SVG_ALERT,
                "icon_bg": "rgba(248, 113, 113, 0.15)" if curr_alert == 2 else ("rgba(251, 191, 36, 0.15)" if curr_alert == 1 else "rgba(52, 211, 153, 0.15)"),
                "icon_color": COLOR_CRITICAL if curr_alert == 2 else (COLOR_WARNING if curr_alert == 1 else COLOR_NORMAL),
                "label": "DEVICE ALERT",
                "value": ALERT_LABELS.get(curr_alert, f"Level {curr_alert}"),
                "unit": "",
                "subtext": "Normal" if curr_alert == 0 else f"+{curr_alert} Severity",
                "subtext_type": "success" if curr_alert == 0 else ("danger" if curr_alert == 2 else "warning"),
            },
            {
                "icon_svg": SVG_FORECAST,
                "icon_bg": "rgba(167, 139, 250, 0.15)",
                "icon_color": COLOR_FORECAST,
                "label": "FORECAST (+1H)",
                "value": f"{forecast_res['prediction']:.1f}" if forecast_res else "--",
                "unit": "°C" if forecast_res else "",
                "subtext": f"{delta_forecast:+.1f}° vs now" if forecast_res else "N/A",
                "subtext_type": "violet",
                "arrow": fc_arrow,
            },
            {
                "icon_svg": SVG_SHIELD,
                "icon_bg": "rgba(248, 113, 113, 0.15)" if (forecast_res and forecast_res['alert_level'] == 2) else ("rgba(251, 191, 36, 0.15)" if (forecast_res and forecast_res['alert_level'] == 1) else "rgba(52, 211, 153, 0.15)"),
                "icon_color": COLOR_CRITICAL if (forecast_res and forecast_res['alert_level'] == 2) else (COLOR_WARNING if (forecast_res and forecast_res['alert_level'] == 1) else COLOR_NORMAL),
                "label": "FORECAST ALERT",
                "value": forecast_res["alert_name"] if forecast_res else "--",
                "unit": "",
                "subtext": "Safe Horizon" if (forecast_res and forecast_res["alert_level"] == 0) else "Risk Horizon",
                "subtext_type": "success" if (forecast_res and forecast_res["alert_level"] == 0) else "warning",
            },
            {
                "icon_svg": SVG_CHECK if anomaly_res.status == "Normal" else SVG_ALERT,
                "icon_bg": "rgba(52, 211, 153, 0.15)" if anomaly_res.status == "Normal" else "rgba(248, 113, 113, 0.15)",
                "icon_color": COLOR_NORMAL if anomaly_res.status == "Normal" else COLOR_CRITICAL,
                "label": "ANOMALY STATUS",
                "value": anomaly_res.status,
                "unit": "",
                "subtext": f"Score {anomaly_res.score:.2f}",
                "subtext_type": "success" if anomaly_res.status == "Normal" else "danger",
            },
        ]

        # Render KPI Cards Grid (Built as ONE string in ui.py)
        render_kpi_grid(kpi_cards, grid_cols=7)

        # --- TIME RANGE SELECTOR & PLOTLY CHARTS ---
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([1, 4])
        with r_col1:
            render_html("<span style='font-size:0.8rem; font-weight:600; text-transform:uppercase; color:#8FA39A; letter-spacing:0.05em;'>Chart Range</span>")
        with r_col2:
            time_window = st.radio(
                "Time Window",
                options=["30 min", "1 hour", "3 hours", "All"],
                index=3,
                horizontal=True,
                label_visibility="collapsed",
                key="chart_time_window",
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            render_chart_header(
                title="Temperature History & +1h Forecast",
                icon_svg=SVG_THERMOMETER,
                icon_color=COLOR_TEMP,
                badge_text=f"Now: {curr_t:.1f} °C",
                badge_bg="rgba(251,146,60,0.12)",
            )
            fig_temp = build_temperature_chart(df, forecast_res, tz_name=tz_name, time_window=time_window)
            st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

        with chart_col2:
            render_chart_header(
                title="Relative Humidity History",
                icon_svg=SVG_DROP,
                icon_color=COLOR_HUMIDITY,
                badge_text=f"Now: {curr_rh:.1f} %",
                badge_bg="rgba(56,189,248,0.12)",
            )
            fig_rh = build_humidity_chart(df, tz_name=tz_name, time_window=time_window)
            st.plotly_chart(fig_rh, use_container_width=True, config={"displayModeBar": False})

        # --- ANOMALY GAUGE CARD & FORECAST MODEL INFO ---
        inf_col1, inf_col2 = st.columns(2)

        with inf_col1:
            render_html(f"""
                <div class="detail-card-title">
                    <span style="color:{COLOR_NORMAL};">{SVG_SHIELD.strip()}</span>
                    <span>Climate Anomaly Scoring (Isolation Forest)</span>
                </div>
            """)
            gauge_fig = build_anomaly_gauge(
                score=anomaly_res.score,
                threshold=anomaly_res.threshold,
                high_threshold=anomaly_res.threshold,
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
            render_anomaly_chips(
                score=anomaly_res.score,
                threshold=anomaly_res.threshold,
                status=anomaly_res.status,
                severity=anomaly_res.severity,
                unusual_features=anomaly_res.unusual_features,
            )

        with inf_col2:
            render_forecast_diagnostics_card(forecast_res, forecast_err)

        # --- EXPANDER: RECENT TELEMETRY TABLE ---
        with st.expander("Recent Telemetry Log (Latest 50 Readings)", expanded=False):
            display_df = df.tail(50).copy()
            if display_df["ts"].dt.tz is not None:
                display_df[f"Time ({tz_name})"] = display_df["ts"].dt.tz_convert(tz_name).dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                display_df[f"Time ({tz_name})"] = display_df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_name).dt.strftime("%Y-%m-%d %H:%M:%S")

            display_df["Alert Label"] = display_df["alert"].map(ALERT_LABELS)
            table_cols = [f"Time ({tz_name})", "T", "RH", "alert", "Alert Label"]
            st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Channel CSV",
                data=csv_data,
                file_name=f"greenhouse_readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon=":material/download:",
            )

    _fragment_live_content()


# -----------------------------------------------------------------------------
# TAB 2: Manual Simulation (Offline Exploration)
# -----------------------------------------------------------------------------
def render_manual_simulation(
    forecast_bundle: Dict[str, Any],
    anomaly_bundle: Dict[str, Any],
    tz_name: str,
) -> None:
    """Render interactive manual simulation tab for offline demonstrations."""
    render_html("""
        <div style="margin-bottom:16px;">
            <div style="font-size:1.2rem; font-weight:700; color:#E6F0EA;">Manual Microclimate Simulator</div>
            <div style="font-size:0.84rem; color:#8FA39A;">
                Simulate arbitrary temperature, humidity, and diurnal conditions to stress-test the ML forecast and Isolation Forest anomaly engines offline.
            </div>
        </div>
    """)

    # Preset button bar
    render_html("<span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#8FA39A; letter-spacing:0.05em;'>Quick Presets</span>")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)

    if "sim_temp" not in st.session_state:
        st.session_state["sim_temp"] = 24.0
    if "sim_rh" not in st.session_state:
        st.session_state["sim_rh"] = 65.0
    if "sim_hour" not in st.session_state:
        st.session_state["sim_hour"] = 14.0

    if pcol1.button("Normal Day (24 °C / 65 %)", icon=":material/sunny:", use_container_width=True):
        st.session_state["sim_temp"] = 24.0
        st.session_state["sim_rh"] = 65.0
        st.session_state["sim_hour"] = 12.0
    if pcol2.button("Heat Wave (36 °C / 30 %)", icon=":material/local_fire_department:", use_container_width=True):
        st.session_state["sim_temp"] = 36.0
        st.session_state["sim_rh"] = 30.0
        st.session_state["sim_hour"] = 15.0
    if pcol3.button("Frost Hazard (1 °C / 80 %)", icon=":material/ac_unit:", use_container_width=True):
        st.session_state["sim_temp"] = 1.0
        st.session_state["sim_rh"] = 80.0
        st.session_state["sim_hour"] = 5.0
    if pcol4.button("High Humidity (28 °C / 95 %)", icon=":material/cloud:", use_container_width=True):
        st.session_state["sim_temp"] = 28.0
        st.session_state["sim_rh"] = 95.0
        st.session_state["sim_hour"] = 18.0

    # Simulation Sliders
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        sim_t = st.slider(
            "Simulated Temperature (°C)",
            min_value=0.0,
            max_value=45.0,
            value=float(st.session_state["sim_temp"]),
            step=0.5,
            key="slider_sim_temp",
        )
    with scol2:
        sim_rh = st.slider(
            "Simulated Humidity (%)",
            min_value=10.0,
            max_value=100.0,
            value=float(st.session_state["sim_rh"]),
            step=1.0,
            key="slider_sim_rh",
        )
    with scol3:
        sim_h = st.slider(
            "Hour of Day (0.0 - 23.5)",
            min_value=0.0,
            max_value=23.5,
            value=float(st.session_state["sim_hour"]),
            step=0.5,
            key="slider_sim_hour",
        )

    st.session_state["sim_temp"] = sim_t
    st.session_state["sim_rh"] = sim_rh
    st.session_state["sim_hour"] = sim_h

    # Build simulated dataframe
    hour_int = int(sim_h)
    min_int = int((sim_h - hour_int) * 60)
    sim_time = pd.Timestamp.now(tz=timezone.utc).floor("min").replace(hour=hour_int, minute=min_int)
    sim_df = pd.DataFrame([{"ts": sim_time, "T": sim_t, "RH": sim_rh}])

    sim_anomaly = score_reading(sim_t, sim_rh, anomaly_bundle)
    sim_esp32_alert = calculate_esp32_alert(sim_t, sim_rh)

    try:
        sim_forecast = predict_next_hour(sim_df, forecast_bundle, tz=tz_name)
    except Exception as exc:
        sim_forecast = None

    # Status Banner for Simulation
    sim_fc_alert = sim_forecast["alert_level"] if sim_forecast else 0
    sim_is_crit = (sim_esp32_alert == 2) or (sim_fc_alert == 2) or (sim_anomaly.status == "Anomaly")
    sim_is_warn = (sim_esp32_alert == 1) or (sim_fc_alert == 1)

    if sim_is_crit:
        s_banner_state = "critical"
        s_banner_title = "SIMULATION: CRITICAL THRESHOLD BREACH"
        s_msg = f"Simulated microclimate parameters (T: {sim_t:.1f} °C, RH: {sim_rh:.1f} %) exceed safe operational thresholds."
    elif sim_is_warn:
        s_banner_state = "warning"
        s_banner_title = "SIMULATION: CAUTION BOUNDARY REACHED"
        s_msg = f"Simulated inputs approach caution tolerance boundaries (T: {sim_t:.1f} °C, RH: {sim_rh:.1f} %)."
    else:
        s_banner_state = "normal"
        s_banner_title = "SIMULATION: OPTIMAL CLIMATE CONDITIONS"
        s_msg = f"Simulated microclimate parameters (T: {sim_t:.1f} °C, RH: {sim_rh:.1f} %, Hour: {hour_int:02d}:{min_int:02d}) are nominal."

    render_status_banner(
        state=s_banner_state,
        title=s_banner_title,
        message=s_msg,
        time_summary=f"Simulated Diurnal Time: {hour_int:02d}:{min_int:02d}",
    )

    # Simulation KPI cards (5 cards)
    delta_sim = (sim_forecast["prediction"] - sim_t) if sim_forecast else 0.0
    sim_arrow = "up" if delta_sim > 0 else ("down" if delta_sim < 0 else None)

    sim_kpis = [
        {
            "icon_svg": SVG_THERMOMETER,
            "icon_bg": "rgba(251, 146, 60, 0.15)",
            "icon_color": COLOR_TEMP,
            "label": "SIMULATED TEMP",
            "value": f"{sim_t:.1f}",
            "unit": "°C",
            "subtext": "Input Value",
            "subtext_type": "neutral",
        },
        {
            "icon_svg": SVG_DROP,
            "icon_bg": "rgba(56, 189, 248, 0.15)",
            "icon_color": COLOR_HUMIDITY,
            "label": "SIMULATED RH",
            "value": f"{sim_rh:.1f}",
            "unit": "%",
            "subtext": "Input Value",
            "subtext_type": "neutral",
        },
        {
            "icon_svg": SVG_VPD,
            "icon_bg": "rgba(52, 211, 153, 0.15)",
            "icon_color": COLOR_NORMAL,
            "label": "CALCULATED VPD",
            "value": f"{sim_anomaly.vpd:.2f}",
            "unit": "kPa",
            "subtext": "Optimal: 0.8 - 1.2",
            "subtext_type": "success" if (0.8 <= sim_anomaly.vpd <= 1.2) else "neutral",
        },
        {
            "icon_svg": SVG_ALERT,
            "icon_bg": "rgba(248, 113, 113, 0.15)" if sim_esp32_alert == 2 else ("rgba(251, 191, 36, 0.15)" if sim_esp32_alert == 1 else "rgba(52, 211, 153, 0.15)"),
            "icon_color": COLOR_CRITICAL if sim_esp32_alert == 2 else (COLOR_WARNING if sim_esp32_alert == 1 else COLOR_NORMAL),
            "label": "ESP32 ALERT RULE",
            "value": ALERT_LABELS.get(sim_esp32_alert, "Unknown"),
            "unit": "",
            "subtext": "Normal" if sim_esp32_alert == 0 else f"+{sim_esp32_alert} Level",
            "subtext_type": "success" if sim_esp32_alert == 0 else ("danger" if sim_esp32_alert == 2 else "warning"),
        },
        {
            "icon_svg": SVG_FORECAST,
            "icon_bg": "rgba(167, 139, 250, 0.15)",
            "icon_color": COLOR_FORECAST,
            "label": "FORECAST (+1H)",
            "value": f"{sim_forecast['prediction']:.1f}" if sim_forecast else "--",
            "unit": "°C" if sim_forecast else "",
            "subtext": f"{delta_sim:+.1f}° vs simulated" if sim_forecast else "N/A",
            "subtext_type": "violet",
            "arrow": sim_arrow,
        },
    ]

    render_kpi_grid(sim_kpis, grid_cols=5)

    # Detailed Evaluation Columns
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        render_html(f"""
            <div class="detail-card-title">
                <span style="color:{COLOR_NORMAL};">{SVG_SHIELD.strip()}</span>
                <span>Simulated Anomaly Scoring</span>
            </div>
        """)
        sim_gauge_fig = build_anomaly_gauge(
            score=sim_anomaly.score,
            threshold=sim_anomaly.threshold,
            high_threshold=sim_anomaly.threshold,
        )
        st.plotly_chart(sim_gauge_fig, use_container_width=True, config={"displayModeBar": False})
        render_anomaly_chips(
            score=sim_anomaly.score,
            threshold=sim_anomaly.threshold,
            status=sim_anomaly.status,
            severity=sim_anomaly.severity,
            unusual_features=sim_anomaly.unusual_features,
        )

    with dcol2:
        if sim_forecast:
            render_html(f"""
                <div class="detail-card">
                    <div class="detail-card-title">
                        <span style="color:{COLOR_FORECAST};">{SVG_FORECAST.strip()}</span>
                        <span>Simulated Forecast Diagnostics</span>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">
                        <div style="background:#0E1613; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px;">
                            <div style="font-size:0.7rem; color:#8FA39A; text-transform:uppercase; font-weight:600;">Projected Target</div>
                            <div style="font-size:0.92rem; font-weight:600; color:{COLOR_FORECAST}; margin-top:2px;">{sim_forecast['forecast_time'].strftime('%H:%M')}</div>
                        </div>
                        <div style="background:#0E1613; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px;">
                            <div style="font-size:0.7rem; color:#8FA39A; text-transform:uppercase; font-weight:600;">Forecast Alert</div>
                            <div style="font-size:0.92rem; font-weight:600; color:#E6F0EA; margin-top:2px;">{sim_forecast['alert_name']} (L{sim_forecast['alert_level']})</div>
                        </div>
                    </div>
                    <div style="background:rgba(167,139,250,0.08); border-left:3px solid #A78BFA; border-radius:6px; padding:10px 12px; font-size:0.82rem; color:#E6F0EA; line-height:1.45;">
                        Operating via <b>{sim_forecast['model_used']}</b>: Computes the 1-hour horizon from simulated point inputs using cyclical trigonometric time harmonics (sin/cos of hour).
                    </div>
                </div>
            """)


# -----------------------------------------------------------------------------
# TAB 3: Models & Architecture
# -----------------------------------------------------------------------------
def render_models_and_about(
    forecast_bundle: Dict[str, Any],
    anomaly_bundle: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
) -> None:
    """Render comprehensive model metrics, architecture flow diagram, and documentation."""
    render_html("""
        <div style="margin-bottom:16px;">
            <div style="font-size:1.2rem; font-weight:700; color:#E6F0EA;">System Architecture & Machine Learning Specifications</div>
            <div style="font-size:0.84rem; color:#8FA39A;">
                End-to-end telemetry pipeline, predictive models, Isolation Forest anomaly scoring, and empirical benchmark evaluations.
            </div>
        </div>
    """)

    # 1. Architecture Flow Diagram
    render_html("<span style='font-size:0.8rem; font-weight:700; text-transform:uppercase; color:#8FA39A; letter-spacing:0.06em;'>End-to-End System Pipeline</span>")
    render_architecture_flow()

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # 2. Forecast Model Specifications
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        f_metrics = forecast_bundle.get("test_metrics", {})
        stat_cards_html = ""
        if f_metrics:
            cards_list = []
            for m_name, vals in f_metrics.items():
                if isinstance(vals, dict):
                    cards_list.append(f'<div class="stat-card"><div class="stat-card-label">{m_name} MAE</div><div class="stat-card-value" style="color:{COLOR_TEMP};">{vals.get("mae", 0.0):.3f}°C</div></div>')
                    cards_list.append(f'<div class="stat-card"><div class="stat-card-label">{m_name} RMSE</div><div class="stat-card-value" style="color:{COLOR_WARNING};">{vals.get("rmse", 0.0):.3f}°C</div></div>')
                    cards_list.append(f'<div class="stat-card"><div class="stat-card-label">{m_name} R²</div><div class="stat-card-value" style="color:{COLOR_NORMAL};">{vals.get("r2", 0.0):.4f}</div></div>')
            stat_cards_html = f'<div class="stat-card-grid">{"".join(cards_list)}</div>'

        t_limits = forecast_bundle.get("thresholds", {})
        limits_html = (
            f'<div style="margin-top:12px; font-size:0.8rem; color:#8FA39A;">'
            f'<b>Alert Thresholds:</b> Critical &le; {t_limits.get("crit_low", 2.0)} °C or &ge; {t_limits.get("crit_high", 35.0)} °C • '
            f'Warning &le; {t_limits.get("warn_low", 5.0)} °C or &ge; {t_limits.get("warn_high", 30.0)} °C'
            f'</div>'
        )

        render_html(f"""
            <div class="detail-card">
                <div class="detail-card-title">
                    <span style="color:{COLOR_FORECAST};">{SVG_FORECAST.strip()}</span>
                    <span>1. Temperature Forecast Model (1-Hour Horizon)</span>
                </div>
                <div style="font-size:0.84rem; color:#E6F0EA; line-height:1.5;">
                    <p><b>Dual-Model Dynamic Switching:</b></p>
                    <ul style="padding-left:18px; color:#8FA39A;">
                        <li><b style="color:#E6F0EA;">Instant Model ({forecast_bundle.get('name_instant', 'Linear/Ridge')}):</b> Active when &lt; 70 min of history exists. Utilizes instantaneous T, RH, and solar diurnal cycle harmonics (sin/cos).</li>
                        <li><b style="color:#E6F0EA;">Lag Model ({forecast_bundle.get('name_lag', 'Ridge/Lasso')}):</b> Automatically engaged when &ge; 70 min of continuous resampled telemetry is available. Incorporates 10m, 30m, and 60m lag momentum.</li>
                        <li><b>Resampling Step:</b> {forecast_bundle.get('step_minutes', 10)} minutes</li>
                        <li><b>Forecast Horizon:</b> {forecast_bundle.get('horizon_minutes', 60)} minutes</li>
                    </ul>
                </div>
                {stat_cards_html}
                {limits_html}
            </div>
        """)

    # 3. Anomaly Model Specifications
    with fcol2:
        stat_cards_a_html = ""
        if metadata and "metrics" in metadata:
            c_metrics = metadata["metrics"].get("climate", {})
            e_metrics = metadata["metrics"].get("climate_event_recall", {})
            stat_cards_a_html = f"""
            <div class="stat-card-grid">
                <div class="stat-card"><div class="stat-card-label">Precision</div><div class="stat-card-value" style="color:{COLOR_NORMAL};">{c_metrics.get('precision', 0.0) * 100:.1f}%</div></div>
                <div class="stat-card"><div class="stat-card-label">Recall</div><div class="stat-card-value" style="color:{COLOR_NORMAL};">{c_metrics.get('recall', 0.0) * 100:.1f}%</div></div>
                <div class="stat-card"><div class="stat-card-label">F1-Score</div><div class="stat-card-value" style="color:{COLOR_FORECAST};">{c_metrics.get('f1', 0.0) * 100:.1f}%</div></div>
                <div class="stat-card"><div class="stat-card-label">Event Recall</div><div class="stat-card-value" style="color:{COLOR_TEMP};">{e_metrics.get('recall', 0.0) * 100:.1f}%</div></div>
            </div>
            """

        render_html(f"""
            <div class="detail-card">
                <div class="detail-card-title">
                    <span style="color:{COLOR_NORMAL};">{SVG_SHIELD.strip()}</span>
                    <span>2. Climate Anomaly Detection (Isolation Forest)</span>
                </div>
                <div style="font-size:0.84rem; color:#E6F0EA; line-height:1.5;">
                    <p><b>Pointwise Multidimensional Inference:</b></p>
                    <ul style="padding-left:18px; color:#8FA39A;">
                        <li><b style="color:#E6F0EA;">Algorithm:</b> Isolation Forest (scikit-learn)</li>
                        <li><b style="color:#E6F0EA;">Input Dimensions:</b> Temperature (T), Relative Humidity (RH), Vapor Pressure Deficit (VPD)</li>
                        <li><b style="color:#E6F0EA;">Threshold:</b> {anomaly_bundle.get('threshold', 0.6315):.4f}</li>
                        <li><b style="color:#E6F0EA;">High Severity Threshold:</b> {anomaly_bundle.get('high_severity_threshold', 0.6315):.4f}</li>
                        <li><b style="color:#E6F0EA;">VPD Formula:</b> <code>VPD = 0.6108*exp(17.27*T/(T+237.3))*(1-RH/100)</code></li>
                    </ul>
                </div>
                {stat_cards_a_html}
                <div style="margin-top:12px; font-size:0.78rem; color:#8FA39A; font-style:italic;">
                    Unusual features are derived from the highest absolute standardized scores post-scaling. They indicate statistical deviation, not agronomic causality.
                </div>
            </div>
        """)

    # 4. Limitations and Warnings Note
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    render_html("""
        <div style="background:#121C18; border:1px solid rgba(255,255,255,0.06); border-left:4px solid #FBBF24; border-radius:12px; padding:16px 20px;">
            <div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#FBBF24; font-size:0.92rem; margin-bottom:6px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                <span>Methodological Limitations & Evaluation Notice</span>
            </div>
            <div style="font-size:0.84rem; color:#E6F0EA; line-height:1.5;">
                The precision, recall, and event detection metrics cited above were evaluated on a controlled synthetic climate fault-injection stress test on held-out chronological data. They do not represent real-world agronomic ground truth. The model was trained without manual anomaly labels, operating on the calibration assumption that baseline microclimate intervals were nominal.
            </div>
        </div>
    """)

    if metadata and "warnings" in metadata:
        with st.expander("Detailed Engineering Warnings & Assumptions", expanded=False):
            for w in metadata["warnings"]:
                st.markdown(f"- {w}")


if __name__ == "__main__":
    main()
