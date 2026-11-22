"""Telegram risk notification module for Smart Greenhouse AIoT.

Provides real-time alerting for greenhouse microclimate risks:
- Triggered when device alert level >= 1, climate anomaly detected, or forecast alert is critical (level 2).
- Does NOT trigger on forecast alert level 1 (warning).
- 10-minute global cooldown preventing message spam across multiple client sessions.
- Automatically sends a 'back to normal' recovery message when conditions stabilize.
- HTML formatted payloads via Telegram Bot API with zero emojis.
- Hardened: credentials read only through get_secret, no sensitive logging, no UI test controls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests

from config import get_secret, log_error

# Global module-level state for spam prevention across client sessions
_NOTIFICATION_STATE: Dict[str, Any] = {
    "last_sent_time": None,      # UTC datetime of last sent alert
    "is_in_alert": False,        # Boolean flag whether greenhouse is currently in alert
    "last_reasons": [],          # Cached list of reasons from the latest active alert
    "cooldown_seconds": 600,     # 10 minutes cooldown between alerts
}


def get_telegram_credentials() -> Tuple[str, str]:
    """Retrieve Telegram bot token and chat ID safely via get_secret."""
    token = get_secret("TELEGRAM_BOT_TOKEN")
    chat_id = get_secret("TELEGRAM_CHAT_ID")
    return token, chat_id


def is_telegram_configured() -> bool:
    """Return True if both Telegram bot token and chat ID are configured in secrets or env."""
    token, chat_id = get_telegram_credentials()
    return bool(token and chat_id)


def send_telegram(text: str) -> bool:
    """Send an HTML-formatted message via Telegram Bot API over HTTPS.

    Never logs tokens, chat IDs, or request URLs. Catches all exceptions cleanly.

    Parameters
    ----------
    text : str
        The HTML formatted text to send.

    Returns
    -------
    bool
        True if the message was sent successfully, False otherwise.
    """
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return bool(data.get("ok", False))
        else:
            log_error(f"TelegramStatus_{response.status_code}")
            return False
    except Exception as exc:
        log_error(type(exc).__name__)
        return False


def evaluate_risk_conditions(
    device_alert: int,
    anomaly_status: str,
    anomaly_severity: str,
    forecast_alert_level: int,
    forecast_temp: Optional[float] = None,
    current_temp: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether current microclimate telemetry triggers an alert.

    Rules:
    - Trigger if device_alert >= 1 (warning or critical from ESP32).
    - Trigger if anomaly_status == 'Anomaly' (Isolation Forest).
    - Trigger if forecast_alert_level == 2 (Critical forecast limit).
    - Do NOT trigger on forecast_alert_level == 1 (Warning forecast).

    Returns
    -------
    Tuple[bool, List[str]]
        (should_trigger, list_of_reasons)
    """
    reasons: List[str] = []

    # 1. Device alert rule (field3 >= 1)
    if device_alert == 2:
        reasons.append("ESP32 Device Alert: Critical Level 2 (Extreme microclimate condition)")
    elif device_alert >= 1:
        reasons.append("ESP32 Device Alert: Warning Level 1 (Boundary limit exceeded)")

    # 2. Climate Anomaly rule
    if str(anomaly_status).strip().lower() == "anomaly":
        sev = str(anomaly_severity).strip().upper() if anomaly_severity else "DETECTED"
        reasons.append(f"Climate Anomaly flagged by Isolation Forest (Severity: {sev})")

    # 3. Forecast Critical rule (Level 2 only; do NOT trigger on Level 1 warning)
    if forecast_alert_level == 2:
        fc_detail = f" ({forecast_temp:.1f} °C)" if forecast_temp is not None else ""
        reasons.append(f"1-Hour Forecast predicts CRITICAL condition{fc_detail}")

    should_alert = len(reasons) > 0
    return should_alert, reasons


def format_alert_message(
    reasons: List[str],
    current_temp: float,
    current_rh: float,
    vpd: float,
    forecast_temp: Optional[float],
    forecast_alert_name: Optional[str],
    anomaly_score: float,
    anomaly_threshold: float,
    local_time_str: str,
    tz_name: str,
) -> str:
    """Format structured HTML risk alert message without emojis."""
    fc_str = f"{forecast_temp:.1f} °C" if forecast_temp is not None else "N/A"
    if forecast_alert_name:
        fc_str += f" ({forecast_alert_name})"

    reasons_formatted = "\n".join([f"• {r}" for r in reasons])

    msg = (
        f"<b>[RISK ALERT] Smart Greenhouse AIoT</b>\n"
        f"<b>Time:</b> {local_time_str} ({tz_name})\n"
        f"<b>Status:</b> Risk Threshold Breached\n\n"
        f"<b>Trigger Reasons:</b>\n"
        f"{reasons_formatted}\n\n"
        f"<b>Current Sensor Telemetry:</b>\n"
        f"• Temperature: {current_temp:.1f} °C\n"
        f"• Relative Humidity: {current_rh:.1f} %\n"
        f"• VPD: {vpd:.2f} kPa\n\n"
        f"<b>Predictive & Anomaly Diagnostics:</b>\n"
        f"• 1-Hour Forecast: {fc_str}\n"
        f"• Anomaly Score: {anomaly_score:.4f} (Threshold: {anomaly_threshold:.4f})"
    )
    return msg


def format_resolved_message(
    current_temp: float,
    current_rh: float,
    vpd: float,
    forecast_temp: Optional[float],
    anomaly_score: float,
    anomaly_threshold: float,
    local_time_str: str,
    tz_name: str,
) -> str:
    """Format structured HTML back-to-normal message without emojis."""
    fc_str = f"{forecast_temp:.1f} °C" if forecast_temp is not None else "N/A"

    msg = (
        f"<b>[RESOLVED] Smart Greenhouse AIoT</b>\n"
        f"<b>Time:</b> {local_time_str} ({tz_name})\n"
        f"<b>Status:</b> Microclimate Returned to Normal\n\n"
        f"All monitored parameters (Temperature, Humidity, VPD, and 1-hour forecast) "
        f"have returned within safe agronomic tolerance boundaries.\n\n"
        f"<b>Current Telemetry:</b>\n"
        f"• Temperature: {current_temp:.1f} °C\n"
        f"• Relative Humidity: {current_rh:.1f} %\n"
        f"• VPD: {vpd:.2f} kPa\n"
        f"• 1-Hour Forecast: {fc_str}\n"
        f"• Anomaly Score: {anomaly_score:.4f} (Threshold: {anomaly_threshold:.4f})"
    )
    return msg


def format_test_message(local_time_str: str, tz_name: str) -> str:
    """Format structured test message without emojis."""
    msg = (
        f"<b>[TEST] Smart Greenhouse AIoT</b>\n"
        f"<b>Time:</b> {local_time_str} ({tz_name})\n\n"
        f"Telegram notification integration is active and verified.\n"
        f"Automated risk alerts will be dispatched on abnormal conditions."
    )
    return msg


def process_live_telemetry_notifications(
    curr_t: float,
    curr_rh: float,
    curr_alert: int,
    anomaly_res: Any,
    forecast_res: Optional[Dict[str, Any]],
    local_time_str: str,
    tz_name: str,
) -> Optional[str]:
    """Process incoming live telemetry, enforce cooldown, and dispatch Telegram messages.

    Parameters
    ----------
    curr_t : float
        Current temperature in °C.
    curr_rh : float
        Current relative humidity in %.
    curr_alert : int
        ESP32 device alert rule level (0, 1, 2).
    anomaly_res : Any
        Result object from score_reading (has status, severity, score, threshold, vpd).
    forecast_res : Optional[Dict[str, Any]]
        Result dictionary from predict_next_hour (has prediction, alert_level, alert_name).
    local_time_str : str
        Formatted local time string (e.g. HH:MM:SS).
    tz_name : str
        Greenhouse timezone identifier.

    Returns
    -------
    Optional[str]
        Action description if a message was dispatched ('alert_sent', 'resolved_sent', None).
    """
    if not is_telegram_configured():
        return None

    # Extract fields safely
    anom_status = getattr(anomaly_res, "status", "Normal")
    anom_sev = getattr(anomaly_res, "severity", "NORMAL")
    anom_score = getattr(anomaly_res, "score", 0.0)
    anom_th = getattr(anomaly_res, "threshold", 0.6315)
    vpd_val = getattr(anomaly_res, "vpd", 0.0)

    fc_alert_lvl = forecast_res.get("alert_level", 0) if forecast_res else 0
    fc_temp = forecast_res.get("prediction", None) if forecast_res else None
    fc_name = forecast_res.get("alert_name", None) if forecast_res else None

    # Evaluate risk conditions
    is_risk, reasons = evaluate_risk_conditions(
        device_alert=curr_alert,
        anomaly_status=anom_status,
        anomaly_severity=anom_sev,
        forecast_alert_level=fc_alert_lvl,
        forecast_temp=fc_temp,
        current_temp=curr_t,
    )

    now_utc = datetime.now(timezone.utc)
    last_sent = _NOTIFICATION_STATE["last_sent_time"]
    cooldown = _NOTIFICATION_STATE["cooldown_seconds"]
    is_in_alert = _NOTIFICATION_STATE["is_in_alert"]

    # CASE 1: In Risk Condition
    if is_risk:
        # Check cooldown
        can_send = False
        if last_sent is None:
            can_send = True
        else:
            elapsed = (now_utc - last_sent).total_seconds()
            if elapsed >= cooldown:
                can_send = True

        if can_send:
            alert_text = format_alert_message(
                reasons=reasons,
                current_temp=curr_t,
                current_rh=curr_rh,
                vpd=vpd_val,
                forecast_temp=fc_temp,
                forecast_alert_name=fc_name,
                anomaly_score=anom_score,
                anomaly_threshold=anom_th,
                local_time_str=local_time_str,
                tz_name=tz_name,
            )
            success = send_telegram(alert_text)
            if success:
                _NOTIFICATION_STATE["last_sent_time"] = now_utc
                _NOTIFICATION_STATE["is_in_alert"] = True
                _NOTIFICATION_STATE["last_reasons"] = reasons
                return "alert_sent"
        return None

    # CASE 2: Returned to Normal Condition
    elif is_in_alert:
        # Everything returned to normal after an active alert state
        resolved_text = format_resolved_message(
            current_temp=curr_t,
            current_rh=curr_rh,
            vpd=vpd_val,
            forecast_temp=fc_temp,
            anomaly_score=anom_score,
            anomaly_threshold=anom_th,
            local_time_str=local_time_str,
            tz_name=tz_name,
        )
        success = send_telegram(resolved_text)
        if success:
            _NOTIFICATION_STATE["is_in_alert"] = False
            _NOTIFICATION_STATE["last_reasons"] = []
            _NOTIFICATION_STATE["last_sent_time"] = now_utc
            return "resolved_sent"

    return None


def reset_notification_state() -> None:
    """Reset module-level notification state (primarily for automated testing)."""
    _NOTIFICATION_STATE["last_sent_time"] = None
    _NOTIFICATION_STATE["is_in_alert"] = False
    _NOTIFICATION_STATE["last_reasons"] = []
