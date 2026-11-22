"""Unit tests for notifier.py: message formatting, triggers, cooldown, and error handling."""

import os
import requests
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import notifier


class DummyAnomalyResult:
    def __init__(self, status="Normal", severity="NORMAL", score=0.49, threshold=0.6315, vpd=1.12):
        self.status = status
        self.severity = severity
        self.score = score
        self.threshold = threshold
        self.vpd = vpd


def test_trigger_rules():
    print("Testing trigger rules...")

    # Rule 1: Device alert >= 1 triggers
    trig, r = notifier.evaluate_risk_conditions(device_alert=1, anomaly_status="Normal", anomaly_severity="NORMAL", forecast_alert_level=0)
    assert trig is True
    assert any("Warning" in x for x in r)

    trig, r = notifier.evaluate_risk_conditions(device_alert=2, anomaly_status="Normal", anomaly_severity="NORMAL", forecast_alert_level=0)
    assert trig is True
    assert any("Critical" in x for x in r)

    # Rule 2: Anomaly triggers
    trig, r = notifier.evaluate_risk_conditions(device_alert=0, anomaly_status="Anomaly", anomaly_severity="HIGH", forecast_alert_level=0)
    assert trig is True
    assert any("Anomaly" in x for x in r)

    # Rule 3: Forecast Level 2 (Critical) triggers
    trig, r = notifier.evaluate_risk_conditions(device_alert=0, anomaly_status="Normal", anomaly_severity="NORMAL", forecast_alert_level=2, forecast_temp=35.5)
    assert trig is True
    assert any("CRITICAL" in x for x in r)

    # Rule 4: Forecast Level 1 (Warning) MUST NOT trigger!
    trig, r = notifier.evaluate_risk_conditions(device_alert=0, anomaly_status="Normal", anomaly_severity="NORMAL", forecast_alert_level=1, forecast_temp=30.5)
    assert trig is False, "Forecast alert level 1 must NOT trigger an alert!"
    assert len(r) == 0

    # Rule 5: All normal does not trigger
    trig, r = notifier.evaluate_risk_conditions(device_alert=0, anomaly_status="Normal", anomaly_severity="NORMAL", forecast_alert_level=0)
    assert trig is False
    assert len(r) == 0

    print("Trigger rules: All PASSED!")


def test_message_formatting_no_emojis():
    print("Testing message formatting (zero emojis & all required fields)...")

    reasons = ["ESP32 Device Alert: Warning Level 1", "Climate Anomaly flagged by Isolation Forest (Severity: HIGH)"]
    alert_msg = notifier.format_alert_message(
        reasons=reasons,
        current_temp=31.2,
        current_rh=38.5,
        vpd=2.85,
        forecast_temp=33.0,
        forecast_alert_name="Warning",
        anomaly_score=0.7250,
        anomaly_threshold=0.6315,
        local_time_str="22:30:00",
        tz_name="Africa/Algiers",
    )

    # Check required fields
    assert "31.2 °C" in alert_msg
    assert "38.5 %" in alert_msg
    assert "2.85 kPa" in alert_msg
    assert "33.0 °C" in alert_msg
    assert "0.7250" in alert_msg
    assert "0.6315" in alert_msg
    assert "22:30:00" in alert_msg
    assert "Africa/Algiers" in alert_msg
    assert "Trigger Reasons:" in alert_msg

    # Check zero emojis
    import re
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    assert len(emoji_pattern.findall(alert_msg)) == 0, "Alert message must contain no emojis!"

    resolved_msg = notifier.format_resolved_message(
        current_temp=24.0,
        current_rh=65.0,
        vpd=1.05,
        forecast_temp=25.0,
        anomaly_score=0.4850,
        anomaly_threshold=0.6315,
        local_time_str="22:45:00",
        tz_name="Africa/Algiers",
    )
    assert "RESOLVED" in resolved_msg
    assert "24.0 °C" in resolved_msg
    assert len(emoji_pattern.findall(resolved_msg)) == 0, "Resolved message must contain no emojis!"

    print("Message formatting: All PASSED!")


def test_cooldown_and_back_to_normal():
    print("Testing 10-minute cooldown and back-to-normal logic...")
    notifier.reset_notification_state()

    mock_send = MagicMock(return_value=True)

    with patch("notifier.send_telegram", mock_send):
        with patch("notifier.is_telegram_configured", return_value=True):
            # 1. First alert at T=0 -> Should send
            res1 = notifier.process_live_telemetry_notifications(
                curr_t=32.0,
                curr_rh=35.0,
                curr_alert=1,
                anomaly_res=DummyAnomalyResult(),
                forecast_res={"prediction": 32.5, "alert_level": 1, "alert_name": "Warning"},
                local_time_str="10:00:00",
                tz_name="UTC",
            )
            assert res1 == "alert_sent"
            assert mock_send.call_count == 1
            assert notifier._NOTIFICATION_STATE["is_in_alert"] is True

            # 2. Second alert 2 minutes later -> Should be suppressed by cooldown
            t0 = notifier._NOTIFICATION_STATE["last_sent_time"]
            # Simulate 2 minutes later
            notifier._NOTIFICATION_STATE["last_sent_time"] = t0 - timedelta(minutes=2)
            res2 = notifier.process_live_telemetry_notifications(
                curr_t=32.5,
                curr_rh=34.0,
                curr_alert=1,
                anomaly_res=DummyAnomalyResult(),
                forecast_res={"prediction": 33.0, "alert_level": 1, "alert_name": "Warning"},
                local_time_str="10:02:00",
                tz_name="UTC",
            )
            assert res2 is None
            assert mock_send.call_count == 1  # Not incremented

            # 3. Third alert 11 minutes later -> Should send (cooldown expired)
            notifier._NOTIFICATION_STATE["last_sent_time"] = datetime.now(timezone.utc) - timedelta(minutes=11)
            res3 = notifier.process_live_telemetry_notifications(
                curr_t=33.0,
                curr_rh=33.0,
                curr_alert=1,
                anomaly_res=DummyAnomalyResult(),
                forecast_res={"prediction": 33.5, "alert_level": 1, "alert_name": "Warning"},
                local_time_str="10:11:00",
                tz_name="UTC",
            )
            assert res3 == "alert_sent"
            assert mock_send.call_count == 2

            # 4. Telemetry returns to Normal -> Should send ONE back-to-normal message
            res4 = notifier.process_live_telemetry_notifications(
                curr_t=24.0,
                curr_rh=65.0,
                curr_alert=0,
                anomaly_res=DummyAnomalyResult(status="Normal"),
                forecast_res={"prediction": 24.5, "alert_level": 0, "alert_name": "Normal"},
                local_time_str="10:15:00",
                tz_name="UTC",
            )
            assert res4 == "resolved_sent"
            assert mock_send.call_count == 3
            assert notifier._NOTIFICATION_STATE["is_in_alert"] is False

            # 5. Telemetry stays Normal -> Should NOT send duplicate resolved message
            res5 = notifier.process_live_telemetry_notifications(
                curr_t=24.1,
                curr_rh=64.8,
                curr_alert=0,
                anomaly_res=DummyAnomalyResult(status="Normal"),
                forecast_res={"prediction": 24.6, "alert_level": 0, "alert_name": "Normal"},
                local_time_str="10:15:20",
                tz_name="UTC",
            )
            assert res5 is None
            assert mock_send.call_count == 3  # Still 3!

    print("Cooldown and Back-to-Normal logic: All PASSED!")


def test_send_telegram_http_mock():
    print("Testing send_telegram HTTP request and error handling...")

    with patch("notifier.get_telegram_credentials", return_value=("fake_token", "fake_chat_id")):
        # Success case
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 999}}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ok = notifier.send_telegram("<b>Test</b>")
            assert ok is True
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["chat_id"] == "fake_chat_id"
            assert call_kwargs["json"]["text"] == "<b>Test</b>"
            assert call_kwargs["json"]["parse_mode"] == "HTML"

        # HTTP error case (e.g. 400 Bad Request)
        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 400
        mock_err_resp.text = "Bad Request: chat not found"
        with patch("requests.post", return_value=mock_err_resp):
            ok_err = notifier.send_telegram("<b>Test</b>")
            assert ok_err is False

        # Timeout case
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
            ok_to = notifier.send_telegram("<b>Test</b>")
            assert ok_to is False

    print("send_telegram HTTP handling: All PASSED!")


if __name__ == "__main__":
    test_trigger_rules()
    test_message_formatting_no_emojis()
    test_cooldown_and_back_to_normal()
    test_send_telegram_http_mock()
    print("\nALL NOTIFIER TESTS COMPLETED SUCCESSFULLY!")
