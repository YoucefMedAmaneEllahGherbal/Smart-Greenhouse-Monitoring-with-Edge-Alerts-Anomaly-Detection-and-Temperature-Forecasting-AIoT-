"""Smoke test script for Smart Greenhouse AIoT modules."""
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure models is in path
MODELS_DIR = Path("models").resolve()
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import joblib
from predict import load_bundle, predict_next_hour
from anomaly import score_reading, calculate_vpd
from thingspeak import calculate_esp32_alert, is_data_stale, get_default_credentials

print("=" * 60)
print("1. TESTING BUNDLE LOADING & WARNING CHECKS")
print("=" * 60)

with warnings.catch_warnings(record=True) as recorded_warnings:
    warnings.simplefilter("always")
    b_forecast = load_bundle("models/greenhouse_forecast_bundle.joblib")
    b_anomaly = joblib.load("models/climate_inference_bundle.joblib")

print(f"Forecast bundle keys: {list(b_forecast.keys())}")
print(f"Anomaly bundle keys: {list(b_anomaly.keys())}")

if recorded_warnings:
    print(f"Recorded {len(recorded_warnings)} warnings during load:")
    for w in recorded_warnings:
        print(f"  - [{w.category.__name__}]: {w.message}")
else:
    print("Zero warnings during bundle loading!")

print("\n" + "=" * 60)
print("2. TESTING ANOMALY DETECTION (anomaly.py)")
print("=" * 60)

# Normal conditions: 24 °C, 65 % RH
res_normal = score_reading(24.0, 65.0, b_anomaly)
print("Normal test (24 °C, 65 % RH):")
print(f"  Status: {res_normal.status}, Severity: {res_normal.severity}, Score: {res_normal.score:.4f}, VPD: {res_normal.vpd:.2f} kPa")
print(f"  Threshold: {res_normal.threshold:.4f}, Unusual features: {res_normal.unusual_features}")
assert res_normal.status == "Normal", f"Expected Normal, got {res_normal.status}"

# Extreme conditions: 40 °C, 20 % RH
res_extreme = score_reading(40.0, 20.0, b_anomaly)
print("Extreme test (40 °C, 20 % RH):")
print(f"  Status: {res_extreme.status}, Severity: {res_extreme.severity}, Score: {res_extreme.score:.4f}, VPD: {res_extreme.vpd:.2f} kPa")
print(f"  Threshold: {res_extreme.threshold:.4f}, Unusual features: {res_extreme.unusual_features}")
assert res_extreme.status == "Anomaly", f"Expected Anomaly, got {res_extreme.status}"
assert res_extreme.severity == "HIGH", f"Expected HIGH severity, got {res_extreme.severity}"

# Test unpacking and dict access
status, sev, sc, th, vpd, feats = res_extreme
assert status == "Anomaly"
assert res_extreme["status"] == "Anomaly"
assert res_extreme.get("score") == sc

print("All anomaly tests PASSED!")

print("\n" + "=" * 60)
print("3. TESTING FORECAST (models/predict.py)")
print("=" * 60)

# Instant model (single reading)
now_utc = datetime.now(timezone.utc)
df_single = pd.DataFrame([{"ts": now_utc, "T": 24.5, "RH": 60.0}])
fc_instant = predict_next_hour(df_single, b_forecast, tz="Africa/Algiers")
print("Instant Model test (1 reading):")
print(f"  Model used: {fc_instant['model_used']}")
print(f"  Prediction: {fc_instant['prediction']:.2f} °C, Alert level: {fc_instant['alert_level']} ({fc_instant['alert_name']})")
print(f"  History minutes: {fc_instant['history_minutes']}, Forecast time: {fc_instant['forecast_time']}")
assert "instant" in fc_instant["model_used"].lower()

# Extreme reading (40 °C)
df_extreme = pd.DataFrame([{"ts": now_utc, "T": 40.0, "RH": 20.0}])
fc_extreme = predict_next_hour(df_extreme, b_forecast, tz="Africa/Algiers")
print("Extreme Instant Model test (40 °C):")
print(f"  Prediction: {fc_extreme['prediction']:.2f} °C, Alert level: {fc_extreme['alert_level']} ({fc_extreme['alert_name']})")
assert fc_extreme["alert_level"] in (1, 2)

# Lag model (80 minutes of readings, 1 reading every 20s = 240 readings)
t_start = now_utc - timedelta(minutes=80)
fake_times = [t_start + timedelta(seconds=20 * i) for i in range(241)]
# Gradual temperature rise
fake_temps = [20.0 + 0.05 * i for i in range(241)]
fake_rhs = [70.0 - 0.05 * i for i in range(241)]
df_lag = pd.DataFrame({"ts": fake_times, "T": fake_temps, "RH": fake_rhs})
fc_lag = predict_next_hour(df_lag, b_forecast, tz="Africa/Algiers")
print("Lag Model test (80 min history):")
print(f"  Model used: {fc_lag['model_used']}")
print(f"  Prediction: {fc_lag['prediction']:.2f} °C, Alert level: {fc_lag['alert_level']} ({fc_lag['alert_name']})")
print(f"  History minutes: {fc_lag['history_minutes']}")
assert "lag" in fc_lag["model_used"].lower()

print("All forecast tests PASSED!")

print("\n" + "=" * 60)
print("4. TESTING THINGSPEAK HELPERS (thingspeak.py)")
print("=" * 60)

# Test ESP32 alert rule
assert calculate_esp32_alert(24.0, 60.0) == 0  # Normal
assert calculate_esp32_alert(31.0, 60.0) == 1  # Warning (T >= 30)
assert calculate_esp32_alert(4.0, 60.0) == 1   # Warning (T <= 5)
assert calculate_esp32_alert(24.0, 92.0) == 1  # Warning (RH >= 90)
assert calculate_esp32_alert(24.0, 35.0) == 1  # Warning (RH <= 40)
assert calculate_esp32_alert(36.0, 60.0) == 2  # Critical (T >= 35)
assert calculate_esp32_alert(1.5, 60.0) == 2   # Critical (T <= 2)
print("ESP32 alert rules: All PASSED!")

# Test stale check
df_fresh = pd.DataFrame([{"ts": datetime.now(timezone.utc) - timedelta(seconds=30), "T": 25.0, "RH": 50.0}])
df_old = pd.DataFrame([{"ts": datetime.now(timezone.utc) - timedelta(minutes=5), "T": 25.0, "RH": 50.0}])
assert is_data_stale(df_fresh, max_minutes=2.0) is False
assert is_data_stale(df_old, max_minutes=2.0) is True
print("Stale data detection: All PASSED!")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
