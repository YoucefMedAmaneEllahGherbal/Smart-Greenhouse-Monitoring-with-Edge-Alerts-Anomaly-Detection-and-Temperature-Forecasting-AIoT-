"""Prediction helper for the Smart Greenhouse Streamlit app."""
import joblib
import numpy as np
import pandas as pd

def load_bundle(path="greenhouse_forecast_bundle.joblib"):
    return joblib.load(path)

def alert_level(t, th):
    if t <= th["crit_low"] or t >= th["crit_high"]:
        return 2
    if t <= th["warn_low"] or t >= th["warn_high"]:
        return 1
    return 0

ALERT_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}

def predict_next_hour(history, bundle, tz="Africa/Algiers"):
    """
    history : DataFrame with columns ['ts', 'T', 'RH'] (ThingSpeak: created_at, field1, field2).
              'ts' may be UTC (ThingSpeak default) or naive local time.
    Returns : dict(prediction, alert_level, alert_name, model_used, history_minutes)
    """
    h = history[["ts", "T", "RH"]].copy()
    h["ts"] = pd.to_datetime(h["ts"])
    if h["ts"].dt.tz is not None:                       # convert UTC -> local greenhouse time
        h["ts"] = h["ts"].dt.tz_convert(tz).dt.tz_localize(None)
    h = h.dropna().sort_values("ts").set_index("ts")
    if h.empty:
        raise ValueError("No valid readings in history")

    # same 10-minute grid as the training data
    step = f"{bundle['step_minutes']}min"
    g = h.resample(step).mean().ffill(limit=2)
    now = g.index[-1]
    hour = now.hour + now.minute / 60

    row = {"T": g["T"].iloc[-1], "RH": g["RH"].iloc[-1],
           "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24)}

    have_lags = len(g) >= 7 and g[["T", "RH"]].iloc[-7:].notna().all().all()
    if have_lags:
        T, RH = g["T"].to_numpy(), g["RH"].to_numpy()
        row.update({"T_lag1": T[-2], "T_lag3": T[-4], "T_lag6": T[-7],
                    "RH_lag3": RH[-4], "RH_lag6": RH[-7],
                    "dT_30": T[-1] - T[-4], "dT_60": T[-1] - T[-7]})
        cols, model, used = bundle["features_lag"], bundle["model_lag"], f"lag ({bundle['name_lag']})"
    else:
        cols, model, used = bundle["features_instant"], bundle["model_instant"], f"instant ({bundle['name_instant']})"

    pred = float(model.predict(pd.DataFrame([row])[cols])[0])
    lvl = alert_level(pred, bundle["thresholds"])
    return {"prediction": pred, "alert_level": lvl, "alert_name": ALERT_NAMES[lvl],
            "model_used": used, "history_minutes": int((g.index[-1] - g.index[0]).total_seconds() // 60),
            "forecast_time": now + pd.Timedelta(minutes=bundle["horizon_minutes"])}
