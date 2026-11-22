"""ThingSpeak data access layer for the Smart Greenhouse AIoT application.

Fetches real-time feeds from ThingSpeak, converts timestamps, parses sensor data,
evaluates device alert conditions, handles network errors gracefully, and detects stale data.
Fully hardened for deployment: HTTPS only, strict credential sanitization, no sensitive logs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import (
    CHANNEL_ID_PATTERN,
    MAX_RESULTS,
    N_READINGS,
    REQUEST_TIMEOUT_SEC,
    get_secret,
    log_error,
)


def get_default_credentials() -> Tuple[str, str]:
    """Retrieve ThingSpeak Channel ID and Read Key securely via get_secret."""
    channel_id = get_secret("THINGSPEAK_CHANNEL_ID")
    read_key = get_secret("THINGSPEAK_READ_KEY")
    return channel_id, read_key


def calculate_esp32_alert(temperature: float, relative_humidity: float) -> int:
    """Calculate device alert level based on ESP32 firmware rules.

    Rules:
    - Critical (2): T >= 35 °C or T <= 2 °C
    - Warning (1): T >= 30 °C or T <= 5 °C or RH >= 90 % or RH <= 40 %
    - Normal (0): otherwise
    """
    if temperature >= 35.0 or temperature <= 2.0:
        return 2
    if temperature >= 30.0 or temperature <= 5.0 or relative_humidity >= 90.0 or relative_humidity <= 40.0:
        return 1
    return 0


@st.cache_data(ttl=15)
def fetch_thingspeak_data(
    channel_id: Optional[str] = None,
    read_key: Optional[str] = None,
    results: int = N_READINGS,
) -> Tuple[pd.DataFrame, str]:
    """Fetch recent sensor feeds from a ThingSpeak channel over HTTPS.

    Parameters
    ----------
    channel_id : str, optional
        The ThingSpeak channel numeric ID (defaults to secret if None).
    read_key : str, optional
        Read API key for private channels (defaults to secret if None).
    results : int, default N_READINGS (capped at MAX_RESULTS)
        Number of readings to fetch.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (DataFrame with columns ['ts', 'T', 'RH', 'alert'], fetch_status)
        Status values: 'ok', 'not_configured', 'unreachable', 'empty'.
    """
    empty_df = pd.DataFrame(columns=["ts", "T", "RH", "alert"])

    if channel_id is None:
        channel_id = get_secret("THINGSPEAK_CHANNEL_ID")
    if read_key is None:
        read_key = get_secret("THINGSPEAK_READ_KEY")

    clean_id = str(channel_id).strip() if channel_id else ""
    clean_key = str(read_key).strip() if read_key else ""

    # Validate channel ID with strict regex: must be 1-12 digits.
    # If invalid, missing, or placeholder, make NO HTTP call.
    if not clean_id or not CHANNEL_ID_PATTERN.match(clean_id):
        return empty_df, "not_configured"

    # HTTPS only, standard certificate verification enforced, sanitize parameters
    url = f"https://api.thingspeak.com/channels/{clean_id}/feeds.json"
    bounded_results = min(max(1, int(results)), MAX_RESULTS)
    params: dict[str, Any] = {"results": bounded_results}
    if clean_key:
        params["api_key"] = clean_key

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except Exception as exc:
        # Catch all network/timeout/connection exceptions and log ONLY type name
        log_error(type(exc).__name__)
        return empty_df, "unreachable"

    if response.status_code != 200:
        # Non-200 responses (404, 401, 403, 500, etc.)
        log_error(f"HTTPStatus_{response.status_code}")
        return empty_df, "unreachable"

    try:
        data = response.json()
    except Exception as exc:
        log_error(type(exc).__name__)
        return empty_df, "unreachable"

    if not isinstance(data, dict) or "feeds" not in data:
        return empty_df, "unreachable"

    feeds = data.get("feeds", [])
    if not feeds:
        return empty_df, "empty"

    raw_df = pd.DataFrame(feeds)
    if "created_at" not in raw_df.columns:
        return empty_df, "unreachable"

    # Parse UTC timestamp safely
    raw_df["ts"] = pd.to_datetime(raw_df["created_at"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["ts"])
    if raw_df.empty:
        return empty_df, "empty"

    # Parse field1 (temperature) and field2 (humidity) strictly coerced
    raw_df["T"] = pd.to_numeric(raw_df.get("field1"), errors="coerce")
    raw_df["RH"] = pd.to_numeric(raw_df.get("field2"), errors="coerce")

    # Replace infinite values and drop missing records
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
    valid_df = raw_df.dropna(subset=["T", "RH"]).copy()
    if valid_df.empty:
        return empty_df, "empty"

    # Parse device alert (field3) with rule-based fallback if field3 is missing or invalid
    if "field3" in valid_df.columns:
        f3_num = pd.to_numeric(valid_df["field3"], errors="coerce")
        valid_f3 = f3_num.dropna()
        if not valid_f3.empty and valid_f3.isin([0, 1, 2]).all():
            valid_df["alert"] = f3_num.fillna(0).astype(int)
        elif "field7" in valid_df.columns and pd.to_numeric(valid_df["field7"], errors="coerce").dropna().isin([0, 1, 2]).all():
            valid_df["alert"] = pd.to_numeric(valid_df["field7"], errors="coerce").fillna(0).astype(int)
        else:
            valid_df["alert"] = [calculate_esp32_alert(t, rh) for t, rh in zip(valid_df["T"], valid_df["RH"])]
    else:
        valid_df["alert"] = [calculate_esp32_alert(t, rh) for t, rh in zip(valid_df["T"], valid_df["RH"])]

    result = valid_df[["ts", "T", "RH", "alert"]].sort_values("ts").reset_index(drop=True)
    return result, "ok"


def get_reading_age_seconds(ts: Any) -> float:
    """Compute the age of a reading in seconds comparing tz-aware UTC timestamps.

    Both the current time and the reading timestamp are strictly tz-aware in UTC.
    All local timezone conversions must be done for display only, never for this check.
    """
    now_utc = datetime.now(timezone.utc)
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts_utc = ts.tz_localize(timezone.utc).to_pydatetime()
        else:
            ts_utc = ts.tz_convert(timezone.utc).to_pydatetime()
    elif isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts_utc = ts.replace(tzinfo=timezone.utc)
        else:
            ts_utc = ts.astimezone(timezone.utc)
    else:
        parsed = pd.to_datetime(ts, utc=True)
        ts_utc = parsed.to_pydatetime()

    return max(0.0, (now_utc - ts_utc).total_seconds())


def get_latest_reading_age(df: pd.DataFrame) -> float:
    """Return the age in seconds of the latest reading in df using tz-aware UTC comparison."""
    if df.empty or "ts" not in df.columns:
        return float("inf")
    latest_ts = df["ts"].max()
    if pd.isna(latest_ts):
        return float("inf")
    return get_reading_age_seconds(latest_ts)


def is_data_stale(df: pd.DataFrame, max_minutes: float = 5.0) -> bool:
    """Return True if the latest reading is older than max_minutes relative to UTC now."""
    age_seconds = get_latest_reading_age(df)
    return age_seconds > (max_minutes * 60.0)
