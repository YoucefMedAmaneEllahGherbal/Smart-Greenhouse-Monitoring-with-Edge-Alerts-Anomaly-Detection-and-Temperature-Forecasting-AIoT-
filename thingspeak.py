"""ThingSpeak data access layer for the Smart Greenhouse AIoT application.

Fetches real-time feeds from ThingSpeak, converts timestamps, parses sensor data,
evaluates device alert conditions, handles errors gracefully, and detects stale data.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Union
import pandas as pd
import requests
import streamlit as st


def get_default_credentials() -> Tuple[str, str]:
    """Retrieve ThingSpeak Channel ID and Read Key from st.secrets or environment variables."""
    channel_id = ""
    read_key = ""

    # Try st.secrets first
    try:
        if "THINGSPEAK_CHANNEL_ID" in st.secrets:
            channel_id = str(st.secrets["THINGSPEAK_CHANNEL_ID"]).strip()
        if "THINGSPEAK_READ_KEY" in st.secrets:
            read_key = str(st.secrets["THINGSPEAK_READ_KEY"]).strip()
    except Exception:
        pass

    # Fallback to environment variables
    if not channel_id:
        channel_id = os.environ.get("THINGSPEAK_CHANNEL_ID", "").strip()
    if not read_key:
        read_key = os.environ.get("THINGSPEAK_READ_KEY", "").strip()

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


@st.cache_data(ttl=4)
def fetch_thingspeak_data(
    channel_id: str,
    read_key: Optional[str] = None,
    results: int = 600
) -> pd.DataFrame:
    """Fetch recent sensor feeds from a ThingSpeak channel.

    Parameters
    ----------
    channel_id : str
        The ThingSpeak channel numeric ID.
    read_key : str, optional
        Read API key for private channels.
    results : int, default 600
        Number of readings to fetch (default ~3 hours at 1 pt / 20 s).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ts', 'T', 'RH', 'alert'].
    """
    empty_df = pd.DataFrame(columns=["ts", "T", "RH", "alert"])

    if not channel_id or not str(channel_id).strip():
        st.warning("Please provide a ThingSpeak Channel ID in the sidebar.")
        return empty_df

    clean_id = str(channel_id).strip()
    url = f"https://api.thingspeak.com/channels/{clean_id}/feeds.json?results={results}"

    clean_key = str(read_key).strip() if read_key else ""
    if clean_key:
        url += f"&api_key={clean_key}"

    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        st.error("Request to ThingSpeak timed out. Please check your internet connection or try again.")
        return empty_df
    except requests.exceptions.RequestException as exc:
        st.error(f"Network error while contacting ThingSpeak: {exc}")
        return empty_df

    if response.status_code == 404:
        st.error(f"ThingSpeak channel '{clean_id}' was not found. Please verify the Channel ID.")
        return empty_df
    if response.status_code in (400, 401, 403) or response.text.strip() == "-1":
        st.error("Unauthorized access to ThingSpeak channel. Please verify the Read API Key.")
        return empty_df
    if response.status_code != 200:
        st.error(f"ThingSpeak returned HTTP {response.status_code}: {response.text[:150]}")
        return empty_df

    try:
        data = response.json()
    except Exception as exc:
        st.error(f"Failed to parse JSON response from ThingSpeak: {exc}")
        return empty_df

    if not isinstance(data, dict) or "feeds" not in data:
        st.warning("Invalid data format received from ThingSpeak.")
        return empty_df

    feeds = data.get("feeds", [])
    if not feeds:
        st.warning("ThingSpeak channel contains no data feeds yet.")
        return empty_df

    raw_df = pd.DataFrame(feeds)
    if "created_at" not in raw_df.columns:
        st.warning("Missing 'created_at' timestamp column in ThingSpeak feed.")
        return empty_df

    # Parse UTC timestamp
    raw_df["ts"] = pd.to_datetime(raw_df["created_at"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["ts"])

    # Parse field1 (temperature) and field2 (humidity)
    raw_df["T"] = pd.to_numeric(raw_df.get("field1"), errors="coerce")
    raw_df["RH"] = pd.to_numeric(raw_df.get("field2"), errors="coerce")

    # Drop rows missing temperature or humidity
    valid_df = raw_df.dropna(subset=["T", "RH"]).copy()
    if valid_df.empty:
        st.warning("All records in feed have missing temperature or humidity readings.")
        return empty_df

    # Parse device alert (field3) with rule-based fallback if field3 is missing or invalid
    if "field3" in valid_df.columns:
        f3_num = pd.to_numeric(valid_df["field3"], errors="coerce")
        # Check if field3 contains valid alert levels {0, 1, 2}
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
    return result


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
