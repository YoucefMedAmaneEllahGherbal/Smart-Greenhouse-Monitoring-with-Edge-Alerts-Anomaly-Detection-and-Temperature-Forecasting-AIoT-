"""Verification of stale data detection with fake timestamps.

Tests the 4 scenarios requested:
1. 30 s old (live)
2. 4 min old (live with a 5 min threshold)
3. 10 min old (stale with a 5 min threshold)
4. 1-hour timezone offset (e.g. UTC+1) representing 30 s old reading (must NOT be stale)
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
from thingspeak import is_data_stale, get_reading_age_seconds, get_latest_reading_age

def run_tests():
    now_utc = datetime.now(timezone.utc)
    print(f"Current UTC time: {now_utc.isoformat()}")

    # Scenario 1: 30 s old
    ts_30s = now_utc - timedelta(seconds=30)
    df_30s = pd.DataFrame({"ts": [ts_30s], "T": [24.0], "RH": [60.0], "alert": [0]})
    age_1 = get_latest_reading_age(df_30s)
    stale_1 = is_data_stale(df_30s, max_minutes=5.0)
    print(f"Test 1 (30s old): Age = {age_1:.1f}s, Stale (5 min threshold)? {stale_1}")
    assert not stale_1, f"Expected Test 1 to be LIVE, but got stale={stale_1}"
    assert 28 <= age_1 <= 35, f"Expected age ~30s, got {age_1}"

    # Scenario 2: 4 min old with 5 min threshold (live)
    ts_4m = now_utc - timedelta(minutes=4)
    df_4m = pd.DataFrame({"ts": [ts_4m], "T": [24.0], "RH": [60.0], "alert": [0]})
    age_2 = get_latest_reading_age(df_4m)
    stale_2 = is_data_stale(df_4m, max_minutes=5.0)
    print(f"Test 2 (4 min old, 5m threshold): Age = {age_2:.1f}s, Stale? {stale_2}")
    assert not stale_2, f"Expected Test 2 to be LIVE, but got stale={stale_2}"
    assert 238 <= age_2 <= 245, f"Expected age ~240s, got {age_2}"

    # Scenario 3: 10 min old with 5 min threshold (stale)
    ts_10m = now_utc - timedelta(minutes=10)
    df_10m = pd.DataFrame({"ts": [ts_10m], "T": [24.0], "RH": [60.0], "alert": [0]})
    age_3 = get_latest_reading_age(df_10m)
    stale_3 = is_data_stale(df_10m, max_minutes=5.0)
    print(f"Test 3 (10 min old, 5m threshold): Age = {age_3:.1f}s, Stale? {stale_3}")
    assert stale_3, f"Expected Test 3 to be STALE, but got stale={stale_3}"
    assert 598 <= age_3 <= 605, f"Expected age ~600s, got {age_3}"

    # Scenario 4: Timestamp with 1-hour timezone offset (e.g. UTC+1) representing 30s ago in wall-clock
    # Local time is UTC + 1 hour. A reading taken 30 seconds ago has:
    # ts_local = (now_utc + 1h) - 30s in timezone(timedelta(hours=1))
    tz_plus_1 = timezone(timedelta(hours=1))
    now_tz1 = datetime.now(tz_plus_1)
    ts_tz1 = now_tz1 - timedelta(seconds=30)
    # Both as datetime and converted pd.Timestamp
    df_tz1 = pd.DataFrame({"ts": [ts_tz1], "T": [24.0], "RH": [60.0], "alert": [0]})
    age_4 = get_latest_reading_age(df_tz1)
    stale_4 = is_data_stale(df_tz1, max_minutes=5.0)
    print(f"Test 4 (UTC+1 offset, 30s ago): Timestamp = {ts_tz1}, Age = {age_4:.1f}s, Stale? {stale_4}")
    assert not stale_4, f"Expected Test 4 to be LIVE, but got stale={stale_4}"
    assert 28 <= age_4 <= 35, f"Expected age ~30s, got {age_4}"

    # Also test pd.Timestamp with tz_localize / tz_convert
    ts_pd_tz = pd.Timestamp(ts_tz1)
    df_pd_tz = pd.DataFrame({"ts": [ts_pd_tz], "T": [24.0], "RH": [60.0], "alert": [0]})
    age_4b = get_latest_reading_age(df_pd_tz)
    stale_4b = is_data_stale(df_pd_tz, max_minutes=5.0)
    print(f"Test 4b (pd.Timestamp UTC+1): Timestamp = {ts_pd_tz}, Age = {age_4b:.1f}s, Stale? {stale_4b}")
    assert not stale_4b
    assert 28 <= age_4b <= 35

    print("\nALL 4 SCENARIOS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
