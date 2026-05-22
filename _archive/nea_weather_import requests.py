"""
ingest_nea_weather.py
=====================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Stage 1b: NEA Weather Data Extraction

What this script does:
  Fetches historical daily rainfall and temperature data from the
  NEA / data.gov.sg API for Singapore weather stations.
  Covers 2013-01-01 to 2020-11-30 (matching SGCharts dengue data window).

Output:
  data/raw/nea_weather.csv
  One row per station per day.

Usage:
  python ingest_nea_weather.py

  This will take ~10-20 minutes to run (fetching ~2,900 days of data).
  Progress is printed as it goes. If it stops, just run again —
  it will skip dates already saved (resume-safe).

Dependencies:
  pip install requests pandas openpyxl
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

# ─── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\data\dengue"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nea_weather.csv")

# Date range — must match your SGCharts dengue data window
DATE_START  = date(2013, 1, 1)
DATE_END    = date(2020, 11, 30)

# Which weather stations to use
# Using 4 stations spread across Singapore for better spatial coverage
# S24 = Changi (East), S43 = Kim Chuan (Central),
# S50 = Clementi (West), S60 = Sentosa (South)
STATIONS = ["S24", "S43", "S50", "S60"]

# API base URLs — no API key needed, fully open
RAINFALL_API    = "https://api.data.gov.sg/v1/environment/rainfall"
TEMPERATURE_API = "https://api.data.gov.sg/v1/environment/air-temperature"

# Polite delay between API calls (seconds) — avoid hammering the server
API_DELAY = 0.5


# ─── Helpers ─────────────────────────────────────────────────────────────────
def date_range(start: date, end: date):
    """Generate all dates from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_reading(api_url: str, query_date: date) -> list[dict]:
    """
    Fetch weather readings for a given date from the NEA API.

    The API returns readings at 5-minute or 1-hour intervals.
    We take the DAILY AVERAGE across all readings for each station.

    Returns list of dicts: [{station_id, station_name, lat, lng, value}, ...]
    """
    date_str = query_date.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            api_url,
            params={"date": date_str},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"    ⚠ API error for {date_str}: {e}")
        return []

    # API response structure:
    # {
    #   "metadata": { "stations": [{id, name, location: {lat, lng}}, ...] },
    #   "items": [
    #     { "timestamp": "...", "readings": [{station_id, value}, ...] },
    #     ...  (one per interval throughout the day)
    #   ]
    # }

    if not data.get("items") or not data.get("metadata"):
        return []

    # Build station lookup: id → {name, lat, lng}
    stations_meta = {
        st["id"]: {
            "name": st.get("name", ""),
            "lat":  st["location"]["latitude"],
            "lng":  st["location"]["longitude"],
        }
        for st in data["metadata"]["stations"]
    }

    # Collect all readings across all time intervals
    station_readings: dict[str, list[float]] = {}
    for item in data["items"]:
        for reading in item.get("readings", []):
            sid = reading["station_id"]
            val = reading["value"]
            if sid not in station_readings:
                station_readings[sid] = []
            station_readings[sid].append(val)

    # Compute daily aggregate per station
    results = []
    for sid, values in station_readings.items():
        if not values:
            continue
        meta = stations_meta.get(sid, {})
        results.append({
            "station_id":   sid,
            "station_name": meta.get("name", sid),
            "lat":          meta.get("lat"),
            "lng":          meta.get("lng"),
            "daily_mean":   round(sum(values) / len(values), 4),
            "daily_max":    round(max(values), 4),
            "daily_min":    round(min(values), 4),
            "n_readings":   len(values),
        })

    return results


def already_fetched_dates() -> set[str]:
    """Return set of dates already in the output file (for resume support)."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        existing = pd.read_csv(OUTPUT_FILE, usecols=["date"])
        return set(existing["date"].unique())
    except Exception:
        return set()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check what's already been fetched
    done_dates = already_fetched_dates()
    total_days = (DATE_END - DATE_START).days + 1
    print(f"Date range: {DATE_START} → {DATE_END} ({total_days} days)")
    print(f"Already fetched: {len(done_dates)} days")
    print(f"Remaining: {total_days - len(done_dates)} days")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 55)

    all_rows = []
    fetched_count = 0

    for current_date in date_range(DATE_START, DATE_END):
        date_str = current_date.strftime("%Y-%m-%d")

        # Skip if already fetched (resume safety)
        if date_str in done_dates:
            continue

        print(f"Fetching {date_str}...", end=" ")

        # ── Rainfall ───────────────────────────────────────────────────────
        rainfall_rows = fetch_reading(RAINFALL_API, current_date)
        time.sleep(API_DELAY)

        # ── Temperature ────────────────────────────────────────────────────
        temp_rows = fetch_reading(TEMPERATURE_API, current_date)
        time.sleep(API_DELAY)

        # ── Combine into one row per station ──────────────────────────────
        # Index by station_id for easy lookup
        rainfall_by_station = {r["station_id"]: r for r in rainfall_rows}
        temp_by_station     = {r["station_id"]: r for r in temp_rows}

        # Get union of all station IDs seen today
        all_station_ids = set(rainfall_by_station) | set(temp_by_station)

        day_rows = []
        for sid in all_station_ids:
            rain = rainfall_by_station.get(sid, {})
            temp = temp_by_station.get(sid, {})

            # Use whichever has metadata
            meta = rain if rain else temp

            row = {
                "date":              date_str,
                "station_id":        sid,
                "station_name":      meta.get("station_name", sid),
                "lat":               meta.get("lat"),
                "lng":               meta.get("lng"),
                # Rainfall (mm)
                "rainfall_mean_mm":  rain.get("daily_mean"),
                "rainfall_max_mm":   rain.get("daily_max"),
                # Temperature (°C)
                "temp_mean_c":       temp.get("daily_mean"),
                "temp_max_c":        temp.get("daily_max"),
                "temp_min_c":        temp.get("daily_min"),
            }
            day_rows.append(row)

        all_rows.extend(day_rows)
        fetched_count += 1
        print(f"✓  ({len(day_rows)} stations)")

        # ── Write to CSV every 30 days (checkpoint) ───────────────────────
        if fetched_count % 30 == 0:
            df_chunk = pd.DataFrame(all_rows)
            # Append to existing file
            header = not os.path.exists(OUTPUT_FILE)
            df_chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=header)
            all_rows = []  # clear buffer
            print(f"  → Checkpoint saved ({fetched_count} days fetched so far)")

    # ── Final write ───────────────────────────────────────────────────────────
    if all_rows:
        df_chunk = pd.DataFrame(all_rows)
        header = not os.path.exists(OUTPUT_FILE)
        df_chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=header)

    # ── Summary ───────────────────────────────────────────────────────────────
    if os.path.exists(OUTPUT_FILE):
        final_df = pd.read_csv(OUTPUT_FILE)
        print("\n" + "=" * 55)
        print(f"Done. Output saved to: {OUTPUT_FILE}")
        print(f"Total rows:    {len(final_df):,}")
        print(f"Total days:    {final_df['date'].nunique()}")
        print(f"Total stations:{final_df['station_id'].nunique()}")
        print("\nSample:")
        print(final_df.head(3).to_string(index=False))
        print("\nColumn summary:")
        print(final_df.describe().to_string())
        print("=" * 55)
        print("Next step: run ingest_sgcharts.py")


if __name__ == "__main__":
    main()