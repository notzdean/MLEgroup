"""
ingest_nea_weather.py
=====================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
NEA Weather Data Ingestion — API only (2013–2020)

NOTE: NEA v2 API coverage starts:
  - Air Temperature:   May 2016
  - Relative Humidity: May 2016
  - Rainfall:          Dec 2016
  Data before these dates will have None for the missing metrics.

USAGE:
  python3 ingest_nea_weather.py

DEPENDENCIES:
  pip3 install requests pandas
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.expanduser("~/Desktop/CS611_dengue/data/dengue")
OUTPUT_FILE = os.path.join(DATA_DIR, "nea_weather.csv")

API_START   = date(2013, 1, 1)
API_END     = date(2020, 11, 30)

BASE_URL    = "https://api-open.data.gov.sg/v2/real-time/api"

DELAY_BETWEEN_CALLS = 3.0   # seconds between each API call
DELAY_BETWEEN_PAGES = 2.0   # seconds between pagination calls
DELAY_ON_RATE_LIMIT = 30.0  # seconds to wait when 429 hit
MAX_RETRIES         = 3     # retries per API call

# ─── API helpers ──────────────────────────────────────────────────────────────
def api_get(url: str, params: dict) -> dict | None:
    """Make one API call with retry logic on 429 rate limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 404:
                return None  # no data for this date — not an error

            if resp.status_code == 429:
                wait = DELAY_ON_RATE_LIMIT * attempt
                print(f"\n    Rate limited — waiting {wait:.0f}s before retry "
                      f"{attempt}/{MAX_RETRIES}...", end=" ", flush=True)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"(failed after {MAX_RETRIES} attempts: {e})", end=" ")
                return None
            time.sleep(DELAY_BETWEEN_CALLS)

    return None


def fetch_metric_for_day(metric: str, query_date: date):
    """
    Fetch one weather metric for one full day, handling pagination.

    metric: 'rainfall' | 'air-temperature' | 'relative-humidity'
    Returns: ({station_id: value}, {station_id: station_name})
    """
    url = f"{BASE_URL}/{metric}"
    date_str = query_date.strftime("%Y-%m-%d")

    by_station: dict[str, list] = {}
    station_meta: dict[str, str] = {}
    pagination_token = None

    while True:
        params = {"date": date_str}
        if pagination_token:
            params["paginationToken"] = pagination_token
            time.sleep(DELAY_BETWEEN_PAGES)

        body = api_get(url, params)
        if not body:
            break

        data = body.get("data", {})

        for st in data.get("stations", []):
            station_meta[st["id"]] = st.get("name", st["id"])

        for reading in data.get("readings", []):
            for entry in reading.get("data", []):
                sid = entry["stationId"]
                by_station.setdefault(sid, []).append(entry["value"])

        pagination_token = data.get("paginationToken")
        if not pagination_token:
            break

    result = {}
    for sid, values in by_station.items():
        if not values:
            continue
        if metric == "rainfall":
            result[sid] = round(sum(values), 4)
        else:
            result[sid] = round(sum(values) / len(values), 4)

    return result, station_meta


# ─── Resume logic ─────────────────────────────────────────────────────────────
def already_fetched_dates() -> set:
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        df = pd.read_csv(OUTPUT_FILE)
        return set(df["date"].unique())
    except Exception:
        return set()


# ─── Main fetch loop ──────────────────────────────────────────────────────────
def fetch_api_range():
    done = already_fetched_dates()
    total = (API_END - API_START).days + 1
    remaining = total - len(done)
    est_hrs = remaining * 3 * DELAY_BETWEEN_CALLS / 3600

    print("=" * 60)
    print("NEA Weather Ingestion — API only (2013–2020)")
    print("=" * 60)
    print(f"  Output:         {OUTPUT_FILE}")
    print(f"  Date range:     {API_START} → {API_END}")
    print(f"  Days to fetch:  {remaining} / {total}")
    print(f"  Est. time:      ~{est_hrs:.1f} hours")
    print(f"  Tip: safe to Ctrl+C and resume later\n")
    print("  NOTE: API data available from ~May 2016 (temp/humidity)")
    print("        and ~Dec 2016 (rainfall). Earlier dates will have")
    print("        None for missing metrics — this is expected.\n")

    all_rows = []
    current = API_START
    checkpoint_counter = 0

    while current <= API_END:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in done:
            current += timedelta(days=1)
            continue

        print(f"  {date_str}...", end=" ", flush=True)

        rain,  rain_meta  = fetch_metric_for_day("rainfall",          current)
        time.sleep(DELAY_BETWEEN_CALLS)
        temp,  temp_meta  = fetch_metric_for_day("air-temperature",   current)
        time.sleep(DELAY_BETWEEN_CALLS)
        humid, humid_meta = fetch_metric_for_day("relative-humidity", current)
        time.sleep(DELAY_BETWEEN_CALLS)

        all_sids = set(rain) | set(temp) | set(humid)
        meta = {**rain_meta, **temp_meta, **humid_meta}

        if not all_sids:
            # Still write a placeholder so this date is marked as done
            all_rows.append({
                "date":         date_str,
                "station":      None,
                "station_id":   None,
                "rainfall_mm":  None,
                "temp_mean_c":  None,
                "humidity_pct": None,
                "source":       "nea_v2_api",
            })
            print("(no data — pre-API coverage)")
        else:
            for sid in all_sids:
                all_rows.append({
                    "date":         date_str,
                    "station":      meta.get(sid, sid),
                    "station_id":   sid,
                    "rainfall_mm":  rain.get(sid),
                    "temp_mean_c":  temp.get(sid),
                    "humidity_pct": humid.get(sid),
                    "source":       "nea_v2_api",
                })
            print(f"✓  ({len(all_sids)} stations, "
                  f"rain={bool(rain)}, temp={bool(temp)}, humid={bool(humid)})")

        current += timedelta(days=1)
        checkpoint_counter += 1

        # Checkpoint every 10 days
        if checkpoint_counter >= 10:
            _save_chunk(all_rows)
            all_rows = []
            checkpoint_counter = 0

    # Save any remaining rows
    if all_rows:
        _save_chunk(all_rows)


def _save_chunk(rows: list):
    if not rows:
        return
    chunk = pd.DataFrame(rows)
    header = not os.path.exists(OUTPUT_FILE)
    chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=header)
    print(f"  → Checkpoint saved ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_api_range()

    # Final summary
    if os.path.exists(OUTPUT_FILE):
        final = pd.read_csv(OUTPUT_FILE)
        # Drop placeholder no-data rows for summary stats
        data_rows = final.dropna(subset=["station_id"])
        print("\n" + "=" * 60)
        print("DONE")
        print(f"  Output:       {OUTPUT_FILE}")
        print(f"  Total rows:   {len(final):,}")
        print(f"  Data rows:    {len(data_rows):,}")
        print(f"  Date range:   {final['date'].min()} → {final['date'].max()}")
        print(f"  Dates:        {final['date'].nunique()}")
        print(f"  Stations:     {data_rows['station'].nunique()}")
        print("\nSample (first rows with rainfall data):")
        print(data_rows.dropna(subset=["rainfall_mm"]).head(3).to_string(index=False))
        print("=" * 60)
        print("\nNext step: ingest_sgcharts.py")


if __name__ == "__main__":
    main()
