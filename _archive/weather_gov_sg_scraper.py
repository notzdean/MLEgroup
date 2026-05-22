"""
weather_gov_sg_scraper.py
=========================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
MSS Historical Weather Data (2013–2020)

HOW IT WORKS:
  Step 1 — Load station list from official MSS station-records.xlsx
            (both Operational and Closed stations included)
            This gives us the definitive list of all 102 station IDs.

  Step 2 — Download 2013–2020 CSVs for every station.
            Checkpoints per station — safe to interrupt and rerun.
            Already-downloaded stations are skipped automatically.

Output:
  ~/Desktop/CS611_dengue/data/dengue/mss_weather_2013_2020.csv
  ~/Desktop/CS611_dengue/data/dengue/stations_list_mss.csv

Usage:
  python3 weather_gov_sg_scraper.py

Requirements:
  pip3 install requests pandas openpyxl

Runtime estimate:
  ~102 stations × 96 months × 1.5s = ~4 hours
  Already downloaded stations are skipped instantly.
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import logging

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

DATA_DIR = os.path.expanduser("~/Desktop/CS611_dengue/data/dengue")

CONFIG = {
    "start_year":  2013,
    "start_month": 1,
    "end_year":    2020,
    "end_month":   11,

    # Official MSS station records file
    "station_records_xlsx": os.path.join(
        os.path.expanduser("~/Desktop/CS611_dengue/data/dengue"),
        "station-records.xlsx"
    ),

    "output_csv":     os.path.join(DATA_DIR, "mss_weather_2013_2020.csv"),
    "stations_csv":   os.path.join(DATA_DIR, "stations_list_mss.csv"),
    "checkpoint_dir": os.path.join(DATA_DIR, "mss_checkpoints"),

    "request_delay": 1.5,
    "timeout":       20,
}

# ─────────────────────────── CONSTANTS ───────────────────────────────────────

BASE_URL = "https://www.weather.gov.sg"
CSV_URL  = BASE_URL + "/files/dailydata/DAILYDATA_{sid}_{year}{month:02d}.csv"

KEEP_COLS = {
    "Year":                         "year",
    "Month":                        "month",
    "Day":                          "day",
    "Daily Rainfall Total (mm)":    "rainfall_mm",
    "Mean Temperature (°C)":        "temp_mean_c",
    "Maximum Temperature (°C)":     "temp_max_c",
    "Minimum Temperature (°C)":     "temp_min_c",
    "Mean Wind Speed (km/h)":       "wind_mean_kmh",
    "Max Wind Speed (km/h)":        "wind_max_kmh",
    "Mean Relative Humidity (%)":   "humidity_mean_pct",
    "Mean Sunshine Duration (hrs)": "sunshine_hrs",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{BASE_URL}/climate-historical-daily/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────── STATION LOADING ─────────────────────────────────

def load_stations_from_xlsx() -> dict:
    """
    Load all station IDs and names from the official MSS station-records.xlsx.
    Includes both operational and closed stations.
    Returns dict of {station_id: station_name}
    """
    xlsx_path = CONFIG["station_records_xlsx"]

    if not os.path.exists(xlsx_path):
        log.error(f"station-records.xlsx not found at: {xlsx_path}")
        log.error("Place the file in your dengue data folder and retry.")
        return {}

    xl = pd.read_excel(xlsx_path, sheet_name=None)
    ops    = xl["Operational Stations"]
    closed = xl["Closed Stations"]

    all_stations = pd.concat([ops, closed], ignore_index=True)
    all_stations.columns = [c.strip() for c in all_stations.columns]
    all_stations = all_stations[["Station ID", "Current Station Name"]].dropna(subset=["Station ID"])
    all_stations["Station ID"] = all_stations["Station ID"].astype(str).str.strip()
    all_stations["Current Station Name"] = all_stations["Current Station Name"].astype(str).str.strip()

    stations = dict(zip(all_stations["Station ID"], all_stations["Current Station Name"]))

    log.info(f"Loaded {len(stations)} stations from station-records.xlsx")
    log.info(f"  Operational: {len(ops)}, Closed: {len(closed)}")

    # Save station list
    pd.DataFrame(
        [{"Station_ID": k, "Station_Name": v} for k, v in stations.items()]
    ).to_csv(CONFIG["stations_csv"], index=False)
    log.info(f"Station list saved → {CONFIG['stations_csv']}\n")

    return stations

# ─────────────────────────── PARSING ─────────────────────────────────────────

def parse_mss_csv(raw_text: str, station_name: str, sid: str) -> pd.DataFrame | None:
    """Parse MSS daily weather CSV. Header is row 0, data starts row 1."""
    try:
        df = pd.read_csv(
            StringIO(raw_text),
            header=0,
            na_values=["—", "-", "\x97", "", " "],
            low_memory=False,
        )
    except Exception as e:
        log.warning(f"  CSV read error for {station_name}: {e}")
        return None

    df.columns = [c.strip() for c in df.columns]

    available = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    if "Year" not in available:
        return None

    df = df[list(available.keys())].rename(columns=available)
    df.insert(0, "station", station_name)
    df.insert(1, "station_id", sid)

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "month", "day"])
    df[["year", "month", "day"]] = df[["year", "month", "day"]].astype(int)

    return df if not df.empty else None

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def iter_months(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def is_checkpoint_valid(checkpoint_file: Path) -> bool:
    try:
        df = pd.read_csv(checkpoint_file, nrows=1, low_memory=False)
        return len(df.columns) <= 15
    except Exception:
        return False


def download_month(session, sid, station_name, year, month) -> pd.DataFrame | None:
    # Strip the S prefix for the URL
    sid_num = sid.lstrip("S")
    url = CSV_URL.format(sid=sid, year=year, month=month)
    try:
        resp = session.get(url, timeout=CONFIG["timeout"])
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  Request failed {url}: {e}")
        return None

    if len(resp.text) < 50:
        return None

    return parse_mss_csv(resp.text, station_name, sid)

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  MSS Weather Scraper — Dengue Project (2013–2020)")
    log.info("  Using official MSS station-records.xlsx")
    log.info("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load official station list ───────────────────────────────────
    stations = load_stations_from_xlsx()
    if not stations:
        return

    # ── Step 2: Download 2013–2020 ────────────────────────────────────────────
    months    = list(iter_months(
        CONFIG["start_year"], CONFIG["start_month"],
        CONFIG["end_year"],   CONFIG["end_month"],
    ))
    total     = len(stations)
    completed = []

    session = requests.Session()
    session.headers.update(HEADERS)

    log.info(f"Stations : {total}")
    log.info(f"Months   : {len(months)} per station")
    log.info(f"Estimated new downloads: {sum(1 for sid in stations if not (checkpoint_dir / f'{sid}.csv').exists())} stations")
    log.info("")

    for idx, (sid, station_name) in enumerate(sorted(stations.items()), 1):
        checkpoint_file = checkpoint_dir / f"{sid}.csv"

        # Use valid existing checkpoint
        if checkpoint_file.exists() and is_checkpoint_valid(checkpoint_file):
            log.info(f"[{idx:>3}/{total}]  CACHED   {station_name} ({sid})")
            completed.append(pd.read_csv(checkpoint_file, low_memory=False))
            continue

        # Delete malformed checkpoint
        if checkpoint_file.exists():
            log.info(f"[{idx:>3}/{total}]  REPAIRING {station_name} ({sid})")
            checkpoint_file.unlink()

        log.info(f"[{idx:>3}/{total}]  Fetching  {station_name} ({sid})")
        frames = []

        for year, month in months:
            df = download_month(session, sid, station_name, year, month)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(CONFIG["request_delay"])

        if frames:
            station_df = pd.concat(frames, ignore_index=True)
            station_df.to_csv(checkpoint_file, index=False)
            completed.append(station_df)
            log.info(f"            ✓ {len(station_df):,} rows")
        else:
            log.warning(f"            ✗ No data for {station_name} ({sid})")

    # ── Step 3: Combine and save ──────────────────────────────────────────────
    if not completed:
        log.error("No data downloaded.")
        return

    combined = pd.concat(completed, ignore_index=True)
    combined.to_csv(CONFIG["output_csv"], index=False)

    log.info("")
    log.info("=" * 60)
    log.info("✓  DONE")
    log.info(f"   Rows     : {len(combined):,}")
    log.info(f"   Stations : {combined['station'].nunique()}")
    log.info(f"   Columns  : {list(combined.columns)}")
    log.info(f"   Output   : {CONFIG['output_csv']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
