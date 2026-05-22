"""
pipeline/ingest_mss_weather.py
================================
Bronze ingestion — MSS weather station records.

Input : data/bronze/raw_mss_weather_2013_2020.csv
        data/bronze/raw_mss_stations_list_mss.csv
Output: data/bronze/BRONZE_mss_weather.csv

Bronze responsibility: load + enforce schema only.
No rows dropped. Cleaning (forward-fill, outlier cap,
column drops) happens in preprocess.py.

Design note
-----------
Temperature and wind columns are retained here even though
EDA showed they are too sparse for feature engineering
(temp: 28.7% coverage, wind: 34.2%). preprocess.py drops them.
MSS chosen over NEA API because MSS extends back to 2013;
NEA API only starts December 2016.
"""

import pandas as pd
from pathlib import Path

ROOT           = Path(__file__).resolve().parent.parent
INPUT_WEATHER  = ROOT / "data" / "bronze" / "raw_mss_weather_2013_2020.csv"
INPUT_STATIONS = ROOT / "data" / "bronze" / "raw_mss_stations_list_mss.csv"
OUTPUT         = ROOT / "data" / "bronze" / "BRONZE_mss_weather.csv"


def main():
    print("=" * 60)
    print("Bronze ingestion — MSS weather station records")
    print("=" * 60)

    # Load weather readings
    df = pd.read_csv(INPUT_WEATHER, low_memory=False)
    print(f"[load]   {len(df):,} rows from {INPUT_WEATHER.name}")

    # Load station metadata and join on station_id
    stations = pd.read_csv(INPUT_STATIONS)
    stations.columns = stations.columns.str.strip()
    stations = stations.rename(columns={
        "Station_ID":   "station_id",
        "Station_Name": "station_name"
    })
    df = df.merge(stations, on="station_id", how="left")
    print(f"[join]   Station metadata joined — {df['station_name'].isna().sum()} unmatched station IDs")

    # Schema enforcement — no dropping
    df["year"]          = pd.to_numeric(df["year"],          errors="coerce").astype("Int64")
    df["month"]         = pd.to_numeric(df["month"],         errors="coerce").astype("Int64")
    df["day"]           = pd.to_numeric(df["day"],           errors="coerce").astype("Int64")
    df["rainfall_mm"]   = pd.to_numeric(df["rainfall_mm"],   errors="coerce")
    df["temp_mean_c"]   = pd.to_numeric(df["temp_mean_c"],   errors="coerce")
    df["temp_max_c"]    = pd.to_numeric(df["temp_max_c"],    errors="coerce")
    df["temp_min_c"]    = pd.to_numeric(df["temp_min_c"],    errors="coerce")
    df["wind_mean_kmh"] = pd.to_numeric(df["wind_mean_kmh"], errors="coerce")
    df["wind_max_kmh"]  = pd.to_numeric(df["wind_max_kmh"],  errors="coerce")
    df["station"]       = df["station"].astype(str)
    df["station_id"]    = df["station_id"].astype(str)

    # Construct date column for convenience (Silver will use this)
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(columns={"year": "year", "month": "month", "day": "day"}),
        errors="coerce"
    )

    # Column order
    cols = ["date", "year", "month", "day", "station_id", "station", "station_name",
            "rainfall_mm", "temp_mean_c", "temp_max_c", "temp_min_c",
            "wind_mean_kmh", "wind_max_kmh"]
    df = df[cols]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"[write]  {len(df):,} rows → {OUTPUT.name}")
    print(f"[done]   Stations: {df['station_id'].nunique()} | "
          f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    main()
