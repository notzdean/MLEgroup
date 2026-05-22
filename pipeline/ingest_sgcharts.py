"""
pipeline/ingest_sgcharts.py
============================
Bronze ingestion — SGCharts dengue cluster snapshots.

Input : data/bronze/raw_sgcharts_dengue.csv
Output: data/bronze/BRONZE_sgcharts_clusters.csv

Bronze responsibility: load + enforce schema only.
No rows are dropped here. All cleaning (bbox filter,
bad lat/lng, deduplication) happens in preprocess.py.
"""

import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "bronze" / "raw_sgcharts_dengue.csv"
OUTPUT = ROOT / "data" / "bronze" / "BRONZE_sgcharts_clusters.csv"


def main():
    print("=" * 60)
    print("Bronze ingestion — SGCharts dengue clusters")
    print("=" * 60)

    df = pd.read_csv(INPUT, low_memory=False)
    print(f"[load]   {len(df):,} rows from {INPUT.name}")

    # Schema enforcement only — no dropping
    df["date"]         = pd.to_datetime(df["date"], errors="coerce")
    df["latitude"]     = pd.to_numeric(df["latitude"],     errors="coerce")
    df["longitude"]    = pd.to_numeric(df["longitude"],    errors="coerce")
    df["cases"]        = pd.to_numeric(df["cases"],        errors="coerce")
    df["recent_cases"] = pd.to_numeric(df["recent_cases"], errors="coerce")
    df["total_cases"]  = pd.to_numeric(df["total_cases"],  errors="coerce")
    df["cluster_id"]   = pd.to_numeric(df["cluster_id"],   errors="coerce")
    df["month_num"]    = pd.to_numeric(df["month_num"],    errors="coerce")
    df["location"]     = df["location"].astype(str)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"[write]  {len(df):,} rows → {OUTPUT.name}")
    print(f"[done]   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
