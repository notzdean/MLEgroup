"""
ingest_sgcharts_dengue.py
=========================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
SGCharts Dengue Cluster Data Ingestion

Reads all 250+ snapshot CSVs from the sgcharts/csv folder and combines
them into a single clean CSV with proper column names and parsed dates.

INPUT CSV COLUMNS (no header in raw files):
  0: cases          — dengue cases at this specific location
  1: location       — street/block address
  2: latitude
  3: longitude
  4: cluster_id     — cluster number within this snapshot
  5: cluster_cases  — total cases in this cluster
  6: cluster_total  — cumulative/area total cases
  7: date_raw       — snapshot date in YYMMDD format (e.g. 150703 = 3 Jul 2015)
  8: epi_week       — epidemiological week number

OUTPUT:
  ~/Desktop/CS611_dengue/data/dengue/sgcharts_dengue.csv

  One row per location per snapshot date, with:
  - Proper date column (YYYY-MM-DD)
  - Cluster-level aggregates preserved
  - Source filename retained for traceability

USAGE:
  python3 ingest_sgcharts_dengue.py

REQUIREMENTS:
  pip3 install pandas
"""

import os
import re
import pandas as pd
from pathlib import Path

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

DATA_DIR    = r"C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\data\dengue"
INPUT_DIR   = r"C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\data\dengue\sgcharts\csv"
OUTPUT_FILE = os.path.join(DATA_DIR, "sgcharts_dengue.csv")

# ─────────────────────────── COLUMN NAMES ────────────────────────────────────

COLUMNS = [
    "cases",
    "location",
    "latitude",
    "longitude",
    "cluster_id",
    "cluster_cases",
    "cluster_total",
    "date_raw",
    "epi_week",
]

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def parse_date(date_raw) -> str | None:
    """
    Convert YYMMDD integer (e.g. 150703) to YYYY-MM-DD string (e.g. 2015-07-03).
    Assumes all dates are in the 2000s.
    """
    try:
        s = str(int(date_raw)).zfill(6)
        year  = 2000 + int(s[0:2])
        month = int(s[2:4])
        day   = int(s[4:6])
        return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return None


def read_csv_file(filepath: Path) -> pd.DataFrame | None:
    """Read one snapshot CSV file and return a cleaned DataFrame."""
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=COLUMNS,
            dtype={"date_raw": str},
            na_values=["", " "],
        )
    except Exception as e:
        print(f"  ✗ Failed to read {filepath.name}: {e}")
        return None

    if df.empty:
        return None

    # Parse date
    df["date"] = df["date_raw"].apply(parse_date)

    # Coerce numeric columns
    for col in ["cases", "latitude", "longitude", "cluster_id",
                "cluster_cases", "cluster_total", "epi_week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean location strings
    df["location"] = df["location"].str.strip().str.lower()

    # Add source filename for traceability
    df["source_file"] = filepath.name

    # Drop date_raw — replaced by date
    df = df.drop(columns=["date_raw"])

    # Reorder columns
    df = df[["date", "epi_week", "cluster_id", "cluster_cases",
             "cluster_total", "cases", "location", "latitude",
             "longitude", "source_file"]]

    return df


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SGCharts Dengue Cluster Ingestion")
    print("=" * 60)

    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        print(f"✗ Input directory not found: {INPUT_DIR}")
        print("  Check that INPUT_DIR points to the folder with your CSVs.")
        return

    # Find all cluster CSVs
    csv_files = sorted(input_dir.glob("*-clusters.csv"))
    if not csv_files:
        # Try without the -clusters suffix
        csv_files = sorted(input_dir.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}\n")

    if not csv_files:
        print("✗ No CSV files found. Check the INPUT_DIR path.")
        return

    # Read and combine all files
    all_frames = []
    for i, filepath in enumerate(csv_files, 1):
        df = read_csv_file(filepath)
        if df is not None and not df.empty:
            all_frames.append(df)
            print(f"  [{i:>3}/{len(csv_files)}]  ✓  {filepath.name:<30}  "
                  f"{len(df):>5} rows  |  date: {df['date'].iloc[0]}")
        else:
            print(f"  [{i:>3}/{len(csv_files)}]  ✗  {filepath.name}  (empty or failed)")

    if not all_frames:
        print("\n✗ No data loaded.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Sort by date then cluster
    combined = combined.sort_values(["date", "cluster_id"]).reset_index(drop=True)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("✓  DONE")
    print(f"   Rows          : {len(combined):,}")
    print(f"   Snapshots     : {combined['date'].nunique()}")
    print(f"   Date range    : {combined['date'].min()} → {combined['date'].max()}")
    print(f"   Unique locs   : {combined['location'].nunique():,}")
    print(f"   Columns       : {list(combined.columns)}")
    print(f"   Output        : {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: main.py Bronze → Silver → Gold pipeline")


if __name__ == "__main__":
    main()
