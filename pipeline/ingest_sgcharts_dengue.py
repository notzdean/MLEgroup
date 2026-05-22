"""
ingest_sgcharts_dengue.py
=========================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
SGCharts Dengue Cluster Data Ingestion

Reads from TWO folders and combines into one clean CSV:

  Folder 1 — csv/
    Jul 2015 → Nov 2020  |  256 files  |  all columns valid

  Folder 2 — incorrect_latitude_longitude/
    May 2013 → Apr 2015  |  123 files  |  .csv and .txt mixed
    NOTE: folder name is misleading — coordinates ARE correct.
          Only cluster_cases = -1 (not recorded in early data).
          All other columns including lat/long are valid.

OUTPUT:
  sgcharts_dengue.csv — covers May 2013 → Nov 2020
  (3-month gap Apr–Jul 2015 between the two folders is unavoidable)

USAGE (Windows):
  python ingest_sgcharts_dengue.py

USAGE (Mac):
  python3 ingest_sgcharts_dengue.py

REQUIREMENTS:
  pip install pandas
"""

import os
import pandas as pd
from pathlib import Path

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

# ── Windows path (active) ────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\data\dengue"

# ── Mac path (comment out Windows above and uncomment this if on Mac) ────────
# BASE_DIR = os.path.expanduser("~/Desktop/CS611_dengue/data/dengue")

INPUT_DIRS = [
    # Folder 1: early data 2013–2015 (has .csv and .txt files)
    os.path.join(BASE_DIR, "sgcharts", "incorrect_latitude_longitude"),
    # Folder 2: main data 2015–2020
    os.path.join(BASE_DIR, "sgcharts", "csv"),
]

OUTPUT_FILE = os.path.join(BASE_DIR, "sgcharts_dengue.csv")

# ─────────────────────────── COLUMN NAMES ────────────────────────────────────

COLUMNS = [
    "cases",           # Number of reported dengue cases at this location
    "location",        # Street address (block level)
    "latitude",
    "longitude",
    "cluster_id",      # Cluster serial number (NOT a unique ID — reused across snapshots)
    "recent_cases",    # Cases with onset in last 2 weeks (NEA); -1 before Dec 2013
    "total_cases",     # Total cases ever reported in this cluster
    "date_raw",        # Date in YYMMDD format
    "month_num",       # Month number (1=Jan, 12=Dec) — NOT epi week
]

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def parse_date(date_raw) -> str | None:
    """
    Convert YYMMDD integer (e.g. 150703) to YYYY-MM-DD (e.g. 2015-07-03).
    """
    try:
        s = str(int(date_raw)).zfill(6)
        year  = 2000 + int(s[0:2])
        month = int(s[2:4])
        day   = int(s[4:6])
        return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return None


def read_file(filepath: Path) -> pd.DataFrame | None:
    """
    Read one snapshot file (.csv or .txt — same format).
    Handles the cluster_cases = -1 issue in 2013-2015 data.
    """
    for encoding in ["utf-8", "windows-1252", "latin-1"]:
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=COLUMNS,
                dtype={"date_raw": str},
                na_values=["", " "],
                encoding=encoding,
            )
            break
        except (UnicodeDecodeError, Exception) as e:
            if encoding == "latin-1":
                print(f"  ✗ Failed to read {filepath.name}: {e}")
                return None
            continue

    if df.empty:
        return None

    # Parse date
    df["date"] = df["date_raw"].apply(parse_date)

    # Coerce numeric columns
    for col in ["cases", "latitude", "longitude", "cluster_id",
                "recent_cases", "total_cases", "month_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix recent_cases = -1 (official placeholder before Dec 2013)
    # Replace with NaN — not recorded, not wrong
    df.loc[df["recent_cases"] == -1, "recent_cases"] = pd.NA

    # Clean location strings
    df["location"] = df["location"].str.strip().str.lower()

    # Add source info for traceability
    df["source_file"]   = filepath.name
    df["source_folder"] = filepath.parent.name

    # Drop date_raw — replaced by date
    df = df.drop(columns=["date_raw"])

    # Reorder
    df = df[["date", "month_num", "cluster_id", "recent_cases",
             "total_cases", "cases", "location", "latitude",
             "longitude", "source_file", "source_folder"]]

    return df


def find_files(folder: str) -> list[Path]:
    """Find all .csv and .txt cluster files in a folder."""
    p = Path(folder)
    if not p.exists():
        print(f"  ⚠ Folder not found: {folder}")
        return []
    files = sorted(list(p.glob("*-clusters.csv")) + list(p.glob("*-clusters.txt")))
    return files

# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SGCharts Dengue Cluster Ingestion")
    print("=" * 60)

    all_frames = []

    for folder in INPUT_DIRS:
        files = find_files(folder)
        folder_name = Path(folder).name
        print(f"\nFolder: {folder_name}")
        print(f"Files found: {len(files)}")

        if not files:
            continue

        for i, filepath in enumerate(files, 1):
            df = read_file(filepath)
            if df is not None and not df.empty:
                all_frames.append(df)
                print(f"  [{i:>3}/{len(files)}]  ✓  {filepath.name:<30}  "
                      f"{len(df):>5} rows  |  {df['date'].iloc[0]}")
            else:
                print(f"  [{i:>3}/{len(files)}]  ✗  {filepath.name}  (empty or failed)")

    if not all_frames:
        print("\n✗ No data loaded.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates (in case files overlap between folders)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "location", "cluster_id"])
    dupes_removed = before - len(combined)

    # Sort by date then cluster
    combined = combined.sort_values(["date", "cluster_id"]).reset_index(drop=True)

    # Save
    os.makedirs(BASE_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("✓  DONE")
    print(f"   Rows           : {len(combined):,}")
    print(f"   Snapshots      : {combined['date'].nunique()}")
    valid_dates = combined['date'].dropna()
    print(f"   Date range     : {valid_dates.min()} → {valid_dates.max()}")
    print(f"   Dupes removed  : {dupes_removed}")
    print(f"   cluster_cases  : {combined['recent_cases'].notna().sum():,} valid  |  "
          f"{combined['recent_cases'].isna().sum():,} NaN (before Dec 2013)")
    print(f"   Unique locs    : {combined['location'].nunique():,}")
    print(f"   Columns        : {list(combined.columns)}")
    print(f"   Output         : {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNote: 3-month gap Apr–Jul 2015 between the two folders is unavoidable.")
    print("Next step: main.py Bronze → Silver → Gold pipeline")


if __name__ == "__main__":
    main()
