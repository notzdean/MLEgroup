"""
pipeline/ingest_population.py
================================
Bronze ingestion — SingStat Census 2020 population by subzone.

Input : data/bronze/raw_singstat_pop_17560.csv  (subzone-level, age breakdown)
        data/bronze/BRONZE_planning_area_population.csv (planning area totals)
Output: data/bronze/BRONZE_subzone_population.csv
        data/bronze/BRONZE_planning_area_population.csv  (passthrough — already clean)

Source file structure
---------------------
raw_singstat_pop_17560.csv is a SingStat Table Builder export with:
  - Rows 0-9  : metadata headers (skipped)
  - Row 10    : column header level 1 (Planning Area/Subzone, then Total/Male/Female)
  - Row 11    : column header level 2 (age groups: Total, 0-4, 5-9 ... 90 & Over)
  - Rows 12+  : data (planning area totals interleaved with subzone rows)
  - Last 10   : footer (skipped)

Planning area rows are identified by "- Total" suffix or no leading spaces.
Subzone rows have 2 leading spaces.

Bronze responsibility: parse the awkward layout + enforce schema.
No imputation or derived columns here — that is Silver's job (preprocess.py).
"""

import pandas as pd
import re
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "bronze" / "raw_singstat_pop_17560.csv"
OUTPUT = ROOT / "data" / "bronze" / "BRONZE_subzone_population.csv"

# Age group columns present in the file (Total sex section only — cols 1-20)
AGE_GROUPS = [
    "total", "age_0_4", "age_5_9", "age_10_14", "age_15_19",
    "age_20_24", "age_25_29", "age_30_34", "age_35_39", "age_40_44",
    "age_45_49", "age_50_54", "age_55_59", "age_60_64", "age_65_69",
    "age_70_74", "age_75_79", "age_80_84", "age_85_89", "age_90_over",
]


def parse_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, low_memory=False)

    # Data rows: row 12 to last data row (before footer ~row 410)
    # Detect footer start: first row where col 0 contains 'Values are shown'
    footer_start = None
    for i, val in enumerate(raw.iloc[:, 0]):
        if isinstance(val, str) and "Values are shown" in val:
            footer_start = i
            break
    if footer_start is None:
        footer_start = len(raw) - 10

    data = raw.iloc[12:footer_start, :21].copy()
    data.columns = ["name"] + AGE_GROUPS
    data = data.reset_index(drop=True)

    # Identify row type
    data["is_planning_area"] = data["name"].str.strip().str.endswith("- Total") | \
                               (~data["name"].str.startswith("  ") & \
                                ~data["name"].str.strip().str.startswith("Total"))
    data["planning_area"] = None

    current_pa = None
    rows = []
    for _, row in data.iterrows():
        name = str(row["name"]).strip()
        if row["is_planning_area"]:
            current_pa = name.replace(" - Total", "").strip()
            continue  # planning area totals go to separate file
        rows.append({
            "planning_area": current_pa,
            "subzone": name,
            **{col: row[col] for col in AGE_GROUPS}
        })

    df = pd.DataFrame(rows)

    # Enforce numeric types
    for col in AGE_GROUPS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def main():
    print("=" * 60)
    print("Bronze ingestion — SingStat subzone population")
    print("=" * 60)

    df = parse_raw(INPUT)
    print(f"[parse]  {len(df):,} subzone rows extracted")
    print(f"         Planning areas: {df['planning_area'].nunique()}")
    print(f"         Sample: {list(df['subzone'].head(5))}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"[write]  {len(df):,} rows → {OUTPUT.name}")
    print(f"[done]   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
