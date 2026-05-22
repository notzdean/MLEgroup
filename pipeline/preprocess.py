"""
pipeline/preprocess.py
========================
Bronze → Silver cleaning for all four data sources.

Reads from data/bronze/, writes to data/silver/.

Per-source rules
----------------

clusters_clean  (from BRONZE_sgcharts_clusters.csv)
  - Drop source_folder == 'incorrect_latitude_longitude' rows
    NOTE: EDA showed these 17,511 rows have valid coordinates BUT the
    pre-2015 snapshots lack reliable cluster-level geometry — we keep
    only the 'csv' folder rows (Jul 2015 onwards) for label creation.
    The full date range for weather lag features still uses 2013 data.
  - Drop rows outside Singapore bbox (lat 1.15-1.48, lng 103.6-104.1)
  - Drop zero-case and null-case rows
  - Drop exact duplicates on (date, cluster_id, location)
  - Normalise location strings → lowercase stripped

weather_clean  (from BRONZE_mss_weather.csv)
  - Drop temp and wind columns (coverage too sparse: temp 28.7%, wind 34.2%)
  - Cap rainfall > 300mm/day → 300 (outlier treatment)
  - Forward-fill gaps ≤ 3 days per station; interpolate longer gaps
  - Aggregate to daily Singapore-wide mean rainfall across all stations
  - Output: one row per date (not per station)

population_clean  (from BRONZE_subzone_population.csv)
  - Standardise subzone names → UPPERCASE, strip punctuation
  - Drop subzones with total population == 0
  - Derive elderly_pct = (age_65_69 + ... + age_90_over) / total
  - Keep planning_area for imputation reference in feature_engineering.py

subzones_clean  (from BRONZE_subzone_geodata.geojson)
  - Load with geopandas
  - Keep only 330 residential subzones (exclude 49 non-residential)
  - Standardise subzone names to canonical UPPERCASE format
  - Compute area_km2 in EPSG:3414 (SVY21), store geometry in EPSG:4326
  - Geometry repair with buffer(0) for invalid polygons
"""

import pandas as pd
import re
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"

SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.48
SG_LNG_MIN, SG_LNG_MAX = 103.6, 104.1

ELDERLY_COLS = [
    "age_65_69", "age_70_74", "age_75_79",
    "age_80_84", "age_85_89", "age_90_over"
]

# Non-residential subzone keywords — used to filter geodata
NON_RESIDENTIAL_KEYWORDS = [
    "AIRPORT", "PORT", "RESERVOIR", "STRAIT", "SEA",
    "ISLAND", "SOUTHERN ISLANDS", "JURONG ISLAND",
    "NORTH-EASTERN ISLANDS", "SEMAKAU"
]


# ── clusters ─────────────────────────────────────────────────────────────────

def clean_clusters() -> pd.DataFrame:
    print("\n[clusters] Bronze → Silver")
    df = pd.read_csv(BRONZE / "BRONZE_sgcharts_clusters.csv", low_memory=False)
    print(f"  Loaded   : {len(df):,} rows")

    # Keep only the reliable 'csv' folder (Jul 2015 onwards)
    df = df[df["source_folder"] == "csv"].copy()
    print(f"  After folder filter : {len(df):,} rows")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Coerce numeric
    for col in ["latitude", "longitude", "cases", "cluster_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop outside SG bbox
    in_bbox = (
        df["latitude"].between(SG_LAT_MIN, SG_LAT_MAX) &
        df["longitude"].between(SG_LNG_MIN, SG_LNG_MAX)
    )
    before = len(df)
    df = df[in_bbox]
    print(f"  Bbox drop: {before - len(df)} rows")

    # Drop zero/null cases
    before = len(df)
    df = df[df["cases"].notna() & (df["cases"] > 0)]
    df["cases"] = df["cases"].astype(int)
    print(f"  Case drop: {before - len(df)} rows (zero or null)")

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["date", "cluster_id", "location"])
    print(f"  Dedup    : {before - len(df)} rows")

    # Normalise location
    df["location"] = df["location"].str.lower().str.strip()

    df = df[["date", "cluster_id", "location", "latitude", "longitude",
             "cases", "recent_cases", "total_cases", "source_file"]].sort_values(
        ["date", "cluster_id"]).reset_index(drop=True)

    out = SILVER / "clusters_clean.csv"
    df.to_csv(out, index=False)
    print(f"  Written  : {len(df):,} rows → {out.name}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── weather ──────────────────────────────────────────────────────────────────

def clean_weather() -> pd.DataFrame:
    print("\n[weather] Bronze → Silver")
    df = pd.read_csv(BRONZE / "BRONZE_mss_weather.csv", low_memory=False)
    print(f"  Loaded   : {len(df):,} rows, {df['station_id'].nunique()} stations")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # Drop temp and wind columns — too sparse for reliable features
    drop_cols = [c for c in df.columns if any(k in c for k in ["temp", "wind"])]
    df = df.drop(columns=drop_cols)
    print(f"  Dropped sparse cols: {drop_cols}")

    # Cap rainfall outliers
    before_cap = (df["rainfall_mm"] > 300).sum()
    df["rainfall_mm"] = df["rainfall_mm"].clip(upper=300)
    print(f"  Capped {before_cap} rainfall readings > 300mm")

    # Aggregate to daily SG-wide mean across all stations
    daily = (df.groupby("date")["rainfall_mm"]
               .mean()
               .reset_index()
               .rename(columns={"rainfall_mm": "rainfall_mean_mm"}))

    # Forward-fill gaps <= 3 days, interpolate the rest
    daily = daily.sort_values("date").set_index("date")
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily.index.name = "date"
    # Forward-fill up to 3 days
    daily["rainfall_mean_mm"] = daily["rainfall_mean_mm"].ffill(limit=3)
    # Linear interpolate remaining
    daily["rainfall_mean_mm"] = daily["rainfall_mean_mm"].interpolate(method="linear")
    daily = daily.reset_index()

    out = SILVER / "weather_clean.csv"
    daily.to_csv(out, index=False)
    print(f"  Written  : {len(daily):,} daily rows → {out.name}")
    print(f"  Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    return daily


# ── population ───────────────────────────────────────────────────────────────

def clean_population() -> pd.DataFrame:
    print("\n[population] Bronze → Silver")
    df = pd.read_csv(BRONZE / "BRONZE_subzone_population.csv", low_memory=False)
    print(f"  Loaded   : {len(df):,} subzones")

    # Standardise names — UPPERCASE, strip punctuation except hyphens
    def canonical(name):
        if pd.isna(name):
            return name
        name = str(name).upper().strip()
        name = re.sub(r"[^\w\s\-]", "", name)
        return re.sub(r"\s+", " ", name).strip()

    df["subzone"]      = df["subzone"].apply(canonical)
    df["planning_area"] = df["planning_area"].apply(canonical)

    # Drop zero-population subzones
    before = len(df)
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0).astype(int)
    df = df[df["total"] > 0]
    print(f"  Dropped {before - len(df)} zero-population subzones")

    # Coerce age columns
    age_cols = [c for c in df.columns if c.startswith("age_")]
    for col in age_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derive elderly_pct
    elderly_cols_present = [c for c in ELDERLY_COLS if c in df.columns]
    df["elderly_pop"] = df[elderly_cols_present].sum(axis=1)
    df["elderly_pct"] = (df["elderly_pop"] / df["total"]).round(4)

    out = SILVER / "population_clean.csv"
    df.to_csv(out, index=False)
    print(f"  Written  : {len(df):,} rows → {out.name}")
    print(f"  Elderly pct range: {df['elderly_pct'].min():.1%} – {df['elderly_pct'].max():.1%}")
    return df


# ── geodata ───────────────────────────────────────────────────────────────────

def clean_geodata():
    print("\n[geodata] Bronze → Silver")
    try:
        import geopandas as gpd
    except ImportError:
        print("  ⚠ geopandas not installed — skipping geodata cleaning.")
        print("  Install with: pip install geopandas")
        return None

    gdf = gpd.read_file(BRONZE / "BRONZE_subzone_geodata.geojson")
    print(f"  Loaded   : {len(gdf)} features")

    # Standardise names
    gdf["SUBZONE_N"]  = gdf["SUBZONE_N"].str.upper().str.strip()
    gdf["PLN_AREA_N"] = gdf["PLN_AREA_N"].str.upper().str.strip()

    # Exclude non-residential subzones
    mask_non_res = gdf["SUBZONE_N"].apply(
        lambda n: any(kw in str(n) for kw in NON_RESIDENTIAL_KEYWORDS)
    )
    before = len(gdf)
    gdf = gdf[~mask_non_res].copy()
    print(f"  Excluded {before - len(gdf)} non-residential subzones → {len(gdf)} remaining")

    # Repair invalid geometries
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid:
        gdf["geometry"] = gdf["geometry"].buffer(0)
        print(f"  Repaired {invalid} invalid geometries")

    # Compute area in EPSG:3414 (SVY21 — metres)
    gdf_proj = gdf.to_crs(epsg=3414)
    gdf["area_km2"] = (gdf_proj.geometry.area / 1e6).round(4)

    # Store in WGS84
    gdf = gdf.to_crs(epsg=4326)

    # Rename columns for consistency
    gdf = gdf.rename(columns={
        "SUBZONE_C": "subzone_code",
        "SUBZONE_N": "subzone_name",
        "PLN_AREA_N": "planning_area",
        "REGION_N": "region",
    })

    out = SILVER / "subzones_clean.geojson"
    SILVER.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    print(f"  Written  : {len(gdf)} subzones → {out.name}")
    print(f"  Area range: {gdf['area_km2'].min()} – {gdf['area_km2'].max()} km²")
    return gdf


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Preprocessing: Bronze → Silver")
    print("=" * 60)

    SILVER.mkdir(parents=True, exist_ok=True)

    clean_clusters()
    clean_weather()
    clean_population()
    clean_geodata()

    print("\n[done] All Silver tables written.")


if __name__ == "__main__":
    main()
