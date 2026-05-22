"""
pipeline/ingest_geodata.py
============================
Bronze ingestion — URA Master Plan 2019 subzone boundaries.

Input : data/bronze/raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson
Output: data/bronze/BRONZE_subzone_geodata.csv  (flat CSV with key fields)
        data/bronze/BRONZE_subzone_geodata.geojson  (geometry preserved)

Bronze responsibility: load + enforce schema + extract key fields.
No geometry repairs, no filtering of non-residential subzones.
That is Silver's job (preprocess.py).

Design note
-----------
332 features in GeoJSON. 49 are non-residential (water bodies, ports,
parks) and will be excluded in preprocess.py. The 330 residential
subzones are the scoring unit for the model.
"""

import json
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "bronze" / "raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"
OUTPUT_CSV     = ROOT / "data" / "bronze" / "BRONZE_subzone_geodata.csv"
OUTPUT_GEOJSON = ROOT / "data" / "bronze" / "BRONZE_subzone_geodata.geojson"


def main():
    print("=" * 60)
    print("Bronze ingestion — URA subzone geodata")
    print("=" * 60)

    with open(INPUT, encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    print(f"[load]   {len(features)} features from {INPUT.name}")

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        rows.append({
            "subzone_code": props.get("SUBZONE_C"),
            "subzone_name": props.get("SUBZONE_N"),
            "planning_area": props.get("PLN_AREA_N"),
            "region": props.get("REGION_N"),
            "geometry_type": feat.get("geometry", {}).get("type"),
        })

    df = pd.DataFrame(rows)
    df["subzone_name"]   = df["subzone_name"].astype(str).str.upper().str.strip()
    df["planning_area"]  = df["planning_area"].astype(str).str.upper().str.strip()
    df["region"]         = df["region"].astype(str).str.upper().str.strip()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[write]  {len(df)} rows → {OUTPUT_CSV.name}")

    # Also write passthrough GeoJSON for use by preprocess.py (geopandas)
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    print(f"[write]  GeoJSON passthrough → {OUTPUT_GEOJSON.name}")

    print(f"[done]   Planning areas: {df['planning_area'].nunique()} | "
          f"Subzones: {len(df)}")


if __name__ == "__main__":
    main()
