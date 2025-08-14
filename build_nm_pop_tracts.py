# build_nm_pop_density_tracts.py
# Python 3.9+   Requires: requests, pandas, geopandas, shapely
# pip install requests pandas geopandas shapely

import os
import io
import zipfile
import requests
from pathlib import Path

import pandas as pd
import geopandas as gpd

# =============================
# CONFIG
# =============================
ACS_YEAR = 2023                         # latest ACS 5-year as of Aug 2025
STATE_FIPS = "35"                       # New Mexico
ACS_VAR = "B01003_001E"                 # Total population (ACS 5-year)
OUT_DIR = Path("out/potential_wells/out/pop_density")
OUT_SHP = OUT_DIR / "nm_pop_density_tracts.shp"
OUT_GPKG = OUT_DIR / "nm_pop_density_tracts.gpkg"
SAVE_CSV = OUT_DIR / "nm_pop_density_tracts.csv"

# TIGER/Line 2024 tract shapefile for NM (state=35)
TIGER_TRACT_ZIP = f"https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_{STATE_FIPS}_tract.zip"

# ACS 5-year API (tracts within NM)
# If you have a key: append &key=YOUR_KEY
ACS_API = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
ACS_QUERY = f"{ACS_API}?get=NAME,{ACS_VAR}&for=tract:*&in=state:{STATE_FIPS}"

SQM_PER_SQMI = 2_589_988.110336  # square meters in a square mile


def dl_unzip(url: str, to_dir: Path) -> Path:
    to_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(to_dir)
    return to_dir


def fetch_acs_population() -> pd.DataFrame:
    key = os.getenv("CENSUS_API_KEY")
    url = f"{ACS_QUERY}&key={key}" if key else ACS_QUERY
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    data = r.json()
    cols, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=cols)

    # Keep strings as-is to preserve zero padding
    for c in ["state", "county", "tract"]:
        df[c] = df[c].astype(str)

    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df.rename(columns={ACS_VAR: "POP"}, inplace=True)
    # numeric pop
    df["POP"] = pd.to_numeric(df["POP"], errors="coerce").fillna(0).astype(int)
    return df[["GEOID", "NAME", "POP"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download & read TIGER tracts (has ALAND in m^2)
    shp_dir = OUT_DIR / "tiger_tracts_2024_nm"
    dl_unzip(TIGER_TRACT_ZIP, shp_dir)
    tracts = gpd.read_file(next(shp_dir.glob("tl_2024_*_tract.shp")))
    # Keep only needed columns to avoid shapefile field name headaches
    keep_cols = [c for c in tracts.columns if c.upper() in {"GEOID", "ALAND", "AWATER", "NAME"} or c == "geometry"]
    tracts = tracts[keep_cols].copy()

    # 2) Fetch ACS population by tract
    pop = fetch_acs_population()

    # Keep only needed columns
    keep_cols = [c for c in tracts.columns if c.upper() in {"GEOID", "ALAND", "AWATER", "NAME", "NAMELSAD"}] + [
        "geometry"]
    tracts = tracts[keep_cols].copy()

    # Create a robust name column
    if "NAME" in tracts.columns:
        tracts = tracts.rename(columns={"NAME": "TRACT_NAME"})
    elif "NAMELSAD" in tracts.columns:
        tracts = tracts.rename(columns={"NAMELSAD": "TRACT_NAME"})
    else:
        tracts["TRACT_NAME"] = tracts["GEOID"]  # fallback


    # 3) Join and compute densities
    g = tracts.merge(pop, on="GEOID", how="left")
    g["ALAND"] = pd.to_numeric(g["ALAND"], errors="coerce").fillna(0)

    # Avoid divide-by-zero
    land_km2 = g["ALAND"] / 1_000_000.0
    land_sqmi = g["ALAND"] / SQM_PER_SQMI
    g["dens_km2"] = g["POP"] / land_km2.replace(0, pd.NA)
    g["dens_sqmi"] = g["POP"] / land_sqmi.replace(0, pd.NA)

    # Round densities for readability (keep full precision if you prefer)
    g["dens_km2"] = g["dens_km2"].round(2)
    g["dens_sqmi"] = g["dens_sqmi"].round(2)

    # 4) Export
    g = g.rename(columns={"dens_km2": "DENS_KM2", "dens_sqmi": "DENS_SQMI"})

    # Ensure the name column exists
    if "TRACT_NAME" not in g.columns:
        # last-resort fallback
        g["TRACT_NAME"] = g["NAME"] if "NAME" in g.columns else g.get("NAMELSAD", g["GEOID"])

    # Choose columns that actually exist to avoid KeyError on some vintages
    want = ["GEOID", "TRACT_NAME", "POP", "ALAND", "AWATER", "DENS_KM2", "DENS_SQMI", "geometry"]
    cols = [c for c in want if c in g.columns]
    shp_out = g[cols]

    # Write GPKG and SHP
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    g.to_file(OUT_GPKG, layer="nm_pop_density_tracts", driver="GPKG")
    shp_out.to_file(OUT_SHP)
    shp_out.drop(columns="geometry").to_csv(SAVE_CSV, index=False)

    # Preserve CRS from TIGER (NAD83)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    g.to_file(OUT_GPKG, layer="nm_pop_density_tracts", driver="GPKG")
    shp_out.to_file(OUT_SHP)

    # Optional CSV (no geometry)
    shp_out.drop(columns="geometry").to_csv(SAVE_CSV, index=False)

    # Quick summary
    print(f"Wrote:\n - {OUT_SHP}\n - {OUT_GPKG}\n - {SAVE_CSV}")
    print(f"ACS {ACS_YEAR} variable {ACS_VAR} joined to TIGER 2024 tracts. "
          f"Densities: people/km² and people/mi².")


if __name__ == "__main__":
    main()
