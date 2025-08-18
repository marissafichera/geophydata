#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package the monitoring-wells map data into:
  1) A single GeoPackage with one layer per dataset
  2) Zipped shapefiles (one zip per layer)

Both QGIS and ArcGIS can read the GeoPackage; the zipped SHPs provide
an extra compatibility path for ArcGIS workflows.

Outputs under: out/potential_wells/package/
"""

from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

# -----------------------------
# INPUTS (adjust if paths differ)
# -----------------------------
COUNTIES   = Path("waterdata/tl_2018_nm_county.shp")
WELL_FILE  = Path("waterdata/OSE_Points_of_Diversion.shp")
LAND_DIR   = Path("nm_land_status_shp")
NM_BOUND   = LAND_DIR / "nm_boundary.shp"  # optional
FED_SHP    = LAND_DIR / "nm_federal_land.shp"
STATE_SHP  = LAND_DIR / "nm_state_trust_land.shp"
TRIBAL_SHP = LAND_DIR / "nm_tribal_land.shp"
PRIV_SHP   = LAND_DIR / "nm_private_land.shp"

POP_SHP    = Path("out/pop_density/nm_pop_density_tracts.shp")
POP_GPKG   = Path("out/pop_density/nm_pop_density_tracts.gpkg")  # layer="nm_pop_density_tracts"
POP_LAYER  = "nm_pop_density_tracts"  # layer name inside GPKG

FINAL_POINTS = Path("out/potential_wells/nm_monitoring_sites_county_2pass.shp")

# -----------------------------
# OUTPUTS
# -----------------------------
BASE_OUT = Path("out/potential_wells/package")
GPKG     = BASE_OUT / "nm_monitoring_package.gpkg"
SHP_DIR  = BASE_OUT / "shp_layers"
ZIP_DIR  = BASE_OUT / "shp_zips"
README  = BASE_OUT / "README.txt"

CRS_METERS = 26913  # UTM 13N used in your analysis & maps

# -----------------------------
# HELPERS
# -----------------------------
def mread(p: Path, **kwargs):
    """Read if exists & non-empty, else return None."""
    if not p or not p.exists():
        print(f"[skip] missing: {p}")
        return None
    try:
        g = gpd.read_file(p, **kwargs)
        if g is not None and not g.empty:
            return g
        print(f"[skip] empty: {p}")
        return None
    except Exception as e:
        print(f"[skip] unreadable {p}: {e}")
        return None

def to_target_crs(g: gpd.GeoDataFrame, epsg=CRS_METERS):
    if g is None: return None
    return g if (g.crs and g.crs.to_epsg() == epsg) else g.to_crs(epsg=epsg)

def explode_polygonal(g: gpd.GeoDataFrame):
    """Explode collections and keep only Polygon/MultiPolygon (for SHP compatibility)."""
    if g is None or g.empty: return g
    gg = g.explode(index_parts=False, ignore_index=True)
    gg = gg[gg.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gg = gg[~gg.geometry.is_empty]
    return gg

def drop_big_dbase_fields(g: gpd.GeoDataFrame, cols=("ALAND", "AWATER")):
    """Drop huge integer fields that overflow dBase in Shapefiles."""
    if g is None or g.empty: return g
    keep = [c for c in g.columns if c not in cols and c != "geometry"]
    cols_out = keep + (["geometry"] if "geometry" in g.columns else [])
    return g[cols_out]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_gpkg_layer(g: gpd.GeoDataFrame, gpkg_path: Path, layer_name: str):
    """Write/overwrite a layer in the GeoPackage."""
    if g is None or g.empty:
        print(f"[skip] {layer_name} -> empty")
        return
    g.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    print(f"[gpkg] {layer_name} ({len(g)} features)")

def write_shp(g: gpd.GeoDataFrame, shp_dir: Path, name: str, polygonal=False, drop_big_ints=False):
    """Write a Shapefile (SHP); optionally polygonize and drop huge ints first."""
    if g is None or g.empty:
        print(f"[skip] SHP {name} -> empty")
        return None
    gg = g
    if polygonal:
        gg = explode_polygonal(gg)
        if gg is None or gg.empty:
            print(f"[skip] SHP {name} -> no polygonal parts")
            return None
    if drop_big_ints:
        gg = drop_big_dbase_fields(gg)
    shp_path = shp_dir / f"{name}.shp"
    gg.to_file(shp_path, driver="ESRI Shapefile")
    print(f"[shp] {name} ({len(gg)} features)")
    return shp_path

def zip_shp(shp_path: Path, out_zip_dir: Path):
    """Zip a single shapefile (all sidecar files) into <name>.zip (one SHP per zip)."""
    if shp_path is None: return
    base = shp_path.with_suffix("")  # strip .shp
    stem = base.name
    parent = base.parent
    # collect all sidecars
    sidecars = list(parent.glob(stem + ".*"))
    out_zip = out_zip_dir / f"{stem}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sidecars:
            z.write(f, arcname=f.name)
    print(f"[zip] {out_zip.name}")
    return out_zip

def write_readme(path: Path, layers, crs_epsg=CRS_METERS):
    txt = []
    txt.append("New Mexico Monitoring Wells – Data Package\n")
    txt.append("Contents:\n")
    txt.append(f"- GeoPackage: {GPKG.name}\n- Zipped Shapefiles: one .zip per layer in shp_zips/\n")
    txt.append(f"CRS: EPSG:{crs_epsg} (UTM Zone 13N, meters)\n")
    txt.append("Layers:\n")
    for name, desc in layers:
        txt.append(f"  - {name}: {desc}\n")
    txt.append("\nHow to use:\n")
    txt.append("  QGIS: Add Layer → Vector → select the GeoPackage and pick layers, or drag the zipped shapefiles in.\n")
    txt.append("  ArcGIS Pro: Add Data → navigate to the GeoPackage or unzip a single layer’s .zip and add the .shp.\n")
    txt.append("\nNotes:\n")
    txt.append("- Shapefile exports drop very large integer fields (e.g., ALAND, AWATER) due to dBase limits.\n")
    txt.append("- Land layers were cleaned to polygon-only for SHP compatibility.\n")
    path.write_text("".join(txt), encoding="utf-8")
    print(f"[ok] wrote {path.name}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(BASE_OUT)
    ensure_dir(SHP_DIR)
    ensure_dir(ZIP_DIR)
    if GPKG.exists():
        GPKG.unlink()  # start fresh so layer list is clean

    # --- Load & prep base layers ---
    counties = to_target_crs(mread(COUNTIES))
    nm = to_target_crs(mread(NM_BOUND))
    if nm is None:
        nm = counties.dissolve().reset_index(drop=True)

    wells = to_target_crs(mread(WELL_FILE))
    if wells is not None:
        wells["USE"]    = wells.get("use_", "").astype(str).str.upper()
        wells["status"] = wells.get("pod_status", "").astype(str).str.upper()
        wells = wells[wells["status"].str.contains("ACT", na=False)]
        irr = wells[wells["USE"] == "IRR"]
        oth = wells[wells["USE"] != "IRR"]
    else:
        irr = None
        oth = None

    # Population layer
    pop = None
    if POP_SHP.exists():
        pop = to_target_crs(mread(POP_SHP))
    elif POP_GPKG.exists():
        pop = to_target_crs(mread(POP_GPKG, layer=POP_LAYER))

    # Land layers
    state   = to_target_crs(mread(STATE_SHP))
    federal = to_target_crs(mread(FED_SHP))
    tribal  = to_target_crs(mread(TRIBAL_SHP))
    private = to_target_crs(mread(PRIV_SHP))

    # Final points (and primary/secondary splits if present)
    points = to_target_crs(mread(FINAL_POINTS))
    p1 = p2 = None
    if points is not None and "site_rank" in points.columns:
        p1 = points[points["site_rank"] == 1]
        p2 = points[points["site_rank"] == 2]

    # --- Write GeoPackage ---
    print("\n=== Writing GeoPackage ===")
    write_gpkg_layer(counties, GPKG, "nm_counties")
    write_gpkg_layer(nm,       GPKG, "nm_boundary")
    write_gpkg_layer(irr,      GPKG, "wells_irrigation")
    write_gpkg_layer(oth,      GPKG, "wells_other")
    write_gpkg_layer(pop,      GPKG, "nm_pop_density_tracts")
    write_gpkg_layer(state,    GPKG, "nm_land_state")
    write_gpkg_layer(federal,  GPKG, "nm_land_federal")
    write_gpkg_layer(tribal,   GPKG, "nm_land_tribal")
    write_gpkg_layer(private,  GPKG, "nm_land_private")
    write_gpkg_layer(points,   GPKG, "monitoring_sites_all")
    write_gpkg_layer(p1,       GPKG, "monitoring_sites_primary")
    write_gpkg_layer(p2,       GPKG, "monitoring_sites_secondary")

    # --- Write Shapefiles + zip each layer ---
    print("\n=== Writing Shapefiles & Zips ===")
    shp_paths = []
    shp_paths += [write_shp(drop_big_dbase_fields(counties), SHP_DIR, "nm_counties", polygonal=True,  drop_big_ints=True)]
    shp_paths += [write_shp(nm,        SHP_DIR, "nm_boundary",     polygonal=True,  drop_big_ints=False)]
    shp_paths += [write_shp(irr,       SHP_DIR, "wells_irrigation", polygonal=False, drop_big_ints=False)]
    shp_paths += [write_shp(oth,       SHP_DIR, "wells_other",      polygonal=False, drop_big_ints=False)]
    shp_paths += [write_shp(pop,       SHP_DIR, "nm_pop_density_tracts", polygonal=True, drop_big_ints=False)]
    shp_paths += [write_shp(state,     SHP_DIR, "nm_land_state",    polygonal=True,  drop_big_ints=False)]
    shp_paths += [write_shp(federal,   SHP_DIR, "nm_land_federal",  polygonal=True,  drop_big_ints=False)]
    shp_paths += [write_shp(tribal,    SHP_DIR, "nm_land_tribal",   polygonal=True,  drop_big_ints=False)]
    shp_paths += [write_shp(private,   SHP_DIR, "nm_land_private",  polygonal=True,  drop_big_ints=False)]
    shp_paths += [write_shp(points,    SHP_DIR, "monitoring_sites_all", polygonal=False, drop_big_ints=False)]
    shp_paths += [write_shp(p1,        SHP_DIR, "monitoring_sites_primary", polygonal=False, drop_big_ints=False)]
    shp_paths += [write_shp(p2,        SHP_DIR, "monitoring_sites_secondary", polygonal=False, drop_big_ints=False)]

    print("\n--- Zipping each shapefile ---")
    for shp in [p for p in shp_paths if p is not None]:
        zip_shp(shp, ZIP_DIR)

    # --- README ---
    layers_list = [
        ("nm_counties", "New Mexico county boundaries"),
        ("nm_boundary", "New Mexico state boundary"),
        ("wells_irrigation", "Active OSE irrigation wells"),
        ("wells_other", "Active OSE non-irrigation wells"),
        ("nm_pop_density_tracts", "Census tract population density"),
        ("nm_land_state", "State trust land"),
        ("nm_land_federal", "Federal land"),
        ("nm_land_tribal", "Tribal land"),
        ("nm_land_private", "Private land"),
        ("monitoring_sites_all", "All selected monitoring sites"),
        ("monitoring_sites_primary", "Primary (Pass 1) sites"),
        ("monitoring_sites_secondary", "Secondary (Pass 2) sites"),
    ]
    write_readme(README, layers_list)

    print("\nDone.")
    print(f"GeoPackage: {GPKG.resolve()}")
    print(f"Zipped SHPs: {ZIP_DIR.resolve()}")
    print(f"README:     {README.resolve()}")

if __name__ == "__main__":
    main()
