# build_nm_public_private_categorical.py
# Outputs: state_trust, federal, tribal, private (as GPKG + optional SHP)

import os, io, re, json, zipfile, tempfile
from pathlib import Path
import requests
import geopandas as gpd
from shapely.ops import unary_union
from shapely import make_valid

OUT_GPKG = Path("nm_land_status.gpkg")
TARGET_CRS = 5070
EXPORT_SHAPEFILES_TOO = True     # <- your preference
SIMPLIFY_TOL = 5.0               # meters; set 0/None to disable

URL_PADUS_NM = "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_NM_GDB_KMZ.zip"
URL_AIANNH = "https://www2.census.gov/geo/tiger/TIGER2024/AIANNH/tl_2024_us_aiannh.zip"
URL_STATES = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
SLO_REST = "https://mapservice.nmstatelands.org/arcgis/rest/services/Public/LandStatus_t/MapServer/0/query"

def dl_unzip(url, workdir: Path) -> Path:
    r = requests.get(url, timeout=300); r.raise_for_status()
    zpath = workdir / Path(url.split("?")[0]).name
    zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as z: z.extractall(workdir)
    return workdir

def find_by_stem(dirpath: Path, stem: str, exts=(".gpkg", ".shp", ".geojson", ".json")) -> Path:
    for ext in exts:
        cand = dirpath / f"{stem}{ext}"
        if cand.exists(): return cand
    for p in dirpath.glob(f"{stem}.*"):
        if p.suffix.lower() in {".shp",".gpkg",".geojson",".json"}: return p
    raise FileNotFoundError(f"No supported file for stem '{stem}' in {dirpath}")

def read_any(path: Path, layer=None) -> gpd.GeoDataFrame:
    if path.suffix.lower()==".gdb":
        if not layer: raise ValueError("FileGDB requires a layer name.")
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path) if layer is None else gpd.read_file(path, layer=layer)

def fix_geoms(gdf):
    gdf=gdf.copy()
    gdf["geometry"]=gdf.geometry.apply(make_valid)
    gdf["geometry"]=gdf.buffer(0)
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

def dissolve_all(gdf):
    return gpd.GeoDataFrame(geometry=[unary_union(gdf.geometry)], crs=gdf.crs)

def overlay_difference(a,b):
    if b is None or b.empty: return a.copy()
    b_d = dissolve_all(b)
    out = gpd.overlay(a,b_d,how="difference")
    return fix_geoms(out)

def overlay_intersection(a,b):
    if a.empty or b.empty:
        return gpd.GeoDataFrame(columns=a.columns, crs=a.crs)
    return gpd.overlay(a,b,how="intersection")

def to_target(gdf):
    if gdf.crs is None: raise ValueError("Layer has no CRS.")
    return gdf.to_crs(TARGET_CRS)

def choose_padus_layer(gdb_path: Path) -> str:
    import fiona
    layers = fiona.listlayers(gdb_path)
    cands = [lyr for lyr in layers if re.search(r"Combined", lyr, re.I)]
    def count(lyr):
        with fiona.open(gdb_path, layer=lyr) as src: return len(src)
    if cands: return max(cands, key=count)
    polys=[]
    for lyr in layers:
        with fiona.open(gdb_path, layer=lyr) as src:
            if str(src.schema.get("geometry","")).upper().endswith("POLYGON"):
                polys.append((lyr,len(src)))
    if not polys: raise RuntimeError("No polygon layer found in PAD-US GDB.")
    return max(polys, key=lambda t:t[1])[0]

def fetch_slo_state_trust():
    params={"where":"1=1","outFields":"*","f":"geojson","outSR":"EPSG:4326","returnExceededLimitFeatures":"true"}
    r=requests.get(SLO_REST, params=params, timeout=300); r.raise_for_status()
    return gpd.read_file(io.BytesIO(r.content))

# --- Heuristic: flag PAD-US features that are Federal (owner/manager) ---
FED_KEYWORDS = [
    "BUREAU OF LAND MANAGEMENT","NATIONAL PARK SERVICE","US FOREST SERVICE","FOREST SERVICE",
    "FISH AND WILDLIFE SERVICE","USFWS","BLM","NPS","USFS","DEPARTMENT OF DEFENSE","DOD",
    "BUREAU OF RECLAMATION","BOR","ARMY CORPS","CORPS OF ENGINEERS","NATIONAL MONUMENT",
    "NATIONAL WILDLIFE REFUGE","NATIONAL FOREST","DEPARTMENT OF THE INTERIOR","DOI",
]
def is_federal_row(row):
    # Common PAD-US fields; across versions names vary
    candidates = []
    for f in ("Owner_Type","Owner","Manager_Type","Manager","Own_Type","Mang_Name","Agency","Agency_Name","Loc_Own"):
        if f in row and row[f] is not None:
            candidates.append(str(row[f]).upper())
    txt = " | ".join(candidates)
    if "FEDERAL" in txt:
        return True
    return any(k in txt for k in FED_KEYWORDS)

def main():
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)

        # --- Downloads
        padus_dir = dl_unzip(URL_PADUS_NM, tdir)
        ai_dir = dl_unzip(URL_AIANNH, tdir)
        st_dir = dl_unzip(URL_STATES, tdir)

        # --- NM boundary
        states = read_any(find_by_stem(st_dir, "tl_2024_us_state"))
        nm = dissolve_all(states.loc[states["STUSPS"]=="NM"])
        nm = to_target(fix_geoms(nm))

        # --- PAD-US
        gdbs = list(padus_dir.rglob("*.gdb"))
        if not gdbs: raise RuntimeError("PAD-US GDB not found.")
        padus = read_any(gdbs[0], layer=choose_padus_layer(gdbs[0]))
        padus = to_target(fix_geoms(padus))

        # --- AIANNH (Tribal)
        aiannh = read_any(find_by_stem(ai_dir, "tl_2024_us_aiannh"))
        aiannh = to_target(fix_geoms(aiannh))

        # --- SLO (State Trust)
        slo = to_target(fix_geoms(fetch_slo_state_trust()))

        # Optional simplify (speeds up overlay)
        if SIMPLIFY_TOL and SIMPLIFY_TOL>0:
            for g in (padus, aiannh, slo):
                if not g.empty:
                    g["geometry"]=g.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

        # Clip to NM early
        padus_nm = overlay_intersection(padus, nm)
        tribal_nm = overlay_intersection(aiannh, nm)
        slo_nm = overlay_intersection(slo, nm)

        # Filter PAD-US to land (exclude marine if labeled)
        for fld in ("FeatClass","GAP_Sts"):
            if fld in padus_nm.columns:
                padus_nm = padus_nm[~padus_nm[fld].astype(str).str.contains("Marine", case=False, na=False)]
                break

        # === NEW: categorical outputs ===
        # 1) Tribal land = AIANNH
        tribal = dissolve_all(tribal_nm) if not tribal_nm.empty else gpd.GeoDataFrame(geometry=[], crs=nm.crs)

        # 2) State land = SLO State Trust (surface estate)
        state_trust = dissolve_all(slo_nm) if not slo_nm.empty else gpd.GeoDataFrame(geometry=[], crs=nm.crs)

        # 3) Federal land = PAD-US features with federal owner/manager,
        #    then subtract any Tribal or SLO to avoid overlaps.
        if padus_nm.empty:
            federal = gpd.GeoDataFrame(geometry=[], crs=nm.crs)
        else:
            padus_nm = padus_nm.copy()
            padus_nm["__is_fed__"] = padus_nm.apply(is_federal_row, axis=1)
            fed_raw = padus_nm.loc[padus_nm["__is_fed__"] == True]
            if not fed_raw.empty:
                federal = dissolve_all(fed_raw)
                # remove overlap with Tribal and SLO (tribal/state should win)
                if not tribal.empty:
                    federal = overlay_difference(federal, tribal)
                if not state_trust.empty:
                    federal = overlay_difference(federal, state_trust)
            else:
                federal = gpd.GeoDataFrame(geometry=[], crs=nm.crs)

        # 4) Private land = NM − (federal ∪ tribal ∪ state_trust)
        public_union_parts = [g for g in (federal, tribal, state_trust) if not g.empty]
        if public_union_parts:
            public_union = gpd.GeoDataFrame(geometry=[unary_union([geom for g in public_union_parts for geom in g.geometry])], crs=nm.crs)
            public_union = fix_geoms(public_union)
        else:
            public_union = gpd.GeoDataFrame(geometry=[], crs=nm.crs)
        private_land = overlay_difference(nm, public_union)

        # --- Exports (GPKG)
        if OUT_GPKG.exists(): OUT_GPKG.unlink()
        nm.to_file(OUT_GPKG, layer="nm_boundary", driver="GPKG")
        federal.to_file(OUT_GPKG, layer="nm_federal_land", driver="GPKG")
        state_trust.to_file(OUT_GPKG, layer="nm_state_trust_land", driver="GPKG")
        tribal.to_file(OUT_GPKG, layer="nm_tribal_land", driver="GPKG")
        private_land.to_file(OUT_GPKG, layer="nm_private_land", driver="GPKG")

        # Also keep the inputs clipped to NM for debugging
        padus_nm.to_file(OUT_GPKG, layer="padus_nm_raw", driver="GPKG")
        aiannh.to_file(OUT_GPKG, layer="tribal_aiannh_raw", driver="GPKG")
        slo.to_file(OUT_GPKG, layer="state_trust_raw", driver="GPKG")

        # --- Optional shapefiles
        if EXPORT_SHAPEFILES_TOO:
            shp = Path("nm_land_status_shp"); shp.mkdir(parents=True, exist_ok=True)
            federal.to_file(shp / "nm_federal_land.shp")
            state_trust.to_file(shp / "nm_state_trust_land.shp")
            tribal.to_file(shp / "nm_tribal_land.shp")
            private_land.to_file(shp / "nm_private_land.shp")

        # Quick summary
        def area_km2(g): return (float(g.area.sum())/1e6) if not g.empty else 0.0
        total = area_km2(nm)
        a_fed, a_state, a_trib, a_priv = map(area_km2, (federal, state_trust, tribal, private_land))
        print(f"NM: {total:,.1f} km² | Federal {a_fed:,.1f} | State Trust {a_state:,.1f} | Tribal {a_trib:,.1f} | Private {a_priv:,.1f}")
        if total:
            print(f"Shares: Federal {a_fed/total*100:.1f}% | State {a_state/total*100:.1f}% | Tribal {a_trib/total*100:.1f}% | Private {a_priv/total*100:.1f}%")

if __name__ == "__main__":
    main()
