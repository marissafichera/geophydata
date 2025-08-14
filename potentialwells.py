#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 50 potential monitoring-well locations for New Mexico with:
  • At least one site per county
  • Priority by density: ALL(irr, other, pop) > ANY2 > ANY1 > fallback
  • Land priority snapping: state -> private -> federal -> tribal (within search radii)
Outputs:
  out/potential_wells/nm_monitoring_sites_50_prioritized.shp
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.errors import GEOSException
from shapely.ops import unary_union
from shapely import make_valid, set_precision
from shapely.prepared import prep
from sklearn.neighbors import KernelDensity
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

# --------------------
# CONFIG / INPUTS
# --------------------
WELL_FILE   = "waterdata/OSE_Points_of_Diversion.shp"
COUNTY_FILE = "waterdata/tl_2018_nm_county.shp"

# Population density (tract polygons you already built)
POP_GPKG  = "out/pop_density/nm_pop_density_tracts.gpkg"   # preferred
POP_SHP   = "out/pop_density/nm_pop_density_tracts.shp"    # fallback
POP_COLS  = ["DENS_SQMI", "DENS_KM2"]                      # first found is used

# Land ownership polygons (already created earlier)
LAND_DIR           = Path("nm_land_status_shp")
STATE_LAND_FILE    = LAND_DIR / "nm_state_trust_land.shp"
PRIVATE_LAND_FILE  = LAND_DIR / "nm_private_land.shp"
FED_LAND_FILE      = LAND_DIR / "nm_federal_land.shp"
TRIBAL_LAND_FILE   = LAND_DIR / "nm_tribal_land.shp"

# Output
OUT_DIR    = Path("out/potential_wells")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SHP    = OUT_DIR / "nm_monitoring_sites_50_prioritized.shp"

# CRS / KDE grid
CRS_METERS = 26913           # UTM 13N
GRID_RES   = 10_000          # grid spacing (m)
BANDWIDTH  = 15_000          # KDE bandwidth (m)
EXCLUDED_BASINS = ['HS', 'LWD', 'SP']

# Thresholds for "high"
TAU_HIGH_IRR   = 0.75        # percentile-based normalized threshold
TAU_HIGH_OTHER = 0.75
TAU_HIGH_POP   = 0.75

# County + statewide seeding
TOTAL_SITES       = 50
MIN_SPACING_KM    = 20       # spacing for statewide extras
COUNTY_FORCE_FALLBACK = True # if a county has nothing high, still place the best available

# Distance gates to avoid KDE long-tail artifacts
D_MAX_AND = 30_000           # Tier ALL: within 30 km of BOTH well types
D_MAX_ANY = 30_000           # Others: within 30 km of EITHER well type

# Land snapping search radii (m), by priority: state->private->federal->tribal
SEARCH_RADII = [0, 5_000, 10_000, 20_000, 40_000]

# --------------------
# UTILITIES
# --------------------
def clean_for_union(gdf, grid=0.1, min_area=1.0):
    g = gdf.copy()
    g["geometry"] = g.geometry.apply(make_valid).buffer(0)
    if grid and grid > 0:
        g["geometry"] = g.geometry.apply(lambda geom: set_precision(geom, grid))
    g = g[~g.geometry.is_empty & g.geometry.notnull()]
    if min_area and min_area > 0:
        g = g[g.geometry.area >= min_area]
    return g

def robust_union(geoms, chunk=500):
    geoms = [g for g in geoms if g and not g.is_empty]
    if not geoms:
        from shapely.geometry import GeometryCollection
        return GeometryCollection()
    try:
        return unary_union(geoms)
    except GEOSException:
        parts = []
        for i in range(0, len(geoms), chunk):
            parts.append(unary_union(geoms[i:i+chunk]))
        return unary_union(parts)

def mread(path):
    path = Path(path)
    if not path.exists(): return None
    try:
        g = gpd.read_file(path)
        return g if not g.empty else None
    except Exception:
        return None

def load_wells_and_counties():
    wells = gpd.read_file(WELL_FILE)
    counties = gpd.read_file(COUNTY_FILE)
    wells['POD_BASIN'] = wells['pod_basin'].fillna("").str.upper()
    wells['USE']       = wells['use_'].fillna("").str.upper()
    wells['status']    = wells['pod_status'].fillna("").str.upper()
    wells = wells[~wells['POD_BASIN'].isin(EXCLUDED_BASINS)]
    wells = wells[wells['status'].str.contains("ACT", na=False, case=False)]
    wells = wells.to_crs(epsg=CRS_METERS)
    counties = counties.to_crs(wells.crs)
    return wells, counties

def counties_grid(counties: gpd.GeoDataFrame):
    xmin, ymin, xmax, ymax = counties.total_bounds
    xs = np.arange(xmin, xmax, GRID_RES)
    ys = np.arange(ymin, ymax, GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy

def kde_intensity(points_gdf, xx, yy, bandwidth=BANDWIDTH):
    """n * pdf so intensities comparable across subsets."""
    if points_gdf is None or points_gdf.empty:
        return np.zeros_like(xx, dtype=float)
    pts = np.vstack([points_gdf.geometry.x, points_gdf.geometry.y]).T
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth).fit(pts)
    pdf = np.exp(kde.score_samples(grid)).reshape(xx.shape)
    return len(pts) * pdf

def norm01(a, lo_p=50, hi_p=99):
    a = a.astype(float)
    lo = np.nanpercentile(a, lo_p)
    hi = np.nanpercentile(a, hi_p)
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a)
    x = (a - lo) / (hi - lo)
    x = np.clip(x, 0, 1); x[~np.isfinite(x)] = 0
    return x

def load_population_layer(target_crs):
    g = None
    gp = Path(POP_GPKG)
    if gp.exists():
        try:
            g = gpd.read_file(gp, layer="nm_pop_density_tracts")
        except Exception:
            g = None
    if g is None:
        gs = Path(POP_SHP)
        if gs.exists():
            g = gpd.read_file(gs)
    if g is None or g.empty:
        return None, None
    if g.crs != target_crs:
        g = g.to_crs(target_crs)
    col = next((c for c in POP_COLS if c in g.columns), None)
    return g, col

def sample_polygon_attr_to_grid(poly_gdf, attr, xx, yy):
    """Map polygon attribute to grid nodes via point-in-polygon."""
    if poly_gdf is None or attr is None:
        return np.zeros_like(xx, dtype=float)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=poly_gdf.crs)
    joined = gpd.sjoin(pts, poly_gdf[[attr, "geometry"]], how="left", predicate="within")
    out = np.zeros(xx.size, dtype=float)
    vals = pd.to_numeric(joined[attr], errors="coerce").fillna(0).to_numpy()
    out[joined.index_left.to_numpy()] = vals
    return out.reshape(xx.shape)

def nearest_distance_raster(points_gdf, xx, yy):
    """Distance (m) from each grid node to nearest point in points_gdf."""
    if points_gdf is None or points_gdf.empty:
        return np.full_like(xx, np.inf, dtype=float)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    pts  = np.vstack([points_gdf.geometry.x, points_gdf.geometry.y]).T
    tree = cKDTree(pts)
    d, _ = tree.query(grid, k=1)
    return d.reshape(xx.shape)

def nonmax_peaks(z, size=5, n=10):
    filt = maximum_filter(z, size=size) == z
    rr, cc = np.where(filt)
    vals = z[rr, cc]
    order = np.argsort(vals)[::-1]
    rr, cc = rr[order][:n], cc[order][:n]
    return list(zip(rr, cc, vals[order][:n]))

def enforce_min_spacing(cands_xy, min_dist):
    kept = []
    for x, y, s in sorted(cands_xy, key=lambda t: t[2], reverse=True):
        if all(np.hypot(x-kx, y-ky) >= min_dist for kx, ky, _ in kept):
            kept.append((x, y, s))
    return kept

# ----- land snapping -----
def best_grid_point_in(inter_geom, xx, yy, Z, p0):
    if inter_geom.is_empty: return None
    ip = prep(inter_geom)
    minx, miny, maxx, maxy = inter_geom.bounds
    xs, ys, zs = xx.ravel(), yy.ravel(), Z.ravel()
    box = (xs>=minx)&(xs<=maxx)&(ys>=miny)&(ys<=maxy)
    idx = np.where(box)[0]
    if idx.size == 0: return inter_geom.representative_point()
    inside = [i for i in idx if ip.contains(Point(xs[i], ys[i])) or inter_geom.touches(Point(xs[i], ys[i]))]
    if not inside: return inter_geom.representative_point()
    best = max(inside, key=lambda i: (zs[i], -Point(xs[i], ys[i]).distance(p0)))
    return Point(xs[best], ys[best])

def load_land_masks(target_crs):
    classes = [
        ("state",   mread(STATE_LAND_FILE)),
        ("private", mread(PRIVATE_LAND_FILE)),
        ("federal", mread(FED_LAND_FILE)),
        ("tribal",  mread(TRIBAL_LAND_FILE)),
    ]
    out = []
    for label, g in classes:
        if g is None:
            out.append((label, None))
            continue
        g = g.to_crs(target_crs)
        g = clean_for_union(g, grid=0.1, min_area=1.0)
        out.append((label, robust_union(list(g.geometry))))
    return out  # list of (label, dissolved geometry)

def snap_points_to_land(points_gdf, land_masks, xx, yy, Z, radii=SEARCH_RADII):
    rows = []
    for _, r in points_gdf.iterrows():
        p, tier, score = r.geometry, r["tier"], r["score"]
        chosen_p, chosen_label, chosen_r = p, "unclassified", None
        for rad in radii:
            hit = False
            for label, dissolved in land_masks:
                if dissolved is None: continue
                if rad == 0 and p.within(dissolved):
                    chosen_p, chosen_label, chosen_r = p, label, 0; hit = True; break
                if rad > 0:
                    inter = dissolved.intersection(p.buffer(rad))
                    if not inter.is_empty:
                        npnt = best_grid_point_in(inter, xx, yy, Z, p)
                        if npnt is not None and npnt.is_valid:
                            chosen_p, chosen_label, chosen_r = npnt, label, rad; hit = True; break
            if hit: break
        rows.append({"tier": tier, "score": float(score),
                     "land_class": chosen_label, "search_radius_m": 0 if chosen_r is None else int(chosen_r),
                     "geometry": chosen_p})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=points_gdf.crs)

# --------------------
# CORE SELECTION
# --------------------
def pick_county_site(county_geom, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK):
    """Return (x,y,score,tier) for best site in county by priority; None if nothing."""
    # boolean mask of grid nodes inside county
    inside = np.fromiter((Point(x, y).within(county_geom) for x, y in zip(xx.ravel(), yy.ravel())),
                         dtype=bool, count=xx.size)
    # 1) ALL-three high
    mask = inside & M_AND3.ravel()
    if mask.any():
        z = S_AND3.ravel()[mask]; xs = xx.ravel()[mask]; ys = yy.ravel()[mask]
        i = np.argmax(z)
        return xs[i], ys[i], z[i], "tier_all3"
    # 2) any two high
    mask = inside & M_AND2.ravel()
    if mask.any():
        z = S_AND2.ravel()[mask]; xs = xx.ravel()[mask]; ys = yy.ravel()[mask]
        i = np.argmax(z)
        return xs[i], ys[i], z[i], "tier_any2"
    # 3) any one high
    mask = inside & M_ANY1.ravel()
    if mask.any():
        z = S_ANY1.ravel()[mask]; xs = xx.ravel()[mask]; ys = yy.ravel()[mask]
        i = np.argmax(z)
        return xs[i], ys[i], z[i], "tier_any1"
    # 4) fallback (optional)
    if COUNTY_FORCE_FALLBACK and inside.any():
        z = S_FALLBACK.ravel()[inside]; xs = xx.ravel()[inside]; ys = yy.ravel()[inside]
        i = np.argmax(z)
        return xs[i], ys[i], z[i], "tier_fallback"
    return None

def pick_statewide_extras(xx, yy, already_pts, need_n, S, M, min_spacing_m, tier_label):
    """Pick additional peaks from surface S within mask M, spaced from already_pts.
       Returns list of (x, y, score, tier_label)."""
    if need_n <= 0:
        return []
    Z = S.copy()
    Z[~M] = -1.0  # invalidate pixels outside mask
    peaks = nonmax_peaks(Z, size=5, n=max(need_n*3, need_n))  # oversample
    cands = [(xx[r, c], yy[r, c], float(val)) for r, c, val in peaks if val > 0]

    # Normalize previous picks to (x, y) for spacing checks (supports 3- or 4-tuples)
    def xy_only(seq):
        out = []
        for t in seq:
            if isinstance(t, (tuple, list)) and len(t) >= 2:
                out.append((t[0], t[1]))
            else:
                try:
                    out.append((t.geometry.x, t.geometry.y))
                except Exception:
                    pass
        return out

    existing_xy = xy_only(already_pts)
    kept = []  # will hold (x, y, score, tier_label)

    for x, y, s in sorted(cands, key=lambda t: t[2], reverse=True):
        kept_xy = xy_only(kept)
        if all(np.hypot(x - x0, y - y0) >= min_spacing_m for x0, y0 in (existing_xy + kept_xy)):
            kept.append((x, y, s, tier_label))
            if len(kept) >= need_n:
                break

    return kept


# --------------------
# MAIN
# --------------------
def main():
    wells, counties = load_wells_and_counties()
    irr = wells[wells['USE'] == 'IRR']
    oth = wells[wells['USE'] != 'IRR']

    # Grid
    xx, yy = counties_grid(counties)

    # Densities (normalized to [0,1] robustly)
    lam_irr   = kde_intensity(irr, xx, yy, BANDWIDTH)
    lam_oth   = kde_intensity(oth, xx, yy, BANDWIDTH)
    irr_n     = norm01(lam_irr)    # 0..1
    oth_n     = norm01(lam_oth)

    # Population to grid, normalized
    pop_gdf, pop_col = load_population_layer(counties.crs)
    POP      = sample_polygon_attr_to_grid(pop_gdf, pop_col, xx, yy) if pop_gdf is not None else np.zeros_like(xx)
    pop_n    = norm01(POP)

    # "high" flags per layer
    irr_h  = irr_n >= TAU_HIGH_IRR
    oth_h  = oth_n >= TAU_HIGH_OTHER
    pop_h  = pop_n >= TAU_HIGH_POP

    # Distance gates (avoid KDE long tails)
    d_irr  = nearest_distance_raster(irr, xx, yy)
    d_oth  = nearest_distance_raster(oth, xx, yy)
    mask_and = (d_irr <= D_MAX_AND) & (d_oth <= D_MAX_AND)
    mask_any = (d_irr <= D_MAX_ANY) | (d_oth <= D_MAX_ANY)

    # Priority surfaces
    # ALL 3 high → score = min(irr, oth, pop)
    S_AND3 = np.minimum.reduce([irr_n, oth_n, pop_n])
    M_AND3 = irr_h & oth_h & pop_h & mask_and

    # ANY 2 high → score = max( min(irr,oth), min(irr,pop), min(oth,pop) )
    pair1  = np.minimum(irr_n, oth_n)
    pair2  = np.minimum(irr_n, pop_n)
    pair3  = np.minimum(oth_n, pop_n)
    S_AND2 = np.maximum.reduce([pair1, pair2, pair3])
    M_AND2 = ((irr_h & oth_h) | (irr_h & pop_h) | (oth_h & pop_h)) & mask_any

    # ANY 1 high → score = max(irr, oth, pop)
    S_ANY1 = np.maximum.reduce([irr_n, oth_n, pop_n])
    M_ANY1 = (irr_h | oth_h | pop_h) & mask_any

    # Fallback score (when county has no high pixels): sum of normalized layers, still prefer near wells
    S_FALLBACK = (irr_n + oth_n + 0.5*pop_n) * (mask_any.astype(float))

    # ---------------- At least one per county ----------------
    picks = []
    for _, row in counties.iterrows():
        res = pick_county_site(row.geometry, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK)
        if res is None:  # extremely unlikely
            continue
        x, y, s, tier = res
        picks.append((x, y, s, tier))

    # Top-up to TOTAL_SITES
    need = max(0, TOTAL_SITES - len(picks))
    min_space_m = MIN_SPACING_KM * 1000.0

    if need > 0:
        picks += pick_statewide_extras(xx, yy, picks, need, S_AND3, M_AND3, min_space_m, "tier_all3")
        need = max(0, TOTAL_SITES - len(picks))
    if need > 0:
        picks += pick_statewide_extras(xx, yy, picks, need, S_AND2, M_AND2, min_space_m, "tier_any2")
        need = max(0, TOTAL_SITES - len(picks))
    if need > 0:
        picks += pick_statewide_extras(xx, yy, picks, need, S_ANY1, M_ANY1, min_space_m, "tier_any1")

    # Trim in case we overshot (shouldn't, but for safety)
    picks = sorted(picks, key=lambda t: t[3]!="tier_all3")  # keep ALL3 first
    # Keep ALL3 first; trim to 50 if needed
    picks = sorted(picks, key=lambda t: t[3] != "tier_all3")[:TOTAL_SITES]

    gdf = gpd.GeoDataFrame(
        {"tier": [t[3] for t in picks],
         "score": [t[2] for t in picks]},
        geometry=[Point(t[0], t[1]) for t in picks],
        crs=counties.crs
    )

    # # Pack into GeoDataFrame with tier & score
    # gdf = gpd.GeoDataFrame(
    #     {"tier": [t for *_, t in picks], "score": [s for _,_,s,_ in picks]},
    #     geometry=[Point(x, y) for x, y, *_ in picks],
    #     crs=counties.crs
    # )

    # ---------------- Land snapping (respect each tier's surface when choosing inside land) ----------------
    # Use Z by tier for best-within-polygon decision
    Z_by_tier = {
        "tier_all3": S_AND3,
        "tier_any2": S_AND2,
        "tier_any1": S_ANY1,
        "tier_fallback": S_FALLBACK
    }
    land_masks = load_land_masks(gdf.crs)

    # Snap per-tier to keep the same score surface
    snapped_parts = []
    for tier_name, part in gdf.groupby("tier"):
        Z = Z_by_tier[tier_name]
        snapped = snap_points_to_land(part, land_masks, xx, yy, Z, radii=SEARCH_RADII)
        snapped_parts.append(snapped)
    out = pd.concat(snapped_parts, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

    # Save
    out.to_file(OUT_SHP)
    print(f"Saved: {OUT_SHP.resolve()}")
    if "land_class" in out.columns:
        print("\nBy land_class:\n", out["land_class"].value_counts())
    if "tier" in out.columns:
        print("\nBy tier:\n", out["tier"].value_counts())

if __name__ == "__main__":
    main()
