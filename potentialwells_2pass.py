#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-pass county selection of potential monitoring wells for New Mexico.

Pass 1: one site per county (priority: ALL3 > ANY2 > ANY1 > fallback)
Pass 2: a second site per county using the same priority, but at least
        SECONDARY_MIN_SPACING_KM away from the county's Pass 1 site.

Then snap all sites to land with priority: state -> private -> federal -> tribal.

Outputs:
  out/potential_wells/nm_monitoring_sites_county_2pass.shp

Also saves two maps:
  out/potential_wells/nm_sites_map1_networks.png
  out/potential_wells/nm_sites_map2_population.png
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------
# CONFIG / INPUTS
# --------------------
WELL_FILE   = "waterdata/OSE_Points_of_Diversion.shp"
COUNTY_FILE = "waterdata/tl_2018_nm_county.shp"

# Population density (tract polygons you already built)
POP_GPKG  = "out/potential_wells/out/pop_density/nm_pop_density_tracts.gpkg"   # preferred
POP_SHP   = "out/potential_wells/out/pop_density/nm_pop_density_tracts.shp"    # fallback
POP_COLS  = ["DENS_SQMI", "DENS_KM2"]                      # first found is used

# Optional NM boundary (else dissolve counties)
NM_BOUND  = "nm_land_status_shp/nm_boundary.shp"

# Land ownership polygons (already created earlier)
LAND_DIR           = Path("nm_land_status_shp")
STATE_LAND_FILE    = LAND_DIR / "nm_state_trust_land.shp"
PRIVATE_LAND_FILE  = LAND_DIR / "nm_private_land.shp"
FED_LAND_FILE      = LAND_DIR / "nm_federal_land.shp"
TRIBAL_LAND_FILE   = LAND_DIR / "nm_tribal_land.shp"

# Output
OUT_DIR    = Path("out/potential_wells")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SHP    = OUT_DIR / "nm_monitoring_sites_county_2pass.shp"
OUT_MAP1   = OUT_DIR / "nm_sites_map1_networks.png"
OUT_MAP2   = OUT_DIR / "nm_sites_map2_population.png"

# CRS / KDE grid
CRS_METERS = 26913           # UTM 13N
GRID_RES   = 10_000          # grid spacing (m)
BANDWIDTH  = 15_000          # KDE bandwidth (m)
EXCLUDED_BASINS = ['HS', 'LWD', 'SP']

# "High" thresholds (percentile-normalized surfaces)
TAU_HIGH_IRR   = 0.75
TAU_HIGH_OTHER = 0.75
TAU_HIGH_POP   = 0.75

# Distance gates to avoid KDE long-tail artifacts (must be near real wells)
D_MAX_AND = 30_000  # for ALL3
D_MAX_ANY = 30_000  # for ANY2/ANY1

# Secondary spacing inside each county
SECONDARY_MIN_SPACING_KM = 10.0  # between pass-1 and pass-2 site
SECONDARY_MIN_SPACING_M  = SECONDARY_MIN_SPACING_KM * 1000.0

# Land snapping search radii (m)
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

def sample_polygon_attr_to_grid(poly_gdf, attr, xx, yy, predicate="within"):
    """
    Map polygon attribute to each grid node via spatial join.
    Returns an array shaped like xx.
    """
    if poly_gdf is None or attr is None:
        return np.zeros_like(xx, dtype=float)

    # grid nodes as points
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=poly_gdf.crs
    )

    joined = gpd.sjoin(
        pts, poly_gdf[[attr, "geometry"]],
        how="left", predicate=predicate
    )

    # Robust to duplicates: collapse multiple matches for the same left point
    idx = joined.index.to_numpy()
    vals = pd.to_numeric(joined[attr], errors="coerce").fillna(0).to_numpy()

    # If duplicates occur (rare with tracts), take the max value per grid node
    df = pd.DataFrame({"i": idx, "v": vals}).groupby("i", as_index=True)["v"].max()

    out = np.zeros(xx.size, dtype=float)
    out[df.index.to_numpy()] = df.to_numpy()
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

def snap_points_to_land(points_gdf, land_masks, xx, yy, Z_by_tier, radii=SEARCH_RADII):
    rows = []
    for _, r in points_gdf.iterrows():
        p, tier, score, rank = r.geometry, r["tier"], r["score"], r["site_rank"]
        Z = Z_by_tier[tier]
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
        rows.append({"site_rank": int(rank), "tier": tier, "score": float(score),
                     "land_class": chosen_label, "search_radius_m": 0 if chosen_r is None else int(chosen_r),
                     "geometry": chosen_p})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=points_gdf.crs)

# --------------------
# COUNTY PICKING
# --------------------
def county_mask(xx, yy, geom):
    return np.fromiter((Point(x, y).within(geom) for x, y in zip(xx.ravel(), yy.ravel())),
                       dtype=bool, count=xx.size).reshape(xx.shape)

def exclude_radius_mask(xx, yy, pts_xy, radius_m):
    """Return boolean mask: True where ALLOWED (farther than radius from any pt)."""
    if not pts_xy:
        return np.ones_like(xx, dtype=bool)
    mask = np.ones_like(xx, dtype=bool)
    for (x0, y0) in pts_xy:
        dx = xx - x0
        dy = yy - y0
        mask &= (dx*dx + dy*dy) >= (radius_m * radius_m)
    return mask

def pick_best_in_county(county_geom, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK,
                        exclude_pts=None, exclude_radius_m=0.0):
    """
    Return (x, y, score, tier_label) for best spot in county, honoring exclusion.
    Priority: ALL3 > ANY2 > ANY1 > Fallback
    """
    inside = county_mask(xx, yy, county_geom)
    allowed = inside
    if exclude_pts:
        allowed &= exclude_radius_mask(xx, yy, exclude_pts, exclude_radius_m)

    def best_from(S, M, tier_label):
        mask = allowed & M
        if not mask.any(): return None
        z = S[mask]; xs = xx[mask]; ys = yy[mask]
        i = np.argmax(z)
        return xs[i], ys[i], float(z[i]), tier_label

    # Priority order
    hit = best_from(S_AND3, M_AND3, "tier_all3")
    if hit:
        return hit

    hit = best_from(S_AND2, M_AND2, "tier_any2")
    if hit:
        return hit

    hit = best_from(S_ANY1, M_ANY1, "tier_any1")
    if hit:
        return hit

    # Fallback: still within 'allowed'
    if allowed.any():
        z = S_FALLBACK[allowed]
        xs = xx[allowed]
        ys = yy[allowed]
        i = np.argmax(z)
        return xs[i], ys[i], float(z[i]), "tier_fallback"
    return None


# --------------------
# MAPS
# --------------------
LAND_COLORS = {
    "state":   "#ff7f00",
    "private": "#33a02c",
    "federal": "#1f78b4",
    "tribal":  "#e31a1c",
    "unclassified": "#6a3d9a",
}
WELL_SYM = {
    "irr_color":  "#33a02c", "irr_alpha": 0.10, "irr_size": 3,
    "oth_color":  "#777777", "oth_alpha": 0.10, "oth_size": 3,
}
MARKER_BY_RANK = {1: "o", 2: "s"}  # primary circle, secondary square
POINT_SIZE = 80

def maybe_read_boundary(path, counties):
    g = mread(path)
    if g is not None:
        g = g.to_crs(counties.crs)
        return g
    return counties.dissolve().reset_index(drop=True).to_crs(counties.crs)

def plot_maps(out_gdf, counties, nm, wells, pop_gdf, pop_col):
    # split wells
    w = wells.copy()
    w["USE"] = w.get("use_", "").astype(str).str.upper()
    w["status"] = w.get("pod_status", "").astype(str).str.upper()
    w = w[w["status"].str.contains("ACT", na=False)]
    irr = w[w["USE"] == "IRR"]
    oth = w[w["USE"] != "IRR"]

    # -------- Map 1: networks + sites ----------
    fig, ax = plt.subplots(figsize=(11, 11))
    nm.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=1)
    counties.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=1.1)

    if not oth.empty:
        oth.plot(ax=ax, color=WELL_SYM["oth_color"], markersize=WELL_SYM["oth_size"],
                 alpha=WELL_SYM["oth_alpha"], zorder=1.2, label="Other wells")
    if not irr.empty:
        irr.plot(ax=ax, color=WELL_SYM["irr_color"], markersize=WELL_SYM["irr_size"],
                 alpha=WELL_SYM["irr_alpha"], zorder=1.2, label="Irrigation wells")

    # plot final sites: color by land_class, marker by site_rank
    for rank in (1, 2):
        part = out_gdf[out_gdf["site_rank"] == rank]
        if part.empty: continue
        for cls, color in LAND_COLORS.items():
            sub = part[part["land_class"] == cls]
            if sub.empty: continue
            sub.plot(ax=ax, color=color, marker=MARKER_BY_RANK[rank], markersize=POINT_SIZE,
                     edgecolor="white", linewidth=0.7, zorder=2.5)

    ax.set_title("Potential Monitoring Wells – Networks & Land Ownership\n(Primary = circles, Secondary = squares)", pad=12)
    ax.set_aspect("equal"); ax.set_axis_off()

    # legend
    handles = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=WELL_SYM["irr_color"],
               markeredgecolor=WELL_SYM["irr_color"], alpha=WELL_SYM["irr_alpha"], markersize=6, label="Irrigation wells"),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=WELL_SYM["oth_color"],
               markeredgecolor=WELL_SYM["oth_color"], alpha=WELL_SYM["oth_alpha"], markersize=6, label="Other wells"),
        Line2D([0],[0], marker='o', color='none', markerfacecolor="#444", markeredgecolor='white',
               markersize=10, label="Primary site"),
        Line2D([0],[0], marker='s', color='none', markerfacecolor="#444", markeredgecolor='white',
               markersize=10, label="Secondary site"),
    ]
    for cls, color in LAND_COLORS.items():
        handles.append(Line2D([0],[0], marker='o', color='none', markerfacecolor=color,
                              markeredgecolor='white', markersize=10, label=f"{cls.title()}"))
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True, framealpha=0.92, title="Legend", borderaxespad=0.0)
    ax.add_artist(leg)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(OUT_MAP1, dpi=300, bbox_inches="tight")
    print(f"Saved Map 1: {OUT_MAP1.resolve()}")

    # -------- Map 2: population + sites ----------
    fig2, ax2 = plt.subplots(figsize=(11, 11))
    vmax = float(np.nanpercentile(pop_gdf[pop_col].astype(float), 99))
    pop_gdf.plot(ax=ax2, column=pop_col, cmap="YlOrRd", legend=True, vmax=vmax,
                 linewidth=0.2, edgecolor="#999999", alpha=0.9, zorder=1)
    nm.boundary.plot(ax=ax2, color="#222222", linewidth=1.2, zorder=2)
    counties.boundary.plot(ax=ax2, color="#555555", linewidth=0.5, zorder=2.1)

    for rank in (1, 2):
        part = out_gdf[out_gdf["site_rank"] == rank]
        if part.empty: continue
        for cls, color in LAND_COLORS.items():
            sub = part[part["land_class"] == cls]
            if sub.empty: continue
            sub.plot(ax=ax2, color=color, marker=MARKER_BY_RANK[rank], markersize=POINT_SIZE,
                     edgecolor="white", linewidth=0.7, zorder=3)

    ax2.set_title("Potential Monitoring Wells – Population Density & Land Ownership\n(Primary = circles, Secondary = squares)", pad=12)
    ax2.set_aspect("equal"); ax2.set_axis_off()

    # small legend for points
    handles2 = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor="#444", markeredgecolor='white', markersize=10, label="Primary"),
        Line2D([0],[0], marker='s', color='none', markerfacecolor="#444", markeredgecolor='white', markersize=10, label="Secondary"),
    ]
    for cls, color in LAND_COLORS.items():
        handles2.append(Line2D([0],[0], marker='o', color='none', markerfacecolor=color,
                               markeredgecolor='white', markersize=10, label=f"{cls.title()}"))
    leg2 = ax2.legend(handles=handles2, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=True, framealpha=0.92, title="Sites", borderaxespad=0.0)
    ax2.add_artist(leg2)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(OUT_MAP2, dpi=300, bbox_inches="tight")
    print(f"Saved Map 2: {OUT_MAP2.resolve()}")

    plt.show()

# --------------------
# MAIN
# --------------------
def main():
    wells, counties = load_wells_and_counties()
    irr = wells[wells['USE'] == 'IRR']
    oth = wells[wells['USE'] != 'IRR']

    # Grid
    xx, yy = counties_grid(counties)

    # Densities normalized
    lam_irr   = kde_intensity(irr, xx, yy, BANDWIDTH)
    lam_oth   = kde_intensity(oth, xx, yy, BANDWIDTH)
    irr_n     = norm01(lam_irr)
    oth_n     = norm01(lam_oth)

    # Population to grid, normalized
    pop_gdf, pop_col = load_population_layer(counties.crs)
    if pop_gdf is None or pop_col is None:
        raise RuntimeError("Population layer not found or missing density column (expected one of DENS_SQMI / DENS_KM2).")
    POP      = sample_polygon_attr_to_grid(pop_gdf, pop_col, xx, yy)
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
    S_AND3 = np.minimum.reduce([irr_n, oth_n, pop_n])
    M_AND3 = irr_h & oth_h & pop_h & mask_and

    pair1  = np.minimum(irr_n, oth_n)
    pair2  = np.minimum(irr_n, pop_n)
    pair3  = np.minimum(oth_n, pop_n)
    S_AND2 = np.maximum.reduce([pair1, pair2, pair3])
    M_AND2 = ((irr_h & oth_h) | (irr_h & pop_h) | (oth_h & pop_h)) & mask_any

    S_ANY1 = np.maximum.reduce([irr_n, oth_n, pop_n])
    M_ANY1 = (irr_h | oth_h | pop_h) & mask_any

    # Fallback still prefers near wells
    S_FALLBACK = (irr_n + oth_n + 0.5*pop_n) * (mask_any.astype(float))

    # -------- Pass 1: one site per county --------
    pass1 = []
    for _, crow in counties.iterrows():
        res = pick_best_in_county(crow.geometry, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK)
        if res is None:
            continue
        x, y, s, tier = res
        pass1.append((x, y, s, tier))

    # -------- Pass 2: second site per county (spaced from pass 1) --------
    pass2 = []
    # build map county -> first point (x,y)
    # use spatial join for robustness (but counties loop works too)
    for (idx, crow), (x1, y1, s1, tier1) in zip(counties.iterrows(), pass1):
        res2 = pick_best_in_county(
            crow.geometry, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK,
            exclude_pts=[(x1, y1)], exclude_radius_m=SECONDARY_MIN_SPACING_M
        )
        if res2 is None:
            # if nothing left after exclusion, relax by ignoring exclusion (rare)
            res2 = pick_best_in_county(crow.geometry, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK)
            if res2 is None:
                continue
        x2, y2, s2, tier2 = res2
        pass2.append((x2, y2, s2, tier2))

    # Pack GDFs
    gdf1 = gpd.GeoDataFrame(
        {"site_rank": [1]*len(pass1), "tier":[t for *_, t in pass1], "score":[s for _,_,s,_ in pass1]},
        geometry=[Point(x, y) for x, y, *_ in pass1], crs=counties.crs
    )
    gdf2 = gpd.GeoDataFrame(
        {"site_rank": [2]*len(pass2), "tier":[t for *_, t in pass2], "score":[s for _,_,s,_ in pass2]},
        geometry=[Point(x, y) for x, y, *_ in pass2], crs=counties.crs
    )
    picks = pd.concat([gdf1, gdf2], ignore_index=True)
    picks = gpd.GeoDataFrame(picks, geometry="geometry", crs=counties.crs)

    # Land snapping (use the proper surface per tier)
    Z_by_tier = {"tier_all3": S_AND3, "tier_any2": S_AND2, "tier_any1": S_ANY1, "tier_fallback": S_FALLBACK}
    land_masks = load_land_masks(picks.crs)
    snapped = snap_points_to_land(picks, land_masks, xx, yy, Z_by_tier, radii=SEARCH_RADII)

    # Save shapefile
    snapped.to_file(OUT_SHP)
    print(f"Saved: {OUT_SHP.resolve()}")
    print("\nBy land_class:\n", snapped["land_class"].value_counts())
    print("\nBy tier:\n", snapped["tier"].value_counts())
    print("\nBy site_rank:\n", snapped["site_rank"].value_counts())

    # Maps
    nm = maybe_read_boundary(NM_BOUND, counties)
    plot_maps(snapped, counties, nm, wells, pop_gdf, pop_col)

if __name__ == "__main__":
    main()
