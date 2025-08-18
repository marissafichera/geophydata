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
WELL_FILE = "waterdata/OSE_Points_of_Diversion.shp"
COUNTY_FILE = "waterdata/tl_2018_nm_county.shp"

# Population density (tract polygons you already built)
POP_GPKG = "out/potential_wells/out/pop_density/nm_pop_density_tracts.gpkg"  # preferred
POP_SHP = "out/potential_wells/out/pop_density/nm_pop_density_tracts.shp"  # fallback
POP_COLS = ["DENS_SQMI", "DENS_KM2"]  # first found is used

# Optional NM boundary (else dissolve counties)
NM_BOUND = "nm_land_status_shp/nm_boundary.shp"

# Land ownership polygons (already created earlier)
LAND_DIR = Path("nm_land_status_shp")
STATE_LAND_FILE = LAND_DIR / "nm_state_trust_land.shp"
PRIVATE_LAND_FILE = LAND_DIR / "nm_private_land.shp"
FED_LAND_FILE = LAND_DIR / "nm_federal_land.shp"
TRIBAL_LAND_FILE = LAND_DIR / "nm_tribal_land.shp"

# Output
OUT_DIR = Path("out/potential_wells")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SHP = OUT_DIR / "nm_monitoring_sites_county_2pass.shp"
OUT_MAP1 = OUT_DIR / "nm_sites_map1_networks.png"
OUT_MAP2 = OUT_DIR / "nm_sites_map2_population.png"

# CRS / KDE grid
CRS_METERS = 26913  # UTM 13N
GRID_RES = 10_000  # grid spacing (m)
BANDWIDTH = 15_000  # KDE bandwidth (m)
EXCLUDED_BASINS = ['HS', 'LWD', 'SP']

# "High" thresholds (percentile-normalized surfaces)
TAU_HIGH_IRR = 0.75
TAU_HIGH_OTHER = 0.75
TAU_HIGH_POP = 0.75

# Distance gates to avoid KDE long-tail artifacts (must be near real wells)
D_MAX_AND = 30_000  # for ALL3
D_MAX_ANY = 30_000  # for ANY2/ANY1

# Secondary spacing inside each county
SECONDARY_MIN_SPACING_KM = 10.0  # between pass-1 and pass-2 site
SECONDARY_MIN_SPACING_M = SECONDARY_MIN_SPACING_KM * 1000.0

# Land snapping search radii (m)
SEARCH_RADII = [0, 50_000]

# tiny tolerance in meters to absorb hairline gaps
ONEDGE_TOL = 25.0

# Toggle land snapping on/off
USE_LAND_SNAPPING = False

# Point styling (ignore land ownership)
POINT_COLOR = "#e31a1c"  # red-ish
POINT_EDGE = "white"
POINT_SIZE_PRIMARY = 90
POINT_SIZE_SECOND = 55


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
            parts.append(unary_union(geoms[i:i + chunk]))
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
    wells['USE'] = wells['use_'].fillna("").str.upper()
    wells['status'] = wells['pod_status'].fillna("").str.upper()
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
    x = np.clip(x, 0, 1);
    x[~np.isfinite(x)] = 0
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
    pts = np.vstack([points_gdf.geometry.x, points_gdf.geometry.y]).T
    tree = cKDTree(pts)
    d, _ = tree.query(grid, k=1)
    return d.reshape(xx.shape)


# ----- land snapping -----
def best_grid_point_in(inter_geom, xx, yy, Z, p0):
    if inter_geom.is_empty: return None
    ip = prep(inter_geom)
    minx, miny, maxx, maxy = inter_geom.bounds
    xs, ys, zs = xx.ravel(), yy.ravel(), Z.ravel()
    box = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
    idx = np.where(box)[0]
    if idx.size == 0: return inter_geom.representative_point()
    inside = [i for i in idx if ip.contains(Point(xs[i], ys[i])) or inter_geom.touches(Point(xs[i], ys[i]))]
    if not inside: return inter_geom.representative_point()
    best = max(inside, key=lambda i: (zs[i], -Point(xs[i], ys[i]).distance(p0)))
    return Point(xs[best], ys[best])


def load_land_masks(target_crs):
    classes = [
        ("state", mread(STATE_LAND_FILE)),
        ("private", mread(PRIVATE_LAND_FILE)),
        ("federal", mread(FED_LAND_FILE)),
        ("tribal", mread(TRIBAL_LAND_FILE)),
    ]
    out = []
    for label, g in classes:
        if g is None:
            out.append((label, None, None, None))  # label, base, buf, prep_buf
            continue
        g = g.to_crs(target_crs)
        g = clean_for_union(g, grid=0.1, min_area=1.0)
        dissolved = robust_union(list(g.geometry))
        # buffer once; reuse
        dissolved_buf = dissolved.buffer(ONEDGE_TOL)
        out.append((label, dissolved, dissolved_buf, prep(dissolved_buf)))
    return out  # tuples: (label, base_geom, buffered_geom, prepared_buffered)


def best_grid_point_in_prepared(prep_poly, xx, yy, Z, p0, radius):
    """Pick highest-Z grid node within circle(p0, radius) that is inside prep_poly."""
    if prep_poly is None:
        return None
    minx, miny, maxx, maxy = p0.x - radius, p0.y - radius, p0.x + radius, p0.y + radius
    xs, ys, zs = xx.ravel(), yy.ravel(), Z.ravel()
    # bbox prefilter
    box = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
    idx = np.where(box)[0]
    if idx.size == 0:
        return None
    # circle filter (avoid sqrt)
    r2 = radius * radius
    dx, dy = xs[idx] - p0.x, ys[idx] - p0.y
    in_circle = (dx * dx + dy * dy) <= r2
    idx = idx[in_circle]
    if idx.size == 0:
        return None
    # land containment (prepared geometry)
    # Note: point-by-point loop is needed; still much cheaper than building intersections
    best_i, best_val, best_dist = None, -1, 1e30
    for i in idx:
        pt = Point(xs[i], ys[i])
        if prep_poly.contains(pt) or prep_poly.covers(pt):
            val = zs[i]
            dist = pt.distance(p0)
            if (val > best_val) or (val == best_val and dist < best_dist):
                best_i, best_val, best_dist = i, val, dist
    if best_i is None:
        return None
    return Point(xs[best_i], ys[best_i])


def snap_points_to_land(points_gdf, land_masks, xx, yy, Z_by_tier, radii=SEARCH_RADII):
    rows = []
    for _, r in points_gdf.iterrows():
        p, tier, score, rank = r.geometry, r["tier"], r["score"], r["site_rank"]
        Z = Z_by_tier[tier]
        chosen_p, chosen_label, chosen_r = p, "unclassified", None

        for rad in radii:
            hit = False
            for label, base_geom, buf_geom, prep_buf in land_masks:
                if prep_buf is None:
                    continue
                if rad == 0:
                    # treat on-boundary as inside using prepared buffered geometry
                    if prep_buf.covers(p):
                        chosen_p, chosen_label, chosen_r = p, label, 0
                        hit = True
                        break
                else:
                    npnt = best_grid_point_in_prepared(prep_buf, xx, yy, Z, p, rad)
                    if npnt is not None and npnt.is_valid:
                        chosen_p, chosen_label, chosen_r = npnt, label, rad
                        hit = True
                        break
            if hit:
                break

        rows.append({
            "site_rank": int(rank),
            "tier": tier,
            "score": float(score),
            "land_class": chosen_label,
            "search_radius_m": 0 if chosen_r is None else int(chosen_r),
            "geometry": chosen_p
        })
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
        mask &= (dx * dx + dy * dy) >= (radius_m * radius_m)
    return mask


# adjust pick_best_in_county to accept a precomputed mask:
def pick_best_in_county_masked(mask_inside, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK,
                               exclude_pts=None, exclude_radius_m=0.0):
    allowed = mask_inside.copy()
    if exclude_pts:
        allowed &= exclude_radius_mask(xx, yy, exclude_pts, exclude_radius_m)

    def best_from(S, M, tier_label):
        mask = allowed & M
        if not mask.any(): return None
        z = S[mask];
        xs = xx[mask];
        ys = yy[mask];
        i = np.argmax(z)
        return xs[i], ys[i], float(z[i]), tier_label

    for S, M, name in [(S_AND3, M_AND3, "tier_all3"), (S_AND2, M_AND2, "tier_any2"), (S_ANY1, M_ANY1, "tier_any1")]:
        hit = best_from(S, M, name)
        if hit: return hit

    if allowed.any():
        z = S_FALLBACK[allowed];
        xs = xx[allowed];
        ys = yy[allowed];
        i = np.argmax(z)
        return xs[i], ys[i], float(z[i]), "tier_fallback"
    return None


# def pick_best_in_county(county_geom, xx, yy, S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK,
#                         exclude_pts=None, exclude_radius_m=0.0):
#     """
#     Return (x, y, score, tier_label) for best spot in county, honoring exclusion.
#     Priority: ALL3 > ANY2 > ANY1 > Fallback
#     """
#     inside = county_mask(xx, yy, county_geom)
#     allowed = inside
#     if exclude_pts:
#         allowed &= exclude_radius_mask(xx, yy, exclude_pts, exclude_radius_m)
#
#     def best_from(S, M, tier_label):
#         mask = allowed & M
#         if not mask.any(): return None
#         z = S[mask]; xs = xx[mask]; ys = yy[mask]
#         i = np.argmax(z)
#         return xs[i], ys[i], float(z[i]), tier_label
#
#     # Priority order
#     hit = best_from(S_AND3, M_AND3, "tier_all3")
#     if hit:
#         return hit
#
#     hit = best_from(S_AND2, M_AND2, "tier_any2")
#     if hit:
#         return hit
#
#     hit = best_from(S_ANY1, M_ANY1, "tier_any1")
#     if hit:
#         return hit
#
#     # Fallback: still within 'allowed'
#     if allowed.any():
#         z = S_FALLBACK[allowed]
#         xs = xx[allowed]
#         ys = yy[allowed]
#         i = np.argmax(z)
#         return xs[i], ys[i], float(z[i]), "tier_fallback"
#     return None


# --------------------
# MAPS
# --------------------
LAND_COLORS = {
    "state": "#ff7f00",
    "private": "purple",
    "federal": "#1f78b4",
    "tribal": "#e31a1c",
    "unclassified": "#6a3d9a",
}
WELL_SYM = {
    "irr_color": "#33a02c", "irr_alpha": 0.10, "irr_size": 3,
    "oth_color": "#777777", "oth_alpha": 0.10, "oth_size": 3,
}
MARKER_BY_RANK = {1: "o", 2: "s"}  # primary circle, secondary square
POINT_SIZE = 80

# --- New/updated plot constants ---
OUT_MAP3 = OUT_DIR / "nm_sites_map3_land_ownership.png"

# both primary & secondary are circles; size distinguishes them
MARKER_CIRCLE = "o"
SIZE_BY_RANK = {1: 100, 2: 65}  # tweak to taste

# land fill styles for Map 3
LAND_FILL = {
    "private": dict(facecolor="white", alpha=0.70, hatch=None, edgecolor="#000000", lw=0.6),
    "federal": dict(facecolor="#1f78b4", alpha=0.12, hatch="//", edgecolor="#1f78b4", lw=0.3),
    "state": dict(facecolor="#ff7f00", alpha=0.12, hatch="xx", edgecolor="#ff7f00", lw=0.3),
    "tribal": dict(facecolor="#e31a1c", alpha=0.12, hatch="\\\\", edgecolor="#e31a1c", lw=0.3),
}


def maybe_read_boundary(path, counties):
    g = mread(path)
    if g is not None:
        g = g.to_crs(counties.crs)
        return g
    return counties.dissolve().reset_index(drop=True).to_crs(counties.crs)


def plot_maps(out_gdf, counties, nm, wells, pop_gdf, pop_col):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # split wells
    w = wells.copy()
    w["USE"] = w.get("use_", "").astype(str).str.upper()
    w["status"] = w.get("pod_status", "").astype(str).str.upper()
    w = w[w["status"].str.contains("ACT", na=False)]
    irr = w[w["USE"] == "IRR"]
    oth = w[w["USE"] != "IRR"]

    # ---------------- Map 1: networks + sites ----------------
    fig, ax = plt.subplots(figsize=(11, 11))
    nm.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=1)
    counties.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=1.1)

    if not oth.empty:
        oth.plot(ax=ax, color="#777777", markersize=3, alpha=0.10, zorder=1.2, label="Other wells")
    if not irr.empty:
        irr.plot(ax=ax, color="#33a02c", markersize=3, alpha=0.10, zorder=1.2, label="Irrigation wells")

    # final sites: same color; size by rank (primary vs secondary)
    for rank in (1, 2):
        part = out_gdf[out_gdf["site_rank"] == rank]
        if part.empty: continue
        size = POINT_SIZE_PRIMARY if rank == 1 else POINT_SIZE_SECOND
        part.plot(ax=ax, color=POINT_COLOR, marker="o", markersize=size,
                  edgecolor=POINT_EDGE, linewidth=0.7, zorder=2.5)

    ax.set_title("Potential Monitoring Wells – Networks\n(Primary = large circles, Secondary = small circles)", pad=12)
    ax.set_aspect("equal");
    ax.set_axis_off()

    # legend WITHOUT land categories
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=WELL_SYM["irr_color"],
               markeredgecolor=WELL_SYM["irr_color"], alpha=WELL_SYM["irr_alpha"],
               markersize=6, label="Irrigation wells"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=WELL_SYM["oth_color"],
               markeredgecolor=WELL_SYM["oth_color"], alpha=WELL_SYM["oth_alpha"],
               markersize=6, label="Other wells"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=10, label="Primary site"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=8, label="Secondary site"),
    ]
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True, framealpha=0.92, title="Legend", borderaxespad=0.0)
    ax.add_artist(leg)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(OUT_MAP1, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
    print(f"Saved Map 1: {OUT_MAP1.resolve()}")

    # ---------------- Map 2: population choropleth + sites (discrete colors) ----------------
    fig2, ax2 = plt.subplots(figsize=(11, 11))

    # 1) Build discrete classes from the population values
    vals = pd.to_numeric(pop_gdf[pop_col], errors="coerce").to_numpy()
    NBINS = 9  # number of color classes (tweak as you like)
    CMAP_NAME = "viridis"  # any matplotlib colormap name

    # Use quantiles for evenly-populated classes (robust for skewed data)
    import numpy as np
    finite = np.isfinite(vals)
    edges = np.quantile(vals[finite], np.linspace(0, 1, NBINS + 1))
    edges[0] = -np.inf;
    edges[-1] = np.inf

    # Class indices: 0..NBINS-1, and -1 for NaN/inf
    class_idx = np.full(vals.shape, -1, dtype=int)
    class_idx[finite] = np.digitize(vals[finite], edges[1:-1], right=True)

    # 2) Turn classes into colors (your pattern)
    import matplotlib.colors as mcolors

    NBINS = 9
    CMAP_NAME = "viridis"

    vals = pd.to_numeric(pop_gdf[pop_col], errors="coerce").to_numpy()
    finite = np.isfinite(vals)

    # quantile edges (robust); switch to linspace for equal-width bins if you prefer
    edges = np.quantile(vals[finite], np.linspace(0, 1, NBINS + 1))
    edges[0] = -np.inf;
    edges[-1] = np.inf

    # class indices: 0..NBINS-1, -1 for NaN/inf
    class_idx = np.full(vals.shape, -1, dtype=int)
    class_idx[finite] = np.digitize(vals[finite], edges[1:-1], right=True)

    # colormap → HEX so each assignment is a single string (no 4-element RGBA broadcast)
    cmap = plt.get_cmap(CMAP_NAME, NBINS)
    colors_hex = [mcolors.to_hex(cmap(i)) for i in range(NBINS)]

    # per-row colors (default light gray for NaN)
    facecolors = np.full(len(pop_gdf), "#dddddd", dtype=object)
    for i in range(NBINS):
        facecolors[class_idx == i] = colors_hex[i]

    # draw
    pop_gdf.plot(
        ax=ax2,
        color=facecolors,
        linewidth=0.2,
        edgecolor="#999999",
        alpha=0.65,
        zorder=1
    )

    # ---------- legend (right) for sites ----------
    handles_sites = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=10, label="Primary site"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=8, label="Secondary site"),
    ]
    leg2 = ax2.legend(
        handles=handles_sites,  # <-- keyword!
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, framealpha=0.92, title="Sites", borderaxespad=0.0
    )
    ax2.add_artist(leg2)

    # ---------- binned legend (bottom) for population ----------
    from matplotlib.patches import Patch
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    NBINS = 8  # choose your bin count
    vals = pop_gdf[pop_col].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    pmin, pmax = np.percentile(vals, [1, 99])

    edges = np.linspace(pmin, pmax, NBINS + 1)
    cmap = cm.get_cmap("viridis", NBINS)
    colors_hex = [mcolors.to_hex(cmap(i)) for i in range(NBINS)]

    # build patches (handles) only — no Artists in here!
    bin_labels = [f"{edges[i]:,.0f}–{edges[i + 1]:,.0f}" for i in range(NBINS)]
    patches = [Patch(facecolor=colors_hex[i], edgecolor="none", label=bin_labels[i])
               for i in range(NBINS)]

    leg_bins = ax2.legend(
        handles=patches,  # <-- keyword!
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=min(NBINS, 5),
        frameon=True, framealpha=0.92,
        title=f"Population density ({'people / sq mi' if pop_col.upper().endswith('SQMI') else 'people / km²'})"
    )
    ax2.add_artist(leg_bins)

    plt.tight_layout(rect=[0, 0.08, 0.82, 1])  # room for bottom legend + right legend
    fig2.savefig(OUT_MAP2, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg2, leg_bins])


    # 3) Draw the polygons using the precomputed colors (with transparency)
    pop_gdf.plot(
        ax=ax2,
        color=facecolors,  # <- supply per-row colors
        linewidth=0.2,
        edgecolor="#999999",
        alpha=0.65,  # translucency so sites pop through
        zorder=1
    )

    # NM & county boundaries
    nm.boundary.plot(ax=ax2, color="#222222", linewidth=1.2, zorder=2)
    counties.boundary.plot(ax=ax2, color="#555555", linewidth=0.5, zorder=2.1)

    # Plot final sites: same color; size by rank (primary vs secondary)
    for rank in (1, 2):
        part = out_gdf[out_gdf["site_rank"] == rank]
        if part.empty:
            continue
        size = POINT_SIZE_PRIMARY if rank == 1 else POINT_SIZE_SECOND
        part.plot(ax=ax2, color=POINT_COLOR, marker="o", markersize=size,
                  edgecolor=POINT_EDGE, linewidth=0.7, zorder=3)

    ax2.set_title(
        "Potential Monitoring Wells – Population Density\n(Primary = large circles, Secondary = small circles)", pad=12)
    ax2.set_aspect("equal");
    ax2.set_axis_off()

    # 4) Legends: point legend on the right, discrete choropleth legend at the bottom
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Sites legend (right)
    site_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=10, label="Primary site"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_COLOR,
               markeredgecolor=POINT_EDGE, markersize=8, label="Secondary site"),
    ]
    leg_sites = ax2.legend(handles=site_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                           frameon=True, framealpha=0.92, title="Sites", borderaxespad=0.0)
    ax2.add_artist(leg_sites)

    # Discrete choropleth legend (bottom, multi-column)
    units = "people / sq mi" if pop_col.upper().endswith("SQMI") else "people / km²"
    bin_labels = [f"{edges[i]:,.0f}–{edges[i + 1]:,.0f}" for i in range(NBINS)]
    patches = [Patch(facecolor=colors_hex[i], edgecolor="none", label=bin_labels[i]) for i in range(NBINS)]
    if np.any(class_idx == -1):
        patches.append(Patch(facecolor="#dddddd", edgecolor="none", label="No data"))

    leg_bins = ax2.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                          ncol=min(NBINS, 5), frameon=True, framealpha=0.92,
                          title=f"Population density ({units})")
    ax2.add_artist(leg_bins)

    plt.tight_layout(rect=[0, 0.07, 0.82, 1])  # leave room for bottom legend and right legend
    fig2.savefig(OUT_MAP2, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg_sites, leg_bins])
    print(f"Saved Map 2: {OUT_MAP2.resolve()}")

    # ---------------- Map 3: land ownership + sites ----------------
    # (keep land legend here)
    def _mr(p):
        g = mread(p);
        return g.to_crs(counties.crs) if g is not None and g.crs != counties.crs else g

    state = _mr(STATE_LAND_FILE)
    federal = _mr(FED_LAND_FILE)
    tribal = _mr(TRIBAL_LAND_FILE)
    private = _mr(PRIVATE_LAND_FILE)

    fig3, ax3 = plt.subplots(figsize=(11, 11))

    for name, layer in [("private", private), ("federal", federal), ("state", state), ("tribal", tribal)]:
        if layer is None or layer.empty: continue
        st = LAND_FILL[name]
        layer.plot(ax=ax3, facecolor=st["facecolor"], alpha=st["alpha"], hatch=st["hatch"],
                   edgecolor=st["edgecolor"], linewidth=st["lw"], zorder=0.8)

    nm.boundary.plot(ax=ax3, color="#222222", linewidth=1.2, zorder=1.2)
    counties.boundary.plot(ax=ax3, color="#555555", linewidth=0.5, zorder=1.3)

    for rank in (1, 2):
        part = out_gdf[out_gdf["site_rank"] == rank]
        if part.empty: continue
        size = POINT_SIZE_PRIMARY if rank == 1 else POINT_SIZE_SECOND
        part.plot(ax=ax3, color=POINT_COLOR, marker="o", markersize=size,
                  edgecolor=POINT_EDGE, linewidth=0.7, zorder=4)

    ax3.set_title("Potential Monitoring Wells – Land Ownership", pad=12)
    ax3.set_aspect("equal");
    ax3.set_axis_off()

    # land legend stays on Map 3
    land_handles = []
    for k in ["state", "federal", "tribal", "private"]:
        lyr = locals().get(k)
        if lyr is None or lyr is False: continue
        st = LAND_FILL[k]
        land_handles.append(Patch(facecolor=st["facecolor"], edgecolor=st["edgecolor"],
                                  hatch=st["hatch"], alpha=st["alpha"], linewidth=st["lw"],
                                  label=f"{k.title()} Land"))
    site_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor="#444",
               markeredgecolor="white", markersize=11, label="Primary site"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor="#444",
               markeredgecolor="white", markersize=8, label="Secondary site"),
    ]
    leg3 = ax3.legend(handles=land_handles + site_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=True, framealpha=0.92, title="Legend", borderaxespad=0.0)
    ax3.add_artist(leg3)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig3.savefig(OUT_MAP3, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg3])
    print(f"Saved Map 3: {OUT_MAP3.resolve()}")

    plt.show()


# --------------------
# MAIN
# --------------------
def main():
    wells, counties = load_wells_and_counties()
    irr = wells[wells['USE'] == 'IRR']
    oth = wells[wells['USE'] != 'IRR']

    # --- Grid for scoring ---
    xx, yy = counties_grid(counties)

    # ✅ Build once; each mask aligns with counties rows
    county_masks = [county_mask(xx, yy, geom) for geom in counties.geometry]

    # Densities normalized
    lam_irr = kde_intensity(irr, xx, yy, BANDWIDTH)
    lam_oth = kde_intensity(oth, xx, yy, BANDWIDTH)
    irr_n = norm01(lam_irr)
    oth_n = norm01(lam_oth)

    # Population to grid, normalized
    pop_gdf, pop_col = load_population_layer(counties.crs)
    if pop_gdf is None or pop_col is None:
        raise RuntimeError(
            "Population layer not found or missing density column (expected one of DENS_SQMI / DENS_KM2).")
    POP = sample_polygon_attr_to_grid(pop_gdf, pop_col, xx, yy)
    pop_n = norm01(POP)

    # "high" flags per layer
    irr_h = irr_n >= TAU_HIGH_IRR
    oth_h = oth_n >= TAU_HIGH_OTHER
    pop_h = pop_n >= TAU_HIGH_POP

    # Distance gates (avoid KDE long tails)
    d_irr = nearest_distance_raster(irr, xx, yy)
    d_oth = nearest_distance_raster(oth, xx, yy)
    mask_and = (d_irr <= D_MAX_AND) & (d_oth <= D_MAX_AND)
    mask_any = (d_irr <= D_MAX_ANY) | (d_oth <= D_MAX_ANY)

    # Priority surfaces
    S_AND3 = np.minimum.reduce([irr_n, oth_n, pop_n])
    M_AND3 = irr_h & oth_h & pop_h & mask_and

    pair1 = np.minimum(irr_n, oth_n)
    pair2 = np.minimum(irr_n, pop_n)
    pair3 = np.minimum(oth_n, pop_n)
    S_AND2 = np.maximum.reduce([pair1, pair2, pair3])
    M_AND2 = ((irr_h & oth_h) | (irr_h & pop_h) | (oth_h & pop_h)) & mask_any

    S_ANY1 = np.maximum.reduce([irr_n, oth_n, pop_n])
    M_ANY1 = (irr_h | oth_h | pop_h) & mask_any

    # Fallback still prefers near wells
    S_FALLBACK = (irr_n + oth_n + 0.5 * pop_n) * (mask_any.astype(float))

    # -------- Pass 1 --------
    pass1 = []
    pass1_by_idx = {}
    for i, mask_inside in enumerate(county_masks):
        res = pick_best_in_county_masked(
            mask_inside, xx, yy,
            S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK
        )
        if res:
            pass1.append(res)
            pass1_by_idx[i] = res

    # -------- Pass 2 --------
    pass2 = []
    for i, mask_inside in enumerate(county_masks):
        prev = pass1_by_idx.get(i)
        if prev:
            x1, y1, *_ = prev
            res2 = pick_best_in_county_masked(
                mask_inside, xx, yy,
                S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK,
                exclude_pts=[(x1, y1)], exclude_radius_m=SECONDARY_MIN_SPACING_M
            )
            if not res2:
                # fallback if exclusion wiped everything out
                res2 = pick_best_in_county_masked(
                    mask_inside, xx, yy,
                    S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK
                )
        else:
            # no pass-1 pick for this county — still try to place one
            res2 = pick_best_in_county_masked(
                mask_inside, xx, yy,
                S_AND3, M_AND3, S_AND2, M_AND2, S_ANY1, M_ANY1, S_FALLBACK
            )

        if res2:
            pass2.append(res2)

    # Pack GDFs
    gdf1 = gpd.GeoDataFrame(
        {"site_rank": [1] * len(pass1), "tier": [t for *_, t in pass1], "score": [s for _, _, s, _ in pass1]},
        geometry=[Point(x, y) for x, y, *_ in pass1], crs=counties.crs
    )
    gdf2 = gpd.GeoDataFrame(
        {"site_rank": [2] * len(pass2), "tier": [t for *_, t in pass2], "score": [s for _, _, s, _ in pass2]},
        geometry=[Point(x, y) for x, y, *_ in pass2], crs=counties.crs
    )
    picks = pd.concat([gdf1, gdf2], ignore_index=True)
    picks = gpd.GeoDataFrame(picks, geometry="geometry", crs=counties.crs)

    # Land snapping (use the proper surface per tier)
    Z_by_tier = {"tier_all3": S_AND3, "tier_any2": S_AND2, "tier_any1": S_ANY1, "tier_fallback": S_FALLBACK}

    if USE_LAND_SNAPPING:
        land_masks = load_land_masks(picks.crs)
        snapped = snap_points_to_land(picks, land_masks, xx, yy, Z_by_tier, radii=SEARCH_RADII)
    else:
        # Skip snapping; keep the selected grid point as-is
        snapped = picks.copy()
        if "land_class" not in snapped.columns:
            snapped["land_class"] = "unclassified"
        if "search_radius_m" not in snapped.columns:
            snapped["search_radius_m"] = 0

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
