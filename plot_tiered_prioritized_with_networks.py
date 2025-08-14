#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------
# INPUTS
# -----------------
COUNTIES   = Path("waterdata/tl_2018_nm_county.shp")
WELL_FILE  = Path("waterdata/OSE_Points_of_Diversion.shp")

# Final points produced by your site-selection script (must have 'land_class')
FINAL_POINTS = Path("out/potential_wells/nm_monitoring_sites_50_prioritized.shp")

# Optional state boundary (fallback = dissolve counties)
NM_BOUND   = Path("nm_land_status_shp/nm_boundary.shp")

# Population density tracts (your build)
POP_GPKG  = Path("out/potential_wells/out/pop_density/nm_pop_density_tracts.gpkg")   # preferred
POP_SHP   = Path("out/potential_wells/out/pop_density/nm_pop_density_tracts.shp")    # fallback
POP_COLS  = ["DENS_SQMI", "DENS_KM2"]  # first that exists is used

# -----------------
# OUTPUTS
# -----------------
OUT1 = Path("out/potential_wells/nm_monitoring_sites_map1_networks.png")
OUT2 = Path("out/potential_wells/nm_monitoring_sites_map2_population.png")
OUT1.parent.mkdir(parents=True, exist_ok=True)

# -----------------
# STYLES
# -----------------
LAND_COLORS = {  # for final points by land ownership
    "state":   "#ff7f00",
    "private": "#33a02c",
    "federal": "#1f78b4",
    "tribal":  "#e31a1c",
    "unclassified": "#6a3d9a",
}
WELL_SYM = {
    "irr_color":  "#33a02c",
    "irr_alpha":  0.10,
    "irr_size":   3,
    "oth_color":  "#777777",
    "oth_alpha":  0.10,
    "oth_size":   3,
}
POINT_SIZE = 80  # final points

# -----------------
# HELPERS
# -----------------
def maybe_read(p: Path):
    if p and p.exists():
        g = gpd.read_file(p)
        return g if not g.empty else None
    return None

def to_crs(g, crs):
    if g is None:
        return None
    if g.crs is None or g.crs != crs:
        return g.to_crs(crs)
    return g

def load_pop_layer(target_crs):
    g = None
    if POP_GPKG.exists():
        try:
            g = gpd.read_file(POP_GPKG, layer="nm_pop_density_tracts")
        except Exception:
            g = None
    if g is None and POP_SHP.exists():
        g = gpd.read_file(POP_SHP)
    if g is None or g.empty:
        return None, None
    if g.crs != target_crs:
        g = g.to_crs(target_crs)
    col = next((c for c in POP_COLS if c in g.columns), None)
    return g, col

# -----------------
# MAIN
# -----------------
def main():
    # Base layers
    counties = gpd.read_file(COUNTIES)
    # Before (bad for GeoDataFrame):
    # nm = maybe_read(NM_BOUND) or counties.dissolve().reset_index(drop=True)

    # After:
    nm_gdf = maybe_read(NM_BOUND)
    nm = nm_gdf if nm_gdf is not None else counties.dissolve().reset_index(drop=True)
    nm = nm.to_crs(counties.crs)

    # Final points
    pts = gpd.read_file(FINAL_POINTS)

    # Well networks (filter to active; split irrigation vs other)
    wells = gpd.read_file(WELL_FILE)
    wells["USE"]    = wells.get("use_", "").astype(str).str.upper()
    wells["status"] = wells.get("pod_status", "").astype(str).str.upper()
    wells = wells[wells["status"].str.contains("ACT", na=False)]
    irr_wells   = wells[wells["USE"] == "IRR"]
    other_wells = wells[wells["USE"] != "IRR"]

    # CRS alignment (use counties crs everywhere)
    target = counties.crs
    nm, pts, wells, irr_wells, other_wells = [to_crs(g, target) for g in (nm, pts, wells, irr_wells, other_wells)]

    # Ensure land_class exists
    if "land_class" not in pts.columns:
        pts["land_class"] = "unclassified"

    # ----------------- MAP 1: networks + final points -----------------
    fig, ax = plt.subplots(figsize=(11, 11))

    # Boundaries
    nm.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=1)
    counties.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=1.1)

    # Well networks
    if not other_wells.empty:
        other_wells.plot(ax=ax, color=WELL_SYM["oth_color"], markersize=WELL_SYM["oth_size"],
                         alpha=WELL_SYM["oth_alpha"], zorder=1.2, label="Other wells")
    if not irr_wells.empty:
        irr_wells.plot(ax=ax, color=WELL_SYM["irr_color"], markersize=WELL_SYM["irr_size"],
                       alpha=WELL_SYM["irr_alpha"], zorder=1.2, label="Irrigation wells")

    # Final points by land_class (color)
    for cls, color in LAND_COLORS.items():
        sub = pts[pts["land_class"] == cls]
        if sub.empty:
            continue
        sub.plot(ax=ax, color=color, marker="o", markersize=POINT_SIZE,
                 edgecolor="white", linewidth=0.7, zorder=2.5, label=cls.title())

    ax.set_title("Potential Monitoring Wells – Networks & Land Ownership", pad=12)
    ax.set_aspect("equal"); ax.set_axis_off()

    # Legend off-map on right
    handles = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=WELL_SYM["irr_color"],
               markeredgecolor=WELL_SYM["irr_color"], alpha=WELL_SYM["irr_alpha"], markersize=6, label="Irrigation wells"),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=WELL_SYM["oth_color"],
               markeredgecolor=WELL_SYM["oth_color"], alpha=WELL_SYM["oth_alpha"], markersize=6, label="Other wells"),
    ]
    for cls, color in LAND_COLORS.items():
        handles.append(
            Line2D([0],[0], marker='o', color='none', markerfacecolor=color, markeredgecolor='white',
                   markersize=10, label=f"{cls.title()} site")
        )
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True, framealpha=0.92, title="Legend", borderaxespad=0.0)
    ax.add_artist(leg)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(OUT1, dpi=300, bbox_inches="tight")
    print(f"Saved Map 1: {OUT1.resolve()}")

    # ----------------- MAP 2: population choropleth + final points -----------------
    fig2, ax2 = plt.subplots(figsize=(11, 11))

    # Load pop tracts
    pop_gdf, pop_col = load_pop_layer(target)
    if pop_gdf is None or pop_col is None:
        raise RuntimeError("Population layer not found or missing density column (expected one of DENS_SQMI / DENS_KM2).")

    # Robust color range (clip to 99th percentile to avoid outliers)
    vmax = float(np.nanpercentile(pop_gdf[pop_col].astype(float), 99))
    pop_gdf.plot(ax=ax2, column=pop_col, cmap="YlOrRd", legend=True, vmax=vmax,
                 linewidth=0.2, edgecolor="#999999", alpha=0.9, zorder=1)

    # Boundaries on top
    nm.boundary.plot(ax=ax2, color="#222222", linewidth=1.2, zorder=2)
    counties.boundary.plot(ax=ax2, color="#555555", linewidth=0.5, zorder=2.1)

    # Final points (same symbology)
    for cls, color in LAND_COLORS.items():
        sub = pts[pts["land_class"] == cls]
        if sub.empty:
            continue
        sub.plot(ax=ax2, color=color, marker="o", markersize=POINT_SIZE,
                 edgecolor="white", linewidth=0.7, zorder=3, label=cls.title())

    ax2.set_title("Potential Monitoring Wells – Population Density & Land Ownership", pad=12)
    ax2.set_aspect("equal"); ax2.set_axis_off()

    # Put a small legend for points only (choropleth already has a legend)
    handles2 = [Line2D([0],[0], marker='o', color='none', markerfacecolor=color, markeredgecolor='white',
                       markersize=10, label=f"{cls.title()} site") for cls, color in LAND_COLORS.items()]
    leg2 = ax2.legend(handles=handles2, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=True, framealpha=0.92, title="Sites", borderaxespad=0.0)
    ax2.add_artist(leg2)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(OUT2, dpi=300, bbox_inches="tight")
    print(f"Saved Map 2: {OUT2.resolve()}")

    plt.show()

if __name__ == "__main__":
    main()
