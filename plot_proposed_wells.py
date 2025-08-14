# plot_proposed_wells_with_networks.py
# Requires: geopandas, matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Inputs (edit if paths differ) ---
POINTS     = Path("out/potential_wells/proposed_monitoring_combined_prioritized.shp")
COUNTIES   = Path("waterdata/tl_2018_nm_county.shp")
WELL_FILE  = Path("waterdata/OSE_Points_of_Diversion.shp")

LAND_DIR   = Path("nm_land_status_shp")
NM_BOUND   = LAND_DIR / "nm_boundary.shp"   # optional; fallback = dissolve counties

FED_SHP    = LAND_DIR / "nm_federal_land.shp"
STATE_SHP  = LAND_DIR / "nm_state_trust_land.shp"
TRIBAL_SHP = LAND_DIR / "nm_tribal_land.shp"
PRIV_SHP   = LAND_DIR / "nm_private_land.shp"

OUT_PNG    = Path("out/potential_wells/proposed_monitoring_combined_prioritized_with_networks.png")

# --- Land polygon styles ---
# --- Land polygon styles (more transparent; private outlined in black) ---
LAND_FILL_STYLES = {
    # private = white fill, BLACK outline so everything else pops
    "private": dict(facecolor="white",   alpha=0.70, hatch=None,  edgecolor=None, lw=0.6),

    # public layers = lighter alpha than before
    "federal": dict(facecolor="#1f78b4", alpha=0.10, hatch="//",  edgecolor="#1f78b4", lw=0.3),
    "state":   dict(facecolor="#ff7f00", alpha=0.10, hatch="xx",  edgecolor="#ff7f00", lw=0.3),
    "tribal":  dict(facecolor="#e31a1c", alpha=0.10, hatch="\\\\", edgecolor="#e31a1c", lw=0.3),
}


# --- Proposed point styles by land_class ---
POINT_STYLES = {
    "state":        dict(color="#ff7f00", marker="^", size=70),
    "private":      dict(color='purple', marker="o", size=60),
    "federal":      dict(color="#1f78b4", marker="s", size=60),
    "tribal":       dict(color="#e31a1c", marker="D", size=65),
    "unclassified": dict(color="#6a3d9a", marker="X", size=70),
}

def maybe_read(path: Path):
    if path.exists():
        gdf = gpd.read_file(path)
        if not gdf.empty:
            return gdf
    print(f"[skip] {path.name} missing or empty")
    return None

def main():
    # --- Load base data
    pts = gpd.read_file(POINTS)
    counties = gpd.read_file(COUNTIES)
    nm = maybe_read(NM_BOUND)
    if nm is None:
        nm = counties.dissolve().reset_index(drop=True)

    # Land categories (optional each)
    federal = maybe_read(FED_SHP)
    state   = maybe_read(STATE_SHP)
    tribal  = maybe_read(TRIBAL_SHP)
    private = maybe_read(PRIV_SHP)

    # Wells
    wells = gpd.read_file(WELL_FILE)

    # --- Harmonize CRS (use counties as anchor)
    target = counties.crs
    def to_target(g):
        return g if g is None or g.crs == target else g.to_crs(target)

    pts, counties, nm = map(to_target, (pts, counties, nm))
    federal, state, tribal, private = map(to_target, (federal, state, tribal, private))
    wells = to_target(wells)

    # --- Prepare well subsets (match your earlier filtering logic)
    wells['USE'] = wells.get('use_', "").fillna("").astype(str).str.upper()
    wells['status'] = wells.get('pod_status', "").fillna("").astype(str).str.upper()
    wells = wells[wells['status'].str.contains("ACT", na=False)]
    irr_wells   = wells[wells['USE'] == 'IRR']
    other_wells = wells[wells['USE'] != 'IRR']

    # Ensure proposed points have land_class
    if "land_class" not in pts.columns:
        pts["land_class"] = "unclassified"

    # --- Figure
    fig, ax = plt.subplots(figsize=(11, 11))

    # 1) Land fills (private first as background white)
    for name, layer in [("private", private), ("federal", federal), ("state", state), ("tribal", tribal)]:
        if layer is None:
            continue
        st = LAND_FILL_STYLES[name]
        layer.plot(
            ax=ax,
            facecolor=st["facecolor"],
            alpha=st["alpha"],
            hatch=st["hatch"],
            edgecolor=st["edgecolor"],
            linewidth=st["lw"],
            zorder=1
        )

    # 2) Boundaries
    nm.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=3)
    counties.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=3)

    # 3) Well networks (under proposed sites)
    if not other_wells.empty:
        other_wells.plot(ax=ax, color="#777777", markersize=2, alpha=0.10, zorder=2, label="Other wells")
    if not irr_wells.empty:
        irr_wells.plot(ax=ax,   color="#33a02c", markersize=2, alpha=0.10, zorder=2, label="Irrigation wells")

    # 4) Proposed prioritized points, colored by land_class (on top)
    for cls, st in POINT_STYLES.items():
        sub = pts[pts["land_class"] == cls]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            color=st["color"],
            marker=st["marker"],
            markersize=st["size"],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=f"{cls.title()} (proposed)"
        )

    # Title / labels / legend
    ax.set_title("Proposed Monitoring Wells (Combined, Prioritized) with Land & Well Networks", pad=12)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Legend: land polygons + well dots + proposed symbols
    land_handles = []
    for k in ["state", "federal", "tribal", "private"]:
        lyr = locals().get(k)
        if lyr is None:
            continue
        st = LAND_FILL_STYLES[k]
        land_handles.append(Patch(facecolor=st["facecolor"], edgecolor=st["edgecolor"],
                                  hatch=st["hatch"], alpha=st["alpha"], linewidth=st["lw"],
                                  label=f"{k.title()} (land)"))

    wells_handles = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor="#33a02c", alpha=0.10,
               markeredgecolor="#33a02c", markersize=6, label="Irrigation wells"),
        Line2D([0],[0], marker='o', color='none', markerfacecolor="#777777", alpha=0.10,
               markeredgecolor="#777777", markersize=6, label="Other wells"),
    ]

    proposed_handles = [
        Line2D([0],[0], marker=POINT_STYLES["state"]["marker"],  color=POINT_STYLES["state"]["color"],
               markerfacecolor=POINT_STYLES["state"]["color"],  markeredgecolor="white", linewidth=0,
               markersize=8, label="Proposed (State)"),
        Line2D([0],[0], marker=POINT_STYLES["private"]["marker"],color=POINT_STYLES["private"]["color"],
               markerfacecolor=POINT_STYLES["private"]["color"],markeredgecolor="white", linewidth=0,
               markersize=8, label="Proposed (Private)"),
        Line2D([0],[0], marker=POINT_STYLES["federal"]["marker"],color=POINT_STYLES["federal"]["color"],
               markerfacecolor=POINT_STYLES["federal"]["color"],markeredgecolor="white", linewidth=0,
               markersize=8, label="Proposed (Federal)"),
        Line2D([0],[0], marker=POINT_STYLES["tribal"]["marker"], color=POINT_STYLES["tribal"]["color"],
               markerfacecolor=POINT_STYLES["tribal"]["color"], markeredgecolor="white", linewidth=0,
               markersize=8, label="Proposed (Tribal)"),
    ]

    # --- Legend outside (right side)
    handles = land_handles + wells_handles + proposed_handles
    labels = [h.get_label() for h in handles]

    leg = ax.legend(
        handles, labels,
        loc="center left",  # anchor the left edge of the legend...
        bbox_to_anchor=(1.02, 0.5),  # ...to a point just outside the axes (x>1)
        frameon=True, framealpha=0.92,
        title="Legend", borderaxespad=0.0
    )

    # Make room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # keep 18% of figure width for legend

    # Save (tight includes the off-axes legend)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
