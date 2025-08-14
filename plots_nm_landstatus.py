# plots_nm_land_status_categories_hatch.py
# Requires: geopandas, matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Folder with shapefiles from your build script
SHAPE_DIR = Path("nm_land_status_shp")
OUT_PNG = Path("nm_land_status_categories_hatch.png")

FILES = {
    "nm_boundary":        "nm_boundary.shp",
    "nm_federal_land":    "nm_federal_land.shp",
    "nm_state_trust":     "nm_state_trust_land.shp",
    "nm_tribal_land":     "nm_tribal_land.shp",
    "nm_private_land":    "nm_private_land.shp",
}

# Styles: fill colors, hatch patterns, edge colors
STYLES = {
    "nm_federal_land": dict(facecolor="#1f78b4", alpha=0.3, hatch="//",  edgecolor="#1f78b4", linewidth=0.8),
    "nm_state_trust":  dict(facecolor="#ff7f00", alpha=0.3, hatch="xx",  edgecolor="#ff7f00", linewidth=0.8),
    "nm_tribal_land":  dict(facecolor="#e31a1c", alpha=0.3, hatch="\\\\", edgecolor="#e31a1c", linewidth=0.8),
    "nm_private_land": dict(facecolor="#33a02c", alpha=0.15, hatch=None, edgecolor="#33a02c", linewidth=0.8),
    "nm_boundary":     dict(facecolor="none", alpha=1.0, hatch=None, edgecolor="#222222", linewidth=1.2),
}

def maybe_read(name):
    path = SHAPE_DIR / FILES[name]
    if not path.exists():
        print(f"[skip] {name}: {path} not found")
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        print(f"[skip] {name}: empty layer")
        return None
    return gdf

def main():
    nm        = maybe_read("nm_boundary")
    federal   = maybe_read("nm_federal_land")
    state_tr  = maybe_read("nm_state_trust")
    tribal    = maybe_read("nm_tribal_land")
    private   = maybe_read("nm_private_land")

    fig, ax = plt.subplots(figsize=(9.5, 9.5))

    # Plot order: private → federal → state → tribal → boundary
    for name, gdf in [
        ("nm_private_land", private),
        ("nm_federal_land", federal),
        ("nm_state_trust",  state_tr),
        ("nm_tribal_land",  tribal),
        ("nm_boundary",     nm),
    ]:
        if gdf is None:
            continue
        style = STYLES[name]
        gdf.plot(ax=ax,
                 facecolor=style.get("facecolor", "none"),
                 alpha=style.get("alpha", 1.0),
                 edgecolor=style.get("edgecolor", "black"),
                 linewidth=style.get("linewidth", 0.8),
                 hatch=style.get("hatch", None))

    ax.set_title("New Mexico Land Categories (Federal / State Trust / Tribal / Private)", pad=12)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Legend using Patch handles
    legend_patches = []
    legend_labels = []
    def add_leg(label, style):
        legend_patches.append(Patch(
            facecolor=style.get("facecolor", "none"),
            edgecolor=style.get("edgecolor", "black"),
            hatch=style.get("hatch", None),
            linewidth=style.get("linewidth", 0.8),
            alpha=style.get("alpha", 1.0)
        ))
        legend_labels.append(label)

    if federal is not None: add_leg("Federal", STYLES["nm_federal_land"])
    if state_tr is not None: add_leg("State Trust", STYLES["nm_state_trust"])
    if tribal is not None: add_leg("Tribal", STYLES["nm_tribal_land"])
    if private is not None: add_leg("Private", STYLES["nm_private_land"])
    if nm is not None: add_leg("NM Boundary", STYLES["nm_boundary"])

    if legend_patches:
        ax.legend(legend_patches, legend_labels, loc="lower left", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG.resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
