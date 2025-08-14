# plot_nm_pop_density_tracts_log.py
# Requires: geopandas, matplotlib; optional: mapclassify
# pip install geopandas matplotlib mapclassify

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np

# --- Inputs ---
GPKG      = Path("out/potential_wells/out/pop_density/nm_pop_density_tracts.gpkg")
SHP       = Path("out/potential_wells/out/pop_density/nm_pop_density_tracts.shp")
COUNTIES  = Path("waterdata/tl_2018_nm_county.shp")
NM_BOUND  = Path("nm_land_status_shp/nm_boundary.shp")

# --- Output ---
OUT_PNG   = Path("out/pop_density/nm_pop_density_map_log.png")

# --- Options ---
USE_MI2      = True        # plot people per square mile (else km²)
SCHEME       = "log_quantiles"  # "log_quantiles", "headtail", "fisherjenks", or "quantiles"
K_CLASSES    = 7           # number of classes for quantiles/fisherjenks
CMAP_NAME    = "viridis"   # try "turbo", "YlGnBu", "PuBuGn", "Spectral_r"
CLIP_HIGH_P  = 99.5        # clip top tail (percentile) to tame outliers

def load_layers():
    g = gpd.read_file(GPKG, layer="nm_pop_density_tracts") if GPKG.exists() else gpd.read_file(SHP)
    counties = gpd.read_file(COUNTIES)
    nm = gpd.read_file(NM_BOUND) if NM_BOUND.exists() else counties.dissolve().reset_index(drop=True)
    # Align CRS
    target = counties.crs
    if g.crs != target: g = g.to_crs(target)
    if nm.crs != target: nm = nm.to_crs(target)
    return g, counties, nm

def format_bin_label(lo, hi):
    def f(x):
        if x is None or np.isnan(x): return "NA"
        return f"{int(round(x)):,}"
    return f"{f(lo)}–{f(hi)}"

def main():
    g, counties, nm = load_layers()
    dens_col = "DENS_SQMI" if (USE_MI2 and "DENS_SQMI" in g.columns) else "DENS_KM2"
    units = "people / mi²" if dens_col == "DENS_SQMI" else "people / km²"

    vals = g[dens_col].astype(float).replace([np.inf, -np.inf], np.nan)
    # clip the crazy tail
    hi = np.nanpercentile(vals, CLIP_HIGH_P)
    vals = vals.clip(upper=hi)

    fig, ax = plt.subplots(figsize=(10.5, 10.5))

    # Try discrete classes with mapclassify
    try:
        import mapclassify as mc
        # choose data to classify
        data_for_class = vals.copy()

        if SCHEME.lower().startswith("log"):
            # log10 transform (ignore zeros/NaN)
            data_for_class = np.log10(data_for_class.replace(0, np.nan))

        valid_mask = ~data_for_class.isna()
        data_valid = data_for_class[valid_mask]

        if data_valid.empty:
            raise ValueError("No valid density values after preprocessing.")

        scheme_lower = SCHEME.lower()
        if scheme_lower == "headtail":
            classifier = mc.HeadTailBreaks(data_valid)   # automatic class count
            edges_t = np.concatenate([[float(data_valid.min())], classifier.bins])
        elif scheme_lower == "fisherjenks":
            classifier = mc.FisherJenks(data_valid, k=K_CLASSES)
            edges_t = np.concatenate([[float(data_valid.min())], classifier.bins])
        elif scheme_lower == "quantiles" or scheme_lower == "log_quantiles":
            classifier = mc.Quantiles(data_valid, k=K_CLASSES)
            edges_t = np.concatenate([[float(data_valid.min())], classifier.bins])
        else:
            classifier = mc.Quantiles(data_valid, k=K_CLASSES)
            edges_t = np.concatenate([[float(data_valid.min())], classifier.bins])

        # If we classified in log space, convert edges back to original units for labels
        if SCHEME.lower().startswith("log"):
            edges = np.power(10, edges_t)
        else:
            edges = edges_t

        # Class indices aligned to valid_mask
        class_idx = vals.copy() * np.nan
        class_idx.loc[valid_mask.index[valid_mask]] = classifier.yb  # yb is 0..k-1

        # Colors
        k = int(np.nanmax(class_idx)) + 1
        cmap = plt.get_cmap(CMAP_NAME, k)
        colors = [cmap(i) for i in range(k)]
        facecolors = np.array(["#dddddd"] * len(g), dtype=object)  # NaN -> light gray
        for i in range(k):
            facecolors[class_idx == i] = colors[i]

        # Plot tracts
        g.plot(ax=ax, color=facecolors, edgecolor="white", linewidth=0.15, zorder=1)

        # Legend (off-map, right)
        patches, labels = [], []
        for i in range(k):
            lo, hi = edges[i], edges[i+1]
            patches.append(Patch(facecolor=colors[i], edgecolor="white"))
            labels.append(format_bin_label(lo, hi))
        leg = ax.legend(
            patches, labels,
            title=f"Population density\n({units})",
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=True, framealpha=0.9, borderaxespad=0.0
        )
        ax.add_artist(leg)
        plt.tight_layout(rect=[0, 0, 0.82, 1])

    except Exception:
        # Fallback: continuous map with log color scale if requested
        from matplotlib.colors import LogNorm
        if SCHEME.lower().startswith("log"):
            v = vals.copy().replace(0, np.nan)
            vmin = np.nanpercentile(v, 2)
            vmax = np.nanpercentile(v, 98)
            norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin*10))
        else:
            v = vals
            vmin = float(np.nanpercentile(v, 2))
            vmax = float(np.nanpercentile(v, 98))
            norm = Normalize(vmin=vmin, vmax=vmax)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_NAME)
        g.plot(ax=ax, column=dens_col, cmap=CMAP_NAME, norm=norm,
               edgecolor="white", linewidth=0.15, zorder=1)

        cax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
        cb = plt.colorbar(mappable, cax=cax)
        cb.set_label(f"Population density ({units})")
        plt.tight_layout(rect=[0, 0, 0.82, 1])

    # overlays
    counties.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=2)
    nm.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=3)

    ax.set_title("New Mexico Population Density by Census Tract", pad=12)
    ax.set_axis_off()
    ax.set_aspect("equal")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved map: {OUT_PNG.resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
