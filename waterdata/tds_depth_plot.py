import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load shapefiles
diss_solids = gpd.read_file(r'W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\NMHydrogeoData\USGS008h\Diss_Solids.shp')
summary_tds = gpd.read_file(r'W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\scratch\summary_tds_all_cl.shp')

# Prepare Diss_Solids data
# Only keep rows where either well_depth or top_of_scr is not zero
diss_solids = diss_solids[(diss_solids['well_depth'] != 0) | (diss_solids['top_of_scr'] != 0)]
diss_solids['depth_for_plot'] = diss_solids.apply(
    lambda row: row['well_depth'] if row['well_depth'] != 0 else row['top_of_scr'], axis=1
)

# Prepare summary_tds data
summary_tds['depth_for_plot'] = summary_tds['well_depth']
summary_tds_filtered = summary_tds[summary_tds['well_depth'] != 0]

# Combine datasets for easier plotting
combined_df = gpd.GeoDataFrame({
    'TDS': pd.concat([diss_solids['TDS_mgL'], summary_tds_filtered['most_recen']]),
    'Depth': pd.concat([diss_solids['depth_for_plot'], summary_tds_filtered['depth_for_plot']])
})

# Function to assign color based on TDS value
def assign_color(tds_value):
    if tds_value < 1000:
        return 'green'
    elif 1000 <= tds_value < 3000:
        return 'yellow'
    elif 3000 <= tds_value < 10000:
        return 'orange'
    elif 10000 <= tds_value <= 35000:
        return 'red'
    else:
        return 'gray'  # For values > 35000 or unexpected

# Apply color classification
combined_df['Color'] = combined_df['TDS'].apply(assign_color)

# Count how many points fall into each water quality zone
zone_counts = {
    'Freshwater (<1,000 mg/L)': (combined_df['TDS'] < 1000).sum(),
    'Slightly Brackish (1,000–3,000 mg/L)': ((combined_df['TDS'] >= 1000) & (combined_df['TDS'] < 3000)).sum(),
    'Moderately Brackish (3,000–10,000 mg/L)': ((combined_df['TDS'] >= 3000) & (combined_df['TDS'] < 10000)).sum(),
    'Heavily Brackish (10,000–35,000 mg/L)': ((combined_df['TDS'] >= 10000) & (combined_df['TDS'] <= 35000)).sum(),
    'Very Saline (>35,000 mg/L)': (combined_df['TDS'] > 35000).sum()
}

# Print the counts
for zone, count in zone_counts.items():
    print(f"{zone}: {count} samples")

# Bar chart for zone counts
plt.figure(figsize=(10, 6))
plt.bar(zone_counts.keys(), zone_counts.values(), color=['green', 'yellow', 'orange', 'red', 'gray'], edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Samples')
plt.title('Sample Counts by Water Quality Zone')
plt.tight_layout()
plt.show()

# Split into two datasets based on TDS value
low_tds = combined_df[combined_df['TDS'] <= 35000]
high_tds = combined_df[combined_df['TDS'] > 35000]

# First Plot: TDS <= 35000
plt.figure(figsize=(10, 7))
plt.scatter(low_tds['TDS'], low_tds['Depth'], c=low_tds['Color'], edgecolors='black', alpha=0.6)
plt.gca().invert_yaxis()
plt.xlabel('TDS (mg/L)')
plt.ylabel('Well Depth (ft)')
plt.title('TDS <= 35,000 vs. Well Depth (Custom Colors)')

# Custom Legend
# Custom Legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='green', label='<1,000 mg/L (Freshwater)'),
    mpatches.Patch(color='yellow', label='1,000-3,000 mg/L (Slightly Brackish)'),
    mpatches.Patch(color='orange', label='3,000-10,000 mg/L (Moderately Brackish)'),
    mpatches.Patch(color='red', label='10,000-35,000 mg/L (Heavily Brackish)')
]
plt.legend(handles=legend_handles, title='TDS Ranges')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Plot: TDS <= 35000 with restricted axes
plt.figure(figsize=(10, 7))
plt.scatter(low_tds['TDS'], low_tds['Depth'], c=low_tds['Color'], edgecolors='black', alpha=0.6)
plt.gca().invert_yaxis()
plt.xlim(0, 10000)
plt.ylim(6000, 0)
plt.xlabel('TDS (mg/L)')
plt.ylabel('Well Depth (ft)')
plt.title('TDS <= 10,000 vs. Well Depth (0-6,000 ft range)')
plt.legend(handles=legend_handles, title='TDS Ranges')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# Second Plot: TDS > 35000
plt.figure(figsize=(10, 7))
plt.scatter(high_tds['TDS'], high_tds['Depth'], c='gray', edgecolors='black', alpha=0.6)
plt.gca().invert_yaxis()
plt.xlabel('TDS (mg/L)')
plt.ylabel('Well Depth (ft)')
plt.title('TDS > 35,000 vs. Well Depth')
plt.grid(True)
plt.tight_layout()
plt.show()
