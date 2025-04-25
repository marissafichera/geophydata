import pandas as pd
import matplotlib.pyplot as plt
import os
import textwrap
import sys
import numpy as np
import re
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

csv_directory = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization\Fig8data'

# Water use classification by sector (lowercase for matching)
high_water_use = {
    "agriculture, forestry, fishing and hunting",
    "mining",
    "utilities",
    "manufacturing",
    "construction"
}

moderate_water_use = {
    "accommodation and food services",
    "arts, entertainment, and recreation",
    "transportation and warehousing",
    "administrative and support and waste management and remediation services"
}

# Default everything else to low water use
def get_water_use_color(sector):
    s = sector.strip().lower()
    if s in high_water_use:
        return "red"
    elif s in moderate_water_use:
        return "yellow"
    else:
        return "green"



def save_plot_by_title(title, output_dir="plots"):
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize the title to create a valid filename
    filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') + ".png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")


def clean_numeric_column(series):
    return (
        series.astype(str)
        .str.replace(r'[^\d\.\-]', '', regex=True)  # remove everything except digits, dot, and minus
        .replace('', np.nan)  # empty strings to NaN
        .astype(float)  # convert to float
    )



def get_gradient_colors(values, cmap_positive='summer_r', cmap_negative='autumn'):
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    colors = []
    for v in values:
        if v >= 0:
            colors.append(cm.get_cmap(cmap_positive)(norm(v)))
        else:
            colors.append(cm.get_cmap(cmap_negative)(norm(v)))
    return colors



def employment_projections():
    # Define the regions and file patterns
    regions = ['Central', 'Northern', 'Southwestern', 'Eastern']
    # regions = ['Eastern2']
    sector_colnames = ["Sector", "Region", "Sector2", "PercentOrNumeric", "22Emp", "32Emp", "Change"]
    subsector_colnames = ["Subsector", "Region", "Subsector2", "PercentOrNumeric", "22Emp", "32Emp", "Change"]

    # Read and process files
    for region in regions:
        # --- Handle Sector files ---
        sector_file = os.path.join(csv_directory, f"Sector_{region}.csv")
        if os.path.exists(sector_file):
            df_sector = pd.read_csv(sector_file, skiprows=1, header=None, sep=None, engine='python')
            df_sector.columns = sector_colnames

            df_sector['22Emp'] = clean_numeric_column(df_sector['22Emp'])
            df_sector['32Emp'] = clean_numeric_column(df_sector['32Emp'])
            df_sector['Change'] = clean_numeric_column(df_sector['Change'])

            # Filter to keep only non-negative and non-null values
            df_sector_valid = df_sector[df_sector['Change'].notnull() &

                                        (df_sector['Sector'].str.strip().str.lower() != "total all industries")]
            df_sector_valid = df_sector_valid.sort_values(by='Change', ascending=True)

            if not df_sector_valid.empty:
                df_sector_valid = df_sector_valid.sort_values(by='Change', ascending=True)
                values = df_sector_valid['Change']
                labels = df_sector_valid['Sector']
                # colors = get_gradient_colors(values)
                colors = [get_water_use_color(sector) for sector in df_sector_valid['Sector']]
                n_rows = len(df_sector_valid)
                fig_height = max(6, 0.4 * n_rows)

                plt.figure(figsize=(10, fig_height))
                # plt.barh(labels, values, color=colors)
                plt.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)

                plt.title(f"Projected Employment Change 2022-2032: {region}")
                plt.xlabel("Projected Change")
                plt.ylabel("Sector")
                ax = plt.gca()
                labels = [label.get_text() for label in ax.get_yticklabels()]
                wrapped_labels = [textwrap.fill(label, 25) for label in labels]  # wrap every 25 characters
                ax.set_yticklabels(wrapped_labels)

                plt.tight_layout()
                save_plot_by_title(f"Sector{region}")

            else:
                print(f"Skipping {sector_file} — no valid (non-negative) 'Change' values or all were excluded.")


        # --- Handle Subsector files ---
        subsector_file = os.path.join(csv_directory, f"Subsector_{region}.csv")
        if os.path.exists(subsector_file):
            df_subsector = pd.read_csv(subsector_file, skiprows=1, header=None, sep=None, engine='python')
            df_subsector.columns = subsector_colnames

            df_subsector['22Emp'] = clean_numeric_column(df_subsector['22Emp'])
            df_subsector['32Emp'] = clean_numeric_column(df_subsector['32Emp'])
            df_subsector['Change'] = clean_numeric_column(df_subsector['Change'])

            # Filter to keep only non-negative and non-null values
            df_subsector_valid = df_subsector[df_subsector['Change'].notnull() &

                                        (df_subsector['Subsector'].str.strip().str.lower() != "total all industries")]
            df_subsector_valid = (
                df_subsector_valid[df_subsector_valid['Change'] > 0]
                .sort_values(by='Change', ascending=False)
                .head(20)
                .sort_values(by='Change')  # To get ascending layout in horizontal bar chart
            )
            df_subsector_valid = df_subsector_valid.sort_values(by='Change', ascending=True)

            if not df_subsector_valid.empty:
                df_subsector_valid = df_subsector_valid.sort_values(by='Change', ascending=True)
                values = df_subsector_valid['Change']
                labels = df_subsector_valid['Subsector']
                colors = get_gradient_colors(values)
                n_rows = len(df_subsector_valid)
                fig_height = max(6, 0.4 * n_rows)

                plt.figure(figsize=(10, fig_height))
                # plt.barh(labels, values, color=colors)
                plt.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)

                plt.title(f"Projected Employment Change 2022-2032: {region}")
                plt.xlabel("Projected Change")
                plt.ylabel("Subsector")
                ax = plt.gca()
                labels = [label.get_text() for label in ax.get_yticklabels()]
                wrapped_labels = [textwrap.fill(label, 25) for label in labels]  # wrap every 25 characters
                ax.set_yticklabels(wrapped_labels)

                plt.tight_layout()
                save_plot_by_title(f"Subsector{region}")

            else:
                print(f"Skipping {sector_file} — no valid (non-negative) 'Change' values or all were excluded.")


    # Show all plots
    plt.show()


def edd_data():
    import pandas as pd
    import geopandas as gpd
    import os

    # === CONFIGURATION ===
    CSV_PATH = 'economicdata\MTGRbyIndustry_FY23Q2_FY25Q2.csv'
    SHAPEFILE_PATH = r"W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\basemap\counties83.shp"
    OUTPUT_FOLDER = 'economicdata'
    FY_START = '23_2'
    FY_END = '25_2'

    # === MAKE SURE OUTPUT FOLDER EXISTS ===
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # === LOAD ECONOMIC DATA ===
    df = pd.read_csv(CSV_PATH)



    # === GET LIST OF COUNTIES FROM COLUMN HEADERS ===
    # === EXTRACT COUNTY NAMES FROM CSV (REMOVE " County" AND CONVERT TO UPPERCASE) ===
    county_columns_csv = [col for col in df.columns if col not in ['Industry', 'FiscalYear_Qtr']]
    county_map = {col: col.replace(' County', '').upper() for col in county_columns_csv}
    df = df.rename(columns=county_map)
    counties = list(county_map.values())

    # === LOAD COUNTY SHAPEFILE ===
    gdf_counties = gpd.read_file(SHAPEFILE_PATH)

    # === PROCESS EACH INDUSTRY ===
    industries = df['Industry'].unique()

    for industry in industries:
        df_ind = df[df['Industry'] == industry]

        # Create pivot table with rows = FiscalYear_Qtr, columns = counties
        df_pivot = df_ind.set_index('FiscalYear_Qtr')[counties]


        # Transpose so counties are rows
        df_transposed = df_pivot.T
        df_transposed.columns = ['FY23_Q2', 'FY25_Q2']
        df_transposed['Industry'] = industry
        df_transposed['County'] = df_transposed.index

        df_transposed['FY23_Q2'] = pd.to_numeric(
            df_transposed['FY23_Q2'].astype(str).str.replace(',', ''), errors='coerce'
        )

        df_transposed['FY25_Q2'] = pd.to_numeric(
            df_transposed['FY25_Q2'].astype(str).str.replace(',', ''), errors='coerce'
        )

        # Compute Growth and YOYChange
        df_transposed['Growth'] = df_transposed['FY25_Q2'] - df_transposed['FY23_Q2']
        df_transposed['YOYChange'] = (df_transposed['Growth'] / df_transposed['FY23_Q2']) * 100

        # Clean up and reorder columns
        df_final = df_transposed[['Industry', 'County', 'FY23_Q2', 'FY25_Q2', 'Growth', 'YOYChange']].reset_index(
            drop=True)

        # === JOIN TO SHAPEFILE ===
        gdf_joined = gdf_counties.merge(df_final, left_on='NAME', right_on='County', how='left')

        # === EXPORT TO FILE ===
        output_path = os.path.join(OUTPUT_FOLDER, f"{industry.replace('/', '_').replace(' ', '_')}_joined.shp")
        gdf_joined.to_file(output_path)
        print(f"Saved: {output_path}")


def main():
    edd_data()


if __name__ == '__main__':
    main()
