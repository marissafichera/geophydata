import pandas as pd
import matplotlib.pyplot as plt
import os
import textwrap

csv_directory = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization\Fig8data'
# Apply dark background globally

import numpy as np

import re
import os


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

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def get_gradient_colors(values, cmap_positive='summer_r', cmap_negative='autumn'):
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    colors = []
    for v in values:
        if v >= 0:
            colors.append(cm.get_cmap(cmap_positive)(norm(v)))
        else:
            colors.append(cm.get_cmap(cmap_negative)(norm(v)))
    return colors



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
            colors = get_gradient_colors(values)
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


        # if not df_sector_valid.empty:
        #     plt.figure(figsize=(10, 6))
        #     plt.barh(
        #         df_sector_valid['Sector'],
        #         df_sector_valid['Change'],
        #         color='skyblue'
        #     )
        #
        #     plt.title(f"{region} Employment Change 2022-2032")
        #     plt.xlabel("Change")
        #     plt.ylabel("Sector")
        #     plt.tight_layout()
        # else:
        #     print(f"Skipping {sector_file} — no valid (non-negative) 'Change' values or all were excluded.")

        # # Only plot if there's data to show
        # if not df_sector_valid.empty:
        #     plt.figure()
        #     plt.pie(
        #         df_sector_valid['Change'],
        #         labels=df_sector_valid['Sector'],
        #         autopct='%1.1f%%',
        #         startangle=90
        #     )
        #     plt.title(f"Sector Change - {region}")
        #     plt.axis('equal')
        #     plt.tight_layout()
        # else:
        #     print(f"Skipping {sector_file} — no valid (non-negative) 'Change' values.")

        # # plot bar chart
        # plt.figure()
        # plt.barh(df_sector['Sector'], df_sector['Change'], color='skyblue')
        # plt.title(f"Sector Change - {region}")
        # plt.xlabel("Change")
        # plt.tight_layout()

        # # Plot pie chart
        # plt.figure()
        # plt.pie(df_sector['Change'], labels=df_sector['Sector'], autopct='%1.1f%%', startangle=90)
        # plt.title(f"Sector Change - {region}")
        # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        # plt.tight_layout()

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
