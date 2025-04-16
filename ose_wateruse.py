import pandas as pd
import os

def table5():
    # Step 1: Load Excel into a DataFrame
    df = pd.read_excel(r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization\OSEWaterUse2020\Table5.xlsx")

    # Step 2: Drop all rows that repeat the header
    expected_cols = ["CN", "County", "Category", "WSW", "WGW", "TW"]
    df_cleaned = df[df.columns.intersection(expected_cols).tolist()]  # ensure columns exist

    # Step 3: Remove all rows that match the header structure (e.g., repeat headers)
    df = df[~((df["CN"] == "CN") & (df["County"] == "County"))]

    # Step 2: Fill missing "County" and "CN" values using the value from the previous row
    df['County'] = df['County'].fillna(method='ffill')
    df['CN'] = df['CN'].fillna(method='ffill')

    # Optional: Save the cleaned result
    df.to_csv("waterdata\cleaned_county_data.csv", index=False)

    return df


def sort_by_category(df):
    # Create output folder
    output_dir = "waterdata\ose_categories"
    os.makedirs(output_dir, exist_ok=True)

    # Group by Category and write each group to its own CSV
    for category, group_df in df.groupby("Category"):
        # Sanitize filename: replace spaces and remove bad characters
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in category).strip().replace(" ", "_")
        filename = f"{safe_name}.csv"

        group_df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Saved {len(df['Category'].unique())} CSV files to '{output_dir}/'")

def make_shapefiles():
    import geopandas as gpd
    import pandas as pd
    import os

    # Paths
    shapefile_path = r"W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\basemap\counties83.shp"
    csv_dir = "waterdata\ose_categories"
    output_dir = r'W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\NMHydrogeoData'
    os.makedirs(output_dir, exist_ok=True)

    # Load the shapefile and normalize the NAME column
    gdf = gpd.read_file(shapefile_path)
    gdf["NAME_clean"] = gdf["NAME"].str.strip().str.lower()

    # Loop through each CSV in the category folder
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            category_name = os.path.splitext(filename)[0]
            csv_path = os.path.join(csv_dir, filename)

            # Load and clean CSV
            df = pd.read_csv(csv_path)
            df["County_clean"] = df["County"].str.strip().str.lower()

            # Join with shapefile
            joined = gdf.merge(df, left_on="NAME_clean", right_on="County_clean", how="left")

            # Drop helper columns
            joined = joined.drop(columns=["NAME_clean", "County_clean"])

            # Output shapefile (or use .gpkg for geopackage)
            output_path = os.path.join(output_dir, f"{category_name}.shp")
            joined.to_file(output_path)

            print(f"Saved: {output_path}")


def main():
    # df = table5()
    df = pd.read_csv(r"waterdata\cleaned_county_data.csv")
    sort_by_category(df)
    make_shapefiles()

if __name__ == '__main__':
    main()
