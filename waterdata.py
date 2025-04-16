import geopandas as gpd
import os
import arcpy as ap
import pandas as pd
from shapely.geometry import Point
import sys

root = r'\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\NMHydrogeoData'
default_gdb = r'\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\NMHydrogeoData.gdb'
ap.env.overwriteOutput = True



def save_script(output_script_folder, script_name='waterdata.txt'):
    base_name, ext = os.path.splitext(script_name)
    output_script_path = os.path.join(output_script_folder, script_name)

    # If the file exists, append an incrementing number to avoid overwrite
    counter = 1
    while os.path.exists(output_script_path):
        new_name = f"{base_name}_{counter}{ext}"
        output_script_path = os.path.join(output_script_folder, new_name)
        counter += 1

    # Get the path of the currently running script
    current_script = os.path.abspath(__file__)

    # Read the current script's contents
    with open(current_script, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Write the content to the new file
    with open(output_script_path, 'w') as out_f:
        out_f.write(script_content)

    print(f"Script saved as: {output_script_path}")


def geojson_to_shapefile(input_geojson, output_shapefile):
    """
    Converts a GeoJSON file to an ESRI Shapefile.

    Parameters:
    - input_geojson (str): Path to the input GeoJSON file.
    - output_shapefile (str): Path to save the output Shapefile.

    Returns:
    - None
    """
    try:
        # Read the GeoJSON into a GeoDataFrame
        gdf = gpd.read_file(input_geojson)

        # Save to shapefile
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')

        print(f"✅ Successfully converted to shapefile: {output_shapefile}")
    except Exception as e:
        print(f"❌ Error: {e}")

def csv_to_shp():
    # Input CSV file path
    input_csv = os.path.join(root, 'USGS008h', 'Diss_Solids.csv')

    # Output Shapefile path
    output_shapefile = os.path.join(root, 'USGS008h', 'Diss_Solids.shp')

    # Read CSV into pandas DataFrame
    df = pd.read_csv(input_csv)

    for c in df.columns:
        print(c)

    sys.exit()

    # Filter for New Mexico (NM) rows
    df_nm = df[df['state_alpha'] == 'NM']

    # Check if DataFrame is not empty
    if df_nm.empty:
        print("No data for New Mexico found in the CSV.")
    else:
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_nm,
            geometry=gpd.points_from_xy(df_nm['dec_long_va'], df_nm['dec_lat_va']),
            crs='EPSG:4326'  # WGS84 Latitude/Longitude
        )

        # Export to Shapefile
        gdf.to_file(output_shapefile)

        print(f"Shapefile successfully created at: {output_shapefile}")


def gdb_to_shapefile(input_gdb, feature_class, output_shapefile):
    """
    Converts a feature class from a geodatabase to a shapefile,
    filtering for rows where 'state_alpha' == 'NM'.

    Parameters:
        input_gdb (str): Path to the input geodatabase (.gdb)
        feature_class (str): Name of the feature class inside the GDB
        output_shapefile (str): Path to the output shapefile (.shp)
    """

    # Build the full path to the feature class
    input_path = r'\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\BrackishWater_DissolvedSolids.gdb'

    # Read the feature class into a GeoDataFrame
    gdf = gpd.read_file(input_path)

    # Filter for New Mexico (NM)
    gdf_nm = gdf[gdf['state_alpha'] == 'NM']

    if gdf_nm.empty:
        print("No features found with state_alpha == 'NM'. No shapefile created.")
    else:
        # Export to Shapefile
        gdf_nm.to_file(output_shapefile)
        print(f"Shapefile created: {output_shapefile}")


def main():
    subroot = 'USGS008h'

    input_gdb = os.path.join(root, 'BrackishWater_DissolvedSolids.gdb')
    feature_class = "Diss_Solids"
    output_shapefile = os.path.join(root, subroot, 'Diss_Solids.shp')

    gdb_to_shapefile(input_gdb, feature_class, output_shapefile)

    # output_script_folder = os.path.join(root, subroot)
    # script_name = 'script.txt'
    #
    # save_script(output_script_folder, script_name)

if __name__ == '__main__':
    main()