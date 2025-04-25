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

def process_usgs_tds_data(subroot, gdf):
    # need to export separately anything with well_depth = 0 AND top_of_scr = 0
    # should export two different shapefiles at end

    gdf_wd0 = gdf[(gdf['well_depth'] == 0) & (gdf['top_of_scr'] == 0)]
    gdf_wd0.to_file(os.path.join(root, subroot, 'USGS_TDS_noWD.shp'))

    gdf_wd = gdf[(gdf['well_depth'] != 0) | (gdf['top_of_scr'] != 0)]
    gdf_wd.to_file(os.path.join(root, subroot, 'USGS_TDS_WD.shp'))


def intersect_analyte_shapefiles(analytes, folder=".", id_col="id", output_path="intersected_records.shp"):
    """
    Reads shapefiles for each analyte and outputs a shapefile of records that exist in all of them by ID.

    Parameters:
        analytes (list of str): List of analyte names (e.g., ["arsenic", "uranium"])
        folder (str): Folder where the shapefiles are located
        id_col (str): Column name used to match records (default: "id")
        output_path (str): Output shapefile path
    """
    common_ids = None
    gdf_dict = {}

    for analyte in analytes:
        path = os.path.join(folder, f"{analyte}_cl.shp")
        if not os.path.exists(path):
            print(f"⚠️ Missing shapefile: {path}")
            return

        gdf = gpd.read_file(path)
        gdf_dict[analyte] = gdf

        ids = set(gdf[id_col])
        common_ids = ids if common_ids is None else common_ids.intersection(ids)

    if not common_ids:
        print("❌ No common IDs found across all shapefiles.")
        return

    # Use the first GeoDataFrame and filter to common IDs
    base_analyte = analytes[0]
    result_gdf = gdf_dict[base_analyte][gdf_dict[base_analyte][id_col].isin(common_ids)]

    # Save to shapefile
    result_gdf.to_file(os.path.join(root, output_path))
    print(f"✅ Output written to: {output_path}")




def intersect_with_optional_union(analytes, folder=".", id_col="id",
                                   union_group={"bicarbonate", "carbonate"},
                                   output_path="intersected_with_union.shp"):
    """
    Reads shapefiles by analyte name and outputs records present in all analytes,
    allowing union logic (OR) for a specified group (like bicarbonate OR carbonate).
    """
    gdf_dict = {}
    common_ids = None
    union_ids = set()

    for analyte in analytes:
        path = os.path.join(folder, f"{analyte}_cl.shp")
        if not os.path.exists(path):
            print(f"⚠️ Missing shapefile: {path}")
            return

        gdf = gpd.read_file(path)
        gdf_dict[analyte] = gdf
        ids = set(gdf[id_col])

        if analyte in union_group:
            union_ids.update(ids)  # OR logic: union of IDs
        else:
            common_ids = ids if common_ids is None else common_ids.intersection(ids)

    # Combine: intersected IDs + union group IDs
    final_ids = common_ids.intersection(union_ids) if union_ids else common_ids

    if not final_ids:
        print("❌ No matching records found with given criteria.")
        return

    # Use one of the GeoDataFrames (any analyte) to extract geometry and IDs
    base_analyte = analytes[0]
    result_gdf = gdf_dict[base_analyte][gdf_dict[base_analyte][id_col].isin(final_ids)]

    # Save result
    result_gdf.to_file(os.path.join(root, output_path))
    print(f"✅ Output written to: {output_path}")

def hydrogeology_studies():
    df = pd.read_csv('HydrogeologyStudiesNM.csv')

    print(df['Agency or Journal'].unique())
    print(df['Agency or Journal'].nunique())
    print(df['Agency or Journal'].value_counts())


def main():
    hydrogeology_studies()

    # analytes = ['calcium',
    #             'sodium',
    #             'potassium',
    #             'magnesium',
    #             'sulfate',
    #             'chloride',
    #             'bicarbonate',
    #             'carbonate']
    #
    # f = r'W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\scratch'
    # intersect_with_optional_union(analytes, folder=f, id_col='id', output_path='piper_HCOorCO.shp')
    # intersect_analyte_shapefiles(analytes, folder=f, output_path="piper_elements.shp")

    # input_gdb = os.path.join(root, 'BrackishWater_DissolvedSolids.gdb')
    # feature_class = "Diss_Solids"
    # output_shapefile = os.path.join(root, subroot, 'Diss_Solids.shp')
    #
    # gdb_to_shapefile(input_gdb, feature_class, output_shapefile)

    # output_script_folder = os.path.join(root, subroot)
    # script_name = 'script.txt'
    #
    # save_script(output_script_folder, script_name)

if __name__ == '__main__':
    main()