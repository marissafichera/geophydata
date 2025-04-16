import geopandas as gpd
import glob
import os

def split_geojson_by_column(input_geojson, output_folder, column_name="source"):
    """
    Splits a GeoJSON file into multiple GeoJSON files based on unique values in a specified column.

    Parameters:
    - input_geojson (str): Path to the input GeoJSON file.
    - output_folder (str): Folder to save the output GeoJSON files.
    - column_name (str): The column to split by (default is "source").

    Returns:
    - None
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the input GeoJSON
    gdf = gpd.read_file(input_geojson)

    # Get unique values in the specified column
    unique_values = gdf[column_name].dropna().unique()

    print(f"Found {len(unique_values)} unique '{column_name}' values.")

    # Split and export
    for value in unique_values:
        subset = gdf[gdf[column_name] == value]
        safe_value = str(value).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_folder, f"{safe_value}.geojson")
        subset.to_file(output_path, driver="GeoJSON")
        print(f"✅ Exported {value} to {output_path}")

    print("\n✅ All exports completed!")


def batch_shapefile_to_geojson(input_folder, output_folder, force_wgs84=True):
    """
    Batch convert all shapefiles in a folder to GeoJSON format.

    Parameters:
    - input_folder (str): Folder containing shapefiles
    - output_folder (str): Folder to save GeoJSON files
    - force_wgs84 (bool): Reproject to WGS84 (EPSG:4326) for web use

    Returns:
    - None
    """
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Find all shapefiles in the input folder (recursive optional)
    shapefiles = glob.glob(os.path.join(input_folder, "*.shp"))

    print(f"Found {len(shapefiles)} shapefiles to convert.")

    for shp_path in shapefiles:
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)

            # Optionally reproject to WGS84
            if force_wgs84:
                gdf = gdf.to_crs(epsg=4326)

            # Define output path
            base_name = os.path.splitext(os.path.basename(shp_path))[0]
            output_geojson = os.path.join(output_folder, f"{base_name}.geojson")

            # Save as GeoJSON
            gdf.to_file(output_geojson, driver='GeoJSON')

            print(f"✅ Converted: {base_name}.shp → {base_name}.geojson")

        except Exception as e:
            print(f"❌ Error converting {shp_path}: {e}")

    print("\n✅ All conversions completed!")




def main():
    # === Example usage ===
    split_geojson_by_column(
        input_geojson=r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\mapbox\watersystems_wells_streams\sites_die.geojson",
        output_folder=r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\mapbox\watersystems_wells_streams",
        column_name="source"
    )

    # # === Example usage ===
    # batch_shapefile_to_geojson(
    #     input_folder=r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\mapbox\watersystems_wells_streams",
    #     output_folder=r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\mapbox\watersystems_wells_streams"
    # )


if __name__ == '__main__':
    main()

