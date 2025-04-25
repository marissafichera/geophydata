import arcpy
from arcpy.sa import *
import os
import arcpy
import numpy as np
import math
from math import sqrt, exp, pi
import sys
import pandas as pd
import geopandas as gpd


# Set environment
arcpy.env.workspace = r"\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\scratch"  # Change to your workspace
arcpy.env.overwriteOutput = True

ROOT = arcpy.env.workspace

extensions = ["Spatial", "3D", "Network", "GeoStats"]  # Add extensions as needed

for ext in extensions:
    status = arcpy.CheckOutExtension(ext)
    print(f"{ext} Extension: {status}")


def csv_to_shapefile(csv_path, shapefile_path, x_field="Longitude", y_field="Latitude", spatial_ref=4326):
    """
    Converts a CSV file to a Shapefile using ArcPy.

    Parameters:
    csv_path (str): Path to the input CSV file.
    shapefile_path (str): Path to the output Shapefile (.shp).
    x_field (str): Name of the longitude field in the CSV.
    y_field (str): Name of the latitude field in the CSV.
    spatial_ref (int or str): EPSG code or spatial reference for projection (default: WGS 1984 - EPSG:4326).

    Returns:
    str: Path to the created Shapefile.
    """

    # Set environment and workspace
    arcpy.env.overwriteOutput = True

    # Convert spatial reference if needed
    spatial_reference = arcpy.SpatialReference(spatial_ref)

    # Load your CSV
    df = pd.read_csv(csv_path)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.longitude, y=df.latitude), crs='EPSG:4326')
    gdf.to_file(shapefile_path)

    # Convert the date column (assuming it's called 'Date') to datetime format
    df['date_mostrecent_conv'] = pd.to_datetime(df['most_recent_date'], format='%d/%m/%Y', errors='coerce')

    # # Drop the original date column
    # df.drop(columns=['most_recent_date'], inplace=True)
    #
    # # Create a temporary layer from the CSV file
    # layer_name = "temp_layer"
    # arcpy.management.MakeXYEventLayer(csv_path, x_field, y_field, layer_name, spatial_reference)
    # print(arcpy.GetInstallInfo()['Version'])
    # environ_details = os.environ
    # fields = arcpy.ListFields(layer_name)
    # for field in fields:
    #     print(f"{field.name} has a type of {field.type} with a length of {field.length}")
    # # Export the layer to a shapefile
    # arcpy.management.CopyFeatures(layer_name, shapefile_path)
    #
    # print(f"Shapefile created successfully: {shapefile_path}")
    return shapefile_path



def is_geographic(feature_class):
    """
    Returns True if the feature class uses a geographic coordinate system.
    """
    sr = arcpy.Describe(feature_class).spatialReference
    return sr.type.lower() == "geographic"


def project_if_geographic(input_fc, target_sr, output_fc):
    """
    Projects the input feature class to the target spatial reference if it is geographic.
    Otherwise, returns the original input_fc.
    """
    if is_geographic(input_fc):
        arcpy.AddMessage("Input {} is geographic. Projecting to target spatial reference...".format(input_fc))
        arcpy.management.Project(input_fc, output_fc, target_sr)
        return output_fc
    else:
        arcpy.AddMessage("Input {} is already projected. No projection needed.".format(input_fc))
        return input_fc


def run_adaptive_kd(
        point_datasets,           # List of point feature classes
        output_dir,               # Folder to save output rasters
        output_combined_raster=None,   # Path to final combined raster
        population_field="NONE",  # Field to weight points, or "NONE"
        cell_size=1000,
        names=None,# Raster resolution in projected units (e.g., meters)
        ):

    density_rasters = []

    # Make sure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    for fc, name in zip(point_datasets, names):
        print(f"➡️  Processing dataset {name}: {fc}")

        out_raster_path = os.path.join(output_dir, f"{name}_adaptive_kd.tif")

        # Leave search_radius=None for adaptive behavior
        kd_raster = KernelDensity(
            in_features=fc,
            population_field=population_field,
            cell_size=cell_size,
            method='GEODESIC',
            search_radius=None  # 👈 Adaptive bandwidth
        )

        kd_raster.save(out_raster_path)
        print(f"✅ Saved: {out_raster_path}")
        density_rasters.append(out_raster_path)

    # print("🧮 Combining rasters with WeightedSum...")
    #
    # # WeightedSum expects a string like: "raster1 1; raster2 1; ..."
    # ws_input = [[f'{r}', 'VALUE', 1] for r in density_rasters]
    # combined = WeightedSum(WSTable(ws_input))
    # combined.save(output_combined_raster)
    #
    # print(f"🎉 Composite adaptive kernel density map saved at:\n{output_combined_raster}")
    #
    # return output_combined_raster, density_rasters


def clip_points_to_boundary(input_fc, clip_fc, output_fc):
    """
    Clips a point feature class to a polygon boundary.

    Parameters:
      input_fc (str): Path to the input points feature class
      clip_fc (str): Path to the polygon clip boundary
      output_fc (str): Path to save the clipped feature class
    """
    out_fc = os.path.join(ROOT, output_fc)
    arcpy.analysis.Clip(in_features=input_fc, clip_features=clip_fc, out_feature_class=out_fc)
    print(f"📍 Clipped {os.path.basename(input_fc)} to boundary → {out_fc}")
    return output_fc


def convert_shapefiles_to_geojson(folder_path, output_folder=None):
    if output_folder is None:
        output_folder = folder_path  # Save GeoJSONs to same folder if not specified

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('cl.shp'):
            shapefile_path = os.path.join(folder_path, filename)

            try:
                # Read shapefile
                gdf = gpd.read_file(shapefile_path)

                # Define output path (same name but .geojson)
                geojson_name = filename.replace('.shp', '.geojson')
                geojson_path = os.path.join(output_folder, geojson_name)

                # Export to GeoJSON
                gdf.to_file(geojson_path, driver='GeoJSON')

                print(f"✅ Converted: {filename} → {geojson_name}")

            except Exception as e:
                print(f"❌ Failed to convert {filename}: {e}")



# Example usage:
def main():
    names = ['summary_tds_all', 'summary_dtw_all']
    names = ['arsenic',
             'bicarbonate',
             'carbonate',
             'chloride',
             'magnesium',
             'nitrate',
             'ph',
             'potassium',
             'sodium',
             'sulfate']
    names = ['calcium']
    waterdata_csvs = [os.path.join('dieoutput', f'{name}', 'output', f'summary.csv') for name in names]

    # Define target spatial reference (for example, NAD83 UTM Zone 13N, EPSG:26913)
    # - using Albers Equal Area Conic EPSG:5070 or potentially ESRI:102008 to cover entire state of NM
    target_spatial_ref = arcpy.SpatialReference(5070)

    input_shps = []
    clip_fc = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\basemap\NM_dataextent_WGS.shp'
    # Project if necessary
    clip_fc_project = project_if_geographic(clip_fc, target_spatial_ref, output_fc=os.path.join(ROOT, 'NM_dataextent_project.shp'))
    for c, name in zip(waterdata_csvs, names):
        shp = (csv_to_shapefile(csv_path=c, shapefile_path=os.path.join(ROOT, f'{name}.shp'), x_field='longitude', y_field='latitude', spatial_ref=4326))
        input_shps.append(clip_points_to_boundary(input_fc=shp, clip_fc=clip_fc_project, output_fc=f'{name}_cl.shp'))
    print(input_shps)

    sys.exit()

    input_datasets = []
    for ishp in input_shps:
        input_datasets.append(os.path.join(ROOT, ishp))

    # Define a temporary workspace folder for intermediate files
    temp_workspace = os.path.join(ROOT, 'temp')
    if not os.path.exists(temp_workspace):
        os.makedirs(temp_workspace)

    input_fcs = []
    for fc, name in zip(input_datasets, names):
        # Define temporary projected feature class path
        projected_fc = os.path.join(temp_workspace, f"{name}_proj.shp")
        # Project if necessary
        input_for_analysis = project_if_geographic(fc, target_spatial_ref, projected_fc)
        input_fcs.append(input_for_analysis)

    # Define the output raster path
    # outrastername = 'composite_adaptive_density'
    # output_density_raster = os.path.join(ROOT, f'{outrastername}.tif')

    run_adaptive_kd(
        point_datasets=input_fcs,
        output_dir=ROOT,
        output_combined_raster=None,
        population_field='NONE',
        cell_size=100,
        names=names,
    )

    # re-project back into WGS1984
    # target_spatial_ref = arcpy.SpatialReference(4326)
    # arcpy.management.DefineProjection(in_dataset=output_combined_raster, coor_system=target_spatial_ref)


if __name__ == '__main__':
    main()

