from idlelib.pyparse import trans

import numpy as np
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.wkt import loads
from shapely.geometry import Point
import pyproj
import sys
import fiona
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import arcpy as ap
import seaborn as sns
import re

root = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData'
default_gdb = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb'
ap.env.overwriteOutput = True


xmin = -12216837
xmax = -11350902
ymin = 3595190
ymax = 4523656

# Create transformer objects
transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Convert to WGS84
lonmin, latmin = transformer.transform(xmin, ymin)
lonmax, latmax = transformer.transform(xmax, ymax)

nm_extent_wgs84 = (lonmin, latmin, lonmax, latmax)

import os
import zipfile


# Function to extract all zip files in the given directory
def extract_zip_files(directory):
    # Loop through every file in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a zip file
        if filename.endswith('.zip'):
            zip_path = os.path.join(directory, filename)

            # Create a folder to extract the contents of the zip file
            extract_folder = os.path.join(directory, os.path.splitext(filename)[0])
            if not os.path.exists(extract_folder):
                os.makedirs(extract_folder)

            # Open and extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                print(f"Extracted {filename} to {extract_folder}")


def copy_raster(in_raster_dir, in_raster_name):
    in_raster_path = os.path.join(in_raster_dir, f'{in_raster_name}.tif')
    print(f'in raster = {in_raster_path}')

    out_gdb = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb'
    out_path = os.path.join(out_gdb, f'{in_raster_name}')
    print(f'out raster path = {out_path}')

    ap.conversion.RasterToGeodatabase(in_raster_path, out_gdb)


def modify_filename(filename):
    print(f'filename = {filename}')
    # Check if the filename starts with 'geophysics'
    if filename.lower().startswith('geophysics'):
        filename = filename[10:]  # Remove 'geophysics' from the start (10 characters)
        # Check if the filename ends with 'USCanada'
        if filename.lower().endswith('uscanada'):
            filename = filename[:-8]  # Remove 'USCanada' from the end (8 characters)

    return f'USGS002{filename}'


def clip_usgs002_data():
    gravtifs = [
        'GeophysicsGravity_HGM_USCanada',
        'GeophysicsGravity_Up30km_HGM_USCanada',
        'GeophysicsGravity_USCanada',
    ]

    gravshps = [
        'DeepGravitySources_Worms_USCanada',
        'ShallowGravitySources_Worms_USCanada'
    ]

    tifs = [
        'GeophysicsGravity_HGM_USCanada',
        'GeophysicsGravity_Up30km_HGM_USCanada',
        'GeophysicsGravity_Up30km_USCanada',
        'GeophysicsGravity_USCanada',
        'GeophysicsMag_RTP_HGM_USCanada',
        'GeophysicsMag_RTP_USCanada',
        'GeophysicsMag_RTP_VD_USCanada',
        'GeophysicsMag_USCanada',
        'GeophysicsMagRTP_DeepSources_USCanada',
        'GeophysicsMagRTP_HGM_DeepSources_USCanada'
    ]

    shps = [
        'DeepGravitySources_Worms_USCanada',
        'ShallowGravitySources_Worms_USCanada',
        'DeepMagSources_Worms_USCanada',
    ]

    dirnames = ['GeophysicsGravity', 'GeophysicsMag']

    for d in dirnames:
        f = os.path.join(root, 'USGS002', d)
        for t in tifs:
            # print(os.path.join(f, t))
            if os.path.isdir(os.path.join(f, t)):
                p = os.path.join(f, t, f'{t}.tif')
                if not os.path.isfile(p):
                    p = os.path.join(f, t, 'USCanadaMagRTP_DeepSources.tif')
                    if not os.path.isfile(p):
                        p = os.path.join(f, t, 'USCanadaMagRTP_HGMDeepSources.tif')

                mf = modify_filename(t)
                output_path = os.path.join(f, t, f'{mf}NM.tif')
                print(f'tif output path = {output_path}')
                with rasterio.open(p) as src:
                    bbox = box(lonmin, latmin, lonmax, latmax)
                    geoms = [bbox]
                    out_image, out_transform = mask(src, geoms, crop=True)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        'driver': 'GTiff',
                        'count': out_image.shape[0],
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        'transform': out_transform
                    })

                    with rasterio.open(output_path, 'w', **out_meta) as dest:
                        dest.write(out_image)
                        rdir = os.path.join(f, t)
                        rname = f'{mf}NM'
                        # copy_raster(in_raster_dir=rdir, in_raster_name=rname)
        for s in shps:
            print(f'shapefile = {s}')
            if os.path.isdir(os.path.join(f, s)):
                p = os.path.join(f, s, f'{s}.shp')
                print(f'shapefile = {p}')
                output_path = os.path.join(f, s, f'{s}_NM.shp')
                gdf = gpd.read_file(p)
                nmgdf = gpd.clip(gdf, nm_extent_wgs84)
                nmgdf.to_file(output_path)
                fc = os.path.join(f, s, f'{s}_NM')

                out_gdb = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb\USGS002'
                fc_to_gdb(in_features=fc, out_gdb=out_gdb)


def fc_to_gdb(in_features, out_gdb=None):
    ap.conversion.FeatureClassToGeodatabase(Input_Features=in_features, Output_Geodatabase=out_gdb)


def stanford_model_data():
    print('importing Stanford thermal model inputs/outputs')
    data = pd.read_csv(r'stanford_thermal_model_inputs_outputs_COMPLETE_VERSION2.csv')
    df = pd.DataFrame(data)

    print('setting geometry column')
    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = gdf['geometry'].apply(loads)
    gdf.set_geometry('geometry', crs='EPSG:3857')

    xmin = -12216837
    xmax = -11350902
    ymin = 3595190
    ymax = 4523656

    nm_extent = [xmin, ymin, xmax, ymax]

    print('clipping to NM extent')
    nmgdf = gpd.clip(gdf, (xmin, ymin, xmax, ymax))

    print('saving to shapefile')
    nmgdf.to_file('nm_stanfordmodel.shp')


def stanford_bht_data(nm_extent, root):
    print('importing Stanford thermal model BHT data')

    data = pd.read_csv('Raw_BHT_aggregated_data.csv')
    df = pd.DataFrame(data)

    print('setting geometry column')
    gdf = gpd.GeoDataFrame(df)
    gdf.geometry = gdf['geometry'].apply(loads)
    gdf.set_geometry('geometry', crs='EPSG:4326')

    print(gdf.geom_type)

    print('clipping to NM extent')
    nmgdf = gdf.clip(nm_extent)

    print('saving to shapefile')
    name = 'nm_stanfordmodel_bht.shp'
    # nmgdf.to_file(name)
    # nmgdf.to_file(os.path.join(root, name))


def kml_to_gdf():
    kmlfolders_pref = 'NGS_GRAV-D_Data_Block'
    kmlfiles_pref = 'NGS_GRAVD_Block'

    sfs = ['CS07_Extent_2', 'CS07_lines_6', 'MS01_Extent', 'MS01_lines', 'MS02_Extent_2', 'MS02_lines_6', 'MS04_Extent',
           'MS04_lines', 'MS05_Extent', 'MS05_lines']
    for s in sfs:
        kmlfolder = f'{kmlfolders_pref}_{s[:4]}'
        kmlname = f'{kmlfiles_pref}_{s}'
        kmlpath = os.path.join(root, 'NOAA001', kmlfolder)
        print(kmlpath)

        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'  # Enable KML support in fiona

        kml = os.path.join(kmlpath, f'{kmlname}.kml')
        with fiona.open(kml) as collection:
            gdf = gpd.GeoDataFrame.from_features(collection)

        gdf.to_file(rf'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\NOAA001\{kmlname}.shp')

def noaa_nm_data():
    # "C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\NOAA002\newmex.xyz"
    name = 'newmex'
    infile = os.path.join(root, 'NOAA002', f'{name}.ast.txt')
    outfile = os.path.join(root, 'NOAA002', name)

    df = pd.read_fwf(infile, header=None)
    # print(df)
    # sys.exit()
    df.columns = ['station_id', 'latitude_deg', 'latitude_dm', 'longitude_deg', 'longitude_dm', 'sea_level_elev_ft',
                  'obs_grav', 'Free_air_anom', 'Bouguer_anom_267', 'terr_corr_outer', 'terr_corr_inner']

    # convert lat/long decimal minutes to decimal degrees
    df['latitude'] = df['latitude_deg'] + (df['latitude_dm'] / 60)
    df['longitude'] = -(np.abs(df['longitude_deg']) + (df['longitude_dm'] / 60))

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

    # print('setting geometry column')
    # gdf.geometry = gdf['geometry'].apply(loads)
    # gdf.set_geometry('geometry', crs='EPSG:4326')

    print(gdf.geom_type)

    # print('clipping to NM extent')
    # nmgdf = gdf.clip(nm_extent_wgs84)

    print('saving')
    gdf.to_file(outfile)
    # nmgdf.to_file(os.path.join(root, name))


def usgs_kml():
    kmlpath = os.path.join(root, 'USGS011')
    kmlname = 'footprint_children'

    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'  # Enable KML support in fiona

    kml = os.path.join(kmlpath, f'{kmlname}.kml')
    with fiona.open(kml) as collection:
        gdf = gpd.GeoDataFrame.from_features(collection)

    gdf.to_file(
        rf'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\USGS011\{kmlname}.shp')

def paces(in_path, out_path):
    df = pd.read_csv(in_path, header=None)
    df.columns = ['ID', 'long', 'lat', 'elevation', 'date-time', 'absgrav']

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat), crs='EPSG:4326')

    # Write the GeoDataFrame to a Shapefile
    gdf.to_file(out_path)
    print(f"Converted {in_path} to {out_path}")

def join_paces_shps(pdir):
    gdfs_to_merge = []

    for i in range(32, 39):
        for j in range(103, 111):
            name = f'N{i}W{j}.shp'
            p = os.path.join(pdir, name)
            gdf = gpd.read_file(p)
            gdfs_to_merge.append(gdf)

    if gdfs_to_merge:
        # Merge all the GeoDataFrames
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs_to_merge, ignore_index=True))

        # Ensure the CRS is consistent
        merged_gdf.set_crs(gdfs_to_merge[0].crs, allow_override=True, inplace=True)

        # Save the merged shapefile
        output_shapefile = os.path.join(pdir, 'pacesmagNM.shp')
        merged_gdf.to_file(output_shapefile)
        print(f"Shapefiles merged into {output_shapefile}")
    else:
        print(f"No shapefiles matching the condition found.")


# Function to go through subdirectories and convert CSV to Shapefile
def convert_csvs_in_directory(parent_directory):
    # Walk through the directory
    for subdir, _, files in os.walk(parent_directory):
        # Look for CSV files in each subdirectory
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                shapefile_name = os.path.splitext(file)[0] + '.shp'
                shapefile_path = os.path.join(parent_directory, shapefile_name)

                # Convert CSV to Shapefile
                paces(csv_path, shapefile_path)


def deal_with_utah_temp_data():

    # Define the Excel file path
    path = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData'\
           r'\NMGeophysicalData\UnivUtah001'
    name = 'Tularosa_PFA_Phase2__temperature_logs.xlsx'
    excel_file = os.path.join(path, name)

    # Create a list to store the data for each sheet
    data = []

    # Read the Excel file
    xls = pd.ExcelFile(excel_file)

    # Loop through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the current sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        print(df)

        # Extract the four specific cell values (change the row and column indices as needed)
        # Example: row 1, column 'A' (adjust indices according to your Excel structure)
        value1 = df.iloc[3, 1]  # Change as per your desired cell
        value2 = df.iloc[6, 1]  # Change as per your desired cell
        value3 = df.iloc[7, 1]  # Change as per your desired cell
        value4 = df.iloc[8, 1]  # Change as per your desired cell
        print(f'values = {value1, value2, value3, value4}'
        )

        # Extract latitude and longitude or other coordinates if available
        # Assuming value3 and value4 are lat, lon for creating points
        easting = value2
        northing = value3

        # Create a geometry point from lat, lon
        point = Point(easting, northing)

        # Add data to the list as a dictionary
        data.append({
            'Well': value1,  # or whatever your first value corresponds to
            'Easting': value2,
            'Northing': value3,
            'Elevation_m': value4,  # second value
            'geometry': point,
        })

    # Create a GeoDataFrame from the data
    gdf = gpd.GeoDataFrame(data)

    # Define the coordinate reference system (CRS), for example, WGS84 (EPSG:4326)
    gdf.set_crs("EPSG:26913", allow_override=True, inplace=True)

    # Save to a shapefile
    output_shapefile = "Tularosa_PFA_Phase2__temperature_logs.shp"
    gdf.to_file(os.path.join(path, output_shapefile))
    fc_to_gdb(os.path.join(path, output_shapefile), out_gdb=os.path.join(default_gdb, 'UnivUtah001'))

    print(f"Shapefile created: {output_shapefile}")

def tgs_data():
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


    # Load the Excel file
    file_path = os.path.join(root, r'TGS\TGSdata_noSJorDB.xlsx')
    excel_file = pd.ExcelFile(file_path)

    # Create a dictionary to hold each sheet as a DataFrame
    sheets_data = {}

    # Iterate over all sheet names and load each sheet into a DataFrame
    for sheet_name in excel_file.sheet_names:
        sheets_data[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

    df1 = sheets_data['Export Well Results']

    # need to do Digital Curves (ARLAS) and Digital Mud Logs (LAS) next, fix script to be better first
    df2 = sheets_data['Digital Curves (LAS+)']
    columns_to_keep = ['UWI', 'Surface Lat', 'Surface Long', 'Top Depth', 'Bottom Depth', 'Product Type Name', 'Data Available', 'Description']
    merged_df = df1.merge(df2, on='UWI', how='inner')
    joined_df = merged_df[columns_to_keep]

    gdf = gpd.GeoDataFrame(joined_df, geometry=gpd.points_from_xy(joined_df['Surface Long'], joined_df['Surface Lat']))
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

    unique_vals = gdf['Data Available'].unique()
    print(f'unique values in digital curves (LAS+) sheet = {unique_vals.shape}')

    # start by combining all porosity logs (or logs that are used to measure porosity)
    # keywords: density, neutron, sonic, gamma ray, acoustic
    keywords_por = ['density', 'neutron', 'sonic', 'gamma ray', 'acoustic']
    pattern = '|'.join(keywords_por)
    poro_df = gdf[gdf['Description'].str.contains(pattern, case=False, na=False)]
    poro_df.to_file(os.path.join(root, 'TGS\TGSproducts_LAS+_poro.shp'))

    # combine all permeability logs
    # keywords: resitivity, induction, laterolog, microlog, spontaneous potential
    keywords_perm = ['resistivity', 'induction', 'laterolog', 'microlog', 'spontaneous potential']
    pattern = '|'.join(keywords_perm)
    perm_df = gdf[gdf['Description'].str.contains(pattern, case=False, na=False)]
    perm_df.to_file(os.path.join(root, 'TGS\TGSproducts_LAS+_perm.shp'))

    # now take group that doesn't contain either
    pattern = '|'.join(keywords_perm + keywords_por)
    other_df = gdf[~gdf['Description'].str.contains(pattern, case=False, na=False)]
    other_df.to_file(os.path.join(root, 'TGS\TGSproducts_LAS+_other.shp'))

    sys.exit()

    # Now sheets_data contains all the sheets in the Excel file as DataFrames
    # Example of accessing the first sheet as a DataFrame
    # df = sheets_data[excel_file.sheet_names[0]]

    # Create geodataframe from Export Product Results sheet - contains all products
    df = sheets_data['Export Product Results']
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Surface Long'], df['Surface Lat']))
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

    # Group by top depth <= 2,000 ft., save to shapefile
    threshold = 2000
    group1 = gdf[gdf['Top Depth'] <= threshold]
    group2 = gdf[gdf['Top Depth'] > threshold]
    group1.to_file(os.path.join(root, 'TGS\TGSproducts_TopDepthLT2kft.shp'))
    group2.to_file(os.path.join(root, 'TGS\TGSproducts_TopDepthGT2kft.shp'))

    # Graph the number of products for each group
    gdfs = [group1, group2]
    labels = ['TGS Wells, Top Depth <= 2,000 ft.', 'TGS Wells, Top Depth > 2,000 ft.']


    for g, l in zip(gdfs, labels):
        category_counts = g['Product Type Name'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        plt.figure()
        ax = sns.barplot(data=category_counts, x='Category', y='Count')
        plt.title(f'{l}')
        plt.xlabel('Category')
        plt.ylabel('Count')

        # Add the count values on top of the bars
        for p in ax.patches:
            # Get the height of each bar
            height = p.get_height()

            # Add text to the bar (centered at the middle of each bar)
            ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  # Slightly above the bar
                    str(int(height)),  # Convert height to integer for count value
                    ha='center',  # Horizontal alignment (centered)
                    va='bottom',  # Vertical alignment (above the bar)
                    fontsize=12,  # Font size of the count
                    color='black')  # Text color

        plt.xticks()
        plt.tight_layout()
        # plt.show()

        total_count = category_counts['Count'].sum()
        print(f'{total_count=}')

    # now group by product type

    labels =['LT2kft', 'GT2kft']
    for l, g in zip(labels, gdfs):
        unique_vals = g['Product Type Name'].unique()
        for v in unique_vals:
            pt_gdf = g[g['Product Type Name'] == v]
            name = re.sub(r'[ ()]', '', v)
            pt_gdf.to_file(os.path.join(root, f'TGS\TGSproducts_{name}{l}.shp'))

    plt.show()

    # Print all sheet names and their corresponding DataFrame shape
    for sheet, df in sheets_data.items():
        print(f"Sheet name: {sheet}, DataFrame shape: {df.shape}")



def main():
    tgs_data()




if __name__ == '__main__':
    main()