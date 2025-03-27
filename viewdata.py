from idlelib.pyparse import trans

import numpy as np
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import points_from_xy
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
import random
# import arcpy as ap
import requests
import json
import os

root = r'\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData'
default_gdb = r'\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb'
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

def geojson_to_shapefile():
    url_ids = [
        '1054',
        '5059',
        '5013',
        '20038',
        '40001',
        '40011',
        '1034',
        '20033',
        '1054',
        '5027',
        '20047',
        '1014',
        '5029',
        '50001',
    ]
    for url_id in url_ids:
        geojson_url = rf'https://mrdata.usgs.gov/earthmri/data-acquisition/project/{url_id}/json'

        # Fetch the GeoJSON data from the URL
        response = requests.get(geojson_url)
        if response.status_code != 200:
            print(f"Failed to retrieve GeoJSON from {geojson_url}")
            return

        geojson_data = response.json()

        # Convert GeoJSON to GeoDataFrame using geopandas
        gdf = gpd.read_file(geojson_url)

        # Write the GeoDataFrame to a shapefile
        shapefile_path = os.path.join('out', 'EarthMRI', f'EarthMRI_{url_id}.shp')
        print(f'{shapefile_path=}')
        gdf.to_file(shapefile_path)
        print(f"Shapefile saved at: {shapefile_path}")

        # Write the entire GeoJSON data to a text file
        geojson_file_path = os.path.join('out', 'EarthMRI', f'EarthMRI_{url_id}.json')
        with open(geojson_file_path, 'w') as geojson_file:
            json.dump(geojson_data, geojson_file, indent=4)

        print(f"GeoJSON saved to: {geojson_file_path}")


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


def sort_tgs_logdata(df, df2, name, label):
    print(f'{df=}')
    print(f'{df2.columns=}')

    if name == 'Export Product Results':
        # extract all the raster and image log data from Export Product Results
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df['Surface Long'], y=df['Surface Lat']), crs='EPSG:4326')
        gdf_rasters = gdf[(gdf['Product Type Name'] == 'SmartRASTER') | (gdf['Product Type Name'] == 'Log Image')]

        # screen TGS data, create separate files for each log type
        keywords = ['caliper',
                    'spontaneous potential',
                    'density', 'neutron',
                    'gamma ray',
                    'resistivity',
                    'conductivity',
                    'microlog']

        for keyword in keywords:
            log_df = gdf_rasters[gdf_rasters['Data Available'].str.contains(keyword, case=False, na=False)]
            log_df.to_file(os.path.join(root, f'TGS\TGS{label}{keyword}.shp'))

        return

    if name != 'Export Product Results':
        df1 = df[['UWI', 'Surface Lat', 'Surface Long']].drop_duplicates()
        # join Export Well Results sheet to other sheet on UWI to get locations
        columns_to_keep = ['UWI', 'Surface Lat', 'Surface Long', 'Top Depth', 'Bottom Depth', 'Product Type Name',
                           'Data Available', 'Description']
        # suffixes = ('_df1', '_df2')
        merged_df = df1.merge(df2, on='UWI', how='inner')
        print(f'{merged_df.columns=}')
        joined_df = merged_df[columns_to_keep]

        # create geodataframe from joined dataframe
        gdf = gpd.GeoDataFrame(joined_df, geometry=gpd.points_from_xy(joined_df['Surface Long'], joined_df['Surface Lat']))
        gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

        # find number of unique values in Data Available column
        # unique_vals = gdf['Data Available'].unique()
        # print(f'unique values in digital curves (LAS+) sheet = {unique_vals.shape}')

        # screen TGS data, create separate files for each log type
        keywords = ['caliper',
                    'spontaneous potential',
                    'density', 'neutron',
                    'gamma ray',
                    'resistivity',
                    'conductivity',
                    'microlog']

        for keyword in keywords:
            log_df = gdf[gdf['Description'].str.contains(keyword, case=False, na=False)]
            log_df.to_file(os.path.join(root, f'TGS\TGS{label}{keyword}.shp'))

        # # start by combining all porosity logs (or logs that are used to measure porosity)
        # # keywords: density, neutron, sonic, gamma ray, acoustic
        # keywords_por = ['density', 'neutron', 'sonic', 'gamma ray', 'acoustic']
        # pattern = '|'.join(keywords_por)
        # poro_df = gdf[gdf['Description'].str.contains(pattern, case=False, na=False)]
        # poro_df.to_file(os.path.join(root, f'TGS\TGS_{label}_poro.shp'))
        #
        # # combine all permeability logs
        # # keywords: resitivity, induction, laterolog, microlog, spontaneous potential
        # keywords_perm = ['resistivity', 'induction', 'laterolog', 'microlog', 'spontaneous potential']
        # pattern = '|'.join(keywords_perm)
        # perm_df = gdf[gdf['Description'].str.contains(pattern, case=False, na=False)]
        # perm_df.to_file(os.path.join(root, f'TGS\TGS_{label}_perm.shp'))
        #
        # # now take group that doesn't contain either
        # pattern = '|'.join(keywords_perm + keywords_por)
        # other_df = gdf[~gdf['Description'].str.contains(pattern, case=False, na=False)]
        # other_df.to_file(os.path.join(root, f'TGS\TGS_{label}_other.shp'))
        #
        # gdfs = [poro_df,
        #         perm_df,
        #         other_df]
        # dfnames = ['poro', 'perm', 'other']
        #
        # for gdf, n in zip(gdfs, dfnames):
        #     # Group by top depth <= 2,000 ft., save to shapefile
        #     threshold = 2000
        #     group1 = gdf[gdf['Top Depth'] <= threshold]
        #     group2 = gdf[gdf['Top Depth'] > threshold]
        #     group1.to_file(os.path.join(root, f'TGS\TGS_{n}{label}TopDepthLT2kft.shp'))
        #     group2.to_file(os.path.join(root, f'TGS\TGS_{n}{label}TopDepthGT2kft.shp'))


def tgs_data():
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


    # Load the Excel file
    file_path = os.path.join(root, r'TGS\TGSdata_noSJorDB.xlsx')
    excel_file = pd.ExcelFile(file_path)

    # Create a dictionary to hold each sheet as a DataFrame
    # sheets_data = {}
    labels = ['rasters', 'LAS', 'LAS+', 'ARLAS', 'MudLAS']

    df1 = pd.read_excel(excel_file, sheet_name='Export Product Results')
    gdf = gpd.GeoDataFrame(df1, geometry=points_from_xy(x=df1['Surface Long'], y=df1['Surface Lat']), crs='EPSG:4326')

    unique_vals = gdf['Product Type Name'].unique()
    for v in unique_vals:
        pt_gdf = gdf[gdf['Product Type Name'] == v]
        name = re.sub(r'[ ()]', '', v)
        pt_gdf.to_file(os.path.join(root, f'TGS\TGSproducts_{name}.shp'))

    sys.exit()

    # Create geodataframe from Export Product Results sheet - contains all products
    # gdf = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1['Surface Long'], df1['Surface Lat']))
    # gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
    # gdf.to_file(os.path.join(root, 'TGS\TGSProducts_NoSJorDB_all.shp'))
    gdf = gpd.read_file(os.path.join(root, 'TGS\TGSProducts_NoSJorDB_all.shp'))

    # Iterate over all sheet names and load each sheet into a DataFrame, then organize digital log data
    for l, sheet_name in zip(labels, excel_file.sheet_names[1:]):
        sheetdata = pd.read_excel(excel_file, sheet_name=sheet_name)
        # sheetdata.to_csv(os.path.join(root, f'TGS\TGSnoSJDB_{sheet_name}.csv'.replace(' ','')))
        sort_tgs_logdata(df=df1, df2=sheetdata, name=sheet_name, label=l)


    # # Group by top depth <= 2,000 ft., save to shapefile
    # threshold = 2000
    # group1 = gdf[gdf['Top Depth'] <= threshold]
    # group2 = gdf[gdf['Top Depth'] > threshold]
    # group1.to_file(os.path.join(root, 'TGS\TGSproducts_TopDepthLT2kft.shp'))
    # group2.to_file(os.path.join(root, 'TGS\TGSproducts_TopDepthGT2kft.shp'))
    #
    # # Graph the number of products for each group
    # gdfs = [group1, group2]
    # labels = ['TGS Wells, Top Depth <= 2,000 ft.', 'TGS Wells, Top Depth > 2,000 ft.']
    #
    #
    # for g, l in zip(gdfs, labels):
    #     category_counts = g['Product Type Name'].value_counts().reset_index()
    #     category_counts.columns = ['Category', 'Count']
    #     plt.figure()
    #     ax = sns.barplot(data=category_counts, x='Category', y='Count')
    #     plt.title(f'{l}')
    #     plt.xlabel('Category')
    #     plt.ylabel('Count')
    #
    #     # Add the count values on top of the bars
    #     for p in ax.patches:
    #         # Get the height of each bar
    #         height = p.get_height()
    #
    #         # Add text to the bar (centered at the middle of each bar)
    #         ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  # Slightly above the bar
    #                 str(int(height)),  # Convert height to integer for count value
    #                 ha='center',  # Horizontal alignment (centered)
    #                 va='bottom',  # Vertical alignment (above the bar)
    #                 fontsize=12,  # Font size of the count
    #                 color='black')  # Text color
    #
    #     plt.xticks()
    #     plt.tight_layout()
    #     # plt.show()
    #
    #     total_count = category_counts['Count'].sum()
    #     print(f'{total_count=}')
    #
    # # now group by product type
    # labels =['LT2kft', 'GT2kft']
    # for l, g in zip(labels, gdfs):
    #     unique_vals = g['Product Type Name'].unique()
    #     for v in unique_vals:
    #         pt_gdf = g[g['Product Type Name'] == v]
    #         name = re.sub(r'[ ()]', '', v)
    #         pt_gdf.to_file(os.path.join(root, f'TGS\TGSproducts_{name}{l}.shp'))
    #
    # plt.show()


def ose_data():
    path = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization'
    df = pd.read_csv(os.path.join(path, 'OSE_PODS_cleaned.csv'))
    pd.set_option('display.max_columns', None)

    # get all the exploration wells and export to shapefile and csv
    exp_df = df[df['use_'] == 'EXP']
    ose_expwells = gpd.GeoDataFrame(exp_df, geometry=points_from_xy(exp_df.easting, exp_df.northing), crs='EPSG:26913')
    ose_expwells.set_crs('EPSG:26913')
    ose_expwells.to_csv(os.path.join(path, 'OSE_EXPwells.csv'))

    # export just the pod_file column to use in OSEFileRetrieval
    ose_expwells_podfile = ose_expwells['db_file']
    ose_expwells_podfile.to_csv(os.path.join(path, 'OSE_EXPwells_dbfile.csv'), index=False)

    # ose_expwells.to_file(os.path.join(path, 'OSE_EXPwells.shp'))
    # print(exp_df.columns.tolist())

    # now create stats for well use permits per groundwater basin
    df_bc = pd.read_csv(os.path.join(path, 'OSE_basincodes.csv'))
    rename_dict = dict(zip(df_bc['Code Value'], df_bc['Code Description']))

    print(df['use_'].unique())
    print(df['pod_basin'].unique())

    basins = df['pod_basin'].unique()
    use_stats_per_basin = {}

    for basin in basins:
        bdf = df[df['pod_basin'] == basin]
        category_counts = bdf['use_'].value_counts()
        use_stats_per_basin[basin] = category_counts
        # category_counts.columns = ['use', 'count']
        # category_counts['basin'] = basin
        # use_stats_per_basin[basin] = category_counts
        print(f'{category_counts=}')
        # plt.figure()
        # ax = sns.barplot(data=category_counts, x='Category', y='Count')
        # plt.title(f'{l}')
        # plt.xlabel('Category')
        # plt.ylabel('Count')
        #
        # # Add the count values on top of the bars
        # for p in ax.patches:
        #     # Get the height of each bar
        #     height = p.get_height()
        #
        #     # Add text to the bar (centered at the middle of each bar)
        #     ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  # Slightly above the bar
        #             str(int(height)),  # Convert height to integer for count value
        #             ha='center',  # Horizontal alignment (centered)
        #             va='bottom',  # Vertical alignment (above the bar)
        #             fontsize=12,  # Font size of the count
        #             color='black')  # Text color
        #
        # plt.xticks()
        # plt.tight_layout()
        # plt.show()

        # total_count = category_counts['count'].sum()
        # print(f'{total_count=}')
    final_df = pd.DataFrame(use_stats_per_basin).fillna(0)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'use'}, inplace=True)
    final_df.rename(columns=rename_dict, inplace=True)

    print(final_df.shape)
    print(df['use_'].unique().shape)
    print(basins.shape)

    final_df.to_csv(os.path.join(path, 'ose_usepermits_per_basin.csv'))
    # plot_ose_data(final_df)


def plot_ose_data(df):
    df = df.drop(index=0).reset_index(drop=True)
    use_labels = df['use_']  # Extract the labels from the first column
    # df = df[['use_', 'Bluewater']]
    df_numeric = df.drop(columns=['use_'])  # Drop the first column for plotting

    print(df_numeric)

    # Assuming df_data is your dataframe after renaming
    # fig, axes = plt.subplots(nrows=5, ncols=9, figsize=(20, 20))  # Adjust grid size as needed
    # axes = axes.flatten()  # Flatten axes array for easy iteration

    num_charts = len(df_numeric.columns)  # Total number of bar charts
    charts_per_fig = 10  # Number of bar charts per figure
    num_figs = int(np.ceil(num_charts / charts_per_fig))  # Number of figures needed

    for fig_idx in range(num_figs):
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))  # 5x2 grid per figure
        axes = axes.flatten()  # Flatten axes for easy iteration

        start_idx = fig_idx * charts_per_fig
        end_idx = min(start_idx + charts_per_fig, num_charts)

        for i, column in enumerate(df_numeric.columns[start_idx:end_idx]):
            axes[i].bar(use_labels, df_numeric[column], color='skyblue')
            axes[i].set_title(column, fontsize=12, pad=10)
            axes[i].tick_params(axis='x', rotation=90, labelsize=8)  # Rotate x-axis labels for readability labels for readability


    # n = 60
    # colors = plt.cm.nipy_spectral(np.linspace(0, 1, n))
    # np.random.shuffle(colors)
    #
    # for i, column in enumerate(df_numeric.columns):
    #     wedges, texts = axes[i].pie(
    #         df_numeric[column],
    #         # autopct='%1.1f%%',
    #         startangle=90,
    #         colors=colors,
    #         labels=None  # Hide labels inside pie to avoid clutter
    #     )
    #     axes[i].set_title(column)
    #
    # # Add a global legend
    # fig.legend(use_labels, loc='lower center')

    # Hide any unused subplots (if columns < total axes)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)  # Adjust hspace and figure boundaries

        # plt.tight_layout()

        # Save the figure as a PNG
        fig.savefig(f'bar_chart_figure_{fig_idx + 1}.png', dpi=300)  # Save with high dpi

        plt.close(fig)  # Close the figure after saving to free up memory

        # plt.show()


def dms_to_decimal(dms_str):
    """Convert degrees:minutes:seconds format to decimal degrees."""
    match = re.match(r'(-?\d+):(\d+):(\d+\.\d+)', dms_str)
    if match:
        degrees, minutes, seconds = map(float, match.groups())
        decimal = abs(degrees) + minutes / 60 + seconds / 3600
        return -decimal if degrees < 0 else decimal
    return None


def parse_edi_files(folder_path=r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\UnivUtah003\EDI_files\Raw_Data'):
    """Read all .edi files in a folder and extract ID, LAT, LONG, ELEV_M."""
    data = []

    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.endswith(".edi"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

                # Extract values from specific lines
                lat_match = re.search(r'LAT=(-?\d+:\d+:\d+\.\d+)', lines[11])
                long_match = re.search(r'LONG=(-?\d+:\d+:\d+\.\d+)', lines[12])
                elev_match = re.search(r'ELEV=([\d\.]+)', lines[13])

                if lat_match and long_match and elev_match:
                    lat_dms = lat_match.group(1)
                    long_dms = long_match.group(1)
                    elev_m = float(elev_match.group(1))

                    # Convert to decimal degrees
                    lat = dms_to_decimal(lat_dms)
                    long = dms_to_decimal(long_dms)

                    # Extract ID from file name (remove .edi extension)
                    file_id = os.path.splitext(file_name)[0]

                    data.append([file_id, lat, long, elev_m])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['ID', 'LAT', 'LONG', 'ELEV_M'])
    gdf = gpd.GeoDataFrame(df, geometry=points_from_xy(x=df.LONG, y=df.LAT), crs='EPSG:4326')
    print(gdf)
    shpname = os.path.join(root, 'UnivUtah003', 'MTrawdata_locs.shp')
    gdf.to_file(shpname)

    save_script(output_script_folder=os.path.join(root, 'UnivUtah003'))


def save_script(output_script_folder, script_name='viewdata.txt'):
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


def nmocd():
    file_path = os.path.join(root, 'NMOCD', 'NMOCDWellLogs')
    df = pd.read_csv(f'{file_path}.csv')
    print(df)
    gdf = gpd.GeoDataFrame(df, geometry=points_from_xy(x=df.Longitude, y=df.Latitude), crs='EPSG:4326')
    gdf.to_file(f'{file_path}.shp')


def USGS012():
    # User inputs
    fcnames = ['Cornudas_Spec', 'Hueco_Spec', 'Middle_Spec', 'Sierra_Blanca_Spec']
    foldernames = ['Cornudas block', 'Hueco block', 'Middle block', 'Sierra Blanca block']
    for fcname, foldername in zip(fcnames, foldernames):
        csv_file = os.path.join(root, rf"USGS012\Aeroradiometric_data\Aeroradiometric_data\{foldername}\DATA\{fcname}.csv")  # Change to your CSV file path
        gdb_path = os.path.join(default_gdb)  # Change to your Geodatabase path
        spatial_reference = ap.SpatialReference(4326)  # WGS 84 (Change if needed)

        # Ensure the geodatabase exists
        if not ap.Exists(gdb_path):
            raise ValueError(f"Geodatabase does not exist: {gdb_path}")

        # Create a table from the CSV
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        table_path = os.path.join(gdb_path, f'{table_name}_tbl')

        if ap.Exists(table_path):
            print(f"Table {table_name} already exists in the geodatabase.")
        else:
            ap.conversion.TableToTable(csv_file, gdb_path, f'{table_name}_tbl')
            print(f"Table f'{table_name}_tbl' created successfully in {gdb_path}.")

        # Check if required fields exist
        fields = [f.name for f in ap.ListFields(table_path)]
        if 'H_LONG' in fields and 'H_LAT' in fields:
            x_field = 'H_LONG'
            y_field = 'H_LAT'
        elif 'LONG' in fields and 'LAT' in fields:
            x_field = 'LONG'
            y_field = ('LAT')
        else:
            raise ValueError("Can't find longitude and latitude column headers, check table")


        # Convert table to feature class
        feature_class_path = os.path.join(gdb_path, fcname)

        if ap.Exists(feature_class_path):
            print(f"Feature class {fcname} already exists. Overwriting...")
            ap.management.Delete(feature_class_path)


        ap.management.XYTableToPoint(
            table_path,
            feature_class_path,
            x_field,
            y_field,
            coordinate_system=spatial_reference
        )

        print(f"Feature class '{fcname}' created successfully in {gdb_path}.")


def csv_to_shapefile(gdf, output_folder):
    shapefile_name = "SSDB_WellHeader_locations.shp"
    gdf.to_file(os.path.join(output_folder, shapefile_name))

    shp = gpd.read_file(os.path.join(output_folder,shapefile_name))
    clip_points_to_boundary(input_fc=shp, output_fc=os.path.join(output_folder, 'SSDB_WellHeader_locs_cl.shp'))
    # Input CSV file
    # csv_file = r"C:\path\to\your\input.csv"
    #
    # # Output folder and shapefile name
    # output_folder = r"C:\path\to\output\folder"

def clip_points_to_boundary(input_fc,
                            output_fc,
                            clip_fc=r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\basemap\NM_dataextent_WGS.shp',
                            ):
    """
    Clips a point feature class to a polygon boundary.

    Parameters:
      input_fc (str): Path to the input points feature class
      clip_fc (str): Path to the polygon clip boundary
      output_fc (str): Path to save the clipped feature class
    """
    print(input_fc)
    sys.exit()
    print(ap.Describe(input_fc).spatialReference)
    print(ap.Describe(clip_fc).spatialReference)

    out_fc = os.path.join(r'W:\statewide\AquiferCharacterization\ArcGIS\Projects\NewMexWells\NewMexWells', output_fc)
    ap.analysis.Clip(in_features=input_fc, clip_features=clip_fc, out_feature_class=out_fc)
    print(f"📍 Clipped {os.path.basename(input_fc)} to boundary → {out_fc}")
    return output_fc


def main():

    tgs_data()

    output_script_folder = os.path.join(root, 'TGS')
    script_name = 'viewdata.txt'

    # USGS012()
    save_script(output_script_folder, script_name)


if __name__ == '__main__':
    main()