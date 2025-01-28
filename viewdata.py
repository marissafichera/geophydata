from idlelib.pyparse import trans

import numpy as np
import os
import pandas as pd
import geopandas as gpd
# import matplotlib.pyplot as plt
from shapely.wkt import loads
import pyproj
import sys
import fiona

root = r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData'

xmin = -12216837
xmax = -11350902
ymin = 3595190
ymax = 4523656

# Create transformer objects
transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Convert to WGS84
lonmin, latmin = transformer.transform(xmin, ymin)
lonmax, latmax = transformer.transform(xmax, ymax)

nm_extent_wgs84 = [lonmin, latmin, lonmax, latmax]


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


def main():
    df = pd.read_csv(r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\USGS015\5f2978e682cef313ed9e82aa\CannonAFB_ERT_Processed_zip\ERT_Locational_Data.txt',
                       sep='\t')
    print(df)
    df.columns = ['Electrode', 'Northing', 'Easting', 'Elevation_m']
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs='EPSG:32614')
    gdf.to_file(r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\USGS015\5f2978e682cef313ed9e82aa\CannonAFB_ERT_Processed_zip\ERT_Locational_Data.shp')








if __name__ == '__main__':
    main()