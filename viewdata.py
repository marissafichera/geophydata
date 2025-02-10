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
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import arcpy as ap


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

nm_extent_wgs84 = (lonmin, latmin, lonmax, latmax)

def copy_raster(in_raster):
    out_gdb = r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb'
    out_path = os.path.join(out_gdb, f'{in_raster}')
    ap.management.CopyRaster(f'{in_raster}.tif', out_path)


def export_features(in_features):
    out_gdb = r'C:\Users\mfichera\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb'
    out_path = os.path.join(out_gdb, in_features)
    ap.conversion.ExportFeatures(in_features=f'{in_features}.shp', out_features=out_path)


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
            print(os.path.join(f, t))
            if os.path.isdir(os.path.join(f, t)):
                p = os.path.join(f, t, f'{t}.tif')
                if not os.path.isfile(p):
                    p = os.path.join(f, t, 'USCanadaMagRTP_DeepSources.tif')
                    if not os.path.isfile(p):
                        p = os.path.join(f, t, 'USCanadaMagRTP_HGMDeepSources.tif')

                output_path = os.path.join(f, t, f'{t}_NM.tif')
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
                        r = os.path.join(f, t, f'{t}_NM')
                        copy_raster(in_raster=output_path)
        for s in shps:
            print(f'shapefile = {s}')
            if os.path.isdir(os.path.join(f, s)):
                p = os.path.join(f, s, f'{s}.shp')
                print(f'shapefile = {p}')
                output_path = os.path.join(f, s, f'{s}_NM.shp')
                gdf = gpd.read_file(p)
                nmgdf = gpd.clip(gdf, nm_extent_wgs84)
                nmgdf.to_file(output_path)
                fc = os.path.join(f, s, f'{s}')
                fc_to_fc(in_features=fc)


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
    clip_usgs002_data()








if __name__ == '__main__':
    main()