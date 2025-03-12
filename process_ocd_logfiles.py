import pandas as pd
import geopandas as gpd
from geopandas import points_from_xy
import sys
import os


def main():
    # read in original NMOCD wells shapefile
    root = r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData\NMOCD'
    gdf = gpd.read_file(os.path.join(root, 'New_Mexico_Oil_and_Gas_Wells.shp'))

    # read in file that lists NMOCD wells which contain log files
    df = pd.read_csv(os.path.join('out', 'NMOCD_wellswithlogs.csv'))
    print(df.shape)
    wells_with_logs = pd.DataFrame(df['API'].unique())
    print(wells_with_logs.shape)
    sys.exit()

    print(wells_with_logs)
    print(f'number of unique exp db_files that contain log files: {wells_with_logs.shape}')

    # join wells_with_logs to df on column db_file - keep only records contained in wells_with_logs
    merged_df = pd.merge(df, wells_with_logs, on='db_file', how='inner')
    print(merged_df)
    print(f'{merged_df.shape}')

    # save to csv and shapefile
    merged_df.to_csv(r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization\SubsurfaceData\OSE\OSE_EXP_withlogfiles.csv')

    gdf = gpd.GeoDataFrame(merged_df, geometry=points_from_xy(merged_df.easting, merged_df.northing), crs='EPSG:26913')
    gdf.to_file(r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\AquiferCharacterization\SubsurfaceData\OSE\OSE_EXP_withlogfiles.shp')


if __name__ == '__main__':
    main()