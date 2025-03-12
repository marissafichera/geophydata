import arcpy
import pandas as pd
import re
import os

# Set the workspace to the geodatabase
gdb_path = r"W:\regional\pecos_slope\ps_data\PecosSlopeModel\__FINAL_MODEL\PecosSlope_3D_ModelInput.gdb\ControlPoints"  # Update this with your actual geodatabase path
arcpy.env.workspace = gdb_path


def summarize_datasources(feature_classes):
    # List to store final results
    summary_data = []
    data_source_ids = set()  # Using a set to ensure uniqueness


    # Process each feature class
    for fc in feature_classes:
        print(f'{fc=}')
        # Convert feature class to pandas DataFrame
        fields = ["MUEBase", "DataSourceID", "UniqueID"]

        # Read data into a list of dictionaries
        data = [row for row in arcpy.da.SearchCursor(fc, fields)]

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=fields)

        # Remove rows where "MUEbase" is NULL
        df = df.dropna(subset=["MUEBase"])


        # Function to clean UniqueID by removing numbers only after "-"
        def clean_unique_id(uid):
            if pd.notna(uid):
                return re.sub(r'-\d+', '', uid)  # Removes numbers following "-"
            return None


        df["DataSourceID"] = df.apply(
            lambda row: clean_unique_id(row["UniqueID"]) if pd.isna(row["DataSourceID"]) else row["DataSourceID"], axis=1)

        # Get unique "DataSourceID" values and store each in its own row
        unique_data_sources = df["DataSourceID"].dropna().unique()
        for ds_id in unique_data_sources:
            summary_data.append({"FeatureClass": fc, "DataSourceID": ds_id})
            data_source_ids.add(ds_id)  # Collect unique DataSourceIDs



    # Create final summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Create a new DataFrame with unique DataSourceIDs
    unique_ds_df = pd.DataFrame(sorted(data_source_ids), columns=["DataSourceID"])

    # Save both DataFrames to CSV
    summary_df.to_csv("feature_class_summary.csv", index=False)
    unique_ds_df.to_csv("unique_datasource_ids.csv", index=False)

    # Print results
    print(summary_df)
    print("\nUnique DataSourceIDs:")
    print(unique_ds_df)

    print("Processing complete. Data saved to 'feature_class_summary.csv' and 'unique_datasource_ids.csv'")

    # Print result
    print(summary_df)

    # Save result to CSV
    summary_df.to_csv(r"C:\Users\mfichera\OneDrive - nmt.edu\Documents\PecosSlope\modelinput_datasources.csv", index=False)

    print("Processing complete. Data saved to modelinput_datasources.csv")


def add_coordinates_to_csv(feature_class, df):
    # first read in feature class to get coordinates
    # Spatial reference (modify as needed)
    spatial_ref = arcpy.Describe(feature_class).spatialReference
    print(f"Feature class is in spatial reference: {spatial_ref.name}")

    # Ensure it's a projected coordinate system (for UTM-based Easting/Northing)
    if not spatial_ref.PCSCode:
        raise ValueError(
            "Feature class is in a geographic coordinate system (lat/lon). Reproject it to a projected system (e.g., UTM) for correct Easting/Northing values.")


    # Open a cursor to read coordinates
    with arcpy.da.SearchCursor(feature_class, ["OID@", "SHAPE@XY"]) as cursor:
        data = []
        for row in cursor:
            oid = row[0]
            easting, northing = row[1]  # X = Easting, Y = Northing
            data.append((oid, easting, northing))

    # Convert coordinates and oid data to DataFrame
    df_coords = pd.DataFrame(data, columns=["OBJECTID", "Easting_dfc", "Northing_dfc"])

    # fill in empty coordinate values on original dataframe
    merged_df = df.merge(df_coords, on='OBJECTID')
    if 'Easting' in df.columns:
        merged_df['Easting'].fillna(merged_df['Easting_dfc'], inplace=True)
        merged_df['Northing'].fillna(merged_df['Northing_dfc'], inplace=True)
        merged_df['Units'].fillna('meters', inplace=True)
        merged_df['Datum'].fillna('NAD83', inplace=True)
    else:
        merged_df['Easting'] = merged_df['Easting_dfc']
        merged_df['Northing'] = merged_df['Northing_dfc']
        merged_df['Units'] = 'meters'
        merged_df['Datum'] = 'NAD83'


    df_fin = merged_df.drop(['Easting_dfc', 'Northing_dfc', 'OID@'], axis=1)
    df_fin = df_fin.rename(columns={'Shape': 'Geometry', 'MUEBase': 'MUEBase_ft'})

    return df_fin


def clean_export_modelinput_data(fc):
    print(f'{fc=}')
    output_csv = rf'ps_csvs\{fc}.csv'

    # Get all field names, including OID
    fields = [f.name for f in arcpy.ListFields(fc)]  # Extract all field names
    fields.insert(0, "OID@")  # Ensure OID is included

    # Read feature class attributes
    data = []
    with arcpy.da.SearchCursor(fc, fields) as cursor:
        for row in cursor:
            data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=fields)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    # arcpy.conversion.ExportTable(fc, output_csv)

    df = pd.read_csv(output_csv)
    df = df.dropna(subset=["MUEBase"])

    # Function to clean UniqueID by removing numbers only after "-"
    def clean_unique_id(uid):
        if pd.notna(uid):
            return re.sub(r'-\d+', '', uid)  # Removes numbers following "-"
        return None

    # fill in empty/missing DataSourceID columns
    df["DataSourceID"] = df.apply(
        lambda row: clean_unique_id(row["UniqueID"]) if pd.isna(row["DataSourceID"]) else row["DataSourceID"],
        axis=1)

    return df


def write_excel_doc(df, fc, output_excel):
        if not os.path.exists(output_excel):
            # export cleaned data to an excel document with sheet name = fc
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=f'{fc}', index=False)
        else:
            with pd.ExcelWriter(output_excel, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=f'{fc}', index=False)


def main():
    # Get list of feature classes
    feature_classes = [fc for fc in arcpy.ListFeatureClasses() if fc.lower().endswith("allbasepts")]
    for feature_class in feature_classes:
        df = clean_export_modelinput_data(feature_class)
        df2 = add_coordinates_to_csv(feature_class, df)
        write_excel_doc(df2, feature_class, output_excel='PecosSlope_ModelInputData_2025.xlsx')


if __name__ == '__main__':
    main()
