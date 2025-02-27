import arcpy
import pandas as pd
import re

# Set the workspace to the geodatabase
gdb_path = r"W:\regional\pecos_slope\ps_data\PecosSlopeModel\__FINAL_MODEL\PecosSlope_3D_ModelInput.gdb\ControlPoints"  # Update this with your actual geodatabase path
arcpy.env.workspace = gdb_path

# Get list of feature classes
feature_classes = [fc for fc in arcpy.ListFeatureClasses() if fc.lower().endswith("allbasepts")]


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
