from arcgis.features import FeatureLayer
import geopandas as gpd
import os

# URL to the ArcGIS REST FeatureLayer
url = "https://services2.arcgis.com/qXZbWTdPDbTjl7Dy/ArcGIS/rest/services/OSE_Aquifer_Test_Wells_view_pub/FeatureServer/0"

# Query the layer
print("Querying feature layer...")
layer = FeatureLayer(url)
features = layer.query(where="1=1", out_fields="*", return_geometry=True)

# Convert to spatial dataframe
sdf = features.sdf
print(f"Retrieved {len(sdf)} records")

# Set output directory
output_dir = r"W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMHydrogeoData\NMHydrogeoData\OSE002h\OSE_Aquifer_Test_Wells_Export"
os.makedirs(output_dir, exist_ok=True)

# === Save to CSV ===
csv_path = os.path.join(output_dir, "OSE_Aquifer_Test_Wells.csv")
sdf.drop(columns="SHAPE").to_csv(csv_path, index=False)
print(f"✅ CSV saved to: {csv_path}")

# === Save to Shapefile ===
shapefile_path = os.path.join(output_dir, "OSE_Aquifer_Test_Wells.shp")
gdf = gpd.GeoDataFrame(sdf, geometry="SHAPE", crs="EPSG:4326")
gdf.to_file(shapefile_path)
print(f"✅ Shapefile saved to: {shapefile_path}")
