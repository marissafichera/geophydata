import arcpy
import os


def update_source():
    # User Inputs
    aprx_path = r"\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData2.aprx"  # Change this to your ArcGIS Pro project
    old_source = r"W:\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb"  # Change to old folder or GDB path
    new_source = r"\\agustin\amp\statewide\AquiferCharacterization\ArcGIS\Projects\NMGeophysicalData\NMGeophysicalData.gdb"  # Change to new folder or GDB path
    target_map_name = 'Map'

    # Load ArcGIS Pro Project
    aprx = arcpy.mp.ArcGISProject(aprx_path)

    # Find the specific map
    target_map = None
    for m in aprx.listMaps():
        if m.name == target_map_name:
            target_map = m
            break

    if not target_map:
        print(f"Map '{target_map_name}' not found in the project.")
    else:
        # Iterate through layers in the selected map
        for lyr in target_map.listLayers():
            if lyr.supports("DATASOURCE"):
                current_source = lyr.dataSource

                # Check if the layer is using the old workspace
                if old_source.lower() in current_source.lower():
                    try:
                        # Detect if the old source is a File Geodatabase (GDB)
                        if old_source.lower().endswith(".gdb"):
                            # Update GDB workspace
                            lyr.findAndReplaceWorkspacePath(old_source, new_source, False)
                            print(f"Updated GDB Layer: {lyr.name} → {new_source}")
                        # else:
                            # Handle folder-based datasets (shapefiles, rasters, etc.)
                            # relative_path = os.path.relpath(current_source, old_source)
                            # new_path = os.path.join(new_source, relative_path)

                            # Ensure new data exists before updating
                            # if arcpy.Exists(new_path):
                            #     lyr.findAndReplaceWorkspacePath(old_source, new_source, False)
                            #     print(f"Updated Folder Layer: {lyr.name} → {new_path}")
                            # else:
                            #     print(f"Skipping {lyr.name}, new source not found: {new_path}")

                    except Exception as e:
                        print(Exception)
                        print(f"Error updating {lyr.name}: {e}")

    # Save changes to the project
    aprx.save()
    print(f"Data sources updated and saved in: {aprx_path}")


def main():
    update_source()

if __name__ == '__main__':
    main()
