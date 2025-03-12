import os
import pandas as pd

def find_unique_tif_files(directory, output_csv="unique_tif_files.csv"):
    unique_names = set()

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            base_name = filename.split("_")[0]  # Extract part before '_'
            unique_names.add(base_name)  # Store unique base names

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(sorted(unique_names), columns=["Unique File Names"])
    df.to_csv(output_csv, index=False)

    print(f"Found {len(unique_names)} unique file names. Saved to {output_csv}.")


def main():
    # Example usage
    directory_path = r"W:\regional\tularosa_JDM\tularosa-JDM researcher\AlyssaBaca\Tularosa-JDM\OCDWellsLogs"  # Change this to your actual directory
    find_unique_tif_files(directory_path)

if __name__ == '__main__':
    main()

