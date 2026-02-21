import os
import pandas as pd

# Directory containing all CSV files (and subdirectories)
input_folder = r"F:\IEC 60870-5-104 SEG Intrusion Detection Dataset"  # Replace with your folder path
output_file = "combined_output.csv"  # Output combined file name

def combine_csvs_recursive(input_folder, output_file):
    # List to hold all dataframes
    combined_data = []

    # Walk through the directory and find all CSV files
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    print(f"Reading {file_path}...")
                    # Read each CSV and append to the list
                    df = pd.read_csv(file_path)
                    combined_data.append(df)
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")

    # Combine all DataFrames
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved to {output_file}")
    else:
        print("No valid CSV files found to combine.")

# Run the function
combine_csvs_recursive(input_folder, output_file)
