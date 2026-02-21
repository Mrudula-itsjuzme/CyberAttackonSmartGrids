import pandas as pd
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the directory where the CSV files are located
base_dir = r"D:\IEC 60870-5-104 SEG Intrusion Detection Dataset\Data"
all_files = []

# Start scanning process
logging.info(f"Scanning directory: {base_dir}")

# Loop through all the folders and read CSV files
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(subdir, file)
            logging.info(f"Reading file: {file_path}")
            try:
                # Use low_memory=False to avoid DtypeWarning
                all_files.append(pd.read_csv(file_path, low_memory=False))
                logging.info(f"Successfully read: {file_path}")
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")

# Check if we found any CSV files
if all_files:
    # Combine all CSV files into one DataFrame
    logging.info("Combining CSV files into a single DataFrame...")
    combined_df = pd.concat(all_files, ignore_index=True)
    logging.info(f"Successfully combined {len(all_files)} files.")

    # Optionally, save the combined DataFrame to a new CSV file
    output_file = "combined_dataset.csv"
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Data successfully saved to {output_file}")
else:
    logging.warning("No CSV files were found or read.")

