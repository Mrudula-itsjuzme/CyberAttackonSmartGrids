import os
import pandas as pd

# Base path to your dataset
base_path = r'D:\IEC 60870-5-104 SEG Intrusion Detection Dataset'

# List of main folders
folders = [
    '20200425_UOWM_IEC104_Dataset_m_sp_na_1_DoS',
    '20200426_UOWM_IEC104_Dataset_c_ci_na_1',
    '20200426_UOWM_IEC104_Dataset_c_ci_na_1_DoS',
    '20200427_UOWM_IEC104_Dataset_c_se_na_1',
    '20200428_UOWM_IEC104_Dataset_c_sc_na_1',
    '20200428_UOWM_IEC104_Dataset_c_se_na_1_DoS',
    '20200429_UOWM_IEC104_Dataset_c_sc_na_1_DoS',
    '20200605_UOWM_IEC104_Dataset_c_rd_na_1',
    '20200605_UOWM_IEC104_Dataset_c_rd_na_1_DoS',
    '20200606_UOWM_IEC104_Dataset_c_rp_na_1',
    '20200606_UOWM_IEC104_Dataset_c_rp_na_1_DoS',
    '20200608_UOWM_IEC104_Dataset_mitm_drop'
]

# Function to load CSV files from a folder and its subfolders
def load_csv_from_subfolders(folder):
    csv_files = []
    
    # Construct the full path to the folder
    folder_path = os.path.join(base_path, folder)
    
    # Walk through all the subdirectories and files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

# Load all the CSV files from the provided directories
all_csv_files = []
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    if os.path.exists(folder_path):
        print(f"Scanning folder: {folder_path}")
        csv_files = load_csv_from_subfolders(folder)
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {folder_path}")
            all_csv_files.extend(csv_files)
        else:
            print(f"No CSV files found in {folder_path}")
    else:
        print(f"Folder not found: {folder_path}")

# Check if we have found any CSV files
if all_csv_files:
    print(f"Total CSV files found: {len(all_csv_files)}")
    
    # Load all CSV data into a single DataFrame
    all_data = pd.concat([pd.read_csv(file) for file in all_csv_files], ignore_index=True)
    
    # Save the combined data to a new CSV file
    combined_csv_path = os.path.join(base_path, 'combined_dataset.csv')
    all_data.to_csv(combined_csv_path, index=False)
    print(f"Data has been successfully combined and saved to: {combined_csv_path}")
else:
    print("No CSV files were found in the specified folders.")
