import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import StandardScaler

# Base path to your dataset
base_path = r'D:\IEC 60870-5-104 SEG Intrusion Detection Dataset'

# List of main folders (these are the directories you want to scan)
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

# Create an empty list to store dataframes
dfs = []

# Process each folder in the dataset
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    if os.path.exists(folder_path):
        print(f"Scanning folder: {folder_path}")
        csv_files = load_csv_from_subfolders(folder)
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {folder_path}")
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    attack_type = os.path.basename(folder).split('_')[4:]
                    df['attack_type'] = '_'.join(attack_type)
                    dfs.append(df)
                    print(f"✅ Processed: {os.path.basename(csv_file)}")
                except Exception as e:
                    print(f"❌ Error processing {csv_file}: {str(e)}")
        else:
            print(f"No CSV files found in {folder_path}")
    else:
        print(f"Folder not found: {folder_path}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(base_path, 'combined_dataset.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n✅ Combined dataset saved to: {output_path}")
    print(f"📊 Total rows: {len(combined_df)}")
    print(f"📊 Total columns: {len(combined_df.columns)}")
else:
    print("❌ No data was processed!")

# Step 2: Data Analysis
print("\n🔍 Starting data analysis...")

# Display basic information about the dataset
print("\n📊 Dataset Preview:")
print(combined_df.head())

print("\n📊 Dataset Shape:")
print(f"Rows, Columns: {combined_df.shape}")

# Basic statistics
print("\n📊 Summary Statistics:")
print(combined_df.describe())

# Check for missing values
print("\n🔍 Missing values per column:")
missing_values = combined_df.isnull().sum()
print(missing_values[missing_values > 0])  # Only show columns with missing values

# Identify numerical and categorical columns
numerical_columns = combined_df.select_dtypes(include=[np.number]).columns
categorical_columns = combined_df.select_dtypes(exclude=[np.number]).columns

print("\n🔢 Numerical columns:")
print(numerical_columns.tolist())
print("\n📝 Categorical columns:")
print(categorical_columns.tolist())

# Correlation analysis
print("\n📊 Creating correlation matrix...")
correlation_matrix = combined_df[numerical_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Categorical data analysis
print("\n📊 Categorical Data Analysis:")
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(combined_df[col].value_counts())
    
    # Create bar plot for categorical variables
    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Distribution plots for numerical variables
print("\n📈 Creating distribution plots for numerical variables...")
for col in numerical_columns[:5]:  # Limiting to first 5 numerical columns to avoid too many plots
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df, x=col, kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# Handle missing values
print("\n🧹 Handling missing values...")
# For numerical columns
combined_df[numerical_columns] = combined_df[numerical_columns].fillna(combined_df[numerical_columns].mean())
# For categorical columns
for col in categorical_columns:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

# Verify no missing values remain
print("\n✅ Missing values after handling:")
print(combined_df.isnull().sum().sum(), "missing values remain")

# Scale numerical features
print("\n⚖️ Scaling numerical features...")
scaler = StandardScaler()
combined_df[numerical_columns] = scaler.fit_transform(combined_df[numerical_columns])

# Save processed dataset
processed_output_path = os.path.join(base_path, 'processed_dataset.csv')
combined_df.to_csv(processed_output_path, index=False)
print(f"\n✅ Processed dataset saved to: {processed_output_path}")

# Final summary
print("\n📋 Analysis Summary:")
print(f"• Total samples: {len(combined_df)}")
print(f"• Features: {len(combined_df.columns)}")
print(f"• Numerical features: {len(numerical_columns)}")
print(f"• Categorical features: {len(categorical_columns)}")

print("\n✅ Analysis complete!")
