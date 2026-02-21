import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define dataset paths
dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
processed_data_path = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"

# Load dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"Loaded dataset! Shape: {df.shape}")
else:
    print("[!] Error: Dataset not found!")
    exit()

# Identify target column
possible_labels = ['Label', 'label', 'target', 'class', 'attack_type']
target_col = next((col for col in df.columns if col in possible_labels), None)

if not target_col:
    print("[!] No valid target column found! Fix the dataset.")
    exit()

print(f"[+] Using '{target_col}' as the target column.")

# Convert categorical variables (including Label) to numeric
print("[+] Converting categorical variables to numeric using Label Encoding...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert strings to numerical labels
    label_encoders[col] = le  # Store encoders in case we need to decode later

# Replace infinite values
print("[+] Handling infinite values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
print("[+] Handling missing values with mean...")
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop all-NaN columns
print("[+] Removing any remaining all-NaN columns...")
df = df.dropna(axis=1, how='all')

# Apply Min-Max Scaling to numeric columns
print("[+] Applying Min-Max Scaling to handle outliers...")
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)  # Ensure target is not scaled

if numeric_cols:
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Ensure no missing values in the target column
df.dropna(subset=[target_col], inplace=True)

# Save processed dataset
df.to_csv(processed_data_path, index=False)
print(f"[+] Processed dataset saved at {processed_data_path}")
